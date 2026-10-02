"""Tests for core/kokoro_tts — the CPU Kokoro TTS backend (2026-07-15, P2).

Pins the fail-closed contract that keeps JARVIS from ever going mute: synthesize()
NEVER raises and returns None on empty text / unavailable engine / render failure /
memoized prior failure, so the caller's edge → pyttsx3 → SAPI5 → silence ladder
takes over. Also pins CI-safety: importing the module must NOT drag in kokoro_onnx
or onnxruntime (the real import lives behind the lazy _engine() seam), so the bare
CI runner (tools/run_tests_ci_sim.py) never loads a heavy native dep.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

from core import config
from core import kokoro_tts as k


def _reset():
    k._ENGINE[0] = None
    k._FAILED[0] = False
    k._PHONEMIZER[0] = None
    k._PHON_OFF[0] = False


class KokoroTtsTests(unittest.TestCase):
    def setUp(self):
        # reset the module singleton/fail latch between tests
        _reset()

    def tearDown(self):
        _reset()

    def test_import_does_not_pull_in_kokoro_onnx(self):
        # the heavy native dep must load lazily, never at import — keeps CI light
        importlib.reload(k)
        self.assertNotIn("kokoro_onnx", sys.modules,
                         "kokoro_onnx must not import at module load (CI safety)")

    def test_empty_text_returns_none(self):
        self.assertIsNone(k.synthesize(""))
        self.assertIsNone(k.synthesize("   "))
        self.assertIsNone(k.synthesize(None))

    def test_unavailable_when_spec_missing(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(k.is_available())

    def test_unavailable_when_models_missing(self):
        with mock.patch.object(k, "_models_present", return_value=False):
            self.assertFalse(k.is_available())

    def test_unavailable_after_memoized_failure(self):
        k._FAILED[0] = True
        self.assertFalse(k.is_available())
        # and _engine short-circuits without retrying
        self.assertIsNone(k._engine())

    def test_synthesize_none_when_unavailable(self):
        with mock.patch.object(k, "is_available", return_value=False):
            self.assertIsNone(k.synthesize("hello sir"))

    def test_synthesize_is_fail_closed_on_engine_none(self):
        # available, but the engine fails to build → render yields nothing → None,
        # never an exception (the whole point of the contract)
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=None):
            self.assertIsNone(k.synthesize("all systems online"))

    def test_synthesize_returns_audio_on_success(self):
        import numpy as np
        fake = mock.Mock()
        fake.create.return_value = (np.zeros(2400, dtype=np.float32), 24000)
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=fake):
            res = k.synthesize("hello")
        self.assertIsNotNone(res)
        audio, sr = res
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.dtype("float32"))
        self.assertEqual(audio.ndim, 1)

    def test_render_exception_is_swallowed(self):
        boom = mock.Mock()
        boom.create.side_effect = RuntimeError("onnx boom")
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=boom):
            self.assertIsNone(k.synthesize("this should not raise"))


# ── speed plan R4: one persistent espeak backend (KOKORO_PERSISTENT_PHONEMIZER) ──
# Characters the fake model "knows"; anything else (here the trailing "ʊ" the
# fake backend appends) must be dropped by kokoro_onnx's post-filter.
_VOCAB = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ,.!?'")}
_SEPARATOR = object()          # stands in for phonemizer's default_separator


class _FakeBackend:
    """phonemizer's EspeakBackend: records every construction and, per
    phonemize call, whether _PHON_LOCK was held (asserting inside the render
    thread would be swallowed by _render's fail-closed except)."""
    built: list = []

    def __init__(self, lang, **kw):
        _FakeBackend.built.append((lang, kw))
        self.calls = []
        self.fail = False

    def phonemize(self, lines, separator=None, strip=None, njobs=None):
        self.calls.append((k._PHON_LOCK.locked(), list(lines),
                           (separator, strip, njobs)))
        if self.fail:
            raise RuntimeError("espeak boom")
        time.sleep(0.005)
        return [ln.lower() + " \u028a" for ln in lines]


class _FakeEngine:
    """kokoro_onnx.Kokoro: records each create() call raw, whether
    _RENDER_LOCK was held, and the most create() calls ever in flight."""

    def __init__(self, fail_phonemes=False):
        self.tokenizer = types.SimpleNamespace(vocab=_VOCAB)
        self.raw = []
        self.render_lock_held = []
        self.fail_phonemes = fail_phonemes
        self._mu = threading.Lock()
        self.active = 0
        self.max_active = 0

    def create(self, *args, **kwargs):
        import numpy as np
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.raw.append((args, kwargs))
            self.render_lock_held.append(k._RENDER_LOCK.locked())
            if self.fail_phonemes and kwargs.get("is_phonemes"):
                raise RuntimeError("onnx boom")
            time.sleep(0.02)
            return np.full(240, 0.25, dtype=np.float32), 24000
        finally:
            with self._mu:
                self.active -= 1


def _fake_modules(test, eng=None):
    """Fake kokoro_onnx / phonemizer modules (CI never has either) for the
    REAL _engine() / _phonemize() to import; Kokoro() builds `eng`."""
    def _kokoro(*_a, **_kw):
        time.sleep(0.05)               # widen the window two threads race in
        return eng
    mods = {
        "kokoro_onnx": types.SimpleNamespace(Kokoro=_kokoro),
        "phonemizer": types.ModuleType("phonemizer"),
        "phonemizer.backend": types.SimpleNamespace(EspeakBackend=_FakeBackend),
        "phonemizer.separator": types.SimpleNamespace(
            default_separator=_SEPARATOR),
        "espeakng_loader": None,       # the dll wiring is skipped (warning)
    }
    mods["phonemizer"].__path__ = []
    p = mock.patch.dict(sys.modules, mods)
    p.start()
    test.addCleanup(p.stop)


def _fake_stack(test, eng, persistent):
    """_fake_modules, plus model files "present" and the persistent flag set
    as given, so the REAL _engine() builds `eng`."""
    _fake_modules(test, eng)
    for p in (mock.patch.object(k, "_models_present", return_value=True),
              mock.patch.object(k, "_tuned_session", return_value=None),
              mock.patch.object(config, "KOKORO_PERSISTENT_PHONEMIZER",
                                persistent)):
        p.start()
        test.addCleanup(p.stop)


class PersistentPhonemizerTests(unittest.TestCase):
    def setUp(self):
        _reset()
        _FakeBackend.built = []

    def tearDown(self):
        _reset()

    def _render_from_two_threads(self, per_thread=3):
        outs = [[], []]
        barrier = threading.Barrier(2)

        def worker(i):
            barrier.wait()
            for n in range(per_thread):
                k._render(f"Hello sir, line {'ab'[i]}{'xyz'[n]}.", 1.0, outs[i])

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(10)
        return outs

    def test_one_backend_and_both_locks_held_from_two_threads(self):
        eng = _FakeEngine()
        _fake_stack(self, eng, persistent=True)
        outs = self._render_from_two_threads()
        self.assertEqual(len(outs[0]) + len(outs[1]), 6)
        # ONE backend, built with the stock call's options, for the session
        self.assertEqual(_FakeBackend.built, [
            (k._LANG, {"preserve_punctuation": True, "with_stress": True})])
        backend = k._PHONEMIZER[0]
        self.assertIsInstance(backend, _FakeBackend)
        self.assertEqual(len(backend.calls), 6)
        self.assertTrue(all(held for held, _l, _o in backend.calls),
                        "_PHON_LOCK must be held on every phonemize call")
        self.assertTrue(all(o == (_SEPARATOR, False, 1)
                            for _h, _l, o in backend.calls))
        # create() never ran twice at once, always under _RENDER_LOCK
        self.assertEqual(eng.max_active, 1)
        self.assertTrue(all(eng.render_lock_held))
        # phonemes in (vocab-filtered, stripped), is_phonemes=True
        for args, kwargs in eng.raw:
            self.assertEqual(kwargs, {"voice": k._VOICE, "speed": 1.0,
                                      "lang": k._LANG, "is_phonemes": True})
            self.assertRegex(args[0], r"^hello sir, line [ab][xyz]\.$")
        self.assertFalse(k._PHON_OFF[0])

    def test_phonemize_mirrors_the_stock_line_handling(self):
        eng = _FakeEngine()
        k._PHONEMIZER[0] = _FakeBackend(k._LANG)
        _fake_stack(self, eng, persistent=True)
        import os
        text = f"  One.{os.linesep}{os.linesep}Two!  "
        # lines split and blank ones dropped like phonemizer.phonemize; the
        # joining line break and the "\u028a" are not in the vocab, so the
        # post-filter drops them, exactly as Tokenizer.phonemize does
        self.assertEqual(k._phonemize(eng, text), "one. two!")
        self.assertEqual(k._PHONEMIZER[0].calls[-1][1], ["One.", "Two!"])
        self.assertEqual(k._phonemize(eng, "   "), "")

    def test_any_failure_latches_back_to_the_stock_call(self):
        for where in ("phonemize", "create"):
            with self.subTest(where=where):
                _reset()
                _fake_modules(self)
                eng = _FakeEngine(fail_phonemes=(where == "create"))
                backend = _FakeBackend(k._LANG)
                backend.fail = (where == "phonemize")
                k._ENGINE[0] = eng
                k._PHONEMIZER[0] = backend
                with mock.patch.object(config, "KOKORO_PERSISTENT_PHONEMIZER",
                                       True):
                    out = []
                    k._render("Hello sir.", 1.0, out)
                    k._render("Hello again.", 1.0, out)
                self.assertTrue(k._PHON_OFF[0])
                self.assertEqual(len(out), 2, "both lines still voiced")
                self.assertEqual(len(backend.calls), 1, "no retry after latch")
                self.assertEqual(eng.raw[-2:], [
                    (("Hello sir.",), {"voice": k._VOICE, "speed": 1.0,
                                       "lang": k._LANG}),
                    (("Hello again.",), {"voice": k._VOICE, "speed": 1.0,
                                         "lang": k._LANG})])

    def test_backend_build_failure_keeps_the_engine_on_the_stock_call(self):
        eng = _FakeEngine()
        _fake_stack(self, eng, persistent=True)
        with mock.patch.object(_FakeBackend, "__init__",
                               side_effect=OSError("dll copy failed")):
            self.assertIs(k._engine(), eng)
        self.assertIsNone(k._PHONEMIZER[0])
        self.assertTrue(k._PHON_OFF[0])
        self.assertFalse(k._FAILED[0])
        out = []
        k._render("Hello sir.", 1.0, out)
        self.assertEqual(len(out), 1)
        self.assertEqual(eng.raw[-1][1], {"voice": k._VOICE, "speed": 1.0,
                                          "lang": k._LANG})

    def test_flag_off_is_the_stock_call_byte_identical(self):
        import numpy as np
        eng = _FakeEngine()
        _fake_stack(self, eng, persistent=False)
        with mock.patch.object(k, "is_available", return_value=True):
            res = k.synthesize("Hello sir.", speed=1.1)
        self.assertEqual(_FakeBackend.built, [], "no backend built")
        self.assertIsNone(k._PHONEMIZER[0])
        # exactly today's call: raw text, no is_phonemes, no render lock
        self.assertEqual(eng.raw, [(("Hello sir.",), {
            "voice": k._VOICE, "speed": 1.1, "lang": k._LANG})])
        self.assertEqual(eng.render_lock_held, [False])
        audio, sr = res
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.tobytes(),
                         np.full(240, 0.25, dtype=np.float32).tobytes())

    def test_render_lock_timeout_returns_none(self):
        _fake_modules(self)
        eng = _FakeEngine()
        k._ENGINE[0] = eng
        k._PHONEMIZER[0] = _FakeBackend(k._LANG)
        with mock.patch.object(config, "KOKORO_PERSISTENT_PHONEMIZER", True), \
             mock.patch.object(k, "_SYNTH_TIMEOUT_S", 0.05):
            self.assertTrue(k._RENDER_LOCK.acquire(timeout=1))
            try:
                self.assertIsNone(k._create(eng, "Hello sir.", 1.0))
                out = []
                k._render("Hello sir.", 1.0, out)
                self.assertEqual(out, [])
            finally:
                k._RENDER_LOCK.release()
        self.assertEqual(eng.raw, [], "create() never ran without the lock")
        self.assertFalse(k._PHON_OFF[0], "a busy lock is not a failure")


# ── espeak-ng dll copies in %TEMP% (2026-10-02) ──────────────────────────────
# phonemizer (phonemizer-fork 3.3.1) copies the dll into a fresh mkdtemp() dir
# for EVERY EspeakWrapper(), and on Windows deletes it only at exit (atexit),
# which JARVIS's TerminateProcess exit skips. The fakes below mirror the real
# construction chain - BaseBackend.__init__ runs the is_available / version /
# supported_languages probes (each a throwaway wrapper), then
# BaseEspeakBackend.__init__ builds the wrapper it keeps - and count copies in
# an injected temp dir. RealEspeakCopyTests checks the same against the real
# library where it is installed (not on the CI runner).
_DLL = "espeak-ng.dll"


def _dll_copies(root):
    import glob
    import os
    return len(glob.glob(os.path.join(root, "tmp*", _DLL)))


class _ChainWrapper:
    """phonemizer.backend.espeak.wrapper.EspeakWrapper: one dll copy each."""

    def __init__(self):
        import os
        import tempfile
        d = tempfile.mkdtemp()             # the injected temp dir (tempfile.tempdir)
        with open(os.path.join(d, _DLL), "wb") as f:
            f.write(b"MZ")
        self.voice = None
        self.library_path = os.path.join(d, _DLL)

    @property
    def version(self):
        return (1, 52, 0)

    def available_voices(self):
        return [types.SimpleNamespace(language="en-gb", name="English (GB)"),
                types.SimpleNamespace(language="en-us", name="English (US)")]

    def set_voice(self, language):
        if language not in {v.language for v in self.available_voices()}:
            raise RuntimeError(f'invalid voice code "{language}"')
        self.voice = language

    def text_to_phonemes(self, line):
        return f"{self.voice}:{line.lower()}"


class _ChainBase:
    """phonemizer.backend.base.BaseBackend.__init__: the three probes."""

    def __init__(self, language, punctuation_marks=None,
                 preserve_punctuation=False, logger=None):
        self.logger = logger or mock.Mock()
        if not self.is_available():
            raise RuntimeError("espeak not installed on your system")
        self.logger.info("initializing backend espeak-%s",
                         ".".join(str(v) for v in self.version()))
        if language not in self.supported_languages():
            raise RuntimeError(f'language "{language}" is not supported')
        self._language = language
        self._preserve_punctuation = preserve_punctuation


class _ChainEspeakBase(_ChainBase):
    """phonemizer.backend.espeak.base.BaseEspeakBackend."""

    def __init__(self, language, punctuation_marks=None,
                 preserve_punctuation=False, logger=None):
        super().__init__(language, punctuation_marks=punctuation_marks,
                         preserve_punctuation=preserve_punctuation,
                         logger=logger)
        self._espeak = _ChainWrapper()

    @classmethod
    def is_available(cls):
        _ChainWrapper()
        return True

    @classmethod
    def version(cls):
        return _ChainWrapper().version


class _ChainEspeakBackend(_ChainEspeakBase):
    """phonemizer.backend.espeak.espeak.EspeakBackend."""

    def __init__(self, language, punctuation_marks=None,
                 preserve_punctuation=False, with_stress=False, tie=False,
                 language_switch="keep-flags", words_mismatch="ignore",
                 logger=None):
        super().__init__(language, punctuation_marks=punctuation_marks,
                         preserve_punctuation=preserve_punctuation,
                         logger=logger)
        self._espeak.set_voice(language)
        self._with_stress = with_stress

    @classmethod
    def supported_languages(cls):
        return {v.language: v.name for v in _ChainWrapper().available_voices()}

    def phonemize(self, lines, separator=None, strip=False, njobs=1):
        return [self._espeak.text_to_phonemes(ln) + " ʊ" for ln in lines]


def _chain_phonemize(text, language, **kw):
    """phonemizer.phonemize(): a fresh backend per call (kokoro_onnx's
    Tokenizer.phonemize, the stock create() path)."""
    return _ChainEspeakBackend(language, **kw).phonemize([text])[0]


class _ChainEngine(_FakeEngine):
    """kokoro_onnx.Kokoro whose stock create() phonemizes through the fake
    phonemizer.phonemize(), as kokoro_onnx 0.4.7 does."""

    def create(self, text, *args, **kwargs):
        if not kwargs.get("is_phonemes"):
            _chain_phonemize(text, kwargs.get("lang"),
                             preserve_punctuation=True, with_stress=True)
        return super().create(text, *args, **kwargs)


def _chain_modules(test, eng):
    """sys.modules fakes for the chain above, so the REAL _engine() /
    _one_copy_backend() import them; Kokoro() builds `eng`."""
    def _pkg(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        return m
    mods = {
        "kokoro_onnx": types.SimpleNamespace(Kokoro=lambda *_a, **_kw: eng),
        "phonemizer": _pkg("phonemizer", phonemize=_chain_phonemize),
        "phonemizer.backend": _pkg("phonemizer.backend",
                                   EspeakBackend=_ChainEspeakBackend),
        "phonemizer.backend.base": types.SimpleNamespace(BaseBackend=_ChainBase),
        "phonemizer.backend.espeak": _pkg("phonemizer.backend.espeak"),
        "phonemizer.backend.espeak.base": types.SimpleNamespace(
            BaseEspeakBackend=_ChainEspeakBase),
        "phonemizer.backend.espeak.wrapper": types.SimpleNamespace(
            EspeakWrapper=_ChainWrapper),
        "phonemizer.separator": types.SimpleNamespace(
            default_separator=_SEPARATOR),
        "espeakng_loader": None,       # the dll wiring is skipped (warning)
    }
    p = mock.patch.dict(sys.modules, mods)
    p.start()
    test.addCleanup(p.stop)


class EspeakDllCopyTests(unittest.TestCase):
    """At most ONE dll copy per process on the persistent path, none per line."""

    def setUp(self):
        import tempfile
        _reset()
        # numpy imported for the first time INSIDE a patch.dict(sys.modules)
        # is dropped again when it ends, and numpy cannot be re-imported.
        importlib.import_module("numpy")
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.temp = self._td.name
        p = mock.patch.object(tempfile, "tempdir", self.temp)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        _reset()

    def _stack(self, persistent):
        eng = _ChainEngine()
        _chain_modules(self, eng)
        for p in (mock.patch.object(k, "_models_present", return_value=True),
                  mock.patch.object(k, "_tuned_session", return_value=None),
                  mock.patch.object(config, "KOKORO_PERSISTENT_PHONEMIZER",
                                    persistent)):
            p.start()
            self.addCleanup(p.stop)
        return eng

    def _speak(self, n):
        out = []
        for i in range(n):
            k._render(f"Line {i}, sir.", 1.0, out)
        self.assertEqual(len(out), n, "every line still voiced")

    def test_stock_call_leaves_four_copies_per_line(self):
        # today's flag-off path, unchanged - and proof the counter sees copies
        self._stack(persistent=False)
        self.assertIsNotNone(k._engine())
        self.assertEqual(_dll_copies(self.temp), 0)
        self._speak(3)
        self.assertEqual(_dll_copies(self.temp), 12)

    def test_persistent_backend_is_one_copy_for_the_process(self):
        eng = self._stack(persistent=True)
        self.assertIs(k._engine(), eng)
        self.assertEqual(_dll_copies(self.temp), 1,
                         "the backend's probes must not each copy the dll")
        # built with the stock call's options (2026-10-02 review: the older
        # _FakeBackend.built pin now only reaches the fallback constructor)
        b = k._PHONEMIZER[0]
        self.assertIsInstance(b, _ChainEspeakBackend)
        self.assertEqual((b._language, b._preserve_punctuation, b._with_stress),
                         (k._LANG, True, True))
        self._speak(25)
        self.assertEqual(_dll_copies(self.temp), 1, "no copy per line")
        self.assertIs(k._engine(), eng)
        self.assertEqual(_dll_copies(self.temp), 1)
        self.assertFalse(k._PHON_OFF[0])
        self.assertTrue(all(kw.get("is_phonemes") for _a, kw in eng.raw))

    def test_one_copy_backend_is_the_stock_backend_on_one_wrapper(self):
        self._stack(persistent=True)
        b = k._one_copy_backend(k._LANG, preserve_punctuation=True,
                                with_stress=True)
        self.assertEqual(_dll_copies(self.temp), 1)
        self.assertIsInstance(b, _ChainEspeakBackend)
        self.assertIsInstance(b._espeak, _ChainWrapper)
        self.assertEqual(b._espeak.voice, k._LANG, "EspeakBackend set-up ran")
        self.assertTrue(b._with_stress)
        self.assertTrue(b._preserve_punctuation)
        self.assertEqual(b.version(), (1, 52, 0))
        stock = _ChainEspeakBackend(k._LANG, preserve_punctuation=True,
                                    with_stress=True)
        self.assertEqual(_dll_copies(self.temp), 5, "stock constructor: four")
        lines = ["Hello sir.", "All systems online!"]
        self.assertEqual(b.phonemize(lines), stock.phonemize(lines))
        with self.assertRaises(RuntimeError):     # language still validated
            k._one_copy_backend("xx-zz")

    def test_falls_back_to_phonemizers_own_constructor(self):
        self._stack(persistent=True)
        with mock.patch.object(k, "_one_copy_backend",
                               side_effect=AttributeError("internals moved")):
            b = k._build_phonemizer()
        self.assertIsInstance(b, _ChainEspeakBackend)
        self.assertEqual(_dll_copies(self.temp), 4)
        self.assertFalse(k._PHON_OFF[0], "still the persistent path")


class RealEspeakCopyTests(unittest.TestCase):
    """The real phonemizer + bundled espeak-ng, in a child process whose
    TEMP/TMP is a fresh directory: the one-copy backend copies the dll once,
    phonemizing never copies it, and its phonemes match phonemizer's own
    backend. Skipped where phonemizer is not installed (the CI runner)."""

    _CHILD = r"""
import glob, json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import espeakng_loader
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import default_separator
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
from core import kokoro_tts as k
n = lambda: len(glob.glob(os.path.join(tempfile.gettempdir(), "tmp*", "*espeak*")))
lines = ["All systems are online, sir.", "It is 3.5 degrees; e.g. at 2:30 p.m.",
         "Hello... world? Yes!"]
r = {"start": n()}
b = k._build_phonemizer()     # the engine's own call, options and all
r["built"] = n()
one = [b.phonemize([ln], separator=default_separator, strip=False, njobs=1)
       for ln in lines * 3]
r["lines"] = n()
s = EspeakBackend(k._LANG, preserve_punctuation=True, with_stress=True)
r["stock"] = n()
r["parity"] = one == [s.phonemize([ln], separator=default_separator,
                                  strip=False, njobs=1) for ln in lines * 3]
print("RESULT " + json.dumps(r))
"""

    def test_real_library_one_copy(self):
        import importlib.util
        import json
        import os
        import subprocess
        import tempfile
        for mod in ("phonemizer", "espeakng_loader"):
            if importlib.util.find_spec(mod) is None:
                self.skipTest(f"{mod} not installed")
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ, TEMP=td, TMP=td, TMPDIR=td,
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            p = subprocess.run([sys.executable, "-c", self._CHILD, repo],
                               env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=120)
        got = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertTrue(got, f"child failed: {p.stderr[-2000:]}")
        r = json.loads(got[-1][len("RESULT "):])
        self.assertEqual(r["start"], 0)
        self.assertEqual(r["built"], 1, "one dll copy for the backend")
        self.assertEqual(r["lines"], 1, "phonemizing never copies the dll")
        self.assertGreater(r["stock"], r["lines"],
                           "the counter must see phonemizer's own copies")
        self.assertTrue(r["parity"], "same phonemes as phonemizer's backend")


if __name__ == "__main__":
    unittest.main()
