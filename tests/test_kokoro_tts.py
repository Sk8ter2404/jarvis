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


if __name__ == "__main__":
    unittest.main()
