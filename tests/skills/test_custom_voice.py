"""Logic tests for skills/custom_voice.py (Coqui XTTS-v2 voice cloning).

XTTS needs a GPU + the `TTS` package + a sample WAV, none of which exist in the
test env — so graceful degradation is the dominant path and is easy to assert.
We also test the pure prosody parsers, config resolution off a faked
bobert_companion, the backend-toggle validation/refusal, and the voice
pre-router regex.

The heavier paths are exercised with FAKE backends injected only inside a
with-block (a fake `TTS` package + `TTS.api.TTS` model class, a fake `torch`
exposing cuda.is_available(), and a fake `librosa`): the model-load cache and
its CUDA-OOM recovery, the full `render()` synthesis+prosody path, and the
`render_stream()` generator (both the inference_stream branch and the
non-streaming fallback). No real model is ever constructed, no GPU touched, no
audio synthesised or played.

ISOLATION: every fake module lives only inside an `inject_modules` with-block
that saves+restores sys.modules (and any parent-package attribute, e.g.
``TTS.api``) on exit. The module-level cache globals custom_voice mutates
(_HAS_TTS_LIB, _TTS_IMPORT_ERROR, _xtts_model, _xtts_loaded_sample,
_xtts_load_error) are reset in tearDown so no state leaks between tests. Real
numpy stays intact.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules. For dotted names
    (``TTS.api``) the leaf is ALSO set as an attribute on its already-imported
    parent package so ``from TTS.api import TTS`` resolves the fake. Restores
    the previous state — including absence — on exit. Pass ``name=None`` to
    force a module to look absent inside the block."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


def _make_tts_pkg(model=_SENTINEL, *, api_raises=False, ctor_raises=None):
    """Build a fake ``TTS`` package + ``TTS.api`` submodule exposing a `TTS`
    class. `model` is the instance returned by ``TTS.api.TTS(...)`` (default: a
    MagicMock). `api_raises` makes ``from TTS.api import TTS`` raise.
    `ctor_raises` is an exception the constructor raises (e.g. a CUDA-OOM)."""
    tts_pkg = types.ModuleType("TTS")
    api_mod = types.ModuleType("TTS.api")

    if not api_raises:
        inst = mock.MagicMock(name="CoquiTTSInstance") if model is _SENTINEL else model

        class _CoquiTTS:
            last_kwargs = None

            def __init__(self, *a, **k):
                _CoquiTTS.last_kwargs = k
                if ctor_raises is not None:
                    raise ctor_raises
                # Copy the prepared instance's behaviour onto self by simply
                # returning it via __new__ trickery is overkill; instead expose
                # the prepared instance as an attribute the test can reach, and
                # proxy .tts to it.
                self._inst = inst

            def tts(self, *a, **k):
                return self._inst.tts(*a, **k)

        api_mod.TTS = _CoquiTTS
        tts_pkg.api = api_mod
        tts_pkg._prepared_instance = inst
    return tts_pkg, api_mod


def _make_torch(cuda_available=False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: None,
    )
    return torch


class _FakeBobert:
    """A stand-in bobert_companion with the config attrs custom_voice reads."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class RateParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_positive_percent(self):
        self.assertAlmostEqual(self.mod._parse_rate("+5%"), 1.05)

    def test_negative_percent(self):
        self.assertAlmostEqual(self.mod._parse_rate("-10%"), 0.90)

    def test_clamped_to_bounds(self):
        self.assertEqual(self.mod._parse_rate("+500%"), 2.0)   # upper clamp
        self.assertEqual(self.mod._parse_rate("-90%"), 0.5)    # lower clamp

    def test_unparseable_is_unity(self):
        self.assertEqual(self.mod._parse_rate("fast"), 1.0)
        self.assertEqual(self.mod._parse_rate(""), 1.0)


class PitchParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_zero_hz_is_zero_semitones(self):
        self.assertEqual(self.mod._parse_pitch_semitones("+0Hz"), 0.0)

    def test_positive_hz_positive_semitones(self):
        st = self.mod._parse_pitch_semitones("+4Hz")
        self.assertGreater(st, 0.0)
        self.assertLess(st, 1.0)   # ~0.34 semitones around a 200 Hz base

    def test_unparseable_is_zero(self):
        self.assertEqual(self.mod._parse_pitch_semitones("higher"), 0.0)


class ConfigResolutionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_backend_from_bobert(self):
        fake = _FakeBobert(TTS_BACKEND="xtts")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            self.assertEqual(self.mod.get_backend(), "xtts")

    def test_backend_from_env_when_bobert_absent(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"TTS_BACKEND": "pyttsx3"}, clear=True):
            self.assertEqual(self.mod.get_backend(), "pyttsx3")

    def test_backend_defaults_to_edge(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_backend(), "edge")

    def test_invalid_backend_value_ignored(self):
        fake = _FakeBobert(TTS_BACKEND="banana")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_backend(), "edge")

    def test_language_default(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_language(), "en")


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_unavailable_when_tts_lib_missing(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=False):
            self.assertFalse(self.mod.is_available())
            reason = self.mod.availability_reason()
        self.assertIn("Coqui TTS isn't installed", reason)

    def test_unavailable_when_no_sample(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path", return_value=""):
            self.assertFalse(self.mod.is_available())
            self.assertIn("No voice sample", self.mod.availability_reason())

    def test_unavailable_when_sample_path_missing_file(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\nope\missing.wav"):
            self.assertFalse(self.mod.is_available())
            self.assertIn("can't find the voice sample", self.mod.availability_reason())

    def test_available_when_lib_and_sample_present(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\voice\sample.wav"), \
             mock.patch.object(self.mod.os.path, "isfile", return_value=True):
            self.assertTrue(self.mod.is_available())
            self.assertEqual(self.mod.availability_reason(), "")


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_empty_text_returns_silence(self):
        audio, sr = self.mod.render("   ")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 1)

    def test_render_raises_when_unavailable(self):
        # availability_reason non-empty → render must raise so the caller
        # (bobert synthesise) can fall back to edge-tts.
        with mock.patch.object(self.mod, "availability_reason",
                               return_value="Coqui TTS isn't installed, sir"):
            with self.assertRaises(RuntimeError) as cm:
                self.mod.render("hello there")
        self.assertIn("Coqui TTS isn't installed", str(cm.exception))


class SetBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("custom_voice")

    def test_invalid_backend_rejected(self):
        out = self.mod.set_backend("banana")
        self.assertIn("isn't a TTS backend", out)
        self.assertIn("edge", out)

    def test_switch_to_edge_confirms_and_sets(self):
        fake = _FakeBobert(TTS_BACKEND="xtts")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            out = self.mod.set_backend("edge")
        self.assertIn("edge-tts", out)
        self.assertEqual(fake.TTS_BACKEND, "edge")

    def test_switch_to_xtts_refused_when_unavailable(self):
        fake = _FakeBobert(TTS_BACKEND="edge")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason",
                               return_value="No voice sample, sir"):
            out = self.mod.set_backend("xtts")
        self.assertIn("No voice sample", out)
        # Backend must NOT have been flipped to xtts on refusal.
        self.assertEqual(fake.TTS_BACKEND, "edge")

    def test_switch_to_xtts_when_available_warms_and_sets(self):
        fake = _FakeBobert(TTS_BACKEND="edge")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            out = self.mod.set_backend("xtts")
        self.assertIn("cloned voice", out)
        self.assertEqual(fake.TTS_BACKEND, "xtts")
        Thread.assert_called_once()   # background warm-up spawned (not started for real)


class MaybeSwitchBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_use_my_voice_routes_to_xtts(self):
        with mock.patch.object(self.mod, "set_backend",
                               return_value="ok-xtts") as sb:
            out = self.mod.maybe_switch_backend("use my voice")
        self.assertEqual(out, "ok-xtts")
        sb.assert_called_once_with("xtts")

    def test_switch_to_edge_voice_routes_to_edge(self):
        with mock.patch.object(self.mod, "set_backend", return_value="ok") as sb:
            self.mod.maybe_switch_backend("switch to the edge voice")
        sb.assert_called_once_with("edge")

    def test_offline_voice_routes_to_pyttsx3(self):
        with mock.patch.object(self.mod, "set_backend", return_value="ok") as sb:
            self.mod.maybe_switch_backend("use the offline voice")
        sb.assert_called_once_with("pyttsx3")

    def test_non_matching_utterance_returns_none(self):
        self.assertIsNone(self.mod.maybe_switch_backend("what time is it"))

    def test_empty_returns_none(self):
        self.assertIsNone(self.mod.maybe_switch_backend(""))


class EnrollSampleActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("custom_voice")

    def test_requires_path(self):
        self.assertIn("format:", self.actions["enroll_xtts_sample"](""))

    def test_missing_file(self):
        out = self.actions["enroll_xtts_sample"](r"C:\nope\missing.wav")
        self.assertIn("no such file", out.lower())

    def test_sets_path_and_invalidates_cache(self):
        fake = _FakeBobert()
        with mock.patch.object(self.mod.os.path, "isfile", return_value=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "_invalidate_model_cache") as inval:
            out = self.actions["enroll_xtts_sample"](r"C:\voice\new.wav")
        self.assertIn("Voice sample updated", out)
        self.assertTrue(getattr(fake, "XTTS_VOICE_SAMPLE", "").endswith("new.wav"))
        inval.assert_called_once()

    def test_list_backends_reports_current(self):
        with mock.patch.object(self.mod, "get_backend", return_value="edge"), \
             mock.patch.object(self.mod, "is_available", return_value=False), \
             mock.patch.object(self.mod, "availability_reason",
                               return_value="Coqui TTS isn't installed"), \
             mock.patch.object(self.mod, "get_sample_path", return_value=""):
            out = self.actions["list_tts_backends"]("")
        self.assertIn("current backend: edge", out)
        self.assertIn("xtts status", out)


# ─────────────────────────────────────────────────────────────────────────
#  Config accessors that read sample-path / language off bobert or env
# ─────────────────────────────────────────────────────────────────────────
class SamplePathAndLanguageTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_sample_path_from_bobert_is_absolutised(self):
        fake = _FakeBobert(XTTS_VOICE_SAMPLE="voice/sample.wav")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            out = self.mod.get_sample_path()
        self.assertTrue(os.path.isabs(out))
        self.assertTrue(out.replace("\\", "/").endswith("voice/sample.wav"))

    def test_sample_path_from_env_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"XTTS_VOICE_SAMPLE": "env/clip.wav"},
                             clear=True):
            out = self.mod.get_sample_path()
        self.assertTrue(os.path.isabs(out))
        self.assertTrue(out.replace("\\", "/").endswith("env/clip.wav"))

    def test_sample_path_empty_when_nothing_set(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_sample_path(), "")

    def test_sample_path_bobert_blank_falls_to_env(self):
        # bc present but XTTS_VOICE_SAMPLE empty -> falls through to env.
        fake = _FakeBobert(XTTS_VOICE_SAMPLE="")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.dict(os.environ, {"XTTS_VOICE_SAMPLE": "e/x.wav"},
                             clear=True):
            out = self.mod.get_sample_path()
        self.assertTrue(out.replace("\\", "/").endswith("e/x.wav"))

    def test_language_from_bobert(self):
        fake = _FakeBobert(XTTS_LANGUAGE="fr")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            self.assertEqual(self.mod.get_language(), "fr")

    def test_language_from_env(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"XTTS_LANGUAGE": "de"}, clear=True):
            self.assertEqual(self.mod.get_language(), "de")

    def test_set_backend_on_bobert_noop_when_absent(self):
        # No bobert loaded: helper must silently do nothing (124->exit).
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            self.mod._set_backend_on_bobert("edge")   # no raise == covered


# ─────────────────────────────────────────────────────────────────────────
#  _probe_tts_lib import probe (cached) + cuda-oom helpers
# ─────────────────────────────────────────────────────────────────────────
class ProbeAndCudaHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")
        self.addCleanup(self._reset_globals)

    def _reset_globals(self):
        # Defensive: the probe / load cache live on module globals. Each test
        # gets a fresh module from load_skill_isolated, but reset anyway so a
        # future shared-module refactor can't leak state across tests.
        self.mod._HAS_TTS_LIB = None
        self.mod._TTS_IMPORT_ERROR = None
        self.mod._xtts_model = None
        self.mod._xtts_loaded_sample = ""
        self.mod._xtts_load_error = None

    def test_probe_true_when_tts_imports(self):
        fake_tts = types.ModuleType("TTS")
        with inject_modules(TTS=fake_tts):
            self.assertTrue(self.mod._probe_tts_lib())
        self.assertTrue(self.mod._HAS_TTS_LIB)

    def test_probe_false_and_records_error_when_missing(self):
        # Force `import TTS` to fail by planting the None-sentinel.
        with inject_modules(TTS=None):
            sys.modules["TTS"] = None  # type: ignore[assignment]
            try:
                self.assertFalse(self.mod._probe_tts_lib())
            finally:
                sys.modules.pop("TTS", None)
        self.assertFalse(self.mod._HAS_TTS_LIB)
        self.assertIsNotNone(self.mod._TTS_IMPORT_ERROR)

    def test_probe_is_cached(self):
        # Pre-set the cache to True; probe must short-circuit without importing.
        self.mod._HAS_TTS_LIB = True
        with mock.patch.dict(sys.modules, {}, clear=False):
            self.assertTrue(self.mod._probe_tts_lib())

    def test_availability_reason_includes_import_error_hint(self):
        self.mod._HAS_TTS_LIB = False
        self.mod._TTS_IMPORT_ERROR = "ImportError: no module named TTS"
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=False):
            reason = self.mod.availability_reason()
        self.assertIn("Coqui TTS isn't installed", reason)
        self.assertIn("no module named TTS", reason)

    def test_is_cuda_oom_matches_on_message(self):
        self.assertTrue(self.mod._is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertTrue(self.mod._is_cuda_oom(RuntimeError("a cuda failure")))
        self.assertFalse(self.mod._is_cuda_oom(ValueError("bad arg")))

    def test_drop_gpu_model_cache_clears_and_calls_empty_cache(self):
        emptied = {"called": False}
        torch = _make_torch()
        torch.cuda.empty_cache = lambda: emptied.__setitem__("called", True)
        self.mod._xtts_model = object()
        self.mod._xtts_loaded_sample = "old.wav"
        with inject_modules(torch=torch):
            self.mod._drop_gpu_model_cache()
        self.assertIsNone(self.mod._xtts_model)
        self.assertEqual(self.mod._xtts_loaded_sample, "")
        self.assertTrue(emptied["called"])

    def test_drop_gpu_model_cache_survives_missing_torch(self):
        # torch is installed in dev, so popping it would re-import the real one.
        # Plant the None-sentinel to force `import torch` to raise inside the
        # helper, exercising its `except Exception: pass` (277-278).
        self.mod._xtts_model = object()
        saved = sys.modules.get("torch", _SENTINEL)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            self.mod._drop_gpu_model_cache()   # must not raise
        finally:
            if saved is _SENTINEL:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved
        self.assertIsNone(self.mod._xtts_model)

    def test_invalidate_model_cache(self):
        self.mod._xtts_model = object()
        self.mod._xtts_loaded_sample = "x.wav"
        self.mod._invalidate_model_cache()
        self.assertIsNone(self.mod._xtts_model)
        self.assertEqual(self.mod._xtts_loaded_sample, "")


# ─────────────────────────────────────────────────────────────────────────
#  _load_xtts_model — construction, caching, CUDA-OOM recovery
# ─────────────────────────────────────────────────────────────────────────
class LoadXttsModelTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")
        self.addCleanup(self._reset)

    def _reset(self):
        self.mod._xtts_model = None
        self.mod._xtts_loaded_sample = ""
        self.mod._xtts_load_error = None

    def test_loads_and_caches_model_cpu(self):
        tts_pkg, api_mod = _make_tts_pkg()
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=_make_torch(cuda_available=False)), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\sample.wav"):
            m1 = self.mod._load_xtts_model()
            # Second call with the same sample returns the cached instance.
            m2 = self.mod._load_xtts_model()
        self.assertIs(m1, m2)
        self.assertEqual(self.mod._xtts_loaded_sample, r"C:\v\sample.wav")
        # gpu flag passed to the ctor was False (cuda unavailable).
        self.assertEqual(api_mod.TTS.last_kwargs.get("gpu"), False)

    def test_passes_gpu_true_when_cuda_available(self):
        tts_pkg, api_mod = _make_tts_pkg()
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=_make_torch(cuda_available=True)), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"):
            self.mod._load_xtts_model()
        self.assertEqual(api_mod.TTS.last_kwargs.get("gpu"), True)

    def test_reloads_when_sample_changes(self):
        tts_pkg, api_mod = _make_tts_pkg()
        paths = iter([r"C:\v\one.wav", r"C:\v\two.wav"])
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=_make_torch()), \
             mock.patch.object(self.mod, "get_sample_path",
                               side_effect=lambda: next(paths)):
            self.mod._load_xtts_model()
            self.mod._load_xtts_model()
        # Sample changed between calls -> the cache key updated to the 2nd path.
        self.assertEqual(self.mod._xtts_loaded_sample, r"C:\v\two.wav")

    def test_import_failure_raises_runtime_error(self):
        tts_pkg, api_mod = _make_tts_pkg(api_raises=True)
        with inject_modules(TTS=tts_pkg), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"):
            # `from TTS.api import TTS` fails -> wrapped RuntimeError.
            with self.assertRaises(RuntimeError) as cm:
                self.mod._load_xtts_model()
        self.assertIn("TTS import failed", str(cm.exception))
        self.assertIsNotNone(self.mod._xtts_load_error)

    def test_ctor_cuda_oom_drops_cache_and_raises(self):
        oom = RuntimeError("CUDA out of memory")
        tts_pkg, api_mod = _make_tts_pkg(ctor_raises=oom)
        dropped = {"called": False}
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=_make_torch(cuda_available=True)), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_drop_gpu_model_cache",
                               side_effect=lambda: dropped.__setitem__("called", True)):
            with self.assertRaises(RuntimeError) as cm:
                self.mod._load_xtts_model()
        self.assertIn("XTTS-v2 load failed", str(cm.exception))
        self.assertTrue(dropped["called"])   # OOM path cleared the dead cache

    def test_ctor_generic_failure_does_not_drop_cache(self):
        tts_pkg, api_mod = _make_tts_pkg(ctor_raises=ValueError("bad model name"))
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=_make_torch(cuda_available=False)), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_drop_gpu_model_cache") as drop:
            with self.assertRaises(RuntimeError):
                self.mod._load_xtts_model()
        drop.assert_not_called()   # non-OOM (or cpu) failure leaves cache alone

    def test_torch_import_failure_defaults_gpu_false(self):
        tts_pkg, api_mod = _make_tts_pkg()
        with inject_modules(TTS=tts_pkg, **{"TTS.api": api_mod}), \
             inject_modules(torch=None), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"):
            sys.modules["torch"] = None  # type: ignore[assignment]
            try:
                self.mod._load_xtts_model()
            finally:
                sys.modules.pop("torch", None)
        self.assertEqual(api_mod.TTS.last_kwargs.get("gpu"), False)


# ─────────────────────────────────────────────────────────────────────────
#  render() — the full synthesis + optional prosody path
# ─────────────────────────────────────────────────────────────────────────
class RenderSuccessTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")
        self.addCleanup(self._reset)

    def _reset(self):
        self.mod._xtts_model = None
        self.mod._xtts_loaded_sample = ""

    def _model_returning(self, samples):
        model = mock.MagicMock(name="model")
        model.tts.return_value = list(samples)
        return model

    def test_render_returns_audio_no_prosody(self):
        model = self._model_returning([0.1, 0.2, 0.3, 0.4])
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "get_language", return_value="en"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            audio, sr = self.mod.render("hello sir", rate="+0%", pitch="+0Hz")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], 4)
        model.tts.assert_called_once()
        # Sample path + language are threaded into the tts() call.
        _, kwargs = model.tts.call_args
        self.assertEqual(kwargs.get("speaker_wav"), r"C:\v\s.wav")
        self.assertEqual(kwargs.get("language"), "en")

    def test_render_downmixes_stereo_output(self):
        # tts() returns a 2-D array -> render must average to mono.
        stereo = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        model = mock.MagicMock()
        model.tts.return_value = stereo
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            audio, sr = self.mod.render("hi")
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.shape[0], 2)

    def test_render_applies_librosa_prosody_when_present(self):
        model = self._model_returning([0.1, 0.2, 0.3, 0.4])
        calls = {"stretch": 0, "shift": 0}
        librosa = types.ModuleType("librosa")

        def _stretch(audio, rate=None):
            calls["stretch"] += 1
            return np.asarray(audio, dtype=np.float32)

        def _shift(audio, sr=None, n_steps=None):
            calls["shift"] += 1
            return np.asarray(audio, dtype=np.float32)

        librosa.effects = types.SimpleNamespace(
            time_stretch=_stretch, pitch_shift=_shift)
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model), \
             inject_modules(librosa=librosa):
            audio, sr = self.mod.render("hi", rate="+20%", pitch="+5Hz")
        self.assertEqual(calls["stretch"], 1)
        self.assertEqual(calls["shift"], 1)
        self.assertEqual(audio.dtype, np.float32)

    def test_render_missing_librosa_returns_raw_audio(self):
        model = self._model_returning([0.5, 0.5, 0.5])
        # librosa IS installed in the dev env, so popping it isn't enough — the
        # `import librosa` inside render() would re-import the real one off
        # disk. Plant the None-sentinel so the import raises ImportError and the
        # prosody block's except-pass falls through with the raw 3-sample audio.
        saved = sys.modules.get("librosa", _SENTINEL)
        sys.modules["librosa"] = None  # type: ignore[assignment]
        try:
            with mock.patch.object(self.mod, "availability_reason", return_value=""), \
                 mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
                 mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
                # rate/pitch non-default so the prosody block is entered, but
                # librosa import fails -> raw audio falls through unchanged.
                audio, sr = self.mod.render("hi", rate="+20%", pitch="+5Hz")
        finally:
            if saved is _SENTINEL:
                sys.modules.pop("librosa", None)
            else:
                sys.modules["librosa"] = saved
        self.assertEqual(audio.shape[0], 3)

    def _fake_librosa(self, calls):
        librosa = types.ModuleType("librosa")

        def _stretch(audio, rate=None):
            calls["stretch"] += 1
            return np.asarray(audio, dtype=np.float32)

        def _shift(audio, sr=None, n_steps=None):
            calls["shift"] += 1
            return np.asarray(audio, dtype=np.float32)

        librosa.effects = types.SimpleNamespace(
            time_stretch=_stretch, pitch_shift=_shift)
        return librosa

    def test_render_rate_only_skips_pitch_shift(self):
        # rate != 1.0 but pitch == 0 -> time_stretch runs, pitch_shift doesn't
        # (covers the 365-true / 367-false branch split).
        model = self._model_returning([0.1, 0.2, 0.3])
        calls = {"stretch": 0, "shift": 0}
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model), \
             inject_modules(librosa=self._fake_librosa(calls)):
            self.mod.render("hi", rate="+15%", pitch="+0Hz")
        self.assertEqual(calls["stretch"], 1)
        self.assertEqual(calls["shift"], 0)

    def test_render_pitch_only_skips_time_stretch(self):
        # rate == 1.0 but pitch != 0 -> pitch_shift runs, time_stretch doesn't
        # (covers the 365-false / 367-true branch split).
        model = self._model_returning([0.1, 0.2, 0.3])
        calls = {"stretch": 0, "shift": 0}
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model), \
             inject_modules(librosa=self._fake_librosa(calls)):
            self.mod.render("hi", rate="+0%", pitch="+6Hz")
        self.assertEqual(calls["stretch"], 0)
        self.assertEqual(calls["shift"], 1)

    def test_render_tts_failure_raises_runtime_error(self):
        model = mock.MagicMock()
        model.tts.side_effect = RuntimeError("synthesis blew up")
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            with self.assertRaises(RuntimeError) as cm:
                self.mod.render("boom")
        self.assertIn("XTTS render failed", str(cm.exception))

    def test_render_tts_cuda_oom_drops_cache_then_raises(self):
        model = mock.MagicMock()
        model.tts.side_effect = RuntimeError("CUDA out of memory")
        dropped = {"called": False}
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model), \
             mock.patch.object(self.mod, "_drop_gpu_model_cache",
                               side_effect=lambda: dropped.__setitem__("called", True)):
            with self.assertRaises(RuntimeError):
                self.mod.render("boom")
        self.assertTrue(dropped["called"])


# ─────────────────────────────────────────────────────────────────────────
#  render_stream() — streaming generator + non-streaming fallback
# ─────────────────────────────────────────────────────────────────────────
class RenderStreamTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")
        self.addCleanup(self._reset)

    def _reset(self):
        self.mod._xtts_model = None
        self.mod._xtts_loaded_sample = ""

    def test_stream_raises_on_availability_failure(self):
        with mock.patch.object(self.mod, "availability_reason",
                               return_value="No voice sample, sir"):
            gen = self.mod.render_stream("hi")
            with self.assertRaises(RuntimeError):
                next(gen)

    def test_stream_falls_back_to_render_when_no_inference_stream(self):
        # Model without a usable .synthesizer.tts_model.inference_stream ->
        # render_stream yields a single (audio, sr) from render().
        model = mock.MagicMock()
        model.synthesizer = None
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "get_language", return_value="en"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model), \
             mock.patch.object(self.mod, "render",
                               return_value=(np.array([0.1, 0.2], dtype=np.float32), 24000)):
            chunks = list(self.mod.render_stream("hi"))
        self.assertEqual(len(chunks), 1)
        audio, sr = chunks[0]
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape[0], 2)

    def test_stream_yields_chunks_from_inference_stream(self):
        # Build a model whose underlying xtts exposes inference_stream + the
        # conditioning-latents call. Chunks are returned as objects with a
        # .detach() (tensor-like) and as a plain array, to cover both arms.
        class _Tensor:
            def __init__(self, arr):
                self._arr = np.asarray(arr, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        xtts = mock.MagicMock()
        xtts.inference_stream.return_value = [
            _Tensor([0.1, 0.2]),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
        xtts.get_conditioning_latents.return_value = ("gpt_latent", "spk_emb")
        model = mock.MagicMock()
        model.synthesizer = types.SimpleNamespace(tts_model=xtts)
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "get_language", return_value="en"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            chunks = list(self.mod.render_stream("hello", language="en"))
        self.assertEqual(len(chunks), 2)
        for arr, sr in chunks:
            self.assertEqual(sr, 24000)
            self.assertEqual(arr.dtype, np.float32)

    def test_stream_downmixes_2d_chunk(self):
        # A 2-D chunk (e.g. [channels, samples]) must be averaged to mono
        # (line 425). shape (2, 3): rows=2 < cols=3 -> mean over axis 0.
        xtts = mock.MagicMock()
        xtts.get_conditioning_latents.return_value = ("g", "s")
        xtts.inference_stream.return_value = [
            np.array([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]], dtype=np.float32),
        ]
        model = mock.MagicMock()
        model.synthesizer = types.SimpleNamespace(tts_model=xtts)
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            chunks = list(self.mod.render_stream("hi"))
        self.assertEqual(len(chunks), 1)
        arr, _sr = chunks[0]
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.shape[0], 3)

    def test_stream_conditioning_failure_raises(self):
        xtts = mock.MagicMock()
        xtts.get_conditioning_latents.side_effect = RuntimeError("cond boom")
        model = mock.MagicMock()
        model.synthesizer = types.SimpleNamespace(tts_model=xtts)
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            gen = self.mod.render_stream("hello")
            with self.assertRaises(RuntimeError) as cm:
                list(gen)
        self.assertIn("XTTS conditioning failed", str(cm.exception))

    def test_stream_inference_failure_raises(self):
        xtts = mock.MagicMock()
        xtts.get_conditioning_latents.return_value = ("g", "s")
        xtts.inference_stream.side_effect = RuntimeError("stream boom")
        model = mock.MagicMock()
        model.synthesizer = types.SimpleNamespace(tts_model=xtts)
        with mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod, "get_sample_path", return_value=r"C:\v\s.wav"), \
             mock.patch.object(self.mod, "_load_xtts_model", return_value=model):
            gen = self.mod.render_stream("hello")
            with self.assertRaises(RuntimeError) as cm:
                list(gen)
        self.assertIn("XTTS stream failed", str(cm.exception))


# ─────────────────────────────────────────────────────────────────────────
#  Remaining branches: warm-up closure, pre-router None, action wrappers
# ─────────────────────────────────────────────────────────────────────────
class RemainingBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("custom_voice")
        self.addCleanup(self._reset)

    def _reset(self):
        self.mod._xtts_model = None
        self.mod._xtts_loaded_sample = ""
        self.mod._xtts_load_error = None

    def test_set_backend_warmup_closure_runs_load(self):
        # Replace Thread with a synchronous stand-in whose start() runs the
        # warm-up target inline, so the closure body (_load_xtts_model call) is
        # covered deterministically without a real background thread.
        fake = _FakeBobert(TTS_BACKEND="edge")

        class _SyncThread:
            def __init__(self, target=None, daemon=None, **k):
                self._target = target

            def start(self):
                if self._target is not None:
                    self._target()

        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod.threading, "Thread", _SyncThread), \
             mock.patch.object(self.mod, "_load_xtts_model") as load:
            out = self.mod.set_backend("xtts")
        self.assertIn("cloned voice", out)
        load.assert_called_once()   # warm-up closure invoked the loader

    def test_set_backend_warmup_closure_swallows_load_error(self):
        # Same synchronous Thread, but the loader raises: the closure's
        # try/except must swallow it (set_backend still returns the
        # confirmation and does not propagate).
        fake = _FakeBobert(TTS_BACKEND="edge")

        class _SyncThread:
            def __init__(self, target=None, daemon=None, **k):
                self._target = target

            def start(self):
                if self._target is not None:
                    self._target()

        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod.threading, "Thread", _SyncThread), \
             mock.patch.object(self.mod, "_load_xtts_model",
                               side_effect=RuntimeError("warm fail")):
            out = self.mod.set_backend("xtts")
        self.assertIn("cloned voice", out)

    def test_maybe_switch_backend_matches_regex_but_no_phrase_returns_none(self):
        # The final `return None` (after a regex match whose text contains none
        # of the dispatch phrase-tokens) is only reachable if .match() succeeds
        # on tokenless text. The compiled groups make that combination
        # unreachable via real input, so we swap the whole _BACKEND_VOICE_RE
        # attribute for a stub whose .match() returns truthy. (re.Pattern.match
        # itself is read-only and can't be patched in place.)
        stub_re = types.SimpleNamespace(match=lambda _s: object())
        with mock.patch.object(self.mod, "_BACKEND_VOICE_RE", stub_re):
            out = self.mod.maybe_switch_backend("some unrelated words")
        self.assertIsNone(out)

    def test_act_set_tts_backend_delegates(self):
        with mock.patch.object(self.mod, "set_backend", return_value="ok") as sb:
            out = self.actions["set_tts_backend"]("edge")
        self.assertEqual(out, "ok")
        sb.assert_called_once_with("edge")

    def test_act_enroll_sample_no_bobert(self):
        # File exists but bobert isn't loaded -> can't persist the path.
        with mock.patch.object(self.mod.os.path, "isfile", return_value=True), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            out = self.actions["enroll_xtts_sample"](r"C:\v\new.wav")
        self.assertIn("bobert_companion isn't loaded", out)

    def test_register_wires_actions(self):
        actions: dict = {}
        self.mod.register(actions)
        self.assertIn("set_tts_backend", actions)
        self.assertIn("list_tts_backends", actions)
        self.assertIn("enroll_xtts_sample", actions)


if __name__ == "__main__":
    unittest.main()
