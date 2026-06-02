"""Unit tests for core/voice_pipeline.py — the voice-subsystem SELECTOR.

This is the CI-critical test module for the low-latency voice wiring: it proves
the decision layer in front of core/realtime_voice.py + core/wake_word.py is
correct WITHOUT any of the optional pip deps and WITHOUT touching real audio.

CI contract (tools/run_tests_ci_sim.py / bare Linux GitHub runner):
  * core.voice_pipeline imports only stdlib + core.config, so it imports cleanly
    on the light-deps runner. numpy is present there but is never needed here.
  * The optional backends (RealtimeSTT/RealtimeTTS/openwakeword/pvporcupine) and
    the heavy core modules (core.realtime_voice/core.wake_word) are ABSENT on CI.
    Every test that needs a particular availability state forces it explicitly:
      - "deps absent"  → patch find_spec to report None (the real CI state).
      - "deps present" → patch find_spec to report a dummy spec AND inject a fake
        core.realtime_voice / core.wake_word into sys.modules, scoped per-test
        and auto-restored, so no real backend is imported.
  * The config flags are forced per-test via mock.patch.object on core.config
    (and os.environ for the env-override paths), never by mutating the real
    constants persistently.

PASS-or-SKIP on both OSes; no personal data; any secret-shaped value is built at
runtime. stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock

from core import config
import core.voice_pipeline as vp


# ──────────────────────────────────────────────────────────────────────
# Helpers: force optional-dep presence/absence + inject fake heavy modules
# ──────────────────────────────────────────────────────────────────────

def _fake_spec(name: str) -> importlib.machinery.ModuleSpec:
    """A throwaway ModuleSpec so find_spec(name) looks 'present' without the
    package existing. loader=None is fine — we never actually import it."""
    return importlib.machinery.ModuleSpec(name, loader=None)


def _patch_specs(testcase, present=(), absent=()):
    """Patch core.voice_pipeline's find_spec view so `present` names resolve and
    `absent` names report None. Any name not listed falls through to the real
    importlib.util.find_spec. Auto-restored via addCleanup."""
    present_set = set(present)
    absent_set = set(absent)
    real = importlib.util.find_spec

    def fake(name, package=None):
        if name in absent_set:
            return None
        if name in present_set:
            return _fake_spec(name)
        return real(name, package)

    p = mock.patch.object(vp.importlib.util, "find_spec", side_effect=fake)
    p.start()
    testcase.addCleanup(p.stop)


def _inject_module(testcase, name: str, module) -> None:
    """Install a fake module under `name` in sys.modules for one test, restoring
    the prior entry (including absence) afterward.

    For a dotted name (e.g. 'core.realtime_voice') the leaf is ALSO bound as an
    attribute on the already-imported parent package. This matters because
    ``from core import realtime_voice`` resolves via the parent's *attribute*
    when the real submodule was imported earlier in the run (which happens in
    the full CI-sim suite) — patching only sys.modules wouldn't shadow it. Both
    the attribute and the sys.modules entry are restored on cleanup."""
    sentinel = object()
    prev = sys.modules.get(name, sentinel)
    sys.modules[name] = module

    # Bind the leaf on the parent package too (dotted names), capturing how to
    # undo it. A list of zero-arg callables keeps the restore logic flat and
    # pyflakes-clean (no conditionally-redefined inner function).
    undo = []
    if "." in name:
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            prev_attr = getattr(parent, leaf, sentinel)
            setattr(parent, leaf, module)
            if prev_attr is sentinel:
                undo.append(lambda: _safe_delattr(parent, leaf))
            else:
                undo.append(lambda: setattr(parent, leaf, prev_attr))

    def restore():
        if prev is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev
        for fn in undo:
            fn()

    testcase.addCleanup(restore)


def _safe_delattr(obj, name: str) -> None:
    try:
        delattr(obj, name)
    except AttributeError:
        pass


def _fake_realtime_voice_module(*, start_result, start_raises=False):
    """Build a fake `core.realtime_voice` exposing the surface the selector uses:
    DEFAULT_STT_MODEL/DEFAULT_TTS_VOICE constants + start_pipeline().

    start_result: the object start_pipeline returns (a sentinel 'pipeline', or
        None to simulate an unavailable-at-start fallback).
    start_raises: if True, start_pipeline raises (drives the selector's except).
    """
    mod = types.ModuleType("core.realtime_voice")
    mod.DEFAULT_STT_MODEL = "base"
    mod.DEFAULT_TTS_VOICE = "en-GB-RyanNeural"
    calls = {}

    def start_pipeline(**kwargs):
        calls.update(kwargs)
        if start_raises:
            raise RuntimeError("start_pipeline boom")
        return start_result

    mod.start_pipeline = start_pipeline
    mod._calls = calls
    return mod


class _FakeDetector:
    """Stand-in for core.wake_word.WakeWordDetector with the attrs the selector
    touches: construction kwargs captured, start() scriptable."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.engine = kwargs.get("engine", "openwakeword")
        self.started = False
        self.start_result = True
        self.start_raises = False
        _FakeDetector.instances.append(self)

    def start(self):
        if self.start_raises:
            raise RuntimeError("detector start boom")
        self.started = True
        return self.start_result


def _fake_wake_word_module(detector_cls=_FakeDetector, ctor_raises=False):
    """Build a fake `core.wake_word` exposing WakeWordDetector + DEFAULT_FRAME_MS."""
    mod = types.ModuleType("core.wake_word")
    mod.DEFAULT_FRAME_MS = 80
    if ctor_raises:
        def _boom(**kwargs):
            raise RuntimeError("ctor boom")
        mod.WakeWordDetector = _boom
    else:
        mod.WakeWordDetector = detector_cls
    return mod


def _clear_voice_env(testcase):
    """Ensure the JARVIS_* voice overrides are absent for a test, restored after.
    Tests that want an override set it explicitly within their own patch.dict."""
    p = mock.patch.dict(
        os.environ,
        {k: v for k, v in os.environ.items()
         if k not in ("JARVIS_VOICE_MODE", "JARVIS_WAKE_WORD_AUTOSTART",
                      "JARVIS_WAKE_WORD_ENGINE")},
        clear=True,
    )
    p.start()
    testcase.addCleanup(p.stop)


# ──────────────────────────────────────────────────────────────────────
# Tiny pure helpers: _as_bool / _cfg / _spec_present
# ──────────────────────────────────────────────────────────────────────

class CoercionTests(unittest.TestCase):
    def test_as_bool_real_bools(self):
        self.assertTrue(vp._as_bool(True))
        self.assertFalse(vp._as_bool(False))

    def test_as_bool_truthy_strings(self):
        for s in ("1", "true", "TRUE", "Yes", "on", " y "):
            self.assertTrue(vp._as_bool(s), s)

    def test_as_bool_falsy_strings(self):
        for s in ("0", "false", "no", "off", "", "  "):
            self.assertFalse(vp._as_bool(s), s)

    def test_as_bool_unknown_uses_default(self):
        self.assertTrue(vp._as_bool("banana", default=True))
        self.assertFalse(vp._as_bool("banana", default=False))
        self.assertFalse(vp._as_bool(None))

    def test_as_bool_numbers(self):
        self.assertTrue(vp._as_bool(1))
        self.assertFalse(vp._as_bool(0))

    def test_cfg_prefers_env_over_config(self):
        _clear_voice_env(self)
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.dict(os.environ, {"JARVIS_VOICE_MODE": "realtime"}):
            self.assertEqual(vp._cfg("VOICE_MODE", "x"), "realtime")

    def test_cfg_blank_env_ignored(self):
        _clear_voice_env(self)
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.dict(os.environ, {"JARVIS_VOICE_MODE": "   "}):
            self.assertEqual(vp._cfg("VOICE_MODE", "x"), "turn_based")

    def test_cfg_falls_back_to_default_when_absent(self):
        _clear_voice_env(self)
        # A name that exists on neither config nor env → the supplied default.
        self.assertEqual(vp._cfg("NONEXISTENT_KNOB_XYZ", "dflt"), "dflt")

    def test_spec_present_handles_value_error(self):
        with mock.patch.object(vp.importlib.util, "find_spec",
                               side_effect=ValueError("half-init")):
            self.assertFalse(vp._spec_present("whatever"))

    def test_spec_present_true_false(self):
        with mock.patch.object(vp.importlib.util, "find_spec",
                               return_value=_fake_spec("x")):
            self.assertTrue(vp._spec_present("x"))
        with mock.patch.object(vp.importlib.util, "find_spec", return_value=None):
            self.assertFalse(vp._spec_present("x"))


# ──────────────────────────────────────────────────────────────────────
# F1: realtime_enabled / realtime_available
# ──────────────────────────────────────────────────────────────────────

class RealtimeEnabledTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)

    def test_default_turn_based_is_disabled(self):
        with mock.patch.object(config, "VOICE_MODE", "turn_based"):
            self.assertFalse(vp.realtime_enabled())

    def test_realtime_flag_enables(self):
        with mock.patch.object(config, "VOICE_MODE", "realtime"):
            self.assertTrue(vp.realtime_enabled())

    def test_case_and_whitespace_insensitive(self):
        with mock.patch.object(config, "VOICE_MODE", "  ReAlTiMe "):
            self.assertTrue(vp.realtime_enabled())

    def test_env_override_enables(self):
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.dict(os.environ, {"JARVIS_VOICE_MODE": "realtime"}):
            self.assertTrue(vp.realtime_enabled())

    def test_env_override_can_force_off(self):
        with mock.patch.object(config, "VOICE_MODE", "realtime"), \
                mock.patch.dict(os.environ, {"JARVIS_VOICE_MODE": "turn_based"}):
            self.assertFalse(vp.realtime_enabled())


class RealtimeAvailableTests(unittest.TestCase):
    def test_absent_deps_report_unavailable(self):
        _patch_specs(self, absent=("RealtimeSTT", "RealtimeTTS"))
        ok, why = vp.realtime_available()
        self.assertFalse(ok)
        self.assertIn("RealtimeSTT", why)
        self.assertIn("RealtimeTTS", why)

    def test_one_missing_reports_it(self):
        _patch_specs(self, present=("RealtimeSTT",), absent=("RealtimeTTS",))
        ok, why = vp.realtime_available()
        self.assertFalse(ok)
        self.assertIn("RealtimeTTS", why)
        self.assertNotIn("RealtimeSTT", why)

    def test_both_present_available(self):
        _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS"))
        ok, why = vp.realtime_available()
        self.assertTrue(ok)
        self.assertEqual(why, "")


# ──────────────────────────────────────────────────────────────────────
# F1: make_realtime_session — the core selection contract
# ──────────────────────────────────────────────────────────────────────

class MakeRealtimeSessionTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)

    def test_flag_off_returns_none_without_probing(self):
        # The hot path: flag off → None, and we must NOT even probe deps.
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.object(vp, "realtime_available") as avail:
            self.assertIsNone(vp.make_realtime_session())
            avail.assert_not_called()

    def test_flag_on_deps_absent_returns_none_and_logs(self):
        with mock.patch.object(config, "VOICE_MODE", "realtime"):
            _patch_specs(self, absent=("RealtimeSTT", "RealtimeTTS"))
            with mock.patch.object(vp, "_log") as log:
                self.assertIsNone(vp.make_realtime_session())
            self.assertTrue(log.called)

    def test_flag_on_deps_present_returns_started_session(self):
        sentinel = object()
        fake_rtv = _fake_realtime_voice_module(start_result=sentinel)
        with mock.patch.object(config, "VOICE_MODE", "realtime"):
            _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS"))
            _inject_module(self, "core.realtime_voice", fake_rtv)
            with mock.patch.object(vp, "_log"):
                got = vp.make_realtime_session(
                    on_user_utterance=lambda t: None, tts_voice="V")
        self.assertIs(got, sentinel)
        # The selector forwarded voice_mode=realtime + our hook + voice.
        self.assertEqual(fake_rtv._calls["voice_mode"], "realtime")
        self.assertEqual(fake_rtv._calls["tts_voice"], "V")

    def test_start_returns_none_falls_back(self):
        fake_rtv = _fake_realtime_voice_module(start_result=None)
        with mock.patch.object(config, "VOICE_MODE", "realtime"):
            _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS"))
            _inject_module(self, "core.realtime_voice", fake_rtv)
            with mock.patch.object(vp, "_log") as log:
                self.assertIsNone(vp.make_realtime_session())
            self.assertTrue(log.called)

    def test_start_raises_is_caught(self):
        fake_rtv = _fake_realtime_voice_module(start_result=None, start_raises=True)
        with mock.patch.object(config, "VOICE_MODE", "realtime"):
            _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS"))
            _inject_module(self, "core.realtime_voice", fake_rtv)
            with mock.patch.object(vp, "_log"):
                self.assertIsNone(vp.make_realtime_session())  # no raise

    def test_default_voice_resolves_from_config_tts_voice(self):
        sentinel = object()
        fake_rtv = _fake_realtime_voice_module(start_result=sentinel)
        with mock.patch.object(config, "VOICE_MODE", "realtime"), \
                mock.patch.object(config, "TTS_VOICE", "en-GB-RyanNeural"):
            _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS"))
            _inject_module(self, "core.realtime_voice", fake_rtv)
            with mock.patch.object(vp, "_log"):
                vp.make_realtime_session()   # no tts_voice override
        self.assertEqual(fake_rtv._calls["tts_voice"], "en-GB-RyanNeural")


# ──────────────────────────────────────────────────────────────────────
# F2: wake_word_autostart_enabled / wake_word_available
# ──────────────────────────────────────────────────────────────────────

class WakeAutostartEnabledTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)

    def test_default_false(self):
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", False):
            self.assertFalse(vp.wake_word_autostart_enabled())

    def test_true_enables(self):
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True):
            self.assertTrue(vp.wake_word_autostart_enabled())

    def test_env_override(self):
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", False), \
                mock.patch.dict(os.environ, {"JARVIS_WAKE_WORD_AUTOSTART": "1"}):
            self.assertTrue(vp.wake_word_autostart_enabled())

    def test_missing_config_constant_defaults_false(self):
        # Older tree without the constant → default False, no crash.
        _clear_voice_env(self)
        with mock.patch.object(vp, "_cfg", return_value=False):
            self.assertFalse(vp.wake_word_autostart_enabled())


class WakeWordAvailableTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)

    def test_engine_off(self):
        ok, why = vp.wake_word_available("off")
        self.assertFalse(ok)
        self.assertIn("off", why)

    def test_openwakeword_present(self):
        _patch_specs(self, present=("openwakeword",))
        ok, _ = vp.wake_word_available("openwakeword")
        self.assertTrue(ok)

    def test_openwakeword_absent(self):
        _patch_specs(self, absent=("openwakeword",))
        ok, why = vp.wake_word_available("openwakeword")
        self.assertFalse(ok)
        self.assertIn("openwakeword", why)

    def test_porcupine_present(self):
        _patch_specs(self, present=("pvporcupine",))
        ok, _ = vp.wake_word_available("porcupine")
        self.assertTrue(ok)

    def test_porcupine_absent(self):
        _patch_specs(self, absent=("pvporcupine",))
        ok, why = vp.wake_word_available("porcupine")
        self.assertFalse(ok)
        self.assertIn("pvporcupine", why)

    def test_unknown_engine(self):
        ok, why = vp.wake_word_available("banana")
        self.assertFalse(ok)
        self.assertIn("unknown engine", why)

    def test_default_engine_is_openwakeword(self):
        with mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                               create=True):
            self.assertEqual(vp.wake_word_engine(), "openwakeword")


# ──────────────────────────────────────────────────────────────────────
# F2: make_wake_detector / wake_detector_or_none
# ──────────────────────────────────────────────────────────────────────

class MakeWakeDetectorTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)
        _FakeDetector.instances.clear()

    def test_flag_off_returns_none_without_probing(self):
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", False), \
                mock.patch.object(vp, "wake_word_available") as avail:
            self.assertIsNone(vp.make_wake_detector())
            avail.assert_not_called()

    def test_flag_on_dep_absent_returns_none(self):
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, absent=("openwakeword",))
            with mock.patch.object(vp, "_log") as log:
                self.assertIsNone(vp.make_wake_detector())
            self.assertTrue(log.called)

    def test_flag_on_dep_present_autostart_false_returns_idle_detector(self):
        fake_ww = _fake_wake_word_module()
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            det = vp.make_wake_detector(autostart=False, wake_words=["jarvis"])
        self.assertIsNotNone(det)
        self.assertFalse(det.started)               # idle (not started)
        self.assertEqual(det.kwargs["wake_words"], ["jarvis"])

    def test_autostart_true_starts_detector(self):
        fake_ww = _fake_wake_word_module()
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            with mock.patch.object(vp, "_log"):
                det = vp.make_wake_detector(autostart=True)
        self.assertIsNotNone(det)
        self.assertTrue(det.started)

    def test_start_failure_returns_none(self):
        class _FailDetector(_FakeDetector):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.start_result = False
        fake_ww = _fake_wake_word_module(detector_cls=_FailDetector)
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            with mock.patch.object(vp, "_log") as log:
                self.assertIsNone(vp.make_wake_detector(autostart=True))
            self.assertTrue(log.called)

    def test_start_raises_is_caught(self):
        class _RaiseDetector(_FakeDetector):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.start_raises = True
        fake_ww = _fake_wake_word_module(detector_cls=_RaiseDetector)
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            with mock.patch.object(vp, "_log"):
                self.assertIsNone(vp.make_wake_detector(autostart=True))

    def test_ctor_raises_is_caught(self):
        fake_ww = _fake_wake_word_module(ctor_raises=True)
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            with mock.patch.object(vp, "_log"):
                self.assertIsNone(vp.make_wake_detector(autostart=False))

    def test_wake_detector_or_none_is_alias(self):
        # Same contract: flag off → None.
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", False):
            self.assertIsNone(vp.wake_detector_or_none())

    def test_threshold_and_device_forwarded(self):
        fake_ww = _fake_wake_word_module()
        with mock.patch.object(config, "WAKE_WORD_AUTOSTART", True), \
                mock.patch.object(config, "WAKE_WORD_ENGINE", "openwakeword",
                                  create=True):
            _patch_specs(self, present=("openwakeword",))
            _inject_module(self, "core.wake_word", fake_ww)
            det = vp.make_wake_detector(autostart=False, threshold=0.8, device=3)
        self.assertEqual(det.kwargs["threshold"], 0.8)
        self.assertEqual(det.kwargs["device"], 3)


# ──────────────────────────────────────────────────────────────────────
# Diagnostics snapshot
# ──────────────────────────────────────────────────────────────────────

class DepsStatusTests(unittest.TestCase):
    def setUp(self):
        _clear_voice_env(self)

    def test_all_absent_shape(self):
        _patch_specs(self, absent=("RealtimeSTT", "RealtimeTTS",
                                   "openwakeword", "pvporcupine"))
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.object(config, "WAKE_WORD_AUTOSTART", False):
            st = vp.deps_status()
        self.assertFalse(st["realtime_enabled"])
        self.assertFalse(st["realtime_available"])
        self.assertFalse(st["wake_word_autostart"])
        self.assertFalse(st["wake_word_available"])
        self.assertEqual(st["realtime_deps"],
                         {"RealtimeSTT": False, "RealtimeTTS": False})
        self.assertEqual(st["wake_word_deps"],
                         {"openwakeword": False, "pvporcupine": False})

    def test_all_present_and_enabled(self):
        _patch_specs(self, present=("RealtimeSTT", "RealtimeTTS",
                                    "openwakeword", "pvporcupine"))
        with mock.patch.object(config, "VOICE_MODE", "realtime"), \
                mock.patch.object(config, "WAKE_WORD_AUTOSTART", True):
            st = vp.deps_status()
        self.assertTrue(st["realtime_enabled"])
        self.assertTrue(st["realtime_available"])
        self.assertTrue(st["wake_word_autostart"])
        self.assertTrue(st["wake_word_available"])


# ──────────────────────────────────────────────────────────────────────
# Import-safety: the module + selectors are total with NO backends present
# ──────────────────────────────────────────────────────────────────────

class ImportSafetyTests(unittest.TestCase):
    """The whole point of the selector: on a bare runner (this is the CI state)
    nothing raises and every selector reports the safe 'off/None' result."""

    def setUp(self):
        _clear_voice_env(self)

    def test_selectors_total_with_default_flags(self):
        # No patching of deps → real find_spec on the real (bare) environment.
        # With defaults the flags are off, so all selectors must return None/
        # False with zero exceptions regardless of what's installed.
        with mock.patch.object(config, "VOICE_MODE", "turn_based"), \
                mock.patch.object(config, "WAKE_WORD_AUTOSTART", False):
            self.assertFalse(vp.realtime_enabled())
            self.assertIsNone(vp.make_realtime_session())
            self.assertFalse(vp.wake_word_autostart_enabled())
            self.assertIsNone(vp.wake_detector_or_none())
            # deps_status must always return a dict, never raise.
            self.assertIsInstance(vp.deps_status(), dict)


if __name__ == "__main__":
    unittest.main()
