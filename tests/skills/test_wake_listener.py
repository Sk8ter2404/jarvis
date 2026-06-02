"""Unit tests for skills/wake_listener.py — the background wake-word listener.

The skill runs a ``core.wake_word.WakeWordDetector`` on its own thread and gates
each wake event on a voice-biometric speaker-ID check before nudging the main
loop awake via ``bobert_companion.proactive_announce``.

CI contract (tools/run_tests_ci_sim.py): ``numpy`` IS on the reduced runner (real
here, used for synthetic audio); ``sounddevice`` / ``openwakeword`` / ``torch``
are NOT. wake_listener imports numpy at module scope and lazy-imports
``core.wake_word`` / ``core.voice_id`` inside helpers, so the skill loads cleanly
under the harness. Every test that needs the detector or voice_id injects a FAKE
(scoped per-test, restored afterwards) — no real detector is ever started, no mic
opened, no model loaded, no real thread run.

Isolation:
  * The skill is loaded fresh per-test via ``load_skill_isolated`` (the harness
    neuters ``Thread.start`` so no daemon thread escapes).
  * Module-level globals the helpers mutate (``_detector``, ``GUEST_MODE_ENABLED``,
    ``VOICE_BIOMETRIC_*``, the voice-buffer state, ``WAKE_WORD_*``) live on the
    freshly-exec'd module object, so they reset between tests automatically; the
    few tests that flip them restore explicitly via addCleanup for safety.
  * ``bobert_companion`` is never imported for real — a tiny fake module is
    injected only where a helper looks it up via ``sys.modules``.
  * ``time.sleep`` is patched out wherever a helper would otherwise block.

stdlib ``unittest`` + ``unittest.mock`` only. No personal data; no real secrets.
"""
from __future__ import annotations

import contextlib
import queue
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


# ─── scoped fake-module injection (mirrors tests/test_voice_id.py) ───────────
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring prior state
    (including absence) on exit. Dotted leaves are also set on the parent
    package. ``None`` value → sys.modules[name]=None so ``import name`` raises;
    for a dotted ``None`` the leaf is ALSO removed from the already-imported
    parent package so ``from pkg import leaf`` (which resolves via the parent's
    attribute, not sys.modules) fails too — without this, a real submodule that
    earlier code already bound as an attribute would leak through the block."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                saved_attr.append(
                    (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                if obj is None:
                    with contextlib.suppress(AttributeError):
                        delattr(parent, leaf)
                else:
                    setattr(parent, leaf, obj)
        if obj is None:
            sys.modules[name] = None
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                with contextlib.suppress(AttributeError):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            if name in missing:
                sys.modules.pop(name, None)
            elif saved_mod.get(name, _SENTINEL) is not _SENTINEL:
                sys.modules[name] = saved_mod[name]


# ─── fakes ───────────────────────────────────────────────────────────────────
class FakeDetector:
    """Stand-in for WakeWordDetector with just the surface wake_listener uses."""

    def __init__(self, running=False, engine="openwakeword", sample_rate=16000,
                 start_ok=True):
        self._running = running
        self._engine = engine
        self.sample_rate = sample_rate
        self._start_ok = start_ok
        self.taps: list = []
        self.started = False
        self.stopped = False
        self._last_event_ts = 0.0
        self.wake_words = ["hey jarvis", "jarvis"]
        self.threshold = 0.5

    def is_running(self):
        return self._running

    def start(self):
        self.started = True
        self._running = bool(self._start_ok)
        return self._start_ok

    def stop(self):
        self.stopped = True
        self._running = False

    def add_tap(self, q):
        self.taps.append(q)

    def remove_tap(self, q):
        with contextlib.suppress(ValueError):
            self.taps.remove(q)

    def status(self):
        return {
            "engine": self._engine,
            "running": self._running,
            "wake_words": self.wake_words,
            "threshold": self.threshold,
            "last_event_ts": self._last_event_ts,
        }


def make_fake_voice_id(available=True, enrolled=("alice",),
                       identify=("alice", 0.91)):
    """Fake core.voice_id module with scriptable availability / enrollment /
    identify_speaker result."""
    mod = types.ModuleType("core.voice_id")
    mod.CONFIDENCE_THRESHOLD = 0.72
    mod.is_available = lambda: available
    mod.list_enrolled = lambda: list(enrolled)

    def _identify(audio, sr):
        return identify
    mod.identify_speaker = _identify
    return mod


def load_listener():
    """Load wake_listener in isolation (threads neutered by the harness)."""
    mod, actions = load_skill_isolated("wake_listener")
    return mod, actions


# ──────────────────────────────────────────────────────────────────────
# register() wiring
# ──────────────────────────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_register_exposes_all_actions(self):
        mod, actions = load_listener()
        for name in ("wake_listener_start", "wake_listener_stop",
                     "wake_listener_status", "wake_listener_configure",
                     "guest_mode_on", "guest_mode_off",
                     "voice_gating_on", "voice_gating_off"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_register_applies_bobert_overrides(self):
        # A bobert_companion exposing VOICE_BIOMETRIC_* / GUEST_MODE should seed
        # the skill defaults at register time.
        bc = types.ModuleType("bobert_companion")
        bc.VOICE_BIOMETRIC_ENABLED = False
        bc.VOICE_BIOMETRIC_THRESHOLD = 0.66
        bc.GUEST_MODE_ENABLED = True
        with inject_modules(bobert_companion=bc):
            mod, _ = load_listener()
            mod._apply_bobert_overrides()
        self.assertFalse(mod.VOICE_BIOMETRIC_ENABLED)
        self.assertEqual(mod.VOICE_BIOMETRIC_THRESHOLD, 0.66)
        self.assertTrue(mod.GUEST_MODE_ENABLED)

    def test_autostart_skipped_when_disabled(self):
        # WAKE_WORD_AUTOSTART defaults False → register must not spawn a start.
        mod, actions = load_listener()
        with mock.patch.object(mod, "wake_listener_start") as start:
            mod.register({})
        start.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# _apply_bobert_overrides edge cases
# ──────────────────────────────────────────────────────────────────────
class ApplyBobertOverridesTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()

    def test_no_bobert_module_is_noop(self):
        with inject_modules(bobert_companion=None, __main__=None):
            # Both lookups miss → early return, defaults unchanged.
            before = self.mod.VOICE_BIOMETRIC_ENABLED
            self.mod._apply_bobert_overrides()
        self.assertEqual(self.mod.VOICE_BIOMETRIC_ENABLED, before)

    def test_non_bool_enabled_ignored(self):
        bc = types.ModuleType("bobert_companion")
        bc.VOICE_BIOMETRIC_ENABLED = "yes"  # not a bool → ignored
        self.mod.VOICE_BIOMETRIC_ENABLED = True
        with inject_modules(bobert_companion=bc):
            self.mod._apply_bobert_overrides()
        self.assertTrue(self.mod.VOICE_BIOMETRIC_ENABLED)

    def test_falls_back_to_main_module(self):
        main = types.ModuleType("__main__fake")
        main.GUEST_MODE_ENABLED = True
        with inject_modules(bobert_companion=None,
                            __main__=main):
            self.mod._apply_bobert_overrides()
        self.assertTrue(self.mod.GUEST_MODE_ENABLED)


# ──────────────────────────────────────────────────────────────────────
# _get_detector — lazy import / construction
# ──────────────────────────────────────────────────────────────────────
class GetDetectorTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))

    def test_returns_cached_detector(self):
        sentinel = FakeDetector()
        self.mod._detector = sentinel
        self.assertIs(self.mod._get_detector(), sentinel)

    def test_constructs_via_core_wake_word(self):
        fake_ww = types.ModuleType("core.wake_word")
        captured = {}

        def _ctor(**kwargs):
            captured.update(kwargs)
            return FakeDetector()
        fake_ww.WakeWordDetector = _ctor
        with inject_modules(**{"core.wake_word": fake_ww}):
            det = self.mod._get_detector()
        self.assertIsInstance(det, FakeDetector)
        self.assertEqual(captured["engine"], self.mod.WAKE_WORD_ENGINE)
        self.assertEqual(captured["wake_words"], self.mod.WAKE_WORDS)
        self.assertIs(captured["on_detect"], self.mod._on_detect)

    def test_import_failure_returns_none(self):
        with inject_modules(**{"core.wake_word": None}):
            self.assertIsNone(self.mod._get_detector())


# ──────────────────────────────────────────────────────────────────────
# wake_listener_start
# ──────────────────────────────────────────────────────────────────────
class StartTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))

    def test_start_when_detector_unavailable(self):
        with mock.patch.object(self.mod, "_get_detector", return_value=None):
            out = self.actions["wake_listener_start"]("")
        self.assertIn("unavailable", out.lower())

    def test_start_when_already_running(self):
        det = FakeDetector(running=True)
        with mock.patch.object(self.mod, "_get_detector", return_value=det):
            out = self.actions["wake_listener_start"]("")
        self.assertIn("already running", out.lower())

    def test_start_success_starts_voice_tap(self):
        det = FakeDetector(running=False, start_ok=True)
        with mock.patch.object(self.mod, "_get_detector", return_value=det), \
             mock.patch.object(self.mod, "_start_voice_tap") as svt:
            out = self.actions["wake_listener_start"]("")
        self.assertTrue(det.started)
        svt.assert_called_once_with(det)
        self.assertIn("Listening", out)

    def test_start_engine_failure_message(self):
        det = FakeDetector(running=False, start_ok=False)
        with mock.patch.object(self.mod, "_get_detector", return_value=det), \
             mock.patch.object(self.mod, "_start_voice_tap") as svt:
            out = self.actions["wake_listener_start"]("")
        self.assertIn("failed to start", out.lower())
        svt.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# wake_listener_stop
# ──────────────────────────────────────────────────────────────────────
class StopTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))

    def test_stop_when_not_initialised(self):
        self.mod._detector = None
        out = self.actions["wake_listener_stop"]("")
        self.assertIn("not running", out.lower())

    def test_stop_when_not_running(self):
        self.mod._detector = FakeDetector(running=False)
        out = self.actions["wake_listener_stop"]("")
        self.assertIn("not running", out.lower())

    def test_stop_running_detector(self):
        det = FakeDetector(running=True)
        self.mod._detector = det
        with mock.patch.object(self.mod, "_stop_voice_tap") as svt:
            out = self.actions["wake_listener_stop"]("")
        self.assertTrue(det.stopped)
        svt.assert_called_once()
        self.assertIsNone(self.mod._detector)  # dropped so next start rebuilds
        self.assertIn("stopped", out.lower())


# ──────────────────────────────────────────────────────────────────────
# wake_listener_status
# ──────────────────────────────────────────────────────────────────────
class StatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))

    def test_status_not_initialised(self):
        self.mod._detector = None
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(enrolled=())}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("not initialised", out.lower())

    def test_status_active_detector(self):
        det = FakeDetector(running=True)
        det._last_event_ts = 0.0
        self.mod._detector = det
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(
                 enrolled=("alice", "bob"))}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("active", out.lower())
        self.assertIn("2 enrolled", out)
        self.assertIn("last hit never", out.lower())

    def test_status_idle_detector(self):
        det = FakeDetector(running=False)
        self.mod._detector = det
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(enrolled=())}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("idle", out.lower())

    def test_status_guest_gate_label(self):
        self.mod._detector = None
        self.mod.GUEST_MODE_ENABLED = True
        self.addCleanup(lambda: setattr(self.mod, "GUEST_MODE_ENABLED", False))
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(enrolled=())}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("gate guest", out.lower())

    def test_status_gate_off_label(self):
        self.mod._detector = None
        self.mod.VOICE_BIOMETRIC_ENABLED = False
        self.addCleanup(
            lambda: setattr(self.mod, "VOICE_BIOMETRIC_ENABLED", True))
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(enrolled=())}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("gate off", out.lower())

    def test_status_voice_id_import_failure_tolerated(self):
        # enrolled count probe fails → falls back to 0 enrolled, no raise.
        self.mod._detector = None
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": None}):
            out = self.actions["wake_listener_status"]("")
        self.assertIn("0 enrolled", out)


# ──────────────────────────────────────────────────────────────────────
# wake_listener_configure
# ──────────────────────────────────────────────────────────────────────
class ConfigureTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_listener()
        # Snapshot the tunables we mutate and restore them after each test.
        self._snap = {
            k: getattr(self.mod, k) for k in (
                "WAKE_WORD_ENGINE", "WAKE_WORD_THRESHOLD", "WAKE_WORDS",
                "WAKE_WORD_DEVICE", "WAKE_WORD_USE_SILERO_VAD",
                "VOICE_BIOMETRIC_ENABLED", "VOICE_BIOMETRIC_THRESHOLD",
                "GUEST_MODE_ENABLED")
        }
        self.addCleanup(self._restore)
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))

    def _restore(self):
        for k, v in self._snap.items():
            setattr(self.mod, k, v)

    def cfg(self, arg):
        return self.actions["wake_listener_configure"](arg)

    def test_usage_when_no_equals(self):
        self.assertIn("Usage", self.cfg("engine"))

    def test_set_engine(self):
        out = self.cfg("engine=porcupine")
        self.assertEqual(self.mod.WAKE_WORD_ENGINE, "porcupine")
        self.assertIn("set to", out.lower())

    def test_engine_empty_defaults_to_openwakeword(self):
        self.cfg("engine=")
        self.assertEqual(self.mod.WAKE_WORD_ENGINE, "openwakeword")

    def test_set_threshold_clamped(self):
        self.cfg("threshold=1.7")
        self.assertEqual(self.mod.WAKE_WORD_THRESHOLD, 1.0)
        self.cfg("threshold=-3")
        self.assertEqual(self.mod.WAKE_WORD_THRESHOLD, 0.0)

    def test_threshold_invalid(self):
        out = self.cfg("threshold=abc")
        self.assertIn("must be a number", out.lower())

    def test_set_words(self):
        self.cfg("words=computer, hey buddy ,")
        self.assertEqual(self.mod.WAKE_WORDS, ["computer", "hey buddy"])

    def test_words_empty_falls_back_to_default(self):
        self.cfg("words=,,")
        self.assertEqual(self.mod.WAKE_WORDS, ["hey jarvis", "jarvis"])

    def test_set_device_int(self):
        self.cfg("device=3")
        self.assertEqual(self.mod.WAKE_WORD_DEVICE, 3)

    def test_set_device_auto_is_none(self):
        self.mod.WAKE_WORD_DEVICE = 5
        self.cfg("device=auto")
        self.assertIsNone(self.mod.WAKE_WORD_DEVICE)

    def test_set_silero_bool(self):
        self.cfg("silero=on")
        self.assertTrue(self.mod.WAKE_WORD_USE_SILERO_VAD)
        self.cfg("silero=off")
        self.assertFalse(self.mod.WAKE_WORD_USE_SILERO_VAD)

    def test_set_voice_gate(self):
        self.cfg("voice_gate=off")
        self.assertFalse(self.mod.VOICE_BIOMETRIC_ENABLED)
        self.cfg("voice_gate=yes")
        self.assertTrue(self.mod.VOICE_BIOMETRIC_ENABLED)

    def test_set_voice_threshold_clamped(self):
        self.cfg("voice_threshold=2")
        self.assertEqual(self.mod.VOICE_BIOMETRIC_THRESHOLD, 1.0)

    def test_voice_threshold_invalid(self):
        out = self.cfg("voice_threshold=xx")
        self.assertIn("must be a number", out.lower())

    def test_set_guest_mode(self):
        self.cfg("guest_mode=true")
        self.assertTrue(self.mod.GUEST_MODE_ENABLED)

    def test_unknown_key(self):
        out = self.cfg("frobnicate=1")
        self.assertIn("unknown", out.lower())

    def test_restart_needed_stops_existing_detector(self):
        det = FakeDetector(running=True)
        self.mod._detector = det
        with mock.patch.object(self.mod, "_stop_voice_tap") as svt:
            out = self.cfg("engine=porcupine")
        self.assertTrue(det.stopped)
        svt.assert_called_once()
        self.assertIsNone(self.mod._detector)
        self.assertIn("restart", out.lower())

    def test_gating_change_does_not_restart(self):
        det = FakeDetector(running=True)
        self.mod._detector = det
        out = self.cfg("guest_mode=on")
        # guest_mode is not a restart-needed key → detector untouched.
        self.assertFalse(det.stopped)
        self.assertIs(self.mod._detector, det)
        self.assertNotIn("restart", out.lower())

    def test_restart_stop_exceptions_swallowed(self):
        det = FakeDetector(running=True)

        def _boom():
            raise RuntimeError("stop boom")
        det.stop = _boom
        self.mod._detector = det
        with mock.patch.object(self.mod, "_stop_voice_tap",
                               side_effect=RuntimeError("tap boom")):
            out = self.cfg("engine=porcupine")  # both exceptions swallowed
        self.assertIsNone(self.mod._detector)
        self.assertIn("restart", out.lower())


# ──────────────────────────────────────────────────────────────────────
# guest / voice-gating toggles
# ──────────────────────────────────────────────────────────────────────
class ToggleTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "GUEST_MODE_ENABLED", False))
        self.addCleanup(
            lambda: setattr(self.mod, "VOICE_BIOMETRIC_ENABLED", True))

    def test_guest_mode_on_off(self):
        self.actions["guest_mode_on"]("")
        self.assertTrue(self.mod.GUEST_MODE_ENABLED)
        self.actions["guest_mode_off"]("")
        self.assertFalse(self.mod.GUEST_MODE_ENABLED)

    def test_voice_gating_on_off(self):
        self.actions["voice_gating_off"]("")
        self.assertFalse(self.mod.VOICE_BIOMETRIC_ENABLED)
        self.actions["voice_gating_on"]("")
        self.assertTrue(self.mod.VOICE_BIOMETRIC_ENABLED)


# ──────────────────────────────────────────────────────────────────────
# _gate_is_strict — the gating precondition logic
# ──────────────────────────────────────────────────────────────────────
class GateIsStrictTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "GUEST_MODE_ENABLED", False))
        self.addCleanup(
            lambda: setattr(self.mod, "VOICE_BIOMETRIC_ENABLED", True))

    def _run(self, **vid_kwargs):
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": make_fake_voice_id(**vid_kwargs)}):
            return self.mod._gate_is_strict()

    def test_not_strict_when_biometric_disabled(self):
        self.mod.VOICE_BIOMETRIC_ENABLED = False
        self.assertFalse(self._run())

    def test_not_strict_in_guest_mode(self):
        self.mod.GUEST_MODE_ENABLED = True
        self.assertFalse(self._run())

    def test_not_strict_when_voice_id_unavailable(self):
        self.assertFalse(self._run(available=False))

    def test_not_strict_when_nobody_enrolled(self):
        self.assertFalse(self._run(enrolled=()))

    def test_strict_when_all_conditions_met(self):
        self.assertTrue(self._run(available=True, enrolled=("alice",)))

    def test_not_strict_when_voice_id_import_fails(self):
        self.mod.VOICE_BIOMETRIC_ENABLED = True
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": None}):
            self.assertFalse(self.mod._gate_is_strict())

    def test_not_strict_when_availability_probe_raises(self):
        vid = make_fake_voice_id()

        def _boom():
            raise RuntimeError("probe boom")
        vid.is_available = _boom
        with mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": vid}):
            self.assertFalse(self.mod._gate_is_strict())


# ──────────────────────────────────────────────────────────────────────
# _on_detect → dispatches gate/announce to a thread
# ──────────────────────────────────────────────────────────────────────
class OnDetectTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()

    def test_on_detect_spawns_gate_thread(self):
        # Capture the Thread target instead of running it (harness already
        # neuters start, but we assert the wiring explicitly).
        created = {}

        class _T:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                created["target"] = target
                created["args"] = args

            def start(self):
                created["started"] = True
        with mock.patch.object(self.mod.threading, "Thread", _T):
            self.mod._on_detect({"phrase": "jarvis", "score": 0.9})
        self.assertIs(created["target"], self.mod._gate_and_announce)
        self.assertEqual(created["args"][0]["phrase"], "jarvis")
        self.assertTrue(created["started"])

    def test_on_detect_missing_fields_default(self):
        # No phrase/score keys → uses defaults, still spawns the worker.
        with mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._on_detect({})
        T.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# _gate_and_announce — the wake-trigger handling / gating + announce
# ──────────────────────────────────────────────────────────────────────
class GateAndAnnounceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(lambda: sys.modules.pop("bobert_companion", None))

    def test_permissive_gate_announces_when_asleep(self):
        # Gate not strict → skip voice ID; bobert asleep → announce fires.
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_called_once()
        # The announce text + source kwarg.
        args, kwargs = announce.call_args
        self.assertEqual(kwargs.get("source"), "wake-listener")

    def test_no_announce_when_awake(self):
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [False]   # awake → no proactive nudge
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_not_called()

    def test_strict_gate_rejects_when_no_speaker_match(self):
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=True), \
             mock.patch.object(self.mod, "_identify_recent_speaker",
                               return_value=(None, 0.3)), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_not_called()  # rejected → no announce

    def test_strict_gate_accepts_on_match_then_announces(self):
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=True), \
             mock.patch.object(self.mod, "_identify_recent_speaker",
                               return_value=("alice", 0.95)), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_called_once()

    def test_sleep_mode_non_list_treated_as_awake(self):
        # Legacy bug guard: _sleep_mode read as _sleep_mode[0]; a bare truthy
        # value (not a list) must be treated as awake (asleep=False).
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = True   # not a list/tuple
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_not_called()

    def test_empty_sleep_list_treated_as_awake(self):
        announce = mock.MagicMock()
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = []   # empty → asleep False
        bc.proactive_announce = announce
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})
        announce.assert_not_called()

    def test_announce_exception_is_swallowed(self):
        def _boom(*a, **k):
            raise RuntimeError("announce boom")
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc.proactive_announce = _boom
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})  # no raise

    def test_no_bobert_module_is_safe(self):
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=None, __main__=None):
            self.mod._gate_and_announce({"phrase": "jarvis"})  # no raise

    def test_announce_skipped_when_callable_missing(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc.proactive_announce = "not-callable"
        with mock.patch.object(self.mod, "_gate_is_strict", return_value=False), \
             inject_modules(bobert_companion=bc):
            self.mod._gate_and_announce({"phrase": "jarvis"})  # no raise


# ──────────────────────────────────────────────────────────────────────
# Voice-tap lifecycle: _start_voice_tap / _drain_voice_tap / _stop_voice_tap
# ──────────────────────────────────────────────────────────────────────
class VoiceTapTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        # Make sure no tap state leaks between tests.
        self.mod._voice_audio_tap = None
        self.mod._voice_buffer_thread = None
        self.mod._voice_buffer_stop = None
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.clear()

    def test_start_voice_tap_attaches_and_spawns(self):
        det = FakeDetector()
        # Harness neuters Thread.start, so the drain thread won't actually run.
        self.mod._start_voice_tap(det)
        self.assertIsNotNone(self.mod._voice_audio_tap)
        self.assertIn(self.mod._voice_audio_tap, det.taps)
        self.assertGreaterEqual(self.mod._voice_buffer_capacity_frames, 8)

    def test_start_voice_tap_idempotent(self):
        det = FakeDetector()
        self.mod._start_voice_tap(det)
        first = self.mod._voice_audio_tap
        self.mod._start_voice_tap(det)  # second call is a no-op
        self.assertIs(self.mod._voice_audio_tap, first)

    def test_start_voice_tap_no_add_tap_method(self):
        # A detector lacking add_tap → bail out cleanly.
        class _NoTap:
            pass
        self.mod._start_voice_tap(_NoTap())
        self.assertIsNone(self.mod._voice_audio_tap)

    def test_start_voice_tap_add_tap_failure_resets_state(self):
        det = FakeDetector()

        def _boom(q):
            raise RuntimeError("add_tap boom")
        det.add_tap = _boom
        self.mod._start_voice_tap(det)
        self.assertIsNone(self.mod._voice_audio_tap)
        self.assertIsNone(self.mod._voice_buffer_stop)

    def test_stop_voice_tap_detaches(self):
        det = FakeDetector()
        self.mod._detector = det
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))
        self.mod._start_voice_tap(det)
        tap = self.mod._voice_audio_tap
        self.mod._stop_voice_tap()
        self.assertIsNone(self.mod._voice_audio_tap)
        self.assertNotIn(tap, det.taps)

    def test_stop_voice_tap_safe_when_not_started(self):
        self.mod._stop_voice_tap()  # nothing attached → no raise
        self.assertIsNone(self.mod._voice_audio_tap)

    def test_drain_voice_tap_moves_frames_into_buffer(self):
        # Drive the drain loop body for exactly a couple iterations by scripting
        # _voice_buffer_stop.is_set() to go False, False, True.
        import threading
        ev = threading.Event()
        self.mod._voice_buffer_stop = ev
        tapq = queue.Queue()
        self.mod._voice_audio_tap = tapq
        self.mod._voice_buffer_capacity_frames = 100
        frame = np.ones(1280, dtype=np.float32)
        tapq.put(frame)

        calls = {"n": 0}

        def _is_set():
            calls["n"] += 1
            return calls["n"] > 2   # run the body twice then stop
        with mock.patch.object(ev, "is_set", side_effect=_is_set):
            self.mod._drain_voice_tap()
        with self.mod._voice_buffer_lock:
            self.assertEqual(len(self.mod._voice_audio_buffer), 1)

    def test_drain_voice_tap_respects_capacity(self):
        import threading
        ev = threading.Event()
        self.mod._voice_buffer_stop = ev
        tapq = queue.Queue()
        self.mod._voice_audio_tap = tapq
        self.mod._voice_buffer_capacity_frames = 2
        for _ in range(5):
            tapq.put(np.ones(10, dtype=np.float32))

        calls = {"n": 0}

        def _is_set():
            calls["n"] += 1
            return calls["n"] > 5
        with mock.patch.object(ev, "is_set", side_effect=_is_set):
            self.mod._drain_voice_tap()
        with self.mod._voice_buffer_lock:
            self.assertLessEqual(len(self.mod._voice_audio_buffer), 2)

    def test_drain_voice_tap_empty_queue_continues(self):
        import threading
        ev = threading.Event()
        self.mod._voice_buffer_stop = ev
        self.mod._voice_audio_tap = queue.Queue()  # empty
        self.mod._voice_buffer_capacity_frames = 10
        calls = {"n": 0}

        def _is_set():
            calls["n"] += 1
            return calls["n"] > 1   # one body pass (hits queue.Empty), then stop
        with mock.patch.object(ev, "is_set", side_effect=_is_set):
            self.mod._drain_voice_tap()  # queue.Empty caught → continue → no raise

    def test_drain_voice_tap_generic_get_exception_continues(self):
        # A non-Empty exception from tap.get is caught by the broad except.
        import threading
        ev = threading.Event()
        self.mod._voice_buffer_stop = ev

        class _BadQueue:
            def get(self, timeout=None):
                raise RuntimeError("queue boom")
        self.mod._voice_audio_tap = _BadQueue()
        self.mod._voice_buffer_capacity_frames = 10
        calls = {"n": 0}

        def _is_set():
            calls["n"] += 1
            return calls["n"] > 1
        with mock.patch.object(ev, "is_set", side_effect=_is_set):
            self.mod._drain_voice_tap()  # RuntimeError caught → continue → stop

    def test_stop_voice_tap_remove_tap_exception_swallowed(self):
        det = FakeDetector()

        def _boom(q):
            raise RuntimeError("remove boom")
        det.remove_tap = _boom
        self.mod._detector = det
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))
        self.mod._start_voice_tap(det)
        self.mod._stop_voice_tap()  # remove_tap raises but is swallowed
        self.assertIsNone(self.mod._voice_audio_tap)

    def test_drain_voice_tap_no_tap_sleeps_and_continues(self):
        import threading
        ev = threading.Event()
        self.mod._voice_buffer_stop = ev
        self.mod._voice_audio_tap = None  # tap is None branch
        calls = {"n": 0}

        def _is_set():
            calls["n"] += 1
            return calls["n"] > 1
        with mock.patch.object(ev, "is_set", side_effect=_is_set), \
             mock.patch.object(self.mod.time, "sleep") as slp:
            self.mod._drain_voice_tap()
        slp.assert_called()  # slept because tap was None


# ──────────────────────────────────────────────────────────────────────
# _snapshot_voice_audio
# ──────────────────────────────────────────────────────────────────────
class SnapshotVoiceAudioTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(self._cleanup)
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.clear()

    def _cleanup(self):
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.clear()

    def test_empty_buffer_returns_none(self):
        self.assertIsNone(self.mod._snapshot_voice_audio(1.0, 16000))

    def test_returns_recent_window(self):
        # 3 frames of 16000 samples each; ask for 1s @16k → last 16000 samples.
        with self.mod._voice_buffer_lock:
            for v in (1.0, 2.0, 3.0):
                self.mod._voice_audio_buffer.append(
                    np.full(16000, v, dtype=np.float32))
        out = self.mod._snapshot_voice_audio(1.0, 16000)
        self.assertEqual(out.size, 16000)
        self.assertTrue(np.allclose(out, 3.0))  # most-recent frame

    def test_returns_all_when_buffer_shorter_than_window(self):
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.append(np.ones(8000, dtype=np.float32))
        out = self.mod._snapshot_voice_audio(10.0, 16000)  # want 160k, have 8k
        self.assertEqual(out.size, 8000)

    def test_concat_failure_returns_none(self):
        # A ragged/bad entry makes np.concatenate raise → None.
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.append(np.ones(10, dtype=np.float32))
        with mock.patch.object(self.mod.np, "concatenate",
                               side_effect=ValueError("ragged")):
            self.assertIsNone(self.mod._snapshot_voice_audio(1.0, 16000))

    def test_all_zero_size_frames_returns_none(self):
        with self.mod._voice_buffer_lock:
            self.mod._voice_audio_buffer.append(np.array([], dtype=np.float32))
        self.assertIsNone(self.mod._snapshot_voice_audio(1.0, 16000))


# ──────────────────────────────────────────────────────────────────────
# _identify_recent_speaker — voice-ID gating call
# ──────────────────────────────────────────────────────────────────────
class IdentifyRecentSpeakerTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()
        self.addCleanup(lambda: setattr(self.mod, "_detector", None))
        # Never actually sleep for the post-wake settle.
        p = mock.patch.object(self.mod.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_returns_none_when_no_audio(self):
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               return_value=None):
            spk, score = self.mod._identify_recent_speaker()
        self.assertIsNone(spk)
        self.assertEqual(score, 0.0)

    def test_returns_speaker_and_score(self):
        audio = np.ones(16000, dtype=np.float32)
        vid = make_fake_voice_id(identify=("alice", 0.93))
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               return_value=audio), \
             mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": vid}):
            spk, score = self.mod._identify_recent_speaker()
        self.assertEqual(spk, "alice")
        self.assertAlmostEqual(score, 0.93)

    def test_swaps_and_restores_confidence_threshold(self):
        audio = np.ones(16000, dtype=np.float32)
        vid = make_fake_voice_id(identify=("bob", 0.8))
        vid.CONFIDENCE_THRESHOLD = 0.72
        seen = {}

        def _identify(a, sr):
            seen["threshold_during"] = vid.CONFIDENCE_THRESHOLD
            return ("bob", 0.8)
        vid.identify_speaker = _identify
        self.mod.VOICE_BIOMETRIC_THRESHOLD = 0.61
        self.addCleanup(
            lambda: setattr(self.mod, "VOICE_BIOMETRIC_THRESHOLD", 0.72))
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               return_value=audio), \
             mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": vid}):
            self.mod._identify_recent_speaker()
        # During the call the module threshold was applied...
        self.assertEqual(seen["threshold_during"], 0.61)
        # ...and the original was restored afterwards.
        self.assertEqual(vid.CONFIDENCE_THRESHOLD, 0.72)

    def test_voice_id_import_failure_returns_none(self):
        audio = np.ones(16000, dtype=np.float32)
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               return_value=audio), \
             mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": None}):
            spk, score = self.mod._identify_recent_speaker()
        self.assertIsNone(spk)
        self.assertEqual(score, 0.0)

    def test_identify_speaker_exception_restores_threshold(self):
        audio = np.ones(16000, dtype=np.float32)
        vid = make_fake_voice_id()
        vid.CONFIDENCE_THRESHOLD = 0.72

        def _boom(a, sr):
            raise RuntimeError("embed boom")
        vid.identify_speaker = _boom
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               return_value=audio), \
             mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": vid}):
            spk, score = self.mod._identify_recent_speaker()
        self.assertIsNone(spk)
        self.assertEqual(score, 0.0)
        self.assertEqual(vid.CONFIDENCE_THRESHOLD, 0.72)  # restored despite raise

    def test_uses_detector_sample_rate(self):
        audio = np.ones(8000, dtype=np.float32)
        self.mod._detector = FakeDetector(sample_rate=8000)
        captured = {}
        vid = make_fake_voice_id()

        def _identify(a, sr):
            captured["sr"] = sr
            return ("alice", 0.9)
        vid.identify_speaker = _identify

        def _snap(window, sr):
            captured["snap_sr"] = sr
            return audio
        with mock.patch.object(self.mod, "_snapshot_voice_audio",
                               side_effect=_snap), \
             mock.patch.object(self.mod, "_ensure_core_on_path"), \
             inject_modules(**{"core.voice_id": vid}):
            self.mod._identify_recent_speaker()
        self.assertEqual(captured["snap_sr"], 8000)
        self.assertEqual(captured["sr"], 8000)


# ──────────────────────────────────────────────────────────────────────
# _ensure_core_on_path
# ──────────────────────────────────────────────────────────────────────
class EnsureCoreOnPathTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_listener()

    def test_inserts_project_dir(self):
        saved = list(sys.path)
        self.addCleanup(lambda: sys.path.__init__(saved))
        # Remove it first so we exercise the insert branch.
        with contextlib.suppress(ValueError):
            sys.path.remove(self.mod._PROJECT_DIR)
        self.mod._ensure_core_on_path()
        self.assertIn(self.mod._PROJECT_DIR, sys.path)

    def test_noop_when_already_present(self):
        if self.mod._PROJECT_DIR not in sys.path:
            sys.path.insert(0, self.mod._PROJECT_DIR)
        n_before = sys.path.count(self.mod._PROJECT_DIR)
        self.mod._ensure_core_on_path()
        self.assertEqual(sys.path.count(self.mod._PROJECT_DIR), n_before)


# ──────────────────────────────────────────────────────────────────────
# register() autostart branch (mic-gating)
# ──────────────────────────────────────────────────────────────────────
class AutostartTests(unittest.TestCase):
    def test_autostart_runs_when_enabled_and_mic_on(self):
        mod, _ = load_listener()
        mod.WAKE_WORD_AUTOSTART = True
        self.addCleanup(lambda: setattr(mod, "WAKE_WORD_AUTOSTART", False))
        bc = types.ModuleType("bobert_companion")
        bc._mic_input_disabled = lambda: False
        # Capture the autostart thread target without running it.
        created = {}

        class _T:
            def __init__(self, target=None, name=None, daemon=None):
                created["target"] = target

            def start(self):
                created["started"] = True
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(mod.threading, "Thread", _T):
            mod.register({})
        self.assertIn("target", created)
        self.assertTrue(created.get("started"))

    def test_autostart_bg_body_calls_start(self):
        # Capture the _bg target the autostart spawns, then run it directly with
        # sleep/start patched so the real body (sleep → wake_listener_start) is
        # exercised without a real thread or delay.
        mod, _ = load_listener()
        mod.WAKE_WORD_AUTOSTART = True
        self.addCleanup(lambda: setattr(mod, "WAKE_WORD_AUTOSTART", False))
        bc = types.ModuleType("bobert_companion")
        bc._mic_input_disabled = lambda: False
        created = {}

        class _T:
            def __init__(self, target=None, name=None, daemon=None):
                created["target"] = target

            def start(self):
                pass
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(mod.threading, "Thread", _T):
            mod.register({})
        with mock.patch.object(mod.time, "sleep") as slp, \
             mock.patch.object(mod, "wake_listener_start") as start:
            created["target"]()  # run the _bg body
        slp.assert_called_once_with(2.0)
        start.assert_called_once_with("")

    def test_autostart_bg_body_swallows_exception(self):
        mod, _ = load_listener()
        mod.WAKE_WORD_AUTOSTART = True
        self.addCleanup(lambda: setattr(mod, "WAKE_WORD_AUTOSTART", False))
        bc = types.ModuleType("bobert_companion")
        bc._mic_input_disabled = lambda: False
        created = {}

        class _T:
            def __init__(self, target=None, name=None, daemon=None):
                created["target"] = target

            def start(self):
                pass
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(mod.threading, "Thread", _T):
            mod.register({})
        with mock.patch.object(mod.time, "sleep"), \
             mock.patch.object(mod, "wake_listener_start",
                               side_effect=RuntimeError("start boom")):
            created["target"]()  # exception inside _bg is caught → no raise

    def test_autostart_skipped_when_mic_disabled(self):
        mod, _ = load_listener()
        mod.WAKE_WORD_AUTOSTART = True
        self.addCleanup(lambda: setattr(mod, "WAKE_WORD_AUTOSTART", False))
        bc = types.ModuleType("bobert_companion")
        bc._mic_input_disabled = lambda: True   # mic hard-off → no autostart
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(mod.threading, "Thread") as T:
            mod.register({})
        T.assert_not_called()

    def test_autostart_skipped_when_engine_off(self):
        mod, _ = load_listener()
        mod.WAKE_WORD_AUTOSTART = True
        mod.WAKE_WORD_ENGINE = "off"
        self.addCleanup(lambda: setattr(mod, "WAKE_WORD_AUTOSTART", False))
        self.addCleanup(
            lambda: setattr(mod, "WAKE_WORD_ENGINE", "openwakeword"))
        with mock.patch.object(mod.threading, "Thread") as T:
            mod.register({})
        T.assert_not_called()


if __name__ == "__main__":
    unittest.main()
