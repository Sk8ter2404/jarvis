"""Logic tests for skills/wellness.py.

Targets the focus-block presence tracker and its gating:
  • _user_present — composite signal (face_tracker OR workshop OR recent input).
  • _gate_reasons — sleep/standby, on-a-call, Bambu-print-active suppressors.
  • _poll_once — the block-start / break-reset / threshold / snooze / gate
    state machine that decides whether a nudge fires (time + presence controlled,
    _enqueue_speech patched so no real speech is queued).
  • _fmt_duration and _pick_nudge_line.
  • the wellness_status action's three branches (no block / running / snoozed).

Every presence source and gate is mocked, so detection is deterministic and no
hardware/OS calls happen.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _LoopBreak(BaseException):
    """Break a `while True` loop from a stubbed sleep without being caught by
    the loop's own `except Exception`."""


class WellnessPresenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")

    def test_user_present_via_face_tracker(self):
        with mock.patch.object(self.mod, "_face_tracker_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_workshop_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_recent_input", return_value=False):
            self.assertTrue(self.mod._user_present())

    def test_user_present_via_workshop(self):
        with mock.patch.object(self.mod, "_face_tracker_at_desk", return_value=None), \
             mock.patch.object(self.mod, "_workshop_mode_active", return_value=True), \
             mock.patch.object(self.mod, "_recent_input", return_value=False):
            self.assertTrue(self.mod._user_present())

    def test_user_present_via_recent_input(self):
        with mock.patch.object(self.mod, "_face_tracker_at_desk", return_value=None), \
             mock.patch.object(self.mod, "_workshop_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_recent_input", return_value=True):
            self.assertTrue(self.mod._user_present())

    def test_user_absent_when_all_signals_negative(self):
        with mock.patch.object(self.mod, "_face_tracker_at_desk", return_value=False), \
             mock.patch.object(self.mod, "_workshop_mode_active", return_value=False), \
             mock.patch.object(self.mod, "_recent_input", return_value=False):
            self.assertFalse(self.mod._user_present())

    def test_recent_input_uses_idle_window(self):
        with mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=10.0):
            self.assertTrue(self.mod._recent_input())
        with mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.mod.RECENT_INPUT_WINDOW + 1):
            self.assertFalse(self.mod._recent_input())


class WellnessGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")

    def test_no_gates_when_all_clear(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_bambu_print_active", return_value=False):
            self.assertEqual(self.mod._gate_reasons(), [])

    def test_gates_collect_all_active_reasons(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True), \
             mock.patch.object(self.mod, "_bambu_print_active", return_value=True):
            reasons = self.mod._gate_reasons()
        self.assertIn("sleep mode", reasons)
        self.assertIn("on a call", reasons)
        self.assertIn("Bambu print active", reasons)

    def test_bambu_active_reads_running_state(self):
        import sys
        fake_bambu = mock.MagicMock()
        fake_bambu._state_lock = None
        fake_bambu._state = {"gcode_state": "RUNNING"}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake_bambu}):
            self.assertTrue(self.mod._bambu_print_active())
        fake_bambu._state = {"gcode_state": "IDLE"}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake_bambu}):
            self.assertFalse(self.mod._bambu_print_active())


class WellnessPollStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        # Reset block-tracking state.
        self.mod._block_started_at[0] = 0.0
        self.mod._last_presence_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0

    def test_block_starts_on_presence(self):
        with mock.patch.object(self.mod, "_user_present", return_value=True):
            self.mod._poll_once()
        self.assertGreater(self.mod._block_started_at[0], 0.0)

    def test_block_resets_after_long_absence(self):
        now = time.time()
        # Seed an in-progress block whose last presence was long ago.
        self.mod._block_started_at[0] = now - 1000
        self.mod._last_presence_at[0] = now - (self.mod.BREAK_RESET_SECONDS + 10)
        with mock.patch.object(self.mod, "_user_present", return_value=False):
            self.mod._poll_once()
        self.assertEqual(self.mod._block_started_at[0], 0.0)

    def test_no_nudge_before_threshold(self):
        # Block just under 90 min → no nudge fired.
        self.mod._block_started_at[0] = time.time() - (self.mod.FOCUS_BLOCK_SECONDS - 60)
        self.mod._last_presence_at[0] = time.time()
        with mock.patch.object(self.mod, "_user_present", return_value=True), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_nudge_fires_after_threshold(self):
        self.mod._block_started_at[0] = time.time() - (self.mod.FOCUS_BLOCK_SECONDS + 60)
        self.mod._last_presence_at[0] = time.time()
        with mock.patch.object(self.mod, "_user_present", return_value=True), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_called_once()
        spoken = enq.call_args[0][0]
        self.assertIn(spoken, self.mod.NUDGE_LINES)
        # Firing stamps the snooze clock.
        self.assertGreater(self.mod._last_nudge_at[0], 0.0)

    def test_gate_blocks_nudge_even_past_threshold(self):
        self.mod._block_started_at[0] = time.time() - (self.mod.FOCUS_BLOCK_SECONDS + 60)
        self.mod._last_presence_at[0] = time.time()
        with mock.patch.object(self.mod, "_user_present", return_value=True), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=["on a call"]), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_snooze_blocks_repeat_nudge(self):
        self.mod._block_started_at[0] = time.time() - (self.mod.FOCUS_BLOCK_SECONDS + 60)
        self.mod._last_presence_at[0] = time.time()
        self.mod._last_nudge_at[0] = time.time() - 60   # nudged a minute ago
        with mock.patch.object(self.mod, "_user_present", return_value=True), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()


class WellnessFormatAndStatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self.mod._block_started_at[0] = 0.0
        self.mod._last_presence_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0

    def test_fmt_duration(self):
        f = self.mod._fmt_duration
        self.assertEqual(f(45), "45s")
        self.assertEqual(f(125), "2m 5s")
        self.assertEqual(f(7325), "2h 2m")

    def test_pick_nudge_line_in_bank(self):
        self.assertIn(self.mod._pick_nudge_line(), self.mod.NUDGE_LINES)

    def test_status_no_block(self):
        with mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=30.0), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]):
            out = self.actions["wellness_status"]("")
        self.assertIn("no active focus block", out.lower())

    def test_status_running_block(self):
        self.mod._block_started_at[0] = time.time() - 1800   # 30 min
        self.mod._last_presence_at[0] = time.time()
        with mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]):
            out = self.actions["wellness_status"]("")
        self.assertIn("focus block running", out.lower())
        self.assertIn("ready to fire", out.lower())

    def test_status_reports_active_gates(self):
        self.mod._block_started_at[0] = time.time() - 1800
        self.mod._last_presence_at[0] = time.time()
        with mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=["on a call"]):
            out = self.actions["wellness_status"]("")
        self.assertIn("on a call", out)


def _fake_ctypes(*, get_input_ok=1, tick=10_000, dw_time=4_000,
                 raise_on_import=False):
    """A ctypes stand-in covering exactly what _get_system_idle_seconds pulls:
    Structure / c_uint / sizeof / windll / byref. GetLastInputInfo populates the
    struct's dwTime (mimicking the by-pointer fill) and returns get_input_ok;
    GetTickCount returns tick. idle_ms = tick - dw_time."""
    ct = types.ModuleType("ctypes")

    class _Structure:
        def __init__(self):
            # subclasses declare _fields_; we don't need real layout.
            self.cbSize = 0
            self.dwTime = 0

    ct.Structure = _Structure
    ct.c_uint = int
    ct.sizeof = lambda _obj: 8
    ct.byref = lambda obj: obj   # pass the object straight through

    user32 = types.SimpleNamespace()
    kernel32 = types.SimpleNamespace()

    def _get_last_input_info(ref):
        if raise_on_import:
            raise RuntimeError("boom")
        ref.dwTime = dw_time
        return get_input_ok

    user32.GetLastInputInfo = _get_last_input_info
    kernel32.GetTickCount = lambda: tick
    ct.windll = types.SimpleNamespace(user32=user32, kernel32=kernel32)
    return ct


class WellnessIdleSecondsTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")

    def test_idle_seconds_success(self):
        ct = _fake_ctypes(tick=10_000, dw_time=4_000)   # 6000 ms idle
        with mock.patch.dict(sys.modules, {"ctypes": ct}):
            self.assertAlmostEqual(self.mod._get_system_idle_seconds(), 6.0)

    def test_idle_seconds_api_returns_zero_is_infinite(self):
        ct = _fake_ctypes(get_input_ok=0)
        with mock.patch.dict(sys.modules, {"ctypes": ct}):
            self.assertEqual(self.mod._get_system_idle_seconds(), float("inf"))

    def test_idle_seconds_negative_delta_is_infinite(self):
        # GetTickCount wrapped (tick < dwTime) → negative millis → inf, never a
        # bogus "user is here" tiny value.
        ct = _fake_ctypes(tick=1_000, dw_time=5_000)
        with mock.patch.dict(sys.modules, {"ctypes": ct}):
            self.assertEqual(self.mod._get_system_idle_seconds(), float("inf"))

    def test_idle_seconds_exception_is_infinite(self):
        ct = _fake_ctypes(raise_on_import=True)
        with mock.patch.dict(sys.modules, {"ctypes": ct}):
            self.assertEqual(self.mod._get_system_idle_seconds(), float("inf"))


class WellnessFaceTrackerTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self._saved = sys.modules.get("skill_face_tracker")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            sys.modules["skill_face_tracker"] = self._saved
        else:
            sys.modules.pop("skill_face_tracker", None)

    def _install(self, snap):
        ft = types.ModuleType("skill_face_tracker")
        if snap is _RAISE:
            ft._snapshot_state = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        elif snap is _NOFUNC:
            pass   # no _snapshot_state attr
        else:
            ft._snapshot_state = lambda: snap
        sys.modules["skill_face_tracker"] = ft

    def test_none_when_module_absent(self):
        sys.modules.pop("skill_face_tracker", None)
        self.assertIsNone(self.mod._face_tracker_at_desk())

    def test_none_when_no_snapshot_func(self):
        self._install(_NOFUNC)
        self.assertIsNone(self.mod._face_tracker_at_desk())

    def test_none_when_snapshot_raises(self):
        self._install(_RAISE)
        self.assertIsNone(self.mod._face_tracker_at_desk())

    def test_none_when_no_sample_yet(self):
        self._install({"last_sample_at": 0})
        self.assertIsNone(self.mod._face_tracker_at_desk())

    def test_true_when_at_monitor(self):
        self._install({"last_sample_at": 123.0, "current_monitor": "left"})
        self.assertTrue(self.mod._face_tracker_at_desk())

    def test_false_when_away(self):
        self._install({"last_sample_at": 123.0, "current_monitor": "away"})
        self.assertFalse(self.mod._face_tracker_at_desk())

    def test_none_when_monitor_unknown_value(self):
        self._install({"last_sample_at": 123.0, "current_monitor": "elsewhere"})
        self.assertIsNone(self.mod._face_tracker_at_desk())


class WellnessWorkshopModeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self._saved = sys.modules.get("skill_workshop_mode")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            sys.modules["skill_workshop_mode"] = self._saved
        else:
            sys.modules.pop("skill_workshop_mode", None)

    def test_false_when_module_absent(self):
        sys.modules.pop("skill_workshop_mode", None)
        self.assertFalse(self.mod._workshop_mode_active())

    def test_true_when_active_flag_set(self):
        wm = types.ModuleType("skill_workshop_mode")
        wm._workshop_active = [True]
        with mock.patch.dict(sys.modules, {"skill_workshop_mode": wm}):
            self.assertTrue(self.mod._workshop_mode_active())

    def test_false_when_flag_clear(self):
        wm = types.ModuleType("skill_workshop_mode")
        wm._workshop_active = [False]
        with mock.patch.dict(sys.modules, {"skill_workshop_mode": wm}):
            self.assertFalse(self.mod._workshop_mode_active())

    def test_false_when_attr_access_raises(self):
        wm = mock.MagicMock()
        type(wm)._workshop_active = mock.PropertyMock(side_effect=RuntimeError("x"))
        with mock.patch.dict(sys.modules, {"skill_workshop_mode": wm}):
            self.assertFalse(self.mod._workshop_mode_active())


class WellnessSleepStandbyGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self._saved = sys.modules.get("bobert_companion")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            sys.modules["bobert_companion"] = self._saved
        else:
            sys.modules.pop("bobert_companion", None)

    def test_false_when_bc_absent(self):
        sys.modules.pop("bobert_companion", None)
        self.assertFalse(self.mod._is_sleep_or_standby())

    def test_true_when_sleep_mode(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [True]
        bc._standby_mode = [False]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_true_when_standby_mode(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [False]
        bc._standby_mode = [True]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertTrue(self.mod._is_sleep_or_standby())

    def test_false_when_both_off(self):
        bc = types.ModuleType("bobert_companion")
        bc._sleep_mode = [False]
        bc._standby_mode = [False]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_false_when_attrs_missing(self):
        # A bobert_companion without the mode flags → AttributeError swallowed.
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertFalse(self.mod._is_sleep_or_standby())


class WellnessInCallGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")

    def _install_gw(self, titles=None, get_all_raises=False):
        gw = types.ModuleType("pygetwindow")
        if get_all_raises:
            gw.getAllWindows = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        else:
            gw.getAllWindows = lambda: [types.SimpleNamespace(title=t)
                                        for t in (titles or [])]
        return gw

    def test_false_when_pygetwindow_missing(self):
        # block_import-style: make the import raise.
        with mock.patch.dict(sys.modules, {"pygetwindow": None}), \
             mock.patch("builtins.__import__",
                        side_effect=self._blocker("pygetwindow")):
            self.assertFalse(self.mod._is_in_call())

    @staticmethod
    def _blocker(name):
        real = __import__

        def _imp(n, *a, **k):
            if n == name:
                raise ImportError(name)
            return real(n, *a, **k)
        return _imp

    def test_true_when_call_window_present(self):
        gw = self._install_gw(titles=["Slack", "Project | Microsoft Teams Meeting"])
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.mod._is_in_call())

    def test_false_when_no_call_window(self):
        gw = self._install_gw(titles=["Notepad", "Chrome"])
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertFalse(self.mod._is_in_call())

    def test_false_when_getallwindows_raises(self):
        gw = self._install_gw(get_all_raises=True)
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertFalse(self.mod._is_in_call())


class WellnessBambuGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self._saved = sys.modules.get("skill_bambu_monitor")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            sys.modules["skill_bambu_monitor"] = self._saved
        else:
            sys.modules.pop("skill_bambu_monitor", None)

    def test_false_when_module_absent(self):
        sys.modules.pop("skill_bambu_monitor", None)
        self.assertFalse(self.mod._bambu_print_active())

    def test_false_when_state_is_none(self):
        bm = types.ModuleType("skill_bambu_monitor")
        bm._state = None
        bm._state_lock = None
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": bm}):
            self.assertFalse(self.mod._bambu_print_active())

    def test_true_when_running_under_lock(self):
        import threading
        bm = types.ModuleType("skill_bambu_monitor")
        bm._state = {"gcode_state": "running"}   # lowercased → upper() == RUNNING
        bm._state_lock = threading.Lock()
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": bm}):
            self.assertTrue(self.mod._bambu_print_active())

    def test_false_when_idle(self):
        bm = types.ModuleType("skill_bambu_monitor")
        bm._state = {"gcode_state": "IDLE"}
        bm._state_lock = None
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": bm}):
            self.assertFalse(self.mod._bambu_print_active())

    def test_false_when_access_raises(self):
        bm = mock.MagicMock()
        type(bm)._state = mock.PropertyMock(side_effect=RuntimeError("boom"))
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": bm}):
            self.assertFalse(self.mod._bambu_print_active())


class WellnessEnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self.tmp = tempfile.mkdtemp(prefix="wellness_speech_")
        self.queue = os.path.join(self.tmp, "pending_speech.json")
        self.mod._SPEECH_QUEUE = self.queue
        self._saved_bc = sys.modules.get("bobert_companion")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self._saved_bc is not None:
            sys.modules["bobert_companion"] = self._saved_bc
        else:
            sys.modules.pop("bobert_companion", None)
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def test_announce_success_short_circuits(self):
        bc = mock.MagicMock()
        bc.proactive_announce.return_value = True
        with mock.patch("importlib.import_module", return_value=bc):
            self.mod._enqueue_speech("nudge via announce")
        bc.proactive_announce.assert_called_once()
        self.assertEqual(bc.proactive_announce.call_args[1]["source"], "wellness")
        self.assertFalse(os.path.exists(self.queue))

    def test_announce_returns_falsey_falls_back_to_file(self):
        # announce present but returns False → fall back to atomic write.
        bc = mock.MagicMock()
        bc.proactive_announce.return_value = False
        with mock.patch("importlib.import_module", return_value=bc):
            self.mod._enqueue_speech("nudge fallback")
        with open(self.queue, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["message"], "nudge fallback")

    def test_import_error_falls_back_and_appends(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        with mock.patch("importlib.import_module",
                        side_effect=ImportError("no bc")):
            self.mod._enqueue_speech("new one")
        with open(self.queue, encoding="utf-8") as f:
            msgs = [d["message"] for d in json.load(f)]
        self.assertEqual(msgs, ["old", "new one"])

    def test_corrupt_queue_treated_as_empty(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("not json at all")
        with mock.patch("importlib.import_module",
                        side_effect=ImportError("no bc")):
            self.mod._enqueue_speech("after garbage")
        with open(self.queue, encoding="utf-8") as f:
            self.assertEqual([d["message"] for d in json.load(f)], ["after garbage"])

    def test_write_failure_is_swallowed(self):
        with mock.patch("importlib.import_module",
                        side_effect=ImportError("no bc")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only fs")):
            self.mod._enqueue_speech("doomed")   # must not raise


class WellnessPickNudgeLineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self.mod._last_phrase_idx[0] = -1

    def test_single_line_bank_returns_it_directly(self):
        with mock.patch.object(self.mod, "NUDGE_LINES", ("only one",)):
            self.assertEqual(self.mod._pick_nudge_line(), "only one")

    def test_avoids_back_to_back_repeat(self):
        # Force the RNG to first re-pick the just-used index, then a new one, so
        # the dedupe loop (idx == last → retry) runs.
        self.mod._last_phrase_idx[0] = 0
        with mock.patch.object(self.mod.random, "randrange",
                               side_effect=[0, 0, 1]):
            line = self.mod._pick_nudge_line()
        self.assertEqual(line, self.mod.NUDGE_LINES[1])
        self.assertEqual(self.mod._last_phrase_idx[0], 1)


class WellnessPollBreakResetEdgeTests(unittest.TestCase):
    """The break-reset branch only resets when a PRIOR presence timestamp
    exists; with no block and no presence yet, _poll_once returns early."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")
        self.mod._block_started_at[0] = 0.0
        self.mod._last_presence_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0

    def test_absent_with_no_prior_presence_is_noop(self):
        # present False, _last_presence_at 0.0 → break-reset guard is skipped,
        # block stays 0, function returns at the "no block" early-out.
        with mock.patch.object(self.mod, "_user_present", return_value=False), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        self.assertEqual(self.mod._block_started_at[0], 0.0)
        enq.assert_not_called()

    def test_absent_within_grace_keeps_block(self):
        # A recent presence (within BREAK_RESET_SECONDS) means a brief glance
        # away does NOT reset the block.
        now = time.time()
        self.mod._block_started_at[0] = now - 1000
        self.mod._last_presence_at[0] = now - 10   # 10s ago, well within grace
        with mock.patch.object(self.mod, "_user_present", return_value=False), \
             mock.patch.object(self.mod, "_gate_reasons", return_value=[]), \
             mock.patch.object(self.mod, "_enqueue_speech"):
            self.mod._poll_once()
        self.assertGreater(self.mod._block_started_at[0], 0.0)   # survived


class WellnessPollLoopTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("wellness")

    def test_poll_loop_runs_poll_once_then_sleeps(self):
        # After the initial delay, the loop calls _poll_once and then sleeps the
        # poll interval; we break on the interval sleep.
        order = []

        def _sleep(secs):
            order.append(secs)
            if secs == self.mod.WELLNESS_POLL_SECONDS:
                raise _LoopBreak
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_poll_once") as poll:
            # _LoopBreak is a BaseException, so the loop's `except Exception`
            # can't swallow it — it propagates out and ends the loop.
            with self.assertRaises(_LoopBreak):
                self.mod._poll_loop()
        poll.assert_called_once()
        self.assertEqual(order[0], self.mod.INITIAL_DELAY_SECONDS)

    def test_poll_loop_swallows_poll_once_exception(self):
        # _poll_once raising Exception is logged, not fatal; the loop proceeds to
        # the interval sleep (where we break).
        def _sleep(secs):
            if secs == self.mod.WELLNESS_POLL_SECONDS:
                raise _LoopBreak
        with mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_poll_once",
                               side_effect=RuntimeError("poll boom")), \
             mock.patch.object(self.mod.logging, "exception") as logexc:
            with self.assertRaises(_LoopBreak):
                self.mod._poll_loop()
        logexc.assert_called()

    def test_poll_loop_outer_handler_catches_fatal_and_returns(self):
        # An Exception escaping the inner try (here: the initial-delay sleep
        # raising) is caught by the OUTER handler, which logs "terminated
        # unexpectedly" and lets the loop return cleanly (no propagation).
        with mock.patch.object(self.mod.time, "sleep",
                               side_effect=RuntimeError("clock exploded")), \
             mock.patch.object(self.mod.logging, "exception") as logexc:
            self.mod._poll_loop()   # returns normally, does NOT raise
        logexc.assert_called()


class WellnessRegisterTests(unittest.TestCase):
    def test_register_adds_action_and_starts_watcher(self):
        mod, actions = load_skill_isolated("wellness")
        self.assertIn("wellness_status", actions)


# Sentinels for the face-tracker installer.
_RAISE = object()
_NOFUNC = object()


class WellnessImportGuardTests(unittest.TestCase):
    def test_path_bootstrap_inserts_project_root(self):
        mod, _ = load_skill_isolated("wellness")
        path = mod.__file__
        proj = os.path.dirname(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("wellness_reexec", path)
        m = importlib.util.module_from_spec(spec)
        m.skill_utils = {}
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.abspath(p) != os.path.abspath(proj)]
            spec.loader.exec_module(m)
            self.assertIn(m._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved


if __name__ == "__main__":
    unittest.main()
