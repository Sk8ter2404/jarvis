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

import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


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


if __name__ == "__main__":
    unittest.main()
