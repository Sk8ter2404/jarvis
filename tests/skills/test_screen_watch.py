"""Logic tests for skills/screen_watch.py.

screen_watch fires a gentle stretch nudge after a long single-window idle
session. Tests cover the deterministic gate logic and the poll state machine:
  • _title_is_ignored (HUD / lock screen / game substrings),
  • _fmt_minutes formatting,
  • _is_sleeping_or_standby + _user_is_away gates (reading injected modules),
  • _poll_once: identity-change resets the stare timer; the nudge fires only
    when stare + idle thresholds are met AND no gate blocks; cooldown
    suppresses a repeat for the same window-identity,
  • the screen_watch_status action's gate readout.

mss / pygetwindow / Win32 idle and the speech enqueue are all mocked — no
real screenshots, no window queries, no pending_speech.json writes.
"""
from __future__ import annotations

import sys
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class ScreenWatchHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self._reset()

    def _reset(self):
        self.mod._current_identity[0] = None
        self.mod._stare_started_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0
        self.mod._last_nudged_id[0] = None

    # ── _title_is_ignored ────────────────────────────────────────────────
    def test_title_ignored_hud_and_lockscreen(self):
        self.assertTrue(self.mod._title_is_ignored("J.A.R.V.I.S HUD"))
        self.assertTrue(self.mod._title_is_ignored("Windows Default Lock Screen"))
        self.assertTrue(self.mod._title_is_ignored("Program Manager"))

    def test_title_not_ignored_for_normal_window(self):
        self.assertFalse(self.mod._title_is_ignored("report.docx - Word"))

    # ── _fmt_minutes ─────────────────────────────────────────────────────
    def test_fmt_minutes(self):
        f = self.mod._fmt_minutes
        self.assertEqual(f(30), "30s")
        self.assertEqual(f(90), "1m 30s")
        self.assertEqual(f(3700), "1h 1m")

    # ── gate helpers ─────────────────────────────────────────────────────
    def test_is_sleeping_reads_bobert_flags(self):
        bc = mock.MagicMock()
        bc._sleep_mode = [True]
        bc._standby_mode = [False]
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertTrue(self.mod._is_sleeping_or_standby())

    def test_is_sleeping_false_when_bc_absent(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            self.assertFalse(self.mod._is_sleeping_or_standby())

    def test_user_is_away_true_only_when_tracker_says_away(self):
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": time.time(),
                                           "current_monitor": "away"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertTrue(self.mod._user_is_away())

    def test_user_is_away_false_when_looking_at_monitor(self):
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": time.time(),
                                           "current_monitor": "left"}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertFalse(self.mod._user_is_away())

    def test_user_is_away_false_when_tracker_unestablished(self):
        # last_sample_at == 0 → tracker hasn't established gaze → don't suppress.
        ft = mock.MagicMock()
        ft._snapshot_state.return_value = {"last_sample_at": 0.0, "current_monitor": None}
        with mock.patch.dict(sys.modules, {"skill_face_tracker": ft}):
            self.assertFalse(self.mod._user_is_away())

    def test_user_is_away_false_when_face_tracker_not_loaded(self):
        sys.modules.pop("skill_face_tracker", None)
        self.assertFalse(self.mod._user_is_away())


class ScreenWatchPollTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("screen_watch")
        self.mod._current_identity[0] = None
        self.mod._stare_started_at[0] = 0.0
        self.mod._last_nudge_at[0] = 0.0
        self.mod._last_nudged_id[0] = None
        self.threshold = self.mod.STARE_THRESHOLD_SECONDS

    def _patch_focus(self, title="report.docx - Word", h="hash123"):
        """Patch the window + hash helpers so _poll_once sees a stable window."""
        return (
            mock.patch.object(self.mod, "_get_focused_window",
                              return_value=(title, (0, 0, 800, 600))),
            mock.patch.object(self.mod, "_hash_window_thumbnail", return_value=h),
        )

    def test_poll_first_sight_sets_stare_timer(self):
        p1, p2 = self._patch_focus()
        with p1, p2:
            self.mod._poll_once()
        self.assertEqual(self.mod._current_identity[0], ("report.docx - Word", "hash123"))
        self.assertGreater(self.mod._stare_started_at[0], 0.0)

    def test_poll_no_window_clears_identity(self):
        self.mod._current_identity[0] = ("old", "h")
        self.mod._stare_started_at[0] = 100.0
        with mock.patch.object(self.mod, "_get_focused_window", return_value=(None, None)):
            self.mod._poll_once()
        self.assertIsNone(self.mod._current_identity[0])
        self.assertEqual(self.mod._stare_started_at[0], 0.0)

    def test_poll_ignored_title_clears_identity(self):
        with mock.patch.object(self.mod, "_get_focused_window",
                               return_value=("JARVIS HUD", (0, 0, 800, 600))):
            self.mod._poll_once()
        self.assertIsNone(self.mod._current_identity[0])

    def test_poll_fires_nudge_when_all_gates_clear(self):
        # Pre-seed an established stare older than the threshold.
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 60), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_called_once()
        self.assertIn("stretch", enq.call_args[0][0].lower())

    def test_poll_suppressed_when_idle_below_threshold(self):
        # User IS actively using the window (idle low) → no nudge.
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=5.0), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_suppressed_when_sleeping(self):
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    def test_poll_cooldown_suppresses_repeat(self):
        ident = ("report.docx - Word", "hash123")
        self.mod._current_identity[0] = ident
        self.mod._stare_started_at[0] = time.time() - (self.threshold + 60)
        # Same identity nudged 1 minute ago → within the 1h cooldown.
        self.mod._last_nudged_id[0] = ident
        self.mod._last_nudge_at[0] = time.time() - 60
        p1, p2 = self._patch_focus()
        with p1, p2, \
             mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds",
                               return_value=self.threshold + 60), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq:
            self.mod._poll_once()
        enq.assert_not_called()

    # ── screen_watch_status action ───────────────────────────────────────
    def test_status_no_window(self):
        self.mod._current_identity[0] = None
        self.assertIn("haven't established", self.actions["screen_watch_status"](""))

    def test_status_lists_open_gates(self):
        self.mod._current_identity[0] = ("report.docx - Word", "h")
        self.mod._stare_started_at[0] = time.time() - 120  # only 2 min stare
        with mock.patch.object(self.mod, "_is_sleeping_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_user_is_away", return_value=False), \
             mock.patch.object(self.mod, "_get_system_idle_seconds", return_value=10.0):
            out = self.actions["screen_watch_status"]("")
        self.assertIn("report.docx - Word", out)
        self.assertIn("stare only", out)   # 2m < 25m threshold
        self.assertIn("idle only", out)


if __name__ == "__main__":
    unittest.main()
