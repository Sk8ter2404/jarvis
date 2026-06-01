"""Logic tests for skills/anticipation_briefing.py.

Covers config clamping, the weekday bucket, the forward-only minute delta, the
spoken-line composer across its action branches (music / Teams / morning /
evening / generic offer), prediction selection (precise-with-lead before
broad-now, confidence floor), the throttle-aware _next_eligible pick, state
pruning, the hard gates, and both registered actions. The scheduler thread is
neutered; bobert_companion / pattern_learning are mocked so nothing real runs.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class AnticipationBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("anticipation_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 5, "lead_min": 15, "conf_floor": 0.5}
        base.update(over)
        return base

    # ── _clamp (pure) ────────────────────────────────────────────────────
    def test_clamp(self):
        c = self.mod._clamp
        self.assertEqual(c(5, 1, 60), 5)
        self.assertEqual(c(0, 1, 60), 1)       # below floor
        self.assertEqual(c(99, 1, 60), 60)     # above ceiling
        self.assertEqual(c("nan", 1, 60), 1)   # uncastable → floor

    # ── _bucket_for_weekday / _minutes_until (pure) ──────────────────────
    def test_bucket_for_weekday(self):
        self.assertEqual(self.mod._bucket_for_weekday(0), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(4), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(5), "weekend")
        self.assertEqual(self.mod._bucket_for_weekday(6), "weekend")

    def test_minutes_until_forward_only(self):
        self.assertEqual(self.mod._minutes_until(600, 540), 60)   # 60 min ahead
        self.assertEqual(self.mod._minutes_until(540, 540), 0)    # now
        self.assertEqual(self.mod._minutes_until(500, 540), 10_000)  # past → sentinel

    # ── _titlecase (pure) ────────────────────────────────────────────────
    def test_titlecase(self):
        self.assertEqual(self.mod._titlecase("sam industries"), "Sam Industries")

    # ── _compose_briefing_line branches ──────────────────────────────────
    def test_compose_music_with_arg(self):
        line = self.mod._compose_briefing_line(
            {"action": "play_music", "common_arg": "michael jackson"})
        self.assertIn("queue your usual", line.lower())
        self.assertIn("Michael Jackson", line)

    def test_compose_music_no_arg(self):
        line = self.mod._compose_briefing_line({"action": "resume_music", "common_arg": ""})
        self.assertIn("usual playlist", line.lower())

    def test_compose_teams_with_name_and_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "check_teams", "common_arg": "sam", "__lead_minutes": 12})
        self.assertIn("Sam sync in 12 minutes", line)
        self.assertIn("last conversation", line.lower())

    def test_compose_teams_no_arg_no_lead(self):
        line = self.mod._compose_briefing_line({"action": "check_teams"})
        self.assertIn("check Teams about now", line)

    def test_compose_morning_briefing_with_lead(self):
        line = self.mod._compose_briefing_line(
            {"action": "morning_briefing", "__lead_minutes": 5})
        self.assertIn("Morning briefing", line)
        self.assertIn("5 minute", line)

    def test_compose_generic_offer(self):
        line = self.mod._compose_briefing_line(
            {"action": "something_else", "offer": "Shall I open your inbox, sir?",
             "__lead_minutes": 3})
        self.assertIn("In about 3 minutes", line)
        self.assertIn("Shall I open your inbox", line)

    def test_compose_empty_when_nothing(self):
        self.assertEqual(self.mod._compose_briefing_line({"action": "x"}), "")

    # ── _select_predictions ──────────────────────────────────────────────
    def test_select_precise_within_lead(self):
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        snap = {"precise": [
            {"key": "p1", "action": "check_teams", "ratio": 0.9,
             "center_minute": cur_min + 10, "tolerance_min": 5, "common_arg": "sam"},
        ], "broad": []}
        preds = self.mod._select_predictions(snap, self._cfg(lead_min=15))
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["key"], "p1")
        self.assertEqual(preds[0]["__lead_minutes"], 10)

    def test_select_filters_below_confidence(self):
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        snap = {"precise": [
            {"key": "low", "action": "check_teams", "ratio": 0.2,
             "center_minute": cur_min + 5, "tolerance_min": 5},
        ], "broad": []}
        self.assertEqual(self.mod._select_predictions(snap, self._cfg(conf_floor=0.5)), [])

    def test_select_precise_before_broad(self):
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        snap = {
            "precise": [{"key": "pr", "action": "check_teams", "ratio": 0.9,
                         "center_minute": cur_min + 5, "tolerance_min": 2}],
            "broad": [{"key": "br", "action": "play_music", "ratio": 0.95,
                       "bucket": self.mod._bucket_for_weekday(now.tm_wday),
                       "hour_window": [now.tm_hour, now.tm_hour + 1]}],
        }
        preds = self.mod._select_predictions(snap, self._cfg())
        self.assertEqual(preds[0]["key"], "pr")   # precise sorts first
        self.assertIn("br", [p["key"] for p in preds])

    def test_select_empty_snapshot(self):
        self.assertEqual(self.mod._select_predictions({}, self._cfg()), [])

    # ── _next_eligible throttle ──────────────────────────────────────────
    def test_next_eligible_skips_throttled_key(self):
        today = time.strftime("%Y-%m-%d", time.localtime())
        pred = {"key": "k1", "action": "check_teams", "common_arg": "sam",
                "__lead_minutes": 10}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[pred]), \
             mock.patch.object(self.mod, "_load_state", return_value={"k1": today}):
            line, p = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")
        self.assertEqual(p, {})

    def test_next_eligible_bypass_returns_line(self):
        pred = {"key": "k1", "action": "check_teams", "common_arg": "sam",
                "__lead_minutes": 10}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={"precise": [], "broad": []}), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[pred]):
            line, p = self.mod._next_eligible(bypass_throttle=True)
        self.assertIn("Sam sync", line)
        self.assertEqual(p["key"], "k1")

    def test_next_eligible_no_snapshot(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value={}):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))

    # ── _prune_state ─────────────────────────────────────────────────────
    def test_prune_state_drops_old(self):
        old_day = time.strftime("%Y-%m-%d", time.localtime(time.time() - 200 * 86400))
        today = time.strftime("%Y-%m-%d", time.localtime())
        state = {"stale": old_day, "fresh": today}
        self.mod._prune_state(state)
        self.assertNotIn("stale", state)
        self.assertIn("fresh", state)

    # ── hard gates ───────────────────────────────────────────────────────
    def test_gate_in_call_detects_window_hint(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Project | Microsoft Teams Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_gate_not_in_call(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Notepad"]):
            self.assertFalse(self.mod._is_in_call())

    # ── actions ──────────────────────────────────────────────────────────
    def test_action_now_suppressed_in_call(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=True):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("call", out.lower())

    def test_action_now_no_match(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})):
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("No prediction matches", out)

    def test_action_now_fires_and_marks(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Sam sync in 5 minutes, sir.", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.actions["anticipation_briefing_now"]("")
        self.assertIn("Sam sync", out)
        mark.assert_called_once()

    def test_action_status_reports_counts(self):
        snap = {"broad": [{}, {}], "precise": [{}]}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_snapshot", return_value=snap), \
             mock.patch.object(self.mod, "_select_predictions", return_value=[]), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["anticipation_briefing_status"]("")
        self.assertIn("2 broad, 1 precise", out)
        self.assertIn("eligible right now", out)


if __name__ == "__main__":
    unittest.main()
