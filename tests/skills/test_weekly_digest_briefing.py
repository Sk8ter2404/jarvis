"""Logic tests for skills/weekly_digest_briefing.py.

Covers config clamping, the ISO-Monday week label, cluster eligibility (in-band
vs lead-window, day-of-week match, confidence floor, ordering), the offer-line
composer, the once-per-week + max-cards throttle in _next_eligible, state
pruning, the in-call gate, and both registered actions. The scheduler thread is
neutered; pattern_learning / bobert_companion are mocked so nothing real runs.
"""
from __future__ import annotations

import datetime
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class WeeklyDigestBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("weekly_digest_briefing")

    def _cfg(self, **over):
        base = {"enabled": True, "poll_min": 15, "lead_min": 30,
                "conf_floor": 0.5, "max_cards": 3}
        base.update(over)
        return base

    # ── _clamp (pure) ────────────────────────────────────────────────────
    def test_clamp(self):
        c = self.mod._clamp
        self.assertEqual(c(15, 1, 60), 15)
        self.assertEqual(c(0, 1, 120), 1)
        self.assertEqual(c(999, 1, 10), 10)
        self.assertEqual(c(None, 1, 10), 1)

    # ── _week_label (pure) ───────────────────────────────────────────────
    def test_week_label_is_monday(self):
        # 2026-06-03 is a Wednesday → Monday on/before is 2026-06-01.
        ts = time.mktime(datetime.datetime(2026, 6, 3, 10, 0).timetuple())
        self.assertEqual(self.mod._week_label(ts), "2026-06-01")
        # A Monday maps to itself.
        ts_mon = time.mktime(datetime.datetime(2026, 6, 1, 0, 0).timetuple())
        self.assertEqual(self.mod._week_label(ts_mon), "2026-06-01")

    # ── _eligible_clusters ───────────────────────────────────────────────
    def test_eligible_in_band(self):
        now = time.localtime()
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.9, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2,
             "offer": "Netflix?"},
        ]}
        out = self.mod._eligible_clusters(digest, self._cfg())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["key"], "c1")

    def test_eligible_wrong_weekday_excluded(self):
        now = time.localtime()
        other_dow = (now.tm_wday + 1) % 7
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.9, "dow": other_dow,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2},
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg()), [])

    def test_eligible_below_confidence_excluded(self):
        now = time.localtime()
        digest = {"clusters": [
            {"key": "c1", "confidence": 0.1, "dow": now.tm_wday,
             "hour_start": now.tm_hour, "hour_end": now.tm_hour + 2},
        ]}
        self.assertEqual(self.mod._eligible_clusters(digest, self._cfg(conf_floor=0.5)), [])

    def test_eligible_lead_window(self):
        now = time.localtime()
        cur_min = now.tm_hour * 60 + now.tm_min
        # Band starts 20 min from now → within a 30-min lead window.
        start_hour = (cur_min + 20) // 60
        digest = {"clusters": [
            {"key": "soon", "confidence": 0.8, "dow": now.tm_wday,
             "hour_start": start_hour, "hour_end": start_hour + 2, "offer": "x"},
        ]}
        # Only meaningful when the +20 lands in the same hour boundary; assert
        # it's NOT rejected outright by computing lead directly.
        out = self.mod._eligible_clusters(digest, self._cfg(lead_min=120))
        # With a generous 120-min lead window the upcoming band is eligible.
        self.assertEqual([c["key"] for c in out], ["soon"])

    def test_eligible_empty_digest(self):
        self.assertEqual(self.mod._eligible_clusters({}, self._cfg()), [])

    # ── _compose_line ────────────────────────────────────────────────────
    def test_compose_line_prefers_offer(self):
        self.assertEqual(self.mod._compose_line({"offer": "Shall I queue Netflix, sir?"}),
                         "Shall I queue Netflix, sir?")

    def test_compose_line_fallback_label(self):
        out = self.mod._compose_line({"label": "Friday Netflix"})
        self.assertIn("Friday Netflix", out)
        self.assertIn("Shall I proceed", out)

    def test_compose_line_empty(self):
        self.assertEqual(self.mod._compose_line({}), "")

    # ── _next_eligible throttle + max cards ──────────────────────────────
    def test_next_eligible_skips_already_fired_this_week(self):
        week = self.mod._week_label()
        cluster = {"key": "k1", "offer": "Netflix?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state",
                               return_value={"k1": {"week": week, "day": "x"}}):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")
        self.assertEqual(c, {})

    def test_next_eligible_respects_max_cards(self):
        today = time.strftime("%Y-%m-%d", time.localtime())
        cluster = {"key": "k2", "offer": "Netflix?"}
        # Already 3 cards surfaced today, cap is 3 → nothing more.
        state = {f"c{i}": {"week": "old", "day": today} for i in range(3)}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg(max_cards=3)), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]), \
             mock.patch.object(self.mod, "_load_state", return_value=state):
            line, c = self.mod._next_eligible(bypass_throttle=False)
        self.assertEqual(line, "")

    def test_next_eligible_bypass_returns_line(self):
        cluster = {"key": "k3", "offer": "Shall I queue Netflix, sir?"}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={"clusters": [cluster]}), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[cluster]):
            line, c = self.mod._next_eligible(bypass_throttle=True)
        self.assertEqual(line, "Shall I queue Netflix, sir?")
        self.assertEqual(c["key"], "k3")

    def test_next_eligible_no_digest(self):
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value={}):
            self.assertEqual(self.mod._next_eligible(bypass_throttle=True), ("", {}))

    # ── _prune_state ─────────────────────────────────────────────────────
    def test_prune_state_drops_old_weeks(self):
        old_week = (datetime.date.fromisoformat(self.mod._week_label())
                    - datetime.timedelta(weeks=20)).isoformat()
        state = {"stale": old_week, "fresh": self.mod._week_label()}
        self.mod._prune_state(state)
        self.assertNotIn("stale", state)
        self.assertIn("fresh", state)

    # ── in-call gate ─────────────────────────────────────────────────────
    def test_in_call_gate(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Zoom Meeting"]):
            self.assertTrue(self.mod._is_in_call())
        with mock.patch.object(self.mod, "_all_window_titles", return_value=["Spotify"]):
            self.assertFalse(self.mod._is_in_call())

    # ── actions ──────────────────────────────────────────────────────────
    def test_action_now_suppressed_sleep(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=True):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("Suppressed", out)
        self.assertIn("sleep", out.lower())

    def test_action_now_no_match(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible", return_value=("", {})):
            out = self.actions["weekly_digest_now"]("")
        self.assertIn("No weekly habit matches", out)

    def test_action_now_fires(self):
        with mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True), \
             mock.patch.object(self.mod, "_next_eligible",
                               return_value=("Netflix, sir?", {"key": "k"})), \
             mock.patch.object(self.mod, "_enqueue_speech", return_value=True), \
             mock.patch.object(self.mod, "_mark_fired") as mark:
            out = self.actions["weekly_digest_now"]("")
        self.assertEqual(out, "Netflix, sir?")
        mark.assert_called_once()

    def test_action_status(self):
        digest = {"clusters": [{}, {}], "computed_at": time.time()}
        with mock.patch.object(self.mod, "_read_config", return_value=self._cfg()), \
             mock.patch.object(self.mod, "_load_digest", return_value=digest), \
             mock.patch.object(self.mod, "_eligible_clusters", return_value=[]), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=True):
            out = self.actions["weekly_digest_status"]("")
        self.assertIn("2 clusters", out)
        self.assertIn("eligible right now", out)


if __name__ == "__main__":
    unittest.main()
