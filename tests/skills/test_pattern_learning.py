"""Logic tests for skills/pattern_learning.py.

The high-value targets are the pure formatting/labelling helpers and the
aggregate()/predictions_for_now()/maybe_pattern_offer_v2() pipeline that turns
an action-event log into broad-window and precise-clock habit predictions.

All disk and time dependencies are controlled: aggregate() is pointed at a
temp JSONL (via patched _LOG_FILE/_AGG_FILE/_STATE_FILE), and the "current
moment" is pinned by patching the skill's time.localtime so window-matching is
deterministic. No SQLite/background thread is exercised (harness neuters
threads; SQLite paths are only hit by the weekly digest, which we drive through
a stubbed connection-less path).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _struct(year, mon, day, hour, minute, wday):
    # time.struct_time positional layout; yday/isdst don't affect the code.
    return time.struct_time((year, mon, day, hour, minute, 0, wday, 1, -1))


class PatternLearningHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")

    def test_bucket_for_weekday(self):
        self.assertEqual(self.mod._bucket_for_weekday(0), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(4), "weekday")
        self.assertEqual(self.mod._bucket_for_weekday(5), "weekend")
        self.assertEqual(self.mod._bucket_for_weekday(6), "weekend")

    def test_format_hour_window(self):
        self.assertEqual(self.mod._format_hour_window(9, 11), "9am-11am")
        self.assertEqual(self.mod._format_hour_window(13, 15), "1pm-3pm")
        self.assertEqual(self.mod._format_hour_window(0, 2), "12am-2am")

    def test_format_clock(self):
        self.assertEqual(self.mod._format_clock(555), "09:15")   # 9*60+15
        self.assertEqual(self.mod._format_clock(0), "00:00")
        self.assertEqual(self.mod._format_clock(23 * 60 + 59), "23:59")

    def test_verb_for_known_and_fallback(self):
        self.assertEqual(self.mod._verb_for("play_music"), "plays music")
        self.assertEqual(self.mod._verb_for("check_teams"), "checks Teams")
        # Unknown action → generic verb derived from the name.
        self.assertEqual(self.mod._verb_for("do_thing"), "runs do thing")

    def test_offer_for_fallback(self):
        self.assertIn("Shall I", self.mod._offer_for("do_thing"))

    def test_titlecase_keeps_small_words_lower(self):
        self.assertEqual(self.mod._titlecase("michael jackson"), "Michael Jackson")
        self.assertEqual(self.mod._titlecase("lord of the rings"),
                         "Lord of the Rings")

    def test_format_hour_band_two_hour_window(self):
        out = self.mod._format_hour_band(20)
        self.assertIn("8 PM", out)
        self.assertIn("10 PM", out)

    def test_cluster_label_contains_day_verb_and_weeks(self):
        label = self.mod._cluster_label(4, 20, "netflix", 4, 4, "")
        self.assertIn("Friday", label)
        self.assertIn("opens Netflix", label)
        self.assertIn("4/4 weeks", label)

    def test_cluster_offer_netflix_specialised(self):
        out = self.mod._cluster_offer(4, 20, "netflix", "")
        self.assertIn("Netflix", out)
        self.assertIn("Friday", out)

    def test_compose_offer_line_music_uses_arg(self):
        out = self.mod._compose_offer_line(
            {"action": "play_music", "common_arg": "michael jackson",
             "offer": "fallback"})
        self.assertIn("Michael Jackson", out)
        self.assertIn("mix", out)

    def test_compose_offer_line_generic_uses_offer(self):
        out = self.mod._compose_offer_line(
            {"action": "check_teams", "offer": "Shall I check Teams, sir?"})
        self.assertEqual(out, "Shall I check Teams, sir?")


class PatternLearningAggregateTests(unittest.TestCase):
    """Drive the full aggregate() pipeline against a synthetic 21-day log."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")
        fd, self.logp = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.aggp = self.logp + ".agg.json"
        self.statep = self.logp + ".state.json"
        self._patches = [
            mock.patch.object(self.mod, "_LOG_FILE", self.logp),
            mock.patch.object(self.mod, "_AGG_FILE", self.aggp),
            mock.patch.object(self.mod, "_STATE_FILE", self.statep),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for path in (self.logp, self.aggp, self.statep):
            try:
                os.unlink(path)
            except OSError:
                pass

    def _write_events(self, events):
        with open(self.logp, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def _synthetic_21_days(self):
        """play_music on ~80% of weekdays 9-11am; check_teams daily ~09:15."""
        fake = []
        base = time.time() - 21 * 86400
        for day in range(21):
            ts = base + day * 86400
            lt = time.localtime(ts)
            wd = lt.tm_wday
            date = time.strftime("%Y-%m-%d", lt)
            dow = time.strftime("%A", lt)
            if wd < 5 and day % 5 != 4:    # 4/5 weekdays = 80%
                fake.append({"ts": ts, "date": date, "dow": dow, "wd": wd,
                             "hour": 9 + (day % 2), "min": 17,
                             "action": "play_music", "arg": "michael jackson"})
            fake.append({"ts": ts, "date": date, "dow": dow, "wd": wd,
                         "hour": 9, "min": 13 + (day % 5),
                         "action": "check_teams", "arg": ""})
        return fake

    def test_aggregate_empty_log(self):
        self._write_events([])
        snap = self.mod.aggregate()
        self.assertEqual(snap["events"], 0)
        self.assertEqual(snap["broad"], [])
        self.assertEqual(snap["precise"], [])

    def test_aggregate_detects_broad_and_precise(self):
        self._write_events(self._synthetic_21_days())
        snap = self.mod.aggregate()
        self.assertGreater(snap["events"], 30)

        # Broad: play_music weekdays 9-11am at ~80%.
        music = next((p for p in snap["broad"] if p["action"] == "play_music"),
                     None)
        self.assertIsNotNone(music, "expected a broad play_music prediction")
        self.assertEqual(music["bucket"], "weekday")
        self.assertEqual(music["hour_window"], [9, 11])
        self.assertGreaterEqual(music["ratio"], 0.5)
        self.assertIn("plays music", music["label"])
        # Argument is carried so the offer can say "your usual <arg> mix".
        self.assertEqual(music["common_arg"], "michael jackson")

        # Precise: check_teams at 09:15 ± a few minutes.
        teams = next((p for p in snap["precise"] if p["action"] == "check_teams"),
                     None)
        self.assertIsNotNone(teams, "expected a precise check_teams prediction")
        self.assertEqual(teams["center_clock"], "09:15")
        self.assertLessEqual(teams["tolerance_min"], 12)
        self.assertGreaterEqual(teams["ratio"], 0.6)

    def test_aggregate_below_min_days_yields_no_broad(self):
        # Only 3 days of data — under MIN_DAYS_OBSERVED (7) so nothing emits
        # even though the action repeats.
        fake = []
        base = time.time() - 3 * 86400
        for day in range(3):
            ts = base + day * 86400
            lt = time.localtime(ts)
            fake.append({"ts": ts, "date": time.strftime("%Y-%m-%d", lt),
                         "dow": time.strftime("%A", lt), "wd": lt.tm_wday,
                         "hour": 10, "min": 0, "action": "play_music",
                         "arg": ""})
        self._write_events(fake)
        snap = self.mod.aggregate()
        self.assertEqual(snap["broad"], [])

    def test_predictions_for_now_matches_window(self):
        snap = {
            "broad": [{"key": "b1", "type": "broad", "bucket": "weekday",
                       "hour_window": [9, 11], "ratio": 0.8,
                       "action": "play_music", "common_arg": "", "offer": "x",
                       "label": "L"}],
            "precise": [{"key": "p1", "type": "precise", "center_minute": 555,
                         "tolerance_min": 5, "ratio": 0.9,
                         "action": "check_teams", "offer": "y", "label": "L2"}],
        }
        # Wednesday 09:15 — both the broad window and precise center match.
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 9, 15, 2)):
            matches = self.mod.predictions_for_now(snap)
        actions = [m["action"] for m in matches]
        self.assertIn("play_music", actions)
        self.assertIn("check_teams", actions)
        # Precise sorts before broad (more specific first).
        self.assertEqual(matches[0]["type"], "precise")

    def test_predictions_for_now_no_match_off_hours(self):
        snap = {
            "broad": [{"key": "b1", "type": "broad", "bucket": "weekday",
                       "hour_window": [9, 11], "ratio": 0.8,
                       "action": "play_music", "offer": "x", "label": "L"}],
            "precise": [],
        }
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 3, 0, 2)):
            self.assertEqual(self.mod.predictions_for_now(snap), [])

    def test_maybe_offer_throttles_once_per_day(self):
        snap = {
            "generated_at": time.time(), "events": 50, "days_span": 21.0,
            "broad": [], "precise": [
                {"key": "precise|check_teams|09:15", "type": "precise",
                 "center_minute": 555, "tolerance_min": 5, "ratio": 0.9,
                 "action": "check_teams",
                 "offer": "Shall I check Teams, sir?", "label": "L"}],
        }
        self.mod._atomic_write_json(self.aggp, snap)
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 9, 15, 2)):
            first = self.mod.maybe_pattern_offer_v2()
            self.assertIn("Teams", first)
            # Second call same day for the same key is throttled to "".
            second = self.mod.maybe_pattern_offer_v2()
            self.assertEqual(second, "")
            # Bypassing the throttle returns the line again.
            bypass = self.mod.maybe_pattern_offer_v2(bypass_throttle=True)
            self.assertIn("Teams", bypass)

    def test_maybe_offer_empty_when_no_snapshot(self):
        # No aggregated file at all → no offer.
        self.assertEqual(self.mod.maybe_pattern_offer_v2(), "")


class PatternLearningActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")

    def test_pattern_predictions_no_snapshot(self):
        with mock.patch.object(self.mod, "_load_aggregated", return_value={}):
            out = self.actions["pattern_predictions"]("")
        self.assertIn("no predictions", out.lower())

    def test_pattern_predictions_renders_labels(self):
        snap = {"events": 120, "days_span": 21.0,
                "broad": [{"label": "You plays music weekdays 80%"}],
                "precise": [{"label": "You checks Teams at 09:15"}]}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap):
            out = self.actions["pattern_predictions"]("")
        self.assertIn("checks Teams at 09:15", out)
        self.assertIn("120 events", out)

    def test_pattern_predictions_weak_data_message(self):
        snap = {"events": 4, "days_span": 1.0, "broad": [], "precise": []}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap):
            out = self.actions["pattern_predictions"]("")
        self.assertIn("no strong patterns", out.lower())

    def test_pattern_offer_now_no_match(self):
        with mock.patch.object(self.mod, "maybe_pattern_offer_v2", return_value=""):
            out = self.actions["pattern_offer_now"]("")
        self.assertIn("no prediction matches", out.lower())

    def test_pattern_aggregate_reports_counts(self):
        fake_snap = {"events": 42, "broad": [{}, {}], "precise": [{}]}
        with mock.patch.object(self.mod, "aggregate", return_value=fake_snap):
            out = self.actions["pattern_aggregate"]("")
        self.assertIn("42 events", out)
        self.assertIn("2 broad", out)
        self.assertIn("1 precise", out)

    def test_pattern_stats_never_aggregated(self):
        with mock.patch.object(self.mod, "_load_aggregated", return_value={}), \
             mock.patch.object(self.mod, "_load_offer_state", return_value={}):
            out = self.actions["pattern_stats"]("")
        self.assertIn("never", out)

    def test_weekly_digest_no_clusters(self):
        with mock.patch.object(self.mod, "compute_weekly_digest",
                               return_value={"clusters": []}):
            out = self.actions["weekly_digest"]("")
        self.assertIn("no weekly habits", out.lower())

    def test_weekly_digest_renders_top_clusters(self):
        digest = {"clusters": [
            {"label": "Friday 8-10 PM: opens Netflix 4/4 weeks"},
            {"label": "Monday 9-11 AM: checks Teams 3/4 weeks"},
        ]}
        with mock.patch.object(self.mod, "compute_weekly_digest",
                               return_value=digest):
            out = self.actions["weekly_digest"]("")
        self.assertIn("Netflix", out)
        self.assertIn("Teams", out)


if __name__ == "__main__":
    unittest.main()
