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
import shutil
import sqlite3
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


class _TempDataBase(unittest.TestCase):
    """Base for tests that exercise the on-disk stores (JSONL + SQLite +
    aggregate/state snapshots). Redirects every module-level path constant at a
    fresh temp dir and resets the module's mutable globals so each test starts
    from a pristine ingestion state. The real project data/ dir is never read
    or written.

    load_skill_isolated() re-execs a fresh module per test, so globals come up
    clean; we still reset defensively in setUp/tearDown so ordering can never
    leak the SQLite once-flags or the debounce cursor between cases.
    """

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")
        self.tmp = tempfile.mkdtemp(prefix="patlearn_test_")
        self.addCleanup(self._cleanup_tmp)
        self.logp = os.path.join(self.tmp, "usage_patterns.jsonl")
        self.dbp = os.path.join(self.tmp, "usage_patterns.sqlite3")
        self.aggp = os.path.join(self.tmp, "usage_patterns_aggregated.json")
        self.statep = os.path.join(self.tmp, "usage_patterns_offer_state.json")
        self._patches = [
            mock.patch.object(self.mod, "_DATA_DIR", self.tmp),
            mock.patch.object(self.mod, "_LOG_FILE", self.logp),
            mock.patch.object(self.mod, "_DB_FILE", self.dbp),
            mock.patch.object(self.mod, "_AGG_FILE", self.aggp),
            mock.patch.object(self.mod, "_STATE_FILE", self.statep),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop_patches)
        self._reset_globals()
        self.addCleanup(self._reset_globals)

    def _stop_patches(self):
        for p in self._patches:
            p.stop()

    def _reset_globals(self):
        # Wipe the SQLite "once" flags + the debounce / rotation counters so a
        # subsequent test re-initialises the schema against ITS temp DB.
        self.mod._db_initialized[0] = False
        self.mod._backfill_done.clear()
        self.mod._writes_since_rotate[0] = 0
        self.mod._last_event["key"] = ""
        self.mod._last_event["ts"] = 0.0

    def _cleanup_tmp(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers ----------------------------------------------------------------
    def _write_jsonl(self, events):
        with open(self.logp, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def _event(self, ts, action="check_teams", arg="", hour=9, minute=15,
               wd=2, date=None):
        lt = time.localtime(ts)
        return {
            "ts": ts,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", lt),
            "date": date or time.strftime("%Y-%m-%d", lt),
            "dow": time.strftime("%A", lt),
            "wd": wd,
            "hour": hour,
            "min": minute,
            "action": action,
            "arg": arg,
        }

    def _db_count(self):
        conn = sqlite3.connect(self.dbp)
        try:
            return conn.execute("SELECT COUNT(1) FROM events").fetchone()[0]
        finally:
            conn.close()


# ─── log_event (JSONL + SQLite mirror, debounce, rotation) ────────────────
class LogEventTests(_TempDataBase):
    def test_log_event_writes_jsonl_and_sqlite(self):
        self.mod.log_event("play_music", "Michael Jackson")
        # JSONL line present and well-formed.
        events = self.mod._load_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "play_music")
        self.assertEqual(events[0]["arg"], "Michael Jackson")
        # arg is normalised/trimmed (strip()) but case preserved in JSONL.
        self.assertIn("wd", events[0])
        self.assertIn("hour", events[0])
        # SQLite mirror got the row too.
        self.assertEqual(self._db_count(), 1)

    def test_log_event_ignores_empty_action(self):
        self.mod.log_event("", "x")
        self.assertEqual(self.mod._load_events(), [])

    def test_log_event_debounces_identical_back_to_back(self):
        t = [1_700_000_000.0]
        with mock.patch.object(self.mod.time, "time", lambda: t[0]):
            self.mod.log_event("check_teams", "")
            # Same key, within DEBOUNCE_SECONDS → dropped.
            t[0] += self.mod.DEBOUNCE_SECONDS / 2.0
            self.mod.log_event("check_teams", "")
            self.assertEqual(len(self.mod._load_events()), 1)
            # Past the debounce window → accepted.
            t[0] += self.mod.DEBOUNCE_SECONDS + 0.1
            self.mod.log_event("check_teams", "")
        self.assertEqual(len(self.mod._load_events()), 2)

    def test_log_event_different_arg_not_debounced(self):
        t = [1_700_000_000.0]
        with mock.patch.object(self.mod.time, "time", lambda: t[0]):
            self.mod.log_event("play_music", "alpha")
            self.mod.log_event("play_music", "beta")   # different key
        self.assertEqual(len(self.mod._load_events()), 2)

    def test_log_event_arg_truncated_to_120(self):
        self.mod.log_event("open_url", "x" * 500)
        ev = self.mod._load_events()[0]
        self.assertEqual(len(ev["arg"]), 120)

    def test_log_event_write_failure_is_swallowed(self):
        # open() raising for the JSONL append must not propagate.
        with mock.patch.object(self.mod, "open", side_effect=OSError("nope"),
                               create=True):
            self.mod.log_event("check_teams", "")  # returns without raising
        # Nothing was logged.
        self.assertEqual(self.mod._load_events(), [])

    def test_log_event_triggers_rotation_and_prune(self):
        # Force the periodic-maintenance branch (every 200 writes) and assert it
        # invokes both the JSONL rotate and the SQLite prune helpers.
        self.mod._writes_since_rotate[0] = 199
        with mock.patch.object(self.mod, "_maybe_rotate") as rot, \
             mock.patch.object(self.mod, "_prune_sqlite_events") as prune:
            self.mod.log_event("check_teams", "")
        rot.assert_called_once()
        prune.assert_called_once()
        self.assertEqual(self.mod._writes_since_rotate[0], 0)

    def test_log_event_sqlite_mirror_failure_does_not_block_jsonl(self):
        # A broken SQLite connection must not stop the JSONL append.
        with mock.patch.object(self.mod, "_connect_db",
                               side_effect=RuntimeError("db down")):
            self.mod.log_event("check_teams", "")
        self.assertEqual(len(self.mod._load_events()), 1)


# ─── _maybe_rotate / _load_events resilience ──────────────────────────────
class RotateAndLoadTests(_TempDataBase):
    def test_maybe_rotate_truncates_to_cap(self):
        with mock.patch.object(self.mod, "MAX_LOG_ENTRIES", 5):
            with open(self.logp, "w", encoding="utf-8") as f:
                for i in range(12):
                    f.write(json.dumps({"action": "a", "n": i}) + "\n")
            self.mod._maybe_rotate()
            with open(self.logp, encoding="utf-8") as f:
                lines = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(lines), 5)
        # Kept the most-recent N.
        self.assertEqual([l["n"] for l in lines], [7, 8, 9, 10, 11])

    def test_maybe_rotate_noop_under_cap(self):
        self._write_jsonl([self._event(1.0) for _ in range(3)])
        self.mod._maybe_rotate()
        self.assertEqual(len(self.mod._load_events()), 3)

    def test_maybe_rotate_missing_file_swallowed(self):
        # No log file present → rotation just returns.
        self.assertFalse(os.path.exists(self.logp))
        self.mod._maybe_rotate()  # no raise

    def test_load_events_skips_corrupt_lines(self):
        with open(self.logp, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._event(1.0)) + "\n")
            f.write("{ this is not json\n")        # corrupt → skipped
            f.write("\n")                            # blank → skipped
            f.write(json.dumps(self._event(2.0)) + "\n")
        events = self.mod._load_events()
        self.assertEqual(len(events), 2)

    def test_load_events_missing_file_returns_empty(self):
        self.assertEqual(self.mod._load_events(), [])


# ─── SQLite plumbing: connect / backfill / prune ──────────────────────────
class SqliteStoreTests(_TempDataBase):
    def test_connect_db_initialises_schema(self):
        conn = self.mod._connect_db()
        self.assertIsNotNone(conn)
        try:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            conn.close()
        self.assertIn("events", names)
        self.assertIn("weekly_summaries", names)
        self.assertTrue(self.mod._db_initialized[0])

    def test_connect_db_returns_none_on_connect_failure(self):
        with mock.patch.object(self.mod.sqlite3, "connect",
                               side_effect=sqlite3.OperationalError("locked")):
            self.assertIsNone(self.mod._connect_db())

    def test_connect_db_schema_init_failure_returns_none(self):
        real_connect = self.mod.sqlite3.connect

        class _BadConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *a, **k):
                if "CREATE TABLE" in sql or "PRAGMA" in sql:
                    raise sqlite3.OperationalError("schema boom")
                return self._inner.execute(sql, *a, **k)

            def close(self):
                self._inner.close()

        def _wrapped(*a, **k):
            return _BadConn(real_connect(*a, **k))

        with mock.patch.object(self.mod.sqlite3, "connect", side_effect=_wrapped):
            self.assertIsNone(self.mod._connect_db())
        # Init flag stays False so a later healthy connect can retry.
        self.assertFalse(self.mod._db_initialized[0])

    def test_backfill_replays_jsonl_into_empty_sqlite(self):
        base = 1_700_000_000.0
        self._write_jsonl([self._event(base + i, action="check_teams")
                           for i in range(4)])
        inserted = self.mod._backfill_sqlite_from_jsonl_if_empty()
        self.assertEqual(inserted, 4)
        self.assertEqual(self._db_count(), 4)
        self.assertTrue(self.mod._backfill_done.is_set())

    def test_backfill_noop_when_already_done_flag_set(self):
        self.mod._backfill_done.set()
        self._write_jsonl([self._event(1.0)])
        self.assertEqual(self.mod._backfill_sqlite_from_jsonl_if_empty(), 0)

    def test_backfill_noop_when_events_table_populated(self):
        self.mod.log_event("check_teams", "")   # seeds one SQLite row
        self.mod._backfill_done.clear()          # pretend we haven't backfilled
        self._write_jsonl([self._event(1.0), self._event(2.0)])
        # Table already has a row → COUNT>0 path → nothing replayed.
        self.assertEqual(self.mod._backfill_sqlite_from_jsonl_if_empty(), 0)

    def test_backfill_noop_when_jsonl_empty(self):
        self.assertEqual(self.mod._backfill_sqlite_from_jsonl_if_empty(), 0)
        self.assertTrue(self.mod._backfill_done.is_set())

    def test_backfill_returns_zero_when_db_unavailable(self):
        with mock.patch.object(self.mod, "_connect_db", return_value=None):
            self.assertEqual(self.mod._backfill_sqlite_from_jsonl_if_empty(), 0)

    def test_prune_sqlite_events_trims_oldest(self):
        conn = self.mod._connect_db()
        try:
            for i in range(10):
                conn.execute(
                    "INSERT INTO events(ts, iso, date, dow, hour, minute, action, arg) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (float(i), "", "2026-01-01", 0, 0, 0, "a", ""))
        finally:
            conn.close()
        self.mod._prune_sqlite_events(max_rows=4)
        self.assertEqual(self._db_count(), 4)
        # The four NEWEST (highest ts) survive.
        conn = sqlite3.connect(self.dbp)
        try:
            kept = {r[0] for r in conn.execute("SELECT ts FROM events").fetchall()}
        finally:
            conn.close()
        self.assertEqual(kept, {6.0, 7.0, 8.0, 9.0})

    def test_prune_sqlite_events_noop_under_cap(self):
        self.mod.log_event("check_teams", "")
        self.mod._prune_sqlite_events(max_rows=100)
        self.assertEqual(self._db_count(), 1)

    def test_prune_sqlite_events_db_unavailable(self):
        with mock.patch.object(self.mod, "_connect_db", return_value=None):
            self.mod._prune_sqlite_events()  # no raise


# ─── weekly digest: SQL path, JSONL fallback, persistence ─────────────────
class WeeklyDigestTests(_TempDataBase):
    def _seed_weekly_events(self, weeks=3):
        """Netflix every Friday ~20:30 across `weeks` distinct weeks, anchored so
        every timestamp falls inside the lookback window relative to `self.now`.
        """
        self.now = 1_700_000_000.0
        # Find a recent Friday at 20:30 local, then step back one week at a time.
        events = []
        # Friday = wd 4. Walk back day-by-day from now to land on a Friday.
        anchor = self.now
        for _ in range(8):
            if time.localtime(anchor).tm_wday == 4:
                break
            anchor -= 86400
        for w in range(weeks):
            ts = anchor - w * 7 * 86400
            lt = time.localtime(ts)
            # Snap to ~20:30 by overriding the hour/minute fields in the row.
            events.append({
                "ts": ts, "iso": "", "date": time.strftime("%Y-%m-%d", lt),
                "dow": time.strftime("%A", lt), "wd": 4, "hour": 20, "min": 30,
                "action": "netflix", "arg": "stranger things",
            })
        return events

    def test_compute_weekly_digest_from_jsonl_fallback(self):
        # No SQLite rows yet → digest falls back to JSONL scan.
        self._write_jsonl(self._seed_weekly_events(weeks=3))
        digest = self.mod.compute_weekly_digest(now=self.now)
        self.assertEqual(digest["lookback_days"], self.mod.WEEKLY_LOOKBACK_DAYS)
        clusters = digest["clusters"]
        self.assertTrue(clusters, "expected at least one weekly cluster")
        nf = next(c for c in clusters if c["action"] == "netflix")
        self.assertEqual(nf["dow"], 4)
        self.assertEqual(nf["hour_start"], 20)        # 20 // 2 * 2
        self.assertEqual(nf["weeks_seen"], 3)
        self.assertIn("Netflix", nf["label"])
        self.assertIn("Netflix", nf["offer"])
        self.assertEqual(nf["common_arg"], "stranger things")
        self.assertGreater(nf["confidence"], 0.0)

    def test_compute_weekly_digest_uses_sqlite_when_present(self):
        # Seed via log_event so rows land in SQLite, then compute from SQL.
        for e in self._seed_weekly_events(weeks=3):
            conn = self.mod._connect_db()
            try:
                conn.execute(
                    "INSERT INTO events(ts, iso, date, dow, hour, minute, action, arg) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (e["ts"], "", e["date"], e["wd"], e["hour"], e["min"],
                     e["action"], e["arg"]))
            finally:
                conn.close()
        # Empty JSONL — so a non-empty result proves the SQL path was used.
        self.assertFalse(os.path.exists(self.logp))
        digest = self.mod.compute_weekly_digest(now=self.now)
        self.assertTrue(any(c["action"] == "netflix" for c in digest["clusters"]))

    def test_compute_weekly_digest_below_min_weeks_filtered(self):
        # Only one week of data → weeks_seen (1) < WEEKLY_MIN_WEEKS (2): dropped.
        self._write_jsonl(self._seed_weekly_events(weeks=1))
        # Need >= WEEKLY_MIN_OCCURRENCES too; duplicate the single Friday 3x.
        rows = self._seed_weekly_events(weeks=1) * 3
        self._write_jsonl(rows)
        digest = self.mod.compute_weekly_digest(now=self.now)
        self.assertEqual(digest["clusters"], [])

    def test_compute_weekly_digest_empty_persists_empty(self):
        self.assertFalse(os.path.exists(self.logp))
        digest = self.mod.compute_weekly_digest(now=1_700_000_000.0)
        self.assertEqual(digest["clusters"], [])
        # An empty digest is still cached (round-trips via load_latest).
        loaded = self.mod.load_latest_weekly_digest()
        self.assertEqual(loaded["clusters"], [])

    def test_weekly_digest_save_and_load_roundtrip(self):
        self._write_jsonl(self._seed_weekly_events(weeks=3))
        computed = self.mod.compute_weekly_digest(now=self.now)
        loaded = self.mod.load_latest_weekly_digest()
        self.assertEqual(loaded["week_start"], computed["week_start"])
        self.assertEqual(len(loaded["clusters"]), len(computed["clusters"]))

    def test_load_latest_weekly_digest_none_when_empty_table(self):
        self.mod._connect_db().close()   # create schema, no rows
        self.assertEqual(self.mod.load_latest_weekly_digest(), {})

    def test_load_latest_weekly_digest_db_unavailable(self):
        with mock.patch.object(self.mod, "_connect_db", return_value=None):
            self.assertEqual(self.mod.load_latest_weekly_digest(), {})

    def test_load_latest_weekly_digest_corrupt_cluster_json(self):
        conn = self.mod._connect_db()
        try:
            conn.execute(
                "INSERT INTO weekly_summaries(week_start, computed_at, cluster_data) "
                "VALUES (?,?,?)", ("2026-05-25", 123.0, "{not json"))
        finally:
            conn.close()
        loaded = self.mod.load_latest_weekly_digest()
        self.assertEqual(loaded["clusters"], [])    # corrupt JSON → []
        self.assertEqual(loaded["week_start"], "2026-05-25")

    def test_save_weekly_digest_db_unavailable(self):
        with mock.patch.object(self.mod, "_connect_db", return_value=None):
            self.mod._save_weekly_digest({"week_start": "x", "computed_at": 0.0,
                                          "clusters": []})  # no raise

    def test_compute_weekly_digest_ignores_events_outside_lookback(self):
        now = 1_700_000_000.0
        old = now - (self.mod.WEEKLY_LOOKBACK_DAYS + 5) * 86400
        self._write_jsonl([self._event(old, action="netflix", wd=4, hour=20)
                           for _ in range(5)])
        digest = self.mod.compute_weekly_digest(now=now)
        self.assertEqual(digest["clusters"], [])


# ─── weekly digest label/offer helpers (media specialisations) ────────────
class WeeklyLabelOfferTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("pattern_learning")

    def test_monday_of_returns_monday(self):
        # 2026-06-03 is a Wednesday; its Monday is 2026-06-01.
        ts = time.mktime(time.struct_time((2026, 6, 3, 12, 0, 0, 2, 1, -1)))
        self.assertEqual(self.mod._monday_of(ts), "2026-06-01")

    def test_format_hour_band_wraps_midnight(self):
        out = self.mod._format_hour_band(23)
        self.assertIn("11 PM", out)
        self.assertIn("1 AM", out)   # (23+2) % 24 = 1

    def test_cluster_offer_music_with_arg(self):
        out = self.mod._cluster_offer(4, 20, "play_music", "michael jackson")
        self.assertIn("Michael Jackson", out)

    def test_cluster_offer_music_without_arg(self):
        out = self.mod._cluster_offer(4, 20, "play_music", "")
        self.assertIn("playlist", out.lower())

    def test_cluster_offer_check_teams(self):
        out = self.mod._cluster_offer(0, 8, "check_teams", "")
        self.assertIn("Teams", out)
        self.assertIn("Monday", out)

    def test_cluster_offer_morning_and_evening_briefing(self):
        m = self.mod._cluster_offer(0, 6, "morning_briefing", "")
        e = self.mod._cluster_offer(0, 18, "evening_briefing", "")
        self.assertIn("morning briefing", m)
        self.assertIn("evening briefing", e)

    def test_cluster_offer_generic_fallback(self):
        out = self.mod._cluster_offer(2, 14, "launch_app", "")
        self.assertIn("Wednesday", out)
        self.assertIn("Shall I proceed", out)

    def test_cluster_label_with_common_arg(self):
        label = self.mod._cluster_label(4, 20, "netflix", 3, 4, "stranger things")
        self.assertIn("Stranger Things", label)
        self.assertIn("3/4 weeks", label)


# ─── snapshot/state loader resilience + offer gating branches ─────────────
class LoaderResilienceTests(_TempDataBase):
    def test_load_aggregated_missing_returns_empty(self):
        self.assertEqual(self.mod._load_aggregated(), {})

    def test_load_aggregated_corrupt_returns_empty(self):
        with open(self.aggp, "w", encoding="utf-8") as f:
            f.write("{ not json")
        self.assertEqual(self.mod._load_aggregated(), {})

    def test_load_aggregated_non_dict_returns_empty(self):
        with open(self.aggp, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.mod._load_aggregated(), {})

    def test_load_offer_state_missing_returns_empty(self):
        self.assertEqual(self.mod._load_offer_state(), {})

    def test_load_offer_state_corrupt_returns_empty(self):
        with open(self.statep, "w", encoding="utf-8") as f:
            f.write("nope{")
        self.assertEqual(self.mod._load_offer_state(), {})

    def test_load_offer_state_non_dict_returns_empty(self):
        with open(self.statep, "w", encoding="utf-8") as f:
            json.dump(["a"], f)
        self.assertEqual(self.mod._load_offer_state(), {})

    def test_save_offer_state_roundtrip(self):
        self.mod._save_offer_state({"k": "2026-06-01"})
        self.assertEqual(self.mod._load_offer_state(), {"k": "2026-06-01"})

    def test_save_offer_state_failure_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._save_offer_state({"k": "v"})  # no raise

    def test_prune_state_drops_stale_entries(self):
        now = time.time()
        old_day = time.strftime("%Y-%m-%d", time.localtime(now - 200 * 86400))
        fresh_day = time.strftime("%Y-%m-%d", time.localtime(now))
        state = {"old": old_day, "fresh": fresh_day, "weird": 12345}
        self.mod._prune_state(state)
        self.assertNotIn("old", state)
        self.assertIn("fresh", state)
        # Non-string values are left untouched (defensive branch).
        self.assertIn("weird", state)

    def test_maybe_offer_skips_keyless_then_matches(self):
        # First match has no key (skipped), second is offered.
        snap = {
            "broad": [
                {"type": "broad", "bucket": "weekday", "hour_window": [9, 11],
                 "ratio": 0.9, "action": "check_teams", "offer": "Keyless",
                 "label": "L"},   # no 'key' → skipped
                {"key": "broad|check_teams|weekday|9-11", "type": "broad",
                 "bucket": "weekday", "hour_window": [9, 11], "ratio": 0.8,
                 "action": "check_teams", "offer": "Shall I check Teams, sir?",
                 "label": "L2"},
            ],
            "precise": [],
        }
        self.mod._atomic_write_json(self.aggp, snap)
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 10, 0, 2)):
            out = self.mod.maybe_pattern_offer_v2()
        self.assertIn("Teams", out)

    def test_maybe_offer_empty_when_match_has_no_offer_line(self):
        # A precise match whose composed line is empty yields "".
        snap = {"broad": [], "precise": [
            {"key": "p1", "type": "precise", "center_minute": 600,
             "tolerance_min": 5, "ratio": 0.9, "action": "noop", "offer": "",
             "label": "L"}]}
        self.mod._atomic_write_json(self.aggp, snap)
        with mock.patch.object(self.mod, "_offer_for", return_value=""), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 10, 0, 2)):
            self.assertEqual(self.mod.maybe_pattern_offer_v2(), "")

    def test_maybe_offer_empty_when_no_matches(self):
        snap = {"broad": [{"key": "b", "type": "broad", "bucket": "weekday",
                           "hour_window": [9, 11], "ratio": 0.8,
                           "action": "x", "offer": "y", "label": "L"}],
                "precise": []}
        self.mod._atomic_write_json(self.aggp, snap)
        # 3am off-hours → predictions_for_now returns [].
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 3, 0, 2)):
            self.assertEqual(self.mod.maybe_pattern_offer_v2(), "")


# ─── predictions_for_now extra branches ───────────────────────────────────
class PredictionsForNowBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("pattern_learning")

    def test_loads_snapshot_when_none_passed(self):
        with mock.patch.object(self.mod, "_load_aggregated", return_value={}):
            self.assertEqual(self.mod.predictions_for_now(None), [])

    def test_skips_broad_with_malformed_window(self):
        snap = {"broad": [
            {"type": "broad", "bucket": "weekday", "hour_window": [9],
             "ratio": 0.9, "action": "x"},          # len != 2 → skipped
            {"type": "broad", "bucket": "weekend", "hour_window": [9, 11],
             "ratio": 0.9, "action": "y"}],          # wrong bucket → skipped
            "precise": []}
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 10, 0, 2)):
            self.assertEqual(self.mod.predictions_for_now(snap), [])

    def test_skips_precise_with_non_int_center(self):
        snap = {"broad": [], "precise": [
            {"type": "precise", "center_minute": None, "tolerance_min": 5,
             "ratio": 0.9, "action": "x"}]}      # center not int → skipped
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 10, 0, 2)):
            self.assertEqual(self.mod.predictions_for_now(snap), [])


# ─── aggregate() deeper branches (precise rejection paths) ────────────────
class AggregatePreciseBranchTests(_TempDataBase):
    def test_precise_rejected_when_spread_too_wide(self):
        # check_teams fires daily but minutes are scattered across the hour, so
        # stddev > STDDEV_THRESHOLD_MIN → no precise prediction.
        base = time.time() - 14 * 86400
        events = []
        for day in range(14):
            ts = base + day * 86400
            lt = time.localtime(ts)
            events.append({
                "ts": ts, "date": time.strftime("%Y-%m-%d", lt),
                "dow": time.strftime("%A", lt), "wd": lt.tm_wday,
                "hour": 9, "min": (day * 7) % 60,    # wide spread
                "action": "check_teams", "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()
        self.assertFalse(any(p["action"] == "check_teams"
                             for p in snap["precise"]))

    def test_precise_rejected_when_ratio_below_threshold(self):
        # Tightly-timed but only on ~half the active days → ratio < 0.6.
        base = time.time() - 20 * 86400
        events = []
        for day in range(20):
            ts = base + day * 86400
            lt = time.localtime(ts)
            d = time.strftime("%Y-%m-%d", lt)
            # A daily "anchor" action on EVERY day so active-day count is high.
            events.append({"ts": ts, "date": d, "dow": "", "wd": lt.tm_wday,
                           "hour": 12, "min": 0, "action": "anchor", "arg": ""})
            if day % 2 == 0:   # check_teams only every other day
                events.append({"ts": ts, "date": d, "dow": "", "wd": lt.tm_wday,
                               "hour": 9, "min": 15, "action": "check_teams",
                               "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()
        self.assertFalse(any(p["action"] == "check_teams"
                             for p in snap["precise"]))

    def test_aggregate_skips_events_with_bad_fields(self):
        # Events missing/!int hour or out-of-range are ignored without raising.
        good_base = time.time() - 10 * 86400
        events = [
            {"ts": good_base, "date": "2026-05-01", "wd": "notint",
             "hour": 9, "min": 0, "action": "x", "arg": ""},     # wd not int
            {"ts": good_base, "date": "2026-05-01", "wd": 0,
             "hour": 99, "min": 0, "action": "x", "arg": ""},    # hour OOR
            {"ts": good_base, "date": "", "wd": 0, "hour": 9,
             "min": 0, "action": "x", "arg": ""},                # no date
            {"ts": good_base, "date": "2026-05-01", "wd": 0, "hour": 9,
             "min": 70, "action": "y", "arg": ""},               # minute OOR
        ]
        self._write_jsonl(events)
        snap = self.mod.aggregate()   # must not raise
        self.assertEqual(snap["broad"], [])
        self.assertEqual(snap["precise"], [])


# ─── _act_pattern_stats age-formatting branches ───────────────────────────
class PatternStatsAgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")

    def test_stats_minutes_ago(self):
        now = 1_700_000_000.0
        snap = {"events": 10, "days_span": 9.0, "generated_at": now - 5 * 60,
                "broad": [{}], "precise": []}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap), \
             mock.patch.object(self.mod, "_load_offer_state", return_value={}), \
             mock.patch.object(self.mod.time, "time", lambda: now):
            out = self.actions["pattern_stats"]("")
        self.assertIn("5 minutes ago", out)
        self.assertIn("1 broad", out)

    def test_stats_hours_ago_and_offered_count(self):
        now = 1_700_000_000.0
        # _act_pattern_stats derives "today" from time.localtime() with no arg
        # (real wall clock), so key the offered-state entries the same way.
        today = time.strftime("%Y-%m-%d", time.localtime())
        snap = {"events": 99, "days_span": 21.0,
                "generated_at": now - 3 * 3600,
                "broad": [{}, {}], "precise": [{}]}
        state = {"k1": today, "k2": today, "k3": "1999-01-01"}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap), \
             mock.patch.object(self.mod, "_load_offer_state", return_value=state), \
             mock.patch.object(self.mod.time, "time", lambda: now):
            out = self.actions["pattern_stats"]("")
        self.assertIn("3 hours ago", out)
        self.assertIn("2 offers surfaced today", out)

    def test_stats_singular_minute_and_offer(self):
        now = 1_700_000_000.0
        today = time.strftime("%Y-%m-%d", time.localtime())
        snap = {"events": 1, "days_span": 0.0, "generated_at": now - 60,
                "broad": [], "precise": []}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap), \
             mock.patch.object(self.mod, "_load_offer_state",
                               return_value={"k": today}), \
             mock.patch.object(self.mod.time, "time", lambda: now):
            out = self.actions["pattern_stats"]("")
        self.assertIn("1 minute ago", out)
        self.assertIn("1 offer surfaced today", out)


# ─── _act_pattern_predictions / aggregate count pluralisation ─────────────
class PatternActionPluralTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("pattern_learning")

    def test_aggregate_action_singular_counts(self):
        with mock.patch.object(self.mod, "aggregate",
                               return_value={"events": 7, "broad": [{}],
                                             "precise": [{}]}):
            out = self.actions["pattern_aggregate"]("")
        self.assertIn("1 broad pattern,", out)     # singular 'pattern'
        self.assertIn("1 precise pattern.", out)

    def test_predictions_action_caps_at_five(self):
        snap = {"events": 50, "days_span": 10.0,
                "broad": [{"label": f"b{i}"} for i in range(4)],
                "precise": [{"label": f"p{i}"} for i in range(4)]}
        with mock.patch.object(self.mod, "_load_aggregated", return_value=snap):
            out = self.actions["pattern_predictions"]("")
        # precise rendered first; only 5 labels total.
        self.assertIn("p0", out)
        self.assertIn("p3", out)
        self.assertIn("b0", out)        # 5th slot = first broad
        self.assertNotIn("b1", out)     # 6th+ dropped


# ─── _act_weekly_digest live (drives compute through temp stores) ─────────
class WeeklyDigestActionTests(_TempDataBase):
    def test_weekly_digest_action_no_data(self):
        out = self.actions["weekly_digest"]("")
        self.assertIn("no weekly habits", out.lower())

    def test_pattern_offer_now_bypasses_throttle(self):
        snap = {"broad": [], "precise": [
            {"key": "precise|check_teams|09:15", "type": "precise",
             "center_minute": 555, "tolerance_min": 5, "ratio": 0.9,
             "action": "check_teams", "offer": "Shall I check Teams, sir?",
             "label": "L"}]}
        self.mod._atomic_write_json(self.aggp, snap)
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2026, 6, 3, 9, 15, 2)):
            out = self.actions["pattern_offer_now"]("")
        self.assertIn("Teams", out)


# ─── background scheduler loop (bounded) ──────────────────────────────────
class AggregatorLoopTests(_TempDataBase):
    def test_loop_bootstraps_then_exits_on_sleep_break(self):
        # time.sleep raises on the 2nd call so we exercise the INITIAL_DELAY
        # sleep + bootstrap aggregate/digest, then break before the while-loop
        # body runs a second sleep. aggregate()/compute_weekly_digest are
        # stubbed so the loop's bootstrap branch is what we assert on.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt   # unwind out of the while True

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "aggregate") as agg, \
             mock.patch.object(self.mod, "compute_weekly_digest") as wk, \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               return_value={}), \
             mock.patch.object(self.mod, "_load_aggregated", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        # No cached snapshot/digest → both bootstrap recomputes fire.
        agg.assert_called()
        wk.assert_called()

    def test_loop_nightly_slot_triggers_aggregate(self):
        # Pin local time to NIGHTLY_HOUR with a stale snapshot so the nightly
        # branch fires inside the first while-iteration, then break.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            # 1st = INITIAL_DELAY, 2nd = top-of-loop sleep → after it, run body.
            if calls["n"] >= 3:
                raise KeyboardInterrupt

        nightly = _struct(2026, 6, 3, self.mod.NIGHTLY_HOUR, 0, 2)
        # Fresh snapshot for the bootstrap check (skip bootstrap recompute),
        # stale 'generated_at' so the in-loop age check (>20h) passes.
        boot_snap = {"generated_at": time.time()}
        loop_snap = {"generated_at": 0.0}
        snaps = [boot_snap, loop_snap, loop_snap]

        def _load_agg():
            return snaps.pop(0) if snaps else loop_snap

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod.time, "localtime", return_value=nightly), \
             mock.patch.object(self.mod, "_load_aggregated", side_effect=_load_agg), \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               return_value={"computed_at": time.time()}), \
             mock.patch.object(self.mod, "aggregate") as agg, \
             mock.patch.object(self.mod, "compute_weekly_digest"):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        agg.assert_called()

    def test_loop_swallows_scheduler_exception(self):
        # An exception inside the loop body is caught; the loop continues until
        # the next sleep raises to end the test.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] == 1:
                return                     # INITIAL_DELAY
            if calls["n"] == 2:
                return                     # top-of-loop sleep #1
            raise KeyboardInterrupt        # end after 2nd loop sleep

        def _boom():
            raise RuntimeError("localtime blew up")

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod.time, "localtime", side_effect=_boom), \
             mock.patch.object(self.mod, "_load_aggregated",
                               return_value={"generated_at": time.time()}), \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               return_value={"computed_at": time.time()}):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        # Reached at least the 3rd sleep → body ran and swallowed the error.
        self.assertGreaterEqual(calls["n"], 3)


# ─── register() ───────────────────────────────────────────────────────────
class RegisterTests(_TempDataBase):
    def test_register_populates_actions_and_starts_thread(self):
        import threading as _thr
        actions = {}
        started = {"v": False}

        # Neuter the daemon thread: construct it but make start() a no-op so the
        # real _aggregator_loop never runs (it would sleep/loop forever).
        with mock.patch.object(_thr.Thread, "start",
                               lambda self: started.__setitem__("v", True)), \
             mock.patch.object(self.mod, "_backfill_sqlite_from_jsonl_if_empty",
                               return_value=0):
            self.mod.register(actions)

        for name in ("pattern_predictions", "pattern_offer_now",
                     "pattern_aggregate", "pattern_stats", "weekly_digest"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))
        self.assertTrue(started["v"])

    def test_register_skips_duplicate_thread_when_already_running(self):
        import threading as _thr
        actions = {}

        class _FakeThread:
            name = "pattern-aggregator"

            def is_alive(self):
                return True

        with mock.patch.object(self.mod.threading, "enumerate",
                               return_value=[_FakeThread()]), \
             mock.patch.object(self.mod, "_backfill_sqlite_from_jsonl_if_empty",
                               return_value=0), \
             mock.patch.object(_thr.Thread, "start") as start:
            self.mod.register(actions)
        # Existing live aggregator → no new thread started.
        start.assert_not_called()
        self.assertIn("pattern_stats", actions)

    def test_register_swallows_backfill_failure(self):
        import threading as _thr
        actions = {}
        with mock.patch.object(self.mod, "_backfill_sqlite_from_jsonl_if_empty",
                               side_effect=RuntimeError("backfill boom")), \
             mock.patch.object(_thr.Thread, "start", lambda self: None):
            self.mod.register(actions)   # must not raise
        self.assertIn("pattern_aggregate", actions)


# ─── atomic write / ensure-dir helpers ────────────────────────────────────
class AtomicWriteTests(_TempDataBase):
    def test_atomic_write_json_roundtrip(self):
        target = os.path.join(self.tmp, "sub", "out.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.mod._atomic_write_json(target, {"a": 1, "b": [2, 3]})
        with open(target, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1, "b": [2, 3]})

    def test_atomic_write_json_cleans_temp_on_dump_failure(self):
        target = os.path.join(self.tmp, "out.json")
        before = set(os.listdir(self.tmp))
        # A non-serialisable payload makes json.dump raise; the .tmp must be
        # unlinked and the error re-raised (no orphan temp file left behind).
        with self.assertRaises(TypeError):
            self.mod._atomic_write_json(target, {"bad": object()})
        after = set(os.listdir(self.tmp))
        self.assertEqual(before, after)        # no leftover .tmp

    def test_ensure_data_dir_swallows_makedirs_error(self):
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("denied")):
            self.mod._ensure_data_dir()    # no raise

    def test_atomic_write_unlink_failure_during_cleanup_reraises(self):
        # json.dump fails AND the tmp-file unlink in the except also fails; the
        # original error must still propagate (inner except is swallowed).
        target = os.path.join(self.tmp, "out.json")
        with mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("cannot unlink")), \
             self.assertRaises(TypeError):
            self.mod._atomic_write_json(target, {"bad": object()})


# ─── remaining defensive / edge branches ──────────────────────────────────
class EdgeBranchTests(_TempDataBase):
    def test_aggregate_broad_window_clipped_at_hour_23(self):
        # An event at hour 23 means window_start can reach 23 → window_end 25,
        # which is > 24 and must be skipped (the only 23:xx window kept is
        # 22-24). Drive >= MIN_DAYS_OBSERVED weekday days so a broad emits.
        events = []
        base = time.time() - 30 * 86400
        seen = 0
        day = 0
        while seen < 9:
            ts = base + day * 86400
            lt = time.localtime(ts)
            day += 1
            if lt.tm_wday >= 5:
                continue
            seen += 1
            events.append({"ts": ts, "date": time.strftime("%Y-%m-%d", lt),
                           "dow": "", "wd": lt.tm_wday, "hour": 23, "min": 30,
                           "action": "check_system", "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()   # must not raise on the hour-23 window
        win = next((p for p in snap["broad"] if p["action"] == "check_system"),
                   None)
        if win is not None:
            # If a window survived it is the 22-24 band, never 23-25.
            self.assertEqual(win["hour_window"], [22, 24])

    def test_aggregate_precise_skips_event_with_bad_minute(self):
        # Build a clean precise cluster for "check_teams", plus one stray row
        # for the SAME action with an out-of-range minute that the precise loop
        # must skip (covers the per-event guard inside the precise pass).
        events = []
        base = time.time() - 20 * 86400
        for day in range(14):
            ts = base + day * 86400
            lt = time.localtime(ts)
            d = time.strftime("%Y-%m-%d", lt)
            events.append({"ts": ts, "date": d, "dow": "", "wd": lt.tm_wday,
                           "hour": 9, "min": 15, "action": "check_teams",
                           "arg": ""})
        # Stray bad-minute row (min=99) — survives broad guards? No: skipped in
        # both passes. Its job is to exercise the precise-loop minute guard.
        events.append({"ts": base, "date": "2026-01-01", "dow": "", "wd": 0,
                       "hour": 9, "min": 99, "action": "check_teams", "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()
        teams = next((p for p in snap["precise"]
                      if p["action"] == "check_teams"), None)
        self.assertIsNotNone(teams)
        # The bad-minute row was excluded from the cluster math.
        self.assertEqual(teams["center_clock"], "09:15")

    def test_aggregate_precise_rejected_low_distinct_days(self):
        # Many tightly-timed occurrences but on FEWER than MIN_OCCURRENCES_PRECISE
        # DISTINCT days → days_with_event guard rejects it (line ~588-589). Pad
        # with an 'anchor' action so total active-day count clears MIN_DAYS.
        events = []
        base = time.time() - 20 * 86400
        for day in range(10):     # 10 distinct active days via anchor
            ts = base + day * 86400
            lt = time.localtime(ts)
            d = time.strftime("%Y-%m-%d", lt)
            events.append({"ts": ts, "date": d, "dow": "", "wd": lt.tm_wday,
                           "hour": 12, "min": 0, "action": "anchor", "arg": ""})
        # check_teams: 6 occurrences but only on 2 distinct days.
        for i in range(6):
            day = i % 2          # only days 0 and 1
            ts = base + day * 86400
            d = time.strftime("%Y-%m-%d", time.localtime(ts))
            events.append({"ts": ts, "date": d, "dow": "", "wd": 0,
                           "hour": 9, "min": 15, "action": "check_teams",
                           "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()
        self.assertFalse(any(p["action"] == "check_teams"
                             for p in snap["precise"]))

    def test_aggregate_precise_rejected_when_few_active_days(self):
        # Tight cluster on 4 distinct days, but the WHOLE log spans only 4
        # active days (< MIN_DAYS_OBSERVED=7) → all_active_days guard rejects.
        events = []
        base = time.time() - 6 * 86400
        for day in range(4):
            ts = base + day * 86400
            d = time.strftime("%Y-%m-%d", time.localtime(ts))
            for _ in range(2):     # ensure >= MIN_OCCURRENCES_PRECISE total
                events.append({"ts": ts, "date": d, "dow": "", "wd": 0,
                               "hour": 9, "min": 15, "action": "check_teams",
                               "arg": ""})
        self._write_jsonl(events)
        snap = self.mod.aggregate()
        self.assertEqual(snap["precise"], [])

    def test_prune_state_swallows_internal_error(self):
        # time.strftime raising inside _prune_state is caught; state unchanged.
        state = {"k": "2026-01-01"}
        with mock.patch.object(self.mod.time, "strftime",
                               side_effect=ValueError("bad fmt")):
            self.mod._prune_state(state)   # no raise
        self.assertEqual(state, {"k": "2026-01-01"})

    def test_weekly_digest_sql_error_falls_back_to_jsonl(self):
        # SQLite query raises → digest falls back to the JSONL scan and still
        # produces clusters (covers the weekly-digest SQL except branch).
        now = 1_700_000_000.0
        anchor = now
        for _ in range(8):
            if time.localtime(anchor).tm_wday == 4:
                break
            anchor -= 86400
        rows = []
        for w in range(3):
            ts = anchor - w * 7 * 86400
            d = time.strftime("%Y-%m-%d", time.localtime(ts))
            rows.append({"ts": ts, "iso": "", "date": d, "dow": "Friday",
                         "wd": 4, "hour": 20, "min": 30, "action": "netflix",
                         "arg": "show"})
        self._write_jsonl(rows)

        class _RaisingConn:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("query boom")

            def close(self):
                pass

        # First _connect_db (the SELECT) raises; later ones (save) work.
        real_connect = self.mod._connect_db
        calls = {"n": 0}

        def _conn():
            calls["n"] += 1
            if calls["n"] == 1:
                return _RaisingConn()
            return real_connect()

        with mock.patch.object(self.mod, "_connect_db", side_effect=_conn):
            digest = self.mod.compute_weekly_digest(now=now)
        self.assertTrue(any(c["action"] == "netflix"
                            for c in digest["clusters"]))

    def test_loop_weekly_slot_triggers_digest(self):
        # Pin local time to the weekly slot (Mon @ WEEKLY_DIGEST_HOUR) with a
        # stale cached digest so the weekly branch fires, then break out.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise KeyboardInterrupt

        # Monday = wd 0. WEEKLY_DIGEST_DOW is 0, WEEKLY_DIGEST_HOUR is 9, but the
        # nightly hour is 3 — pin hour to the weekly hour so only weekly fires.
        weekly_when = _struct(2026, 6, 1, self.mod.WEEKLY_DIGEST_HOUR, 0, 0)

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=weekly_when), \
             mock.patch.object(self.mod, "_load_aggregated",
                               return_value={"generated_at": time.time()}), \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               return_value={"computed_at": 0.0}), \
             mock.patch.object(self.mod, "aggregate"), \
             mock.patch.object(self.mod, "compute_weekly_digest") as wk:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        wk.assert_called()

    def test_loop_bootstrap_recomputes_when_snapshot_stale(self):
        # Bootstrap branch: a cached snapshot older than MAX_AGGREGATE_AGE_SECS
        # forces an immediate aggregate() before the while-loop. Break on the
        # 2nd sleep (top of loop) so we only assert the bootstrap.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt

        stale = {"generated_at": 1.0}   # ancient
        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_load_aggregated", return_value=stale), \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               return_value={"computed_at": time.time()}), \
             mock.patch.object(self.mod, "aggregate") as agg, \
             mock.patch.object(self.mod, "compute_weekly_digest"):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        agg.assert_called()

    def test_loop_initial_aggregate_failure_swallowed(self):
        # The bootstrap aggregate() raising must be caught so the loop proceeds
        # to the weekly bootstrap and then the while-loop.
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt

        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_load_aggregated",
                               side_effect=RuntimeError("agg load boom")), \
             mock.patch.object(self.mod, "load_latest_weekly_digest",
                               side_effect=RuntimeError("wk load boom")):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._aggregator_loop()
        # Got past both bootstrap try-blocks to the loop's 2nd sleep.
        self.assertGreaterEqual(calls["n"], 2)


# ─── more SQLite/IO error-path coverage ───────────────────────────────────
class SqliteErrorPathTests(_TempDataBase):
    def test_backfill_skips_unserialisable_row_but_commits_rest(self):
        # One JSONL row makes the INSERT raise (per-row try/except → continue);
        # the others still backfill and the transaction commits.
        good = [self._event(1_700_000_000.0 + i, action="check_teams")
                for i in range(3)]
        self._write_jsonl(good)
        real_conn = self.mod._connect_db

        class _Wrap:
            def __init__(self, inner):
                self._inner = inner
                self._n = 0

            def execute(self, sql, *a, **k):
                if sql.startswith("INSERT INTO events"):
                    self._n += 1
                    if self._n == 2:      # 2nd row blows up
                        raise sqlite3.IntegrityError("bad row")
                return self._inner.execute(sql, *a, **k)

            def close(self):
                self._inner.close()

        def _factory():
            return _Wrap(real_conn())

        with mock.patch.object(self.mod, "_connect_db", side_effect=_factory):
            inserted = self.mod._backfill_sqlite_from_jsonl_if_empty()
        self.assertEqual(inserted, 2)        # 3 attempted, 1 skipped
        self.assertEqual(self._db_count(), 2)

    def test_backfill_rolls_back_on_commit_failure(self):
        self._write_jsonl([self._event(1.0)])
        real_conn = self.mod._connect_db

        class _Wrap:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *a, **k):
                if sql == "COMMIT":
                    raise sqlite3.OperationalError("commit boom")
                return self._inner.execute(sql, *a, **k)

            def close(self):
                self._inner.close()

        with mock.patch.object(self.mod, "_connect_db",
                               side_effect=lambda: _Wrap(real_conn())):
            # Outer except catches the re-raised COMMIT error → returns 0.
            self.assertEqual(self.mod._backfill_sqlite_from_jsonl_if_empty(), 0)

    def test_prune_sqlite_events_query_error_swallowed(self):
        real_conn = self.mod._connect_db

        class _Wrap:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *a, **k):
                if sql.startswith("SELECT COUNT"):
                    raise sqlite3.OperationalError("count boom")
                return self._inner.execute(sql, *a, **k)

            def close(self):
                self._inner.close()

        with mock.patch.object(self.mod, "_connect_db",
                               side_effect=lambda: _Wrap(real_conn())):
            self.mod._prune_sqlite_events(max_rows=1)   # no raise

    def test_load_events_outer_open_error_returns_empty(self):
        # The file exists (so we pass the existence check) but open() raises:
        # the outer try/except returns [].
        self._write_jsonl([self._event(1.0)])
        with mock.patch.object(self.mod, "open",
                               side_effect=OSError("locked"), create=True):
            self.assertEqual(self.mod._load_events(), [])

    def test_weekly_digest_jsonl_fallback_skips_malformed_rows(self):
        # No SQLite rows → JSONL fallback. Mix valid Friday-netflix rows with
        # malformed ones (bad ts/dow/action) that the fallback loop must skip.
        now = 1_700_000_000.0
        anchor = now
        for _ in range(8):
            if time.localtime(anchor).tm_wday == 4:
                break
            anchor -= 86400
        rows = []
        for w in range(3):
            ts = anchor - w * 7 * 86400
            d = time.strftime("%Y-%m-%d", time.localtime(ts))
            rows.append({"ts": ts, "iso": "", "date": d, "dow": "Friday",
                         "wd": 4, "hour": 20, "min": 30, "action": "netflix",
                         "arg": "show"})
        rows.append({"ts": "notnum", "wd": 4, "hour": 20, "action": "x"})  # bad ts
        rows.append({"ts": now, "wd": "x", "hour": 20, "action": "y"})      # bad wd
        rows.append({"ts": now, "wd": 4, "hour": 20, "action": 123})        # bad action
        self._write_jsonl(rows)
        digest = self.mod.compute_weekly_digest(now=now)
        self.assertTrue(any(c["action"] == "netflix" for c in digest["clusters"]))

    def test_weekly_digest_filters_below_min_occurrences(self):
        # Two distinct weeks (clears WEEKLY_MIN_WEEKS) but only 2 total hits
        # (< WEEKLY_MIN_OCCURRENCES=3) → cluster dropped by the occ guard.
        now = 1_700_000_000.0
        anchor = now
        for _ in range(8):
            if time.localtime(anchor).tm_wday == 4:
                break
            anchor -= 86400
        rows = []
        for w in range(2):
            ts = anchor - w * 7 * 86400
            d = time.strftime("%Y-%m-%d", time.localtime(ts))
            rows.append({"ts": ts, "iso": "", "date": d, "dow": "Friday",
                         "wd": 4, "hour": 20, "min": 30, "action": "netflix",
                         "arg": ""})
        self._write_jsonl(rows)
        digest = self.mod.compute_weekly_digest(now=now)
        self.assertEqual(digest["clusters"], [])


if __name__ == "__main__":
    unittest.main()
