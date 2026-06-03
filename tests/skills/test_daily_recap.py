"""Logic tests for skills/daily_recap.py.

Covers the pure formatting helpers (app-name normaliser, number word, duration
formatter, title-caser), the session-log miner's regex extraction, the
pattern-JSONL supplement, today's-tasks counter, and the big _build_recap
text-assembly across its many branches (top app, prints, Teams, music, tasks,
empty-day fallback) with _scan_session_logs mocked. No real logs/network.

Also covers the I/O helpers (speech queue, config + persistent state, today's
log/pattern readers), the card popup, the manual + scheduled fire paths, the
register() wiring, and the background scheduler loop driven through a sleep that
raises (so the daemon's branches run without ever blocking).
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
import time
import unittest
from collections import Counter
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _empty_report(**over):
    rpt = {
        "voice_count": 0, "action_counts": Counter(), "music_titles": Counter(),
        "app_minutes": Counter(), "teams_alerts": 0, "teams_vips": Counter(),
        "print_started": 0, "print_finished": 0, "print_failed": 0,
    }
    rpt.update(over)
    return rpt


class DailyRecapTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_recap")

    # ── _normalize_app_name (pure) ───────────────────────────────────────
    def test_normalize_app_name(self):
        n = self.mod._normalize_app_name
        self.assertEqual(n("Bambu Studio - part.3mf"), "Bambu Studio")
        self.assertEqual(n("some VSCode window"), "VS Code")
        self.assertEqual(n("Microsoft Teams | Chat"), "Microsoft Teams")
        self.assertEqual(n("random notepad"), "")
        self.assertEqual(n(""), "")

    # ── _number_word / _format_duration_minutes / _titlecase (pure) ──────
    def test_number_word(self):
        self.assertEqual(self.mod._number_word(1), "one")
        self.assertEqual(self.mod._number_word(11), "eleven")
        self.assertEqual(self.mod._number_word(42), "42")

    def test_format_duration_minutes(self):
        f = self.mod._format_duration_minutes
        self.assertEqual(f(0), "")
        self.assertEqual(f(5), "5 minutes")
        self.assertEqual(f(1), "1 minute")
        self.assertEqual(f(60), "1 hour")
        self.assertEqual(f(160), "2 hours 40 minutes")

    def test_titlecase(self):
        self.assertEqual(self.mod._titlecase("michael jackson essentials"),
                         "Michael Jackson Essentials")
        self.assertEqual(self.mod._titlecase("best of the doors"),
                         "Best of the Doors")   # small words stay lowercase mid-string

    # ── _scan_session_logs regex mining ──────────────────────────────────
    def test_scan_session_logs_extracts_actions_music_voice(self):
        log = (
            "[09:00:01] [action] play_music: 'Michael Jackson Essentials'\n"
            "  You:    play something\n"
            "[09:01:00] [action] focus_window: Bambu Studio - bracket.3mf\n"
            "Incoming call from Sam Industries on Teams\n"
            "Print complete, sir\n"
        )
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["s.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            rpt = self.mod._scan_session_logs()
        self.assertEqual(rpt["voice_count"], 1)
        self.assertEqual(rpt["action_counts"]["play_music"], 1)
        self.assertEqual(rpt["music_titles"]["michael jackson essentials"], 1)
        self.assertEqual(rpt["print_finished"], 1)
        self.assertEqual(rpt["teams_alerts"], 1)
        self.assertEqual(rpt["teams_vips"]["Sam Industries"], 1)
        self.assertGreaterEqual(rpt["app_minutes"]["Bambu Studio"], 1)

    # ── _supplement_with_pattern_jsonl ───────────────────────────────────
    def test_supplement_with_pattern_jsonl(self):
        events = [{"action": "play_music", "arg": "The Doors"},
                  {"action": "see_screen", "arg": ""}]
        rpt = _empty_report()
        with mock.patch.object(self.mod, "_todays_pattern_events", return_value=events):
            self.mod._supplement_with_pattern_jsonl(rpt)
        self.assertEqual(rpt["action_counts"]["play_music"], 1)
        self.assertEqual(rpt["music_titles"]["the doors"], 1)

    # ── _count_tasks_completed_today ─────────────────────────────────────
    def test_count_tasks_completed_today(self):
        today = time.strftime("%Y-%m-%d")
        todo = f"- [x] shipped {today}\n- [ ] todo {today}\n- [x] old 1999-01-01\n"
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=todo)):
            self.assertEqual(self.mod._count_tasks_completed_today(), 1)

    # ── _build_recap assembly ────────────────────────────────────────────
    def test_build_recap_rich_day(self):
        report = _empty_report(
            app_minutes=Counter({"Bambu Studio": 160}),
            print_finished=1,
            teams_alerts=4, teams_vips=Counter({"Sam": 4}),
            action_counts=Counter({"play_music": 11}),
            music_titles=Counter({"michael jackson essentials": 11}),
        )
        with mock.patch.object(self.mod, "_scan_session_logs", return_value=report), \
             mock.patch.object(self.mod, "_supplement_with_pattern_jsonl"), \
             mock.patch.object(self.mod, "_bambu_now", return_value={}), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=0):
            out = self.mod._build_recap()
        self.assertTrue(out.startswith("[intent:briefing]"))
        self.assertIn("2 hours 40 minutes in Bambu Studio", out)
        self.assertIn("completed one print", out)
        self.assertIn("took four Teams calls including one from Sam", out)
        self.assertIn("11 Michael Jackson Essentials tracks", out)
        self.assertIn("Shall I queue the same morning briefing for tomorrow?", out)

    def test_build_recap_empty_day_uses_voice_fallback(self):
        report = _empty_report(voice_count=0)
        with mock.patch.object(self.mod, "_scan_session_logs", return_value=report), \
             mock.patch.object(self.mod, "_supplement_with_pattern_jsonl"), \
             mock.patch.object(self.mod, "_bambu_now", return_value={}), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=0):
            out = self.mod._build_recap()
        self.assertIn("nothing of note made it onto the record today, sir", out)

    def test_build_recap_in_flight_print(self):
        report = _empty_report()
        bambu = {"gcode_state": "RUNNING", "filename": "gear.3mf"}
        with mock.patch.object(self.mod, "_scan_session_logs", return_value=report), \
             mock.patch.object(self.mod, "_supplement_with_pattern_jsonl"), \
             mock.patch.object(self.mod, "_bambu_now", return_value=bambu), \
             mock.patch.object(self.mod, "_bambu_strip", return_value="gear"), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=0):
            out = self.mod._build_recap()
        self.assertIn("the H2D is still printing 'gear'", out)

    def test_build_recap_tasks_and_failed_print(self):
        report = _empty_report(print_failed=2)
        with mock.patch.object(self.mod, "_scan_session_logs", return_value=report), \
             mock.patch.object(self.mod, "_supplement_with_pattern_jsonl"), \
             mock.patch.object(self.mod, "_bambu_now", return_value={}), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=3):
            out = self.mod._build_recap()
        self.assertIn("had two prints fail on you", out)
        self.assertIn("cleared three tasks from the queue", out)

    # ── _bambu_now / _bambu_strip cross-skill reads ──────────────────────
    def test_bambu_now_reads_state(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"gcode_state": "FINISH", "filename": "x.3mf"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_now()["gcode_state"], "FINISH")

    def test_bambu_now_absent(self):
        import sys
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_bambu_monitor", None)
            self.assertEqual(self.mod._bambu_now(), {})

    # ── daily_recap action ───────────────────────────────────────────────
    def test_action_returns_and_enqueues(self):
        mod, actions = load_skill_isolated("daily_recap")
        with mock.patch.object(mod, "_build_recap", return_value="[intent:briefing] Sir, today..."), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_show_card_safe"), \
             mock.patch.object(mod, "_save_last_fired_date"):
            out = actions["daily_recap"]("")
        self.assertIn("Sir, today", out)
        enq.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("daily_recap")
        with mock.patch.object(mod, "_build_recap", side_effect=RuntimeError("boom")):
            out = actions["daily_recap"]("")
        self.assertIn("failed", out.lower())


class EnqueueSpeechTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_enqueue_routes_through_bobert_announcer(self):
        bc = mock.MagicMock()
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            self.mod._enqueue_speech("hello sir")
        bc.proactive_announce.assert_called_once_with("hello sir", source="recap")

    def test_enqueue_falls_back_to_atomic_write_when_no_announcer(self):
        # bobert_companion importable but lacks proactive_announce → file path.
        bc = mock.MagicMock(spec=[])  # no attributes → getattr returns None
        with tempfile.TemporaryDirectory() as d:
            queue = os.path.join(d, "pending_speech.json")
            with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", queue):
                self.mod._enqueue_speech("queued line")
            with open(queue, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data[-1]["message"], "queued line")

    def test_enqueue_appends_to_existing_queue(self):
        bc = mock.MagicMock(spec=[])
        with tempfile.TemporaryDirectory() as d:
            queue = os.path.join(d, "pending_speech.json")
            with open(queue, "w", encoding="utf-8") as f:
                json.dump([{"ts": 1.0, "message": "old"}], f)
            with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", queue):
                self.mod._enqueue_speech("new line")
            with open(queue, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual([d["message"] for d in data], ["old", "new line"])

    def test_enqueue_recovers_from_corrupt_queue_file(self):
        bc = mock.MagicMock(spec=[])
        with tempfile.TemporaryDirectory() as d:
            queue = os.path.join(d, "pending_speech.json")
            with open(queue, "w", encoding="utf-8") as f:
                f.write("{ not json")
            with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", queue):
                self.mod._enqueue_speech("after corruption")
            with open(queue, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data[-1]["message"], "after corruption")

    def test_enqueue_write_failure_is_swallowed(self):
        bc = mock.MagicMock(spec=[])
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._enqueue_speech("doomed")  # must not raise

    def test_enqueue_announcer_raising_falls_back_to_file(self):
        # import_module succeeds, announcer is callable but raises → the broad
        # except swallows it and we drop through to the atomic file write.
        bc = mock.MagicMock()
        bc.proactive_announce.side_effect = RuntimeError("announce broke")
        with tempfile.TemporaryDirectory() as d:
            queue = os.path.join(d, "pending_speech.json")
            with mock.patch.object(self.mod.importlib, "import_module", return_value=bc), \
                 mock.patch.object(self.mod, "_SPEECH_QUEUE", queue):
                self.mod._enqueue_speech("recovered line")
            with open(queue, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data[-1]["message"], "recovered line")


class ConfigAndStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_read_config_from_bobert(self):
        bc = mock.MagicMock()
        bc.DAILY_RECAP_ENABLED = False
        bc.DAILY_RECAP_HOUR = 7
        bc.DAILY_RECAP_MINUTE = 5
        with mock.patch.object(self.mod.importlib, "import_module", return_value=bc):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": False, "hour": 7, "minute": 5})

    def test_read_config_defaults_when_bobert_absent(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no bobert")):
            cfg = self.mod._read_config()
        self.assertEqual(cfg, {"enabled": True, "hour": 22, "minute": 30})

    def test_load_last_fired_date_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_load_last_fired_date_reads_value(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data='{"last_fired_date": "2026-06-01"}')):
            self.assertEqual(self.mod._load_last_fired_date(), "2026-06-01")

    def test_load_last_fired_date_corrupt_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{not json")):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_save_last_fired_date_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "daily_recap_state.json")
            with mock.patch.object(self.mod, "_STATE_FILE", state):
                self.mod._save_last_fired_date("2026-06-02")
                self.assertEqual(self.mod._load_last_fired_date(), "2026-06-02")

    def test_save_last_fired_date_failure_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("nope")):
            self.mod._save_last_fired_date("2026-06-02")  # no raise


class TodaysReadersTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_todays_log_paths_globs_sorted(self):
        with mock.patch.object(self.mod.glob, "glob",
                               return_value=["b.log", "a.log"]) as g:
            out = self.mod._todays_log_paths()
        self.assertEqual(out, ["a.log", "b.log"])
        g.assert_called_once()

    def test_todays_pattern_events_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._todays_pattern_events(), [])

    def test_todays_pattern_events_filters_to_today(self):
        today = datetime.date.today().isoformat()
        data = (json.dumps({"date": today, "action": "play_music"}) + "\n"
                + json.dumps({"date": "1999-01-01", "action": "old"}) + "\n"
                + "\n"                       # blank line skipped
                + "{ corrupt line\n")        # corrupt skipped
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=data)):
            out = self.mod._todays_pattern_events()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["action"], "play_music")

    def test_todays_pattern_events_read_error_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._todays_pattern_events(), [])

    def test_count_tasks_missing_file_zero(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_tasks_completed_today(), 0)

    def test_count_tasks_read_error_zero(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._count_tasks_completed_today(), 0)


class ScanSessionLogsDeepTests(unittest.TestCase):
    """Exercise the launch_app / open_url / dwell-remark / print-fail / read-fail
    branches of the miner that the headline test doesn't reach."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_launch_app_open_url_and_dwell_and_failed_print(self):
        log = (
            "[10:00:00] [action] launch_app: OrcaSlicer\n"
            "[10:05:00] [action] open_url: https://news.example.com\n"
            "[10:06:00] [action] open_url: youtube.com/watch\n"
            "you've been in Fusion 360 for 2 hours and 10 minutes, sir\n"
            "Print started, sir\n"
            "the print has failed, sir\n"
        )
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["s.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            rpt = self.mod._scan_session_logs()
        # launch_app credited OrcaSlicer; open_url with no app-hint → "Browser".
        self.assertGreaterEqual(rpt["app_minutes"]["OrcaSlicer"], 1)
        self.assertGreaterEqual(rpt["app_minutes"]["Browser"], 1)
        self.assertGreaterEqual(rpt["app_minutes"]["YouTube"], 1)
        # Dwell remark synthesises 2*60 minute-buckets for Fusion 360.
        self.assertEqual(rpt["app_minutes"]["Fusion 360"], 120)
        self.assertEqual(rpt["print_started"], 1)
        self.assertEqual(rpt["print_failed"], 1)

    def test_scan_swallows_unreadable_log(self):
        # open() raising for a path is caught per-file (continue).
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["bad.log"]), \
             mock.patch("builtins.open", side_effect=OSError("permission denied")):
            rpt = self.mod._scan_session_logs()
        self.assertEqual(rpt["voice_count"], 0)


class SupplementEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_supplement_noop_when_no_events(self):
        rpt = _empty_report()
        with mock.patch.object(self.mod, "_todays_pattern_events", return_value=[]):
            self.mod._supplement_with_pattern_jsonl(rpt)
        self.assertEqual(rpt["action_counts"], Counter())

    def test_supplement_skips_event_without_action(self):
        rpt = _empty_report()
        events = [{"action": "", "arg": "x"}]
        with mock.patch.object(self.mod, "_todays_pattern_events", return_value=events):
            self.mod._supplement_with_pattern_jsonl(rpt)
        self.assertEqual(rpt["action_counts"], Counter())


class BambuCrossSkillTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_bambu_now_uses_lock_when_present(self):
        fake = mock.MagicMock()
        fake._state_lock = mock.MagicMock()
        fake._state_lock.__enter__ = mock.MagicMock(return_value=None)
        fake._state_lock.__exit__ = mock.MagicMock(return_value=False)
        fake._state = {"gcode_state": "RUNNING"}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_now()
        self.assertEqual(out["gcode_state"], "RUNNING")
        fake._state_lock.__enter__.assert_called_once()

    def test_bambu_now_state_none_returns_empty(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = None
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_now(), {})

    def test_bambu_now_exception_returns_empty(self):
        fake = mock.MagicMock()
        type(fake)._state = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_now(), {})

    def test_bambu_strip_absent_returns_input(self):
        sys.modules.pop("skill_bambu_monitor", None)
        self.assertEqual(self.mod._bambu_strip("gear.3mf"), "gear.3mf")

    def test_bambu_strip_delegates_to_monitor(self):
        fake = mock.MagicMock()
        fake._strip_filename.return_value = "gear"
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_strip("gear.3mf"), "gear")

    def test_bambu_strip_exception_returns_input(self):
        fake = mock.MagicMock()
        fake._strip_filename.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_strip("gear.3mf"), "gear.3mf")


class BuildRecapBranchTests(unittest.TestCase):
    """Cover the music-count, multi-print, in-flight-no-name, single-task and
    voice-fallback phrasings of _build_recap not hit by the headline cases."""

    def _run(self, report, bambu=None, strip="x", tasks=0):
        with mock.patch.object(self.mod, "_scan_session_logs", return_value=report), \
             mock.patch.object(self.mod, "_supplement_with_pattern_jsonl"), \
             mock.patch.object(self.mod, "_bambu_now", return_value=bambu or {}), \
             mock.patch.object(self.mod, "_bambu_strip", return_value=strip), \
             mock.patch.object(self.mod, "_count_tasks_completed_today",
                               return_value=tasks):
            return self.mod._build_recap()

    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_music_generic_count_when_no_dominant_title(self):
        # 4 plays spread over 4 distinct titles → no title >= max(2,2)=2, so the
        # generic "played N tracks" branch fires.
        report = _empty_report(
            action_counts=Counter({"play_music": 4}),
            music_titles=Counter({"a": 1, "b": 1, "c": 1, "d": 1}),
        )
        out = self._run(report)
        self.assertIn("played 4 tracks", out)

    def test_music_single_track_no_titles(self):
        report = _empty_report(action_counts=Counter({"play_music": 1}))
        out = self._run(report)
        self.assertIn("played one track", out)

    def test_music_multiple_no_titles(self):
        report = _empty_report(action_counts=Counter({"play_music": 3}))
        out = self._run(report)
        self.assertIn("played 3 tracks", out)

    def test_completed_multiple_prints(self):
        report = _empty_report(print_finished=2)
        out = self._run(report)
        self.assertIn("completed two prints", out)

    def test_single_failed_print(self):
        report = _empty_report(print_failed=1)
        out = self._run(report)
        self.assertIn("had one print fail on you", out)

    def test_in_flight_without_filename(self):
        report = _empty_report()
        out = self._run(report, bambu={"gcode_state": "PREPARE", "filename": ""},
                        strip="")
        self.assertIn("the H2D is still in flight", out)

    def test_single_teams_call(self):
        report = _empty_report(teams_alerts=1)
        out = self._run(report)
        self.assertIn("took one Teams call", out)

    def test_single_task_cleared(self):
        report = _empty_report()
        out = self._run(report, tasks=1)
        self.assertIn("cleared one task from the queue", out)

    def test_single_voice_interaction_fallback(self):
        report = _empty_report(voice_count=1)
        out = self._run(report)
        self.assertIn("just one voice interaction reached me today", out)

    def test_many_voice_interactions_fallback(self):
        report = _empty_report(voice_count=12)
        out = self._run(report)
        self.assertIn("12 voice interactions reached me today", out)

    def test_single_part_only_uses_one_chunk_sentence(self):
        # Exactly one chunk (top app, below-10-min is filtered, so use prints).
        report = _empty_report(print_finished=1)
        out = self._run(report)
        # len(parts)==1 path: "Sir, today completed one print."
        self.assertIn("Sir, today completed one print.", out)

    def test_top_app_below_threshold_is_omitted(self):
        # 5 minutes of samples < 10 → no app chunk; falls to voice fallback.
        report = _empty_report(app_minutes=Counter({"Bambu Studio": 5}),
                               voice_count=2)
        out = self._run(report)
        self.assertNotIn("Bambu Studio", out)
        self.assertIn("2 voice interactions", out)

    def test_in_flight_chunk_is_tail_uses_and_join(self):
        # App + in-flight print → the tail chunk starts with "and ", exercising
        # the comma-join branch at line ~644 (the tail already carries its own
        # "and ", so the join must NOT insert a second one).
        report = _empty_report(app_minutes=Counter({"Bambu Studio": 160}))
        out = self._run(report, bambu={"gcode_state": "RUNNING",
                                       "filename": "gear.3mf"}, strip="gear")
        self.assertIn("Bambu Studio, and the H2D is still printing 'gear'", out)
        # No doubled "and": the chunk before the tail ends with a single ", and".
        self.assertNotIn("and and", out)

    def test_multi_chunk_failed_print_and_tasks(self):
        report = _empty_report(print_failed=2, voice_count=0)
        out = self._run(report, tasks=2)
        self.assertIn("had two prints fail on you", out)
        self.assertIn("cleared two tasks from the queue", out)


class CardAndFireTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    def test_show_card_safe_calls_hud(self):
        hud = mock.MagicMock()
        with mock.patch.object(self.mod.importlib, "import_module", return_value=hud):
            self.mod._show_card_safe()
        hud.show_card.assert_called_once_with("recap")

    def test_show_card_safe_swallows_failure(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no hud_card")):
            self.mod._show_card_safe()  # must not raise

    def test_show_card_safe_inserts_project_dir_on_path(self):
        # Force the "_PROJECT_DIR not in sys.path" branch by removing it first.
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path if p != self.mod._PROJECT_DIR]
            with mock.patch.object(self.mod.importlib, "import_module",
                                   side_effect=ImportError("no hud_card")):
                self.mod._show_card_safe()
            self.assertIn(self.mod._PROJECT_DIR, sys.path)
        finally:
            sys.path[:] = saved

    def test_fire_recap_enqueues_and_stamps(self):
        with mock.patch.object(self.mod, "_build_recap", return_value="[intent:briefing] X"), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod, "_show_card_safe") as card, \
             mock.patch.object(self.mod, "_save_last_fired_date") as save:
            out = self.mod._fire_recap("scheduled")
        self.assertEqual(out, "[intent:briefing] X")
        enq.assert_called_once()
        card.assert_called_once()
        save.assert_called_once()


class RegisterTests(unittest.TestCase):
    def test_register_disabled_skips_scheduler(self):
        mod, _ = load_skill_isolated("daily_recap", register=False)
        actions = {}
        import threading as _thr
        with mock.patch.object(mod, "_read_config",
                               return_value={"enabled": False, "hour": 22, "minute": 30}), \
             mock.patch.object(_thr.Thread, "start") as start:
            mod.register(actions)
        self.assertIn("daily_recap", actions)
        start.assert_not_called()

    def test_register_enabled_starts_scheduler(self):
        mod, _ = load_skill_isolated("daily_recap", register=False)
        actions = {}
        import threading as _thr
        started = {"v": False}
        with mock.patch.object(mod, "_read_config",
                               return_value={"enabled": True, "hour": 22, "minute": 30}), \
             mock.patch.object(_thr.Thread, "start",
                               lambda self: started.__setitem__("v", True)):
            mod.register(actions)
        self.assertTrue(started["v"])
        self.assertIn("daily_recap", actions)


class SchedulerLoopTests(unittest.TestCase):
    """Drive _scheduler_loop one iteration at a time. time.sleep raises after a
    set number of calls to unwind the `while True`, so each branch runs without
    the daemon ever blocking. INITIAL_DELAY sleep is call #1."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("daily_recap")

    @staticmethod
    def _sleep_breaking_at(calls, n):
        """A time.sleep stand-in that returns until its `n`-th call, then raises
        KeyboardInterrupt to unwind the `while True`. Letting the in-branch POLL
        sleep return first means the branch's trailing `continue` executes (one
        full loop turn) before the next iteration raises."""
        def _sleep(_s):
            calls["n"] += 1
            if calls["n"] >= n:
                raise KeyboardInterrupt
        return _sleep

    @staticmethod
    def _fixed_now(dt):
        class _DT(datetime.datetime):
            @classmethod
            def now(cls):
                return dt
        return _DT

    def test_loop_disabled_continues(self):
        calls = {"n": 0}
        # Break on the 2nd POLL sleep (call #3) so the 1st branch's `continue`
        # runs and the loop turns over once.
        with mock.patch.object(self.mod.time, "sleep",
                               self._sleep_breaking_at(calls, 3)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": False, "hour": 22, "minute": 30}), \
             mock.patch.object(self.mod, "_fire_recap") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        fire.assert_not_called()
        self.assertGreaterEqual(calls["n"], 3)

    def test_loop_already_fired_today_continues(self):
        today = datetime.date.today().isoformat()
        calls = {"n": 0}
        with mock.patch.object(self.mod.time, "sleep",
                               self._sleep_breaking_at(calls, 3)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": True, "hour": 22, "minute": 30}), \
             mock.patch.object(self.mod, "_load_last_fired_date", return_value=today), \
             mock.patch.object(self.mod, "_fire_recap") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        fire.assert_not_called()

    def test_loop_before_scheduled_time_continues(self):
        calls = {"n": 0}
        fixed_now = datetime.datetime(2026, 6, 2, 8, 0, 0)  # 08:00 < 22:30
        with mock.patch.object(self.mod.time, "sleep",
                               self._sleep_breaking_at(calls, 3)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": True, "hour": 22, "minute": 30}), \
             mock.patch.object(self.mod, "_load_last_fired_date", return_value=""), \
             mock.patch.object(self.mod.datetime, "datetime", self._fixed_now(fixed_now)), \
             mock.patch.object(self.mod, "_fire_recap") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        fire.assert_not_called()

    def test_loop_past_catchup_window_marks_done(self):
        calls = {"n": 0}
        # Schedule at 02:00; now is 23:00 → ~21 h past the same-day slot, well
        # beyond the 120-min catch-up window, so the loop marks today done
        # rather than barging in. (A 22:30 slot can never be >2 h past without
        # crossing midnight, which resets `scheduled` forward — so this branch
        # is only reachable for an early-morning slot.)
        fixed_now = datetime.datetime(2026, 6, 3, 23, 0, 0)
        with mock.patch.object(self.mod.time, "sleep",
                               self._sleep_breaking_at(calls, 3)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": True, "hour": 2, "minute": 0}), \
             mock.patch.object(self.mod, "_load_last_fired_date", return_value=""), \
             mock.patch.object(self.mod.datetime, "datetime", self._fixed_now(fixed_now)), \
             mock.patch.object(self.mod, "_save_last_fired_date") as save, \
             mock.patch.object(self.mod, "_fire_recap") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        save.assert_called()
        fire.assert_not_called()

    def test_loop_fires_within_window(self):
        calls = {"n": 0}
        # 30 min past the 22:30 slot → within catch-up, fire.
        fixed_now = datetime.datetime(2026, 6, 2, 23, 0, 0)
        with mock.patch.object(self.mod.time, "sleep",
                               self._sleep_breaking_at(calls, 3)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"enabled": True, "hour": 22, "minute": 30}), \
             mock.patch.object(self.mod, "_load_last_fired_date", return_value=""), \
             mock.patch.object(self.mod.datetime, "datetime", self._fixed_now(fixed_now)), \
             mock.patch.object(self.mod, "_fire_recap") as fire:
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        fire.assert_called_with("scheduled")

    def test_loop_swallows_body_exception(self):
        calls = {"n": 0}

        def _sleep(_s):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt
        with mock.patch.object(self.mod.time, "sleep", _sleep), \
             mock.patch.object(self.mod, "_read_config",
                               side_effect=RuntimeError("cfg blew up")):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._scheduler_loop()
        # Body raised, was caught by the except, then the except's sleep raised.
        self.assertGreaterEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
