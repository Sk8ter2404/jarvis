"""Logic tests for skills/daily_recap.py.

Covers the pure formatting helpers (app-name normaliser, number word, duration
formatter, title-caser), the session-log miner's regex extraction, the
pattern-JSONL supplement, today's-tasks counter, and the big _build_recap
text-assembly across its many branches (top app, prints, Teams, music, tasks,
empty-day fallback) with _scan_session_logs mocked. No real logs/network.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
