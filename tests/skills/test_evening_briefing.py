"""Logic tests for skills/evening_briefing.py.

Covers the pure text helpers (tomorrow-weather phrasing, count humaniser,
wttr/Open-Meteo parsers, the session-log pattern scanner + dry observation),
the cross-skill Bambu read, today's-tasks counter, and the full _build_briefing
assembly with every external source mocked. The scheduler thread is neutered;
the speech enqueue is mocked so pending_speech.json is never written.
"""
from __future__ import annotations

import time
import unittest
from collections import Counter
from unittest import mock

from tests._skill_harness import load_skill_isolated


class EveningBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("evening_briefing")

    # ── _phrase_tomorrow (pure) ──────────────────────────────────────────
    def test_phrase_tomorrow_with_desc(self):
        self.assertEqual(
            self.mod._phrase_tomorrow(18, 9, "Partly Cloudy"),
            "tomorrow looks like a high of 18, low of 9, and partly cloudy",
        )

    def test_phrase_tomorrow_without_desc(self):
        self.assertEqual(
            self.mod._phrase_tomorrow(18, 9, ""),
            "tomorrow looks like a high of 18 and a low of 9",
        )

    # ── _humanize_count (pure) ───────────────────────────────────────────
    def test_humanize_count(self):
        h = self.mod._humanize_count
        self.assertEqual(h(2), "twice")
        self.assertEqual(h(3), "three times")
        self.assertEqual(h(4), "four times")
        self.assertEqual(h(11), "11 times")

    # ── _tomorrow_weather_from_wttr (parses mocked JSON) ─────────────────
    def test_tomorrow_weather_from_wttr_parses(self):
        payload = {
            "weather": [
                {"maxtempC": "20", "mintempC": "10", "hourly": []},   # today
                {"maxtempC": "18", "mintempC": "9",
                 "hourly": [{"time": "1200",
                             "weatherDesc": [{"value": "Sunny"}]}]},  # tomorrow
            ]
        }
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = __import__("json").dumps(payload).encode()
        fake_resp.__enter__ = lambda s: fake_resp
        fake_resp.__exit__ = lambda *a: False
        with mock.patch.object(self.mod.urllib.request, "urlopen", return_value=fake_resp):
            out = self.mod._tomorrow_weather_from_wttr()
        self.assertIn("high of 18", out)
        self.assertIn("low of 9", out)
        self.assertIn("sunny", out)

    def test_tomorrow_weather_from_wttr_network_fail(self):
        with mock.patch.object(self.mod.urllib.request, "urlopen",
                               side_effect=OSError("no net")):
            self.assertEqual(self.mod._tomorrow_weather_from_wttr(), "")

    def test_fetch_tomorrow_weather_falls_back_to_open_meteo(self):
        with mock.patch.object(self.mod, "_tomorrow_weather_from_wttr", return_value=""), \
             mock.patch.object(self.mod, "_tomorrow_weather_from_open_meteo",
                               return_value="tomorrow looks like a high of 5 and a low of 1"):
            out = self.mod._fetch_tomorrow_weather()
        self.assertIn("high of 5", out)

    # ── _scan_today_for_patterns + _dry_observation ──────────────────────
    def test_scan_today_for_patterns(self):
        log = (
            "[10:00:00] [action] play_music: 'Michael Jackson'\n"
            "  You:    play Michael Jackson\n"
            "  You:    play Michael Jackson please\n"
            "[10:05:00] [action] see_screen:\n"
        )
        with mock.patch.object(self.mod, "_todays_log_paths", return_value=["x.log"]), \
             mock.patch("builtins.open", mock.mock_open(read_data=log)):
            actions, plays, you = self.mod._scan_today_for_patterns()
        self.assertEqual(you, 2)
        self.assertEqual(actions["play_music"], 1)
        self.assertEqual(plays["michael jackson"], 2)  # filler "please" stripped

    def test_dry_observation_play_pattern(self):
        plays = Counter({"michael jackson": 4})
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), plays, 10)):
            out = self.mod._dry_observation()
        self.assertIn("'play michael jackson'", out)
        self.assertIn("four times", out)
        self.assertIn("pattern emerges", out.lower())

    def test_dry_observation_repeated_action(self):
        actions = Counter({"check_weather": 5, "see_screen": 99})  # boring excluded
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(actions, Counter(), 10)):
            out = self.mod._dry_observation()
        self.assertIn("check weather", out)
        self.assertIn("five times", out)

    def test_dry_observation_nothing(self):
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), Counter(), 0)):
            self.assertEqual(self.mod._dry_observation(), "")

    def test_dry_observation_below_threshold(self):
        # 2 plays is under DRY_OBS_MIN_COUNT (3) → no remark.
        with mock.patch.object(self.mod, "_scan_today_for_patterns",
                               return_value=(Counter(), Counter({"jazz": 2}), 5)):
            self.assertEqual(self.mod._dry_observation(), "")

    # ── _count_tasks_completed_today ─────────────────────────────────────
    def test_count_tasks_completed_today(self):
        today = time.strftime("%Y-%m-%d")
        todo = (f"- [x] done one {today}\n"
                f"- [x] old task 1999-01-01\n"
                f"- [ ] pending {today}\n"
                f"- [x] done two {today}\n")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=todo)):
            self.assertEqual(self.mod._count_tasks_completed_today(), 2)

    # ── _bambu_status (cross-skill read) ─────────────────────────────────
    def test_bambu_status_running(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "RUNNING",
                       "filename": "p.3mf", "layer_num": 5, "total_layer": 50,
                       "mc_remaining": 45}
        fake._strip_filename = lambda s: "p"
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("still printing", out)
        self.assertIn("layer 5 of 50", out)

    def test_bambu_status_failed(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "FAILED", "filename": ""}
        fake._strip_filename = lambda s: ""
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIn("failure", self.mod._bambu_status().lower())

    def test_bambu_status_absent(self):
        import sys
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_bambu_monitor", None)
            self.assertEqual(self.mod._bambu_status(), "")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_full(self):
        with mock.patch.object(self.mod, "_count_voice_interactions_today", return_value=5), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=2), \
             mock.patch.object(self.mod, "_bambu_status", return_value="the H2D is still printing"), \
             mock.patch.object(self.mod, "_fetch_tomorrow_weather",
                               return_value="tomorrow looks like a high of 18"), \
             mock.patch.object(self.mod, "_first_meeting_tomorrow",
                               return_value="your first meeting tomorrow is at 9 AM"), \
             mock.patch.object(self.mod, "_dry_observation",
                               return_value="you said 'play X' four times today, sir."), \
             mock.patch.object(self.mod, "_fetch_news", return_value="Today's headlines, sir. Y."), \
             mock.patch.object(self.mod, "_fetch_tomorrow_umbrella", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("Good evening, sir. 5 voice interactions", out)
        self.assertIn("2 tasks cleared", out)
        self.assertIn("Currently, the H2D is still printing", out)
        self.assertIn("For tomorrow,", out)
        self.assertTrue(out.startswith("[intent:briefing]"))  # news included

    def test_build_briefing_quiet_day(self):
        with mock.patch.object(self.mod, "_count_voice_interactions_today", return_value=0), \
             mock.patch.object(self.mod, "_count_tasks_completed_today", return_value=0), \
             mock.patch.object(self.mod, "_bambu_status", return_value=""), \
             mock.patch.object(self.mod, "_fetch_tomorrow_weather", return_value=""), \
             mock.patch.object(self.mod, "_first_meeting_tomorrow", return_value=""), \
             mock.patch.object(self.mod, "_dry_observation", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_tomorrow_umbrella", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("A quiet day on the voice channel", out)
        self.assertFalse(out.startswith("[intent:briefing]"))

    # ── evening_briefing action ──────────────────────────────────────────
    def test_action_returns_and_enqueues(self):
        mod, actions = load_skill_isolated("evening_briefing")
        with mock.patch.object(mod, "_build_briefing", return_value="Good evening, sir."), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_show_card_safe"), \
             mock.patch.object(mod, "_save_last_fired_date"):
            out = actions["evening_briefing"]("")
        self.assertEqual(out, "Good evening, sir.")
        enq.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("evening_briefing")
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["evening_briefing"]("")
        self.assertIn("failed", out.lower())


if __name__ == "__main__":
    unittest.main()
