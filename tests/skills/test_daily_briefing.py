"""Logic tests for skills/daily_briefing.py.

Covers the briefing-text assembly (time phrase + weather/meeting/bambu extras,
"nothing remarkable" empty path), the weather/meeting formatters that sit on
top of briefing_sources, the cross-skill Bambu/face-tracker reads, and the
manual daily_briefing action. The scheduler thread is neutered by the harness;
the speech enqueue is mocked so no pending_speech.json is written.
"""
from __future__ import annotations

import datetime
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class DailyBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("daily_briefing")

    # ── _format_time_phrase (pure) ───────────────────────────────────────
    def test_format_time_phrase(self):
        st = time.struct_time((2026, 6, 1, 13, 5, 0, 0, 152, -1))
        self.assertEqual(self.mod._format_time_phrase(st), "1:05 PM")
        st2 = time.struct_time((2026, 6, 1, 0, 0, 0, 0, 152, -1))
        self.assertEqual(self.mod._format_time_phrase(st2), "12:00 AM")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_with_all_extras(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value="14 degrees and clear"), \
             mock.patch.object(self.mod, "_first_meeting_today",
                               return_value="your first meeting today is at 9:30 AM"), \
             mock.patch.object(self.mod, "_bambu_status", return_value="the H2D is mid-print"):
            out = self.mod._build_briefing()
        self.assertIn("Good morning, sir", out)
        self.assertIn("14 degrees and clear", out)
        self.assertIn("9:30 AM", out)
        self.assertIn("mid-print", out)
        self.assertTrue(out.endswith("."))

    def test_build_briefing_nothing_remarkable(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_first_meeting_today", return_value=""), \
             mock.patch.object(self.mod, "_bambu_status", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("nothing remarkable to report", out)

    # ── _fetch_weather (over briefing_sources) ───────────────────────────
    def test_fetch_weather_phrase(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 14, "desc": "Overcast", "source": "wttr"}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(),
                             "outside temperature is 14 degrees and overcast")

    def test_fetch_weather_cached_suffix(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 9, "desc": "", "source": "cache", "stale": True}
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._fetch_weather()
        self.assertIn("9 degrees", out)
        self.assertIn("(cached)", out)

    def test_fetch_weather_degrades_when_sources_missing(self):
        with mock.patch.object(self.mod, "_briefing_sources", return_value=None):
            self.assertEqual(self.mod._fetch_weather(), "")

    def test_fetch_weather_bad_temp(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"desc": "rain"}  # no temp_c
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._fetch_weather(), "")

    # ── _first_meeting_today formatting ──────────────────────────────────
    def test_first_meeting_with_organizer_and_subject(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = {
            "start": datetime.datetime(2026, 6, 1, 9, 30),
            "organizer": "Sam Industries <sam@x.com>",
            "subject": "Design review",
        }
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            out = self.mod._first_meeting_today()
        self.assertIn("9:30 AM", out)
        self.assertIn("Sam Industries", out)
        self.assertIn("Design review", out)

    def test_first_meeting_none(self):
        bs = mock.MagicMock()
        bs.get_first_meeting_data.return_value = None
        with mock.patch.object(self.mod, "_briefing_sources", return_value=bs):
            self.assertEqual(self.mod._first_meeting_today(), "")

    # ── _bambu_status (cross-skill read) ─────────────────────────────────
    def test_bambu_status_finished_recently(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time() - 600, "gcode_state": "FINISH",
                       "filename": "bracket.3mf"}
        fake._strip_filename = lambda s: "bracket"
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("finished printing", out)
        self.assertIn("bracket", out)

    def test_bambu_status_running_with_layers(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": time.time(), "gcode_state": "RUNNING",
                       "filename": "part.gcode", "layer_num": 10, "total_layer": 100,
                       "mc_remaining": 90}
        fake._strip_filename = lambda s: "part"
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._bambu_status()
        self.assertIn("mid-print", out)
        self.assertIn("layer 10 of 100", out)
        self.assertIn("1 hour and 30 minutes remaining", out)

    def test_bambu_status_not_loaded(self):
        import sys
        # Ensure the monitor module is absent.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_bambu_monitor", None)
            self.assertEqual(self.mod._bambu_status(), "")

    def test_bambu_status_no_data(self):
        fake = mock.MagicMock()
        fake._state_lock = None
        fake._state = {"last_update": 0.0}
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._bambu_status(), "")

    # ── _user_at_desk (cross-skill read) ─────────────────────────────────
    def test_user_at_desk_present(self):
        fake = mock.MagicMock()
        fake._snapshot_state.return_value = {"last_sample_at": time.time(),
                                             "current_monitor": "middle_or_top"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), True)

    def test_user_at_desk_away(self):
        fake = mock.MagicMock()
        fake._snapshot_state.return_value = {"last_sample_at": time.time(),
                                             "current_monitor": "away"}
        import sys
        with mock.patch.dict(sys.modules, {"skill_face_tracker": fake}):
            self.assertIs(self.mod._user_at_desk(), False)

    def test_user_at_desk_unknown_when_not_loaded(self):
        import sys
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_face_tracker", None)
            self.assertIsNone(self.mod._user_at_desk())

    # ── state load/save ──────────────────────────────────────────────────
    def test_load_last_fired_missing(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_last_fired_date(), "")

    def test_load_last_fired_reads_value(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open",
                        mock.mock_open(read_data='{"last_fired_date": "2026-06-01"}')):
            self.assertEqual(self.mod._load_last_fired_date(), "2026-06-01")

    # ── daily_briefing action ────────────────────────────────────────────
    def test_action_returns_and_enqueues(self):
        mod, actions = load_skill_isolated("daily_briefing")
        with mock.patch.object(mod, "_build_briefing", return_value="Good morning, sir. test."), \
             mock.patch.object(mod, "_enqueue_speech") as enq, \
             mock.patch.object(mod, "_save_last_fired_date") as save:
            out = actions["daily_briefing"]("")
        self.assertEqual(out, "Good morning, sir. test.")
        enq.assert_called_once_with("Good morning, sir. test.")
        save.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("daily_briefing")
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["daily_briefing"]("")
        self.assertIn("failed", out.lower())


if __name__ == "__main__":
    unittest.main()
