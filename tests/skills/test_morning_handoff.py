"""Logic tests for skills/morning_handoff.py.

Covers the chained-briefing sections (weather / calendar+mail / Teams-VIP
callout / print / news) with sibling skills mocked, the ordinal helper, the
full _build_handoff stitch, the overnight-print phrase + skew wording, the
predictive-setup readback assembly (with all hardware launchers stubbed), the
same-day suppression + chain entry, and the registered actions. No real
network / audio / window / printer access — every external primitive is mocked.
"""
from __future__ import annotations

import os
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class MorningHandoffTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_handoff")

    # ── _ordinal (pure) ──────────────────────────────────────────────────
    def test_ordinal(self):
        o = self.mod._ordinal
        self.assertEqual(o(1), "1st")
        self.assertEqual(o(13), "13th")
        self.assertEqual(o(23), "23rd")
        self.assertEqual(o(30), "30th")

    # ── _section_weather ─────────────────────────────────────────────────
    def test_section_weather(self):
        bs = mock.MagicMock()
        bs.get_weather_data.return_value = {"temp_c": 18, "desc": "Overcast", "source": "wttr"}
        with mock.patch.object(self.mod, "_import_skill", return_value=bs):
            out = self.mod._section_weather()
        self.assertEqual(out, "18 degrees and overcast in your area.")

    def test_section_weather_degraded(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_weather(), "")

    # ── _section_calendar (mail + meeting) ───────────────────────────────
    def test_section_calendar_combines_mail_and_meeting(self):
        import datetime
        ms = mock.MagicMock()
        ms.get_unread_mail_count.return_value = 3
        ms.get_first_meeting.return_value = {
            "start": datetime.datetime(2026, 6, 1, 9, 30),
            "subject": "Design review", "organizer": "Sam Co <w@x.com>"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        self.assertTrue(out.startswith("From Outlook:"))
        self.assertIn("3 unread emails", out)
        self.assertIn("9:30 AM", out)
        self.assertIn("with Sam", out)
        self.assertIn("Design review", out)

    def test_section_calendar_empty(self):
        ms = mock.MagicMock()
        ms.get_unread_mail_count.return_value = 0
        ms.get_first_meeting.return_value = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    # ── _section_teams_vip ───────────────────────────────────────────────
    def test_teams_vip_emphasis(self):
        # VIP emphasis fires only when the configured JARVIS_VIP_NAME is the
        # visible sender.
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 2, "Sam Industries")
        with mock.patch.dict(os.environ, {"JARVIS_VIP_NAME": "Sam"}), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            out = self.mod._section_teams_vip()
        self.assertIn("Sam Industries", out)
        self.assertIn("2 unread messages", out)

    def test_teams_single_non_vip(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (True, 1, "Alex")
        with mock.patch.dict(os.environ, {"JARVIS_VIP_NAME": "Sam"}), \
             mock.patch.object(self.mod, "_import_skill", return_value=tn):
            out = self.mod._section_teams_vip()
        self.assertEqual(out, "One unread message on Teams from Alex, sir.")

    def test_teams_nothing_unread(self):
        tn = mock.MagicMock()
        tn._ask_vision_for_teams_state.return_value = (False, 0, "")
        with mock.patch.object(self.mod, "_import_skill", return_value=tn):
            self.assertEqual(self.mod._section_teams_vip(), "")

    # ── _section_print ───────────────────────────────────────────────────
    def test_section_print_finished(self):
        bm = mock.MagicMock()
        # bm._state_lock is used in a `with` block → give it a real lock.
        import threading
        bm._state_lock = threading.Lock()
        bm._state = {"gcode_state": "FINISH", "filename": "bracket.3mf",
                     "last_update": time.time()}
        bm._strip_filename = lambda s: "bracket"
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            out = self.mod._section_print()
        self.assertIn("overnight print", out.lower())
        self.assertIn("bracket", out)
        self.assertIn("ready", out.lower())

    def test_section_print_idle(self):
        bm = mock.MagicMock()
        import threading
        bm._state_lock = threading.Lock()
        bm._state = {"last_update": 0.0}
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            self.assertEqual(self.mod._section_print(), "")

    # ── _build_handoff stitch ────────────────────────────────────────────
    def test_build_handoff_chains_sections(self):
        with mock.patch.object(self.mod, "_section_weather", return_value="18 degrees and clear."), \
             mock.patch.object(self.mod, "_section_calendar", return_value="From Outlook: 1 unread email."), \
             mock.patch.object(self.mod, "_section_teams_vip", return_value=""), \
             mock.patch.object(self.mod, "_section_print", return_value=""), \
             mock.patch.object(self.mod, "_section_news", return_value="Today's headlines, sir. X."):
            out = self.mod._build_handoff(setup_line="Workshop is yours, sir.")
        self.assertTrue(out.startswith("[intent:briefing] Good morning, sir."))
        self.assertIn("Workshop is yours, sir.", out)
        self.assertIn("18 degrees and clear.", out)
        self.assertIn("From Outlook: 1 unread email.", out)
        self.assertIn("Anything else I should know, sir?", out)

    def test_build_handoff_section_crash_is_skipped(self):
        # The real code logs fn.__name__ on crash, so the stub needs one.
        boom = mock.MagicMock(side_effect=RuntimeError("boom"), __name__="_section_weather")
        with mock.patch.object(self.mod, "_section_weather", boom), \
             mock.patch.object(self.mod, "_section_calendar", return_value=""), \
             mock.patch.object(self.mod, "_section_teams_vip", return_value=""), \
             mock.patch.object(self.mod, "_section_print", return_value=""), \
             mock.patch.object(self.mod, "_section_news", return_value=""):
            out = self.mod._build_handoff()
        # Crashed section is dropped, the rest of the chain still assembles.
        self.assertIn("Good morning, sir.", out)
        self.assertIn("Anything else I should know, sir?", out)

    # ── _overnight_print_phrase ──────────────────────────────────────────
    def test_overnight_print_phrase_finished_with_skew(self):
        bm = mock.MagicMock()
        bm.get_last_print_completion_summary.return_value = {
            "finish_phrase": "4:12 AM", "delta_minutes": 120}  # 2h under estimate
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            phrase, was_active = self.mod._overnight_print_phrase(time.time())
        self.assertTrue(was_active)
        self.assertIn("finished at 4:12 AM", phrase)
        self.assertIn("2 hours under estimate", phrase)

    def test_overnight_print_phrase_running(self):
        bm = mock.MagicMock()
        bm.get_last_print_completion_summary.return_value = None
        import threading
        bm._state_lock = threading.Lock()
        bm._state = {"gcode_state": "RUNNING", "layer_num": 47, "total_layer": 312}
        bm._format_minutes = lambda m: "18 minutes"
        with mock.patch.object(self.mod, "_import_skill", return_value=bm):
            phrase, was_active = self.mod._overnight_print_phrase(time.time())
        self.assertTrue(was_active)
        self.assertIn("still printing", phrase)
        self.assertIn("layer 47 of 312", phrase)

    def test_overnight_print_phrase_none(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._overnight_print_phrase(time.time()), ("", False))

    # ── _predictive_morning_setup readback ───────────────────────────────
    def test_predictive_setup_readback_assembly(self):
        with mock.patch.object(self.mod, "_focus_middle_monitor", return_value=True), \
             mock.patch.object(self.mod, "_morning_pattern_apps", return_value=set()), \
             mock.patch.object(self.mod, "_overnight_print_phrase",
                               return_value=("your overnight print finished at 4:12 AM", True)), \
             mock.patch.object(self.mod, "_open_chrome_apple_music", return_value=True), \
             mock.patch.object(self.mod, "_launch_named_app", return_value=True), \
             mock.patch.object(self.mod, "_set_master_volume", return_value=True), \
             mock.patch.object(self.mod.time, "sleep"):
            out = self.mod._predictive_morning_setup(now_ts=time.time())
        self.assertIn("Workshop is yours, sir.", out)
        self.assertIn("Apple Music is queued", out)
        self.assertIn("Teams is up", out)
        # Print finished overnight → "next layer file?" sign-off.
        self.assertIn("next layer file", out.lower())

    def test_predictive_setup_disabled(self):
        with mock.patch.object(self.mod, "PREDICTIVE_SETUP_ENABLED", False):
            self.assertEqual(self.mod._predictive_morning_setup(), "")

    # ── same-day suppression + chain entry ───────────────────────────────
    def test_handoff_already_fired_today(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._handoff_already_fired_today())
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertFalse(self.mod._handoff_already_fired_today())

    def test_fire_handoff_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_handoff_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_handoff") as build:
            out = self.mod._fire_handoff("auto", force=False)
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_fire_from_chain_fires_after_delay(self):
        with mock.patch.object(self.mod, "_handoff_already_fired_today", return_value=False), \
             mock.patch.object(self.mod.time, "sleep") as slp, \
             mock.patch.object(self.mod, "_fire_handoff", return_value="briefing!") as fire:
            out = self.mod._fire_from_chain("chain")
        slp.assert_called_once()
        fire.assert_called_once()
        self.assertEqual(out, "briefing!")

    # ── registered actions ───────────────────────────────────────────────
    def test_action_morning_handoff(self):
        mod, actions = load_skill_isolated("morning_handoff")
        with mock.patch.object(mod, "_fire_handoff", return_value="[intent:briefing] Good morning."):
            out = actions["morning_handoff"]("")
        self.assertIn("Good morning", out)

    def test_action_predictive_setup_aliases(self):
        mod, actions = load_skill_isolated("morning_handoff")
        # All three aliases bind to the same predictive setup.
        for name in ("predictive_morning_setup", "setup_workspace", "workspace_setup"):
            self.assertIn(name, actions)
        with mock.patch.object(mod, "_predictive_morning_setup", return_value="Workshop is yours, sir."):
            out = actions["setup_workspace"]("")
        self.assertEqual(out, "Workshop is yours, sir.")

    def test_action_predictive_setup_no_op_message(self):
        mod, actions = load_skill_isolated("morning_handoff")
        with mock.patch.object(mod, "_predictive_morning_setup", return_value=""):
            out = actions["predictive_morning_setup"]("")
        self.assertIn("already in order", out.lower())


if __name__ == "__main__":
    unittest.main()
