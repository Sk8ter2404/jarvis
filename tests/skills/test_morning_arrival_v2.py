"""Logic tests for skills/morning_arrival_v2.py.

v2 is presence-gated. Tests cover the morning-window check, the six data-source
section formatters (weather / teams / news intro-strip / print / deliveries /
calendar) with their sibling skills mocked, the JARVIS-cadence composition with
the 60-second TTS budget + drop order, same-day + chain-already-fired
suppression, the chain entry, and the manual action. The presence-watcher
thread is neutered by the harness; nothing real (network/face/printer) runs.
"""
from __future__ import annotations

import datetime
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class MorningArrivalV2Tests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_arrival_v2")

    # ── _within_morning_window (pure-ish) ────────────────────────────────
    def test_within_morning_window(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=time.struct_time((2026, 6, 1, 8, 0, 0, 0, 152, -1))):
            self.assertTrue(self.mod._within_morning_window())
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=time.struct_time((2026, 6, 1, 15, 0, 0, 0, 152, -1))):
            self.assertFalse(self.mod._within_morning_window())

    # ── _section_teams ───────────────────────────────────────────────────
    def test_section_teams_plural_with_sender(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 3, "top_sender": "Sam"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_teams()
        self.assertEqual(out, "3 new Teams messages, one from Sam")

    def test_section_teams_single(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 1, "top_sender": "Alex"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "one new Teams message from Alex")

    def test_section_teams_zero(self):
        ms = mock.MagicMock()
        ms.get_teams_unread_count.return_value = {"count": 0}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_teams(), "")

    def test_section_teams_skill_absent(self):
        with mock.patch.object(self.mod, "_import_skill", return_value=None):
            self.assertEqual(self.mod._section_teams(), "")

    # ── _section_news strips the intro greeting ──────────────────────────
    def test_section_news_strips_intro(self):
        nb = mock.MagicMock()
        nb.get_news_text.return_value = "Today's headlines, sir. Rates rise. Storm clears."
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            out = self.mod._section_news()
        self.assertEqual(out, "Rates rise. Storm clears.")

    def test_section_news_empty(self):
        nb = mock.MagicMock()
        nb.get_news_text.return_value = ""
        with mock.patch.object(self.mod, "_import_skill", return_value=nb):
            self.assertEqual(self.mod._section_news(), "")

    # ── _section_deliveries drops sentinel responses ─────────────────────
    def test_section_deliveries_real_result(self):
        aot = mock.MagicMock()
        aot.action_check_orders.return_value = "Your headphones arrive tomorrow."
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "Your headphones arrive tomorrow.")

    def test_section_deliveries_sentinel_dropped(self):
        aot = mock.MagicMock()
        aot.action_check_orders.return_value = "No active Amazon orders right now."
        with mock.patch.object(self.mod, "_import_skill", return_value=aot):
            self.assertEqual(self.mod._section_deliveries(), "")

    # ── _section_calendar ────────────────────────────────────────────────
    def test_section_calendar_subject(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = {
            "start": datetime.datetime(2026, 6, 1, 10, 30),
            "subject": "Standup", "organizer": "me"}
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            out = self.mod._section_calendar()
        self.assertEqual(out, "Standup at 10:30 AM")

    def test_section_calendar_none(self):
        ms = mock.MagicMock()
        ms.get_first_meeting.return_value = None
        with mock.patch.object(self.mod, "_import_skill", return_value=ms):
            self.assertEqual(self.mod._section_calendar(), "")

    # ── _compose_briefing JARVIS cadence ─────────────────────────────────
    def test_compose_briefing_full(self):
        parts = {"weather": "Bring an umbrella, sir", "teams": "one new Teams message from Sam",
                 "calendar": "a sync at 10", "print": "the H2D finished overnight",
                 "deliveries": "Headphones arrive today.", "news": "Markets up."}
        out = self.mod._compose_briefing(parts)
        self.assertTrue(out.startswith("[intent:briefing] Good morning, sir."))
        self.assertIn("Bring an umbrella, sir.", out)
        self.assertIn("One new Teams message from Sam.", out)   # capitalised
        self.assertIn("You have a sync at 10.", out)
        self.assertIn("The H2D finished overnight.", out)

    def test_compose_briefing_nothing_overnight(self):
        parts = {k: "" for k in ("weather", "teams", "calendar", "print", "deliveries", "news")}
        out = self.mod._compose_briefing(parts)
        self.assertIn("Nothing of note overnight.", out)

    # ── TTS budget ───────────────────────────────────────────────────────
    def test_estimate_tts_strips_tag(self):
        secs = self.mod._estimate_tts_seconds("[intent:briefing] " + ("y" * 60))
        self.assertAlmostEqual(secs, 4.0, places=3)

    def test_compose_within_budget_drops_news_first(self):
        news_blob = "z" * 400
        parts = {"weather": "Dry today.", "teams": "teams ping", "calendar": "a sync at 10",
                 "print": "print done", "deliveries": "pkg today", "news": news_blob}
        with mock.patch.object(self.mod, "TTS_BUDGET_SECONDS", 10.0):
            out = self.mod._compose_within_budget(parts)
        # news is first in TTS_DROP_ORDER → dropped; calendar (last) survives.
        self.assertNotIn(news_blob, out)
        self.assertIn("You have a sync at 10.", out)

    # ── suppression ──────────────────────────────────────────────────────
    def test_already_fired_today(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod, "_load_state", return_value={"last_fired_date": today}):
            self.assertTrue(self.mod._already_fired_today())
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertFalse(self.mod._already_fired_today())

    def test_fire_arrival_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing") as build:
            out = self.mod._fire_arrival("auto", force=False)
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_fire_arrival_suppressed_when_chain_already_briefed(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=False), \
             mock.patch.object(self.mod, "_chain_morning_briefing_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing") as build:
            out = self.mod._fire_arrival("auto", force=False)
        self.assertEqual(out, "")
        build.assert_not_called()

    def test_fire_arrival_force_bypasses_suppression(self):
        with mock.patch.object(self.mod, "_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_build_briefing", return_value="[intent:briefing] hi"), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_mark_fired"):
            out = self.mod._fire_arrival("manual", force=True)
        self.assertEqual(out, "[intent:briefing] hi")

    # ── manual action ────────────────────────────────────────────────────
    def test_action_returns_text(self):
        mod, actions = load_skill_isolated("morning_arrival_v2")
        with mock.patch.object(mod, "_fire_arrival", return_value="[intent:briefing] Good morning."):
            out = actions["morning_arrival_v2"]("")
        self.assertIn("Good morning", out)

    def test_action_no_content(self):
        mod, actions = load_skill_isolated("morning_arrival_v2")
        with mock.patch.object(mod, "_fire_arrival", return_value=""):
            out = actions["morning_arrival_v2"]("")
        self.assertIn("no content", out.lower())


if __name__ == "__main__":
    unittest.main()
