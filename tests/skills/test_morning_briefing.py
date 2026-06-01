"""Logic tests for skills/morning_briefing.py.

Covers the pure helpers (ordinal, pending-task count, weather phrase, bed-time
remark, Outlook summary timeout wrapper), the full _build_briefing assembly
with every external source mocked, the same-day fired flag, the chain entry
point's TOCTOU/suppression behaviour, and the manual action. No network, no
real pending_speech.json writes.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class MorningBriefingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("morning_briefing")

    # ── _ordinal (pure) ──────────────────────────────────────────────────
    def test_ordinal(self):
        o = self.mod._ordinal
        self.assertEqual(o(1), "1st")
        self.assertEqual(o(2), "2nd")
        self.assertEqual(o(3), "3rd")
        self.assertEqual(o(4), "4th")
        self.assertEqual(o(11), "11th")   # teens are all "th"
        self.assertEqual(o(12), "12th")
        self.assertEqual(o(21), "21st")
        self.assertEqual(o(22), "22nd")

    # ── _count_pending_tasks ─────────────────────────────────────────────
    def test_count_pending_tasks(self):
        todo = "- [ ] one\n- [x] done\n- [ ] two\nnot a task\n- [ ] three\n"
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=todo)):
            self.assertEqual(self.mod._count_pending_tasks(), 3)

    def test_count_pending_tasks_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._count_pending_tasks(), 0)

    # ── _fetch_weather (briefing_sources) ────────────────────────────────
    def test_fetch_weather_phrase(self):
        fake_bs = mock.MagicMock()
        fake_bs.get_weather_data.return_value = {"temp_c": 18, "desc": "Overcast", "source": "wttr"}
        import sys
        # _fetch_weather does `from . import briefing_sources` first; inject a
        # package-style module name so that import resolves to our fake.
        with mock.patch.dict(sys.modules, {"skill_morning_briefing.briefing_sources": fake_bs,
                                           "briefing_sources": fake_bs}):
            out = self.mod._fetch_weather()
        self.assertIn("18 degrees and overcast", out)

    def test_fetch_weather_degraded(self):
        fake_bs = mock.MagicMock()
        fake_bs.get_weather_data.return_value = None
        import sys
        with mock.patch.dict(sys.modules, {"skill_morning_briefing.briefing_sources": fake_bs,
                                           "briefing_sources": fake_bs}):
            self.assertEqual(self.mod._fetch_weather(), "")

    # ── _bed_remark ──────────────────────────────────────────────────────
    def test_bed_remark_late_night(self):
        # mtime at 02:30 local → triggers the dry "pace yourself" remark.
        late = time.mktime(time.struct_time((2026, 6, 1, 2, 30, 0, 0, 152, -1)))
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["s.log"]), \
             mock.patch.object(self.mod.os.path, "getmtime", return_value=late):
            out = self.mod._bed_remark()
        self.assertIn("2:30 AM", out)
        self.assertIn("pace yourself", out)

    def test_bed_remark_daytime_silent(self):
        noon = time.mktime(time.struct_time((2026, 6, 1, 12, 0, 0, 0, 152, -1)))
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{}")), \
             mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir", return_value=["s.log"]), \
             mock.patch.object(self.mod.os.path, "getmtime", return_value=noon):
            self.assertEqual(self.mod._bed_remark(), "")

    # ── _outlook_summary timeout wrapper ─────────────────────────────────
    def test_outlook_summary_returns_blocking_result(self):
        with mock.patch.object(self.mod, "_outlook_summary_blocking",
                               return_value="one unread email"):
            self.assertEqual(self.mod._outlook_summary(), "one unread email")

    def test_outlook_summary_swallows_blocking_error(self):
        with mock.patch.object(self.mod, "_outlook_summary_blocking",
                               side_effect=RuntimeError("graph down")):
            self.assertEqual(self.mod._outlook_summary(), "")

    # ── _build_briefing assembly ─────────────────────────────────────────
    def test_build_briefing_full(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value="18 degrees and clear in your area"), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=3), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value="Today's headlines, sir. X."), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value="bring an umbrella"), \
             mock.patch.object(self.mod, "_outlook_summary", return_value="one unread email"), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("Good morning, sir", out)
        self.assertIn("18 degrees and clear", out)
        self.assertIn("3 tasks queued", out)
        self.assertIn("From Outlook: one unread email", out)
        self.assertIn("bring an umbrella", out)
        # News present → leading briefing-intent tag.
        self.assertTrue(out.startswith("[intent:briefing]"))

    def test_build_briefing_empty_queue_phrase(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=0), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value=""), \
             mock.patch.object(self.mod, "_outlook_summary", return_value=""), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("mercifully empty", out)
        # No news → no intent tag.
        self.assertFalse(out.startswith("[intent:briefing]"))

    def test_build_briefing_single_task(self):
        with mock.patch.object(self.mod, "_fetch_weather", return_value=""), \
             mock.patch.object(self.mod, "_count_pending_tasks", return_value=1), \
             mock.patch.object(self.mod, "_bed_remark", return_value=""), \
             mock.patch.object(self.mod, "_fetch_news", return_value=""), \
             mock.patch.object(self.mod, "_fetch_umbrella_alert", return_value=""), \
             mock.patch.object(self.mod, "_outlook_summary", return_value=""), \
             mock.patch.object(self.mod, "_fetch_robot_volunteer", return_value=""):
            out = self.mod._build_briefing()
        self.assertIn("one task queued", out)

    # ── same-day fired flag ──────────────────────────────────────────────
    def test_already_fired_today_true(self):
        today = time.strftime("%Y-%m-%d")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=today)):
            self.assertTrue(self.mod._briefing_already_fired_today())

    def test_already_fired_today_stale(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="1999-01-01")):
            self.assertFalse(self.mod._briefing_already_fired_today())

    def test_already_fired_today_no_flag(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertFalse(self.mod._briefing_already_fired_today())

    # ── _fire_from_chain suppression ─────────────────────────────────────
    def test_fire_from_chain_suppressed_when_already_fired(self):
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=True), \
             mock.patch.object(self.mod, "_fire_briefing") as fire:
            self.mod._fire_from_chain("test")
        fire.assert_not_called()

    def test_fire_from_chain_fires_after_delay(self):
        # First check (pre) and second check (post-delay) both False → fires.
        with mock.patch.object(self.mod, "_briefing_already_fired_today", return_value=False), \
             mock.patch.object(self.mod.time, "sleep") as slp, \
             mock.patch.object(self.mod, "_fire_briefing") as fire:
            self.mod._fire_from_chain("chain pick")
        slp.assert_called_once()  # the pre-fire delay
        fire.assert_called_once()

    # ── morning_briefing action ──────────────────────────────────────────
    def test_action_builds_and_marks(self):
        mod, actions = load_skill_isolated("morning_briefing")
        with mock.patch.object(mod, "_build_briefing", return_value="Good morning, sir."), \
             mock.patch.object(mod, "_show_card_safe"), \
             mock.patch.object(mod, "_mark_briefing_fired_today") as mark:
            out = actions["morning_briefing"]("")
        self.assertEqual(out, "Good morning, sir.")
        mark.assert_called_once()

    def test_action_handles_exception(self):
        mod, actions = load_skill_isolated("morning_briefing")
        with mock.patch.object(mod, "_build_briefing", side_effect=RuntimeError("boom")):
            out = actions["morning_briefing"]("")
        self.assertIn("failed", out.lower())


if __name__ == "__main__":
    unittest.main()
