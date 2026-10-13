"""Logic tests for skills/hours_reminder.py: the Monday hours reminder.

Anchor 2026-09-28 (Monday): one pay period is Sep 28 - Oct 11, so the next
starts Oct 12 and that Monday's reminder covers the period that just ended.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from tests._skill_harness import load_skill_isolated


class FakeScheduler:
    def __init__(self, running=True):
        self.running = running
        self.calls = []
        self.cancelled = []

    def is_available(self):
        return True

    def status(self):
        return {"running": self.running}

    def schedule_cron(self, **kw):
        self.calls.append(("cron", kw))
        return kw.get("job_id")

    def schedule_once(self, **kw):
        self.calls.append(("once", kw))
        return kw.get("job_id")

    def cancel_job(self, jid):
        self.cancelled.append(jid)
        return True


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("JARVIS_DATA_DIR")
        self._dir = tempfile.mkdtemp(prefix="jarvis_hours_test_")
        os.environ["JARVIS_DATA_DIR"] = self._dir
        self.said = []
        self.say_ok = True

        def say_aloud(text):
            self.said.append(text)
            if self.say_ok:
                return "Queued to speak, sir."
            return "I couldn't queue that line, sir."

        self.mod, self.actions = load_skill_isolated(
            "hours_reminder", actions={"say_aloud": say_aloud})
        self.cfg = {"HOURS_REMINDER_ENABLED": True,
                    "HOURS_REMINDER_ANCHOR": "2026-09-28",
                    "HOURS_REMINDER_TIME": "09:00"}
        p = mock.patch.object(
            self.mod, "_cfg", lambda name, default: self.cfg.get(name, default))
        p.start()
        self.addCleanup(p.stop)

    def at(self, y, mo, d, h=9, mi=30):
        p = mock.patch.object(self.mod, "_now",
                              lambda: datetime(y, mo, d, h, mi))
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)


class StatusActionTests(_Base):
    def test_registered_and_verbatim(self):
        self.assertIn("hours_status", self.actions)
        self.assertIn("hours_reminder_fire", self.actions)
        self.assertIn("hours_status", self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_hours_week(self):
        self.at(2026, 10, 12)
        out = self.actions["hours_status"]("")
        self.assertTrue(out.startswith("Yes, sir, it's hours week."))
        self.assertIn("Monday, September 28 to Sunday, October 11", out)

    def test_off_week_names_the_next_hours_day(self):
        self.at(2026, 10, 20)
        out = self.actions["hours_status"]("")
        self.assertTrue(out.startswith("Not this week, sir."))
        self.assertIn("Monday, October 26", out)

    def test_anchor_setting_is_honoured(self):
        self.cfg["HOURS_REMINDER_ANCHOR"] = "2026-10-05"
        self.at(2026, 10, 12)
        self.assertTrue(self.actions["hours_status"]("").startswith("Not this"))
        self.at(2026, 10, 19)
        self.assertTrue(self.actions["hours_status"]("").startswith("Yes"))

    def test_status_answers_even_when_the_reminder_is_off(self):
        self.cfg["HOURS_REMINDER_ENABLED"] = False
        self.at(2026, 10, 12)
        self.assertTrue(self.actions["hours_status"]("").startswith("Yes"))


class FireTests(_Base):
    def test_speaks_on_the_hours_monday_with_the_dates(self):
        self.at(2026, 10, 12, 9, 0)
        out = self.actions["hours_reminder_fire"]("")
        self.assertEqual(self.said, [
            "Time to put in your hours, sir, for Monday, September 28 to "
            "Sunday, October 11."])
        self.assertEqual(out, self.said[0])

    def test_silent_in_the_off_week(self):
        self.at(2026, 10, 19, 9, 0)
        self.assertEqual(self.actions["hours_reminder_fire"](""), "")
        self.assertEqual(self.said, [])

    def test_once_per_period(self):
        self.at(2026, 10, 12, 9, 0)
        self.actions["hours_reminder_fire"]("")
        self.actions["hours_reminder_fire"]("")
        self.assertEqual(len(self.said), 1)

    def test_next_period_reminds_again(self):
        self.at(2026, 10, 12, 9, 0)
        self.actions["hours_reminder_fire"]("")
        self.at(2026, 10, 26, 9, 0)
        self.actions["hours_reminder_fire"]("")
        self.assertEqual(len(self.said), 2)
        self.assertIn("Monday, October 12 to Sunday, October 25", self.said[1])

    def test_catch_up_later_in_the_week_still_speaks(self):
        self.at(2026, 10, 14, 11, 0)
        self.actions["hours_reminder_fire"]("")
        self.assertEqual(len(self.said), 1)
        self.assertIn("September 28", self.said[0])

    def test_not_before_the_reminder_time_or_late_at_night(self):
        self.at(2026, 10, 12, 8, 0)
        self.assertEqual(self.actions["hours_reminder_fire"](""), "")
        self.at(2026, 10, 12, 22, 0)
        self.assertEqual(self.actions["hours_reminder_fire"](""), "")
        self.assertEqual(self.said, [])

    def test_disabled_setting_silences_it(self):
        self.cfg["HOURS_REMINDER_ENABLED"] = False
        self.at(2026, 10, 12, 9, 0)
        self.assertEqual(self.actions["hours_reminder_fire"](""), "")
        self.assertEqual(self.said, [])

    def test_unspoken_reminder_is_not_marked_done(self):
        self.say_ok = False
        self.at(2026, 10, 12, 9, 0)
        self.assertEqual(self.actions["hours_reminder_fire"](""), "")
        self.say_ok = True
        self.actions["hours_reminder_fire"]("")
        self.assertEqual(len(self.said), 2)   # the retry went out
        self.actions["hours_reminder_fire"]("")
        self.assertEqual(len(self.said), 2)   # and is now marked

    def test_explicit_now_speaks_even_off_week_and_does_not_mark(self):
        self.at(2026, 10, 20, 9, 0)
        out = self.actions["hours_reminder_fire"]("now")
        self.assertTrue(out.startswith("Not this week"))
        self.assertEqual(len(self.said), 1)


class ArmTests(_Base):
    def test_arms_the_monday_cron_job(self):
        sched = FakeScheduler()
        self.at(2026, 10, 20, 9, 30)       # off week: no catch-up
        line = self.mod._arm_with(sched, self.actions)
        self.assertIn("armed Mondays 09:00", line)
        kind, kw = sched.calls[0]
        self.assertEqual(kind, "cron")
        self.assertEqual(kw["action"], "hours_reminder_fire")
        self.assertEqual((kw["hour"], kw["minute"]), (9, 0))
        self.assertEqual(kw["day_of_week"], "mon")
        self.assertEqual(kw["job_id"], "hours_reminder_monday")
        self.assertEqual(len(sched.calls), 1)

    def test_reminder_time_setting(self):
        self.cfg["HOURS_REMINDER_TIME"] = "07:45"
        sched = FakeScheduler()
        self.at(2026, 10, 20)
        self.mod._arm_with(sched, self.actions)
        self.assertEqual(
            (sched.calls[0][1]["hour"], sched.calls[0][1]["minute"]), (7, 45))

    def test_catch_up_job_when_due_and_unsaid(self):
        sched = FakeScheduler()
        self.at(2026, 10, 13, 10, 0)       # Tuesday of an hours week
        self.mod._arm_with(sched, self.actions)
        self.assertEqual([c[0] for c in sched.calls], ["cron", "once"])
        self.assertEqual(sched.calls[1][1]["job_id"], "hours_reminder_catchup")

    def test_no_catch_up_once_reminded(self):
        self.at(2026, 10, 13, 10, 0)
        self.mod._mark_reminded("2026-10-12")
        sched = FakeScheduler()
        self.mod._arm_with(sched, self.actions)
        self.assertEqual([c[0] for c in sched.calls], ["cron"])

    def test_disabled_removes_the_jobs(self):
        self.cfg["HOURS_REMINDER_ENABLED"] = False
        sched = FakeScheduler()
        line = self.mod._arm_with(sched, self.actions)
        self.assertIn("disabled", line)
        self.assertEqual(sched.calls, [])
        self.assertEqual(sorted(sched.cancelled),
                         ["hours_reminder_catchup", "hours_reminder_monday"])

    def test_arm_failure_is_reported_not_raised(self):
        sched = FakeScheduler()
        sched.schedule_cron = mock.Mock(side_effect=RuntimeError("no store"))
        self.at(2026, 10, 20)
        self.assertIn("could not arm", self.mod._arm_with(sched, self.actions))

    def test_boot_waits_for_the_scheduler_then_arms(self):
        sched = FakeScheduler()
        states = iter([False, False, True])
        self.at(2026, 10, 20)
        with mock.patch("core.scheduler.is_available", lambda: True), \
             mock.patch("core.scheduler.status",
                        lambda: {"running": next(states)}), \
             mock.patch("core.scheduler.schedule_cron", sched.schedule_cron), \
             mock.patch("core.scheduler.schedule_once", sched.schedule_once), \
             mock.patch("core.scheduler.cancel_job", sched.cancel_job), \
             mock.patch.object(self.mod.time, "sleep", lambda s: None):
            self.mod._arm_when_ready(self.actions)
        self.assertEqual(sched.calls[0][0], "cron")

    def test_no_apscheduler_falls_back_to_polling(self):
        started = []
        with mock.patch("core.scheduler.is_available", lambda: False), \
             mock.patch.object(self.mod, "_start_polling",
                               lambda a: started.append(True)):
            self.mod._arm_when_ready(self.actions)
        self.assertEqual(started, [True])

    def test_register_never_blocks_boot(self):
        # load_skill_isolated neuters Thread.start: register() returned.
        self.assertIn("hours_status", self.actions)


if __name__ == "__main__":
    unittest.main()
