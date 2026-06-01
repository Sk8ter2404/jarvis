"""Logic tests for skills/repo_robot.py.

Tracks the REPO Robot project from a JSON state file + todo/log scans. The
skill reads real files at the project root, so every test either patches
_load_state to inject a controlled dict or redirects the file constants to temp
paths — nothing real is read or written. _save_state is mocked wherever the
morning volunteer would persist a stamp.

Covered: date parsing + natural-language formatting, robot-keyword line
detection, todo pending/done counting, blocker/parts derivations, the three
voice actions, and the morning-volunteer trigger logic (part-arrived and
blocker-resolved branches, plus the once-per-day guard).
"""
from __future__ import annotations

import datetime
import os
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _iso(d):
    return d.isoformat()


class DateHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_parse_date_valid(self):
        self.assertEqual(self.mod._parse_date("2026-06-01"),
                         datetime.date(2026, 6, 1))

    def test_parse_date_with_time_suffix(self):
        self.assertEqual(self.mod._parse_date("2026-06-01T12:30:00"),
                         datetime.date(2026, 6, 1))

    def test_parse_date_invalid_and_none(self):
        self.assertIsNone(self.mod._parse_date("not a date"))
        self.assertIsNone(self.mod._parse_date(None))

    def test_natural_days(self):
        today = datetime.date.today()
        self.assertEqual(self.mod._natural_days(today), "today")
        self.assertEqual(self.mod._natural_days(today - datetime.timedelta(days=1)),
                         "yesterday")
        self.assertEqual(self.mod._natural_days(today - datetime.timedelta(days=3)),
                         "3 days ago")
        self.assertEqual(self.mod._natural_days(today + datetime.timedelta(days=2)),
                         "in 2 days")


class RobotLineTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_matches_keywords(self):
        self.assertTrue(self.mod._is_robot_line("Flash the ESP32 firmware"))
        self.assertTrue(self.mod._is_robot_line("solder the SG90 servo wiring"))
        self.assertTrue(self.mod._is_robot_line("REPO Robot eye alignment"))

    def test_ignores_unrelated(self):
        self.assertFalse(self.mod._is_robot_line("buy groceries and milk"))
        # 'eye' alone must not match (deliberately conservative).
        self.assertFalse(self.mod._is_robot_line("I see what you mean"))


class CountTodoTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_counts_pending_and_done(self):
        todo = (
            "- [ ] Flash ESP32 with new firmware\n"
            "- [x] Order the servo for the robot arm\n"
            "- [ ] buy unrelated groceries\n"          # not robot → ignored
            "- [X] wire up the robot eye\n"
            "- [ ] reschedule dentist\n"                # not robot → ignored
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jarvis_todo.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(todo)
            with mock.patch.object(self.mod, "_TODO_FILE", path):
                pending, done = self.mod._count_todo_tasks()
        self.assertEqual(pending, 1)   # only "Flash ESP32..."
        self.assertEqual(done, 2)      # servo + robot eye

    def test_missing_file_returns_zeroes(self):
        with mock.patch.object(self.mod, "_TODO_FILE",
                               os.path.join(tempfile.gettempdir(), "nope_xyz.md")):
            self.assertEqual(self.mod._count_todo_tasks(), (0, 0))


class DerivedViewTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_open_blockers_filters_resolved(self):
        state = {"blockers": [
            {"text": "waiting on parts", "resolved": False},
            {"text": "fixed one", "resolved": True},
            "garbage-not-a-dict",
        ]}
        out = self.mod._open_blockers(state)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "waiting on parts")

    def test_just_arrived_parts_within_window(self):
        today = datetime.date.today()
        state = {"parts_on_order": [
            {"part": "SG90 servo", "eta": _iso(today - datetime.timedelta(days=1)),
             "arrived": False},                                   # arrived → include
            {"part": "ESP32", "eta": _iso(today - datetime.timedelta(days=20)),
             "arrived": False},                                   # too old → exclude
            {"part": "wires", "eta": _iso(today - datetime.timedelta(days=1)),
             "arrived": True},                                    # already flagged → exclude
            {"part": "future", "eta": _iso(today + datetime.timedelta(days=3)),
             "arrived": False},                                   # not yet due → exclude
        ]}
        arrived = self.mod._just_arrived_parts(state)
        self.assertEqual([p["part"] for p in arrived], ["SG90 servo"])


class RobotStatusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("repo_robot")

    def test_empty_state(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            out = self.actions["robot_status"]("")
        self.assertIn("state file is empty", out.lower())

    def test_status_assembles_bits(self):
        state = {
            "next_step": "flash the firmware",
            "blockers": [{"text": "b1", "resolved": False},
                         {"text": "b2", "resolved": False}],
            "parts_on_order": [{"part": "servo", "arrived": False}],
        }
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_count_todo_tasks", return_value=(3, 1)), \
             mock.patch.object(self.mod, "_recent_log_mentions", return_value=0):
            out = self.actions["robot_status"]("")
        self.assertIn("flash the firmware", out)
        self.assertIn("2 blockers", out)
        self.assertIn("1 part on order", out)
        self.assertIn("3 todo items pending", out)
        self.assertIn("1 robot task completed", out)   # done tail

    def test_blocker_action_none(self):
        with mock.patch.object(self.mod, "_load_state",
                               return_value={"blockers": []}):
            out = self.actions["robot_blocker"]("")
        self.assertIn("No active blockers", out)

    def test_blocker_action_single(self):
        state = {"blockers": [{"text": "PSU undersized",
                               "since": "2026-05-30", "resolved": False}]}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["robot_blocker"]("")
        self.assertIn("One blocker", out)
        self.assertIn("PSU undersized", out)

    def test_next_step_action_with_blocker_warning(self):
        state = {"next_step": "mount the head",
                 "blockers": [{"text": "x", "resolved": False}]}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["next_robot_step"]("")
        self.assertIn("mount the head", out)
        self.assertIn("1 open blocker", out)

    def test_next_step_action_none_recorded(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            out = self.actions["next_robot_step"]("")
        self.assertIn("No next step recorded", out)


class MorningVolunteerTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_part_arrived_volunteer(self):
        today = datetime.date.today()
        state = {
            "next_step": "wire the servo",
            "parts_on_order": [{"part": "SG90 servo",
                                "eta": _iso(today - datetime.timedelta(days=1)),
                                "arrived": False}],
        }
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_save_state") as save:
            out = self.mod.get_morning_volunteer_text()
        self.assertIn("SG90 servo", out)
        self.assertIn("arrived", out)
        # Persists the once-per-day stamp.
        save.assert_called_once()
        self.assertEqual(save.call_args[0][0]["last_volunteered_on"], today.isoformat())

    def test_blocker_resolved_volunteer(self):
        today = datetime.date.today()
        state = {
            "next_step": "continue assembly",
            "parts_on_order": [],
            "blockers": [{"text": "missing bracket", "resolved": True,
                          "resolved_at": _iso(today - datetime.timedelta(days=1))}],
        }
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.mod.get_morning_volunteer_text()
        self.assertIn("missing bracket", out)
        self.assertIn("clear", out)

    def test_no_volunteer_without_next_step(self):
        today = datetime.date.today()
        state = {"next_step": "",
                 "parts_on_order": [{"part": "x",
                                     "eta": _iso(today), "arrived": False}]}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            self.assertEqual(self.mod.get_morning_volunteer_text(), "")

    def test_already_volunteered_today_is_silent(self):
        today = datetime.date.today().isoformat()
        state = {"next_step": "do thing", "last_volunteered_on": today,
                 "parts_on_order": [{"part": "x", "eta": today, "arrived": False}]}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            self.assertEqual(self.mod.get_morning_volunteer_text(), "")


if __name__ == "__main__":
    unittest.main()
