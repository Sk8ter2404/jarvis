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
import time
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

    def test_no_volunteer_when_state_empty(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            self.assertEqual(self.mod.get_morning_volunteer_text(), "")

    def test_blocker_resolved_too_old_no_volunteer(self):
        today = datetime.date.today()
        state = {
            "next_step": "go",
            "parts_on_order": [],
            "blockers": [
                # First entry is unresolved → the loop `continue`s past it
                # (exercises the not-resolved guard) before reaching the stale
                # resolved one whose date is out of the 3-day window.
                {"text": "still open", "resolved": False},
                "not-a-dict",
                {"text": "old one", "resolved": True,
                 "resolved_at": _iso(today - datetime.timedelta(days=10))},
            ],
        }
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_save_state"):
            self.assertEqual(self.mod.get_morning_volunteer_text(), "")


class StateIOTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_load_state_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_state(), {})

    def test_load_state_reads_dict(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"next_step": "flash"}')
            with mock.patch.object(self.mod, "_STATE_FILE", path):
                self.assertEqual(self.mod._load_state(), {"next_step": "flash"})

    def test_load_state_non_dict_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("[1, 2, 3]")
            with mock.patch.object(self.mod, "_STATE_FILE", path):
                self.assertEqual(self.mod._load_state(), {})

    def test_load_state_corrupt_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data="{ bad json")):
            self.assertEqual(self.mod._load_state(), {})

    def test_save_state_roundtrips_and_stamps_updated_at(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with mock.patch.object(self.mod, "_STATE_FILE", path):
                ok = self.mod._save_state({"next_step": "go"})
                self.assertTrue(ok)
                reloaded = self.mod._load_state()
        self.assertEqual(reloaded["next_step"], "go")
        self.assertIn("updated_at", reloaded)

    def test_save_state_delegates_to_atomic_io(self):
        # The write goes through the shared core.atomic_io helper (mkstemp +
        # os.replace) rather than a fixed "<state>.tmp" sibling, so two
        # concurrent saves can't clobber each other's temp file.
        with mock.patch.object(self.mod.os, "makedirs", return_value=None), \
             mock.patch.object(self.mod, "_atomic_write_json") as aw:
            ok = self.mod._save_state({"next_step": "go"})
        self.assertTrue(ok)
        aw.assert_called_once()
        # Called with (_STATE_FILE, the stamped state dict).
        args, _kwargs = aw.call_args
        self.assertEqual(args[0], self.mod._STATE_FILE)
        self.assertEqual(args[1]["next_step"], "go")
        self.assertIn("updated_at", args[1])

    def test_save_state_write_failure_returns_false(self):
        # An atomic-writer failure (e.g. read-only fs, permission denied) is
        # caught and reported as False rather than propagating.
        with mock.patch.object(self.mod.os, "makedirs", return_value=None), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("read-only fs")):
            self.assertFalse(self.mod._save_state({"x": 1}))


class TodoScanResilienceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_count_todo_read_error_swallowed(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.mod._count_todo_tasks(), (0, 0))


class RecentLogMentionsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_no_logs_dir_returns_zero(self):
        with mock.patch.object(self.mod.os.path, "isdir", return_value=False):
            self.assertEqual(self.mod._recent_log_mentions(), 0)

    def test_counts_recent_robot_lines_only(self):
        with tempfile.TemporaryDirectory() as d:
            recent = os.path.join(d, "session_recent.log")
            stale = os.path.join(d, "session_old.log")
            notlog = os.path.join(d, "notes.txt")
            with open(recent, "w", encoding="utf-8") as f:
                f.write("Flashed the ESP32\nbought milk\nrobot eye wired\n")
            with open(stale, "w", encoding="utf-8") as f:
                f.write("servo soldering\n")
            with open(notlog, "w", encoding="utf-8") as f:
                f.write("esp32 mention in non-log\n")
            # Make `stale` old; keep `recent` fresh.
            old = time.time() - (self.mod._RECENT_HOURS + 5) * 3600
            os.utime(stale, (old, old))
            with mock.patch.object(self.mod, "_LOGS_DIR", d):
                count = self.mod._recent_log_mentions()
        # Only the two robot lines in the recent .log file.
        self.assertEqual(count, 2)

    def test_per_file_read_error_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            good = os.path.join(d, "session_a.log")
            with open(good, "w", encoding="utf-8") as f:
                f.write("esp32 flash\n")
            with mock.patch.object(self.mod, "_LOGS_DIR", d), \
                 mock.patch.object(self.mod.os.path, "getmtime",
                                   side_effect=OSError("stat fail")):
                # getmtime raising for each file → inner except → skipped.
                self.assertEqual(self.mod._recent_log_mentions(), 0)

    def test_listdir_error_swallowed(self):
        with mock.patch.object(self.mod.os.path, "isdir", return_value=True), \
             mock.patch.object(self.mod.os, "listdir",
                               side_effect=OSError("denied")):
            self.assertEqual(self.mod._recent_log_mentions(), 0)


class StatusTailAndBlockerBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("repo_robot")

    def test_status_includes_flash_date_and_log_tail(self):
        today = datetime.date.today()
        state = {
            "next_step": "calibrate",
            "last_firmware_flash": {"at": _iso(today - datetime.timedelta(days=2))},
        }
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_count_todo_tasks", return_value=(0, 0)), \
             mock.patch.object(self.mod, "_recent_log_mentions", return_value=4):
            out = self.actions["robot_status"]("")
        self.assertIn("last flash 2 days ago", out)
        self.assertIn("Recent log mentions: 4", out)

    def test_status_no_bits_uses_dash_lead(self):
        # Empty-ish but non-empty dict (so not the "state file is empty" path),
        # with nothing trackable → "no tracked state yet".
        state = {"misc": "value"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_count_todo_tasks", return_value=(0, 0)), \
             mock.patch.object(self.mod, "_recent_log_mentions", return_value=0):
            out = self.actions["robot_status"]("")
        self.assertIn("no tracked state yet", out)

    def test_blocker_action_multiple(self):
        state = {"blockers": [
            {"text": "PSU undersized", "since": "2026-05-30", "resolved": False},
            {"text": "missing bracket", "resolved": False},
        ]}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["robot_blocker"]("")
        self.assertIn("2 blockers", out)
        self.assertIn("PSU undersized", out)
        self.assertIn("missing bracket", out)

    def test_next_step_action_clean_no_blockers(self):
        state = {"next_step": "mount the head", "blockers": []}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["next_robot_step"]("")
        self.assertEqual(out, "Next, sir: mount the head.")


class JustArrivedEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("repo_robot")

    def test_part_without_eta_skipped(self):
        state = {"parts_on_order": [
            {"part": "no-eta part", "arrived": False},      # missing eta → skip
            "garbage",                                       # not a dict → skip
        ]}
        self.assertEqual(self.mod._just_arrived_parts(state), [])


class RegisterTests(unittest.TestCase):
    def test_register_wires_actions(self):
        mod, _ = load_skill_isolated("repo_robot", register=False)
        actions = {}
        mod.register(actions)
        for name in ("robot_status", "robot_blocker", "next_robot_step"):
            self.assertIn(name, actions)

    def test_register_swallows_makedirs_failure(self):
        mod, _ = load_skill_isolated("repo_robot", register=False)
        actions = {}
        with mock.patch.object(mod.os, "makedirs",
                               side_effect=OSError("denied")):
            mod.register(actions)   # must not raise
        self.assertIn("robot_status", actions)


if __name__ == "__main__":
    unittest.main()
