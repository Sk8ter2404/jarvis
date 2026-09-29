"""Tests for skills/project_status.py — the GROUNDED answer to "what am I
working on?".

2026-09-29 live sweep: that question was answered from auto-learned topics
that were Whisper mis-hearings of TV audio. This skill answers from a list the
owner keeps (data/projects_status.json) and — the half that matters most —
says plainly when there is no list instead of inventing one.

Generic fixtures only. The data file is redirected to a temp dir through
JARVIS_DATA_DIR (core/paths resolves it at call time), so nothing here reads
or writes the real data/.

    python -m unittest tests.skills.test_project_status
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_EXAMPLE = os.path.join(_PROJECT_ROOT, "tools", "projects_status.example.json")
_TODAY = datetime.date(2026, 9, 28)


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="project_status_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.tmp})
        env.start()
        self.addCleanup(env.stop)
        self.mod, self.actions = load_skill_isolated("project_status")
        today = mock.patch.object(self.mod, "_today", return_value=_TODAY)
        today.start()
        self.addCleanup(today.stop)
        self.path = os.path.join(self.tmp, "projects_status.json")

    def _write(self, payload):
        with open(self.path, "w", encoding="utf-8") as fh:
            if isinstance(payload, str):
                fh.write(payload)
            else:
                json.dump(payload, fh)

    def ask(self, arg=""):
        return self.actions["project_status"](arg)


class RegistrationTests(_Base):
    def test_registered_and_spoken_verbatim(self):
        self.assertIn("project_status", self.actions)
        # Computed-and-dropped is the same defect wearing a different hat.
        self.assertIn("project_status", self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_reads_the_staging_aware_data_path(self):
        self.assertEqual(os.path.normcase(self.mod._path()),
                         os.path.normcase(self.path))


class NeverInventsTests(_Base):
    """No list -> say so. Never guess, never fall back to learned topics."""

    def test_absent_file(self):
        self.assertEqual(self.ask(), self.mod.NO_LIST)
        self.assertIn("won't guess", self.ask())

    def test_empty_list_and_missing_key(self):
        self._write({"projects": []})
        self.assertEqual(self.ask(), self.mod.NO_LIST)
        self._write({})
        self.assertEqual(self.ask(), self.mod.NO_LIST)

    def test_entries_without_names_do_not_count(self):
        self._write({"projects": [{"status": "half done"}, "junk", {"name": " "}]})
        self.assertEqual(self.ask(), self.mod.NO_LIST)

    def test_unreadable_file_is_disclosed(self):
        for bad in ("{ not json", "[1, 2]", '{"projects": "x"}'):
            with self.subTest(bad=bad):
                self._write(bad)
                self.assertEqual(self.ask(), self.mod.UNREADABLE)

    def test_a_query_with_no_list_still_does_not_invent(self):
        self.assertEqual(self.ask("garden shed"), self.mod.NO_LIST)


class ReadBackTests(_Base):
    def _list(self):
        self._write({"projects": [
            {"name": "Garden shed", "status": "Frame is up; roof next",
             "updated": "2026-09-20", "next": "Order shingles"},
            {"name": "Bird feeder", "status": "Printing the second prototype",
             "updated": "2026-09-27"},
            {"name": "Bookshelf", "status": "done", "updated": "2025-08-30"},
            {"name": "Kite", "updated": "not a date"},
        ]})

    def test_lists_active_projects_newest_first(self):
        self._list()
        out = self.ask()
        self.assertTrue(out.startswith("You have three active projects, sir."))
        self.assertLess(out.index("Bird feeder"), out.index("Garden shed"))
        self.assertIn("frame is up; roof next (updated 8 days ago); "
                      "next, order shingles.", out)
        self.assertIn("(updated yesterday)", out)
        self.assertIn("Kite: no status recorded.", out)
        # Finished work is not current work.
        self.assertNotIn("Bookshelf", out)
        self.assertIn("One more is marked finished.", out)

    def test_generic_arguments_are_not_treated_as_a_name(self):
        self._list()
        self.assertEqual(self.ask("what am I working on lately"), self.ask())

    def test_one_named_project(self):
        self._list()
        self.assertEqual(
            self.ask("the garden sheds"),
            "Garden shed, sir: frame is up; roof next (updated 8 days ago); "
            "next, order shingles.")
        self.assertIn("last updated August 30, 2025",
                      self.ask("bookshelf"))

    def test_unknown_name_is_honest(self):
        self._list()
        out = self.ask("robot arm")
        self.assertTrue(out.startswith("I don't see robot arm on your "
                                       "project list, sir."))
        self.assertIn("Garden shed", out)

    def test_substring_is_not_a_match(self):
        self._write({"projects": [{"name": "Carburettor rebuild",
                                   "status": "ordering parts"}]})
        self.assertTrue(self.ask("car").startswith("I don't see car"))

    def test_all_finished(self):
        self._write({"projects": [{"name": "Bookshelf", "status": "Finished."}]})
        self.assertIn("marked finished", self.ask())

    def test_long_lists_are_capped(self):
        self._write({"projects": [{"name": f"Project {chr(65 + i)}",
                                   "status": "going"} for i in range(8)]})
        out = self.ask()
        self.assertTrue(out.startswith("You have eight active projects, sir."))
        self.assertIn("And three more on the list.", out)
        self.assertEqual(out.count(": going"), self.mod._MAX_READ)


class ShippedExampleTests(_Base):
    """tools/projects_status.example.json is the documented starting point —
    it must parse under the skill's own loader and use only fake content."""

    def test_example_file_matches_the_schema(self):
        projects, state = self.mod.load_projects(_EXAMPLE)
        self.assertEqual(state, "ok")
        self.assertEqual(len(projects), 3)
        for p in projects:
            self.assertTrue(p["name"])
            self.assertIsNotNone(p["updated"])
        shutil.copy(_EXAMPLE, self.path)
        self.assertTrue(self.ask().startswith("You have two active projects"))


if __name__ == "__main__":
    unittest.main()
