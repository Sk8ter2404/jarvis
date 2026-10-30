"""core.recall_actions (v3.0.10, RECALL_ACTION_GROUNDING).

Live 2026-10-29 18:54:51: 14 s after the robot's speak action ran ("The
robot said it, sir."), sir asked whether it had been sent to the robot and
JARVIS called it "a bit of a misinterpretation". A recall / surprise
question now gets a prompt note listing the actions of the last minute.
Pure; stdlib unittest only.

    python tools/run_tests.py recall_actions
"""
from __future__ import annotations

import unittest

from core import recall_actions as ra

NOW = 1_000_000.0
_SPOKE = {"action": "robot_speak", "arg": "a line", "at": NOW - 14,
          "result": "The robot said it, sir."}


class IsRecallQuestionTests(unittest.TestCase):
    def test_recall_and_surprise_lines(self):
        for text in ("Wait, what? They sent it over to the robot?",
                     "did you send that to the robot?",
                     "Huh?", "what?", "hold on, was that the robot",
                     "why did it say that", "it said that?",
                     "Jarvis, did that go through?"):
            with self.subTest(text=text):
                self.assertTrue(ra.is_recall_question(text))

    def test_ordinary_lines(self):
        for text in ("what's the weather", "tell me a joke",
                     "what time is it?", "play some jazz", "", None,
                     "x" * 300):
            with self.subTest(text=text):
                self.assertFalse(ra.is_recall_question(text))


class RecallNoteTests(unittest.TestCase):
    def test_live_case_lists_the_action_and_result(self):
        note = ra.recall_action_note(
            "Wait, what? They sent it over to the robot?", [_SPOKE], now=NOW)
        self.assertIn("robot_speak('a line') -> 'The robot said it, sir.'",
                      note)
        self.assertIn("14 s ago", note)
        self.assertIn("never deny", note)

    def test_nothing_recent_or_not_a_question(self):
        old = dict(_SPOKE, at=NOW - 120)
        self.assertEqual(ra.recall_action_note("Wait, what?", [old],
                                               now=NOW), "")
        self.assertEqual(ra.recall_action_note("Wait, what?", [], now=NOW),
                         "")
        self.assertEqual(ra.recall_action_note("tell me a joke", [_SPOKE],
                                               now=NOW), "")

    def test_synthetic_and_bad_entries_skipped(self):
        junk = [{"action": "_unverified_claim", "at": NOW - 1},
                {"action": "x", "at": "soon"}, "junk", None]
        self.assertEqual(ra.recall_action_note("huh?", junk, now=NOW), "")

    def test_capped_entries_and_length(self):
        hist = [{"action": f"act{i}", "arg": "a" * 200, "result": "r" * 200,
                 "at": NOW - 50 + i} for i in range(6)]
        note = ra.recall_action_note("did you do that?", hist, now=NOW)
        listed = [ln for ln in note.splitlines() if ln.startswith("- ")]
        self.assertLessEqual(len(listed), ra.RECALL_MAX_ENTRIES)
        self.assertIn("act5", note)                 # the newest is kept
        self.assertLessEqual(sum(len(ln) for ln in listed),
                             ra.RECALL_MAX_CHARS + 200)

    def test_never_raises(self):
        class Bad:
            def __iter__(self):
                raise RuntimeError("boom")
        self.assertEqual(ra.recall_action_note("huh?", Bad(), now=NOW), "")


if __name__ == "__main__":
    unittest.main()
