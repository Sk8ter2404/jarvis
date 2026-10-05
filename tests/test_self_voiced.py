"""core/self_voiced.py - the one honest line JARVIS says for a self-voiced
action (a device chat) that returned without saying anything (2026-10-05).

Light tier (stdlib only). The monolith wiring - which paths say the line, the
bounded wait for another chat - is tests/monolith/test_monolith_self_voiced_
silent.py.

    python -m unittest tests.test_self_voiced
"""
from __future__ import annotations

import unittest

from core import dialogue
from core import self_voiced as sv


class SilentLineTests(unittest.TestCase):

    def test_a_reason_jarvis_knows_is_named(self):
        self.assertEqual(
            sv.silent_line("Chat not started: mic_muted."),
            "I'm afraid the chat didn't start, sir; the microphone is muted.")
        self.assertEqual(
            sv.silent_line("Banter finished: 0 lines, device_lost."),
            "I'm afraid the banter didn't start, sir; the device stopped "
            "answering.")

    def test_a_skills_private_reason_is_never_guessed_at(self):
        for res in ("Chat not started: binding.",
                    "Chat not started: unreachable.",
                    "Chat not started: api_missing."):
            with self.subTest(res=res):
                self.assertEqual(sv.silent_line(res),
                                 "I'm afraid the chat didn't start, sir.")

    def test_an_unparseable_result_still_gets_a_line(self):
        for res in (None, "", 42, "ok", "x" * 500,
                    "This is a very long subject phrase not started: error."):
            with self.subTest(res=res):
                line = sv.silent_line(res)
                self.assertTrue(line.startswith("I'm afraid "), line)
                self.assertIn("didn't start, sir", line)

    def test_subject(self):
        self.assertEqual(sv.subject_of("Chat not started: x."), "the chat")
        self.assertEqual(sv.subject_of("Desk Chat finished: 0 lines, error."),
                         "the desk chat")
        self.assertEqual(sv.subject_of("nothing to parse"), "that")
        self.assertEqual(sv.subject_of(None), "that")

    def test_the_last_reason_word_wins(self):
        self.assertEqual(sv.reason_of("Chat finished: 0 lines, error."),
                         "error")
        self.assertEqual(sv.reason_of("active, then device_busy"),
                         "device_busy")
        self.assertEqual(sv.reason_of("Chat not started: binding."), "")
        self.assertEqual(sv.reason_of(None), "")

    def test_owner_stops(self):
        for r in ("wake", "owner_stop", "interrupted"):
            with self.subTest(r=r):
                self.assertTrue(sv.stopped_by_owner(
                    f"Chat finished: 0 lines, {r}."))
        self.assertFalse(sv.stopped_by_owner("Chat finished: 0 lines, error."))
        self.assertFalse(sv.stopped_by_owner(None))

    def test_every_dialogue_reason_is_covered(self):
        # core.dialogue.REASONS: how a run ends. Each is a clause the owner
        # hears, an owner stop (nothing added) or "done".
        for r in dialogue.REASONS:
            with self.subTest(r=r):
                self.assertTrue(r in sv.REASON_CLAUSES
                                or r in sv.OWNER_STOP_REASONS or r == "done")

    def test_stopped_while_waiting_reads_as_an_owner_stop(self):
        # Nothing is added for it, like any owner stop before a word.
        self.assertTrue(sv.stopped_by_owner(sv.STOPPED_WHILE_WAITING))

    def test_spoke_then_raised_is_neither_a_crash_nor_a_failure_line(self):
        # So no failure round re-runs a chat that already talked, and no
        # terminal line is added after it.
        from core import failure_markers as fm
        for exc in (RuntimeError("x"), ValueError(), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__):
                res = sv.spoke_then_raised(exc)
                self.assertIn(type(exc).__name__, res)
                self.assertNotIn(fm.ACTION_CRASH_MARK, res)
                self.assertEqual(fm.terminal_failure_text(res), "")
                self.assertFalse(sv.stopped_by_owner(res))

    def test_lines_are_spoken_sentences(self):
        lines = [sv.BUSY_LINE] + [
            sv.silent_line(f"Chat not started: {r}.") for r in sv.REASON_CLAUSES]
        for line in lines:
            with self.subTest(line=line):
                self.assertTrue(line.endswith("."))
                self.assertIn("sir", line)
                self.assertNotIn("_", line)       # no raw reason words
                self.assertNotIn("[", line)
                self.assertLess(len(line.split()), 25)


if __name__ == "__main__":
    unittest.main()
