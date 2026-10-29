"""core/followup_window.py 3.0.7: keeps_extension + addressed_form.

Live 2026-10-28 19:08-19:12 (PC audio muted, wake-word mode, follow-up
window 20 s): room speech scoring 0.61-0.67 against the owner's voiceprint
was admitted line after line and each line extended the window. His own
commands score as low (0.61-0.73 on some that session), so the voice score
alone cannot refuse a follow-up: an unsure voice may not EXTEND the window,
and an unsure line that is not shaped like a command or question to JARVIS
is left unanswered. The fixtures below are neutral stand-ins, not log text.

Pure: no monolith.

    python -m unittest tests.test_followup_window_unsure
"""
from __future__ import annotations

import unittest

from core import learn_gate as lg
from core import followup_window as fw


class KeepsExtensionTests(unittest.TestCase):
    def test_only_a_matched_voice_or_no_voice_id_extends(self):
        self.assertTrue(fw.keeps_extension(lg.OWNER))
        self.assertTrue(fw.keeps_extension(lg.UNAVAILABLE))
        self.assertFalse(fw.keeps_extension(lg.UNSURE))
        self.assertFalse(fw.keeps_extension(lg.NOT_OWNER))
        self.assertFalse(fw.keeps_extension(None))

    def test_the_verdict_strings_match_the_learn_gate(self):
        # followup_window repeats them to stay import-free.
        self.assertEqual(fw._OWNER, lg.OWNER)
        self.assertEqual(fw._UNAVAILABLE, lg.UNAVAILABLE)


class AddressedFormTests(unittest.TestCase):
    def test_commands_are_kept(self):
        for text in ("Close every window except the editor, I'm heading out.",
                     "Volume up.", "Turn the lamp off.", "please open the notes",
                     "Okay, stop the timer.", "Don't do that again.",
                     "Be quiet for a while.", "Let's start the recording.",
                     "Mute yourself.", "I need you to read the last message.",
                     "We'd like you to dim the lights."):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "command")

    def test_questions_are_kept(self):
        for text in ("What's next on the list?", "Where?", "Is it raining?",
                     "how long until the print is done",
                     "Can you turn it off.", "Could you check the server"):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "question")

    def test_short_continuations_of_the_last_request_are_kept(self):
        for text in ("And tomorrow.", "Same for the kitchen.",
                     "Ten minutes instead.", "A bit quieter.",
                     "A little louder", "Much slower."):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "continuation")

    def test_auxiliary_questions_without_a_mark_are_kept(self):
        for text in ("Is it going to rain tomorrow.", "Are you there",
                     "Did you get that", "Is there anything new",
                     "Does that work"):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "question")

    def test_long_or_unshaped_lines_are_not_continuations(self):
        for text in ("And then he said it was all fine, honestly.",
                     "It was a bit quieter", "That was a lot louder than before",
                     "Instead.", "And", "Same"):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "")

    def test_continuation_form_never_raises(self):
        self.assertFalse(fw.continuation_form(None))
        self.assertFalse(fw.continuation_form(12))
        self.assertFalse(fw.continuation_form([]))

    def test_the_name_anywhere_is_kept(self):
        self.assertEqual(fw.addressed_form("Thanks, Jarvis."), "named")
        self.assertEqual(fw.addressed_form("so what does jarvis think"), "named")

    def test_statements_fragments_and_asides_are_not(self):
        for text in ("That turned out really well, everyone.",
                     "See you can't do it.",           # "see" is not a command
                     "I'm going to put the box on the top shelf.",
                     "Should the river carry me home",  # sung line, no "?"
                     "Did a nice job on that one.",     # auxiliary, no "?"
                     "Oh, it did that by itself. I was about to switch it off.",
                     "I'll cover that on the next slide.",
                     "Goodbye.", "Seems so.", "The shelf is crooked.",
                     "You're supposed to be on the left.",
                     "Jervis, you are meant to stay muted.",  # misheard name
                     "", "   ", None):
            with self.subTest(text=text):
                self.assertEqual(fw.addressed_form(text), "")

    def test_never_raises(self):
        self.assertEqual(fw.addressed_form(object()), "")
        self.assertEqual(fw.addressed_form(12), "")


if __name__ == "__main__":
    unittest.main()
