"""Tests for core/yes_no.py - the one classifier every spoken yes/no answer
goes through (the high-risk action confirmation, the autocorrect pick and the
shutdown prompt; 2026-10-01).

stdlib unittest only; CI-safe (light tier).
    python -m unittest tests.test_yes_no
"""
from __future__ import annotations

import unittest

from core import yes_no


class NormalizeTests(unittest.TestCase):
    def test_punctuation_and_wake_word_dropped(self):
        cases = {
            "Yes.": "yes",
            "No.": "no",
            "Jarvis, no.": "no",
            "Hey Jarvis, go ahead!": "go ahead",
            "okay jarvis, yes": "yes",
            "No, thanks.": "no thanks",
            "Yes, please.": "yes",
            "Yes, sir.": "yes",
            "Don't.": "dont",
            "Don’t do it": "dont do it",
            "Shut-down": "shut down",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(yes_no.normalize(text), want)

    def test_a_lone_wake_word_is_kept(self):
        self.assertEqual(yes_no.normalize("Jarvis."), "jarvis")
        self.assertEqual(yes_no.normalize("Okay, Jarvis."), "okay")

    def test_never_raises(self):
        self.assertEqual(yes_no.normalize(None), "")
        self.assertEqual(yes_no.normalize(12), "12")


class ClassifyReplyTests(unittest.TestCase):
    def _all(self, texts, want):
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(yes_no.classify_reply(text), want)

    def test_natural_yes_replies(self):
        # The confirmation gate used to CANCEL on every one of these but the
        # bare "yes": its list was yes/confirm/do it/go ahead/proceed only.
        self._all(("Yes.", "Yeah.", "Yep, do it", "Yup", "Sure.", "Okay.",
                   "ok", "Jarvis, yes.", "Hey Jarvis, go ahead", "Do it.",
                   "Proceed.", "Confirm", "Confirmed.", "Absolutely.",
                   "Yes, please.", "Sure thing.", "Okay, go ahead.",
                   "Sure, why not.", "Yes, no problem", "of course",
                   "yes and turn off the lights"), "yes")

    def test_lookalikes_are_never_yes(self):
        # A raw startswith confirmed a delete / purchase on each of these.
        for text in ("Yesterday we went to the store", "Confirmation number 5",
                     "Doing it tomorrow", "Goal", "Okay, what's the weather?",
                     "Fine, whatever you think about lunch"):
            with self.subTest(text=text):
                self.assertNotEqual(yes_no.classify_reply(text), "yes")

    def test_hedged_yes_is_a_no(self):
        self._all(("Do it later", "Go ahead and cancel it", "Yes, but wait",
                   "Yeah, I don't think so", "Absolutely not",
                   "Sure, but not now", "Yes, after lunch"), "no")

    def test_refusals(self):
        self._all(("No.", "Nope", "Nah", "Jarvis, no.", "No, thanks.",
                   "Cancel", "Stop", "Wait!", "Hold on", "Don't.", "Not now",
                   "Never mind"), "no")

    def test_unrelated(self):
        self._all(("what time is it", "turn on the lights", "", None,
                   "Jarvis.", "I don't know"), "other")


class SharedWordListTests(unittest.TestCase):
    def test_the_routers_share_one_vocabulary(self):
        # Every word the old per-router lists accepted is still a yes / no.
        for w in ("yes", "yeah", "yep", "yup", "sure", "ok", "okay",
                  "confirm", "proceed"):
            self.assertIn(w, yes_no.YES_WORDS)
        for w in ("no", "nope", "nah", "cancel", "stop", "nevermind"):
            self.assertIn(w, yes_no.NO_WORDS)
        self.assertFalse(yes_no.YES_WORDS & yes_no.NO_WORDS)


if __name__ == "__main__":
    unittest.main()
