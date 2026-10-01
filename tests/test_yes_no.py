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
                   "Yes, delete it", "Yes I am", "Yes, I'm sure",
                   "Please, go ahead.", "Yes, that's right",
                   "Yes, go ahead and do it now"), "yes")

    def test_more_natural_yes_replies(self):
        # 2026-10-01 review: these were "other" (cancel + route on) - a
        # second, safe-but-annoying version of the cancel B001 reported.
        # "I'm sure." / "I am." answer the pushback "Are you certain?".
        self._all(("Please do.", "Go for it.", "I'm sure.", "I am.",
                   "I am certain", "Jarvis, I am sure.", "Okay, go for it",
                   "Sounds good."), "yes")

    def test_a_sentence_that_starts_with_a_yes_word_is_not_a_yes(self):
        # 2026-10-01 review, probed through the real confirmation gate: on
        # the first cut a strong yes word followed by ANY hedge-free sentence
        # was a yes, so each of these RAN a queued delete / purchase /
        # dangerous shell command (the base code declined them all).
        self._all(("Yeah, I saw that movie last week.",
                   "Yep, that is what she said.",
                   "Absolutely, the game was great.",
                   "Ya know what I mean?",
                   "Correct me if I am wrong, the file is big",
                   "Yeah right",
                   "Yeah, actually delete the other one",
                   "Yeah, I saw it", "Yeah, I know",
                   "yes and turn off the lights",
                   "I am going to bed", "I am hungry",
                   "Please do the dishes", "Please."), "other")

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


class CallerVocabularyTests(unittest.TestCase):
    """extra_yes / extra_no: the draft gates' "send it" and the printer
    wizard's "that's right" / "wrong" ride on the same rule (2026-10-01)."""

    SEND = (("send",), ("ship", "it"))

    def test_extra_yes_leads_follow_the_soft_rule(self):
        for text in ("Send it.", "Okay, send it", "Yes, ship it", "send"):
            with self.subTest(text=text):
                self.assertEqual(
                    yes_no.classify_reply(text, extra_yes=self.SEND), "yes")
        for text in ("Send it to the whole team instead of him",
                     "Yeah, I saw it", "send it later"):
            with self.subTest(text=text):
                self.assertNotEqual(
                    yes_no.classify_reply(text, extra_yes=self.SEND), "yes")

    def test_extra_no_words_refuse(self):
        self.assertEqual(yes_no.classify_reply("Wrong.", extra_no=("wrong",)),
                         "no")
        self.assertEqual(yes_no.classify_reply("Wrong."), "other")

    def test_without_extras_a_caller_word_is_not_a_yes(self):
        self.assertEqual(yes_no.classify_reply("Send it."), "other")


class HedgeWordsTests(unittest.TestCase):
    def test_hedges_found_idioms_skipped(self):
        # The shutdown prompt reads the words after its "no" with this.
        self.assertEqual(yes_no.hedge_words(["wait"]), ["wait"])
        self.assertEqual(yes_no.hedge_words("cancel that".split()),
                         ["cancel"])
        self.assertEqual(yes_no.hedge_words(["thanks"]), [])
        self.assertEqual(yes_no.hedge_words("no problem".split()), [])
        self.assertEqual(yes_no.hedge_words(None), [])


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
