"""core/claim_validator.py: PASSIVE completion claims (2026-10-02 live).

Parakeet, primary speech-to-text since 11:54 that day, writes the imperative
"close" as "closed". The owner asked JARVIS to close an app; the transcript
read as a past-tense sentence and the local brain answered in the passive
voice ("Very good, sir. <the app> has been closed.") with NO action token.
The reactive detector only knew first-person claims ("I've closed ...",
"Closing it now ..."), so nothing ran, nothing was corrected, and the owner
heard a completion that never happened.

The owner's past-tense words were NOT what let it through: no rule grounds a
claim on the owner's sentence, so "Jarvis closed <app>" never counts as the
action having run. The passive voice simply had no pattern.

These pin the new rule in both directions with made-up fixtures of the same
shape: a passive completion claim about the owner's target is caught when no
action ran; questions, recall of earlier turns, third-party facts and a
report of an action that did run are not. Pure module, stdlib only.

    python -m unittest tests.test_claim_validator_passive
"""
from __future__ import annotations

import unittest

from core import claim_validator as cv


def _flag(text, *, ran=(), user=""):
    return cv.find_unverified_claim(text, ran_actions=ran, user_text=user)


class PassiveClaimTests(unittest.TestCase):
    def test_the_live_shape_is_caught(self):
        # Same shape as the live turn: a misheard imperative, then a passive
        # completion claim with no [ACTION:] token.
        got = _flag("[intent:confirmation] Very good, sir. Notepad has been "
                    "closed.", user="Jarvis closed notepad.")
        self.assertIsNotNone(got)
        self.assertIn("closed", got)

    def test_passive_completion_claims_about_the_target(self):
        cases = (
            ("Your message has been sent, sir.", "send the message to the team"),
            ("The timer has been set for ten minutes, sir.",
             "set a timer for ten minutes"),
            ("Spotify has been opened on your left monitor.", "open spotify"),
            ("Everything else has been closed, sir.",
             "close everything except the editor"),
            ("All the windows have now been closed, sir.", "close all windows"),
            ("It's been done, sir.", "rename the folder"),
            ("That has been taken care of, sir.", "archive those emails"),
            ("The lights have been turned off, sir.", "lights off"),
            ("The calculator is now closed, sir.", "close the calculator"),
            ("Paint's been shut down, sir.", "jarvis closed paint"),
            ("The music has been paused.", "pause the music"),
            ("The spreadsheet was successfully closed, sir.",
             "close the spreadsheet"),
        )
        for text, user in cases:
            with self.subTest(text=text):
                self.assertIsNotNone(_flag(text, user=user))

    def test_an_unknown_turn_still_catches_a_pronoun_or_target_subject(self):
        for text in ("It has been closed, sir.", "The window has been closed.",
                     "Your email has been sent."):
            with self.subTest(text=text):
                self.assertIsNotNone(_flag(text))

    def test_a_past_tense_owner_sentence_grounds_nothing(self):
        # "Jarvis closed notepad" is the owner's (misheard) words, not a run.
        for text in ("I've closed Notepad, sir.", "Notepad has been closed.",
                     "Done, sir."):
            with self.subTest(text=text):
                self.assertIsNotNone(_flag(text, user="Jarvis closed notepad."))

    def test_a_report_of_an_action_that_ran_is_grounded(self):
        self.assertIsNone(_flag("Notepad has been closed, sir.",
                                ran=["close_window"],
                                user="close notepad"))
        self.assertIsNone(_flag("Your message has been sent, sir.",
                                ran=["send_message"],
                                user="send the message"))
        self.assertIsNone(_flag("It's been done, sir.", ran=["rename_file"],
                                user="rename the folder"))

    def test_a_different_family_does_not_ground_it(self):
        self.assertIsNotNone(_flag("Notepad has been closed, sir.",
                                   ran=["get_time"], user="close notepad"))


class PassiveFalsePositiveTests(unittest.TestCase):
    def test_questions_are_not_claims(self):
        for text in ("Has Notepad been closed?",
                     "Notepad has been closed?",
                     "Should it be closed, sir?"):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user="close notepad"))

    def test_an_owner_question_is_recall_not_a_claim(self):
        # A status / recall question answered from the conversation: the
        # reply reports state, it does not claim to act now.
        for user in ("did you close notepad?", "has notepad been closed",
                     "is my message sent?", "was the timer set"):
            with self.subTest(user=user):
                self.assertIsNone(_flag("Notepad has been closed, sir. Your "
                                        "message has been sent and the "
                                        "timer has been set.", user=user))

    def test_recall_of_an_earlier_turn_is_not_a_claim(self):
        for text in ("Notepad has been closed since this morning, sir.",
                     "Your message has been sent already, an hour ago.",
                     "The timer has been set earlier today, sir.",
                     "Notepad has been closed previously, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user="close notepad"))

    def test_third_party_facts_are_not_claims(self):
        for text, user in (
                ("The old bridge has been closed for repairs since 2019.",
                 "plan my drive to work"),
                ("The museum has been closed for decades, sir.",
                 "open my notes"),
                ("The record has been set by a sprinter from Jamaica.",
                 "open the sports news"),
                ("Jazz has been played in New Orleans for over a century.",
                 "play some jazz"),
                ("The case has been closed by the court.", "close notepad")):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user=user))

    def test_negations_and_failures_are_not_claims(self):
        for text in ("Notepad hasn't been closed, sir.",
                     "Notepad has not been closed yet, sir.",
                     "I'm afraid Notepad could not be closed, sir.",
                     "Notepad can't be closed from here, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user="close notepad"))

    def test_noun_uses_of_the_words_are_not_claims(self):
        for text in ("The deal has been a success, sir.",
                     "Your inbox has been busy today, sir.",
                     "The weather has been lovely this week."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user="check my inbox"))


class CompletedClaimTests(unittest.TestCase):
    """find_completed_claim: the streaming flush never voices a claim that
    something ALREADY happened before the reply's actions run. Progressive
    narration ahead of a token ("Opening it now") still flushes."""

    def _done(self, text, *, ran=(), user=""):
        return cv.find_completed_claim(text, ran_actions=ran, user_text=user)

    def test_completed_claims(self):
        for text, user in (
                ("I've closed every other window for you.", ""),
                ("I've taken the liberty of closing everything else.", ""),
                ("I sent the message, sir.", ""),
                ("Sent, sir.", ""),
                ("Very good, sir. Notepad has been closed.", "close notepad"),
                ("Done, sir.", "")):
            with self.subTest(text=text):
                self.assertIsNotNone(self._done(text, user=user))

    def test_narration_ahead_of_a_token_is_not_a_completion(self):
        for text in ("Opening Spotify now, sir.", "I'll send it right away.",
                     "Sending the rover out now.", "Let me take a look.",
                     "I'm closing it for you now.",
                     "The tower opened in 1889, sir.",
                     "Has Notepad been closed?"):
            with self.subTest(text=text):
                self.assertIsNone(self._done(text, user="close notepad"))

    def test_a_completion_grounded_by_an_action_that_ran(self):
        self.assertIsNone(self._done("Done, sir.", ran=["volume_up"]))
        self.assertIsNone(self._done("I've closed it, sir.",
                                     ran=["close_window"]))
        self.assertIsNotNone(self._done("I've closed it, sir.",
                                        ran=["get_time"]))


if __name__ == "__main__":
    unittest.main()
