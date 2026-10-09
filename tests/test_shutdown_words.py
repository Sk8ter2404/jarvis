"""The owner's own shutdown words (live 2026-10-06 22:28-22:30).

Four tries, zero shutdowns:

  1. "Jarvis shut down."            -> the overnight prompt was armed (fine).
  2. "Jarvis shut down no overnight protocol."
                                    -> "unrelated reply": the "no" sat
     mid-sentence. The prompt was cancelled and the model SAID "I've
     cancelled the overnight protocol and am powering down now." with
     nothing run - the claim check knew "shutting down", not "powering
     down".
  3. "Jarvis shut down with no overnight protocol."
                                    -> the model emitted shutdown_jarvis, but
     core.action_risk.asked_for_self_termination() read "with" as an object
     and held it for a yes.
  4. "Jarvis, yes, that is what I want."
                                    -> core.yes_no called it "other" (too many
     words after the yes): the held shutdown was cancelled.

Pinned here on the pure modules (core/action_risk.py, core/yes_no.py,
core/claim_validator.py) with the owner's exact words plus variants - and
the speech that must still NOT shut JARVIS down. The routers themselves are
pinned in tests/monolith/test_monolith_shutdown_words.py.

stdlib unittest only; CI-safe (light tier).
    python -m unittest tests.test_shutdown_words
"""
from __future__ import annotations

import unittest

from core import action_risk as ar
from core import claim_validator as cv
from core import yes_no

# The owner's exact four utterances, as transcribed live.
SAID_1 = "Jarvis shut down."
SAID_2 = "Jarvis shut down no overnight protocol."
SAID_3 = "Jarvis shut down with no overnight protocol."
SAID_4 = "Jarvis, yes, that is what I want."
# The reply spoken after (2) with no action run.
REPLY_2 = ("Very good, sir. I've cancelled the overnight protocol and am "
           "powering down now.")

# Speech that is NOT the owner asking JARVIS to go.
NOT_ASKING = (
    "Jarvis, don't shut down.",
    "Don't shut down",
    "Jarvis, do not shut down with no overnight protocol.",
    "No, don't shut down.",
    "Never go offline, Jarvis.",
    "Don't turn yourself off.",
    "should I shut down my PC?",
    "Should I shut down?",
    "Jarvis, shut down the printer.",
    "shut down the robot",
    "Jarvis, shut down the browser.",
    "Jarvis, shut down the printer with no overnight protocol.",
    'He said "shut down now."',
    'The sign said "shut down with no overnight protocol."',
    "The factory had to shut down with no warning.",
    "They shut down the highway without the permit.",
    "Shut down? No.",
    "It shut down last night.",
)


class AskedForSelfTerminationTests(unittest.TestCase):
    def test_the_owners_words_ask_for_a_shutdown(self):
        for said in (SAID_1, SAID_2, SAID_3,
                     "Jarvis, shut down without the protocol.",
                     "shut down without the overnight protocol",
                     "Jarvis, shut down, no overnight.",
                     "power down with no overnight protocol",
                     "Jarvis, shut down now.",
                     "shut down please",
                     "Jarvis, shut down for the night.",
                     "shut down right now",
                     "Jarvis, go offline with no upgrade."):
            for name in ("shutdown_jarvis", "exit_jarvis", "shut_down"):
                with self.subTest(said=said, name=name):
                    self.assertTrue(ar.asked_for_self_termination(name, said))

    def test_speech_that_is_not_the_owner_asking_does_not(self):
        for said in NOT_ASKING:
            for name in ("shutdown_jarvis", "exit_jarvis", "shut_down"):
                with self.subTest(said=said, name=name):
                    self.assertFalse(
                        ar.asked_for_self_termination(name, said))

    def test_a_decline_of_something_else_is_not_a_qualifier(self):
        for said in ("shut down with no warning", "shut down no",
                     "shut down not now", "shut down with chrome",
                     "shut down without saving"):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.asked_for_self_termination("shutdown_jarvis", said))

    def test_negation_and_other_subjects_reach_restart_and_overnight_too(self):
        for name, said in (("restart", "don't restart"),
                           ("restart", "Should I restart?"),
                           ("restart", "it restarted? no, did it restart"),
                           ("start_overnight_upgrade", "I can't sleep"),
                           ("start_overnight_upgrade", "don't go to bed yet")):
            with self.subTest(name=name, said=said):
                self.assertFalse(ar.asked_for_self_termination(name, said))
        for name, said in (("restart", "Jarvis, restart."),
                           ("start_overnight_upgrade", "Goodnight, Jarvis."),
                           ("shutdown_jarvis", "Jarvis, I said shut down."),
                           ("shutdown_jarvis", "can you shut down")):
            with self.subTest(name=name, said=said):
                self.assertTrue(ar.asked_for_self_termination(name, said))


class ShutdownDecliningOvernightTests(unittest.TestCase):
    def test_a_shutdown_that_says_no_to_the_overnight_protocol(self):
        for said in (SAID_2, SAID_3,
                     "Jarvis, shut down, no overnight protocol.",
                     "Hey Jarvis, shut down now with no overnight protocol, "
                     "please.",
                     "shut down without the protocol",
                     "Power down without the overnight protocol.",
                     "Jarvis, just shut down, no overnight.",
                     "No overnight protocol, just shut down.",
                     "Jarvis, without the overnight protocol, shut down."):
            with self.subTest(said=said):
                self.assertTrue(ar.shutdown_declining_overnight(said))
                self.assertTrue(
                    ar.shutdown_declining_overnight(said, alone_ok=True))

    def test_the_decline_alone_counts_only_while_the_question_is_open(self):
        for said in ("Without the overnight protocol.",
                     "Jarvis, with no overnight protocol.",
                     "without the protocol, please", "No overnight protocol."):
            with self.subTest(said=said):
                self.assertFalse(ar.shutdown_declining_overnight(said))
                self.assertTrue(
                    ar.shutdown_declining_overnight(said, alone_ok=True))

    def test_anything_else_is_not_that_shape(self):
        for said in NOT_ASKING + (
                SAID_1, SAID_4, "", None, "No.",
                "shut down with the overnight protocol",
                "Jarvis, shut down, yes, overnight protocol.",
                "no overnight protocol tomorrow, remind me",
                "without the overnight protocol I would be lost"):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.shutdown_declining_overnight(said, alone_ok=True))


class OwnerYesTests(unittest.TestCase):
    def test_the_owners_yes_is_a_yes(self):
        for said in (SAID_4, "Jarvis, yes please do.", "Jarvis, yes, do it.",
                     "Yes, that's what I want.", "Yes, that is what I said.",
                     "Yeah, that's what I meant.", "That's what I want.",
                     "Jarvis, yes, that's what I asked for."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "yes")

    def test_lookalikes_stay_other_or_no(self):
        for said in ("Yep, that is what she said.",
                     "Yes, that is what I want to know about the weather",
                     "That's what I want to know."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "other")
        for said in ("No, that is what I want to avoid.",
                     "Yes, that is what I want, but later.",
                     "No, don't."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "no")


class PoweringDownClaimTests(unittest.TestCase):
    def test_the_live_reply_is_an_unverified_claim(self):
        self.assertTrue(cv.find_unverified_claim(REPLY_2, user_text=SAID_2))

    def test_other_ways_of_saying_it(self):
        for reply in ("Powering down now, sir.",
                      "Very good, sir. Powering down.",
                      "I'm powering down now, sir.",
                      "I'll power down now, sir.",
                      "Understood, sir. I'm shutting myself down.",
                      "Shutting myself down, sir.",
                      "Going offline now, sir.",
                      "I'm going offline, sir.",
                      "I'm turning myself off now.",
                      "Understood, sir. I'm shutting down now."):
            with self.subTest(reply=reply):
                self.assertTrue(cv.find_unverified_claim(reply,
                                                         user_text=SAID_3))

    def test_grounded_offers_and_facts_are_not_claims(self):
        self.assertIsNone(cv.find_unverified_claim(
            "Powering down now, sir.", ran_actions=["shutdown_jarvis"],
            user_text=SAID_3))
        self.assertIsNone(cv.find_unverified_claim(
            "I've powered off the lamp, sir.",
            ran_actions=["smart_home_control"],
            user_text="turn off the lamp"))
        for reply in ("Shall I power down, sir?",
                      "Would you like me to go offline for the night?",
                      "Powering down a PC properly takes a few seconds.",
                      "The power went down for an hour last night, sir.",
                      "That would shut me down completely, sir. Say yes if "
                      "that is what you want."):
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_unverified_claim(reply,
                                                           user_text=SAID_1))


if __name__ == "__main__":
    unittest.main()
