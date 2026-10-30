"""core/claim_validator: self-change promises and the ack-claim drop
(v3.0.9, live 2026-10-29).

THE LIVE TURNS (session_2026-10-28_20-18-06.log, JARVIS 3.0.7):
  18:56:26  "Jarvis, when you make the <robot> talk, you need to put a
            little bit of delay in there ..." -> "Very good, sir. I'll build
            in a slight pause before the <robot> speaks.";
  18:56:56  "Jarvis, you need to make him finish talking before you speak."
            -> "Understood, sir. I'll ensure the <robot> completes its
            delivery before I chime in.";
  18:57:07  "Try it again." -> "Certainly, sir. I'll ensure he finishes his
            business before I interrupt.";
  18:53:47  "No, not right now." -> "Very good, sir. On it." (withheld as an
            unverified claim; the forced follow-up round re-ran an action).

Pinned (stdlib only, light tier):
  * find_self_change_promise flags the live promises and the documented
    shapes, and NOT: a promise an action this turn keeps, contrition, memory
    claims, offers, questions, action promises (lights, lists, reminders);
    the robot's own speak action keeps no promise;
  * drop_self_change_promises keeps the rest and its leading tags, and says
    whether anything of substance is left;
  * is_behaviour_directive reads the owner's live directives;
  * drop_unverified_claims leaves "Very good, sir." from the live reply.

"deskbot" stands in for the owner's robot (public code never names it).

    python tools/run_tests.py self_change_promises
"""
from __future__ import annotations

import os
import sys
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import claim_validator as cv  # noqa: E402

LIVE_REPLIES = (
    "Very good, sir. I'll build in a slight pause before the deskbot speaks.",
    "Understood, sir. I'll ensure the deskbot completes its delivery before "
    "I chime in.",
    "Certainly, sir. I'll ensure he finishes his business before I "
    "interrupt.",
)
PROMISES = LIVE_REPLIES + (
    "I'll add a short delay before I respond.",
    "I'll make sure to wait for him before I speak.",
    "From now on, I'll wait for him to finish talking.",
    "Next time I'll let him finish before I speak.",
    "I'll adjust my timing, sir.",
    "I'll be sure to give him a moment before I speak.",
    "I'll wait until he finishes talking.",
    "I will tweak my delivery, sir.",
    "Going forward, I'll hold off until the deskbot has finished talking.",
)
NOT_PROMISES = (
    "I'll remind you at 5, sir.",
    "I'll be more careful, sir.",
    "I'll do better next time, sir.",
    "I'll try to be quicker next time.",
    "I'll remember that, sir.",
    "I'll keep that in mind for next time.",
    "Shall I add a pause before the deskbot speaks?",
    "I can add a pause if you'd like.",
    "Would you like me to wait before I speak?",
    "I'll let you know when it finishes.",
    "I'll turn on the lights.",
    "I'll add it to your shopping list.",
    "See you next time, sir.",
    "I'll see you next time, sir.",
    "I'll set a reminder for that in the future.",
    "I'll give you a moment, sir.",
    "I'll pause the music for you.",
    "I can't change my timing myself, sir.",
    "I'm afraid I won't be able to wait before I speak.",
    "The deskbot paused before it spoke.",
    # Review 2026-10-29: contrition and ordinary plans with "next time".
    "Apologies, sir. I'll listen more carefully next time.",
    "I'll get it right next time, sir.",
    "I'll double-check next time.",
    "I'll ask him next time he's online.",
    "I'll save that for next time.",
    "I'll introduce you next time.",
    "I'll have it ready for you next time.",
    "I'll pay more attention next time.",
    "I'll wait for you next time, sir.",
    "I'll be listening more carefully, sir.",
    "",
)


class FindTests(unittest.TestCase):
    def test_the_live_replies_are_promises(self):
        for reply in LIVE_REPLIES:
            with self.subTest(reply=reply):
                self.assertIsNotNone(cv.find_self_change_promise(reply))

    def test_the_documented_shapes(self):
        for reply in PROMISES:
            with self.subTest(reply=reply):
                self.assertIsNotNone(cv.find_self_change_promise(reply),
                                     reply)

    def test_must_not_fire(self):
        for reply in NOT_PROMISES:
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_self_change_promise(reply), reply)

    def test_a_real_action_this_turn_may_keep_it(self):
        reply = "I'll make sure to add a pause before I speak, sir."
        self.assertIsNotNone(cv.find_self_change_promise(reply))
        for ran in (["add_reminder"], ["start_timer"], ["set_volume"]):
            with self.subTest(ran=ran):
                self.assertIsNone(cv.find_self_change_promise(
                    reply, ran_actions=ran))

    def test_the_robot_speak_action_keeps_no_promise(self):
        reply = LIVE_REPLIES[0]
        for ran in (["deskbot_speak"], ["_unverified_claim"]):
            with self.subTest(ran=ran):
                self.assertIsNotNone(cv.find_self_change_promise(
                    reply, ran_actions=ran))

    def test_never_raises(self):
        self.assertIsNone(cv.find_self_change_promise(None))
        self.assertEqual(cv.drop_self_change_promises(None), (None, [], True))


class ApologyTests(unittest.TestCase):
    def test_an_apology_is_kept_and_nothing_is_noted(self):
        reply = ("[intent:apology] Apologies, sir. I'll listen more "
                 "carefully next time.")
        self.assertEqual(cv.drop_self_change_promises(reply),
                         (reply, [], True))
        reply = "Very good, sir. I'll save that for next time."
        self.assertEqual(cv.drop_self_change_promises(reply),
                         (reply, [], True))


class DropTests(unittest.TestCase):
    def test_the_live_reply_leaves_courtesy_only(self):
        kept, flagged, substance = cv.drop_self_change_promises(
            "[intent:confirmation] " + LIVE_REPLIES[0])
        self.assertEqual(kept, "[intent:confirmation] Very good, sir.")
        self.assertEqual(len(flagged), 1)
        self.assertIn("pause", flagged[0])
        self.assertFalse(substance)

    def test_other_content_is_kept(self):
        kept, flagged, substance = cv.drop_self_change_promises(
            "The deskbot said it, sir. I'll wait for him to finish talking "
            "next time.")
        self.assertEqual(kept, "The deskbot said it, sir.")
        self.assertTrue(flagged)
        self.assertTrue(substance)

    def test_no_promise_is_unchanged(self):
        text = "Very good, sir. I'll remind you at 5."
        self.assertEqual(cv.drop_self_change_promises(text), (text, [], True))


class DirectiveTests(unittest.TestCase):
    def test_the_live_directives(self):
        for text in (
                "Jarvis, when you make the deskbot talk, you need to put a "
                "little bit of delay in there because you say that it said "
                "it and he's talking and you can't hear him.",
                "Jarvis, you need to make him finish talking before you "
                "speak.",
                "From now on wait for him.", "make sure you let him finish",
                "next time give him a second", "I want you to pause first",
                "you should wait for the deskbot", "you gotta slow down"):
            with self.subTest(text=text):
                self.assertTrue(cv.is_behaviour_directive(text), text)

    def test_not_directives(self):
        for text in ("Try it again.", "make him say hi",
                     "can you wait for him?", "when is my next class",
                     "", None):
            with self.subTest(text=text):
                self.assertFalse(cv.is_behaviour_directive(text), text)

    def test_the_lines(self):
        self.assertEqual(cv.SELF_CHANGE_HONEST_LINE,
                         "I can't change that myself, sir.")
        self.assertEqual(cv.SELF_CHANGE_NOTED_LINE,
                         "I can't change that myself, sir - I've noted it "
                         "for Claude.")


class AckClaimDropTests(unittest.TestCase):
    def test_the_live_reply_keeps_very_good(self):
        self.assertEqual(cv.drop_unverified_claims(
            "[intent:confirmation] Very good, sir. On it.",
            user_text="No, not right now."),
            "[intent:confirmation] Very good, sir.")

    def test_only_the_claim_left_is_empty(self):
        self.assertEqual(cv.drop_unverified_claims(
            "On it.", user_text="No, not right now."), "")
        self.assertEqual(cv.ACK_CLAIM_LINE, "Very good, sir.")

    def test_every_claim_goes(self):
        self.assertEqual(cv.drop_unverified_claims(
            "Very good, sir. I've paused the music. On it.",
            user_text="okay"), "Very good, sir.")

    def test_never_raises(self):
        self.assertEqual(cv.drop_unverified_claims(None), "")


if __name__ == "__main__":
    unittest.main()
