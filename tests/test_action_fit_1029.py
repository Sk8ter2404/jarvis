"""core/action_fit.py rules 6 and 7 as of v3.0.9 (live 2026-10-29).

THE LIVE TURNS (session_2026-10-28_20-18-06.log, JARVIS 3.0.7):
  18:54:35 / 18:55:16  "Jarvis, say <line>." ran the robot's speak action,
                       so the robot said it, not JARVIS;
  18:55:35  "I just want to say it, sir. Or if it's say <line>." (the
            recogniser's "I just want you to say it") - the same;
  18:55:26  "All right." ran the robot's speak action again with the same
            line (rule 6 only logged it, in shadow);
  18:53:47  "No, not right now." declined JARVIS's offer and the follow-up
            round re-ran the previous turn's pause_music.

Pinned (stdlib only, light tier):
  * rule 6: the robot speaks only when the owner's words make the ROBOT the
    speaker (a robot word or a third-person form); "say X", "can you say
    X", "I want you to say X" are JARVIS's; robot_speaker=False is the
    v3.0.8 rule 6; the no-speech-word branch is unchanged;
  * rule 7: the same (name, arg) again for a bare acknowledgement or
    decline; never for yes / sure / again / a request, never after an
    accepted offer, never with ack_repeat=False;
  * check_rule names the rule that fired; check() is its reason;
  * the robot word is ONE pattern shared with core/onscreen_refs.

The live words and action named the owner's robot; "deskbot" is a made-up
stand-in of the same shape, and the spoken lines are neutral.

    python tools/run_tests.py action_fit_1029
"""
from __future__ import annotations

import os
import sys
import unittest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import action_fit as af  # noqa: E402
from core import onscreen_refs as onscreen  # noqa: E402
from core import robot_words  # noqa: E402

SPEAK = "deskbot_speak"
LINE = "hello there"

# The live lines that asked JARVIS himself to speak.
LIVE_JARVIS_SAYS = (
    "Jarvis, say hello there.",
    "I just want to say it, sir. Or if it's say hello there.",
)
# More ways of asking JARVIS himself.
JARVIS_SAYS = LIVE_JARVIS_SAYS + (
    "say hello there", "can you say goodnight", "you say it",
    "I want you to say hello", "Jarvis, could you say that again for me",
    "I get it, say hello",
)
# The robot is the speaker: these still fit.
ROBOT_SAYS = (
    "have the bot say hi", "make him talk", "tell the robot I aced my exam",
    "let him introduce himself", "Jarvis, have it say hello",
    "can the robot say goodnight", "Jarvis, make the deskbut talk.",
    "get him to tell a joke", "tell him to say hi", "can he say hi",
    "he should say good morning", "deskbot, greet my roommate",
    "make the desk bot roast me", "have it read this out",
)
BARE_ACKS = (
    "All right.", "No, not right now.", "alright", "Okay.", "OK, sir.",
    "cool", "Nice.", "great", "Good.", "fine", "got it", "Thanks, Jarvis.",
    "thank you", "no", "nope", "nah", "not now", "not yet", "maybe later",
    "never mind", "forget it", "no thanks", "That's fine.", "I'm good",
    "all good", "Perfect.", "Understood, sir.", "Jarvis, no.",
    "okay, thanks", "that’s fine, thanks",
)
NOT_BARE = (
    "yes", "yeah", "yep", "sure", "please", "go ahead", "do it", "again",
    "okay play it", "ok, say it again", "all right, turn it off",
    "Alright?", "good night", "no, the other one", "Jarvis.", "sir", "",
    None,
)


class RobotSpeakerTests(unittest.TestCase):
    """Rule 6: the robot speaks only when sir makes the robot the
    speaker."""

    def test_the_live_lines_are_refused(self):
        for text in LIVE_JARVIS_SAYS:
            with self.subTest(text=text):
                rule, reason = af.check_rule(text, SPEAK, LINE)
                self.assertEqual(rule, af.RULE_ROBOT_SPEECH)
                self.assertIn("asked you (JARVIS) to say it", reason)
                self.assertIn("say the words yourself", reason)
                self.assertIn(SPEAK, reason)

    def test_asking_jarvis_himself_is_refused(self):
        for text in JARVIS_SAYS:
            with self.subTest(text=text):
                self.assertTrue(af.asks_robot_speech(text), text)
                self.assertFalse(af.asks_robot_speaker(text), text)
                self.assertIsNotNone(af.check(text, SPEAK, LINE))

    def test_the_robot_as_speaker_fits(self):
        for text in ROBOT_SAYS:
            with self.subTest(text=text):
                self.assertTrue(af.asks_robot_speaker(text), text)
                self.assertEqual(af.check_rule(text, SPEAK, LINE), (0, None))
                self.assertIsNone(af.check(text, SPEAK, LINE))

    def test_off_is_the_v308_rule(self):
        # robot_speaker=False: any speech word fits, as in v3.0.8.
        for text in JARVIS_SAYS:
            with self.subTest(text=text):
                self.assertEqual(af.check_rule(text, SPEAK, LINE,
                                               robot_speaker=False),
                                 (0, None))

    def test_no_speech_word_keeps_its_own_reason(self):
        text = "We need to give the deskbot more voice lines."
        for flag in (True, False):
            with self.subTest(robot_speaker=flag):
                rule, reason = af.check_rule(text, SPEAK, LINE,
                                             robot_speaker=flag)
                self.assertEqual(rule, af.RULE_ROBOT_SPEECH)
                self.assertIn("asked nobody to say anything", reason)

    def test_other_actions_are_not_rule_six(self):
        for nm in ("deskbot_banter", "tts_speak", "say_hello"):
            with self.subTest(nm=nm):
                self.assertEqual(af.check_rule("Jarvis, say hello there.",
                                               nm, LINE), (0, None))

    def test_no_owner_words_never_flags(self):
        self.assertEqual(af.check_rule("", SPEAK, LINE), (0, None))
        self.assertFalse(af.asks_robot_speaker(None))

    def test_the_robot_word_is_shared_with_onscreen_refs(self):
        self.assertIs(onscreen._ROBOT_WORD_STRICT,
                      robot_words.ROBOT_WORD_STRICT)
        self.assertIs(af.ROBOT_WORD_STRICT, robot_words.ROBOT_WORD_STRICT)
        # Common words ending in "bot" name no robot.
        for text in ("say hi to the abbot", "say it to the chatbot"):
            with self.subTest(text=text):
                self.assertFalse(af.asks_robot_speaker(text), text)


class BareAckTests(unittest.TestCase):
    def test_bare_acknowledgements_and_declines(self):
        for text in BARE_ACKS:
            with self.subTest(text=text):
                self.assertTrue(af.is_bare_ack(text), text)

    def test_never_bare(self):
        for text in NOT_BARE:
            with self.subTest(text=text):
                self.assertFalse(af.is_bare_ack(text), text)


class AckRepeatTests(unittest.TestCase):
    """Rule 7: the same action and argument again for a bare
    acknowledgement or decline."""

    PRIOR = [(SPEAK, LINE), ("pause_music", "")]

    def test_the_live_all_right_is_refused(self):
        rule, reason = af.check_rule("All right.", SPEAK, LINE, self.PRIOR)
        self.assertEqual(rule, af.RULE_ACK_REPEAT)
        self.assertIn("acknowledged or declined", reason)
        self.assertIn(f"{SPEAK}({LINE})", reason)

    def test_the_live_decline_does_not_rerun_pause_music(self):
        rule, reason = af.check_rule("No, not right now.", "pause_music", "",
                                     self.PRIOR)
        self.assertEqual(rule, af.RULE_ACK_REPEAT)
        self.assertIn("pause_music", reason)

    def test_a_request_or_a_yes_still_runs(self):
        for text in ("yes", "sure", "again", "okay play it", "say it again"):
            with self.subTest(text=text):
                rule, _reason = af.check_rule(text, "pause_music", "",
                                              self.PRIOR)
                self.assertNotEqual(rule, af.RULE_ACK_REPEAT)

    def test_an_accepted_offer_runs(self):
        self.assertEqual(af.check_rule("okay", "pause_music", "", self.PRIOR,
                                       accepted_offer=True), (0, None))

    def test_another_argument_or_no_prior_runs(self):
        self.assertEqual(af.check_rule("All right.", "pause_music", "chrome",
                                       self.PRIOR), (0, None))
        self.assertEqual(af.check_rule("All right.", "pause_music", ""),
                         (0, None))

    def test_off_is_no_rule_seven(self):
        self.assertEqual(af.check_rule("No, not right now.", "pause_music",
                                       "", self.PRIOR, ack_repeat=False),
                         (0, None))
        # With rule 7 off, "All right." to the robot falls to rule 6 (it
        # asks nobody to say anything), as in v3.0.8.
        rule, _reason = af.check_rule("All right.", SPEAK, LINE, self.PRIOR,
                                      ack_repeat=False)
        self.assertEqual(rule, af.RULE_ROBOT_SPEECH)


class CheckRuleTests(unittest.TestCase):
    def test_check_is_the_reason_of_check_rule(self):
        cases = (("Jarvis, get out of the way for just a second.",
                  "game_mode_on", ""),
                 ("Jarvis, say hello there.", SPEAK, LINE),
                 ("have the bot say hi", SPEAK, "hi"))
        for text, nm, arg in cases:
            with self.subTest(text=text):
                self.assertEqual(af.check(text, nm, arg),
                                 af.check_rule(text, nm, arg)[1])

    def test_rule_numbers(self):
        self.assertEqual(af.check_rule(
            "Jarvis, get out of the way for just a second.",
            "game_mode_on")[0], af.RULE_MODE_SWAP)
        self.assertEqual(af.check_rule("He does not sound happy.",
                                       "recognize_face")[0],
                         af.RULE_FACE_LOOK)
        self.assertEqual(af.check_rule(
            "Jarvis, turn on all smart home devices.",
            "list_smart_home_devices")[0], af.RULE_STATUS_FOR_COMMAND)

    def test_a_fault_reads_as_fits(self):
        class Boom:
            def __str__(self):
                raise RuntimeError("boom")
        self.assertEqual(af.check_rule("say hi", Boom()), (0, None))


if __name__ == "__main__":
    unittest.main()
