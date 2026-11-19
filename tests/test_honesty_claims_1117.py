"""v3.0.26 (overhaul lane A, honesty): claims of switching something on or
off, setting it, starting or stopping it - and a decline with a reason.

Live 2026-11-16 02:02:35 "turn off lakework mode" (the recogniser's "wake-word
mode"): the first reply "I've turned off wake-word listening." was withheld
by the claim check, but the follow-up round said "Understood, sir. I've
disabled wake-word listening." and it WAS spoken - core/claim_validator had
no enable / disable / activate / deactivate family, so neither the in-turn
check nor the turn checker saw a claim in it, and nothing ran.

Live 2026-11-17 00:23:45 "Want the full list, sir?" -> "Not right now.
That'll take forever." took two withheld "On it" rounds and a close-out:
core.action_fit.is_bare_decline needs the WHOLE line to be ack words.

Pure: core modules only, no monolith (light-deps CI tier). The monolith
wiring is tests/monolith/test_monolith_honesty_1117.py.

    python tools/run_tests.py honesty_claims_1117
"""
from __future__ import annotations

import unittest

from core import action_fit as af
from core import claim_validator as cv
from core import config
from core import stt_vocab
from core import turn_checker as tc

# The live shapes (the owner's words reduced to the command phrase).
LIVE_ASK = "Jarvis turn off wake word mode."
LIVE_FIRST = ("[intent:confirmation] As you wish, sir. I've turned off "
              "wake-word listening.")
LIVE_FOLLOWUP = ("[intent:confirmation] Understood, sir. I've disabled "
                 "wake-word listening.")
LIVE_DECLINE = "Not right now. That'll take forever."
REGISTERED = ("wake_word_mode_on", "wake_word_mode_off",
              "wake_word_mode_status", "get_time", "set_volume",
              "smart_home_control", "pause_music", "standby_mode")


def claim(text, user="", ran=()):
    return cv.find_unverified_claim(text, ran_actions=ran, user_text=user)


class ToggleFamilyTests(unittest.TestCase):
    def test_the_live_follow_up_line_is_a_claim(self):
        self.assertEqual(claim(LIVE_FOLLOWUP, LIVE_ASK), "i've disabled")
        # ...and the first reply still is (the turn family).
        self.assertEqual(claim(LIVE_FIRST, LIVE_ASK), "i've turned off")

    def test_the_stream_gate_holds_it_back(self):
        # _SentenceFlushBuffer never voices a completed claim early.
        self.assertTrue(cv.find_completed_claim(LIVE_FOLLOWUP,
                                                user_text=LIVE_ASK))

    def test_the_turn_checker_sees_it_on_its_own(self):
        v = tc.check_turn(LIVE_ASK, LIVE_FOLLOWUP, [], [], REGISTERED)
        self.assertEqual(v.kind, tc.SAID_NO_ACTION)

    def test_every_first_person_form_is_a_claim(self):
        for text in ("I've disabled it, sir.", "I have enabled guest mode.",
                     "I just activated the standby protocol.",
                     "I've deactivated focus mode.",
                     "I've re-enabled wake-word mode, sir.",
                     "I'm enabling it now, sir.",
                     "I'll disable that for you.",
                     "Let me activate it.",
                     "Enabling guest mode now, sir.",
                     "Disabled, sir.",
                     "Very good, sir, deactivating it now."):
            with self.subTest(text=text):
                self.assertIsNotNone(claim(text, "do it"))

    def test_a_passive_now_disabled_is_a_claim(self):
        self.assertEqual(
            claim("Wake-word mode is now disabled, sir.", LIVE_ASK),
            "wake-word mode is now disabled")
        self.assertIsNotNone(claim("It has been enabled, sir.",
                                   "turn on focus mode"))

    def test_state_offers_and_nouns_are_not_claims(self):
        for user, text in (
                ("is wake word mode on", "Wake-word mode is disabled at the "
                                         "moment, sir."),
                ("why is it on", "It was enabled by default, sir."),
                ("that is annoying", "Shall I disable it?"),
                ("that is annoying", "I can disable it if you like, sir."),
                ("what happens then", "Disabling it would mean I answer "
                                      "everything, sir."),
                ("what helps", "Activated charcoal is great for that, sir."),
                ("did you", "I haven't enabled it yet, sir."),
                ("is it on", "It's enabled by default, sir."),
                ("what is it", "Enabled by default, it keeps the TV out.")):
            with self.subTest(text=text):
                self.assertIsNone(claim(text, user))

    def test_grounded_only_by_a_mode_action(self):
        self.assertIsNone(claim(LIVE_FOLLOWUP, LIVE_ASK,
                                ran=("wake_word_mode_off",)))
        self.assertIsNone(claim("I've activated the standby protocol, sir.",
                                "standby", ran=("standby_mode",)))
        # A device switched by the smart-home action (the turn family's).
        self.assertIsNone(claim("I've activated the porch lights, sir.",
                                "porch lights on",
                                ran=("smart_home_control",)))
        # An unrelated action grounds nothing.
        self.assertEqual(claim(LIVE_FOLLOWUP, LIVE_ASK, ran=("get_time",)),
                         "i've disabled")


class SetFamilyTests(unittest.TestCase):
    def test_claims(self):
        for text in ("I've set the volume to 40 percent, sir.",
                     "I set it to 72, sir.",
                     "Setting the brightness to half now, sir.",
                     "I'll set your thermostat at 70.",
                     "I've adjusted the volume, sir.",
                     "I've changed the output device, sir."):
            with self.subTest(text=text):
                self.assertIsNotNone(claim(text, "do it"))

    def test_not_claims(self):
        for text in ("I've set the record straight, sir.",
                     "You're all set, sir.",
                     "Setting goals helps, sir.",
                     "I've adjusted my tone, sir.",
                     "I've set aside some time for you.",
                     "Shall I set it to 40?"):
            with self.subTest(text=text):
                self.assertIsNone(claim(text, "hm"))

    def test_grounded_by_the_setter(self):
        self.assertIsNone(claim("I've set the volume to 40 percent, sir.",
                                "volume 40", ran=("set_volume",)))
        # A report of a device or media action that ran (_GROUND_ALSO).
        self.assertIsNone(claim("I've set the lamp to 40 percent, sir.",
                                "dim the lamp", ran=("smart_home_control",)))
        self.assertIsNone(claim("I've set the playlist to shuffle, sir.",
                                "shuffle it", ran=("spotify_shuffle",)))
        self.assertIsNotNone(claim("I've set the volume to 40 percent, sir.",
                                   "volume 40", ran=("get_time",)))


class StartStopFamilyTests(unittest.TestCase):
    def test_claims(self):
        for text in ("I've stopped the music, sir.",
                     "Starting the backup now, sir.",
                     "I'll stop it for you.",
                     "I've started a scan, sir.",
                     "I'm stopping the timer now."):
            with self.subTest(text=text):
                self.assertIsNotNone(claim(text, "do it"))

    def test_not_claims(self):
        for text in ("Starting the morning with a coffee, I see.",
                     "Let me stop you there, sir.",
                     "I stopped by the archive earlier.",
                     "The rain stopped an hour ago.",
                     "Stopping the war took decades.",
                     "Starting a business takes courage, sir."):
            with self.subTest(text=text):
                self.assertIsNone(claim(text, "hm"))

    def test_grounded_by_a_media_or_timer_action(self):
        self.assertIsNone(claim("I've stopped the music, sir.", "stop",
                                ran=("pause_music",)))
        self.assertIsNotNone(claim("I've stopped the music, sir.", "stop",
                                   ran=("get_time",)))


class DeclineWithReasonTests(unittest.TestCase):
    def test_the_live_line(self):
        self.assertTrue(af.is_decline_with_reason(LIVE_DECLINE))
        self.assertTrue(af.is_decline_with_reason("Jarvis, " + LIVE_DECLINE))

    def test_the_strict_predicates_are_unchanged(self):
        # Action-fit rule 7 still asks these, and they still say no.
        self.assertFalse(af.is_bare_ack(LIVE_DECLINE))
        self.assertFalse(af.is_bare_decline(LIVE_DECLINE))
        self.assertTrue(af.is_bare_decline("No, not right now."))

    def test_declines(self):
        for text in ("No thanks, I'm busy.", "Okay, no thanks. I'm good.",
                     "Nah, it's fine.", "No, you're good.",
                     "Not now, maybe later. Thanks.", "Nope. Not interested.",
                     "No thanks, I'm all set.", "No, not right now.",
                     "No. That'll take a while, sir.",
                     "Not right now, I'm heading out.",
                     "Not now, another time."):
            with self.subTest(text=text):
                self.assertTrue(af.is_decline_with_reason(text))

    def test_requests_corrections_and_questions_are_not(self):
        for text in ("No. Play the next one.", "No, play the next one",
                     "No. I meant the other one.",
                     "No. The lights are still on.",
                     "No, I want the weather.", "No. It's too loud.",
                     "No, it's the wrong song.",
                     "No. Why would you do that?",
                     "No, I'm watching TV, turn it down.",
                     "No. That's not what I asked.", "No, the other one.",
                     "No. Just turn it off.", "No, open Spotify instead.",
                     "Not right now?", "No. I'd rather you play jazz.",
                     "No, I'm going to bed, so wake me at seven.",
                     "Not now. Set a timer for ten minutes.", "No. Try again.",
                     "No, it is still playing.", "No. Do it again.",
                     # no decline at the head
                     "All right. That'll take forever.", "Great, I'm busy.",
                     "That'll take forever.", ""):
            with self.subTest(text=text):
                self.assertFalse(af.is_decline_with_reason(text))

    def test_never_raises(self):
        self.assertFalse(af.is_decline_with_reason(None))
        self.assertFalse(af.is_decline_with_reason(object()))


class SttAndConfigTests(unittest.TestCase):
    def fix(self, text):
        return stt_vocab.apply_replacements(text, config.STT_REPLACEMENTS)

    def test_lakework_is_the_wake_word(self):
        self.assertEqual(self.fix("Jarvis turn off lakework mode."),
                         "Jarvis turn off wake word mode.")
        self.assertEqual(self.fix("lake word mode on"), "wake word mode on")
        self.assertEqual(self.fix("lake work mode off"), "wake word mode off")
        self.assertEqual(self.fix("lakeword off"), "wake word off")

    def test_lake_work_alone_is_left_alone(self):
        self.assertEqual(self.fix("lots of lake work this summer"),
                         "lots of lake work this summer")

    def test_the_built_in_names_stay(self):
        self.assertEqual(config.STT_REPLACEMENTS["yetsi"], "Etsy")

    def test_defaults(self):
        self.assertIs(config.TURN_CHECK_LOCAL_CORRECTION, True)
        self.assertIs(config.ACK_CLAIM_NO_FOLLOWUP, True)
        self.assertIn("haven't actually done that", tc.LOCAL_CORRECTION_LINE)
        self.assertIn("LOCAL_CORRECTION_LINE", tc.__all__)


if __name__ == "__main__":
    unittest.main()
