"""core/pronoun_switch.py — "turn it off" with nothing for "it" to mean.

Live 2026-10-01: "Jarvis, turn it off" reached the local model with no device
or media in play; it invented [ACTION: shutdown] and the action-name corrector
mapped that onto shutdown_jarvis. A bare pronoun on/off command with no recent
context now gets a short question instead of a guess.
"""
from __future__ import annotations

import unittest

from core import pronoun_switch as ps


class SwitchStateTests(unittest.TestCase):
    def test_bare_pronoun_switch_commands(self):
        cases = {
            "Jarvis, turn it off.": "off",
            "turn that off": "off",
            "turn this one off please": "off",
            "Turn it on": "on",
            "switch it off": "off",
            "shut that off": "off",
            "can you turn it off": "off",
            "could you please switch that on?": "on",
            "turn off that one": "off",
            "turn them off": "off",
            "hey jarvis turn it off now": "off",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(ps.switch_state(text), want)

    def test_anything_that_names_its_target_or_says_more_is_not_one(self):
        for text in ("turn off the lamp", "turn the music off",
                     "turn it off and on again", "turn it down",
                     "turn it up", "shut down", "turn off jarvis",
                     "shut it down", "what is it", "it is off",
                     "turn it off in ten minutes", "", None, 42):
            with self.subTest(text=text):
                self.assertIsNone(ps.switch_state(text))


class ClarifyingQuestionTests(unittest.TestCase):
    def test_the_question_mirrors_the_verb_and_state(self):
        self.assertEqual(ps.clarifying_question("Jarvis, turn it off."),
                         "Turn what off, sir?")
        self.assertEqual(ps.clarifying_question("switch that on"),
                         "Switch what on, sir?")
        self.assertEqual(ps.clarifying_question("shut it off"),
                         "Shut what off, sir?")

    def test_any_other_utterance_gets_the_generic_question(self):
        for text in ("do the thing", "", None):
            with self.subTest(text=text):
                self.assertEqual(ps.clarifying_question(text),
                                 ps.GENERIC_QUESTION)


class ReferentQuestionTests(unittest.TestCase):
    def test_no_context_at_all_asks(self):
        self.assertEqual(ps.referent_question("Jarvis, turn it off."),
                         "Turn what off, sir?")

    def test_stale_context_asks(self):
        self.assertEqual(
            ps.referent_question(
                "turn it off",
                prior_turn_age_s=ps.REFERENT_WINDOW_S + 1,
                last_action_age_s=ps.REFERENT_WINDOW_S + 1,
                media_age_s=ps.MEDIA_REFERENT_WINDOW_S + 1),
            "Turn what off, sir?")

    def test_a_recent_turn_is_a_referent_the_model_can_resolve(self):
        self.assertIsNone(ps.referent_question("turn it off",
                                               prior_turn_age_s=30.0))

    def test_a_recent_action_is_a_referent(self):
        self.assertIsNone(ps.referent_question("turn it off",
                                               last_action_age_s=60.0))

    def test_media_jarvis_started_is_a_referent(self):
        # Music JARVIS put on twenty minutes ago is still "it".
        self.assertIsNone(ps.referent_question("turn it off",
                                               media_age_s=20 * 60.0))

    def test_media_playing_now_is_a_referent(self):
        # Review 2026-10-02: Spotify the owner started by hand (SMTC reports
        # it playing) is "it" too - he asked "Turn what off, sir?" before.
        self.assertIsNone(ps.referent_question("turn it off",
                                               media_playing=True))
        self.assertIsNotNone(ps.referent_question("turn it off",
                                                  media_playing=False))

    def test_jarvis_speaking_a_moment_ago_is_a_referent(self):
        # A proactive line ("the chamber light is still on, sir"), a timer
        # going off: what JARVIS just said is what "it" means.
        self.assertIsNone(ps.referent_question("turn it off",
                                               spoke_age_s=30.0))
        self.assertIsNotNone(ps.referent_question(
            "turn it off", spoke_age_s=ps.REFERENT_WINDOW_S + 60))

    def test_point_to_control_owns_pronoun_commands(self):
        # With pointing on, "turn that off" resolves by where he points
        # (skills/kinect_pointing via the smart-home router) - never ask.
        self.assertIsNone(ps.referent_question("turn that off",
                                               pointing_enabled=True))

    def test_a_named_target_never_asks(self):
        self.assertIsNone(ps.referent_question("turn off the lamp"))

    def test_garbage_ages_are_treated_as_no_context(self):
        self.assertEqual(
            ps.referent_question("turn it off", prior_turn_age_s="soon",
                                 last_action_age_s=-5.0),
            "Turn what off, sir?")


if __name__ == "__main__":
    unittest.main()
