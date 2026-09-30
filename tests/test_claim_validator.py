"""Tests for core/claim_validator.py — the reactive hallucinated-execution
check parse_and_run_actions runs on a reply that carries no [ACTION:] token.

Every rule is pinned in BOTH directions: the live false positive is gone, and
a real hallucinated claim is still caught. Pure module, stdlib only, so this
runs on the light-deps CI runner too.

    python -m unittest tests.test_claim_validator
"""
from __future__ import annotations

import unittest

from core import claim_validator as cv


def _flag(text, *, ran=(), user=""):
    return cv.find_unverified_claim(text, ran_actions=ran, user_text=user)


# ════════════════════════════════════════════════════════════════════════════
#  (A) third-party facts are never a claim; first-person claims still are
# ════════════════════════════════════════════════════════════════════════════
class ThirdPartyFactTests(unittest.TestCase):
    # The live v2.0.115 reply that forced "I can't actually move the moon".
    MOON = ("[intent:amused] If I may say so, sir... the moon is actually "
            "moving about 1.5 inches away from us every year.")

    def test_live_moon_fact_is_not_a_claim(self):
        self.assertIsNone(_flag(self.MOON))
        self.assertIsNone(_flag(self.MOON, user="tell me something interesting"))

    def test_third_party_progressive_facts_are_not_claims(self):
        for text in (
                "The moon is moving away from Earth at about 3.8 cm a year.",
                "The shop is closing soon, sir.",
                "The song playing on the radio is by a band from the 70s.",
                "Things are looking up, sir.",
                "Spotify is playing a jazz playlist right now.",
                "The window is on your left monitor, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text))

    def test_gerund_subjects_and_participial_openers_are_not_claims(self):
        for text in (
                "Moving house is one of the most stressful life events, sir.",
                "Playing chess regularly can reduce the risk of dementia.",
                "Searching for water on Mars has been a priority for decades.",
                "Launching a rocket costs about 60 million dollars.",
                "Opening a bottle of champagne releases carbon dioxide.",
                "Moving about 4 cm a year, the moon slowly drifts away.",
                "The moon, moving at 4 cm a year, slowly drifts away.",
                "Opened in 1889, the tower was the tallest structure on Earth.",
                "Sent to space in 1977, the probe is now past the heliosphere.",
                "Closed on Sundays, I'm afraid, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text))

    def test_idioms_questions_offers_and_retractions_are_not_claims(self):
        for text in (
                "Moving on, sir.",
                "Well done, sir.",
                "Well played, sir.",
                "Let me play devil's advocate for a moment, sir.",
                "Let me know if you need anything else.",
                "Switching gears, sir, your print is at 84%.",
                "Shall I open it for you, sir?",
                "I'll open it if you'd like, sir.",
                "I can't actually move the moon, sir.",
                "I haven't opened it yet, sir.",
                "I'm afraid I couldn't open that, sir.",
                "I've been thinking about that, sir.",
                "Running the numbers now.",
                "It is 2:53 PM on Tuesday, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text))

    def test_first_person_claims_are_still_caught(self):
        # The four shapes the fix must keep: perfect, future, subjectless
        # narration, "I have sent" — plus the classic "Restarting now".
        for text, phrase in (
                ("I've moved it to your left monitor, sir.", "i've moved"),
                ("I'll move the window now.", "i'll move"),
                ("Moving it now, sir.", "moving"),
                ("I have sent the email, sir.", "i have sent"),
                ("Restarting now, sir.", "restarting"),
                ("I've turned off the lights.", "i've turned off"),
                ("Opening Spotify now, sir.", "opening"),
                ("I'm opening the file now.", "i'm opening"),
                ("Very good, sir, opening your calendar.", "opening"),
                ("Excellent choice, sir, playing it now.", "playing"),
                ("Right away, sir — launching the game.", "launching"),
                ("Sent, sir.", "sent"),
                ("Queued three songs, sir.", "queued"),
                ("Setting a timer for five minutes, sir.", "setting a timer"),
                ("I'll remind you in ten minutes.", "i'll remind you"),
                ("Let me take a look.", "let me take a look"),
                ("Taking a screenshot now.", "taking a screenshot"),
                ("Turning the lights off, sir.", "turning the lights off"),
                ("Playing 'Can't Stop' now, sir.", "playing"),
                ("Restarting, it'll be about a minute, sir.", "restarting"),
                ("Opening the app, it'll be up shortly.", "opening")):
            with self.subTest(text=text):
                self.assertEqual(_flag(text), phrase)

    def test_word_boundaries_hold(self):
        # "reopening"/"unmoving" contain claim verbs but are other words.
        self.assertIsNone(_flag("The museum is reopening next week, sir."))
        self.assertIsNone(_flag("An unmoving statue, sir."))


# ════════════════════════════════════════════════════════════════════════════
#  (B) an acknowledgement preface on an answer is not claimed execution
# ════════════════════════════════════════════════════════════════════════════
class AcknowledgementTests(unittest.TestCase):
    LIVE = ("[intent:confirmation] On it, sir. You asked about the desk "
            "speaker's battery.")

    def test_live_ack_preface_on_an_answer_is_not_a_claim(self):
        self.assertIsNone(_flag(self.LIVE, user="what did I just ask you"))

    def test_ack_preface_on_other_answers(self):
        for user, reply in (
                ("tell me something interesting",
                 "On it, sir. Octopuses have three hearts."),
                ("what's the date tomorrow",
                 "Right away, sir. Tomorrow is Wednesday the 30th."),
                ("could you tell me who wrote Dune",
                 "Done, sir — Frank Herbert wrote it in 1965.")):
            with self.subTest(user=user):
                self.assertIsNone(_flag(reply, user=user))

    def test_bare_acks_are_still_claims_even_on_a_question(self):
        for reply, phrase in (("On it, sir.", "on it"),
                              ("Done, sir.", "done"),
                              ("Right away, sir.", "right away"),
                              ("[intent:confirmation] On it.", "on it"),
                              ("Of course, sir. Right away.", "right away"),
                              ("Consider it done, sir.", "consider it done")):
            for user in ("", "turn off the lights", "what time is it"):
                with self.subTest(reply=reply, user=user):
                    self.assertEqual(_flag(reply, user=user), phrase)

    def test_ack_plus_content_answering_a_command_is_still_a_claim(self):
        self.assertEqual(
            _flag("On it, sir. The lights are off.", user="turn off the lights"),
            "on it")
        self.assertEqual(
            _flag("Done, sir. Spotify is open.", user="can you open spotify"),
            "done")

    def test_ack_plus_content_with_unknown_turn_is_still_a_claim(self):
        # No owner utterance known (proactive path / direct call): exactly
        # the old behaviour — the acknowledgement counts as a claim.
        self.assertEqual(
            _flag("On it, sir. You asked about the battery."), "on it")

    def test_ack_plus_promise_on_a_question_is_still_a_claim(self):
        for reply, phrase in (
                ("On it, sir. I'll have that for you in a moment.", "on it"),
                ("Right away, sir. It should be ready shortly.", "right away")):
            with self.subTest(reply=reply):
                self.assertEqual(_flag(reply, user="what's the time"), phrase)

    def test_ack_plus_action_claim_on_a_question_is_still_a_claim(self):
        self.assertEqual(
            _flag("Right away, sir. Opening the forecast now.",
                  user="what's the weather like"),
            "opening")
        self.assertEqual(
            _flag("On it, sir. I've turned off the lights.",
                  user="are the lights on"),
            "i've turned off")


# ════════════════════════════════════════════════════════════════════════════
#  follow-up rounds: a claim that reads back this turn's result is grounded
# ════════════════════════════════════════════════════════════════════════════
class GroundingTests(unittest.TestCase):
    CMD = "play some jazz"

    def test_summary_of_a_same_family_action_is_grounded(self):
        for ran, reply in (
                (["play_music"], "Playing Blue Horizon by the Example Quartet, sir."),
                (["smart_home_control"], "I've turned off the lights, sir."),
                (["web_search"], "I searched the web and found three results."),
                (["set_timer"], "I've set a timer for five minutes, sir."),
                (["open_url"], "Opened the page for you, sir.")):
            with self.subTest(ran=ran):
                self.assertIsNone(_flag(reply, ran=ran, user=self.CMD))

    def test_claim_of_a_different_family_is_still_caught(self):
        for ran, reply, phrase in (
                (["get_time"], "Playing Blue Horizon by the Example Quartet, sir.",
                 "playing"),
                (["get_time"], "I've turned off the lights, sir.",
                 "i've turned off"),
                (["web_search"], "Opening the first result now, sir.",
                 "opening")):
            with self.subTest(ran=ran, reply=reply):
                self.assertEqual(_flag(reply, ran=ran, user=self.CMD), phrase)

    def test_completion_ack_after_a_real_action_is_grounded(self):
        self.assertIsNone(_flag("Done, sir.", ran=["smart_home_control"],
                                user="turn off the lights"))
        self.assertIsNone(_flag("Done, sir. The lights are off.",
                                ran=["smart_home_control"],
                                user="turn off the lights"))

    def test_completion_ack_with_nothing_run_is_a_claim(self):
        self.assertEqual(_flag("Done, sir.", ran=[],
                               user="turn off the lights"), "done")

    def test_bare_pending_ack_is_never_grounded(self):
        # "On it" promises something NEW — even after an earlier action ran.
        self.assertEqual(_flag("On it, sir.", ran=["get_time"],
                               user="what time is it"), "on it")


# ════════════════════════════════════════════════════════════════════════════
#  owner-question detection + the spoken preface strip
# ════════════════════════════════════════════════════════════════════════════
class LooksLikeQuestionTests(unittest.TestCase):
    def test_questions(self):
        for text in ("what did I just ask you", "what's the date tomorrow",
                     "tell me something interesting", "how's the weather",
                     "is it going to rain", "do you know what time it is",
                     "could you tell me the time", "jarvis, what time is it",
                     "remind me what I said", "have you seen my keys?",
                     "who wrote dune?"):
            with self.subTest(text=text):
                self.assertTrue(cv.looks_like_question(text))

    def test_commands(self):
        for text in ("turn off the lights", "can you open spotify",
                     "could you play some jazz", "do the dishes",
                     "have a look at my screen", "play some jazz",
                     "tell me when the print finishes", "remind me to stretch",
                     "", "   "):
            with self.subTest(text=text):
                self.assertFalse(cv.looks_like_question(text))


class LibertyAndLookupClaimTests(unittest.TestCase):
    """Live v2.0.131 (2026-09-29): "tell me something interesting" ->
    "I've taken the liberty of searching for something truly fascinating,
    but ..." with no [ACTION:] and no search. The persona's initiative opener
    wrapped round an action gerund is a first-person claim; so are "I've
    checked" / "I looked it up" / "I've done a quick search"."""

    LIVE = ("[intent:dry_wit] I've taken the liberty of searching for "
            "something truly fascinating, but I'm afraid your recent decision "
            "to skip dinner is proving far more interesting, sir.")

    def test_live_liberty_search_is_a_claim(self):
        self.assertEqual(_flag(self.LIVE, user="tell me something interesting"),
                         "i've taken the liberty of searching")

    def test_liberty_and_lookup_claims_are_caught(self):
        for text in (
                "I've taken the liberty of checking the weather, sir.",
                "I took the liberty of looking that up, sir.",
                "I'm taking the liberty of opening Spotify.",
                "I've checked the logs, sir. All clean.",
                "I checked your calendar, sir.",
                "I've double-checked the numbers.",
                "Let me check.",
                "Checked, sir.",
                "Checking now, sir.",
                "I've done a quick search, sir. Nothing of note.",
                "I'm verifying it now."):
            with self.subTest(text=text):
                self.assertIsNotNone(
                    _flag(text, user="tell me something interesting"))

    def test_third_party_and_idiomatic_uses_still_pass(self):
        for text in (
                "Octopuses have three hearts, sir.",
                "Scientists checked the data again in 1998.",
                "The committee looked into the matter for a year.",
                "Last time I checked, Pluto was a dwarf planet, sir.",
                "Checking your tyre pressure monthly is wise, sir.",
                "Checked in 1990, the records showed nothing.",
                "You should check the oil before a long drive.",
                "I've taken the liberty of assuming you'd like tea, sir.",
                "Shall I check the forecast?",
                "I can check if you'd like, sir.",
                "Searching for water on Mars has been a priority for decades."):
            with self.subTest(text=text):
                self.assertIsNone(
                    _flag(text, user="tell me something interesting"))

    def test_checking_its_own_arithmetic_is_not_an_action(self):
        # Review 2026-09-29: a conversion answer that re-checks its maths
        # must not cost a correction round; nor is "checking account" a verb.
        user = "convert 100 degrees fahrenheit to celsius"
        for text in (
                "Let me double-check my math: 100 degrees Fahrenheit is 37.8 "
                "degrees Celsius, sir.",
                "I've double-checked the arithmetic, sir: 37.8 degrees "
                "Celsius.",
                "I checked the conversion twice, sir: 37.8 degrees Celsius.",
                "Checking accounts typically pay little interest, sir."):
            with self.subTest(text=text):
                self.assertIsNone(_flag(text, user=user))
        self.assertIsNotNone(_flag("I checked the logs, sir.", user=user))

    def test_a_check_is_grounded_by_any_action_that_ran(self):
        self.assertIsNone(_flag("I've checked the logs, sir. All clean.",
                                ran=("read_logs",)))
        self.assertIsNone(_flag(self.LIVE, ran=("web_search",)))
        # ...but a search claim is not grounded by an unrelated action.
        self.assertIsNotNone(_flag(self.LIVE, ran=("get_time",)))


class StripAckPrefaceTests(unittest.TestCase):
    def test_strips_the_live_preface_and_keeps_the_intent_tag(self):
        out = cv.strip_ack_preface(AcknowledgementTests.LIVE,
                                   "what did I just ask you")
        self.assertEqual(
            out, "[intent:confirmation] You asked about the desk speaker's "
                 "battery.")

    def test_strips_without_tags(self):
        self.assertEqual(
            cv.strip_ack_preface("Done, sir — Frank Herbert wrote it.",
                                 "who wrote dune"),
            "Frank Herbert wrote it.")

    def test_leaves_everything_else_alone(self):
        for text, user in (
                ("On it, sir. The lights are off.", "turn off the lights"),
                ("On it, sir.", "what time is it"),
                ("On it, sir. I'll have it in a moment.", "what time is it"),
                ("On it, sir. Opening the forecast now.", "what's the weather"),
                ("Very good, sir. It is Tuesday.", "what day is it"),
                ("It is Tuesday, sir.", "what day is it"),
                ("", "what day is it")):
            with self.subTest(text=text):
                self.assertEqual(cv.strip_ack_preface(text, user), text)


if __name__ == "__main__":
    unittest.main()
