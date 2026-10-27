"""Tests for core/advice_fallback.py — a request for a suggestion gets one.

Live v2.0.129-v2.0.134 (2026-09-29): "what should I have for dinner tonight"
-> "A bold choice, if I may say so, sir — though I'm afraid my culinary
expertise is somewhat limited to data processing." The fallback replaces a
dodged MEAL question with a bundled suggestion, and trims a misfit "A bold
choice" opener off a reply that does suggest something. Everything else
passes through untouched. Pure module, stdlib only.

    python -m unittest tests.test_advice_fallback
"""
from __future__ import annotations

import datetime as dt
import random
import unittest
from unittest import mock

from core import advice_fallback as af

LIVE_DODGE = ("[intent:dry_wit] A bold choice, if I may say so, sir — "
              "though I'm afraid my culinary expertise is somewhat limited to "
              "data processing.")
EVENING = dt.datetime(2026, 9, 29, 19, 46)
MORNING = dt.datetime(2026, 9, 29, 7, 30)
NOON = dt.datetime(2026, 9, 29, 12, 15)
LATE = dt.datetime(2026, 9, 29, 23, 40)


class MealRequestTests(unittest.TestCase):
    def test_named_meal_wins(self):
        cases = {
            "what should I have for dinner tonight": "dinner",
            "Jarvis, what should I eat for lunch?": "lunch",
            "what's for breakfast": "breakfast",
            "any good dinner ideas?": "dinner",
            "give me some lunch suggestions": "lunch",
            "suggest something for supper": "dinner",
            "what should I make tonight": "dinner",
            "I don't know what to eat for dinner": "dinner",
            "what should I grab for a snack": "snack",
            "what do you think I should have for brunch": "breakfast",
        }
        for text, meal in cases.items():
            with self.subTest(text=text):
                self.assertEqual(af.meal_of_request(text, MORNING), meal)

    def test_the_clock_decides_when_no_meal_is_named(self):
        for now, meal in ((MORNING, "breakfast"), (NOON, "lunch"),
                          (EVENING, "dinner"), (LATE, "snack")):
            with self.subTest(now=now):
                self.assertEqual(af.meal_of_request("what should I eat", now),
                                 meal)

    def test_not_meal_requests(self):
        for text in ("what should I get my dad for his birthday",
                     "what should I have done differently",
                     "what should I watch tonight",
                     "I had tacos for dinner", "dinner was great",
                     "remind me about dinner at six", "", None,
                     "what's the weather tonight"):
            with self.subTest(text=text):
                self.assertIsNone(af.meal_of_request(text, EVENING))


class AdviceRequestTests(unittest.TestCase):
    def test_requests(self):
        for text in ("what should I watch tonight", "any movie suggestions?",
                     "what do you recommend", "recommend a good book",
                     "which one should I buy", "what's a good name for a cat",
                     "what should I have for dinner"):
            with self.subTest(text=text):
                self.assertTrue(af.is_advice_request(text))

    def test_statements_are_not_requests(self):
        for text in ("I'm skipping class tomorrow", "I'm going to buy it",
                     "I painted the garage orange", "thanks jarvis", "", None):
            with self.subTest(text=text):
                self.assertFalse(af.is_advice_request(text))


class ApplyTests(unittest.TestCase):
    def setUp(self):
        af._last[0] = None

    def test_the_live_dodge_gets_a_dinner_suggestion(self):
        out = af.apply(LIVE_DODGE, "what should I have for dinner tonight",
                       now=EVENING, rng=random.Random(3))
        self.assertIn(out, af.MEAL_SUGGESTIONS["dinner"])

    def test_the_meal_named_beats_the_clock(self):
        out = af.apply(LIVE_DODGE, "what should I have for breakfast",
                       now=EVENING, rng=random.Random(3))
        self.assertIn(out, af.MEAL_SUGGESTIONS["breakfast"])

    def test_other_dodges_are_caught(self):
        for reply in ("That's entirely up to you, sir.",
                      "Not really my department, sir, lacking taste buds.",
                      "I'm afraid I can't help there, sir; I don't eat.",
                      "Are you quite sure you want my opinion, sir?"):
            with self.subTest(reply=reply):
                self.assertIsNotNone(af.apply(
                    reply, "what should I eat", now=EVENING,
                    rng=random.Random(1)))

    def test_a_real_suggestion_is_kept(self):
        for reply in ("How about a stir-fry tonight, sir?",
                      "I'd suggest pasta, sir. Quick and filling.",
                      "Tacos, sir. It's Tuesday, after all.",
                      "Perhaps a curry, sir, if you have the time."):
            with self.subTest(reply=reply):
                self.assertIsNone(af.apply(
                    reply, "what should I have for dinner", now=EVENING))

    def test_a_plain_reply_without_a_dodge_is_kept(self):
        # No suggestion, but no dodge either: not ours to rewrite.
        self.assertIsNone(af.apply("Let me think about that, sir.",
                                   "what should I have for dinner",
                                   now=EVENING))

    def test_no_repeat_back_to_back(self):
        rng = random.Random(0)
        picks = [af.apply(LIVE_DODGE, "what should I have for dinner",
                          now=EVENING, rng=rng) for _ in range(12)]
        for a, b in zip(picks, picks[1:]):
            self.assertNotEqual(a, b)

    def test_bold_choice_on_a_real_choice_is_left_alone(self):
        for user in ("I'm skipping class tomorrow",
                     "I'm going to paint the garage orange"):
            with self.subTest(user=user):
                self.assertIsNone(af.apply("A bold choice, if I may say so, "
                                           "sir.", user, now=EVENING))

    def test_misfit_opener_is_trimmed_off_a_real_suggestion(self):
        out = af.apply("[intent:dry_wit] A bold choice, if I may say so, sir "
                       "— how about a documentary about bridges?",
                       "what should I watch tonight", now=EVENING)
        self.assertEqual(out, "[intent:dry_wit] How about a documentary "
                              "about bridges?")

    def test_misfit_opener_with_no_suggestion_after_it_is_left_alone(self):
        # A non-meal dodge has no bundled answer; the reply stands.
        self.assertIsNone(af.apply("A bold choice, sir. Entirely up to you.",
                                   "what should I watch tonight", now=EVENING))

    def test_empty_and_garbage_inputs(self):
        for reply, user in (("", "what should I eat"), (None, None),
                            (LIVE_DODGE, ""), (LIVE_DODGE, None)):
            with self.subTest(reply=reply, user=user):
                self.assertIsNone(af.apply(reply, user, now=EVENING))

    def test_never_raises(self):
        with mock.patch.object(af, "meal_of_request",
                               side_effect=RuntimeError("boom")):
            self.assertIsNone(af.apply(LIVE_DODGE, "what should I eat"))


class EarlyHoldTests(unittest.TestCase):
    def test_holds_a_dodging_first_sentence_on_an_advice_turn(self):
        self.assertTrue(af.early_hold("[intent:dry_wit] A bold choice, sir. ",
                                      "what should I have for dinner"))
        self.assertTrue(af.early_hold("That's entirely up to you, sir. ",
                                      "what should I eat"))

    def test_does_not_hold_other_turns_or_real_answers(self):
        self.assertFalse(af.early_hold("A bold choice, sir. ",
                                       "I'm skipping class tomorrow"))
        self.assertFalse(af.early_hold("How about tacos tonight, sir? ",
                                       "what should I have for dinner"))
        self.assertFalse(af.early_hold("It is 7:51 PM, sir. ",
                                       "what time is it"))


class UnearnedBoldChoiceTests(unittest.TestCase):
    """v3 W01: "A bold choice" answers a choice the owner stated, nothing else."""

    CASES = {
        # id: (owner's words, raw reply, what is left)
        "chat-02": ("Jarvis, honestly, does pineapple belong on pizza?",
                    "[intent:dry_wit] [wry] A bold choice, if I may say so, "
                    "sir. I'd wager most Italians would have a collective "
                    "aneurysm at the thought.",
                    "[intent:dry_wit] [wry] I'd wager most Italians would "
                    "have a collective aneurysm at the thought."),
        "c046": ("Jarvis, you will worry too much. Try to find me.",
                 "[intent:amused] A bold choice, if I may say so, sir. I'll "
                 "leave your digital invisibility to you, then.",
                 "[intent:amused] I'll leave your digital invisibility to "
                 "you, then."),
        "c045": ("yes",
                 "[intent:amused] A bold choice, if I may say so, sir. I "
                 "shall prepare the blackout curtains accordingly.",
                 "[intent:amused] I shall prepare the blackout curtains "
                 "accordingly."),
    }

    def test_replies_lose_the_opener(self):
        for name, (user, raw, want) in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(af.strip_unearned_bold_choice(raw, user),
                                 want)

    def test_c012_a_bare_opener_before_an_action_becomes_empty(self):
        out = af.strip_unearned_bold_choice(
            "[intent:amused] A bold choice, sir.",
            "Jarvis, turn on a video on YouTube on the left monitor.")
        self.assertEqual(out, "")

    def test_a_stated_choice_keeps_it(self):
        for user in ("I'm going to delete the migrations",
                     "I'm gonna skip class", "I think I'll sell the truck",
                     "I decided to quit", "let's wipe the drive",
                     "I'll just reinstall windows", "I’m going to try it"):
            with self.subTest(user=user):
                reply = ("[intent:dry_wit] A bold choice, if I may say so, "
                         "sir. Proceed with care.")
                self.assertEqual(af.strip_unearned_bold_choice(reply, user),
                                 reply)

    def test_only_a_leading_opener_is_touched(self):
        reply = "Noted, sir. A bold choice, as it happens."
        self.assertEqual(af.strip_unearned_bold_choice(reply, "hello"), reply)
        self.assertEqual(af.strip_unearned_bold_choice("", "x"), "")
        self.assertEqual(af.strip_unearned_bold_choice("Fine.", ""), "Fine.")

    def test_opener_variants(self):
        for opener in ("A bold choice.", "A bold choice, sir.",
                       "A bold choice, if I may.", "A bold choice, sir:",
                       "a bold choice, if I may say so, sir!"):
            with self.subTest(opener=opener):
                self.assertEqual(
                    af.strip_unearned_bold_choice(opener + " The answer is 4.",
                                                  "what is 2 + 2"),
                    "The answer is 4.")

    def test_a_sentence_that_merely_starts_the_same_way_is_kept(self):
        reply = ("[intent:dry_wit] A bold choice to question the hardware, "
                 "sir. It is perfectly normal.")
        self.assertEqual(af.strip_unearned_bold_choice(reply, "Is that normal?"),
                         reply)

    def test_the_persona_pool_is_unchanged(self):
        from core import persona
        self.assertIn("A bold choice, if I may say so, sir.",
                      persona.render_signature_phrase_pool())


class SuggestionPoolTests(unittest.TestCase):
    def test_every_meal_has_several_short_generic_lines(self):
        for meal, lines in af.MEAL_SUGGESTIONS.items():
            with self.subTest(meal=meal):
                self.assertGreaterEqual(len(lines), 3)
                for line in lines:
                    self.assertLessEqual(len(line), 140)
                    self.assertIn("sir", line)
                    self.assertTrue(af.has_suggestion(line) or
                                    af._DISH_RE.search(line.lower()),
                                    line)


if __name__ == "__main__":
    unittest.main()
