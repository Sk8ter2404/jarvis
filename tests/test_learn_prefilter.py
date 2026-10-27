"""core/learn_prefilter.py (v3 W09): which owner turns can teach JARVIS.

Pure and table-driven. GENERIC sentences only, plus one fixture of real
owner turns from the session logs (commands and chat, no personal detail,
tests/_learn_prefilter_turns.json) that pins the skip rate.

    python -m unittest tests.test_learn_prefilter
"""
from __future__ import annotations

import json
import os
import unittest

from core import learn_prefilter as lp

# Turns that state something about the owner, ask to be remembered, set a
# preference or correct JARVIS. Every one must reach the extractor.
KEEP = [
    "Jarvis, I have gigabit up and down.",     # a real [learned] fact, 10-18
    "remember that my sister's birthday is in March",
    "Jarvis, remember I take my coffee black",
    "please remember the garage code changed",
    "call me Tony from now on",
    "my name is Alex",
    "Jarvis, my name's Alex by the way",
    "don't forget I have a dentist appointment on Friday",
    "I am a senior at the university",
    "I'm studying manufacturing engineering",
    "I'm a vegan now",
    "I like jazz",
    "I love working late at night",
    "I prefer metric units for printing",
    "I work at an IT company",
    "I live near the lake",
    "I have a paper due this Friday that must be submitted in person",
    "I have two cats and a dog",
    "I always eat lunch at noon",
    "I never use the left monitor for video",
    "I usually go to bed around eleven",
    "My printer is a Bambu H2D",
    "my sister lives in Ohio",
    "My favorite color is green",
    "And I enjoy cleaning the house on Sundays",
    "I bought a new standing desk last week",
    "from now on call the lamp the reading light",
    "no, I said Accelo not Excel",
    "that's wrong, it's the left monitor",
    "Actually it's Brayden with a y",
    "I meant the other window",
    "Jarvis, turn on air mouse. I work nights so keep it quiet.",
    "we have a gigabit connection at home",
    "the construction downstairs is replacing the floors this week",
]

# Commands, thanks, questions, polite requests, one-to-three-word turns.
SKIP = [
    "open Chrome",
    "Jarvis, open Teams on the top monitor.",
    "Jarvis plays Skrillex on YouTube.",
    "Jarvis paused the music.",
    "pause the music",
    "turn the volume down to twenty percent",
    "turn my fan on",
    "close all of that out for me",
    "play some Michael Jackson on YouTube",
    "hand tracking off",
    "Jarvis card mode.",
    "volume fifty",
    "thanks",
    "Thank you.",
    "thanks a lot that was great",
    "okay",
    "yes",
    "yes please",
    "never mind",
    "what time is it",
    "Jarvis, what's the weather today?",
    "what's my battery at?",
    "how's the printer doing?",
    "is the server up?",
    "do you remember what I said yesterday",
    "what's my name?",
    "can you open Fortnite for me?",
    "could you turn the lights off",
    "please close the window",
    "I want you to be able to do all this stuff. Pass that over to Claude.",
    "I need you to check my calendar",
    "go ahead and shut down",
    "tell Claude to finish the report",
    "send that over to Claude",
    "remind me to buy filament",
    "I'm back",
    "I have a question",
    "Jarvis",
    "",
    "   ",
]


class DecisionTests(unittest.TestCase):
    def test_turns_that_can_teach_are_kept(self):
        for t in KEEP:
            with self.subTest(turn=t):
                ok, why = lp.decide(t, "", ())
                self.assertTrue(ok, f"{t!r} was skipped ({why})")

    def test_turns_that_cannot_teach_are_skipped(self):
        for t in SKIP:
            with self.subTest(turn=t):
                ok, why = lp.decide(t, "", ())
                self.assertFalse(ok, f"{t!r} was kept ({why})")

    def test_worth_learning_is_the_bool_of_decide(self):
        for t in KEEP + SKIP:
            self.assertEqual(lp.worth_learning(t), lp.decide(t)[0])

    def test_reason_is_a_fixed_word_never_the_text(self):
        for t in KEEP + SKIP:
            if not t.strip():
                continue
            why = lp.decide(t)[1]
            self.assertRegex(why, r"^[a-z-]+$")
            self.assertNotIn(t.lower(), why)

    def test_the_wake_word_does_not_count_as_words(self):
        # "Jarvis, thanks" is one word, not two.
        self.assertFalse(lp.worth_learning("Jarvis, thanks"))
        self.assertFalse(lp.worth_learning("hey Jarvis open YouTube"))
        self.assertTrue(lp.worth_learning("hey Jarvis, I like jazz"))

    def test_curly_apostrophes_match(self):
        self.assertTrue(lp.worth_learning("I’m studying robotics at school"))
        self.assertTrue(lp.worth_learning("my name’s Alex"))

    def test_a_remembering_reply_keeps_an_unclear_turn(self):
        self.assertFalse(lp.worth_learning("open the report", "Opening it."))
        self.assertTrue(lp.worth_learning("open the report on Mondays",
                                          "I'll remember that."))

    def test_question_about_the_owner_is_not_a_statement(self):
        self.assertFalse(lp.worth_learning("what do I like to drink"))
        self.assertFalse(lp.worth_learning("do I have a meeting today"))
        self.assertFalse(lp.worth_learning("is my printer finished?"))

    def test_explicit_name_inside_a_question_is_still_kept(self):
        self.assertTrue(lp.worth_learning("can you call me Tony"))

    def test_an_action_turn_with_nothing_to_learn_is_skipped(self):
        # No verb or question shape, but an action ran: nothing to learn.
        t = "the left monitor then the lamp"
        self.assertTrue(lp.worth_learning(t, "", ()))
        self.assertFalse(lp.worth_learning(t, "", ("window_move",)))
        ok, why = lp.decide(t, "", {"window_move"})
        self.assertEqual((ok, why), (False, "action-turn"))

    def test_an_action_does_not_hide_a_stated_fact(self):
        self.assertTrue(lp.worth_learning(
            "I work nights so dim the lamp", "", ("lights_set",)))
        self.assertTrue(lp.worth_learning(
            "close all windows, my name is Alex", "", ("close_all",)))

    def test_never_raises_on_odd_input(self):
        for bad in (None, 5, b"x", ["a"], {"a": 1}, object()):
            self.assertIsInstance(lp.worth_learning(bad), bool)
        self.assertTrue(lp.worth_learning("I like jazz", None, None))
        self.assertTrue(lp.worth_learning("I like jazz", 7, 7))
        self.assertTrue(lp.worth_learning("I like jazz", "", [None, 3]))

    def test_error_inside_decide_keeps_the_turn(self):
        orig = lp._decide
        lp._decide = lambda *a, **k: 1 / 0
        try:
            self.assertEqual(lp.decide("open chrome"), (True, "error"))
        finally:
            lp._decide = orig

    def test_a_very_long_turn_is_judged_quickly(self):
        import time
        t0 = time.perf_counter()
        lp.decide("open " + "the window and " * 3000)
        self.assertLess(time.perf_counter() - t0, 2.0)


class ModeTests(unittest.TestCase):
    def test_normalize_mode(self):
        for raw, want in (("off", "off"), ("shadow", "shadow"), ("on", "on"),
                          ("ON", "on"), (" Shadow ", "shadow"),
                          (True, "on"), (False, "off"), (None, "off"),
                          ("yes", "off"), ("", "off"), (3, "off")):
            with self.subTest(raw=raw):
                self.assertEqual(lp.normalize_mode(raw), want)


FIXTURE = os.path.join(os.path.dirname(__file__), "_learn_prefilter_turns.json")


class LogFixtureTests(unittest.TestCase):
    """Owner turns lifted from the 2026-10-02..10-26 session logs (the 24
    current-era logs plus the archive). The one owner turn in them that
    produced a [learned] fact line with its text on record ("I have gigabit
    up and down") is in KEEP above; the other fact lines came from overheard
    speech whose text is never logged. The rest must mostly be skipped: the
    point of the pre-filter is that most turns are commands and questions."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as fh:
            cls.turns = json.load(fh)

    def test_fixture_is_real_sized(self):
        self.assertGreaterEqual(len(self.turns), 150)

    def test_at_least_half_of_the_rest_is_skipped(self):
        skipped = sum(1 for t in self.turns if not lp.worth_learning(t))
        rate = skipped / len(self.turns)
        self.assertGreaterEqual(rate, 0.50, f"skip rate {rate:.0%}")

    def test_no_turn_that_states_a_fact_about_the_owner_is_skipped(self):
        first_person = ("i have ", "i use ", "i hate ", "i need something",
                        "i feel", "my ")
        for t in self.turns:
            low = t.lower()
            body = low.split(",", 1)[-1].strip() if low.startswith(
                "jarvis,") else low
            if any(body.startswith(p) for p in first_person) and \
                    len(body.split()) > 4:
                with self.subTest(turn=t):
                    self.assertTrue(lp.worth_learning(t))


if __name__ == "__main__":
    unittest.main()
