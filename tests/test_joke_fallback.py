"""Tests for core/joke_fallback.py — a refused joke request gets a joke.

Live v2.0.131 (2026-09-29): "tell me a short joke" -> "I'm afraid I've run
out of material, sir; my humor processors seem to have hit a bit of a wall."
The fallback fires only for an explicit joke request whose reply refuses AND
carries no joke; everything else passes through untouched. Pure module,
stdlib only.

    python -m unittest tests.test_joke_fallback
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

from core import joke_fallback as jf

LIVE_REFUSAL = ("[intent:dry_wit] I'm afraid I've run out of material, sir; "
                "my humor processors seem to have hit a bit of a wall.")


class JokeRequestTests(unittest.TestCase):
    def test_explicit_requests(self):
        for text in ("tell me a short joke", "Tell me a joke.",
                     "Jarvis, tell me a joke", "hey jarvis tell me another joke",
                     "can you tell me a funny joke please",
                     "could you give me a quick joke", "got any good jokes?",
                     "do you know any jokes", "make me laugh", "joke please",
                     "tell us a dad joke"):
            with self.subTest(text=text):
                self.assertTrue(jf.is_joke_request(text))

    def test_not_requests(self):
        for text in ("are you still making jokes at me", "that was a bad joke",
                     "stop joking around", "tell me something interesting",
                     "tell me another one", "no more jokes", "", None,
                     "what's the weather", "is this a joke to you"):
            with self.subTest(text=text):
                self.assertFalse(jf.is_joke_request(text))


class RefusalAndContentTests(unittest.TestCase):
    def test_refusals(self):
        for reply in (LIVE_REFUSAL,
                      "I'm afraid I'm fresh out of jokes, sir.",
                      "I can't think of a single one right now, sir.",
                      "I'm not in the mood, sir.",
                      "I think I'll pass this time, sir.",
                      "My comedy subroutines are offline, sir.",
                      "I'm afraid I don't have any good jokes, sir."):
            with self.subTest(reply=reply):
                self.assertTrue(jf.looks_like_refusal(reply))

    def test_jokes_are_not_refusals_or_have_content(self):
        for reply in ("Why did the robot cross the road, sir? Because it was "
                      "programmed by the chicken.",
                      "A neutron walks into a bar and asks how much for a "
                      "drink. 'For you, no charge.'",
                      "Knock knock, sir."):
            with self.subTest(reply=reply):
                self.assertTrue(jf.has_joke_content(reply)
                                or not jf.looks_like_refusal(reply))
                self.assertIsNone(jf.apply(reply, "tell me a joke"))


class ApplyTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(jf, "_last", [None])
        p.start()
        self.addCleanup(p.stop)

    def test_live_refusal_gets_a_bundled_joke(self):
        out = jf.apply(LIVE_REFUSAL, "tell me a short joke",
                       rng=random.Random(1))
        self.assertIn(out, jf.FALLBACK_JOKES)

    def test_refusal_with_a_joke_inside_is_kept(self):
        reply = ("I'm afraid I've run out of fresh material, sir, but here's "
                 "an old one: why did the scarecrow win an award? He was "
                 "outstanding in his field.")
        self.assertIsNone(jf.apply(reply, "tell me a joke"))

    def test_review_refusal_shapes_get_a_joke(self):
        # Review 2026-09-29: a question BACK to the owner is not a joke
        # setup, and a bare "because" is not a punchline.
        for reply in ("A joke, sir? I'm afraid I've run out of material.",
                      "I'm afraid I've run out of material, sir, because "
                      "you've heard them all.",
                      "I'd rather not, sir.",
                      "My humour circuits are offline tonight, sir."):
            with self.subTest(reply=reply):
                self.assertIn(jf.apply(reply, "tell me a joke",
                                       rng=random.Random(2)),
                              jf.FALLBACK_JOKES)

    def test_review_real_jokes_with_persona_colour_are_kept(self):
        # A pun that opens like a refusal, and a joke behind a self-mocking
        # "humour circuits" preamble, must be spoken as the model wrote them.
        for reply in ("I don't have any jokes about paper, sir. They're "
                      "tearable.",
                      "I'm afraid my humour circuits are a bit rusty, sir, "
                      "but here goes: I told my router we needed to talk. It "
                      "went quiet."):
            with self.subTest(reply=reply):
                self.assertIsNone(jf.apply(reply, "tell me a joke"))

    def test_refusal_on_a_non_joke_turn_is_kept(self):
        self.assertIsNone(jf.apply(LIVE_REFUSAL, "what's the printer doing"))
        self.assertIsNone(jf.apply(LIVE_REFUSAL, ""))

    def test_never_the_same_joke_twice_in_a_row(self):
        rng = random.Random(3)
        prev = None
        for _ in range(30):
            out = jf.apply(LIVE_REFUSAL, "tell me a joke", rng=rng)
            self.assertNotEqual(out, prev)
            prev = out

    def test_list_is_short_clean_and_addresses_sir(self):
        self.assertLessEqual(len(jf.FALLBACK_JOKES), 12)
        self.assertEqual(len(set(jf.FALLBACK_JOKES)), len(jf.FALLBACK_JOKES))
        for joke in jf.FALLBACK_JOKES:
            with self.subTest(joke=joke):
                self.assertIn("sir", joke)
                self.assertLess(len(joke), 120)
                self.assertFalse(jf.looks_like_refusal(joke))

    def test_never_raises(self):
        self.assertIsNone(jf.apply(None, None))
        self.assertIsNone(jf.apply(123, object()))


if __name__ == "__main__":
    unittest.main()
