"""Monolith wiring for open-ended requests (2026-09-29 quality sweep, v2.0.131).

Three live turns got a non-answer or a misroute:

  (A) "tell me a short joke" -> "I'm afraid I've run out of material, sir; my
      humor processors seem to have hit a bit of a wall." A refusal. The
      dispatch now swaps a refusal of an explicit joke request for a bundled
      one-liner (core/joke_fallback.py) and logs [joke-fallback].
  (B) "tell me something interesting" -> "I've taken the liberty of searching
      for something truly fascinating, but ..." with no [ACTION:]. The
      reactive claim check now flags the "taken the liberty of <gerund>" shape,
      so the correction round runs.
  (C) "convert 100 degrees fahrenheit to celsius" -> correct answer AND
      [ACTION: weather_briefing]: the preemptive injector read "37.8 degrees
      Celsius" as the weather stated from memory. A conversion turn no longer
      injects weather_briefing; the weather from memory still does.

Every LLM call, action and speech output is stubbed. Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_open_ended
"""
from __future__ import annotations

import random
import unittest

from core import joke_fallback
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

JOKE_REFUSAL = ("[intent:dry_wit] I'm afraid I've run out of material, sir; "
                "my humor processors seem to have hit a bit of a wall.")
REAL_JOKE = ("[intent:dry_wit] Why did the robot cross the road, sir? "
             "Because it was programmed by the chicken.")
LIBERTY_SEARCH = ("[intent:dry_wit] I've taken the liberty of searching for "
                  "something truly fascinating, but I'm afraid your recent "
                  "decision to skip dinner is proving far more interesting, "
                  "sir.")
CONVERSION = ("[intent:confirmation] 100 degrees Fahrenheit is about 37.8 "
              "degrees Celsius, sir.")
CONVERSION_SHORT = "[intent:confirmation] That comes to 37.8 degrees Celsius, sir."


# ════════════════════════════════════════════════════════════════════════════
#  (A) a refused joke request gets a joke
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class JokeFallbackDispatchTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(joke_fallback, "_last", [None])
        self._p(joke_fallback, "random", random.Random(7))

    def test_refusal_is_replaced_by_a_bundled_joke(self):
        self._dispatch("tell me a short joke", JOKE_REFUSAL)
        self.assertEqual(len(self.spoken), 1)
        self.assertIn(self.spoken[0], joke_fallback.FALLBACK_JOKES)
        self.assertNotIn("run out of material", self.spoken[0])

    def test_fallback_is_logged(self):
        import contextlib
        import io
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=JOKE_REFUSAL)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch("tell me a joke")
        self.assertIn("[joke-fallback]", buf.getvalue())

    def test_history_keeps_the_joke_not_the_refusal(self):
        bc = self.bc
        history = [{"role": "user", "content": "tell me a joke"},
                   {"role": "assistant", "content": JOKE_REFUSAL}]
        self._p(bc, "conversation_history", history)
        self._dispatch("tell me a joke", JOKE_REFUSAL)
        self.assertIn(history[-1]["content"], joke_fallback.FALLBACK_JOKES)

    def test_a_real_joke_is_spoken_unchanged(self):
        self._dispatch("tell me a short joke", REAL_JOKE)
        self.assertEqual(self.spoken, [REAL_JOKE])

    def test_a_refusal_of_something_else_is_left_alone(self):
        reply = ("[intent:bad_news] I'm afraid I've hit a bit of a wall with "
                 "the printer, sir.")
        self._dispatch("what's the printer doing", reply)
        self.assertEqual(self.spoken, [reply])


# ════════════════════════════════════════════════════════════════════════════
#  (B) "I've taken the liberty of searching" with no action is a claim
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class LibertySearchClaimTests(_Base):
    def test_liberty_search_gets_the_correction_round(self):
        self._dispatch("tell me something interesting", LIBERTY_SEARCH,
                       ["Octopuses have three hearts, sir."])
        self.gfr.assert_called_once()
        self.assertEqual(self._followup_names(), ["_unverified_claim"])

    def test_a_plain_fact_still_needs_no_second_round(self):
        fact = "[intent:dry_wit] Octopuses have three hearts, sir."
        self._dispatch("tell me something interesting", fact)
        self.gfr.assert_not_called()
        self.assertEqual(self.spoken, [fact])


# ════════════════════════════════════════════════════════════════════════════
#  (C) a temperature conversion never runs weather_briefing
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class ConversionIsNotWeatherTests(_Base):
    def setUp(self):
        super().setUp()
        self._stub("weather_briefing", "Currently 64 degrees and clear, sir.")

    def test_conversion_turn_does_not_run_weather_briefing(self):
        for reply in (CONVERSION, CONVERSION_SHORT):
            with self.subTest(reply=reply):
                self.calls["weather_briefing"].clear()
                self.spoken.clear()
                self._dispatch("convert 100 degrees fahrenheit to celsius",
                               reply)
                self.assertEqual(self.calls["weather_briefing"], [])
                self.assertEqual(self.spoken, [reply])

    def test_both_scale_answer_is_not_weather_even_without_the_turn(self):
        self.assertIsNone(self.bc._detect_preemptive_hallucination(CONVERSION))

    def test_detector_takes_the_owner_turn_into_account(self):
        bc = self.bc
        self.assertIsNone(bc._detect_preemptive_hallucination(
            CONVERSION_SHORT,
            user_text="what's 100 fahrenheit in celsius"))
        self.assertEqual(
            bc._detect_preemptive_hallucination(CONVERSION_SHORT)[:2],
            ("inject", "weather_briefing"))

    def test_weather_from_memory_is_still_injected(self):
        for reply in ("It's 64 degrees Fahrenheit outside, sir.",
                      "It's 72 and sunny, sir.",
                      "Currently 58 degrees and cloudy, sir."):
            with self.subTest(reply=reply):
                self.calls["weather_briefing"].clear()
                self._dispatch("how's it looking outside", reply)
                self.assertEqual(len(self.calls["weather_briefing"]), 1)

    def test_weather_question_answered_in_both_scales_is_still_injected(self):
        # Review 2026-09-29: a both-scales reply is a conversion only when the
        # owner did not ask about the weather or a temperature reading.
        for user, reply in (
                ("what's the temperature", "It's 68°F / 20°C, sir."),
                ("how cold is it",
                 "68 degrees Fahrenheit, or 20 degrees Celsius, sir."),
                ("what's the weather in 90210 in celsius",
                 "It's 20 degrees Celsius, sir.")):
            with self.subTest(user=user):
                self.calls["weather_briefing"].clear()
                self._dispatch(user, reply)
                self.assertEqual(len(self.calls["weather_briefing"]), 1)

    def test_other_categories_still_fire_on_a_conversion_turn(self):
        self._stub("get_time", "current time is 01:47 AM")
        self._dispatch("convert 100 degrees fahrenheit to celsius",
                       "It's 1:47 AM, sir, and 100 F is 37.8 C.")
        self.assertEqual(len(self.calls["get_time"]), 1)
        self.assertEqual(self.calls["weather_briefing"], [])


if __name__ == "__main__":
    unittest.main()
