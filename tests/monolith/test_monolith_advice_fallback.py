"""The monolith wiring of core/advice_fallback.py (v2.0.135).

Live v2.0.129-v2.0.134 (2026-09-29): "what should I have for dinner tonight"
-> "[intent:dry_wit] A bold choice, if I may say so, sir — though I'm afraid my
culinary expertise is somewhat limited to data processing." Dispatch now swaps
that dodge for a bundled suggestion and logs [advice-fallback]; the sentence
flush does not voice the dodge early.

    python -m unittest tests.monolith.test_monolith_advice_fallback
"""
from __future__ import annotations

import contextlib
import io
import random
import types
import unittest
from unittest import mock

from core import advice_fallback
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

DODGE = ("[intent:dry_wit] A bold choice, if I may say so, sir — though "
         "I'm afraid my culinary expertise is somewhat limited to data "
         "processing.")
REAL = "[intent:confirmation] How about a stir-fry tonight, sir? Ten minutes."
ALL_SUGGESTIONS = tuple(s for lines in advice_fallback.MEAL_SUGGESTIONS.values()
                        for s in lines)


@requires_monolith
class AdviceFallbackDispatchTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(advice_fallback, "_last", [None])
        self._p(advice_fallback, "random", random.Random(5))

    def test_the_dodge_is_replaced_by_a_suggestion(self):
        self._dispatch("what should I have for dinner tonight", DODGE)
        self.assertEqual(len(self.spoken), 1)
        self.assertIn(self.spoken[0], advice_fallback.MEAL_SUGGESTIONS["dinner"])
        self.assertNotIn("bold choice", self.spoken[0].lower())

    def test_the_fallback_is_logged(self):
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=DODGE)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch("what should I have for dinner")
        self.assertIn("[advice-fallback]", buf.getvalue())

    def test_history_keeps_the_suggestion_not_the_dodge(self):
        history = [{"role": "user", "content": "what should I have for dinner"},
                   {"role": "assistant", "content": DODGE}]
        self._p(self.bc, "conversation_history", history)
        self._dispatch("what should I have for dinner", DODGE)
        self.assertIn(history[-1]["content"], ALL_SUGGESTIONS)

    def test_a_real_suggestion_is_spoken_unchanged(self):
        self._dispatch("what should I have for dinner", REAL)
        self.assertEqual(self.spoken, [REAL])

    def test_bold_choice_about_a_real_choice_is_left_alone(self):
        reply = "[intent:dry_wit] A bold choice, if I may say so, sir."
        self._dispatch("I'm painting the garage orange", reply)
        self.assertEqual(self.spoken, [reply])

    def test_a_joke_refusal_still_goes_to_the_joke_fallback(self):
        from core import joke_fallback
        self._p(joke_fallback, "_last", [None])
        self._p(joke_fallback, "random", random.Random(7))
        self._dispatch("tell me a joke", "[intent:dry_wit] I'm afraid I've run "
                       "out of material, sir.")
        self.assertIn(self.spoken[0], joke_fallback.FALLBACK_JOKES)


class _InlineThread:
    def __init__(self, target=None, args=(), kwargs=None, **_kw):
        self._t, self._a, self._k = target, args, kwargs or {}

    def start(self):
        if self._t:
            self._t(*self._a, **self._k)

    def join(self, *_a, **_k):
        return None

    def is_alive(self):
        return False


@requires_monolith
class AdviceEarlyFlushHoldTests(_Base):
    def _feed(self, user_text, chunks):
        bc = self.bc
        spoken: list[str] = []
        prev = bc._begin_turn_grounding(user_text)
        try:
            with mock.patch.object(bc, "threading",
                                   types.SimpleNamespace(Thread=_InlineThread)):
                buf = bc._SentenceFlushBuffer(speak_fn=spoken.append)
                for c in chunks:
                    buf.feed(c)
        finally:
            bc._end_turn_grounding(prev)
        return spoken

    def test_a_dodging_first_sentence_is_not_voiced_early(self):
        spoken = self._feed("what should I have for dinner",
                            ["A bold choice, sir. ", "Entirely up to you. ",
                             "tail"])
        self.assertEqual(spoken, [])

    def test_a_real_suggestion_still_streams(self):
        spoken = self._feed("what should I have for dinner",
                            ["How about tacos tonight, sir? ",
                             "Quick and cheap. ", "tail"])
        self.assertEqual(spoken[0], "How about tacos tonight, sir?")

    def test_other_turns_still_stream(self):
        spoken = self._feed("I'm painting the garage orange",
                            ["A bold choice, sir. ", "I approve. ", "tail"])
        self.assertEqual(spoken[0], "A bold choice, sir.")


if __name__ == "__main__":
    unittest.main()
