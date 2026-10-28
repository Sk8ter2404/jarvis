"""Monolith wiring for two misheard-speech gates (v3.0.1, 2026-10-27).

THE LIVE INCIDENTS (v3.0.0 session log, 2026-10-27):

  20:39:40  The TV had just come on. Parakeet found no words in a capture;
            the 'empty' rescue handed it to Whisper, which made it "Thank
            you." JARVIS had spoken 14 s earlier, so the R10 reply rule kept
            it and JARVIS answered ("As you wish, sir."); the owner: "I
            didn't say that." Fix: _transcribe_capture records the rescue
            reason (_last_stt_rescue) and _noise_verdict hands
            primary_heard_nothing to core.speech_filter.hallucination_verdict.

  20:41:26  A capture at 1.25x the VAD threshold came back as "Enough and
            awesome." and the model web-searched it verbatim. Fix:
            _unclear_turn (core.speech_filter.unclear_reason) asks the owner
            to repeat instead of acting.

Synthetic context only; the transcripts are the two generic live lines.

    python -m unittest tests.monolith.test_monolith_misheard_gates
"""
from __future__ import annotations

import inspect
import time
import unittest
from unittest import mock

from tests.monolith.test_monolith_noise_gate import _Base
from tests.test_speech_filter import UnclearReasonTests


class PhantomThankYouTests(_Base):
    def _rescue(self, why):
        self._p(self.bc, "_last_stt_rescue", [why])

    def test_the_live_phantom_is_not_answered(self):
        self._owner_spoke()
        line_at = self._jarvis_says("Understood, sir. I'll keep it brief.")
        self._rescue("empty")
        answered, log = self._turn("Thank you.", at=line_at + 14.0,
                                   peak=0.1523)
        self.assertFalse(answered, "a Whisper-only 'Thank you.' was answered")
        self.assertIn("[noise] ignored (the primary decoder heard no words)",
                      log)
        self.assertNotIn("thank", log.lower())

    def test_a_reply_parakeet_heard_is_still_answered(self):
        self._owner_spoke()
        line_at = self._jarvis_says("It is seven forty-two, sir.")
        self._rescue("")
        answered, log = self._turn("Thank you.", at=line_at + 2.5,
                                   peak=0.0096)
        self.assertTrue(answered)
        self.assertNotIn("[noise]", log)

    def test_a_no_wake_rescue_is_not_the_signal(self):
        self._owner_spoke()
        line_at = self._jarvis_says("It is seven forty-two, sir.")
        self._rescue("no-wake")
        answered, _log = self._turn("Thank you.", at=line_at + 2.5,
                                    peak=0.0096)
        self.assertTrue(answered)

    def test_an_answer_to_a_question_jarvis_just_asked_is_kept(self):
        # Review of 3.0.1: Parakeet missing a quiet "yeah" right after a
        # question must not turn the owner's answer into noise.
        self._owner_spoke()
        line_at = self._jarvis_says("Shall I put it on the TV, sir?")
        self._rescue("empty")
        answered, log = self._turn("Yeah.", at=line_at + 2.0, peak=0.0300)
        self.assertTrue(answered)
        self.assertNotIn("primary decoder", log)

    def test_the_primary_route_records_the_rescue_reason(self):
        bc = self.bc
        fake = mock.Mock()
        fake.run.return_value = ("Thank you.", {"no_speech_prob": 0.1,
                                                "avg_logprob": -0.4})
        fake.last_rescue = "empty"
        self._p(bc, "_parakeet_primary", fake)
        self._p(bc, "_last_stt_rescue", ["stale"])
        self._p(bc, "_stt_r6_route", lambda: "primary")
        self._p(bc, "_tail_probe_start", lambda a: None)
        self.assertEqual(bc._transcribe_capture(object())[0], "Thank you.")
        self.assertEqual(bc._last_stt_rescue[0], "empty")
        # A Whisper-routed capture clears it: never a stale reason.
        self._p(bc, "_stt_r6_route", lambda: None)
        self._p(bc, "transcribe", lambda a: ("hi", {}))
        self._p(bc, "_SPECULATIVE_STT", False, create=True)
        bc._transcribe_capture(object())
        self.assertEqual(bc._last_stt_rescue[0], "")


class UnclearTurnTests(_Base):
    def setUp(self):
        super().setUp()
        self.spoken = self._p(self.bc, "_speak")
        self._p(self.bc, "_unclear_cue_at", [0.0])
        self._p(self.bc, "UNCLEAR_UTTERANCE_GATE", True)

    def _u(self, text, peak=0.0100, injected=False, at=None):
        self._p(self.bc, "_last_recording_peak", peak)
        at = time.monotonic() if at is None else at
        return self._quiet(self.bc._unclear_turn, text, injected, now=at)

    def test_the_live_line_is_asked_about_not_searched(self):
        self._owner_spoke()
        out, log = self._u("Enough and awesome.")
        self.assertTrue(out)
        self.spoken.assert_called_once_with(self.bc.UNCLEAR_CUE)
        self.assertIn("[unclear] not acting (3 words", log)
        self.assertIn("asking sir to repeat", log)
        self.assertNotIn("awesome", log.lower(), "the transcript was logged")

    def test_the_cue_is_rate_limited(self):
        self._owner_spoke()
        now = time.monotonic()
        self._u("Enough and awesome.", at=now)
        out, log = self._u("Enough and awesome.", at=now + 5.0)
        self.assertTrue(out)
        self.assertEqual(self.spoken.call_count, 1)
        self.assertIn("dropped", log)

    def test_with_no_conversation_it_is_dropped_silently(self):
        self.bc._last_owner_turn_at[0] = 0.0
        out, _log = self._u("Enough and awesome.")
        self.assertTrue(out)
        self.spoken.assert_not_called()

    def test_real_short_commands_still_act(self):
        self._owner_spoke()
        acted = [t for t in UnclearReasonTests.REAL if self._u(t)[0]]
        self.assertEqual(acted, [])
        self.spoken.assert_not_called()

    def test_an_answer_to_a_question_is_never_judged(self):
        self._owner_spoke()
        line_at = time.monotonic()      # _speak is mocked here: set the
        self._p(self.bc, "_last_jarvis_line",   # cell it would leave
                [line_at, True])
        out, _log = self._u("Enough and awesome.", at=line_at + 3.0)
        self.assertFalse(out)
        self.spoken.assert_not_called()
        # Once the reply window has passed, the gate applies again.
        out, _log = self._u("Enough and awesome.", at=line_at + 60.0)
        self.assertTrue(out)

    def test_loud_typed_and_switched_off_are_untouched(self):
        self._owner_spoke()
        self.assertFalse(self._u("Enough and awesome.", peak=0.0405)[0])
        self.assertFalse(self._u("Enough and awesome.", injected=True)[0])
        self._p(self.bc, "UNCLEAR_UTTERANCE_GATE", False)
        self.assertFalse(self._u("Enough and awesome.")[0])
        self.spoken.assert_not_called()

    def test_main_runs_it_after_the_media_gate_before_the_turn(self):
        src = inspect.getsource(self.bc.main)
        self.assertEqual(src.count("_unclear_turn(text, _injected_text is not None)"), 1)
        gate = src.index("_unclear_turn(text, _injected_text is not None)")
        self.assertLess(src.index("_media_voice_gate(text, _injected_text"),
                        gate)
        self.assertLess(gate, src.index('print(f"  You:    {text}")'))


if __name__ == "__main__":
    unittest.main()
