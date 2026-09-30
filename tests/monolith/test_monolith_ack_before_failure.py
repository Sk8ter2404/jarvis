"""No "Right away, sir." before a refusal (2026-09-30).

THE LIVE TURN (2026-09-29 21:58): "send the robot exploring for a minute" ->
the model replied "[ACTION: <robot>_explore, 60] Right away, sir.", the action
REFUSED, and the owner heard "Right away, sir." and then the failure
follow-up's "I'm afraid the ... firmware isn't capable ...".

Drives the REAL _run_llm_dispatch_body -> parse_and_run_actions -> failure
follow-up loop with stub actions and a recording _speak (the answer-first
harness), plus the REAL _SentenceFlushBuffer for the streamed half:

  * a pure acknowledgement is not voiced when an action of the same reply
    failed or refused - before OR after the [ACTION:] marker; it stays in
    conversation_history; content after it is still spoken;
  * a successful action keeps it; answer-first is unchanged;
  * follow-up rounds apply the same rule, but only while the next round will
    still report the failure;
  * the streaming flush never voices a lone acknowledgement: it waits for the
    next sentence, so "Right away, sir. [ACTION: x]" is decided after x ran;
  * an acknowledgement that DID stream early is logged honestly.

Names are generic. No real audio, no LLM.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_ack_before_failure
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_answer_first import _Base

_ACK = "Right away, sir."
_REFUSED = ("I couldn't start exploring: the rover's firmware can't do that "
            "yet.")
_FOLLOWUP = "I'm afraid the rover's firmware isn't capable of exploring, sir."


class _InlineThread:
    """threading.Thread stand-in that runs the target on start()."""

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self._target, self._args = target, args

    def start(self):
        if self._target is not None:
            self._target(*self._args)

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


@requires_monolith
class _AckBase(_Base):
    def setUp(self):
        super().setUp()
        self.explore_result = [_REFUSED]
        self.bc.ACTIONS["rover_explore"] = (
            lambda a="": self.explore_result[0])
        self.followup.side_effect = [_FOLLOWUP, ""]

    def _buffer(self, spoken):
        return self.bc._SentenceFlushBuffer(speak_fn=spoken.append)

    def _stream(self, chunks):
        """Feed ``chunks`` through a real flush buffer the way _call_llm
        does; returns (early_spoken, buffer)."""
        early: list = []
        buf = self._buffer(early)
        with mock.patch.object(self.bc, "threading",
                               mock.Mock(Thread=_InlineThread)):
            for c in chunks:
                buf.feed(c)
        return early, buf


class LiveTurnTests(_AckBase):
    def test_live_turn_ack_after_the_marker_is_not_spoken(self):
        reply = f"[ACTION: rover_explore, 60] {_ACK}"
        printed = self._run(reply, text="send the rover exploring for a minute")
        self.assertEqual(self.spoken, [_FOLLOWUP])
        self.assertIn(_ACK, self._history_text(),
                      "only the audio goes; the reply is remembered")
        self.assertIn("[ack-hold] dropped the acknowledgement (3 words)",
                      printed)

    def test_ack_before_the_marker_is_not_spoken(self):
        self._run(f"{_ACK} [ACTION: rover_explore, 60]")
        self.assertEqual(self.spoken, [_FOLLOWUP])

    def test_confident_variants_are_not_spoken(self):
        for ack in ("On it, sir.", "Consider it done, sir.",
                    "Certainly, sir.", "Of course, sir. Right away.",
                    "As you wish, sir.", "Will do, sir.",
                    "Straight away, sir."):
            with self.subTest(ack=ack):
                self.spoken.clear()
                self.followup.side_effect = [_FOLLOWUP, ""]
                self._run(f"[ACTION: rover_explore, 60] {ack}")
                self.assertEqual(self.spoken, [_FOLLOWUP])

    def test_success_keeps_the_ack(self):
        self.explore_result[0] = "Exploring for sixty seconds."
        printed = self._run(f"[ACTION: rover_explore, 60] {_ACK}")
        self.assertEqual(self.spoken, [_ACK])
        self.assertNotIn("[ack-hold]", printed)

    def test_content_after_the_ack_is_still_spoken(self):
        tail = "The rover has a full battery."
        self._run(f"[ACTION: rover_explore, 60] {_ACK} {tail}")
        self.assertEqual(self.spoken, [tail, _FOLLOWUP])

    def test_leading_tags_stay_on_the_remaining_content(self):
        tail = "The rover has a full battery."
        self._run(f"[intent:acknowledge] {_ACK} {tail} "
                  f"[ACTION: rover_explore, 60]")
        self.assertIn(tail, self.spoken[0])
        self.assertNotIn("Right away", self.spoken[0])

    def test_a_non_ack_lead_is_spoken(self):
        lead = "Sending the rover out to explore, sir."
        self._run(f"{lead} [ACTION: rover_explore, 60]")
        self.assertEqual(self.spoken, [lead, _FOLLOWUP])

    def test_verbatim_refusal_counts_too(self):
        # An honest refusal voiced verbatim is deliberately NOT a
        # FAILURE_MARKERS failure ("won't"), but the acknowledgement before
        # it is just as false.
        bc = self.bc
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS) | {"rover_explore"})
        refusal = "I won't send the rover out while it is charging, sir."
        self.explore_result[0] = refusal
        self._run(f"{_ACK} [ACTION: rover_explore, 60]")
        self.assertEqual(self.spoken, [refusal])

    def test_answer_first_is_unchanged(self):
        # A successful verbatim answer: answer-first drops the lead exactly
        # as before, and this rule never reports anything.
        printed = self._run("One moment, sir. [ACTION: weather_briefing]")
        self.assertEqual(len(self.spoken), 1)
        self.assertIn("[answer-first]", printed)
        self.assertNotIn("[ack-hold]", printed)


class FollowupRoundTests(_AckBase):
    def test_followup_ack_before_a_new_failure_is_not_spoken(self):
        self.followup.side_effect = [
            f"{_ACK} [ACTION: rover_explore, 60]", _FOLLOWUP, ""]
        self._run("One moment, sir. [ACTION: see_screen]",
                  text="look at the screen and send the rover out")
        self.assertNotIn(_ACK, self.spoken)
        self.assertEqual(self.spoken[-1], _FOLLOWUP)

    def test_followup_ack_is_kept_when_nothing_would_report_it(self):
        # The same (action, result) already failed this chain, so the loop
        # stops after this round: dropping the acknowledgement would leave
        # the round silent. It is spoken as before.
        self.followup.side_effect = [
            f"{_ACK} [ACTION: rover_explore, 60]", _FOLLOWUP, ""]
        self._run(f"[ACTION: rover_explore, 60] {_ACK}")
        self.assertEqual(self.spoken, [_ACK])


class StreamedAckTests(_AckBase):
    """The flush buffer holds a lone acknowledgement back."""

    def test_ack_then_marker_is_not_voiced_early(self):
        early, buf = self._stream([_ACK + " ", "[ACTION: rover_explore, 60]"])
        self.assertEqual(early, [])
        self.assertEqual(buf.spoken_prefix, "")
        self.assertTrue(buf._stopped)

    def test_ack_is_released_with_the_next_sentence(self):
        early, buf = self._stream([_ACK + " ", "Sending the rover out now. ",
                                   "[ACTION: rover_explore, 60]"])
        self.assertEqual(early, [_ACK, "Sending the rover out now."])
        self.assertEqual(buf.spoken_prefix,
                         _ACK + " Sending the rover out now. ")

    def test_consecutive_acks_are_released_as_one_line(self):
        early, _buf = self._stream(["Very good, sir. ", "Right away. ",
                                    "Sending it now. "])
        self.assertEqual(early, ["Very good, sir. Right away.",
                                 "Sending it now."])

    def test_ack_held_when_the_next_sentence_is_gated(self):
        early, buf = self._stream([_ACK + " ", "It's 1:47 AM right now, sir. "])
        self.assertEqual(early, [])
        self.assertTrue(buf._stopped)

    def test_streamed_turn_never_voices_the_ack(self):
        # End to end: stream -> flush buffer -> dispatch, as _call_llm wires
        # it. The acknowledgement streams FIRST, then the marker; the action
        # fails; the owner hears only the failure follow-up.
        bc = self.bc
        reply = f"{_ACK} [ACTION: rover_explore, 60]"

        def during_llm():
            early, buf = self._stream([_ACK + " ", "[ACTION: rover_explore",
                                       ", 60]"])
            self.spoken.extend(early)
            bc._stream_spoken_prefix[0] = buf.spoken_prefix

        self._run(reply, during_llm=during_llm)
        self.assertEqual(self.spoken, [_FOLLOWUP])

    def test_an_ack_that_did_stream_is_logged(self):
        bc = self.bc
        head = f"{_ACK} Sending the rover out now. "
        bc._stream_spoken_prefix[0] = head
        printed = self._run(f"{head}[ACTION: rover_explore, 60]")
        self.assertIn("[ack-hold] 'Right away, sir.' was already voiced", printed)
        self.assertEqual(self.spoken, [_FOLLOWUP])


@requires_monolith
class HelperTests(_AckBase):
    def test_is_confident_ack(self):
        f = self.bc._is_confident_ack
        for s in ("Right away, sir.", "On it.", "Consider it done, sir.",
                  "Certainly.", "As you wish.", "Will do, sir.",
                  "Yes, sir. Right away.", "Checking the rover for you now."):
            self.assertTrue(f(s), s)
        for s in ("The rover is out of battery.", "Right away, sir, 60 s.",
                  "Right away?", "", "I can't do that, sir.",
                  "I won't do that.", "I'm afraid, sir."):
            self.assertFalse(f(s), s)

    def test_drop_needs_a_failed_or_refused_real_action(self):
        d = self.bc._drop_ack_before_failure
        ok = [("rover_explore", "Exploring.", False)]
        bad = [("rover_explore", _REFUSED, False)]
        synthetic = [("_unverified_claim", "didn't happen", False)]
        deferred = [("rover_explore", "⚠  PUSHBACK: can't, sir", False)]
        self.assertEqual(d(_ACK, ok), (_ACK, 0))
        self.assertEqual(d(_ACK, bad), ("", 3))
        self.assertEqual(d(_ACK, synthetic), (_ACK, 0))
        self.assertEqual(d(_ACK, deferred), (_ACK, 0))
        self.assertEqual(d(_ACK, []), (_ACK, 0))
        self.assertEqual(d("[wry] On it. Battery is low.", bad),
                         ("[wry] Battery is low.", 2))

    def test_the_followup_failure_test_is_the_shared_one(self):
        # The loop's _is_failure and the acknowledgement drop must ask ONE
        # question (the stale-duplicate rule).
        import inspect
        src = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertIn("return _action_result_failed(result)", src)
        self.assertTrue(self.bc._action_result_failed("It FAILED."))
        self.assertFalse(self.bc._action_result_failed(None))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
