"""v3.0.26 (overhaul lane A, honesty) - the monolith wiring.

Drives the REAL _run_llm_dispatch (turn ledger open, so a claim is withheld
as it is live), parse_and_run_actions, the follow-up loop, the close-out and
the turn checker, with every LLM call, action and speech output stubbed.

  * Live 2026-11-16 02:02:35-44: "turn off lakework mode" (wake-word mode).
    The first reply "I've turned off wake-word listening." was withheld, the
    follow-up round's "I've disabled wake-word listening." was SPOKEN, and
    nothing ran; the turn checker logged said_no_action and could do
    nothing. Now the follow-up's claim is withheld too (the toggle family,
    tests/test_honesty_claims_1117.py), the turn ends on the honest close-
    out, and - when sir's own words name the wake word - the narrated past
    tense runs wake_word_mode_off instead (_PREEMPT_OWNER_GATES).
  * TURN_CHECK_LOCAL_CORRECTION: a claim that REACHED sir (here: voiced early
    by the stream) with no cloud retry gets "Correction, sir - ..."; a
    withheld one does not.
  * Live 2026-11-17 00:23:45: "Want the full list, sir?" -> "Not right now.
    That'll take forever." ends at "Very good, sir." with no follow-up round
    and no turn-checker retry.
  * The two read-backs whose own tokens never run (the turn checker's retry
    read-back, a confirmed action's read-back) never speak a claim of an
    action nothing ran.

Generic fixtures only. No audio, no LLM, no network.

    python -m unittest tests.monolith.test_monolith_honesty_1117
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

ASK_HEARD_WRONG = "Jarvis turn off lakework mode."   # before the STT fix
ASK = "Jarvis turn off wake word mode."               # after it
FIRST = ("[intent:confirmation] As you wish, sir. I've turned off wake-word "
         "listening.")
FOLLOWUP = ("[intent:confirmation] Understood, sir. I've disabled wake-word "
            "listening.")
QUESTION = ("Quite a bit, sir: I can set timers and play music. Want the "
            "full list, sir?")
DECLINE = "Not right now. That'll take forever."
ON_IT = "[intent:confirmation] On it, sir."


class _HonestyBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "off")
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_autocorrect_choice", [])
        self._stub("wake_word_mode_off", "Wake-word mode off, sir.")
        self._stub("wake_word_mode_on", "Wake-word mode on, sir.")
        self.log = ""

    def _fake_llm(self, reply, early=""):
        """A canned brain reply honouring _call_llm's history contract (the
        turn checker's retry looks the turn up there); ``early`` = the part
        the stream voiced before the reply was complete."""
        bc = self.bc

        def llm(user_text):
            if early:
                bc._stream_spoken_prefix[0] = early
            bc.conversation_history.append({"role": "user",
                                            "content": user_text})
            bc.conversation_history.append({"role": "assistant",
                                            "content": reply})
            return reply
        return llm

    def _dispatch(self, user_text, first_reply, followups=()):
        bc = self.bc
        self._p(bc, "get_response_with_animation",
                side_effect=self._fake_llm(first_reply))
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = bc._run_llm_dispatch(user_text)
            self.assertTrue(bc._turn_check_flush(10.0), "worker stuck")
        self.log = buf.getvalue()
        return out

    def _followup_inputs(self):
        return [[n for n, _r in c.args[0]] for c in self.gfr.call_args_list]


@requires_monolith
class WakeWordLiveTurnTests(_HonestyBase):
    def test_the_follow_up_lie_is_never_spoken(self):
        # The live transcript, which never names the wake word: nothing may
        # run on a guess, and nothing false may be said.
        self._dispatch(ASK_HEARD_WRONG, FIRST, [FOLLOWUP])
        for s in self.spoken:
            self.assertNotIn("disabled", s)
            self.assertNotIn("turned off", s)
        self.assertEqual(self.calls["wake_word_mode_off"], [])
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])
        self.assertIn("reply claims 'i've disabled'", self.log)
        self.assertEqual(self.spoken, [self.bc._CLOSE_OUT_GENERIC])

    def test_sirs_own_ask_runs_the_setter_the_reply_narrates(self):
        self._dispatch(ASK, FIRST, ["Done, sir."])
        self.assertEqual(self.calls["wake_word_mode_off"], [""])
        self.assertIn("injecting [ACTION: wake_word_mode_off]", self.log)
        self.assertNotIn("_unverified_claim",
                         sum(self._followup_inputs(), []))
        self.assertNotIn(self.bc._CLOSE_OUT_GENERIC, self.spoken)

    def test_the_owner_gate(self):
        bc = self.bc

        def got(reply, user):
            out = bc._detect_preemptive_hallucination(reply, user_text=user)
            return out[1] if out else None

        done_off = "I've disabled wake-word listening, sir."
        self.assertEqual(got(done_off, ASK), "wake_word_mode_off")
        self.assertEqual(got("Wake-word mode is now off, sir.", ASK),
                         "wake_word_mode_off")
        for user in (ASK_HEARD_WRONG, "what's the weather",
                     "is wake word mode on?",
                     "Jarvis, turn wake word mode back on.", ""):
            with self.subTest(user=user):
                self.assertIsNone(got(done_off, user))
        done_on = "I've re-enabled wake-word mode, sir."
        self.assertEqual(got(done_on, "Jarvis, wake word mode back on."),
                         "wake_word_mode_on")
        self.assertIsNone(got(done_on, ASK))
        # The progressive rows are as they were (ungated, audit A37).
        self.assertEqual(got("Disabling wake word mode, sir.", ""),
                         "wake_word_mode_off")
        self.assertIsNone(got("I'm always listening, sir.", ASK))


@requires_monolith
class DeclineWithReasonTurnTests(_HonestyBase):
    def _asked_first(self, line=QUESTION):
        self.bc.conversation_history.extend(
            [{"role": "user", "content": "Jarvis, what all can you do?"},
             {"role": "assistant", "content": line}])

    def test_the_live_decline_ends_the_turn(self):
        self._asked_first()
        self._dispatch(DECLINE, ON_IT, ["[intent:confirmation] On it."])
        self.gfr.assert_not_called()
        self.assertEqual(self.spoken, ["Very good, sir."])
        self.assertIn("sir declined with a reason - dropped the claim",
                      self.log)
        self.assertNotIn("[close-out]", self.log)

    def test_no_retry_on_claude_for_a_decline(self):
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "on")
        self._p(bc, "_chat_cloud_allowed", return_value=True)
        oneshot = self._p(bc, "_claude_oneshot", return_value="Shh.")
        tmp = tempfile.mkdtemp(prefix="jarvis_honesty_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self._p(bc, "_turn_check_log_path",
                return_value=os.path.join(tmp, "turn_check.jsonl"))
        self._asked_first()
        self._dispatch(DECLINE, ON_IT)
        oneshot.assert_not_called()
        self.assertNotIn(bc._turn_checker.ONE_MOMENT_LINE, self.spoken)
        self.assertIn("[turn-check] skipped - sir only acknowledged or "
                      "declined", self.log)

    def test_without_a_question_before_it_is_the_old_path(self):
        self._asked_first("Your timer is set for ten minutes, sir.")
        self._dispatch(DECLINE, ON_IT)
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])

    def test_a_failure_report_after_a_check_in_is_the_old_path(self):
        # review 2026-11-19: a check-in question is not an offer, and "No,
        # it didn't work." reports a failure - the claim is held back and
        # the follow-up round may still emit the token.
        self._asked_first("Did that fix it, sir?")
        self._dispatch("No, it didn't work.", ON_IT)
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])
        self.assertNotIn("sir declined with a reason", self.log)

    def test_a_choice_after_an_either_or_question_is_the_old_path(self):
        self._asked_first("The first track or the second, sir?")
        self._dispatch("No, it's the second one.",
                       "Playing the second one, sir.")
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])
        self.assertNotIn("sir declined with a reason", self.log)

    def test_a_reason_decline_after_a_check_in_is_the_old_path(self):
        # Only an OFFER before it counts, not any question.
        self._asked_first("Is that better, sir?")
        self._dispatch(DECLINE, ON_IT)
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])

    def test_a_request_after_the_no_is_the_old_path(self):
        self._asked_first()
        self._dispatch("No. Play the next one.", ON_IT)
        self.assertEqual(self._followup_inputs()[0], ["_unverified_claim"])

    def test_an_open_offer_declined_with_a_reason(self):
        bc = self.bc
        prev = bc._begin_turn_grounding(DECLINE)
        self.addCleanup(bc._end_turn_grounding, prev)
        bc._note_turn_offer_open("Shall I read the full list, sir?")
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned, results = bc.parse_and_run_actions(ON_IT)
        self.assertEqual(results, [])
        self.assertEqual(cleaned, "Very good, sir.")

    def test_rule_7_keeps_the_strict_predicate(self):
        # action-fit rule 7 (the same action again for a bare ack) is not
        # widened: a reason sentence still makes the line "not bare".
        from core import action_fit as af
        self.assertEqual(af.check_rule(DECLINE, "pause_music", "",
                                       [("pause_music", "")])[0], 0)


@requires_monolith
class LocalCorrectionTests(_HonestyBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "on")
        self._p(bc, "TURN_CHECK_LOCAL_CORRECTION", True)
        self.cloud = self._p(bc, "_chat_cloud_allowed", return_value=False)
        self._p(bc, "_cloud_boost_on", return_value=False)
        self._p(bc, "_turn_check_cloud_always", return_value=False)
        self.oneshot = self._p(bc, "_claude_oneshot", return_value="")
        tmp = tempfile.mkdtemp(prefix="jarvis_honesty_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self._p(bc, "_turn_check_log_path",
                return_value=os.path.join(tmp, "turn_check.jsonl"))
        self.line = bc._turn_checker.LOCAL_CORRECTION_LINE

    def _streamed_claim(self, followup):
        """A progressive claim voiced EARLY by the stream (the flush lets
        "Enabling X now" through ahead of its token), then no token."""
        bc = self.bc
        early = "Enabling focus mode now, sir. "
        self._p(bc, "get_response_with_animation",
                side_effect=self._fake_llm(early.strip(), early=early))
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=[followup] + [None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch("Jarvis, focus mode on.")
            self.assertTrue(bc._turn_check_flush(10.0))
        self.log = buf.getvalue()

    def test_a_withheld_claim_gets_no_correction(self):
        self._dispatch(ASK_HEARD_WRONG, FIRST, [FOLLOWUP])
        self.assertNotIn(self.line, self.spoken)
        self.assertIn("[turn-fix] the claim never reached sir", self.log)
        self.oneshot.assert_not_called()

    def test_a_spoken_claim_is_corrected(self):
        self._streamed_claim("Focus mode keeps notifications quiet while "
                             "you work, sir.")
        self.assertEqual(self.spoken[-1], self.line)
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         self.line)
        self.assertIn("[turn-fix] the claim was spoken and nothing ran - "
                      "correcting it locally", self.log)
        self.oneshot.assert_not_called()

    def test_an_honest_line_after_it_needs_no_correction(self):
        self._streamed_claim("I'm afraid focus mode isn't available, sir.")
        self.assertNotIn(self.line, self.spoken)

    def test_the_flag_off_only_logs_the_verdict(self):
        self._p(self.bc, "TURN_CHECK_LOCAL_CORRECTION", False)
        self._streamed_claim("Focus mode keeps notifications quiet, sir.")
        self.assertNotIn(self.line, self.spoken)
        self.assertNotIn("[turn-fix]", self.log)
        self.assertIn("[turn-check] said_no_action", self.log)

    def test_shadow_only_says_it_would(self):
        self._p(self.bc, "TURN_CHECK_MODE", "shadow")
        self._streamed_claim("Focus mode keeps notifications quiet, sir.")
        self.assertNotIn(self.line, self.spoken)
        self.assertIn("[turn-fix] would correct locally", self.log)

    def test_a_cloud_retry_comes_first(self):
        self.cloud.return_value = True
        self._p(self.bc, "_publish_turn_brain")
        self._streamed_claim("Focus mode keeps notifications quiet, sir.")
        self.oneshot.assert_called_once()
        self.assertNotIn(self.line, self.spoken)
        self.assertNotIn("[turn-fix]", self.log)


@requires_monolith
class UnrunReadBackTests(_HonestyBase):
    def test_the_retry_read_back_drops_an_unrun_claim(self):
        bc = self.bc
        self._p(bc, "TURN_CHECK_MODE", "on")
        self._p(bc, "_chat_cloud_allowed", return_value=True)
        self._p(bc, "_publish_turn_brain")
        self._p(bc, "_claude_oneshot",
                return_value="[ACTION: weather_briefing] Checking, sir.")
        tmp = tempfile.mkdtemp(prefix="jarvis_honesty_")
        self.addCleanup(shutil.rmtree, tmp, True)
        self._p(bc, "_turn_check_log_path",
                return_value=os.path.join(tmp, "turn_check.jsonl"))
        self._stub("weather_briefing", "Mild and clear.")
        self._stub("lamp_off", "Desk lamp off.")
        with mock.patch.object(bc, "INFORMATIVE_ACTIONS",
                               set(bc.INFORMATIVE_ACTIONS)
                               | {"weather_briefing"}):
            self._dispatch("turn off the desk lamp", "Right away, sir.",
                           ["", "It is mild out, sir. I've also turned off "
                                "the lamp. [ACTION: lamp_off]"])
        self.assertIn("It is mild out, sir.", self.spoken)
        self.assertFalse(any("turned off the lamp" in s for s in self.spoken),
                         self.spoken)
        self.assertEqual(self.calls["lamp_off"], [])
        self.assertIn("the retry read-back: not speaking the claim", self.log)
        # The history the next turn's model reads holds the read-back as it
        # was said: no withheld claim, no unrun token (f27-a item 3).
        hist = [m.get("content", "") for m in bc.conversation_history
                if m.get("role") == "assistant"]
        self.assertFalse(any("turned off the lamp" in h for h in hist), hist)
        self.assertFalse(any("lamp_off" in h for h in hist), hist)
        self.assertIn("It is mild out, sir.", hist)

    def test_the_confirmed_read_back_drops_an_unrun_claim(self):
        bc = self.bc
        self._stub("read_thing_x", "The thing reads 42.")
        self._stub("lamp_off", "Desk lamp off.")
        self._p(bc, "_needs_confirmation", lambda n, a: n == "read_thing_x")
        self._p(bc, "get_followup_response",
                return_value=("The thing reads 42, sir. I've also turned off "
                              "the lamp. [ACTION: lamp_off]"))
        with mock.patch.object(bc, "INFORMATIVE_ACTIONS",
                               set(bc.INFORMATIVE_ACTIONS)
                               | {"read_thing_x"}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _c, results = bc.parse_and_run_actions(
                    "[ACTION: read_thing_x]")
                self.assertIn("REQUIRES CONFIRMATION", results[0][1])
                self.spoken.clear()
                self.assertTrue(bc.handle_confirmation_response("yes"))
        self.assertEqual(self.calls["read_thing_x"], [""])
        self.assertEqual(self.calls["lamp_off"], [])
        self.assertIn("The thing reads 42, sir.", self.spoken)
        self.assertFalse(any("turned off the lamp" in s for s in self.spoken),
                         self.spoken)
        self.assertIn("the confirmed read-back: not speaking the claim",
                      buf.getvalue())


if __name__ == "__main__":
    import unittest
    unittest.main()
