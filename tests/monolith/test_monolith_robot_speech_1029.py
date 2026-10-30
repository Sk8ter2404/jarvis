"""Monolith wiring for the v3.0.9 lane-A fixes (live 2026-10-29, JARVIS
3.0.7, session_2026-10-28_20-18-06.log).

THE LIVE TURNS:
  18:54:35 / 18:55:16  "Jarvis, say <line>." ran the robot's speak action:
                       the robot said it, not JARVIS;
  18:55:26  "All right." ran it again with the same line ("[action-fit]
            would block", and it ran: ACTION_FIT_MODE is 'shadow');
  18:53:47  "No, not right now." -> "Very good, sir. On it." -> the reply
            was withheld as an unverified claim and the forced follow-up
            round ran the previous turn's pause_music twice;
  18:56:56  "Jarvis, you need to make him finish talking before you speak."
            -> "Understood, sir. I'll ensure the robot completes its
            delivery before I chime in." (JARVIS cannot change that).

Pinned against the REAL parse_and_run_actions / _run_llm_dispatch (every
action, LLM call and speech output stubbed; nothing is written):
  * rule 6 and rule 7 are ENFORCED while ACTION_FIT_MODE is 'shadow' (the
    handler is not called; the result is "refused (action-fit)"), the robot
    as speaker still runs, an accepted offer still runs, and 'off' turns
    both off; ACTION_FIT_ROBOT_SPEAK_ENFORCE / ACTION_FIT_ACK_REPEAT_ENFORCE
    False are the v3.0.8 path exactly;
  * ACK_CLAIM_NO_FOLLOWUP: the live decline speaks "Very good, sir." with
    no synthetic result and no follow-up round; False is the old path;
  * SELF_CHANGE_PROMISE_CHECK: the promise is dropped, the owner's directive
    is noted through note_for_claude and the honest line is said; a non-
    directive says only "I can't change that myself, sir."; a real action
    keeps the promise; the stream flush holds it; False is the old path.

"deskbot" stands in for the owner's robot; the spoken lines are neutral.

    python -m unittest tests.monolith.test_monolith_robot_speech_1029
"""
from __future__ import annotations

import contextlib
import io
import time
from unittest import mock

from core import dev_notes
from core import onscreen_refs
from tests._monolith_harness import requires_monolith
from tests.monolith import test_monolith_action_fit as fit_t
from tests.monolith import test_monolith_claim_validation as cv_t

SPEAK = "deskbot_speak"
LINE = "hello there"
SAY_LIVE = "Jarvis, say hello there."
SAID = "The deskbot said it, sir."

DECLINE_LIVE = "No, not right now."
DECLINE_REPLY = "[intent:confirmation] Very good, sir. On it."
OFFER = "Shall I note that for Claude to add, sir?"

DIRECTIVE_LIVE = ("Jarvis, you need to make him finish talking before you "
                  "speak.")
PROMISE_LIVE = ("[intent:confirmation] Understood, sir. I'll ensure the "
                "deskbot completes its delivery before I chime in.")


class _FitBase(fit_t._Base):
    def setUp(self):
        super().setUp()
        self.speak = mock.Mock(return_value=SAID)
        self.play = mock.Mock(return_value="Playing jazz, sir.")
        self.acts[SPEAK] = self.speak
        self.acts["play_music"] = self.play

    def _turn(self, reply, user_text, *, mode="shadow", prior=(),
              offer=""):
        bc = self.bc
        self._p(bc, "ACTION_FIT_MODE", mode)
        for name, arg in prior:
            self.history.append({"action": name, "arg": arg, "result": "ok",
                                 "at": time.time() - 10})
        prev = bc._begin_turn_grounding(user_text)
        out = io.StringIO()
        try:
            if offer:
                bc._note_turn_accepted_offer(offer)
            with contextlib.redirect_stdout(out):
                res = bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)
        self.log = out.getvalue()
        return res


@requires_monolith
class RobotSpeechEnforceTests(_FitBase):
    def test_the_live_say_is_refused_in_shadow(self):
        _c, results = self._turn(f"Understood, sir. [ACTION: {SPEAK}, {LINE}]",
                                 SAY_LIVE)
        self.speak.assert_not_called()
        name, res, _info = results[0]
        self.assertEqual(name, SPEAK)
        self.assertTrue(res.startswith("refused (action-fit): not run"))
        self.assertIn("say the words yourself", res)
        self.assertTrue(self.bc._action_result_failed(res))
        self.assertIn(f"[action-fit] blocked {SPEAK}", self.log)

    def test_the_robot_as_speaker_runs(self):
        for text in ("Jarvis, have the bot say hello there.",
                     "make him say hello there", "can the robot say hi"):
            with self.subTest(text=text):
                self.speak.reset_mock()
                self._turn(f"[ACTION: {SPEAK}, {LINE}]", text)
                self.speak.assert_called_once_with(LINE)
                self.assertNotIn("[action-fit]", self.log)

    def test_the_flag_off_is_the_old_path(self):
        self._p(self.bc, "ACTION_FIT_ROBOT_SPEAK_ENFORCE", False, create=True)
        _c, results = self._turn(f"Understood, sir. [ACTION: {SPEAK}, {LINE}]",
                                 SAY_LIVE)
        self.speak.assert_called_once_with(LINE)
        self.assertEqual(results[0][1], SAID)
        self.assertNotIn("[action-fit]", self.log)

    def test_the_flag_off_keeps_the_v308_shadow_line(self):
        # No speech word at all: v3.0.8 logged it in shadow and ran it.
        self._p(self.bc, "ACTION_FIT_ROBOT_SPEAK_ENFORCE", False, create=True)
        self._p(self.bc, "ACTION_FIT_ACK_REPEAT_ENFORCE", False, create=True)
        self._turn(f"[ACTION: {SPEAK}, {LINE}]", "All right.",
                   prior=[(SPEAK, LINE)])
        self.speak.assert_called_once_with(LINE)
        self.assertIn(f"[action-fit] would block {SPEAK}", self.log)

    def test_mode_off_turns_it_off(self):
        self._turn(f"[ACTION: {SPEAK}, {LINE}]", SAY_LIVE, mode="off")
        self.speak.assert_called_once_with(LINE)
        self.assertNotIn("[action-fit]", self.log)

    def test_other_rules_stay_in_shadow(self):
        game_on = self.acts["game_mode_on"]
        self._turn("Certainly, sir. [ACTION: game_mode_on]",
                   "Jarvis, get out of the way for just a second.")
        game_on.assert_called_once()
        self.assertIn("[action-fit] would block game_mode_on", self.log)


@requires_monolith
class AckRepeatEnforceTests(_FitBase):
    def test_the_live_all_right_is_refused_in_shadow(self):
        _c, results = self._turn(f"Very good, sir. [ACTION: {SPEAK}, {LINE}]",
                                 "All right.", prior=[(SPEAK, LINE)])
        self.speak.assert_not_called()
        self.assertIn("acknowledged or declined", results[0][1])
        self.assertIn(f"[action-fit] blocked {SPEAK}", self.log)

    def test_a_repeat_after_a_decline_is_refused(self):
        _c, results = self._turn("On it. [ACTION: play_music, jazz]",
                                 DECLINE_LIVE, prior=[("play_music", "jazz")])
        self.play.assert_not_called()
        self.assertTrue(results[0][1].startswith("refused (action-fit)"))

    def test_an_accepted_offer_runs(self):
        self._turn("On it. [ACTION: play_music, jazz]", "okay",
                   prior=[("play_music", "jazz")],
                   offer="Shall I play some jazz again, sir?")
        self.play.assert_called_once_with("jazz")
        self.assertNotIn("[action-fit]", self.log)

    def test_a_request_runs(self):
        self._turn("On it. [ACTION: play_music, jazz]", "okay, play it again",
                   prior=[("play_music", "jazz")])
        self.play.assert_called_once_with("jazz")

    def test_the_flag_off_is_the_old_path(self):
        self._p(self.bc, "ACTION_FIT_ACK_REPEAT_ENFORCE", False, create=True)
        _c, results = self._turn("On it. [ACTION: play_music, jazz]",
                                 DECLINE_LIVE, prior=[("play_music", "jazz")])
        self.play.assert_called_once_with("jazz")
        self.assertEqual(results[0][1], "Playing jazz, sir.")
        self.assertNotIn("[action-fit]", self.log)


class _ClaimBase(cv_t._Base):
    def _parse_in_turn(self, reply, user, *, ran=(), offer=""):
        bc = self.bc
        prev = bc._begin_turn_grounding(user)
        self.addCleanup(bc._end_turn_grounding, prev)
        if offer:
            bc._note_turn_accepted_offer(offer)
        for name, result in ran:
            bc._note_turn_action_ran(name, result)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()


@requires_monolith
class AckClaimTests(_ClaimBase):
    def test_the_live_decline_speaks_very_good_and_no_round(self):
        cleaned, results, printed = self._parse_in_turn(DECLINE_REPLY,
                                                        DECLINE_LIVE)
        self.assertEqual(cleaned, "[intent:confirmation] Very good, sir.")
        self.assertEqual(results, [])
        self.assertIn("sir only acknowledged or declined", printed)

    def test_only_the_claim_says_very_good(self):
        cleaned, results, _printed = self._parse_in_turn("On it, sir.",
                                                         "Okay.")
        self.assertEqual(cleaned, "Very good, sir.")
        self.assertEqual(results, [])

    def test_the_live_turn_runs_no_follow_up_round(self):
        self._stub("pause_music", "Chrome is already paused, sir")
        self._dispatch(DECLINE_LIVE, DECLINE_REPLY,
                       ["On it. [ACTION: pause_music]"])
        self.gfr.assert_not_called()
        self.assertEqual(self.calls["pause_music"], [])
        self.assertFalse(any("On it" in s for s in self.spoken), self.spoken)

    def test_an_accepted_offer_is_the_old_path(self):
        cleaned, results, _printed = self._parse_in_turn(
            DECLINE_REPLY, "okay", offer=OFFER)
        self.assertEqual(cleaned, "")
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])

    def test_a_request_is_the_old_path(self):
        cleaned, results, _printed = self._parse_in_turn(
            DECLINE_REPLY, "pause the music")
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])

    def test_the_flag_off_is_the_old_path(self):
        self._p(self.bc, "ACK_CLAIM_NO_FOLLOWUP", False, create=True)
        cleaned, results, printed = self._parse_in_turn(DECLINE_REPLY,
                                                        DECLINE_LIVE)
        self.assertEqual(cleaned, "")
        self.assertEqual([n for n, _r, _i in results], ["_unverified_claim"])
        self.assertNotIn("acknowledged or declined", printed)


@requires_monolith
class SelfChangeTests(_ClaimBase):
    def setUp(self):
        super().setUp()
        self._stub("note_for_claude", dev_notes.SPOKEN_LINE)

    def test_the_live_directive_is_noted_and_said_honestly(self):
        cleaned, results, printed = self._parse_in_turn(PROMISE_LIVE,
                                                        DIRECTIVE_LIVE)
        self.assertEqual(cleaned, "[intent:confirmation] I can't change that "
                                  "myself, sir - I've noted it for Claude.")
        self.assertEqual(results, [])            # no follow-up round
        self.assertEqual(self.calls["note_for_claude"],
                         [onscreen_refs.clean(DIRECTIVE_LIVE)])
        self.assertIn("dropped the self-change promise", printed)
        self.assertIn("note_for_claude", self.bc._turn_actions_ran())

    def test_a_second_round_notes_once(self):
        bc = self.bc
        prev = bc._begin_turn_grounding(DIRECTIVE_LIVE)
        self.addCleanup(bc._end_turn_grounding, prev)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._self_change_note(DIRECTIVE_LIVE)
            bc._self_change_note(DIRECTIVE_LIVE)
        self.assertEqual(len(self.calls["note_for_claude"]), 1)

    def test_a_non_directive_says_the_plain_line(self):
        cleaned, results, _printed = self._parse_in_turn(
            "[intent:confirmation] Certainly, sir. I'll ensure he finishes "
            "his business before I interrupt.", "Try it again.")
        self.assertEqual(cleaned,
                         "[intent:confirmation] I can't change that myself, "
                         "sir.")
        self.assertEqual(results, [])
        self.assertEqual(self.calls["note_for_claude"], [])

    def test_a_note_that_fails_says_the_plain_line(self):
        self._stub("note_for_claude", dev_notes.GUEST_LINE)
        cleaned, _results, _printed = self._parse_in_turn(PROMISE_LIVE,
                                                          DIRECTIVE_LIVE)
        self.assertEqual(cleaned, "[intent:confirmation] I can't change that "
                                  "myself, sir.")

    def test_a_real_action_keeps_the_promise(self):
        reply = "I'll make sure to add a pause before I speak, sir."
        cleaned, _results, printed = self._parse_in_turn(
            reply, "remind me to stretch at five",
            ran=[("add_reminder", "Reminder set for 5 PM.")])
        self.assertIn(reply, cleaned)
        self.assertNotIn("self-change", printed)
        self.assertEqual(self.calls["note_for_claude"], [])

    def test_contrition_and_offers_are_untouched(self):
        for reply in ("I'll be more careful, sir.",
                      "Shall I add a pause before the deskbot speaks, sir?"):
            with self.subTest(reply=reply):
                cleaned, _results, printed = self._parse_in_turn(
                    reply, DIRECTIVE_LIVE)
                self.assertIn(reply, cleaned)
                self.assertNotIn("self-change", printed)

    def test_the_flag_off_is_the_old_path(self):
        self._p(self.bc, "SELF_CHANGE_PROMISE_CHECK", False, create=True)
        cleaned, results, printed = self._parse_in_turn(PROMISE_LIVE,
                                                        DIRECTIVE_LIVE)
        self.assertEqual(cleaned, PROMISE_LIVE)
        self.assertEqual(results, [])
        self.assertNotIn("self-change", printed)
        self.assertEqual(self.calls["note_for_claude"], [])

    def test_outside_an_owner_turn_nothing_changes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned, _results = self.bc.parse_and_run_actions(PROMISE_LIVE)
        self.assertEqual(cleaned, PROMISE_LIVE)

    def test_a_check_fault_changes_nothing(self):
        self._p(self.bc._claim_validator, "drop_self_change_promises",
                side_effect=RuntimeError("boom"))
        cleaned, _results, printed = self._parse_in_turn(PROMISE_LIVE,
                                                         DIRECTIVE_LIVE)
        self.assertEqual(cleaned, PROMISE_LIVE)
        self.assertIn("self-change check failed", printed)

    def _flush(self, chunks):
        bc = self.bc
        prev = bc._begin_turn_grounding(DIRECTIVE_LIVE)
        self.addCleanup(bc._end_turn_grounding, prev)
        spoken: list[str] = []
        with mock.patch.object(bc, "threading", mock.Mock(
                Thread=cv_t._InlineThread)):
            buf = bc._SentenceFlushBuffer(speak_fn=spoken.append)
            for c in chunks:
                buf.feed(c)
        return spoken

    def test_the_stream_flush_holds_the_promise(self):
        chunks = ("I'll build in a slight pause before the deskbot speaks, "
                  "sir. ", "It should help a great deal, I think. ", "tail")
        self.assertEqual(self._flush(chunks), [])
        self._p(self.bc, "SELF_CHANGE_PROMISE_CHECK", False, create=True)
        self.assertNotEqual(self._flush(chunks), [])
