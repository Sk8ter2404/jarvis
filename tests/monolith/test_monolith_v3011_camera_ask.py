"""v3.0.11 lane V, monolith wiring: he makes up what he sees, and looking fails
(live 2026-11-05 14:57-14:59).

Against the REAL monolith with every action and LLM call stubbed: a camera ask
runs see_user with his words and speaks the vision answer with no brain call; a
retry follows the look; a claim the brain still makes is not spoken; the same
lead line is not said twice in a chain; "what did you say?" says the last line.

    python tools/run_tests.py monolith.test_monolith_v3011_camera_ask
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import core.config as cfg
from core import camera_ask, claim_validator
from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated
from tests.monolith.test_monolith_claim_validation import _Base

ASK = "Jarvis, what am I holding?"
ANSWER = "[local-vision] You are holding a can of Coke."


@requires_monolith
class _CameraBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "_focus_changed_recently", return_value=False)
        self._hist = self._p(bc, "_action_history", [])
        self._stub("see_user", ANSWER)
        self._stub("see_screen", "Windows on screen: a browser")
        self._stub("recall_screen", "Recorded: nothing")
        for name in ("CAMERA_ASK_ROUTE", "LOOK_RETRY_ROUTE", "CAMERA_CLAIM_CHECK",
                     "CHAIN_LEAD_DEDUPE", "REPEAT_LAST_LINE",
                     "ACTION_FIT_CAMERA_ASK_ENFORCE"):
            self._p(bc, name, True, create=True)

    def _spoken(self):
        return [c.args[0] for c in self.bc._speak.call_args_list]


class CameraAskTurnTests(_CameraBase):
    def test_live_turn_runs_the_look_and_speaks_the_answer(self):
        self._dispatch(ASK, "You're holding a box of Tic Tacs, sir.")
        self.assertEqual(self.calls["see_user"], ["what am I holding?"])
        self.bc.get_response_with_animation.assert_not_called()
        self.gfr.assert_not_called()
        self.assertEqual(self._spoken(), ["You are holding a can of Coke."])

    def test_the_answer_joins_the_history(self):
        self._dispatch(ASK, "x")
        self.assertIn({"role": "assistant",
                       "content": "You are holding a can of Coke."},
                      list(self.bc.conversation_history))

    def test_a_failed_look_says_so_with_no_brain_round(self):
        self._stub("see_user", "failed to look: no webcam frames available yet")
        self._dispatch(ASK, "x", followups=["You're holding a can of Coke, sir."])
        self.gfr.assert_not_called()
        self.assertEqual(self._spoken(), [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_a_look_that_raises_says_so_with_no_brain_round(self):
        def boom(_arg=""):
            raise RuntimeError("no camera lib")
        self._actions["see_user"] = boom
        with mock.patch.dict("os.environ", {"JARVIS_BUG_AUTO_CAPTURE": "0"}):
            self._dispatch(ASK, "x", followups=["You're holding a can of Coke, sir."])
        self.gfr.assert_not_called()
        self.assertEqual(self._spoken(), [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_flag_off_the_brain_answers(self):
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._p(self.bc, "CAMERA_CLAIM_CHECK", False, create=True)
        self._dispatch(ASK, "You're holding a box of Tic Tacs, sir.")
        self.bc.get_response_with_animation.assert_called_once()
        self.assertEqual(self.calls["see_user"], [])

    def test_screen_ask_is_left_to_the_brain(self):
        for words in ("what's on my screen", "what is this page", "what's this song"):
            self.assertIsNone(self._quiet(self.bc._camera_ask_route_reply, words), words)

    def test_pc_control_off_no_route(self):
        self._p(self.bc, "PC_CONTROL_ENABLED", False)
        self.assertIsNone(self._quiet(self.bc._utterance_route_reply, ASK))

    def test_an_unregistered_action_is_no_route(self):
        self._actions.pop("see_user")
        self.assertIsNone(self._quiet(self.bc._camera_ask_route_reply, ASK))


class _FakeCv2(types.ModuleType):
    def __init__(self):
        super().__init__("cv2")

    def imencode(self, _ext, _frame):
        buf = mock.Mock()
        buf.tobytes.return_value = b"PNG"
        return True, buf


class RealSeeUserTurnTests(_CameraBase):
    """The real _act_see_user under the real skills/face_tracker.py gaze wrapper:
    lane tests that stubbed see_user never saw what the owner would hear."""

    def _real_look(self, vision=ANSWER, age=0.5, cameras=True, monitor="right"):
        bc = self.bc
        now = bc.time.time()
        frame = mock.Mock()
        frame.copy.return_value = frame
        cams = [{"index": 0, "label": "cam", "look_x": 0.5}] if cameras else []
        self._p(cfg, "CAMERAS", cams)
        self._p(bc, "_camera_last_seen", {0: now} if cameras else {})
        self._p(bc, "_camera_latest_frame", {0: frame} if cameras else {})
        self._p(bc, "_camera_last_frame_at", {0: now - age} if cameras else {})
        self._p(bc, "_camera_last_read_error", {})
        self.vision = self._p(bc, "ask_vision", return_value=vision)
        for name in ("CAMERA_ASK_ROUTE", "LOOK_RETRY_ROUTE", "CAMERA_CLAIM_CHECK"):
            self._p(cfg, name, True, create=True)
        modules = mock.patch.dict(sys.modules, {"cv2": _FakeCv2()})
        modules.start()
        self.addCleanup(modules.stop)
        mod, _ = load_skill_isolated("face_tracker", register=False)
        mod._state.update({"current_monitor": monitor})
        self.calls["see_user"] = []
        self._actions["see_user"] = mod._build_see_user_wrapper(bc._act_see_user)

    def test_the_gaze_note_is_never_spoken(self):
        self._real_look()
        self._dispatch(ASK, "x")
        self.assertEqual(self._spoken(), ["You are holding a can of Coke."])
        self.gfr.assert_not_called()

    def test_a_stale_note_and_the_gaze_note_are_never_spoken(self):
        self._real_look(age=12.0)
        self._dispatch(ASK, "x")
        self.assertEqual(self._spoken(), ["You are holding a can of Coke."])

    def test_no_look_is_the_honest_line_and_no_brain(self):
        coke = "You're holding a can of Coke, sir."
        for kw in ({"vision": "(local vision unavailable - Ollama not reachable)"},
                   {"vision": "(vision failed: HTTP 500)"},
                   {"cameras": False}):
            with self.subTest(**kw):
                self.bc._speak.reset_mock()
                self._real_look(**kw)
                self._dispatch(ASK, "x", followups=[coke])
                self.gfr.assert_not_called()
                self.assertEqual(self._spoken(),
                                 [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_no_webcam_frames_yet_is_the_honest_line(self):
        self._real_look()
        self._p(self.bc, "_camera_last_seen", {})
        self._p(self.bc, "_camera_latest_frame", {})
        self._dispatch(ASK, "x", followups=["You're holding a can of Coke, sir."])
        self.gfr.assert_not_called()
        self.assertEqual(self._spoken(),
                         [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_a_frame_ten_minutes_old_is_not_answered_as_current(self):
        self._real_look(age=600.0)
        self._dispatch(ASK, "x", followups=["You're holding a can of Coke, sir."])
        self.vision.assert_not_called()
        self.gfr.assert_not_called()
        self.assertEqual(self._spoken(), [camera_ask.STALE_FRAME_LINE])

    def test_i_cant_make_it_out_is_spoken_as_it_stands(self):
        for line in ("I can't make out what you're holding.",
                     "You couldn't be holding anything but a can."):
            with self.subTest(line=line):
                self.bc._speak.reset_mock()
                self._real_look(vision=line)
                self._dispatch(ASK, "x", followups=["You're holding a phone, sir."])
                self.bc.get_response_with_animation.assert_not_called()
                self.gfr.assert_not_called()
                self.assertEqual(self._spoken(), [line])

    def test_a_failed_look_never_backs_the_brains_claim(self):
        # The original incident, off the route: the look found no picture and the
        # brain's follow-up "You're holding a can of Coke, sir." was spoken.
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._real_look(vision="(local vision unavailable - Ollama not reachable)")
        self._dispatch("what do you think of what I'm holding?",
                       "You're holding a can of Coke, sir.",
                       followups=["You're holding a can of Coke, sir."])
        self.assertEqual(self._spoken(),
                         [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_the_brains_own_failed_look_backs_nothing(self):
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._real_look(vision="(local vision unavailable - Ollama not reachable)")
        self._dispatch("what do you think of what I'm holding?",
                       "[ACTION: see_user, what am I holding?]",
                       followups=["You're holding a can of Coke, sir."])
        said = " ".join(self._spoken())
        self.assertNotIn("Coke", said)
        self.assertIn(claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE, said)

    def test_a_failed_look_in_the_history_backs_no_claim(self):
        self._hist.append({"action": "see_user", "arg": "q", "at": self.bc.time.time(),
                           "result": "failed to look: no cameras configured\n\n"
                                     "[gaze: currently looking at the right monitor]"})
        self._hist.append({"action": "see_user", "arg": "q", "at": self.bc.time.time(),
                           "result": ANSWER})
        self.assertEqual(self._quiet(self.bc._camera_claim_grounding)[:1], [ANSWER])


class LookAgainTests(_CameraBase):
    def _ran(self, name, arg, turn=None):
        self._hist.append({"action": name, "arg": arg, "result": "x",
                           "at": self.bc.time.time(), "turn": turn})

    def test_after_a_camera_ask(self):
        self._ran("see_user", "what am I holding?")
        self._dispatch("Jarvis, look again.", "Apologies, sir. [ACTION: recall_screen]")
        self.assertEqual(self.calls["see_user"], ["what am I holding?"])
        self.assertEqual(self.calls["recall_screen"], [])
        self.assertEqual(self.calls["see_screen"], [])
        self.bc.get_response_with_animation.assert_not_called()

    def test_after_a_screen_ask(self):
        self._ran("see_screen", "read this page")
        self.assertEqual(self._quiet(self.bc._camera_ask_route_reply, "look again"),
                         "[ACTION: see_screen, read this page]")

    def test_a_bare_retry_only_right_after_the_look(self):
        self._ran("see_user", "what am I holding?", turn=4)
        self._p(self.bc, "_last_line_turn", [5])
        self.assertEqual(self._quiet(self.bc._camera_ask_route_reply, "try again"),
                         "[ACTION: see_user, what am I holding?]")
        # "any messages I missed" in between: a shortcut, so no row, one more turn.
        self._p(self.bc, "_last_line_turn", [6])
        self.assertIsNone(self._quiet(self.bc._camera_ask_route_reply, "try again"))
        self.assertEqual(self._quiet(self.bc._camera_ask_route_reply, "look again"),
                         "[ACTION: see_user, what am I holding?]")

    def test_a_retry_of_a_question_about_the_past_stays_one(self):
        self._ran("recall_screen", "ten minutes ago")
        self.assertEqual(self._quiet(self.bc._camera_ask_route_reply, "look again"),
                         "[ACTION: recall_screen, ten minutes ago]")

    def test_rule_8_refuses_the_monitors_in_shadow(self):
        self._p(self.bc, "ACTION_FIT_MODE", "shadow", create=True)
        self._p(self.bc, "LOOK_RETRY_ROUTE", False, create=True)
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._dispatch(ASK, "[ACTION: see_screen, what am I holding?]",
                       followups=["I'll have to look, sir."])
        self.assertEqual(self.calls["see_screen"], [])


@requires_monolith
class HistoryTurnTests(MonolithGlobalsTestCase):
    def test_a_row_carries_its_owner_turn(self):
        bc = self.bc
        hist = []
        for name, value in (("_action_history", hist), ("_last_line_turn", [9])):
            patcher = mock.patch.object(bc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        bc.record_action_history("see_user", "q", "r")
        self.assertEqual(hist[-1]["turn"], 9)


class ReplayLookTests(_CameraBase):
    def test_do_it_again_says_the_answer(self):
        self._hist.append({"action": "see_user", "arg": "what am I holding?",
                           "result": "x", "at": self.bc.time.time()})
        self._stub("see_user", ANSWER + "\n\n[gaze: currently looking at the right monitor]")
        self.assertEqual(self._quiet(self.bc._act_replay_last_action),
                         "You are holding a can of Coke.")


class CameraClaimTurnTests(_CameraBase):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)

    def test_his_ask_runs_the_look_instead(self):
        self._dispatch("what do you think of what I'm holding?",
                       "You're holding a box of Tic Tacs, sir.",
                       followups=["You are holding a can of Coke, sir."])
        self.assertEqual(self.calls["see_user"],
                         ["what do you think of what I'm holding?"])
        self.assertNotIn("You're holding a box of Tic Tacs, sir.", self._spoken())
        self.assertIn("You are holding a can of Coke, sir.", self._spoken())

    def _about_it(self, *followups):
        self._dispatch("what do you think of what I'm holding?",
                       "You're holding a smartphone, sir.", followups=followups)

    def test_a_look_that_raises_never_speaks_the_claim(self):
        def boom(_arg=""):
            raise RuntimeError("no camera lib")
        self._actions["see_user"] = boom
        self._p(self.bc, "record_action_error")
        with mock.patch.dict("os.environ", {"JARVIS_BUG_AUTO_CAPTURE": "0"}):
            self._about_it("You're holding a smartphone, sir.")
        self.assertEqual(self._spoken(),
                         [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])
        self.gfr.assert_not_called()

    def test_a_look_goes_through_the_normal_gates(self):
        self._p(self.bc, "_action_fit_refusal", return_value="refused by a gate")
        self._about_it("You're holding a smartphone, sir.")
        self.assertEqual(self.calls["see_user"], [])
        self.assertEqual(self._spoken(),
                         [claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE])

    def test_a_look_that_runs_keeps_its_bookkeeping(self):
        rec = self._p(self.bc, "record_action_history")
        self._about_it("You are holding a can of Coke, sir.")
        self.assertEqual(self.calls["see_user"],
                         ["what do you think of what I'm holding?"])
        self.assertEqual(rec.call_args.args[:2],
                         ("see_user", "what do you think of what I'm holding?"))

    def test_a_failed_look_is_not_run_again_for_the_next_invention(self):
        self._stub("see_user", "no webcam frames, the look failed")
        self._about_it("You're holding a smartphone, sir.",
                       "You're holding a smartphone, sir.")
        self.assertEqual(len(self.calls["see_user"]), 1)
        said = " ".join(self._spoken())
        self.assertNotIn("smartphone", said)
        self.assertIn(claim_validator.CAMERA_CLAIM_LOOK_FAILED_LINE, said)

    def test_a_crash_after_the_claim_is_found_does_not_speak_it(self):
        self._p(self.bc, "_camera_claim_look_tried", side_effect=RuntimeError)
        self._about_it("x")
        self.assertEqual(self._spoken(),
                         [claim_validator.CAMERA_CLAIM_HONEST_LINE])

    def test_an_unrelated_line_does_not_back_an_invention(self):
        self._dispatch("Jarvis, can you see me?", "I can, sir.")
        self.bc._speak.reset_mock()
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._dispatch("how is it going", "You're holding a can of Coke, sir.")
        self.assertEqual(self._spoken(), [claim_validator.CAMERA_CLAIM_HONEST_LINE])

    def test_no_ask_says_it_has_not_looked(self):
        self._dispatch("how is it going", "You're holding a box of Tic Tacs, sir.")
        self.assertEqual(self.calls["see_user"], [])
        self.assertEqual(self._spoken(), [claim_validator.CAMERA_CLAIM_HONEST_LINE])

    def test_ordinary_speech_passes(self):
        for line in ("You're holding up well, sir.", "You are holding a meeting at 3."):
            self.bc._speak.reset_mock()
            self._dispatch("how is it going", line)
            self.assertEqual(self._spoken(), [line])

    def test_flush_buffer_holds_it(self):
        frame_prev = self.bc._begin_turn_grounding("how is it going")
        try:
            self.assertIsNotNone(self.bc._claim_validator.find_camera_claim(
                "You are holding a smartphone, sir.",
                ran_actions=self.bc._turn_actions_ran(),
                user_text=self.bc._turn_user_text(),
                grounding=self.bc._camera_claim_grounding()))
        finally:
            self.bc._end_turn_grounding(frame_prev)

    def test_flag_off_spoken_as_before(self):
        self._p(self.bc, "CAMERA_CLAIM_CHECK", False, create=True)
        line = "You're holding a box of Tic Tacs, sir."
        self._dispatch("how is it going", line)
        self.assertEqual(self._spoken(), [line])


class RecordsPromiseTurnTests(_CameraBase):
    LIVE = ("My apologies, sir; I'll update my visual database to include 'can' "
            "instead of 'smartphone'.")

    def setUp(self):
        super().setUp()
        self._p(cfg, "SELF_CHANGE_RECORDS_CHECK", True, create=True)
        self._p(self.bc, "SELF_CHANGE_PROMISE_CHECK", True, create=True)

    def test_live_line_is_answered_honestly(self):
        self._dispatch("Jarvis, this is a coke can, dude.", self.LIVE)
        said = " ".join(self._spoken())
        self.assertNotIn("update my visual database", said)
        self.assertIn("My apologies, sir.", said)
        self.assertIn(claim_validator.SELF_CHANGE_RECORDS_LINE, said)


class ChainLeadTests(_CameraBase):
    def test_live_chain_says_it_once(self):
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._p(self.bc, "LOOK_RETRY_ROUTE", False, create=True)
        self._stub("see_screen", "Windows: one", "Windows: two", "Windows: three")
        self._dispatch("what do the windows say",
                       "Apologies, sir. [ACTION: recall_screen]",
                       followups=["Apologies, sir. [ACTION: see_screen, windows]",
                                  "Apologies, sir. [ACTION: see_screen, windows two]",
                                  "Apologies, sir. There are three windows."])
        said = self._spoken()
        self.assertEqual(said.count("Apologies, sir."), 1, said)
        self.assertIn("There are three windows.", said)

    def test_flag_off_each_round_says_it(self):
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._p(self.bc, "LOOK_RETRY_ROUTE", False, create=True)
        self._p(self.bc, "CHAIN_LEAD_DEDUPE", False, create=True)
        self._stub("see_screen", "Windows: one", "Windows: two")
        self._dispatch("what do the windows say",
                       "Apologies, sir. [ACTION: recall_screen]",
                       followups=["Apologies, sir. [ACTION: see_screen, windows]"])
        self.assertEqual(self._spoken().count("Apologies, sir."), 2)


class RepeatLastLineTests(_CameraBase):
    LINE = "I'd wager so, sir. You're quite clearly here."

    def setUp(self):
        super().setUp()
        self._last = self._p(self.bc, "_last_line", self.bc._repeat_line.LastLine())
        self._p(self.bc, "set_state")

    def test_says_it_word_for_word_with_no_brain(self):
        self._last.note(self.LINE, turn=1)
        with mock.patch.object(self.bc, "get_response_with_animation") as brain:
            self.assertTrue(self._quiet(self.bc._run_voice_shortcuts,
                                        "Jarvis, what did you say?"))
        brain.assert_not_called()
        self.assertEqual(self._spoken(), [self.LINE])

    def test_nothing_said_yet_falls_through(self):
        self.assertFalse(self._quiet(self.bc._repeat_last_line_shortcut,
                                     "what did you say?"))

    def test_other_lines_fall_through(self):
        self._last.note(self.LINE, turn=1)
        self.assertFalse(self._quiet(self.bc._repeat_last_line_shortcut, ASK))

    def test_flag_off(self):
        self._last.note(self.LINE, turn=1)
        self._p(self.bc, "REPEAT_LAST_LINE", False, create=True)
        self.assertFalse(self._quiet(self.bc._repeat_last_line_shortcut,
                                     "what did you say?"))

    def test_recall_note_stays_out(self):
        self._p(cfg, "REPEAT_LAST_LINE", True, create=True)
        self._p(self.bc, "RECALL_ACTION_GROUNDING", True, create=True)
        self._hist.append({"action": "recognize_face", "arg": "", "result": "That's you",
                           "at": self.bc.time.time()})
        self.assertEqual(self._quiet(self.bc._recall_action_note,
                                     "Jarvis, what did you say?"), "")


if __name__ == "__main__":
    unittest.main()
