"""v3.0.11 lane V, monolith wiring: he makes up what he sees, and looking fails
(live 2026-11-05 14:57-14:59).

Against the REAL monolith with every action and LLM call stubbed: a camera ask
runs see_user with his words and speaks the vision answer with no brain call; a
retry follows the look; a claim the brain still makes is not spoken; the same
lead line is not said twice in a chain; "what did you say?" says the last line.

    python tools/run_tests.py monolith.test_monolith_v3011_camera_ask
"""
from __future__ import annotations

import unittest
from unittest import mock

import core.config as cfg
from core import claim_validator
from tests._monolith_harness import requires_monolith
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

    def test_a_failed_look_goes_to_the_follow_up_round(self):
        self._stub("see_user", "no webcam frames available yet")
        self._dispatch(ASK, "x", followups=["I have no picture yet, sir."])
        self.gfr.assert_called()
        self.assertNotIn("no webcam frames available yet", self._spoken())

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


class LookAgainTests(_CameraBase):
    def _ran(self, name, arg):
        self._hist.append({"action": name, "arg": arg, "result": "x",
                           "at": self.bc.time.time()})

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

    def test_rule_8_refuses_the_monitors_in_shadow(self):
        self._p(self.bc, "ACTION_FIT_MODE", "shadow", create=True)
        self._p(self.bc, "LOOK_RETRY_ROUTE", False, create=True)
        self._p(self.bc, "CAMERA_ASK_ROUTE", False, create=True)
        self._dispatch(ASK, "[ACTION: see_screen, what am I holding?]",
                       followups=["I'll have to look, sir."])
        self.assertEqual(self.calls["see_screen"], [])


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
