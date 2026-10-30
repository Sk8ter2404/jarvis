"""Monolith wiring for the v3.0.6 routing fixes (live 2026-10-28,
session_2026-10-28_17-21-18.log), against the REAL monolith with every LLM
call stubbed and no real program launched:

  * B1 17:26:20 "We need to give the deskbot more voice lines." -> the
    brain sent [ACTION: deskbot_speak, I am ready for more dialogue.] and
    the robot spoke. (The live words and action named the owner's robot;
    "deskbot" is a made-up stand-in of the same shape, because public code
    never names the device.) Now the screen route (_utterance_route_reply ->
    _screen_route_reply -> core.dispatcher.screen_route -> core.onscreen_refs.
    claude_note -> dev_todo_note) notes it for Claude before the brain.
  * B2 17:25:16 "Jarvis, can you open it up so I can help you?" -> the brain
    sent [ACTION: open_app, PowerShell] and PowerShell opened. Now the real
    launch_app handler (core.actions._act_launch_app) says "Open what, sir?"
    and opens nothing.

    python -m unittest tests.monolith.test_monolith_routing_306
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

LIVE_TODO = "We need to give the deskbot more voice lines."
LIVE_TODO_BRAIN = ("[intent:confirmation] Very good, sir. "
                   "[ACTION: deskbot_speak, I am ready for more dialogue.]")
NOTED = "Noted for Claude, sir."

LIVE_OPEN = "Jarvis, can you open it up so I can help you?"
LIVE_OPEN_BRAIN = ("[intent:confirmation] Certainly, sir. "
                   "[ACTION: open_app, PowerShell]")
LIVE_OPEN_HISTORY = [
    {"role": "user", "content": "Jarvis, how's the server?"},
    {"role": "assistant", "content": "[ACTION: server_status]"},
]


@requires_monolith
class DevTodoTurnTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._stub("note_for_claude", NOTED)
        self._stub("deskbot_speak", "The Desk Bot said it, sir.")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=LIVE_TODO_BRAIN)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def test_the_live_turn_is_noted_and_the_robot_stays_quiet(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(LIVE_TODO)
        self.assertEqual(self.calls["note_for_claude"],
                         ["We need to give the deskbot more voice lines"])
        self.assertEqual(self.calls["deskbot_speak"], [])
        self.llm.assert_not_called()

    def test_setting_off_leaves_it_to_the_brain(self):
        from core import config as cfg
        with mock.patch.object(cfg, "DEV_TODO_NOTES", False):
            got = self._quiet(self.bc._utterance_route_reply, LIVE_TODO)
        self.assertIsNone(got)


@requires_monolith
class UnresolvedOpenTurnTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "conversation_history", list(LIVE_OPEN_HISTORY))
        # The real _call_llm appends the user line and then the brain's raw
        # reply to conversation_history BEFORE parse_and_run_actions runs the
        # action (bobert_companion.py _call_llm), so the reply naming
        # PowerShell is already in the history the launch handler reads.
        # A bare return_value hid that (review of B2).
        self.brain_reply = LIVE_OPEN_BRAIN

        def _brain(user_text, *a, **k):
            hist = bc.conversation_history
            hist.append({"role": "user", "content": user_text})
            hist.append({"role": "assistant", "content": self.brain_reply})
            return self.brain_reply

        self.llm = self._p(bc, "get_response_with_animation",
                           side_effect=_brain)
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=[None] * 8)
        from core import actions as A
        self.popen = self._p(A.subprocess, "Popen")
        self.startfile = self._p(A.os, "startfile", create=True)
        self._p(A, "_site_shortcut_url", return_value=None)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_the_live_turn_asks_and_opens_nothing(self):
        self._run(LIVE_OPEN)
        self.popen.assert_not_called()
        self.startfile.assert_not_called()
        self.assertTrue(any("Open what, sir?" in t for t in self.spoken),
                        self.spoken)
        self.assertFalse(any("launched" in t.lower() for t in self.spoken))

    def test_a_named_app_still_launches(self):
        self._run("Jarvis, open PowerShell")
        self.assertTrue(self.popen.called or self.startfile.called)
        self.assertFalse(any("Open what" in t for t in self.spoken))

    def test_the_reply_in_history_names_the_app_and_still_asks(self):
        self._run(LIVE_OPEN)
        self.assertTrue(any("open_app, PowerShell" in m["content"]
                            for m in self.bc.conversation_history))
        self.popen.assert_not_called()

    def test_setting_off_is_the_old_behaviour(self):
        from core import config as cfg
        with mock.patch.object(cfg, "UNRESOLVED_OPEN_ASK", False):
            self._run(LIVE_OPEN)
        self.assertTrue(self.popen.called or self.startfile.called)


LIVE_DECLINE_ASK = "How's the master sheet looking?"
LIVE_DECLINE = "[intent:briefing] I'm afraid I've no way to check that, sir."


@requires_monolith
class DeclineRecheckTurnTests(_Base):
    """B4 17:26:00: the flat decline now ends on an offer a yes can take."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self.hist = [{"role": "user", "content": LIVE_DECLINE_ASK},
                     {"role": "assistant", "content": LIVE_DECLINE}]
        self._p(bc, "conversation_history", self.hist)
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=LIVE_DECLINE)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_the_live_decline_offers_a_note_and_a_yes_can_take_it(self):
        from core import prompt_router as pr
        # Against the REAL registry: no action name matches "master sheet".
        self.assertEqual(pr.nearest_action(LIVE_DECLINE_ASK,
                                           list(self.bc.ACTIONS)), "")
        self._run(LIVE_DECLINE_ASK)
        said = " ".join(self.spoken)
        self.assertIn("no way to check that", said)
        self.assertIn(pr.DECLINE_NOTE_OFFER, said)
        self.assertTrue(self.hist[-1]["content"].endswith(
            pr.DECLINE_NOTE_OFFER), self.hist[-1])
        # ... and it is the open offer a plain "yes" answers.
        self.assertIn("for Claude", self.bc._open_offer.peek())
        self.assertTrue(self.bc._offer_is_last_said(
            self.bc._open_offer.peek()))

    def test_shadow_and_off_leave_the_reply_alone(self):
        from core import prompt_router as pr
        for mode in ("shadow", "off"):
            with self.subTest(mode=mode):
                self.spoken.clear()
                self._p(self.bc, "DECLINE_RECHECK_MODE", mode)
                self._run(LIVE_DECLINE_ASK)
                self.assertNotIn(pr.DECLINE_NOTE_OFFER, " ".join(self.spoken))


@requires_monolith
class RobotSpeechFitTests(_Base):
    """B7: action-fit rule 6 in the REAL _runner (shadow by default): the
    live line's deskbot_speak still runs and is logged as a misfit - since
    v3.0.9 only with ACTION_FIT_ROBOT_SPEAK_ENFORCE False; on (the default)
    rule 6 is enforced even in shadow."""

    def setUp(self):
        super().setUp()
        self._stub("deskbot_speak", "The Desk Bot said it, sir.")

    def _dispatch(self, user_text, mode):
        bc = self.bc
        self._p(bc, "ACTION_FIT_MODE", mode)
        prev = bc._begin_turn_grounding(user_text)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                bc.parse_and_run_actions(LIVE_TODO_BRAIN)
        finally:
            bc._end_turn_grounding(prev)
        return out.getvalue()

    def test_shadow_runs_and_logs_the_live_misfit(self):
        # The v3.0.6 shadow behaviour, kept exactly behind
        # ACTION_FIT_ROBOT_SPEAK_ENFORCE False (v3.0.9).
        self._p(self.bc, "ACTION_FIT_ROBOT_SPEAK_ENFORCE", False, create=True)
        log = self._dispatch(LIVE_TODO, "shadow")
        self.assertEqual(self.calls["deskbot_speak"],
                         ["I am ready for more dialogue."])
        self.assertIn("[action-fit] would block deskbot_speak", log)

    def test_shadow_enforces_rule_six_since_309(self):
        # v3.0.9 (live 2026-10-29): rule 6 is enforced even in shadow.
        log = self._dispatch(LIVE_TODO, "shadow")
        self.assertEqual(self.calls["deskbot_speak"], [])
        self.assertIn("[action-fit] blocked deskbot_speak", log)

    def test_on_does_not_run_it(self):
        log = self._dispatch(LIVE_TODO, "on")
        self.assertEqual(self.calls["deskbot_speak"], [])
        self.assertIn("[action-fit] blocked deskbot_speak", log)

    def test_an_asked_line_is_not_a_misfit(self):
        log = self._dispatch("Jarvis, make him talk.", "shadow")
        self.assertEqual(len(self.calls["deskbot_speak"]), 1)
        self.assertNotIn("[action-fit]", log)

    def test_config_default_is_still_shadow(self):
        from core import config as cfg
        self.assertEqual(cfg.ACTION_FIT_MODE, "shadow")


LIVE_YT = "Jarvis play vision quest on the left monitor."
LIVE_YT_BRAIN = ("[intent:confirmation] Right away, sir. "
                 "[ACTION: youtube_play, vision quest | monitor:left]")


@requires_monolith
class YoutubeMonitorArgTurnTests(_Base):
    """B6 17:28:11: the REAL youtube_play handler (core.actions) behind the
    REAL parse loop: the monitor never reaches the YouTube search."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.sap = self._p(bc, "_streaming_auto_play",
                           return_value="playing 'vision quest' on YouTube")
        self._p(bc, "_JARVIS_MEDIA_WINDOW_HWND", {})

    def test_the_live_token_searches_the_title_only(self):
        prev = self.bc._begin_turn_grounding(LIVE_YT)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.bc.parse_and_run_actions(LIVE_YT_BRAIN)
        finally:
            self.bc._end_turn_grounding(prev)
        self.sap.assert_called_once_with("youtube", "vision quest")


if __name__ == "__main__":
    unittest.main()


@requires_monolith
class DeclineRecheckLifecycleTests(_Base):
    """Review of B4: against the REAL ACTIONS registry a decline of a
    question about the server / PC shutting down is never answered with an
    offer to shut JARVIS down or restart him."""

    def test_no_lifecycle_action_is_offered_by_name(self):
        from core import prompt_router as pr
        names = list(self.bc.ACTIONS)
        self.assertIn("shut_down", names)
        for ask in ("is the server going to shut down tonight?",
                    "can you check whether the computer will restart?"):
            with self.subTest(ask=ask):
                got = pr.decline_recheck(LIVE_DECLINE, ask, names)
                self.assertNotIn("shut down", got)
                self.assertNotIn("restart", got)
                self.assertEqual(got, pr.DECLINE_NOTE_OFFER)
