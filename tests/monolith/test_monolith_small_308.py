"""Monolith wiring for two v3.0.8 note-route fixes, against the REAL monolith
with every action and LLM call stubbed (nothing is written or spoken):

  * PASS_FUTURE_REFERS_BACK: "I'll hand it to Claude" made a note reading
    "I'll" (known since 3.0.5). Now the screen route (_utterance_route_reply
    -> _screen_route_reply -> core.dispatcher.screen_route -> core.
    onscreen_refs.claude_note) notes the PREVIOUS request, before the brain.
  * DEV_TODO_ROBOT_STRICT: the 3.0.6 robot word took any word ending in
    "bot", so "we need to fix the chatbot" was noted for Claude and never
    reached the brain. Now it is the brain's; the robot shapes still note.
    ("deskbot" is a made-up stand-in; public code never names the device.)

Fails on v3.0.7 (1561e62).

    python -m unittest tests.monolith.test_monolith_small_308
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from core import onscreen_refs as O
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

NOTED = "Noted for Claude, sir."
BRAIN = "[intent:confirmation] Very good, sir."


@requires_monolith
class SmallNoteRouteTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._stub("note_for_claude", NOTED)
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=BRAIN)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_a_future_hand_off_notes_the_previous_request(self):
        self._run("I'll hand it to Claude.")
        self.assertEqual(self.calls["note_for_claude"], [O._PASS_REFERS_BACK])
        self.llm.assert_not_called()

    def test_a_future_hand_off_after_a_task_notes_the_task(self):
        got = self._quiet(self.bc._utterance_route_reply,
                          "Find the manual, or I'll pass it to Claude")
        self.assertEqual(got, "[ACTION: note_for_claude, Find the manual]")

    def test_the_chatbot_reaches_the_brain(self):
        self._run("We need to fix the chatbot.")
        self.assertEqual(self.calls["note_for_claude"], [])
        self.llm.assert_called()

    def test_other_common_bot_words_reach_the_brain(self):
        for said in ("we need to make the Talbot more reliable",
                     "the abbot needs more lines",
                     "we should make the robot vacuum smarter"):
            with self.subTest(said=said):
                got = self._quiet(self.bc._utterance_route_reply, said) or ""
                self.assertNotIn("note_for_claude", got)

    def test_the_robot_shapes_still_note(self):
        for said in ("We need to give the deskbot more voice lines.",
                     "we need to fix the desk bot",
                     "the bot needs more voice lines"):
            with self.subTest(said=said):
                got = self._quiet(self.bc._utterance_route_reply, said) or ""
                self.assertTrue(got.startswith("[ACTION: note_for_claude, "),
                                said)

    def test_settings_off_are_the_old_behaviour(self):
        from core import config as cfg
        with mock.patch.object(cfg, "PASS_FUTURE_REFERS_BACK", False):
            got = self._quiet(self.bc._utterance_route_reply,
                              "I'll hand it to Claude")
        self.assertEqual(got, "[ACTION: note_for_claude, I'll]")
        with mock.patch.object(cfg, "DEV_TODO_ROBOT_STRICT", False):
            got = self._quiet(self.bc._utterance_route_reply,
                              "we need to fix the chatbot")
        self.assertEqual(got,
                         "[ACTION: note_for_claude, we need to fix the chatbot]")


if __name__ == "__main__":
    unittest.main()
