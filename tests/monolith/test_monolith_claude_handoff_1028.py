"""Monolith wiring for the Claude hand-off the STT mis-heard (2026-10-28).

Live 15:58 (session_2026-10-28_12-55-29.log): after a manual search JARVIS
asked "shall I read them to you?", and the owner said "No, just pass over to
Claude for me." Parakeet heard "Clog"; no route claimed it, the brain sent
[ACTION: browser_task, pass over to Clog] and the browser agent searched
DuckDuckGo for "Clog" for ~47 s (actions_done=48839 ms) before JARVIS asked
whether 'Clog' was a website.

Pinned here, against the REAL monolith (every action and LLM call stubbed;
nothing is browsed or written): the live turn's route
(_utterance_route_reply -> _screen_route_reply -> core.dispatcher.
screen_route -> core.onscreen_refs.claude_note) claims the hand-off as
note_for_claude BEFORE the brain, so browser_task never runs; a nozzle clog
or a cloud backup is still the brain's.

    python -m unittest tests.monolith.test_monolith_claude_handoff_1028
"""
from __future__ import annotations

import contextlib
import io
import unittest

from core import onscreen_refs as O
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

LIVE = "No, just pass over to Clog for me."
LIVE_BRAIN = ("[intent:confirmation] On it, sir. "
              "[ACTION: browser_task, pass over to Clog]")
NOTED = "Noted for Claude, sir."


@requires_monolith
class HandOffTurnTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._stub("note_for_claude", NOTED)
        self._stub("browser_task", "The request 'pass over to Clog' is too "
                                   "ambiguous for me to act upon.")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=LIVE_BRAIN)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_route_reply_claims_the_live_hand_off(self):
        got = self._quiet(self.bc._utterance_route_reply, LIVE)
        self.assertEqual(got, f"[ACTION: note_for_claude, {O._PASS_REFERS_BACK}]")

    def test_the_live_turn_notes_it_and_never_browses(self):
        self._run(LIVE)
        self.assertEqual(self.calls["note_for_claude"], [O._PASS_REFERS_BACK])
        self.assertEqual(self.calls["browser_task"], [])
        self.llm.assert_not_called()

    def test_the_owners_other_phrasings_take_the_same_route(self):
        for said in ("No, just pass over to Claude for me.",
                     "just pass it over to Claude for me",
                     "hand this to Claude please",
                     "send that over to Claude now",
                     "pass it on to Claude for me please"):
            with self.subTest(said=said):
                self.assertEqual(
                    self._quiet(self.bc._utterance_route_reply, said),
                    f"[ACTION: note_for_claude, {O._PASS_REFERS_BACK}]")

    def test_a_question_about_claude_reaches_the_brain(self):
        # Review 2026-10-28: this wrote note_for_claude("What gets") and the
        # LLM never ran.
        self._run("What gets sent to Claude?")
        self.assertEqual(self.calls["note_for_claude"], [])
        self.llm.assert_called()

    def test_printer_and_cloud_talk_stays_with_the_brain(self):
        for said in ("the nozzle is about to clog", "clear the clog",
                     "is it clogged", "send it to the cloud",
                     "upload that to the cloud", "back it up to the cloud"):
            with self.subTest(said=said):
                got = self._quiet(self.bc._utterance_route_reply, said) or ""
                self.assertNotIn("note_for_claude", got)


if __name__ == "__main__":
    unittest.main()
