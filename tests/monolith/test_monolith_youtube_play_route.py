"""Monolith wiring for the built-in "play X on YouTube" route (NEW #7, 2026-10-02).

Live 21:43:17 (2026-10-01): "Jarvis plays <artist> essentials on YouTube"
went to the model, which emitted [ACTION: youtube] (the search action): the
results page opened, nothing played, and JARVIS said "Right away, sir."

_utterance_route_reply now asks core.dispatcher.youtube_play_route FIRST, so
the turn runs youtube_play and the model is never called. It is a built-in
route, not a skill's: SKILL_ROUTES_ENABLED (the skills' switch) does not turn
it off, but PC control off does (no action may run then). Synthetic titles.

    python -m unittest tests.monolith.test_monolith_youtube_play_route
"""
from __future__ import annotations

import contextlib
import io
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

LIVE = "Jarvis plays Artist Name essentials on YouTube."


@requires_monolith
class YouTubePlayRouteWiringTests(_Base):

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "SKILL_ROUTES_ENABLED", True, create=True)
        self.history: list = []
        self._p(bc, "conversation_history", self.history)
        self.play = self._stub("youtube_play", "Playing it on YouTube now.")
        self.search = self._stub("youtube", "searching YouTube for it")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value="[ACTION: youtube, Artist Name essentials]"
                                        " Right away, sir.")
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(text)
        return buf.getvalue()

    def test_route_reply_claims_the_live_sentence(self):
        with contextlib.redirect_stdout(io.StringIO()):
            got = self.bc._utterance_route_reply(LIVE)
        self.assertEqual(got, "[ACTION: youtube_play, Artist Name essentials]")

    def test_the_live_turn_plays_and_never_asks_the_model(self):
        self._run(LIVE)
        self.assertEqual(self.calls["youtube_play"], ["Artist Name essentials"])
        self.assertEqual(self.calls["youtube"], [],
                         "the SEARCH action ran - the results page opens and "
                         "nothing plays (the 21:43 failure)")
        self.llm.assert_not_called()

    def test_the_skills_switch_does_not_disable_the_built_in_route(self):
        self._p(self.bc, "SKILL_ROUTES_ENABLED", False, create=True)
        self._run(LIVE)
        self.assertEqual(self.calls["youtube_play"], ["Artist Name essentials"])
        self.llm.assert_not_called()

    def test_pc_control_off_leaves_the_turn_to_the_model(self):
        self._p(self.bc, "PC_CONTROL_ENABLED", False)
        self._run(LIVE)
        self.llm.assert_called_once()
        self.assertEqual(self.calls["youtube_play"], [])

    def test_a_search_request_still_goes_to_the_model(self):
        self._run("Jarvis, search YouTube for lofi beats")
        self.llm.assert_called_once()
        self.assertEqual(self.calls["youtube_play"], [])

    def test_no_youtube_play_action_means_no_route(self):
        self._actions.pop("youtube_play", None)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.bc._utterance_route_reply(LIVE))

    def test_a_skill_route_still_runs_after_the_built_in_one_declines(self):
        self._stub("desk_chat", "Chat finished.")
        self.assertTrue(self.bc.register_utterance_route(
            lambda t: "[ACTION: desk_chat, pizza]" if "desk" in t else None,
            "desk device"))
        self._run("talk to the desk device about pizza")
        self.assertEqual(self.calls["desk_chat"], ["pizza"])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
