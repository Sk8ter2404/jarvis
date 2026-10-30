"""v3.0.9 C: "resume the movie" never reaches play_music (monolith wiring).

Live 2026-10-29 19:12 "Jarvis Resume movie." while a movie sat paused in
Chrome: the brain wrote [ACTION: play_music, resume], Apple Music opened a
song called "Resume", 45 s went by and the movie stayed paused.

Pinned here, against the REAL monolith (every action and LLM call stubbed;
nothing is played, paused or browsed): the live turn's route
(_utterance_route_reply -> _video_transport_route_reply -> core.dispatcher.
video_transport_route) claims it as resume_music BEFORE the brain; with
PLAY_MUSIC_TRANSPORT_WORDS off no route claims it (the old path); the local
cheat sheet carries the one-line hint only with the flag on.

    python tools/run_tests.py monolith.test_monolith_v309_transport_route
"""
from __future__ import annotations

import contextlib
import io
import unittest

import core.config as cfg
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

LIVE = "Jarvis Resume movie."
LIVE_BRAIN = "As you wish, sir. [ACTION: play_music, resume]"
HINT = "never play_music"


@requires_monolith
class VideoTransportRouteTurnTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", True, create=True)
        self._stub("resume_music", "Resumed Chrome, sir.")
        self._stub("pause_music", "Paused Chrome, sir.")
        self._stub("play_music", "opened 'Resume' on Apple Music")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=LIVE_BRAIN)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_route_reply_claims_the_live_utterance(self):
        self.assertEqual(self._quiet(self.bc._utterance_route_reply, LIVE),
                         "[ACTION: resume_music]")

    def test_the_live_turn_resumes_and_never_asks_the_brain(self):
        self._run(LIVE)
        self.assertEqual(self.calls["resume_music"], [""])
        self.assertEqual(self.calls["play_music"], [])
        self.llm.assert_not_called()

    def test_other_phrasings_take_the_same_route(self):
        for said, tok in (("resume the video", "[ACTION: resume_music]"),
                          ("Jarvis, continue the show.",
                           "[ACTION: resume_music]"),
                          ("unpause the episode please",
                           "[ACTION: resume_music]"),
                          ("Jarvis Pause movie.", "[ACTION: pause_music]")):
            with self.subTest(said=said):
                self.assertEqual(
                    self._quiet(self.bc._utterance_route_reply, said), tok)

    def test_music_and_titles_stay_with_the_brain(self):
        for said in ("resume", "resume the music", "play the movie Up",
                     "stop the movie", "what movie is this"):
            with self.subTest(said=said):
                got = self._quiet(self.bc._utterance_route_reply, said) or ""
                self.assertNotIn("resume_music", got)
                self.assertNotIn("pause_music", got)

    def test_flag_off_no_route(self):
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", False)
        self.assertIsNone(
            self._quiet(self.bc._video_transport_route_reply, LIVE))
        got = self._quiet(self.bc._utterance_route_reply, LIVE) or ""
        self.assertNotIn("resume_music", got)

    def test_unregistered_action_no_route(self):
        del self._actions["resume_music"]
        self.assertIsNone(
            self._quiet(self.bc._video_transport_route_reply, LIVE))

    def test_flag_is_in_the_turn_flag_list(self):
        names = [v for k, v in vars(self.bc).items()
                 if isinstance(v, tuple) and "NOTE_CLAIM_CHECK" in v]
        self.assertTrue(any("PLAY_MUSIC_TRANSPORT_WORDS" in t for t in names))


@requires_monolith
class CheatSheetHintTests(_Base):
    def _sheet(self):
        bc = self.bc
        self._p(bc, "_LOCAL_CHEATSHEET_CACHE", [None])
        return bc._local_cheatsheet()

    def test_hint_with_the_flag_on(self):
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", True, create=True)
        sheet = self._sheet()
        line = next(ln for ln in sheet.splitlines() if HINT in ln)
        self.assertIn("resume_music / pause_music / next_song", line)
        # Right under the play_music line.
        lines = sheet.splitlines()
        i = lines.index(line)
        self.assertIn("[ACTION: play_music,", lines[i - 1])

    def test_flag_off_is_the_old_sheet(self):
        self._p(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", False)
        self.assertNotIn(HINT, self._sheet())
        self.assertEqual(self.bc._play_music_transport_hint(), "")


if __name__ == "__main__":
    unittest.main()
