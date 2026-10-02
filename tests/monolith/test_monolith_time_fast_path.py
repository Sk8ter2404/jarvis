"""Monolith wiring: "what time is it" is answered with no LLM (2026-10-02).

Live 10:27:33 "Jarvis, what time is it?" went through the LLM ("One moment,
sir. [ACTION: get_time]", a second round, ~4 s). The fast paths had no
grammar for the LOCAL time (core/fast_paths.local_time_answer adds it; see
tests/test_fast_paths_local_time.py for the grammar). These drive the REAL
_run_voice_shortcuts - the path main() takes for every accepted utterance,
wake word and all - with the clock frozen and the LLM wired to fail if
reached.

    python -m unittest tests.monolith.test_monolith_time_fast_path
"""
from __future__ import annotations

import datetime as dt

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_fast_paths import _Base

FRI = dt.datetime(2026, 10, 2, 10, 27)


@requires_monolith
class TimeFastPathTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_fast_path_now", return_value=FRI)

    def test_the_live_shapes_never_reach_the_llm(self):
        for text in ("Jarvis, what time is it?", "jarvis what time is it",
                     "What time is it, Jarvis?", "Jarvis what's the time"):
            with self.subTest(text=text):
                self._speak.reset_mock()
                self.bc.conversation_history.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled)
                self._speak.assert_called_once_with("It's 10:27 AM, sir.")
                self.assertIn("[fast-path] time", log)
                self.assertIn("JARVIS: It's 10:27 AM, sir.", log)
                self.llm.assert_not_called()
                self.get_resp.assert_not_called()
                self.arm.assert_not_called()
                self.assertEqual(
                    self.bc.conversation_history[-2:],
                    [{"role": "user", "content": text},
                     {"role": "assistant", "content": "It's 10:27 AM, sir."}])

    def test_a_meeting_time_still_goes_to_the_llm(self):
        handled, log = self._turn("what time is my meeting")
        self.assertFalse(handled)
        self._speak.assert_not_called()
        self.assertNotIn("[fast-path]", log)
