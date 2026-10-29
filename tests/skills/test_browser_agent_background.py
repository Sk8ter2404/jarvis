"""skills/browser_agent.py background runs (v3.0.8 F3): the pure parts.

The live turn (dispatch -> action -> verbatim line -> announce -> spoken stop)
is tests/monolith/test_monolith_v308_browser_background.py. Here: the stop
grammar, the spoken form of a result, and the one-voicing rule when a run
ends while its caller still waits inline. No browser, no loop, no network.

    python -m unittest tests.skills.test_browser_agent_background
"""
from __future__ import annotations

import concurrent.futures
import sys
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class BackgroundPureTests(unittest.TestCase):
    def setUp(self):
        mods = mock.patch.dict(sys.modules)
        mods.start()
        self.addCleanup(mods.stop)
        self.mod, self.actions = load_skill_isolated("browser_agent")

    def test_stop_grammar(self):
        for t in ("stop", "Jarvis, stop.", "E-stop", "emergency stop", "halt",
                  "never mind", "Nevermind", "forget it", "cancel that",
                  "stop the browser task", "stop browsing", "okay stop please",
                  "cancel the search"):
            with self.subTest(t=t):
                self.assertTrue(self.mod.is_stop_request(t))
        for t in ("stop the music", "cancel the timer", "stop listening",
                  "stop playing that song", "never mind the weather, what "
                  "time is it", "", None):
            with self.subTest(t=t):
                self.assertFalse(self.mod.is_stop_request(t))

    def test_spoken_result_drops_urls_and_stays_short(self):
        said = self.mod._spoken_result(
            "Best: Item A (https://a.example/x?y=1) - cheap. See "
            "www.b.example/z too.")
        self.assertNotIn("http", said)
        self.assertNotIn("www", said)
        self.assertIn("Item A - cheap.", said)
        long = ("This is one sentence of the answer. " * 40).strip()
        self.assertLessEqual(len(self.mod._spoken_result(long)),
                             self.mod._BG_ANNOUNCE_CHARS)
        self.assertTrue(self.mod._bg_announcement(
            "Browser agent failed, sir — x").startswith("Browser agent"))

    def test_a_run_ending_during_the_inline_wait_is_voiced_once(self):
        fut = concurrent.futures.Future()
        announce = mock.Mock()
        st = self.mod._state

        def _submit(task, max_steps, headless):
            st["current_fut"] = fut
            return fut

        real_result = fut.result

        def _result(timeout=None):
            # The run ends while the caller waits: the done-callback fires
            # here, inside the inline window.
            if not fut.done():
                fut.set_result("Done quickly.")
            return real_result(timeout=timeout)

        with mock.patch.object(self.mod, "_submit", side_effect=_submit), \
                mock.patch.object(self.mod, "_announce", announce), \
                mock.patch.object(fut, "result", side_effect=_result):
            out = self.mod._run_task_background("task", max_steps=3,
                                                headless=False)
        self.assertEqual(out, "Done quickly.")
        announce.assert_not_called()


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
