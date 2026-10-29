"""v3.0.6 fast paths, through the REAL entry point a live turn uses
(bobert_companion._run_voice_shortcuts), with the LLM booby-trapped.

Live 2026-10-28 human test (17:53-18:10):
  * 17:55:44 "How long is left on that timer?" and 18:03:33 "Do I have any
    timers or reminders running?" both went through the model (~4.3-5.1 s to
    first word) because the timer-list grammar knew neither "that timer" nor
    "timers or reminders". They are now the store's own line (list_timers ->
    describe_timers).
  * 18:02:40 "What can you do for me?" got the model's improvised list. It is
    now a fixed overview of the registered skills (core.fast_paths
    .capability_reply), each phrase kept only while an action backs it.

    python -m unittest tests.monolith.test_monolith_v306_fast_paths
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_diag_fixes import _ShortcutBase


@requires_monolith
class LiveTimerQuestionsReadTheStore(_ShortcutBase):
    LINE = "One timer is running, sir: number 1, in 2 minutes 48 seconds."

    def setUp(self):
        super().setUp()
        self.lister = mock.Mock(return_value=self.LINE)
        patcher = mock.patch.dict(self.bc.ACTIONS, {"list_timers": self.lister})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_live_timer_questions_never_reach_the_model(self):
        for text in ("How long is left on that timer?",
                     "Do I have any timers or reminders running?",
                     "Jarvis, any timers or reminders?",
                     "what's left on this timer"):
            with self.subTest(text=text):
                self.lister.reset_mock()
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, f"{text!r} reached the LLM")
                self.lister.assert_called_once()
                self.assertEqual(self.spoken, [self.LINE])
                self.assertIn("[fast-path] timers", log)
        self.llm.assert_not_called()


@requires_monolith
class LiveCapabilityQuestion(_ShortcutBase):
    def _acts(self, names):
        acts = {n: mock.Mock(return_value="") for n in names}
        patcher = mock.patch.dict(self.bc.ACTIONS, acts)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_what_can_you_do_for_me_is_a_fixed_true_overview(self):
        self._acts(("set_timer", "weather_briefing", "check_print",
                    "note_for_claude"))
        handled, log = self._turn("What can you do for me?")
        self.assertTrue(handled)
        self.assertIn("[fast-path] capabilities", log)
        self.assertEqual(len(self.spoken), 1)
        said = self.spoken[0]
        for phrase in ("set timers and reminders", "tell you the weather",
                       "keep an eye on the 3D printer", "pass a note to Claude"):
            self.assertIn(phrase, said)
        # The live improvised claims are never voiced.
        self.assertNotIn("manage your schedule", said)
        self.assertNotIn("proficient", said)
        self.llm.assert_not_called()

    def test_phrase_without_a_registered_action_is_left_out(self):
        acts = dict(self.bc.ACTIONS)
        for n in ("check_print",):
            acts.pop(n, None)
        acts["set_timer"] = mock.Mock()
        with mock.patch.object(self.bc, "ACTIONS", acts):
            handled, _ = self._turn("what can you do")
        self.assertTrue(handled)
        self.assertNotIn("3D printer", self.spoken[0])
        self.assertIn("set timers and reminders", self.spoken[0])

    def test_review_f9_no_home_server_or_cameras_claimed_unconfigured(self):
        # server_status is always registered; with SERVER_WATCH_HOST "" it only
        # says "I'm not watching a server yet". see_screen alone is no camera.
        import core.config as cfg
        self._acts(("system_pulse", "see_screen", "draft_reply",
                    "server_status"))
        acts = {n: self.bc.ACTIONS[n] for n in
                ("system_pulse", "see_screen", "draft_reply", "server_status")}
        with mock.patch.object(self.bc, "ACTIONS", acts),              mock.patch.object(cfg, "SERVER_WATCH_HOST", "", create=True):
            handled, log = self._turn("What can you do for me?")
        self.assertTrue(handled)
        said = self.spoken[0]
        self.assertIn("check this PC", said)
        self.assertNotIn("server", said)
        self.assertNotIn("camera", said)
        self.llm.assert_not_called()

    def test_requests_fall_through(self):
        for text in ("what can you do about the noise",
                     "what can you do with this file"):
            with self.subTest(text=text):
                _, log = self._turn(text)
                self.assertNotIn("[fast-path] capabilities", log)


if __name__ == "__main__":
    unittest.main()
