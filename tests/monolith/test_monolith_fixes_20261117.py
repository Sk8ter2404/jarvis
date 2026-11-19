"""The 2026-11-17 review fixes through the REAL monolith paths
(JARVIS-Research/review_20261117_logs_hands_brain.json, logs finding 1):

  * ACTION_FIT_GAME_MODE_ENFORCE: Dell 11-17 12:46 "Jarvis is running on
    Quinn36B" -> [ACTION: game_mode_on] ran while action-fit only logged
    "would block". Now it does not run, in 'shadow' too, and JARVIS says why;
  * "open YouTube." -> [ACTION: youtube_search_direct, YouTube] (16:12:48,
    a random Short) opens the YouTube home page instead;
  * AWAY_RECAP_ROUTE: "did you log anything while I was at school today?"
    (16:02:13) drains the speech queue with no brain call.

Every action is a stub (nothing launches, nothing plays); no LLM, no audio.
Paraphrased fixtures.

    python -m unittest tests.monolith.test_monolith_fixes_20261117
"""
from __future__ import annotations

import time
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

DELL_LINE = "Jarvis is running on Quinn36B"


@requires_monolith
class GameModeDispatchTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "ACTION_FIT_MODE", "shadow", create=True)
        self._p(bc, "ACTION_FIT_GAME_MODE_ENFORCE", True, create=True)
        self._p(bc, "_action_history", [])
        self._stub("game_mode_on", "Game mode on, sir.")
        self._stub("system_pulse", "CPU 12 percent.")

    def _said(self):
        return " ".join(self.spoken)

    def test_the_dell_line_does_not_run_game_mode(self):
        self._dispatch(DELL_LINE, "Very good, sir. [ACTION: game_mode_on]",
                       followups=["Understood, sir."])
        self.assertEqual(self.calls["game_mode_on"], [])
        self.assertIn("skipped game mode", self._said())

    def test_game_words_still_run_it(self):
        self._dispatch("Jarvis, time to play Fortnite.",
                       "Very good, sir. [ACTION: game_mode_on]",
                       followups=["Done, sir."])
        self.assertEqual(self.calls["game_mode_on"], [""])

    def test_performance_runs_it(self):
        self._dispatch("Jarvis, boost performance.",
                       "Very good, sir. [ACTION: game_mode_on]",
                       followups=["Done, sir."])
        self.assertEqual(self.calls["game_mode_on"], [""])

    def test_flag_off_shadow_runs_it_as_before(self):
        self._p(self.bc, "ACTION_FIT_GAME_MODE_ENFORCE", False, create=True)
        self._dispatch(DELL_LINE, "Very good, sir. [ACTION: game_mode_on]",
                       followups=["Done, sir."])
        self.assertEqual(self.calls["game_mode_on"], [""])

    def test_fit_mode_off_turns_it_off_too(self):
        self._p(self.bc, "ACTION_FIT_MODE", "off", create=True)
        self._dispatch(DELL_LINE, "Very good, sir. [ACTION: game_mode_on]",
                       followups=["Done, sir."])
        self.assertEqual(self.calls["game_mode_on"], [""])

    def test_beside_another_action_the_other_still_runs(self):
        self._dispatch(DELL_LINE,
                       "[ACTION: game_mode_on] [ACTION: system_pulse]",
                       followups=["CPU is at 12 percent, sir."])
        self.assertEqual(self.calls["game_mode_on"], [])
        self.assertEqual(self.calls["system_pulse"], [""])


@requires_monolith
class YouTubeHomeDispatchTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "_UTTERANCE_ROUTES", [])
        self._p(self.bc, "ACTION_FIT_MODE", "off", create=True)
        self._stub("open_url", "opened https://www.youtube.com")
        self._stub("youtube_search_direct", "opened a random Short")

    def test_open_youtube_opens_the_home_page(self):
        self._dispatch("open YouTube.",
                       "Right away, sir. [ACTION: youtube_search_direct, YouTube]",
                       followups=["YouTube is open, sir."])
        self.assertEqual(self.calls["youtube_search_direct"], [])
        self.assertEqual(self.calls["open_url"], ["https://www.youtube.com"])

    def test_a_real_search_still_searches(self):
        self._dispatch("Jarvis, some lofi beats please.",
                       "[ACTION: youtube_search_direct, lofi beats]",
                       followups=["Playing, sir."])
        self.assertEqual(self.calls["youtube_search_direct"], ["lofi beats"])
        self.assertEqual(self.calls["open_url"], [])


@requires_monolith
class AwayRecapShortcutTests(_Base):
    ASK = "did you log anything while I was at school today?"

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "AWAY_RECAP_ROUTE", True, create=True)
        self._p(bc, "_pending_drain_spoken", [None])
        self._p(bc, "_last_away_recap", [None])
        self.appended = []
        self._p(bc, "_append_turn",
                side_effect=lambda u, r: self.appended.append((u, r)))
        self._p(bc, "_pending_queue_nonempty", return_value=False)

    def _drain_says(self, *lines, source="recap:presence"):
        def fake(*_a, **_k):
            for ln in lines:
                self.bc._speak(ln)
                self.bc._note_pending_spoken({"source": source}, ln)
            return bool(lines)
        self._p(self.bc, "_speak_pending", side_effect=fake)

    def test_the_queued_recap_is_the_answer(self):
        recap = "While you were away, sir: credit at $7.10; and memory at 89 percent."
        self._drain_says(recap)
        self.assertTrue(self._quiet(self.bc._away_recap_shortcut, self.ASK))
        self.assertEqual(self.spoken, [recap])
        self.assertEqual(self.appended, [(self.ASK, recap)])
        self.assertIsNone(self.bc._pending_drain_spoken[0])
        self.assertEqual(self.bc._last_away_recap[0][1], recap)

    def test_nothing_queued_is_said_honestly(self):
        self._drain_says()
        self.assertTrue(self._quiet(self.bc._away_recap_shortcut, self.ASK))
        from core import owner_presence as op
        self.assertEqual(self.spoken, [op.AWAY_NOTHING_WAITING_LINE])

    def test_a_recap_already_given_is_said_again(self):
        recap = "While you were away, sir: the print finished."
        self.bc._last_away_recap[0] = (time.time() - 600, recap)
        self._drain_says()
        self.assertTrue(self._quiet(self.bc._away_recap_shortcut, self.ASK))
        self.assertEqual(self.spoken, [recap])

    def test_held_lines_are_not_called_nothing(self):
        self._drain_says()
        self._p(self.bc, "_pending_queue_nonempty", return_value=True)
        self.assertTrue(self._quiet(self.bc._away_recap_shortcut, self.ASK))
        from core import owner_presence as op
        self.assertEqual(self.spoken, [op.AWAY_STILL_HELD_LINE])

    def test_other_lines_and_flag_off_fall_through(self):
        self._drain_says("x")
        self.assertFalse(self._quiet(self.bc._away_recap_shortcut,
                                     "what time is it?"))
        self._p(self.bc, "AWAY_RECAP_ROUTE", False, create=True)
        self.assertFalse(self._quiet(self.bc._away_recap_shortcut, self.ASK))
        self.assertEqual(self.spoken, [])

    def test_the_voice_shortcuts_take_it(self):
        self._drain_says("While you were away, sir: the print finished.")
        self._p(self.bc, "_alarm_ringing_shortcut", return_value=False)
        self._p(self.bc, "_browser_bg_stop_shortcut", return_value=False)
        self._p(self.bc, "_repeat_last_line_shortcut", return_value=False)
        self.assertTrue(self._quiet(self.bc._run_voice_shortcuts, self.ASK))
        self.assertEqual(self.spoken,
                         ["While you were away, sir: the print finished."])


if __name__ == "__main__":
    unittest.main()
