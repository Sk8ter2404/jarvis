"""The follow-up loop stops going in circles (2026-10-02 live).

Live 12:00:18-12:01:04, one turn: close_window failed ("no window matching
..."), see_screen, close_window failed ("could not close"), see_screen,
close_window failed ("no window matching '<title> | left'"), see_screen ...
until the follow-up loop gave up. Every failure carried a DIFFERENT argument
and result, and the repeat guard keyed on (action, result), so none of them
counted as a repeat.

The guard now keys on the action NAME: when the same action fails in a
second round of one chain the chain stops, and the existing honest close-out
("I'm afraid I couldn't finish that one, sir.") is spoken once - the attempt
that failed again was never reported, so the close-out speaks even when an
earlier round said something. The acknowledgement in front of that failed
attempt ("Certainly, sir.") is not spoken. One retry of a failed action
still runs; different actions failing once each, and a failure followed by a
success, keep the chain going; the depth cap and the loop / no-progress
guards are unchanged.

Drives the REAL _run_llm_dispatch with stub actions, a canned LLM and a
recording _speak. Made-up fixtures of the live shape; no audio, no LLM, no
real windows.

    python -m unittest tests.monolith.test_monolith_failure_circles
"""
from __future__ import annotations

import contextlib
import io

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base


@requires_monolith
class FailureCircleTests(_Base):
    def setUp(self):
        super().setUp()
        self._stub("see_screen", "Notepad is visible on the left monitor.")

    def _run(self, user_text, first, followups):
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value=first)
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bc._run_llm_dispatch(user_text)
        return buf.getvalue()

    def test_the_live_circle_stops_at_the_second_close_failure(self):
        bc = self.bc
        self._stub("close_window", "no window matching 'notepad.exe'",
                   "could not close",
                   "no window matching 'Notepad | left'")
        printed = self._run(
            "close notepad",
            "[intent:confirmation] Certainly, sir. "
            "[ACTION: close_window, notepad.exe]",
            ["[intent:bad_news] I couldn't find a window by that name, sir. "
             "[ACTION: see_screen, notepad]",
             "[intent:confirmation] Certainly, sir. "
             "[ACTION: close_window, Notepad]",
             "[intent:bad_news] Windows is being stubborn, sir. "
             "[ACTION: see_screen, notepad]",
             "[intent:confirmation] Right away, sir. "
             "[ACTION: close_window, Notepad | left]"])
        self.assertEqual(self.calls["close_window"],
                         ["notepad.exe", "Notepad"])
        self.assertEqual(self.gfr.call_count, 2)
        self.assertIn("close_window failed again", printed)
        self.assertIn("[close-out] the chain stopped (repeating failure)",
                      printed)
        self.assertEqual(self.spoken[-1], bc._CLOSE_OUT_GENERIC)
        self.assertEqual(self.spoken.count(bc._CLOSE_OUT_GENERIC), 1)
        self.assertNotIn("[intent:confirmation] Certainly, sir.", self.spoken)

    def test_one_retry_that_succeeds_keeps_going(self):
        self._stub("close_window", "could not close", "closed: Notepad")
        printed = self._run(
            "close notepad", "[ACTION: close_window, notepad]",
            ["Let me try the full title. [ACTION: close_window, Notepad]",
             "Notepad is closed, sir."])
        self.assertEqual(self.calls["close_window"], ["notepad", "Notepad"])
        self.assertNotIn("failed again", printed)
        self.assertNotIn("[close-out]", printed)

    def test_different_actions_failing_once_each_keep_going(self):
        self._stub("focus_window", "no window matching 'notes'")
        self._stub("close_window", "could not close")
        printed = self._run(
            "close my notes", "[ACTION: focus_window, notes]",
            ["[ACTION: close_window, notes]",
             "I'm afraid the notes window won't close, sir."])
        self.assertEqual(self.gfr.call_count, 2)
        self.assertNotIn("failed again", printed)
        self.assertEqual(self.spoken[-1],
                         "I'm afraid the notes window won't close, sir.")

    def test_the_same_failure_twice_still_stops(self):
        bc = self.bc
        self._stub("close_window", "could not close")
        printed = self._run(
            "close notepad", "[ACTION: close_window, notepad]",
            ["[ACTION: close_window, notepad]",
             "[ACTION: close_window, notepad]"])
        self.assertEqual(len(self.calls["close_window"]), 2)
        self.assertIn("failed again", printed)
        self.assertEqual(self.spoken, [bc._CLOSE_OUT_GENERIC])

    def test_close_out_line_speaks_for_a_repeat_stop_after_substance(self):
        bc = self.bc
        self.assertEqual(
            bc._chain_close_out_line(cut="repeating failure", rounds=2,
                                     spoke_substance=True, barged=False),
            bc._CLOSE_OUT_GENERIC)
        # The other cuts keep their rule: something real was said, no line.
        for cut in ("loop detected", "no new progress", "depth cap",
                    "ended on a promise"):
            with self.subTest(cut=cut):
                self.assertEqual(
                    bc._chain_close_out_line(cut=cut, rounds=2,
                                             spoke_substance=True,
                                             barged=False), "")
        self.assertEqual(
            bc._chain_close_out_line(cut="repeating failure", rounds=2,
                                     spoke_substance=True, barged=True), "")
