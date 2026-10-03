"""Monolith wiring for "close all windows except X" (2026-10-03).

Live 17:22 the owner asked JARVIS to close every window but the Claude app.
The brain had no action for it: it ran list_windows, then minimize_window on
one of his folders, the JARVIS HUD, the JARVIS Reticle, a media player,
"Program Manager" and "Windows Input Experience" - and closed nothing.

Pinned here, against the REAL monolith (every action, LLM call, window and
Win32 read stubbed; nothing real is closed or minimized):
  * close_all_windows_except / minimize_all_windows_except are registered,
    their one-line summary is spoken verbatim, a bulk close is never replayed
    by "do that again" and never reached by an autocorrect guess;
  * the request is routed to the action without the brain
    (_utterance_route_reply -> core.dispatcher.window_keep_route);
  * more than PUSHBACK_MAX_CLOSE_WINDOWS windows to close asks first ("Close
    7 windows, sir? Say yes."), and the yes runs it and speaks its summary;
  * a confirmed TERMINAL result is spoken without its marker prefix.
The window-scope half (no JARVIS / shell windows in a lookup) is pinned in
tests/monolith/test_monolith_window_scope.py.

    python -m unittest tests.monolith.test_monolith_close_all_windows_except
"""
from __future__ import annotations

import contextlib
import io
import unittest

from core import failure_markers as fm
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

# The live request, paraphrased.
LIVE = "Jarvis, close every window except for Claude."
CLOSE = "close_all_windows_except"
MINIMIZE = "minimize_all_windows_except"
SUMMARY = "Closed 2 windows, sir; kept Claude."
# What the brain answered live, titles generic.
IMPROVISED = ("[intent:confirmation] Certainly, sir. "
              "[ACTION: minimize_window, Downloads - File Explorer] "
              "[ACTION: minimize_window, JARVIS HUD] "
              "[ACTION: minimize_window, JARVIS Reticle] "
              "[ACTION: minimize_window, Program Manager]")


@requires_monolith
class RegistrationTests(_Base):
    def test_both_actions_are_registered_to_core_actions(self):
        import core.actions as A
        self.assertIs(self.bc.ACTIONS[CLOSE], A._act_close_all_windows_except)
        self.assertIs(self.bc.ACTIONS[MINIMIZE],
                      A._act_minimize_all_windows_except)

    def test_the_summary_is_spoken_verbatim(self):
        for name in (CLOSE, MINIMIZE):
            with self.subTest(name=name):
                self.assertIn(name, self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)
                self.assertNotIn(name, self.bc.INFORMATIVE_ACTIONS)
        self.assertEqual(self.bc._verbatim_result_text(CLOSE, SUMMARY),
                         SUMMARY)

    def test_a_bulk_close_is_never_replayed_or_guessed(self):
        self.assertIn(CLOSE, self.bc._DESTRUCTIVE_REPLAY_ACTIONS)
        self.assertTrue(self.bc._autocorrect_protected(CLOSE))


@requires_monolith
class RouteTurnTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self.close_all = self._stub(CLOSE, SUMMARY)
        self._stub(MINIMIZE, "Minimized 2 windows, sir; kept Claude.")
        self._stub("minimize_window", "minimized: JARVIS HUD")
        self._stub("list_windows", "Open windows:\n  - Claude")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=IMPROVISED)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(text)

    def test_route_reply_claims_the_request(self):
        got = self._quiet(self.bc._utterance_route_reply, LIVE)
        self.assertEqual(got, f"[ACTION: {CLOSE}, Claude]")

    def test_the_live_turn_closes_without_the_brain_and_never_minimizes(self):
        self._run(LIVE)
        self.assertEqual(self.calls[CLOSE], ["Claude"])
        self.assertEqual(self.calls["minimize_window"], [])
        self.assertEqual(self.calls["list_windows"], [])
        self.llm.assert_not_called()
        self.assertIn(SUMMARY, self.spoken)

    def test_minimize_request_runs_the_minimize_variant(self):
        self._run("Jarvis, minimise everything but Claude")
        self.assertEqual(self.calls[MINIMIZE], ["Claude"])
        self.assertEqual(self.calls[CLOSE], [])
        self.llm.assert_not_called()

    def test_pc_control_off_leaves_the_turn_to_the_model(self):
        self._p(self.bc, "PC_CONTROL_ENABLED", False)
        self.assertIsNone(self._quiet(self.bc._utterance_route_reply, LIVE))

    def test_no_action_registered_means_no_route(self):
        self._actions.pop(CLOSE, None)
        self.assertIsNone(self._quiet(self.bc._utterance_route_reply, LIVE))


@requires_monolith
class PushbackTests(_Base):
    def setUp(self):
        real = self.bc._jarvis_pushback
        super().setUp()
        bc = self.bc
        self._p(bc, "_jarvis_pushback", real)
        self._p(bc, "PUSHBACK_ENABLED", True)
        self._p(bc, "PUSHBACK_MAX_CLOSE_WINDOWS", 5)
        self.closing: list = []
        self.preview = self._p(bc, "_close_all_windows_except_preview",
                               side_effect=lambda arg: list(self.closing))
        self._p(bc, "_UTTERANCE_ROUTES", [])
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_confirmation_at", [0.0])
        self.close_all = self._stub(CLOSE, "Closed 7 windows, sir; kept "
                                           "Claude.")
        self.llm = self._p(bc, "get_response_with_animation",
                           return_value=IMPROVISED)
        self._p(bc, "get_followup_response", side_effect=[None] * 8)

    def _titles(self, n):
        return [f"Window {i} - Notepad" for i in range(n)]

    def test_more_than_the_bar_asks(self):
        self.closing = self._titles(7)
        got = self.bc._jarvis_pushback(CLOSE, "Claude")
        self.assertIsNotNone(got)
        self.assertEqual(got[0], "Close 7 windows, sir? Say yes.")
        self.preview.assert_called_once_with("Claude")

    def test_at_or_under_the_bar_runs(self):
        self.closing = self._titles(5)
        self.assertIsNone(self.bc._jarvis_pushback(CLOSE, "Claude"))

    def test_unsaved_work_is_named(self):
        self.closing = self._titles(6) + ["*report.txt - Notepad"]
        phrase, _why = self.bc._jarvis_pushback(CLOSE, "Claude")
        self.assertEqual(phrase, "Close 7 windows, sir, including your "
                                 "unsaved Notepad project? Say yes.")

    def test_pushback_off_never_asks(self):
        self._p(self.bc, "PUSHBACK_ENABLED", False)
        self.closing = self._titles(9)
        self.assertIsNone(self.bc._jarvis_pushback(CLOSE, "Claude"))

    def test_minimizing_never_asks(self):
        self.closing = self._titles(9)
        self.assertIsNone(self.bc._jarvis_pushback(MINIMIZE, "Claude"))
        self.preview.assert_not_called()

    def test_the_turn_asks_and_a_yes_runs_it(self):
        self.closing = self._titles(7)
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(LIVE)
        self.assertEqual(self.calls[CLOSE], [], "closed before the yes")
        self.llm.assert_not_called()
        self.assertIn("Close 7 windows, sir? Say yes.", " ".join(self.spoken))
        self.assertEqual(list(self.bc._pending_confirmation),
                         [(CLOSE, "Claude")])
        self.spoken.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc.handle_confirmation_response(
                "Jarvis, yes."))
        self.assertEqual(self.calls[CLOSE], ["Claude"])
        self.assertEqual(self.spoken, ["Closed 7 windows, sir; kept Claude."])

    def test_a_no_closes_nothing(self):
        self.closing = self._titles(7)
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._run_llm_dispatch(LIVE)
            self.bc.handle_confirmation_response("No.")
        self.assertEqual(self.calls[CLOSE], [])


@requires_monolith
class ConfirmedTerminalLineTests(_Base):
    def test_a_confirmed_terminal_result_is_spoken_without_its_marker(self):
        line = ("Task Manager runs as administrator, sir; Windows won't let "
                "me close it from here.")
        self._stub(CLOSE, fm.TERMINAL_FAILURE_PREFIX + line)
        self._p(self.bc, "_pending_confirmation", [(CLOSE, "Claude")])
        self._p(self.bc, "_pending_confirmation_at", [0.0])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc.handle_confirmation_response("yes"))
        self.assertEqual(self.spoken, [line])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
