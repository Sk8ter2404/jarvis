"""A TERMINAL failure is spoken once and ends the chain (2026-10-02 live).

Live 12:00:18-12:01:04: close_window on Task Manager (an elevated window a
normal-integrity JARVIS cannot close) failed, and every failure started
another follow-up round - see_screen, close_window, see_screen ... - with
different arguments each time until the depth cap. The owner never heard
why.

close_window now returns an honest line behind
core.failure_markers.TERMINAL_FAILURE_PREFIX (tests/test_close_window_
elevated.py). The dispatcher speaks that line verbatim (the same path as a
SPEAK_RESULT_VERBATIM_ACTIONS read-out, whatever the action), drops the
acknowledgement in front of it, and stops the follow-up loop: no LLM round
can retry what Windows refused.

Drives the REAL _run_llm_dispatch with stub actions, a canned LLM and a
recording _speak. No audio, no LLM, no real windows.

    python -m unittest tests.monolith.test_monolith_terminal_failure
"""
from __future__ import annotations

from core import failure_markers as fm
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

_LINE = ("Task Manager runs as administrator, sir; Windows won't let me "
         "close it from here.")
_TERMINAL = fm.TERMINAL_FAILURE_PREFIX + _LINE


@requires_monolith
class TerminalFailureTests(_Base):
    def setUp(self):
        super().setUp()
        self._stub("close_window", _TERMINAL)
        self._stub("see_screen", "Task Manager is on the left monitor.")
        bc = self.bc
        n0 = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(n0, None)))

    def _dispatch(self, user_text, first_reply, followups=()):
        """As the base, but the canned LLM records the turn in
        conversation_history the way _call_llm does."""
        bc = self.bc

        def fake_llm(text):
            bc.conversation_history.append({"role": "user", "content": text})
            bc.conversation_history.append({"role": "assistant",
                                            "content": first_reply})
            return first_reply

        self._p(bc, "get_response_with_animation", side_effect=fake_llm)
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        return self._quiet(bc._run_llm_dispatch, user_text)

    def test_the_line_is_spoken_and_no_follow_up_round_runs(self):
        self._dispatch("close task manager",
                       "[intent:confirmation] Certainly, sir. "
                       "[ACTION: close_window, Task Manager]",
                       ["[ACTION: see_screen, task manager]",
                        "[ACTION: close_window, Task Manager | left]"])
        self.gfr.assert_not_called()
        self.assertEqual(self.spoken, [_LINE])
        self.assertEqual(self.calls["close_window"], ["Task Manager"])
        self.assertEqual(self.calls["see_screen"], [])
        self.assertEqual(self.bc.conversation_history[-1],
                         {"role": "assistant", "content": _LINE})

    def test_a_terminal_result_in_a_follow_up_round_ends_the_chain(self):
        # The live opening: an unverified claim (never spoken), then the
        # follow-up round's real action, which Windows refuses.
        self._dispatch("Jarvis closed the task manager app.",
                       "[intent:confirmation] Very good, sir. The task "
                       "manager app has been closed.",
                       ["[intent:confirmation] Right away, sir. "
                        "[ACTION: close_window, taskmgr.exe]",
                        "[ACTION: see_screen, task manager]",
                        "[ACTION: close_window, Task Manager]"])
        self.assertEqual(self.gfr.call_count, 1)
        self.assertEqual(self.calls["close_window"], ["taskmgr.exe"])
        self.assertEqual(self.calls["see_screen"], [])
        self.assertEqual(self.spoken, [_LINE])
        for s in self.spoken:
            self.assertNotIn("has been closed", s)

    def test_verbatim_text_is_the_line_for_any_action(self):
        bc = self.bc
        self.assertEqual(bc._verbatim_result_text("close_window", _TERMINAL),
                         _LINE)
        self.assertEqual(bc._verbatim_result_text("close_window",
                                                  "could not close"), "")
        self.assertTrue(bc._action_result_failed(_TERMINAL))

    def test_an_ordinary_failure_still_gets_its_follow_up_round(self):
        self._stub("close_window", "could not close")
        self._dispatch("close notepad", "[ACTION: close_window, notepad]",
                       ["I'm afraid Notepad wouldn't close, sir."])
        self.gfr.assert_called_once()
        self.assertEqual(self.spoken,
                         ["I'm afraid Notepad wouldn't close, sir."])
