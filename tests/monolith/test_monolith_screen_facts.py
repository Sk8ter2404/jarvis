"""Monolith wiring for the 2026-10-27 20:41 screen turns (v3.0.1).

With the REAL monolith and every action, LLM call and speech stubbed:

  * a reply written after a see_screen read EARLIER this turn that states a
    release / date / number the read text does not carry is not spoken and
    gets a synthetic _ungrounded_screen_fact result; the live reply against
    the live page is kept (core.claim_validator.find_ungrounded_screen_fact);
  * the third see_screen / read_screen of one owner turn is refused before it
    runs, unless LOOP_GUARD_MODE is 'off' (core.loop_guard.screen_read_refusal).

    python -m unittest tests.monolith.test_monolith_screen_facts
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.test_screen_fact_grounding import (LIVE_PAGE, LIVE_REPLY,
                                              LIVE_USER, WINDOWS_ONLY)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_speak", lambda *a, **k: None)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_write_hud_state", lambda **k: None)
        self._p(bc, "record_session_action", lambda *a, **k: None)
        self._p(bc, "record_action_history", lambda *a, **k: None)
        self._p(bc, "record_action_error", lambda *a, **k: None)
        self._p(bc, "_cmd_autocorrect", None)
        self._p(bc, "_draft_preview_gate", None)
        self._p(bc, "_processing_filler", mock.Mock())
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self._p(bc, "MISSION_NARRATION_ENABLED", False)
        self._p(bc, "MID_TASK_STATUS_ENABLED", False)
        self._p(bc, "_needs_confirmation", lambda n, a: False)
        self._p(bc, "_jarvis_pushback", lambda n, a: None)
        self._p(bc, "_append_turn", lambda *a, **k: None)
        self._p(bc, "LOOP_GUARD_MODE", "shadow")
        self.reads: list[str] = []
        self._actions = dict(bc.ACTIONS)

        def see(arg=""):
            self.reads.append(arg)
            return "Read from the screen's own text:\n" + LIVE_PAGE

        self._actions["see_screen"] = see
        self._p(bc, "ACTIONS", self._actions)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _run(self, user_text, earlier, reply):
        """parse_and_run_actions(reply) inside an owner turn in which the
        (name, result) pairs of ``earlier`` already ran."""
        bc = self.bc
        prev = bc._begin_turn_grounding(user_text)
        try:
            for name, result in earlier:
                bc._note_turn_action_ran(name, result)
            with contextlib.redirect_stdout(io.StringIO()):
                return bc.parse_and_run_actions(reply)
        finally:
            bc._end_turn_grounding(prev)


class ScreenFactTests(_Base):
    def test_live_reply_after_the_live_read_is_kept(self):
        cleaned, results = self._run(
            LIVE_USER, [("web_search", "opened Google search"),
                        ("see_screen", LIVE_PAGE)], LIVE_REPLY)
        self.assertIn("released yesterday", cleaned)
        self.assertNotIn("_ungrounded_screen_fact", [n for n, _r, _i in results])

    def test_live_reply_after_a_read_without_it_is_not_spoken(self):
        cleaned, results = self._run(
            "Jarvis, is it time to watch Vision Quest yet?",
            [("see_screen", WINDOWS_ONLY)], LIVE_REPLY)
        self.assertNotIn("released", cleaned)
        self.assertNotIn("dropped", cleaned)
        synth = [(n, r) for n, r, _i in results
                 if n == "_ungrounded_screen_fact"]
        self.assertEqual(len(synth), 1)
        self.assertIn("'released'", synth[0][1])

    def test_no_screen_read_earlier_nothing_is_judged(self):
        cleaned, results = self._run(
            LIVE_USER, [("web_search", "opened Google search")], LIVE_REPLY)
        self.assertIn("released yesterday", cleaned)
        self.assertEqual(results, [])

    def test_a_retry_marker_is_a_loop_action(self):
        import inspect
        src = inspect.getsource(self.bc._run_llm_dispatch_body)
        self.assertIn('"_ungrounded_screen_fact"}', src)


class ScreenReadCapTests(_Base):
    LIVE_TURN3 = "Enough and awesome."
    REPLY = ("On it. [ACTION: see_screen, left] It seems to be a mix of "
             "synonyms, sir.")

    def test_third_read_in_the_live_turn_is_refused(self):
        cleaned, results = self._run(
            self.LIVE_TURN3, [("web_search", "opened Google search"),
                              ("see_screen", LIVE_PAGE),
                              ("see_screen", LIVE_PAGE)], self.REPLY)
        self.assertEqual(self.reads, [])
        self.assertEqual(len(results), 1)
        name, result, info = results[0]
        self.assertEqual(name, "see_screen")
        self.assertTrue(result.startswith("refused:"))
        self.assertTrue(self.bc._action_result_failed(result))

    def test_second_read_runs(self):
        self._run(self.LIVE_TURN3, [("see_screen", LIVE_PAGE)], self.REPLY)
        self.assertEqual(self.reads, ["left"])

    def test_off_switch_lets_the_third_read_run(self):
        self._p(self.bc, "LOOP_GUARD_MODE", "off")
        self._run(self.LIVE_TURN3, [("see_screen", LIVE_PAGE),
                                    ("see_screen", LIVE_PAGE)], self.REPLY)
        self.assertEqual(self.reads, ["left"])


if __name__ == "__main__":
    unittest.main()
