"""Monolith wiring for the 2026-10-01 diagnostic-batch fixes (claude/diag-fixes).

Each class pins one defect from the 09-05 heavy live diagnostic / the 10-01
audit, at the monolith level (the light-tier halves live in tests/test_*.py).
Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_diag_fixes
"""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _ShortcutBase(MonolithGlobalsTestCase):
    """_run_voice_shortcuts with every earlier shortcut standing aside and
    the LLM booby-trapped (the test_monolith_fast_paths pattern)."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "set_state")
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self.llm = self._p(bc, "_call_llm",
                           side_effect=AssertionError("LLM called"))
        self._p(bc, "maybe_replay_last_action", return_value=None)
        router = types.ModuleType("core.mode_router")
        router.maybe_handle_mode_toggle = lambda _t: None
        router.controlled_dispatch = lambda _t, _a: None
        router.is_in_controlled_mode = lambda: False
        disp = types.ModuleType("core.dispatcher")
        disp.resolve_and_dispatch = lambda _t, _a: None
        voice = types.SimpleNamespace(maybe_switch_backend=lambda _t: None)
        patcher = mock.patch.dict(bc.sys.modules, {
            "core.mode_router": router, "core.dispatcher": disp,
            "skill_custom_voice": voice})
        patcher.start()
        self.addCleanup(patcher.stop)
        bc.conversation_history.clear()

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _turn(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            handled = self.bc._run_voice_shortcuts(text)
        return handled, buf.getvalue()


# ── item 6: "are you ok" / "run a system check" run the real self-check ────
class SelfCheckShortcutTests(_ShortcutBase):
    SUMMARY = ("Sir, nothing is reporting a failure, but I am not able to call "
               "the system nominal: the microphone did not run. "
               "(2.1s sweep, 20 probes.)")

    def setUp(self):
        super().setUp()
        self.diag = mock.Mock(return_value=self.SUMMARY)
        patcher = mock.patch.dict(self.bc.ACTIONS, {"are_you_ok": self.diag})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_are_you_ok_runs_the_self_diagnostic_and_speaks_it(self):
        for text in ("are you ok", "Jarvis, are you okay?", "run a system check"):
            with self.subTest(text=text):
                self.diag.reset_mock()
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, f"{text!r} fell through to the LLM")
                self.diag.assert_called_once()
                self.assertEqual(self.spoken[-1], self.SUMMARY)
                self.assertIn("[fast-path] self-check", log)
                self.assertIn(f"JARVIS: {self.SUMMARY}", log)
        self.llm.assert_not_called()

    def test_the_turn_is_recorded(self):
        self._turn("are you ok")
        self.assertEqual(self.bc.conversation_history[-2:], [
            {"role": "user", "content": "are you ok"},
            {"role": "assistant", "content": self.SUMMARY}])

    def test_a_diagnostic_that_raises_is_reported_honestly(self):
        self.diag.side_effect = RuntimeError("probe table missing")
        handled, _log = self._turn("are you ok")
        self.assertTrue(handled)
        self.assertIn("self-check", self.spoken[-1].lower())
        self.assertNotIn("nominal", self.spoken[-1].lower())

    def test_no_diagnostic_skill_falls_through(self):
        # Nothing to run: the shortcut stands aside rather than invent a pass.
        with mock.patch.dict(self.bc.ACTIONS, clear=False) as acts:
            for name in ("are_you_ok", "run_diagnostic", "self_diagnostic",
                         "system_check"):
                acts.pop(name, None)
            with mock.patch.object(self.bc, "_run_fast_paths",
                                   return_value=False):
                handled, _log = self._turn("are you ok")
        self.assertFalse(handled)

    def test_disabled_with_the_fast_paths(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        handled, _log = self._turn("are you ok")
        self.assertFalse(handled)
        self.diag.assert_not_called()

    def test_ordinary_turns_never_run_it(self):
        with mock.patch.object(self.bc, "_run_fast_paths", return_value=False):
            for text in ("are you ok with that", "check the system",
                         "what's the weather"):
                with self.subTest(text=text):
                    handled, _ = self._turn(text)
                    self.assertFalse(handled)
        self.diag.assert_not_called()



# ── item 4: list_timers never makes things up ───────────────────────────────
class TimerListShortcutTests(_ShortcutBase):
    """A whole-utterance timer question is answered from the timer store
    with no LLM, so the model never gets the chance to invent a timer."""

    LINE = "One timer is running, sir: number 1, 'tea', in 4 minutes."

    def setUp(self):
        super().setUp()
        self.lister = mock.Mock(return_value=self.LINE)
        patcher = mock.patch.dict(self.bc.ACTIONS, {"list_timers": self.lister})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_timer_questions_are_answered_from_the_store(self):
        for text in ("what timers do I have", "list my timers",
                     "Jarvis, any timers running?", "do I have any reminders",
                     "how much time is left on my timer"):
            with self.subTest(text=text):
                self.lister.reset_mock()
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, f"{text!r} reached the LLM")
                self.lister.assert_called_once()
                self.assertEqual(self.spoken, [self.LINE])
                self.assertIn("[fast-path] timers", log)
        self.llm.assert_not_called()

    def test_setting_or_cancelling_a_timer_is_not_a_listing(self):
        with mock.patch.object(self.bc, "_run_fast_paths", return_value=False):
            for text in ("set a timer for 5 minutes", "cancel my timer",
                         "what time is it"):
                with self.subTest(text=text):
                    handled, _ = self._turn(text)
                    self.assertFalse(handled)
        self.lister.assert_not_called()

    def test_no_timer_skill_falls_through(self):
        with mock.patch.dict(self.bc.ACTIONS) as acts:
            acts.pop("list_timers", None)
            with mock.patch.object(self.bc, "_run_fast_paths",
                                   return_value=False):
                handled, _ = self._turn("what timers do I have")
        self.assertFalse(handled)


@requires_monolith
class TimerClaimTests(MonolithGlobalsTestCase):
    """How the invention happened: the local model answered "what timers do
    I have" in its OWN words — with no token at all (no guard knew a timer
    claim), or as prose in front of [ACTION: list_timers] (answer-first
    drops only pure acknowledgements, so the invented sentence was spoken
    before the real list). A timer-state claim now injects list_timers, and
    whenever list_timers runs the model's own timer claims are dropped: the
    store's line is the only answer voiced."""

    REAL = "No timers are running, sir."

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.lister = mock.Mock(return_value=self.REAL)
        acts = dict(bc.ACTIONS)
        acts["list_timers"] = self.lister
        for name, value in (("ACTIONS", acts),
                            ("_speak", lambda *a, **k: None),
                            ("_write_hud_state", lambda **k: None),
                            ("record_session_action", lambda *a, **k: None),
                            ("record_action_history", lambda *a, **k: None),
                            ("_cmd_autocorrect", None),
                            ("PC_CONTROL_ENABLED", True),
                            ("_needs_confirmation", lambda n, a: False),
                            ("_jarvis_pushback", lambda n, a: None)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _run(self, reply):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = self.bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()

    def test_invented_prose_next_to_the_token_is_dropped(self):
        cleaned, results, log = self._run(
            "One moment, sir. You have two timers running, sir: tea in 4 "
            "minutes and the laundry in 20. [ACTION: list_timers]")
        self.lister.assert_called_once()
        self.assertEqual([r[0] for r in results], ["list_timers"])
        self.assertNotIn("two timers", cleaned)
        self.assertNotIn("laundry", cleaned)
        self.assertIn("One moment, sir.", cleaned)
        self.assertIn("[timers]", log)

    def test_a_tokenless_timer_claim_runs_the_real_list(self):
        for reply in ("You have a 5-minute tea timer running, sir.",
                      "There are no timers running right now, sir.",
                      "Your tea timer has 4 minutes left, sir.",
                      "It's 5 minutes left on your timer, sir."):
            with self.subTest(reply=reply):
                self.lister.reset_mock()
                verdict = self.bc._detect_preemptive_hallucination(reply)
                self.assertEqual(verdict[:2], ("inject", "list_timers"))
                cleaned, _results, _log = self._run(reply)
                self.lister.assert_called_once()
                self.assertEqual(cleaned, "",
                                 "the invented timer line was still voiced")

    def test_timer_prose_that_claims_no_state_is_left_alone(self):
        for reply in ("Timer set for 5 minutes, sir.",
                      "I'll remind you in 5 minutes, sir.",
                      "You have 3 unread emails, sir.",
                      "Your timer is up, sir.",
                      "Setting a timer is easy, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(self.bc._TIMER_STATE_CLAIM_RE.search(reply))

    def test_other_actions_keep_their_prose(self):
        other = mock.Mock(return_value="done")
        self.bc.ACTIONS["noop_x"] = other
        cleaned, _results, _log = self._run(
            "You have two timers running, sir. [ACTION: noop_x]")
        self.assertIn("two timers", cleaned)


if __name__ == "__main__":
    unittest.main()
