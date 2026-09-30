"""Monolith wiring for the deterministic fast paths (core/fast_paths.py,
core/date_math.py; 2026-09-29).

The fast path answers relative-date questions, "what did I just ask you" and
"what's my name" right before the LLM dispatch, inside _run_voice_shortcuts,
which main() runs for voice AND typed / injected turns. These tests show it
answers without calling the LLM, records the turn in conversation_history like
a normal turn, never arms the processing filler, logs one "[fast-path] <kind>"
line, and stands down when FAST_PATHS_ENABLED is False. The clock is frozen
through _fast_path_now (CI runs in UTC). Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_fast_paths
"""
from __future__ import annotations

import contextlib
import datetime as dt
import inspect
import io
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

TUE = dt.datetime(2026, 9, 29, 14, 56)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._speak = self._p(bc, "_speak")
        self._p(bc, "set_state")
        self._p(bc, "_fast_path_now", return_value=TUE)
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self._p(bc, "USER_NAME", "Alex")
        # The LLM must never be reached on a fast-path turn.
        self.llm = self._p(bc, "_call_llm",
                           side_effect=AssertionError("LLM called"))
        self.get_resp = self._p(bc, "get_response_with_animation",
                                side_effect=AssertionError("LLM called"))
        self.arm = self._p(bc._processing_filler, "arm")
        # Every earlier shortcut stands aside (as in test_monolith_sec7).
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
        """What main() does with an accepted utterance: the shortcuts first,
        the LLM dispatch only when none handled it."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            handled = self.bc._run_voice_shortcuts(text)
        return handled, buf.getvalue()


class FastPathAnswersTests(_Base):
    def test_date_tomorrow_is_answered_without_the_llm(self):
        handled, log = self._turn("what's the date tomorrow")
        self.assertTrue(handled)
        reply = "Tomorrow is Wednesday, September 30, 2026, sir."
        self._speak.assert_called_once_with(reply)
        self.llm.assert_not_called()
        self.get_resp.assert_not_called()
        self.assertIn("[fast-path] date", log)
        # One line, the kind only.
        self.assertEqual(log.count("[fast-path]"), 1)

    def test_the_reply_is_in_the_transcript_log(self):
        # Live 2026-09-29: fast-path replies were spoken but never printed as
        # a "JARVIS:" line, so the session log (and the run-jarvis driver)
        # showed no answer at all for them.
        _handled, log = self._turn("what's the date tomorrow")
        self.assertIn("JARVIS: Tomorrow is Wednesday, September 30, 2026, sir.", log)

    def test_days_until_christmas_and_friday(self):
        for text, reply in (
                ("how many days until Christmas",
                 "Christmas is 87 days away, on Friday, December 25, sir."),
                ("how long until Friday",
                 "Friday is 3 days away, on October 2, sir.")):
            with self.subTest(text=text):
                self._speak.reset_mock()
                handled, log = self._turn(text)
                self.assertTrue(handled)
                self._speak.assert_called_once_with(reply)
                self.assertIn("[fast-path] days-until", log)
        self.llm.assert_not_called()

    def test_turn_is_recorded_so_follow_ups_work(self):
        self.bc.conversation_history.extend([
            {"role": "user", "content": "good afternoon"},
            {"role": "assistant", "content": "Good afternoon, sir."}])
        self._turn("what's the date tomorrow")
        self.assertEqual(self.bc.conversation_history[-2:], [
            {"role": "user", "content": "what's the date tomorrow"},
            {"role": "assistant",
             "content": "Tomorrow is Wednesday, September 30, 2026, sir."}])

    def test_the_processing_filler_is_never_armed(self):
        # Handled before _run_llm_dispatch, the only place that arms it
        # ("Processing, sir." on a voice turn).
        handled, _ = self._turn("how many days until Christmas")
        self.assertTrue(handled)
        self.arm.assert_not_called()
        self._speak.assert_called_once()

    def test_last_utterance_excludes_the_current_one(self):
        self.bc.conversation_history.extend([
            {"role": "user", "content": "Jarvis, open the project notes."},
            {"role": "assistant", "content": "Opening them now, sir."}])
        handled, log = self._turn("what did I just ask you")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            'You asked me: "open the project notes", sir.')
        self.assertIn("[fast-path] last-utterance", log)
        # Asked again: still the real question, never the recall question.
        self._speak.reset_mock()
        self._turn("what did I just ask you")
        self._speak.assert_called_once_with(
            'You asked me: "open the project notes", sir.')
        self.llm.assert_not_called()

    def test_last_utterance_with_nothing_earlier_says_so(self):
        handled, _ = self._turn("what was my last question")
        self.assertTrue(handled)
        self._speak.assert_called_once_with(
            "There's no earlier question from you in this conversation, "
            "sir.")

    def test_owner_name_comes_from_user_name(self):
        handled, log = self._turn("what's my name")
        self.assertTrue(handled)
        self._speak.assert_called_once_with("Your name is Alex, sir.")
        self.assertIn("[fast-path] owner-name", log)
        self.llm.assert_not_called()


class FastPathFallThroughTests(_Base):
    def test_disabled_flag_falls_through_to_the_llm(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        for text in ("what's the date tomorrow", "what's my name",
                     "what did I just ask you"):
            with self.subTest(text=text):
                handled, log = self._turn(text)
                self.assertFalse(handled)
                self.assertNotIn("[fast-path]", log)
        self._speak.assert_not_called()
        self.assertEqual(self.bc.conversation_history, [])

    def test_no_configured_name_falls_through(self):
        self._p(self.bc, "USER_NAME", "")
        handled, _ = self._turn("what's my name")
        self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_identity_questions_stay_with_the_camera_route(self):
        for text in ("who am I", "who is this", "who am I looking at",
                     "do you recognize me"):
            with self.subTest(text=text):
                handled, _ = self._turn(text)
                self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_commands_fall_through(self):
        for text in ("remind me tomorrow to call the office",
                     "what's the weather tomorrow", "play music until friday",
                     "pause until tomorrow", "wake me up tomorrow"):
            with self.subTest(text=text):
                handled, _ = self._turn(text)
                self.assertFalse(handled)
        self._speak.assert_not_called()

    def test_a_matcher_error_falls_through(self):
        self._p(self.bc._fast_paths, "match",
                side_effect=RuntimeError("boom"))
        handled, log = self._turn("what's the date tomorrow")
        self.assertFalse(handled)
        self.assertIn("[fast-path] failed", log)
        self._speak.assert_not_called()

    def test_earlier_shortcuts_keep_precedence(self):
        self._p(self.bc, "maybe_replay_last_action",
                return_value="Done again, sir.")
        handled, log = self._turn("what's the date tomorrow")
        self.assertTrue(handled)
        self._speak.assert_called_once_with("Done again, sir.")
        self.assertNotIn("[fast-path]", log)


class FastPathWiringTests(_Base):
    """main() cannot run in a test (see test_monolith_turn_timing)."""

    def test_fast_path_is_the_last_shortcut_before_the_llm(self):
        src = inspect.getsource(self.bc._run_voice_shortcuts)
        self.assertTrue(src.rstrip().endswith("return _run_fast_paths(text)"))
        main = inspect.getsource(self.bc.main)
        self.assertLess(main.index("if _run_voice_shortcuts(text):"),
                        main.index("reply = _run_llm_dispatch(text"))

    def test_local_guard_keeps_whats_my_name_off_the_camera(self):
        guard = self.bc._LOCAL_NEVER_GUESS_GUARD
        self.assertIn("\"What's my name\" is NOT a camera look", guard)
        self.assertIn("Never guess a name.", guard)


if __name__ == "__main__":
    unittest.main()
