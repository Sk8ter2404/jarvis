"""Monolith wiring for the 2026-10-09 invented-volume turn.

Live 17:53: "Jarvis, why can't I hear my YouTube video?" -> JARVIS spoke
"I'm afraid the volume is currently set to 20%, sir. [ACTION: system_pulse]".
Nothing read the volume. Pinned here, with the REAL monolith and every action,
LLM call and speech stubbed:

  * the question is answered by audio_check's reading, with no LLM;
  * the live reply's invented number is never flushed early, and is dropped
    from the spoken reply with a synthetic _ungrounded_reading result;
  * a reading an action really returned this turn is kept and spoken.

    python -m unittest tests.monolith.test_monolith_grounded_audio
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

LIVE_USER = "Jarvis, why can't I hear my YouTube video?"
LIVE_REPLY = ("I'm afraid the volume is currently set to 20%, sir. "
              "[ACTION: system_pulse]")
PULSE = ("CPU 20% (8 cores) | GPU 31% at 52°C | RAM 58% of 32 GB | "
         "12 windows open")
AUDIO_READ = ("The system volume is at 20 percent, not muted, and sound is "
              "going to Speakers, sir.")


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken: list[str] = []
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
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
        self.calls: dict[str, list[str]] = {}
        self._actions = dict(bc.ACTIONS)
        self._p(bc, "ACTIONS", self._actions)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _stub(self, name, result):
        self.calls[name] = []

        def fn(arg=""):
            self.calls[name].append(arg)
            return result

        self._actions[name] = fn

    def _quiet(self, fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    def _in_turn(self, user_text, fn, *a, **k):
        bc = self.bc
        prev = bc._begin_turn_grounding(user_text)
        try:
            return self._quiet(fn, *a, **k)
        finally:
            bc._end_turn_grounding(prev)


class AudioCheckRouteTests(_Base):
    def test_audio_check_is_registered(self):
        self.assertIn("audio_check", self.bc.ACTIONS)

    def test_live_question_is_answered_by_the_reading(self):
        self._stub("audio_check", AUDIO_READ)
        self._p(self.bc, "FAST_PATHS_ENABLED", True)
        handled = self._quiet(self.bc._run_audio_check_shortcut, LIVE_USER)
        self.assertTrue(handled)
        self.assertEqual(self.calls["audio_check"], [LIVE_USER])
        self.assertEqual(self.spoken, [AUDIO_READ])

    def test_a_command_is_not_routed(self):
        self._stub("audio_check", AUDIO_READ)
        self._p(self.bc, "FAST_PATHS_ENABLED", True)
        self.assertFalse(self._quiet(self.bc._run_audio_check_shortcut,
                                     "turn the volume up"))
        self.assertEqual(self.calls["audio_check"], [])


class InventedReadingTests(_Base):
    def test_live_sentence_is_never_flushed_early(self):
        buf = self.bc._SentenceFlushBuffer(speak_fn=lambda t: None)
        stopped = self._in_turn(
            LIVE_USER, buf._gate_stops,
            "I'm afraid the volume is currently set to 20%, sir.")
        self.assertTrue(stopped)

    def test_live_reply_drops_the_invented_volume(self):
        self._stub("system_pulse", PULSE)
        cleaned, results = self._in_turn(
            LIVE_USER, self.bc.parse_and_run_actions, LIVE_REPLY)
        self.assertNotIn("20%", cleaned)
        names = [n for n, _r, _i in results]
        self.assertIn("system_pulse", names)
        self.assertIn("_ungrounded_reading", names)
        warn = dict((n, r) for n, r, _i in results)["_ungrounded_reading"]
        self.assertIn("audio_check", warn)

    def test_outside_an_owner_turn_nothing_is_dropped(self):
        # A proactive remark has no follow-up round and may quote a sensor
        # line from its own prompt: it is left as it was.
        self._stub("system_pulse", PULSE)
        cleaned, results = self._quiet(self.bc.parse_and_run_actions,
                                       LIVE_REPLY)
        self.assertIn("20%", cleaned)
        self.assertNotIn("_ungrounded_reading",
                         [n for n, _r, _i in results])

    def test_a_real_reading_is_kept(self):
        self._stub("audio_check", AUDIO_READ)
        reply = ("[ACTION: audio_check, why can't I hear] The volume is at "
                 "20 percent, sir.")
        cleaned, results = self._in_turn(
            LIVE_USER, self.bc.parse_and_run_actions, reply)
        self.assertIn("20 percent", cleaned)
        self.assertNotIn("_ungrounded_reading",
                         [n for n, _r, _i in results])

    def test_followup_quoting_an_earlier_result_is_kept(self):
        bc = self.bc
        prev = bc._begin_turn_grounding(LIVE_USER)
        try:
            bc._note_turn_action_ran("audio_check", AUDIO_READ)
            cleaned, results = self._quiet(
                bc.parse_and_run_actions,
                "The system volume is at 20 percent, sir, and nothing is "
                "muted.")
            buf = bc._SentenceFlushBuffer(speak_fn=lambda t: None)
            stopped = self._quiet(buf._gate_stops,
                                  "The system volume is at 20 percent, sir.")
        finally:
            bc._end_turn_grounding(prev)
        self.assertIn("20 percent", cleaned)
        self.assertEqual(results, [])
        self.assertFalse(stopped)


if __name__ == "__main__":
    unittest.main()
