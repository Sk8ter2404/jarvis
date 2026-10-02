"""Audit P1-1 (2026-10-01): the ~64 s voice-loop watchdog stall.

The main-loop watchdog trips when nothing has ticked ``_heartbeat()`` for
``_MAIN_LOOP_HEARTBEAT_TIMEOUT`` seconds. Two stretches of LEGITIMATE work on
the main loop ran far longer than that with no tick:

  * transcribe()'s no-VAD retry: when Silero VAD finds nothing, the clip is
    decoded again with the VAD off - at beam 5, with faster-whisper's
    temperature fallback (up to 6 decodes, best-of-5 sampling) on whatever
    the buffer held, however long. On noise that fallback is exactly what
    runs to the end.
  * parse_and_run_actions: a chain of actions ran back to back with no tick
    between them (the follow-up round is not wrapped in the thinking
    animation that ticks during an LLM call).

Nothing here loads a Whisper model or runs a real action.
"""
from __future__ import annotations

import contextlib
import threading
import time
import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _Seg:
    def __init__(self, text):
        self.text = text
        self.no_speech_prob = 0.1
        self.avg_logprob = -0.2


@requires_monolith
class NoVadRetryIsBoundedTests(MonolithGlobalsTestCase):

    def _transcribe(self, fake_stt, seconds):
        bc = self.bc
        audio = np.zeros(int(bc.SAMPLE_RATE * seconds), dtype=np.float32)
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(bc, "_ensure_whisper"))
            st.enter_context(mock.patch.object(bc, "_stt", fake_stt))
            st.enter_context(mock.patch.object(bc, "_stt_engine", "faster_whisper"))
            st.enter_context(mock.patch.object(bc, "STT_HOTWORDS", "", create=True))
            st.enter_context(mock.patch.object(bc, "STT_REPLACEMENTS", {}, create=True))
            return bc.transcribe(audio)

    def test_first_pass_is_unchanged(self):
        """Control: the VAD pass keeps its beam and its filter."""
        info = mock.Mock(no_speech_prob=0.1)
        fake = mock.Mock()
        fake.transcribe.return_value = (iter([_Seg("hello")]), info)
        text, _conf = self._transcribe(fake, 3.0)
        self.assertEqual(text, "hello")
        first = fake.transcribe.call_args_list[0]
        self.assertTrue(first.kwargs["vad_filter"])
        self.assertEqual(first.kwargs["beam_size"], 5)
        self.assertEqual(fake.transcribe.call_count, 1)

    def test_retry_is_one_greedy_pass(self):
        """A quiet "JARVIS" the VAD rejected is still recovered - by ONE
        greedy decode: beam 1 and no temperature fallback."""
        info = mock.Mock(no_speech_prob=0.1)
        fake = mock.Mock()
        fake.transcribe.side_effect = [(iter([]), info),
                                       (iter([_Seg("jarvis")]), info)]
        text, _conf = self._transcribe(fake, 3.0)
        self.assertEqual(text, "jarvis")
        retry = fake.transcribe.call_args_list[1]
        self.assertFalse(retry.kwargs["vad_filter"])
        self.assertEqual(retry.kwargs["beam_size"], 1)
        self.assertEqual(retry.kwargs.get("temperature"), 0.0)

    def test_retry_is_skipped_for_a_long_buffer(self):
        """A long buffer the VAD found no speech in is room noise; decoding
        it again is the stall, not a rescue."""
        info = mock.Mock(no_speech_prob=0.9)
        fake = mock.Mock()
        fake.transcribe.side_effect = [(iter([]), info),
                                       (iter([_Seg("hallucinated")]), info)]
        # 20 s: twice the retry's length limit (_STT_NO_VAD_RETRY_MAX_AUDIO_S).
        text, conf = self._transcribe(fake, 20.0)
        self.assertEqual(fake.transcribe.call_count, 1)
        self.assertEqual(text, "")
        self.assertEqual(conf["avg_logprob"], -10.0)

    def test_retry_still_runs_at_the_length_limit(self):
        info = mock.Mock(no_speech_prob=0.1)
        fake = mock.Mock()
        fake.transcribe.side_effect = [(iter([]), info),
                                       (iter([_Seg("jarvis")]), info)]
        limit = getattr(self.bc, "_STT_NO_VAD_RETRY_MAX_AUDIO_S", 10.0)
        text, _conf = self._transcribe(fake, limit)
        self.assertEqual(fake.transcribe.call_count, 2)
        self.assertEqual(text, "jarvis")


@requires_monolith
class DispatchHeartbeatTests(MonolithGlobalsTestCase):

    def setUp(self):
        bc = self.bc
        self.acts = dict(bc.ACTIONS)
        for name, value in (("ACTIONS", self.acts),
                            ("_speak", lambda *a, **k: None),
                            ("_write_hud_state", lambda **k: None),
                            ("record_session_action", lambda *a, **k: None),
                            ("record_action_history", lambda *a, **k: None),
                            ("record_action_error", lambda *a, **k: None),
                            ("_cmd_autocorrect", None),
                            ("PC_CONTROL_ENABLED", True),
                            ("_needs_confirmation", lambda n, a: False),
                            ("_jarvis_pushback", lambda n, a: None),
                            ("_publish_main_loop_heartbeat",
                             lambda *a, **k: None)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.ages = []

        def _probe(_arg):
            self.ages.append(time.time() - bc._main_loop_heartbeat[0])
            return "probed"

        def _slow(_arg):
            # Stands in for an action that ran ~50 s: the beat is that old
            # when it returns.
            bc._main_loop_heartbeat[0] = time.time() - 50.0
            return "slow done"

        self.acts["hb_probe"] = _probe
        self.acts["hb_slow"] = _slow

    def test_each_action_ticks_the_watchdog_before_it_runs(self):
        bc = self.bc
        bc._main_loop_heartbeat[0] = time.time() - 100.0
        bc.parse_and_run_actions("[ACTION: hb_probe, a] [ACTION: hb_slow, b] "
                                 "[ACTION: hb_probe, c]")
        self.assertEqual(len(self.ages), 2)
        for age in self.ages:
            self.assertLess(age, 5.0, "an action started on a stale heartbeat")

    def test_a_long_last_action_does_not_leave_the_beat_stale(self):
        bc = self.bc
        bc.parse_and_run_actions("[ACTION: hb_slow, x]")
        self.assertLess(time.time() - bc._main_loop_heartbeat[0], 5.0,
                        "the reply after a long action starts on a stale beat")

    def test_a_dispatch_off_the_main_thread_never_ticks(self):
        """The beat means "the MAIN loop is alive"; an off-thread dispatch must
        not hide a wedged main loop."""
        bc = self.bc
        stale = time.time() - 100.0
        bc._main_loop_heartbeat[0] = stale
        t = threading.Thread(
            target=lambda: bc.parse_and_run_actions("[ACTION: hb_probe, a]"))
        t.start()
        t.join(10)
        self.assertEqual(len(self.ages), 1)
        self.assertEqual(bc._main_loop_heartbeat[0], stale)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
