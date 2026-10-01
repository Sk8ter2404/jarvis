"""Monolith wiring for device dialogues (core/dialogue.py): the generic hooks
a skill uses to run a short scripted back-and-forth between JARVIS and a
talking device.

Covers:
  * request_tts_interrupt accepts a non-acoustic stop while a dialogue runs
    (the device may be talking while JARVIS is silent), still refuses without
    one and during the boot grace;
  * _dialogue_ready's reasons; the session (filler cancel, flag cleared on an
    exception, dialogue bracket, re-prime, holds, one at a time);
  * handle.stop from another thread cuts a line, stopped() reasons;
  * _speak_line: interrupted / failed-while-capture-live / muted / spoken;
  * _listen_for_stop: refused during playback, a reply, another Path-B
    capture and outside a dialogue; the stream is closed BEFORE the owner is
    released, both before return; a stop word stops the dialogue; transcripts
    are never printed and never reach a learner;
  * the self-echo gate (core/self_echo.py): _speak_line registers a line
    exactly like _speak, and the stop-listen is never gated by it (the
    owner's words inside the SELF_ECHO_TAIL_S tail still reach the dialogue);
  * the generic Path-B double-open fix in get_mic_buffer;
  * the holds: _speak_pending keeps the queue, should_be_proactive is False,
    a non-wake transcript is dropped and a wake-prefixed / typed one passes;
    the learners return early during a dialogue;
  * SELF_VOICED actions: nothing else is spoken for an all-self-voiced reply
    (prose, quip, verbatim, follow-up; a failure-marker result is ignored),
    mixed replies keep their speech, the proactive path never voices one,
    the chain dispatcher gets the predicate, the speak sets stay disjoint;
  * _local_complete: no local-mode directive, the sampling options and
    format, a 400 retries once without format, the local_only guard;
  * the skill_utils keys and their JarvisServices wrappers.

GENERIC fixtures only ("desk device"). No real audio, no LLM, no network.

Run locally (full-deps tier):
    python -m unittest tests.monolith.test_monolith_dialogue
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_SRC = "desk device"


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import device_speech_filter as dsf
        self.dsf = dsf
        dsf._reset_cache_for_tests()
        self.addCleanup(dsf._reset_cache_for_tests)
        # A normal, awake, post-boot, non-staging session.
        self._p(bc, "_is_staging", return_value=False)
        self._p(bc, "_mic_input_disabled", return_value=False)
        self._p(bc, "_session_start_time", time.time() - 1000.0)
        self._p(bc, "DIALOGUE_ENABLED", True)
        self._p(bc, "DIALOGUE_STOP_LISTEN", True)
        self._p(bc, "_tts_muted", [False])
        self._p(bc, "_mic_muted", [False])
        self._p(bc, "_sleep_mode", [False])
        self._p(bc, "_standby_mode", [False])
        self._p(bc, "_realtime_session", [None])
        self._p(bc, "_dialogue_active", [False])
        self._p(bc, "_dialogue_current", [None])
        self._p(bc, "_speech_hold_until", [0.0])
        self._p(bc, "_turn_hold_until", [0.0])
        self._p(bc, "_turn_hold_reason", [""])
        self._p(bc, "_tts_interrupt_seq", [0])
        self._p(bc, "_tts_playback_active", [False])
        self._p(bc, "_tts_reply_active", [False])
        self._p(bc, "_pathb_mic_active", [False])
        self._p(bc, "_record_speech_active", [False])
        self._p(bc, "_ambient_stream_active", [0])
        self.reprime = self._p(bc, "_reprime_after_background")
        self.filler = mock.MagicMock()
        self._p(bc, "_processing_filler", self.filler)
        bc._tts_interrupt.clear()
        self.addCleanup(bc._tts_interrupt.clear)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _hide_wake_listener(self):
        """Take the wake listener out of sys.modules for this test only (a
        bare pop leaked into every later test of the run)."""
        patcher = mock.patch.dict(sys.modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("skill_wake_listener", None)

    def _quiet(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()


# ── request_tts_interrupt ────────────────────────────────────────────────
class InterruptGateTests(_Base):
    def test_accepted_while_a_dialogue_runs_with_nothing_playing(self):
        self.bc._dialogue_active[0] = True
        ok, _ = self._quiet(self.bc.request_tts_interrupt, "tray", False)
        self.assertTrue(ok)
        self.assertEqual(self.bc._tts_interrupt_seq[0], 1)
        self.assertTrue(self.bc._tts_interrupt.is_set())

    def test_refused_without_a_dialogue_or_playback(self):
        ok, _ = self._quiet(self.bc.request_tts_interrupt, "tray", False)
        self.assertFalse(ok)
        self.assertEqual(self.bc._tts_interrupt_seq[0], 0)

    def test_refused_during_the_boot_grace(self):
        self.bc._dialogue_active[0] = True
        self.bc._session_start_time = time.time()
        ok, log = self._quiet(self.bc.request_tts_interrupt, "tray", False)
        self.assertFalse(ok)
        self.assertIn("boot grace", log)

    def test_acoustic_echo_gate_still_applies(self):
        self.bc._dialogue_active[0] = True
        self._p(self.bc, "_barge_in_wake_enabled", return_value=True)
        self._p(self.bc, "_tts_current_text", ["well, jarvis, indeed"])
        ok, _ = self._quiet(self.bc.request_tts_interrupt, "wake-word", True)
        self.assertFalse(ok)


# ── readiness + session ──────────────────────────────────────────────────
class ReadyTests(_Base):
    def test_ready_when_all_clear(self):
        self.assertEqual(self.bc._dialogue_ready(), "")

    def test_reasons(self):
        bc = self.bc
        cases = [
            ("DIALOGUE_ENABLED", False, "disabled"),
            ("_tts_muted", [True], "tts_muted"),
            ("_mic_muted", [True], "mic_muted"),
            ("_sleep_mode", [True], "sleep"),
            ("_standby_mode", [True], "sleep"),
            ("_realtime_session", [object()], "realtime_voice"),
            ("_dialogue_active", [True], "active"),
        ]
        for name, value, reason in cases:
            with mock.patch.object(bc, name, value):
                self.assertEqual(bc._dialogue_ready(), reason, name)
        with mock.patch.object(bc, "_is_staging", return_value=True):
            self.assertEqual(bc._dialogue_ready(), "staging")
        with mock.patch.object(bc, "_session_start_time", time.time()):
            self.assertEqual(bc._dialogue_ready(), "boot_grace")


class SessionTests(_Base):
    def test_enter_and_exit(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC, max_s=30) as h:
                self.assertTrue(bc._dialogue_active[0])
                self.assertIs(bc._dialogue_current[0], h)
                self.assertTrue(self.dsf.dialogue_active())
                self.assertEqual(self.dsf.dialogue_source(), _SRC)
                self.assertIsNone(h.stopped())
                self.assertEqual(bc._dialogue_ready(), "active")
        self.filler.cancel.assert_called_once_with("dialogue")
        self.assertFalse(bc._dialogue_active[0])
        self.assertIsNone(bc._dialogue_current[0])
        # The 4 s tail: the other listeners still hold.
        self.assertTrue(self.dsf.dialogue_active())
        self.assertTrue(bc._dialogue_gate_active())
        self.reprime.assert_called_once_with("dialogue")

    def test_refused_raises_with_the_reason(self):
        bc = self.bc
        bc._tts_muted[0] = True
        with self.assertRaises(bc.DialogueUnavailable) as cm:
            with bc._dialogue_session(_SRC):
                self.fail("must not enter")
        self.assertEqual(cm.exception.reason, "tts_muted")
        self.assertFalse(bc._dialogue_active[0])

    def test_one_at_a_time(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC):
                with self.assertRaises(bc.DialogueUnavailable) as cm:
                    with bc._dialogue_session(_SRC):
                        pass
                self.assertEqual(cm.exception.reason, "active")

    def test_flag_cleared_on_exception(self):
        bc = self.bc
        with self._quiet_ctx():
            with self.assertRaises(RuntimeError):
                with bc._dialogue_session(_SRC):
                    raise RuntimeError("skill bug")
        self.assertFalse(bc._dialogue_active[0])
        self.assertIsNone(bc._dialogue_current[0])
        self.reprime.assert_called_once_with("dialogue")

    def test_holds_applied_at_exit(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC) as h:
                h.hold_after(12.0, "device_lost")
                h.hold_after(3.0, "other")          # the longest wins
                self.assertFalse(bc._speech_hold_active())   # not yet
        self.assertTrue(bc._speech_hold_active())
        self.assertGreater(bc._turn_hold_until[0], time.monotonic() + 11.0)
        self.assertEqual(bc._turn_hold_reason[0], "device_lost")

    def test_stopped_reasons(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC, max_s=30) as h:
                bc._tts_muted[0] = True
                self.assertEqual(h.stopped(), "tts_muted")
                bc._tts_muted[0] = False
                bc._mic_muted[0] = True
                self.assertEqual(h.stopped(), "mic_muted")
                bc._mic_muted[0] = False
                bc._sleep_mode[0] = True
                self.assertEqual(h.stopped(), "sleep")
                bc._sleep_mode[0] = False
                h._t0 -= 31.0
                self.assertEqual(h.stopped(), "expired")
                h._t0 += 31.0
                bc._tts_interrupt_seq[0] += 1          # tray / wake barge
                self.assertEqual(h.stopped(), "interrupted")

    def test_stop_from_another_thread_bumps_the_seq_once(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC) as h:
                results = []
                t = threading.Thread(
                    target=lambda: results.append(h.stop("device_lost")))
                t.start()
                t.join(5)
                self.assertEqual(results, [True])
                self.assertFalse(h.stop("owner_stop"))
                self.assertEqual(h.stopped(), "device_lost")
                self.assertEqual(bc._tts_interrupt_seq[0], 1)
                self.assertTrue(bc._tts_interrupt.is_set())

    def test_stop_after_the_session_ended_never_cuts_a_later_reply(self):
        """A slow stop-listen verdict (or a late device watcher) that calls
        stop() after the session exited must not interrupt whatever JARVIS
        is saying by then."""
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC) as h:
                pass
            bc._tts_playback_active[0] = True      # an unrelated reply plays
            self.assertFalse(h.stop("owner_stop"))
        self.assertEqual(bc._tts_interrupt_seq[0], 0)
        self.assertFalse(bc._tts_interrupt.is_set())

    def test_a_stop_during_a_device_line_leaves_no_stale_interrupt(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC) as h:
                h.stop("owner_stop")               # nothing was playing
                self.assertTrue(bc._tts_interrupt.is_set())
        self.assertFalse(bc._tts_interrupt.is_set())
        self.assertEqual(bc._tts_interrupt_seq[0], 1)   # the seq still moved

    def test_hold_after_that_races_the_exit_still_applies(self):
        bc = self.bc
        with self._quiet_ctx():
            with bc._dialogue_session(_SRC) as h:
                pass
            self.assertFalse(bc._speech_hold_active())
            h.hold_after(12.0, "device_lost")      # the watcher was late
        self.assertTrue(bc._speech_hold_active())
        self.assertGreater(bc._turn_hold_until[0], time.monotonic() + 11.0)
        self.assertEqual(bc._turn_hold_reason[0], "device_lost")

    def test_handoff_turn_is_reserved(self):
        with self._quiet_ctx():
            with self.bc._dialogue_session(_SRC) as h:
                self.assertFalse(h.handoff_turn("hello"))

    @contextlib.contextmanager
    def _quiet_ctx(self):
        with contextlib.redirect_stdout(io.StringIO()):
            yield


# ── _speak_line ──────────────────────────────────────────────────────────
class SpeakLineTests(_Base):
    def test_spoken(self):
        spk = self._p(self.bc, "_speak", return_value=True)
        self.assertEqual(self.bc._speak_line("Hello.", mood="wry"), "spoken")
        spk.assert_called_once_with("Hello.", mood="wry")

    def test_interrupted_when_the_seq_moves(self):
        bc = self.bc

        def speak(_t, mood=None):
            bc._tts_interrupt_seq[0] += 1
            return True
        self._p(bc, "_speak", side_effect=speak)
        self.assertEqual(bc._speak_line("Hello."), "interrupted")

    def test_failed_when_a_capture_stays_live(self):
        bc = self.bc
        spk = self._p(bc, "_speak", return_value=True)
        h = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_current[0] = h
        bc._dialogue_active[0] = True
        bc._pathb_mic_active[0] = True
        out, _ = self._quiet(bc._speak_line, "Hello.")
        self.assertEqual(out, "failed")
        spk.assert_not_called()

    def test_waits_for_a_capture_to_release(self):
        bc = self.bc
        seen = []
        self._p(bc, "_speak",
                side_effect=lambda t, mood=None: seen.append(
                    bc._pathb_mic_active[0]) or True)
        bc._dialogue_current[0] = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_active[0] = True
        bc._pathb_mic_active[0] = True
        threading.Timer(0.1, bc._pathb_mic_active.__setitem__,
                        args=(0, False)).start()
        self.assertEqual(bc._speak_line("Hello."), "spoken")
        self.assertEqual(seen, [False])

    def test_muted_and_staging(self):
        bc = self.bc
        bc._tts_muted[0] = True
        self.assertEqual(bc._speak_line("Hello."), "muted")
        bc._tts_muted[0] = False
        with mock.patch.object(bc, "_is_staging", return_value=True):
            self.assertEqual(bc._speak_line("Hello."), "staging")

    def test_speak_failure(self):
        self._p(self.bc, "_speak", return_value=False)
        self.assertEqual(self.bc._speak_line("Hello."), "failed")

    def test_stop_racing_the_start_of_playback_is_reasserted(self):
        """A stop() that lands after the line began but before playback was
        live is wiped by the play path's clear; the line must still be cut."""
        bc = self.bc
        h = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_current[0] = h
        bc._dialogue_active[0] = True
        cut = threading.Event()

        def speak(_t, mood=None):
            h.stop("device_lost")          # accepted: dialogue active
            bc._tts_interrupt.clear()      # the play path clears at start
            bc._tts_playback_active[0] = True
            try:
                cut.wait(2.0)              # "playing" until interrupted
            finally:
                bc._tts_playback_active[0] = False
            return True

        def watch():
            while not cut.is_set():
                if bc._tts_interrupt.is_set():
                    cut.set()
                time.sleep(0.01)
        w = threading.Thread(target=watch, daemon=True)
        w.start()
        self._p(bc, "_speak", side_effect=speak)
        t0 = time.monotonic()
        out, _ = self._quiet(bc._speak_line, "A long line.")
        cut.set()
        self.assertEqual(out, "interrupted")
        self.assertLess(time.monotonic() - t0, 1.5)

    def test_stop_that_bumped_the_seq_before_the_snapshot_is_still_cut(self):
        """The stop landed between the entry's stop-state read and its seq
        snapshot (so seq0 already includes the bump): the line is still cut
        and reported interrupted."""
        bc = self.bc
        h = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_current[0] = h
        bc._dialogue_active[0] = True
        cut = threading.Event()
        real_seq = bc._tts_interrupt_seq

        class _Seq(list):
            reads = 0

            def __getitem__(self, i):
                _Seq.reads += 1
                if _Seq.reads == 1:            # the snapshot read
                    h.stop("device_lost")      # lands just before it
                return list.__getitem__(self, i)
        seq = _Seq(real_seq)
        self._p(bc, "_tts_interrupt_seq", seq)

        def speak(_t, mood=None):
            bc._tts_interrupt.clear()          # the play path clears at start
            bc._tts_playback_active[0] = True
            try:
                cut.wait(2.0)
            finally:
                bc._tts_playback_active[0] = False
            return True

        def watch():
            while not cut.is_set():
                if bc._tts_interrupt.is_set():
                    cut.set()
                time.sleep(0.01)
        threading.Thread(target=watch, daemon=True).start()
        self._p(bc, "_speak", side_effect=speak)
        t0 = time.monotonic()
        out, _ = self._quiet(bc._speak_line, "A long line.")
        cut.set()
        self.assertEqual(out, "interrupted")
        self.assertLess(time.monotonic() - t0, 1.5)


# ── _listen_for_stop ─────────────────────────────────────────────────────
class _FakeStream:
    instances = []

    def __init__(self, samplerate, channels, dtype, blocksize, device,
                 callback):
        self.callback = callback
        self.closed = False
        _FakeStream.instances.append(self)

    def start(self):
        import numpy as np
        for _ in range(8):
            self.callback(np.ones((1024, 1), dtype=np.float32) * 0.2,
                          1024, None, None)


@requires_monolith
class _ListenBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        _FakeStream.instances = []
        self.stream_cls = self._p(bc.sd, "InputStream",
                                  side_effect=_FakeStream)
        self._p(bc, "get_input_device", return_value=0)
        self.close_flags = []

        def close(stream, timeout_sec=2.0):
            self.close_flags.append(bc._pathb_mic_active[0])
            stream.closed = True
        self._p(bc, "_safe_close_stream", side_effect=close)
        self.learn = self._p(bc, "learn_from_turn")
        self.amb = self._p(bc, "_ambient_learn_from_gated")
        self._p(bc, "_note_input_open_failure")
        self._hide_wake_listener()
        self.h = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_current[0] = self.h
        bc._dialogue_active[0] = True

    def listen(self, **kw):
        return self.bc._listen_for_stop(lambda: True, beat_s=0.05,
                                        max_s=2.0, **kw)


class ListenForStopTests(_ListenBase):
    def test_refused_outside_a_dialogue(self):
        self.bc._dialogue_active[0] = False
        cap = self.listen()
        self.assertFalse(cap.available)
        self.stream_cls.assert_not_called()

    def test_refused_while_playback_reply_or_path_b_is_live(self):
        for cell in ("_tts_playback_active", "_tts_reply_active",
                     "_pathb_mic_active", "_record_speech_active"):
            getattr(self.bc, cell)[0] = True
            cap = self.listen()
            self.assertFalse(cap.available, cell)
            getattr(self.bc, cell)[0] = False
        self.bc._ambient_stream_active[0] = 1
        self.assertFalse(self.listen().available)
        self.stream_cls.assert_not_called()

    def test_refused_when_mic_muted_or_stop_listen_off(self):
        self.bc._mic_muted[0] = True
        self.assertFalse(self.listen().available)
        self.bc._mic_muted[0] = False
        with mock.patch.object(self.bc, "DIALOGUE_STOP_LISTEN", False):
            self.assertFalse(self.listen().available)
        self.stream_cls.assert_not_called()

    def test_stream_closed_then_released_before_return(self):
        self._p(self.bc, "transcribe", return_value=("", {}))
        cap = self.listen()
        self.assertTrue(cap.available)
        self.assertEqual(len(_FakeStream.instances), 1)
        self.assertTrue(_FakeStream.instances[0].closed)
        self.assertEqual(self.close_flags, [True])     # closed while owned
        self.assertFalse(self.bc._pathb_mic_active[0])  # released on return
        self.assertTrue(cap.beat_voiced)
        self.assertEqual(cap.result(2.0), ("", ""))

    def test_the_capture_reports_whether_it_held_voice(self):
        # 2026-10-01 review: the Runner skips its verdict wait for a capture
        # with no voice in it, so _listen_for_stop must say which it was.
        self._p(self.bc, "transcribe", return_value=("", {}))
        self.assertTrue(self.listen().voiced)

        class _Silent(_FakeStream):
            def start(self):
                import numpy as np
                for _ in range(8):
                    self.callback(np.zeros((1024, 1), dtype=np.float32),
                                  1024, None, None)
        self.stream_cls.side_effect = _Silent
        cap = self.listen()
        self.assertTrue(cap.available)
        self.assertFalse(cap.voiced)
        cap.result(2.0)

    def test_own_stream_yields_when_playback_starts_elsewhere(self):
        """Another thread starts playing mid-capture (a timer, say): the
        stop-listen closes its own stream at once instead of staying open
        under JARVIS's voice until the device finishes."""
        bc = self.bc
        self._p(bc, "transcribe", return_value=("", {}))
        timer = threading.Timer(0.2, bc._tts_playback_active.__setitem__,
                                args=(0, True))
        timer.start()
        self.addCleanup(timer.cancel)
        t0 = time.monotonic()
        cap = bc._listen_for_stop(lambda: False, beat_s=0.05, max_s=3.0)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertTrue(_FakeStream.instances[0].closed)
        self.assertFalse(bc._pathb_mic_active[0])
        cap.result(2.0)

    def test_stop_word_stops_the_dialogue_quietly(self):
        heard = "please stop right now marmalade"
        self._p(self.bc, "transcribe", return_value=(heard, {}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cap = self.listen()
            kind = cap.result(3.0)
        self.assertEqual(kind, ("stop", ""))
        self.assertEqual(self.h.stopped(), "owner_stop")
        self.assertNotIn("marmalade", buf.getvalue())
        self.learn.assert_not_called()
        self.amb.assert_not_called()

    def test_wake_word_stops_with_wake(self):
        self._p(self.bc, "transcribe", return_value=("hey jarvis", {}))
        with contextlib.redirect_stdout(io.StringIO()):
            cap = self.listen()
            self.assertEqual(cap.result(3.0)[0], "wake")
        self.assertEqual(self.h.stopped(), "wake")

    def test_expected_device_line_is_not_speech(self):
        self.dsf.expect(_SRC, "The toast is late again today.",
                        window_s=5.0)
        self._p(self.bc, "transcribe",
                return_value=("the toast is late again today", {}))
        cap = self.listen()
        self.assertEqual(cap.result(3.0), ("", ""))
        self.assertIsNone(self.h.stopped())

    def test_other_speech_is_reported_but_not_acted_on(self):
        self._p(self.bc, "transcribe", return_value=("what a lovely day",
                                                     {}))
        cap = self.listen()
        self.assertEqual(cap.result(3.0), ("speech", "what a lovely day"))
        self.assertIsNone(self.h.stopped())

    def test_voiced_beat_is_transcribed_on_its_own(self):
        calls = []

        def tr(audio):
            calls.append(len(audio))
            return (("the toast is late", {}) if len(calls) == 1
                    else ("that's enough", {}))
        self._p(self.bc, "transcribe", side_effect=tr)
        polls = []

        def until():                  # the device finishes after 3 frames
            polls.append(1)
            return len(polls) > 3
        with contextlib.redirect_stdout(io.StringIO()):
            cap = self.bc._listen_for_stop(until, beat_s=0.05, max_s=2.0)
            self.assertTrue(cap.beat_voiced)
            self.assertEqual(cap.result(3.0)[0], "stop")
        self.assertEqual(len(calls), 2)
        self.assertLess(calls[1], calls[0])


# ── the self-echo gate (core/self_echo.py) ───────────────────────────────
_JARVIS_LINE = "The kettle has strong opinions today."


class SelfEchoInterplayTests(_ListenBase):
    """Dialogues next to the self-echo gate (core/self_echo.py).

    (a) _speak_line IS _speak: a dialogue line gets the same self-echo
        registration as any other line — the audible-playback window
        (play_with_lipsync's playback_begin / playback_end) and the content
        memory counted from the END of the line (remember / refresh).
    (b) The stop-listen is never gated by self-echo: the owner's words heard
        inside the SELF_ECHO_TAIL_S tail right after a JARVIS dialogue line
        still reach the dialogue, and the stop-listen publishes no capture
        timing a later main-loop gate could pick up."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import self_echo as se
        self.se = se
        se._reset_for_tests()
        self.addCleanup(se._reset_for_tests)
        self.clock = [500.0]
        self._p(se, "_clock", lambda: self.clock[0])
        self._p(bc, "SELF_ECHO_FILTER_ENABLED", True)
        self._p(bc, "SELF_ECHO_WINDOW_S", 20.0)
        self._p(bc, "SELF_ECHO_TAIL_S", 0.8)
        self._p(bc, "_last_capture_window", [None])
        # The real _speak down to play_with_lipsync's device body.
        self._p(bc, "_tts_layer", None)
        self._p(bc, "_sentence_tts_plan", lambda text: None)
        self._p(bc, "synthesise",
                side_effect=lambda t: (bc.np.zeros(2400, dtype="float32"),
                                       24000))
        self._p(bc, "set_state")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_heartbeat")

    def _say(self, speak) -> list:
        """Speak _JARVIS_LINE through ``speak`` from t=500.0; the fake device
        body lasts 1.5 s. Returns playback_live() as seen while it played."""
        self.se._reset_for_tests()
        self.clock[0] = 500.0
        during = []

        def body(audio, sr):
            during.append(self.se.playback_live())
            self.clock[0] += 1.5
        with mock.patch.object(self.bc, "_play_with_lipsync_body",
                               side_effect=body), \
                contextlib.redirect_stdout(io.StringIO()):
            speak(_JARVIS_LINE)
        return during

    def _observe(self, speak) -> dict:
        bc = self.bc
        during = self._say(speak)
        live_after = self.se.playback_live()
        # A main-loop capture open through the line, speech 0.5 s in.
        self.clock[0] = 502.0
        bc._last_capture_window[0] = (499.0, 500.5, 502.0, 500.3)
        timing, _ = self._quiet(bc._self_echo_ignored, "what a lovely day")
        # 19 s after the line ENDED (20.5 s after it began): still an echo
        # only if the window restarted at the end of playback.
        self.clock[0] = 520.5
        bc._last_capture_window[0] = None
        content, _ = self._quiet(bc._self_echo_ignored, _JARVIS_LINE.lower())
        return {"registered_while_playing": during, "live_after": live_after,
                "timing_gate": timing, "content_gate_from_end": content}

    def test_speak_line_registers_exactly_like_speak(self):
        bc = self.bc
        results = []

        def via_speak_line(text):
            results.append(bc._speak_line(text))
        want = {"registered_while_playing": [True], "live_after": False,
                "timing_gate": True, "content_gate_from_end": True}
        self.assertEqual(self._observe(lambda t: bc._speak(t)), want)
        self.assertEqual(self._observe(via_speak_line), want)
        self.assertEqual(results, ["spoken"])

    def test_stop_listen_hears_the_owner_inside_the_self_echo_tail(self):
        bc, se = self.bc, self.se
        self.assertEqual(self._say(lambda t: bc._speak_line(t)), [True])
        self.clock[0] = 501.6            # the line ended 0.1 s ago
        # Control: the tail is armed — a main-loop capture that was open
        # through the line and heard speech now would be dropped.
        bc._last_capture_window[0] = (499.0, 501.6, 502.4, 501.5)
        self.assertTrue(self._quiet(bc._self_echo_ignored,
                                    "what a lovely day")[0])
        bc._last_capture_window[0] = None
        # A gate that swallows everything, and spies on the self-echo API:
        # the stop-listen must consult none of them.
        gate = self._p(bc, "_self_echo_ignored", return_value=True)
        overlap = self._p(se, "capture_overlap", wraps=se.capture_overlap)
        match = self._p(se, "match", wraps=se.match)
        for heard, verdict, stopped in (
                ("what a lovely day", ("speech", "what a lovely day"), None),
                ("please stop right now", ("stop", ""), "owner_stop"),
                ("hey jarvis", ("wake", ""), "wake")):
            h = bc._DialogueHandle(_SRC, 30)
            bc._dialogue_current[0] = h
            with mock.patch.object(bc, "transcribe",
                                   return_value=(heard, {})), \
                    contextlib.redirect_stdout(io.StringIO()):
                cap = self.listen()
                self.assertTrue(cap.available, heard)
                self.assertEqual(cap.result(3.0), verdict, heard)
            self.assertEqual(h.stopped(), stopped, heard)
        gate.assert_not_called()
        overlap.assert_not_called()
        match.assert_not_called()
        self.assertIsNone(bc._last_capture_window[0])


class PathBDoubleOpenTests(_Base):
    def test_second_path_b_capture_is_denied(self):
        bc = self.bc
        self._hide_wake_listener()
        stream = self._p(bc.sd, "InputStream")
        self._p(bc, "get_input_device", return_value=0)
        bc._pathb_mic_active[0] = True          # another Path B is live
        self.assertIsNone(bc._get_mic_buffer_impl(0.1))
        stream.assert_not_called()
        self.assertTrue(bc._pathb_mic_active[0])  # never cleared by us


# ── holds + learners ─────────────────────────────────────────────────────
# ── a dialogue started OFF the main thread vs the main loop's microphone ────
class _LiveMic:
    """A fake input stream that keeps delivering quiet frames from its own
    thread until it is closed: a microphone somebody is listening on. Counts
    how many are open at once (two open captures on one device is the WASAPI
    double-open the rest of the capture code exists to prevent)."""
    lock = threading.Lock()
    open_now = 0
    max_open = 0
    instances: list = []

    def __init__(self, samplerate=16000, channels=1, dtype="float32",
                 blocksize=1024, device=None, callback=None):
        self.callback = callback
        self.closed = False
        self.started = False
        self._stop = threading.Event()
        _LiveMic.instances.append(self)

    @classmethod
    def reset(cls):
        cls.open_now = 0
        cls.max_open = 0
        cls.instances = []

    def start(self):
        import numpy as np
        with _LiveMic.lock:
            _LiveMic.open_now += 1
            _LiveMic.max_open = max(_LiveMic.max_open, _LiveMic.open_now)
        self.started = True
        frame = np.full((1024, 1), 0.001, dtype=np.float32)

        def feed():
            while not self._stop.wait(0.02):
                try:
                    self.callback(frame, 1024, None, None)
                except Exception:
                    return
        threading.Thread(target=feed, name="fake-mic", daemon=True).start()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        if self.started:
            with _LiveMic.lock:
                _LiveMic.open_now -= 1


@requires_monolith
class WebStartedDialogueMicTests(_Base):
    """A dialogue started from a WEB PANEL action runs on a web-server thread,
    not inside a voice turn, while the main loop may still be listening in
    record_speech. The dialogue's stop-listen (_listen_for_stop) is refused
    while record_speech holds the microphone, so before the fix the owner
    could not stop a web-started dialogue by voice. The main loop's capture
    now yields for as long as a dialogue on ANOTHER thread runs
    (_dialogue_holds_mic). Fake microphone only; generic fixtures."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        _LiveMic.reset()
        self.addCleanup(lambda: [m.close() for m in list(_LiveMic.instances)])
        self.stream_cls = self._p(bc.sd, "InputStream", side_effect=_LiveMic)
        self._p(bc, "get_input_device", return_value=0)
        self._p(bc, "_safe_close_stream",
                side_effect=lambda s, timeout_sec=2.0: (
                    s.close() if s is not None else None))
        self._p(bc, "_input_backoff_wait", return_value=False)
        self._p(bc, "_filler_capture_mark")
        self._p(bc, "_note_live_capture")
        self._p(bc, "_input_open_succeeded")
        self._p(bc, "_heartbeat")
        self._p(bc, "HUD_ENABLED", False)
        self._p(bc, "VAD_THRESHOLD", 0.5)          # quiet frames never trip
        self._p(bc, "_process_capture_chunk",
                side_effect=lambda data, sr, skip_ns=False: data)
        from core import audio_processor as ap
        for name in ("note_vad_poll", "note_raw_rms", "note_vad_active"):
            self._p(ap, name)
        self._p(ap, "seconds_since_audible_chunk", return_value=0.0)
        self._p(bc, "_dialogue_mic_yield_logged", [None], create=True)
        self._p(bc, "learn_from_turn")
        self._p(bc, "_ambient_learn_from_gated")
        self._p(bc, "_note_input_open_failure")
        self._p(bc, "set_state")
        self._hide_wake_listener()                 # Path B: a stream of its own
        bc._watchdog_reset_signal.clear()

    def _foreign_handle(self):
        """A dialogue handle built on ANOTHER thread (a web-server thread)."""
        box = []
        t = threading.Thread(
            target=lambda: box.append(self.bc._DialogueHandle(_SRC, 30)),
            name="web-handler")
        t.start()
        t.join(5.0)
        return box[0]

    def _run_foreign_dialogue(self):
        h = self._foreign_handle()
        self.bc._dialogue_current[0] = h
        self.bc._dialogue_active[0] = True
        return h

    def test_holds_mic_only_for_a_dialogue_on_another_thread(self):
        bc = self.bc
        self.assertFalse(bc._dialogue_holds_mic())             # no dialogue
        bc._dialogue_current[0] = bc._DialogueHandle(_SRC, 30)  # this thread
        bc._dialogue_active[0] = True
        self.assertFalse(bc._dialogue_holds_mic(),
                         "a dialogue must never yield the mic to itself")
        bc._dialogue_current[0] = self._foreign_handle()
        self.assertTrue(bc._dialogue_holds_mic())
        bc._dialogue_active[0] = False
        self.assertFalse(bc._dialogue_holds_mic())

    def test_record_speech_opens_nothing_while_a_foreign_dialogue_runs(self):
        bc = self.bc
        self._run_foreign_dialogue()
        out, log = self._quiet(bc.record_speech, timeout=0.2)
        self.assertIsNone(out)
        self.stream_cls.assert_not_called()
        self.assertFalse(bc._record_speech_active[0])
        self.assertIn("yields the microphone", log)

    def test_capture_utterance_takes_no_capture_while_a_foreign_dialogue_runs(self):
        bc = self.bc
        rec = self._p(bc, "record_speech", return_value=None)
        self._p(bc, "_speak_pending", return_value=False)
        self._run_foreign_dialogue()
        out, _ = self._quiet(bc._capture_utterance, None, {})
        self.assertIsNone(out)
        rec.assert_not_called()
        # an injected (typed) turn still passes during the dialogue
        out, _ = self._quiet(bc._capture_utterance, "what time is it", {})
        self.assertEqual(out[0], "what time is it")

    def test_a_voice_turn_dialogue_still_captures_on_its_own_thread(self):
        bc = self.bc
        bc._dialogue_current[0] = bc._DialogueHandle(_SRC, 30)
        bc._dialogue_active[0] = True
        out, _ = self._quiet(bc.record_speech, timeout=0.2)
        self.assertIsNone(out)                       # quiet: timed out
        self.assertEqual(self.stream_cls.call_count, 1)
        self.assertTrue(_LiveMic.instances[0].closed)

    def test_the_yield_is_logged_once_per_dialogue(self):
        bc = self.bc
        self._run_foreign_dialogue()
        _, log = self._quiet(lambda: [bc._dialogue_mic_yield_wait(0.0)
                                      for _ in range(5)])
        self.assertEqual(log.count("yields the microphone"), 1)

    def test_a_web_started_dialogue_gets_the_mic_from_a_listening_main_loop(self):
        """THE BUG, end to end: the main loop is inside record_speech (the
        fake mic is live) when a dialogue starts on another thread. The
        capture hands the microphone over, the dialogue's stop-listen gets it,
        and the owner's "stop" ends the dialogue. Never two streams at once."""
        bc = self.bc
        self._p(bc, "transcribe", return_value=("please stop", {}))
        rec_out = {}

        def main_loop_listen():
            rec_out["audio"] = bc.record_speech(timeout=4.0)
            rec_out["at"] = time.monotonic()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main = threading.Thread(target=main_loop_listen, name="main-loop")
            main.start()
            self.addCleanup(main.join, 6.0)
            deadline = time.monotonic() + 3.0
            while (not bc._record_speech_active[0]
                   and time.monotonic() < deadline):
                time.sleep(0.01)
            self.assertTrue(bc._record_speech_active[0],
                            "the fake main loop never started listening")
            # This test thread plays the web-server thread.
            t0 = time.monotonic()
            with bc._dialogue_session(_SRC) as h:
                cap = bc._listen_for_stop(lambda: True, beat_s=0.05,
                                          max_s=1.0)
                verdict = cap.result(3.0) if cap.available else None
                stopped = h.stopped()
            main.join(6.0)
        self.assertTrue(cap.available,
                        "the stop-listen could not get the microphone from "
                        "the listening main loop")
        self.assertEqual(verdict, ("stop", ""))
        self.assertEqual(stopped, "owner_stop")
        self.assertFalse(main.is_alive())
        self.assertIsNone(rec_out.get("audio"))
        self.assertLess(rec_out["at"] - t0, 2.0,
                        "record_speech kept the mic after the dialogue began")
        self.assertEqual(_LiveMic.max_open, 1,
                         "two captures were open on the microphone at once")
        self.assertEqual(len(_LiveMic.instances), 2)   # main loop, then listen
        self.assertFalse(bc._record_speech_active[0])
        self.assertFalse(bc._pathb_mic_active[0])
        self.assertIn("a dialogue needs the microphone", buf.getvalue())


class HoldTests(_Base):
    def test_speak_pending_keeps_the_queue_while_held(self):
        bc = self.bc
        tmp = tempfile.mkdtemp(prefix="dlg_pending_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        path = os.path.join(tmp, "pending_speech.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"message": "A reminder, sir."}], f)
        self._p(bc, "PENDING_SPEECH_PATH", path)
        spk = self._p(bc, "_speak")
        bc._speech_hold_until[0] = time.monotonic() + 10.0
        self.assertFalse(bc._speak_pending())
        self.assertTrue(os.path.exists(path))
        spk.assert_not_called()

    def test_should_be_proactive_false_while_held_or_in_dialogue(self):
        bc = self.bc
        self._p(bc, "PROACTIVE_ENABLED", True)
        bc._speech_hold_until[0] = time.monotonic() + 10.0
        self.assertFalse(bc.should_be_proactive())
        bc._speech_hold_until[0] = 0.0
        bc._dialogue_active[0] = True
        self.assertFalse(bc.should_be_proactive())

    def test_turn_hold_drops_non_wake_mic_turns_only(self):
        bc = self.bc
        bc._turn_hold_until[0] = time.monotonic() + 10.0
        bc._turn_hold_reason[0] = "device_lost"
        out, log = self._quiet(bc._dialogue_hold_ignored, "what time is it")
        self.assertTrue(out)
        self.assertIn("[dialogue-hold] ignored (device_lost)", log)
        self.assertNotIn("what time", log)
        self.assertFalse(bc._dialogue_hold_ignored("jarvis, what time is it"))
        self.assertFalse(bc._dialogue_hold_ignored("what time is it",
                                                   injected=True))
        bc._turn_hold_until[0] = 0.0
        self.assertFalse(bc._dialogue_hold_ignored("what time is it"))

    def test_main_loop_and_standby_sites_consult_the_hold(self):
        import inspect
        src = inspect.getsource(self.bc)
        self.assertEqual(src.count("_dialogue_hold_ignored(text, "), 2)

    def test_learners_return_early_during_a_dialogue(self):
        bc = self.bc
        self._p(bc, "LEARN_EVERY_TURN", True)
        self._p(bc, "AMBIENT_LISTEN_ENABLED", True)
        bc._dialogue_active[0] = True
        pending = list(bc._learn_pending)
        live = bc._learn_worker_live[0]
        bc.learn_from_turn("talk about toast", "", {})
        self.assertEqual(list(bc._learn_pending), pending)
        self.assertEqual(bc._learn_worker_live[0], live)
        media = self._p(bc, "_ambient_media_is_playing", return_value=False)
        self.assertIsNone(bc._ambient_learn_from_gated("the toast", {}))
        media.assert_not_called()

    def test_learner_tail_uses_the_filter_bracket(self):
        tok = self.dsf.begin_dialogue(_SRC)
        self.assertTrue(self.bc._dialogue_gate_active())
        self.dsf.end_dialogue(tok, tail_s=0.0)
        self.assertFalse(self.bc._dialogue_gate_active())


# ── DIALOGUE_LOST_HOLD_S (the Settings knob nothing used to read) ────────
class LostHoldKnobTests(_Base):
    """A dialogue that ends because the device went away (stop("device_lost"))
    holds speech and non-wake turns for DIALOGUE_LOST_HOLD_S, read at the
    dialogue's END. Before, nothing in the tree read the knob: the hold was
    only whatever seconds a skill passed to hold_after(), so the Settings
    value never reached this path."""

    def _end_with(self, reason, *, hold=None):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            with bc._dialogue_session(_SRC) as h:
                if hold is not None:
                    h.hold_after(*hold)
                if reason:
                    h.stop(reason)
        return time.monotonic()

    def test_device_lost_holds_for_the_knob_without_a_hold_after(self):
        bc = self.bc
        self._p(bc, "DIALOGUE_LOST_HOLD_S", 7.0)
        t = self._end_with("device_lost")
        self.assertTrue(bc._speech_hold_active())
        self.assertEqual(bc._turn_hold_reason[0], "device_lost")
        self.assertAlmostEqual(bc._turn_hold_until[0] - t, 7.0, delta=0.5)
        self.assertAlmostEqual(bc._speech_hold_until[0] - t, 7.0, delta=0.5)

    def test_the_knob_is_read_at_each_dialogues_end(self):
        bc = self.bc
        self._p(bc, "DIALOGUE_LOST_HOLD_S", 3.0)
        t = self._end_with("device_lost")
        self.assertAlmostEqual(bc._turn_hold_until[0] - t, 3.0, delta=0.5)
        bc._speech_hold_until[0] = bc._turn_hold_until[0] = 0.0
        bc.DIALOGUE_LOST_HOLD_S = 20.0          # changed on the live module
        t = self._end_with("device_lost")
        self.assertAlmostEqual(bc._turn_hold_until[0] - t, 20.0, delta=0.5)

    def test_a_longer_skill_hold_still_wins_and_a_shorter_one_is_raised(self):
        bc = self.bc
        self._p(bc, "DIALOGUE_LOST_HOLD_S", 5.0)
        t = self._end_with("device_lost", hold=(30.0, "skill"))
        self.assertAlmostEqual(bc._turn_hold_until[0] - t, 30.0, delta=0.5)
        self.assertEqual(bc._turn_hold_reason[0], "skill")
        bc._speech_hold_until[0] = bc._turn_hold_until[0] = 0.0
        t = self._end_with("device_lost", hold=(1.0, "skill"))
        self.assertAlmostEqual(bc._turn_hold_until[0] - t, 5.0, delta=0.5)
        self.assertEqual(bc._turn_hold_reason[0], "device_lost")

    def test_other_endings_and_a_zero_knob_hold_nothing(self):
        bc = self.bc
        self._p(bc, "DIALOGUE_LOST_HOLD_S", 9.0)
        for reason in (None, "owner_stop", "wake", "device_busy"):
            with self.subTest(reason=reason):
                self._end_with(reason)
                self.assertFalse(bc._speech_hold_active())
                self.assertEqual(bc._turn_hold_until[0], 0.0)
        bc.DIALOGUE_LOST_HOLD_S = 0
        self._end_with("device_lost")
        self.assertFalse(bc._speech_hold_active())
        self.assertEqual(bc._turn_hold_until[0], 0.0)

    def test_a_bad_knob_falls_back_and_is_capped(self):
        bc = self.bc
        for value, want in (("junk", 12.0), (float("nan"), 12.0),
                            (-4.0, 0.0), (1e6, 120.0), (None, 12.0)):
            with self.subTest(value=value):
                self._p(bc, "DIALOGUE_LOST_HOLD_S", value)
                self.assertEqual(bc._dialogue_lost_hold_s(), want)

    def test_the_shipped_default_matches_the_settings_row(self):
        from core import config
        from tools import settings_window as sw
        self.assertEqual(config.DIALOGUE_LOST_HOLD_S, 12.0)
        self.assertEqual(sw.SCHEMA["DIALOGUE_LOST_HOLD_S"]["default"],
                         config.DIALOGUE_LOST_HOLD_S)


# ── self-voiced actions ──────────────────────────────────────────────────
class SelfVoicedTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "SELF_VOICED_ACTIONS", set())
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS))
        self._p(bc, "INFORMATIVE_ACTIONS", set(bc.INFORMATIVE_ACTIONS))
        self.spoken = []
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "maybe_glance_response", return_value=None)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_stream_spoken_prefix", [""])
        self.quip = self._p(bc, "_apply_quip_layer",
                            side_effect=lambda s, r: s + " QUIP")
        self.verbatim = self._p(bc, "_speak_verbatim_results",
                                return_value=False)
        self.followup = self._p(bc, "get_followup_response",
                                return_value="Following up.")
        self._p(bc, "_trim_conversation_history")

    def _turn(self, spoken, results):
        bc = self.bc
        self._p(bc, "get_response_with_animation", return_value="reply")
        self._p(bc, "parse_and_run_actions", side_effect=[
            (spoken, results), ("", [])])
        with contextlib.redirect_stdout(io.StringIO()):
            bc._run_llm_dispatch_body("talk to the desk device")

    def test_register_and_query(self):
        bc = self.bc
        self.assertTrue(bc.register_self_voiced("Desk_Chat"))
        self.assertTrue(bc.is_self_voiced("desk_chat"))
        self.assertFalse(bc.is_self_voiced("other"))
        self.assertFalse(bc.register_self_voiced(""))

    def test_all_self_voiced_reply_speaks_nothing_else(self):
        self.bc.register_self_voiced("desk_chat")
        self._turn("On it, sir.", [("desk_chat",
                                    "Dialogue finished: 4 lines, done.",
                                    True)])
        self.assertEqual(self.spoken, [])
        self.quip.assert_not_called()
        self.verbatim.assert_not_called()
        self.followup.assert_not_called()

    def test_failure_marker_result_is_not_reported(self):
        self.bc.register_self_voiced("desk_chat")
        self._turn("", [("desk_chat", "It failed and could not start.",
                         False)])
        self.followup.assert_not_called()
        self.assertEqual(self.spoken, [])

    def test_mixed_reply_keeps_its_speech(self):
        self.bc.register_self_voiced("desk_chat")
        self._turn("Right away, sir.",
                   [("desk_chat", "It failed.", False),
                    ("see_screen", "A text editor.", True)])
        self.assertTrue(self.spoken)
        self.quip.assert_called()
        self.followup.assert_called_once()
        informative = self.followup.call_args[0][0]
        self.assertEqual([n for n, _ in informative], ["see_screen"])

    def test_unregistered_action_unchanged(self):
        self._turn("Right away, sir.", [("desk_chat", "It failed.", False)])
        self.assertTrue(self.spoken)
        self.followup.assert_called_once()

    def test_proactive_path_never_voices_a_self_voiced_result(self):
        bc = self.bc
        bc.register_self_voiced("desk_chat")
        self._p(bc, "pause_face_tracking")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_thinking_loop")
        self._p(bc, "generate_proactive_comment", return_value="Hm.")
        self._p(bc, "_reset_see_screen_budget")
        self._p(bc, "parse_and_run_actions", return_value=(
            "", [("desk_chat", "Dialogue finished: 4 lines, done.", True),
                 ("weather_briefing", "Mild.", False)]))
        hist = len(bc.conversation_history)
        self.addCleanup(lambda: bc.conversation_history.__delitem__(
            slice(hist, None)))
        with contextlib.redirect_stdout(io.StringIO()):
            bc._do_proactive_turn({})
        results = self.quip.call_args[0][1]
        self.assertEqual([r[0] for r in results], ["weather_briefing"])

    def test_chain_resolver_never_sees_a_self_voiced_action(self):
        import types
        bc = self.bc
        seen = []
        fake_disp = types.ModuleType("core.dispatcher")
        fake_disp.resolve_and_dispatch = (
            lambda _t, a: seen.append(dict(a)) or None)
        fake_router = types.ModuleType("core.mode_router")
        fake_router.maybe_handle_mode_toggle = lambda _t: None
        fake_router.controlled_dispatch = lambda _t, _a: None
        fake_router.is_in_controlled_mode = lambda: False
        self._p(bc, "maybe_replay_last_action", return_value=None)
        self._p(bc, "_run_fast_paths", return_value=False)
        self._p(bc, "ACTIONS", {"desk_chat": lambda a="": "",
                                "play_music": lambda a="": ""})
        mods = {"core.dispatcher": fake_disp, "core.mode_router": fake_router,
                "skill_custom_voice": types.SimpleNamespace(
                    maybe_switch_backend=lambda _t: None)}
        with mock.patch.dict(bc.sys.modules, mods):
            bc._run_voice_shortcuts("play jazz and have a chat")
            self.assertEqual(sorted(seen[-1]), ["desk_chat", "play_music"])
            bc.register_self_voiced("desk_chat")
            bc._run_voice_shortcuts("play jazz and have a chat")
            self.assertEqual(sorted(seen[-1]), ["play_music"])

    def test_speak_sets_stay_disjoint(self):
        bc = self.bc
        verb = next(iter(bc.SPEAK_RESULT_VERBATIM_ACTIONS))
        info = next(iter(bc.INFORMATIVE_ACTIONS))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(bc.register_self_voiced(verb))
            self.assertFalse(bc.register_self_voiced(info))
        self.assertNotIn(verb.lower(), bc.SELF_VOICED_ACTIONS)

        class Mod:
            SELF_VOICED_ACTIONS = ("desk_chat", verb)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            bc._collect_skill_speak_sets(Mod, "desk")
        self.assertIn("desk_chat", bc.SELF_VOICED_ACTIONS)
        self.assertNotIn(verb.lower(), bc.SELF_VOICED_ACTIONS)
        self.assertIn("REFUSED self-voiced routing for 1", buf.getvalue())

        class Mod2:
            SPEAK_VERBATIM_ACTIONS = ("desk_chat",)
            INFORMATIVE_ACTIONS = ("desk_chat",)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._collect_skill_speak_sets(Mod2, "desk2")
        self.assertNotIn("desk_chat", bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertNotIn("desk_chat", bc.INFORMATIVE_ACTIONS)
        self.assertFalse(bc.SELF_VOICED_ACTIONS
                         & bc.SPEAK_RESULT_VERBATIM_ACTIONS)
        self.assertFalse(bc.SELF_VOICED_ACTIONS & bc.INFORMATIVE_ACTIONS)


# ── a confirmed self-voiced action (the confirmation gate's "yes") ────────
class ConfirmedSelfVoicedTests(_Base):
    """A self-voiced action (a device dialogue) that went through the
    confirmation gate: the owner says "yes", the action runs and speaks every
    line itself — and handle_confirmation_response used to add "Done." on top
    of it (live 2026-09-30). The accept path follows the main path's rule
    (_all_self_voiced): nothing else is spoken for a self-voiced action."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "SELF_VOICED_ACTIONS", set())
        self._p(bc, "SPEAK_RESULT_VERBATIM_ACTIONS",
                set(bc.SPEAK_RESULT_VERBATIM_ACTIONS))
        self._p(bc, "INFORMATIVE_ACTIONS", set(bc.INFORMATIVE_ACTIONS))
        self.spoken = []
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        self.followup = self._p(bc, "get_followup_response",
                                return_value="Following up.")
        self.ran = []
        acts = dict(bc.ACTIONS)
        acts["desk_chat"] = lambda a="": (
            self.ran.append("desk_chat")
            or "Dialogue finished: 4 lines, done.")
        acts["desk_lamp"] = lambda a="": self.ran.append("desk_lamp") or "ok"
        self._p(bc, "ACTIONS", acts)
        self._p(bc, "_pending_confirmation", [])
        bc.register_self_voiced("desk_chat")

    def _confirm(self, *names, answer="yes"):
        for n in names:
            self.bc._pending_confirmation.append((n, ""))
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc.handle_confirmation_response(answer)

    def test_confirmed_self_voiced_action_gets_no_done(self):
        self.assertTrue(self._confirm("desk_chat"))
        self.assertEqual(self.ran, ["desk_chat"])
        self.assertEqual(self.spoken, [],
                         "JARVIS added feedback on top of a self-voiced "
                         "action that already did its own talking")
        self.followup.assert_not_called()
        self.assertEqual(self.bc._pending_confirmation, [])

    def test_a_failure_marker_result_is_not_reported_either(self):
        self.bc.ACTIONS["desk_chat"] = lambda a="": (
            "It failed and could not start.")
        self._confirm("desk_chat")
        self.assertEqual(self.spoken, [])

    def test_mixed_confirmation_says_done_for_the_plain_action_only(self):
        self._confirm("desk_chat", "desk_lamp")
        self.assertEqual(self.ran, ["desk_chat", "desk_lamp"])
        self.assertEqual(self.spoken, ["Done."])

    def test_a_raising_self_voiced_action_is_still_reported(self):
        def boom(_a=""):
            raise RuntimeError("transport gone")
        self.bc.ACTIONS["desk_chat"] = boom
        self._confirm("desk_chat")
        self.assertEqual(len(self.spoken), 1)
        self.assertIn("ran into an error", self.spoken[0])

    def test_a_deferral_result_is_not_swallowed(self):
        prefix = self.bc._ANSWER_FIRST_DEFERRED_PREFIXES[0]
        self.bc.ACTIONS["desk_chat"] = lambda a="": prefix + " not now"
        self._confirm("desk_chat")
        self.assertTrue(self.spoken)

    def test_unregistered_action_still_says_done(self):
        self.bc.SELF_VOICED_ACTIONS.clear()
        self._confirm("desk_chat")
        self.assertEqual(self.spoken, ["Done."])

    def test_declining_still_says_cancelled(self):
        self._confirm("desk_chat", answer="no")
        self.assertEqual(self.ran, [])
        self.assertEqual(self.spoken, ["Cancelled."])


# ── _local_complete ──────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status, content="", text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._content = content
        self.text = text

    def json(self):
        return {"message": {"content": self._content}}


class LocalCompleteTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test:4b")
        self._p(bc, "_ollama_has_model", return_value=True)
        self.after = self._p(bc, "_after_local_post")
        self.post = self._p(bc.requests, "post",
                            return_value=_Resp(200, '{"lines": []}'))

    def call(self, **kw):
        args = dict(temperature=0.9, top_p=0.95, repeat_penalty=1.1,
                    json_schema={"type": "object"})
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._local_complete(
                "You write a short exchange.",
                [{"role": "user", "content": "Topic: toast."}], **args)

    def test_payload_has_no_directive_and_carries_the_options(self):
        self.assertEqual(self.call(), '{"lines": []}')
        payload = self.post.call_args.kwargs["json"]
        system = payload["messages"][0]["content"]
        self.assertEqual(system, "You write a short exchange.")
        self.assertNotIn("YOU ARE RUNNING ON THE LOCAL MODEL", system)
        self.assertEqual(payload["options"]["temperature"], 0.9)
        self.assertEqual(payload["options"]["top_p"], 0.95)
        self.assertEqual(payload["options"]["repeat_penalty"], 1.1)
        self.assertEqual(payload["options"]["num_predict"], 220)
        self.assertEqual(payload["format"], {"type": "object"})
        self.assertEqual(payload["model"], "gemma-test:4b")
        self.assertEqual(payload["messages"][1]["content"], "Topic: toast.")
        self.after.assert_called_once()

    def test_http_400_retries_once_without_format(self):
        self.post.side_effect = [_Resp(400, text="format unsupported"),
                                 _Resp(200, "ok text")]
        self.assertEqual(self.call(), "ok text")
        self.assertEqual(self.post.call_count, 2)
        self.assertIn("format", self.post.call_args_list[0].kwargs["json"])
        self.assertNotIn("format", self.post.call_args_list[1].kwargs["json"])

    def test_error_and_empty_return_none(self):
        self.post.return_value = _Resp(500)
        self.assertIsNone(self.call())
        self.post.return_value = _Resp(200, "   ")
        self.assertIsNone(self.call())
        self.post.side_effect = RuntimeError("down")
        self.assertIsNone(self.call())

    def test_local_only_refuses_non_loopback_and_cloud_tags(self):
        with mock.patch.object(self.bc, "LOCAL_LLM_BASE_URL",
                               "http://llm.example.com:11434"):
            self.assertIsNone(self.call())
            self.post.assert_not_called()
            self.post.return_value = _Resp(200, "remote ok")
            self.assertEqual(self.call(local_only=False), "remote ok")
        self.post.reset_mock()
        with mock.patch.object(self.bc, "_get_local_llm_model",
                               return_value="bigmodel:120b-cloud"):
            self.assertIsNone(self.call())
        self.post.assert_not_called()

    def test_local_llm_off_returns_none(self):
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", False):
            self.assertIsNone(self.call())
        self.post.assert_not_called()


# ── skill_utils / services ───────────────────────────────────────────────
class SkillUtilsKeysTests(_Base):
    KEYS = ("dialogue_ready", "dialogue_session", "speak_line",
            "listen_for_stop", "local_complete", "register_self_voiced",
            "is_self_voiced")

    def test_keys_present_and_wired(self):
        su = self.bc.skill_utils
        for k in self.KEYS:
            self.assertIn(k, su)
            self.assertTrue(callable(su[k]))
        self.assertEqual(su["dialogue_ready"](), "")
        with mock.patch.object(self.bc, "SELF_VOICED_ACTIONS", set()):
            self.assertTrue(su["register_self_voiced"]("desk_chat"))
            self.assertTrue(su["is_self_voiced"]("desk_chat"))

    def test_services_wrappers(self):
        from core.services import JarvisServices
        svc = JarvisServices.from_skill_utils(self.bc.skill_utils)
        self.assertEqual(svc.dialogue_ready(), "")
        with mock.patch.object(self.bc, "SELF_VOICED_ACTIONS", set()):
            self.assertTrue(svc.register_self_voiced("desk_chat"))
            self.assertTrue(svc.is_self_voiced("desk_chat"))
        with contextlib.redirect_stdout(io.StringIO()):
            with svc.dialogue_session(_SRC, max_s=20) as h:
                self.assertAlmostEqual(h.max_s, 20.0)
        empty = JarvisServices.from_skill_utils({})
        self.assertEqual(empty.dialogue_ready(), "disabled")
        self.assertEqual(empty.speak_line("x"), "failed")
        self.assertIsNone(empty.listen_for_stop(lambda: True))
        self.assertIsNone(empty.local_complete("s", []))
        self.assertFalse(empty.register_self_voiced("x"))
        with self.assertRaises(RuntimeError):
            empty.dialogue_session(_SRC)

    def test_wired_and_unwired_refusals_are_one_class_with_a_reason(self):
        # The monolith's refusal and the unwired wrapper's "disabled" are the
        # SAME class (core.dialogue.DialogueUnavailable), so one except
        # clause / one getattr(exc, "reason") serves both paths.
        from core import dialogue as dlg
        from core.services import JarvisServices
        self.assertIs(self.bc.DialogueUnavailable, dlg.DialogueUnavailable)
        caught = []
        self.bc._tts_muted[0] = True
        wired = JarvisServices.from_skill_utils(self.bc.skill_utils)
        for svc in (wired, JarvisServices.from_skill_utils({})):
            try:
                with svc.dialogue_session(_SRC):
                    self.fail("must not enter")
            except dlg.DialogueUnavailable as exc:
                caught.append(getattr(exc, "reason", None))
        self.assertEqual(caught, ["tts_muted", "disabled"])


# ── a background mic buffer yields to a starting dialogue ────────────────
@requires_monolith
class BackgroundBufferYieldsTests(_Base):
    """Live 2026-10-01 13:48: "Jarvis, talk to <the robot>" ended "0 lines,
    error" with "[dialogue] mic capture still live; line not spoken". The
    standby lyric loop's 3 s get_mic_buffer (Path B) had started just before
    the dialogue; _speak_line waits only ~1 s for the mic, so the FIRST line
    failed and the banter with it. Path B now yields once a dialogue is active,
    as it already yields to record_speech."""

    def _capture(self, flip_dialogue):
        import numpy as np
        bc = self.bc
        self._p(bc, "_record_speech_active", [False])
        self._p(bc, "_pathb_mic_active", [False])
        if hasattr(self, "_hide_wake_listener"):
            self._hide_wake_listener()
        dlg = bc._dialogue_active
        frame = np.ones((1024, 1), dtype=np.float32) * 0.2

        class _Stream:
            def __init__(s, *a, **k):
                s.cb = k["callback"]

            def start(s):
                s.cb(frame, 1024, None, None)
                if flip_dialogue:
                    dlg[0] = True            # the banter starts mid-capture
                s.cb(frame, 1024, None, None)

        self._p(bc.sd, "InputStream", side_effect=_Stream)
        self._p(bc, "get_input_device", return_value=3)
        owned_at_close = []
        self._p(bc, "_safe_close_stream",
                side_effect=lambda st, *a, **k: owned_at_close.append(
                    bc._pathb_mic_active[0]))
        t0 = time.monotonic()
        out = bc.get_mic_buffer(0.25, sample_rate=16000)
        return out, time.monotonic() - t0, owned_at_close

    def test_the_buffer_lets_go_when_a_dialogue_starts(self):
        out, took, owned_at_close = self._capture(flip_dialogue=True)
        self.assertIsNone(out)                     # yielded before collecting
        self.assertLess(took, 1.0)                 # well inside _speak_line's 1 s
        self.assertEqual(owned_at_close, [True])   # closed, THEN released
        self.assertFalse(self.bc._pathb_mic_active[0])

    def test_without_a_dialogue_the_buffer_is_unchanged(self):
        out, _took, _owned = self._capture(flip_dialogue=False)
        self.assertIsNotNone(out)
        self.assertEqual(out.size, 2048)


if __name__ == "__main__":
    unittest.main()
