"""Smart Turn end of turn wired into the REAL record_speech (speed plan R7,
2026-10-02 — the SHADOW release).

core/endpointing.py's models and EotDecider are covered CI-light by
tests/test_endpointing.py. This file drives the real capture loop through
tests/test_speculative_stt.py's fake mic stream and its _run_capture — reused,
never re-implemented (the speculative-STT "green by re-implementation" trap)
— with the real SileroStream / SmartTurn classes over FAKE onnxruntime
sessions (test_endpointing's _StreamSession / _TurnSession: p 0.9 for a loud
window, a scripted Smart Turn p). No model file, no real model run.

Pinned here:
  * shadow returns byte-identical audio to off, on a turn it WOULD have cut;
  * one numeric [eot-shadow] line per ACCEPTED owner turn (held from the
    capture's end until the main loop's "You:" line, so a capture the loop
    drops — TV refused in wake-word mode, a standby line with no wake word —
    logs none), and eot / st_p / st_n on the [turn-timing] line;
  * Silero and Smart Turn hear what Whisper hears: the PROCESSED clip from
    its first pre-roll sample; Smart Turn the clip so far x ONE gain (as
    Whisper's clip gets, from the peak so far), Silero each chunk x the gain
    so far; each capture starts Silero afresh;
  * a mid-sentence pause after a would-be end reports resumed=1;
  * mode 'off' never imports a model runtime; no capture loads a model (only
    the boot warmer does);
  * a fault in the decider, Silero, Smart Turn or the hooks themselves (even
    one whose text cannot be formatted) never breaks a capture or leaves the
    mic claimed, and turns Smart Turn off for the session with one line;
  * a Smart Turn check never holds the capture thread past one chunk (64 ms)
    — one that does turns Smart Turn off — and runs after the speculative
    snapshot of its chunk;
  * in-turn captures (no smart_endpoint; off the main loop's thread) never
    build a decider, and only the two main-loop listens pass it;
  * mode 'on' (not shipped) ends a complete turn early, with eot=st.

Run (full-deps tier):
    python -m unittest tests.monolith.test_monolith_smart_turn
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import inspect
import io
import os
import re
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
# Modules, not names: discovery must not collect their TestCases again here.
from tests import test_endpointing as _ept
from tests import test_speculative_stt as _spec

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_CHUNK = _spec._CHUNK                 # 1,024 samples = 64 ms
_PRE = _spec._PRE_BUFFER              # 12 pre-roll chunks join at the trip
_MS = 64
_LOUD = 0.1     # RMS: loud for the fake Silero at ANY gain (|mean| 0.09)
# A complete turn: room tone (fills the pre-roll), 30 voiced chunks, then
# silence well past the 21-chunk break.
_TURN = [0.001] * 20 + [_LOUD] * 30 + [0.0] * 25
# The same words with a mid-sentence pause (8 silent chunks) before 20 more.
_PAUSED = [0.001] * 20 + [_LOUD] * 30 + [0.0] * 8 + [_LOUD] * 20 + [0.0] * 25
# A turn that starts soft and gets loud: the capture's auto-gain falls from
# x10 (peak 0.02) to x2.5 (peak 0.1) half-way through.
_RISING = [0.001] * 20 + [0.02] * 10 + [_LOUD] * 20 + [0.0] * 25
# Smart Turn first runs at the 4th silent chunk (SMART_TURN_MIN_SILENCE_S).
_FIRE_MS = (_PRE + 30 + 4) * _MS                       # 2,944
_TURN_MS = (_PRE + 30 + 21) * _MS                      # 4,032
_PAUSED_MS = (_PRE + 30 + 8 + 20 + 21) * _MS           # 5,824
_SHADOW_LINE = re.compile(r"\[eot-shadow\] fire_ms=(?:\d+|-) "
                          r"p=(?:[01]\.\d{3}|-) resumed=[01] actual_ms=\d+")


class _RecTurn(_ept.ep.SmartTurn):
    """SmartTurn that keeps a copy of every clip it is asked about (and, when
    `events` is a list, notes ("check", clip chunks) in it)."""

    def __init__(self, heard, *a, **k):
        super().__init__(*a, **k)
        self.heard = heard
        self.events = None

    def predict(self, audio, sample_rate=16000):
        import numpy as np
        self.heard.append(np.array(audio, dtype=np.float32, copy=True))
        if self.events is not None:
            self.events.append(("check", len(audio) // _CHUNK))
        return super().predict(audio, sample_rate)


class _NoStr(Exception):
    """An exception whose text itself cannot be formatted."""

    def __str__(self):
        raise ValueError("str() of this exception fails")


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    MODE = "shadow"
    P = 0.99

    def setUp(self):
        super().setUp()
        import numpy as np
        from core import turn_timing as tt
        self.np, self.tt = np, tt
        bc = self.bc
        bc._spec_stt_reset()
        self.addCleanup(bc._spec_stt_reset)
        ep = bc._endpointing
        self.vsess = _ept._StreamSession()
        self.tsess = _ept._TurnSession(ps=self.P)
        self.loads = {"silero": 0, "turn": 0}
        self.silero_load_error = None
        self.heard_by_turn = []
        self._p(bc, "_eot_stream", ep.SileroStream(
            model_path="fake.onnx", session_factory=self._silero_factory))
        self._p(bc, "_eot_turn", _RecTurn(
            self.heard_by_turn, "fake.onnx",
            session_factory=self._turn_factory,
            features=lambda x: np.zeros((80, 800), np.float32),
            clock=self.tsess.clock))
        self._p(bc, "_eot_state", {"ready": False, "off": "", "logged": False})
        # The held [eot-shadow] line starts empty, and a Smart Turn check is
        # timed on the fake session's clock (each run costs tsess.costs).
        self._p(bc, "_eot_shadow_pending", [None])
        self._p(bc, "_eot_clock", self.tsess.clock)
        self._p(bc, "SMART_TURN_MODE", self.MODE)
        self.timing = tt.TurnTiming(print_fn=lambda line: None)
        self._p(bc, "_turn_timing", self.timing)
        # Every EotDecider a capture builds.
        self.made = []
        made = self.made

        class Spy(ep.EotDecider):
            def __init__(s, *a, **k):
                super().__init__(*a, **k)
                made.append(s)

        self.Spy = Spy
        self._p(ep, "EotDecider", Spy)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _silero_factory(self, _path):
        self.loads["silero"] += 1
        if self.silero_load_error is not None:
            raise self.silero_load_error
        return self.vsess

    def _turn_factory(self, _path):
        self.loads["turn"] += 1
        return self.tsess

    def _warm(self):
        """The real boot warmer, over the fake sessions."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc._warm_smart_turn()
        self.assertTrue(self.bc._eot_state["ready"])
        del self.vsess.feeds[:]           # the warm window is no turn's

    def _capture(self, rms_seq, smart=True, accept=True, **extra):
        """(audio, log) of ONE real record_speech capture of `rms_seq`, as
        the main loop's listen calls it (smart_endpoint=`smart`). `accept`:
        the main loop then accepts it as the owner's turn — its "You:" line,
        after which _eot_shadow_flush() logs the turn's [eot-shadow] line."""
        bc = self.bc
        real = bc.record_speech

        def rec(timeout=None, **kw):
            return real(timeout, smart_endpoint=smart, **extra, **kw)

        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", rec), \
                contextlib.redirect_stdout(out):
            audio, _decodes = _spec.SpeculativeRealCaptureLoopTests \
                ._run_capture(self, rms_seq)
            if accept:
                bc._eot_shadow_flush()
        return audio, out.getvalue()

    def _flush(self):
        """What _eot_shadow_flush() logs now (the main loop's "You:")."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.bc._eot_shadow_flush()
        return out.getvalue()

    def _auto_gain_on(self):
        """The shipped auto-gain knobs, whatever this box's settings say."""
        from core import config as cfg
        for name, val in (("CAPTURE_AUTO_GAIN_ENABLED", True),
                          ("CAPTURE_AUTO_GAIN_TARGET_PEAK", 0.25),
                          ("CAPTURE_AUTO_GAIN_MAX", 10.0),
                          ("CAPTURE_AUTO_GAIN_NOISE_FLOOR", 0.005)):
            self._p(cfg, name, val)

    def _silero_heard(self, feeds=None):
        """Every sample the capture stream's Silero was fed, in order."""
        feeds = self.vsess.feeds if feeds is None else feeds
        return self.np.concatenate(
            [f["input"][:, _ept.ep.CONTEXT:].reshape(-1) for f in feeds])

    def _off_audio(self, rms_seq):
        """The same capture with no Smart Turn at all (mode 'off')."""
        with mock.patch.object(self.bc, "SMART_TURN_MODE", "off"):
            audio, log = self._capture(rms_seq)
        self.assertNotIn("[eot", log)
        return audio

    @staticmethod
    def _shadow_lines(log):
        return [ln.strip() for ln in log.splitlines() if "[eot-shadow]" in ln]

    @staticmethod
    def _eot_lines(log):
        return [ln.strip() for ln in log.splitlines() if "[eot]" in ln]

    def _turn_fields(self, since):
        """The [turn-timing] fields of the turn that begins at this capture's
        VAD break, the way _capture_utterance begins it."""
        self.timing.begin_voice(since)
        return self.tt.parse_line(self.timing.emit())


class ShadowCaptureTests(_Base):
    def test_shadow_returns_audio_identical_to_off(self):
        # Mutation-proven (2026-10-02): a shadow mode that breaks where Smart
        # Turn fires (_EotCapture.feed returning True on the fire) turns this
        # red — the shadow audio is cut 1,088 ms short.
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.assertEqual(self.made, [])
        audio, log = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        # It WOULD have cut this turn, so "identical" is not vacuous.
        self.assertEqual(self.made[0].record().fire_ms, _FIRE_MS)
        self.assertEqual(len(audio), len(off))
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(audio.dtype, off.dtype)

    def test_one_numeric_shadow_line_and_the_turn_stats(self):
        self._warm()
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        lines = self._shadow_lines(log)
        self.assertEqual(lines, [f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 "
                                 f"resumed=0 actual_ms={_TURN_MS}"])
        # actual_ms is the returned clip, pre-roll included.
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "0.99", "1"))
        self.assertEqual(d["vad_break"], "0")
        self.assertEqual(self._eot_lines(log), [])

    def test_the_shadow_line_never_carries_what_was_said(self):
        # The whole line is numbers in a fixed shape: no field can hold text.
        # Whatever the turn did (fired, never fired, Smart Turn failed), the
        # line is a full match of that shape.
        self._warm()
        logs = [self._capture(_TURN)[1], self._capture(_PAUSED)[1]]
        self.tsess.ps = [0.2]
        logs.append(self._capture(_TURN)[1])
        self.tsess.fail = RuntimeError("the owner said: open the pod bay")
        logs.append(self._capture(_TURN)[1])
        lines = [ln for log in logs for ln in self._shadow_lines(log)]
        self.assertEqual(len(lines), 4)
        for ln in lines:
            self.assertRegex(ln, r"^" + _SHADOW_LINE.pattern + r"$")
        self.assertNotIn("pod bay", "".join(lines))

    def test_a_pause_after_a_would_be_end_is_resumed_and_kept(self):
        np = self.np
        self._warm()
        off = self._off_audio(_PAUSED)
        audio, log = self._capture(_PAUSED)
        self.assertTrue(np.array_equal(audio, off))   # the words after it
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 resumed=1 "
            f"actual_ms={_PAUSED_MS}"])

    def test_below_the_threshold_it_keeps_checking_and_never_fires(self):
        self._warm()
        self.tsess.ps = [0.5]
        since = self.timing.now()
        _audio, log = self._capture(_TURN)
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=0.500 resumed=0 actual_ms={_TURN_MS}"])
        d = self._turn_fields(since)
        # Checks at silent chunks 4, 9, 14 and 19 (five apart).
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "0.5", "4"))

    def test_silero_and_smart_turn_hear_what_whisper_hears(self):
        # Whisper gets apply_capture_auto_gain(clip, peak), and the clip is
        # the PROCESSED capture (_process_capture_chunk), not the raw mic.
        # Silero must get the same samples from the clip's FIRST pre-roll
        # sample on, and Smart Turn the clip so far — not the raw or the
        # un-gained capture. The processing stage here is a fixed x0.5, so a
        # hook fed the raw chunk shows (_run_capture alone turns processing
        # off: processed == raw there).
        # Mutation-proven (2026-10-02 repair): feeding `data` instead of
        # chunks[-1] in record_speech's voiced and silent branches turns this
        # red; with processing off it stayed green.
        np = self.np
        bc = self.bc
        self._auto_gain_on()
        self._p(bc, "_process_capture_chunk",
                lambda d, sr, skip_ns=False: d * np.float32(0.5))
        self._warm()
        audio, _log = self._capture(_TURN)
        raw = np.concatenate([_spec._mic_frame(r) for r in _TURN])
        self.assertTrue(np.array_equal(
            audio, raw[8 * _CHUNK:8 * _CHUNK + len(audio)] * np.float32(0.5)))
        whisper, gain = bc.apply_capture_auto_gain(
            audio, bc._last_recording_peak)
        self.assertGreater(gain, 2.0)          # a real gain, not x1
        silero = self._silero_heard()
        # Silero is fed up to the would-be end (shadow stops listening there).
        self.assertEqual(len(silero), _FIRE_MS * 16)
        self.assertTrue(np.array_equal(silero, whisper[:len(silero)]))
        self.assertEqual(len(self.heard_by_turn), 1)
        self.assertTrue(np.array_equal(self.heard_by_turn[0],
                                       whisper[:_FIRE_MS * 16]))

    def test_smart_turn_hears_the_clip_at_one_gain_as_whisper_does(self):
        # A turn that starts soft and gets loud. Whisper's clip is scaled
        # ONCE, by the capture's final peak (x2.5 here). Smart Turn must hear
        # exactly that — the clip so far x the gain from the peak so far,
        # which is what Whisper would get had the turn ended at the check —
        # not each chunk x the gain at ITS time (x10 for the soft start, ~4x
        # too loud). Silero is streamed, so it hears each chunk x the gain so
        # far.
        np = self.np
        bc = self.bc
        self._auto_gain_on()
        self._warm()
        audio, log = self._capture(_RISING)
        whisper, gain = bc.apply_capture_auto_gain(
            audio, bc._last_recording_peak)
        self.assertLess(gain, 3.0)             # the loud part's gain
        fire = (_PRE + 30 + 4) * _CHUNK        # pre-roll, voice, 4 silent
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms={fire * 1000 // 16000} p=0.990 resumed=0 "
            f"actual_ms={len(audio) * 1000 // 16000}"])
        self.assertEqual(len(self.heard_by_turn), 1)
        heard = self.heard_by_turn[0]
        self.assertEqual(len(heard), fire)
        self.assertTrue(np.array_equal(heard, whisper[:fire]))
        # Silero: the soft start at the running gain (x10, not x2.5).
        soft = slice(0, (_PRE + 10) * _CHUNK)
        running, g10 = bc.apply_capture_auto_gain(
            audio[soft], float(np.sqrt(np.mean(audio[_PRE * _CHUNK:
                                                     (_PRE + 1) * _CHUNK]
                                               ** 2))))
        self.assertEqual(g10, 10.0)
        self.assertTrue(np.array_equal(self._silero_heard()[soft], running))

    def test_each_capture_starts_silero_afresh(self):
        # The capture stream's Silero carries its recurrent state and context
        # across chunks; a new capture must not start from the previous one's
        # (which may have been the TV). The fake session makes it visible:
        # h_out = h_in + windows, c_out = c_in - windows.
        # Mutation-proven (2026-10-02 repair): deleting _eot_begin's
        # _eot_stream.reset() turns this red.
        np = self.np
        self._warm()
        self._capture(_TURN)
        n = len(self.vsess.feeds)
        self.assertGreater(n, 1)
        self.assertTrue(np.any(self.vsess.feeds[-1]["h"] != 0))
        self._capture(_TURN)
        first = self.vsess.feeds[n]
        self.assertTrue(np.all(first["h"] == 0))
        self.assertTrue(np.all(first["c"] == 0))

    def test_a_capture_cut_at_the_utterance_ceiling_is_eot_max(self):
        self._warm()
        self._p(self.bc, "MAX_RECORDING_SECS", -1.0)
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        n = (_PRE + 1) * _MS                  # pre-roll + the tripping chunk
        self.assertEqual(len(audio) * 1000 // 16000, n)
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={n}"])
        self.assertEqual(self._turn_fields(since)["eot"], "max")

    def test_a_capture_with_no_speech_prints_nothing(self):
        # A listen that times out is no turn: no line, no stats.
        self._warm()
        bc = self.bc
        real = bc.record_speech
        out = io.StringIO()
        with mock.patch.object(
                bc, "record_speech",
                lambda timeout=None, **kw: real(0.0, smart_endpoint=True)), \
                contextlib.redirect_stdout(out):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, [0.001] * 5, tail_silence=0)
        self.assertIsNone(audio)
        self.assertEqual(self._shadow_lines(out.getvalue()), [])


class OwnerTurnLineTests(_Base):
    """One [eot-shadow] line per OWNER TURN, not per capture: the main loop's
    idle listens capture everything the room says (TV, other people, JARVIS
    himself), and a line per capture made the shadow data — the resumed=1
    rate tools/turn_latency_report.py counts, the fire rate — describe the
    room, not the owner. The line is held from the capture's end and logged
    right after the main loop's "You:" line; the next loop pass drops it."""

    def test_the_capture_holds_its_line_until_the_turn_is_accepted(self):
        self._warm()
        _audio, log = self._capture(_TURN, accept=False)
        self.assertEqual(self._shadow_lines(log), [])
        self.assertEqual(self._shadow_lines(self._flush()), [
            f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 resumed=0 "
            f"actual_ms={_TURN_MS}"])
        self.assertEqual(self._flush(), "")            # once per turn

    def test_the_next_loop_pass_drops_a_turn_that_was_never_accepted(self):
        self._warm()
        self._capture(_TURN, accept=False)
        self.bc._tt_loop_top(None)
        self.assertEqual(self._flush(), "")
        self._capture(_TURN, accept=False)
        self.bc._tt_loop_top("a typed command")
        self.assertEqual(self._flush(), "")

    def test_an_owner_capture_replaces_the_line_an_in_turn_one_keeps_it(self):
        bc = self.bc
        self._warm()
        self._capture(_TURN, accept=False)
        self.tsess.ps = [0.2]
        self._capture(_TURN, accept=False)             # the newer turn
        self._capture(_TURN, smart=False, accept=False)  # an in-turn capture
        self.assertEqual(self._shadow_lines(self._flush()), [
            f"[eot-shadow] fire_ms=- p=0.200 resumed=0 actual_ms={_TURN_MS}"])
        # An owner listen that hears nothing (no line of its own) still
        # drops the line before it.
        self._capture(_TURN, accept=False)
        real = bc.record_speech
        with mock.patch.object(
                bc, "record_speech",
                lambda timeout=None, **kw: real(0.0, smart_endpoint=True)), \
                contextlib.redirect_stdout(io.StringIO()):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, [0.001] * 5, tail_silence=0)
        self.assertIsNone(audio)
        self.assertEqual(self._flush(), "")

    def _utterance(self, rms_seq, heard_text):
        """One pass of the main loop's normal-mode turn, its control flow as
        main() has it (main() cannot run in a test — the wiring test below
        pins it there): the loop top, _capture_utterance through the REAL
        record_speech and fake mic (Whisper's text faked), the wake-word gate
        (_bg_gate_for_turn), then the accepted turn's "You:" mark. Returns
        (refused, log)."""
        bc = self.bc
        real = bc.record_speech
        kws = []

        def listen(*a, **kw):
            kws.append(dict(kw))
            return real(*a, **kw)

        def main_loop_pass(timeout=None, **_kw):
            bc._tt_loop_top(None)
            with mock.patch.object(bc, "record_speech", listen):
                return bc._capture_utterance(None, {})

        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_transcribe_capture", return_value=(
            heard_text, {"no_speech_prob": 0.0, "avg_logprob": -0.1}))
        self._p(bc, "_require_wake_runtime", True)
        from core.followup_window import FollowupWindow
        self._p(bc, "_followup_window", FollowupWindow(0))
        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", main_loop_pass), \
                contextlib.redirect_stdout(out):
            cap, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, rms_seq)
            self.assertEqual(cap[0], heard_text)
            refused, _why = bc._bg_gate_for_turn(cap[0], False)
            if not refused:
                bc._tt("mark", "you")
                bc._eot_shadow_flush()
            else:
                bc._tt_loop_top(None)     # main() `continue`s to the top
                bc._eot_shadow_flush()    # (no "You:" — nothing to log)
        self.assertEqual(kws[0].get("smart_endpoint"), True)
        return refused, out.getvalue()

    def test_a_refused_capture_logs_no_line_an_accepted_one_logs_one(self):
        self._warm()
        refused, log = self._utterance(_TURN, "what is on the telly")
        self.assertTrue(refused)                       # wake-word mode
        self.assertEqual(self._shadow_lines(log), [])
        refused, log = self._utterance(_TURN, "Jarvis, what time is it")
        self.assertFalse(refused)
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms={_FIRE_MS} p=0.990 resumed=0 "
            f"actual_ms={_TURN_MS}"])
        self.assertEqual(len(self.made), 2)            # both were measured

    def test_a_standby_capture_that_wakes_nothing_logs_no_line(self):
        bc = self.bc
        self._warm()
        for name in ("_sleep_mode", "_standby_mode"):
            self._p(bc, name, [True])
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_audio_music_feed")
        for name in ("_device_speech_ignored", "_dialogue_hold_ignored",
                     "_self_echo_ignored"):
            self._p(bc, name, return_value=False)
        self._p(bc, "_ambient_learning", [False])
        self._p(bc, "_transcribe_capture", return_value=(
            "just the telly", {"no_speech_prob": 0.0, "avg_logprob": -0.1}))
        real = bc.record_speech
        kws = []

        def listen(*a, **kw):
            kws.append(dict(kw))
            return real(*a, **kw)

        def standby_pass(timeout=None, **_kw):
            bc._tt_loop_top(None)
            with mock.patch.object(bc, "record_speech", listen):
                return bc._handle_sleep_standby(None)

        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", standby_pass), \
                contextlib.redirect_stdout(out):
            cap, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, _TURN)
            bc._tt_loop_top(None)
            bc._eot_shadow_flush()
        self.assertIsNone(cap)
        self.assertEqual(kws[0].get("smart_endpoint"), True)
        self.assertEqual(len(self.made), 1)
        self.assertEqual(self._shadow_lines(out.getvalue()), [])

    def test_main_logs_the_line_right_after_the_accepted_turns_you_mark(self):
        src = inspect.getsource(self.bc.main)
        mark = '_tt("mark", "you")'
        flush = "_eot_shadow_flush()"
        self.assertEqual((src.count(mark), src.count(flush)), (1, 1))
        self.assertLess(src.index('print(f"  You:    {text}")'),
                        src.index(mark))
        between = src[src.index(mark) + len(mark):src.index(flush)]
        self.assertEqual(between.strip(), "")
        # ...and the loop top that drops an unaccepted turn's line runs
        # before either capture of the pass.
        top = src.index("_tt_loop_top(_injected_text)")
        self.assertLess(top, src.index("_handle_sleep_standby(_injected_text)"))
        self.assertLess(top, src.index("_capture_utterance(_injected_text, "))


class ModeOffTests(_Base):
    MODE = "off"

    @contextlib.contextmanager
    def _import_spy(self):
        """Every import STATEMENT of onnxruntime / faster_whisper (even one
        already in sys.modules) while the block runs."""
        tried = []
        real_import = builtins.__import__

        def spy(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split(".")[0] in ("onnxruntime",
                                                     "faster_whisper"):
                tried.append(name.split(".")[0])
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", spy):
            yield tried

    def test_mode_off_never_imports_a_model_runtime(self):
        bc = self.bc
        ep = bc._endpointing
        np = self.np
        # The REAL loaders (default sessions) over a model path that does not
        # exist, and marked warm: in 'off', the mode alone must keep them
        # untouched.
        self._p(bc, "_eot_stream", ep.SileroStream(model_path="x.onnx"))
        self._p(bc, "_eot_turn", ep.SmartTurn("x.onnx"))
        bc._eot_state["ready"] = True
        with self._import_spy() as tried:
            off, log = self._capture(_TURN)
        self.assertEqual((tried, self.made), ([], []))
        self.assertNotIn("[eot", log)
        self.assertEqual(bc._eot_stream.failed, "")
        # The spy is live: the same capture in shadow does try to load
        # Silero (onnxruntime) — and, Silero being unavailable, the capture
        # is unchanged, ends nothing early and turns Smart Turn off once.
        bc.SMART_TURN_MODE = "shadow"
        with self._import_spy() as tried:
            audio, log = self._capture(_TURN)
        self.assertIn("onnxruntime", tried)
        self.assertNotIn("faster_whisper", tried)   # Smart Turn never asked
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(len(self._eot_lines(log)), 1)
        self.assertIn("smart turn off for this session (silero: load failed",
                      log)
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)            # none for this one
        self.assertNotIn("[eot", log2)

    def test_mode_off_registers_no_boot_warmer(self):
        bc = self.bc
        reg = []
        self._p(bc, "_boot_warmers", reg)
        for mode, want in (("off", []), ("true", []), ("", []),
                           (" Shadow ", [("smart-turn", bc._warm_smart_turn)]),
                           ("on", [("smart-turn", bc._warm_smart_turn)])):
            del reg[:]
            with mock.patch.object(bc, "SMART_TURN_MODE", mode):
                bc._eot_register_warmer()
            self.assertEqual(reg, want, mode)


@requires_monolith
class ShippedRegistrationTests(MonolithGlobalsTestCase):
    def test_the_import_registered_the_warmer_iff_mode_is_not_off(self):
        bc = self.bc
        entry = ("smart-turn", bc._warm_smart_turn)
        if bc._eot_mode() == "off":
            self.assertNotIn(entry, bc._boot_warmers)
        else:
            self.assertIn(entry, bc._boot_warmers)
        self.assertLessEqual(
            sum(1 for n, _ in bc._boot_warmers if n == "smart-turn"), 1)

    def test_a_check_is_timed_on_the_real_clock(self):
        # The tests time checks on the fake session clock; the shipped one is
        # the wall-clock-independent perf counter.
        self.assertIs(self.bc._eot_clock, time.perf_counter)


class FaultTests(_Base):
    def test_a_decider_fault_never_breaks_a_capture_and_latches_off(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)

        class Boom(self.Spy):
            def update(s, chunk, silence_n):
                if s._chunks >= 20:
                    raise RuntimeError("decider bug")
                return super().update(chunk, silence_n)

        self._p(self.bc._endpointing, "EotDecider", Boom)
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (capture hook failed: "
            "RuntimeError: decider bug)"])
        built = len(self.made)
        audio2, log2 = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio2, off))
        self.assertEqual(len(self.made), built)       # no decider any more
        self.assertNotIn("[eot", log2)

    def test_a_decider_that_cannot_be_built_never_breaks_a_capture(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)

        def boom(*a, **k):
            raise TypeError("bad knob")

        self._p(self.bc._endpointing, "EotDecider", boom)
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertIn("smart turn off for this session (capture hook "
                      "failed: TypeError: bad knob)", log)

    def _no_str_fault(self, where):
        """A capture whose Smart Turn hook raises _NoStr at `where`: the
        capture is unchanged, the mic is released, and the one [eot] line
        names the exception's type (its text cannot be formatted)."""
        np = self.np
        bc = self.bc
        self._warm()
        off = self._off_audio(_TURN)
        saved = bc._record_speech_active[0]
        self.addCleanup(bc._record_speech_active.__setitem__, 0, saved)
        Spy = self.Spy

        class Bad(Spy):
            def __init__(s, *a, **k):
                if where == "build":
                    raise _NoStr()
                super().__init__(*a, **k)

            def update(s, chunk, silence_n):
                if where == "feed" and s._chunks >= 20:
                    raise _NoStr()
                return super().update(chunk, silence_n)

            def record(s, eot="rms"):
                if where == "finish":
                    raise _NoStr()
                return super().record(eot)

        self._p(bc._endpointing, "EotDecider", Bad)
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertFalse(bc._record_speech_active[0])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (capture hook failed: "
            "_NoStr)"])
        self.assertEqual(self._shadow_lines(log), [])

    def test_an_unprintable_fault_building_the_decider_never_breaks_it(self):
        self._no_str_fault("build")

    def test_an_unprintable_fault_feeding_the_decider_never_breaks_it(self):
        self._no_str_fault("feed")

    def test_an_unprintable_fault_finishing_the_turn_never_breaks_it(self):
        self._no_str_fault("finish")

    def test_a_raise_out_of_the_hook_setup_still_frees_the_mic(self):
        # _eot_begin never raises; if it ever did, the capture's own finally
        # (stream closed, then _record_speech_active dropped) must still run:
        # a stuck flag defers every device re-enumeration and refuses every
        # get_mic_buffer claim for the life of the process (record_speech's
        # H-8 note).
        bc = self.bc
        self._warm()
        saved = bc._record_speech_active[0]
        self.addCleanup(bc._record_speech_active.__setitem__, 0, saved)
        marks = []
        real_mark = bc._filler_capture_mark

        def mark(*a, **k):
            marks.append(k.get("wait", False))
            return real_mark(*a, **k)

        self._p(bc, "_filler_capture_mark", mark)
        self._p(bc, "_eot_begin", side_effect=RuntimeError("setup bug"))
        with self.assertRaises(RuntimeError):
            self._capture(_TURN)
        self.assertFalse(bc._record_speech_active[0])
        # The entry mark, then the capture's finally (stream closed first).
        self.assertEqual(marks, [True, False])

    def test_a_fault_in_the_capture_predict_hook_latches_off_once(self):
        # A fault in R7's own predict closure (around SmartTurn.predict,
        # which itself never raises) used to be swallowed by the decider for
        # that turn only: no line, and every later turn built a decider and
        # logged p=- forever.
        np = self.np
        bc = self.bc
        self._warm()
        off = self._off_audio(_TURN)
        self._p(bc._eot_turn, "predict", side_effect=RuntimeError("hook bug"))
        logs = []
        for _ in range(3):
            audio, log = self._capture(_TURN)
            self.assertTrue(np.array_equal(audio, off))
            logs.append(log)
        self.assertEqual(
            [ln for log in logs for ln in self._eot_lines(log)],
            ["[eot] smart turn off for this session (capture hook failed: "
             "RuntimeError: hook bug)"])
        self.assertEqual(len(self.made), 1)
        self.assertEqual(self._shadow_lines(logs[0]), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])

    def test_silero_failing_mid_session_changes_nothing_and_latches(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.vsess.fail = RuntimeError("ort")
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self.heard_by_turn, [])      # never asked
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (silero: run failed: "
            "RuntimeError: ort)"])
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("rms", "-", "0"))
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertNotIn("[eot", log2)

    def test_smart_turn_failing_mid_session_changes_nothing_and_latches(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.tsess.fail = RuntimeError("ort")
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (smart turn: run failed: "
            "RuntimeError: ort)"])
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertNotIn("[eot", log2)


class CaptureThreadCostTests(_Base):
    """A Smart Turn check runs on the capture thread, inside one 64 ms chunk.
    It must never make the loop fall behind the microphone — a check late in
    the silence would delay the fixed 21-chunk end of the turn, which shadow
    promises to leave alone — nor delay the speculative snapshot of its
    chunk."""

    def test_a_check_longer_than_one_chunk_turns_smart_turn_off(self):
        np = self.np
        self._warm()
        off = self._off_audio(_TURN)
        self.tsess.costs = [0.070]       # the turn's first check: 70 ms
        audio, log = self._capture(_TURN)
        self.assertTrue(np.array_equal(audio, off))
        self.assertEqual(self._eot_lines(log), [
            "[eot] smart turn off for this session (too slow: a check took "
            "70 ms, over one 64 ms capture chunk)"])
        # That check's p is not used: no would-be end, no p.
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms=- p=- resumed=0 actual_ms={_TURN_MS}"])
        self.assertEqual(self.bc._eot_turn.failed, "")   # 70 < its 150 ms
        _a, log2 = self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertNotIn("[eot", log2)

    def test_a_check_inside_one_chunk_is_kept(self):
        self._warm()
        self.tsess.costs = [0.060]
        self.tsess.ps = [0.5]
        _audio, log = self._capture(_TURN)
        _audio, log2 = self._capture(_TURN)
        self.assertEqual(self._eot_lines(log + log2), [])
        self.assertEqual(len(self.made), 2)
        self.assertEqual(self._shadow_lines(log2), [
            f"[eot-shadow] fire_ms=- p=0.500 resumed=0 actual_ms={_TURN_MS}"])

    def test_the_speculative_snapshot_goes_before_a_check_on_its_chunk(self):
        # Speculative STT (off by default) snapshots at silent chunk 7; a
        # Smart Turn check landing on the same chunk must not delay it.
        bc = self.bc
        self._warm()
        self._p(bc, "_SPECULATIVE_STT", True)
        # The first check, at SMART_TURN_MIN_SILENCE_S, on silent chunk 7.
        self._p(bc, "SMART_TURN_MIN_SILENCE_S", 7 * _MS / 1000.0)
        events = []
        bc._eot_turn.events = events
        real_start = bc._spec_stt_start

        def start(snapshot, peak_rms, n_chunks):
            events.append(("spec", n_chunks))
            return real_start(snapshot, peak_rms, n_chunks)

        self._p(bc, "_spec_stt_start", start)
        _audio, log = self._capture(_TURN)
        n = _PRE + 30 + 7
        self.assertEqual(events, [("spec", n), ("check", n)])
        self.assertEqual(self._shadow_lines(log), [
            f"[eot-shadow] fire_ms={n * _MS} p=0.990 resumed=0 "
            f"actual_ms={_TURN_MS}"])


class WarmerTests(_Base):
    def _run_warmers(self):
        bc = self.bc
        self._p(bc, "_boot_warmers", [])
        self._p(bc, "_boot_warmers_started", [False])
        bc._eot_register_warmer()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            th = bc._run_boot_warmers()
            th.join(10)
        return [ln.strip() for ln in out.getvalue().splitlines()]

    def test_no_capture_loads_a_model_before_the_warmer(self):
        _audio, log = self._capture(_TURN)
        self.assertEqual((self.made, self.loads), ([], {"silero": 0,
                                                        "turn": 0}))
        self.assertEqual(log.count("[eot"), 0)
        lines = self._run_warmers()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^\[warm\] smart-turn ok \(\d+ ms\)$")
        self.assertEqual(self.loads, {"silero": 1, "turn": 1})
        self._capture(_TURN)
        self.assertEqual(len(self.made), 1)
        self.assertEqual(self.loads, {"silero": 1, "turn": 1})

    def test_a_silero_load_failure_is_its_one_line_and_skips_smart_turn(self):
        self.silero_load_error = OSError("gone")
        lines = self._run_warmers()
        self.assertEqual(lines, ["[warm] smart-turn failed (RuntimeError: "
                                 "silero: load failed: OSError: gone)"])
        self.assertEqual(self.loads, {"silero": 1, "turn": 0})
        st = self.bc._eot_state
        self.assertFalse(st["ready"])
        self.assertIn("silero: load failed", st["off"])
        _audio, log = self._capture(_TURN)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", log)                 # the warm line was it

    def test_a_slow_smart_turn_warm_latches_off(self):
        self.tsess.costs = [0.05, 0.2]
        lines = self._run_warmers()
        self.assertEqual(lines, ["[warm] smart-turn failed (RuntimeError: "
                                 "smart turn: too slow: warm call 200 ms > "
                                 "150 ms)"])
        self.assertFalse(self.bc._eot_state["ready"])
        self._capture(_TURN)
        self.assertEqual(self.made, [])


class InTurnCaptureTests(_Base):
    def test_a_capture_without_smart_endpoint_never_builds_a_decider(self):
        # bambu_setup's spoken access code, draft confirmations: a plain
        # record_speech(timeout=...).
        self._warm()
        audio, log = self._capture(_TURN, smart=False)
        self.assertEqual(self.made, [])
        self.assertEqual(self.vsess.feeds, [])
        self.assertNotIn("[eot", log)
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)

    def test_an_off_thread_capture_never_builds_a_decider(self):
        # A dashboard-run action asking the owner something captures off the
        # main loop's thread (_record_speech_offthread), even if a caller
        # passed smart_endpoint.
        self._warm()
        bc = self.bc
        real = bc.record_speech
        reentry = []
        box = {}

        def rec(timeout=None, **kw):
            if threading.current_thread() is threading.main_thread():
                th = threading.Thread(
                    target=lambda: box.setdefault(
                        "audio", real(timeout, smart_endpoint=True)),
                    name="action-capture")
                th.start()
                th.join(30)
                return box.get("audio")
            reentry.append(dict(kw))     # _record_speech_offthread's call
            return real(timeout, **kw)

        out = io.StringIO()
        with mock.patch.object(bc, "record_speech", rec), \
                contextlib.redirect_stdout(out):
            audio, _ = _spec.SpeculativeRealCaptureLoopTests._run_capture(
                self, _TURN)
        self.assertEqual(reentry, [{"_offthread_claimed": True}])
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", out.getvalue())
        # And the claimed re-entry ignores the flag itself.
        audio, log = self._capture(_TURN, _offthread_claimed=True)
        self.assertEqual(self.made, [])
        self.assertNotIn("[eot", log)

    def test_only_the_two_main_loop_listens_pass_smart_endpoint(self):
        path = os.path.join(_ROOT, "bobert_companion.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        sites = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and any(k.arg == "smart_endpoint"
                                for k in node.keywords)):
                    callee = getattr(node.func, "id", None) or getattr(
                        node.func, "attr", None)
                    kw = [k for k in node.keywords
                          if k.arg == "smart_endpoint"][0]
                    sites.append((fn.name, callee,
                                  getattr(kw.value, "value", None)))
        # Both listens sit directly in their (top-level) functions.
        self.assertEqual(sorted(set(sites)), [
            ("_capture_utterance", "record_speech", True),
            ("_handle_sleep_standby", "record_speech", True)])
        for sub in ("core", "skills"):
            for dirpath, _dirs, files in os.walk(os.path.join(_ROOT, sub)):
                for name in files:
                    if not name.endswith(".py"):
                        continue
                    with open(os.path.join(dirpath, name),
                              encoding="utf-8", errors="replace") as fh:
                        self.assertNotIn("smart_endpoint", fh.read(), name)


class OnModeTests(_Base):
    MODE = "on"

    def test_on_ends_a_complete_turn_at_the_first_check(self):
        self._warm()
        since = self.timing.now()
        audio, log = self._capture(_TURN)
        self.assertEqual(len(audio) * 1000 // 16000, _FIRE_MS)
        self.assertEqual(self._shadow_lines(log), [])
        d = self._turn_fields(since)
        self.assertEqual((d["eot"], d["st_p"], d["st_n"]), ("st", "0.99", "1"))
        self.assertEqual(d["vad_break"], "0")         # t0 is this break

    def test_on_below_the_threshold_is_the_fixed_wait(self):
        self._warm()
        self.tsess.ps = [0.5]
        since = self.timing.now()
        audio, _log = self._capture(_TURN)
        self.assertEqual(len(audio) * 1000 // 16000, _TURN_MS)
        self.assertEqual(self._turn_fields(since)["eot"], "rms")


if __name__ == "__main__":
    unittest.main()
