"""The capture chain wired into the REAL record_speech and the main loop's
capture phase (v3.0.26 overhaul, core/capture_chain.py).

record_speech runs for real over tests/test_speculative_stt.py's fake mic
stream (reused, never re-implemented); only the hardware / UI edges are
stubbed. Pinned here:

  * VAD floor (VAD_FLOOR_MODE 'on'; ships 'shadow' until one live day's
    '[vad-floor]' line shows 0 turns lost): a main-loop listen's capture
    starts only over the measured floor (capped at 1.25 x VAD_THRESHOLD),
    the chunks under it stay in the pre-roll (the onset is kept), and the
    floor never applies to an in-turn capture, around JARVIS's playback, or
    before ~10 s of idle audio was measured; 'shadow' starts on
    VAD_THRESHOLD and books what 'on' would have lost; 'off' is the old gate;
    VAD_START_CONFIRM_CHUNKS 2 drops a lone blip;
  * VOICE_ID_CLIP 'raw' (shipped): record_speech keeps the frames before
    _process_capture_chunk beside the processed clip, sample for sample, and
    _capture_utterance / _handle_sleep_standby publish THAT as
    _last_capture_audio (voice-ID, the media gate's probe key) while STT
    still hears the processed clip;
  * the auto-gain line is debug-only and every boost is booked;
  * STT_RESCUE_EMPTY 'speech' (shipped): Parakeet's empty text is re-decoded
    by Whisper only when the speech detector finds speech (or cannot tell),
    in the live rescue and the shadow's would-be rescue alike.

Run (full-deps tier):
    python tools/run_tests.py monolith.test_monolith_capture_chain
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
# A module, not names: discovery must not collect its TestCases again here.
from tests import test_speculative_stt as _spec

_CHUNK = _spec._CHUNK


@requires_monolith
class _CaptureBase(MonolithGlobalsTestCase):
    # The floor mode these tests run under; None = the shipped default.
    FLOOR_MODE = "on"

    def setUp(self):
        super().setUp()
        if self.FLOOR_MODE is not None:
            self._p(self.bc, "VAD_FLOOR_MODE", self.FLOOR_MODE)
        import numpy as np
        from core import capture_chain as cc
        self.np, self.cc = np, cc
        bc = self.bc
        bc._spec_stt_reset()
        self.addCleanup(bc._spec_stt_reset)
        bc._vad_floor.reset()
        bc._vad_book.reset()
        self.addCleanup(setattr, bc, "_last_recording_peak",
                        bc._last_recording_peak)
        self.decoded = []

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _prime(self, level, n=None):
        """The floor as ~10+ s of idle listening at `level` measured it."""
        for _ in range(n or self.cc.MIN_SAMPLES + 16):
            self.bc._vad_floor.note_idle(level)

    def _capture(self, rms_seq, *, smart=True, timeout=0.6, tail=30,
                 processed=None):
        """ONE real record_speech capture of `rms_seq` (per-chunk RMS), as
        the main loop's listen calls it (smart_endpoint=`smart`); None when
        no capture started within `timeout`. `processed`: a stand-in for
        _process_capture_chunk (else processing is off: processed == raw)."""
        bc = self.bc
        frames = ([_spec._mic_frame(r) for r in rms_seq]
                  + [_spec._mic_frame(0.0)] * tail)
        self.frames = frames

        class _SDShim:
            PortAudioError = bc.sd.PortAudioError

            @staticmethod
            def InputStream(**kw):
                return _spec._FakeMicStream(frames, **kw)

        self._p(bc, "_mic_input_disabled", lambda: False)
        self._p(bc, "sd", _SDShim)
        self._p(bc, "get_input_device", lambda: None)
        self._p(bc, "_safe_close_stream", lambda _s: None)
        self._p(bc, "pause_face_tracking", lambda *a, **k: None)
        self._p(bc, "set_state", lambda *a, **k: None)
        self._p(bc, "_report_silent_mic", lambda *a, **k: None)
        self._p(bc, "_transcribe_impl",
                lambda a: (self.decoded.append(a) or ("stub", {})))
        if processed is None:
            bc._audio_master_enabled[0] = False
        else:
            self._p(bc, "_process_capture_chunk", processed)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            audio = bc.record_speech(timeout=timeout, smart_endpoint=smart)
        self.log = out.getvalue()
        return audio

    def _vad_meta(self):
        return (self.bc._last_capture_meta[0] or {}).get("vad")

    def _preroll(self):
        return self.bc._last_capture_preroll[0][1]


class VadFloorShippedDefaultTests(_CaptureBase):
    """The shipped mode is 'shadow': a quiet utterance over VAD_THRESHOLD
    still starts a capture (nothing can go deaf), and the would-be hold is
    only counted."""
    FLOOR_MODE = None

    def test_shipped_mode_is_shadow_and_starts_on_the_base(self):
        self.assertEqual(self.bc.VAD_FLOOR_MODE, "shadow")
        self._prime(0.0045)
        self.assertIsNotNone(self._capture([0.009] * 20))
        self.assertAlmostEqual(self._vad_meta()["start_thr"], 0.008)
        self.assertTrue(self._vad_meta()["would_hold"])


class VadFloorCaptureTests(_CaptureBase):
    def test_noise_under_the_measured_floor_starts_no_capture(self):
        # The 11-17 17:06 room: idle chunks ~0.0045, noise blips at 0.009 -
        # over VAD_THRESHOLD (0.008), under the floor's 0.010.
        self._prime(0.0045)
        self.assertIsNone(self._capture([0.009] * 20))
        self.assertNotIn("Recording", self.log)
        self.assertEqual(self.bc._vad_book.held, 1)     # one episode

    def test_the_same_noise_starts_a_capture_with_the_floor_off(self):
        self._prime(0.0045)
        self._p(self.bc, "VAD_FLOOR_MODE", "off")
        self.assertIsNotNone(self._capture([0.009] * 20))

    def test_speech_over_the_floor_keeps_its_onset_in_the_pre_roll(self):
        self._prime(0.0045)
        audio = self._capture([0.009] * 4 + [0.03] * 10)
        self.assertIsNotNone(audio)
        # 4 held chunks (pre-roll) + 10 voiced + the 21-chunk hangover.
        self.assertEqual(len(audio), (4 + 10 + 21) * _CHUNK)
        self.assertEqual(self._preroll(), 4 * _CHUNK)
        self.np.testing.assert_array_equal(audio[:_CHUNK], self.frames[0])
        meta = self._vad_meta()
        self.assertAlmostEqual(meta["start_thr"], 0.010)
        self.assertFalse(meta["would_hold"])

    def test_a_long_held_onset_is_kept_past_the_plain_pre_roll(self):
        # 20 chunks (1.3 s) of quiet speech over VAD_THRESHOLD but under the
        # floor's start threshold, then louder: the capture starts late, yet
        # it begins where VAD_THRESHOLD would have (the v2.0.104 lesson).
        self._prime(0.0045)
        audio = self._capture([0.009] * 20 + [0.03] * 6)
        self.assertIsNotNone(audio)
        self.assertEqual(self._preroll(), 20 * _CHUNK)
        self.np.testing.assert_array_equal(audio[:_CHUNK], self.frames[0])

    def test_a_held_episode_keeps_at_most_the_cap(self):
        # Steady noise over the base for a minute must not grow the ring
        # without bound: the clip starts _VAD_HELD_KEEP_MAX chunks past the
        # plain pre-roll, no earlier.
        self._prime(0.0045)
        audio = self._capture([0.009] * 100 + [0.03] * 4)
        self.assertIsNotNone(audio)
        self.assertEqual(self._preroll(),
                         (12 + self.bc._VAD_HELD_KEEP_MAX) * _CHUNK)

    def test_an_in_turn_capture_never_uses_the_floor(self):
        self._prime(0.0045)
        n = len(self.bc._vad_floor)
        self.assertIsNotNone(self._capture([0.009] * 20, smart=False))
        self.assertIsNone(self._vad_meta())
        self.assertEqual(len(self.bc._vad_floor), n)    # nor samples it

    def test_jarvis_playing_keeps_the_base_and_samples_nothing(self):
        self._prime(0.0045)
        n = len(self.bc._vad_floor)
        self._p(self.bc, "_vad_playback_recent", lambda: True)
        self.assertIsNotNone(self._capture([0.009] * 20))
        self.assertAlmostEqual(self._vad_meta()["start_thr"], 0.008)
        self.assertEqual(len(self.bc._vad_floor), n)

    def test_an_unmeasured_floor_starts_on_the_base(self):
        self.assertIsNotNone(self._capture([0.009] * 20))
        self.assertAlmostEqual(self._vad_meta()["start_thr"], 0.008)

    def test_the_cap_holds_in_a_loud_room(self):
        self._prime(0.02)                      # 2.5 x 0.02 = 0.05 wanted
        self.assertIsNotNone(self._capture([0.0105] * 10))
        self.assertAlmostEqual(self._vad_meta()["start_thr"], 0.010)

    def test_idle_main_listen_chunks_feed_the_floor(self):
        self.assertIsNotNone(self._capture([0.004] * 10 + [0.03] * 3))
        # 10 idle chunks and the tripping one; nothing once recording.
        self.assertEqual(len(self.bc._vad_floor), 11)

    def test_shadow_starts_on_the_base_and_books_what_on_would_lose(self):
        self._p(self.bc, "VAD_FLOOR_MODE", "shadow")
        self._prime(0.0045)
        self.assertIsNotNone(self._capture([0.009] * 20))
        meta = self._vad_meta()
        self.assertAlmostEqual(meta["start_thr"], 0.008)
        self.assertTrue(meta["would_hold"])
        self.bc._vad_floor_note_turn()
        self.bc._vad_floor_note_turn()         # one turn, counted once
        self.assertEqual(self.bc._vad_book.would_hold_turns, 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.bc._vad_book_tick(force=True)
        self.assertIn("[vad-floor] shadow:", out.getvalue())
        self.assertIn("1 would not have started (1 of them turns)",
                      out.getvalue())

    def test_confirm_two_chunks_drops_a_lone_blip_and_keeps_a_pair(self):
        self._p(self.bc, "VAD_START_CONFIRM_CHUNKS", 2)
        self.assertIsNone(self._capture([0.03] + [0.0] * 5))
        audio = self._capture([0.0, 0.03, 0.03, 0.03])
        self.assertIsNotNone(audio)
        # the first loud chunk waited in the pre-roll
        self.assertEqual(self._preroll(), 2 * _CHUNK)

    def test_one_chunk_trips_are_booked_at_the_shipped_default(self):
        self.assertIsNotNone(self._capture([0.03]))
        self.assertTrue(self._vad_meta()["single"])
        self.assertIsNotNone(self._capture([0.03] * 3))
        self.assertFalse(self._vad_meta()["single"])
        self.assertEqual(self.bc._vad_book.single, 1)
        self.assertEqual(self.bc._vad_book.captures, 2)

    def test_end_of_speech_is_not_touched(self):
        # Speech that dips under the floor's 0.010 but stays over
        # VAD_THRESHOLD is still voice once recording: the capture does not
        # end sooner than with the floor off.
        seq = [0.03] * 3 + [0.009] * 15 + [0.03] * 2
        self._prime(0.0045)
        on = self._capture(seq)
        self._p(self.bc, "VAD_FLOOR_MODE", "off")
        off = self._capture(seq)
        self.assertEqual(len(on), len(off))


class VoiceClipTests(_CaptureBase):
    def _x3(self, d, sr=16000, skip_ns=False, **_kw):
        """A stand-in for noise suppression + AGC: a different signal."""
        return (d * self.np.float32(3.0)).astype(self.np.float32)

    def test_record_speech_keeps_the_raw_frames_beside_the_processed_clip(self):
        np = self.np
        audio = self._capture([0.0] * 3 + [0.02] * 8, processed=self._x3)
        raw = self.bc._last_capture_raw[0]
        self.assertIsNotNone(raw)
        self.assertEqual(len(raw), len(audio))
        expect = np.concatenate(self.frames[:len(audio) // _CHUNK])
        np.testing.assert_array_equal(raw, expect)
        np.testing.assert_allclose(audio, raw * 3.0, rtol=1e-6)
        # the auto-gain basis is the processed clip's own level
        self.assertAlmostEqual(self.bc._last_recording_clip_peak[0],
                               self.cc.peak_chunk_rms(audio))
        self.assertIs(self.bc._voice_clip_for(audio), raw)
        self._p(self.bc, "VOICE_ID_CLIP", "processed")
        self.assertIs(self.bc._voice_clip_for(audio), audio)

    def test_a_clip_that_does_not_line_up_falls_back_to_itself(self):
        audio = self._capture([0.02] * 4, processed=self._x3)
        cut = audio[:-10]
        self.assertIs(self.bc._voice_clip_for(cut), cut)
        self.assertIsNone(self.bc._voice_clip_for(None))

    def _run_capture_utterance(self, raw, proc):
        bc = self.bc
        seen = {}

        def fake_record(timeout=None, **_kw):
            bc._last_capture_raw[0] = raw
            return proc

        def fake_transcribe(a):
            seen["stt"] = a
            return "turn on the desk lamp", {"avg_logprob": -0.1}

        self._p(bc, "_get_realtime_session", return_value=None)
        self._p(bc, "record_speech", side_effect=fake_record)
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "resume_face_tracking")
        self._p(bc, "_heartbeat")
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_media_probe_read", return_value=None)
        self._p(bc, "_transcribe_capture", side_effect=fake_transcribe)
        bc._last_recording_peak = 0.012
        with contextlib.redirect_stdout(io.StringIO()):
            out = bc._capture_utterance(None, memory=mock.Mock())
        return out, seen

    def test_capture_utterance_publishes_the_raw_clip_for_voice_id(self):
        np = self.np
        raw = np.tile(_spec._mic_frame(0.012), 8).astype(np.float32)
        proc = raw * np.float32(4.0)
        out, seen = self._run_capture_utterance(raw, proc)
        self.assertEqual(out[0], "turn on the desk lamp")
        bc = self.bc
        self.assertIs(bc._last_capture_audio, raw)       # voice-ID's clip
        self.assertIs(bc._last_capture_stt[0], proc)      # STT's clip
        self.assertEqual(bc._media_probe[0]["audio_id"], id(raw))
        # STT heard the processed clip, auto-gained from ITS level
        want, _g = bc.apply_capture_auto_gain(proc, 0.012)
        np.testing.assert_array_equal(seen["stt"], want)

    def _fake_ambient(self):
        import sys
        import types
        got = []
        mod = types.ModuleType("skill_ambient_listen")
        mod.note_main_transcript = lambda *a: got.append(a) or True
        self._p(sys, "modules", dict(sys.modules, skill_ambient_listen=mod))
        return got

    def test_capture_utterance_hands_its_transcript_to_the_ambient_tap(self):
        np = self.np
        raw = np.tile(_spec._mic_frame(0.012), 8).astype(np.float32)
        proc = raw * np.float32(4.0)
        got = self._fake_ambient()
        self.bc._capture_spans.append((1.0, 2.0))
        self._run_capture_utterance(raw, proc)
        self.assertEqual(len(got), 1)
        text, conf, peak, clip, sr, span = got[0]
        self.assertEqual(text, "turn on the desk lamp")
        self.assertEqual(conf, {"avg_logprob": -0.1})
        self.assertAlmostEqual(peak, 0.012)
        self.assertIs(clip, raw)
        self.assertEqual((sr, span), (16000, (1.0, 2.0)))

    def test_only_a_decoded_capture_joins_the_offered_spans(self):
        # A capture the main loop dropped is a span (covers the batch) but
        # never an offered one; offering a transcript marks the last span.
        bc = self.bc
        bc._capture_spans.clear()
        bc._capture_open_since[0] = bc.time.monotonic() - 1.0
        bc._capture_span_close()
        t0 = bc.time.monotonic() - 60.0
        self.assertEqual(len(bc.capture_spans_since(t0)), 1)
        self.assertEqual(bc.capture_spans_offered_since(t0), [])
        bc._ambient_offer_main_transcript("", {})
        self.assertEqual(len(bc.capture_spans_offered_since(t0)), 1)
        self.assertEqual(bc.capture_spans_offered_since(
            bc.time.monotonic() + 60.0), [])

    def test_standby_publishes_the_raw_clip_and_hands_over_its_text(self):
        np = self.np
        bc = self.bc
        raw = np.tile(_spec._mic_frame(0.012), 8).astype(np.float32)
        proc = raw * np.float32(4.0)
        got = self._fake_ambient()
        seen = {}

        def fake_record(timeout=None, **_kw):
            bc._last_capture_raw[0] = raw
            return proc

        def fake_transcribe(a):
            seen["stt"] = a
            return "the weather is nice today", {"avg_logprob": -0.2}

        bc._sleep_mode[0] = True
        self._p(bc, "record_speech", side_effect=fake_record)
        self._p(bc, "_speak_pending", return_value=False)
        self._p(bc, "set_state")
        self._p(bc, "_heartbeat")
        self._p(bc, "_audio_music_feed")
        self._p(bc, "_media_probe_read", return_value=None)
        self._p(bc, "_alarm_is_ringing", return_value=False)
        self._p(bc, "_standby_wake_detected", return_value=None)
        self._p(bc, "_transcribe_capture", side_effect=fake_transcribe)
        bc._last_recording_peak = 0.012
        with contextlib.redirect_stdout(io.StringIO()):
            bc._handle_sleep_standby(None)
        self.assertIs(bc._last_capture_audio, raw)
        self.assertIs(bc._last_capture_stt[0], proc)
        want, _g = bc.apply_capture_auto_gain(proc, 0.012)
        np.testing.assert_array_equal(seen["stt"], want)
        self.assertEqual([g[0] for g in got], ["the weather is nice today"])

    def test_processed_setting_restores_the_old_clip(self):
        np = self.np
        raw = np.tile(_spec._mic_frame(0.012), 8).astype(np.float32)
        proc = raw * np.float32(4.0)
        self._p(self.bc, "VOICE_ID_CLIP", "processed")
        self._run_capture_utterance(raw, proc)
        self.assertIs(self.bc._last_capture_audio, proc)

    def test_the_auto_gain_line_is_debug_only_and_every_boost_is_booked(self):
        np = self.np
        bc = self.bc
        clip = np.tile(_spec._mic_frame(0.02), 4).astype(np.float32)
        bc._last_recording_peak = 0.012
        bc._debug_mode[0] = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _a, g = bc._capture_auto_gain_logged(clip)
        self.assertGreater(g, 1.0)
        self.assertNotIn("[auto-gain]", out.getvalue())
        self.assertEqual(len(bc._vad_book.gains), 1)
        bc._debug_mode[0] = True
        with contextlib.redirect_stdout(out):
            bc._capture_auto_gain_logged(clip)
        self.assertIn("[auto-gain] quiet input rms=0.0120", out.getvalue())
        self.assertIn("(clip basis)", out.getvalue())


class EmptyRescueTests(_CaptureBase):
    def _rescue(self, speech_s, mode=None, text=""):
        bc = self.bc
        calls = []

        def fake_seconds(audio, sample_rate=16000, start=0):
            calls.append(len(audio))
            return speech_s
        self._p(bc._tail_vad, "speech_seconds", side_effect=fake_seconds)
        self._p(bc, "_whisper_bg_loading", return_value=False)
        if mode is not None:
            self._p(bc, "STT_RESCUE_EMPTY", mode)
        audio = self.np.zeros(16000, self.np.float32)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            why = bc._parakeet_rescue(text, audio)
        self.out = out.getvalue()
        return why, len(calls)

    def test_shipped_mode_is_speech(self):
        from core import config
        self.assertEqual(config.STT_RESCUE_EMPTY, "speech")
        self.assertEqual(config.STT_RESCUE_EMPTY_MIN_SPEECH_S, 0.25)

    def test_no_speech_keeps_the_empty_text_and_says_so_once(self):
        self.assertEqual(self._rescue(0.0), ("", 1))
        self.assertIn("[stt] empty rescue skipped: no speech in the capture "
                      "(1 so far", self.out)
        self.assertEqual(self.bc._vad_book.rescue_skipped, 1)
        self.assertEqual(self._rescue(0.1), ("", 1))     # under 0.25 s
        self.assertEqual(self.out, "")                   # not every time

    def test_speech_or_unknown_rescues(self):
        self.assertEqual(self._rescue(0.25), ("empty", 1))
        self.assertEqual(self._rescue(None), ("empty", 1))
        self.assertEqual(self.bc._vad_book.rescue_run, 2)

    def test_always_and_off(self):
        self.assertEqual(self._rescue(0.0, mode="always"), ("empty", 0))
        self.assertEqual(self._rescue(5.0, mode="off"), ("", 0))

    def test_speech_awake_never_rescues_while_asleep(self):
        bc = self.bc
        bc._sleep_mode[0] = True
        self.addCleanup(bc._sleep_mode.__setitem__, 0, False)
        self.assertEqual(self._rescue(5.0, mode="speech_awake"), ("", 0))
        bc._sleep_mode[0] = False
        self.assertEqual(self._rescue(5.0, mode="speech_awake"), ("empty", 1))
        self.assertEqual(self._rescue(0.0, mode="speech_awake"), ("", 1))

    def test_a_text_is_never_asked_about_speech(self):
        self._p(self.bc, "_parakeet_wake_lost", return_value=False)
        self.assertEqual(self._rescue(0.0, text="turn the lights off"),
                         ("", 0))

    def test_the_shadow_would_be_rescue_follows_the_same_rule(self):
        bc = self.bc
        self._p(bc._tail_vad, "speech_seconds", return_value=0.0)
        audio = self.np.zeros(16000, self.np.float32)
        self.assertEqual(bc._parakeet_shadow_rescue("", audio, {}), "")
        self._p(bc, "STT_RESCUE_EMPTY", "always")
        self.assertEqual(bc._parakeet_shadow_rescue("", audio, {}), "empty")


if __name__ == "__main__":
    unittest.main()
