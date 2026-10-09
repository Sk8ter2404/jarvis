"""Listening over media, monolith wiring (2026-10-05).

Owner, 10-05 ~02:00: "it seems like he's constantly listening, especially
when videos are playing, and he can't hear me - even in wake word mode."

Drives the REAL record_speech against fake streams that feed its REAL
callbacks (the bus's, or its own per-capture one) - no re-implemented loop
(the 2026-09-06 lesson), no device. What is faked: the PortAudio stream, the
device resolution, the UI edges (face tracking, state, the silent-mic
reporter), the canceller (a deterministic stand-in where the test is about
the WIRING; tests/test_media_aec.py covers the canceller itself), STT and
voice-ID. Frames carry index markers, so a test can say which frame a clip
starts with.

Mutation-tested (each load-bearing call deleted by hand -> the named test
goes red): the bus pre-roll read (_bus_chunks in _capture_frame) ->
test_the_pre_roll_comes_from_the_ring; the B2 carry (_segment_cut) ->
test_a_media_segment_overlaps_the_next; the canceller's det / keep in
_capture_frame -> test_the_cancelled_signal_decides_and_the_linear_is_kept;
the D1 re-seat (_pregate_take_hit) -> test_a_confirmed_wake_starts_the_capture;
the veto (_wake_vetoed) -> test_the_video_saying_the_name_is_vetoed.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_CHUNK = 1024
_SR = 16000


def _frame(value: float) -> np.ndarray:
    """A chunk whose every sample is +/-value (RMS == |value|): its first
    sample IS the marker."""
    x = np.full(_CHUNK, abs(value), np.float32)
    x[1::2] *= -1
    return x


def _quiet(i: int) -> np.ndarray:          # under VAD 0.008 for i < 70
    return _frame(0.0001 * (i + 1))


def _loud(i: int) -> np.ndarray:          # well over VAD 0.008
    return _frame(0.03 + 0.0001 * i)


def _marker(clip, k):
    return round(float(abs(clip[k * _CHUNK])), 5)


@requires_monolith
class _Base(MonolithGlobalsTestCase):

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _edges(self):
        bc = self.bc
        self._p(bc, "_mic_input_disabled", lambda: False)
        self._p(bc, "get_input_device", lambda: 0)
        self._p(bc, "pause_face_tracking", lambda *a, **k: None)
        self._p(bc, "set_state", lambda *a, **k: None)
        self._p(bc, "_report_silent_mic", lambda *a, **k: None)
        self._p(bc, "_filler_capture_mark", lambda *a, **k: None)
        self._p(bc, "_note_live_capture", lambda *a, **k: None)
        self._p(bc, "HUD_ENABLED", False)
        prev = bc._audio_master_enabled[0]
        self.addCleanup(bc._audio_master_enabled.__setitem__, 0, prev)
        bc._audio_master_enabled[0] = False     # chunks pass through as-is
        self.addCleanup(setattr, bc, "_last_recording_peak",
                        bc._last_recording_peak)
        prev_mute = bc._mic_muted[0]
        self.addCleanup(bc._mic_muted.__setitem__, 0, prev_mute)
        bc._mic_muted[0] = False
        self.addCleanup(bc._spec_stt_reset)

    def _bus_on(self):
        """MIC_BUS_MODE 'on' with a fake stream. Returns the list of
        streams the bus opened."""
        bc = self.bc
        from core import mic_bus as mb
        self._edges()
        self._p(bc, "MIC_BUS_MODE", "on", create=True)
        self._p(mb, "DEAD_AFTER_S", 1.0)
        streams = []

        class FeedStream:
            latency = 0.0

            def __init__(s, *a, callback=None, device=None, **k):
                s.cb, s.device, s.closed = callback, device, False
                streams.append(s)

            def start(s):
                pass

            def stop(s):
                pass

            def close(s):
                s.closed = True

        self._p(bc.sd, "InputStream", FeedStream)
        self._p(bc, "_safe_close_stream",
                lambda st, *a, **k: st.close() if st is not None else None)
        self.addCleanup(self._bus_off)
        return streams

    def _bus_off(self):
        bc = self.bc
        bus = bc._mic_bus_obj[0]
        if bus is not None:
            bus.shutdown()
        bc._mic_bus_obj[0] = None
        bc._mic_bus_active[0] = False
        bc._mic_bus_carry[0] = None

    def _open_bus(self, streams):
        bus = self.bc._mic_bus_get()
        ok, err = bus.ensure(0)
        self.assertTrue(ok, err)
        return bus, streams[-1]

    def _feed_now(self, bus, stream, frames):
        n0 = bus.n_written
        for f in frames:
            stream.cb(f.reshape(-1, 1), len(f), None, None)
        end = time.monotonic() + 5
        while bus.n_written < n0 + len(frames) * _CHUNK:
            self.assertLess(time.monotonic(), end, "the DSP thread stalled")
            time.sleep(0.005)

    def _feed_when_read(self, bus, stream, frames, subs=1):
        """Feed ``frames`` once a capture subscribes, never faster than it
        reads (a test must not overflow a subscription)."""
        def run():
            end = time.monotonic() + 10
            while (bus.status()["subscribers"] < subs
                   and time.monotonic() < end):
                time.sleep(0.002)
            for f in frames:
                while (any(s.qsize() > 16 for s in list(bus._subs))
                       and time.monotonic() < end):
                    time.sleep(0.002)
                stream.cb(f.reshape(-1, 1), len(f), None, None)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.addCleanup(t.join, 5)
        return t

    def _record(self, timeout=5.0):
        with mock.patch("builtins.print"):
            return self.bc.record_speech(timeout=timeout)


# ── B1: the always-open microphone ────────────────────────────────────────
class MicBusCaptureTests(_Base):

    def test_the_pre_roll_comes_from_the_ring(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        # Heard BEFORE the capture started: a per-capture stream never had it.
        self._feed_now(bus, st, [_quiet(i) for i in range(15)])
        self._feed_when_read(bus, st, [_loud(i) for i in range(10)]
                             + [_quiet(60)] * 40)
        clip = self._record()
        self.assertIsNotNone(clip)
        self.assertEqual(_marker(clip, 0), round(0.0001 * 4, 5),
                         "the pre-roll is the ring's last 12 chunks (3..14)")
        self.assertEqual(_marker(clip, 11), round(0.0001 * 15, 5))
        self.assertEqual(_marker(clip, 12), round(0.03, 5))
        # The stream stays open after the capture; nothing reopened.
        self.assertEqual(len(streams), 1)
        self.assertFalse(streams[0].closed)
        self.assertTrue(bc._mic_bus_active[0])
        self.assertFalse(bc._record_speech_active[0])

    def test_the_next_capture_reuses_the_same_stream(self):
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        for _ in range(2):
            self._feed_when_read(bus, st, [_loud(1)] * 3 + [_quiet(1)] * 40)
            self.assertIsNotNone(self._record())
        self.assertEqual(len(streams), 1)

    def test_mute_closes_the_stream_and_frees_the_device(self):
        bc = self.bc
        streams = self._bus_on()
        bus, _st = self._open_bus(streams)
        bc._mic_muted[0] = True
        bus.wake()
        end = time.monotonic() + 3
        while bus.is_open() and time.monotonic() < end:
            time.sleep(0.01)
        self.assertTrue(streams[0].closed)
        self.assertFalse(bc._mic_bus_active[0])
        self.assertIsNone(self._record(timeout=0.1))   # muted: nothing opens

    def test_a_failed_open_is_booked_with_the_r10_backoff(self):
        bc = self.bc
        self._bus_on()
        booked = []
        self._p(bc.sd, "InputStream",
                mock.Mock(side_effect=bc.sd.PortAudioError("gone")))
        self._p(bc, "_input_open_failed", lambda e, d: booked.append(e))
        self._p(bc, "_note_input_open_failure", lambda *a, **k: None)
        self._p(bc, "_usb_storm_note_audio_drop", lambda *a, **k: None)
        self.assertIsNone(self._record(timeout=0.1))
        self.assertEqual(len(booked), 1)
        self.assertFalse(bc._record_speech_active[0])
        self.assertFalse(bc._mic_bus_active[0])

    def test_off_is_the_per_capture_stream_exactly(self):
        bc = self.bc
        self._edges()
        self._p(bc, "MIC_BUS_MODE", "off", create=True)
        opened = []

        class Legacy:
            def __init__(s, *a, callback=None, **k):
                opened.append(s)
                s.cb = callback

            def start(s):
                for f in [_loud(0)] * 3 + [_quiet(0)] * 40:
                    s.cb(f.reshape(-1, 1), len(f), None, None)

        self._p(bc.sd, "InputStream", Legacy)
        self._p(bc, "_safe_close_stream", lambda *a, **k: None)
        self.assertIsNotNone(self._record())
        self.assertEqual(len(opened), 1)
        self.assertIsNone(bc._mic_bus_obj[0])          # never built

    def test_the_record_taps_get_the_bus_between_captures(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        self.assertTrue(bc.add_record_tap(tap := __import__("queue").Queue()))
        self.addCleanup(bc.remove_record_tap, tap)
        self._feed_now(bus, st, [_quiet(i) for i in range(3)])
        got = [tap.get(timeout=2) for _ in range(3)]
        self.assertEqual(round(float(abs(got[2][0])), 5), 0.0003)

    def test_get_mic_buffer_taps_the_bus_and_never_opens_a_second_stream(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)

        def feeder():
            end = time.monotonic() + 5
            while not bc._record_speech_taps and time.monotonic() < end:
                time.sleep(0.002)
            for i in range(6):
                st.cb(_quiet(i).reshape(-1, 1), _CHUNK, None, None)

        threading.Thread(target=feeder, daemon=True).start()
        buf = bc.get_mic_buffer(0.2)
        self.assertIsNotNone(buf)
        self.assertEqual(len(buf), int(0.2 * _SR))
        self.assertEqual(len(streams), 1)

    def test_a_device_refresh_suspends_and_resumes_the_bus(self):
        bc = self.bc
        streams = self._bus_on()
        bus, _st = self._open_bus(streams)
        terminated = []
        with mock.patch.object(bc.sd, "_terminate",
                               side_effect=lambda: terminated.append(
                                   bc._mic_bus_active[0])), \
                mock.patch.object(bc.sd, "_initialize"), \
                mock.patch.object(bc.sd, "query_devices",
                                  return_value={"name": "FakeMic"}), \
                mock.patch.object(bc, "_pick_device",
                                  return_value=(0, "FakeMic")), \
                mock.patch.object(bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(bc, "SPEAKER_INDEX", None), \
                mock.patch("builtins.print"):
            bc._device_cache["checked_at"] = 0.0
            bc._refresh_devices(force=True)
        self.assertEqual(terminated, [False],
                         "the reinit ran, with the bus's stream closed")
        self.assertTrue(streams[0].closed)
        self.assertEqual(bus.status()["suspended"], False)   # resumed
        ok, _ = bus.ensure(0)
        self.assertTrue(ok)
        self.assertEqual(len(streams), 2)

    def test_the_owner_cell_defers_a_reinit_like_any_capture(self):
        bc = self.bc
        bc._mic_bus_active[0] = True
        try:
            with bc._pa_gate:
                self.assertTrue(bc._pa_mic_capture_live())
        finally:
            bc._mic_bus_active[0] = False

    def test_the_legacy_barge_in_never_opens_beside_the_bus(self):
        bc = self.bc
        self._p(bc, "_mic_bus_open", lambda: True)
        self._p(bc, "_mic_input_disabled", lambda: False)
        opened = mock.Mock()
        self._p(bc.sd, "InputStream", opened)
        self.assertIsNone(bc._start_barge_in_listener())
        opened.assert_not_called()


# ── B2: overlapping segments over media ───────────────────────────────────
class SegmentTests(_Base):

    def test_a_media_segment_overlaps_the_next(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "MEDIA_SEGMENT_S", 0.5, create=True)
        self._p(bc._lm, "SEGMENT_OVERLAP_S", 4 * _CHUNK / _SR)
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        self._p(bc, "_require_wake_runtime", True)
        self._p(bc, "_media_aec_effective", lambda: False)
        bus, st = self._open_bus(streams)
        self._feed_when_read(bus, st, [_loud(i) for i in range(60)])
        first = self._record()
        self.assertIsNotNone(first)
        self.assertEqual(bc._last_capture_meta[0]["why"], "segment")
        n_first = len(first) // _CHUNK
        self.assertLessEqual(n_first, 10)            # cut at ~0.5 s of audio
        carry = bc._mic_bus_carry[0]
        self.assertIsNotNone(carry)
        last = _marker(first, n_first - 1)
        self._feed_when_read(bus, st, [_loud(100 + i) for i in range(10)]
                             + [_quiet(0)] * 40)
        second = self._record()
        self.assertIsNotNone(second)
        # It starts 4 chunks before the first one's cut: no gap, no split.
        want = round(last - 0.0001 * 3, 5)
        self.assertEqual(_marker(second, 0), want)

    def test_no_segments_when_the_echo_is_cancelled(self):
        bc = self.bc
        self._p(bc, "_bus_mode_on", lambda: True)
        self._p(bc, "_require_wake_runtime", True)
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        self._p(bc, "_media_aec_effective", lambda: True)
        self.assertFalse(bc._segment_active_now())
        self._p(bc, "_media_aec_effective", lambda: False)
        self.assertTrue(bc._segment_active_now())


# ── C1 / C2: what the canceller hands the capture ─────────────────────────
class _FakeAEC:
    """lin = raw * lin_gain, sup = raw * sup_gain."""

    def __init__(self, lin_gain, sup_gain):
        self.lin_gain, self.sup_gain = lin_gain, sup_gain
        self.sessions = 0

    def process(self, x, t=None):
        x = np.asarray(x, np.float32)
        return x * self.lin_gain, x * self.sup_gain

    def new_session(self):
        self.sessions += 1

    def take_not_converging(self):
        return False

    def erle_db(self):
        return 25.0

    def status(self):
        return {"converging": True}


class CancelledSignalTests(_Base):

    def _aec(self, fake, mode="on"):
        bc = self.bc
        self._p(bc, "MEDIA_AEC_MODE", mode, create=True)
        self._p(bc, "_media_aec_get", lambda: fake)
        self._p(bc, "_media_aec_bypass", lambda: "")
        self._p(bc, "_media_aec_effective", lambda: mode == "on")

    def test_the_cancelled_signal_decides_and_the_linear_is_kept(self):
        streams = self._bus_on()
        self._aec(_FakeAEC(lin_gain=0.5, sup_gain=1.0))
        bus, st = self._open_bus(streams)
        self._feed_when_read(bus, st, [_loud(i) for i in range(4)]
                             + [_quiet(0)] * 40)
        clip = self._record()
        self.assertIsNotNone(clip)
        k = len(clip) // _CHUNK - 21 - 4   # the 4 loud chunks
        self.assertAlmostEqual(abs(float(clip[k * _CHUNK])), 0.015, places=4,
                               msg="STT gets the LINEAR (echo-removed) frame")

    def test_video_alone_never_starts_a_capture(self):
        streams = self._bus_on()
        self._aec(_FakeAEC(lin_gain=0.0, sup_gain=0.0))
        bus, st = self._open_bus(streams)
        self._feed_when_read(bus, st, [_loud(i) for i in range(30)]
                             + [_quiet(0)] * 40)
        self.assertIsNone(self._record(timeout=1.0))

    def test_shadow_measures_but_the_capture_is_today_s(self):
        streams = self._bus_on()
        self._aec(_FakeAEC(lin_gain=0.0, sup_gain=0.0), mode="shadow")
        bus, st = self._open_bus(streams)
        self._feed_when_read(bus, st, [_loud(i) for i in range(4)]
                             + [_quiet(0)] * 40)
        clip = self._record()
        self.assertIsNotNone(clip)
        self.assertGreater(float(np.abs(clip).max()), 0.02)

    def test_the_per_capture_stream_uses_the_canceller_too(self):
        bc = self.bc
        self._edges()
        self._p(bc, "MIC_BUS_MODE", "off", create=True)
        fake = _FakeAEC(lin_gain=0.0, sup_gain=0.0)
        self._aec(fake)

        class Legacy:
            def __init__(s, *a, callback=None, **k):
                s.cb = callback

            def start(s):
                # Trailing quiet: a broken canceller ends the capture (red)
                # instead of hanging it.
                for f in [_loud(0)] * 30 + [_quiet(0)] * 40:
                    s.cb(f.reshape(-1, 1), len(f), None, None)

        self._p(bc.sd, "InputStream", Legacy)
        self._p(bc, "_safe_close_stream", lambda *a, **k: None)
        self._p(bc, "_media_aec_obj", [fake])
        self.assertIsNone(self._record(timeout=0.5))
        self.assertEqual(fake.sessions, 1)     # a new stream, a new session


# ── D1 / D2 / D3: the pre-gate's trigger, veto and duck ──────────────────
class PregateTests(_Base):

    def test_a_confirmed_wake_starts_the_capture(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_quiet(i) for i in range(30)])
        bc._pregate_hit[0] = {"n0": 10 * _CHUNK, "t": time.monotonic(),
                              "ok": True, "seq": 7, "score": 0.4}
        self._feed_when_read(bus, st, [_quiet(40)] * 2 + [_loud(1)] * 3
                             + [_quiet(41)] * 40)
        clip = self._record()
        self.assertIsNotNone(clip)
        self.assertEqual(_marker(clip, 0), round(0.0001 * 11, 5),
                         "the capture starts at the hit minus the pre-roll")
        meta = bc._last_capture_meta[0]
        self.assertTrue(meta["pregate"])
        self.assertEqual(meta["pregate_seq"], 7)
        self.assertIsNone(bc._pregate_hit[0])          # consumed

    def test_shadow_never_re_seats(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "shadow", create=True)
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_quiet(i) for i in range(30)])
        bc._pregate_hit[0] = {"n0": 10 * _CHUNK, "t": time.monotonic(),
                              "ok": True, "seq": 7, "score": 0.4}
        self._feed_when_read(bus, st, [_loud(1)] * 3 + [_quiet(41)] * 40)
        clip = self._record()
        self.assertEqual(_marker(clip, 0), round(0.0001 * 19, 5))

    def _confirm_env(self, decoded):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_loud(i % 50) for i in range(60)])
        self._p(bc, "_wake_confirm_decode", lambda a: (decoded, {}))
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: ("owner", 0.88, ""))
        self._p(bc, "_media_aec_effective", lambda: False)
        return bus

    def test_the_confirm_hands_record_speech_the_cut(self):
        bc = self.bc
        bus = self._confirm_env("Jarvis, pause the music.")
        t_hit = bus.last_frame_time() - 1.5
        with mock.patch("builtins.print"):
            self.assertEqual(bc._pregate_confirm_one(t_hit, 0.33, False),
                             "confirmed")
        hit = bc._pregate_hit[0]
        self.assertEqual(hit["n0"], 60 * _CHUNK - int(3.0 * _SR))

    def test_a_cut_that_is_not_addressed_is_dropped(self):
        bc = self.bc
        bus = self._confirm_env("and that is the recipe")
        with mock.patch("builtins.print"):
            self.assertEqual(bc._pregate_confirm_one(
                bus.last_frame_time() - 1.5, 0.33, False), "dropped")
        self.assertIsNone(bc._pregate_hit[0])

    def test_the_video_saying_the_name_is_vetoed(self):
        bc = self.bc
        bus = self._confirm_env("Jarvis, pause the music.")
        t_hit = bus.last_frame_time() - 1.5
        loop = bc._lm.ScoreTrack()
        loop.add(t_hit + 0.4, 0.6)                 # the PC said "Jarvis"
        fake = mock.Mock(track=loop)
        self._p(bc, "_loopveto_get", lambda: fake)
        self._p(bc, "WAKE_LOOPBACK_VETO", "on", create=True)
        with mock.patch("builtins.print"):
            self.assertEqual(bc._pregate_confirm_one(t_hit, 0.5, False),
                             "vetoed")
            self._p(bc, "WAKE_LOOPBACK_VETO", "shadow", create=True)
            self.assertEqual(bc._pregate_confirm_one(t_hit, 0.5, False),
                             "confirmed")
        self.assertEqual(bc._listen_counter.snapshot().get("would_veto"), 1)

    def test_a_loudness_capture_led_by_the_video_s_name_is_dropped(self):
        bc = self.bc
        loop = bc._lm.ScoreTrack()
        self._p(bc, "_loopveto_get", lambda: mock.Mock(track=loop))
        self._p(bc, "WAKE_LOOPBACK_VETO", "on", create=True)
        self._p(bc, "_capture_clip_t0", lambda: 100.0)
        conf = {"word_t": [[0, 0.2], [1, 0.6]], "n_words": 3}
        loop.add(100.5, 0.7)
        with mock.patch("builtins.print"):
            self.assertTrue(bc._wake_turn_checks("Jarvis, pause it", conf))
            # The owner's own name, the PC silent: kept.
            self.assertFalse(bc._wake_turn_checks("Hey Jarvis, stop", {
                "word_t": [[0, 9.0], [1, 9.3]], "n_words": 3}))
            # Not led by the name: nothing to check.
            self.assertFalse(bc._wake_turn_checks("pause it", conf))

    def test_the_wake_duck_is_its_own_owner(self):
        bc = self.bc
        ducker = bc._AudioDucker()
        calls = []
        self._p(ducker, "duck", lambda: calls.append("duck"))
        self._p(ducker, "restore", lambda: calls.append("restore"))
        self._p(bc, "_audio_ducker", ducker)
        self._p(bc, "WAKE_DUCK_MODE", "on", create=True)
        self._p(bc._lm, "DUCK_CAP_S", 0.2)
        bc._wake_duck_start()
        self.assertEqual(ducker._owner_holds, {"wake": 1})
        bc._wake_duck_start()                       # exclusive: not twice
        self.assertEqual(ducker._holds, 1)
        ducker.release()                            # an anonymous release
        self.assertEqual(ducker._owner_holds, {"wake": 1},
                         "another owner's release never drops the wake hold")
        end = time.monotonic() + 3
        while ducker._owner_holds and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(ducker._owner_holds, {})   # the 8 s cap (0.2 here)
        self.assertEqual(calls.count("duck"), 1)

    def test_the_owner_release_waits_for_jarvis_to_finish_speaking(self):
        bc = self.bc
        ducker = bc._AudioDucker()
        restores = []
        self._p(ducker, "restore", lambda: restores.append(1))
        ducker.hold(owner="wake")
        prev = bc._tts_playback_active[0]
        self.addCleanup(bc._tts_playback_active.__setitem__, 0, prev)
        bc._tts_playback_active[0] = True
        ducker.release(owner="wake")
        self.assertEqual(restores, [], "no swell under his voice")
        ducker.release(owner="wake")                # held none: a no-op
        self.assertEqual(ducker._holds, 0)

    def test_no_duck_into_a_headset(self):
        bc = self.bc
        ducker = mock.Mock()
        self._p(bc, "_audio_ducker", ducker)
        self._p(bc, "WAKE_DUCK_MODE", "on", create=True)
        self._p(bc, "_loopback_obj",
                [mock.Mock(endpoint="Headset Earphone (ACME HS-1 Wireless)")])
        bc._wake_duck_start()
        ducker.hold.assert_not_called()


# ── A2: the sentence re-anchor at the wake-word-mode refusal ─────────────
class ReanchorTests(_Base):
    TEXT = "Okay so that was close. Jarvis, pause the music."

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_last_capture_audio", np.ones(10 * _SR, np.float32) * 0.01)
        self._p(bc, "_last_capture_sr", _SR)
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)

    def _conf(self):
        return {"word_t": [[i, 1.0 * i] for i in range(9)], "n_words": 9}

    def test_shadow_counts_and_logs_numbers_only(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "shadow", create=True)
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: ("unsure", 0.51, ""))
        with mock.patch("builtins.print") as p:
            self.assertIsNone(bc._wake_reanchor(self.TEXT, self._conf()))
        line = " ".join(str(c.args[0]) for c in p.call_args_list)
        self.assertIn("[wake-reanchor] shadow", line)
        self.assertIn("name at word 6 of 9", line)
        for word in ("pause", "music", "close"):
            self.assertNotIn(word, line)
        self.assertEqual(bc._listen_counter.snapshot()["reanchor_would"], 1)

    def test_on_cuts_at_the_name_checks_the_voice_and_re_decodes(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        seen = {}

        def voice(audio, sr, **k):
            seen["len"] = len(audio)
            return ("owner", 0.8, "")

        self._p(bc, "_owner_voice_over_media", voice)
        self._p(bc, "_wake_confirm_decode",
                lambda a: ("Jarvis, pause the music.", {}))
        with mock.patch("builtins.print"):
            out = bc._wake_reanchor(self.TEXT, self._conf())
        self.assertEqual(out, "Jarvis, pause the music.")
        # Cut from the name (5.0 s) minus 0.3 s.
        self.assertEqual(seen["len"], int(10 * _SR) - int(4.7 * _SR))
        self.assertEqual(len(bc._last_capture_audio), seen["len"])

    def test_a_taken_sentence_reads_its_turn_peak_at_the_cut(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: ("owner", 0.8, ""))
        self._p(bc, "_wake_confirm_decode",
                lambda a: ("Jarvis, pause the music.", {}))
        self._p(bc, "_capture_clip_t0", lambda: 100.0)
        peaks = []
        self._p(bc, "_pregate_turn_peak",
                lambda t0, text: peaks.append(t0))
        vetoed = mock.Mock(return_value=False)
        self._p(bc, "_wake_vetoed", vetoed)
        with mock.patch("builtins.print"):
            out = bc._wake_reanchor(self.TEXT, self._conf())
            self.assertFalse(bc._wake_turn_checks(out, self._conf()))
            # The next (plain) turn reads the capture's own start again.
            self.assertFalse(bc._wake_turn_checks(out, self._conf()))
        # A3's peak is read where the command starts (name 5.0 s - 0.3 s).
        self.assertAlmostEqual(peaks[0], 104.7, places=3)
        self.assertEqual(peaks[1], 100.0)
        # The re-anchor vetoed at the name's own time; the turn did not
        # check it again (the second, plain turn did).
        self.assertEqual(vetoed.call_count, 2)
        self.assertAlmostEqual(vetoed.call_args_list[0].args[0], 105.0,
                               places=3)
        self.assertIsNone(bc._reanchor_taken[0])

    def test_on_refuses_someone_else_s_voice(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: (bc._learn_gate_mod.NOT_OWNER, 0.2, ""))
        decode = mock.Mock()
        self._p(bc, "_wake_confirm_decode", decode)
        with mock.patch("builtins.print"):
            self.assertIsNone(bc._wake_reanchor(self.TEXT, self._conf()))
        decode.assert_not_called()

    def test_on_needs_word_timing_and_a_clean_re_decode(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: ("owner", 0.8, ""))
        self._p(bc, "_wake_confirm_decode",
                lambda a: ("the music.", {}))
        with mock.patch("builtins.print"):
            self.assertIsNone(bc._wake_reanchor(self.TEXT, {}))
            self.assertIsNone(bc._wake_reanchor(self.TEXT, self._conf()))
        self.assertEqual(bc._listen_counter.snapshot()["reanchor_failed"], 2)

    def test_off_and_an_addressed_line_do_nothing(self):
        bc = self.bc
        self._p(bc, "WAKE_REANCHOR_MODE", "off", create=True)
        self.assertIsNone(bc._wake_reanchor(self.TEXT, self._conf()))
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        self.assertIsNone(bc._wake_reanchor("Jarvis, stop", self._conf()))

    def test_the_main_loop_asks_only_at_the_wake_word_mode_refusal(self):
        import inspect
        src = inspect.getsource(self.bc.main)
        gate = src.index("_bg_gate, _bg_why = _bg_gate_for_turn(")
        ask = src.index("_ra_text = _wake_reanchor(text, conf)")
        refuse = src.index('print(f"  [bg-audio] {_bg_why} — ignoring non-wake')
        self.assertLess(gate, ask)
        self.assertLess(ask, refuse)
        self.assertIn('_bg_why == "wake-word mode"', src[gate:ask])
        self.assertIn("_injected_text is None", src[gate:ask])


# ── the shared media test, the minute line, A3 ───────────────────────────
class MediaTestAndLogTests(_Base):

    def test_one_reading_serves_the_music_gate_and_the_new_stages(self):
        bc = self.bc
        reads = []

        def read_state():
            reads.append(1)
            return {"music": False, "peak": 0.2, "room": False,
                    "playing": True}

        self._p(bc, "_music_read_state", read_state)
        bc._music_state["at"] = None
        self._p(bc, "MUSIC_GATE_MODE", "off", create=True)
        self.assertTrue(bc._pc_media_playing())
        self.assertTrue(bc._pc_media_playing())        # cached (TTL)
        self.assertEqual(len(reads), 1)
        self.assertFalse(bc._music_now(),
                         "the music gate's own mode still rules its answer")

    def test_the_minute_line(self):
        bc = self.bc
        now = [1000.0]
        bc._listen_counter = bc._lm.MinuteCounter(clock=lambda: now[0])
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        bc._listen_note("captures", 2)
        bc._listen_note("refused", 2)
        with mock.patch("builtins.print") as p:
            bc._listen_tick()
            now[0] = 1061.0
            bc._listen_tick()
        lines = [str(c.args[0]) for c in p.call_args_list]
        self.assertTrue(any("[listen-media] 61 s with media" in s
                            and "captures 2" in s for s in lines), lines)

    def test_the_shadow_turn_peak(self):
        bc = self.bc
        track = bc._lm.ScoreTrack()
        track.add(50.5, 0.07)
        track.add(51.2, 0.43)
        track.add(56.0, 0.9)                # outside [start-1, +3]
        self._p(bc, "_pregate_obj", [mock.Mock(track=track)])
        with mock.patch("builtins.print") as p:
            bc._pregate_turn_peak(50.0, "Hay Jarvis, pause")
        line = str(p.call_args.args[0])
        self.assertIn("peak score 0.43 (form hey)", line)
        self.assertNotIn("pause", line)

    def test_the_ambient_listener_s_hit_reaches_the_books(self):
        bc = self.bc
        bc._listen_ambient_wake_hit()
        self.assertEqual(len(bc._ambient_match._hits), 1)

    def test_boot_line_names_every_mode(self):
        bc = self.bc
        self._p(bc, "_pregate_get", lambda: None)
        self._p(bc, "_media_aec_get", lambda: None)
        with mock.patch("builtins.print") as p:
            bc._listen_media_boot()
        line = str(p.call_args.args[0])
        for part in ("re-anchor", "pre-gate", "bus", "aec", "veto",
                     "wake duck", "barge-in"):
            self.assertIn(part, line)

    def test_a_stale_media_reading_is_not_playing(self):
        bc = self.bc
        bc._music_state.update(playing=True, at=time.monotonic() - 60.0)
        self.assertFalse(bc._pc_media_playing(refresh=False),
                         "nothing refreshed it for a minute: not a 'yes'")
        bc._music_state["at"] = time.monotonic()
        self.assertTrue(bc._pc_media_playing(refresh=False))

    def test_the_tick_checks_a_missing_reference(self):
        """Failure mode 7: the PC plays (the meter) but the loopback hears
        nothing - protected playback. The minute tick notices it, the
        canceller stops counting as effective (B2 / D3 stand in), and it
        clears when the loopback hears audio again."""
        bc = self.bc
        quiet = [True]

        class Loop:
            running = True
            endpoint = "Speakers (ACME USB Audio)"
            n_written = 16000

            def rms_recent(self, seconds=1.0):
                return 0.0 if quiet[0] else 0.02

        self._p(bc, "MEDIA_AEC_MODE", "on", create=True)
        self._p(bc, "_loopback_obj", [Loop()])
        self._p(bc, "_media_aec_obj", [_FakeAEC(1.0, 1.0)])
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        bc._device_cache["last_out_name"] = "Speakers (ACME USB Audio)"
        with mock.patch("builtins.print") as p:
            bc._listen_tick()                       # quiet: the clock starts
            self.assertFalse(bc._aec_ref_missing[0])
            self.assertTrue(bc._media_aec_effective())
            bc._aec_ref_quiet_since[0] = time.monotonic() - 6.0
            bc._listen_tick()
            self.assertTrue(bc._aec_ref_missing[0])
            self.assertFalse(bc._media_aec_effective())
            quiet[0] = False
            bc._listen_tick()
            self.assertFalse(bc._aec_ref_missing[0])
        lines = " ".join(str(c.args[0]) for c in p.call_args_list)
        self.assertIn("cannot hear", lines)
        self.assertNotIn("not the one the canceller listens to", lines)

    def test_jarvis_speaking_elsewhere_is_said_once(self):
        bc = self.bc

        class Loop:
            running = True
            endpoint = "Speakers (ACME USB Audio)"
            n_written = 16000

            def rms_recent(self, seconds=1.0):
                return 0.02

        self._p(bc, "_loopback_obj", [Loop()])
        self._p(bc, "_pc_media_playing", lambda refresh=True: False)
        bc._device_cache["last_out_name"] = "Monitor (ACME HDMI)"
        with mock.patch("builtins.print") as p:
            bc._media_aec_check_reference()
            bc._media_aec_check_reference()
        said = [c for c in p.call_args_list
                if "not the one the canceller listens to" in str(c.args[0])]
        self.assertEqual(len(said), 1)

    def test_the_loopback_never_starts_with_the_canceller_off(self):
        bc = self.bc
        self._p(bc, "MEDIA_AEC_MODE", "off", create=True)
        self.assertIsNone(bc._media_aec_get())
        self.assertIsNone(bc._loopback_obj[0])


if __name__ == "__main__":
    unittest.main()
