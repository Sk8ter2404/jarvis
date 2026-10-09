"""Listening over media - the 2026-10-09 review of claude/listen-over-media.

Two reviews found what the first suite missed: a late wake confirm replayed
the command just acted on, the always-open mic handed JARVIS's own last
words to the next capture (and to the ambient listener), a segment's
overlap outlived the turn it came from, a wedged stream open made him deaf
without a word, his headset barge-in switched off with the bus, a dead
loopback still counted as a reference, the echo canceller "decided" on a
per-capture stream that re-anchors at every open, and 33 of 49 mutants of
the wiring stayed green. Each test here drives the REAL code (record_speech
through the bus's fake stream, the real gates) and fails on the code the
reviews read (3dc27a1) - see the review notes in the class docstrings.

The harness is tests/monolith/test_monolith_listen_media._Base (fake
streams feeding the real callbacks, frames carrying index markers).
"""
from __future__ import annotations

import ast
import inspect
import queue
import textwrap
import threading
import time
import unittest
from unittest import mock

import numpy as np

from tests.monolith.test_monolith_listen_media import (
    _Base, _FakeAEC, _frame, _quiet, _loud, _marker, _CHUNK, _SR)


def _jarvis(i: int) -> np.ndarray:      # JARVIS's own voice heard by the mic
    return _frame(0.05 + 0.0001 * i)


class _ReviewBase(_Base):

    def _turn_boundary(self, turn: bool) -> None:
        """The main loop's top-of-iteration: the previous capture became a
        turn (``turn``) or not."""
        bc = self.bc
        bc._turn_in_progress[0] = bool(turn)
        with mock.patch("builtins.print"):
            bc._note_turn_boundary()

    def _no_confirm_thread(self):
        """_pregate_on_trigger would start the never-exiting confirm worker:
        a stand-in that reads as alive keeps the queue observable."""
        self._p(self.bc, "_pregate_confirm_thread",
                [mock.Mock(is_alive=lambda: True)])
        q = self.bc._pregate_confirm_q
        while not q.empty():
            q.get_nowait()
        self.addCleanup(lambda: [q.get_nowait() for _ in range(q.qsize())])
        return q


# ── #1 a late wake confirm must never replay a command already acted on ──
class StaleWakeConfirmTests(_ReviewBase):
    """Review 1, finding 1 (HIGH): _pregate_take_hit only checked that a
    confirmed hit was under 10 s old. A confirm that landed after "Jarvis,
    pause the music" had become a turn re-seated the NEXT capture on that
    same command and force-started it with nobody speaking - it ran twice
    (play/pause toggles the music back on)."""

    def _a(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_quiet(i) for i in range(5)])
        n_cmd = bus.n_written                      # the command starts here
        t = self._feed_when_read(bus, st, [_loud(i) for i in range(10)]
                                 + [_quiet(1)] * 40)
        self.assertIsNotNone(self._record())
        t.join(5)
        self.assertFalse(bc._last_capture_meta[0]["pregate"])
        return bus, st, n_cmd

    def _late_hit(self, n_cmd):
        # The detector fired at the end of "Jarvis" (5 chunks in); its cut
        # starts at the command.
        self.bc._pregate_hit[0] = {"n0": n_cmd, "n_hit": n_cmd + 5 * _CHUNK,
                                   "t": time.monotonic() - 3.0, "ok": True,
                                   "seq": 1, "score": 0.4}

    def test_a_confirm_for_a_command_already_acted_on_is_dropped(self):
        bc = self.bc
        bus, st, n_cmd = self._a()
        self._turn_boundary(turn=True)            # capture A became a turn
        self._late_hit(n_cmd)
        self._feed_when_read(bus, st, [_quiet(2)] * 60)
        self.assertIsNone(self._record(timeout=2.0),
                          "silence alone must not replay the command")
        self.assertIsNone(bc._pregate_hit[0])      # consumed, not reused
        self.assertEqual(bc._listen_counter.snapshot().get("pregate_stale"),
                         1)

    def test_a_late_confirm_still_rescues_a_refused_capture(self):
        """D1's purpose kept: capture A was refused (no turn), so the late
        confirm re-seats the next capture at the name."""
        bc = self.bc
        bus, st, n_cmd = self._a()
        self._turn_boundary(turn=False)
        self._late_hit(n_cmd)
        self._feed_when_read(bus, st, [_quiet(2)] * 60)
        b = self._record(timeout=2.0)
        self.assertIsNotNone(b)
        self.assertTrue(bc._last_capture_meta[0]["pregate"])
        self.assertEqual(_marker(b, 0), round(0.03, 5))

    def test_a_name_heard_while_he_spoke_is_dropped(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        bus, st = self._open_bus(streams)
        ptok = bc._self_echo.playback_begin("Turning it up, sir.")
        self._feed_now(bus, st, [_jarvis(i) for i in range(10)])
        n_hit = bus.n_written - 2 * _CHUNK         # inside his own speech
        bc._self_echo.playback_end(ptok)
        bc._pregate_hit[0] = {"n0": n_hit - 8 * _CHUNK, "n_hit": n_hit,
                              "t": time.monotonic(), "ok": True, "seq": 2,
                              "score": 0.4}
        self._feed_when_read(bus, st, [_quiet(2)] * 60)
        self.assertIsNone(self._record(timeout=2.0))


# ── #3 the bus's pre-roll never reaches back into JARVIS's own voice ──────
class PrerollAfterPlaybackTests(_ReviewBase):
    """Review 1, finding 3: a bus capture's pre-roll is ring audio from
    before record_speech started - right after he speaks, his own tail -
    and both self-echo layers are bounded at the capture's open, so his
    "...sir." merged into the owner's next "Jarvis, turn it up" and the
    wake rule refused the line."""

    def test_his_last_words_never_seed_the_next_capture(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        line = "Pausing the music for you now, sir."
        tok = bc._self_echo.remember(line)
        ptok = bc._self_echo.playback_begin(line)
        self._feed_now(bus, st, [_jarvis(i) for i in range(12)])
        bc._self_echo.playback_end(ptok)
        bc._self_echo.refresh(tok)
        self._feed_when_read(bus, st, [_loud(i) for i in range(6)]
                             + [_quiet(0)] * 40)
        clip = self._record()
        self.assertIsNotNone(clip)
        marks = [_marker(clip, k) for k in range(len(clip) // _CHUNK)]
        self.assertFalse([m for m in marks if m >= 0.05],
                         "none of his own voice in the clip")
        self.assertEqual(marks[0], round(0.03, 5))

    def test_an_older_playback_leaves_the_pre_roll_whole(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        now = bc._self_echo.now()
        ptok = bc._self_echo.playback_begin("Good evening, sir.", at=now - 9)
        bc._self_echo.playback_end(ptok, at=now - 8)
        self._feed_now(bus, st, [_quiet(i) for i in range(15)])
        self._feed_when_read(bus, st, [_loud(i) for i in range(4)]
                             + [_quiet(60)] * 40)
        clip = self._record()
        self.assertEqual(_marker(clip, 0), round(0.0001 * 4, 5),
                         "the 12-chunk pre-roll from the ring")
        self.assertEqual(bc._last_capture_preroll[0][1], 12 * _CHUNK)


# ── #4 a segment's overlap never outlives the turn it came from ──────────
class SegmentCarryTests(_ReviewBase):
    """Review 1, finding 4: the carry left by a media segment was used by
    the first capture after the turn - also when that segment's command
    had been accepted and JARVIS had replied since. The next capture was
    force-started on ring audio holding his reply, nobody speaking."""

    def _cut(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "MEDIA_SEGMENT_S", 0.5, create=True)
        self._p(bc._lm, "SEGMENT_OVERLAP_S", 4 * _CHUNK / _SR)
        self.media = [True]
        self._p(bc, "_pc_media_playing",
                lambda refresh=True: self.media[0])
        self._p(bc, "_require_wake_runtime", True)
        self._p(bc, "_media_aec_effective", lambda: False)
        bus, st = self._open_bus(streams)
        t = self._feed_when_read(bus, st, [_loud(i) for i in range(40)])
        self.assertIsNotNone(self._record())
        t.join(5)
        self.assertEqual(bc._last_capture_meta[0]["why"], "segment")
        self.assertIsNotNone(bc._mic_bus_carry[0])
        self.media[0] = False                      # the video paused
        return bus, st

    def test_a_turn_from_the_cut_capture_drops_the_carry(self):
        bc = self.bc
        bus, st = self._cut()
        self._turn_boundary(turn=True)
        self.assertIsNone(bc._mic_bus_carry[0])
        self._feed_when_read(bus, st, [_quiet(0)] * 60)
        self.assertIsNone(self._record(timeout=2.0))

    def test_his_reply_since_the_cut_drops_the_carry(self):
        bc = self.bc
        bus, st = self._cut()
        ptok = bc._self_echo.playback_begin("I can't find that, sir.")
        self._feed_now(bus, st, [_jarvis(i) for i in range(15)])
        bc._self_echo.playback_end(ptok)
        self._feed_when_read(bus, st, [_quiet(0)] * 60)
        self.assertIsNone(self._record(timeout=2.0),
                          "his reply must not start a capture")

    def test_a_refused_segment_still_carries_its_overlap(self):
        bc = self.bc
        bus, st = self._cut()
        self._turn_boundary(turn=False)
        self.assertIsNotNone(bc._mic_bus_carry[0])
        self._feed_when_read(bus, st, [_quiet(0)] * 60)
        second = self._record(timeout=2.0)
        self.assertIsNotNone(second)
        self.assertGreaterEqual(_marker(second, 0), 0.03)


# ── #2 a wedged stream open is booked, logged and spoken ─────────────────
class WedgedBusOpenTests(_ReviewBase):
    """Review 1, finding 2: an open that never returns on the bus's owner
    thread made every capture wait 2.5 s and skip silently - no line, no
    R10 booking, no spoken warning - for ever. The owner cell stays held
    on purpose (a native open is in flight: no PortAudio reinit under
    it)."""

    def test_a_wedged_open_is_booked_logged_and_spoken(self):
        bc = self.bc
        self._bus_on()
        release = threading.Event()
        self.addCleanup(release.set)
        opens = []

        class Wedged:
            def __init__(s, *a, **k):
                opens.append(1)
                release.wait(30)            # a Pa_OpenStream that hangs
                raise RuntimeError("released by the test")

        self._p(bc.sd, "InputStream", Wedged)
        self._p(bc, "_note_input_open_failure", lambda *a, **k: None)
        self._p(bc, "MIC_SILENT_WARN_SECONDS", 0.0)
        spoken = []
        self._p(bc, "_report_silent_mic",
                lambda age, now: spoken.append(age) or True)
        self.addCleanup(bc._input_open_backoff.reset)
        with mock.patch("builtins.print"):
            bus = bc._mic_bus_get()
        printed = []
        try:
            with mock.patch("builtins.print",
                            side_effect=lambda *a, **k: printed.append(
                                " ".join(str(x) for x in a))):
                r1 = bc.record_speech(timeout=1.0)
                r2 = bc.record_speech(timeout=3.0)
            fails = bc._input_open_backoff.fails
            stalled = bus.open_stalled_s()
            held = bool(bc._mic_bus_active[0])
        finally:
            release.set()
            end = time.monotonic() + 3
            while bc._mic_bus_active[0] and time.monotonic() < end:
                time.sleep(0.01)
        self.assertIsNone(r1)
        self.assertIsNone(r2)
        self.assertEqual(len(opens), 1, "never a second open beside it")
        self.assertGreaterEqual(fails, 2, "each attempt booked (R10)")
        self.assertTrue([p for p in printed if "microphone open" in p],
                        printed)
        self.assertTrue(spoken, "the owner is told")
        self.assertGreater(stalled, 2.0)
        self.assertTrue(held, "the owner cell covers the native open")


# ── #5 his headset barge-in keeps working with the bus open ──────────────
class BusBargeInTests(_ReviewBase):
    """Review 1, finding 5: _start_barge_in_listener returned None whenever
    the bus was open, and its only replacement (WAKE_BARGEIN_MODE) ships
    'off' - his enabled headset barge-in was gone."""

    def test_the_barge_in_rule_runs_on_the_bus(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        self.addCleanup(setattr, bc, "_barge_in_interrupted", False)
        lst = bc._start_barge_in_listener()
        self.assertIsInstance(lst, bc._BusBargeIn)
        self.addCleanup(lst.close)
        self.assertEqual(len(streams), 1, "no second stream on the mic")
        self._feed_now(bus, st, [_quiet(0)] * 3)
        self.assertFalse(bc._barge_in_interrupted)
        self._feed_now(bus, st, [_loud(i) for i in range(3)])
        self.assertTrue(bc._barge_in_interrupted)
        lst.close()
        bc._barge_in_interrupted = False
        self._feed_now(bus, st, [_loud(i) for i in range(3)])
        self.assertFalse(bc._barge_in_interrupted, "closed: it stops")

    def test_a_pregate_hit_during_his_speech_is_ignored_without_bargein(self):
        """The bus pre-gate's own barge-in stays off at the defaults."""
        bc = self.bc
        q = self._no_confirm_thread()
        self.assertFalse(bc._bargein_bus_on())
        self._p(bc, "_tts_playback_active", [True])
        self._p(bc, "_mic_bus_obj", [mock.Mock(is_open=lambda: True)])
        bc._pregate_on_trigger(time.monotonic(), 0.95)
        self.assertTrue(q.empty())


# ── #6 the bus never hands his own voice to the record taps ──────────────
class BusTapPlaybackTests(_ReviewBase):
    """Review 1, finding 6: the bus fanned every frame to the record taps,
    also while JARVIS spoke; the ambient listener (on in his settings)
    transcribed his voice into the buffer the learners read - the "test
    data became facts" class."""

    def test_no_tap_frame_while_he_is_audible(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        tap = queue.Queue()
        self.assertTrue(bc.add_record_tap(tap))
        self.addCleanup(bc.remove_record_tap, tap)
        self._p(bc, "_tts_playback_active", [True])
        self._feed_now(bus, st, [_jarvis(i) for i in range(4)])
        self.assertTrue(tap.empty(), "main-loop speech")
        bc._tts_playback_active[0] = False
        ptok = bc._self_echo.playback_begin("At your service, sir.")
        self._feed_now(bus, st, [_jarvis(i) for i in range(4)])
        self.assertTrue(tap.empty(), "another thread's playback")
        now = bc._self_echo.now()
        bc._self_echo._playbacks[ptok] = [now - 2.0, now - 1.0, False]
        self._feed_now(bus, st, [_quiet(i) for i in range(3)])
        got = [tap.get(timeout=2) for _ in range(3)]
        self.assertEqual(round(float(abs(got[2][0])), 5), 0.0003)

    def test_the_tail_after_his_playback_is_held_back(self):
        bc = self.bc
        now = bc._self_echo.now()
        ptok = bc._self_echo.playback_begin("Done, sir.", at=now - 1.0)
        bc._self_echo.playback_end(ptok, at=now - 0.1)
        self.assertTrue(bc._jarvis_audible_now())
        bc._self_echo._playbacks[ptok][1] = now - 0.5
        self.assertFalse(bc._jarvis_audible_now())

    def test_mute_and_a_private_capture_close_the_taps(self):
        bc = self.bc
        self.assertTrue(bc._mic_bus_tap_allowed())
        self._p(bc, "_offthread_capture_holds_mic", lambda: True)
        self.assertFalse(bc._mic_bus_tap_allowed())
        self._p(bc, "_offthread_capture_holds_mic", lambda: False)
        self._p(bc, "_mic_muted", [True])
        self.assertFalse(bc._mic_bus_tap_allowed())


# ── #7 the 10 s cap applies only where the video would run the capture ──
class PregateCapTests(_ReviewBase):
    """Review 1, finding 7: every capture that took a confirmed hit was cut
    at 10 s, even in a quiet room - the rest of a long command landed in a
    capture without the wake word. Scaled: cap 0.6 s, real-time feed."""

    def _run(self, media):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        self._p(bc._lm, "PREGATE_CAP_S", 0.6)
        self._p(bc, "_pc_media_playing", lambda refresh=True: media)
        self._p(bc, "_media_aec_effective", lambda: False)
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_quiet(i) for i in range(4)])
        bc._pregate_hit[0] = {"n0": bus.n_written - 2 * _CHUNK,
                              "t": time.monotonic(), "ok": True, "seq": 3,
                              "score": 0.4}

        def paced():
            end = time.monotonic() + 10
            while (bus.status()["subscribers"] < 1
                   and time.monotonic() < end):
                time.sleep(0.002)
            for f in [_loud(i) for i in range(25)] + [_quiet(0)] * 30:
                st.cb(f.reshape(-1, 1), len(f), None, None)
                time.sleep(_CHUNK / _SR)
        t = threading.Thread(target=paced, daemon=True)
        t.start()
        self.addCleanup(t.join, 10)
        clip = self._record(timeout=6.0)
        return clip, dict(bc._last_capture_meta[0])

    def test_a_quiet_room_keeps_the_whole_command(self):
        clip, meta = self._run(media=False)
        self.assertTrue(meta["pregate"])
        self.assertNotEqual(meta["why"], "max")
        self.assertGreaterEqual(len(clip) // _CHUNK, 25)

    def test_over_media_the_cap_still_ends_it(self):
        clip, meta = self._run(media=True)
        self.assertTrue(meta["pregate"])
        self.assertEqual(meta["why"], "max")
        self.assertLess(len(clip) // _CHUNK, 25)


# ── #8 a loopback that stops delivering is no reference ──────────────────
class DeadLoopbackTests(_ReviewBase):
    """Review 1, finding 8: the missing-reference check read the last
    second WRITTEN, so a reader that stopped kept its last loud second for
    ever: never flagged, still "effective" (B2 segments off), nothing
    cancelled."""

    def _lb(self, age_s):
        from core import loopback_ref as lbr
        clock = [1000.0]
        lb = lbr.LoopbackReference(clock=lambda: clock[0])
        lb.write(np.full(_SR, 0.2, np.float32), t=clock[0])   # loud music
        lb._run.set()
        clock[0] += age_s
        return lb

    def _env(self, lb):
        bc = self.bc
        self._p(bc, "_loopback_obj", [lb])
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        self._p(bc, "MEDIA_AEC_MODE", "on", create=True)
        self._p(bc, "_bus_mode_on", lambda: True)
        self._p(bc, "_require_wake_runtime", True)
        bc._aec_ref_missing[0] = False
        bc._aec_tts_device_checked[0] = True
        bc._aec_ref_quiet_since[0] = time.monotonic() - 30.0
        fake = mock.Mock()
        fake.status.return_value = {"converging": True}
        self._p(bc, "_media_aec_obj", [fake])

    def test_a_dead_loopback_is_flagged_and_not_effective(self):
        bc = self.bc
        self._env(self._lb(age_s=60.0))
        self.assertFalse(bc._media_aec_effective(), "no fresh reference")
        with mock.patch("builtins.print") as p:
            bc._media_aec_check_reference()
        self.assertTrue(bc._aec_ref_missing[0])
        self.assertIn("cannot hear", " ".join(str(c.args[0])
                                              for c in p.call_args_list))
        self.assertTrue(bc._segment_active_now(), "B2 stands in")

    def test_a_live_loopback_stays_effective(self):
        bc = self.bc
        self._env(self._lb(age_s=0.05))
        with mock.patch("builtins.print"):
            bc._media_aec_check_reference()
        self.assertFalse(bc._aec_ref_missing[0])
        self.assertTrue(bc._media_aec_effective())
        self.assertFalse(bc._segment_active_now())


# ── #11 the bus goes back to the selected mic ────────────────────────────
class BusFallbackDeviceTests(_ReviewBase):
    """Review 1, finding 11: when the selected mic failed to open, the bus
    retried on the system default but booked itself as open on the
    selected index, so the selected mic was never tried again (a stream
    per capture retried it on the very next capture)."""

    def test_the_bus_retries_the_selected_mic(self):
        bc = self.bc
        from core import mic_bus as mb
        self._bus_on()
        made = []
        fail_selected = [True]

        class Picky:
            latency = 0.0

            def __init__(s, *a, callback=None, device=None, **k):
                if device == 0 and fail_selected[0]:
                    raise bc.sd.PortAudioError("busy for a moment")
                s.cb, s.device, s.closed = callback, device, False
                made.append(s)

            def start(s):
                pass

            def stop(s):
                pass

            def close(s):
                s.closed = True

        self._p(bc.sd, "InputStream", Picky)
        self._p(bc, "_note_input_open_failure", lambda *a, **k: None)
        with mock.patch("builtins.print"):
            bus = bc._mic_bus_get()
        ok, err = bus.ensure(0)
        self.assertTrue(ok, err)
        self.assertIsNone(made[-1].device, "opened on the system default")
        self.assertIsNone(bus.status()["device_actual"])
        fail_selected[0] = False
        ok, _ = bus.ensure(0)              # inside FALLBACK_RETRY_S: kept
        self.assertTrue(ok)
        self.assertEqual(len(made), 1)
        self._p(mb, "FALLBACK_RETRY_S", 0.0)
        ok, _ = bus.ensure(0)              # the window is over: retried
        self.assertTrue(ok)
        self.assertEqual(made[-1].device, 0, "back on the selected mic")
        self.assertTrue(made[0].closed)
        self.assertEqual(bus.status()["device_actual"], 0)


# ── F1 the canceller decides only on the continuous bus stream ───────────
class CancellerSettleTests(_ReviewBase):
    """Review 2, F1: after a (re)anchor the canceller's first frames still
    carry the video; on the bus (reopen, dropped frames, a reference gap)
    the suppressed copy that starts captures is silence meanwhile."""

    def test_the_suppressed_copy_is_silent_while_it_settles(self):
        bc = self.bc
        settling = [True]

        class Aec(_FakeAEC):
            def settling(self, n=0):
                return settling[0]

        aec = Aec(lin_gain=0.5, sup_gain=0.5)
        self._p(bc, "_media_aec_get", lambda: aec)
        self._p(bc, "_media_aec_bypass", lambda: "")
        x = _loud(0)
        lin, sup = bc._mic_bus_process(x, time.monotonic())
        self.assertEqual(float(np.abs(sup).max()), 0.0)
        self.assertAlmostEqual(float(np.abs(lin).max()), 0.015, places=4)
        settling[0] = False
        _lin, sup = bc._mic_bus_process(x, time.monotonic())
        self.assertAlmostEqual(float(np.abs(sup).max()), 0.015, places=4)


# ── #10 a segment cut never splits a re-anchored command in two turns ────
class ReanchorSegmentTests(_ReviewBase):
    """Review 1, finding 10: a command that crosses a 12 s segment cut. When
    the name sits in the overlap the NEXT capture starts with, the
    re-anchor leaves it to that capture (whole there; here it may be cut
    mid-command, and taking both would run it twice)."""
    TEXT = "Okay so that was close. Jarvis, pause the music."

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "_last_capture_audio",
                np.ones(10 * _SR, np.float32) * 0.01)
        self._p(bc, "_last_capture_sr", _SR)
        self._p(bc, "_pc_media_playing", lambda refresh=True: True)
        self._p(bc, "WAKE_REANCHOR_MODE", "on", create=True)
        self._p(bc, "_owner_voice_over_media",
                lambda *a, **k: ("owner", 0.8, ""))
        self._p(bc, "_wake_confirm_decode",
                lambda a: ("Jarvis, pause the music.", {}))

    def _conf(self, name_t):
        # Word 5 is "Jarvis"; words before it at 0..4 s, after it 0.3 s apart.
        t = [0.0, 1.0, 2.0, 3.0, 4.0, name_t,
             name_t + 0.3, name_t + 0.6, name_t + 0.9]
        return {"word_t": [[i, v] for i, v in enumerate(t)], "n_words": 9}

    def _meta(self, why):
        self._p(self.bc, "_last_capture_meta", [{"why": why, "bus": True,
                                                "pregate": False,
                                                "pregate_seq": 0}])

    def test_a_name_in_the_overlap_is_left_to_the_next_segment(self):
        self._meta("segment")
        with mock.patch("builtins.print") as p:
            self.assertIsNone(self.bc._wake_reanchor(self.TEXT,
                                                     self._conf(8.6)))
        self.assertIn("in the next segment",
                      " ".join(str(c.args[0]) for c in p.call_args_list))

    def test_an_earlier_name_is_taken_from_a_segment(self):
        self._meta("segment")
        with mock.patch("builtins.print"):
            self.assertIsNotNone(self.bc._wake_reanchor(self.TEXT,
                                                        self._conf(5.0)))

    def test_a_late_name_is_taken_when_no_segment_cut_the_capture(self):
        self._meta("silence")
        with mock.patch("builtins.print"):
            self.assertIsNotNone(self.bc._wake_reanchor(self.TEXT,
                                                        self._conf(8.6)))

    def test_a_cut_under_0_4_s_is_refused(self):
        self._meta("silence")
        with mock.patch("builtins.print") as p:
            self.assertIsNone(self.bc._wake_reanchor(self.TEXT,
                                                     self._conf(9.95)))
        self.assertIn("too short",
                      " ".join(str(c.args[0]) for c in p.call_args_list))


# ── review 2, F3: the wiring 33 mutants walked through ───────────────────
class MainLoopWiringTests(_ReviewBase):
    """main() is the boot entrypoint and an endless loop (no unit run), so
    its listening-over-media wiring is checked on its AST - the shape, not
    a source-string position (M01, M02, M03)."""

    @classmethod
    def _main_tree(cls, bc):
        return ast.parse(textwrap.dedent(inspect.getsource(bc.main)))

    def test_boot_builds_the_listening_modes_right_before_the_loop(self):
        tree = self._main_tree(self.bc)
        found = False
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for i, st in enumerate(body[:-1]):
                if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and getattr(st.value.func, "id", "")
                        == "_listen_media_boot"):
                    nxt = body[i + 1]
                    self.assertIsInstance(nxt, ast.While)
                    self.assertIs(getattr(nxt.test, "value", None), True,
                                  "the main loop follows the boot call")
                    found = True
        self.assertTrue(found, "_listen_media_boot() is called in main()")

    def test_a_re_anchored_line_becomes_the_turn_s_text(self):
        tree = self._main_tree(self.bc)
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            t = node.test
            if not (isinstance(t, ast.Compare)
                    and getattr(t.left, "id", "") == "_ra_text"
                    and isinstance(t.ops[0], ast.IsNot)):
                continue
            for st in node.body:
                if (isinstance(st, ast.Assign)
                        and getattr(st.targets[0], "id", "") == "text"
                        and getattr(st.value, "id", "") == "_ra_text"):
                    found = True
        self.assertTrue(found, "if _ra_text is not None: text = _ra_text")

    def test_a_wake_led_mic_turn_is_checked_and_dropped_on_a_veto(self):
        tree = self._main_tree(self.bc)
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            calls = [n for n in ast.walk(node.test)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", "") == "_wake_turn_checks"]
            if not calls:
                continue
            args = [getattr(a, "id", "") for a in calls[0].args]
            self.assertEqual(args, ["text", "conf"])
            self.assertTrue(any(isinstance(s, ast.Continue)
                                for s in node.body))
            found = True
        self.assertTrue(found, "if ... _wake_turn_checks(text, conf): "
                               "... continue")


class TurnBoundaryTests(_ReviewBase):
    """M04 / M05: the minute line (and the missing-reference check) and
    the wake duck's release run at every turn boundary."""

    def test_the_boundary_ticks_the_minute_line_and_the_duck(self):
        bc = self.bc
        tick = self._p(bc, "_listen_tick")
        duck = self._p(bc, "_wake_duck_turn_over")
        bc._note_turn_boundary()
        tick.assert_called_once_with()
        duck.assert_called_once_with()


class CaptureSignalTests(_ReviewBase):
    """M06, M07, M08, M09, M12: what a capture hands on."""

    def test_the_tts_only_canceller_retires_under_the_media_canceller(self):
        bc = self.bc
        seen = []

        class Proc:
            def process(self, chunk, **kw):
                seen.append(kw)
                return chunk

        fake_mod = mock.Mock()
        fake_mod.get_processor.return_value = Proc()
        self._p(bc, "_audio_processor", fake_mod)
        self._p(bc, "_audio_master_enabled", [True])
        self._p(bc, "_audio_aec_enabled", [True])
        x = _loud(0)
        bc._process_capture_chunk(x, _SR)
        bc._process_capture_chunk(x, _SR, media_aec=True)
        self.assertTrue(seen[0]["enable_aec"])
        self.assertFalse(seen[1]["enable_aec"])

    def test_the_silent_mic_check_reads_the_raw_mic(self):
        bc = self.bc
        from core import audio_processor as ap
        streams = self._bus_on()
        self._p(bc, "MEDIA_AEC_MODE", "on", create=True)
        self._p(bc, "_media_aec_get",
                lambda: _FakeAEC(lin_gain=0.5, sup_gain=0.5))
        self._p(bc, "_media_aec_bypass", lambda: "")
        self._p(bc, "_media_aec_effective", lambda: True)
        raws = []
        self._p(ap, "note_raw_rms", lambda rms, ts=None: raws.append(rms))
        bus, st = self._open_bus(streams)
        self._feed_when_read(bus, st, [_loud(0)] * 4 + [_quiet(0)] * 40)
        self.assertIsNotNone(self._record())
        self.assertAlmostEqual(max(raws), 0.03, places=4)

    def test_a_per_capture_clip_never_starts_before_its_stream(self):
        bc = self.bc
        self._edges()
        self._p(bc, "MIC_BUS_MODE", "off", create=True)

        class Legacy:
            def __init__(s, *a, callback=None, **k):
                s.cb = callback

            def start(s):
                for f in [_loud(0)] * 4 + [_quiet(0)] * 40:
                    s.cb(f.reshape(-1, 1), len(f), None, None)

        self._p(bc.sd, "InputStream", Legacy)
        self._p(bc, "_safe_close_stream", lambda *a, **k: None)
        self.assertIsNotNone(self._record())
        win = bc._last_capture_window[0]
        self.assertGreaterEqual(win[3], win[0])

    def test_a_seeded_capture_has_no_pre_roll_offset(self):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        bus, st = self._open_bus(streams)
        self._feed_now(bus, st, [_quiet(i) for i in range(30)])
        bc._pregate_hit[0] = {"n0": 10 * _CHUNK, "t": time.monotonic(),
                              "ok": True, "seq": 7, "score": 0.4}
        self._feed_when_read(bus, st, [_loud(1)] * 3 + [_quiet(41)] * 40)
        clip = self._record()
        self.assertTrue(bc._last_capture_meta[0]["pregate"])
        self.assertEqual(bc._last_capture_preroll[0], (len(clip), 0))

    def test_a_bus_that_closes_under_a_capture_ends_it(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        out = {}

        def run():
            with mock.patch("builtins.print"):
                out["clip"] = bc.record_speech(timeout=8.0)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        end = time.monotonic() + 5
        while bus.status()["subscribers"] < 1 and time.monotonic() < end:
            time.sleep(0.002)
        for i in range(6):
            st.cb(_loud(i).reshape(-1, 1), _CHUNK, None, None)
        time.sleep(0.3)
        # The stream closes under the capture (a device refresh / a dead
        # stream): the capture must end as a closed stream would.
        self.addCleanup(bus.resume)
        self.assertTrue(bus.suspend(timeout=3.0))
        t.join(5)
        self.assertFalse(t.is_alive(), "the capture ended")
        self.assertIsNotNone(out.get("clip"))
        self.assertEqual(bc._last_capture_meta[0]["why"], "bus")


class BusOwnershipTests(_ReviewBase):
    """M14, M18, M21, M24: one stream per device, one feed per frame."""

    def test_path_b_never_opens_beside_the_bus_owner(self):
        bc = self.bc
        self._edges()
        self._p(bc, "_mic_bus_open", lambda: False)   # its open in flight
        self._p(bc, "_mic_bus_active", [True])
        opened = mock.Mock()
        self._p(bc.sd, "InputStream", opened)
        self.assertIsNone(bc.get_mic_buffer(0.1))
        opened.assert_not_called()

    def test_the_bus_claim_yields_to_another_capture(self):
        bc = self.bc
        for cell in ("_pathb_mic_active", "_enroll_capture_active",
                     "_diag_capture_active"):
            with self.subTest(cell=cell):
                self._p(bc, cell, [True])
                self.assertFalse(bc._mic_bus_claim())
                self.assertFalse(bc._mic_bus_active[0])
                getattr(bc, cell)[0] = False
        self.assertTrue(bc._mic_bus_claim())
        bc._pa_release_owner(bc._mic_bus_active)

    def test_the_pre_gate_is_fed_once_per_frame(self):
        bc = self.bc
        streams = self._bus_on()
        bus, st = self._open_bus(streams)
        fed = []

        class Pg:
            tap = queue.Queue()
            trigger = None

            def feed(self, x, t=None):
                fed.append(float(abs(np.asarray(x)[0])))

        pg = Pg()
        bc.add_record_tap(pg.tap)                   # the bus-off feed
        self.addCleanup(bc.remove_record_tap, pg.tap)
        self._p(bc, "_pregate_obj", [pg])
        bc._pregate_attach()
        self.assertNotIn(pg.tap, bc._record_speech_taps)
        bc._pregate_attach()                        # idempotent
        self._feed_now(bus, st, [_quiet(i) for i in range(3)])
        end = time.monotonic() + 2
        while len(fed) < 3 and time.monotonic() < end:
            time.sleep(0.005)
        self.assertEqual(len(fed), 3)
        self.assertTrue(pg.tap.empty())

    def test_shadow_never_queues_a_confirm(self):
        bc = self.bc
        q = self._no_confirm_thread()
        self._p(bc, "_mic_bus_obj", [mock.Mock(is_open=lambda: True)])
        self._p(bc, "_tts_playback_active", [False])
        self._p(bc, "_tts_reply_active", [False])
        self._p(bc, "WAKE_PREGATE_MODE", "shadow", create=True)
        bc._pregate_on_trigger(time.monotonic(), 0.9)
        self.assertTrue(q.empty())
        self._p(bc, "WAKE_PREGATE_MODE", "on", create=True)
        bc._pregate_on_trigger(time.monotonic(), 0.9)
        self.assertEqual(q.qsize(), 1)


class ShadowIsInertTests(_ReviewBase):
    """M20, M22: MEDIA_AEC_MODE 'shadow' changes nothing a consumer hears -
    the taps and the pre-gate get the raw mic."""

    def _shadow_bus(self, mode):
        bc = self.bc
        streams = self._bus_on()
        self._p(bc, "MEDIA_AEC_MODE", mode, create=True)
        self._p(bc, "_media_aec_get",
                lambda: _FakeAEC(lin_gain=0.5, sup_gain=0.25))
        self._p(bc, "_media_aec_bypass", lambda: "")
        self._p(bc, "_loopback_stale", lambda: False)
        self._p(bc, "_media_aec_obj", [_FakeAEC(1.0, 1.0)])
        return self._open_bus(streams)

    def test_the_taps_get_the_raw_mic_in_shadow(self):
        bc = self.bc
        bus, st = self._shadow_bus("shadow")
        tap = queue.Queue()
        bc.add_record_tap(tap)
        self.addCleanup(bc.remove_record_tap, tap)
        self._feed_now(bus, st, [_loud(0)])
        self.assertAlmostEqual(float(abs(tap.get(timeout=2)[0])), 0.03,
                               places=5)

    def test_the_pre_gate_hears_the_raw_mic_in_shadow(self):
        bc = self.bc
        bus, st = self._shadow_bus("shadow")
        fed = []

        class Pg:
            tap = queue.Queue()
            trigger = None

            def feed(self, x, t=None):
                fed.append(float(abs(np.asarray(x)[0])))

        self._p(bc, "_pregate_obj", [Pg()])
        bc._pregate_attach()
        self._feed_now(bus, st, [_loud(0)])
        end = time.monotonic() + 2
        while not fed and time.monotonic() < end:
            time.sleep(0.005)
        self.assertAlmostEqual(fed[0], 0.03, places=5)

    def test_on_and_effective_the_pre_gate_hears_the_suppressed_copy(self):
        bc = self.bc
        bus, st = self._shadow_bus("on")
        fed = []

        class Pg:
            tap = queue.Queue()
            trigger = None

            def feed(self, x, t=None):
                fed.append(float(abs(np.asarray(x)[0])))

        self._p(bc, "_pregate_obj", [Pg()])
        bc._pregate_attach()
        self.assertTrue(bc._media_aec_effective())
        self._feed_now(bus, st, [_loud(0)])
        end = time.monotonic() + 2
        while not fed and time.monotonic() < end:
            time.sleep(0.005)
        self.assertAlmostEqual(fed[0], 0.0075, places=5)


class SmallerGuardsTests(_ReviewBase):
    """M30, M36."""

    def test_no_cancellation_into_a_headset(self):
        bc = self.bc

        class Lb:
            n_written = 16000
            endpoint = "Headset Earphone (ACME HS-1 Wireless)"

        self._p(bc, "_loopback_obj", [Lb()])
        self.assertEqual(bc._media_aec_bypass(), "headset")
        Lb.endpoint = "Speakers (ACME USB Audio)"
        self.assertEqual(bc._media_aec_bypass(), "")

    def test_a_release_by_an_owner_holding_none_drops_nothing(self):
        bc = self.bc
        ducker = bc._AudioDucker()
        restores = []
        self._p(ducker, "restore", lambda: restores.append(1))
        ducker.hold(owner="other")
        ducker.release(owner="wake")                # holds none
        self.assertEqual(ducker._owner_holds, {"other": 1})
        self.assertEqual(ducker._holds, 1)
        self.assertEqual(restores, [])


class DefaultsTests(_ReviewBase):
    """Review 1, finding 9: at the shipped defaults boot must not load the
    wake detector (openWakeWord: +~100 MB, sklearn, a second OpenMP
    runtime) - WAKE_PREGATE_MODE ships 'off' until its own canary."""

    def test_boot_at_defaults_loads_no_detector(self):
        bc = self.bc
        self._p(bc, "_pregate_obj", [None])
        made = []

        class FakeWorker:
            def __init__(s, name, **k):
                made.append(name)
                s.tap = queue.Queue()
                s.trigger = None

            def start(s):
                return True

        self._p(bc._wake_pregate_mod, "PregateWorker", FakeWorker)
        self.assertEqual(bc._pregate_mode(), "off")
        with mock.patch("builtins.print"):
            bc._listen_media_boot()
        self.assertEqual(made, [])
        self.assertIsNone(bc._pregate_obj[0])
        # 'shadow' (the owner's flip) builds it on the record tap.
        self._p(bc, "WAKE_PREGATE_MODE", "shadow", create=True)
        with mock.patch("builtins.print"):
            bc._listen_media_boot()
        self.assertEqual(made, ["mic"])
        tap = bc._pregate_obj[0].tap
        self.addCleanup(bc.remove_record_tap, tap)
        self.assertIn(tap, bc._record_speech_taps)


if __name__ == "__main__":
    unittest.main()
