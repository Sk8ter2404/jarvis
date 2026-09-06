"""Monolith tests for the camera-preview producer's survival machinery.

WHAT THIS FILE IS DEFENDING (measured on the owner's rig, 2026-09-05, two
consecutive sessions). All three camera tiles went dark ~30 minutes in and
never came back until a restart, and the session log said NOTHING about it.
py-spy against the live PID showed the producer thread ALIVE and parked inside
a C-level DirectShow call for 68+ minutes:

    Thread-67 (_face_tracking_thread)
        _open_tile_capture        (bobert_companion.py:4429)   <- cap.set(...)
        _read_side_tile_webcams   (bobert_companion.py:4561)
        _compose_kinect_preview   (bobert_companion.py:5101)
        _hud_kinect_preview_write (bobert_companion.py:5195)
        _face_tracking_thread     (bobert_companion.py:6119)

so ONE sick USB webcam stopped the loop that publishes ALL THREE previews, and
did it silently. Three claims are pinned here, and each test is written to FAIL
if the corresponding guarantee is removed:

  1. THE PRODUCER CANNOT STOP QUIETLY. Every exit shape - clean return,
     Exception, MemoryError, and the genuinely-silent SystemExit /
     KeyboardInterrupt / bare BaseException that today slip past
     ``except Exception`` and past the release+"Stopped" statements after the
     ``while`` - must reach the log with its reason and (for exceptions) its
     traceback. ProducerCannotStopSilentlyTests asserts that inside
     ``assertLogs(ERROR)``, so a missing log line is a test failure, not a
     silent pass.
  2. THE PRODUCER CANNOT HANG QUIETLY EITHER - which is what actually
     happened, and which no exit handler can ever catch. The watchdog turns a
     missing heartbeat into a loud line carrying the stuck thread's real Python
     stack.
  3. ONE SICK CAMERA IS BENCHED, NOT RETRIED FOREVER, and the healthy cameras
     keep publishing while it is.

Nothing here opens a real camera: cv2 is mocked at the exact call sites, and
the wedge is simulated with an Event so the "wedged" worker can be released and
joined at the end of the test.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith, load_monolith


class _FakeCap:
    """Stand-in for cv2.VideoCapture. Optionally blocks inside ``set()`` -
    which is the exact line the live producer was measured wedged on.

    A BARE ``_FakeCap()`` NOW DELIVERS FRAMES FOREVER (changed 2026-09-05).
    It used to open and then return ``(False, None)`` from every read, which
    was a fine stand-in while "opened" was the whole definition of a usable
    camera. It is not any more: the openers moved to Media Foundation, and a
    device another process is holding reports isOpened() True on MSMF,
    accepts set(), reads the requested resolution back out of get() — and
    delivers nothing (measured 2026-09-05: 0 frames of 20, against
    DirectShow's honest isOpened() == False). So _open_tile_capture() and the
    face-track producer now require a frame before they will hand a handle
    back, and a fake that never reads models a BROKEN camera, not a healthy
    one. Tests that want the broken shape ask for it: ``delivers=False``."""

    _NO_FRAMES = object()

    def __init__(self, opened=True, block_event=None, frames=_NO_FRAMES,
                 delivers=True):
        self._opened = opened
        self._block = block_event
        self._scripted = frames is not _FakeCap._NO_FRAMES
        self._frames = list(frames or []) if self._scripted else []
        self._delivers = delivers
        self.released = False
        self.set_calls = []
        self.reads = 0

    def isOpened(self):
        return self._opened

    def set(self, prop, val):
        self.set_calls.append((prop, val))
        if self._block is not None:
            # Wait, exactly like a wedged DirectShow media-type renegotiation.
            self._block.wait(30.0)
        return True

    def read(self):
        self.reads += 1
        if self._scripted:
            if self._frames:
                return True, self._frames.pop(0)
            return False, None
        if not self._delivers:
            return False, None
        import numpy as _np
        return True, _np.zeros((8, 8, 3), dtype=_np.uint8)

    def release(self):
        self.released = True


@requires_monolith
class ProducerCannotStopSilentlyTests(MonolithGlobalsTestCase):
    """PRIORITY 1: the loop must never die quietly, whatever kills it."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run_supervisor(self, raiser):
        """Drive _face_tracking_thread with a body that does `raiser`. Returns
        (printed_text, log_records, raised_exception_or_None)."""
        printed: list[str] = []
        raised = None

        def _body():
            if raiser is not None:
                raise raiser

        with mock.patch.object(self.bc, "_face_tracking_thread_body", _body), \
             mock.patch.object(self.bc, "_hud_camera_preview_remove"), \
             mock.patch.object(self.bc, "print",
                               lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
                               create=True):
            with self.assertLogs(level=logging.ERROR) as cm:
                try:
                    self.bc._face_tracking_thread()
                except BaseException as e:      # noqa: BLE001 - that IS the point
                    raised = e
        return "\n".join(printed), cm.records, raised

    def test_every_exit_shape_is_logged_with_its_reason(self):
        """THE test this task exists for: FAILS if any exit path is silent.

        assertLogs raises when nothing is logged at ERROR, so deleting the
        supervisor's `finally` (or narrowing `except BaseException` back to
        `except Exception`) fails this outright rather than passing quietly."""
        cases = [
            ("clean return",       None,                    "RETURNED without the stop event"),
            ("RuntimeError",       RuntimeError("boom"),     "RuntimeError"),
            ("MemoryError",        MemoryError(),            "MemoryError"),
            # The three that produce ABSOLUTELY NOTHING today: `except
            # Exception` does not catch them, the release + "Stopped" lines
            # after the `while` are plain statements so they are skipped, and
            # threading's default excepthook returns early for SystemExit.
            ("SystemExit",         SystemExit(1),            "SystemExit"),
            ("KeyboardInterrupt",  KeyboardInterrupt(),      "KeyboardInterrupt"),
            ("bare BaseException", BaseException("raw"),     "BaseException"),
        ]
        for name, exc, needle in cases:
            with self.subTest(exit=name):
                printed, records, raised = self._run_supervisor(exc)
                joined = printed + "\n" + "\n".join(r.getMessage() for r in records)
                self.assertIn("STOPPED", joined,
                              f"{name} exit produced no stop line")
                self.assertIn(needle, joined,
                              f"{name} exit did not name its reason")
                if exc is None:
                    self.assertIsNone(raised)
                else:
                    # Re-raised after logging: the supervisor reports, it does
                    # not swallow.
                    self.assertIs(raised, exc)

    def test_exception_exit_carries_a_traceback(self):
        """A logged reason without the traceback is half a diagnostic."""
        printed, records, raised = self._run_supervisor(RuntimeError("kaboom"))
        self.assertIsInstance(raised, RuntimeError)
        self.assertTrue(any(r.exc_info for r in records),
                        "no log record carried exc_info (traceback lost)")

    def test_captures_are_released_on_a_baseexception_exit(self):
        """The old teardown sat after the `while` as plain statements, so a
        BaseException skipped it and leaked every handle. It is now in the
        supervisor's `finally`."""
        cap = _FakeCap()
        entry = {"cam": {"index": 7, "label": "Test cam"}, "cap": cap}

        def _body():
            self.bc._face_track_caps[0] = [entry]
            raise SystemExit(2)

        with mock.patch.object(self.bc, "_face_tracking_thread_body", _body), \
             mock.patch.object(self.bc, "_hud_camera_preview_remove"), \
             self.assertLogs(level=logging.ERROR):
            with self.assertRaises(SystemExit):
                self.bc._face_tracking_thread()
        self.assertTrue(cap.released, "capture leaked on the SystemExit path")
        self.assertIsNone(entry["cap"])

    def test_preview_file_is_dropped_on_every_exit_path(self):
        with mock.patch.object(self.bc, "_face_tracking_thread_body",
                               lambda: (_ for _ in ()).throw(SystemExit(3))), \
             mock.patch.object(self.bc, "_hud_camera_preview_remove") as rm, \
             self.assertLogs(level=logging.ERROR):
            with self.assertRaises(SystemExit):
                self.bc._face_tracking_thread()
        rm.assert_called_once()


@requires_monolith
class ProducerWatchdogTests(MonolithGlobalsTestCase):
    """PRIORITY 1 (the half an exit handler can never cover): the producer did
    not exit, it HUNG. Only a heartbeat + watchdog can see that."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _set_heartbeat(self, at, stage="kinect composite preview write",
                       tid=None):
        self.bc._face_track_heartbeat.update({
            "at": at, "stage": stage, "iters": 42,
            "tid": threading.get_ident() if tid is None else tid,
        })

    def test_quiet_while_the_producer_is_moving(self):
        now = time.time()
        self._set_heartbeat(now - 1.0)
        self.assertFalse(self.bc._face_track_watchdog_check(now))

    def test_reports_a_stall_and_names_the_stage(self):
        now = time.time()
        self._set_heartbeat(now - 600.0)
        with self.assertLogs(level=logging.ERROR) as cm:
            self.assertTrue(self.bc._face_track_watchdog_check(now))
        msg = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("STALLED", msg)
        self.assertIn("kinect composite preview write", msg)
        self.assertIn("600s", msg)

    def test_the_stall_report_carries_the_stuck_threads_real_stack(self):
        """This is the fact that cost the owner an evening of inference: with
        it in the log, the wedge names itself."""
        now = time.time()
        self._set_heartbeat(now - 90.0)
        with self.assertLogs(level=logging.ERROR) as cm:
            self.bc._face_track_watchdog_check(now)
        msg = "\n".join(r.getMessage() for r in cm.records)
        # The dump is of THIS thread, so this test function must appear in it.
        self.assertIn("test_the_stall_report_carries_the_stuck_threads_real_stack",
                      msg)

    def test_stall_is_not_re_reported_every_tick(self):
        now = time.time()
        self._set_heartbeat(now - 90.0)
        with self.assertLogs(level=logging.ERROR):
            self.assertTrue(self.bc._face_track_watchdog_check(now))
        # A second look one tick later must stay quiet...
        self.assertFalse(self.bc._face_track_watchdog_check(now + 5.0))
        # ...but it must speak again after the re-warn interval.
        later = now + self.bc._FACE_TRACK_STALL_REWARN_S + 1.0
        with self.assertLogs(level=logging.ERROR):
            self.assertTrue(self.bc._face_track_watchdog_check(later))

    def test_recovery_is_announced(self):
        now = time.time()
        self._set_heartbeat(now - 90.0)
        with self.assertLogs(level=logging.ERROR):
            self.bc._face_track_watchdog_check(now)
        self._set_heartbeat(now)
        with self.assertLogs(level=logging.WARNING) as cm:
            self.assertFalse(self.bc._face_track_watchdog_check(now))
        self.assertIn("moving again",
                      "\n".join(r.getMessage() for r in cm.records))

    def test_silent_before_the_producer_has_ever_beaten(self):
        self.bc._face_track_heartbeat.update({"at": 0.0, "stage": "not started"})
        self.assertFalse(self.bc._face_track_watchdog_check(time.time()))

    def test_watchdog_never_raises_on_garbage_state(self):
        """A watchdog that can throw is a watchdog that stops watching."""
        self.bc._face_track_heartbeat["at"] = "not a number"
        self.assertFalse(self.bc._face_track_watchdog_check(time.time()))

    def test_liveness_snapshot_reports_the_age_and_stall_flag(self):
        now = time.time()
        self._set_heartbeat(now - 45.0)
        snap = self.bc.get_face_track_liveness(now)
        self.assertAlmostEqual(snap["age_s"], 45.0, places=3)
        self.assertTrue(snap["stalled"])
        self.assertEqual(snap["stage"], "kinect composite preview write")

    def test_watchdog_thread_stops_when_told(self):
        stop = threading.Event()
        t = threading.Thread(target=self.bc._face_track_watchdog,
                             args=(stop, 0.01), daemon=True)
        t.start()
        stop.set()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())


@requires_monolith
class CameraQuarantineTests(MonolithGlobalsTestCase):
    """PRIORITY 2: a camera that keeps needing rescue gets benched, on a
    backoff, and healed by a real frame - never by a successful open."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_two_strikes_do_not_bench_a_camera(self):
        """A blip must survive. The measured cadence gives a sick camera a
        rescue cycle every ~10 s; a USB power-save hiccup gets at most one."""
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES - 1):
            self.assertFalse(
                self.bc._camera_note_sick_cycle(0, "Right webcam", "read fail",
                                                now + i))
        self.assertFalse(self.bc._camera_is_quarantined(0, now + 5))

    def test_third_strike_benches_it_loudly(self):
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES - 1):
            self.bc._camera_note_sick_cycle(0, "Right webcam", "read fail", now + i)
        with self.assertLogs(level=logging.WARNING) as cm:
            self.assertTrue(
                self.bc._camera_note_sick_cycle(0, "Right webcam", "read fail",
                                                now + 20))
        msg = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("QUARANTINE", msg)
        self.assertIn("Right webcam", msg)
        self.assertIn("keeps", msg.lower())     # "...every OTHER camera keeps"
        self.assertTrue(self.bc._camera_is_quarantined(0, now + 21))

    def test_strikes_decay_outside_the_window(self):
        now = time.time()
        self.bc._camera_note_sick_cycle(0, "cam", "r", now)
        self.bc._camera_note_sick_cycle(0, "cam", "r", now + 1)
        # Third strike arrives long after the window - the first two are gone,
        # so this is strike #1 again and must NOT bench.
        late = now + self.bc._CAMERA_QUARANTINE_WINDOW_S + 10
        self.assertFalse(self.bc._camera_note_sick_cycle(0, "cam", "r", late))
        self.assertFalse(self.bc._camera_is_quarantined(0, late))

    def test_bench_expires_so_a_transient_fault_self_heals(self):
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(0, "cam", "r", now + i)
        with self.bc._camera_quarantine_lock:
            until = self.bc._camera_quarantine[0]["until"]
            backoff = self.bc._camera_quarantine[0]["backoff"]
        self.assertEqual(backoff, self.bc._CAMERA_QUARANTINE_BASE_S)
        self.assertTrue(self.bc._camera_is_quarantined(0, now + 10))
        self.assertTrue(self.bc._camera_is_quarantined(0, until - 1))
        self.assertFalse(self.bc._camera_is_quarantined(0, until + 1))

    def test_backoff_doubles_per_offence_and_is_capped(self):
        """Not a hot loop, and not a permanent ban."""
        now = time.time()
        seen = []
        for round_no in range(6):
            base = now + round_no * 10_000       # each round in its own window
            for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
                self.bc._camera_note_sick_cycle(0, "cam", "r", base + i)
            with self.bc._camera_quarantine_lock:
                seen.append(self.bc._camera_quarantine[0]["backoff"])
            # Expire the bench so the next round can strike again.
            self.bc._camera_quarantine[0]["until"] = 0.0
        self.assertEqual(seen[0], self.bc._CAMERA_QUARANTINE_BASE_S)
        self.assertEqual(seen[1], self.bc._CAMERA_QUARANTINE_BASE_S * 2)
        self.assertEqual(seen[2], self.bc._CAMERA_QUARANTINE_BASE_S * 4)
        self.assertLessEqual(max(seen), self.bc._CAMERA_QUARANTINE_MAX_S)
        self.assertEqual(seen[-1], self.bc._CAMERA_QUARANTINE_MAX_S)

    def test_only_a_real_frame_lifts_the_bench(self):
        """The sick camera OPENED fine every 10 s and still delivered nothing,
        so an open must never count as health."""
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(0, "Right webcam", "r", now + i)
        self.assertTrue(self.bc._camera_is_quarantined(0, now + 5))
        with self.assertLogs(level=logging.WARNING) as cm:
            self.bc._camera_note_healthy(0, "Right webcam")
        self.assertIn("quarantine lifted",
                      "\n".join(r.getMessage() for r in cm.records))
        self.assertFalse(self.bc._camera_is_quarantined(0, now + 5))
        # Strikes and backoff reset too, so the next fault gets the full
        # tolerance again rather than an instant re-bench.
        with self.bc._camera_quarantine_lock:
            self.assertEqual(self.bc._camera_quarantine[0]["strikes"], [])
            self.assertEqual(self.bc._camera_quarantine[0]["backoff"], 0.0)

    def test_already_benched_camera_does_not_re_escalate(self):
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(0, "cam", "r", now + i)
        self.assertFalse(self.bc._camera_note_sick_cycle(0, "cam", "r", now + 6))
        with self.bc._camera_quarantine_lock:
            self.assertEqual(self.bc._camera_quarantine[0]["count"], 1)

    def test_bookkeeping_never_raises(self):
        self.assertFalse(self.bc._camera_note_sick_cycle(object(), "cam", "r"))
        self.assertFalse(self.bc._camera_is_quarantined(object()))
        self.bc._camera_note_healthy(object())      # must not raise

    def test_quarantine_is_surfaced_on_get_camera_health(self):
        """PRIORITY 3: reuse the existing health surface, do not invent a new
        channel."""
        now = time.time()
        idx = self.bc.CAMERAS[1]["index"]
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(idx, "Right webcam",
                                            "read failures", now + i)
        health = self.bc.get_camera_health()
        self.assertIn(idx, health)
        self.assertTrue(health[idx]["quarantined"])
        self.assertIn("read failures", health[idx]["quarantine_reason"])
        self.assertEqual(health[idx]["quarantine_count"], 1)
        # A camera that has never misbehaved reports the honest false.
        other = self.bc.CAMERAS[0]["index"]
        self.assertFalse(health[other]["quarantined"])
        self.assertIsNone(health[other]["quarantine_reason"])
        # And the dict stays STRICTLY index-keyed (no new string keys).
        self.assertTrue(all(isinstance(k, int) for k in health))


@requires_monolith
class BoundedCameraIoTests(MonolithGlobalsTestCase):
    """The wedge itself: an unbounded DirectShow call under the global camera
    lock. Bounded now, so the producer walks away instead of parking."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_healthy_open_still_returns_the_capture_with_one_open(self):
        cap = _FakeCap()
        with mock.patch.object(self.bc.cv2, "VideoCapture",
                               return_value=cap) as vc:
            got = self.bc._open_tile_capture(3)
        self.assertIs(got, cap)
        self.assertEqual(vc.call_count, 1, "healthy path must not add opens")
        self.assertFalse(cap.released)

    def test_a_wedged_open_no_longer_parks_the_producer(self):
        """THE regression test for the measured hang: cap.set() never returns.

        Before the fix this call never came back and took every camera tile
        with it. It must now return None promptly and leave the loop free."""
        gate = threading.Event()
        cap = _FakeCap(block_event=gate)
        try:
            with mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap), \
                 mock.patch.object(self.bc, "_CAMERA_OPEN_TIMEOUT_S", 0.3):
                t0 = time.monotonic()
                with self.assertLogs(level=logging.WARNING) as cm:
                    got = self.bc._open_tile_capture(0)
                elapsed = time.monotonic() - t0
            self.assertIsNone(got)
            self.assertLess(elapsed, 5.0,
                            "the bounded open did not bound anything")
            self.assertIn("wedged", "\n".join(r.getMessage() for r in cm.records))
        finally:
            gate.set()
            time.sleep(0.2)
        # The abandoned worker tears its own handle down (inside the lock), so
        # a late-returning open does not leak a device.
        for _ in range(50):
            if cap.released:
                break
            time.sleep(0.05)
        self.assertTrue(cap.released,
                        "abandoned open worker leaked its capture handle")

    def test_repeated_wedges_bench_the_camera(self):
        gate = threading.Event()
        try:
            with mock.patch.object(self.bc.cv2, "VideoCapture",
                                   side_effect=lambda *a, **k: _FakeCap(block_event=gate)), \
                 mock.patch.object(self.bc, "_CAMERA_OPEN_TIMEOUT_S", 0.1):
                for _ in range(self.bc._CAMERA_QUARANTINE_STRIKES):
                    self.bc._open_tile_capture(0)
            self.assertTrue(self.bc._camera_is_quarantined(0))
        finally:
            gate.set()
            time.sleep(0.3)

    def test_a_benched_camera_is_never_opened_again(self):
        now = time.time()
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(0, "Right webcam", "r", now + i)
        with mock.patch.object(self.bc.cv2, "VideoCapture") as vc:
            self.assertIsNone(self.bc._open_tile_capture(0))
        vc.assert_not_called()

    def test_guarded_release_never_blocks_the_producer_on_a_busy_lock(self):
        """A plain `with _camera_io_lock:` around a release is how the producer
        wedged a SECOND time once an abandoned worker owned the lock."""
        holder_in = threading.Event()
        holder_out = threading.Event()

        def _hold():
            with self.bc._camera_io_lock:
                holder_in.set()
                holder_out.wait(10.0)

        t = threading.Thread(target=_hold, daemon=True)
        t.start()
        self.assertTrue(holder_in.wait(5.0))
        cap = _FakeCap()
        try:
            with self.assertLogs(level=logging.WARNING) as cm:
                ok = self.bc._release_capture_guarded(cap, 0, "Right webcam",
                                                      timeout=0.2)
            self.assertFalse(ok)
            self.assertFalse(cap.released,
                             "released OUTSIDE the lock - that is the heap "
                             "corruption this guard exists to avoid")
            self.assertIn("could not release",
                          "\n".join(r.getMessage() for r in cm.records))
            # Queued (strong reference kept), so CPython never runs the
            # destructor off-lock - but see DeferredCameraReleaseTests: queued
            # is NOT abandoned, the reaper still owes this handle a release.
            self.assertIn(cap, [i["cap"]
                                for i in self.bc._camera_pending_releases])
        finally:
            holder_out.set()
            t.join(timeout=5.0)

    def test_the_loops_own_hd_opener_is_bounded_too(self):
        """The loop's _open_capture is the SAME shape that wedged - cv2
        .VideoCapture + cap.set() under _camera_io_lock - and the session log
        shows it hitting the sick device every ~10 s. Which of the two wedged
        first was luck, so both are bounded. (_open_capture is a closure inside
        the producer body, so the wiring is asserted at the source, exactly as
        OpenCaptureByNameTests already does for the name resolution.)"""
        import inspect
        src = inspect.getsource(self.bc._face_tracking_thread_body)
        # Renamed from _dshow_open on 2026-09-05: the opener is no longer
        # DirectShow-specific, it goes through _camera_open().
        self.assertIn("_open_capture_bounded(idx, _backend_open", src)
        self.assertIn("_CAMERA_LOOP_OPEN_TIMEOUT_S", src)
        # Generous on purpose: cv2 can legitimately spend 20-30 s retrying a
        # dead index, so the cap must sit ABOVE that or it would abandon opens
        # that were going to succeed.
        self.assertGreater(self.bc._CAMERA_LOOP_OPEN_TIMEOUT_S, 30.0)

    def test_guarded_release_releases_when_the_lock_is_free(self):
        cap = _FakeCap()
        self.assertTrue(self.bc._release_capture_guarded(cap, 0, "cam"))
        self.assertTrue(cap.released)
        self.assertNotIn(cap, [i["cap"]
                               for i in self.bc._camera_pending_releases])


@requires_monolith
class SickCameraDoesNotStopTheOthersTests(MonolithGlobalsTestCase):
    """The headline behaviour: one benched camera, everything else publishing.

    This exercises the exact call chain that wedged -
    _read_side_tile_webcams -> _open_tile_capture - with the right webcam
    (index 0, the real culprit) benched and the left one (index 2) healthy."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        import numpy as np
        self.np = np
        self.left_frame = np.full((8, 8, 3), 11, dtype=np.uint8)
        self.bc._kinect_tile_caps.update({"left": None, "right": None})
        self.bc._kinect_tile_frames.update({"left": None, "right": None})
        self.bc._kinect_tile_last_read.update({"left": 0.0, "right": 0.0})

    def _bench_right(self, now):
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(0, "Right webcam", "read fail",
                                            now + i)

    def test_left_keeps_publishing_while_right_is_benched(self):
        now = time.time()
        self._bench_right(now)
        with self.bc._camera_state_lock:
            self.bc._camera_latest_frame[2] = self.left_frame
            self.bc._camera_last_frame_at[2] = now
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 2, "right": 0}), \
             mock.patch.object(self.bc, "_open_tile_capture") as opener:
            out = self.bc._read_side_tile_webcams(now)
        self.assertIs(out["left"], self.left_frame,
                      "the healthy tile stopped publishing")
        self.assertIsNone(out["right"], "a benched camera must show no frame")
        opener.assert_not_called()

    def test_the_benched_camera_is_not_reopened_by_the_compositor(self):
        """This is the precise regression. The fast path declines the moment
        the loop's cached frame goes stale (2.0 s) - i.e. exactly when the
        camera is sick - and the fallback then re-opened the wedging device on
        every composite."""
        now = time.time()
        self._bench_right(now)
        with self.bc._camera_state_lock:
            # Deliberately STALE: older than _KINECT_PREVIEW_TILE_REUSE_MAX_AGE,
            # so without the gate this would fall through to the open.
            self.bc._camera_latest_frame[0] = self.left_frame
            self.bc._camera_last_frame_at[0] = (
                now - self.bc._KINECT_PREVIEW_TILE_REUSE_MAX_AGE - 5.0)
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 2, "right": 0}), \
             mock.patch.object(self.bc, "_open_tile_capture") as opener:
            out = self.bc._read_side_tile_webcams(now)
        self.assertIsNone(out["right"])
        for call in opener.call_args_list:
            self.assertNotEqual(call.args[0], 0,
                                "re-opened the camera it had already benched")

    def test_benching_drops_the_tiles_own_handle_without_racing_a_release(self):
        now = time.time()
        self._bench_right(now)
        cap = _FakeCap()
        self.bc._kinect_tile_caps["right"] = cap
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 2, "right": 0}), \
             mock.patch.object(self.bc, "_open_tile_capture"):
            self.bc._read_side_tile_webcams(now)
        self.assertTrue(cap.released)
        self.assertIsNone(self.bc._kinect_tile_caps["right"])

    def test_the_healthy_fast_path_is_untouched_and_opens_nothing(self):
        """No regression: the side tiles still reuse the face-track loop's
        cached frames, and the number of DirectShow opens stays at zero."""
        now = time.time()
        right_frame = self.np.full((8, 8, 3), 22, dtype=self.np.uint8)
        with self.bc._camera_state_lock:
            self.bc._camera_latest_frame[2] = self.left_frame
            self.bc._camera_last_frame_at[2] = now
            self.bc._camera_latest_frame[0] = right_frame
            self.bc._camera_last_frame_at[0] = now
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name") as res, \
             mock.patch.object(self.bc, "_open_tile_capture") as opener:
            out = self.bc._read_side_tile_webcams(now)
        self.assertIs(out["left"], self.left_frame)
        self.assertIs(out["right"], right_frame)
        opener.assert_not_called()
        # The fast path must not even need a DirectShow name enumeration.
        res.assert_not_called()

    def test_the_bench_expires_and_the_camera_comes_back(self):
        """A genuinely transient fault still self-heals - on a backoff, not a
        hot 10-second reopen loop."""
        now = time.time()
        self._bench_right(now)
        with self.bc._camera_quarantine_lock:
            until = self.bc._camera_quarantine[0]["until"]
        # The bench really is ~a minute long, not a 10-second hot retry.
        self.assertGreaterEqual(until - now, self.bc._CAMERA_QUARANTINE_BASE_S)
        after = until + 1.0
        # Two frames: the mocked opener serves BOTH slots from this one fake,
        # and the left slot reads first.
        cap = _FakeCap(frames=[self.np.full((8, 8, 3), 33, dtype=self.np.uint8),
                               self.np.full((8, 8, 3), 44, dtype=self.np.uint8)])
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 2, "right": 0}), \
             mock.patch.object(self.bc, "_open_tile_capture",
                               return_value=cap) as opener:
            out = self.bc._read_side_tile_webcams(after)
        # The tile opener takes the CAMERAS name as well as the index since
        # 2026-09-05: matching by name is what lets Media Foundation be given
        # the right device without a DirectShow enumeration (the two backends
        # do not index the same list).
        self.assertIn(0, [c.args[0] for c in opener.call_args_list])
        self.assertIsNotNone(out["right"], "the retry never produced a frame")


# --- the reentrancy invariant -------------------------------------------
#
# _open_capture_bounded moved the `with _camera_io_lock:` acquire OFF the
# caller and ONTO a throwaway worker thread. _camera_io_lock is an RLock, so
# the old code - which called the opener while already holding it - was fine:
# reentrant on the SAME thread. It is not fine any more. The worker is a
# DIFFERENT thread and cannot take a lock its own joiner is sitting on, so any
# caller holding the lock across a bounded open is guaranteed to burn the whole
# timeout, get None, log a fabricated "DirectShow is wedged" and score a
# quarantine strike against a device that never blocked.
#
# The soft-wake path did exactly that (measured 2026-09-05: the full
# _CAMERA_LOOP_OPEN_TIMEOUT_S stall on EVERY wake, no camera involved), which
# killed the "woke via release+reopen" recovery outright and froze all three
# tiles past _CAMERA_PREVIEW_STALE_S while the producer sat on the process-wide
# lock. These tests pin the rule: the first EXECUTES the deadlock so it is
# proven rather than asserted, the second walks the SHIPPING module's AST so it
# fails again wherever the shape comes back, the third proves the checker has
# teeth.

_BOUNDED_OPENERS = ("_open_capture_bounded", "_open_capture", "_open_tile_capture")


def _holds_camera_io_lock(node) -> bool:
    """True if `node` is a ``with _camera_io_lock:`` statement."""
    import ast
    if not isinstance(node, (ast.With, ast.AsyncWith)):
        return False
    return any(isinstance(i.context_expr, ast.Name)
               and i.context_expr.id == "_camera_io_lock"
               for i in node.items)


def _bounded_opens_under_the_lock(source: str):
    """Every call to a bounded opener sitting LEXICALLY inside a
    ``with _camera_io_lock:`` block, as (enclosing_function, opener, lineno).

    Deliberately does NOT carry the lock state into a nested function body: a
    closure merely DEFINED under the lock (_dshow_open, _do_open) is invoked by
    the worker thread, which is the whole point of the design."""
    import ast
    tree = ast.parse(source)
    hits = []

    def walk(node, fn_name, under_lock):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, child.name, False)   # fresh lock state per body
                continue
            if isinstance(child, ast.Lambda):
                continue
            if under_lock and isinstance(child, ast.Call):
                fn = child.func
                if isinstance(fn, ast.Name) and fn.id in _BOUNDED_OPENERS:
                    hits.append((fn_name, fn.id, child.lineno))
            walk(child, fn_name, under_lock or _holds_camera_io_lock(child))

    walk(tree, "<module>", False)
    return hits


@requires_monolith
class BoundedOpenNeverRunsUnderTheCameraLockTests(MonolithGlobalsTestCase):

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_a_bounded_open_cannot_finish_while_the_caller_holds_the_lock(self):
        """EXECUTED, not asserted: the exact deadlock the wake path shipped.

        `opener` stands in for _dshow_open / _do_open - it takes
        _camera_io_lock itself, which _open_capture_bounded's docstring
        REQUIRES of every opener. With the lock free it returns instantly;
        called from a thread that already holds the lock it can only time out.
        """
        ran = {"n": 0}
        made = _FakeCap()

        def opener():
            with self.bc._camera_io_lock:
                ran["n"] += 1
                return made

        # Control: lock free -> the worker gets in, ONE open, prompt success.
        # Drain first: earlier tests in this file deliberately strand gated
        # workers on _camera_io_lock, and a control that merely queued behind
        # one would time the WRONG thing.
        self.assertTrue(self.bc._camera_io_lock.acquire(timeout=20.0),
                        "camera I/O lock still held by an earlier test")
        self.bc._camera_io_lock.release()
        t0 = time.monotonic()
        got = self.bc._open_capture_bounded(0, opener, label="control",
                                            timeout=10.0)
        control_elapsed = time.monotonic() - t0
        self.assertIs(got, made)
        self.assertEqual(ran["n"], 1)
        self.assertLess(control_elapsed, 5.0,
                        "the control open was not prompt - test is unsound")

        # The pre-fix wake shape: this thread already owns the RLock.
        ran["n"] = 0
        t0 = time.monotonic()
        with self.bc._camera_io_lock:
            with self.assertLogs(level=logging.WARNING) as cm:
                got = self.bc._open_capture_bounded(0, opener,
                                                    label="wake path",
                                                    timeout=0.6)
            held_elapsed = time.monotonic() - t0
            self.assertIsNone(got, "the worker somehow took a lock this thread "
                                   "holds - RLock is not cross-thread reentrant")
            self.assertEqual(ran["n"], 0, "opener ran while the lock was held?")
            self.assertGreaterEqual(held_elapsed, 0.6,
                                    "it must have burned the FULL timeout")
            # It must SAY something - a give-up that logs nothing is the
            # silent-death bug class this whole file exists for. The exact
            # wording belongs to _open_capture_bounded and is asserted
            # there, not pinned here.
            self.assertTrue(cm.records, "the give-up was silent")
        # ...and that log line was a lie: nothing was wedged. The opener sails
        # through the instant this thread drops the lock.
        for _ in range(50):
            if ran["n"]:
                break
            time.sleep(0.05)
        self.assertEqual(ran["n"], 1,
                         "the 'wedged' opener was only ever waiting on us")

    def test_no_bounded_opener_is_called_while_holding_the_camera_io_lock(self):
        """THE regression test. Before the fix this found

            _face_tracking_thread_body -> _open_capture   (the soft-wake reopen)

        which guaranteed a _CAMERA_LOOP_OPEN_TIMEOUT_S stall on every wake.
        Reads the SHIPPING file, so it catches the shape anywhere in the
        module - not only where it happened to be."""
        source = io.open(self.bc.__file__, encoding="utf-8").read()
        hits = _bounded_opens_under_the_lock(source)
        self.assertEqual(
            hits, [],
            "bounded opener(s) called while _camera_io_lock is held - the "
            "worker thread can never acquire it, so these calls can only "
            "time out: "
            + "; ".join("%s() calls %s() at %s:%d" % (fn, op, self.bc.__file__, ln)
                        for fn, op, ln in hits))

    def test_the_invariant_checker_actually_catches_the_shape(self):
        """A checker that finds nothing is worthless unless it CAN find
        something. Feed it the pre-fix shape; it must fire."""
        pre_fix = ("def producer():\n"
                   "    with _camera_io_lock:\n"
                   "        cap.release()\n"
                   "        new_c = _open_capture(cam)\n")
        self.assertEqual(_bounded_opens_under_the_lock(pre_fix),
                         [("producer", "_open_capture", 4)])
        # ...and the shipping shape - closure defined under the lock, opener
        # called outside it - must NOT trip it.
        fixed = ("def producer():\n"
                 "    _release_capture_guarded(cap, 0, 'x')\n"
                 "    new_c = _open_capture(cam)\n"
                 "def opener():\n"
                 "    with _camera_io_lock:\n"
                 "        return cv2.VideoCapture(0)\n")
        self.assertEqual(_bounded_opens_under_the_lock(fixed), [])


def _open_capture_code(bc):
    """The loop's ``_open_capture`` code object.

    It is a closure inside ``_face_tracking_thread_body``, so it cannot be
    imported - it can only be reached through the enclosing function's
    constants."""
    for const in bc._face_tracking_thread_body.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == "_open_capture":
            return const
    raise AssertionError(
        "_open_capture is gone from _face_tracking_thread_body - this file's "
        "extraction is stale, not passing")


def _extract_open_capture(bc):
    """Hand back the loop's REAL ``_open_capture`` as a callable.

    Rebinding the SHIPPED code object to the monolith's globals runs the real
    thing rather than a paraphrase of it, which matters here: the whole point
    of these tests is what the producer PRINTS, and a copy would print whatever
    the copy says. Legal only because the code object has no free variables
    (everything it touches is a module global) - asserted, so a refactor that
    adds one fails loudly instead of quietly testing nothing."""
    code = _open_capture_code(bc)
    assert code.co_freevars == (), (
        f"_open_capture grew free variables {code.co_freevars!r} - re-derive "
        f"this extraction instead of trusting it")
    return types.FunctionType(code, bc.__dict__, "_open_capture")


# "opened <cam> at index N", in any wording.
_CLAIMS_AN_OPEN = re.compile(r"opened .*at index")


@requires_monolith
class AnOpenedLineMeansTheProducerHoldsThatCameraTests(MonolithGlobalsTestCase):
    """``  [face-track] opened <cam> at index N`` is LOAD-BEARING.

    It is the line the owner reconstructs camera timelines from ("read failure
    #25 -> opened -> woke"), so it must mean exactly one thing: this call
    handed the producer a capture the producer is now using.

    Bounding the open broke that. The announcement lived INSIDE ``_dshow_open``
    - i.e. on the throwaway worker, and before ``_open_capture_bounded._run``
    ever looks at ``abandoned`` - so a worker whose joiner had already walked
    away still printed it, for a handle ``_run`` released and discarded one
    line later. Measured sequence, in this order:

        [face-track] usb 2.0 camera (index 0) did not finish opening within
            0.5s - DirectShow is wedged inside the open. Abandoning the attempt
        [face-track] opened usb 2.0 camera at index 0        <- LIE
        <handle released and thrown away>

    and because the lie landed LAST it read as the camera having come back.
    These tests fail if the announcement ever moves back onto the worker."""

    CAM = {"index": 4, "label": "usb 2.0 camera"}

    def setUp(self):
        self.open_capture = _extract_open_capture(self.bc)
        self.printed: list[str] = []

    def _print_patch(self):
        return mock.patch.object(
            self.bc, "print",
            lambda *a, **k: self.printed.append(" ".join(str(x) for x in a)),
            create=True)

    def _claims(self):
        return [ln for ln in self.printed if _CLAIMS_AN_OPEN.search(ln)]

    def test_a_kept_handle_is_announced_exactly_once(self):
        """The honest half: an open the loop goes on to USE is still reported,
        with the index actually opened (the name-shuffle signal)."""
        cap = _FakeCap()
        with mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap), \
             mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
             self._print_patch():
            got = self.open_capture(dict(self.CAM))
        self.assertIs(got, cap)
        self.assertFalse(cap.released)
        self.assertEqual(
            self._claims(),
            ["  [face-track] opened usb 2.0 camera at index 4"],
            f"expected exactly one honest open line, got {self.printed!r}")

    def test_a_failed_open_claims_nothing(self):
        cap = _FakeCap(opened=False)
        with mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap), \
             mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
             self._print_patch():
            got = self.open_capture(dict(self.CAM))
        self.assertIsNone(got)
        self.assertEqual(self._claims(), [],
                         f"claimed an open that never happened: {self.printed!r}")

    def test_an_abandoned_open_never_says_it_opened_the_camera(self):
        """THE regression test. The worker wedges, the joiner abandons it, and
        THEN the wedge clears and the open succeeds - the exact live sequence.
        Nothing in that transcript may claim the camera opened, because the
        handle is released and discarded."""
        gate = threading.Event()
        cap = _FakeCap(block_event=gate)
        try:
            with mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap), \
                 mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
                 mock.patch.object(self.bc, "_CAMERA_LOOP_OPEN_TIMEOUT_S", 0.3), \
                 self._print_patch():
                got = self.open_capture(dict(self.CAM))
                self.assertIsNone(got, "the bounded open did not bound anything")
                self.assertTrue(
                    any("Abandoning the attempt" in ln for ln in self.printed),
                    f"no abandon line at all: {self.printed!r}")
                # Now let the wedged DirectShow call finish. It opens fine -
                # that is exactly what the sick device DOES - and the worker
                # must throw the handle away without claiming a success.
                gate.set()
                for _ in range(100):
                    if cap.released:
                        break
                    time.sleep(0.05)
            self.assertTrue(cap.released,
                            "abandoned worker leaked its late handle")
            self.assertEqual(
                self._claims(), [],
                f"claimed an open for a handle it threw away: {self.printed!r}")
            # It must not go silent either: the late completion is a real
            # diagnostic (this camera OPENS fine and then fails to READ), so it
            # is reported - in words that cannot be misread as a recovery.
            late = [ln for ln in self.printed
                    if "finished opening AFTER the attempt was abandoned" in ln]
            self.assertEqual(len(late), 1,
                             f"late completion unreported: {self.printed!r}")
            self.assertIn("NOT using it", late[0])
            abandon_at = next(i for i, ln in enumerate(self.printed)
                              if "Abandoning the attempt" in ln)
            self.assertGreater(self.printed.index(late[0]), abandon_at)
        finally:
            gate.set()

    def test_the_worker_side_opener_cannot_print_at_all(self):
        """Structural belt: ``_backend_open`` runs on the abandonable worker,
        so it must not be able to announce anything. Asserted on the code
        object, so it cannot be defeated by re-wording the message.

        RENAMED from ``_dshow_open`` on 2026-09-05, when the opener stopped
        being DirectShow-specific and moved to _camera_open()."""
        opener = [c for c in _open_capture_code(self.bc).co_consts
                  if isinstance(c, types.CodeType)
                  and c.co_name == "_backend_open"]
        self.assertEqual(len(opener), 1, "_backend_open moved or was renamed")
        self.assertNotIn(
            "print", opener[0].co_names,
            "_backend_open prints from the throwaway worker again - an "
            "abandoned open will narrate itself as a success")

    def test_the_shared_opener_cannot_print_either(self):
        """The same rule, one level down.

        _backend_open now delegates to the module-level _camera_open(), so the
        structural check above would pass while _camera_open printed the
        backend decision straight into the session transcript from the
        throwaway worker. It did, briefly, on 2026-09-05. Everything
        _camera_open has to say goes through logging instead."""
        self.assertNotIn(
            "print", self.bc._camera_open.__code__.co_names,
            "_camera_open prints from the throwaway open worker - an abandoned "
            "open will narrate itself into the transcript after the abandon "
            "line, exactly as an 'opened' line used to")



class _LockWitnessCap(_FakeCap):
    """A _FakeCap that records, at release() time, whether _camera_io_lock was
    held by SOMEBODY.

    _camera_io_lock is an RLock and reentrancy is per-thread, so asking from the
    releasing thread would always say "free" even while we hold it. The only
    honest question is asked from a different thread: if a 0.05 s acquire there
    fails, the lock is held."""

    def __init__(self, io_lock, **kw):
        super().__init__(**kw)
        self._io_lock = io_lock
        self.lock_was_held_at_release = None

    def release(self):
        box: dict = {}

        def _probe():
            got = self._io_lock.acquire(timeout=0.05)
            box["free"] = bool(got)
            if got:
                self._io_lock.release()

        t = threading.Thread(target=_probe, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.lock_was_held_at_release = (box.get("free") is False)
        super().release()


@requires_monolith
class DeferredCameraReleaseTests(MonolithGlobalsTestCase):
    """A BUSY camera I/O lock is not a WEDGED one, and a live capture handed to
    the guard must not be leaked for the rest of the session.

    THE DEFECT THIS PINS (found reviewing the resilience work above, 2026-09-05).
    _release_capture_guarded gave up after _CAMERA_IO_LOCK_TIMEOUT_S = 5.0 s,
    appended the LIVE handle to _camera_abandoned_handles - a list that was
    appended to in exactly one place and drained in none - and printed "the
    camera I/O lock is held by a wedged DirectShow call". Both halves were
    unfounded:

      * THE CAUSE WAS NEVER ESTABLISHED. The acquire timeout proves only that we
        did not get the lock in 5 s. skills/self_diagnostic._probe_webcam_locked
        holds that same lock across cv2.VideoCapture on indices 0, 1 AND 2 plus
        a PowerShell PnP query, every 30 minutes, concurrently with the tracker -
        its own acquire(2.5) was measured returning False at 01:22:53, 01:52:53
        and 02:22:53. _probe_camera_index holds it for its whole per-index
        budget and budgets its OWN wait at (timeout + 0.5) x CAMERA_PROBE_MAX
        (~42 s) for exactly that reason. Routine contention, announced as a
        DirectShow wedge.
      * THE HANDLE WAS LIVE. Parking it kept its DirectShow filter graph - and
        so the USB device - open for the remaining life of the process, so every
        later reopen of that index had to contend with a graph JARVIS itself
        still held. That is this camera's documented failure mode, made
        permanent by the code meant to survive it.

    Every test in this class fails against that code: three raise AttributeError
    (no queue, no reaper, nothing to drain) and the message tests fail on the
    wording.

    Nothing here opens a real camera - _FakeCap only, and the "busy" lock is a
    thread holding the real RLock, released and joined in every finally.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def tearDown(self):
        # Never leave a reaper spinning on a queue the next test owns. (The
        # harness also restores _camera_pending_releases in place, which makes
        # any surviving reaper drain-to-empty and exit on its own.)
        try:
            with self.bc._camera_pending_lock:
                self.bc._camera_pending_releases[:] = []
        except Exception:
            pass
        super().tearDown()

    # -- helpers ---------------------------------------------------------
    def _busy_io_lock(self):
        """Hold _camera_io_lock from another thread until the returned Event is
        set. Returns (release_event, thread) - the caller MUST set + join."""
        holder_in = threading.Event()
        holder_out = threading.Event()

        def _hold():
            with self.bc._camera_io_lock:
                holder_in.set()
                holder_out.wait(30.0)

        t = threading.Thread(target=_hold, daemon=True, name="test-lock-holder")
        t.start()
        self.assertTrue(holder_in.wait(5.0), "helper never took the lock")
        return holder_out, t

    def _wait_until(self, pred, timeout=15.0, msg=""):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            time.sleep(0.02)
        self.fail(msg or "condition never became true")

    def _queued_caps(self):
        with self.bc._camera_pending_lock:
            return [i["cap"] for i in self.bc._camera_pending_releases]

    # -- the headline ----------------------------------------------------
    def test_a_busy_lock_does_not_leak_the_camera_for_the_session(self):
        """THE defect, asserted PURELY on behaviour.

        Deliberately touches no symbol the fix introduced - only
        _release_capture_guarded, the real _camera_io_lock and the handle's own
        released flag - so it fails against the old code because the camera was
        never released, not because a name is missing. The old code could not
        pass it at all: _camera_abandoned_handles had one append site and zero
        drain sites, so a handle that went in never came out.

        Runs at the real _CAMERA_RELEASE_RETRY_S cadence (~2 s) for the same
        reason: a test that has to reach inside the fix to go fast is not
        testing the shipped behaviour."""
        cap = _FakeCap()
        free_it, holder = self._busy_io_lock()
        try:
            ok = self.bc._release_capture_guarded(cap, 0, "Right webcam",
                                                  timeout=0.2)
            # 1. the producer was NOT blocked, and nothing was released
            #    off-lock (that is the heap corruption, 0xc0000374).
            self.assertFalse(ok)
            self.assertFalse(cap.released,
                             "released while another thread held "
                             "_camera_io_lock - the exact overlapping teardown "
                             "the lock exists to prevent")

            # 2. the lock frees - the self-diagnostic's 30-minute webcam probe
            #    finishes, the slow open returns - and the release must happen.
            free_it.set()
            holder.join(timeout=5.0)
            self._wait_until(
                lambda: cap.released,
                msg="the camera was NEVER released once the lock freed - its "
                    "DirectShow filter graph holds the USB device for the rest "
                    "of the session, so every later reopen of that index has to "
                    "contend with a graph JARVIS itself still holds")
        finally:
            free_it.set()
            holder.join(timeout=5.0)

    def test_the_release_queue_is_drained_not_merely_appended_to(self):
        """The purest form of the defect: the old list was appended to in one
        place and drained in none. Once a handle has been released it must also
        leave the queue, or the "leak" is just relocated."""
        cap = _FakeCap()
        self.assertTrue(self.bc._defer_camera_release(cap, 0, "Right webcam"))
        self.assertIn(cap, self._queued_caps())
        self.assertEqual(1, self.bc._drain_camera_pending_releases(timeout=5.0))
        self.assertTrue(cap.released)
        self.assertNotIn(cap, self._queued_caps())

    def test_the_deferred_release_still_happens_inside_the_camera_io_lock(self):
        """The fix must not buy the leak back with a heap corruption: the
        deferred release is still performed INSIDE _camera_io_lock, exactly
        like every other release site in the file."""
        cap = _LockWitnessCap(self.bc._camera_io_lock)
        self.assertTrue(self.bc._defer_camera_release(cap, 0, "Right webcam"))
        self.assertEqual(1, self.bc._drain_camera_pending_releases(timeout=5.0))
        self.assertTrue(cap.released)
        self.assertIs(True, cap.lock_was_held_at_release,
                      "the deferred release ran with _camera_io_lock FREE - an "
                      "off-lock release is what heap-corrupts DirectShow")
        self.assertEqual([], self._queued_caps())

    def test_a_still_busy_lock_keeps_a_strong_reference_and_retries(self):
        """While the lock stays busy the handle must stay queued: dropping the
        last reference would let CPython run the destructor - which calls
        release() OFF-lock - behind our back."""
        cap = _FakeCap()
        free_it, holder = self._busy_io_lock()
        try:
            self.assertTrue(self.bc._defer_camera_release(cap, 0, "Right webcam"))
            self.assertEqual(0, self.bc._drain_camera_pending_releases(timeout=0.2))
            self.assertFalse(cap.released)
            self.assertIn(cap, self._queued_caps(),
                          "handle dropped from the queue while still "
                          "unreleased - nothing holds it now")
        finally:
            free_it.set()
            holder.join(timeout=5.0)

    # -- honest reporting ------------------------------------------------
    def test_the_timeout_message_does_not_name_a_cause_it_never_measured(self):
        """The old line said "the camera I/O lock is held by a wedged DirectShow
        call". All that was measured is that an acquire timed out - and the most
        likely holder is JARVIS's own 30-minute webcam probe."""
        cap = _FakeCap()
        free_it, holder = self._busy_io_lock()
        try:
            with self.assertLogs(level=logging.WARNING) as cm:
                self.bc._release_capture_guarded(cap, 0, "Right webcam",
                                                 timeout=0.2)
            said = "\n".join(r.getMessage() for r in cm.records)
            self.assertNotIn("wedged", said.lower(),
                             f"still diagnosing a wedge it never observed: {said!r}")
            self.assertIn("busy", said.lower())
            self.assertIn("not identified", said.lower(),
                          f"does not admit the holder is unknown: {said!r}")
            self.assertIn("deferred", said.lower(),
                          f"does not say the release is still owed: {said!r}")
        finally:
            free_it.set()
            holder.join(timeout=5.0)

    def test_the_escalation_message_hedges_because_it_is_still_an_inference(self):
        """After _CAMERA_RELEASE_ESCALATE_S a wedge is a defensible reading -
        but a busy lock is still all that was measured, so the line must say
        LIKELY, and must not claim the handle was abandoned."""
        cap = _FakeCap()
        free_it, holder = self._busy_io_lock()
        try:
            self.assertTrue(self.bc._defer_camera_release(cap, 0, "Right webcam"))
            with mock.patch.object(self.bc, "_CAMERA_RELEASE_ESCALATE_S", 0.0):
                with self.assertLogs(level=logging.WARNING) as cm:
                    self.bc._drain_camera_pending_releases(timeout=0.2)
            said = "\n".join(r.getMessage() for r in cm.records)
            self.assertIn("likely", said.lower(),
                          f"states a wedge as fact: {said!r}")
            self.assertIn("actually been measured", said.lower())
            self.assertIn("not abandoned", said.lower(),
                          f"does not say the handle is still owed a release: {said!r}")
        finally:
            free_it.set()
            holder.join(timeout=5.0)

    def test_the_escalation_threshold_sits_above_every_legitimate_holder(self):
        """The escalation line claims the wait is longer than any legitimate
        holder. That is only true if the threshold sits above the ~30 s a dead
        CAP_DSHOW index legitimately burns - the same measurement
        _CAMERA_LOOP_OPEN_TIMEOUT_S is sized on."""
        self.assertGreater(self.bc._CAMERA_RELEASE_ESCALATE_S, 30.0)
        self.assertLess(self.bc._CAMERA_RELEASE_RETRY_S,
                        self.bc._CAMERA_RELEASE_ESCALATE_S)


@requires_monolith
class ProducerBeatsWhileOpeningTests(MonolithGlobalsTestCase):
    """PRIORITY 1b: THE WATCHDOG MUST NOT CRY WOLF.

    ``_CAMERA_LOOP_OPEN_TIMEOUT_S`` (35 s) is deliberately LONGER than
    ``_FACE_TRACK_STALL_WARN_S`` (30 s), and on purpose: this file and
    ``_probe_camera_index`` both document that ``cv2.VideoCapture(CAP_DSHOW)``
    can LEGITIMATELY spend 20-30 s in its internal retry loop on an index with
    no device behind it, so a lower cap would abandon opens that were going to
    succeed. But the producer used to go completely SILENT for that whole
    open - the supervisor beat ``"starting"`` once and ``_open_capture_bounded``
    then sat in a single blind ``t.join(timeout=35.0)``.

    Measured against the real functions, 2026-09-05 (no camera opened; the
    "opener" was a pure-Python function parked on an Event)::

        _FACE_TRACK_STALL_WARN_S    = 30.0
        _CAMERA_LOOP_OPEN_TIMEOUT_S = 35.0
        heartbeat moved during the open? False  (at delta = 0.000 s)
        watchdog at t=30.1s -> fired? True

    So unplugging index 0 turned an ordinary boot - one that finishes a few
    seconds later with "Could not open Right webcam (index 0) - skipping" -
    into an ERROR reading "WEDGED CAMERA CALL, not a hardware failure" with a
    full thread stack dump, five seconds BEFORE the open it was complaining
    about was even allowed to give up. The same fired on every backoff-driven
    recovery reopen of an unplugged camera, re-shouted every 120 s. A watchdog
    that fires on a condition this same file calls legitimate is noise, and
    this one line is supposed to be the authoritative answer to "did the
    producer wedge?".

    The fix is a heartbeat INSIDE the wait, every
    ``_CAMERA_OPEN_BEAT_INTERVAL_S``; the open's own timeout - not the
    watchdog - is what reports a wedge on this path, with a better message.
    These tests fail if that beat is removed while the constants stay
    inverted. Nothing here opens a real camera.
    """

    CAM = {"index": 0, "label": "Right webcam"}

    def setUp(self):
        self.open_capture = _extract_open_capture(self.bc)

    def _armed_at(self, at):
        """Heartbeat exactly as the supervisor leaves it: one 'starting' beat
        and a clean stall state."""
        self.bc._face_track_heartbeat.update(
            {"at": at, "stage": "starting", "iters": 0,
             "tid": threading.get_ident()})
        self.bc._face_track_stall_state.update(
            {"warned_at": 0.0, "stalled": False, "since": 0.0})

    def _wait_released(self, cap, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not cap.released:
            time.sleep(0.02)

    # -- the regression ---------------------------------------------------
    def test_the_watchdog_stays_quiet_across_a_legitimate_dshow_open(self):
        """THE regression test, replayed at 1/100 scale with the REAL watchdog
        thread, the REAL producer ``_open_capture`` and the REAL bounded open.
        Only the three wall-clock constants are scaled, and they keep the same
        ORDER as shipped (stall threshold BELOW the open cap) - which is the
        whole point: that ordering is only survivable because the wait beats.

        Before the fix this raised, because the watchdog logged an ERROR
        saying the producer had not moved, at stage 'starting' after 0
        iterations, partway through an open the code had not yet given up on."""
        gate = threading.Event()
        cap = _FakeCap(block_event=gate)     # wedges inside .set(), like the live one
        wd_stop = threading.Event()
        wd = None
        self._armed_at(time.time())
        try:
            with mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
                 mock.patch.object(self.bc, "_FACE_TRACK_STALL_WARN_S", 0.30), \
                 mock.patch.object(self.bc, "_CAMERA_LOOP_OPEN_TIMEOUT_S", 0.80), \
                 mock.patch.object(self.bc, "_CAMERA_OPEN_BEAT_INTERVAL_S", 0.02), \
                 mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap):
                wd = threading.Thread(target=self.bc._face_track_watchdog,
                                      args=(wd_stop, 0.02), daemon=True,
                                      name="test-face-track-watchdog")
                wd.start()
                with self.assertNoLogs(level=logging.ERROR):
                    got = self.open_capture(self.CAM)
                wd_stop.set()
                wd.join(timeout=5.0)
            # The open WAS abandoned - the producer walked away on time. The
            # watchdog simply did not misreport that as a wedge of everything.
            self.assertIsNone(got)
            self.assertFalse(self.bc._face_track_stall_state["stalled"])
        finally:
            wd_stop.set()
            if wd is not None:
                wd.join(timeout=5.0)
            gate.set()
            self._wait_released(cap)

    def test_without_the_beat_that_same_replay_DOES_cry_wolf(self):
        """Negative control, so the test above cannot pass for the wrong
        reason. Same timeline, same constants, same wedge - but the open is
        entered with ``beat=None``, which is exactly what the producer did
        before the fix. The watchdog must fire, naming stage 'starting'."""
        gate = threading.Event()
        cap = _FakeCap(block_event=gate)
        wd_stop = threading.Event()
        wd = None
        self._armed_at(time.time())

        def _opener():
            with self.bc._camera_io_lock:
                c = self.bc.cv2.VideoCapture(0, self.bc.cv2.CAP_DSHOW)
                c.set(self.bc.cv2.CAP_PROP_FRAME_WIDTH, 1280)
                return c

        try:
            with mock.patch.object(self.bc, "_FACE_TRACK_STALL_WARN_S", 0.30), \
                 mock.patch.object(self.bc, "_CAMERA_OPEN_BEAT_INTERVAL_S", 0.02), \
                 mock.patch.object(self.bc.cv2, "VideoCapture", return_value=cap):
                wd = threading.Thread(target=self.bc._face_track_watchdog,
                                      args=(wd_stop, 0.02), daemon=True,
                                      name="test-face-track-watchdog")
                wd.start()
                with self.assertLogs(level=logging.ERROR) as cm:
                    self.bc._open_capture_bounded(0, _opener,
                                                  label="Right webcam",
                                                  timeout=0.80)
                wd_stop.set()
                wd.join(timeout=5.0)
            blob = "\n".join(r.getMessage() for r in cm.records)
            self.assertIn("STALLED", blob)
            self.assertIn("starting", blob)
        finally:
            wd_stop.set()
            if wd is not None:
                wd.join(timeout=5.0)
            gate.set()
            self._wait_released(cap)

    # -- what the beat actually says --------------------------------------
    def test_the_wait_beats_repeatedly_and_names_the_camera_it_waits_on(self):
        """A beat that just said "alive" would be a lie of omission: the point
        is that liveness / see_user can tell 'parked on THIS open, N s in'
        from 'looping normally'."""
        gate = threading.Event()
        beats: list[tuple[float, str]] = []

        def _parked():
            gate.wait(30.0)
            return None

        try:
            with mock.patch.object(self.bc, "_CAMERA_OPEN_BEAT_INTERVAL_S", 0.02):
                self.bc._open_capture_bounded(
                    0, _parked, label="Right webcam", timeout=0.30,
                    beat=lambda stage: beats.append((time.monotonic(), stage)))
        finally:
            gate.set()
        self.assertGreaterEqual(
            len(beats), 5,
            f"the producer went quiet inside the open again: {beats!r}")
        gaps = [b[0] - a[0] for a, b in zip(beats, beats[1:])]
        self.assertLess(max(gaps), 0.15, f"beat gap too wide: {gaps!r}")
        for _, stage in beats:
            self.assertIn("Right webcam", stage)
            self.assertIn("index 0", stage)

    def test_a_bounded_open_with_no_beat_never_touches_the_heartbeat(self):
        """The heartbeat records the CALLING thread's ident for the watchdog's
        stack dump, so only the producer may stamp it. A caller that passes no
        beat must leave it completely alone - otherwise a helper could forge
        liveness the producer does not have and point the dump at the wrong
        stack."""
        t0 = time.time() - 12.0
        self._armed_at(t0)
        self.bc._open_capture_bounded(7, lambda: None, label="side tile",
                                      timeout=0.2)
        self.assertEqual(self.bc._face_track_heartbeat["at"], t0)
        self.assertEqual(self.bc._face_track_heartbeat["stage"], "starting")

    def test_a_beat_that_raises_cannot_break_the_open(self):
        """A liveness call must never cost a camera."""
        cap = _FakeCap()

        def _boom(_stage):
            raise RuntimeError("beat exploded")

        got = self.bc._open_capture_bounded(0, lambda: cap, label="cam",
                                            timeout=2.0, beat=_boom)
        self.assertIs(got, cap)

    # -- healthy path is untouched ----------------------------------------
    def test_the_healthy_open_still_returns_promptly_with_exactly_one_open(self):
        """Healthy-path regression: the sliced join must not add DirectShow
        traffic or latency to an open that returns immediately."""
        cap = _FakeCap()
        self._armed_at(time.time())
        with mock.patch.object(self.bc, "KINECT_AS_CAMERA", False), \
             mock.patch.object(self.bc.cv2, "VideoCapture",
                               return_value=cap) as vc:
            t0 = time.monotonic()
            got = self.open_capture(self.CAM)
            elapsed = time.monotonic() - t0
        self.assertIs(got, cap)
        self.assertEqual(vc.call_count, 1, "healthy path must not add opens")
        self.assertFalse(cap.released)
        self.assertLess(elapsed, 1.0,
                        "the beat loop added latency to a healthy open")

    # -- the invariant itself ---------------------------------------------
    def test_the_shipped_constants_leave_no_silent_stretch(self):
        """Either the producer beats through an open, or the stall threshold
        sits above the open cap. Today it is the former - assert it, because
        the constants alone are INVERTED and that is only safe with the beat."""
        bc = self.bc
        self.assertLess(bc._FACE_TRACK_STALL_WARN_S,
                        bc._CAMERA_LOOP_OPEN_TIMEOUT_S,
                        "constants un-inverted - if that was deliberate this "
                        "test is stale, but do NOT drop the beat as well")
        self.assertLess(
            bc._CAMERA_OPEN_BEAT_INTERVAL_S * 3.0, bc._FACE_TRACK_STALL_WARN_S,
            "the open-wait beat is no longer frequent enough to keep the "
            "watchdog quiet through a legitimate 20-30 s CAP_DSHOW block")

    def test_the_producers_opener_really_is_the_thing_that_beats(self):
        """Asserted on the CODE OBJECT of the shipped closure, so re-wording a
        message cannot defeat it and a paraphrase in a test cannot satisfy it."""
        names = _open_capture_code(self.bc).co_names
        self.assertIn("_face_track_beat", names,
                      "_open_capture no longer beats - the initial "
                      "`for cam in CAMERAS` sweep has no other beat between "
                      "cameras, so a boot goes silent for a full open window")
        self.assertIn("_open_capture_bounded", names)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
