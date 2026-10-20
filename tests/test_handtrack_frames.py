"""core/handtrack_frames.py: the JARVIS -> tracker frame hand-off. In-process
round trip of the shared-memory header and pixels, the heartbeat gate (nothing
is copied while the tracker is away), the rate cap, torn-frame rejection and
the never-raises contract of the camera-loop hook. No camera, no tracker."""
import os
import sys
import tempfile
import time
import unittest
import uuid
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import numpy as np
    import cv2
except Exception:                                    # pragma: no cover
    np = cv2 = None

from core import handtrack_frames as hf  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@unittest.skipIf(np is None or cv2 is None, "needs numpy + cv2")
class WriterTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.alive = os.path.join(self.dir.name, ".handtrack_alive")
        self.clock = Clock()
        self.wall = [time.time()]
        self.w = hf.HandTrackFrames(clock=self.clock, wall=lambda: self.wall[0],
                                    alive_file=self.alive,
                                    shm_name="jv_ht_test_" + uuid.uuid4().hex[:8])
        self.reader = None

    def tearDown(self):
        if self.reader is not None:
            self.reader.close()
        self.w.close()
        self.dir.cleanup()

    def heartbeat(self):
        with open(self.alive, "w") as f:
            f.write("{}")
        os.utime(self.alive, (self.wall[0], self.wall[0]))

    def attach(self):
        from multiprocessing import shared_memory
        self.reader = shared_memory.SharedMemory(name=self.w._shm_name)
        return self.reader

    @staticmethod
    def frame(value=50, w=1280, h=720):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, :, 0] = value
        f[:, :, 1] = 200
        f[10:30, 10:50, 2] = 255
        return f

    def test_header_round_trip(self):
        head = hf.pack_header(7, 1234567, 640, 480, 3)
        self.assertEqual(len(head), hf.HEADER_SIZE)
        d = hf.unpack_header(head)
        self.assertEqual((d["seq"], d["timestamp_ms"], d["width"], d["height"],
                          d["channels"], d["seq_begin"]), (7, 1234567, 640, 480, 3, 7))
        self.assertIsNone(hf.unpack_header(b"NOTMAGIC" + bytes(56)))
        self.assertIsNone(hf.unpack_header(b"short"))

    def test_nothing_is_copied_without_a_tracker(self):
        self.assertFalse(self.w.active())
        self.assertFalse(self.w.publish(self.frame()))
        self.assertIsNone(self.w._shm)                    # no segment even created
        hf.publish_frame({"primary": True}, self.frame(), setting="primary")  # no raise

    def test_publish_round_trip_keeps_the_aspect(self):
        # 1280x720 (16:9) -> 640x360, NOT stretched to 640x480: a stretched
        # frame squeezes every hand ~25 % narrower and skews finger angles.
        self.heartbeat()
        src = self.frame(77)
        self.assertTrue(self.w.publish(src))
        shm = self.attach()
        head = hf.unpack_header(shm.buf)
        self.assertEqual((head["seq"], head["width"], head["height"], head["channels"]),
                         (1, 640, 360, 3))
        self.assertEqual(head["seq_begin"], head["seq"])
        got = hf.read_frame(shm.buf, 0)
        self.assertIsNotNone(got)
        seq, ts, frame = got
        self.assertEqual(seq, 1)
        self.assertEqual(frame.shape, (360, 640, 3))
        want = cv2.resize(src, (640, 360), interpolation=cv2.INTER_AREA)
        self.assertTrue((frame == want).all())
        self.assertAlmostEqual(ts / 1000.0, self.wall[0], delta=1.0)
        self.assertIsNone(hf.read_frame(shm.buf, 1))      # no new frame yet

    def test_seq_increments_per_frame_and_rate_is_capped(self):
        self.heartbeat()
        self.assertTrue(self.w.publish(self.frame(1)))
        self.assertFalse(self.w.publish(self.frame(2)))   # same instant: capped
        self.clock.t += 0.05
        self.assertTrue(self.w.publish(self.frame(3)))
        self.assertEqual(self.w.status()["seq"], 2)
        shm = self.attach()
        seq, _ts, frame = hf.read_frame(shm.buf, 1)
        self.assertEqual(seq, 2)
        self.assertEqual(int(frame[100, 100, 0]), 3)

    def test_native_size_frame_is_copied_as_is(self):
        self.heartbeat()
        src = self.frame(9, 640, 480)
        self.assertTrue(self.w.publish(src))
        shm = self.attach()
        self.assertTrue((hf.read_frame(shm.buf, 0)[2] == src).all())

    def test_fit_size(self):
        self.assertEqual(hf.fit_size(1280, 720), (640, 360))
        self.assertEqual(hf.fit_size(1920, 1080), (640, 360))
        self.assertEqual(hf.fit_size(640, 480), (640, 480))
        self.assertEqual(hf.fit_size(1024, 1024), (480, 480))
        self.assertEqual(hf.fit_size(320, 240), (320, 240))      # never enlarged
        self.assertEqual(hf.fit_size(0, 0), (640, 480))

    def test_size_can_change_between_frames(self):
        self.heartbeat()
        self.assertTrue(self.w.publish(self.frame(5)))            # 16:9
        self.clock.t += 0.05
        self.assertTrue(self.w.publish(self.frame(6, 640, 480)))  # 4:3
        shm = self.attach()
        seq, _ts, frame = hf.read_frame(shm.buf, 1)
        self.assertEqual((seq, frame.shape), (2, (480, 640, 3)))
        self.assertEqual(int(frame[200, 200, 0]), 6)

    def test_oversized_header_is_rejected(self):
        self.heartbeat()
        self.w.publish(self.frame())
        shm = self.attach()
        import struct
        struct.pack_into("<II", shm.buf, hf.OFF_WH, 800, 600)
        self.assertIsNone(hf.read_frame(shm.buf, 0))

    def test_torn_frame_is_rejected(self):
        self.heartbeat()
        self.w.publish(self.frame())
        shm = self.attach()
        import struct
        struct.pack_into("<Q", shm.buf, hf.OFF_SEQ_BEGIN, 2)   # writer mid-copy
        self.assertIsNone(hf.read_frame(shm.buf, 0))
        struct.pack_into("<Q", shm.buf, hf.OFF_SEQ_BEGIN, 1)
        self.assertIsNotNone(hf.read_frame(shm.buf, 0))

    def test_seq_is_published_after_the_pixels(self):
        # While the copy runs the header still says "seq_begin = new, seq = old":
        # observe the buffer at the moment the pixels are written.
        self.heartbeat()
        self.w.publish(self.frame(1))
        self.clock.t += 0.1
        seen = {}
        real_view = self.w._view

        class Spy:
            def __setitem__(_s, key, value):
                head = hf.unpack_header(self.w._shm.buf)
                seen["seq"], seen["begin"] = head["seq"], head["seq_begin"]
                real_view[key] = value

        self.w._view = Spy()
        self.assertTrue(self.w.publish(self.frame(2)))
        self.assertEqual((seen["seq"], seen["begin"]), (1, 2))
        self.assertEqual(hf.unpack_header(self.w._shm.buf)["seq"], 2)
        self.w._view = real_view

    def test_stale_heartbeat_stops_publishing_and_idle_segment_closes(self):
        self.heartbeat()
        self.assertTrue(self.w.publish(self.frame()))
        self.wall[0] += 10.0                               # tracker stopped touching
        self.clock.t += 1.0
        self.assertFalse(self.w.active())
        self.assertFalse(self.w.publish(self.frame()))
        self.assertIsNotNone(self.w._shm)                  # not closed yet
        self.clock.t += hf.IDLE_CLOSE_S + 1
        self.assertFalse(self.w.publish(self.frame()))
        self.assertIsNone(self.w._shm)                     # now closed

    def test_alive_check_is_throttled(self):
        self.heartbeat()
        with mock.patch.object(hf, "file_age_s", wraps=hf.file_age_s) as age:
            for _ in range(50):
                self.w.active()
            self.assertEqual(age.call_count, 1)
            self.clock.t += 0.6
            self.w.active()
            self.assertEqual(age.call_count, 2)

    def test_failure_never_reaches_the_camera_loop(self):
        self.heartbeat()
        self.assertFalse(self.w.publish("not a frame"))
        self.assertFalse(self.w.publish(np.zeros((10, 10), np.uint8)))   # 2-D
        boom = mock.Mock(side_effect=RuntimeError("shm exploded"))
        w2 = hf.HandTrackFrames(clock=self.clock, wall=lambda: self.wall[0],
                                alive_file=self.alive, shm_factory=boom)
        self.assertFalse(w2.publish(self.frame()))
        self.assertIn("RuntimeError", w2.status()["last_error"])
        with mock.patch.object(hf.WRITER, "active", side_effect=OSError("x")):
            hf.publish_frame({"primary": True}, self.frame())          # swallowed


class CameraMatchTests(unittest.TestCase):
    def test_settings(self):
        primary = {"index": 0, "label": "Webcam", "primary": True}
        other = {"index": 2, "label": "Right webcam", "primary": False}
        self.assertTrue(hf.camera_matches("primary", primary))
        self.assertTrue(hf.camera_matches("", primary))
        self.assertFalse(hf.camera_matches("primary", other))
        self.assertTrue(hf.camera_matches("right", other, "right"))
        self.assertFalse(hf.camera_matches("right", primary, "left"))
        self.assertTrue(hf.camera_matches("2", other))
        self.assertFalse(hf.camera_matches("kinect", primary))      # not published
        self.assertFalse(hf.camera_matches(None, other))

    def test_publish_frame_picks_the_configured_camera(self):
        with mock.patch.object(hf.WRITER, "active", return_value=True), \
                mock.patch.object(hf.WRITER, "publish") as pub:
            hf.publish_frame({"index": 0, "primary": True}, "F", setting="primary")
            self.assertEqual(pub.call_count, 1)
            hf.publish_frame({"index": 1, "primary": False, "label": "Right webcam"},
                             "F", setting="primary")
            self.assertEqual(pub.call_count, 1)
            hf.publish_frame({"index": 1, "primary": False, "label": "Right webcam"},
                             "F", setting="right")
            self.assertEqual(pub.call_count, 2)

    def test_no_work_without_a_tracker(self):
        with mock.patch.object(hf.WRITER, "active", return_value=False), \
                mock.patch.object(hf, "camera_matches") as cm:
            hf.publish_frame({"primary": True}, "F")
            cm.assert_not_called()


class FakeCap:
    """read() -> (True, frame) and advances the fake clock one camera frame."""
    def __init__(self, clock, frames, dt=1 / 30, fail_after=None, boom=False):
        self.clock, self.frames, self.dt = clock, frames, dt
        self.fail_after, self.boom = fail_after, boom
        self.reads = 0

    def read(self):
        if self.boom:
            raise RuntimeError("driver hiccup")
        self.reads += 1
        self.clock.t += self.dt
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, self.frames[min(self.reads - 1, len(self.frames) - 1)]


@unittest.skipIf(np is None, "needs numpy")
class BurstTests(unittest.TestCase):
    """While the tracker is alive the camera loop keeps reading the tracker's
    camera ~0.15 s per pass (live v2.0.194: only 5.8 fps reached it)."""
    CAM = {"index": 0, "primary": True}

    def setUp(self):
        self.clock = Clock()
        self.frames = [np.full((4, 4, 3), i, np.uint8) for i in range(1, 40)]
        p = mock.patch.object(hf.WRITER, "publish")
        self.pub = p.start()
        self.addCleanup(p.stop)

    def _active(self, on=True):
        p = mock.patch.object(hf.WRITER, "active", return_value=on)
        p.start()
        self.addCleanup(p.stop)

    def test_reads_the_camera_for_the_budget_and_publishes_each_frame(self):
        self._active()
        cap = FakeCap(self.clock, self.frames)
        beats = []
        first = np.zeros((4, 4, 3), np.uint8)
        newest, n = hf.burst(cap, self.CAM, first, beat=lambda: beats.append(1),
                             setting="primary", clock=self.clock)
        self.assertEqual(n, cap.reads)
        self.assertGreaterEqual(n, 4)                    # 0.15 s of a 30 fps camera
        self.assertLessEqual(n, 6)
        self.assertEqual(self.pub.call_count, n)
        self.assertIs(newest, self.frames[n - 1])        # detection gets the newest
        self.assertEqual(len(beats), n)                  # the loop watchdog stays fed

    def test_no_tracker_means_no_extra_reads(self):
        self._active(False)
        cap = FakeCap(self.clock, self.frames)
        first = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(hf.burst(cap, self.CAM, first, setting="primary",
                                  clock=self.clock), (first, 0))
        self.assertEqual(cap.reads, 0)

    def test_other_cameras_are_left_alone(self):
        self._active()
        cap = FakeCap(self.clock, self.frames)
        other = {"index": 2, "primary": False, "label": "Left webcam"}
        self.assertEqual(hf.burst(cap, other, "F", setting="primary",
                                  clock=self.clock)[1], 0)
        self.assertEqual(cap.reads, 0)

    def test_a_failed_read_ends_the_burst(self):
        self._active()
        cap = FakeCap(self.clock, self.frames, fail_after=2)
        newest, n = hf.burst(cap, self.CAM, "F", setting="primary", clock=self.clock)
        self.assertEqual((n, cap.reads), (2, 3))
        self.assertIs(newest, self.frames[1])

    def test_never_raises(self):
        self._active()
        cap = FakeCap(self.clock, self.frames, boom=True)
        self.assertEqual(hf.burst(cap, self.CAM, "F", setting="primary",
                                  clock=self.clock), ("F", 0))
        self.assertEqual(hf.burst(None, self.CAM, "F", setting="primary"), ("F", 0))


class PacedCap:
    """A capture whose read() blocks for one frame interval in REAL time, like
    MSMF with BUFFERSIZE=1. It records when each read finished and how many
    threads were inside read() at once (must never exceed 1)."""

    def __init__(self, fps=30.0, fail_after=None, read_s=None):
        import threading as _th
        self.dt = 1.0 / fps if read_s is None else read_s
        self.fail_after = fail_after
        self.reads = 0
        self.read_at = []
        self.inside = 0
        self.max_inside = 0
        self.released = False
        self.read_after_release = False
        self.readers = set()
        self._lock = _th.Lock()
        self._th = _th

    def read(self):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            self.readers.add(self._th.get_ident())
            if self.released:
                self.read_after_release = True
        try:
            time.sleep(self.dt)
            with self._lock:
                self.reads += 1
                n = self.reads
            self.read_at.append(time.monotonic())
            if self.fail_after is not None and n > self.fail_after:
                return False, None
            return True, np.full((4, 4, 3), n % 250 + 1, np.uint8)
        finally:
            with self._lock:
                self.inside -= 1

    def release(self):
        with self._lock:
            if self.inside:
                raise AssertionError("released while a read was in flight")
            self.released = True


@unittest.skipIf(np is None, "needs numpy")
class FramePumpTests(unittest.TestCase):
    """STEADY FRAMES (2026-10-20): while the tracker is alive its camera is read
    by ONE pump thread at the camera's own rate, every frame published evenly,
    while the face loop takes the newest frame instead of reading. With no
    tracker nothing changes. The pump never opens or releases a camera."""
    CAM = {"index": 0, "primary": True}

    def setUp(self):
        self.pub_at = []
        self.writer = mock.Mock()
        self.writer.active.return_value = True
        self.writer.publish.side_effect = lambda f: self.pub_at.append(time.monotonic())
        self.clock = Clock()
        self.pump = hf.FramePump(writer=self.writer, clock=self.clock,
                                 log=lambda s: None)
        self.addCleanup(self.pump.detach)

    def serve(self, cap, cam=None, **kw):
        return self.pump.serve(cap, cam or self.CAM, setting="primary", **kw)

    def test_no_tracker_means_none_and_no_thread(self):
        self.writer.active.return_value = False
        cap = PacedCap()
        for _ in range(5):
            self.assertIsNone(self.serve(cap))
        time.sleep(0.05)
        self.assertEqual((cap.reads, self.pump.starts), (0, 0))
        self.assertFalse(self.pump.bound())
        self.writer.publish.assert_not_called()

    def test_no_tracker_costs_one_active_check(self):
        self.writer.active.return_value = False
        with mock.patch.object(hf, "is_hand_camera") as ihc:
            self.assertIsNone(self.serve(PacedCap()))
            ihc.assert_not_called()
        self.assertEqual(self.writer.active.call_count, 1)

    def test_frames_reach_the_tracker_evenly_while_the_loop_is_slow(self):
        # The face loop takes a frame only every ~0.17 s (face pass, previews,
        # other cameras): the tracker must still get every camera frame.
        cap = PacedCap(fps=30.0)
        t_end = time.monotonic() + 0.9
        got = []
        while time.monotonic() < t_end:
            r = self.serve(cap)
            self.assertIsNotNone(r)
            got.append(r)
            time.sleep(0.17)
        self.pump.detach(cap)
        self.assertTrue(all(ok for ok, _f in got))
        self.assertGreaterEqual(len(self.pub_at), 18)        # ~27 at 30 fps
        gaps = sorted(b - a for a, b in zip(self.pub_at, self.pub_at[1:]))
        # Even: the typical gap is one camera frame, and even the worst one is
        # shorter than the loop's 0.17 s pass (the old burst left exactly that
        # hole). Generous bounds so a loaded test machine cannot flake it.
        self.assertLess(gaps[len(gaps) // 2], 0.05, gaps)
        self.assertLess(gaps[-1], 0.15, gaps)
        self.assertEqual(cap.max_inside, 1)                  # one reader, ever
        self.assertEqual(len(cap.readers), 1)                # ...and not the loop
        self.assertEqual(self.pump.starts, 1)

    def test_the_loop_gets_the_newest_frame_and_never_the_same_one_twice(self):
        cap = PacedCap(fps=50.0)
        a = self.serve(cap)
        time.sleep(0.1)
        b = self.serve(cap)
        self.assertTrue(a[0] and b[0])
        self.assertIsNot(a[1], b[1])
        self.assertGreaterEqual(int(b[1][0, 0, 0]) - int(a[1][0, 0, 0]), 2)

    def test_detach_waits_for_the_read_in_flight_before_a_release(self):
        cap = PacedCap(read_s=0.25)
        self.serve(cap)                         # one frame, the reader is in read()
        time.sleep(0.05)
        self.assertEqual(cap.inside, 1)
        self.assertTrue(self.pump.detach(cap))
        self.assertEqual(cap.inside, 0)
        cap.release()                           # raises if a read is in flight
        time.sleep(0.3)
        self.assertFalse(cap.read_after_release)
        self.assertFalse(self.pump.bound())

    def test_detach_of_another_handle_is_a_no_op(self):
        cap = PacedCap()
        self.serve(cap)
        self.assertFalse(self.pump.detach(PacedCap()))
        self.assertTrue(self.pump.bound())

    def test_a_failed_read_is_reported_once_then_the_loop_reads_itself(self):
        cap = PacedCap(fps=60.0, fail_after=2)
        results = [self.serve(cap) for _ in range(4)]
        self.assertEqual([r[0] for r in results[:2]], [True, True])
        self.assertEqual(results[2], (False, None))     # the loop counts it
        self.assertIsNone(results[3])                   # re-arm back-off
        self.assertFalse(self.pump.bound())
        self.clock.t += hf.PUMP_REARM_S + 0.1
        cap.fail_after = None
        self.assertTrue(self.serve(cap)[0])             # re-armed
        self.assertEqual(self.pump.starts, 2)

    def test_another_camera_is_never_served_and_its_handle_freed(self):
        cap = PacedCap()
        self.assertTrue(self.serve(cap)[0])
        other = {"index": 2, "primary": False, "label": "Left webcam"}
        self.assertIsNone(self.serve(PacedCap(), other))
        self.assertTrue(self.pump.bound())               # still on the primary
        # HAND_TRACK_CAMERA moved off this camera: the pump lets go of it
        # before the loop reads it again.
        self.assertIsNone(self.pump.serve(cap, self.CAM, setting="2"))
        self.assertFalse(self.pump.bound())

    def test_tracker_leaving_stops_the_reader(self):
        cap = PacedCap()
        self.assertTrue(self.serve(cap)[0])
        self.writer.active.return_value = False
        self.assertIsNone(self.serve(cap))
        self.assertFalse(self.pump.bound())
        n = cap.reads
        time.sleep(0.1)
        self.assertEqual(cap.reads, n)

    def test_a_reopened_handle_moves_the_reader(self):
        old, new = PacedCap(), PacedCap()
        self.assertTrue(self.serve(old)[0])
        self.assertTrue(self.serve(new)[0])
        self.assertEqual(old.inside, 0)
        n = old.reads
        time.sleep(0.1)
        self.assertEqual(old.reads, n)
        self.assertEqual(self.pump.starts, 2)

    def test_kinect_slots_are_not_pumped(self):
        cap = PacedCap()
        self.assertIsNone(self.serve(cap, pumpable=False))
        self.assertEqual(self.pump.starts, 0)

    def test_accept_and_on_frame_hooks(self):
        seen = []
        self.pump.accept = lambda f: int(f[0, 0, 0]) % 2 == 0
        self.pump.on_frame = lambda cam, f: seen.append(cam["index"])
        cap = PacedCap(fps=60.0)
        for _ in range(6):
            self.serve(cap)
        self.pump.detach(cap)
        self.assertGreaterEqual(len(seen), 6)
        self.assertLess(self.writer.publish.call_count, len(seen))
        self.assertGreater(self.writer.publish.call_count, 0)

    def test_never_raises(self):
        class Boom:
            def read(self):
                raise RuntimeError("driver hiccup")
        r = self.serve(Boom())
        self.assertEqual(r, (False, None))
        self.assertIsNone(self.serve(Boom()))           # back-off
        self.writer.active.side_effect = RuntimeError("x")
        self.assertIsNone(self.serve(PacedCap()))

    def test_the_pump_never_opens_or_releases_a_camera(self):
        import inspect
        src = inspect.getsource(hf.FramePump)
        for bad in ("VideoCapture", ".release(", "_camera_open", "open_capture"):
            self.assertNotIn(bad, src)


class MonolithHookTests(unittest.TestCase):
    def test_the_burst_follows_the_publish_and_skips_the_nap(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"), encoding="utf-8") as f:
            src = f.read()
        i = src.index("_handtrack_frames.publish_frame(cam, frame)")
        j = src.index("_handtrack_frames.burst(", i)
        self.assertLess(j - i, 800)
        self.assertEqual(src.count("_handtrack_frames.burst("), 1)
        self.assertIn("if not _ht_burst_ran:", src)
        self.assertEqual(src.count("_ht_burst_ran = False"), 1)

    def test_one_guarded_call_in_the_camera_loop(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"), encoding="utf-8") as f:
            src = f.read()
        call = "_handtrack_frames.publish_frame(cam, frame)"
        self.assertEqual(src.count(call), 1)
        i = src.index(call)
        # (2026-10-20) a frame the frame pump served was already handed over
        self.assertIn("if not _black and not _ht_pumped:", src[i - 80:i])
        self.assertEqual(src.count("from core import handtrack_frames as _handtrack_frames"), 1)


if __name__ == "__main__":
    unittest.main()
