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


class MonolithHookTests(unittest.TestCase):
    def test_one_guarded_call_in_the_camera_loop(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"), encoding="utf-8") as f:
            src = f.read()
        call = "_handtrack_frames.publish_frame(cam, frame)"
        self.assertEqual(src.count(call), 1)
        i = src.index(call)
        self.assertIn("if not _black:", src[i - 80:i])
        self.assertEqual(src.count("from core import handtrack_frames as _handtrack_frames"), 1)


if __name__ == "__main__":
    unittest.main()
