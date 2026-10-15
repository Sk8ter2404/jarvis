"""core/handtrack_depth.py - the Kinect colour + depth hand-off to the hand tracker
(Phase 2, 2026-10-15). FAKES ONLY: a fake Kinect bridge, a fake colour writer, an
in-memory fake shared-memory segment and a fake clock. Nothing here opens a
Kinect or any camera, and no frame in these tests is a real picture.

Covers: publish only while the tracker is alive AND the Kinect is selected, the
rate cap, "never raises into the caller" (and never opens the sensor), the depth
segment header + seqlock, the registration math on synthetic data (nominal FOV
mapping, SDK-mapper fit, nearest-wins, range gating), the binary wire message and
the palm-depth sampler. Paths are built with os.path so the tests run on Linux CI.
"""
import os
import struct
import sys
import types
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import numpy as np
except Exception:                                    # pragma: no cover
    np = None

from core import handtrack_depth as hd  # noqa: E402


class FakeShm:
    """Just enough of multiprocessing.shared_memory.SharedMemory."""

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.buf = memoryview(bytearray(size))
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeFrames:
    """The colour-segment writer (core.handtrack_frames.WRITER) stand-in."""

    def __init__(self, active=True):
        self._active = active
        self.published = []
        self.seq = 0
        self.raise_on_publish = False
        self.accept = True

    def active(self):
        return self._active

    def publish(self, frame):
        if self.raise_on_publish:
            raise RuntimeError("boom")
        if not self.accept:
            return False
        self.seq += 1
        self.published.append(frame)
        return True

    def status(self):
        return {"seq": self.seq}


def color_frame(value=7):
    return np.full((1080, 1920, 3), value, np.uint8)


def depth_frame(mm=1500):
    return np.full((424, 512), mm, np.uint16)


class FakeBridge:
    """The few audio.kinect_bridge accessors the feed may use. Every call is
    counted so tests can prove the feed did NOT touch the sensor."""

    def __init__(self):
        self.open = True
        self.enabled = True
        self.color_clock = 1.0
        self.depth_clock = 1.0
        self.color = color_frame()
        self.depth = depth_frame()
        self.calls = []
        self.color_raises = False
        self.color_mapper = None
        self.depth_mapper = None
        self.depth_new = True

    def get_stream_health(self):
        self.calls.append("health")
        return {"open": self.open, "enabled": self.enabled}

    def frame_clock(self, stream):
        self.calls.append("clock")
        return self.color_clock if stream == "color" else self.depth_clock

    def get_color_bgr(self, require_new=True):
        self.calls.append(("color", require_new))
        if self.color_raises:
            raise RuntimeError("pump exploded")
        return self.color

    def get_depth(self, require_new=False):
        self.calls.append(("depth", require_new))
        if require_new and not self.depth_new:
            return None
        return self.depth

    def get_color_space_mapper(self):
        self.calls.append("cmapper")
        return self.color_mapper

    def get_depth_space_mapper(self):
        self.calls.append("dmapper")
        return self.depth_mapper


def make_feed(selected=True, depth=False, active=True, session=None, bridge=None):
    clock = Clock()
    br = bridge or FakeBridge()
    fr = FakeFrames(active=active)
    dw = hd.DepthWriter(shm_factory=FakeShm, shm_name="jv_depth_test")
    feed = hd.KinectFeed(
        bridge=br, frames=fr, depth_writer=dw, clock=clock,
        setting=lambda: "kinect" if selected else "primary",
        depth_on=lambda: depth, session=session)
    return feed, br, fr, dw, clock


# ── the segment ───────────────────────────────────────────────────────────
@unittest.skipIf(np is None, "needs numpy")
class SegmentTests(unittest.TestCase):
    def test_header_layout_and_round_trip(self):
        head = hd.pack_header(7, 1234567, 160, 90, hd.METHOD_SDK_FIT, 42)
        self.assertEqual(len(head), hd.HEADER_SIZE)
        self.assertEqual(head[:8], b"JVHD0001")
        # the documented offsets
        self.assertEqual(struct.unpack_from("<Q", head, 8)[0], 7)          # seq
        self.assertEqual(struct.unpack_from("<Q", head, 16)[0], 1234567)   # timestamp_ms
        self.assertEqual(struct.unpack_from("<II", head, 24), (160, 90))   # width, height
        self.assertEqual(struct.unpack_from("<I", head, 32)[0], 2)         # bytes/sample
        self.assertEqual(struct.unpack_from("<I", head, 36)[0], 1)         # method
        self.assertEqual(struct.unpack_from("<Q", head, 40)[0], 7)         # seq_begin
        self.assertEqual(struct.unpack_from("<Q", head, 48)[0], 42)        # color_seq
        d = hd.unpack_header(head)
        self.assertEqual((d["seq"], d["width"], d["height"], d["method"],
                          d["color_seq"], d["seq_begin"]), (7, 160, 90, 1, 42, 7))
        self.assertIsNone(hd.unpack_header(b"NOTMAGIC" + bytes(56)))
        self.assertIsNone(hd.unpack_header(b"short"))

    def test_publish_and_read_round_trip(self):
        dw = hd.DepthWriter(shm_factory=FakeShm)
        plane = (np.arange(90 * 160, dtype=np.uint32) % 4000 + 500).astype(np.uint16).reshape(90, 160)
        self.assertTrue(dw.publish(plane, color_seq=11, method=hd.METHOD_NOMINAL))
        buf = dw._shm.buf
        got = hd.read_depth(buf, 0)
        self.assertIsNotNone(got)
        seq, color_seq, ts, out = got
        self.assertEqual((seq, color_seq), (1, 11))
        self.assertTrue((out == plane).all())
        self.assertGreater(ts, 1_600_000_000_000)
        self.assertIsNone(hd.read_depth(buf, 1))              # nothing new
        self.assertTrue(dw.publish(plane + 1, 12, hd.METHOD_SDK_FIT))
        self.assertEqual(hd.read_depth(buf, 1)[:2], (2, 12))
        self.assertEqual(hd.unpack_header(buf)["method"], hd.METHOD_SDK_FIT)

    def test_a_torn_or_half_written_plane_is_skipped(self):
        dw = hd.DepthWriter(shm_factory=FakeShm)
        dw.publish(np.full((90, 160), 900, np.uint16), 1, 0)
        buf = dw._shm.buf
        # the writer announced the next frame (seq_begin = seq + 1) and is mid-copy
        struct.pack_into("<Q", buf, hd.OFF_SEQ_BEGIN, 2)
        self.assertIsNone(hd.read_depth(buf, 0))
        struct.pack_into("<Q", buf, hd.OFF_SEQ_BEGIN, 1)      # finished
        self.assertIsNotNone(hd.read_depth(buf, 0))

    def test_the_writer_never_raises_and_refuses_bad_planes(self):
        dw = hd.DepthWriter(shm_factory=FakeShm)
        self.assertFalse(dw.publish(np.zeros((90, 160), np.uint8), 1, 0))      # wrong dtype
        self.assertFalse(dw.publish(np.zeros((400, 400), np.uint16), 1, 0))    # too big
        self.assertFalse(dw.publish(None, 1, 0))                               # garbage

        def boom(name, size):
            raise OSError("no shm here")
        dw2 = hd.DepthWriter(shm_factory=boom)
        self.assertFalse(dw2.publish(np.zeros((90, 160), np.uint16), 1, 0))
        self.assertIn("no shm here", dw2.last_error)

    def test_close_releases_the_segment(self):
        dw = hd.DepthWriter(shm_factory=FakeShm)
        dw.publish(np.zeros((90, 160), np.uint16), 1, 0)
        shm = dw._shm
        dw.close()
        self.assertTrue(shm.closed and shm.unlinked)
        self.assertFalse(dw.is_open)
        dw.close()                                            # idempotent


# ── the wire message ──────────────────────────────────────────────────────
@unittest.skipIf(np is None, "needs numpy")
class WireTests(unittest.TestCase):
    def test_layout_matches_the_documented_format(self):
        plane = np.arange(6, dtype=np.uint16).reshape(2, 3) + 1000
        msg = hd.pack_wire(plane, color_seq=0x0102030405060708, mirrored=True)
        self.assertEqual(len(msg), 24 + 2 * 3 * 2)
        self.assertEqual(msg[:4], b"JVDP")
        self.assertEqual(msg[4], 1)                           # version
        self.assertEqual(msg[5] & 1, 1)                       # mirrored flag
        self.assertEqual(struct.unpack_from("<HH", msg, 6), (3, 2))     # w, h
        self.assertEqual(struct.unpack_from("<H", msg, 10)[0], 0)       # reserved
        self.assertEqual(struct.unpack_from("<Q", msg, 12)[0], 0x0102030405060708)
        self.assertEqual(struct.unpack_from("<HH", msg, 20), (500, 4500))
        self.assertEqual(struct.unpack_from("<6H", msg, 24), (1000, 1001, 1002, 1003, 1004, 1005))
        self.assertEqual(hd.pack_wire(plane, 1, mirrored=False)[5] & 1, 0)

    def test_round_trip_and_rejects(self):
        plane = (np.arange(90 * 160) % 4000).astype(np.uint16).reshape(90, 160)
        msg = hd.pack_wire(plane, 99)
        out = hd.unpack_wire(msg)
        self.assertEqual((out["width"], out["height"], out["seq"]), (160, 90, 99))
        self.assertTrue((out["plane"] == plane).all())
        self.assertIsNone(hd.unpack_wire(b"\xff\xd8\xff\xe0" + bytes(100)))   # a JPEG
        self.assertIsNone(hd.unpack_wire(msg[:-2]))                           # truncated
        self.assertIsNone(hd.unpack_wire(b"JVDP"))
        bad = bytearray(msg)
        bad[4] = 9
        self.assertIsNone(hd.unpack_wire(bytes(bad)))                         # version

    def test_a_depth_message_is_never_mistaken_for_a_jpeg(self):
        self.assertNotEqual(hd.WIRE_MAGIC[:2], b"\xff\xd8")


# ── registration ──────────────────────────────────────────────────────────
@unittest.skipIf(np is None, "needs numpy")
class RegistrationTests(unittest.TestCase):
    def test_nominal_mapping_follows_the_fields_of_view(self):
        import math
        c = hd.nominal_calibration()
        self.assertEqual(c.method, hd.METHOD_NOMINAL)
        self.assertEqual((c.bx, c.by), (0.0, 0.0))               # the baseline is NOT modelled
        # the optical centre maps to the colour centre
        self.assertAlmostEqual(c.kx * 256 + c.ox, 960.0, places=6)
        self.assertAlmostEqual(c.ky * 212 + c.oy, 540.0, places=6)
        # a ray 20 degrees right of the axis: pinhole tan() on both sensors
        fdx = 256 / math.tan(math.radians(70.6 / 2))
        fcx = 960 / math.tan(math.radians(84.1 / 2))
        t = math.tan(math.radians(20))
        u = 256 + fdx * t
        self.assertAlmostEqual(c.kx * u + c.ox, 960 + fcx * t, places=3)
        # the depth FOV edge (35.3 deg) lies INSIDE the colour frame horizontally
        # (42 deg) and outside it vertically (30 deg vs 26.9 deg)
        self.assertGreater(c.kx * 0 + c.ox, 0)
        self.assertLess(c.ky * 0 + c.oy, 0)

    def test_a_flat_wall_registers_with_unknown_borders(self):
        out = hd.register_depth(depth_frame(1500), hd.nominal_calibration())
        self.assertEqual(out.shape, (90, 160))
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(int(out[45, 80]), 1500)
        # colour is wider than depth: the outer columns have no depth (0 = unknown)
        self.assertEqual(int(out[45, 0]), 0)
        self.assertEqual(int(out[45, 159]), 0)
        cols = np.where((out > 0).any(axis=0))[0]
        self.assertEqual((int(cols.min()), int(cols.max())), (17, 142))
        # depth is taller than colour: every row has depth in the middle
        self.assertTrue(bool((out[:, 80] > 0).all()))
        self.assertTrue(set(np.unique(out)) <= {0, 1500})

    def test_range_gate_zero_means_unknown(self):
        c = hd.nominal_calibration()
        for mm in (0, 400, 499, 4501, 8000):
            self.assertEqual(int(hd.register_depth(depth_frame(mm), c).max()), 0, mm)
        self.assertEqual(int(hd.register_depth(depth_frame(500), c)[45, 80]), 500)
        self.assertEqual(int(hd.register_depth(depth_frame(4500), c)[45, 80]), 4500)

    def test_the_nearest_sample_wins_a_cell(self):
        c = hd.nominal_calibration()
        d = depth_frame(3000)
        d[210:214, 254:258] = 700                              # a hand-sized blob at the optical centre
        out = hd.register_depth(d, c)
        self.assertEqual(int(out[45, 80]), 700)
        self.assertEqual(int(out[10, 40]), 3000)

    def test_parallax_term_shifts_near_depth_more_than_far(self):
        c = hd.Calib(*(lambda n: (n.kx, n.ox, 100.0, n.ky, n.oy, 0.0))(hd.nominal_calibration()))
        near = hd.register_depth(depth_frame(1000), c)
        far = hd.register_depth(depth_frame(2000), c)
        nc = np.where((near > 0).any(axis=0))[0]
        fc = np.where((far > 0).any(axis=0))[0]
        # bx/Z colour pixels: 100 px at 1 m vs 50 px at 2 m -> 50/1920*160 = ~4 cells apart
        shift = int(nc.min()) - int(fc.min())
        self.assertIn(shift, (4, 5))
        # nominal (no baseline) puts both at the same place
        n0 = hd.nominal_calibration()
        a = np.where((hd.register_depth(depth_frame(1000), n0) > 0).any(axis=0))[0]
        b = np.where((hd.register_depth(depth_frame(2000), n0) > 0).any(axis=0))[0]
        self.assertEqual(int(a.min()), int(b.min()))

    def test_fit_recovers_a_known_sdk_mapping(self):
        # A synthetic Kinect: colour px = Ac*(-X/Z) + Bc/Z + 960, depth px = Ad*(X/Z) + 256.
        def cm(x, y, z):
            return (1081.0 * (-x / z) + 56.0 / z + 960.0, 1081.0 * (-y / z) + 540.0)

        def dm(x, y, z):
            return (-365.0 * (x / z) + 256.0, 365.0 * (-y / z) + 212.0)
        c = hd.fit_calibration(cm, dm)
        self.assertIsNotNone(c)
        self.assertEqual(c.method, hd.METHOD_SDK_FIT)
        self.assertAlmostEqual(c.kx, 1081.0 / 365.0, places=6)
        self.assertAlmostEqual(c.bx, 56.0, places=4)
        self.assertAlmostEqual(c.by, 0.0, places=4)
        self.assertLess(c.residual_px, 0.01)
        # ... and registering with it puts the optical centre at the colour centre
        self.assertAlmostEqual(c.kx * 256 + c.ox + c.bx / 2.0, 960.0 + 56.0 / 2.0, places=3)

    def test_fit_refuses_when_it_cannot_be_trusted(self):
        self.assertIsNone(hd.fit_calibration(lambda *a: None, lambda *a: None))
        # a mapper that works only for a few points
        calls = {"n": 0}

        def flaky(x, y, z):
            calls["n"] += 1
            return (960.0, 540.0) if calls["n"] < 4 else None
        self.assertIsNone(hd.fit_calibration(flaky, flaky))
        # a wildly non-linear mapping: residual above the limit
        wild = lambda x, y, z: (960 + 1500 * (x / z) ** 3 + 900 * (x / z), 540 + 700 * (y / z))  # noqa: E731
        dm = lambda x, y, z: (256 + 365 * (x / z), 212 + 365 * (y / z))                           # noqa: E731
        self.assertIsNone(hd.fit_calibration(wild, dm))
        nan = lambda x, y, z: (float("nan"), float("nan"))                                        # noqa: E731
        self.assertIsNone(hd.fit_calibration(nan, nan))
        self.assertIsNone(hd.fit_calibration(lambda *a: 1 / 0, lambda *a: 1 / 0))             # raises

    def test_fit_makes_at_most_60_mapper_calls(self):
        n = {"c": 0, "d": 0}

        def cm(x, y, z):
            n["c"] += 1
            return (960.0 - 1081 * x / z, 540.0 - 1081 * y / z)

        def dm(x, y, z):
            n["d"] += 1
            return (256.0 + 365 * x / z, 212.0 - 365 * y / z)
        hd.fit_calibration(cm, dm)
        self.assertLessEqual(n["c"] + n["d"], 60)

    def test_palm_depth_is_a_windowed_median(self):
        plane = np.zeros((90, 160), np.uint16)
        plane[40:50, 75:85] = 800
        self.assertAlmostEqual(hd.sample_median_m(plane, 0.5, 0.5), 0.8)
        plane[45, 80] = 100                                    # an outlier does not move a median
        self.assertAlmostEqual(hd.sample_median_m(plane, 0.5, 0.5), 0.8)
        self.assertIsNone(hd.sample_median_m(plane, 0.05, 0.05))           # all unknown
        self.assertIsNone(hd.sample_median_m(plane, 1.4, 0.5))             # outside
        self.assertIsNone(hd.sample_median_m(None, 0.5, 0.5))              # never raises
        sparse = np.zeros((90, 160), np.uint16)
        sparse[45, 80] = 900
        self.assertIsNone(hd.sample_median_m(sparse, 0.5, 0.5))            # < 5 valid samples


# ── the feed ──────────────────────────────────────────────────────────────
@unittest.skipIf(np is None, "needs numpy")
class FeedGateTests(unittest.TestCase):
    def test_not_selected_is_one_compare_and_touches_nothing(self):
        feed, br, fr, dw, _ = make_feed(selected=False)
        self.assertEqual(feed.step(), "not selected")
        self.assertEqual(br.calls, [])
        self.assertEqual(fr.published, [])
        self.assertFalse(dw.is_open)

    def test_no_tracker_no_publish_and_the_sensor_is_not_asked(self):
        feed, br, fr, _dw, _ = make_feed(active=False)
        self.assertEqual(feed.step(), "tracker away")
        self.assertEqual(br.calls, [])
        self.assertEqual(fr.published, [])

    def test_publishes_the_kinect_colour_when_selected_and_the_tracker_is_alive(self):
        feed, br, fr, dw, _ = make_feed()
        self.assertEqual(feed.step(), "")
        self.assertEqual(len(fr.published), 1)
        self.assertEqual(fr.published[0].shape, (1080, 1920, 3))
        self.assertEqual(feed.color_frames, 1)
        self.assertFalse(dw.is_open)                           # depth is off: no segment at all
        self.assertNotIn(("depth", True), br.calls)
        self.assertNotIn(("depth", False), br.calls)

    def test_it_never_opens_the_kinect(self):
        feed, br, fr, _dw, _ = make_feed()
        br.open = False
        self.assertEqual(feed.step(), "kinect not open")
        self.assertEqual([c for c in br.calls if isinstance(c, tuple)], [])   # no pixel read
        self.assertEqual(fr.published, [])
        br.open, br.enabled = True, False
        feed2, br2, fr2, _, _ = make_feed()
        br2.enabled = False
        self.assertEqual(feed2.step(), "kinect not open")
        self.assertEqual(fr2.published, [])

    def test_only_the_bridges_read_accessors_are_used(self):
        feed, br, _fr, _dw, _ = make_feed(depth=True)
        feed.step()
        names = {c if isinstance(c, str) else c[0] for c in br.calls}
        self.assertLessEqual(names, {"health", "clock", "color", "depth", "cmapper", "dmapper"})
        self.assertIn(("color", False), br.calls)              # the composite's own call: no frame consumed away
        self.assertNotIn(("color", True), br.calls)

    def test_rate_cap(self):
        feed, br, fr, _dw, clock = make_feed()
        self.assertEqual(feed.step(), "")
        br.color_clock += 1
        self.assertEqual(feed.step(), "rate cap")              # same instant
        clock.t += 0.02
        self.assertEqual(feed.step(), "rate cap")
        clock.t += hd.MIN_GAP_S
        self.assertEqual(feed.step(), "")
        self.assertEqual(len(fr.published), 2)
        # a loop spinning at 1 kHz for one second still yields at most ~16 frames
        n0 = len(fr.published)
        for _ in range(1000):
            clock.t += 0.001
            br.color_clock += 1
            feed.step()
        self.assertLessEqual(len(fr.published) - n0, int(1.0 / hd.MIN_GAP_S) + 1)
        self.assertGreaterEqual(1.0 / hd.MIN_GAP_S, 15.0)
        self.assertLessEqual(hd.MAX_FPS, 15.0)

    def test_a_frame_already_sent_is_not_sent_again(self):
        feed, br, fr, _dw, clock = make_feed()
        feed.step()
        clock.t += 1.0
        self.assertEqual(feed.step(), "no new colour")         # the sensor's frame clock did not move
        self.assertEqual(len(fr.published), 1)
        br.color_clock += 0.033
        self.assertEqual(feed.step(), "")
        self.assertEqual(len(fr.published), 2)

    def test_no_colour_frame_is_a_reason_not_an_error(self):
        feed, br, fr, _dw, _ = make_feed()
        br.color = None
        self.assertEqual(feed.step(), "no colour frame")
        self.assertEqual(fr.published, [])
        self.assertEqual(feed.last_error, "")

    def test_a_writer_that_declines_is_not_counted(self):
        feed, _br, fr, _dw, _ = make_feed()
        fr.accept = False
        self.assertEqual(feed.step(), "writer declined")
        self.assertEqual(feed.color_frames, 0)

    def test_never_raises_into_the_caller_and_backs_off(self):
        feed, br, fr, _dw, clock = make_feed()
        br.color_raises = True
        self.assertEqual(feed.step(), "error")                 # did not raise
        self.assertIn("pump exploded", feed.last_error)
        br.color_raises = False
        clock.t += 1.0
        self.assertEqual(feed.step(), "backing off")           # not hammering a sick bridge
        clock.t += hd.ERROR_BACKOFF_S
        self.assertEqual(feed.step(), "")
        # a colour writer that raises, a health call that raises
        fr.raise_on_publish = True
        clock.t += 1.0
        br.color_clock += 1
        self.assertEqual(feed.step(), "error")
        br.get_stream_health = mock.Mock(side_effect=RuntimeError("x"))
        clock.t += 10.0
        self.assertEqual(feed.step(), "error")

    def test_the_module_entry_point_never_raises(self):
        with mock.patch.object(hd.FEED, "step", side_effect=RuntimeError("boom")):
            self.assertEqual(hd.publish_kinect(), "error")
        with mock.patch.object(hd, "FEED", None):
            self.assertEqual(hd.publish_kinect(), "error")
            self.assertEqual(hd.feed_status()["selected"], False)

    def test_defaults_are_off(self):
        from core import config
        self.assertFalse(config.HOLO_DEPTH_ENABLED)
        self.assertEqual(config.HAND_TRACK_CAMERA, "primary")
        self.assertFalse(hd.KinectFeed(session=None).kinect_selected())

    def test_a_kinect_hologram_selects_the_kinect_without_any_setting(self):
        up = [False]
        feed, _br, fr, _dw, _ = make_feed(selected=False, depth=False,
                                          session=lambda: up[0])
        self.assertEqual(feed.step(), "not selected")
        up[0] = True
        self.assertTrue(feed.kinect_selected() and feed.depth_wanted())
        self.assertEqual(feed.step(), "")
        self.assertEqual(len(fr.published), 1)

    def test_effective_setting_blocks_the_webcams_during_a_kinect_hologram(self):
        with mock.patch.object(hd, "holo_session", return_value=True):
            self.assertEqual(hd.effective_camera_setting("primary"), "kinect")
        with mock.patch.object(hd, "holo_session", return_value=False):
            self.assertEqual(hd.effective_camera_setting("primary"), "primary")
        with mock.patch.object(hd, "holo_session", side_effect=RuntimeError):
            self.assertEqual(hd.effective_camera_setting("right"), "right")
        from core import handtrack_frames as hf
        cam = {"primary": True, "index": 0}
        with mock.patch.object(hd, "holo_session", return_value=True):
            self.assertFalse(hf.is_hand_camera(cam))           # "kinect" matches no webcam
        with mock.patch.object(hd, "holo_session", return_value=False):
            self.assertTrue(hf.is_hand_camera(cam, "primary"))

    def test_idle_tracker_closes_the_depth_segment(self):
        feed, _br, _fr, dw, clock = make_feed(depth=True)
        feed.step()
        self.assertTrue(dw.is_open)
        feed._frames.accept = True
        feed._frames._active = False
        feed.step()
        clock.t += hd.IDLE_CLOSE_S + 1
        feed.step()
        self.assertFalse(dw.is_open)


@unittest.skipIf(np is None, "needs numpy")
class FeedDepthTests(unittest.TestCase):
    def test_depth_is_registered_to_the_colour_frame_it_follows(self):
        feed, br, fr, dw, _ = make_feed(depth=True)
        feed.step()
        self.assertEqual(feed.depth_frames, 1)
        got = hd.read_depth(dw._shm.buf, 0)
        seq, color_seq, _ts, plane = got
        self.assertEqual(color_seq, fr.seq)                    # paired with colour frame #1
        self.assertEqual(plane.shape, (hd.OUT_H, hd.OUT_W))
        self.assertEqual(int(plane[45, 80]), 1500)
        self.assertEqual(hd.unpack_header(dw._shm.buf)["method"], hd.METHOD_NOMINAL)
        self.assertIn(("depth", True), br.calls)               # the bridge's freshness rule

    def test_depth_off_means_no_depth_calls_and_no_segment(self):
        feed, br, _fr, dw, _ = make_feed(depth=False)
        feed.step()
        self.assertFalse(dw.is_open)
        self.assertFalse([c for c in br.calls if isinstance(c, tuple) and c[0] == "depth"])
        self.assertNotIn("cmapper", br.calls)

    def test_no_new_depth_frame_publishes_no_plane(self):
        feed, br, _fr, dw, _ = make_feed(depth=True)
        br.depth_new = False
        self.assertEqual(feed.step(), "")                      # colour still goes
        self.assertEqual(feed.depth_frames, 0)
        self.assertFalse(dw.is_open)

    def test_uses_the_sdk_mapper_fit_when_the_bridge_has_one(self):
        feed, br, _fr, dw, _ = make_feed(depth=True)
        br.color_mapper = lambda x, y, z: (960.0 - 1081 * x / z + 56.0 / z, 540.0 - 1081 * y / z)
        br.depth_mapper = lambda x, y, z: (256.0 + 365 * x / z, 212.0 - 365 * y / z)
        feed.step()
        self.assertEqual(hd.unpack_header(dw._shm.buf)["method"], hd.METHOD_SDK_FIT)
        self.assertEqual(feed.status()["registration"]["method"], "sdk-fit")
        self.assertTrue(feed.status()["depth_active"])

    def test_falls_back_to_the_nominal_mapping_and_retries_the_fit_later(self):
        feed, br, _fr, dw, clock = make_feed(depth=True)
        feed.step()                                            # no mappers exposed
        self.assertEqual(feed.status()["registration"]["method"], "nominal-fov")
        n_before = br.calls.count("cmapper")
        br.color_clock += 1
        clock.t += 1.0
        feed.step()
        self.assertEqual(br.calls.count("cmapper"), n_before)  # not retried every frame
        br.color_mapper = lambda x, y, z: (960.0 - 1081 * x / z, 540.0 - 1081 * y / z)
        br.depth_mapper = lambda x, y, z: (256.0 + 365 * x / z, 212.0 - 365 * y / z)
        br.color_clock += 1
        clock.t += hd.FIT_RETRY_S + 1
        feed.step()
        self.assertEqual(feed.status()["registration"]["method"], "sdk-fit")
        # a good fit is kept: the mappers are not called again
        n = br.calls.count("cmapper")
        br.color_clock += 1
        clock.t += hd.FIT_RETRY_S + 1
        feed.step()
        self.assertEqual(br.calls.count("cmapper"), n)

    def test_a_mapper_that_raises_falls_back_quietly(self):
        feed, br, _fr, dw, _ = make_feed(depth=True)
        br.get_color_space_mapper = mock.Mock(side_effect=RuntimeError("com"))
        self.assertEqual(feed.step(), "")
        self.assertEqual(feed.status()["registration"]["method"], "nominal-fov")

    def test_depth_active_is_honest_about_age(self):
        feed, _br, _fr, _dw, clock = make_feed(depth=True)
        self.assertFalse(feed.depth_active())
        feed.step()
        self.assertTrue(feed.depth_active())
        clock.t += 5
        self.assertFalse(feed.depth_active())


# ── the bridge accessor and the "never touches the pump" contract ─────────
class BridgeContractTests(unittest.TestCase):
    def test_frame_clock_is_a_pure_peek(self):
        from audio import kinect_bridge as kb
        rt = types.SimpleNamespace(_last_color_frame_time=12.5, _last_depth_frame_time=None)
        with mock.patch.object(kb, "_runtime", [rt]):
            self.assertEqual(kb.frame_clock("color"), 12.5)
            self.assertIsNone(kb.frame_clock("depth"))
            self.assertIsNone(kb.frame_clock("body"))          # only colour / depth
            self.assertIsNone(kb.frame_clock("../x"))
        with mock.patch.object(kb, "_runtime", [None]):
            self.assertIsNone(kb.frame_clock("color"))         # no runtime: no open attempt
        with mock.patch.object(kb, "_runtime", [types.SimpleNamespace(_last_color_frame_time="x")]):
            self.assertIsNone(kb.frame_clock("color"))         # never raises

    def test_frame_clock_does_not_open_or_read_anything(self):
        from audio import kinect_bridge as kb
        rt = types.SimpleNamespace(_last_color_frame_time=1.0)
        with mock.patch.object(kb, "_runtime", [rt]), \
                mock.patch.object(kb, "get_runtime", side_effect=AssertionError("opened")), \
                mock.patch.object(kb, "get_color_bgr", side_effect=AssertionError("read")):
            self.assertEqual(kb.frame_clock("color"), 1.0)

    def test_the_pump_and_the_frame_accessors_know_nothing_about_the_hand_off(self):
        import inspect
        from audio import kinect_bridge as kb
        for fn in (kb._pump_tick, kb._prime_color_frame, kb._body_pump_loop,
                   kb._read_and_cache_bodies, kb.get_color_bgr, kb.get_depth,
                   kb.start_body_pump, kb.stop_body_pump):
            src = inspect.getsource(fn)
            for name in ("handtrack", "hand_track", "holo_", "KinectFeed"):
                self.assertNotIn(name, src, (fn.__name__, name))
        with open(os.path.join(_ROOT, "audio", "kinect_bridge.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("import handtrack", text)
        self.assertNotIn("from core import handtrack", text)

    def test_the_face_loop_calls_the_feed_once_per_pass_and_only_when_the_camera_is_on(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("from core import handtrack_depth as _handtrack_depth", src)
        i = src.index("_handtrack_depth.publish_kinect()")
        self.assertEqual(src.count("_handtrack_depth.publish_kinect()"), 1)
        self.assertIn("if not camera_off:", src[i - 80:i])
        j = src.index("# Read all cameras, run face detection on each", i)
        self.assertLess(j - i, 200)                            # right before the per-camera loop
        # it sits in the face-tracking loop, after the per-pass camera_off decision
        self.assertGreater(i, src.rindex("camera_off = (_face_track_camera_off.is_set()", 0, i))

    def test_the_module_imports_nothing_from_jarvis_at_load_time(self):
        with open(os.path.join(_ROOT, "core", "handtrack_depth.py"), encoding="utf-8") as f:
            src = f.read()
        top = src.split("\nclass Calib", 1)[0]
        for line in top.splitlines():
            if line.startswith(("import ", "from ")):
                self.assertTrue(line.split()[1] in ("__future__", "math", "struct", "time", "typing"), line)

    def test_no_windows_only_path_literals(self):
        with open(os.path.join(_ROOT, "core", "handtrack_depth.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotRegex(src, r"[A-Za-z]:\\\\")


if __name__ == "__main__":
    unittest.main()
