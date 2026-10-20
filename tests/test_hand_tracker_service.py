"""tools/hand_tracker_service.py with a FAKE HandLandmarker and a fake hub:
the websocket protocol messages, mirroring, frame subscription, the idle
"hands gone" message, loopback-only binding, and the exit rules (JARVIS pid /
heartbeat). No camera, no mediapipe, no network."""
import asyncio
import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import numpy as np
    import cv2
except Exception:                                    # pragma: no cover
    np = cv2 = None

from tools import hand_tracker_service as svc  # noqa: E402
from tests.test_handtrack_gestures import make_hand  # noqa: E402


class FakeHub:
    def __init__(self):
        self.texts = []
        self.jpegs = []
        self.depths = []
        self.frames_wanted = False
        self.depth_wanted = False
        self.clients = 0

    def broadcast(self, text=None, jpeg=None, depth=None):
        if text is not None:
            self.texts.append(json.loads(text))
        if jpeg is not None:
            self.jpegs.append(jpeg)
        if depth is not None:
            self.depths.append(depth)

    def wants_frames(self):
        return self.frames_wanted

    def wants_depth(self):
        return self.depth_wanted

    def client_count(self):
        return self.clients


class FakeImage:
    def __init__(self, image_format=None, data=None):
        self.data = data


class FakeMp:
    Image = FakeImage
    ImageFormat = types.SimpleNamespace(SRGB=1)


def lm_objects(points):
    return [types.SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in points]


class FakeLandmarker:
    """Returns a prepared HandLandmarkerResult-shaped object per call."""

    def __init__(self):
        self.hands = []            # list of (hand dict from make_hand, score)
        self.images = []
        self.stamps = []

    def detect_for_video(self, image, ms):
        self.images.append(image.data)
        self.stamps.append(ms)
        r = types.SimpleNamespace(hand_landmarks=[], handedness=[],
                                  hand_world_landmarks=[])
        for hd in self.hands:
            r.hand_landmarks.append(lm_objects(hd["lm"]))
            r.handedness.append([types.SimpleNamespace(
                category_name=hd["side"], score=hd["score"])])
            r.hand_world_landmarks.append(lm_objects(
                [[(p[0] - 0.5) * 0.2, (p[1] - 0.5) * 0.2, 0.0] for p in hd["lm"]]))
        return r


@unittest.skipIf(cv2 is None, "needs numpy + cv2")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.args = svc.parse_args(["--alive", os.path.join(self.dir.name, "alive.json")])
        self.t = svc.Tracker(self.args)
        self.hub = FakeHub()
        self.t.hub = self.hub
        self.lm = FakeLandmarker()
        self.t.landmarker = self.lm
        self.t._cv2, self.t._mp = cv2, FakeMp
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.frame[:, :100, 2] = 255                         # red stripe on the left

    def tearDown(self):
        self.dir.cleanup()

    def process(self, seq=1):
        self.t._process(seq, 1700000000000 + seq, self.frame)

    def test_frame_is_mirrored_before_detection(self):
        self.process()
        seen = self.lm.images[0]                             # RGB, flipped
        self.assertEqual(seen.shape, (480, 640, 3))
        self.assertEqual(int(seen[0, 639, 0]), 255)          # stripe is on the right now
        self.assertEqual(int(seen[0, 0, 0]), 0)

    def test_hands_message_matches_the_protocol(self):
        self.lm.hands = [make_hand("open", side="Right", cx=400, score=0.93)]
        self.process(5)
        msg = self.hub.texts[-1]
        self.assertEqual(msg["type"], "hands")
        self.assertEqual((msg["seq"], msg["w"], msg["h"]), (5, 640, 480))
        self.assertEqual(msg["t"], 1700000000005)
        self.assertEqual(len(msg["hands"]), 1)
        h = msg["hands"][0]
        self.assertEqual(sorted(h), sorted(["id", "side", "score", "lm", "world",
                                            "state", "pinch", "palm", "palm_facing",
                                            "size", "ok"]))
        # 2026-10-20 recognition: size over the frame height, ok = may trigger
        self.assertAlmostEqual(h["size"], 100.0 / 480, delta=0.03)
        self.assertIsInstance(h["ok"], bool)
        self.assertEqual(h["side"], "Right")
        self.assertAlmostEqual(h["score"], 0.93)
        self.assertEqual((len(h["lm"]), len(h["lm"][0])), (21, 3))
        self.assertEqual((len(h["world"]), len(h["world"][0])), (21, 3))
        self.assertIn(h["state"], ("open", "fist", "pinch", "point", "unknown"))
        self.assertTrue(0.0 <= h["pinch"] <= 1.0)
        self.assertEqual(len(h["palm"]), 2)
        self.assertIsInstance(h["palm_facing"], bool)
        self.assertEqual(h["state"], "open")
        self.assertTrue(h["palm_facing"])
        # coordinates are rounded to 4 places (small messages at 30/s)
        self.assertEqual(h["lm"][0], [round(v, 4) for v in h["lm"][0]])

    def test_no_hands_is_an_empty_list(self):
        self.process()
        self.assertEqual(self.hub.texts[-1]["hands"], [])

    def test_two_hands_have_distinct_ids(self):
        self.lm.hands = [make_hand("open", side="Left", cx=200),
                         make_hand("open", side="Right", cx=440)]
        self.process()
        ids = [h["id"] for h in self.hub.texts[-1]["hands"]]
        self.assertEqual(sorted(ids), [0, 1])

    def test_gesture_events_are_sent_as_gesture_messages(self):
        self.lm.hands = [make_hand("pinch")]
        for i in range(30):
            self.process(i + 1)
        evs = [m for m in self.hub.texts if m["type"] == "gesture"]
        self.assertEqual([e["name"] for e in evs][:1], ["pinch_start"])
        for e in evs:
            self.assertIn(e["side"], ("Left", "Right", "both"))
            self.assertEqual(len(e["pos"]), 2)

    def test_jpeg_only_for_subscribers(self):
        self.process(1)
        self.assertEqual(self.hub.jpegs, [])
        self.hub.frames_wanted = True
        self.process(2)
        self.assertEqual(len(self.hub.jpegs), 1)
        jpg = self.hub.jpegs[0]
        self.assertEqual(jpg[:2], b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (480, 640, 3))
        self.assertGreater(int(decoded[0, 639, 2]), 200)     # mirrored stripe, red

    def test_detector_timestamps_strictly_increase(self):
        for i in range(5):
            self.process(i + 1)
        self.assertEqual(self.lm.stamps, sorted(set(self.lm.stamps)))

    def test_idle_sends_one_empty_hands_message(self):
        self.lm.hands = [make_hand("open")]
        self.process(1)
        self.assertEqual(len(self.hub.texts[-1]["hands"]), 1)
        self.t._last_frame_at = time.monotonic() - 1.0
        n = len(self.hub.texts)
        self.t._idle(time.monotonic())
        self.t._idle(time.monotonic())
        self.assertEqual(len(self.hub.texts), n + 1)
        self.assertEqual(self.hub.texts[-1]["hands"], [])


class FakeDepthShm:
    def __init__(self, buf):
        self.buf = buf

    def close(self):
        pass


@unittest.skipIf(cv2 is None, "needs numpy + cv2")
class DepthTests(unittest.TestCase):
    """Kinect depth (Phase 2): the tracker relays the registered plane to clients
    that subscribed to it and gives each hand a palm depth "z". Fake segment, fake
    clock; the plane is built with the writer half of core/handtrack_depth.py."""

    def setUp(self):
        from core import handtrack_depth as hd
        from tests.test_handtrack_depth import FakeShm
        self.hd = hd
        self.dir = tempfile.TemporaryDirectory()
        self.args = svc.parse_args(["--alive", os.path.join(self.dir.name, "alive.json")])
        self.t = svc.Tracker(self.args)
        self.hub = FakeHub()
        self.t.hub = self.hub
        self.lm = FakeLandmarker()
        self.t.landmarker = self.lm
        self.t._cv2, self.t._mp = cv2, FakeMp
        self.frame = np.zeros((360, 640, 3), np.uint8)
        self.writer = hd.DepthWriter(shm_factory=FakeShm)
        self.now_s = [1_700_000_000.0]
        self.t._wall = lambda: self.now_s[0]
        self.writer._wall = lambda: self.now_s[0]
        self.writer._open()
        self.t.depth_shm = FakeDepthShm(self.writer._shm.buf)

    def tearDown(self):
        self.dir.cleanup()

    def plane(self, fill=0, boxes=()):
        p = np.full((90, 160), fill, np.uint16)
        for (y0, y1, x0, x1, v) in boxes:
            p[y0:y1, x0:x1] = v
        return p

    def send(self, plane, color_seq=1):
        self.assertTrue(self.writer.publish(plane, color_seq, self.hd.METHOD_NOMINAL))
        self.t._poll_depth()

    def process(self, seq=1):
        self.t._process(seq, int(self.now_s[0] * 1000), self.frame)

    def test_the_plane_is_mirrored_and_relayed_only_to_depth_subscribers(self):
        left_wall = self.plane(3000, [(0, 90, 0, 20, 700)])          # something nearer, on the LEFT
        self.send(left_wall)
        self.assertEqual(self.hub.depths, [])                          # nobody subscribed
        self.hub.depth_wanted = True
        self.send(left_wall, color_seq=2)
        self.assertEqual(len(self.hub.depths), 1)
        msg = self.hd.unpack_wire(self.hub.depths[0])
        self.assertEqual((msg["width"], msg["height"], msg["seq"]), (160, 90, 2))
        self.assertTrue(msg["mirrored"])
        self.assertEqual(int(msg["plane"][45, 0]), 3000)              # the blob is on the RIGHT now,
        self.assertEqual(int(msg["plane"][45, 159]), 700)             # like the mirrored picture
        self.assertEqual(msg["near_mm"], 500)
        self.assertEqual(msg["far_mm"], 4500)

    def test_one_message_per_new_plane(self):
        self.hub.depth_wanted = True
        self.send(self.plane(2000))
        self.t._poll_depth()
        self.t._poll_depth()
        self.assertEqual(len(self.hub.depths), 1)
        self.send(self.plane(2100))
        self.assertEqual(len(self.hub.depths), 2)

    def test_hands_carry_z_when_depth_is_fresh(self):
        self.lm.hands = [make_hand("open", side="Right", cx=320, cy=180)]
        self.process(1)
        h0 = self.hub.texts[-1]
        self.assertNotIn("depth", h0)                                  # no plane yet: no z keys at all
        self.assertNotIn("z", h0["hands"][0])
        self.send(self.plane(2000, [(0, 90, 0, 160, 800)]))
        self.process(2)
        msg = self.hub.texts[-1]
        self.assertTrue(msg["depth"])
        z = msg["hands"][0]["z"]
        self.assertAlmostEqual(z, 0.8, places=3)

    def test_palm_z_is_sampled_where_the_palm_is_in_mirrored_space(self):
        # the palm sits near the picture's RIGHT edge once mirrored
        self.lm.hands = [make_hand("open", side="Right", cx=0.9 * 640, cy=180)]
        self.process(1)
        px = self.hub.texts[-1]["hands"][0]["palm"][0]
        self.assertGreater(px, 0.8)
        near_right = self.plane(2500, [(0, 90, 0, 40, 600)])           # raw: blob on the LEFT
        self.send(near_right)                                          # mirrored: on the RIGHT
        self.process(2)
        self.assertAlmostEqual(self.hub.texts[-1]["hands"][0]["z"], 0.6, places=3)

    def test_unknown_palm_depth_is_null_not_zero(self):
        self.lm.hands = [make_hand("open", side="Right", cx=320, cy=180)]
        self.send(self.plane(0))                                       # plane of unknown depth
        self.process(1)
        msg = self.hub.texts[-1]
        self.assertTrue(msg["depth"])
        self.assertIsNone(msg["hands"][0]["z"])

    def test_a_stale_plane_stops_the_z_keys(self):
        self.lm.hands = [make_hand("open", side="Right", cx=320, cy=180)]
        self.send(self.plane(1000))
        self.assertTrue(self.t.depth_fresh())
        self.now_s[0] += self.hd.FRESH_DEPTH_S + 0.1
        self.assertFalse(self.t.depth_fresh())
        self.process(1)
        self.assertNotIn("z", self.hub.texts[-1]["hands"][0])
        self.assertNotIn("depth", self.hub.texts[-1])

    def test_depth_never_breaks_the_colour_path(self):
        self.lm.hands = [make_hand("open", side="Right", cx=320, cy=180)]
        self.t.depth_shm = FakeDepthShm(memoryview(bytearray(10)))     # garbage segment
        self.t._poll_depth()                                           # no raise
        self.process(1)
        self.assertEqual(len(self.hub.texts[-1]["hands"]), 1)
        self.t.depthmod = None                                         # module failed to load
        self.t.depth_shm = None
        self.t._poll_depth()
        self.process(2)
        self.assertEqual(len(self.hub.texts[-1]["hands"]), 1)

    def test_heartbeat_says_whether_depth_is_flowing(self):
        self.t._beat(10.0)
        with open(self.t.args.alive, encoding="utf-8") as f:
            self.assertFalse(json.load(f)["depth"])
        self.send(self.plane(1000))
        self.t._last_beat = 0.0
        self.t._beat(20.0)
        with open(self.t.args.alive, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["depth"])

    def test_the_depth_segment_name_is_an_argument_with_a_default(self):
        self.assertEqual(svc.parse_args([]).depth_shm_name, "jarvis_handtrack_depth")
        self.assertEqual(svc.parse_args(["--depth-shm-name", "x"]).depth_shm_name, "x")
        self.assertEqual(self.hd.DEPTH_SHM_NAME, "jarvis_handtrack_depth")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.life = os.path.join(self.dir.name, "jarvis_alive")
        self.alive = os.path.join(self.dir.name, "alive.json")
        self.args = svc.parse_args(["--alive", self.alive, "--jarvis-alive", self.life,
                                    "--jarvis-pid", "4321", "--stale-s", "5",
                                    "--grace-s", "20"])
        self.t = svc.Tracker(self.args)
        self.t._last_exit_check = -10.0

    def tearDown(self):
        self.dir.cleanup()

    def gone(self):
        self.t._last_exit_check = -10.0
        return self.t._jarvis_gone(time.monotonic())

    def touch(self, age=0.0):
        with open(self.life, "w") as f:
            f.write("1")
        t = time.time() - age
        os.utime(self.life, (t, t))

    def test_runs_while_jarvis_is_alive(self):
        self.touch()
        with mock.patch.object(svc, "pid_alive", return_value=True):
            self.assertIsNone(self.gone())

    def test_exits_when_the_jarvis_process_is_gone(self):
        self.touch()
        with mock.patch.object(svc, "pid_alive", return_value=False):
            self.assertIn("gone", self.gone())

    def test_exits_when_the_heartbeat_goes_stale(self):
        with mock.patch.object(svc, "pid_alive", return_value=True):
            self.touch(age=2)
            self.assertIsNone(self.gone())
            self.touch(age=9)
            self.assertIn("stale", self.gone())

    def test_exits_when_jarvis_removes_the_heartbeat(self):
        with mock.patch.object(svc, "pid_alive", return_value=True):
            self.touch()
            self.assertIsNone(self.gone())
            os.remove(self.life)
            self.assertIn("removed", self.gone())

    def test_waits_for_a_heartbeat_that_has_not_appeared_yet(self):
        with mock.patch.object(svc, "pid_alive", return_value=True):
            self.assertIsNone(self.gone())                    # within the grace
            self.t.started -= 60
            self.assertIn("never appeared", self.gone())

    def test_check_is_throttled(self):
        with mock.patch.object(svc, "pid_alive", return_value=True) as p:
            self.touch()
            self.t._last_exit_check = -10.0
            self.t._jarvis_gone(time.monotonic())
            self.t._jarvis_gone(time.monotonic())
            self.assertEqual(p.call_count, 1)

    def test_heartbeat_file_carries_a_status_line(self):
        self.t._last_beat = 0.0
        self.t.hands_now = 2
        self.t._beat(time.monotonic())
        with open(self.alive, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(d["hands"], 2)
        self.assertEqual(d["port"], 8793)
        self.assertFalse(d["attached"])
        self.assertEqual(d["pid"], os.getpid())

    def test_refuses_a_non_loopback_host(self):
        args = svc.parse_args(["--host", "0.0.0.0", "--alive", self.alive])
        t = svc.Tracker(args)
        with mock.patch.object(t, "load_model") as lm:
            self.assertEqual(t.run(), 2)
        lm.assert_not_called()

    def test_missing_model_is_a_clean_failure(self):
        args = svc.parse_args(["--model", os.path.join(self.dir.name, "nope.task"),
                               "--alive", self.alive])
        t = svc.Tracker(args)
        mp_stub = types.SimpleNamespace(__version__="x")
        with mock.patch.dict(sys.modules, {
                "mediapipe": mp_stub,
                "mediapipe.tasks": types.ModuleType("mediapipe.tasks"),
                "mediapipe.tasks.python": types.SimpleNamespace(
                    BaseOptions=object, vision=object)}):
            self.assertEqual(t.run(), 3)


class HubTests(unittest.TestCase):
    def make(self):
        hub = svc.Hub("127.0.0.1", 0)
        a = {"q": asyncio.Queue(maxsize=2), "frames": False}
        b = {"q": asyncio.Queue(maxsize=2), "frames": True}
        hub.clients = {"a": a, "b": b}
        return hub, a, b

    def test_slow_clients_drop_the_oldest_message(self):
        hub, a, _b = self.make()
        for i in range(5):
            hub._fanout(f"m{i}", None)
        self.assertEqual([a["q"].get_nowait(), a["q"].get_nowait()], ["m3", "m4"])

    def test_jpeg_goes_only_to_subscribers(self):
        hub, a, b = self.make()
        hub._fanout(None, b"JPEG")
        self.assertTrue(a["q"].empty())
        self.assertEqual(b["q"].get_nowait(), b"JPEG")

    def test_text_goes_to_everyone(self):
        hub, a, b = self.make()
        hub._fanout("hello", None)
        self.assertEqual(a["q"].get_nowait(), "hello")
        self.assertEqual(b["q"].get_nowait(), "hello")

    def test_origin_policy_is_localhost_only(self):
        ok = ["http://127.0.0.1:8788", "http://localhost", "https://localhost:3000",
              "http://[::1]:80"]
        bad = ["https://evil.example", "http://127.0.0.1.evil.example",
               "http://localhost.evil.example:80", "null", "file://"]
        for o in ok:
            self.assertTrue(svc.ORIGIN_RE.match(o), o)
        for o in bad:
            self.assertFalse(svc.ORIGIN_RE.match(o), o)

    def test_depth_goes_only_to_depth_subscribers(self):
        hub = svc.Hub("127.0.0.1", 0)
        a = {"q": asyncio.Queue(maxsize=4), "frames": True, "depth": False}
        b = {"q": asyncio.Queue(maxsize=4), "frames": False, "depth": True}
        old = {"q": asyncio.Queue(maxsize=4), "frames": True}          # a pre-depth client state
        hub.clients = {"a": a, "b": b, "c": old}
        hub._fanout(None, None, b"JVDP-plane")
        self.assertTrue(a["q"].empty())
        self.assertTrue(old["q"].empty())
        self.assertEqual(b["q"].get_nowait(), b"JVDP-plane")
        hub._fanout(None, b"JPEG")                                     # old call shape still works
        self.assertEqual(a["q"].get_nowait(), b"JPEG")
        self.assertTrue(b["q"].empty())
        self.assertTrue(hub.wants_depth())
        b["depth"] = False
        self.assertFalse(hub.wants_depth())

    def test_subscribe_keys_are_independent(self):
        st = {"frames": False, "depth": False}
        svc.Hub.apply_subscribe(st, {"type": "subscribe", "frames": True})
        self.assertEqual(st, {"frames": True, "depth": False})
        svc.Hub.apply_subscribe(st, {"type": "subscribe", "depth": True})
        self.assertEqual(st, {"frames": True, "depth": True})         # frames untouched
        svc.Hub.apply_subscribe(st, {"type": "subscribe", "frames": False, "depth": True})
        self.assertEqual(st, {"frames": False, "depth": True})
        svc.Hub.apply_subscribe(st, {"type": "subscribe"})             # a bare subscribe: frames off, as ever
        self.assertEqual(st["frames"], False)
        svc.Hub.apply_subscribe(st, {"type": "subscribe", "depth": False, "frames": True})
        self.assertEqual(st, {"frames": True, "depth": False})

    def test_last_hands_message_is_replayed_to_new_clients(self):
        hub = svc.Hub("127.0.0.1", 0)
        hub.loop = mock.Mock()
        hub.broadcast('{"type":"hands","hands":[]}')
        hub.broadcast('{"type":"gesture","name":"clap"}')
        self.assertEqual(hub.last_hands_text, '{"type":"hands","hands":[]}')


class SourceTests(unittest.TestCase):
    def test_server_binds_the_loopback_address_only(self):
        self.assertEqual(svc.parse_args([]).host, "127.0.0.1")
        self.assertEqual(svc.parse_args([]).port, 8793)
        self.assertNotIn("0.0.0.0", ("127.0.0.1", "::1", "localhost"))
        self.assertEqual(svc.LOOPBACK, ("127.0.0.1", "::1", "localhost"))


if __name__ == "__main__":
    unittest.main()
