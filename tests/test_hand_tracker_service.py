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
        self.frames_wanted = False
        self.clients = 0

    def broadcast(self, text=None, jpeg=None):
        if text is not None:
            self.texts.append(json.loads(text))
        if jpeg is not None:
            self.jpegs.append(jpeg)

    def wants_frames(self):
        return self.frames_wanted

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
                                            "state", "pinch", "palm", "palm_facing"]))
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
