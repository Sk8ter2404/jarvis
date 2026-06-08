"""Tests for audio.kinect_bridge — the lazy, graceful Xbox Kinect v2 client.

The whole point of this module is to NEVER touch pykinect2 / the Kinect Runtime
at import time and to gate every accessor behind the enabled flag + a real
sensor open, returning graceful sentinels (None / [] / a present:False dict)
rather than raising. These tests exercise all of that WITHOUT a real sensor,
real pykinect2, or even numpy/cv2 necessarily being importable on the host:

  * import_pykinect2() is monkeypatched to return fake PyKinectV2 +
    PyKinectRuntime modules, so _open_runtime_locked() never execs real source.
    A test can also make it raise ImportError to drive the "pykinect2 not
    installed" branch.
  * numpy is injected as a light fake for the frame-reshape tests when the real
    one isn't present; cv2 is faked for the PNG path.

The module must still IMPORT on CI (it does no top-level pykinect2/numpy/cv2
import) — verified by the import at module top. stdlib unittest + mock.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from audio import kinect_bridge as kb


# ─────────────────────────────────────────────────────────────────────────
# fake pykinect2 modules + a fake runtime
# ─────────────────────────────────────────────────────────────────────────
def _fake_pk2_module():
    """A stand-in PyKinectV2 carrying just the FrameSourceTypes_* constants the
    bridge ORs together."""
    m = types.ModuleType("pykinect2.PyKinectV2")
    m.FrameSourceTypes_Color = 1
    m.FrameSourceTypes_Infrared = 2
    m.FrameSourceTypes_Depth = 8
    m.FrameSourceTypes_Body = 32
    return m


class _FakeJoint:
    def __init__(self, x, y, z, state=2):
        self.Position = types.SimpleNamespace(x=x, y=y, z=z)
        self.TrackingState = state


class _FakeBody:
    """A tracked/untracked body whose .joints is indexable 0.._JOINT_COUNT-1.

    hand_right_state / hand_left_state mirror the Kinect ints the real
    PyKinectRuntime sets on each body (0 Unknown,1 NotTracked,2 Open,3 Closed,
    4 Lasso). Optional so existing tests (which don't set them) still get the
    bridge's "unknown" degrade via getattr.

    tracking_id mirrors the stable per-person id the real PyKinectRuntime sets
    from body.TrackingId (a large nonzero 64-bit int). Optional: when omitted,
    the fake has no tracking_id attribute at all, so the bridge's getattr
    fallback to the slot index is exercised — matching the list-based fakes."""
    def __init__(self, tracked, joints=None, hand_right_state=None,
                 hand_left_state=None, tracking_id=None):
        self.is_tracked = tracked
        if tracking_id is not None:
            self.tracking_id = tracking_id
        # Default: a plausible upright, facing skeleton ~2 m away.
        if joints is None:
            joints = {}
        full = []
        for idx in range(kb._JOINT_COUNT):
            name = kb._JOINT_NAMES[idx]
            if name in joints:
                x, y, z = joints[name]
            else:
                # head high, spine lower, shoulders equidistant in z
                y = 0.6 if name == "head" else 0.0
                x = 0.0
                z = 2.0
            full.append(_FakeJoint(x, y, z))
        self.joints = full
        # Only attach the hand-state attrs when explicitly provided, so the
        # "absent attribute → unknown" degrade path is still exercised by the
        # tests that omit them.
        if hand_right_state is not None:
            self.hand_right_state = hand_right_state
        if hand_left_state is not None:
            self.hand_left_state = hand_left_state


def _as_body_array(bodies):
    """Wrap a sequence of fake bodies in the SAME container the real
    PyKinectRuntime hands back: a numpy ``ndarray(dtype=object)`` (see
    KinectBodyFrameData.bodies = numpy.ndarray((max_body_count), dtype=object)).

    Faithfulness matters here: a plain Python list has a well-defined truth
    value, so any guard that boolean-tests ``frame.bodies`` (e.g. the old
    ``not frame.bodies``) passes on a list yet raises the ambiguous-truth
    ValueError on the real ndarray — the exact production bug that killed every
    gesture. Wrapping fixtures as an ndarray makes the harness exercise the real
    type. Falls back to the original sequence when numpy isn't importable (CI
    pre-imports numpy, so this practically always returns an ndarray)."""
    try:
        import numpy as np
    except Exception:   # pragma: no cover - numpy is present on dev + CI
        return bodies
    arr = np.empty(len(bodies), dtype=object)
    for i, b in enumerate(bodies):
        arr[i] = b
    return arr


class _FakeBodyFrame:
    def __init__(self, bodies):
        # Mirror hardware: real .bodies is an ndarray(dtype=object), not a list.
        self.bodies = _as_body_array(bodies)


class _FakeRuntime:
    """Mimics PyKinectRuntime's frame-readiness + getter surface. Each frame is
    'new' once, then has_new_* returns False until re-armed."""
    def __init__(self, source_flags=0, *, color=None, depth=None,
                 infrared=None, bodies=None, has_infrared_getter=True):
        self.source_flags = source_flags
        self._color = color
        self._depth = depth
        self._infrared = infrared
        self._bodies = bodies
        self._new = {"color": color is not None, "depth": depth is not None,
                     "infrared": infrared is not None, "body": bodies is not None}
        self.closed = False
        if not has_infrared_getter:
            # Emulate the real installed build, which lacks this getter entirely.
            self.get_last_infrared_frame = None

    # readiness
    def has_new_color_frame(self): return self._new["color"]
    def has_new_depth_frame(self): return self._new["depth"]
    def has_new_infrared_frame(self): return self._new["infrared"]
    def has_new_body_frame(self): return self._new["body"]

    # getters
    def get_last_color_frame(self):
        self._new["color"] = False
        return self._color

    def get_last_depth_frame(self):
        self._new["depth"] = False
        return self._depth

    def get_last_infrared_frame(self):
        self._new["infrared"] = False
        return self._infrared

    def get_last_body_frame(self):
        self._new["body"] = False
        return _FakeBodyFrame(self._bodies) if self._bodies is not None else None

    def close(self):
        self.closed = True


def _patch_loader(test, runtime):
    """Make import_pykinect2() return a fake PyKinectV2 plus a runtime module
    whose PyKinectRuntime(flags) yields `runtime`. Returns nothing; restores on
    cleanup."""
    pk2 = _fake_pk2_module()
    rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")

    def _ctor(flags):
        runtime.source_flags = flags
        return runtime
    rt_mod.PyKinectRuntime = _ctor
    p = mock.patch.object(kb, "import_pykinect2", lambda: (pk2, rt_mod))
    p.start()
    test.addCleanup(p.stop)


class _BridgeBase(unittest.TestCase):
    def setUp(self):
        # Reset module singletons + flags so state never leaks across tests.
        self.addCleanup(self._reset)
        kb._runtime[0] = None
        kb._open_error[0] = None
        kb._negative_until[0] = 0.0
        # Drop the shared body-frame cache so each test starts COLD: the first
        # get_bodies() then does a direct read of the test's armed frame instead
        # of being served a previous test's still-fresh cached bodies.
        kb._body_cache[0] = None
        kb._body_cache_at[0] = 0.0
        self._orig_enabled = kb._ENABLED
        kb._ENABLED = True   # most tests assume opted-in; disabled tests flip it
        # The production open-path now polls rt.has_new_color/body_frame() for
        # ~2.5s and only caches a runtime that actually STREAMS (rejecting a
        # dead sensor handle from a release-race). These fakes don't model a
        # live frame pump, so treat any opened fake runtime as streaming —
        # otherwise get_runtime would reject it and return None. No test here
        # exercises the rejection path, so this can't mask intended behaviour.
        _streams = mock.patch.object(kb, "_runtime_streams",
                                     lambda *a, **k: True)
        _streams.start()
        self.addCleanup(_streams.stop)

    def _reset(self):
        try:
            kb.close()
        except Exception:
            pass
        kb._runtime[0] = None
        kb._open_error[0] = None
        kb._negative_until[0] = 0.0
        kb._body_cache[0] = None
        kb._body_cache_at[0] = 0.0
        kb._ENABLED = self._orig_enabled

    def _inject(self, name, module):
        old = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, old) if old is not None
            else sys.modules.pop(name, None))


# ─────────────────────────────────────────────────────────────────────────
# set_enabled gating
# ─────────────────────────────────────────────────────────────────────────
class EnabledGateTests(_BridgeBase):
    def test_set_get_enabled_coerces_bool(self):
        kb.set_enabled("yes")
        self.assertIs(kb.get_enabled(), True)
        kb.set_enabled(0)
        self.assertIs(kb.get_enabled(), False)

    def test_disabled_runtime_short_circuits_without_loader(self):
        kb.set_enabled(False)
        # Loader must NOT be called when disabled — patch it to blow up if so.
        with mock.patch.object(kb, "import_pykinect2",
                               side_effect=AssertionError("loader touched")):
            rt, reason = kb.get_runtime()
        self.assertIsNone(rt)
        self.assertIn("disabled", reason)

    def test_disabled_available_is_false(self):
        kb.set_enabled(False)
        ok, reason = kb.available()
        self.assertFalse(ok)
        self.assertIn("disabled", reason)

    def test_set_enabled_false_closes_open_runtime(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        self.assertIsNotNone(kb.get_runtime()[0])
        kb.set_enabled(False)   # should tear the sensor down
        self.assertTrue(rt.closed)
        self.assertIsNone(kb._runtime[0])


# ─────────────────────────────────────────────────────────────────────────
# available() — present / absent paths + negative cache
# ─────────────────────────────────────────────────────────────────────────
class AvailableTests(_BridgeBase):
    def test_available_true_when_sensor_opens(self):
        _patch_loader(self, _FakeRuntime())
        ok, reason = kb.available()
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_available_false_when_pykinect2_absent(self):
        with mock.patch.object(kb, "import_pykinect2",
                               side_effect=ImportError("pykinect2")):
            ok, reason = kb.available()
        self.assertFalse(ok)
        self.assertIn("pykinect2 not installed", reason)

    def test_available_false_when_sensor_open_raises(self):
        pk2 = _fake_pk2_module()
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")

        def _ctor(_flags):
            raise RuntimeError("no sensor")
        rt_mod.PyKinectRuntime = _ctor
        with mock.patch.object(kb, "import_pykinect2", lambda: (pk2, rt_mod)):
            ok, reason = kb.available()
        self.assertFalse(ok)
        self.assertIn("could not open Kinect sensor", reason)

    def test_negative_result_is_cached(self):
        calls = {"n": 0}

        def _boom():
            calls["n"] += 1
            raise ImportError("pykinect2")
        with mock.patch.object(kb, "import_pykinect2", side_effect=_boom):
            ok1, _ = kb.available()
            ok2, _ = kb.available()   # within the cache window → no re-probe
        self.assertFalse(ok1)
        self.assertFalse(ok2)
        self.assertEqual(calls["n"], 1)   # loader hit only once

    def test_runtime_opened_with_all_four_sources(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        kb.get_runtime()
        # Color(1)|Infrared(2)|Depth(8)|Body(32) = 43
        self.assertEqual(rt.source_flags, 1 | 2 | 8 | 32)

    def test_runtime_is_singleton(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        a, _ = kb.get_runtime()
        b, _ = kb.get_runtime()
        self.assertIs(a, b)
        self.assertIs(a, rt)


# ─────────────────────────────────────────────────────────────────────────
# frame accessors — reshape correctness + graceful None
# ─────────────────────────────────────────────────────────────────────────
class ColorFrameTests(_BridgeBase):
    def test_color_reshapes_flat_bgra_to_bgr(self):
        np = _require_numpy(self)
        # Build a flat BGRA buffer with a known pixel so we can verify the
        # alpha is dropped and channel order preserved.
        flat = np.zeros(1920 * 1080 * 4, dtype=np.uint8)
        # pixel (0,0): B=10 G=20 R=30 A=40
        flat[0:4] = [10, 20, 30, 40]
        _patch_loader(self, _FakeRuntime(color=flat))
        bgr = kb.get_color_bgr()
        self.assertIsNotNone(bgr)
        self.assertEqual(bgr.shape, (1080, 1920, 3))
        self.assertEqual(tuple(int(v) for v in bgr[0, 0]), (10, 20, 30))

    def test_color_none_when_no_new_frame(self):
        rt = _FakeRuntime(color=None)   # has_new_color_frame False
        _patch_loader(self, rt)
        self.assertIsNone(kb.get_color_bgr())

    def test_color_wrong_size_returns_none(self):
        np = _require_numpy(self)
        flat = np.zeros(100, dtype=np.uint8)   # not 8294400
        _patch_loader(self, _FakeRuntime(color=flat))
        self.assertIsNone(kb.get_color_bgr())

    def test_color_none_when_disabled(self):
        kb.set_enabled(False)
        self.assertIsNone(kb.get_color_bgr())

    def test_color_png_encodes(self):
        np = _require_numpy(self)
        flat = np.zeros(1920 * 1080 * 4, dtype=np.uint8)
        _patch_loader(self, _FakeRuntime(color=flat))
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.imencode = lambda ext, img: (True, np.frombuffer(b"\x89PNGDATA",
                                                                  dtype=np.uint8))
        self._inject("cv2", fake_cv2)
        png = kb.get_color_png()
        self.assertIsInstance(png, bytes)
        self.assertTrue(png.startswith(b"\x89PNG"))


class DepthInfraredTests(_BridgeBase):
    def test_depth_reshapes_to_512x424(self):
        np = _require_numpy(self)
        flat = np.arange(512 * 424, dtype=np.uint16)
        _patch_loader(self, _FakeRuntime(depth=flat))
        d = kb.get_depth()
        self.assertIsNotNone(d)
        self.assertEqual(d.shape, (424, 512))
        self.assertEqual(str(d.dtype), "uint16")

    def test_depth_none_when_unavailable(self):
        _patch_loader(self, _FakeRuntime(depth=None))
        self.assertIsNone(kb.get_depth())

    def test_infrared_normalises_uint16_to_uint8(self):
        np = _require_numpy(self)
        flat = np.full(512 * 424, 1000, dtype=np.uint16)
        flat[0] = 2000   # the peak → maps to 255
        _patch_loader(self, _FakeRuntime(infrared=flat))
        ir = kb.get_infrared_gray()
        self.assertIsNotNone(ir)
        self.assertEqual(ir.shape, (424, 512))
        self.assertEqual(str(ir.dtype), "uint8")
        self.assertEqual(int(ir.flat[0]), 255)   # the peak pixel

    def test_infrared_none_when_getter_absent(self):
        # Mirrors the installed pykinect2 build: no get_last_infrared_frame.
        np = _require_numpy(self)
        flat = np.full(512 * 424, 1000, dtype=np.uint16)
        rt = _FakeRuntime(infrared=flat, has_infrared_getter=False)
        _patch_loader(self, rt)
        self.assertIsNone(kb.get_infrared_gray())


# ─────────────────────────────────────────────────────────────────────────
# body / presence
# ─────────────────────────────────────────────────────────────────────────
class PresenceTests(_BridgeBase):
    def test_get_bodies_returns_only_tracked(self):
        # 2 tracked of 6 slots — the rest untracked.
        bodies = [
            _FakeBody(True, {"head": (0, 0.6, 1.8), "spine_shoulder": (0, 0.0, 1.8)}),
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.5), "spine_shoulder": (0, 0.0, 2.5)}),
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
        ]
        _patch_loader(self, _FakeRuntime(bodies=bodies))
        got = kb.get_bodies()
        self.assertEqual(len(got), 2)
        self.assertIn("head", got[0]["joints"])
        self.assertEqual(got[0]["head"], (0.0, 0.6, 1.8))
        # joint tuples are (x, y, z, tracking_state)
        self.assertEqual(len(got[0]["joints"]["head"]), 4)

    def test_get_bodies_handles_numpy_object_array(self):
        # REGRESSION: on real hardware PyKinectRuntime.bodies is a length-6
        # numpy ndarray(dtype=object), not a list. The old guard did
        # `not getattr(frame, "bodies", None)` which calls bool() on that
        # array → ValueError("truth value of an array ... is ambiguous"),
        # swallowed by the broad except → get_bodies() returned [] on EVERY
        # frame, silently killing gestures/presence/head-yaw/hand-states.
        # This drives the exact ndarray shape to prove the guard no longer
        # bool()s the array.
        np = _require_numpy(self)
        raw = np.empty(6, dtype=object)
        raw[0] = _FakeBody(True, {"head": (0, 0.6, 1.8),
                                  "spine_shoulder": (0, 0.0, 1.8)})
        raw[1] = _FakeBody(False)
        raw[2] = _FakeBody(True, {"head": (0, 0.6, 2.5),
                                  "spine_shoulder": (0, 0.0, 2.5)})
        for i in range(3, 6):
            raw[i] = _FakeBody(False)
        _patch_loader(self, _FakeRuntime(bodies=raw))
        got = kb.get_bodies()
        # 2 tracked of the 6 ndarray slots survive — proves we iterated the
        # array (didn't bail to []) AND didn't raise on the truthiness check.
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0]["head"], (0.0, 0.6, 1.8))

    def test_get_presence_works_with_numpy_object_array(self):
        # Downstream proof: presence (and thus gestures/head-yaw, which all
        # consume get_bodies) sees the tracked bodies when the frame carries
        # the real ndarray-of-object body buffer.
        np = _require_numpy(self)
        raw = np.empty(6, dtype=object)
        raw[0] = _FakeBody(True, {"head": (0, 0.6, 1.8),
                                  "spine_shoulder": (0, 0.0, 1.8)})
        for i in range(1, 6):
            raw[i] = _FakeBody(False)
        _patch_loader(self, _FakeRuntime(bodies=raw))
        pres = kb.get_presence()
        self.assertTrue(pres["present"])
        self.assertEqual(pres["count"], 1)
        self.assertEqual(pres["nearest_m"], 1.8)

    def test_get_bodies_handles_real_ndarray_bodies(self):
        # REGRESSION (gestures-completely-dead): the real PyKinectRuntime returns
        # frame.bodies as a 6-long numpy ndarray(dtype=object) with the tracked
        # body at an arbitrary, usually non-zero, slot — NOT a Python list. A
        # guard that boolean-tests that array (the old `not frame.bodies`) raises
        # ValueError "truth value of an array … is ambiguous", which get_bodies'
        # blanket except swallows to [] — silently killing every body/gesture.
        # Build bodies exactly as hardware does and assert the one tracked body
        # survives. RED before the line-497 fix (raises→[]→len 0); GREEN after.
        np = _require_numpy(self)
        frame_bodies = np.array(
            [_FakeBody(False)] * 4 + [_FakeBody(True, {"head": (0, 0.6, 2.0)})]
            + [_FakeBody(False)], dtype=object)   # tracked at slot 4, mirrors live
        self.assertEqual(frame_bodies.shape, (6,))   # exactly the hardware shape
        _patch_loader(self, _FakeRuntime(bodies=frame_bodies))
        got = kb.get_bodies()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], 4)            # preserves the real slot index
        self.assertIn("head", got[0]["joints"])

    def test_get_bodies_id_uses_stable_tracking_id(self):
        # The real Kinect carries a stable .tracking_id that follows a person
        # across frames; a body can sit at ANY of the 6 slots (live: the lone
        # tracked body was at slot 4). The emitted 'id' must be that stable
        # tracking_id, NOT the volatile enumerate slot index.
        bodies = [
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 1.8),
                             "spine_shoulder": (0, 0.0, 1.8)},
                      tracking_id=72057594037928001),  # slot 4
            _FakeBody(False),
        ]
        _patch_loader(self, _FakeRuntime(bodies=bodies))
        got = kb.get_bodies()
        self.assertEqual(len(got), 1)
        # id is the tracking_id, not the slot index 4.
        self.assertEqual(got[0]["id"], 72057594037928001)

    def test_get_bodies_id_is_stable_across_slot_migration(self):
        # Same person (same tracking_id), two consecutive frames where the
        # Kinect moved them from slot 1 to slot 3. The slot index churns but the
        # emitted id must stay put — that's the whole point of the fix.
        tid = 72057594037928123
        frame_a = [
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.0),
                             "spine_shoulder": (0, 0.0, 2.0)}, tracking_id=tid),
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
            _FakeBody(False),
        ]
        frame_b = [
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.0),
                             "spine_shoulder": (0, 0.0, 2.0)}, tracking_id=tid),
            _FakeBody(False), _FakeBody(False),
        ]
        rt = _FakeRuntime(bodies=frame_a)
        _patch_loader(self, rt)
        id_a = kb.get_bodies()[0]["id"]
        # Re-arm the runtime with the migrated frame.
        rt._bodies = frame_b
        rt._new["body"] = True
        id_b = kb.get_bodies()[0]["id"]
        self.assertEqual(id_a, tid)
        self.assertEqual(id_b, tid)
        self.assertEqual(id_a, id_b)   # identity survives the slot move

    def test_get_bodies_id_falls_back_to_slot_when_no_tracking_id(self):
        # List-based / older fakes that carry no tracking_id attribute must
        # degrade to the enumerate slot index (here the 1st tracked body is at
        # slot 2, the 2nd at slot 4 → ids 2 and 4).
        bodies = [
            _FakeBody(False), _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 1.8),
                             "spine_shoulder": (0, 0.0, 1.8)}),  # slot 2
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.5),
                             "spine_shoulder": (0, 0.0, 2.5)}),  # slot 4
            _FakeBody(False),
        ]
        _patch_loader(self, _FakeRuntime(bodies=bodies))
        got = kb.get_bodies()
        self.assertEqual([b["id"] for b in got], [2, 4])

    def test_get_bodies_id_falsy_tracking_id_degrades_to_slot(self):
        # A tracking_id of 0 (or any falsy value) is not a valid Kinect id for a
        # tracked body; the guard must fall back to the slot index rather than
        # emit a misleading 0. The lone tracked body sits at slot 1 → id 1.
        bodies = [
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 1.8),
                             "spine_shoulder": (0, 0.0, 1.8)}, tracking_id=0),
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
            _FakeBody(False),
        ]
        _patch_loader(self, _FakeRuntime(bodies=bodies))
        got = kb.get_bodies()
        self.assertEqual(got[0]["id"], 1)   # slot index, not the falsy 0

    def test_get_presence_counts_two_and_picks_nearest(self):
        bodies = [
            _FakeBody(True, {"head": (0, 0.6, 1.8), "spine_shoulder": (0, 0.0, 1.8)}),
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.5), "spine_shoulder": (0, 0.0, 2.5)}),
            _FakeBody(False), _FakeBody(False), _FakeBody(False),
        ]
        _patch_loader(self, _FakeRuntime(bodies=bodies))
        pres = kb.get_presence()
        self.assertTrue(pres["present"])
        self.assertEqual(pres["count"], 2)
        self.assertEqual(pres["nearest_m"], 1.8)   # nearer of 1.8 / 2.5
        self.assertIn("ts", pres)

    def test_get_presence_empty_when_no_bodies(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(False)] * 6))
        pres = kb.get_presence()
        self.assertFalse(pres["present"])
        self.assertEqual(pres["count"], 0)
        self.assertIsNone(pres["nearest_m"])

    def test_get_bodies_empty_when_runtime_none(self):
        kb.set_enabled(False)
        self.assertEqual(kb.get_bodies(), [])

    def test_get_presence_graceful_when_runtime_none(self):
        kb.set_enabled(False)
        pres = kb.get_presence()
        self.assertEqual(pres["present"], False)
        self.assertEqual(pres["count"], 0)

    def test_get_presence_swallows_get_bodies_error(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(True)]))
        with mock.patch.object(kb, "get_bodies",
                               side_effect=RuntimeError("frame glitch")):
            pres = kb.get_presence()
        self.assertEqual(pres["present"], False)
        self.assertEqual(pres["count"], 0)

    def test_facing_true_for_upright_squared_body(self):
        # head above spine, shoulders equidistant in z → facing.
        body = _FakeBody(True, {
            "head": (0, 0.6, 2.0), "spine_shoulder": (0, 0.0, 2.0),
            "shoulder_left": (-0.2, 0.4, 2.0), "shoulder_right": (0.2, 0.4, 2.0),
        })
        _patch_loader(self, _FakeRuntime(bodies=[body]))
        self.assertTrue(kb.get_presence()["facing"])

    def test_facing_false_for_side_on_body(self):
        # big z-gap between shoulders → side-on, not facing.
        body = _FakeBody(True, {
            "head": (0, 0.6, 2.0), "spine_shoulder": (0, 0.0, 2.0),
            "shoulder_left": (-0.1, 0.4, 1.6), "shoulder_right": (0.1, 0.4, 2.4),
        })
        _patch_loader(self, _FakeRuntime(bodies=[body]))
        self.assertFalse(kb.get_presence()["facing"])


# ─────────────────────────────────────────────────────────────────────────
# hand states (the air-mouse keystone accessor)
# ─────────────────────────────────────────────────────────────────────────
class HandStateTests(_BridgeBase):
    def test_state_name_maps_enum(self):
        # 0/1 → unknown, 2 → open, 3 → closed, 4 → lasso; junk → unknown.
        self.assertEqual(kb._hand_state_name(0), "unknown")
        self.assertEqual(kb._hand_state_name(1), "unknown")  # NotTracked
        self.assertEqual(kb._hand_state_name(2), "open")
        self.assertEqual(kb._hand_state_name(3), "closed")
        self.assertEqual(kb._hand_state_name(4), "lasso")
        self.assertEqual(kb._hand_state_name(None), "unknown")
        self.assertEqual(kb._hand_state_name("x"), "unknown")

    def test_get_bodies_carries_hand_states(self):
        body = _FakeBody(True, {"head": (0, 0.6, 1.8)},
                         hand_right_state=3, hand_left_state=2)
        _patch_loader(self, _FakeRuntime(bodies=[body]))
        got = kb.get_bodies()
        self.assertEqual(got[0]["hand_right"], "closed")
        self.assertEqual(got[0]["hand_left"], "open")

    def test_get_bodies_hand_state_defaults_unknown_when_absent(self):
        # A body without the attrs (older build) degrades to "unknown".
        body = _FakeBody(True, {"head": (0, 0.6, 1.8)})
        _patch_loader(self, _FakeRuntime(bodies=[body]))
        got = kb.get_bodies()
        self.assertEqual(got[0]["hand_right"], "unknown")
        self.assertEqual(got[0]["hand_left"], "unknown")

    def test_get_hand_states_nearest_body(self):
        # Two bodies at different depths; the nearer (1.5 m) wins.
        near = _FakeBody(True, {"head": (0, 0.6, 1.5),
                                "spine_shoulder": (0, 0.0, 1.5)},
                         hand_right_state=2, hand_left_state=3)
        far = _FakeBody(True, {"head": (0, 0.6, 3.0),
                               "spine_shoulder": (0, 0.0, 3.0)},
                        hand_right_state=3, hand_left_state=2)
        _patch_loader(self, _FakeRuntime(bodies=[far, near]))
        states = kb.get_hand_states()
        self.assertTrue(states["tracked"])
        self.assertEqual(states["right"], "open")    # the NEAR body's right
        self.assertEqual(states["left"], "closed")
        self.assertIn("ts", states)

    def test_get_hand_states_unknown_when_no_sensor(self):
        kb.set_enabled(False)
        states = kb.get_hand_states()
        self.assertFalse(states["tracked"])
        self.assertEqual(states["right"], "unknown")
        self.assertEqual(states["left"], "unknown")

    def test_get_hand_states_unknown_when_no_body(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(False)] * 6))
        states = kb.get_hand_states()
        self.assertFalse(states["tracked"])
        self.assertEqual(states["right"], "unknown")

    def test_get_hand_states_swallows_get_bodies_error(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(True)]))
        with mock.patch.object(kb, "get_bodies",
                               side_effect=RuntimeError("frame glitch")):
            states = kb.get_hand_states()
        self.assertFalse(states["tracked"])
        self.assertEqual(states["right"], "unknown")


# ─────────────────────────────────────────────────────────────────────────
# head-facing yaw (joint-derived gaze; the Kinect v2 Face API is absent on
# this pykinect2 build, so facing is recovered from the shoulder line)
# ─────────────────────────────────────────────────────────────────────────
class HeadYawTests(_BridgeBase):
    # Raw joints handed straight to _body_facing_yaw are (x, y, z, state) — the
    # same shape get_bodies() produces — so the tracked-state gate (>=1) passes.
    def _square(self):
        return {"shoulder_left": (-0.2, 0.4, 2.0, 2), "shoulder_right": (0.2, 0.4, 2.0, 2),
                "head": (0.0, 0.6, 2.0, 2)}

    def _turn_sensor_right(self):
        # Looking at a monitor on the sensor's RIGHT: LEFT shoulder forward
        # (smaller z), RIGHT shoulder back (larger z), head shifted +x.
        return {"shoulder_left": (-0.18, 0.4, 1.85, 2),
                "shoulder_right": (0.18, 0.4, 2.15, 2), "head": (0.12, 0.6, 2.0, 2)}

    def _turn_sensor_left(self):
        return {"shoulder_left": (-0.18, 0.4, 2.15, 2),
                "shoulder_right": (0.18, 0.4, 1.85, 2), "head": (-0.12, 0.6, 2.0, 2)}

    def test_facing_yaw_zero_when_square(self):
        yaw = kb._body_facing_yaw(self._square())
        self.assertIsNotNone(yaw)
        self.assertLess(abs(yaw), 3.0)   # ~0°

    def test_facing_yaw_positive_when_turned_sensor_right(self):
        self.assertGreater(kb._body_facing_yaw(self._turn_sensor_right()), 8.0)

    def test_facing_yaw_negative_when_turned_sensor_left(self):
        self.assertLess(kb._body_facing_yaw(self._turn_sensor_left()), -8.0)

    def test_facing_yaw_none_without_shoulders(self):
        # No shoulders and no head → can't estimate.
        self.assertIsNone(kb._body_facing_yaw({"spine_mid": (0, 0, 2.0, 2)}))

    def test_facing_yaw_ignores_untracked_shoulders(self):
        # Shoulders present but NotTracked (state 0) → no shoulder yaw; with no
        # other usable joint the estimate is None.
        joints = {"shoulder_left": (-0.2, 0.4, 1.8, 0),
                  "shoulder_right": (0.2, 0.4, 2.2, 0)}
        self.assertIsNone(kb._body_facing_yaw(joints))

    def test_get_presence_includes_nearest_head_yaw(self):
        # Two bodies; the NEAREST (1.5 m, turned right) supplies head_yaw_deg.
        near = _FakeBody(True, {
            "head": (0.12, 0.6, 1.5), "spine_shoulder": (0, 0.0, 1.5),
            "shoulder_left": (-0.18, 0.4, 1.42), "shoulder_right": (0.18, 0.4, 1.58)})
        far = _FakeBody(True, {
            "head": (0, 0.6, 3.0), "spine_shoulder": (0, 0.0, 3.0),
            "shoulder_left": (-0.2, 0.4, 3.0), "shoulder_right": (0.2, 0.4, 3.0)})
        _patch_loader(self, _FakeRuntime(bodies=[far, near, _FakeBody(False),
                                                 _FakeBody(False), _FakeBody(False),
                                                 _FakeBody(False)]))
        pres = kb.get_presence()
        self.assertEqual(pres["nearest_m"], 1.5)
        self.assertIsNotNone(pres["head_yaw_deg"])
        self.assertGreater(pres["head_yaw_deg"], 0.0)   # nearest is turned right

    def test_get_head_yaw_returns_nearest(self):
        body = _FakeBody(True, {
            "head": (0.12, 0.6, 1.5), "spine_shoulder": (0, 0.0, 1.5),
            "shoulder_left": (-0.18, 0.4, 1.42), "shoulder_right": (0.18, 0.4, 1.58)})
        _patch_loader(self, _FakeRuntime(bodies=[body, _FakeBody(False),
                                                 _FakeBody(False), _FakeBody(False),
                                                 _FakeBody(False), _FakeBody(False)]))
        yaw = kb.get_head_yaw()
        self.assertIsNotNone(yaw)
        self.assertGreater(yaw, 0.0)

    def test_get_head_yaw_none_when_disabled(self):
        kb.set_enabled(False)
        self.assertIsNone(kb.get_head_yaw())

    def test_get_head_yaw_none_when_no_body(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(False)] * 6))
        self.assertIsNone(kb.get_head_yaw())

    def test_get_presence_head_yaw_none_when_no_body(self):
        _patch_loader(self, _FakeRuntime(bodies=[_FakeBody(False)] * 6))
        self.assertIsNone(kb.get_presence()["head_yaw_deg"])


# ─────────────────────────────────────────────────────────────────────────
# lifecycle
# ─────────────────────────────────────────────────────────────────────────
class CloseTests(_BridgeBase):
    def test_close_idempotent_with_no_runtime(self):
        kb._runtime[0] = None
        kb.close()
        kb.close()   # must not raise

    def test_close_releases_and_clears_singleton(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        kb.get_runtime()
        kb.close()
        self.assertTrue(rt.closed)
        self.assertIsNone(kb._runtime[0])
        kb.close()   # second call is a no-op


# ─────────────────────────────────────────────────────────────────────────
# PART B: stale-runtime guard (body stream can't go quiet once warmed)
# ─────────────────────────────────────────────────────────────────────────
class BodyStaleGuardTests(_BridgeBase):
    def setUp(self):
        super().setUp()
        # Isolate the staleness clock between tests.
        self._orig_last = kb._last_body_frame_at[0]
        self.addCleanup(lambda: kb._last_body_frame_at.__setitem__(0, self._orig_last))

    def test_not_stale_when_no_runtime_open(self):
        kb._runtime[0] = None
        kb._last_body_frame_at[0] = 0.0
        # Nothing to reset when the runtime isn't even open.
        self.assertFalse(kb._body_frame_is_stale(now=10_000.0))

    def test_not_stale_before_window_elapses(self):
        kb._runtime[0] = object()
        kb._last_body_frame_at[0] = 100.0
        # Within BODY_STALE_RESET_SEC → not stale.
        self.assertFalse(
            kb._body_frame_is_stale(now=100.0 + kb.BODY_STALE_RESET_SEC - 0.5))

    def test_stale_after_window_elapses(self):
        kb._runtime[0] = object()
        kb._last_body_frame_at[0] = 100.0
        self.assertTrue(
            kb._body_frame_is_stale(now=100.0 + kb.BODY_STALE_RESET_SEC + 0.5))

    def test_not_stale_when_clock_unseeded(self):
        # A runtime open but no frame yet observed (clock 0) must NOT trip — the
        # reopen window is seeded by the open path, not judged against epoch 0.
        kb._runtime[0] = object()
        kb._last_body_frame_at[0] = 0.0
        self.assertFalse(kb._body_frame_is_stale(now=10_000.0))

    def test_reset_if_body_stale_clears_runtime_and_closes_it(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        self.assertIsNotNone(kb.get_runtime()[0])
        # Force the clock far into the past so the stream looks stale.
        kb._last_body_frame_at[0] = 1.0
        did = kb.reset_if_body_stale(now=1.0 + kb.BODY_STALE_RESET_SEC + 1.0)
        self.assertTrue(did)
        self.assertIsNone(kb._runtime[0])   # next get_runtime() will reopen
        self.assertTrue(rt.closed)          # old handle released

    def test_reset_noop_when_not_stale(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._last_body_frame_at[0] = 1000.0
        did = kb.reset_if_body_stale(now=1000.5)   # fresh → no reset
        self.assertFalse(did)
        self.assertIsNotNone(kb._runtime[0])
        self.assertFalse(rt.closed)

    def test_reset_reopens_a_live_stream_on_next_get_runtime(self):
        # End-to-end: a stale reset followed by get_runtime() must reopen (the
        # whole point — the next open re-binds a live stream).
        rt1 = _FakeRuntime()
        _patch_loader(self, rt1)
        first, _ = kb.get_runtime()
        kb._last_body_frame_at[0] = 1.0
        kb.reset_if_body_stale(now=1.0 + kb.BODY_STALE_RESET_SEC + 1.0)
        self.assertIsNone(kb._runtime[0])
        second, _ = kb.get_runtime()   # reopens via the patched loader
        self.assertIsNotNone(second)

    def test_get_bodies_stamps_freshness_on_new_frame(self):
        # A real consumer read (get_bodies) must advance the staleness clock so a
        # healthy, actively-read stream never trips the reset.
        body = _FakeBody(True, {"head": (0, 0.6, 1.8)})
        _patch_loader(self, _FakeRuntime(bodies=[body]))
        kb._last_body_frame_at[0] = 0.0
        kb.get_bodies()
        self.assertGreater(kb._last_body_frame_at[0], 0.0)

    def test_note_body_frame_seen_uses_injected_now(self):
        kb.note_body_frame_seen(now=12345.0)
        self.assertEqual(kb._last_body_frame_at[0], 12345.0)


# ─────────────────────────────────────────────────────────────────────────
# PART B: always-on body-frame pump (keeps the body pipe flowing)
# ─────────────────────────────────────────────────────────────────────────
class BodyPumpTests(_BridgeBase):
    def setUp(self):
        super().setUp()
        self._orig_last = kb._last_body_frame_at[0]
        self.addCleanup(lambda: kb._last_body_frame_at.__setitem__(0, self._orig_last))
        self.addCleanup(kb.stop_body_pump)

    def test_pump_tick_noop_when_disabled(self):
        kb._ENABLED = False
        kb._runtime[0] = object()
        kb._last_body_frame_at[0] = 0.0
        kb._pump_tick()   # disabled → does nothing, must not raise
        self.assertEqual(kb._last_body_frame_at[0], 0.0)

    def test_pump_tick_no_stamp_when_runtime_cant_open(self):
        # The pump now OWNS the open (so the cache warms with no consumer), but
        # when the sensor can't be opened it must stamp nothing and not raise.
        # Patch the loader to fail so this holds regardless of whether pykinect2
        # happens to be installed on the host.
        kb._runtime[0] = None
        kb._last_body_frame_at[0] = 0.0
        with mock.patch.object(kb, "import_pykinect2",
                               side_effect=ImportError("pykinect2")):
            kb._pump_tick()   # tries to open, can't → nothing to pump
        self.assertEqual(kb._last_body_frame_at[0], 0.0)
        self.assertIsNone(kb._runtime[0])

    def test_pump_tick_drains_and_stamps_on_new_frame(self):
        rt = _FakeRuntime(bodies=[_FakeBody(True, {"head": (0, 0.6, 1.8)})])
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._last_body_frame_at[0] = 0.0
        self.assertTrue(rt.has_new_body_frame())
        kb._pump_tick()
        # Pump consumed the frame (readiness cleared) + stamped freshness.
        self.assertFalse(rt.has_new_body_frame())
        self.assertGreater(kb._last_body_frame_at[0], 0.0)

    def test_pump_tick_resets_when_stale(self):
        rt = _FakeRuntime()   # no body frame ready → never re-stamps freshness
        _patch_loader(self, rt)
        kb.get_runtime()
        # Seed the clock in the deep past so the stale reset fires this tick.
        kb._last_body_frame_at[0] = time_monotonic_minus(kb.BODY_STALE_RESET_SEC + 5.0)
        kb._pump_tick()
        self.assertIsNone(kb._runtime[0])   # stale → runtime dropped for reopen

    def test_start_body_pump_singleton_no_double(self):
        kb._ENABLED = True
        # Don't actually run the loop thread — capture starts.
        started = []

        class _FakeThread:
            def __init__(self, *a, **k):
                started.append(k.get("name"))
            def start(self):
                pass
            def is_alive(self):
                return True   # pretend it's running so the 2nd call no-ops

        with mock.patch.object(kb.threading, "Thread", _FakeThread):
            first = kb.start_body_pump()
            second = kb.start_body_pump()
        self.assertTrue(first)         # started a new pump
        self.assertFalse(second)       # singleton: no second thread
        self.assertEqual(started, ["kinect-body-pump"])

    def test_start_body_pump_noop_when_disabled(self):
        kb._ENABLED = False
        self.assertFalse(kb.start_body_pump())

    def test_set_enabled_true_starts_pump(self):
        _patch_loader(self, _FakeRuntime())
        started = {"n": 0}
        with mock.patch.object(kb, "start_body_pump",
                               side_effect=lambda: started.__setitem__("n", started["n"] + 1)):
            kb.set_enabled(True)
        self.assertEqual(started["n"], 1)

    def test_set_enabled_false_stops_pump(self):
        stopped = {"n": 0}
        with mock.patch.object(kb, "stop_body_pump",
                               side_effect=lambda: stopped.__setitem__("n", stopped["n"] + 1)):
            kb.set_enabled(False)
        self.assertGreaterEqual(stopped["n"], 1)


# ─────────────────────────────────────────────────────────────────────────
# PART C: shared body-frame cache (the gesture/air-mouse starvation fix)
# ─────────────────────────────────────────────────────────────────────────
# THE BUG: the Kinect body frame is single-consumer — get_last_body_frame()
# clears has_new_body_frame(), so the FIRST reader each frame consumes it and
# every other reader that tick got []. With the skeleton renderer + pump +
# gesture poller (18 Hz) + air-mouse poller (30 Hz) all calling get_bodies(),
# the pollers starved → gestures/air-mouse never fired. THE FIX: the pump is the
# sole frame reader; it parses once into a module cache and every consumer reads
# the cache. These tests prove the contract WITHOUT a real sensor.
class SharedBodyCacheTests(_BridgeBase):
    def setUp(self):
        super().setUp()
        # Isolate the staleness clock too (some assertions touch it).
        self._orig_last = kb._last_body_frame_at[0]
        self.addCleanup(lambda: kb._last_body_frame_at.__setitem__(0, self._orig_last))

    def _one_body_runtime(self):
        return _FakeRuntime(bodies=[_FakeBody(
            True, {"head": (0, 0.6, 1.8), "spine_shoulder": (0, 0.0, 1.8)})])

    def test_many_get_bodies_between_frames_return_same_cache_never_empty(self):
        # THE REGRESSION: the sensor has ONE new frame; the pump reads+caches it,
        # then many get_bodies() calls arrive before the next sensor frame. Every
        # one must return the SAME tracked body from the cache — NEVER a spurious
        # [] (which is exactly what starved the gesture + air-mouse pollers).
        rt = self._one_body_runtime()
        _patch_loader(self, rt)
        kb.get_runtime()
        # The pump performs the SOLE read, consuming the single new-frame flag.
        kb._pump_tick()
        self.assertFalse(rt.has_new_body_frame())   # flag consumed by the pump
        # Now 50 consumer reads with NO new sensor frame in between.
        results = [kb.get_bodies() for _ in range(50)]
        self.assertTrue(all(len(r) == 1 for r in results),
                        "a consumer read returned [] between sensor frames — "
                        "the starvation bug")
        for r in results:
            self.assertEqual(r[0]["head"], (0.0, 0.6, 1.8))

    def test_mixed_consumers_share_one_frame(self):
        # get_bodies / get_hand_states / get_presence / get_head_yaw all read the
        # SAME shared cache, so the gesture poller, air-mouse, and overlay reading
        # in the same frame ALL see the body — no consumer loses the race.
        near = _FakeBody(True,
                         {"head": (0.12, 0.6, 1.5), "spine_shoulder": (0, 0.0, 1.5),
                          "shoulder_left": (-0.18, 0.4, 1.42),
                          "shoulder_right": (0.18, 0.4, 1.58)},
                         hand_right_state=3, hand_left_state=2)
        rt = _FakeRuntime(bodies=[near])
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._pump_tick()                       # sole read → cache populated
        self.assertFalse(rt.has_new_body_frame())
        # Four different consumers, one shared frame, all see the body.
        self.assertEqual(len(kb.get_bodies()), 1)
        self.assertEqual(kb.get_hand_states()["right"], "closed")
        self.assertTrue(kb.get_presence()["present"])
        self.assertIsNotNone(kb.get_head_yaw())

    def test_cache_refreshes_on_new_frame(self):
        # When a NEW sensor frame arrives the pump re-reads and the cache reflects
        # the new bodies — a 1-body frame, then re-armed to 2 bodies.
        rt = _FakeRuntime(bodies=[_FakeBody(
            True, {"head": (0, 0.6, 1.8), "spine_shoulder": (0, 0.0, 1.8)})])
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._pump_tick()
        self.assertEqual(len(kb.get_bodies()), 1)
        # Re-arm the runtime with a 2-body frame (a new sensor frame).
        rt._bodies = [
            _FakeBody(True, {"head": (0, 0.6, 1.8), "spine_shoulder": (0, 0.0, 1.8)}),
            _FakeBody(True, {"head": (0, 0.6, 2.5), "spine_shoulder": (0, 0.0, 2.5)}),
        ]
        rt._new["body"] = True
        kb._pump_tick()                       # sole reader picks up the new frame
        self.assertEqual(len(kb.get_bodies()), 2)

    def test_stale_cache_returns_empty(self):
        # A cache entry older than BODY_CACHE_FRESH_SEC must NOT be served: with no
        # new sensor frame to fall back on, get_bodies() returns []. (In prod the
        # 30 Hz pump keeps it fresh; this guards the staleness boundary.)
        rt = _FakeRuntime()   # no new body frame pending
        _patch_loader(self, rt)
        kb.get_runtime()
        # Seed the cache with a body but stamp it well in the past.
        kb._body_cache[0] = [{"id": 1, "joints": {}, "head": None}]
        kb._body_cache_at[0] = kb.time.monotonic() - (kb.BODY_CACHE_FRESH_SEC + 1.0)
        self.assertEqual(kb.get_bodies(), [])   # stale → not served, no new frame

    def test_get_cached_bodies_fresh_vs_stale_boundary(self):
        # _get_cached_bodies serves within the window, drops past it. Use injected
        # `now` so the boundary is exact and host-clock-independent.
        kb._body_cache[0] = [{"id": 7}]
        kb._body_cache_at[0] = 100.0
        fresh = kb._get_cached_bodies(now=100.0 + kb.BODY_CACHE_FRESH_SEC - 0.01)
        self.assertIsNotNone(fresh)
        self.assertEqual(fresh[0]["id"], 7)
        stale = kb._get_cached_bodies(now=100.0 + kb.BODY_CACHE_FRESH_SEC + 0.01)
        self.assertIsNone(stale)

    def test_get_cached_bodies_none_when_cold(self):
        kb._body_cache[0] = None      # never populated
        kb._body_cache_at[0] = 1.0
        self.assertIsNone(kb._get_cached_bodies(now=1.0))

    def test_get_cached_bodies_returns_copy_not_shared_ref(self):
        # Consumers get a shallow copy so they can't mutate the shared cache list.
        kb._body_cache[0] = [{"id": 1}]
        kb._body_cache_at[0] = 50.0
        got = kb._get_cached_bodies(now=50.0)
        self.assertIsNotNone(got)
        got.append({"id": 999})
        # The shared cache list is unchanged by the consumer's mutation.
        self.assertEqual(len(kb._body_cache[0]), 1)

    def test_empty_bodies_cached_as_list_not_none(self):
        # A frame with NO tracked bodies caches [] (a real "no one in view"),
        # distinct from None (cold). A consumer reads [] from the cache, and the
        # cell is a list so it is served (fresh) rather than re-reading.
        rt = _FakeRuntime(bodies=[_FakeBody(False)] * 6)
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._pump_tick()
        self.assertEqual(kb._body_cache[0], [])         # [] cached, not None
        self.assertIsNotNone(kb._body_cache[0])
        self.assertEqual(kb.get_bodies(), [])

    def test_read_and_cache_serves_fresh_cache_when_no_new_frame(self):
        # _read_and_cache_bodies called when NO new frame is pending returns the
        # still-fresh cache (so a direct caller racing the pump in the same frame
        # gets the body, not []).
        kb._runtime[0] = _FakeRuntime()         # has_new_body_frame() False
        kb._body_cache[0] = [{"id": 3, "joints": {}}]
        kb._body_cache_at[0] = kb.time.monotonic()   # fresh
        got = kb._read_and_cache_bodies()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], 3)

    def test_read_and_cache_empty_when_no_frame_and_cold_cache(self):
        kb._runtime[0] = _FakeRuntime()         # no new frame
        kb._body_cache[0] = None                # cold
        self.assertEqual(kb._read_and_cache_bodies(), [])

    def test_read_and_cache_empty_when_runtime_down(self):
        kb.set_enabled(False)
        self.assertEqual(kb._read_and_cache_bodies(), [])

    def test_pump_tick_populates_cache_with_no_consumer(self):
        # The pump must warm the cache on its own (no consumer call needed) so the
        # first gesture/air-mouse read after enable already has data.
        rt = self._one_body_runtime()
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._body_cache[0] = None
        kb._pump_tick()                         # only the pump ran
        self.assertIsNotNone(kb._body_cache[0])
        self.assertEqual(len(kb._body_cache[0]), 1)

    def test_stop_body_pump_clears_cache(self):
        kb._body_cache[0] = [{"id": 1}]
        kb._body_cache_at[0] = kb.time.monotonic()
        kb.stop_body_pump()
        self.assertIsNone(kb._body_cache[0])    # dropped so a closed sensor → []

    def test_parse_body_frame_pure_parses_tracked(self):
        # The extracted pure parser turns a raw frame into the public shape with
        # no sensor/runtime contact (the pump feeds it the sole frame).
        frame = _FakeBodyFrame([
            _FakeBody(False),
            _FakeBody(True, {"head": (0, 0.6, 2.0), "spine_shoulder": (0, 0.0, 2.0)}),
        ])
        out = kb._parse_body_frame(frame)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["head"], (0.0, 0.6, 2.0))

    def test_parse_body_frame_empty_on_none(self):
        self.assertEqual(kb._parse_body_frame(None), [])

    def test_get_bodies_direct_read_when_no_pump(self):
        # A lone consumer with the cache cold (no pump warmed it yet) still works:
        # get_bodies() falls back to a one-shot direct read of the armed frame.
        rt = self._one_body_runtime()
        _patch_loader(self, rt)
        kb.get_runtime()
        kb._body_cache[0] = None                # cold — no pump has run
        got = kb.get_bodies()                   # must do the fallback direct read
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["head"], (0.0, 0.6, 1.8))


def time_monotonic_minus(seconds: float) -> float:
    """A monotonic timestamp `seconds` in the past (for seeding the stale clock
    in pump tests that call the real time.monotonic inside _pump_tick)."""
    import time as _t
    return _t.monotonic() - seconds


# ─────────────────────────────────────────────────────────────────────────
# PART A: color-space mapper accessor (the skeleton-overlay projection seam)
# ─────────────────────────────────────────────────────────────────────────
class ColorSpaceMapperTests(_BridgeBase):
    def test_mapper_none_when_runtime_absent(self):
        kb.set_enabled(False)
        self.assertIsNone(kb.get_color_space_mapper())

    def test_mapper_none_when_runtime_lacks_mapper(self):
        # A runtime without a ._mapper (an older/odd build) → None, not a crash.
        rt = _FakeRuntime()   # no _mapper attribute
        _patch_loader(self, rt)
        self.assertIsNone(kb.get_color_space_mapper())

    def test_mapper_projects_via_runtime_mapper(self):
        # Wire a fake runtime ._mapper + a PyKinectV2 carrying a CameraSpacePoint
        # so get_color_space_mapper returns a working (x,y,z)->(px,py) callable.
        captured = {}

        class _ColorPt:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class _FakeMapper:
            def MapCameraPointToColorSpace(self, pt):
                # Echo the camera point's x/y scaled, proving the point was built.
                captured["pt"] = (pt.x, pt.y, pt.z)
                return _ColorPt(pt.x * 2.0, pt.y * 3.0)

        class _CamPt:
            x = 0.0
            y = 0.0
            z = 0.0

        rt = _FakeRuntime()
        rt._mapper = _FakeMapper()
        pk2 = _fake_pk2_module()
        pk2.CameraSpacePoint = _CamPt
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")
        rt_mod.PyKinectRuntime = lambda flags: rt
        with mock.patch.object(kb, "import_pykinect2", lambda: (pk2, rt_mod)):
            kb.get_runtime()
            mapper = kb.get_color_space_mapper()
            self.assertIsNotNone(mapper)
            out = mapper(5.0, 7.0, 2.0)
        self.assertEqual(out, (10.0, 21.0))         # x*2, y*3
        self.assertEqual(captured["pt"], (5.0, 7.0, 2.0))

    def test_mapper_callable_swallows_per_point_error(self):
        class _FakeMapper:
            def MapCameraPointToColorSpace(self, pt):
                raise RuntimeError("COM glitch")

        class _CamPt:
            x = 0.0
            y = 0.0
            z = 0.0

        rt = _FakeRuntime()
        rt._mapper = _FakeMapper()
        pk2 = _fake_pk2_module()
        pk2.CameraSpacePoint = _CamPt
        rt_mod = types.ModuleType("pykinect2.PyKinectRuntime")
        rt_mod.PyKinectRuntime = lambda flags: rt
        with mock.patch.object(kb, "import_pykinect2", lambda: (pk2, rt_mod)):
            kb.get_runtime()
            mapper = kb.get_color_space_mapper()
            # A throwing COM call degrades to None for that point, never raises.
            self.assertIsNone(mapper(1.0, 2.0, 3.0))


# ─────────────────────────────────────────────────────────────────────────
# KinectCapture drop-in shim
# ─────────────────────────────────────────────────────────────────────────
class KinectCaptureTests(_BridgeBase):
    def test_isopened_true_when_sensor_opens(self):
        _patch_loader(self, _FakeRuntime())
        cap = kb.KinectCapture()
        self.assertTrue(cap.isOpened())

    def test_isopened_false_when_disabled(self):
        kb.set_enabled(False)
        cap = kb.KinectCapture()
        self.assertFalse(cap.isOpened())

    def test_read_returns_ret_bgr(self):
        np = _require_numpy(self)
        flat = np.zeros(1920 * 1080 * 4, dtype=np.uint8)
        _patch_loader(self, _FakeRuntime(color=flat))
        cap = kb.KinectCapture()
        ret, frame = cap.read()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (1080, 1920, 3))

    def test_read_uses_last_frame_even_when_not_new(self):
        # require_new=False path: a second read still yields the frame.
        np = _require_numpy(self)
        flat = np.zeros(1920 * 1080 * 4, dtype=np.uint8)
        _patch_loader(self, _FakeRuntime(color=flat))
        cap = kb.KinectCapture()
        cap.read()                  # consumes the "new" flag
        ret, frame = cap.read()     # still returns the last frame
        self.assertTrue(ret)
        self.assertIsNotNone(frame)

    def test_read_false_none_when_no_frame(self):
        _patch_loader(self, _FakeRuntime(color=None))
        cap = kb.KinectCapture()
        ret, frame = cap.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)

    def test_get_reports_kinect_geometry(self):
        _patch_loader(self, _FakeRuntime())
        cap = kb.KinectCapture()
        self.assertEqual(cap.get(3), 1920.0)   # CAP_PROP_FRAME_WIDTH
        self.assertEqual(cap.get(4), 1080.0)   # CAP_PROP_FRAME_HEIGHT

    def test_set_returns_false_without_raising(self):
        _patch_loader(self, _FakeRuntime())
        cap = kb.KinectCapture()
        self.assertFalse(cap.set(3, 1280))

    def test_release_does_not_close_shared_runtime(self):
        rt = _FakeRuntime()
        _patch_loader(self, rt)
        cap = kb.KinectCapture()
        cap.release()
        # The shared singleton must stay alive for other consumers.
        self.assertFalse(rt.closed)
        self.assertIsNotNone(kb._runtime[0])
        self.assertFalse(cap.isOpened())


# ─────────────────────────────────────────────────────────────────────────
# numpy helper — skip a reshape test if numpy genuinely isn't importable
# (it's pre-imported on CI, so this practically never skips)
# ─────────────────────────────────────────────────────────────────────────
def _require_numpy(test):
    try:
        import numpy as np
        return np
    except Exception:   # pragma: no cover - numpy is present on dev + CI
        test.skipTest("numpy not importable")


if __name__ == "__main__":
    unittest.main()
