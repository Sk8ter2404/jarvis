"""The opt-in Kinect ghost-body filter (audio/kinect_bridge.filter_ghost_bodies,
KINECT_GHOST_FILTER_ENABLED, default OFF; backlog "Kinect sees ghost
skeletons", 2026-10-28).

Every behaviour test drives the LIVE entry point: a fake runtime hands the body
pump (_pump_tick -> _read_and_cache_bodies) a frame, and the test reads what
every consumer reads - get_bodies() / get_presence(). Joint data and depth
rasters are synthetic: a plausible standing adult about 2 m from the sensor,
and the same skeleton bent into shapes no person can make.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from audio import kinect_bridge as kb
from core import config as cfg
from tests.test_kinect_bridge import _BridgeBase, _FakeBody, _FakeRuntime, _patch_loader

try:
    import numpy as np
except Exception:   # pragma: no cover - numpy is present on dev + CI
    np = None


# A plausible standing adult, metres, camera space (x right, y up, z away).
_PERSON = {
    "head": (0.0, 0.65, 2.00), "neck": (0.0, 0.50, 2.02),
    "spine_shoulder": (0.0, 0.42, 2.03), "spine_mid": (0.0, 0.15, 2.05),
    "spine_base": (0.0, -0.12, 2.05),
    "shoulder_left": (-0.18, 0.38, 2.05), "elbow_left": (-0.22, 0.10, 2.05),
    "wrist_left": (-0.24, -0.14, 2.00), "hand_left": (-0.24, -0.20, 1.98),
    "hand_tip_left": (-0.24, -0.26, 1.98), "thumb_left": (-0.21, -0.20, 1.97),
    "shoulder_right": (0.18, 0.38, 2.05), "elbow_right": (0.22, 0.10, 2.05),
    "wrist_right": (0.24, -0.14, 2.00), "hand_right": (0.24, -0.20, 1.98),
    "hand_tip_right": (0.24, -0.26, 1.98), "thumb_right": (0.21, -0.20, 1.97),
    "hip_left": (-0.09, -0.15, 2.05), "knee_left": (-0.10, -0.58, 2.00),
    "ankle_left": (-0.10, -0.98, 2.08), "foot_left": (-0.10, -1.03, 1.98),
    "hip_right": (0.09, -0.15, 2.05), "knee_right": (0.10, -0.58, 2.00),
    "ankle_right": (0.10, -0.98, 2.08), "foot_right": (0.10, -1.03, 1.98),
}


def _person(dx=0.0, **over):
    j = {k: (x + dx, y, z) for k, (x, y, z) in _PERSON.items()}
    j.update(over)
    return j


def _stretched():
    """Limbs a metre long: a skeleton fitted onto furniture / noise."""
    return _person(elbow_left=(-0.9, 0.10, 2.05), wrist_left=(-1.8, 0.0, 2.0),
                   elbow_right=(0.9, 0.10, 2.05), wrist_right=(1.8, 0.0, 2.0),
                   knee_left=(-0.10, -1.40, 2.0), ankle_left=(-0.10, -2.4, 2.0))


def _flat():
    """Every joint at exactly one depth: a body painted on a surface."""
    return {k: (x, y, 2.0) for k, (x, y, _z) in _PERSON.items()}


def _mapper(x, y, z):
    return (256.0 + x * 150.0, 212.0 - y * 150.0)


def _raster(background_mm, body_mm=None):
    """A 512x424 depth frame: ``background_mm`` everywhere, and (optionally)
    the person's silhouette painted at ``body_mm`` over the joints' pixel box
    (inside the ring depth_corroborate samples)."""
    if np is not None:
        r = np.full((kb.DEPTH_H, kb.DEPTH_W), background_mm, dtype=np.uint16)
        if body_mm is not None:
            r[110:372, 214:299] = body_mm
        return r
    r = [[background_mm] * kb.DEPTH_W for _ in range(kb.DEPTH_H)]
    if body_mm is not None:
        for y in range(110, 372):
            for x in range(214, 299):
                r[y][x] = body_mm
    return r


class _GhostBase(_BridgeBase):
    def setUp(self):
        super().setUp()
        reset = getattr(kb, "reset_ghost_filter", None)
        if callable(reset):
            reset()
            self.addCleanup(reset)

    def _on(self, recheck=1.0):
        for name, val in (("KINECT_GHOST_FILTER_ENABLED", True),
                          ("KINECT_GHOST_DEPTH_RECHECK_SEC", recheck)):
            p = mock.patch.object(cfg, name, val, create=True)
            p.start()
            self.addCleanup(p.stop)

    def _depth(self, raster):
        """Serve ``raster`` (or None) as the depth frame, with the test mapper;
        returns the call counter."""
        calls = {"n": 0}

        def _get_depth(*_a, **_k):
            calls["n"] += 1
            return raster
        for name, fn in (("get_depth", _get_depth),
                         ("get_depth_space_mapper", lambda: _mapper)):
            p = mock.patch.object(kb, name, fn)
            p.start()
            self.addCleanup(p.stop)
        return calls

    def _pump(self, *joint_sets):
        """One body frame through the live pump; returns get_bodies()."""
        bodies = [_FakeBody(True, joints=j, tracking_id=100 + i)
                  for i, j in enumerate(joint_sets)]
        rt = _FakeRuntime(bodies=bodies)
        _patch_loader(self, rt)
        kb.get_runtime()
        with redirect_stdout(io.StringIO()) as out:
            kb._pump_tick()
        self.out = out.getvalue()
        return kb.get_bodies()


class GhostFilterOffTests(_GhostBase):
    def test_setting_ships_off(self):
        self.assertIs(cfg.KINECT_GHOST_FILTER_ENABLED, False)
        self.assertEqual(cfg.KINECT_GHOST_DEPTH_RECHECK_SEC, 1.0)

    def test_off_publishes_every_body_as_before(self):
        # Flag off: a body with impossible limbs still reaches the consumers,
        # and no depth frame is read.
        calls = self._depth(_raster(1000, 2000))
        got = self._pump(_stretched(), _flat())
        self.assertEqual(len(got), 2)
        self.assertEqual(kb.get_presence()["count"], 2)
        self.assertEqual(calls["n"], 0)


class GhostFilterJointTests(_GhostBase):
    def test_keeps_a_plausible_person(self):
        self._on()
        self._depth(None)                       # no depth frame: joints only
        got = self._pump(_person())
        self.assertEqual(len(got), 1)
        self.assertEqual(self.out, "")

    def test_drops_impossible_limb_lengths(self):
        self._on()
        self._depth(None)
        self.assertEqual(self._pump(_stretched()), [])
        self.assertFalse(kb.get_presence()["present"])
        self.assertIn("impossible limb lengths", self.out)

    def test_drops_a_flat_body(self):
        self._on()
        self._depth(None)
        self.assertEqual(self._pump(_flat()), [])
        self.assertIn("flat", self.out)

    def test_ghost_beside_a_person_leaves_presence_count_one(self):
        self._on()
        self._depth(None)
        got = self._pump(_person(), _stretched())
        self.assertEqual([b["id"] for b in got], [100])
        self.assertEqual(kb.get_presence()["count"], 1)

    def test_one_glitched_bone_does_not_drop_a_real_person(self):
        # A single wrist jump makes one bad bone out of 16: still a person.
        rep = kb.body_plausibility(
            {"joints": {k: (x, y, z, 2) for k, (x, y, z)
                        in _person(wrist_left=(-0.24, -0.90, 2.0)).items()}})
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["limbs"]["bad"], 1)


class GhostFilterDepthTests(_GhostBase):
    def test_keeps_a_person_occluding_the_room(self):
        self._on()
        self._depth(_raster(3500, 2030))
        self.assertEqual(len(self._pump(_person())), 1)

    def test_drops_a_reflection_framed_by_nearer_geometry(self):
        # The mirror / TV-screen case: the body reads at the folded path
        # length (2 m), the bezel or frame around it at 1 m.
        self._on()
        self._depth(_raster(1000, 2030))
        self.assertEqual(self._pump(_person()), [])
        self.assertIn("depth suspect", self.out)

    def test_drops_a_flat_depth_blob(self):
        # The surround sits at the body's own depth: part of a surface.
        self._on()
        self._depth(_raster(2030, 2030))
        self.assertEqual(self._pump(_person()), [])
        self.assertIn("flat depth blob", self.out)

    def test_drops_a_skeleton_with_no_mass_behind_it(self):
        self._on()
        self._depth(_raster(5000))
        self.assertEqual(self._pump(_person()), [])
        self.assertIn("depth uncorroborated", self.out)

    def test_unknown_depth_never_drops(self):
        self._on()
        self._depth(_raster(0))                 # all "no return"
        self.assertEqual(len(self._pump(_person())), 1)

    def test_depth_is_rechecked_at_most_once_per_window(self):
        self._on(recheck=60.0)
        calls = self._depth(_raster(1000, 2030))
        self.assertEqual(self._pump(_person()), [])
        self.assertEqual(self._pump(_person()), [])      # cached verdict
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.out, "")                   # announced once

    def test_recheck_zero_skips_the_depth_check(self):
        self._on(recheck=0.0)
        calls = self._depth(_raster(1000, 2030))
        self.assertEqual(len(self._pump(_person())), 1)
        self.assertEqual(calls["n"], 0)

    def test_status_reports_the_last_frame(self):
        self._on()
        self._depth(None)
        self._pump(_person(), _stretched())
        st = kb.ghost_filter_status()
        self.assertTrue(st["enabled"])
        self.assertEqual((st["checked"], st["dropped"]), (2, 1))


@unittest.skipIf(np is None, "numpy needed for the real get_depth path")
class GhostFilterFrozenDepthTests(_GhostBase):
    """The REAL get_depth() behind the filter: a dead depth stream keeps serving
    one frozen buffer, which must never be used to judge live bodies."""

    def _pump_rt(self, depth_raster, depth_is_new, joints=None):
        rt = _FakeRuntime(
            bodies=[_FakeBody(True, joints=joints or _person(),
                              tracking_id=100)],
            depth=depth_raster.ravel())
        rt._new["depth"] = depth_is_new
        _patch_loader(self, rt)
        kb.get_runtime()
        p = mock.patch.object(kb, "get_depth_space_mapper", lambda: _mapper)
        p.start()
        self.addCleanup(p.stop)
        with redirect_stdout(io.StringIO()) as out:
            kb._pump_tick()
        self.out = out.getvalue()
        return kb.get_bodies()

    def test_frozen_depth_does_not_drop_a_live_body(self):
        # Empty room (wall at 3.5 m) in a stale buffer, no new depth frame:
        # on a fresh frame this body would be "uncorroborated"; on a frozen one
        # the verdict is unknown and the person stays.
        self._on()
        got = self._pump_rt(_raster(3500), depth_is_new=False)
        self.assertEqual(len(got), 1)
        self.assertEqual(self.out, "")
        self.assertTrue(kb.get_presence()["present"])

    def test_frozen_depth_is_not_remembered_as_a_verdict(self):
        self._on(recheck=60.0)
        self._pump_rt(_raster(3500), depth_is_new=False)
        self.assertEqual(kb._ghost_depth_seen, {})

    def test_a_new_depth_frame_is_still_used(self):
        # Same scene, but the frame is genuinely new: the ghost is dropped.
        self._on()
        self.assertEqual(self._pump_rt(_raster(3500), depth_is_new=True), [])
        self.assertIn("depth uncorroborated", self.out)

    def test_a_just_received_frame_is_reused_then_expires(self):
        self._on()
        raster = _raster(3500)
        self.assertIs(kb._ghost_fresh_depth(), None)
        with mock.patch.object(kb, "get_depth", return_value=raster):
            self.assertIs(kb._ghost_fresh_depth(), raster)
        with mock.patch.object(kb, "get_depth", return_value=None):
            self.assertIs(kb._ghost_fresh_depth(), raster)       # within window
            kb._ghost_depth_frame[1] -= kb.GHOST_DEPTH_MAX_AGE_SEC + 1.0
            self.assertIs(kb._ghost_fresh_depth(), None)         # expired


if __name__ == "__main__":
    unittest.main()
