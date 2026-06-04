"""Tests for audio/kinect_pointing — the PURE pointing geometry + the
calibration store. No hardware, no monolith, no real data file.

Covers:
  * arm_direction: builds a unit ray shoulder→hand, picks the pointing arm
    (more-extended / raised), returns None when joints aren't tracked,
  * angle_between: correctness (0 / 90 / 180 / unnormalised inputs / None),
  * PointingStore.put/list/remove/device_for round-trip on a TMP path,
  * resolve: returns the right target inside the cone, None outside it, and the
    CLOSEST of two competing targets,
  * average_direction: averages steady frames, rejects an outlier, returns None
    on too-few samples.

The store always uses a throwaway path (tempfile), so the real
data/kinect_pointing.json is never read or written by this suite.

stdlib unittest only.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

from audio import kinect_pointing as kp


# ─── synthetic body builders ────────────────────────────────────────────────
def _body(side="right", *, shoulder=(0.0, 0.0, 2.0), hand=(1.0, 0.0, 2.0),
          state=2, with_elbow=True, with_tip=True, distance_m=2.0,
          other=None):
    """One get_bodies()-shaped body with a single tracked arm on `side`.
    `state` sets the tracking_state on every joint of that arm. `other`, if
    given, is a dict of extra joints merged in (e.g. the opposite arm)."""
    j = {}
    j[f"shoulder_{side}"] = (shoulder[0], shoulder[1], shoulder[2], state)
    if with_elbow:
        mid = ((shoulder[0] + hand[0]) / 2, (shoulder[1] + hand[1]) / 2,
               (shoulder[2] + hand[2]) / 2)
        j[f"elbow_{side}"] = (mid[0], mid[1], mid[2], state)
    j[f"hand_{side}"] = (hand[0], hand[1], hand[2], state)
    if with_tip:
        j[f"hand_tip_{side}"] = (hand[0] + 0.05, hand[1], hand[2], state)
    if other:
        j.update(other)
    return {"id": 0, "joints": j, "distance_m": distance_m,
            "head": None, "facing": None}


def _arm_joints(side, *, shoulder, hand, state=2):
    """Just the joints for one arm (to merge as the 'other' arm)."""
    mid = ((shoulder[0] + hand[0]) / 2, (shoulder[1] + hand[1]) / 2,
           (shoulder[2] + hand[2]) / 2)
    return {
        f"shoulder_{side}": (shoulder[0], shoulder[1], shoulder[2], state),
        f"elbow_{side}": (mid[0], mid[1], mid[2], state),
        f"hand_{side}": (hand[0], hand[1], hand[2], state),
        f"hand_tip_{side}": (hand[0] + 0.05, hand[1], hand[2], state),
    }


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


# ─── angle_between ──────────────────────────────────────────────────────────
class AngleBetweenTests(unittest.TestCase):
    def test_same_direction_zero(self):
        self.assertAlmostEqual(kp.angle_between((1, 0, 0), (1, 0, 0)), 0.0, places=5)

    def test_orthogonal_ninety(self):
        self.assertAlmostEqual(kp.angle_between((1, 0, 0), (0, 1, 0)), 90.0, places=5)

    def test_opposite_oneeighty(self):
        self.assertAlmostEqual(kp.angle_between((1, 0, 0), (-1, 0, 0)), 180.0, places=4)

    def test_unnormalised_inputs(self):
        # Magnitude must not matter — only direction.
        self.assertAlmostEqual(kp.angle_between((5, 0, 0), (0, 9, 0)), 90.0, places=5)

    def test_none_and_zero_vectors(self):
        self.assertIsNone(kp.angle_between(None, (1, 0, 0)))
        self.assertIsNone(kp.angle_between((1, 0, 0), None))
        self.assertIsNone(kp.angle_between((0, 0, 0), (1, 0, 0)))

    def test_known_angle(self):
        d = (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)
        self.assertAlmostEqual(kp.angle_between((1, 0, 0), d), 30.0, places=4)


# ─── arm_direction ──────────────────────────────────────────────────────────
class ArmDirectionTests(unittest.TestCase):
    def test_right_arm_points_plus_x(self):
        b = _body(side="right", shoulder=(0.0, 0.0, 2.0), hand=(1.0, 0.0, 2.0))
        ray = kp.arm_direction(b)
        self.assertIsNotNone(ray)
        origin, d = ray
        # Origin is the shoulder.
        self.assertAlmostEqual(origin[0], 0.0, places=5)
        # Direction is a unit vector toward +x.
        self.assertAlmostEqual(d[0], 1.0, places=2)
        self.assertAlmostEqual(d[1], 0.0, places=2)
        self.assertAlmostEqual(math.sqrt(d[0]**2 + d[1]**2 + d[2]**2), 1.0, places=5)

    def test_none_when_arm_untracked(self):
        # Whole arm inferred (state 1) → below MIN_TRACKING_STATE → no ray.
        b = _body(side="right", state=1)
        self.assertIsNone(kp.arm_direction(b))

    def test_falls_back_to_left_when_only_left_tracked(self):
        # Right arm untracked (state 0), left arm tracked and pointing up-left.
        right = _arm_joints("right", shoulder=(0.0, 0.0, 2.0),
                            hand=(0.6, 0.0, 2.0), state=0)
        b = _body(side="left", shoulder=(-0.2, 0.0, 2.0), hand=(-1.0, 0.3, 2.0),
                  other=right)
        ray = kp.arm_direction(b)
        self.assertIsNotNone(ray)
        _, d = ray
        # Left arm pointed toward -x → direction x is negative.
        self.assertLess(d[0], 0.0)

    def test_picks_more_extended_arm(self):
        # Left arm fully extended (long), right arm short/relaxed. The extended
        # arm is the deliberate point and must win.
        short_right = _arm_joints("right", shoulder=(0.2, 0.0, 2.0),
                                  hand=(0.45, -0.4, 2.0))   # short + downward
        b = _body(side="left", shoulder=(-0.2, 0.0, 2.0),
                  hand=(-1.4, 0.1, 2.0), other=short_right)   # long, level
        ray = kp.arm_direction(b)
        self.assertIsNotNone(ray)
        _, d = ray
        self.assertLess(d[0], 0.0)   # the long LEFT arm (-x) was chosen

    def test_elbow_to_hand_fallback_when_shoulder_missing(self):
        # No shoulder joint at all → ray built from elbow→hand still works.
        j = {
            "elbow_right": (0.0, 0.0, 2.0, 2),
            "hand_right": (0.0, 1.0, 2.0, 2),       # straight up
            "hand_tip_right": (0.0, 1.05, 2.0, 2),
        }
        b = {"id": 0, "joints": j, "distance_m": 2.0}
        ray = kp.arm_direction(b)
        self.assertIsNotNone(ray)
        origin, d = ray
        self.assertAlmostEqual(origin[1], 0.0, places=5)   # origin = elbow
        self.assertAlmostEqual(d[1], 1.0, places=2)        # points +y

    def test_bad_input_returns_none(self):
        self.assertIsNone(kp.arm_direction(None))
        self.assertIsNone(kp.arm_direction({}))
        self.assertIsNone(kp.arm_direction({"joints": {}}))
        self.assertIsNone(kp.arm_direction("nonsense"))


# ─── PointingStore ──────────────────────────────────────────────────────────
class _StoreBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="jarvis_point_test_")
        self.path = os.path.join(self._tmpdir, "kinect_pointing.json")
        self.store = kp.PointingStore(path=self.path)

    def tearDown(self):
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
            os.rmdir(self._tmpdir)
        except OSError:
            pass


class StoreRoundTripTests(_StoreBase):
    def test_put_creates_file_and_lists(self):
        self.assertFalse(os.path.exists(self.path))
        ok = self.store.put("desk lamp", (1.0, 0.0, 0.1), device="Office Lamp")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.path))   # durable write
        targets = self.store.list_targets()
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["name"], "desk lamp")
        self.assertEqual(targets[0]["device"], "Office Lamp")
        # Stored direction is normalised.
        d = targets[0]["dir"]
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in d)), 1.0, places=5)

    def test_device_for_falls_back_to_name(self):
        self.store.put("fan", (0.0, 0.0, 1.0))   # no explicit device
        self.assertEqual(self.store.device_for("fan"), "fan")

    def test_device_for_case_insensitive(self):
        self.store.put("Desk Lamp", (1.0, 0.0, 0.0), device="Lamp A")
        self.assertEqual(self.store.device_for("desk lamp"), "Lamp A")

    def test_remove_target(self):
        self.store.put("fan", (0.0, 0.0, 1.0))
        self.assertTrue(self.store.remove_target("FAN"))   # case-insensitive
        self.assertEqual(self.store.list_targets(), [])
        self.assertFalse(self.store.remove_target("fan"))  # already gone

    def test_put_overwrites(self):
        self.store.put("lamp", (1.0, 0.0, 0.0))
        self.store.put("lamp", (0.0, 1.0, 0.0))
        targets = self.store.list_targets()
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0]["dir"][1], 1.0, places=4)

    def test_put_rejects_empty_name_or_zero_vector(self):
        self.assertFalse(self.store.put("", (1.0, 0.0, 0.0)))
        self.assertFalse(self.store.put("x", (0.0, 0.0, 0.0)))

    def test_corrupt_file_reads_as_empty(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        self.assertEqual(self.store.list_targets(), [])
        # And a put still works (overwrites the garbage).
        self.assertTrue(self.store.put("lamp", (1.0, 0.0, 0.0)))
        self.assertEqual(len(self.store.list_targets()), 1)


class StoreResolveTests(_StoreBase):
    def test_resolves_within_cone(self):
        self.store.put("desk lamp", (1.0, 0.0, 0.0))      # +x
        self.store.put("fan", (0.0, 0.0, 1.0))            # +z
        # A direction 5° off +x resolves to the desk lamp.
        d = (math.cos(math.radians(5)), math.sin(math.radians(5)), 0.0)
        self.assertEqual(self.store.resolve(d), "desk lamp")
        # Straight +z resolves to the fan.
        self.assertEqual(self.store.resolve((0.0, 0.0, 1.0)), "fan")

    def test_none_outside_cone(self):
        self.store.put("desk lamp", (1.0, 0.0, 0.0))
        # 45° off the only target → outside the ~18° cone → None.
        d = (math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0)
        self.assertIsNone(self.store.resolve(d))

    def test_closest_of_two_wins(self):
        # Two targets 30° apart; aim 10° from A (so 20° from B). A is closer and
        # inside the cone; B is outside it. A must win.
        self.store.put("A", _unit((1.0, 0.0, 0.0)))
        self.store.put("B", (math.cos(math.radians(30)),
                             math.sin(math.radians(30)), 0.0))
        aim = (math.cos(math.radians(10)), math.sin(math.radians(10)), 0.0)
        self.assertEqual(self.store.resolve(aim), "A")

    def test_closest_when_both_in_cone(self):
        # Both targets within the cone of the aim; the nearer one wins.
        self.store.put("near", (math.cos(math.radians(3)),
                                math.sin(math.radians(3)), 0.0))
        self.store.put("far", (math.cos(math.radians(15)),
                               math.sin(math.radians(15)), 0.0))
        aim = (1.0, 0.0, 0.0)   # 3° from near, 15° from far — both inside ~18°
        self.assertEqual(self.store.resolve(aim), "near")

    def test_resolve_none_direction(self):
        self.store.put("A", (1.0, 0.0, 0.0))
        self.assertIsNone(self.store.resolve(None))

    def test_resolve_empty_store(self):
        self.assertIsNone(self.store.resolve((1.0, 0.0, 0.0)))


# ─── average_direction ──────────────────────────────────────────────────────
class AverageDirectionTests(unittest.TestCase):
    def test_averages_steady_frames(self):
        # Several nearly-identical +x directions → mean ~ +x.
        dirs = [(1.0, 0.02 * i, 0.0) for i in range(-3, 4)]   # 7 frames
        avg = kp.average_direction(dirs)
        self.assertIsNotNone(avg)
        self.assertAlmostEqual(avg[0], 1.0, places=1)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in avg)), 1.0, places=5)

    def test_rejects_outlier(self):
        # Six tight +x frames plus one wild +y outlier; the outlier is dropped,
        # so the mean stays essentially +x (not pulled toward +y).
        dirs = [(1.0, 0.0, 0.0)] * 6 + [(0.0, 1.0, 0.0)]
        avg = kp.average_direction(dirs)
        self.assertIsNotNone(avg)
        self.assertGreater(avg[0], 0.99)
        self.assertLess(abs(avg[1]), 0.05)

    def test_none_when_too_few_samples(self):
        # Below CALIBRATE_MIN_SAMPLES valid frames → None ("couldn't get a
        # steady read").
        few = [(1.0, 0.0, 0.0)] * (kp.CALIBRATE_MIN_SAMPLES - 1)
        self.assertIsNone(kp.average_direction(few))

    def test_none_filtered_out(self):
        # None frames don't count toward the sample minimum.
        dirs = [None] * 10 + [(1.0, 0.0, 0.0)] * 2
        self.assertIsNone(kp.average_direction(dirs))


if __name__ == "__main__":
    unittest.main()
