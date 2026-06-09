"""Tests for audio.kinect_skeleton — the pure skeleton→color-space projector.

PART A's skeleton overlay projects the Kinect body's camera-space joints to
color-space pixels and lists the bone segments to stroke. That geometry is a
pure function fed an injectable mapper, so these tests drive it with fabricated
joints + a trivial fake mapper — NO pykinect2, NO cv2, NO sensor. They assert:

  * tracked joints project; untracked (below min state) joints are dropped;
  * non-finite (inf/NaN) and wildly off-frame projections are discarded;
  * a ColorSpacePoint-like object (.x/.y) AND a plain (px, py) tuple are both
    accepted from the mapper;
  * bone segments are emitted only for bones whose BOTH endpoints projected;
  * malformed joints / a throwing mapper degrade to omission, never raise.

stdlib unittest only — runs on the light-deps CI runner (cv2 is blocked there).
"""
from __future__ import annotations

import unittest

from audio import kinect_skeleton as ks


TRACKED = 2      # PyKinect TrackingState fully tracked
INFERRED = 1
NOT_TRACKED = 0


def _identity_mapper(x, y, z):
    """Trivial mapper: treat camera (x, y) as already-pixel, ignore z. Returns a
    plain (px, py) tuple — the sequence form the projector must accept."""
    return (x, y)


class _CSP:
    """A ColorSpacePoint stand-in carrying float .x/.y (the real pykinect2
    ColorSpacePoint shape the production mapper returns)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ProjectJointsTests(unittest.TestCase):
    def test_tracked_joints_project_to_pixels(self):
        joints = {
            "head": (100.0, 200.0, 2.0, TRACKED),
            "neck": (110.0, 260.0, 2.0, TRACKED),
        }
        pts = ks.project_body_joints(joints, _identity_mapper)
        self.assertEqual(pts["head"], (100, 200))
        self.assertEqual(pts["neck"], (110, 260))

    def test_untracked_joint_is_dropped_by_default(self):
        # Default min_tracking_state=1 → NotTracked(0) dropped, Inferred(1) kept.
        joints = {
            "head": (100.0, 200.0, 2.0, TRACKED),
            "hand_left": (300.0, 100.0, 2.0, NOT_TRACKED),
            "hand_right": (320.0, 110.0, 2.0, INFERRED),
        }
        pts = ks.project_body_joints(joints, _identity_mapper)
        self.assertIn("head", pts)
        self.assertNotIn("hand_left", pts)     # NotTracked dropped
        self.assertIn("hand_right", pts)       # Inferred kept (keeps figure whole)

    def test_min_tracking_state_two_drops_inferred(self):
        joints = {"hand_right": (320.0, 110.0, 2.0, INFERRED)}
        pts = ks.project_body_joints(joints, _identity_mapper,
                                     min_tracking_state=2)
        self.assertEqual(pts, {})

    def test_bare_xyz_joint_without_state_is_projected(self):
        # A (x, y, z) tuple with no tracking_state is treated as tracked enough.
        pts = ks.project_body_joints({"head": (50.0, 60.0, 2.0)}, _identity_mapper)
        self.assertEqual(pts["head"], (50, 60))

    def test_infinite_projection_is_dropped(self):
        pts = ks.project_body_joints({"head": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: (float("inf"), 5.0))
        self.assertEqual(pts, {})

    def test_nan_projection_is_dropped(self):
        pts = ks.project_body_joints({"head": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: (10.0, float("nan")))
        self.assertEqual(pts, {})

    def test_wildly_off_frame_projection_is_dropped(self):
        # Many frame-widths away → a bad point we must not stroke a bone to.
        pts = ks.project_body_joints({"head": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: (99999.0, 99999.0))
        self.assertEqual(pts, {})

    def test_just_off_frame_projection_is_kept(self):
        # A joint a little past the edge (raised arm) is within the margin → kept.
        px = ks.COLOR_W + 100
        pts = ks.project_body_joints({"hand_right": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: (px, 540.0))
        self.assertEqual(pts["hand_right"], (px, 540))

    def test_color_space_point_object_accepted(self):
        pts = ks.project_body_joints({"head": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: _CSP(640.4, 360.6))
        # rounded to ints
        self.assertEqual(pts["head"], (640, 361))

    def test_mapper_returning_none_omits_joint(self):
        pts = ks.project_body_joints({"head": (0.0, 0.0, 2.0, TRACKED)},
                                     lambda *a: None)
        self.assertEqual(pts, {})

    def test_throwing_mapper_omits_joint_not_raises(self):
        def _boom(*_a):
            raise RuntimeError("COM glitch")
        # One throwing joint must not abort the whole skeleton.
        joints = {"head": (10.0, 20.0, 2.0, TRACKED)}
        pts = ks.project_body_joints(joints, _boom)
        self.assertEqual(pts, {})

    def test_malformed_joint_tuple_is_skipped(self):
        joints = {
            "head": (10.0, 20.0, 2.0, TRACKED),
            "neck": ("oops",),            # too short / wrong type
            "spine_mid": None,            # None joint
        }
        pts = ks.project_body_joints(joints, _identity_mapper)
        self.assertEqual(set(pts), {"head"})

    def test_non_dict_joints_returns_empty(self):
        self.assertEqual(ks.project_body_joints(None, _identity_mapper), {})
        self.assertEqual(ks.project_body_joints([1, 2, 3], _identity_mapper), {})

    def test_non_callable_mapper_returns_empty(self):
        self.assertEqual(
            ks.project_body_joints({"head": (0, 0, 2, TRACKED)}, None), {})


class BoneSegmentTests(unittest.TestCase):
    def test_bone_drawn_only_when_both_endpoints_present(self):
        # head + neck present (a real BONE pair) → 1 segment; an isolated joint
        # with no partnered neighbour yields none.
        points = {"head": (100, 200), "neck": (110, 260)}
        segs = ks.iter_bone_segments(points)
        self.assertIn(((100, 200), (110, 260)), segs)
        self.assertEqual(len(segs), 1)   # only the head-neck bone has both ends

    def test_missing_endpoint_skips_that_bone(self):
        # Only 'head' present — the head→neck bone needs 'neck', so no segment.
        segs = ks.iter_bone_segments({"head": (100, 200)})
        self.assertEqual(segs, [])

    def test_full_skeleton_segment_count_matches_present_bones(self):
        # Project a plausible full upper body; every bone whose ends are present
        # should appear exactly once and no bone should reference a missing joint.
        names = ("head", "neck", "spine_shoulder", "spine_mid", "spine_base",
                 "shoulder_left", "elbow_left", "wrist_left", "hand_left",
                 "shoulder_right", "elbow_right", "wrist_right", "hand_right")
        points = {n: (i * 10, i * 10) for i, n in enumerate(names)}
        segs = ks.iter_bone_segments(points)
        expected = sum(1 for a, b in ks.BONES if a in points and b in points)
        self.assertEqual(len(segs), expected)
        self.assertGreater(len(segs), 8)   # a real upper-body skeleton has many

    def test_bones_topology_is_self_consistent(self):
        # Every BONE endpoint must be one of the bridge's friendly joint names so
        # a projected {name: pt} dict can actually look them up.
        from audio import kinect_bridge as kb
        valid = set(kb._JOINT_NAMES)
        for a, b in ks.BONES:
            self.assertIn(a, valid, f"bone parent {a!r} not a known joint")
            self.assertIn(b, valid, f"bone child {b!r} not a known joint")

    def test_no_duplicate_bones(self):
        seen = set()
        for a, b in ks.BONES:
            key = frozenset((a, b))
            self.assertNotIn(key, seen, f"duplicate bone {a}-{b}")
            seen.add(key)


# ── B2 preview-feedback geometry: controlling-hand pick, radius, alpha blend ──
class ControllingHandPointTests(unittest.TestCase):
    def test_prefers_right_hand(self):
        points = {"hand_right": (100, 200), "hand_left": (300, 400)}
        self.assertEqual(ks.controlling_hand_point(points), (100, 200))

    def test_falls_back_to_left_then_wrist(self):
        # No right hand → left hand.
        self.assertEqual(
            ks.controlling_hand_point({"hand_left": (300, 400)}), (300, 400))
        # No hand at all → a wrist (hand tip dropped to untracked).
        self.assertEqual(
            ks.controlling_hand_point({"wrist_right": (50, 60)}), (50, 60))

    def test_none_when_no_hand_or_wrist_present(self):
        # A body with only torso joints → no hand circle.
        self.assertIsNone(ks.controlling_hand_point({"head": (10, 10),
                                                     "spine_mid": (10, 50)}))
        self.assertIsNone(ks.controlling_hand_point({}))
        self.assertIsNone(ks.controlling_hand_point(None))

    def test_prefer_side_follows_extended_hand(self):
        # The reach-to-engage air-mouse passes the live which-hand: with both
        # hands projected, prefer_side="left" draws the circle on the LEFT hand
        # (the extended one), overriding the default right-first order.
        points = {"hand_right": (100, 200), "hand_left": (300, 400)}
        self.assertEqual(
            ks.controlling_hand_point(points, prefer_side="left"), (300, 400))
        self.assertEqual(
            ks.controlling_hand_point(points, prefer_side="right"), (100, 200))

    def test_prefer_side_falls_back_to_wrist_then_default(self):
        # Preferred side's hand absent → its wrist; then the default order.
        self.assertEqual(
            ks.controlling_hand_point({"wrist_left": (5, 6),
                                       "hand_right": (100, 200)},
                                      prefer_side="left"), (5, 6))
        # Preferred side has nothing projected → default order (right hand).
        self.assertEqual(
            ks.controlling_hand_point({"hand_right": (100, 200)},
                                      prefer_side="left"), (100, 200))

    def test_prefer_side_none_keeps_default_order(self):
        # prefer_side=None (disengaged) → the existing right-first behaviour.
        points = {"hand_right": (100, 200), "hand_left": (300, 400)}
        self.assertEqual(
            ks.controlling_hand_point(points, prefer_side=None), (100, 200))

    def test_fallback_false_requires_preferred_side(self):
        # The "no controller → no ring" contract the bright engaged ring uses:
        # with fallback=False the PREFERRED side must project, else None — never an
        # arbitrary (e.g. always-right) fallback.
        points = {"hand_right": (100, 200), "hand_left": (300, 400)}
        # Preferred side projects → its point.
        self.assertEqual(
            ks.controlling_hand_point(points, prefer_side="left",
                                      fallback=False), (300, 400))
        # Preferred (left) side absent, only the right hand → None (no wrong-hand).
        self.assertIsNone(
            ks.controlling_hand_point({"hand_right": (100, 200)},
                                      prefer_side="left", fallback=False))
        # Preferred side's wrist still counts (hand tip dropped).
        self.assertEqual(
            ks.controlling_hand_point({"wrist_left": (5, 6)},
                                      prefer_side="left", fallback=False), (5, 6))

    def test_fallback_false_without_side_is_none(self):
        # No prefer_side + fallback=False → nothing to prefer → None.
        points = {"hand_right": (100, 200), "hand_left": (300, 400)}
        self.assertIsNone(
            ks.controlling_hand_point(points, prefer_side=None, fallback=False))


class HandCircleRadiusTests(unittest.TestCase):
    def test_scales_with_width_and_has_a_floor(self):
        big = ks.hand_circle_radius(1920)
        small = ks.hand_circle_radius(640)
        self.assertGreater(big, small)            # scales with frame width
        self.assertGreaterEqual(small, 12)        # never vanishes
        # Bad input → the safe default, never a raise.
        self.assertEqual(ks.hand_circle_radius("nope"), 42)


class BlendColorTests(unittest.TestCase):
    def test_alpha_zero_keeps_base(self):
        self.assertEqual(ks.blend_color((10, 20, 30), (200, 200, 200), 0.0),
                         (10, 20, 30))

    def test_alpha_one_is_full_over(self):
        self.assertEqual(ks.blend_color((10, 20, 30), (200, 100, 50), 1.0),
                         (200, 100, 50))

    def test_half_blend_is_midpoint(self):
        out = ks.blend_color((0, 0, 0), (100, 200, 40), 0.5)
        self.assertEqual(out, (50, 100, 20))

    def test_clamps_and_tolerates_bad_alpha(self):
        # Out-of-range alpha is clamped; channels stay within 0..255.
        out = ks.blend_color((0, 0, 0), (300, 300, 300), 5.0)
        for ch in out:
            self.assertTrue(0 <= ch <= 255)


if __name__ == "__main__":
    unittest.main()
