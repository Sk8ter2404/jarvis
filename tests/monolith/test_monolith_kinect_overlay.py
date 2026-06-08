"""Monolith tests for PART A — the Kinect color + LIVE SKELETON HUD preview.

Covers the compositor wired into bobert_companion:

  * _composite_preview_image  — webcam tiles laid down the right edge of the
                                Kinect color+skeleton base (pure cv2/numpy);
  * _draw_skeleton_on_color   — bones+joints stroked over the color frame via
                                the bridge's color-space mapper (count returned);
  * _compose_kinect_preview   — None when no Kinect frame; composed otherwise,
                                reusing already-cached webcam frames;
  * _hud_kinect_preview_write — off when the flag is disabled; writes the
                                composite via the shared preview writer when on;
  * _resolve_webcam_indices_by_name — pygrabber name→index map (mocked), cached,
                                graceful when pygrabber is absent;
  * _hud_kinect_skeleton_overlay_enabled — reflects the config flags.

Imported ONCE via the cached harness; @requires_monolith so it SKIPS on the
light-deps CI runner (cv2/numpy absent) and RUNS in the local full tier. Nothing
opens a real Kinect, camera, or writes the real preview file (the writer is
mocked, or a temp path is used).
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _np():
    import numpy as np
    return np


@requires_monolith
class CompositePreviewImageTests(MonolithGlobalsTestCase):
    def test_no_tiles_returns_base_unchanged(self):
        np = _np()
        base = np.full((1080, 1920, 3), 7, dtype=np.uint8)
        out = self.bc._composite_preview_image(base, [])
        # Same object back (no tiles → nothing to composite).
        self.assertIs(out, base)

    def test_tiles_laid_on_right_edge_without_mutating_base(self):
        np = _np()
        base = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tile = np.full((480, 640, 3), 255, dtype=np.uint8)
        out = self.bc._composite_preview_image(base, [tile, tile])
        # Base must NOT be scribbled on (we copy first).
        self.assertEqual(int(base.max()), 0)
        # The composite has bright pixels somewhere on the right side (a tile +
        # its border were drawn).
        self.assertGreater(int(out.max()), 0)
        right_half = out[:, out.shape[1] // 2:, :]
        self.assertGreater(int(right_half.max()), 0)
        self.assertEqual(out.shape, base.shape)

    def test_oversized_tiles_do_not_overflow_or_raise(self):
        np = _np()
        base = np.zeros((200, 400, 3), dtype=np.uint8)
        # A tall tile that, scaled to ~20% width, would exceed the base height
        # must be clipped (the stack stops) rather than raise.
        tall = np.full((4000, 100, 3), 200, dtype=np.uint8)
        out = self.bc._composite_preview_image(base, [tall])
        self.assertEqual(out.shape, base.shape)


@requires_monolith
class DrawSkeletonTests(MonolithGlobalsTestCase):
    def _fake_mapper(self):
        # Camera (x, y) → a pixel near frame centre, deterministic + on-frame.
        return lambda x, y, z: (960 + x * 100.0, 540 - y * 100.0)

    def test_draws_bodies_and_returns_count(self):
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bodies = [{
            "joints": {
                "head": (0.0, 1.0, 2.0, 2),
                "neck": (0.0, 0.7, 2.0, 2),
                "spine_shoulder": (0.0, 0.5, 2.0, 2),
                "shoulder_right": (0.3, 0.4, 2.0, 2),
                "elbow_right": (0.5, 0.2, 2.0, 2),
            }
        }]
        with mock.patch.object(self.bc._kinect_bridge, "get_color_space_mapper",
                               return_value=self._fake_mapper()):
            drawn = self.bc._draw_skeleton_on_color(color, bodies)
        self.assertEqual(drawn, 1)
        # Something was actually stroked onto the frame.
        self.assertGreater(int(color.max()), 0)

    def test_returns_zero_when_no_mapper(self):
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with mock.patch.object(self.bc._kinect_bridge, "get_color_space_mapper",
                               return_value=None):
            drawn = self.bc._draw_skeleton_on_color(color, [{"joints": {}}])
        self.assertEqual(drawn, 0)
        self.assertEqual(int(color.max()), 0)   # nothing drawn

    def test_returns_zero_for_empty_bodies(self):
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with mock.patch.object(self.bc._kinect_bridge, "get_color_space_mapper",
                               return_value=self._fake_mapper()):
            self.assertEqual(self.bc._draw_skeleton_on_color(color, []), 0)


@requires_monolith
class ComposeKinectPreviewTests(MonolithGlobalsTestCase):
    def test_none_when_no_kinect_color_frame(self):
        with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                               return_value=None):
            self.assertIsNone(self.bc._compose_kinect_preview(now=123.0))

    def test_composes_with_skeleton_and_real_webcam_tiles(self):
        # B1: the side tiles are the REAL webcams (captured directly), NOT the
        # Kinect cache. Mock the direct reader to hand back two webcam frames.
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        left = np.full((480, 640, 3), 90, dtype=np.uint8)
        right = np.full((480, 640, 3), 200, dtype=np.uint8)
        bodies = [{"joints": {"head": (0.0, 1.0, 2.0, 2),
                              "neck": (0.0, 0.7, 2.0, 2)}}]
        with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                               return_value=color), \
                mock.patch.object(self.bc._kinect_bridge, "get_bodies",
                                  return_value=bodies), \
                mock.patch.object(self.bc._kinect_bridge,
                                  "get_color_space_mapper",
                                  return_value=lambda x, y, z: (960, 540)), \
                mock.patch.object(self.bc, "_read_side_tile_webcams",
                                  return_value={"left": left, "right": right}):
            out = self.bc._compose_kinect_preview(now=10.0)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, color.shape)

    def test_does_not_read_kinect_cache_for_tiles(self):
        # The compose path must NOT pull side tiles from _camera_latest_frame
        # (the Kinect frame on the owner's rig). Seed a GARISH all-255 frame in
        # the cache; the real-webcam reader returns a distinct value (77). The
        # composed right-edge tiles must show the WEBCAM (77), proving the Kinect
        # cache was ignored — if the old code leaked, a solid 255 block would
        # appear there instead.
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        webcam = np.full((480, 640, 3), 77, dtype=np.uint8)
        kinect_marker = np.full((480, 640, 3), 255, dtype=np.uint8)  # would-be leak
        with self.bc._camera_state_lock:
            self.bc._camera_latest_frame[0] = kinect_marker
            self.bc._camera_latest_frame[1] = kinect_marker
        real_lock = self.bc._camera_state_lock
        try:
            with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                                   return_value=color), \
                    mock.patch.object(self.bc._kinect_bridge, "get_bodies",
                                      return_value=[]), \
                    mock.patch.object(self.bc, "_read_side_tile_webcams",
                                      return_value={"left": webcam, "right": webcam}):
                out = self.bc._compose_kinect_preview(now=10.0)
        finally:
            with real_lock:
                self.bc._camera_latest_frame.pop(0, None)
                self.bc._camera_latest_frame.pop(1, None)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, color.shape)
        # Inspect the right-edge tile strip (where the side tiles are laid). The
        # webcam value 77 must be present there; a solid 255 Kinect block must
        # NOT be (the base outside the tiles is 0, so 255 could only come from a
        # leaked Kinect cache frame).
        right_strip = out[:, int(out.shape[1] * 0.78):, :]
        self.assertTrue(bool((right_strip == 77).any()),
                        "webcam tile (77) should appear on the right edge")
        # No 3x3 solid-255 region anywhere in the strip (would be the Kinect leak).
        self.assertFalse(bool((right_strip == 255).all(axis=2).any()),
                         "a solid Kinect (255) tile must NOT appear")

    def test_off_webcam_shows_placeholder_not_kinect(self):
        # B1: a slot whose webcam won't open → a dim 'off' placeholder tile, NOT
        # the Kinect frame. The reader returns None for that slot.
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                               return_value=color), \
                mock.patch.object(self.bc._kinect_bridge, "get_bodies",
                                  return_value=[]), \
                mock.patch.object(self.bc, "_read_side_tile_webcams",
                                  return_value={"left": None, "right": None}):
            out = self.bc._compose_kinect_preview(now=10.0)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, color.shape)
        # The placeholder is a near-black panel (value 28) with grey text; it is
        # NOT a bright Kinect-ish frame. Its max stays low (text grey ~140).
        ph = self.bc._placeholder_tile("Left webcam")
        self.assertLessEqual(int(ph.max()), 145)
        self.assertGreaterEqual(int(ph.min()), 0)

    def test_does_not_mutate_bridge_frame_buffer(self):
        # The compose path must copy the bridge's color frame before drawing.
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                               return_value=color), \
                mock.patch.object(self.bc._kinect_bridge, "get_bodies",
                                  return_value=[{"joints": {"head": (0.0, 1.0, 2.0, 2),
                                                            "neck": (0.0, 0.7, 2.0, 2)}}]), \
                mock.patch.object(self.bc._kinect_bridge, "get_color_space_mapper",
                                  return_value=lambda x, y, z: (960, 540)), \
                mock.patch.object(self.bc, "_read_side_tile_webcams",
                                  return_value={"left": None, "right": None}):
            self.bc._compose_kinect_preview(now=10.0)
        # The original frame the bridge handed out is untouched.
        self.assertEqual(int(color.max()), 0)


@requires_monolith
class HudKinectPreviewWriteTests(MonolithGlobalsTestCase):
    def test_off_when_overlay_flag_disabled(self):
        with mock.patch.object(self.bc, "_hud_kinect_skeleton_overlay_enabled",
                               return_value=False):
            write = mock.Mock(return_value=True)
            with mock.patch.object(self.bc, "_hud_camera_preview_write", write):
                self.assertFalse(self.bc._hud_kinect_preview_write(now=1.0))
            write.assert_not_called()

    def test_writes_composite_when_enabled(self):
        np = _np()
        composed = np.full((1080, 1920, 3), 9, dtype=np.uint8)
        write = mock.Mock(return_value=True)
        with mock.patch.object(self.bc, "_hud_kinect_skeleton_overlay_enabled",
                               return_value=True), \
                mock.patch.object(self.bc, "_compose_kinect_preview",
                                  return_value=composed), \
                mock.patch.object(self.bc, "_hud_camera_preview_write", write):
            ok = self.bc._hud_kinect_preview_write(now=2.0)
        self.assertTrue(ok)
        write.assert_called_once()
        # The COMPOSITE (not a raw webcam frame) was handed to the writer.
        self.assertIs(write.call_args.args[0], composed)

    def test_false_when_no_kinect_frame(self):
        with mock.patch.object(self.bc, "_hud_kinect_skeleton_overlay_enabled",
                               return_value=True), \
                mock.patch.object(self.bc, "_compose_kinect_preview",
                                  return_value=None):
            self.assertFalse(self.bc._hud_kinect_preview_write(now=3.0))


@requires_monolith
class OverlayEnabledFlagTests(MonolithGlobalsTestCase):
    def test_enabled_requires_both_flags(self):
        import core.config as cfg
        with mock.patch.object(cfg, "KINECT_SKELETON_OVERLAY_ENABLED", True,
                               create=True), \
                mock.patch.object(cfg, "KINECT_ENABLED", True, create=True):
            self.assertTrue(self.bc._hud_kinect_skeleton_overlay_enabled())

    def test_disabled_when_kinect_off(self):
        import core.config as cfg
        with mock.patch.object(cfg, "KINECT_SKELETON_OVERLAY_ENABLED", True,
                               create=True), \
                mock.patch.object(cfg, "KINECT_ENABLED", False, create=True):
            self.assertFalse(self.bc._hud_kinect_skeleton_overlay_enabled())

    def test_disabled_when_overlay_flag_off(self):
        import core.config as cfg
        with mock.patch.object(cfg, "KINECT_SKELETON_OVERLAY_ENABLED", False,
                               create=True), \
                mock.patch.object(cfg, "KINECT_ENABLED", True, create=True):
            self.assertFalse(self.bc._hud_kinect_skeleton_overlay_enabled())


@requires_monolith
class ResolveWebcamNamesTests(MonolithGlobalsTestCase):
    def setUp(self):
        # Reset the resolver cache before each test so name resolution re-runs.
        self.bc._kinect_preview_webcam_idx.clear()
        self.bc._kinect_preview_webcam_resolved[0] = False
        self.addCleanup(self.bc._kinect_preview_webcam_idx.clear)
        self.addCleanup(lambda: self.bc._kinect_preview_webcam_resolved.__setitem__(0, False))

    def _fake_pygrabber(self, names):
        """Install a fake pygrabber.dshow_graph.FilterGraph whose
        get_input_devices() returns `names` (index == list position)."""
        mod = types.ModuleType("pygrabber")
        sub = types.ModuleType("pygrabber.dshow_graph")

        class _FG:
            def get_input_devices(self_inner):
                return list(names)
        sub.FilterGraph = _FG
        mod.dshow_graph = sub
        return mod, sub

    def test_resolves_left_right_by_name(self):
        mod, sub = self._fake_pygrabber(
            ["USB 2.0 Camera", "Kinect V2 Video Sensor", "Fullhan Webcam"])
        with mock.patch.dict("sys.modules",
                             {"pygrabber": mod, "pygrabber.dshow_graph": sub}):
            got = self.bc._resolve_webcam_indices_by_name()
        # 'Fullhan Webcam' is at index 2 → left; 'USB 2.0 Camera' at 0 → right.
        self.assertEqual(got.get("left"), 2)
        self.assertEqual(got.get("right"), 0)

    def test_result_is_cached(self):
        mod, sub = self._fake_pygrabber(["Fullhan Webcam", "USB 2.0 Camera"])
        with mock.patch.dict("sys.modules",
                             {"pygrabber": mod, "pygrabber.dshow_graph": sub}):
            self.bc._resolve_webcam_indices_by_name()
        # Second call must NOT re-enumerate (pygrabber now absent) — cache holds.
        with mock.patch.dict("sys.modules", {}, clear=False):
            import sys
            saved = sys.modules.pop("pygrabber", None)
            try:
                again = self.bc._resolve_webcam_indices_by_name()
            finally:
                if saved is not None:
                    sys.modules["pygrabber"] = saved
        self.assertEqual(again.get("left"), 0)
        self.assertEqual(again.get("right"), 1)

    def test_graceful_when_pygrabber_absent(self):
        # Importing pygrabber raises → returns {} and marks resolved (no retry).
        import builtins
        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name.startswith("pygrabber"):
                raise ImportError("no pygrabber")
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", _boom):
            got = self.bc._resolve_webcam_indices_by_name()
        self.assertEqual(got, {})
        self.assertTrue(self.bc._kinect_preview_webcam_resolved[0])


# ══════════════════════════════════════════════════════════════════════════
#  B1 — the side tiles are the REAL WEBCAMS, captured directly (NOT the Kinect)
# ══════════════════════════════════════════════════════════════════════════
@requires_monolith
class SideTileWebcamReadTests(MonolithGlobalsTestCase):
    def setUp(self):
        # Start every test with no held tile handles + fresh throttle clocks.
        self.bc._release_side_tile_webcams()
        self.addCleanup(self.bc._release_side_tile_webcams)

    class _FakeCap:
        """A stand-in cv2.VideoCapture: yields a solid frame per read()."""
        def __init__(self, value):
            self._np = __import__("numpy")
            self._value = value
            self.released = False

        def read(self):
            return True, self._np.full((480, 640, 3), self._value,
                                       dtype=self._np.uint8)

        def release(self):
            self.released = True

    def test_reads_the_named_webcam_indices_not_the_kinect(self):
        # 'left'→idx 5, 'right'→idx 6 (resolved by DirectShow NAME). The reader
        # must open EXACTLY those indices — the webcams, never the Kinect frame.
        opened = []

        def _fake_open(idx):
            opened.append(idx)
            return self._FakeCap(100 + idx)

        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 5, "right": 6}), \
                mock.patch.object(self.bc, "_open_tile_capture", _fake_open):
            out = self.bc._read_side_tile_webcams(now=1000.0)
        self.assertEqual(sorted(opened), [5, 6])           # opened the webcams
        self.assertIsNotNone(out["left"])
        self.assertIsNotNone(out["right"])
        # The frames are the webcam values (105 / 106), never a Kinect frame.
        self.assertEqual(int(out["left"].mean()), 105)
        self.assertEqual(int(out["right"].mean()), 106)

    def test_unresolved_name_yields_none_for_placeholder(self):
        # No 'right' webcam name resolved → that slot is None (→ placeholder), and
        # NEVER falls back to the Kinect cache.
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 5}), \
                mock.patch.object(self.bc, "_open_tile_capture",
                                  lambda idx: self._FakeCap(120)):
            out = self.bc._read_side_tile_webcams(now=2000.0)
        self.assertIsNotNone(out["left"])
        self.assertIsNone(out["right"])                    # → placeholder

    def test_open_failure_yields_none_for_placeholder(self):
        # The webcam won't open (off / covered / locked) → None → placeholder.
        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 5, "right": 6}), \
                mock.patch.object(self.bc, "_open_tile_capture",
                                  lambda idx: None):
            out = self.bc._read_side_tile_webcams(now=3000.0)
        self.assertIsNone(out["left"])
        self.assertIsNone(out["right"])

    def test_throttle_reuses_cached_frame_between_reads(self):
        # A second call within the read interval must NOT re-open the device —
        # it returns the cached frame (low-rate capture).
        calls = {"n": 0}

        def _fake_open(idx):
            calls["n"] += 1
            return self._FakeCap(130)

        with mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 5}), \
                mock.patch.object(self.bc, "_open_tile_capture", _fake_open):
            self.bc._read_side_tile_webcams(now=4000.0)
            opens_after_first = calls["n"]
            # Immediately again (same now) → cached, no new open/read.
            out = self.bc._read_side_tile_webcams(now=4000.0)
        self.assertEqual(calls["n"], opens_after_first)    # no re-open
        self.assertIsNotNone(out["left"])


# ══════════════════════════════════════════════════════════════════════════
#  B2 — the air-mouse hand circle: BLUE engaged / ORANGE closed / grey idle
# ══════════════════════════════════════════════════════════════════════════
@requires_monolith
class HandCircleDrawTests(MonolithGlobalsTestCase):
    def _inject_air_mouse_state(self, engaged, grip):
        """Install a fake skill_kinect_air_mouse exposing the state getter +
        colour helper the monolith reads."""
        import sys
        from skills import kinect_air_mouse as real  # the real pure helpers
        sk = types.ModuleType("skill_kinect_air_mouse")
        sk.get_air_mouse_state = lambda: {"engaged": engaged, "grip": grip}
        sk.hand_circle_color_for = real.hand_circle_color_for
        sk.HAND_CIRCLE_COLOR_ENGAGED = real.HAND_CIRCLE_COLOR_ENGAGED
        sk.HAND_CIRCLE_COLOR_CLOSED = real.HAND_CIRCLE_COLOR_CLOSED
        sk.HAND_CIRCLE_COLOR_IDLE = real.HAND_CIRCLE_COLOR_IDLE
        old = sys.modules.get("skill_kinect_air_mouse")
        sys.modules["skill_kinect_air_mouse"] = sk
        self.addCleanup(
            lambda: sys.modules.__setitem__("skill_kinect_air_mouse", old)
            if old is not None else sys.modules.pop("skill_kinect_air_mouse", None))
        return sk

    def _dominant_circle_color(self, before, after):
        """Return the BGR mean of the pixels that CHANGED between two frames —
        i.e. the colour the circle painted."""
        np = _np()
        diff = np.any(after != before, axis=2)
        if not diff.any():
            return None
        changed = after[diff]
        return tuple(int(round(c)) for c in changed.mean(axis=0))  # (B, G, R)

    def test_engaged_open_draws_blue_ring(self):
        np = _np()
        self._inject_air_mouse_state(engaged=True, grip="open")
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        points = {"hand_right": (960, 540)}
        before = color.copy()
        drew = self.bc._draw_hand_circle_on_color(color, points)
        self.assertTrue(drew)
        b, g, r = self._dominant_circle_color(before, color)
        self.assertGreater(b, r)            # BGR blue: B channel dominates
        self.assertGreater(b, g - 1)

    def test_closed_draws_orange_ring(self):
        np = _np()
        self._inject_air_mouse_state(engaged=True, grip="closed")
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        points = {"hand_right": (960, 540)}
        before = color.copy()
        drew = self.bc._draw_hand_circle_on_color(color, points)
        self.assertTrue(drew)
        b, g, r = self._dominant_circle_color(before, color)
        self.assertGreater(r, b)            # BGR orange/amber: R dominates B

    def test_blue_and_orange_are_distinct(self):
        np = _np()
        pts = {"hand_right": (960, 540)}
        c1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._inject_air_mouse_state(engaged=True, grip="open")
        b1 = c1.copy(); self.bc._draw_hand_circle_on_color(c1, pts)
        engaged_col = self._dominant_circle_color(b1, c1)
        c2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._inject_air_mouse_state(engaged=True, grip="closed")
        b2 = c2.copy(); self.bc._draw_hand_circle_on_color(c2, pts)
        closed_col = self._dominant_circle_color(b2, c2)
        self.assertNotEqual(engaged_col, closed_col)

    def test_no_hand_point_draws_nothing(self):
        np = _np()
        self._inject_air_mouse_state(engaged=True, grip="open")
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # No hand/wrist joint among the points → no circle.
        drew = self.bc._draw_hand_circle_on_color(color, {"head": (10, 10)})
        self.assertFalse(drew)
        self.assertEqual(int(color.max()), 0)

    def test_disengaged_still_draws_only_faint_grey(self):
        np = _np()
        self._inject_air_mouse_state(engaged=False, grip="open")
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        points = {"hand_right": (960, 540)}
        before = color.copy()
        drew = self.bc._draw_hand_circle_on_color(color, points)
        # Idle hint is grey: roughly equal channels where it painted.
        if drew:
            b, g, r = self._dominant_circle_color(before, color)
            self.assertAlmostEqual(b, r, delta=25)   # grey ≈ equal channels


# ══════════════════════════════════════════════════════════════════════════
#  B3 — the gesture pop badge: appears on a fresh gesture, gone once it fades
# ══════════════════════════════════════════════════════════════════════════
@requires_monolith
class GesturePopDrawTests(MonolithGlobalsTestCase):
    def _inject_gesture_state(self, gesture, label, ts):
        import sys
        from skills import kinect_gestures as real
        sk = types.ModuleType("skill_kinect_gestures")
        sk.get_last_gesture = lambda: {"gesture": gesture, "label": label,
                                       "ts": ts}
        sk.gesture_pop_alpha = real.gesture_pop_alpha
        sk.GESTURE_POP_TTL_SECONDS = real.GESTURE_POP_TTL_SECONDS
        old = sys.modules.get("skill_kinect_gestures")
        sys.modules["skill_kinect_gestures"] = sk
        self.addCleanup(
            lambda: sys.modules.__setitem__("skill_kinect_gestures", old)
            if old is not None else sys.modules.pop("skill_kinect_gestures", None))
        return sk

    def test_fresh_gesture_draws_a_badge(self):
        import time as _t
        np = _np()
        # Fire 'just now' (monotonic) so the badge is at full opacity.
        self._inject_gesture_state("wave", "WAVE", _t.monotonic())
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        drew = self.bc._draw_gesture_pop_on_color(color, now=_t.monotonic())
        self.assertTrue(drew)
        self.assertGreater(int(color.max()), 0)     # the amber badge was drawn

    def test_stale_gesture_draws_nothing(self):
        import time as _t
        np = _np()
        # Fired well beyond the TTL → fully faded → nothing drawn.
        self._inject_gesture_state("wave", "WAVE", _t.monotonic() - 10.0)
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        drew = self.bc._draw_gesture_pop_on_color(color, now=_t.monotonic())
        self.assertFalse(drew)
        self.assertEqual(int(color.max()), 0)

    def test_no_gesture_yet_draws_nothing(self):
        import time as _t
        np = _np()
        self._inject_gesture_state(None, "", 0.0)
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        drew = self.bc._draw_gesture_pop_on_color(color, now=_t.monotonic())
        self.assertFalse(drew)
        self.assertEqual(int(color.max()), 0)


if __name__ == "__main__":
    unittest.main()
