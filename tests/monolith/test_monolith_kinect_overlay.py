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

    def test_composes_with_skeleton_and_cached_webcam_tiles(self):
        np = _np()
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        webcam = np.full((480, 640, 3), 123, dtype=np.uint8)
        bodies = [{"joints": {"head": (0.0, 1.0, 2.0, 2),
                              "neck": (0.0, 0.7, 2.0, 2)}}]
        # Seed a cached webcam frame at index 0 and resolve 'left' → 0.
        with self.bc._camera_state_lock:
            self.bc._camera_latest_frame[0] = webcam
        try:
            with mock.patch.object(self.bc._kinect_bridge, "get_color_bgr",
                                   return_value=color), \
                    mock.patch.object(self.bc._kinect_bridge, "get_bodies",
                                      return_value=bodies), \
                    mock.patch.object(self.bc._kinect_bridge,
                                      "get_color_space_mapper",
                                      return_value=lambda x, y, z: (960, 540)), \
                    mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                                      return_value={"left": 0}):
                out = self.bc._compose_kinect_preview(now=10.0)
        finally:
            with self.bc._camera_state_lock:
                self.bc._camera_latest_frame.pop(0, None)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, color.shape)

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
                mock.patch.object(self.bc, "_resolve_webcam_indices_by_name",
                                  return_value={}):
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


if __name__ == "__main__":
    unittest.main()
