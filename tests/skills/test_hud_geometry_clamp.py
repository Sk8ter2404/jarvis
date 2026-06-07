"""Headless tests for the unified HUD's saved-geometry validation in
``hud/jarvis_unified_hud.py``.

WHY THIS EXISTS
  The unified HUD is a frameless, no-taskbar window whose position + size are
  persisted to ``unified_hud_geometry.json`` and restored on the next launch.
  The restore used to trust that JSON as-is — no max-size and no on-screen
  check — so after a monitor-layout change (an unplugged display, a resolution
  swap) the saved rect could land FULLY off every screen. With no frame and no
  taskbar button the HUD was then unreachable; the only recovery was to hand-
  delete the geometry file.

  ``_validate_geometry`` closes that hole: it clamps the restored width/height
  to ``[MIN_*, MAX_*]`` and, if the (size-clamped) rect overlaps no available
  screen, snaps its top-left back to the CLI anchor. It is pure arithmetic —
  no Qt, no display — so we load the HUD source with PyQt6 genuinely ABSENT
  (its own ``except ImportError`` stub path makes the Qt names harmless and
  ``_HAS_PYQT6`` False) and call the helper directly. This mirrors the load
  harness in ``tests/skills/test_hud_camera_preview.py``.

ISOLATION
  • PyQt6 imports are blocked for the duration of the load (a fake
    ``__import__`` raises ImportError for the ``PyQt6`` package), restored on
    cleanup; any previously-cached PyQt6 submodules are hidden during the load
    and restored after, so a dev box that *has* PyQt6 still exercises the
    headless path.
  • The source is loaded from its file path under a synthetic module name and
    dropped on cleanup → pristine module globals. No project file is read or
    written (the helpers under test touch only their arguments).

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hud",
)


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source from hud/<filename> with PyQt6 import blocked, so the
    module takes its graceful-degrade path (``_HAS_PYQT6`` False, Qt names
    stubbed). Restores sys.modules + the real importer on cleanup."""
    path = os.path.join(_HUD_DIR, filename)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] == "PyQt6":
            raise ImportError(f"[test] PyQt6 blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n)
              for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    with mock.patch("builtins.__import__", side_effect=_imp):
        spec.loader.exec_module(module)
    testcase.assertFalse(module._HAS_PYQT6,
                         "PyQt6 should be blocked → headless degrade path")
    return module


class UnifiedHudGeometryClampTests(unittest.TestCase):
    MOD_NAME = "_ju_hud_geomclamp_under_test"

    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, "jarvis_unified_hud.py", self.MOD_NAME)
        # A single 2560x1440 screen at the origin for the on/off-screen tests.
        self.screens = [(0, 0, 2560, 1440)]
        self.default_xy = (2280, -1400)   # the module's CLI default anchor

    # ── the core contract: off-screen snaps back, on-screen is preserved ──────
    def test_offscreen_rect_snaps_back_to_anchor(self):
        # Saved rect sits far to the right of the only screen (a monitor that has
        # since been unplugged) → off EVERY screen → reset to the CLI anchor,
        # while the (in-bounds) size is preserved exactly.
        geo = {"x": 9000, "y": 5000, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), self.default_xy)
        self.assertEqual((out["w"], out["h"]), (420, 560))

    def test_onscreen_rect_is_preserved(self):
        # Fully inside the screen → left exactly where the user put it.
        geo = {"x": 100, "y": 200, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual(out, {"x": 100, "y": 200, "w": 420, "h": 560})

    def test_partially_onscreen_rect_is_preserved(self):
        # Straddling the right bezel — still has real pixels on the screen, so it
        # counts as reachable and the position is NOT moved (intentional placement
        # across a monitor edge must survive).
        geo = {"x": 2400, "y": 100, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), (2400, 100))

    def test_negative_origin_screen_keeps_rect(self):
        # The shipped MONITORS 'top' panel lives at negative y. A rect on a
        # negative-origin screen must be recognised as on-screen, not snapped.
        screens = [(0, -1440, 2560, 1440)]
        geo = {"x": 200, "y": -1300, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), (200, -1300))

    # ── size clamping ─────────────────────────────────────────────────────────
    def test_oversized_rect_is_clamped_to_max(self):
        # A stale geometry from a since-removed huge monitor must not resurrect a
        # monster bar even though its top-left is on-screen.
        geo = {"x": 0, "y": 0, "w": 99999, "h": 99999}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual(out["w"], self.mod.MAX_W)
        self.assertEqual(out["h"], self.mod.MAX_H)
        # Top-left still on-screen → position preserved.
        self.assertEqual((out["x"], out["y"]), (0, 0))

    def test_undersized_rect_is_clamped_to_min(self):
        # (_load_saved_geometry already rejects sub-MIN sizes, but the validator
        # must be self-contained.) A tiny saved size floors at MIN_*.
        geo = {"x": 10, "y": 10, "w": 1, "h": 1}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual(out["w"], self.mod.MIN_W)
        self.assertEqual(out["h"], self.mod.MIN_H)

    def test_clamped_size_changes_offscreen_verdict(self):
        # The on-screen test uses the CLAMPED size, not the raw one: a rect whose
        # huge raw width would overlap the screen but whose clamped width does not
        # is treated as off-screen and snapped back.
        # Screen spans x=[0,2560). Put the left edge well to the right of it; the
        # raw width (8000) would reach back over the screen, the clamped MAX_W
        # (900) cannot.
        geo = {"x": 4000, "y": 100, "w": 8000, "h": 560}
        out = self.mod._validate_geometry(geo, self.screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), self.default_xy)
        self.assertEqual(out["w"], self.mod.MAX_W)

    # ── union of multiple screens ─────────────────────────────────────────────
    def test_on_second_screen_of_union_is_preserved(self):
        # Two screens side by side; a rect on the SECOND one is on-screen.
        screens = [(0, 0, 2560, 1440), (2560, 0, 2560, 1440)]
        geo = {"x": 3000, "y": 100, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), (3000, 100))

    def test_in_gap_between_screens_snaps_back(self):
        # A non-contiguous layout (a gap between two monitors) — a rect fully in
        # the gap overlaps neither screen → snap back.
        screens = [(0, 0, 1920, 1080), (4000, 0, 1920, 1080)]
        geo = {"x": 2500, "y": 100, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, screens, self.default_xy)
        self.assertEqual((out["x"], out["y"]), self.default_xy)

    # ── degenerate inputs ─────────────────────────────────────────────────────
    def test_empty_screen_list_skips_onscreen_test(self):
        # No screens known (Qt + MONITORS both unavailable) → only the size is
        # clamped; position is trusted (better than refusing to show at all).
        geo = {"x": 9000, "y": 9000, "w": 420, "h": 560}
        out = self.mod._validate_geometry(geo, [], self.default_xy)
        self.assertEqual((out["x"], out["y"]), (9000, 9000))
        self.assertEqual((out["w"], out["h"]), (420, 560))

    # ── the _rects_overlap primitive ──────────────────────────────────────────
    def test_rects_overlap_true_when_sharing_area(self):
        self.assertTrue(self.mod._rects_overlap(0, 0, 100, 100, 50, 50, 100, 100))

    def test_rects_overlap_false_when_disjoint(self):
        self.assertFalse(self.mod._rects_overlap(0, 0, 100, 100, 200, 0, 100, 100))

    def test_rects_overlap_edge_touch_is_not_overlap(self):
        # Sharing only an edge (x reaches exactly the neighbour's left) is zero
        # area → NOT reachable.
        self.assertFalse(self.mod._rects_overlap(0, 0, 100, 100, 100, 0, 100, 100))

    # ── constant sanity ───────────────────────────────────────────────────────
    def test_max_bounds_are_above_min_bounds(self):
        self.assertGreater(self.mod.MAX_W, self.mod.MIN_W)
        self.assertGreater(self.mod.MAX_H, self.mod.MIN_H)

    def test_max_bounds_match_cli_default_clamp(self):
        # The validator's MAX_* must be the same 900x1100 the CLI-default path
        # clamps to, so the two restore paths agree.
        self.assertEqual((self.mod.MAX_W, self.mod.MAX_H), (900, 1100))


if __name__ == "__main__":
    unittest.main()
