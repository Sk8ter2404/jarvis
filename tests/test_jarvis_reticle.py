"""Work-area (taskbar-aware) positioning tests for ``hud/jarvis_reticle.py``.

WHY THIS EXISTS
  The reticle overlay is a full-virtual-screen, topmost, click-through tkinter
  window. The Windows taskbar is topmost too, so the band of the overlay that
  overlaps the taskbar is occluded: a reticle drawn for a click near the bottom
  screen edge was sliced off behind the tray (confirmed by screenshot — only
  the top sliver showed). The fix makes the overlay taskbar-aware by trimming
  its bottom edge to the primary monitor's *work area* (top of the taskbar)
  instead of spanning the full screen, so bottom-edge reticles render fully
  above the tray. This guards the pure geometry math behind that fix.

  shrinks past taskbar → height trimmed so the window ends at the work-area top.
  no work area known   → geometry returned verbatim (query failed / non-Win).
  no overlap           → geometry untouched (taskbar already below the window).
  top/x/width fixed    → only the height ever changes; the top never moves and
                         the height is never grown.

ISOLATION
  ``_clamp_to_work_area`` is pure (ints in, ints out) and
  ``_primary_work_area_bottom`` is best-effort and side-effect-free, so no Tk
  root is ever constructed — the module is loaded with ``importlib`` exactly
  like the sibling jarvis_hud test (stdlib tkinter imports fine on the headless
  runner; we simply never build a window). On the Linux CI runner
  ``_primary_work_area_bottom`` takes its non-Windows guard and returns None.

stdlib ``unittest`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)


def _load_reticle(testcase):
    """Load hud/jarvis_reticle.py under a synthetic module name. tkinter is not
    blocked — the module imports it at top level (stdlib tk imports fine on the
    runner) but no Tk root is ever constructed by these tests."""
    path = os.path.join(_HUD_DIR, "jarvis_reticle.py")
    mod_name = "_jarvis_reticle_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class ClampToWorkAreaTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_reticle(self)

    def test_trims_height_to_taskbar_top(self):
        # 1440-tall primary screen with a 48px taskbar → work area ends at 1392.
        # The overlay must stop there, not span the full 1440 behind the tray.
        x, y, w, h = self.mod._clamp_to_work_area(0, 0, 2560, 1440, 1392)
        self.assertEqual((x, y, w), (0, 0, 2560))
        self.assertEqual(h, 1392)

    def test_none_work_area_returns_geometry_verbatim(self):
        # Query failed / non-Windows → leave the full-screen geometry untouched.
        geom = (0, 0, 2560, 1440)
        self.assertEqual(self.mod._clamp_to_work_area(*geom, None), geom)

    def test_no_overlap_leaves_geometry_untouched(self):
        # Taskbar top at/below the window bottom → nothing to trim.
        geom = (0, 0, 2560, 1440)
        self.assertEqual(self.mod._clamp_to_work_area(*geom, 1440), geom)
        self.assertEqual(self.mod._clamp_to_work_area(*geom, 2000), geom)

    def test_never_moves_top_or_grows_height(self):
        # Top edge, x and width are invariant; height only ever shrinks.
        x, y, w, h = self.mod._clamp_to_work_area(0, 0, 2560, 1440, 1392)
        self.assertEqual(y, 0)               # top never moves
        self.assertLessEqual(h, 1440)        # never grows
        self.assertEqual((x, w), (0, 2560))  # x / width untouched

    def test_multimonitor_negative_origin_span(self):
        # A virtual desktop whose primary taskbar sits at y=1392 inside a span
        # that starts on a monitor placed above the primary (negative origin):
        # only the bottom is trimmed, the negative origin is preserved.
        x, y, w, h = self.mod._clamp_to_work_area(-2560, -120, 7680, 1560, 1392)
        self.assertEqual((x, y, w), (-2560, -120, 7680))
        self.assertEqual(h, 1392 - (-120))   # ends exactly at the taskbar top

    def test_floor_keeps_canvas_at_least_one_pixel(self):
        # Degenerate input (taskbar one pixel below the top) still yields a
        # construct-safe, positive height.
        _, _, _, h = self.mod._clamp_to_work_area(0, 0, 2560, 1440, 1)
        self.assertGreaterEqual(h, 1)


class PrimaryWorkAreaBottomTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_reticle(self)

    def test_contract_is_int_or_none(self):
        val = self.mod._primary_work_area_bottom()
        self.assertTrue(val is None or isinstance(val, int))

    def test_non_windows_returns_none(self):
        # On the Linux CI runner (and the ci-sim, which flips sys.platform to
        # "linux") the non-Windows guard fires and no ctypes.windll is touched.
        if sys.platform != "win32":
            self.assertIsNone(self.mod._primary_work_area_bottom())


if __name__ == "__main__":
    unittest.main()
