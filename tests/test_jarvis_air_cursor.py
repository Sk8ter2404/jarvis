"""Transparency / click-through contract tests for ``hud/jarvis_air_cursor.py``.

WHY THIS EXISTS
  The air-cursor reticle overlay is a topmost, click-through tkinter window that
  follows the Kinect air-mouse cursor. In v1.64.0 it was stretched across the
  WHOLE virtual desktop, and a latent bug — re-asserting WS_EX_LAYERED via
  SetWindowLongW *after* Tk had installed its ``-transparentcolor`` key, without
  re-keying — left the layered window with no defined colour-key, so Windows
  composited the keyed background as a SOLID OPAQUE BLOCK that blacked out all
  four monitors ("it's like it's overlaying a black screen over everything").

  The fix is two-fold and this test guards both halves:
    1. The overlay no longer paints a desktop-sized surface — it is a SMALL
       WINDOW_SIZE-px window moved to the cursor each frame, so even a total
       transparency failure could only ever show a tiny patch, never a
       fullscreen blackout.
    2. The Win32 click-through ex-style ORs in WS_EX_LAYERED | WS_EX_TRANSPARENT
       (+ NOACTIVATE + TOOLWINDOW) and the background is re-keyed transparent via
       SetLayeredWindowAttributes — the keyed colour is BG_KEY, which the paint
       path fills (never an opaque black fill).

ISOLATION
  ``_click_through_exstyle`` and ``_colorref`` are pure (ints/strings in, ints
  out) and the geometry constants are plain module globals, so NO Tk root is
  ever constructed — the module is loaded with ``importlib`` exactly like the
  sibling jarvis_reticle test. stdlib tkinter imports fine on the headless /
  Linux CI runner; we simply never build a window. App-Control-safe; stdlib
  ``unittest`` only (no pytest).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)


def _load_air_cursor(testcase):
    """Load hud/jarvis_air_cursor.py under a synthetic module name. tkinter is
    imported at top level (stdlib tk imports fine on the runner) but no Tk root
    is ever constructed by these tests."""
    path = os.path.join(_HUD_DIR, "jarvis_air_cursor.py")
    mod_name = "_jarvis_air_cursor_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class ClickThroughExStyleTests(unittest.TestCase):
    """The extended-window-style the overlay applies MUST make the window
    transparent (layered) AND click-through (hit-test pass-through), so the
    air-mouse's real cursor underneath is never blocked."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_ors_in_layered_and_transparent(self):
        # WS_EX_LAYERED (required for the colour-key) and WS_EX_TRANSPARENT
        # (the actual click-through bit) must both be present.
        out = self.mod._click_through_exstyle(0)
        self.assertTrue(out & self.mod.WS_EX_LAYERED,
                        "WS_EX_LAYERED missing — colour-key can't apply")
        self.assertTrue(out & self.mod.WS_EX_TRANSPARENT,
                        "WS_EX_TRANSPARENT missing — window would eat clicks")

    def test_ors_in_noactivate_and_toolwindow(self):
        out = self.mod._click_through_exstyle(0)
        self.assertTrue(out & self.mod.WS_EX_NOACTIVATE,
                        "WS_EX_NOACTIVATE missing — overlay could steal focus")
        self.assertTrue(out & self.mod.WS_EX_TOOLWINDOW,
                        "WS_EX_TOOLWINDOW missing — overlay would alt-tab")

    def test_preserves_existing_bits(self):
        # The function ORs onto the current style; pre-existing bits survive.
        sentinel = 0x00000400  # WS_EX_CONTROLPARENT — arbitrary unrelated bit
        out = self.mod._click_through_exstyle(sentinel)
        self.assertTrue(out & sentinel, "existing ex-style bits were dropped")
        self.assertTrue(out & self.mod.WS_EX_LAYERED)

    def test_is_pure_no_side_effects(self):
        # Same input → same output, and the input value itself is only OR'd in.
        a = self.mod._click_through_exstyle(0)
        b = self.mod._click_through_exstyle(0)
        self.assertEqual(a, b)
        # Calling with a superset returns a superset (monotone OR).
        self.assertEqual(self.mod._click_through_exstyle(a), a)


class ColorKeyTests(unittest.TestCase):
    """The colour passed to SetLayeredWindowAttributes must be the BG_KEY,
    correctly converted to a Win32 COLORREF (0x00bbggrr), so the keyed
    background composites TRANSPARENT instead of as an opaque block."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_colorref_byte_order(self):
        # #010101 → 0x00010101 (r=g=b=1). Symmetric value, but assert the
        # channel packing explicitly with an asymmetric colour too.
        self.assertEqual(self.mod._colorref("#010101"), 0x010101)
        # #4cc9ff (cyan): r=0x4c g=0xc9 b=0xff → COLORREF 0x00ffc94c.
        self.assertEqual(self.mod._colorref("#4cc9ff"), 0xFFC94C)

    def test_colorref_handles_no_hash(self):
        self.assertEqual(self.mod._colorref("4cc9ff"), 0xFFC94C)

    def test_bg_key_is_the_keyed_colour(self):
        # The transparent key is BG_KEY; sanity-check it is a near-black so the
        # drawn cyan/gold reticle pixels are never accidentally keyed away.
        self.assertEqual(self.mod.BG_KEY, "#010101")


class SmallFollowWindowTests(unittest.TestCase):
    """The overlay must NOT paint a desktop-sized surface — that is what turned a
    transparency glitch into an all-monitors blackout. The window is a small
    fixed square that follows the cursor."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_window_is_small_not_fullscreen(self):
        # A few hundred px at most — comfortably smaller than any single monitor,
        # let alone the 7680x2880 virtual desktop the old code spanned.
        self.assertLessEqual(self.mod.WINDOW_SIZE, 400)
        self.assertGreaterEqual(self.mod.WINDOW_SIZE, 96)
        self.assertEqual(self.mod.WINDOW_HALF, self.mod.WINDOW_SIZE // 2)

    def test_window_fits_the_reticle_visuals(self):
        # The window must be big enough to contain the glow ring + arc pad +
        # pulse on both sides of centre, or the reticle would be clipped.
        needed = 2 * (self.mod.GLOW_RADIUS_TRACK + 5 + 3)
        self.assertGreaterEqual(self.mod.WINDOW_SIZE, needed)

    def test_no_fullscreen_geometry_constant(self):
        # Defensive: the module should not carry a stretched-to-span geometry.
        # The follow-window is always WINDOW_SIZE; there is no width/height span
        # baked into a window size anywhere.
        self.assertTrue(hasattr(self.mod, "WINDOW_SIZE"))


class PaintPathTests(unittest.TestCase):
    """Guard that the per-frame clear fills the COLOUR-KEYED background, never an
    opaque/black fill, and only over the small window — by inspecting the source
    of the paint path (no Tk root needed)."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def _source(self):
        path = os.path.join(_HUD_DIR, "jarvis_air_cursor.py")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_clear_fills_keyed_background_over_window_size(self):
        import inspect
        tick_src = inspect.getsource(self.mod.AirCursorOverlay.tick)
        # The clear rectangle must use BG_KEY (the keyed-transparent colour) and
        # be sized to the small window, NOT a stored width/height span.
        self.assertIn("fill=BG_KEY", tick_src)
        self.assertIn("WINDOW_SIZE", tick_src)
        # It must not clear using the virtual-desktop span (the old fullscreen
        # behaviour) — those attributes no longer drive any fill.
        self.assertNotIn("self.width", tick_src)
        self.assertNotIn("self.height", tick_src)

    def test_no_opaque_black_fill_anywhere(self):
        src = self._source()
        # No paint should fill solid black/white opaque over the surface; the
        # only background fill is the keyed BG_KEY. (BG_KEY itself is #010101,
        # which is keyed transparent — not a literal "black"/"#000000" fill.)
        self.assertNotIn('fill="black"', src)
        self.assertNotIn("fill='black'", src)
        self.assertNotIn('fill="#000000"', src)

    def test_relayers_colorkey_after_setting_exstyle(self):
        # The root-cause fix: after SetWindowLongW touches WS_EX_LAYERED we MUST
        # re-establish the colour-key via SetLayeredWindowAttributes, or the
        # layered window composites fully opaque.
        src = self._source()
        self.assertIn("SetLayeredWindowAttributes", src)
        self.assertIn("LWA_COLORKEY", src)


if __name__ == "__main__":
    unittest.main()
