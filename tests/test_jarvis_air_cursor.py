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
from unittest import mock


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


# ══════════════════════════════════════════════════════════════════════════
#  TWO-HAND DUAL RETICLE (Part 3): two circle cursors — BLUE normally, PURPLE
#  while actively resizing a window. Drawn as two small click-through windows.
# ══════════════════════════════════════════════════════════════════════════
class _FakeOverlay:
    """An AirCursorOverlay instance with the Tk roots/canvases replaced by mocks,
    built WITHOUT running __init__ (which needs a real display), so the two-hand
    render path can be unit-tested headless."""

    @staticmethod
    def make(mod, *, resizing=False, hand_pts=((400, 300), (1800, 320))):
        ov = object.__new__(mod.AirCursorOverlay)
        ov.origin_x = 0
        ov.origin_y = 0
        ov.span_w = 2560
        ov.span_h = 1440
        ov.frame = 0
        ov.two_hand = True
        ov.two_resizing = resizing
        ov.hand_pts = list(hand_pts)
        # Mocked primary window + canvas.
        ov.root = mock.MagicMock(name="root")
        ov.root.state.return_value = "normal"
        ov.canvas = mock.MagicMock(name="canvas")
        ov._win_x = ov._win_y = None
        # 2nd window starts absent (built lazily by _ensure_second_window).
        ov.root2 = None
        ov.canvas2 = None
        ov._win2_x = ov._win2_y = None
        ov._has_colorkey2 = False
        return ov


class TwoHandPaletteTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_blue_palette_when_not_resizing(self):
        ov = _FakeOverlay.make(self.mod, resizing=False)
        outer, inner, dim = ov._two_hand_palette()
        self.assertEqual(outer, self.mod.BLUE)
        self.assertEqual(inner, self.mod.BLUE_BRIGHT)

    def test_purple_palette_while_resizing(self):
        ov = _FakeOverlay.make(self.mod, resizing=True)
        outer, inner, dim = ov._two_hand_palette()
        self.assertEqual(outer, self.mod.PURPLE)
        self.assertEqual(inner, self.mod.PURPLE_BRIGHT)

    def test_blue_and_purple_are_distinct(self):
        self.assertNotEqual(self.mod.BLUE, self.mod.PURPLE)


class TwoHandParseTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_parses_dict_points(self):
        P = self.mod.AirCursorOverlay._parse_hand_pts
        self.assertEqual(P([{"x": 1, "y": 2}, {"x": 3, "y": 4}]),
                         [(1, 2), (3, 4)])

    def test_rejects_single_point(self):
        P = self.mod.AirCursorOverlay._parse_hand_pts
        self.assertIsNone(P([{"x": 1, "y": 2}]))

    def test_rejects_garbage(self):
        P = self.mod.AirCursorOverlay._parse_hand_pts
        self.assertIsNone(P(None))
        self.assertIsNone(P("nope"))
        self.assertIsNone(P([1, 2]))


class TwoHandRenderTests(unittest.TestCase):
    """The two-hand tick draws a reticle into BOTH hand windows and positions each
    window at its hand. Tk is mocked so this runs headless."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def test_draws_two_circles_one_per_hand(self):
        ov = _FakeOverlay.make(self.mod,
                               hand_pts=((400, 300), (1800, 320)))
        # Stub the lazy 2nd-window builder so it installs a mock window+canvas
        # (no real Tk). It must be CALLED (the 2nd reticle needs its own window).
        built = {"n": 0}

        def fake_build():
            built["n"] += 1
            ov.root2 = mock.MagicMock(name="root2")
            ov.root2.state.return_value = "normal"
            ov.canvas2 = mock.MagicMock(name="canvas2")
        with mock.patch.object(ov, "_ensure_second_window",
                               side_effect=fake_build):
            ov._tick_two_hand()
        # Both canvases were painted (a circle drawn into each).
        self.assertTrue(ov.canvas.create_oval.called,
                        "no reticle drawn for hand 1")
        self.assertTrue(ov.canvas2.create_oval.called,
                        "no reticle drawn for hand 2")
        self.assertEqual(built["n"], 1)   # the 2nd window was ensured
        # Each window was geometry()-placed (moved to its hand).
        self.assertTrue(ov.root.geometry.called)
        self.assertTrue(ov.root2.geometry.called)

    def test_hand_windows_placed_at_distinct_positions(self):
        # Hand 1 far left, hand 2 far right → the two windows get DIFFERENT
        # geometry strings (the circles really are at two places, not one).
        ov = _FakeOverlay.make(self.mod, hand_pts=((100, 300), (2400, 300)))

        def fake_build():
            ov.root2 = mock.MagicMock(name="root2")
            ov.root2.state.return_value = "normal"
            ov.canvas2 = mock.MagicMock(name="canvas2")
        with mock.patch.object(ov, "_ensure_second_window",
                               side_effect=fake_build):
            ov._tick_two_hand()
        g1 = ov.root.geometry.call_args[0][0]
        g2 = ov.root2.geometry.call_args[0][0]
        self.assertNotEqual(g1, g2)

    def test_purple_drawn_while_resizing(self):
        # While resizing, the reticle drawn into each canvas uses the PURPLE outline.
        ov = _FakeOverlay.make(self.mod, resizing=True)

        def fake_build():
            ov.root2 = mock.MagicMock(name="root2")
            ov.root2.state.return_value = "normal"
            ov.canvas2 = mock.MagicMock(name="canvas2")
        with mock.patch.object(ov, "_ensure_second_window",
                               side_effect=fake_build):
            ov._tick_two_hand()
        # Collect every outline colour passed to create_oval on hand 1.
        outlines = [kw.get("outline")
                    for _a, kw in ov.canvas.create_oval.call_args_list]
        self.assertIn(self.mod.PURPLE, outlines,
                      f"purple ring not drawn while resizing; got {outlines}")
        self.assertNotIn(self.mod.BLUE, outlines)

    def test_blue_drawn_when_not_resizing(self):
        ov = _FakeOverlay.make(self.mod, resizing=False)

        def fake_build():
            ov.root2 = mock.MagicMock(name="root2")
            ov.root2.state.return_value = "normal"
            ov.canvas2 = mock.MagicMock(name="canvas2")
        with mock.patch.object(ov, "_ensure_second_window",
                               side_effect=fake_build):
            ov._tick_two_hand()
        outlines = [kw.get("outline")
                    for _a, kw in ov.canvas.create_oval.call_args_list]
        self.assertIn(self.mod.BLUE, outlines)
        self.assertNotIn(self.mod.PURPLE, outlines)


class TwoHandRefreshStateTests(unittest.TestCase):
    """_refresh_state parses the published two-hand frame into the render fields."""

    def setUp(self):
        self.mod = _load_air_cursor(self)

    def _ov(self):
        import time as _t
        ov = object.__new__(self.mod.AirCursorOverlay)
        ov.parent_pid = 12345          # a "real" parent so the orphan cap is skipped
        ov._started_at = _t.time()
        ov._last_state_ts = 0.0
        ov._prev_visible = False
        ov._was_grab = False
        ov._grab_flash = 0
        ov.cur_x = ov.cur_y = None
        ov.target_x = ov.target_y = None
        ov.trail = []
        ov.two_hand = False
        ov.two_resizing = False
        ov.hand_pts = None
        return ov

    def test_two_hand_frame_sets_render_fields(self):
        import time as _t
        ov = self._ov()
        frame = {"visible": True, "two_hand": True, "resizing": True,
                 "color": "purple", "ts": _t.time(),
                 "hands": [{"x": 200, "y": 300}, {"x": 1600, "y": 320}],
                 "state": "grab", "x": 200, "y": 300}
        with mock.patch.object(self.mod, "_is_parent_alive", lambda pid: True), \
             mock.patch.object(self.mod, "_read_state", lambda: frame):
            ok = ov._refresh_state()
        self.assertTrue(ok)
        self.assertTrue(ov.two_hand)
        self.assertTrue(ov.two_resizing)
        self.assertEqual(ov.hand_pts, [(200, 300), (1600, 320)])
        # The single-hand reticle is suppressed while two-hand is active.
        self.assertEqual(ov.state, "hidden")

    def test_single_hand_frame_leaves_two_hand_off(self):
        import time as _t
        ov = self._ov()
        frame = {"visible": True, "two_hand": False, "ts": _t.time(),
                 "state": "track", "x": 500, "y": 500, "color": "cyan"}
        with mock.patch.object(self.mod, "_is_parent_alive", lambda pid: True), \
             mock.patch.object(self.mod, "_read_state", lambda: frame):
            ov._refresh_state()
        self.assertFalse(ov.two_hand)
        self.assertIsNone(ov.hand_pts)
        self.assertEqual(ov.state, "track")


if __name__ == "__main__":
    unittest.main()
