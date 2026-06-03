"""Regression tests for two runtime bugs JARVIS itself queued from a live
session (2026-06-02):

  1. Printer-status actions (check_print / how_is_the_print / print_details)
     returned a result that was logged but never SPOKEN, because they weren't in
     INFORMATIVE_ACTIONS — so the result->speech follow-up loop never fired
     (unlike check_credits, which is listed and speaks correctly).

  2. Vision-click targeting overshot on a >100%-scaled multi-monitor rig:
     find_click_target added a NATIVE-pixel offset to a LOGICAL monitor origin
     without scaling, so clicks landed too far right/down.

Monolith-tier (full-deps): run locally; skip on the light-deps CI runner.
    python -m unittest tests.monolith.test_monolith_runtime_bugfixes
"""
from __future__ import annotations

import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class PrinterStatusInformativeTests(MonolithGlobalsTestCase):
    def test_printer_status_actions_are_informative(self):
        # Without these in INFORMATIVE_ACTIONS the dispatch follow-up loop breaks
        # immediately and the printer status is logged but never voiced.
        for name in ("check_print", "how_is_the_print", "print_details"):
            self.assertIn(name, self.bc.INFORMATIVE_ACTIONS,
                          f"{name} must be informative so its result is spoken")

    def test_check_credits_still_informative(self):
        # The reference behaviour we're matching — guard against accidental removal.
        self.assertIn("check_credits", self.bc.INFORMATIVE_ACTIONS)


@requires_monolith
class FindClickTargetScalingTests(MonolithGlobalsTestCase):
    @staticmethod
    def _png(w: int, h: int) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (0, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()

    def test_native_pixels_scaled_to_logical_before_adding_origin(self):
        """A target at the native-pixel CENTRE of a 3840x2160 capture of a
        2560x1440-LOGICAL monitor (150% scale) whose origin is (-2560, 0) must
        click at the LOGICAL centre (-1280, 720) — not the un-scaled (-640, 1080)
        the old code produced."""
        bc = self.bc
        png_lowres = self._png(1568, 882)     # Pass-1 downscale (max_dim 1568)
        png_native = self._png(3840, 2160)    # Pass-2 full-res (native pixels)

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            # Pass-1 (full image) -> centre; Pass-2 (the 500x500 crop) -> centre.
            return (784, 441) if (w, h) == (1568, 882) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (-2560, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("a sidebar item", monitor="qa")

        self.assertIsNotNone(pt)
        # native centre (1920,1080) x (2560/3840, 1440/2160) -> (1280,720); + origin
        self.assertAlmostEqual(pt[0], -1280, delta=2)
        self.assertAlmostEqual(pt[1], 720, delta=2)
        # And definitively NOT the old buggy native-added coordinate.
        self.assertNotEqual((pt[0], pt[1]), (-640, 1080))

    def test_no_scale_when_native_equals_logical(self):
        """At 100% scale (native == logical) the scaling is a no-op, so an
        un-scaled single-monitor setup can't regress."""
        bc = self.bc
        png_lowres = self._png(1280, 720)
        png_native = self._png(2560, 1440)   # == logical below

        def fake_shot(monitor=None, max_dim=1568):
            return png_lowres if max_dim <= 1568 else png_native

        def fake_vision(_desc, _png, w, h):
            return (640, 360) if (w, h) == (1280, 720) else (250, 250)

        with mock.patch.dict(bc.MONITORS, {"qa": (0, 0, 2560, 1440)}), \
             mock.patch.object(bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(bc, "_query_vision_for_coords", side_effect=fake_vision):
            pt = bc.find_click_target("x", monitor="qa")

        self.assertIsNotNone(pt)
        # native == logical -> the returned point equals the native refined coord
        # plus the (0,0) origin, i.e. no scale distortion.
        self.assertTrue(0 <= pt[0] <= 2560 and 0 <= pt[1] <= 1440)


if __name__ == "__main__":
    unittest.main()
