"""The focused-window capture honours a full-resolution cap (NEW #11, 2026-10-02).

see_screen's "read this page" path now captures only the focused window
(core/actions._see_screen_focused_window). The capture it reuses,
_capture_focused_window_png, hard-coded a 1568-px downscale (the glance
size), so a maximised window on a 2560-px monitor still lost 40 % of its
pixels - the page text the owner asked about. It now takes ``max_dim``; the
glance keeps 1568. mss and the window rect are faked; no screen is read.

    python -m unittest tests.monolith.test_monolith_focused_window_capture
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)

W, H = 2576, 1408      # a maximised window on a 2560x1440 monitor (frame incl.)


class _Grab:
    def __init__(self, w, h):
        self.size = (w, h)
        self.bgra = bytes(w * h * 4)


class _FakeSct:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self, region):
        return _Grab(region["width"], region["height"])


@requires_monolith
class FocusedWindowCaptureSizeTests(MonolithGlobalsTestCase):

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _capture(self, **kw):
        bc = self.bc
        fake_mss = types.ModuleType("mss")
        fake_mss.MSS = _FakeSct
        fake_mss.mss = _FakeSct
        with mock.patch.dict(sys.modules, {"mss": fake_mss}), \
             mock.patch.object(bc, "screenshot_privacy_block_reason",
                               return_value=None), \
             mock.patch.object(bc, "_privacy_blocklist_match",
                               return_value=None), \
             mock.patch.dict(bc._focused_window_state,
                             {"rect": (0, 0, W, H), "title": "Article"}), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                png = bc._capture_focused_window_png(**kw)
            except TypeError as e:
                self.fail(f"the capture takes no size cap ({e}) - it is always "
                          "downscaled to the 1568-px glance size")
        self.assertIsInstance(png, bytes)
        from PIL import Image
        return Image.open(io.BytesIO(png)).size

    def test_full_resolution_cap_keeps_every_pixel(self):
        self.assertEqual(self._capture(max_dim=2600), (W, H))

    def test_the_glance_default_is_unchanged(self):
        w, h = self._capture()
        self.assertEqual(max(w, h), 1568)

    def test_a_junk_cap_falls_back_to_the_glance_size(self):
        w, h = self._capture(max_dim="big")
        self.assertEqual(max(w, h), 1568)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
