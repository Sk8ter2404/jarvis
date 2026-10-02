"""Ctrl+scroll resize on ``hud/jarvis_hud.py`` actually scales the rings.

WHY THIS EXISTS
  Ctrl+mouse-wheel (and the right-click Smaller/Larger items) called
  ``_set_scale``, which resized the window and ran ``canvas.scale()`` ONCE. But
  every tick does ``canvas.delete("all")`` and redraws from the fixed
  HUD_W / HUD_H / R_OUTER / RING_CX constants, so 50 ms later the rings snapped
  back to their base size inside a bigger (or clipped) window: the resize did
  nothing visible (GUI_REVIEW B9). Each frame is now scaled by ``self._scale``
  after it is drawn, and text point sizes follow the same factor.

ISOLATION
  No Tk root is built (the headless CI runner has no display). The stand-in
  ``self`` is a real ``HUD`` made with ``HUD.__new__`` (``__init__`` skipped);
  ``root`` is a MagicMock and ``canvas`` is a small recording canvas that keeps
  every item's coordinates and applies ``scale()`` exactly as Tk does
  (coordinates only, fonts untouched). STATE_FILE / CONFIG_FILE / CONTROL_FILE
  are redirected to a per-test temp dir, so no real project file is touched.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)

# Scalar animation accumulators the render reads (__init__ is skipped).
_RENDER_ATTRS = dict(
    last_cpu=0.0, last_ram=0.0, last_mic=0.0, last_amp=0.0,
    _phase=0.0, _halo_phase=0.0,
    _action_reveal_frame=0, _action_at_start=0.0,
    _focused_window="", role="prod",
    _scale_min=0.5, _scale_max=2.5,
)


class _RecordingCanvas:
    """Keeps the display list like a Tk canvas: create_* append an item,
    delete("all") clears it, scale() rescales every item's coordinates about
    (x0, y0) and leaves fonts alone. Anything else (config, bind...) no-ops."""

    def __init__(self):
        self.items = []

    def _add(self, kind, coords, kw):
        flat = []
        for c in coords:
            flat.extend(c if isinstance(c, (tuple, list)) else [c])
        self.items.append({"kind": kind, "coords": [float(v) for v in flat],
                           "kw": kw})
        return len(self.items)

    def create_oval(self, *c, **kw):
        return self._add("oval", c, kw)

    def create_arc(self, *c, **kw):
        return self._add("arc", c, kw)

    def create_line(self, *c, **kw):
        return self._add("line", c, kw)

    def create_rectangle(self, *c, **kw):
        return self._add("rectangle", c, kw)

    def create_polygon(self, *c, **kw):
        return self._add("polygon", c, kw)

    def create_text(self, *c, **kw):
        return self._add("text", c, kw)

    def delete(self, tag):
        if tag == "all":
            self.items = []

    def scale(self, tag, x0, y0, sx, sy):
        for it in self.items:
            cs = it["coords"]
            for i in range(0, len(cs) - 1, 2):
                cs[i] = x0 + (cs[i] - x0) * sx
                cs[i + 1] = y0 + (cs[i + 1] - y0) * sy

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _load_jarvis_hud(testcase):
    path = os.path.join(_HUD_DIR, "jarvis_hud.py")
    mod_name = "_jarvis_hud_scale_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class JarvisHudScaleTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_jarvis_hud(self)
        self.tmp = tempfile.mkdtemp(prefix="jarvis_hud_scale_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for attr, name in (("STATE_FILE", "hud_state.json"),
                           ("CONFIG_FILE", "hud_config.json"),
                           ("CONTROL_FILE", "jarvis_hud_control.json")):
            p = mock.patch.object(self.mod, attr,
                                  os.path.join(self.tmp, name))
            p.start()
            self.addCleanup(p.stop)
        with open(self.mod.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"visible": True, "state": "Idle"}, f)

    def _hud(self, scale=1.0):
        s = self.mod.HUD.__new__(self.mod.HUD)
        s.parent_pid = 4321
        s._hidden = False
        s._user_hidden = False
        s._prev_state_visible = None
        s._closing = False
        s.frame = 0
        s.root = mock.MagicMock()
        s.root.winfo_x.return_value = 100
        s.root.winfo_y.return_value = 50
        s.canvas = _RecordingCanvas()
        for k, v in _RENDER_ATTRS.items():
            setattr(s, k, v)
        s._scale = scale
        return s

    def _tick(self, hud):
        with mock.patch.object(self.mod, "_is_parent_alive",
                               return_value=True), \
                mock.patch.object(self.mod, "_HAS_PSUTIL", False), \
                mock.patch.object(self.mod, "_HAS_GW", False):
            return self.mod.HUD._tick_body(hud)

    def _outer_ring_width(self, hud):
        """Width of the widest oval on the canvas now: the CPU ring's
        full-circle track (radius R_OUTER), the outermost ring drawn."""
        return max(it["coords"][2] - it["coords"][0]
                   for it in hud.canvas.items if it["kind"] == "oval")

    def _background(self, hud):
        return hud.canvas.items[0]["coords"]

    def test_frame_is_drawn_at_the_persisted_scale(self):
        hud = self._hud(scale=2.0)
        self._tick(hud)
        m = self.mod
        self.assertAlmostEqual(self._outer_ring_width(hud),
                               2 * m.R_OUTER * 2.0, places=3)
        self.assertEqual(self._background(hud),
                         [0.0, 0.0, m.HUD_W * 2.0, m.HUD_H * 2.0])

    def test_ctrl_wheel_resize_survives_the_next_tick(self):
        # The owner-visible bug: one notch up grows the window, then the next
        # tick redrew the rings at their base size.
        hud = self._hud(scale=1.0)
        self._tick(hud)
        base = self._outer_ring_width(hud)
        self.mod.HUD._on_wheel_resize(
            hud, types.SimpleNamespace(delta=120, x_root=0, y_root=0))
        self.assertAlmostEqual(hud._scale, 1.1)
        for _ in range(2):
            self._tick(hud)
            self.assertAlmostEqual(self._outer_ring_width(hud), base * 1.1,
                                   places=3)

    def test_text_sizes_follow_the_scale(self):
        small, big = self._hud(scale=1.0), self._hud(scale=2.0)
        self._tick(small)
        self._tick(big)

        def sizes(h):
            return sorted(it["kw"]["font"][1] for it in h.canvas.items
                          if it["kind"] == "text")

        self.assertTrue(sizes(small))
        self.assertEqual(sizes(big), [s * 2 for s in sizes(small)])

    def test_resizing_an_already_scaled_frame_does_not_compound(self):
        hud = self._hud(scale=1.1)
        self._tick(hud)
        self.mod.HUD._set_scale(hud, 1.2)
        # The frame on screen, rescaled from 1.1 to 1.2 - not 1.1 * 1.2.
        self.assertAlmostEqual(self._outer_ring_width(hud),
                               2 * self.mod.R_OUTER * 1.2, places=3)

    def test_scale_one_draws_the_base_layout(self):
        hud = self._hud(scale=1.0)
        self._tick(hud)
        m = self.mod
        self.assertAlmostEqual(self._outer_ring_width(hud), 2 * m.R_OUTER,
                               places=3)
        self.assertEqual(self._background(hud), [0.0, 0.0, m.HUD_W, m.HUD_H])


if __name__ == "__main__":
    unittest.main()
