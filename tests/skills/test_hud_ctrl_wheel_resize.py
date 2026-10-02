"""Ctrl+mouse-wheel resizes the live HUD (``hud/jarvis_unified_hud.py``).

WHY THIS EXISTS
  v2.0.165 made Ctrl+scroll resize the OLD corner HUD (``hud/jarvis_hud.py``,
  ``tests/test_jarvis_hud_scale.py``), but the monolith has launched the
  unified HUD since 2026-05-30, and that one could only be resized by its
  corner grip: a Ctrl+wheel did nothing at all, while the prompt told the
  model "Ctrl+wheel to resize". Now:
    * Ctrl+wheel grows / shrinks the window by WHEEL_STEP per notch, aspect
      ratio kept, and the rings, arcs and text scale with it (paintEvent
      lays everything out from ``_paint_scale(width)``);
    * the width stays inside WHEEL_MIN_W..WHEEL_MAX_W (the widths where the
      drawing scale follows the window) and the height inside MIN_H..MAX_H;
    * the size persists the way a grip size does (resize -> resizeEvent ->
      the debounced ``_save_geometry`` -> unified_hud_geometry.json);
    * a plain wheel (no Ctrl) still goes to QWidget, as before.

THREE LAYERS
  * the pure size arithmetic ``_ctrl_wheel_size`` / ``_paint_scale``;
  * the ``wheelEvent`` handler, headless: the module is loaded with PyQt6
    blocked (its own ImportError stub path), the HUD is ``object.__new__``'d
    and driven with fake events and a fake ``Qt`` namespace;
  * a REAL offscreen render (only where PyQt6 is installed, the local full
    tier): a subprocess builds the widget on the offscreen platform, sends
    real ``QWheelEvent``s, records the reactor radius and draw scale each
    paint uses, and reads back the geometry file the debounced save wrote. A
    subprocess keeps Qt out of the test process entirely.

ISOLATION
  No real window, display, file or device: the offscreen subprocess repoints
  every file the HUD reads or writes into a per-test temp dir and stops the
  data timers before the first tick (so no nvidia-smi poll runs).

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_HUD_DIR = os.path.join(_PROJECT_DIR, "hud")
_HUD_FILE = "jarvis_unified_hud.py"

NOTCH = 120


def _load_hud_no_pyqt(testcase, mod_name):
    """Load hud/jarvis_unified_hud.py with PyQt6 blocked, so the module takes
    its headless degrade path (``_HAS_PYQT6`` False, Qt names stubbed).
    sys.modules and the real importer are restored on cleanup."""
    path = os.path.join(_HUD_DIR, _HUD_FILE)
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
    testcase.assertFalse(module._HAS_PYQT6)
    return module


# ════════════════════════════════════════════════════════════════════════════
#  The size arithmetic
# ════════════════════════════════════════════════════════════════════════════
class CtrlWheelSizeTests(unittest.TestCase):
    def setUp(self):
        self.m = _load_hud_no_pyqt(self, "_ju_wheel_size_ut")

    def _walk(self, w, h, delta, limit=60):
        """Every size a run of identical wheel events visits, until the
        helper reports no change."""
        seq = [(w, h)]
        for _ in range(limit):
            nxt = self.m._ctrl_wheel_size(w, h, delta)
            if nxt is None:
                return seq
            w, h = nxt
            seq.append(nxt)
        self.fail(f"the wheel never reached a bound: {seq[-3:]}")

    def test_one_notch_up_grows_ten_percent_keeping_the_aspect(self):
        self.assertEqual(self.m._ctrl_wheel_size(420, 560, NOTCH), (462, 616))

    def test_one_notch_down_shrinks_and_undoes_a_notch_up(self):
        self.assertEqual(self.m._ctrl_wheel_size(420, 560, -NOTCH), (382, 509))
        self.assertEqual(self.m._ctrl_wheel_size(462, 616, -NOTCH), (420, 560))

    def test_growing_stops_exactly_at_the_max_width(self):
        seq = self._walk(420, 560, NOTCH)
        self.assertEqual(seq[-1][0], self.m.WHEEL_MAX_W)
        for w, h in seq:
            self.assertLessEqual(w, self.m.WHEEL_MAX_W)
            self.assertLessEqual(h, self.m.MAX_H)
        # Every step strictly grows both sides, aspect within rounding.
        for (w0, h0), (w1, h1) in zip(seq, seq[1:]):
            self.assertGreater(w1, w0)
            self.assertGreater(h1, h0)
            self.assertAlmostEqual(h1 / w1, 560 / 420, delta=0.01)

    def test_shrinking_stops_exactly_at_the_min_width(self):
        seq = self._walk(420, 560, -NOTCH)
        self.assertEqual(seq[-1][0], self.m.WHEEL_MIN_W)
        for w, h in seq:
            self.assertGreaterEqual(w, self.m.WHEEL_MIN_W)
            self.assertGreaterEqual(h, self.m.MIN_H)
            self.assertGreaterEqual(w, self.m.MIN_W)

    def test_height_bounds_hold_for_a_tall_window(self):
        m = self.m
        seq = self._walk(500, 1050, NOTCH)
        self.assertEqual(seq[-1][1], m.MAX_H)
        self.assertLess(seq[-1][0], m.WHEEL_MAX_W)
        seq = self._walk(400, 400, -NOTCH)
        self.assertEqual(seq[-1][1], m.MIN_H)

    def test_a_touchpad_delta_is_a_fraction_of_a_notch(self):
        m = self.m
        w, h = m._ctrl_wheel_size(420, 560, 30)
        self.assertGreater(w, 420)
        self.assertLess(w, 462)
        # Four quarter-notches land where one full notch does (+-rounding).
        for _ in range(3):
            w, h = m._ctrl_wheel_size(w, h, 30)
        self.assertAlmostEqual(w, 462, delta=2)
        self.assertAlmostEqual(h, 616, delta=2)

    def test_huge_deltas_clamp_instead_of_raising(self):
        m = self.m
        w, h = m._ctrl_wheel_size(420, 560, 10 ** 9)
        self.assertEqual(w, m.WHEEL_MAX_W)
        self.assertLessEqual(h, m.MAX_H)
        w, h = m._ctrl_wheel_size(420, 560, -10 ** 9)
        self.assertEqual(w, m.WHEEL_MIN_W)
        self.assertGreaterEqual(h, m.MIN_H)

    def test_no_change_reports_none(self):
        m = self.m
        for args in ((420, 560, 0), (420, 560, 0.0), (420, 560, None),
                     (420, 560, "up"), (0, 560, NOTCH), (420, -1, NOTCH),
                     (420, 560, float("nan")), (420, 560, float("inf")),
                     (m.WHEEL_MAX_W, 900, NOTCH),
                     (m.WHEEL_MIN_W, 500, -NOTCH)):
            with self.subTest(args=args):
                self.assertIsNone(m._ctrl_wheel_size(*args))

    def test_a_grip_size_past_a_bound_is_never_snapped(self):
        # The corner grip has no maximum. Ctrl+wheel up there does nothing;
        # Ctrl+wheel down shrinks one normal step toward the range.
        m = self.m
        self.assertIsNone(m._ctrl_wheel_size(850, 1000, NOTCH))
        self.assertEqual(m._ctrl_wheel_size(850, 1000, -NOTCH), (773, 909))
        self.assertIsNone(m._ctrl_wheel_size(500, 1200, NOTCH))

    def test_every_wheel_step_scales_the_drawing_with_the_window(self):
        # The point of the port (and of the v2.0.165 fix to the old HUD): a
        # notch must scale the rings and text, not just the window. Across
        # the whole wheel range the paint scale is the window's own ratio,
        # never pinned by the paint clamp.
        m = self.m
        sizes = (list(reversed(self._walk(420, 560, -NOTCH)))
                 + self._walk(420, 560, NOTCH)[1:])
        for (w0, _h0), (w1, _h1) in zip(sizes, sizes[1:]):
            with self.subTest(step=(w0, w1)):
                self.assertAlmostEqual(m._paint_scale(w1) / m._paint_scale(w0),
                                       w1 / w0, places=9)

    def test_wheel_bounds_sit_inside_the_window_and_paint_bounds(self):
        m = self.m
        self.assertGreaterEqual(m.WHEEL_MIN_W, m.MIN_W)
        self.assertLessEqual(m.WHEEL_MAX_W, m.MAX_W)
        self.assertLess(m.WHEEL_MIN_W, 420)
        self.assertGreater(m.WHEEL_MAX_W, 420)
        self.assertGreaterEqual(m.WHEEL_MIN_W / m.PAINT_BASE_W, m.PAINT_S_MIN)
        self.assertLessEqual(m.WHEEL_MAX_W / m.PAINT_BASE_W, m.PAINT_S_MAX)

    def test_paint_scale_keeps_the_original_clamp(self):
        m = self.m
        self.assertEqual(m._paint_scale(420), 1.0)
        self.assertEqual(m._paint_scale(100), 0.78)
        self.assertEqual(m._paint_scale(5000), 1.7)

    def test_paint_event_takes_its_scale_from_the_shared_helper(self):
        # One copy of the clamp: the wheel range is derived from the same
        # constants, so the two cannot drift apart.
        src = textwrap.dedent(inspect.getsource(self.m.UnifiedHud.paintEvent))
        calls = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_paint_scale"]
        self.assertEqual(len(calls), 1)
        consts = {n.value for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Constant)}
        self.assertNotIn(0.78, consts)
        self.assertNotIn(1.7, consts)


# ════════════════════════════════════════════════════════════════════════════
#  The wheelEvent handler, headless
# ════════════════════════════════════════════════════════════════════════════
_CTRL, _SHIFT, _ALT = 0x04000000, 0x02000000, 0x08000000
_FAKE_QT = types.SimpleNamespace(KeyboardModifier=types.SimpleNamespace(
    NoModifier=0, ControlModifier=_CTRL, ShiftModifier=_SHIFT,
    AltModifier=_ALT))


class _Delta:
    def __init__(self, dy):
        self._dy = dy

    def x(self):
        return 0

    def y(self):
        return self._dy


class _FakeWheel:
    def __init__(self, dy, mods):
        self._dy, self._mods = dy, mods
        self.accepted = None

    def modifiers(self):
        return self._mods

    def angleDelta(self):
        return _Delta(self._dy)

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class WheelEventHandlerTests(unittest.TestCase):
    def setUp(self):
        self.m = _load_hud_no_pyqt(self, "_ju_wheel_handler_ut")
        p = mock.patch.object(self.m, "Qt", _FAKE_QT)
        p.start()
        self.addCleanup(p.stop)
        # The QWidget stub stands in for the Qt base class: a plain wheel
        # must reach its wheelEvent, exactly as it did before the port.
        self.base_wheel = mock.Mock(name="QWidget.wheelEvent")
        p = mock.patch.object(self.m.QWidget, "wheelEvent", self.base_wheel,
                              create=True)
        p.start()
        self.addCleanup(p.stop)

    def _hud(self, w=420, h=560):
        hud = object.__new__(self.m.UnifiedHud)
        hud.width = lambda: w
        hud.height = lambda: h
        hud.resize = mock.Mock(name="resize")
        return hud

    def test_ctrl_wheel_up_grows_the_window(self):
        hud, ev = self._hud(), _FakeWheel(NOTCH, _CTRL)
        hud.wheelEvent(ev)
        hud.resize.assert_called_once_with(462, 616)
        self.assertTrue(ev.accepted)
        self.base_wheel.assert_not_called()

    def test_ctrl_wheel_down_shrinks_the_window(self):
        hud, ev = self._hud(), _FakeWheel(-NOTCH, _CTRL)
        hud.wheelEvent(ev)
        hud.resize.assert_called_once_with(382, 509)
        self.assertTrue(ev.accepted)

    def test_ctrl_with_another_modifier_still_resizes(self):
        hud, ev = self._hud(), _FakeWheel(NOTCH, _CTRL | _SHIFT)
        hud.wheelEvent(ev)
        hud.resize.assert_called_once_with(462, 616)

    def test_ctrl_wheel_at_a_bound_is_consumed_without_a_resize(self):
        hud = self._hud(self.m.WHEEL_MAX_W, 952)
        ev = _FakeWheel(NOTCH, _CTRL)
        hud.wheelEvent(ev)
        hud.resize.assert_not_called()
        self.assertTrue(ev.accepted)
        self.base_wheel.assert_not_called()

    def test_plain_wheel_keeps_todays_behaviour(self):
        for mods in (0, _SHIFT, _ALT, _SHIFT | _ALT):
            for dy in (NOTCH, -NOTCH):
                with self.subTest(mods=hex(mods), dy=dy):
                    self.base_wheel.reset_mock()
                    hud, ev = self._hud(), _FakeWheel(dy, mods)
                    hud.wheelEvent(ev)
                    hud.resize.assert_not_called()
                    self.base_wheel.assert_called_once_with(ev)
                    self.assertIsNone(ev.accepted)   # left to QWidget

    def test_the_wheel_size_persists_like_a_grip_size(self):
        """resize() delivers a resizeEvent; the grip's own path (resizeEvent
        -> debounced _save_geometry) then writes the wheel size, and the next
        launch restores it."""
        m = self.m
        tmp = tempfile.mkdtemp(prefix="ju_wheel_geom_")
        self.addCleanup(shutil.rmtree, tmp, True)
        p = mock.patch.object(m, "GEOMETRY_FILE",
                              os.path.join(tmp, "unified_hud_geometry.json"))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(m.QWidget, "resizeEvent", create=True)
        p.start()
        self.addCleanup(p.stop)

        geo = {"x": 40, "y": 30, "w": 420, "h": 560}
        hud = object.__new__(m.UnifiedHud)
        hud.width = lambda: geo["w"]
        hud.height = lambda: geo["h"]
        hud.btn_close = mock.Mock()
        hud.grip = mock.Mock()
        hud._save_timer = mock.Mock()
        hud.geometry = lambda: types.SimpleNamespace(
            x=lambda: geo["x"], y=lambda: geo["y"],
            width=lambda: geo["w"], height=lambda: geo["h"])

        def fake_resize(w, h):           # what QWidget.resize does
            geo["w"], geo["h"] = w, h
            hud.resizeEvent(object())
        hud.resize = fake_resize

        hud.wheelEvent(_FakeWheel(NOTCH, _CTRL))
        hud._save_timer.start.assert_called()
        hud._save_geometry()             # the debounce timer firing
        self.assertEqual(m._load_saved_geometry(),
                         {"x": 40, "y": 30, "w": 462, "h": 616})
        restored = m._validate_geometry(m._load_saved_geometry(), [], (0, 0))
        self.assertEqual((restored["w"], restored["h"]), (462, 616))


# ════════════════════════════════════════════════════════════════════════════
#  Real offscreen render (PyQt6 installed only), in a subprocess
# ════════════════════════════════════════════════════════════════════════════
_WHEEL_SCRIPT = r"""
import importlib.util, json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root, tmp = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QTimer, QEventLoop
from PyQt6.QtGui import QWheelEvent
app = QApplication(["ctrl-wheel-resize"])

spec = importlib.util.spec_from_file_location(
    "_ju_wheel_render", os.path.join(root, "hud", "jarvis_unified_hud.py"))
ju = importlib.util.module_from_spec(spec)
sys.modules["_ju_wheel_render"] = ju
spec.loader.exec_module(ju)
for name, fn in (("HUD_STATE_FILE", "hud_state.json"),
                 ("BAMBU_STATE_FILE", "bambu_overlay_state.json"),
                 ("CONTROL_FILE", "unified_hud_state.json"),
                 ("GEOMETRY_FILE", "unified_hud_geometry.json"),
                 ("CAMERA_PREVIEW_FILE", "preview.jpg")):
    setattr(ju, name, os.path.join(tmp, fn))

class Slow:
    def snapshot(self):
        return {}

w = ju.UnifiedHud(0, Slow())
w.timer.stop(); w.cam_timer.stop()      # no data ticks: no nvidia-smi poll
calls = []
_orig = w._draw_reactor
def _rec(p, cx, cy, R, accent, s):
    calls.append((R, s))
    return _orig(p, cx, cy, R, accent, s)
w._draw_reactor = _rec
w.setGeometry(100, 100, 420, 560)
w.show()
app.processEvents()

C = Qt.KeyboardModifier.ControlModifier
NONE = Qt.KeyboardModifier.NoModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier

def wheel(dy, mods):
    ev = QWheelEvent(QPointF(60, 60), QPointF(w.mapToGlobal(QPoint(60, 60))),
                     QPoint(0, 0), QPoint(0, dy), Qt.MouseButton.NoButton,
                     mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)
    app.processEvents()
    return ev.isAccepted()

def snap():
    del calls[:]
    w.grab()
    R, s = calls[-1]
    return {"w": w.width(), "h": w.height(), "R": R, "s": s}

def settle(ms=900):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

out = {"base": snap()}
out["ctrl_up_accepted"] = wheel(120, C)
out["up"] = snap()
out["plain_accepted"] = wheel(120, NONE)
out["plain"] = snap()
out["shift_accepted"] = wheel(-120, SHIFT)
out["shift"] = snap()
settle()
with open(ju.GEOMETRY_FILE, encoding="utf-8") as f:
    out["saved"] = json.load(f)
out["loaded"] = ju._load_saved_geometry()
wheel(-120, C)
out["down"] = snap()
for _ in range(20):
    wheel(120, C)
out["max"] = snap()
for _ in range(20):
    wheel(-120, C)
out["min"] = snap()
out["bounds"] = [getattr(ju, n, None)
                 for n in ("WHEEL_MIN_W", "WHEEL_MAX_W", "PAINT_BASE_W")]
w.hide(); w.deleteLater()
print("RESULT " + json.dumps(out))
"""


def _pyqt6_installed() -> bool:
    try:
        return importlib.util.find_spec("PyQt6") is not None
    except (ImportError, ValueError):
        return False


@unittest.skipUnless(_pyqt6_installed(), "PyQt6 not installed (light tier)")
class RealOffscreenWheelTests(unittest.TestCase):
    """A real QWheelEvent on the real widget: Ctrl+wheel resizes it, the
    paint that follows draws the reactor and text at the new scale, a plain
    wheel changes nothing, and the debounced save writes the new size."""

    result: dict = {}

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.mkdtemp(prefix="ju_wheel_render_")
        cls.addClassCleanup(shutil.rmtree, tmp, True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run(
            [sys.executable, "-c", _WHEEL_SCRIPT, _PROJECT_DIR, tmp],
            capture_output=True, text=True, timeout=120, env=env,
            creationflags=flags)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT ")), None)
        if line is None:
            raise AssertionError(f"wheel subprocess failed (rc={proc.returncode}):"
                                 f"\n{proc.stdout}\n{proc.stderr}")
        cls.result = json.loads(line[len("RESULT "):])

    def test_ctrl_wheel_grows_the_window_and_the_drawing(self):
        r = self.result
        self.assertEqual((r["base"]["w"], r["base"]["h"]), (420, 560), r)
        self.assertTrue(r["ctrl_up_accepted"], r)
        self.assertEqual((r["up"]["w"], r["up"]["h"]), (462, 616), r)
        self.assertAlmostEqual(r["up"]["R"] / r["base"]["R"], 1.1, places=3)
        self.assertAlmostEqual(r["up"]["s"] / r["base"]["s"], 1.1, places=3)

    def test_ctrl_wheel_down_shrinks_it_back(self):
        r = self.result
        self.assertLess(r["down"]["w"], r["up"]["w"], r)
        self.assertLess(r["down"]["R"], r["up"]["R"], r)
        self.assertEqual((r["down"]["w"], r["down"]["h"]), (420, 560), r)
        self.assertAlmostEqual(r["down"]["R"], r["base"]["R"], places=3)
        self.assertAlmostEqual(r["down"]["s"], r["base"]["s"], places=3)

    def test_plain_wheel_changes_nothing(self):
        r = self.result
        self.assertFalse(r["plain_accepted"], r)
        self.assertFalse(r["shift_accepted"], r)
        for key in ("plain", "shift"):
            self.assertEqual(r[key], r["up"], r)

    def test_the_wheel_size_is_saved_and_restored(self):
        r = self.result
        self.assertEqual((r["saved"]["w"], r["saved"]["h"]), (462, 616), r)
        self.assertEqual((r["loaded"]["w"], r["loaded"]["h"]), (462, 616), r)

    def test_min_and_max_bounds_with_the_drawing_still_scaling(self):
        r = self.result
        lo, hi, base = r["bounds"]
        self.assertEqual(r["max"]["w"], hi, r)
        self.assertEqual(r["min"]["w"], lo, r)
        self.assertAlmostEqual(r["max"]["s"], hi / base, places=6)
        self.assertAlmostEqual(r["min"]["s"], lo / base, places=6)
        self.assertGreater(r["max"]["R"], r["up"]["R"])
        self.assertLess(r["min"]["R"], r["base"]["R"])


if __name__ == "__main__":
    unittest.main()
