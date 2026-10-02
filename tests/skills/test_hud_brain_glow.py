"""Brain glow, HUD side: the main HUD (hud/jarvis_unified_hud.py), the
holographic overlay (hud/jarvis_holo.py) and both arc reactors
(hud/arc_reactor_status_hud.py — the status ring, and
hud/holo_workshop_canvas.py — the "show the arc reactor" canvas) read
hud_state.json's ``brain`` key and tint their glow with the colour of the
brain that is answering, plus a brief label with its name.

The contract every HUD must keep: a MISSING, None or garbage ``brain`` key
never breaks the HUD — it simply renders its normal (state-coloured) look.

Three layers:
  * SOURCE: each HUD imports core.brain_glow.hud_brain inside a try/except
    with a no-op fallback (a missing helper = no glow, never a dead HUD);
  * HEADLESS readers (PyQt6 blocked, the modules' own ImportError path):
    the Qt HUDs' refresh stores the parsed brain, tolerating garbage;
    the tkinter HUDs run a whole frame against a recording fake canvas and
    must draw the brain ring + label only when a valid brain is present;
  * REAL RENDER (only where PyQt6 is installed — the local full tier): a
    subprocess renders the two Qt HUDs offscreen with an Opus brain, with no
    brain and with a garbage brain, and the halo pixel must turn violet only
    for the real brain. A subprocess keeps Qt out of the test process
    entirely (no QApplication outlives a test, no Qt teardown at exit).

stdlib unittest + unittest.mock only (no pytest); nothing under the project
tree is written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

from core import brain_glow as BG

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_HUD_DIR = os.path.join(_PROJECT_DIR, "hud")

_GLOW_HUDS = ("jarvis_unified_hud.py", "jarvis_holo.py",
              "arc_reactor_status_hud.py", "holo_workshop_canvas.py")

# Label window effectively unbounded: discovery imports this module long
# before its tests run in a full-suite pass, so a short window would already
# have expired (the expiry case patches time.time explicitly instead).
_OPUS = BG.brain_state("claude-opus-5-5", "cloud", label_s=1e9)
_GARBAGE = [None, "opus", 5, [], {}, {"color": "violet"},
            {"color": "#12345", "name": "x"}, {"color": None},
            {"color": "#B05CFF", "label_until": "soon", "name": None}]


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load(testcase, filename: str, mod_name: str, block_pyqt: bool):
    """Load hud/<filename> under a synthetic name (dropped on cleanup).
    block_pyqt=True forces the module's PyQt6-absent stub path."""
    path = os.path.join(_HUD_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    hidden = {}
    if block_pyqt:
        hidden = {n: sys.modules.pop(n) for n in list(sys.modules)
                  if n.split(".")[0] == "PyQt6"}

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    if block_pyqt:
        real_import = __import__

        def _imp(name, *a, **k):
            if name.split(".")[0] == "PyQt6":
                raise ImportError(f"[test] PyQt6 blocked: {name}")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=_imp):
            spec.loader.exec_module(module)
        testcase.assertFalse(module._HAS_PYQT6)
    else:
        spec.loader.exec_module(module)
    return module


# ════════════════════════════════════════════════════════════════════════════
#  Source invariants
# ════════════════════════════════════════════════════════════════════════════
class SourceInvariantTests(unittest.TestCase):
    def test_each_glow_hud_imports_the_reader_fail_open(self):
        for fn in _GLOW_HUDS:
            with self.subTest(hud=fn):
                src = _read(os.path.join(_HUD_DIR, fn))
                self.assertIn("from core.brain_glow import hud_brain", src)
                # The import sits in a try with a no-op fallback.
                i = src.index("from core.brain_glow import hud_brain")
                block = src[max(0, i - 200): i + 400]
                self.assertIn("try:", block)
                self.assertIn("except Exception", block)
                self.assertIn("def _hud_brain", block)

    def test_the_launched_hud_is_one_of_the_glow_huds(self):
        # bobert_companion._launch_hud spawns the unified HUD today; if that
        # ever moves, the new target must read the brain key too.
        mono = _read(os.path.join(_PROJECT_DIR, "bobert_companion.py"))
        import re
        m = re.search(r"^def _launch_hud\(", mono, re.M)
        self.assertIsNotNone(m)
        body = mono[m.end(): m.end() + 6000]
        hit = re.search(r'"hud",\s*"([^"]+\.py)"', body)
        self.assertIsNotNone(hit)
        self.assertIn(hit.group(1), _GLOW_HUDS)


# ════════════════════════════════════════════════════════════════════════════
#  Qt HUD readers, headless
# ════════════════════════════════════════════════════════════════════════════
class UnifiedHudReaderTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load(self, "jarvis_unified_hud.py", "_ju_brain_glow_ut", True)

    def _bare(self):
        hud = object.__new__(self.mod.UnifiedHud)
        hud.parent_pid = 0
        hud.frame = 0
        hud.gpu_util = None
        hud._gpu_sampling = False
        hud._gpu_cached_at = time.time() + 3600.0
        hud._last_net = None
        hud._last_net_at = None
        hud._refresh_camera_preview = lambda: False
        hud.brain = None
        return hud

    def _refresh(self, hud, state):
        def fake_read_json(path):
            return dict(state) if path == self.mod.HUD_STATE_FILE else {}
        with mock.patch.object(self.mod, "_read_json", side_effect=fake_read_json), \
             mock.patch.object(self.mod, "_is_parent_alive", return_value=True), \
             mock.patch.object(self.mod, "_control_says_off", return_value=False):
            return hud._refresh()

    def test_refresh_reads_the_brain(self):
        hud = self._bare()
        self.assertTrue(self._refresh(hud, {"state": "Idle", "brain": _OPUS}))
        self.assertEqual(hud.brain.color, _OPUS["color"])
        self.assertEqual(hud._glow_hex(), _OPUS["color"])

    def test_missing_or_garbage_brain_is_none(self):
        for bad in _GARBAGE[:-1]:
            with self.subTest(brain=bad):
                hud = self._bare()
                hud.brain = BG.hud_brain({"brain": _OPUS})   # stale value
                self.assertTrue(self._refresh(hud, {"state": "Idle", "brain": bad}))
                self.assertIsNone(hud.brain)
                self.assertIsNone(hud._glow_hex())
        hud = self._bare()
        self.assertTrue(self._refresh(hud, {"state": "Idle"}))
        self.assertIsNone(hud.brain)

    def test_reader_failure_never_breaks_refresh(self):
        hud = self._bare()
        with mock.patch.object(self.mod, "_hud_brain", side_effect=RuntimeError("x")):
            self.assertTrue(self._refresh(hud, {"state": "Idle", "brain": _OPUS}))
        self.assertIsNone(hud.brain)

    def test_paint_path_uses_the_glow(self):
        import inspect
        src = inspect.getsource(self.mod.UnifiedHud._draw_reactor)
        self.assertIn("_glow_hex", src)
        paint = inspect.getsource(self.mod.UnifiedHud.paintEvent)
        self.assertIn("_draw_brain_label", paint)

    def test_no_ring_asleep_or_in_standby(self):
        """2026-10-02 review: the brain glow at full brightness in standby /
        sleep took away the asleep cue."""
        for st in ("Standby", "Sleep", "sleeping"):
            with self.subTest(state=st):
                hud = self._bare()
                self.assertTrue(self._refresh(hud, {"state": st,
                                                    "brain": _OPUS}))
                self.assertIsNotNone(hud.brain)
                self.assertIsNone(hud._glow_hex())


class ArcReactorReaderTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load(self, "arc_reactor_status_hud.py", "_arc_brain_glow_ut",
                         True)

    def _bare(self):
        sc = object.__new__(self.mod.ArcReactorStatusScene)
        sc.parent_pid = 0
        sc.frame = 0
        sc.gpu_util_pct = None
        sc._gpu_sampling = False
        sc._gpu_cached_at = time.time() + 3600.0
        sc._last_net_bytes = None
        sc._last_net_at = None
        sc.update = lambda *a, **k: None
        sc.brain = None
        return sc

    def _refresh(self, sc, state):
        def fake_read_json(path):
            return dict(state) if path == self.mod.HUD_STATE_FILE else {}
        with mock.patch.object(self.mod, "_read_json", side_effect=fake_read_json), \
             mock.patch.object(self.mod, "_is_parent_alive", return_value=True), \
             mock.patch.object(self.mod, "_control_says_off", return_value=False):
            return sc.refresh_data()

    def test_refresh_reads_the_brain(self):
        sc = self._bare()
        self.assertTrue(self._refresh(sc, {"state": "Idle", "brain": _OPUS}))
        self.assertEqual(sc._glow_hex(), _OPUS["color"])

    def test_missing_or_garbage_brain_is_none(self):
        for bad in _GARBAGE[:-1]:
            with self.subTest(brain=bad):
                sc = self._bare()
                self.assertTrue(self._refresh(sc, {"state": "Idle", "brain": bad}))
                self.assertIsNone(sc._glow_hex())

    def test_paint_path_uses_the_glow(self):
        import inspect
        src = inspect.getsource(self.mod.ArcReactorStatusScene.drawBackground)
        self.assertIn("_glow_hex", src)

    def test_no_ring_asleep_or_in_standby(self):
        for st in ("Standby", "Sleep"):
            with self.subTest(state=st):
                sc = self._bare()
                self.assertTrue(self._refresh(sc, {"state": st,
                                                   "brain": _OPUS}))
                self.assertIsNone(sc._glow_hex())


# ════════════════════════════════════════════════════════════════════════════
#  tkinter HUDs: one frame against a recording fake canvas
# ════════════════════════════════════════════════════════════════════════════
def _colours(canvas_mock):
    out = []
    for name, args, kwargs in canvas_mock.method_calls:
        for k in ("outline", "fill"):
            v = kwargs.get(k)
            if isinstance(v, str) and v.startswith("#"):
                out.append(v.lower())
    return out


def _texts(canvas_mock):
    return [kwargs.get("text", "") for name, args, kwargs in canvas_mock.method_calls
            if name == "create_text"]


def _ovals(canvas_mock):
    return sum(1 for name, _a, _k in canvas_mock.method_calls if name == "create_oval")


class _TkFrameMixin:
    FILE = ""
    CLS = ""
    STATE_ATTR = ""

    def _make(self):
        try:
            has_tk = importlib.util.find_spec("tkinter") is not None
        except (ImportError, ValueError):   # pragma: no cover
            has_tk = False
        if not has_tk:   # pragma: no cover - no Tk on this runner
            self.skipTest("tkinter unavailable")
        mod = _load(self, self.FILE, "_tk_brain_glow_" + self.CLS, False)
        cls = getattr(mod, self.CLS)
        with mock.patch.object(mod.tk, "Tk"), mock.patch.object(mod.tk, "Canvas"), \
             mock.patch.object(cls, "tick"):
            obj = cls(0, 0, self.W, self.H, 0)
        return mod, obj

    def _frame(self, state, now=None):
        mod, obj = self._make()
        state_path = getattr(mod, self.STATE_ATTR)

        def fake_read_json(path):
            return dict(state) if path == state_path else {}
        patches = [mock.patch.object(mod, "_read_json", side_effect=fake_read_json),
                   mock.patch.object(mod, "_is_parent_alive", return_value=True)]
        if now is not None:
            patches.append(mock.patch.object(mod.time, "time", return_value=now))
        for p in patches:
            p.start()
        try:
            obj.canvas.reset_mock()
            obj._tick_body()      # NOT tick(): its wrapper swallows errors
        finally:
            for p in reversed(patches):
                p.stop()
        return obj.canvas

    def test_brain_draws_ring_and_label_in_its_colour(self):
        base = self._frame({"state": "Idle"})
        glow = self._frame({"state": "Idle", "brain": _OPUS})
        self.assertIn(_OPUS["color"].lower(), _colours(glow))
        self.assertTrue(any("OPUS 5.5" in t for t in _texts(glow)), _texts(glow))
        self.assertGreater(_ovals(glow), _ovals(base))
        self.assertNotIn(_OPUS["color"].lower(), _colours(base))
        self.assertFalse(any("OPUS" in t for t in _texts(base)))

    def test_label_is_brief_ring_stays(self):
        later = _OPUS["label_until"] + 30.0
        base = self._frame({"state": "Idle"}, now=later)
        glow = self._frame({"state": "Idle", "brain": _OPUS}, now=later)
        self.assertFalse(any("OPUS" in t for t in _texts(glow)))
        self.assertGreater(_ovals(glow), _ovals(base))

    def test_garbage_brain_renders_the_normal_frame(self):
        base = self._frame({"state": "Idle"})
        for bad in _GARBAGE[:-1]:
            with self.subTest(brain=bad):
                c = self._frame({"state": "Idle", "brain": bad})
                self.assertEqual(_ovals(c), _ovals(base))
                self.assertEqual(len(_texts(c)), len(_texts(base)))

    def test_no_brain_ring_or_label_asleep(self):
        """Asleep / in standby the overlay keeps its at-rest look: the brain
        ring and label are not drawn (2026-10-02 review)."""
        for st in ("Standby", "Sleep"):
            with self.subTest(state=st):
                base = self._frame({"state": st})
                glow = self._frame({"state": st, "brain": _OPUS})
                self.assertNotIn(_OPUS["color"].lower(), _colours(glow))
                self.assertEqual(_ovals(glow), _ovals(base))
                self.assertFalse(any("OPUS" in t for t in _texts(glow)))

    def test_every_state_renders_with_a_brain(self):
        for st in ("Idle", "Listening", "Thinking", "Speaking", "Standby"):
            with self.subTest(state=st):
                self._frame({"state": st, "brain": _OPUS,
                             "tts_amplitude": 0.7, "mic_level": 0.4})


class HoloOverlayFrameTests(_TkFrameMixin, unittest.TestCase):
    FILE, CLS, STATE_ATTR = "jarvis_holo.py", "HoloHUD", "STATE_FILE"
    W, H = 2560, 1440


class WorkshopCanvasFrameTests(_TkFrameMixin, unittest.TestCase):
    FILE, CLS, STATE_ATTR = ("holo_workshop_canvas.py", "WorkshopCanvas",
                             "HUD_STATE_FILE")
    W, H = 320, 320


# ════════════════════════════════════════════════════════════════════════════
#  Real offscreen render (PyQt6 installed only), in a subprocess
# ════════════════════════════════════════════════════════════════════════════
_RENDER_SCRIPT = r"""
import importlib.util, json, os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
root = sys.argv[1]
sys.path.insert(0, root)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtCore import QRectF
app = QApplication(["brain-glow-render"])
from core import brain_glow as BG

def load(fn, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "hud", fn))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

opus = BG.brain_state("claude-opus-5-5", "cloud", label_s=60.0)
sonnet = BG.brain_state("claude-sonnet-5-5", "cloud", label_s=60.0)
# (state, hud_state) per case. The ring pixel sits on the brain ring; the halo
# pixel on the state-coloured halo outside it.
cases = {"opus": ("idle", {"brain": opus}), "none": ("idle", {}),
         "garbage": ("idle", {"brain": {"color": "violet"}}),
         "garbage2": ("idle", {"brain": "opus"}),
         "think_sonnet": ("thinking", {"brain": sonnet}),
         "think_none": ("thinking", {}),
         "standby_opus": ("standby", {"brain": opus}),
         "standby_none": ("standby", {})}
out = {"unified": {}, "arc": {}}

def rgb(c):
    return [c.red(), c.green(), c.blue()]

ju = load("jarvis_unified_hud.py", "_ju_render")
class Slow:
    def snapshot(self): return {}
for key, (state, hud_state) in cases.items():
    w = ju.UnifiedHud(0, Slow())
    w.resize(420, 560)
    w.state = state
    w.brain = ju._hud_brain(hud_state, time.time())
    img = w.grab().toImage()
    W, s = 420.0, 1.0
    title_h = 34.0 * s; top = title_h + 12 * s
    size = min(W - 28.0 * s, 560 * 0.40); R = size * 0.42
    cx, cy = W / 2.0, top + size / 2.0
    out["unified"][key] = {
        "ring": rgb(img.pixelColor(int(round(cx - ju.BRAIN_RING_R * R)), int(cy))),
        "halo": rgb(img.pixelColor(int(cx - 1.23 * R), int(cy)))}
    w.timer.stop(); w.cam_timer.stop(); w.deleteLater()

ar = load("arc_reactor_status_hud.py", "_arc_render")
for key, (state, hud_state) in cases.items():
    sc = ar.ArcReactorStatusScene(320, 320, 0)
    sc.state = state
    sc.brain = ar._hud_brain(hud_state, time.time())
    img = QImage(320, 320, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    sc.render(p, QRectF(0, 0, 320, 320), QRectF(0, 0, 320, 320))
    p.end()
    d_ring = sc.R_OUTER * ar.BRAIN_RING_R / 2 ** 0.5
    d_halo = sc.R_GLOW * 0.95 / 2 ** 0.5
    out["arc"][key] = {
        "ring": rgb(img.pixelColor(int(round(sc.cx - d_ring)),
                                   int(round(sc.cy - d_ring)))),
        "halo": rgb(img.pixelColor(int(round(sc.cx - d_halo)),
                                   int(round(sc.cy - d_halo))))}
print("RESULT " + json.dumps(out))
"""


def _pyqt6_installed() -> bool:
    try:
        return importlib.util.find_spec("PyQt6") is not None
    except (ImportError, ValueError):
        return False


@unittest.skipUnless(_pyqt6_installed(), "PyQt6 not installed (light tier)")
class RealRenderTests(unittest.TestCase):
    """The brain RING turns violet for an Opus brain and stays the normal
    look with no brain or a garbage brain; the HALO keeps the state colour
    whatever the brain (2026-10-02 review: Sonnet's gold on the halo read as
    "thinking"); asleep / in standby there is no ring — and nothing raises."""

    result: dict = {}

    @classmethod
    def setUpClass(cls):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if sys.platform == "win32" else 0
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run(
            [sys.executable, "-c", _RENDER_SCRIPT, _PROJECT_DIR],
            capture_output=True, text=True, timeout=120, env=env,
            creationflags=flags)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT ")), None)
        if line is None:
            raise AssertionError(f"render subprocess failed (rc={proc.returncode}):"
                                 f"\n{proc.stdout}\n{proc.stderr}")
        cls.result = json.loads(line[len("RESULT "):])

    def _check(self, hud):
        px = self.result[hud]
        ring = {k: v["ring"] for k, v in px.items()}
        halo = {k: v["halo"] for k, v in px.items()}
        # Opus violet (#B05CFF) has far more red than the idle cyan (#4CC9FF)
        # and reads violet: blue well above green.
        self.assertGreater(ring["opus"][0], ring["none"][0] + 20, px)
        self.assertGreater(ring["opus"][2], ring["opus"][1] + 60, px)
        # Garbage renders exactly like no brain at all.
        self.assertEqual(ring["garbage"], ring["none"], px)
        self.assertEqual(ring["garbage2"], ring["none"], px)
        # The halo is the STATE's: a brain never recolours it.
        self.assertEqual(halo["opus"], halo["none"], px)
        self.assertEqual(halo["think_sonnet"], halo["think_none"], px)
        # Asleep / in standby: no ring, the dim at-rest look.
        self.assertEqual(ring["standby_opus"], ring["standby_none"], px)

    def test_unified_hud_ring(self):
        self._check("unified")

    def test_arc_reactor_ring(self):
        self._check("arc")


if __name__ == "__main__":
    unittest.main()
