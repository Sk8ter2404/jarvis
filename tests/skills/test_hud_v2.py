"""Logic tests for ``skills/holographic_overlay/hud_v2.py`` — the Stark-style
status-ring renderer that the holographic_overlay manager spawns as its own
PyQt6 subprocess.

WHY A BESPOKE FAKE-Qt LAYER
  ``hud_v2.py`` does ``from PyQt6.QtCore import ...`` / ``QtGui`` / ``QtWidgets``
  at module top level, builds its colour palette from ``QColor(...)`` at import,
  and subclasses ``QGraphicsScene`` / ``QWidget``. PyQt6 is NOT on the CI runner
  (ubuntu-latest, light dep set), and even where it IS installed a renderer test
  must never spin up a real Qt event loop / display. So this module INJECTS a
  fake ``PyQt6`` package (QtCore/QtGui/QtWidgets) into ``sys.modules`` before
  loading the source, then drives the pure logic — the layout math, the
  state→colour/brightness decisions, the control-file read+apply in
  ``refresh_data``, the meeting/now-playing formatting, the GPU sensor read with
  its nvidia-smi + pulse_strip fallback, and the giant ``drawBackground`` paint
  pass (called once per permutation against a mock painter, never a real GPU).

  The fakes are real Python classes (not bare MagicMocks) so ``super().__init__``
  works and the subclasses construct; enum members are real ``int`` values so the
  source's ``int(Qt.AlignmentFlag.AlignCenter | ...)`` and ``setAlpha(int(...))``
  arithmetic runs exactly as under real Qt. ``QColor`` tracks its rgba so a test
  can assert which palette colour a code path chose.

ISOLATION CONTRACT
  • The fake PyQt6 modules are installed per-test in ``setUp`` and removed in
    ``addCleanup`` (save/restore of the prior ``sys.modules`` entries, including
    absence) — never a leaked module-level write. The freshly-loaded source
    module is likewise dropped on cleanup.
  • The source is loaded from its file path under a synthetic module name; it is
    NOT imported through the package (which would drag in the manager) and the
    real monolith is never booted.
  • ``hud_v2``'s data-source files (hud_state.json, bambu_overlay_state.json,
    the control file) are module-level absolute paths into the real project root;
    every test that exercises a read points those globals at a per-test temp dir,
    so no real project file is read or written. Globals mutated on the loaded
    module don't leak because the module itself is dropped each test.
  • ``psutil`` is the one real CI dep the source uses; tests that assert on CPU/
    RAM patch it on the loaded module so the numbers are deterministic.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


_SENTINEL = object()

# hud_v2.py lives next to the holographic_overlay package __init__.
_HUD_V2_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "skills", "holographic_overlay", "hud_v2.py",
)


# ═══════════════════════════════════════════════════════════════════════════
#  Fake PyQt6 layer
#
#  Each fake is the minimum surface hud_v2.py touches. Constructors swallow any
#  args; methods the source calls on instances it did NOT subclass-override
#  resolve through ``__getattr__`` to a no-op MagicMock. Geometry/colour holders
#  keep their values so layout/colour assertions are possible.
# ═══════════════════════════════════════════════════════════════════════════
class _AutoMock:
    """Base for fake Qt classes: arbitrary attribute access yields a callable
    no-op (a MagicMock), so ``self.setSceneRect(...)`` / ``painter.setPen(...)``
    etc. work without enumerating Qt's whole API. Subclass ``__init__`` swallows
    all positional/keyword args. Real instance attributes set by the source (or
    by a fake subclass) shadow the auto-mock, so they win."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        # Only reached for attributes not found normally. Cache the mock so the
        # same attr returns a stable object across calls.
        m = mock.MagicMock(name=name)
        object.__setattr__(self, name, m)
        return m


class _IntFlag(int):
    """An int subclass standing in for a Qt enum member. Real int value so the
    source's ``int(flag)`` and ``flag | other`` behave; carries a name for
    debugging."""

    def __new__(cls, value, label=""):
        obj = super().__new__(cls, value)
        obj._label = label
        return obj

    def __repr__(self):  # pragma: no cover - debugging aid only
        return f"_IntFlag({int(self)}, {self._label!r})"


class _EnumNS:
    """A tiny namespace whose attributes are ``_IntFlag`` members, so
    ``Qt.AlignmentFlag.AlignCenter`` is an int and ``A | B`` is an int."""

    def __init__(self, **members):
        self._counter = 1
        for name, val in members.items():
            setattr(self, name, _IntFlag(val, name))

    def __getattr__(self, name):
        # Auto-vivify any enum member we didn't pre-declare with a unique bit.
        val = _IntFlag(1 << (self.__dict__.setdefault("_counter", 1)), name)
        self._counter = self.__dict__["_counter"] + 1
        object.__setattr__(self, name, val)
        return val


class _FakeQColor(_AutoMock):
    """Tracks rgba so a test can assert which palette colour a branch chose.
    Supports ``QColor(r,g,b[,a])``, ``QColor(named)`` copy-construct, and
    ``setAlpha``."""

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], _FakeQColor):
            src = args[0]
            self.r, self.g, self.b, self.a = src.r, src.g, src.b, src.a
        elif len(args) >= 3:
            self.r, self.g, self.b = args[0], args[1], args[2]
            self.a = args[3] if len(args) >= 4 else 255
        else:
            self.r = self.g = self.b = 0
            self.a = 255

    def setAlpha(self, a):
        self.a = int(a)

    def rgba(self):
        return (self.r, self.g, self.b, self.a)

    def __eq__(self, other):
        return isinstance(other, _FakeQColor) and self.rgba() == other.rgba()

    def __hash__(self):
        return hash(self.rgba())


class _FakeQRectF(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class _FakeQPointF(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class _FakeQFont(_AutoMock):
    Weight = _EnumNS(Thin=100, Normal=400, DemiBold=600, Bold=700)

    def __init__(self, *args):
        super().__init__()
        self.args = args


class _FakeQPen(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class _FakeQBrush(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.args = args


class _FakeQRadialGradient(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.stops = []

    def setColorAt(self, pos, color):
        self.stops.append((pos, color))


class _FakeQPainter(_AutoMock):
    RenderHint = _EnumNS(Antialiasing=1, TextAntialiasing=2)


class _FakeQGraphicsScene(_AutoMock):
    """Real base so ``StarkStatusRingScene`` can ``super().__init__`` and inherit
    ``setSceneRect`` / ``update`` (auto-mocked)."""


class _FakeQWidget(_AutoMock):
    pass


class _FakeQGraphicsView(_AutoMock):
    # The source reads ``self.view.Shape.NoFrame`` — give Shape a member.
    Shape = _EnumNS(NoFrame=0)


class _FakeQApplication(_AutoMock):
    _instance = None

    def __init__(self, *args):
        super().__init__()
        type(self)._instance = self

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class _FakeQTimer(_AutoMock):
    def __init__(self, *args):
        super().__init__()
        self.timeout = mock.MagicMock()
        self.interval = None
        self.started = False

    def setInterval(self, ms):
        self.interval = ms

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class _FakeQGraphicsDropShadowEffect(_AutoMock):
    pass


class _FakeQt:
    """The ``Qt`` namespace — only the enum groups hud_v2 references."""
    PenStyle = _EnumNS(NoPen=0, SolidLine=1)
    BrushStyle = _EnumNS(NoBrush=0, SolidPattern=1)
    PenCapStyle = _EnumNS(FlatCap=0, SquareCap=16, RoundCap=32)
    AlignmentFlag = _EnumNS(
        AlignLeft=1, AlignRight=2, AlignHCenter=4, AlignVCenter=128,
        AlignCenter=132,
    )
    WindowType = _EnumNS(
        FramelessWindowHint=2048, WindowStaysOnTopHint=262144, Tool=10,
    )
    WidgetAttribute = _EnumNS(
        WA_TranslucentBackground=120, WA_TransparentForMouseEvents=51,
    )
    ScrollBarPolicy = _EnumNS(ScrollBarAlwaysOff=1)


def _build_fake_pyqt6():
    """Construct the three fake PyQt6 submodules + parent package. Returned as a
    dict ``{name: module}`` ready to splice into ``sys.modules``."""
    pkg = types.ModuleType("PyQt6")
    pkg.__path__ = []  # mark as a package so ``import PyQt6.QtCore`` is allowed

    qtcore = types.ModuleType("PyQt6.QtCore")
    qtcore.Qt = _FakeQt
    qtcore.QTimer = _FakeQTimer
    qtcore.QRectF = _FakeQRectF
    qtcore.QPointF = _FakeQPointF

    qtgui = types.ModuleType("PyQt6.QtGui")
    qtgui.QPainter = _FakeQPainter
    qtgui.QColor = _FakeQColor
    qtgui.QPen = _FakeQPen
    qtgui.QBrush = _FakeQBrush
    qtgui.QFont = _FakeQFont
    qtgui.QRadialGradient = _FakeQRadialGradient

    qtwidgets = types.ModuleType("PyQt6.QtWidgets")
    qtwidgets.QApplication = _FakeQApplication
    qtwidgets.QWidget = _FakeQWidget
    qtwidgets.QGraphicsView = _FakeQGraphicsView
    qtwidgets.QGraphicsScene = _FakeQGraphicsScene
    qtwidgets.QGraphicsDropShadowEffect = _FakeQGraphicsDropShadowEffect

    pkg.QtCore = qtcore
    pkg.QtGui = qtgui
    pkg.QtWidgets = qtwidgets
    return {
        "PyQt6": pkg,
        "PyQt6.QtCore": qtcore,
        "PyQt6.QtGui": qtgui,
        "PyQt6.QtWidgets": qtwidgets,
    }


def _install_fake_pyqt6(testcase):
    """Splice the fake PyQt6 modules into ``sys.modules`` for the duration of a
    test, restoring prior state (incl. absence) on cleanup. Reset the fake
    QApplication singleton so cross-test state never leaks."""
    fakes = _build_fake_pyqt6()
    saved = {name: sys.modules.get(name, _SENTINEL) for name in fakes}

    def restore():
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        _FakeQApplication._instance = None

    for name, modobj in fakes.items():
        sys.modules[name] = modobj
    testcase.addCleanup(restore)


def _load_hud_v2(testcase):
    """Load hud_v2.py fresh from its path with fake PyQt6 already installed.
    Registers it under a unique synthetic name and drops it on cleanup so each
    test gets pristine module globals (palette, frame counters, file paths)."""
    mod_name = f"_hud_v2_under_test_{id(testcase)}_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(mod_name, _HUD_V2_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class _HudBase(unittest.TestCase):
    """Installs fake PyQt6, loads a fresh hud_v2 module, and redirects the
    module's data-source file globals into a throwaway temp dir."""

    def setUp(self):
        _install_fake_pyqt6(self)
        self.hud = _load_hud_v2(self)
        self.assertTrue(self.hud._HAS_PYQT6,
                        "fake PyQt6 should make _HAS_PYQT6 True")
        # ``psutil`` is a CI dep, but on a matrix slice where it is genuinely
        # absent the source leaves the name unbound (``_HAS_PSUTIL`` False). The
        # psutil-path tests force ``_HAS_PSUTIL=True`` to drive those branches,
        # so give the module a stand-in ``psutil`` to patch when the real one
        # didn't bind — keeps the suite running (never skipping) regardless of
        # whether psutil is importable in the current environment.
        if not hasattr(self.hud, "psutil"):
            self.hud.psutil = mock.MagicMock(name="psutil_standin")
        self.tmp = tempfile.mkdtemp(prefix="hud_v2_test_")
        self.addCleanup(self._cleanup_tmp)
        # Point every data-source path at the temp dir (absent unless a test
        # writes it). Mutating these on the loaded module is safe — the module
        # is dropped on cleanup, so nothing leaks.
        self.hud_state = os.path.join(self.tmp, "hud_state.json")
        self.bambu_state = os.path.join(self.tmp, "bambu_overlay_state.json")
        self.control = os.path.join(self.tmp, "control.json")
        self.hud.HUD_STATE_FILE = self.hud_state
        self.hud.BAMBU_STATE_FILE = self.bambu_state
        self.hud.CONTROL_FILE = self.control

    def _cleanup_tmp(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    # ── helpers ──────────────────────────────────────────────────────────
    def _write(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _new_scene(self, w=460, h=340, pid=0):
        return self.hud.StarkStatusRingScene(w, h, pid)

    def _paint(self, scene):
        """Drive the full paint pass once against a mock painter + rect."""
        painter = _FakeQPainter()
        rect = self.hud.QRectF(0.0, 0.0, scene.w, scene.h)
        scene.drawBackground(painter, rect)
        return painter


# ═══════════════════════════════════════════════════════════════════════════
#  Module import + palette construction
# ═══════════════════════════════════════════════════════════════════════════
class ModuleImportTests(_HudBase):
    def test_palette_colors_constructed(self):
        # With fake PyQt6, the palette globals are real fake QColors (not None).
        for name in ("CYAN", "AMBER", "RED", "GREEN_SOFT", "TEXT_COLOR",
                     "PANEL_DARK"):
            col = getattr(self.hud, name)
            self.assertIsNotNone(col, f"{name} should be a colour")
            self.assertTrue(hasattr(col, "rgba"))

    def test_cyan_palette_rgba_matches_spec(self):
        self.assertEqual(self.hud.CYAN.rgba(), (76, 201, 255, 255))
        self.assertEqual(self.hud.AMBER.rgba(), (255, 179, 71, 255))
        self.assertEqual(self.hud.RED.rgba(), (255, 91, 91, 255))
        # PANEL_DARK carries a non-opaque alpha.
        self.assertEqual(self.hud.PANEL_DARK.rgba(), (4, 8, 13, 215))

    def test_constants(self):
        self.assertEqual(self.hud.TICK_MS, 500)
        self.assertEqual(self.hud.TRACK_REFRESH_TICKS, 12)
        self.assertEqual(self.hud.CALENDAR_REFRESH_TICKS, 240)
        self.assertEqual(self.hud.GPU_CACHE_SECONDS, 4.0)

    def test_pyqt6_absent_degrades_gracefully(self):
        # Re-exec the source with PyQt6 import blocked (and any cached PyQt6
        # submodules hidden) so the `except ImportError` guard runs. The
        # module MUST import cleanly: hud_v2 is a Qt subprocess that can't
        # render without PyQt6, but importing it has to succeed so main() can
        # print an install hint and exit 2 — the launcher treats a
        # fast-exiting subprocess as "not engaged" rather than crashing on an
        # import-time traceback. The guard stubs the QGraphicsScene / QWidget
        # base classes and the palette globals fall back to None.
        real_import = __import__

        def _imp(name, *a, **k):
            if name.split(".")[0] == "PyQt6":
                raise ImportError("blocked PyQt6")
            return real_import(name, *a, **k)

        hidden = {n: sys.modules.pop(n)
                  for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
        mod_name = f"_hud_v2_noqt_{id(self)}"
        spec = importlib.util.spec_from_file_location(mod_name, _HUD_V2_PATH)
        m = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = m
        self.addCleanup(lambda: sys.modules.pop(mod_name, None))
        try:
            with mock.patch("builtins.__import__", side_effect=_imp):
                spec.loader.exec_module(m)        # must NOT raise
        finally:
            sys.modules.update(hidden)

        # The ImportError guard ran: flag False, palette degraded to None.
        self.assertFalse(m._HAS_PYQT6)
        self.assertIsNone(m.CYAN)
        self.assertIsNone(m.PANEL_DARK)
        # The renderer classes are still defined (on stub bases) so the module
        # object is whole — referencing them must not raise NameError.
        self.assertIs(m.StarkStatusRingScene.__bases__[0], object)
        self.assertIs(m.StarkStatusRingWindow.__bases__[0], object)
        # End-to-end: main() takes the graceful-degrade path — prints the
        # install hint and returns 2 without constructing a QApplication.
        with mock.patch.object(m.sys, "argv", ["hud_v2.py"]), \
                mock.patch.object(m, "_print_install_hint") as hint:
            rc = m.main()
        self.assertEqual(rc, 2)
        hint.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
#  _is_parent_alive
# ═══════════════════════════════════════════════════════════════════════════
class ParentAliveTests(_HudBase):
    def test_nonpositive_pid_always_alive(self):
        # pid <= 0 means "no parent supplied" → treat as alive.
        self.assertTrue(self.hud._is_parent_alive(0))
        self.assertTrue(self.hud._is_parent_alive(-5))

    def test_uses_psutil_when_present(self):
        with mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "pid_exists",
                                  return_value=True) as pe:
            self.assertTrue(self.hud._is_parent_alive(1234))
        pe.assert_called_once_with(1234)

    def test_psutil_says_dead(self):
        with mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "pid_exists",
                                  return_value=False):
            self.assertFalse(self.hud._is_parent_alive(1234))

    def test_psutil_exception_defaults_alive(self):
        # A psutil hiccup must not kill the renderer — default to alive.
        with mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "pid_exists",
                                  side_effect=RuntimeError("boom")):
            self.assertTrue(self.hud._is_parent_alive(1234))

    def test_oskill_fallback_alive(self):
        with mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(self.hud.os, "kill",
                                  return_value=None) as k:
            self.assertTrue(self.hud._is_parent_alive(4321))
        k.assert_called_once_with(4321, 0)

    def test_oskill_fallback_dead(self):
        with mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(self.hud.os, "kill",
                                  side_effect=ProcessLookupError()):
            self.assertFalse(self.hud._is_parent_alive(4321))

    def test_oskill_permission_error_treated_dead(self):
        # The source's except tuple includes PermissionError → returns False.
        with mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(self.hud.os, "kill",
                                  side_effect=PermissionError()):
            self.assertFalse(self.hud._is_parent_alive(4321))


# ═══════════════════════════════════════════════════════════════════════════
#  _read_json / _control_says_off
# ═══════════════════════════════════════════════════════════════════════════
class ReadJsonTests(_HudBase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(self.hud._read_json(self.hud_state), {})

    def test_valid_json_parsed(self):
        self._write(self.hud_state, {"state": "Listening", "x": 1})
        self.assertEqual(self.hud._read_json(self.hud_state),
                         {"state": "Listening", "x": 1})

    def test_corrupt_json_returns_empty(self):
        with open(self.hud_state, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.hud._read_json(self.hud_state), {})

    def test_json_null_coerced_to_empty_dict(self):
        # ``json.load`` of literal ``null`` is None → source's ``or {}`` kicks in.
        with open(self.hud_state, "w", encoding="utf-8") as f:
            f.write("null")
        self.assertEqual(self.hud._read_json(self.hud_state), {})

    def test_control_says_off_true(self):
        self._write(self.control, {"mode": "OFF"})   # case-insensitive
        self.assertTrue(self.hud._control_says_off())

    def test_control_says_off_false_when_on(self):
        self._write(self.control, {"mode": "on"})
        self.assertFalse(self.hud._control_says_off())

    def test_control_says_off_false_when_absent(self):
        self.assertFalse(self.hud._control_says_off())

    def test_control_says_off_handles_missing_mode_key(self):
        self._write(self.control, {"other": 1})
        self.assertFalse(self.hud._control_says_off())

    def test_control_says_off_handles_non_string_mode(self):
        # ``mode`` not a str → ``(data.get("mode") or "")`` keeps it falsey-safe.
        self._write(self.control, {"mode": None})
        self.assertFalse(self.hud._control_says_off())


# ═══════════════════════════════════════════════════════════════════════════
#  _sample_now_playing_safe — module lookup + import + failure swallow
# ═══════════════════════════════════════════════════════════════════════════
class NowPlayingSafeTests(_HudBase):
    def _inject(self, **attrs):
        """Install a fake ``skill_apple_music_intel`` module; restore on cleanup."""
        prev = sys.modules.get("skill_apple_music_intel", _SENTINEL)
        m = types.ModuleType("skill_apple_music_intel")
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules["skill_apple_music_intel"] = m

        def restore():
            if prev is _SENTINEL:
                sys.modules.pop("skill_apple_music_intel", None)
            else:
                sys.modules["skill_apple_music_intel"] = prev
        self.addCleanup(restore)
        return m

    def test_returns_sample_from_loaded_module(self):
        self._inject(_sample_now_playing=lambda: {"title": "T", "artist": "A"})
        self.assertEqual(self.hud._sample_now_playing_safe(),
                         {"title": "T", "artist": "A"})

    def test_none_when_module_absent_and_unimportable(self):
        # Pin the absent-sentinel so a deferred import can't reach a real module.
        prev = sys.modules.get("skill_apple_music_intel", _SENTINEL)
        sys.modules["skill_apple_music_intel"] = None
        self.addCleanup(
            lambda: (sys.modules.pop("skill_apple_music_intel", None)
                     if prev is _SENTINEL
                     else sys.modules.__setitem__(
                         "skill_apple_music_intel", prev)))
        with mock.patch.object(self.hud.importlib, "import_module",
                               side_effect=ImportError("nope")):
            self.assertIsNone(self.hud._sample_now_playing_safe())

    def test_none_when_module_lacks_function(self):
        self._inject(other=1)        # no _sample_now_playing attribute
        self.assertIsNone(self.hud._sample_now_playing_safe())

    def test_swallows_sampler_exception(self):
        def boom():
            raise RuntimeError("itunes COM hiccup")
        self._inject(_sample_now_playing=boom)
        self.assertIsNone(self.hud._sample_now_playing_safe())

    def test_imports_when_not_yet_in_sys_modules(self):
        # Module absent from sys.modules → source falls to import_module.
        prev = sys.modules.get("skill_apple_music_intel", _SENTINEL)
        sys.modules.pop("skill_apple_music_intel", None)
        self.addCleanup(
            lambda: None if prev is _SENTINEL
            else sys.modules.__setitem__("skill_apple_music_intel", prev))
        fake = types.ModuleType("skill_apple_music_intel")
        fake._sample_now_playing = lambda: {"title": "Imported"}
        with mock.patch.object(self.hud.importlib, "import_module",
                               return_value=fake) as imp:
            out = self.hud._sample_now_playing_safe()
        imp.assert_called_once_with("skill_apple_music_intel")
        self.assertEqual(out, {"title": "Imported"})


# ═══════════════════════════════════════════════════════════════════════════
#  _get_first_meeting_safe — is_configured gate, today/14-day fallback
# ═══════════════════════════════════════════════════════════════════════════
class FirstMeetingSafeTests(_HudBase):
    def _inject(self, **attrs):
        prev = sys.modules.get("skill_ms_graph", _SENTINEL)
        m = types.ModuleType("skill_ms_graph")
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules["skill_ms_graph"] = m

        def restore():
            if prev is _SENTINEL:
                sys.modules.pop("skill_ms_graph", None)
            else:
                sys.modules["skill_ms_graph"] = prev
        self.addCleanup(restore)
        return m

    def test_none_when_module_absent(self):
        prev = sys.modules.get("skill_ms_graph", _SENTINEL)
        sys.modules["skill_ms_graph"] = None
        self.addCleanup(
            lambda: (sys.modules.pop("skill_ms_graph", None)
                     if prev is _SENTINEL
                     else sys.modules.__setitem__("skill_ms_graph", prev)))
        with mock.patch.object(self.hud.importlib, "import_module",
                               side_effect=ImportError):
            self.assertIsNone(self.hud._get_first_meeting_safe())

    def test_none_when_not_configured(self):
        self._inject(is_configured=lambda: False,
                     get_first_meeting=lambda when: {"subject": "X"})
        self.assertIsNone(self.hud._get_first_meeting_safe())

    def test_none_when_is_configured_raises(self):
        def boom():
            raise RuntimeError
        self._inject(is_configured=boom,
                     get_first_meeting=lambda when: {"subject": "X"})
        self.assertIsNone(self.hud._get_first_meeting_safe())

    def test_none_when_no_get_first_meeting(self):
        self._inject(is_configured=lambda: True)   # missing get_first_meeting
        self.assertIsNone(self.hud._get_first_meeting_safe())

    def test_returns_today_meeting(self):
        calls = []

        def gfm(when):
            calls.append(when)
            return {"subject": "Standup"} if when == "today" else None
        self._inject(is_configured=lambda: True, get_first_meeting=gfm)
        self.assertEqual(self.hud._get_first_meeting_safe(),
                         {"subject": "Standup"})
        self.assertEqual(calls, ["today"])     # short-circuits, no 14-day call

    def test_falls_back_to_next_14_days(self):
        def gfm(when):
            return None if when == "today" else {"subject": "Later"}
        self._inject(is_configured=lambda: True, get_first_meeting=gfm)
        self.assertEqual(self.hud._get_first_meeting_safe(),
                         {"subject": "Later"})

    def test_no_is_configured_attr_still_queries(self):
        # is_configured is optional; absent → straight to get_first_meeting.
        self._inject(get_first_meeting=lambda when: {"subject": "NoGate"})
        self.assertEqual(self.hud._get_first_meeting_safe(),
                         {"subject": "NoGate"})

    def test_swallows_get_first_meeting_exception(self):
        def gfm(when):
            raise RuntimeError("graph 500")
        self._inject(is_configured=lambda: True, get_first_meeting=gfm)
        self.assertIsNone(self.hud._get_first_meeting_safe())


# ═══════════════════════════════════════════════════════════════════════════
#  _format_meeting — relative-time string buckets
# ═══════════════════════════════════════════════════════════════════════════
class FormatMeetingTests(_HudBase):
    def test_none_event(self):
        self.assertEqual(self.hud._format_meeting(None), ("", ""))

    def test_empty_dict(self):
        self.assertEqual(self.hud._format_meeting({}), ("", ""))

    def test_subject_only_when_no_start(self):
        self.assertEqual(self.hud._format_meeting({"subject": "Solo"}),
                         ("Solo", ""))

    def test_subject_stripped(self):
        self.assertEqual(
            self.hud._format_meeting({"subject": "  Spaced  "}),
            ("Spaced", ""))

    def test_start_not_datetime_returns_blank_when(self):
        self.assertEqual(
            self.hud._format_meeting({"subject": "S", "start": "2026-01-01"}),
            ("S", ""))

    def test_minutes_bucket(self):
        # Pad +30 s so the few microseconds between this now() and the one
        # inside _format_meeting can't tip ``secs // 60`` down a minute.
        start = (datetime.datetime.now()
                 + datetime.timedelta(minutes=12, seconds=30))
        subj, when = self.hud._format_meeting({"subject": "Standup",
                                               "start": start})
        self.assertEqual(subj, "Standup")
        self.assertEqual(when, "in 12 min")

    def test_now_bucket_under_a_minute(self):
        start = datetime.datetime.now() + datetime.timedelta(seconds=30)
        _subj, when = self.hud._format_meeting({"subject": "S", "start": start})
        self.assertEqual(when, "now")

    def test_now_bucket_recent_past(self):
        # secs <= -60 → "now" (meeting just started).
        start = datetime.datetime.now() - datetime.timedelta(minutes=5)
        _subj, when = self.hud._format_meeting({"subject": "S", "start": start})
        self.assertEqual(when, "now")

    def test_hours_bucket(self):
        start = datetime.datetime.now() + datetime.timedelta(hours=3)
        _subj, when = self.hud._format_meeting({"subject": "S", "start": start})
        self.assertTrue(when.startswith("in 3."))
        self.assertTrue(when.endswith(" h"))

    def test_days_bucket(self):
        start = datetime.datetime.now() + datetime.timedelta(days=2, hours=1)
        _subj, when = self.hud._format_meeting({"subject": "S", "start": start})
        self.assertEqual(when, "in 2 d")

    def test_aware_datetime_is_normalised(self):
        # An aware start must still produce a sane delta (source strips tzinfo).
        start = (datetime.datetime.now().astimezone()
                 + datetime.timedelta(minutes=20))
        _subj, when = self.hud._format_meeting({"subject": "S", "start": start})
        self.assertTrue(when.endswith("min"))


# ═══════════════════════════════════════════════════════════════════════════
#  Scene construction + layout math
# ═══════════════════════════════════════════════════════════════════════════
class SceneLayoutTests(_HudBase):
    def test_construct_primes_defaults(self):
        s = self._new_scene(460, 340, pid=99)
        self.assertEqual(s.w, 460.0)
        self.assertEqual(s.h, 340.0)
        self.assertEqual(s.parent_pid, 99)
        self.assertEqual(s.frame, 0)
        self.assertEqual(s.state, "idle")
        self.assertEqual(s.cpu_pct, 0.0)
        self.assertFalse(s.bambu_active)
        self.assertEqual(s.track_title, "")
        self.assertEqual(s.cal_subject, "")

    def test_layout_centers_and_scales_off_short_axis(self):
        s = self._new_scene(460, 340)
        ref = min(460.0, 340.0)
        self.assertEqual(s.cx, 230.0)
        self.assertEqual(s.cy, 170.0)
        self.assertAlmostEqual(s.R_OUTER, ref * 0.36)
        self.assertAlmostEqual(s.R_INNER_BAMBU, ref * 0.28)
        self.assertAlmostEqual(s.R_CORE, ref * 0.18)
        self.assertAlmostEqual(s.R_HUB, ref * 0.10)
        self.assertAlmostEqual(s.R_GLOW, ref * 0.48)

    def test_layout_uses_height_when_taller_is_narrow(self):
        # Tall-narrow panel → ref is the width.
        s = self._new_scene(200, 800)
        self.assertAlmostEqual(s.R_OUTER, 200.0 * 0.36)
        self.assertEqual(s.cx, 100.0)
        self.assertEqual(s.cy, 400.0)

    def test_resize_scene_recomputes(self):
        s = self._new_scene(460, 340)
        s.resize_scene(800, 600)
        self.assertEqual(s.w, 800.0)
        self.assertEqual(s.h, 600.0)
        self.assertEqual(s.cx, 400.0)
        self.assertAlmostEqual(s.R_OUTER, 600.0 * 0.36)

    def test_construct_primes_psutil_when_present(self):
        # Re-construct with psutil present and assert cpu_percent primed once.
        with mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "cpu_percent") as cp:
            self._new_scene()
        cp.assert_called_once_with(interval=None)

    def test_construct_swallows_psutil_prime_error(self):
        with mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "cpu_percent",
                                  side_effect=RuntimeError):
            # Must not raise.
            self._new_scene()


# ═══════════════════════════════════════════════════════════════════════════
#  Pure helpers: _accent_for_state, _color_for_metric, _fraction, _truncate
# ═══════════════════════════════════════════════════════════════════════════
class HelperTests(_HudBase):
    def test_accent_for_state_map(self):
        s = self._new_scene()
        cases = {
            "listening": self.hud.AMBER,
            "speaking": self.hud.AMBER_BRIGHT,
            "thinking": self.hud.CYAN_BRIGHT,
            "standby": self.hud.CYAN_DIM,
            "sleep": self.hud.CYAN_DIM,
            "idle": self.hud.CYAN,
            "anything_else": self.hud.CYAN,
        }
        for state, expected in cases.items():
            s.state = state
            self.assertIs(s._accent_for_state(), expected, state)

    def test_color_for_metric_thresholds(self):
        cfm = self.hud.StarkStatusRingScene._color_for_metric
        self.assertIs(cfm(95.0, 75.0, 90.0), self.hud.RED)    # >= crit
        self.assertIs(cfm(90.0, 75.0, 90.0), self.hud.RED)    # == crit
        self.assertIs(cfm(80.0, 75.0, 90.0), self.hud.AMBER)  # >= warn
        self.assertIs(cfm(75.0, 75.0, 90.0), self.hud.AMBER)  # == warn
        self.assertIs(cfm(50.0, 75.0, 90.0), self.hud.CYAN)   # below warn

    def test_fraction_clamps(self):
        frac = self.hud.StarkStatusRingScene._fraction
        self.assertEqual(frac(50.0, 100.0), 0.5)
        self.assertEqual(frac(-10.0, 100.0), 0.0)     # clamped low
        self.assertEqual(frac(150.0, 100.0), 1.0)     # clamped high
        self.assertEqual(frac(50.0, 0.0), 0.0)        # full_at <= 0 guard
        self.assertEqual(frac(50.0, -1.0), 0.0)

    def test_truncate(self):
        trunc = self.hud.StarkStatusRingScene._truncate
        self.assertEqual(trunc("short", 10), "short")        # unchanged
        self.assertEqual(trunc("exactly10!", 10), "exactly10!")
        out = trunc("this is a long line", 10)
        self.assertEqual(len(out), 10)
        self.assertTrue(out.endswith("…"))
        # max_chars <= 1 takes the hard-slice branch (no ellipsis room).
        self.assertEqual(trunc("abcdef", 1), "a")
        self.assertEqual(trunc("abcdef", 0), "")

    def test_truncate_rstrips_before_ellipsis(self):
        # "hello " → trailing space stripped before the ellipsis is appended.
        self.assertEqual(self.hud.StarkStatusRingScene._truncate("hello world", 7),
                         "hello…")


# ═══════════════════════════════════════════════════════════════════════════
#  _read_gpu_temp — cache, nvidia-smi parse, pulse_strip fallback
# ═══════════════════════════════════════════════════════════════════════════
class GpuTempTests(_HudBase):
    def test_cache_returns_prev_within_window(self):
        s = self._new_scene()
        s.gpu_temp_c = 55.0
        s._gpu_cached_at = 1000.0
        with mock.patch.object(self.hud.time, "time", return_value=1001.0), \
                mock.patch.object(self.hud.shutil, "which") as which:
            self.assertEqual(s._read_gpu_temp(), 55.0)
        which.assert_not_called()       # still inside the 4 s cache window

    def test_nvidia_smi_parse_max_temp(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        out = mock.MagicMock()
        out.stdout = "61\n67\n"
        with mock.patch.object(self.hud.time, "time", return_value=10_000.0), \
                mock.patch.object(self.hud.shutil, "which",
                                  return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(self.hud.subprocess, "run", return_value=out):
            self.assertEqual(s._read_gpu_temp(), 67.0)   # max of the two

    def test_nvidia_smi_absent_falls_back_to_pulse_strip(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        self._write(self.hud_state, {"pulse_strip": "CPU 12% GPU 49C RAM 30%"})
        with mock.patch.object(self.hud.time, "time", return_value=20_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            self.assertEqual(s._read_gpu_temp(), 49.0)

    def test_pulse_strip_decimal_temp(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        self._write(self.hud_state, {"pulse_strip": "GPU 48.5C"})
        with mock.patch.object(self.hud.time, "time", return_value=21_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            self.assertEqual(s._read_gpu_temp(), 48.5)

    def test_no_nvidia_no_strip_returns_none(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        with mock.patch.object(self.hud.time, "time", return_value=22_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            self.assertIsNone(s._read_gpu_temp())

    def test_strip_without_gpu_token_returns_none(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        self._write(self.hud_state, {"pulse_strip": "CPU 12% RAM 30%"})
        with mock.patch.object(self.hud.time, "time", return_value=23_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            self.assertIsNone(s._read_gpu_temp())

    def test_nvidia_smi_empty_output_falls_through(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        out = mock.MagicMock()
        out.stdout = "\n  \n"          # no digit lines
        self._write(self.hud_state, {"pulse_strip": "GPU 40C"})
        with mock.patch.object(self.hud.time, "time", return_value=24_000.0), \
                mock.patch.object(self.hud.shutil, "which",
                                  return_value="nvidia-smi"), \
                mock.patch.object(self.hud.subprocess, "run", return_value=out):
            # No temps from nvidia-smi → pulse_strip fallback used.
            self.assertEqual(s._read_gpu_temp(), 40.0)

    def test_subprocess_exception_falls_through_to_strip(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        self._write(self.hud_state, {"pulse_strip": "GPU 51C"})
        with mock.patch.object(self.hud.time, "time", return_value=25_000.0), \
                mock.patch.object(self.hud.shutil, "which",
                                  return_value="nvidia-smi"), \
                mock.patch.object(self.hud.subprocess, "run",
                                  side_effect=OSError("spawn failed")):
            self.assertEqual(s._read_gpu_temp(), 51.0)

    def test_updates_cache_timestamp(self):
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        with mock.patch.object(self.hud.time, "time", return_value=30_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            s._read_gpu_temp()
        self.assertEqual(s._gpu_cached_at, 30_000.0)

    def test_malformed_strip_decimal_swallowed_returns_none(self):
        # A pulse_strip whose GPU value collects multiple dots ("4.8.5") makes
        # float() raise inside the parse -> the inner `except Exception: pass`
        # swallows it and the function returns None.
        s = self._new_scene()
        s._gpu_cached_at = 0.0
        self._write(self.hud_state, {"pulse_strip": "GPU 4.8.5C"})
        with mock.patch.object(self.hud.time, "time", return_value=26_000.0), \
                mock.patch.object(self.hud.shutil, "which", return_value=None):
            self.assertIsNone(s._read_gpu_temp())


# ═══════════════════════════════════════════════════════════════════════════
#  refresh_data — the QTimer-driven state pull + close-decision
# ═══════════════════════════════════════════════════════════════════════════
class RefreshDataTests(_HudBase):
    def _scene_no_slow(self, **kw):
        """A scene whose slow track/calendar refreshers are stubbed out so
        refresh_data tests stay offline and deterministic."""
        s = self._new_scene(**kw)
        return s

    def test_returns_false_when_parent_dead(self):
        s = self._new_scene(pid=555)
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=False):
            self.assertFalse(s.refresh_data())

    def test_returns_false_when_control_off(self):
        s = self._new_scene()
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=True):
            self.assertFalse(s.refresh_data())

    def test_happy_path_reads_state_and_returns_true(self):
        s = self._scene_no_slow()
        self._write(self.hud_state, {"state": "Listening",
                                     "tts_amplitude": 0.4, "mic_level": 0.7})
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            ok = s.refresh_data()
        self.assertTrue(ok)
        self.assertEqual(s.state, "listening")     # lowercased
        self.assertEqual(s.tts_amp, 0.4)
        self.assertEqual(s.mic_level, 0.7)
        self.assertEqual(s.frame, 1)               # frame advanced

    def test_state_defaults_to_idle_when_absent(self):
        s = self._scene_no_slow()
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertEqual(s.state, "idle")

    def test_bad_amplitude_values_coerced_to_zero(self):
        s = self._scene_no_slow()
        self._write(self.hud_state, {"state": "Idle",
                                     "tts_amplitude": "loud",
                                     "mic_level": "quiet"})
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertEqual(s.tts_amp, 0.0)
        self.assertEqual(s.mic_level, 0.0)

    def test_reads_psutil_metrics(self):
        s = self._scene_no_slow()
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "cpu_percent",
                                  return_value=42.0), \
                mock.patch.object(self.hud.psutil, "virtual_memory") as vm, \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            vm.return_value.percent = 63.0
            s.refresh_data()
        self.assertEqual(s.cpu_pct, 42.0)
        self.assertEqual(s.ram_pct, 63.0)

    def test_psutil_exception_swallowed(self):
        s = self._scene_no_slow()
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", True), \
                mock.patch.object(self.hud.psutil, "cpu_percent",
                                  side_effect=RuntimeError), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            ok = s.refresh_data()
        self.assertTrue(ok)        # error swallowed, still returns True

    def test_bambu_active_states_parsed(self):
        s = self._scene_no_slow()
        self._write(self.bambu_state, {"gcode_state": "running",
                                       "mc_percent": 73})
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertEqual(s.bambu_gcode, "RUNNING")    # upper-cased
        self.assertTrue(s.bambu_active)
        self.assertEqual(s.bambu_percent, 73)

    def test_bambu_inactive_state(self):
        s = self._scene_no_slow()
        self._write(self.bambu_state, {"gcode_state": "FINISH",
                                       "mc_percent": "garbage"})
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertFalse(s.bambu_active)
        self.assertEqual(s.bambu_percent, 0)    # bad mc_percent → 0

    def test_track_refresh_on_frame_zero(self):
        s = self._scene_no_slow()
        s.frame = 0       # frame % TRACK_REFRESH_TICKS == 0 → sample now-playing
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value={"title": " Song ",
                                                "artist": " Band "}), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertEqual(s.track_title, "Song")     # stripped
        self.assertEqual(s.track_artist, "Band")

    def test_track_cleared_when_sample_none(self):
        s = self._scene_no_slow()
        s.frame = 0
        s.track_title = "stale"
        s.track_artist = "old"
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        self.assertEqual(s.track_title, "")
        self.assertEqual(s.track_artist, "")

    def test_track_not_refreshed_off_cadence(self):
        s = self._scene_no_slow()
        s.frame = 1       # 1 % 12 != 0 → sampler NOT called
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe") as nps, \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value=None):
            s.refresh_data()
        nps.assert_not_called()

    def test_calendar_refresh_on_cadence(self):
        s = self._scene_no_slow()
        # (frame + 6) % 240 == 0 → frame 234 triggers the calendar refresh.
        s.frame = 234
        start = (datetime.datetime.now()
                 + datetime.timedelta(minutes=8, seconds=30))
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe",
                                  return_value={"subject": "Sync",
                                                "start": start}) as gfm:
            s.refresh_data()
        gfm.assert_called_once()
        self.assertEqual(s.cal_subject, "Sync")
        self.assertEqual(s.cal_when, "in 8 min")

    def test_calendar_not_refreshed_off_cadence(self):
        s = self._scene_no_slow()
        s.frame = 0       # (0+6) % 240 != 0 → calendar NOT queried
        with mock.patch.object(self.hud, "_is_parent_alive", return_value=True), \
                mock.patch.object(self.hud, "_control_says_off",
                                  return_value=False), \
                mock.patch.object(self.hud, "_HAS_PSUTIL", False), \
                mock.patch.object(s, "_read_gpu_temp", return_value=None), \
                mock.patch.object(self.hud, "_sample_now_playing_safe",
                                  return_value=None), \
                mock.patch.object(self.hud, "_get_first_meeting_safe") as gfm:
            s.refresh_data()
        gfm.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
#  drawBackground — drive the full paint pass across state/data permutations.
#  Painter is a mock, so this exercises the geometry/branch logic (the bulk of
#  the file) without a real GPU surface.
# ═══════════════════════════════════════════════════════════════════════════
class DrawBackgroundTests(_HudBase):
    def test_paints_idle_default(self):
        s = self._new_scene()
        painter = self._paint(s)        # must not raise
        self.assertTrue(painter.drawEllipse.called)

    def test_paints_every_speech_state(self):
        for state in ("idle", "listening", "thinking", "speaking",
                      "standby", "sleep", "weird"):
            s = self._new_scene()
            s.state = state
            s.tts_amp = 0.6
            s.mic_level = 0.5
            self._paint(s)              # each state's hub-brightness branch

    def test_paints_with_active_bambu_running(self):
        s = self._new_scene()
        s.bambu_active = True
        s.bambu_gcode = "RUNNING"
        s.bambu_percent = 55
        self._paint(s)

    def test_paints_with_bambu_paused(self):
        s = self._new_scene()
        s.bambu_active = True
        s.bambu_gcode = "PAUSE"
        s.bambu_percent = 30
        self._paint(s)

    def test_paints_with_bambu_prepare(self):
        s = self._new_scene()
        s.bambu_active = True
        s.bambu_gcode = "PREPARE"
        s.bambu_percent = 0
        self._paint(s)

    def test_paints_with_gpu_temp_present(self):
        s = self._new_scene()
        s.gpu_temp_c = 78.0      # drives the hot-GPU chip + coloured arc
        s.cpu_pct = 95.0         # crit → RED arc + chip
        s.ram_pct = 80.0         # warn → AMBER
        self._paint(s)

    def test_paints_with_gpu_temp_absent(self):
        s = self._new_scene()
        s.gpu_temp_c = None      # "GPU — °C" placeholder chip + dim arc
        self._paint(s)

    def test_paints_with_track_full(self):
        s = self._new_scene()
        s.track_title = "Title"
        s.track_artist = "Artist"
        self._paint(s)

    def test_paints_with_track_title_only(self):
        s = self._new_scene()
        s.track_title = "Solo Title"
        s.track_artist = ""
        self._paint(s)

    def test_paints_with_track_artist_only(self):
        s = self._new_scene()
        s.track_title = ""
        s.track_artist = "Only Artist"
        self._paint(s)

    def test_paints_with_long_track_truncated(self):
        s = self._new_scene()
        s.track_artist = "A Very Long Artist Name Indeed"
        s.track_title = "An Equally Lengthy Track Title That Overflows The Row"
        self._paint(s)

    def test_paints_with_meeting_and_when(self):
        s = self._new_scene()
        s.cal_subject = "Standup"
        s.cal_when = "in 14 min"
        self._paint(s)

    def test_paints_with_meeting_no_when(self):
        s = self._new_scene()
        s.cal_subject = "All-day offsite"
        s.cal_when = ""
        self._paint(s)

    def test_paints_with_empty_state_string(self):
        # state == "" exercises the ``self.state.upper() if self.state else``
        # ternary's else branch and the centre-label fallback.
        s = self._new_scene()
        s.state = ""
        self._paint(s)

    def test_paints_advances_decorative_tick_each_frame(self):
        s = self._new_scene()
        s.frame = 7        # nonzero spin/pulse phase
        self._paint(s)

    def test_paints_full_house(self):
        # Everything on at once — the busiest single frame.
        s = self._new_scene()
        s.state = "speaking"
        s.tts_amp = 0.9
        s.cpu_pct = 88.0
        s.ram_pct = 91.0
        s.gpu_temp_c = 84.0
        s.bambu_active = True
        s.bambu_gcode = "RUNNING"
        s.bambu_percent = 99
        s.track_title = "Song"
        s.track_artist = "Band"
        s.cal_subject = "Demo"
        s.cal_when = "now"
        s.frame = 41
        painter = self._paint(s)
        self.assertTrue(painter.drawText.called)

    def test_paints_tiny_scene_clamps_chip_positions(self):
        # A very small scene forces the ``max(8.0, ...)`` / ``min(w-88, ...)``
        # chip-position clamps to bite.
        s = self._new_scene(120, 120)
        s.gpu_temp_c = 60.0
        self._paint(s)


# ═══════════════════════════════════════════════════════════════════════════
#  Window + _on_tick + main() + install hint
# ═══════════════════════════════════════════════════════════════════════════
class WindowTests(_HudBase):
    def _make_window(self):
        return self.hud.StarkStatusRingWindow(100, 200, 460, 340, parent_pid=0)

    def test_window_constructs_with_scene_and_timer(self):
        win = self._make_window()
        self.assertIsInstance(win.scene, self.hud.StarkStatusRingScene)
        self.assertEqual(win.parent_pid, 0)
        # Timer was created, interval set to TICK_MS, and started.
        self.assertEqual(win.timer.interval, self.hud.TICK_MS)
        self.assertTrue(win.timer.started)

    def test_on_tick_keeps_running_when_alive(self):
        win = self._make_window()
        with mock.patch.object(win.scene, "refresh_data", return_value=True):
            win._on_tick()
        # Still running — timer not stopped (started flag remains True).
        self.assertTrue(win.timer.started)

    def test_on_tick_stops_and_quits_when_dead(self):
        win = self._make_window()
        app = self.hud.QApplication.instance()
        with mock.patch.object(win.scene, "refresh_data", return_value=False), \
                mock.patch.object(app, "quit") as quit_:
            win._on_tick()
        self.assertFalse(win.timer.started)     # timer.stop() called
        quit_.assert_called_once()

    def test_translucent_attr_failure_is_swallowed(self):
        # The source wraps ONLY the click-through (WA_TransparentForMouseEvents)
        # setAttribute in try/except — some Windows builds raise on it. Make a
        # real setAttribute that raises for exactly that flag and assert the
        # window still constructs (the WA_TranslucentBackground call must still
        # succeed). Patched on the fake base so it overrides the auto-mock.
        transparent = self.hud.Qt.WidgetAttribute.WA_TransparentForMouseEvents

        def fake_set_attribute(self2, attr, on=True):
            if attr == transparent:
                raise RuntimeError("unsupported flag")
            return None

        with mock.patch.object(_FakeQWidget, "setAttribute", fake_set_attribute,
                               create=True):
            win = self._make_window()      # must not raise
        self.assertIsNotNone(win)


class MainEntryTests(_HudBase):
    def test_print_install_hint_writes_stderr(self):
        buf = []
        with mock.patch.object(self.hud.sys, "stderr") as err:
            err.write = lambda s: buf.append(s)
            self.hud._print_install_hint()
        # The print() emitted at least one chunk mentioning PyQt6.
        self.assertTrue(any("PyQt6" in s for s in buf))

    def test_main_returns_2_when_pyqt_absent(self):
        argv = ["hud_v2.py"]
        with mock.patch.object(self.hud, "_HAS_PYQT6", False), \
                mock.patch.object(self.hud.sys, "argv", argv), \
                mock.patch.object(self.hud, "_print_install_hint") as hint:
            rc = self.hud.main()
        self.assertEqual(rc, 2)
        hint.assert_called_once()

    def test_main_constructs_window_and_runs_app(self):
        argv = ["hud_v2.py", "--x", "10", "--y", "20",
                "--width", "300", "--height", "200", "--parent-pid", "0"]
        fake_app = mock.MagicMock()
        fake_app.exec.return_value = 0
        with mock.patch.object(self.hud, "_HAS_PYQT6", True), \
                mock.patch.object(self.hud.sys, "argv", argv), \
                mock.patch.object(self.hud, "QApplication",
                                  return_value=fake_app) as app_cls, \
                mock.patch.object(self.hud, "StarkStatusRingWindow") as win_cls:
            rc = self.hud.main()
        self.assertEqual(rc, 0)
        app_cls.assert_called_once()
        win_cls.assert_called_once()
        win_cls.return_value.show.assert_called_once()
        fake_app.exec.assert_called_once()

    def test_main_parses_custom_geometry_into_window(self):
        argv = ["hud_v2.py", "--x", "111", "--y", "-222",
                "--width", "333", "--height", "444", "--parent-pid", "777"]
        fake_app = mock.MagicMock()
        fake_app.exec.return_value = 5
        with mock.patch.object(self.hud, "_HAS_PYQT6", True), \
                mock.patch.object(self.hud.sys, "argv", argv), \
                mock.patch.object(self.hud, "QApplication",
                                  return_value=fake_app), \
                mock.patch.object(self.hud, "StarkStatusRingWindow") as win_cls:
            rc = self.hud.main()
        self.assertEqual(rc, 5)
        args = win_cls.call_args.args
        self.assertEqual(args, (111, -222, 333, 444, 777))


if __name__ == "__main__":
    unittest.main()
