"""Logic tests for skills/holographic_overlay (package skill).

This is a *visual* skill: ``skills/holographic_overlay/__init__.py`` is the
~996-statement manager that drives EIGHT on-screen reactor surfaces, each a
separate PyQt/tkinter renderer it spawns as its own subprocess:

  • fullscreen overlay (hud/jarvis_holo.py)
  • workshop canvas (hud/holo_workshop_canvas.py)
  • bambu H2D corner overlay (hud/bambu_h2d_overlay.py)
  • workshop HUD (hud/workshop_hud.py)
  • workshop print monitor (hud/workshop_print_monitor.py)
  • holographic HUD v2 (hud/holographic_hud_v2.py)
  • arc-reactor status HUD (hud/arc_reactor_status_hud.py)
  • Stark status ring (skills/holographic_overlay/hud_v2.py)

The actual Qt rendering lives in those *child-process* scripts. The manager
``__init__.py`` itself imports ONLY stdlib (json/logging/os/subprocess/sys/
threading/time) — no Qt — so its testable surface is pure non-visual logic:
geometry/layout math, alive/dormant state machines, the atomic control-file
writers, launch/shutdown lifecycles (missing-script + Popen-failure + idempotent
paths), the auto-show watcher decisions, every registered voice action, and the
config-flag auto-launch branches in register().

ISOLATION CONTRACT
  • Every test loads a FRESH module via ``load_skill_isolated`` (the harness
    re-execs it, so module globals — _*_PROCESS, _*_WATCHER_STARTED,
    _*_USER_OFF — start clean each test). ``subprocess.Popen`` is mocked across
    the load so register()'s auto-launch (workshop_hud is on by default) can
    never spawn a real HUD, and the harness no-ops Thread.start so watcher
    threads never run.
  • The manager's control/state JSON files normally live in the real project
    dir; the base class redirects every one of them into a per-test temp dir so
    no real project file is read or written, then removes the dir on cleanup.
  • ``bobert_companion`` is never booted: tests that need it inject a fake via a
    save/restore of sys.modules (mirroring test_self_diagnostic's contract) and
    restore the prior state — including absence — on cleanup.
  • Qt is irrelevant to this module (it shells out), so there are no fake-Qt
    modules to inject here; the proof the manager works with Qt absent is that
    the whole suite imports + exercises it using ONLY stdlib + mocks, exactly
    as the CI runner (no PyQt) would.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install/replace entries in ``sys.modules`` for the with-block,
    restoring the previous state (including ABSENCE) on exit.

    A value of ``None`` installs the import-system's "known-absent" sentinel
    (``sys.modules[name] = None``), which makes a deferred ``import name`` raise
    ImportError *without* re-loading the real module from disk. This is the
    isolation lever that keeps a deferred ``import bobert_companion`` from
    booting the real ~14K-line monolith on a dev box that has it: every test
    either pins a fake module or pins the None-sentinel. Mirrors the
    save/restore helper in test_self_diagnostic.
    """
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        sys.modules[name] = obj          # obj may be None (absent-sentinel)
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _fake_proc(alive=True):
    """A fake Popen handle. ``poll()`` returns None while 'alive', else 0.
    terminate/kill/wait are recorded no-ops so shutdown logic can be asserted
    without a real process or OS handle."""
    proc = mock.MagicMock(name="FakePopen")
    proc.poll.return_value = None if alive else 0
    proc.terminate = mock.Mock()
    proc.kill = mock.Mock()
    proc.wait = mock.Mock()
    return proc


def _fake_bobert(**flags):
    """A stand-in ``bobert_companion`` module. Attributes set via ``flags``
    pin specific config (e.g. ``WORKSHOP_HUD_AUTO_LAUNCH=False``); anything not
    set is simply absent, so the skill's ``getattr(_bc, FLAG, default)`` reads
    fall through to the documented defaults. Installing this (instead of letting
    a deferred ``import bobert_companion`` reach disk) is what guarantees the
    real monolith is NEVER imported during a test."""
    bc = types.ModuleType("bobert_companion")
    for k, v in flags.items():
        setattr(bc, k, v)
    return bc


def _load_isolated(bobert=None):
    """Load the package skill in isolation.

    • ``subprocess.Popen`` is mocked so any auto-launch can't spawn a real HUD.
    • A fake ``bobert_companion`` is installed for the whole load so register()'s
      deferred ``import bobert_companion`` never boots the real monolith. By
      default the fake pins ``WORKSHOP_HUD_AUTO_LAUNCH=False`` so the default
      auto-launch is suppressed during the common setUp (no Popen, no real
      control-file writes); the watcher-start flags are left absent so the
      watchers still arm on their True defaults.
    • ``open``/``os.replace`` are stubbed during the load so that even if a
      surface does auto-launch, its atomic control-file write can't touch a real
      project file (paths aren't redirected until after the module exists).
    """
    if bobert is None:
        bobert = _fake_bobert(WORKSHOP_HUD_AUTO_LAUNCH=False)
    with inject_modules(bobert_companion=bobert), \
            mock.patch.object(subprocess, "Popen") as popen, \
            mock.patch("builtins.open", mock.mock_open()), \
            mock.patch.object(os, "replace"):
        popen.return_value = _fake_proc(alive=True)
        mod, actions = load_skill_isolated("holographic_overlay")
    return mod, actions


class _HoloBase(unittest.TestCase):
    """Loads a fresh isolated module and redirects ALL of the manager's
    control/state JSON file paths into a throwaway temp dir, so no test ever
    touches a real project file. Module globals are fresh per load (the harness
    re-execs), so there is no cross-test process/flag leakage to undo.
    """

    # Every ``*_FILE`` / ``*_SCRIPT`` global the manager references. We point
    # the JSON control/state files at temp paths; script paths are pointed at a
    # temp dir too so os.path.exists checks are deterministic (absent unless a
    # test creates them).
    _STATE_FILE_ATTRS = (
        "_WORKSHOP_STATE_FILE", "_HUD_STATE_FILE", "_BAMBU_OVERLAY_STATE_FILE",
        "_WORKSHOP_HUD_CONTROL_FILE", "_WORKSHOP_PRINT_MONITOR_CONTROL_FILE",
        "_ARC_STATUS_CONTROL_FILE", "_STARK_STATUS_CONTROL_FILE",
    )
    _SCRIPT_ATTRS = (
        "_OVERLAY_SCRIPT", "_WORKSHOP_SCRIPT", "_BAMBU_OVERLAY_SCRIPT",
        "_WORKSHOP_HUD_SCRIPT", "_WORKSHOP_PRINT_MONITOR_SCRIPT",
        "_HOLO_HUD_V2_SCRIPT", "_ARC_STATUS_SCRIPT", "_STARK_STATUS_SCRIPT",
    )

    def setUp(self):
        self.mod, self.actions = _load_isolated()
        # Keep a fake bobert_companion installed for the WHOLE test so any
        # deferred ``import bobert_companion`` inside a helper under test
        # (_get_monitor_rect / _maybe_start_* / register paths) resolves to the
        # fake and never boots the real monolith. Saved/restored so the dev
        # box's real module state is untouched after the test.
        self._saved_bc = sys.modules.get("bobert_companion", _SENTINEL)
        sys.modules["bobert_companion"] = _fake_bobert()
        self.addCleanup(self._restore_bc)
        self.tmp = tempfile.mkdtemp(prefix="holo_test_")
        self.addCleanup(self._cleanup_tmp)
        # Redirect control/state files into the temp dir.
        for attr in self._STATE_FILE_ATTRS:
            setattr(self.mod, attr,
                    os.path.join(self.tmp, attr.strip("_").lower() + ".json"))
        # Redirect renderer-script paths into the temp dir so existence is
        # deterministic (a test opts a script "present" by touching the path).
        for attr in self._SCRIPT_ATTRS:
            setattr(self.mod, attr,
                    os.path.join(self.tmp, attr.strip("_").lower() + ".py"))

    def _restore_bc(self):
        if self._saved_bc is _SENTINEL:
            sys.modules.pop("bobert_companion", None)
        else:
            sys.modules["bobert_companion"] = self._saved_bc

    def _cleanup_tmp(self):
        for fn in os.listdir(self.tmp):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(self.tmp, fn))
        with contextlib.suppress(OSError):
            os.rmdir(self.tmp)

    # ── helpers ──────────────────────────────────────────────────────────
    def _touch(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# renderer stub\n")

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  register() — action wiring + auto-launch config branches
# ═══════════════════════════════════════════════════════════════════════════
class RegisterTests(_HoloBase):
    def test_registers_every_documented_action_alias(self):
        # A broad slice across all eight surfaces must be wired.
        expected = (
            # fullscreen overlay
            "show_holographic_overlay", "show_holo", "hud_on", "holographic_on",
            "hide_holographic_overlay", "hide_holo", "hud_off", "dismiss_holo",
            "holographic_off", "toggle_holographic_overlay", "toggle_holo",
            "holographic_status",
            # workshop canvas
            "arc_reactor", "arc_reactor_on", "arc_reactor_off",
            "arc_reactor_pulse", "holo_workshop_canvas", "holo_workshop",
            "workshop_canvas",
            # bambu overlay
            "bambu_h2d_overlay", "bambu_overlay", "bambu_overlay_on",
            "show_bambu_overlay", "bambu_overlay_off", "hide_bambu_overlay",
            "bambu_overlay_toggle", "bambu_overlay_status",
            # workshop hud
            "workshop_hud", "show_workshop_hud", "workshop_hud_on",
            "workshop_hud_off", "workshop_hud_toggle", "workshop_hud_status",
            "hide_workshop_hud",
            # workshop print monitor
            "workshop_print_monitor", "workshop_print_monitor_on",
            "show_workshop_print_monitor", "workshop_print_monitor_off",
            "hide_workshop_print_monitor", "workshop_print_monitor_toggle",
            "workshop_print_monitor_status", "print_hud", "print_hud_on",
            "print_hud_off", "workshop_print_hud", "workshop_print_hud_on",
            "workshop_print_hud_off",
            # holo hud v2
            "holographic_hud_v2", "holo_hud_v2", "holo_hud_v2_on",
            "show_holo_hud_v2", "holo_hud_v2_off", "hide_holo_hud_v2",
            "holo_hud_v2_toggle", "holo_hud_v2_status", "arc_reactor_hud",
            # arc-reactor status hud
            "arc_reactor_status_hud", "arc_reactor_status",
            "arc_reactor_status_on", "arc_reactor_status_off",
            "arc_reactor_status_toggle", "arc_reactor_status_status",
            "status_hud", "status_hud_on", "show_status_hud", "status_hud_off",
            "hide_status_hud", "status_ring", "status_ring_on",
            "status_ring_off", "pulse_hud", "pulse_hud_on", "pulse_hud_off",
            # stark status ring
            "stark_status_ring", "stark_status_ring_on",
            "stark_status_ring_off", "stark_status_ring_toggle",
            "stark_status_ring_status", "hud_v2", "hud_v2_on", "show_hud_v2",
            "hud_v2_off", "hide_hud_v2", "hud_v2_toggle", "status_ring_v2",
            "status_ring_v2_on", "status_ring_v2_off", "show_status_ring_v2",
            "hide_status_ring_v2",
        )
        for name in expected:
            self.assertIn(name, self.actions, f"missing action: {name}")

    def test_does_not_register_hide_hud(self):
        # Explicit source contract: 'hide_hud' is owned by the monolith and
        # must NOT be shadowed here (it would reroute the main HUD closer).
        self.assertNotIn("hide_hud", self.actions)

    def test_every_registered_action_is_callable(self):
        for name, fn in self.actions.items():
            self.assertTrue(callable(fn), f"{name} not callable")

    @staticmethod
    def _load_with_flags(**flags):
        """Re-load the skill with a fake bobert carrying ``flags`` and every
        renderer script reported present, so auto-launch lifecycle branches in
        register() run. Popen + file IO are stubbed so nothing real spawns or is
        written. Returns the freshly loaded module."""
        bc = _fake_bobert(**flags)
        with inject_modules(bobert_companion=bc), \
                mock.patch("os.path.exists", return_value=True), \
                mock.patch("builtins.open", mock.mock_open()), \
                mock.patch.object(os, "replace"), \
                mock.patch.object(subprocess, "Popen") as popen:
            popen.return_value = _fake_proc(alive=True)
            mod, _actions = load_skill_isolated("holographic_overlay")
        return mod

    def test_workshop_hud_auto_launches_by_default(self):
        # WORKSHOP_HUD_AUTO_LAUNCH defaults True → register() spawns it once.
        # (Pass no flags → the getattr default of True applies.)
        mod = self._load_with_flags()
        self.assertIsNotNone(mod._WORKSHOP_HUD_PROCESS)

    def test_workshop_hud_auto_launch_suppressed_by_flag(self):
        mod = self._load_with_flags(WORKSHOP_HUD_AUTO_LAUNCH=False)
        self.assertIsNone(mod._WORKSHOP_HUD_PROCESS)

    def test_overlay_auto_launch_when_flag_true(self):
        mod = self._load_with_flags(HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH=True,
                                    WORKSHOP_HUD_AUTO_LAUNCH=False)
        self.assertIsNotNone(mod._OVERLAY_PROCESS)        # overlay launched
        self.assertIsNone(mod._WORKSHOP_HUD_PROCESS)      # hud suppressed

    def test_optional_surfaces_auto_launch_when_all_flags_true(self):
        mod = self._load_with_flags(
            HOLO_HUD_V2_AUTO_LAUNCH=True,
            HOLO_ARC_REACTOR_STATUS_AUTO_LAUNCH=True,
            HOLO_STARK_STATUS_RING_AUTO_LAUNCH=True,
            WORKSHOP_HUD_AUTO_LAUNCH=False,
        )
        self.assertIsNotNone(mod._HOLO_HUD_V2_PROCESS)
        self.assertIsNotNone(mod._ARC_STATUS_PROCESS)
        self.assertIsNotNone(mod._STARK_STATUS_PROCESS)

    def test_optional_surfaces_dormant_by_default(self):
        # All three optional surfaces default OFF (flags absent → getattr False).
        mod = self._load_with_flags(WORKSHOP_HUD_AUTO_LAUNCH=False)
        self.assertIsNone(mod._HOLO_HUD_V2_PROCESS)
        self.assertIsNone(mod._ARC_STATUS_PROCESS)
        self.assertIsNone(mod._STARK_STATUS_PROCESS)

    def test_register_starts_watchers_when_enabled(self):
        # The three watcher-starters should fire during register() on their
        # default-True flags. With Thread.start no-op'd by the harness, the
        # *_STARTED flags still flip. (setUp's load uses a fake bobert with the
        # watcher flags absent → defaults True.)
        self.assertTrue(self.mod._WATCHER_STARTED)
        self.assertTrue(self.mod._BAMBU_WATCHER_STARTED)
        self.assertTrue(self.mod._WORKSHOP_PRINT_MONITOR_WATCHER_STARTED)


# ═══════════════════════════════════════════════════════════════════════════
#  _get_monitor_rect — bobert MONITORS resolution + fallbacks
# ═══════════════════════════════════════════════════════════════════════════
class MonitorRectTests(_HoloBase):
    def test_fallback_when_import_fails(self):
        with inject_modules(bobert_companion=None):
            self.assertEqual(self.mod._get_monitor_rect(), (0, 0, 2560, 1440))

    def test_reads_bobert_top_monitor(self):
        bc = mock.MagicMock()
        bc.MONITORS = {"top": (100, 200, 1920, 1080)}
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._get_monitor_rect(),
                             (100, 200, 1920, 1080))

    def test_fallback_when_no_top_key(self):
        bc = mock.MagicMock()
        bc.MONITORS = {"bottom": (0, 0, 1, 1)}
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._get_monitor_rect(), (0, 0, 2560, 1440))

    def test_fallback_when_monitors_attr_missing(self):
        bc = types.ModuleType("bobert_companion")  # no MONITORS attr at all
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._get_monitor_rect(), (0, 0, 2560, 1440))

    def test_fallback_when_rect_too_short(self):
        bc = mock.MagicMock()
        bc.MONITORS = {"top": (0, 0)}            # < 4 elements
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._get_monitor_rect(), (0, 0, 2560, 1440))

    def test_coerces_float_rect_to_int(self):
        bc = mock.MagicMock()
        bc.MONITORS = {"top": (10.7, 20.2, 1920.9, 1080.5)}
        with inject_modules(bobert_companion=bc):
            self.assertEqual(self.mod._get_monitor_rect(), (10, 20, 1920, 1080))


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry math for every surface
# ═══════════════════════════════════════════════════════════════════════════
class GeometryTests(_HoloBase):
    @contextlib.contextmanager
    def _rect(self, rect):
        with mock.patch.object(self.mod, "_get_monitor_rect",
                               return_value=rect):
            yield

    # ── workshop canvas: bottom-right, quarter-size clamp ─────────────────
    def test_workshop_geometry_anchors_bottom_right(self):
        with self._rect((0, 0, 2560, 1440)):
            x, y, w, h = self.mod._resolve_workshop_geometry()
        self.assertLessEqual(x + w, 2560)
        self.assertLessEqual(y + h, 1440)
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)
        # default 320 clamp: min(320, max(160, mw//4)) == 320 for a wide mon.
        self.assertEqual(w, self.mod._WORKSHOP_W)
        self.assertEqual(h, self.mod._WORKSHOP_H)

    def test_workshop_geometry_small_monitor_clamps_to_quarter(self):
        # mw//4 == 100 < 160 floor → w = 160; same for h.
        with self._rect((0, 0, 400, 400)):
            _x, _y, w, h = self.mod._resolve_workshop_geometry()
        self.assertEqual(w, 160)
        self.assertEqual(h, 160)

    def test_workshop_geometry_offset_origin(self):
        with self._rect((100, 50, 2560, 1440)):
            x, y, w, h = self.mod._resolve_workshop_geometry()
        self.assertEqual(x, 100 + 2560 - w - self.mod._WORKSHOP_MARGIN)
        self.assertEqual(y, 50 + 1440 - h - self.mod._WORKSHOP_MARGIN)

    # ── bambu overlay: top-right, slides down under workshop HUD ───────────
    def test_bambu_geometry_top_right(self):
        with self._rect((0, 0, 2560, 1440)), \
                mock.patch.object(self.mod, "_workshop_hud_is_alive",
                                  return_value=False):
            x, y, w, h = self.mod._resolve_bambu_overlay_geometry()
        self.assertEqual(w, self.mod._BAMBU_OVERLAY_W)
        self.assertEqual(h, self.mod._BAMBU_OVERLAY_H)
        self.assertEqual(x, 2560 - w - self.mod._BAMBU_OVERLAY_MARGIN)
        self.assertEqual(y, self.mod._BAMBU_OVERLAY_MARGIN)

    def test_bambu_geometry_stacks_below_workshop_hud(self):
        with self._rect((0, 0, 2560, 1440)), \
                mock.patch.object(self.mod, "_workshop_hud_is_alive",
                                  return_value=True):
            _x, y, _w, _h = self.mod._resolve_bambu_overlay_geometry()
        self.assertEqual(
            y, self.mod._BAMBU_OVERLAY_MARGIN + self.mod._WORKSHOP_HUD_H + 12)

    # ── workshop HUD: top-right ───────────────────────────────────────────
    def test_workshop_hud_geometry_top_right(self):
        with self._rect((0, 0, 2000, 1200)):
            x, y, w, h = self.mod._resolve_workshop_hud_geometry()
        self.assertEqual(w, self.mod._WORKSHOP_HUD_W)
        self.assertEqual(h, self.mod._WORKSHOP_HUD_H)
        self.assertEqual(x, 2000 - w - self.mod._WORKSHOP_HUD_MARGIN)
        self.assertEqual(y, self.mod._WORKSHOP_HUD_MARGIN)

    # ── workshop print monitor: top-center ────────────────────────────────
    def test_print_monitor_geometry_centers_horizontally(self):
        with self._rect((0, 0, 2000, 1000)):
            x, y, w, _h = self.mod._resolve_workshop_print_monitor_geometry()
        self.assertEqual(x, (2000 - w) // 2)
        self.assertEqual(y, self.mod._WORKSHOP_PRINT_MONITOR_MARGIN)

    def test_print_monitor_geometry_offset_origin(self):
        with self._rect((300, 40, 2000, 1000)):
            x, y, w, _h = self.mod._resolve_workshop_print_monitor_geometry()
        self.assertEqual(x, 300 + (2000 - w) // 2)
        self.assertEqual(y, 40 + self.mod._WORKSHOP_PRINT_MONITOR_MARGIN)

    # ── holo HUD v2: centered, clamped to 45%/60% of monitor ──────────────
    def test_holo_hud_v2_geometry_centered_clamped_wide(self):
        with self._rect((0, 0, 2560, 1440)):
            x, y, w, h = self.mod._resolve_holo_hud_v2_geometry()
        # min(640, max(400, int(2560*0.45)=1152)) → 640 default.
        self.assertEqual(w, self.mod._HOLO_HUD_V2_W)
        self.assertEqual(h, self.mod._HOLO_HUD_V2_H)
        self.assertEqual(x, (2560 - w) // 2)
        self.assertEqual(y, self.mod._HOLO_HUD_V2_MARGIN)

    def test_holo_hud_v2_geometry_narrow_uses_floor(self):
        # mw*0.45 == 180 < 400 floor → w = 400; mh*0.6 == 240 < 420 → h = 420.
        with self._rect((0, 0, 400, 400)):
            _x, _y, w, h = self.mod._resolve_holo_hud_v2_geometry()
        self.assertEqual(w, 400)
        self.assertEqual(h, 420)

    # ── arc status HUD: top-left ──────────────────────────────────────────
    def test_arc_status_geometry_top_left(self):
        with self._rect((0, 0, 2560, 1440)):
            x, y, w, h = self.mod._resolve_arc_status_geometry()
        self.assertEqual(w, self.mod._ARC_STATUS_W)
        self.assertEqual(h, self.mod._ARC_STATUS_H)
        self.assertEqual(x, self.mod._ARC_STATUS_MARGIN)
        self.assertEqual(y, self.mod._ARC_STATUS_MARGIN)

    def test_arc_status_geometry_offset_origin(self):
        with self._rect((50, 60, 2560, 1440)):
            x, y, _w, _h = self.mod._resolve_arc_status_geometry()
        self.assertEqual(x, 50 + self.mod._ARC_STATUS_MARGIN)
        self.assertEqual(y, 60 + self.mod._ARC_STATUS_MARGIN)

    # ── stark status ring: top-center, stacks below print monitor ─────────
    def test_stark_status_geometry_top_center(self):
        with self._rect((0, 0, 2560, 1440)), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=False):
            x, y, w, h = self.mod._resolve_stark_status_geometry()
        self.assertEqual(w, self.mod._STARK_STATUS_W)
        self.assertEqual(h, self.mod._STARK_STATUS_H)
        self.assertEqual(x, (2560 - w) // 2)
        self.assertEqual(y, self.mod._STARK_STATUS_MARGIN)

    def test_stark_status_geometry_stacks_below_print_monitor(self):
        with self._rect((0, 0, 2560, 1440)), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=True):
            _x, y, _w, _h = self.mod._resolve_stark_status_geometry()
        self.assertEqual(
            y, self.mod._STARK_STATUS_MARGIN
            + self.mod._WORKSHOP_PRINT_MONITOR_H + 12)


# ═══════════════════════════════════════════════════════════════════════════
#  *_is_alive state machine (shared shape across all eight surfaces)
# ═══════════════════════════════════════════════════════════════════════════
class IsAliveTests(_HoloBase):
    # (alive-fn name, process-global name)
    SURFACES = (
        ("_overlay_is_alive", "_OVERLAY_PROCESS"),
        ("_workshop_is_alive", "_WORKSHOP_PROCESS"),
        ("_bambu_overlay_is_alive", "_BAMBU_OVERLAY_PROCESS"),
        ("_workshop_hud_is_alive", "_WORKSHOP_HUD_PROCESS"),
        ("_workshop_print_monitor_is_alive", "_WORKSHOP_PRINT_MONITOR_PROCESS"),
        ("_holo_hud_v2_is_alive", "_HOLO_HUD_V2_PROCESS"),
        ("_arc_status_is_alive", "_ARC_STATUS_PROCESS"),
        ("_stark_status_is_alive", "_STARK_STATUS_PROCESS"),
    )

    def test_none_process_is_not_alive(self):
        for fn_name, proc_attr in self.SURFACES:
            setattr(self.mod, proc_attr, None)
            self.assertFalse(getattr(self.mod, fn_name)(),
                             f"{fn_name} should be dead when proc is None")

    def test_running_process_is_alive(self):
        for fn_name, proc_attr in self.SURFACES:
            setattr(self.mod, proc_attr, _fake_proc(alive=True))
            self.assertTrue(getattr(self.mod, fn_name)(),
                            f"{fn_name} should be alive when poll() is None")

    def test_exited_process_is_not_alive(self):
        for fn_name, proc_attr in self.SURFACES:
            setattr(self.mod, proc_attr, _fake_proc(alive=False))
            self.assertFalse(getattr(self.mod, fn_name)(),
                             f"{fn_name} should be dead when poll() returns rc")


# ═══════════════════════════════════════════════════════════════════════════
#  Atomic control-file writers (merge-update + os.replace + failure swallow)
# ═══════════════════════════════════════════════════════════════════════════
class ControlFileWriterTests(_HoloBase):
    # (writer fn name, control-file attr)
    WRITERS = (
        ("_write_workshop_state", "_WORKSHOP_STATE_FILE"),
        ("_write_workshop_hud_control", "_WORKSHOP_HUD_CONTROL_FILE"),
        ("_write_workshop_print_monitor_control",
         "_WORKSHOP_PRINT_MONITOR_CONTROL_FILE"),
        ("_write_arc_status_control", "_ARC_STATUS_CONTROL_FILE"),
        ("_write_stark_status_control", "_STARK_STATUS_CONTROL_FILE"),
    )

    def test_writer_creates_file_with_payload(self):
        for fn_name, path_attr in self.WRITERS:
            path = getattr(self.mod, path_attr)
            getattr(self.mod, fn_name)(mode="on")
            self.assertEqual(self._read_json(path).get("mode"), "on")

    def test_writer_merges_into_existing(self):
        for fn_name, path_attr in self.WRITERS:
            path = getattr(self.mod, path_attr)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"keep": 1, "mode": "off"}, f)
            getattr(self.mod, fn_name)(mode="on", extra="z")
            data = self._read_json(path)
            self.assertEqual(data["mode"], "on")     # overwritten
            self.assertEqual(data["keep"], 1)        # preserved
            self.assertEqual(data["extra"], "z")     # added

    def test_writer_tolerates_corrupt_existing(self):
        for fn_name, path_attr in self.WRITERS:
            path = getattr(self.mod, path_attr)
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            # Should not raise; corrupt content discarded, new payload written.
            getattr(self.mod, fn_name)(mode="on")
            self.assertEqual(self._read_json(path).get("mode"), "on")

    def test_writer_swallows_io_errors(self):
        # os.replace raising must not propagate (best-effort writer contract).
        for fn_name, _path_attr in self.WRITERS:
            with mock.patch.object(self.mod.os, "replace",
                                   side_effect=OSError("disk full")):
                getattr(self.mod, fn_name)(mode="on")  # no exception

    def test_workshop_state_force_visible_field(self):
        self.mod._write_workshop_state(mode="pulse", force_visible=False)
        data = self._read_json(self.mod._WORKSHOP_STATE_FILE)
        self.assertEqual(data["mode"], "pulse")
        self.assertFalse(data["force_visible"])


# ═══════════════════════════════════════════════════════════════════════════
#  Launch / shutdown lifecycle — generic across the eight surfaces.
#
#  Each surface exposes _launch_X()/_shutdown_X() returning (ok, msg) and is
#  guarded by a missing-script check, an idempotent already-alive check, a
#  Popen-failure reset, and a terminate→kill→wait shutdown. We drive every one
#  of those paths.
# ═══════════════════════════════════════════════════════════════════════════
class LaunchShutdownTests(_HoloBase):
    # (launch, shutdown, alive, script-attr, process-attr, *args-for-launch)
    SURFACES = (
        ("_launch_overlay", "_shutdown_overlay", "_overlay_is_alive",
         "_OVERLAY_SCRIPT", "_OVERLAY_PROCESS", ()),
        ("_launch_workshop", "_shutdown_workshop", "_workshop_is_alive",
         "_WORKSHOP_SCRIPT", "_WORKSHOP_PROCESS", ("on",)),
        ("_launch_bambu_overlay", "_shutdown_bambu_overlay",
         "_bambu_overlay_is_alive", "_BAMBU_OVERLAY_SCRIPT",
         "_BAMBU_OVERLAY_PROCESS", ()),
        ("_launch_workshop_hud", "_shutdown_workshop_hud",
         "_workshop_hud_is_alive", "_WORKSHOP_HUD_SCRIPT",
         "_WORKSHOP_HUD_PROCESS", ()),
        ("_launch_workshop_print_monitor", "_shutdown_workshop_print_monitor",
         "_workshop_print_monitor_is_alive", "_WORKSHOP_PRINT_MONITOR_SCRIPT",
         "_WORKSHOP_PRINT_MONITOR_PROCESS", ()),
        ("_launch_holo_hud_v2", "_shutdown_holo_hud_v2",
         "_holo_hud_v2_is_alive", "_HOLO_HUD_V2_SCRIPT",
         "_HOLO_HUD_V2_PROCESS", ()),
        ("_launch_arc_status", "_shutdown_arc_status", "_arc_status_is_alive",
         "_ARC_STATUS_SCRIPT", "_ARC_STATUS_PROCESS", ()),
        ("_launch_stark_status", "_shutdown_stark_status",
         "_stark_status_is_alive", "_STARK_STATUS_SCRIPT",
         "_STARK_STATUS_PROCESS", ()),
    )

    def test_launch_missing_script_refuses(self):
        for launch, _sd, _alive, script_attr, proc_attr, args in self.SURFACES:
            setattr(self.mod, proc_attr, None)
            # Script path points at a non-existent temp file (never touched).
            ok, msg = getattr(self.mod, launch)(*args)
            self.assertFalse(ok, f"{launch} should fail on missing script")
            self.assertIn("missing", msg.lower())
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_launch_success_spawns_and_stores_process(self):
        for launch, _sd, _alive, script_attr, proc_attr, args in self.SURFACES:
            setattr(self.mod, proc_attr, None)
            self._touch(getattr(self.mod, script_attr))
            proc = _fake_proc(alive=True)
            with mock.patch.object(subprocess, "Popen", return_value=proc) as p:
                ok, msg = getattr(self.mod, launch)(*args)
            self.assertTrue(ok, f"{launch} should succeed: {msg}")
            self.assertIs(getattr(self.mod, proc_attr), proc)
            # Spawned with the project's python + the script path + parent-pid.
            argv = p.call_args.args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1], getattr(self.mod, script_attr))
            self.assertIn("--parent-pid", argv)

    def test_launch_idempotent_when_already_alive(self):
        # Re-launching a live surface must NOT spawn a second process. Most
        # surfaces say "already engaged"; the workshop canvas instead reports
        # the live mode ("engaged"/"pulsing") — both are success with no
        # respawn, which is the contract we assert.
        for launch, _sd, _alive, script_attr, proc_attr, args in self.SURFACES:
            setattr(self.mod, proc_attr, _fake_proc(alive=True))
            self._touch(getattr(self.mod, script_attr))
            with mock.patch.object(subprocess, "Popen") as p:
                ok, msg = getattr(self.mod, launch)(*args)
            self.assertTrue(ok, f"{launch} idempotent call should succeed")
            self.assertTrue(msg, f"{launch} should return a message")
            p.assert_not_called()      # no second spawn

    def test_launch_popen_failure_resets_process(self):
        for launch, _sd, _alive, script_attr, proc_attr, args in self.SURFACES:
            setattr(self.mod, proc_attr, None)
            self._touch(getattr(self.mod, script_attr))
            with mock.patch.object(subprocess, "Popen",
                                   side_effect=OSError("denied")):
                ok, msg = getattr(self.mod, launch)(*args)
            self.assertFalse(ok, f"{launch} should report failure")
            self.assertIn("failed", msg.lower())
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_shutdown_when_not_running_is_noop_ok(self):
        for _launch, sd, _alive, _script, proc_attr, _args in self.SURFACES:
            setattr(self.mod, proc_attr, None)
            ok, msg = getattr(self.mod, sd)()
            self.assertTrue(ok)
            self.assertIn("isn't currently engaged", msg)
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_shutdown_terminates_live_process(self):
        # All surfaces terminate()+wait() the live process and clear the global;
        # the user-facing verb varies ("dismissed" for most, "disengaged" for
        # the workshop canvas), so we assert the lifecycle, not the wording.
        for _launch, sd, _alive, _script, proc_attr, _args in self.SURFACES:
            proc = _fake_proc(alive=True)
            setattr(self.mod, proc_attr, proc)
            ok, msg = getattr(self.mod, sd)()
            self.assertTrue(ok)
            self.assertTrue(msg)
            proc.terminate.assert_called_once()
            proc.wait.assert_called()                 # waited for handle release
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_shutdown_escalates_to_kill_when_wait_times_out(self):
        for _launch, sd, _alive, _script, proc_attr, _args in self.SURFACES:
            proc = _fake_proc(alive=True)
            # First wait (after terminate) raises → escalate to kill, wait again
            proc.wait.side_effect = [subprocess.TimeoutExpired("x", 2.0), None]
            setattr(self.mod, proc_attr, proc)
            ok, _msg = getattr(self.mod, sd)()
            self.assertTrue(ok)
            proc.terminate.assert_called_once()
            proc.kill.assert_called_once()
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_shutdown_swallows_terminate_error(self):
        for _launch, sd, _alive, _script, proc_attr, _args in self.SURFACES:
            proc = _fake_proc(alive=True)
            proc.terminate.side_effect = OSError("already gone")
            setattr(self.mod, proc_attr, proc)
            ok, _msg = getattr(self.mod, sd)()   # must not raise
            self.assertTrue(ok)
            self.assertIsNone(getattr(self.mod, proc_attr))

    def test_shutdown_swallows_kill_and_post_kill_wait_errors(self):
        # Exercise the deepest escalation: terminate's wait times out → kill()
        # itself raises (swallowed) → the post-kill wait ALSO raises (swallowed).
        # The surface must still report success and clear its process global.
        for _launch, sd, _alive, _script, proc_attr, _args in self.SURFACES:
            proc = _fake_proc(alive=True)
            proc.wait.side_effect = [subprocess.TimeoutExpired("x", 2.0),
                                     OSError("handle vanished")]
            proc.kill.side_effect = OSError("kill raced exit")
            setattr(self.mod, proc_attr, proc)
            ok, _msg = getattr(self.mod, sd)()   # must not raise
            self.assertTrue(ok)
            proc.terminate.assert_called_once()
            proc.kill.assert_called_once()
            self.assertIsNone(getattr(self.mod, proc_attr))


# ═══════════════════════════════════════════════════════════════════════════
#  Launch side-effects unique to specific surfaces
# ═══════════════════════════════════════════════════════════════════════════
class LaunchSideEffectTests(_HoloBase):
    def test_workshop_launch_writes_reset_state_file(self):
        self._touch(self.mod._WORKSHOP_SCRIPT)
        self.mod._WORKSHOP_PROCESS = None
        with mock.patch.object(subprocess, "Popen",
                               return_value=_fake_proc()):
            ok, _msg = self.mod._launch_workshop("on")
        self.assertTrue(ok)
        data = self._read_json(self.mod._WORKSHOP_STATE_FILE)
        self.assertEqual(data["mode"], "on")
        self.assertFalse(data["force_visible"])

    def test_workshop_pulse_mode_message_and_argv(self):
        self._touch(self.mod._WORKSHOP_SCRIPT)
        self.mod._WORKSHOP_PROCESS = None
        with mock.patch.object(subprocess, "Popen",
                               return_value=_fake_proc()) as p:
            ok, msg = self.mod._launch_workshop("pulse")
        self.assertTrue(ok)
        self.assertIn("pulsing", msg.lower())
        self.assertIn("pulse", p.call_args.args[0])      # --mode pulse in argv

    def test_workshop_invalid_mode_coerced_to_on(self):
        self._touch(self.mod._WORKSHOP_SCRIPT)
        self.mod._WORKSHOP_PROCESS = None
        with mock.patch.object(subprocess, "Popen",
                               return_value=_fake_proc()) as p:
            ok, msg = self.mod._launch_workshop("nonsense")
        self.assertTrue(ok)
        argv = p.call_args.args[0]
        self.assertEqual(argv[argv.index("--mode") + 1], "on")
        self.assertNotIn("pulsing", msg.lower())

    def test_workshop_alive_updates_mode_without_respawn(self):
        self.mod._WORKSHOP_PROCESS = _fake_proc(alive=True)
        with mock.patch.object(subprocess, "Popen") as p:
            ok, msg = self.mod._launch_workshop("pulse")
        self.assertTrue(ok)
        self.assertIn("pulsing", msg.lower())
        p.assert_not_called()
        # Live state file updated so the running canvas can switch presentation.
        self.assertEqual(
            self._read_json(self.mod._WORKSHOP_STATE_FILE)["mode"], "pulse")

    def test_shutdown_workshop_writes_off_state(self):
        self.mod._WORKSHOP_PROCESS = None
        ok, _msg = self.mod._shutdown_workshop()
        self.assertTrue(ok)
        # Even with nothing running, the "off" intent is recorded for any
        # canvas that missed a tick.
        self.assertEqual(
            self._read_json(self.mod._WORKSHOP_STATE_FILE)["mode"], "off")

    def test_workshop_hud_alive_rewrites_on_control(self):
        self.mod._WORKSHOP_HUD_PROCESS = _fake_proc(alive=True)
        with mock.patch.object(subprocess, "Popen") as p:
            ok, msg = self.mod._launch_workshop_hud()
        self.assertTrue(ok)
        self.assertIn("already", msg.lower())
        p.assert_not_called()
        self.assertEqual(
            self._read_json(self.mod._WORKSHOP_HUD_CONTROL_FILE)["mode"], "on")

    def test_arc_status_alive_rewrites_on_control(self):
        self.mod._ARC_STATUS_PROCESS = _fake_proc(alive=True)
        with mock.patch.object(subprocess, "Popen") as p:
            ok, msg = self.mod._launch_arc_status()
        self.assertTrue(ok)
        self.assertIn("already", msg.lower())
        p.assert_not_called()

    def test_stark_status_alive_rewrites_on_control(self):
        self.mod._STARK_STATUS_PROCESS = _fake_proc(alive=True)
        with mock.patch.object(subprocess, "Popen") as p:
            ok, msg = self.mod._launch_stark_status()
        self.assertTrue(ok)
        self.assertIn("already", msg.lower())
        p.assert_not_called()

    def test_shutdown_hud_writes_off_control(self):
        # Each control-file-backed surface records mode=off on shutdown.
        cases = (
            ("_shutdown_workshop_hud", "_WORKSHOP_HUD_PROCESS",
             "_WORKSHOP_HUD_CONTROL_FILE"),
            ("_shutdown_workshop_print_monitor",
             "_WORKSHOP_PRINT_MONITOR_PROCESS",
             "_WORKSHOP_PRINT_MONITOR_CONTROL_FILE"),
            ("_shutdown_arc_status", "_ARC_STATUS_PROCESS",
             "_ARC_STATUS_CONTROL_FILE"),
            ("_shutdown_stark_status", "_STARK_STATUS_PROCESS",
             "_STARK_STATUS_CONTROL_FILE"),
        )
        for sd, proc_attr, ctrl_attr in cases:
            setattr(self.mod, proc_attr, None)
            getattr(self.mod, sd)()
            self.assertEqual(
                self._read_json(getattr(self.mod, ctrl_attr))["mode"], "off")


# ═══════════════════════════════════════════════════════════════════════════
#  Registered actions — dispatch, REFUSED prefixing, toggle, status strings
# ═══════════════════════════════════════════════════════════════════════════
class ActionDispatchTests(_HoloBase):
    # ── arc_reactor dispatcher ────────────────────────────────────────────
    def test_arc_reactor_on_synonyms_launch(self):
        for arg in ("", "on", "engage", "show", "start"):
            with mock.patch.object(self.mod, "_launch_workshop",
                                   return_value=(True, "online")) as launch:
                self.actions["arc_reactor"](arg)
            launch.assert_called_once_with("on")

    def test_arc_reactor_off_synonyms_shutdown(self):
        for arg in ("off", "disengage", "hide", "stop", "dismiss"):
            with mock.patch.object(self.mod, "_shutdown_workshop",
                                   return_value=(True, "disengaged")) as sd:
                self.actions["arc_reactor"](arg)
            sd.assert_called_once()

    def test_arc_reactor_pulse_synonyms(self):
        for arg in ("pulse", "pulsing", "throb"):
            with mock.patch.object(self.mod, "_launch_workshop",
                                   return_value=(True, "pulsing")) as launch:
                self.actions["arc_reactor"](arg)
            launch.assert_called_once_with("pulse")

    def test_arc_reactor_unknown_arg_defaults_on(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(True, "online")) as launch:
            self.actions["arc_reactor"]("flibbertigibbet")
        launch.assert_called_once_with("on")

    def test_arc_reactor_refused_on_failure(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(False, "boom")):
            out = self.actions["arc_reactor"]("on")
        self.assertTrue(out.startswith("REFUSED:"))

    # ── direct arc_reactor aliases ────────────────────────────────────────
    def test_arc_reactor_direct_aliases(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(True, "x")) as launch:
            self.actions["arc_reactor_on"]("")
            launch.assert_called_once_with("on")
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(True, "x")) as launch:
            self.actions["arc_reactor_pulse"]("")
            launch.assert_called_once_with("pulse")
        with mock.patch.object(self.mod, "_shutdown_workshop",
                               return_value=(True, "x")) as sd:
            self.actions["arc_reactor_off"]("")
            sd.assert_called_once()

    def test_arc_reactor_on_alias_refuses_on_failure(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(False, "script missing")):
            out = self.actions["arc_reactor_on"]("")
        self.assertTrue(out.startswith("REFUSED:"))

    # ── fullscreen overlay actions ────────────────────────────────────────
    def test_overlay_show_hide_toggle(self):
        with mock.patch.object(self.mod, "_launch_overlay",
                               return_value=(True, "online")) as launch:
            self.assertIn("online", self.actions["show_holographic_overlay"](""))
            launch.assert_called_once()
        with mock.patch.object(self.mod, "_shutdown_overlay",
                               return_value=(True, "dismissed")) as sd:
            self.assertIn("dismissed", self.actions["hide_holographic_overlay"](""))
            sd.assert_called_once()

    def test_overlay_toggle_routes_by_alive(self):
        with mock.patch.object(self.mod, "_overlay_is_alive", return_value=True), \
                mock.patch.object(self.mod, "_shutdown_overlay",
                                  return_value=(True, "off")) as sd:
            self.actions["toggle_holographic_overlay"]("")
        sd.assert_called_once()
        with mock.patch.object(self.mod, "_overlay_is_alive", return_value=False), \
                mock.patch.object(self.mod, "_launch_overlay",
                                  return_value=(True, "on")) as launch:
            self.actions["toggle_holographic_overlay"]("")
        launch.assert_called_once()

    def test_overlay_status_alive_reports_geometry(self):
        with mock.patch.object(self.mod, "_overlay_is_alive", return_value=True), \
                mock.patch.object(self.mod, "_get_monitor_rect",
                                  return_value=(0, 0, 1920, 1080)):
            out = self.actions["holographic_status"]("")
        self.assertIn("1920x1080", out)
        self.assertIn("engaged", out)

    def test_overlay_status_dormant(self):
        with mock.patch.object(self.mod, "_overlay_is_alive", return_value=False):
            out = self.actions["holographic_status"]("")
        self.assertIn("not currently engaged", out)

    # ── toggle routing for the remaining surfaces ─────────────────────────
    def test_all_toggles_route_by_alive_state(self):
        # (toggle action, alive-fn, on-launch fn, off-shutdown fn)
        cases = (
            ("workshop_hud_toggle", "_workshop_hud_is_alive",
             "_launch_workshop_hud", "_shutdown_workshop_hud"),
            ("workshop_print_monitor_toggle", "_workshop_print_monitor_is_alive",
             "_launch_workshop_print_monitor", "_shutdown_workshop_print_monitor"),
            ("holo_hud_v2_toggle", "_holo_hud_v2_is_alive",
             "_launch_holo_hud_v2", "_shutdown_holo_hud_v2"),
            ("arc_reactor_status_toggle", "_arc_status_is_alive",
             "_launch_arc_status", "_shutdown_arc_status"),
            ("stark_status_ring_toggle", "_stark_status_is_alive",
             "_launch_stark_status", "_shutdown_stark_status"),
        )
        for action, alive_fn, launch_fn, sd_fn in cases:
            # dormant → launch
            with mock.patch.object(self.mod, alive_fn, return_value=False), \
                    mock.patch.object(self.mod, launch_fn,
                                      return_value=(True, "on")) as launch:
                self.actions[action]("")
            launch.assert_called_once()
            # alive → shutdown
            with mock.patch.object(self.mod, alive_fn, return_value=True), \
                    mock.patch.object(self.mod, sd_fn,
                                      return_value=(True, "off")) as sd:
                self.actions[action]("")
            sd.assert_called_once()

    # ── status strings for each surface (alive + dormant) ─────────────────
    def test_status_strings_when_alive(self):
        # (status action, alive-fn, substring)
        cases = (
            ("workshop_hud_status", "_workshop_hud_is_alive", "workshop HUD"),
            ("holo_hud_v2_status", "_holo_hud_v2_is_alive", "HUD v2"),
            ("arc_reactor_status_status", "_arc_status_is_alive",
             "arc reactor status HUD"),
            ("stark_status_ring_status", "_stark_status_is_alive",
             "Stark status ring"),
        )
        for action, alive_fn, sub in cases:
            with mock.patch.object(self.mod, alive_fn, return_value=True), \
                    mock.patch.object(self.mod, "_get_monitor_rect",
                                      return_value=(0, 0, 1600, 900)):
                out = self.actions[action]("")
            self.assertIn(sub, out)
            self.assertIn("engaged", out)

    def test_status_strings_when_dormant(self):
        cases = (
            ("workshop_hud_status", "_workshop_hud_is_alive"),
            ("holo_hud_v2_status", "_holo_hud_v2_is_alive"),
            ("arc_reactor_status_status", "_arc_status_is_alive"),
            ("stark_status_ring_status", "_stark_status_is_alive"),
        )
        for action, alive_fn in cases:
            with mock.patch.object(self.mod, alive_fn, return_value=False):
                out = self.actions[action]("")
            self.assertIn("not currently engaged", out)

    def test_on_off_aliases_refused_prefix(self):
        # A representative on-alias from each surface REFUSES on launcher fail.
        cases = (
            ("workshop_hud_on", "_launch_workshop_hud"),
            ("holo_hud_v2_on", "_launch_holo_hud_v2"),
            ("arc_reactor_status_on", "_launch_arc_status"),
            ("stark_status_ring_on", "_launch_stark_status"),
            ("show_status_hud", "_launch_arc_status"),
        )
        for action, launch_fn in cases:
            with mock.patch.object(self.mod, launch_fn,
                                   return_value=(False, "missing")):
                out = self.actions[action]("")
            self.assertTrue(out.startswith("REFUSED:"), action)


# ═══════════════════════════════════════════════════════════════════════════
#  Bambu overlay — USER_OFF latch, status, toggle, watcher decisions
# ═══════════════════════════════════════════════════════════════════════════
class BambuOverlayTests(_HoloBase):
    def test_bambu_is_active_states(self):
        for gs in ("RUNNING", "running", "PAUSE", "pause", "PREPARE"):
            self.assertTrue(self.mod._bambu_is_active({"gcode_state": gs}), gs)
        for gs in ("FINISH", "FAILED", "IDLE", ""):
            self.assertFalse(self.mod._bambu_is_active({"gcode_state": gs}), gs)
        self.assertFalse(self.mod._bambu_is_active({}))

    def test_read_bambu_overlay_state_missing_file(self):
        self.assertEqual(self.mod._read_bambu_overlay_state(), {})

    def test_read_bambu_overlay_state_parses(self):
        with open(self.mod._BAMBU_OVERLAY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"gcode_state": "RUNNING"}, f)
        self.assertEqual(
            self.mod._read_bambu_overlay_state(), {"gcode_state": "RUNNING"})

    def test_read_bambu_overlay_state_corrupt_returns_empty(self):
        with open(self.mod._BAMBU_OVERLAY_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{nope")
        self.assertEqual(self.mod._read_bambu_overlay_state(), {})

    def test_on_clears_user_off_and_launches(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = True
        with mock.patch.object(self.mod, "_launch_bambu_overlay",
                               return_value=(True, "engaged")):
            out = self.actions["bambu_overlay_on"]("")
        self.assertIn("engaged", out)
        self.assertFalse(self.mod._BAMBU_OVERLAY_USER_OFF)

    def test_off_sets_user_off_latch(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = False
        with mock.patch.object(self.mod, "_shutdown_bambu_overlay",
                               return_value=(True, "dismissed")):
            self.actions["bambu_overlay_off"]("")
        self.assertTrue(self.mod._BAMBU_OVERLAY_USER_OFF)

    def test_toggle_routes_by_alive(self):
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                               return_value=True), \
                mock.patch.object(self.mod, "_shutdown_bambu_overlay",
                                  return_value=(True, "off")) as sd:
            self.actions["bambu_overlay_toggle"]("")
        sd.assert_called_once()
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                               return_value=False), \
                mock.patch.object(self.mod, "_launch_bambu_overlay",
                                  return_value=(True, "on")) as launch:
            self.actions["bambu_overlay_toggle"]("")
        launch.assert_called_once()

    def test_status_engaged(self):
        with open(self.mod._BAMBU_OVERLAY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"gcode_state": "RUNNING"}, f)
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                               return_value=True):
            out = self.actions["bambu_overlay_status"]("")
        self.assertIn("engaged", out)
        self.assertIn("RUNNING", out)

    def test_status_manual_off(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = True
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                               return_value=False):
            out = self.actions["bambu_overlay_status"]("")
        self.assertIn("manual", out)

    def test_status_dormant(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = False
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                               return_value=False):
            out = self.actions["bambu_overlay_status"]("")
        self.assertIn("dormant", out)

    def test_clear_user_off_helper(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = True
        self.mod._clear_user_off()
        self.assertFalse(self.mod._BAMBU_OVERLAY_USER_OFF)


# ═══════════════════════════════════════════════════════════════════════════
#  Workshop print monitor — USER_OFF latch, status (4 branches), helpers
# ═══════════════════════════════════════════════════════════════════════════
class WorkshopPrintMonitorTests(_HoloBase):
    def test_on_clears_user_off(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = True
        with mock.patch.object(self.mod, "_launch_workshop_print_monitor",
                               return_value=(True, "engaged")):
            self.actions["workshop_print_monitor_on"]("")
        self.assertFalse(self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF)

    def test_off_sets_user_off(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = False
        with mock.patch.object(self.mod, "_shutdown_workshop_print_monitor",
                               return_value=(True, "dismissed")):
            self.actions["workshop_print_monitor_off"]("")
        self.assertTrue(self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF)

    def test_status_engaged_lists_context(self):
        with mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                               return_value=True), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=True), \
                mock.patch.object(self.mod, "_get_monitor_rect",
                                  return_value=(0, 0, 1600, 900)):
            out = self.actions["workshop_print_monitor_status"]("")
        self.assertIn("engaged", out)
        self.assertIn("printer active", out)
        self.assertIn("workshop mode", out)

    def test_status_engaged_no_context(self):
        with mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                               return_value=True), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=False), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=False), \
                mock.patch.object(self.mod, "_get_monitor_rect",
                                  return_value=(0, 0, 1600, 900)):
            out = self.actions["workshop_print_monitor_status"]("")
        self.assertIn("engaged", out)
        # No active context → none of the parenthetical context labels appear.
        self.assertNotIn("printer active", out)
        self.assertNotIn("workshop mode", out)

    def test_status_manual_off(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = True
        with mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                               return_value=False), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=False), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=False):
            out = self.actions["workshop_print_monitor_status"]("")
        self.assertIn("manual", out)

    def test_status_initialising_when_active_but_not_alive(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = False
        with mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                               return_value=False), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=False):
            out = self.actions["workshop_print_monitor_status"]("")
        self.assertIn("initialising", out)

    def test_status_dormant(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = False
        with mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                               return_value=False), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=False), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=False):
            out = self.actions["workshop_print_monitor_status"]("")
        self.assertIn("dormant", out)

    def test_clear_user_off_helper(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = True
        self.mod._clear_workshop_print_monitor_user_off()
        self.assertFalse(self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF)

    def test_bambu_print_active_for_monitor(self):
        with mock.patch.object(self.mod, "_read_bambu_overlay_state",
                               return_value={"gcode_state": "RUNNING"}):
            self.assertTrue(self.mod._bambu_print_is_active_for_monitor())
        with mock.patch.object(self.mod, "_read_bambu_overlay_state",
                               return_value={}):
            self.assertFalse(self.mod._bambu_print_is_active_for_monitor())
        with mock.patch.object(self.mod, "_read_bambu_overlay_state",
                               return_value={"gcode_state": "FINISH"}):
            self.assertFalse(self.mod._bambu_print_is_active_for_monitor())

    # ── _workshop_mode_is_active — reads a foreign skill module ───────────
    def test_workshop_mode_active_from_list_cell(self):
        fake = types.ModuleType("workshop_mode")
        fake._workshop_active = [True]
        with inject_modules(workshop_mode=fake):
            sys.modules.pop("skills.workshop_mode", None)
            self.assertTrue(self.mod._workshop_mode_is_active())

    def test_workshop_mode_inactive_from_list_cell(self):
        fake = types.ModuleType("workshop_mode")
        fake._workshop_active = [False]
        with inject_modules(workshop_mode=fake,
                            **{"skills.workshop_mode": None}):
            self.assertFalse(self.mod._workshop_mode_is_active())

    def test_workshop_mode_scalar_cell(self):
        fake = types.ModuleType("workshop_mode")
        fake._workshop_active = True            # not subscriptable
        with inject_modules(workshop_mode=fake,
                            **{"skills.workshop_mode": None}):
            self.assertTrue(self.mod._workshop_mode_is_active())

    def test_workshop_mode_missing_attr_is_false(self):
        fake = types.ModuleType("workshop_mode")  # no _workshop_active
        with inject_modules(workshop_mode=fake,
                            **{"skills.workshop_mode": None}):
            self.assertFalse(self.mod._workshop_mode_is_active())

    def test_workshop_mode_module_absent_is_false(self):
        # Neither skills.workshop_mode nor workshop_mode importable.
        with inject_modules(**{"skills.workshop_mode": None,
                               "workshop_mode": None}), \
                mock.patch.object(self.mod, "sys") as fake_sys:
            fake_sys.modules = {}
            # importlib.import_module will fail for both forms → False.
            self.assertFalse(self.mod._workshop_mode_is_active())

    def test_workshop_mode_outer_exception_is_false(self):
        # An unexpected error while reading the foreign module's flag (here a
        # property whose access raises) is caught by the outer guard → False,
        # so the watcher leans toward inaction rather than crashing.
        class _Boom(types.ModuleType):
            @property
            def _workshop_active(self):
                raise RuntimeError("exploded reading flag")
        fake = _Boom("workshop_mode")
        with inject_modules(workshop_mode=fake,
                            **{"skills.workshop_mode": None}):
            self.assertFalse(self.mod._workshop_mode_is_active())


# ═══════════════════════════════════════════════════════════════════════════
#  Watcher decision logic (one deterministic iteration each, no real thread)
# ═══════════════════════════════════════════════════════════════════════════
class WatcherTests(_HoloBase):
    @contextlib.contextmanager
    def _one_iteration(self, stop_attr):
        """Make the named stop-Event's ``wait`` return False once (run a single
        loop body) then True (exit). No real sleeping or threading."""
        ev = getattr(self.mod, stop_attr)
        with mock.patch.object(ev, "wait", side_effect=[False, True]):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _quiet():
        """Silence the ``logging.exception`` output the watchers emit when they
        swallow a loop-body error, so the deliberate error-path tests don't
        spew tracebacks into the test report. Restored on exit."""
        import logging
        root = logging.getLogger()
        prev = root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(prev)

    @contextlib.contextmanager
    def _n_iterations(self, stop_attr, n, times):
        """Run exactly ``n`` loop bodies (``wait`` → False n times then True),
        with ``time.time`` returning successive values from ``times`` so the
        elapsed-since-active math (grace / linger windows) is deterministic.
        ``times`` must supply one value per ``time.time()`` call across the n
        bodies (each body reads the clock once)."""
        ev = getattr(self.mod, stop_attr)
        with mock.patch.object(ev, "wait", side_effect=([False] * n) + [True]), \
                mock.patch.object(self.mod.time, "time", side_effect=list(times)):
            yield

    # ── _read_jarvis_state ────────────────────────────────────────────────
    def test_read_jarvis_state_missing_file_idle(self):
        self.assertEqual(self.mod._read_jarvis_state(), "idle")

    def test_read_jarvis_state_parses_lowercased(self):
        with open(self.mod._HUD_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"state": "THINKING"}, f)
        self.assertEqual(self.mod._read_jarvis_state(), "thinking")

    def test_read_jarvis_state_corrupt_idle(self):
        with open(self.mod._HUD_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad")
        self.assertEqual(self.mod._read_jarvis_state(), "idle")

    def test_read_jarvis_state_missing_key_idle(self):
        with open(self.mod._HUD_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"other": 1}, f)
        self.assertEqual(self.mod._read_jarvis_state(), "idle")

    # ── auto-show watcher ─────────────────────────────────────────────────
    def test_auto_watcher_launches_when_thinking(self):
        with self._one_iteration("_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  return_value="thinking"), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod, "_launch_workshop") as launch:
            self.mod._auto_show_watcher()
        launch.assert_called_once_with("on")

    def test_auto_watcher_skips_launch_when_already_alive(self):
        with self._one_iteration("_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  return_value="speaking"), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_workshop") as launch:
            self.mod._auto_show_watcher()
        launch.assert_not_called()

    def test_auto_watcher_idle_does_nothing_without_prior_active(self):
        # last_active_at starts at 0 → idle branch can't hide anything.
        with self._one_iteration("_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  return_value="idle"), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_shutdown_workshop") as sd:
            self.mod._auto_show_watcher()
        sd.assert_not_called()

    def test_auto_watcher_iteration_swallows_errors(self):
        # An exception inside the loop body is logged, not raised.
        with self._quiet(), self._one_iteration("_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  side_effect=RuntimeError("boom")):
            self.mod._auto_show_watcher()   # must return cleanly

    def test_auto_watcher_hides_after_grace_when_mode_on(self):
        # iter1 active (records last_active_at); iter2 idle past the grace
        # window with the live state file mode=="on" → auto-launched canvas is
        # retired via _shutdown_workshop.
        with open(self.mod._WORKSHOP_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"mode": "on"}, f)
        t0 = 1000.0
        with self._n_iterations(
                "_WATCHER_STOP", 2,
                [t0, t0 + self.mod._AUTO_HIDE_GRACE_S + 1]), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  side_effect=["thinking", "idle"]), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_workshop"), \
                mock.patch.object(self.mod, "_shutdown_workshop") as sd:
            self.mod._auto_show_watcher()
        sd.assert_called_once()

    def test_auto_watcher_hides_when_state_read_raises(self):
        # iter2 past grace, but reading the live state file to learn the mode
        # raises → the watcher's inner guard swallows it and keeps the default
        # mode "on", so the auto-launched canvas is still retired.
        with open(self.mod._WORKSHOP_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"mode": "on"}, f)
        t0 = 1000.0
        with self._n_iterations(
                "_WATCHER_STOP", 2,
                [t0, t0 + self.mod._AUTO_HIDE_GRACE_S + 1]), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  side_effect=["thinking", "idle"]), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod.json, "load",
                                  side_effect=ValueError("corrupt")), \
                mock.patch.object(self.mod, "_launch_workshop"), \
                mock.patch.object(self.mod, "_shutdown_workshop") as sd:
            self.mod._auto_show_watcher()
        sd.assert_called_once()

    def test_auto_watcher_keeps_canvas_when_mode_pulse(self):
        # Same timing, but the user set mode=="pulse" → the watcher leaves the
        # canvas alone (only auto-retires canvases it auto-launched in "on").
        with open(self.mod._WORKSHOP_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"mode": "pulse"}, f)
        t0 = 1000.0
        with self._n_iterations(
                "_WATCHER_STOP", 2,
                [t0, t0 + self.mod._AUTO_HIDE_GRACE_S + 1]), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  side_effect=["speaking", "idle"]), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_workshop"), \
                mock.patch.object(self.mod, "_shutdown_workshop") as sd:
            self.mod._auto_show_watcher()
        sd.assert_not_called()

    def test_auto_watcher_within_grace_does_not_hide(self):
        # iter2 idle but still INSIDE the grace window → no shutdown yet.
        t0 = 1000.0
        with self._n_iterations("_WATCHER_STOP", 2, [t0, t0 + 0.1]), \
                mock.patch.object(self.mod, "_read_jarvis_state",
                                  side_effect=["thinking", "idle"]), \
                mock.patch.object(self.mod, "_workshop_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_workshop"), \
                mock.patch.object(self.mod, "_shutdown_workshop") as sd:
            self.mod._auto_show_watcher()
        sd.assert_not_called()

    # ── bambu overlay watcher ─────────────────────────────────────────────
    def test_bambu_watcher_launches_on_active_print(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = False
        with self._one_iteration("_BAMBU_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_bambu_overlay_state",
                                  return_value={"gcode_state": "RUNNING"}), \
                mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod, "_launch_bambu_overlay") as launch:
            self.mod._bambu_overlay_watcher()
        launch.assert_called_once()

    def test_bambu_watcher_respects_user_off(self):
        self.mod._BAMBU_OVERLAY_USER_OFF = True
        with self._one_iteration("_BAMBU_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_bambu_overlay_state",
                                  return_value={"gcode_state": "RUNNING"}), \
                mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod, "_launch_bambu_overlay") as launch:
            self.mod._bambu_overlay_watcher()
        launch.assert_not_called()

    def test_bambu_watcher_iteration_swallows_errors(self):
        with self._quiet(), self._one_iteration("_BAMBU_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_read_bambu_overlay_state",
                                  side_effect=RuntimeError("boom")):
            self.mod._bambu_overlay_watcher()

    def test_bambu_watcher_retires_after_linger_and_rearms(self):
        # iter1 active RUNNING (records last_active_at, saw_active_once);
        # iter2 FINISH past the linger window → overlay torn down + USER_OFF
        # cleared so the next print re-engages auto mode.
        self.mod._BAMBU_OVERLAY_USER_OFF = True   # should be cleared on retire
        t0 = 2000.0
        with self._n_iterations(
                "_BAMBU_WATCHER_STOP", 2,
                [t0, t0 + self.mod._BAMBU_OVERLAY_LINGER_S + 1]), \
                mock.patch.object(self.mod, "_read_bambu_overlay_state",
                                  side_effect=[{"gcode_state": "RUNNING"},
                                               {"gcode_state": "FINISH"}]), \
                mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_bambu_overlay"), \
                mock.patch.object(self.mod, "_shutdown_bambu_overlay") as sd:
            self.mod._bambu_overlay_watcher()
        sd.assert_called_once()
        self.assertFalse(self.mod._BAMBU_OVERLAY_USER_OFF)   # auto-mode re-armed

    def test_bambu_watcher_within_linger_keeps_overlay(self):
        t0 = 2000.0
        with self._n_iterations("_BAMBU_WATCHER_STOP", 2, [t0, t0 + 1.0]), \
                mock.patch.object(self.mod, "_read_bambu_overlay_state",
                                  side_effect=[{"gcode_state": "RUNNING"},
                                               {"gcode_state": "FINISH"}]), \
                mock.patch.object(self.mod, "_bambu_overlay_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_bambu_overlay"), \
                mock.patch.object(self.mod, "_shutdown_bambu_overlay") as sd:
            self.mod._bambu_overlay_watcher()
        sd.assert_not_called()

    # ── workshop print monitor watcher ────────────────────────────────────
    def test_print_monitor_watcher_launches_when_printer_active(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = False
        with self._one_iteration("_WORKSHOP_PRINT_MONITOR_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=False), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod,
                                  "_launch_workshop_print_monitor") as launch:
            self.mod._workshop_print_monitor_watcher()
        launch.assert_called_once()

    def test_print_monitor_watcher_launches_when_workshop_active(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = False
        with self._one_iteration("_WORKSHOP_PRINT_MONITOR_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=False), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod,
                                  "_launch_workshop_print_monitor") as launch:
            self.mod._workshop_print_monitor_watcher()
        launch.assert_called_once()

    def test_print_monitor_watcher_respects_user_off(self):
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = True
        with self._one_iteration("_WORKSHOP_PRINT_MONITOR_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  return_value=True), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=False), \
                mock.patch.object(self.mod,
                                  "_launch_workshop_print_monitor") as launch:
            self.mod._workshop_print_monitor_watcher()
        launch.assert_not_called()

    def test_print_monitor_watcher_iteration_swallows_errors(self):
        with self._quiet(), \
                self._one_iteration("_WORKSHOP_PRINT_MONITOR_WATCHER_STOP"), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  side_effect=RuntimeError("boom")):
            self.mod._workshop_print_monitor_watcher()

    def test_print_monitor_watcher_retires_after_linger(self):
        # iter1 active; iter2 idle past the linger window → widget retired and
        # auto-show re-armed for the next session.
        self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF = True
        t0 = 3000.0
        stop = "_WORKSHOP_PRINT_MONITOR_WATCHER_STOP"
        with self._n_iterations(
                stop, 2,
                [t0, t0 + self.mod._WORKSHOP_PRINT_MONITOR_LINGER_S + 1]), \
                mock.patch.object(self.mod, "_bambu_print_is_active_for_monitor",
                                  side_effect=[True, False]), \
                mock.patch.object(self.mod, "_workshop_mode_is_active",
                                  side_effect=[False, False]), \
                mock.patch.object(self.mod, "_workshop_print_monitor_is_alive",
                                  return_value=True), \
                mock.patch.object(self.mod, "_launch_workshop_print_monitor"), \
                mock.patch.object(self.mod,
                                  "_shutdown_workshop_print_monitor") as sd:
            self.mod._workshop_print_monitor_watcher()
        sd.assert_called_once()
        self.assertFalse(self.mod._WORKSHOP_PRINT_MONITOR_USER_OFF)


# ═══════════════════════════════════════════════════════════════════════════
#  Watcher-starter guards (single-shot + config flag)
# ═══════════════════════════════════════════════════════════════════════════
class WatcherStarterTests(_HoloBase):
    def test_auto_watcher_starts_once(self):
        self.mod._WATCHER_STARTED = False
        with mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_auto_watcher()
            self.mod._maybe_start_auto_watcher()   # second call is a no-op
        self.assertTrue(self.mod._WATCHER_STARTED)
        Thread.assert_called_once()

    def test_auto_watcher_disabled_by_flag(self):
        self.mod._WATCHER_STARTED = False
        bc = types.ModuleType("bobert_companion")
        bc.HOLO_WORKSHOP_AUTO_ON_THINK = False
        with inject_modules(bobert_companion=bc), \
                mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_auto_watcher()
        Thread.assert_not_called()
        self.assertFalse(self.mod._WATCHER_STARTED)

    def test_bambu_watcher_starts_once(self):
        self.mod._BAMBU_WATCHER_STARTED = False
        with mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_bambu_watcher()
            self.mod._maybe_start_bambu_watcher()
        Thread.assert_called_once()
        self.assertTrue(self.mod._BAMBU_WATCHER_STARTED)

    def test_bambu_watcher_disabled_by_flag(self):
        self.mod._BAMBU_WATCHER_STARTED = False
        bc = types.ModuleType("bobert_companion")
        bc.BAMBU_OVERLAY_AUTO_WHILE_PRINTING = False
        with inject_modules(bobert_companion=bc), \
                mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_bambu_watcher()
        Thread.assert_not_called()

    def test_print_monitor_watcher_starts_once(self):
        self.mod._WORKSHOP_PRINT_MONITOR_WATCHER_STARTED = False
        with mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_workshop_print_monitor_watcher()
            self.mod._maybe_start_workshop_print_monitor_watcher()
        Thread.assert_called_once()
        self.assertTrue(self.mod._WORKSHOP_PRINT_MONITOR_WATCHER_STARTED)

    def test_print_monitor_watcher_disabled_by_flag(self):
        self.mod._WORKSHOP_PRINT_MONITOR_WATCHER_STARTED = False
        bc = types.ModuleType("bobert_companion")
        bc.WORKSHOP_PRINT_MONITOR_AUTO_LAUNCH = False
        with inject_modules(bobert_companion=bc), \
                mock.patch.object(self.mod.threading, "Thread") as Thread:
            self.mod._maybe_start_workshop_print_monitor_watcher()
        Thread.assert_not_called()

    def test_starters_default_enabled_when_bobert_import_fails(self):
        # With bobert_companion absent (None-sentinel → ImportError), each
        # starter's config-read raises and is swallowed, leaving ``enabled``
        # at its True default → the watcher still arms. Covers the import-guard
        # except path in all three starters.
        cases = (
            ("_maybe_start_auto_watcher", "_WATCHER_STARTED"),
            ("_maybe_start_bambu_watcher", "_BAMBU_WATCHER_STARTED"),
            ("_maybe_start_workshop_print_monitor_watcher",
             "_WORKSHOP_PRINT_MONITOR_WATCHER_STARTED"),
        )
        for starter, flag in cases:
            setattr(self.mod, flag, False)
            with inject_modules(bobert_companion=None), \
                    mock.patch.object(self.mod.threading, "Thread") as Thread:
                getattr(self.mod, starter)()
            Thread.assert_called_once()
            self.assertTrue(getattr(self.mod, flag))


# ═══════════════════════════════════════════════════════════════════════════
#  register() auto-launch config-read guards — import failure is swallowed
# ═══════════════════════════════════════════════════════════════════════════
class RegisterConfigGuardTests(_HoloBase):
    def test_register_survives_bobert_import_failure(self):
        # Every auto-launch flag is read inside try/except import bobert. With
        # bobert absent (None-sentinel), all those reads raise+swallow and
        # register() completes with the documented defaults: workshop_hud
        # auto-launches (default True), the opt-in surfaces stay dormant.
        with inject_modules(bobert_companion=None), \
                mock.patch("os.path.exists", return_value=True), \
                mock.patch("builtins.open", mock.mock_open()), \
                mock.patch.object(os, "replace"), \
                mock.patch.object(subprocess, "Popen") as popen:
            popen.return_value = _fake_proc(alive=True)
            mod, actions = load_skill_isolated("holographic_overlay")
        # Sanity: registration still happened and default auto-launch ran.
        self.assertIn("stark_status_ring", actions)
        self.assertIsNotNone(mod._WORKSHOP_HUD_PROCESS)
        self.assertIsNone(mod._OVERLAY_PROCESS)
        self.assertIsNone(mod._STARK_STATUS_PROCESS)


if __name__ == "__main__":
    unittest.main()
