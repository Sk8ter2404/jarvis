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
from unittest import mock


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


# ══════════════════════════════════════════════════════════════════════════
#  BLACKOUT BACKSTOP (P2-7): the reticle now actively re-keys the colour-key via
#  Win32 (like the air-cursor), and its degraded-alpha fallback is LOW (faint),
#  not a near-opaque sheet over four monitors.
# ══════════════════════════════════════════════════════════════════════════
class ClickThroughBackstopTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_reticle(self)

    def test_exstyle_ors_in_layered_and_transparent(self):
        out = self.mod._click_through_exstyle(0)
        self.assertTrue(out & self.mod.WS_EX_LAYERED)
        self.assertTrue(out & self.mod.WS_EX_TRANSPARENT)
        self.assertTrue(out & self.mod.WS_EX_NOACTIVATE)
        self.assertTrue(out & self.mod.WS_EX_TOOLWINDOW)

    def test_exstyle_preserves_existing_bits(self):
        sentinel = 0x00000400
        out = self.mod._click_through_exstyle(sentinel)
        self.assertTrue(out & sentinel)
        self.assertTrue(out & self.mod.WS_EX_LAYERED)

    def test_colorref_byte_order(self):
        # #4cc9ff → COLORREF 0x00ffc94c (0x00bbggrr).
        self.assertEqual(self.mod._colorref("#4cc9ff"), 0xFFC94C)
        self.assertEqual(self.mod._colorref("010101"), 0x010101)

    def test_source_relayers_colorkey_after_exstyle(self):
        # The root-cause fix: after SetWindowLongW touches WS_EX_LAYERED the code
        # MUST re-establish the colour-key via SetLayeredWindowAttributes(COLORKEY)
        # or the full-desktop layered window composites OPAQUE (the blackout).
        path = os.path.join(_HUD_DIR, "jarvis_reticle.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("SetLayeredWindowAttributes", src)
        self.assertIn("LWA_COLORKEY", src)
        self.assertIn("_make_click_through_win32", src)

    def test_degraded_alpha_fallback_is_faint_not_a_sheet(self):
        # The no-colour-key fallback alpha must be LOW (~0.25) so a degraded
        # reticle is faint, never a dim sheet over four monitors. Assert the
        # source uses a low alpha and NOT the old 0.85.
        path = os.path.join(_HUD_DIR, "jarvis_reticle.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"-alpha", 0.25', src)
        self.assertNotIn('"-alpha", 0.85', src)


# ══════════════════════════════════════════════════════════════════════════
#  PID-RECYCLE GUARD + STALE-STATE EXIT (P0-2)
# ══════════════════════════════════════════════════════════════════════════
class _FakeProc:
    def __init__(self, create_time):
        self._ct = create_time

    def create_time(self):
        return self._ct


class _FakePsutil:
    def __init__(self, *, exists=True, create_time=1000.0, raise_on_process=False):
        self._exists = exists
        self._ct = create_time
        self._raise = raise_on_process

    def pid_exists(self, pid):
        return self._exists

    def Process(self, pid):
        if self._raise:
            raise RuntimeError("no such process")
        return _FakeProc(self._ct)


class ParentAliveRecycleTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_reticle(self)
        # 2026-07-12: _is_parent_alive consults the AUTHORITATIVE
        # core.parent_watch layer first (real Win32 syscalls — it correctly
        # reads the fake pid 4242 as dead). Stub it inconclusive-alive so
        # these tests keep exercising the psutil fallback + recycle-guard
        # semantics beneath it; parent_watch has its own suite.
        import core.parent_watch as _pw
        p = mock.patch.object(_pw, "parent_is_alive", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def _with_psutil(self, fake):
        self.addCleanup(setattr, self.mod, "psutil", getattr(self.mod, "psutil", None))
        self.addCleanup(setattr, self.mod, "_HAS_PSUTIL", self.mod._HAS_PSUTIL)
        self.mod.psutil = fake
        self.mod._HAS_PSUTIL = True

    def test_matching_start_time_is_alive(self):
        self._with_psutil(_FakePsutil(exists=True, create_time=1234.5))
        self.assertTrue(self.mod._is_parent_alive(4242, 1234.5))

    def test_recycled_pid_is_dead(self):
        self._with_psutil(_FakePsutil(exists=True, create_time=9999.0))
        self.assertFalse(self.mod._is_parent_alive(4242, 1234.5))

    def test_no_start_time_pid_exists_only(self):
        self._with_psutil(_FakePsutil(exists=True, create_time=42.0))
        self.assertTrue(self.mod._is_parent_alive(4242))
        self._with_psutil(_FakePsutil(exists=False))
        self.assertFalse(self.mod._is_parent_alive(4242))

    def test_unreadable_live_start_time_is_alive(self):
        self._with_psutil(_FakePsutil(exists=True, raise_on_process=True))
        self.assertTrue(self.mod._is_parent_alive(4242, 1234.5))


class StaleStateExitTests(unittest.TestCase):
    """_should_exit closes the overlay when the host stops updating
    hud_reticles.json (crashed without a clean shutdown). Built WITHOUT __init__
    so no Tk root / real file is needed."""

    def setUp(self):
        self.mod = _load_reticle(self)

    def _ov(self, *, parent_pid=4242, last_mtime=1000.0):
        ov = object.__new__(self.mod.ReticleOverlay)
        ov.parent_pid = parent_pid
        ov.parent_start = None
        ov._started_at = 0.0
        ov._last_state_mtime = last_mtime
        return ov

    def test_live_parent_does_not_exit_on_stale_state(self):
        # 2026-07-08 fix: a live REAL parent must NOT self-exit just because
        # hud_reticles.json went stale. A healthy JARVIS only rewrites that file
        # on a UI-automation action, so a voice-only / idle stretch >
        # STATE_STALE_EXIT_S is normal, not a crash — and nothing re-spawns a
        # self-exited overlay, so click/type reticles would silently stop for the
        # rest of the session. Trust the parent-liveness check for a real parent.
        ov = self._ov(last_mtime=1000.0)
        now = 1000.0 + self.mod.STATE_STALE_EXIT_S + 5.0
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: True), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: 1000.0):
            self.assertFalse(ov._should_exit(now))

    def test_orphan_exits_when_state_file_stale(self):
        # An ORPHAN (no real parent to trust) still exits on a stale state file.
        # started_at within ORPHAN_MAX_LIFETIME_S of now so the orphan-cap path
        # doesn't pre-empt — this asserts the stale path specifically. 2026-07-08.
        ov = self._ov(parent_pid=0, last_mtime=1000.0)
        ov._started_at = 1000.0
        now = 1000.0 + self.mod.STATE_STALE_EXIT_S + 5.0
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: True), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: 1000.0):
            self.assertTrue(ov._should_exit(now))

    def test_does_not_exit_while_state_fresh(self):
        ov = self._ov(last_mtime=1000.0)
        now = 1000.0 + 5.0
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: True), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: now):
            self.assertFalse(ov._should_exit(now))
        # The freshest-seen mtime advanced (so a later check measures from it).
        self.assertEqual(ov._last_state_mtime, now)

    def test_exits_when_parent_dead(self):
        ov = self._ov()
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: False), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: 1e12):
            self.assertTrue(ov._should_exit(2000.0))

    def test_orphan_cap_exits_after_lifetime(self):
        ov = self._ov(parent_pid=0)
        ov._started_at = 0.0
        now = self.mod.ORPHAN_MAX_LIFETIME_S + 10.0
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: True), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: now):
            self.assertTrue(ov._should_exit(now))

    def test_state_file_mtime_missing_is_zero(self):
        # A missing file yields 0.0 (no exception), and a 0 last-mtime never
        # trips the stale exit (we only measure once the file has been seen).
        with mock.patch.object(self.mod.os.path, "getmtime",
                               side_effect=OSError("gone")):
            self.assertEqual(self.mod._state_file_mtime(), 0.0)
        ov = self._ov(last_mtime=0.0)
        with mock.patch.object(self.mod, "_is_parent_alive", lambda *a: True), \
             mock.patch.object(self.mod, "_state_file_mtime", lambda: 0.0):
            self.assertFalse(ov._should_exit(1e9))


if __name__ == "__main__":
    unittest.main()
