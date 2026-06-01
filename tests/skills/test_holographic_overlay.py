"""Logic tests for skills/holographic_overlay (package skill).

This is a visual skill: it manages ~8 PyQt/tkinter HUD subprocesses + watcher
threads. Per the test plan, visual skills with little non-visual logic get a
few focused tests covering: loads/registers cleanly, the pure geometry/state
helpers, and a couple of action dispatches with subprocess.Popen mocked so no
real window is ever spawned.

subprocess.Popen is patched throughout so register()'s auto-launch and the
action launchers never start a real process. Watcher threads are neutered by
the harness (Thread.start no-op).
"""
from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _load():
    """Load the package skill with subprocess.Popen mocked so register()'s
    auto-launch (workshop_hud is on by default) can't spawn a real HUD."""
    with mock.patch.object(subprocess, "Popen") as popen:
        popen.return_value = mock.MagicMock(poll=mock.Mock(return_value=None))
        mod, actions = load_skill_isolated("holographic_overlay")
    return mod, actions


class HolographicLoadTests(unittest.TestCase):
    def test_registers_core_action_aliases(self):
        mod, actions = _load()
        # A representative slice of the documented voice triggers must be wired.
        for name in ("show_holographic_overlay", "hide_holographic_overlay",
                     "arc_reactor", "arc_reactor_on", "bambu_overlay_status",
                     "workshop_hud_status", "stark_status_ring"):
            self.assertIn(name, actions)

    def test_does_not_register_hide_hud(self):
        # Explicit comment in the source: 'hide_hud' is owned by the monolith
        # and must NOT be registered here (it would shadow the main HUD closer).
        _mod, actions = _load()
        self.assertNotIn("hide_hud", actions)


class HolographicGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load()

    def test_get_monitor_rect_fallback_when_import_fails(self):
        # Force the bobert import to fail → 1440p fallback rect.
        with mock.patch.dict("sys.modules", {"bobert_companion": None}):
            rect = self.mod._get_monitor_rect()
        self.assertEqual(rect, (0, 0, 2560, 1440))

    def test_get_monitor_rect_reads_bobert_top(self):
        import sys
        bc = mock.MagicMock()
        bc.MONITORS = {"top": (100, 200, 1920, 1080)}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertEqual(self.mod._get_monitor_rect(), (100, 200, 1920, 1080))

    def test_workshop_geometry_anchors_bottom_right(self):
        with mock.patch.object(self.mod, "_get_monitor_rect",
                               return_value=(0, 0, 2560, 1440)):
            x, y, w, h = self.mod._resolve_workshop_geometry()
        # Anchored to the bottom-right of the monitor, inside its bounds.
        self.assertLess(x + w, 2560 + 1)
        self.assertLess(y + h, 1440 + 1)
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)

    def test_print_monitor_geometry_centers_horizontally(self):
        with mock.patch.object(self.mod, "_get_monitor_rect",
                               return_value=(0, 0, 2000, 1000)):
            x, y, w, _h = self.mod._resolve_workshop_print_monitor_geometry()
        # Centered: left margin ≈ right margin.
        self.assertEqual(x, (2000 - w) // 2)


class HolographicStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load()

    # ── _bambu_is_active ─────────────────────────────────────────────────
    def test_bambu_is_active_states(self):
        self.assertTrue(self.mod._bambu_is_active({"gcode_state": "RUNNING"}))
        self.assertTrue(self.mod._bambu_is_active({"gcode_state": "pause"}))   # case-insens
        self.assertTrue(self.mod._bambu_is_active({"gcode_state": "PREPARE"}))
        self.assertFalse(self.mod._bambu_is_active({"gcode_state": "FINISH"}))
        self.assertFalse(self.mod._bambu_is_active({}))

    # ── _read_jarvis_state ───────────────────────────────────────────────
    def test_read_jarvis_state_defaults_idle(self):
        with mock.patch("os.path.exists", return_value=False):
            self.assertEqual(self.mod._read_jarvis_state(), "idle")

    # ── _act_arc_reactor dispatch ────────────────────────────────────────
    def test_arc_reactor_on_dispatch(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(True, "Arc reactor online, sir.")) as launch:
            out = self.actions["arc_reactor"]("on")
        self.assertIn("online", out)
        launch.assert_called_once_with("on")

    def test_arc_reactor_off_dispatch(self):
        with mock.patch.object(self.mod, "_shutdown_workshop",
                               return_value=(True, "Arc reactor disengaged, sir.")) as sd:
            out = self.actions["arc_reactor"]("off")
        self.assertIn("disengaged", out)
        sd.assert_called_once()

    def test_arc_reactor_unknown_arg_defaults_on(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(True, "Arc reactor online, sir.")) as launch:
            self.actions["arc_reactor"]("flibbertigibbet")
        launch.assert_called_once_with("on")   # permissive: unknown → on

    def test_action_returns_refused_prefix_on_failure(self):
        with mock.patch.object(self.mod, "_launch_workshop",
                               return_value=(False, "script missing")):
            out = self.actions["arc_reactor_on"]("")
        self.assertTrue(out.startswith("REFUSED:"))

    # ── status when not alive ────────────────────────────────────────────
    def test_overlay_status_when_dormant(self):
        with mock.patch.object(self.mod, "_overlay_is_alive", return_value=False):
            out = self.actions["holographic_overlay_status"]("") \
                if "holographic_overlay_status" in self.actions else self.mod._act_status("")
        self.assertIn("not currently engaged", out)

    def test_bambu_overlay_status_dormant(self):
        with mock.patch.object(self.mod, "_bambu_overlay_is_alive", return_value=False):
            self.mod._BAMBU_OVERLAY_USER_OFF = False
            out = self.actions["bambu_overlay_status"]("")
        self.assertIn("dormant", out)


class HolographicLaunchTests(unittest.TestCase):
    """Exercise a launcher's missing-script + Popen-failure paths without
    spawning anything real."""
    def setUp(self):
        self.mod, self.actions = _load()
        self.mod._OVERLAY_PROCESS = None

    def test_launch_overlay_missing_script(self):
        with mock.patch("os.path.exists", return_value=False):
            ok, msg = self.mod._launch_overlay()
        self.assertFalse(ok)
        self.assertIn("missing", msg)

    def test_launch_overlay_popen_failure(self):
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch.object(subprocess, "Popen", side_effect=OSError("denied")):
            ok, msg = self.mod._launch_overlay()
        self.assertFalse(ok)
        self.assertIn("failed to launch", msg)
        self.assertIsNone(self.mod._OVERLAY_PROCESS)   # reset on failure

    def test_launch_overlay_success(self):
        fake_proc = mock.MagicMock(poll=mock.Mock(return_value=None))
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch.object(subprocess, "Popen", return_value=fake_proc):
            ok, msg = self.mod._launch_overlay()
        self.assertTrue(ok)
        self.assertIn("online", msg)


if __name__ == "__main__":
    unittest.main()
