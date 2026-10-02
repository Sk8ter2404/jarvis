"""Logic tests for skills/globe.py (the holographic globe).

The skill shells out to hud/globe_hud.py and talks to it through a control
file, so everything here runs with NO display and NO real subprocess:
  • the one launch seam, ``_spawn``, is replaced by a FakeLauncher that
    records argv and hands back a fake process handle;
  • the control file and the monitor layout are redirected per test
    (a temp dir; core.config.MONITORS / HUD_MONITOR patched);
  • the bundled city table is the real one (hud/data/world_cities.json), so
    the lookup tests exercise the data that ships.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

from core import config as core_config
from core.failure_markers import FAILURE_MARKERS

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MONITORS = {
    "left":   (-1920, 0, 1920, 1080),
    "middle": (0, 0, 2560, 1440),
    "top":    (0, -1440, 2560, 1440),
}


def _is_failure(reply: str) -> bool:
    low = reply.lower()
    return any(m in low for m in FAILURE_MARKERS)


class FakeLauncher:
    """Stands in for subprocess.Popen: records every argv, returns a handle
    whose poll() is None until terminate()/kill()."""

    def __init__(self, fail: Exception = None):
        self.calls = []
        self.procs = []
        self.fail = fail

    def __call__(self, argv):
        if self.fail is not None:
            raise self.fail
        self.calls.append(list(argv))
        proc = mock.MagicMock(name=f"FakeGlobe{len(self.procs)}")
        proc.poll.return_value = None

        def _die(*_a, **_k):
            proc.poll.return_value = 0
        proc.terminate.side_effect = _die
        proc.kill.side_effect = _die
        self.procs.append(proc)
        return proc


def _args(argv) -> dict:
    """{'--x': '…', …} from a recorded HUD argv."""
    return {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}


class _GlobeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_globe_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.mod, self.actions = load_skill_isolated("globe")
        self.control_path = os.path.join(self.tmp, "globe_hud_state.json")
        self.launcher = FakeLauncher()
        for target, value in (("_CONTROL_FILE", self.control_path),
                              ("_spawn", self.launcher)):
            p = mock.patch.object(self.mod, target, value)
            p.start()
            self.addCleanup(p.stop)
        for name, value in (("MONITORS", dict(_MONITORS)),
                            ("HUD_MONITOR", "top")):
            p = mock.patch.object(core_config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def control(self) -> dict:
        with open(self.control_path, "r", encoding="utf-8") as f:
            return json.load(f)


class ShowGlobeTests(_GlobeCase):
    def test_show_launches_the_hud_centred_on_the_hud_monitor(self):
        reply = self.actions["show_globe"]("")
        self.assertEqual(reply, "Globe up on the top monitor, sir.")
        self.assertEqual(len(self.launcher.calls), 1)
        argv = self.launcher.calls[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(os.path.normpath(argv[1]),
                         os.path.join(_ROOT, "hud", "globe_hud.py"))
        side = int(1440 * 0.62)
        self.assertEqual(_args(argv), {
            "--x": str((2560 - side) // 2),
            "--y": str(-1440 + (1440 - side) // 2),
            "--width": str(side), "--height": str(side),
            "--parent-pid": str(os.getpid())})
        self.assertEqual(self.control(), {"mode": "on", "pins": []})

    def test_monitor_name_passes_through(self):
        for spoken in ("| left", "left", "the left monitor", "Left Screen"):
            with self.subTest(spoken=spoken):
                self.launcher.calls.clear()
                self.mod._GLOBE_PROCESS = None
                self.mod._GLOBE_MONITOR = None
                reply = self.actions["show_globe"](spoken)
                self.assertEqual(reply, "Globe up on the left monitor, sir.")
                a = _args(self.launcher.calls[0])
                side = int(1080 * 0.62)
                self.assertEqual((a["--x"], a["--y"], a["--width"]),
                                 (str(-1920 + (1920 - side) // 2),
                                  str((1080 - side) // 2), str(side)))

    def test_main_means_the_primary_monitor(self):
        self.actions["show_globe"]("main")
        a = _args(self.launcher.calls[0])
        self.assertEqual(a["--x"], str((2560 - int(1440 * 0.62)) // 2))
        self.assertEqual(self.mod._GLOBE_MONITOR, "middle")

    def test_unknown_monitor_is_refused_without_launching(self):
        reply = self.actions["show_globe"]("kitchen")
        self.assertIn("kitchen", reply)
        self.assertIn("left, middle, top", reply)
        self.assertTrue(_is_failure(reply))
        self.assertEqual(self.launcher.calls, [])

    def test_single_instance(self):
        self.actions["show_globe"]("")
        self.assertEqual(self.actions["show_globe"](""),
                         "The globe is already up, sir.")
        self.assertEqual(self.actions["show_globe"]("top"),
                         "The globe is already up, sir.")
        self.assertEqual(len(self.launcher.calls), 1)
        self.launcher.procs[0].terminate.assert_not_called()

    def test_a_dead_globe_is_relaunched(self):
        self.actions["show_globe"]("")
        self.launcher.procs[0].poll.return_value = 1    # closed with Escape
        self.assertEqual(self.actions["show_globe"](""),
                         "Globe up on the top monitor, sir.")
        self.assertEqual(len(self.launcher.calls), 2)

    def test_another_monitor_moves_the_globe_and_keeps_its_pins(self):
        self.actions["show_globe"]("")
        self.actions["globe_pin"]("Tokyo")
        reply = self.actions["show_globe"]("left")
        self.assertEqual(reply, "Globe moved to the left monitor, sir.")
        self.launcher.procs[0].terminate.assert_called_once()
        self.assertEqual(len(self.launcher.calls), 2)
        self.assertEqual(_args(self.launcher.calls[1])["--x"],
                         str(-1920 + (1920 - int(1080 * 0.62)) // 2))
        self.assertEqual([p["label"] for p in self.control()["pins"]],
                         ["Tokyo"])
        self.assertEqual(self.control()["mode"], "on")

    def test_a_fresh_globe_drops_a_stale_sessions_pins(self):
        with open(self.control_path, "w", encoding="utf-8") as f:
            json.dump({"mode": "off", "seq": 4, "pins": [
                {"lat": 1.0, "lon": 2.0, "label": "old"}]}, f)
        self.actions["show_globe"]("")
        self.assertEqual(self.control()["pins"], [])
        self.assertEqual(self.control()["mode"], "on")

    def test_default_falls_back_to_the_primary_without_the_hud_monitor(self):
        with mock.patch.object(core_config, "MONITORS",
                               {"left": _MONITORS["left"],
                                "middle": _MONITORS["middle"]}):
            self.assertEqual(self.actions["show_globe"](""),
                             "Globe up on the middle monitor, sir.")

    def test_launch_failure_is_reported_honestly(self):
        with mock.patch.object(self.mod, "_spawn",
                               FakeLauncher(fail=OSError("no python"))):
            reply = self.actions["show_globe"]("")
        self.assertIn("couldn't start the globe", reply)
        self.assertTrue(_is_failure(reply))
        self.assertFalse(self.mod._globe_is_alive())

    def test_missing_renderer_is_refused(self):
        with mock.patch.object(self.mod, "_GLOBE_SCRIPT",
                               os.path.join(self.tmp, "nope.py")):
            reply = self.actions["show_globe"]("")
        self.assertTrue(reply.startswith("REFUSED"))
        self.assertEqual(self.launcher.calls, [])


class HideGlobeTests(_GlobeCase):
    def test_hide_when_not_running(self):
        self.assertEqual(self.actions["hide_globe"](""),
                         "The globe isn't up, sir.")
        self.assertEqual(self.launcher.calls, [])
        # A globe this process lost track of is still asked to close.
        self.assertEqual(self.control()["mode"], "off")

    def test_hide_terminates_and_signals_off(self):
        self.actions["show_globe"]("")
        proc = self.launcher.procs[0]
        self.assertEqual(self.actions["hide_globe"](""), "Globe dismissed, sir.")
        proc.terminate.assert_called_once()
        proc.wait.assert_called()
        self.assertEqual(self.control()["mode"], "off")
        self.assertFalse(self.mod._globe_is_alive())
        self.assertEqual(self.actions["hide_globe"](""),
                         "The globe isn't up, sir.")

    def test_hide_escalates_to_kill_when_terminate_hangs(self):
        self.actions["show_globe"]("")
        proc = self.launcher.procs[0]
        proc.terminate.side_effect = None            # ignores terminate()
        proc.wait.side_effect = [subprocess.TimeoutExpired("globe", 2.0), 0]
        self.actions["hide_globe"]("")
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)


class GlobePinTests(_GlobeCase):
    def test_pin_opens_the_globe_and_records_the_city(self):
        self.assertEqual(self.actions["globe_pin"]("Tokyo"), "Pinned Tokyo, sir.")
        self.assertEqual(len(self.launcher.calls), 1)
        ctl = self.control()
        self.assertEqual(ctl["mode"], "on")
        self.assertEqual(ctl["seq"], 1)
        (pin,) = ctl["pins"]
        self.assertEqual(pin["label"], "Tokyo")
        self.assertAlmostEqual(pin["lat"], 35.7, delta=0.3)
        self.assertAlmostEqual(pin["lon"], 139.7, delta=0.3)

    def test_pins_keep_their_order_and_bump_the_focus_seq(self):
        self.actions["show_globe"]("")
        self.actions["globe_pin"]("London")
        self.actions["globe_pin"]("New York")
        self.assertEqual(len(self.launcher.calls), 1)    # no relaunch
        ctl = self.control()
        self.assertEqual([p["label"] for p in ctl["pins"]],
                         ["London", "New York"])
        self.assertEqual(ctl["seq"], 2)

    def test_repinning_moves_the_pin_to_the_end_without_a_duplicate(self):
        for place in ("London", "Paris", "London"):
            self.actions["globe_pin"](place)
        ctl = self.control()
        self.assertEqual([p["label"] for p in ctl["pins"]], ["Paris", "London"])
        self.assertEqual(ctl["seq"], 3)

    def test_lookup_ignores_case_accents_and_punctuation(self):
        for spoken, name in (("TOKYO", "Tokyo"), ("tokyo", "Tokyo"),
                             ("São Paulo", "São Paulo"),
                             ("sao paulo", "São Paulo"),
                             ("SAO-PAULO", "São Paulo"),
                             ("zurich", "Zürich"), ("Zürich", "Zürich"),
                             ("reykjavik", "Reykjavík"),
                             ("København", "Copenhagen"),
                             ("copenhagen", "Copenhagen"),
                             ("washington dc", "Washington, D.C."),
                             ("Washington, D.C.", "Washington, D.C."),
                             ("New York City", "New York"),
                             ("the new york", "New York"),
                             ("Paris, France", "Paris"),
                             ("Bombay", "Mumbai"), ("  london  ", "London")):
            with self.subTest(spoken=spoken):
                row = self.mod._find_city(spoken)
                self.assertIsNotNone(row, spoken)
                self.assertEqual(row[0], name)

    def test_label_override(self):
        self.assertEqual(self.actions["globe_pin"]("Tokyo | the conference"),
                         "Pinned the conference, sir.")
        self.assertEqual(self.control()["pins"][0]["label"], "the conference")

    def test_coordinates(self):
        for spoken, lat, lon in (("35.7, 139.7", 35.7, 139.7),
                                 ("35.7 139.7", 35.7, 139.7),
                                 ("-33.9,18.4", -33.9, 18.4),
                                 ("33.9S 18.4E", -33.9, 18.4),
                                 ("51.5N, 0.1W", 51.5, -0.1),
                                 ("64.1° N, 21.9° W", 64.1, -21.9)):
            with self.subTest(spoken=spoken):
                self.assertEqual(self.mod._parse_latlon(spoken), (lat, lon))
        self.assertEqual(self.actions["globe_pin"]("35.7, 139.7"),
                         "Pinned 35.7, 139.7, sir.")
        self.assertEqual(self.control()["pins"][-1]["lat"], 35.7)

    def test_out_of_range_coordinates_are_not_a_place(self):
        self.assertIsNone(self.mod._parse_latlon("95, 20"))
        self.assertIsNone(self.mod._parse_latlon("45, 200"))
        reply = self.actions["globe_pin"]("95, 20")
        self.assertTrue(_is_failure(reply))
        self.assertEqual(self.launcher.calls, [])

    def test_unknown_city_is_an_honest_failure(self):
        reply = self.actions["globe_pin"]("Atlantis")
        self.assertIn("Atlantis", reply)
        self.assertIn("major cities", reply)
        self.assertIn(str(len(self.mod._load_cities())), reply)
        self.assertTrue(_is_failure(reply))   # → the failure follow-up
        self.assertEqual(self.launcher.calls, [])
        self.assertFalse(os.path.exists(self.control_path))

    def test_empty_pin_is_a_format_error(self):
        reply = self.actions["globe_pin"]("  ")
        self.assertTrue(reply.startswith("format:"))
        self.assertEqual(self.launcher.calls, [])

    def test_pin_fails_honestly_when_the_globe_cannot_open(self):
        with mock.patch.object(self.mod, "_spawn",
                               FakeLauncher(fail=OSError("boom"))):
            reply = self.actions["globe_pin"]("Tokyo")
        self.assertTrue(_is_failure(reply))
        self.assertNotIn("Pinned", reply)

    def test_pins_are_capped(self):
        names = [row[0] for row in self.mod._load_cities()[:20]]
        for name in names:
            self.actions["globe_pin"](name)
        pins = self.control()["pins"]
        self.assertEqual(len(pins), self.mod._MAX_PINS)
        self.assertEqual(pins[-1]["label"], names[-1])


class GlobeClearTests(_GlobeCase):
    def test_clear_keeps_the_globe_up(self):
        self.actions["globe_pin"]("Tokyo")
        self.assertEqual(self.actions["globe_clear"](""), "Pins cleared, sir.")
        self.assertEqual(self.control()["pins"], [])
        self.launcher.procs[0].terminate.assert_not_called()

    def test_clear_when_not_running(self):
        self.assertEqual(self.actions["globe_clear"](""),
                         "The globe isn't up, sir.")
        self.assertEqual(self.launcher.calls, [])


class RegistrationTests(unittest.TestCase):
    def test_register_exposes_the_actions_and_launches_nothing(self):
        with mock.patch("subprocess.Popen") as popen:
            mod, actions = load_skill_isolated("globe")
        self.assertEqual(set(actions),
                         {"show_globe", "hide_globe", "globe_pin", "globe_clear"})
        popen.assert_not_called()
        self.assertIsNone(mod._GLOBE_PROCESS)

    def test_the_skill_never_imports_tkinter(self):
        with open(os.path.join(_ROOT, "skills", "globe.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertNotIn("tkinter", names)

    def test_the_action_sweep_never_opens_a_real_globe(self):
        # tools/action_smoke.py calls every registered action on the live
        # desktop; the spawning ones must be on its deny list.
        from tools import action_smoke
        _mod, actions = load_skill_isolated("globe")
        for name in ("show_globe", "globe_pin", "globe_clear"):
            self.assertTrue(action_smoke._spawns_desktop_windows(
                name, actions[name]), name)
        self.assertFalse(action_smoke._spawns_desktop_windows(
            "hide_globe", actions["hide_globe"]))


class ControlFileContractTests(_GlobeCase):
    """What the skill writes is what the HUD reads (hud/globe_hud.parse_pins).
    Imports the HUD module (tkinter is imported but no window is built)."""

    def _hud(self):
        if importlib.util.find_spec("tkinter") is None:
            self.skipTest("tkinter not installed")
        spec = importlib.util.spec_from_file_location(
            "globe_hud_under_test", os.path.join(_ROOT, "hud", "globe_hud.py"))
        hud = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hud)
        return hud

    def test_pins_round_trip_into_the_hud(self):
        hud = self._hud()
        self.actions["globe_pin"]("London")
        self.actions["globe_pin"]("Tokyo | HQ")
        pins = hud.parse_pins(self.control())
        self.assertEqual([p[2] for p in pins], ["London", "HQ"])
        self.assertAlmostEqual(pins[1][0], 35.7, delta=0.3)
        self.assertEqual(hud.MAX_PINS, self.mod._MAX_PINS)

    def test_hud_drops_malformed_pins(self):
        hud = self._hud()
        self.assertEqual(hud.parse_pins({"pins": [
            {"lat": "x", "lon": 1}, {"lat": 95, "lon": 0}, {"lon": 3},
            "junk", {"lat": 10, "lon": 20, "label": None}]}),
            [(10.0, 20.0, "")])
        self.assertEqual(hud.parse_pins({"pins": "nope"}), [])
        self.assertEqual(hud.parse_pins([]), [])


if __name__ == "__main__":
    unittest.main()
