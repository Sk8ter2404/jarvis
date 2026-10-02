"""DIES ON OPEN (R11), wired into the monolith.

THE LIVE SEQUENCE (2026-09-29 19:39-19:59, v2.0.134): the Kinect dropped off
USB within about a second of every open, and the camera gate's backoff ladder
reopened it at 30/60/120/300/600 s and then every 10 minutes - a USB
re-enumeration (and an audio device-list change) each time. The gate now puts
such a device on a slow retry and says so once (tests/test_camera_gate_dies_
on_open.py pins the rule). This file pins the WIRING: the owner knob is
shipped, mirrored in the Settings schema and the template, and read by the
gate the monolith builds; and the owner's "use the Kinect again"
(camera_unquarantine) clears the slow retry, not only a culprit quarantine.

Nothing opens a camera or the Kinect: the gate is driven by hand on a frozen
clock, with a fake device list. Names are synthetic.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_camera_storm import _FakeBackend, _StormBase

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_KNOB = "CAMERA_DIES_ON_OPEN_RETRY_S"


@requires_monolith
class DiesOnOpenKnobTests(_StormBase):

    def test_config_ships_it_as_a_float(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        self.assertIs(type(lits.get(_KNOB)), float, _KNOB)
        self.assertEqual(lits[_KNOB], 1800.0)

    def test_settings_window_and_template_mirror_it(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "jarvis_settings_window_dies_on_open",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            example = json.load(fh)
        self.assertTrue(_KNOB in sw.SCHEMA, f"{_KNOB} is not in the schema")
        self.assertEqual(sw.SCHEMA[_KNOB]["type"], "float")
        self.assertEqual(sw.SCHEMA[_KNOB]["default"], 1800.0)
        self.assertEqual(example.get(_KNOB), 1800.0)
        self.assertIs(type(example[_KNOB]), float)

    def test_the_gate_the_monolith_builds_reads_it(self):
        bc = self.bc
        with mock.patch.object(bc, _KNOB, 900.0, create=True):
            g = bc._make_camera_gate()
        self.assertEqual(getattr(g, "dies_on_open_retry_s", None), 900.0)
        self.assertEqual(getattr(bc._camera_gate, "dies_on_open_retry_s", None),
                         getattr(bc, _KNOB, None))


@requires_monolith
class DiesOnOpenSurvivesRestartWiringTests(_StormBase):
    """2026-10-01: the production gate saves its dies-on-open runs to the
    staging-aware data dir and restores them at boot; the test harness never
    gets a state file it did not ask for."""

    def test_production_path_is_a_data_file_but_none_under_the_harness(self):
        import tempfile
        bc = self.bc
        self.assertIsNone(bc._camera_gate_doo_state_path())
        d = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"JARVIS_TEST_MODE": "0"}), \
             mock.patch("core.paths.data_dir", return_value=d):
            # Still None: the test run's live-data guard is armed.
            self.assertIsNone(bc._camera_gate_doo_state_path())
            with mock.patch.dict(sys.modules):
                sys.modules.pop("tests.live_data_guard", None)
                self.assertEqual(bc._camera_gate_doo_state_path(),
                                 os.path.join(d, "camera_gate_doo.json"))
        self.assertIsNone(getattr(bc._camera_gate, "_doo_state_path", None))

    def test_the_monolith_gate_restores_a_saved_run(self):
        import tempfile
        bc = self.bc
        path = os.path.join(tempfile.mkdtemp(), "camera_gate_doo.json")
        now = self.clock.time()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "devices": {"kinect": {
                "count": 3, "retry_s": 1800.0, "until": now + 1200.0,
                "said_at": now - 60.0}}}, fh)
        g = bc._make_camera_gate(clock=self.clock.time, doo_state_path=path)
        self.assertTrue(g.dies_on_open("kinect"))
        d = g.begin("kinect", "kinect-bridge")
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "backoff")

    def test_setup_logging_re_logs_the_restored_run(self):
        # 2026-10-01 review: the gate restores at IMPORT, before setup_logging
        # Tees stdout into the session log, so under pythonw the "restored
        # ... held for X more" line was lost. setup_logging re-logs it.
        import tempfile
        bc = self.bc
        path = os.path.join(tempfile.mkdtemp(), "camera_gate_doo.json")
        now = self.clock.time()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "devices": {"kinect": {
                "count": 3, "retry_s": 1800.0, "until": now + 1200.0,
                "said_at": now - 60.0}}}, fh)
        with mock.patch.object(bc, "_camera_gate_log"):
            g = bc._make_camera_gate(clock=self.clock.time, doo_state_path=path)
        logged: list = []
        with mock.patch.object(bc, "_camera_gate", g), \
             mock.patch.object(bc, "_camera_gate_log", logged.append):
            bc._camera_gate_log_restored_runs()
        self.assertEqual(len(logged), 1, logged)
        self.assertIn("kinect: restored from the last run", logged[0])
        self.assertIn("held for", logged[0])
        # And setup_logging is what calls it.
        with mock.patch.object(bc, "_camera_gate_log_restored_runs") as relog, \
             mock.patch.object(bc, "LOGS_DIR", tempfile.mkdtemp()), \
             mock.patch.object(bc, "LOGGING_ENABLED", True), \
             mock.patch.object(bc, "_cleanup_old_logs"), \
             mock.patch.object(bc, "_log_file_handle", None), \
             mock.patch.object(bc, "_log_file_path", None), \
             mock.patch.object(bc.sys, "stdout", bc.sys.stdout), \
             mock.patch.object(bc.sys, "stderr", bc.sys.stderr), \
             mock.patch.object(bc.sys, "excepthook", bc.sys.excepthook), \
             mock.patch("faulthandler.enable"):
            try:
                bc.setup_logging()
            finally:
                if bc._log_file_handle is not None:
                    bc._log_file_handle.close()
        relog.assert_called_once_with()


@requires_monolith
class UseTheKinectAgainTests(_StormBase):
    """camera_unquarantine -> camera_gate_lift_quarantine must reach a device
    on the slow dies-on-open retry, which is not a quarantine."""

    def _dies_on_open(self, g, times: int):
        for _ in range(times):
            for _i in range(500):
                d = g.begin("kinect", "kinect-bridge")
                if d.allowed:
                    break
                self.clock.advance(min(max(0.5, d.wait_s), 60.0))
            self.assertTrue(d.allowed, d)
            self.clock.advance(0.3)
            g.end("kinect", "kinect-bridge", True)
            g.hold("kinect", "kinect-bridge")
            self.clock.advance(4.0)
            g.unhold("kinect", "kinect-bridge")
            g.note_drop("kinect", "kinect-bridge")

    def test_use_the_kinect_again_clears_the_slow_retry(self):
        bc = self.bc
        g = bc._make_camera_gate(clock=self.clock.time)
        backend = _FakeBackend(self.clock, names=lambda: (
            "SynthCam One", "Synth Kinect Sensor"))
        slow = getattr(g, "dies_on_open", lambda *_a: False)
        with mock.patch.object(bc, "_camera_gate", g, create=True), \
             mock.patch.object(bc, "_camera_backend", backend), \
             mock.patch.object(bc, "proactive_announce",
                               side_effect=lambda m, *a, **k:
                               self.spoken.append(m) or True), \
             mock.patch("builtins.print"):
            self._dies_on_open(g, 3)
            self.assertTrue(slow("kinect"), "three dies-on-open did not put "
                                            "the Kinect on the slow retry")
            self.assertEqual(g.begin("kinect", "kinect-bridge").reason,
                             "backoff")
            self.assertEqual(
                self.spoken,
                ["The Kinect's stream keeps dying a few seconds after every "
                 "start, sir. If it's dropping off USB, check its power "
                 "supply. I'll only retry it every thirty minutes."])
            self.assertEqual(bc.get_camera_gate_status()["dies_on_open"]
                             ["kinect"]["label"], "the Kinect")
            from skills import camera_system
            with mock.patch.object(camera_system, "_bc", return_value=bc):
                said = camera_system.camera_unquarantine("kinect")
            self.assertIn("the Kinect is back in use", said)
            self.assertFalse(slow("kinect"))
            self.assertTrue(g.begin("kinect", "kinect-bridge").allowed,
                            "the owner said to use it again; the slow "
                            "retry still held it")
            # Only the owner's words are said back: no second notice.
            self.assertEqual(len(self.spoken), 1)

    def test_a_webcam_quarantine_lift_is_unchanged(self):
        bc = self.bc
        g = bc._make_camera_gate(clock=self.clock.time)
        g.culprit_threshold = 1
        backend = _FakeBackend(self.clock)
        with mock.patch.object(bc, "_camera_gate", g, create=True), \
             mock.patch.object(bc, "_camera_backend", backend), \
             mock.patch.object(bc, "proactive_announce", return_value=True), \
             mock.patch("builtins.print"):
            g.begin("name:synthcam one", "face-track")
            g.end("name:synthcam one", "face-track", True)
            self.clock.advance(0.5)
            g.note_drop("name:synthcam one", "face-track")
            g.note_drop("name:synthcam two", "face-track")
            self.assertTrue(g.quarantined("name:synthcam one"))
            self.assertEqual(bc.camera_gate_lift_quarantine("kinect"), [])
            self.assertTrue(g.quarantined("name:synthcam one"))
            lifted = bc.camera_gate_lift_quarantine("")
            self.assertEqual(len(lifted), 1)
            self.assertFalse(g.quarantined("name:synthcam one"))


@requires_monolith
class VerdictSurvivesARestartWiringTests(_StormBase):
    """B083 (2026-10-01): the live process remembers the dies-on-open verdict
    in its data dir; a staging process never touches the live file. Merged
    with live-logs B056: the ONE mechanism is the gate's doo_state_path, from
    _camera_gate_doo_state_path() (staging-aware core.paths.data_file)."""

    _dies_on_open = UseTheKinectAgainTests._dies_on_open

    def _drive(self, *, staging: bool):
        from core import paths
        bc = self.bc
        # The Kinect is ON the (fake) device list: this machine's real one
        # must not decide whether the gate lets the bridge open it.
        backend = _FakeBackend(self.clock, names=lambda: (
            "SynthCam One", "Synth Kinect Sensor"))
        with mock.patch.object(paths, "is_staging", return_value=staging), \
             mock.patch.object(bc, "_camera_backend", backend), \
             mock.patch.object(bc, "proactive_announce", return_value=True), \
             mock.patch("builtins.print"):
            g = bc._make_camera_gate(
                clock=self.clock.time,
                doo_state_path=bc._camera_gate_doo_state_path())
            self._dies_on_open(g, 3)
        return g

    def test_live_remembers_and_staging_does_not(self):
        import shutil
        import tempfile
        from core import paths
        tmp = tempfile.mkdtemp(prefix="jarvis_doo_wiring_")
        self.addCleanup(shutil.rmtree, tmp, True)
        live = os.path.join(tmp, "data", "camera_gate_doo.json")
        os.makedirs(os.path.dirname(live))
        # A production process: not the test harness, no data-dir override,
        # the project rooted in the temp dir.
        with mock.patch.dict(os.environ, {"JARVIS_TEST_MODE": "0"}), \
             mock.patch.dict(sys.modules), \
             mock.patch.object(paths, "PROJECT_DIR", tmp):
            os.environ.pop(paths.DATA_DIR_ENV, None)
            sys.modules.pop("tests.live_data_guard", None)
            self._drive(staging=True)
            self.assertFalse(os.path.exists(live),
                             "a staging process wrote the live verdict file")
            self._drive(staging=False)
        with open(live, encoding="utf-8") as fh:
            self.assertIn("kinect", json.load(fh)["devices"])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
