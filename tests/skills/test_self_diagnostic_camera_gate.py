"""The self-diagnostic's webcam scan and soft wake ask the camera gate first
(2026-09-29).

During the owner's hub-reset storm no producer camera had a fresh frame, so
this probe's SCAN is exactly the branch that runs: it opens Media Foundation
indices itself (index 0 is the Kinect's MF interface) and then soft-wakes
whatever it found - two more opens into a bus that is resetting. With the gate
holding the devices it must open NOTHING and report UNVERIFIED (never a
failure: a repair task for a device we were forbidden to open would be a lie).

The monolith is replaced by a small fake module exposing only the gate
accessors this skill reads, backed by the real core/camera_gate.CameraGate on
a frozen clock. No camera is opened: _open_probe_capture is a counting fake.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from core import camera_gate as cg
from tests._skill_harness import load_skill_isolated


class _Clock:
    def __init__(self):
        self.t = 9000.0

    def __call__(self):
        return self.t


def _stub_cv2():
    mod = types.ModuleType("cv2")
    mod.CAP_DSHOW = 700
    mod.CAP_MSMF = 1400
    return mod


def _fake_bc(gate):
    bc = types.ModuleType("bobert_companion")
    bc._camera_io_lock = None
    bc.camera_gate_key_for_scan = lambda idx, backend: f"msmf:synth-{idx}"
    bc.camera_gate_begin = lambda key, comp, **kw: gate.begin(key, comp)
    bc.camera_gate_end = lambda key, comp, ok, **kw: gate.end(key, comp, ok, **kw)
    bc.camera_gate_cancel = lambda key, comp: gate.cancel(key, comp)
    bc.get_camera_gate_status = gate.snapshot
    return bc


class SelfDiagnosticAsksTheGateTests(unittest.TestCase):

    def setUp(self):
        self.clock = _Clock()
        self.gate = cg.CameraGate(clock=self.clock, log=lambda _m: None,
                                  announce=lambda _m: None)
        self.mod, _ = load_skill_isolated("self_diagnostic", register=False)
        for p in (mock.patch.dict(sys.modules, {
                      "cv2": _stub_cv2(),
                      "bobert_companion": _fake_bc(self.gate)}),):
            p.start()
            self.addCleanup(p.stop)
        self.opened: list = []

        def _open(idx, backend, require_frame=0.0):
            self.opened.append(idx)
            return None
        p = mock.patch.object(self.mod, "_open_probe_capture", _open)
        p.start()
        self.addCleanup(p.stop)
        # The PnP / locker fallbacks below the scan must not run for real.
        for name, val in (("_windows_camera_pnp_devices", None),
                          ("_windows_camera_hardware_count", None),
                          ("_camera_lock_suspects", [])):
            p = mock.patch.object(self.mod, name, return_value=val)
            p.start()
            self.addCleanup(p.stop)

    def _storm(self):
        self.gate.note_drop("name:synth-a", "face-track")
        self.gate.note_drop("name:synth-b", "face-track")
        self.assertTrue(self.gate.storm_active())

    def test_a_storm_cool_down_means_the_scan_opens_nothing(self):
        self._storm()
        res = self.mod._probe_webcam()
        self.assertEqual(self.opened, [], "the scan opened a camera mid-storm")
        # UNVERIFIED - neither a pass nor a failure: ok False, tested False
        # (so no repair task and no spoken failure), with its own short cause.
        self.assertFalse(res.get("ok"), res)
        self.assertFalse(res.get("tested", True), res)
        details = res.get("details") or {}
        self.assertEqual(details.get("unverified_short_cause"),
                         self.mod._UNVERIFIED_SHORT_CAUSES["camera_gate"])
        self.assertTrue(details.get("usb_storm_active"), details)
        self.assertEqual(sorted(details.get("camera_gate_held", {})),
                         ["0", "1", "2"])

    def test_the_wake_is_refused_during_a_storm(self):
        self._storm()
        ok, note = self.mod._attempt_camera_wake(1, backend="msmf")
        self.assertFalse(ok)
        self.assertIn("camera gate", note)
        self.assertEqual(self.opened, [])

    def test_every_allowed_scan_open_is_reported_back(self):
        res = self.mod._probe_webcam()
        self.assertEqual(self.opened, [0], "the stagger should hold indices 1 "
                                           "and 2 inside the same second")
        self.assertFalse(res.get("tested", True), res)
        snap = self.gate.snapshot()["devices"]
        self.assertEqual(snap["msmf:synth-0"]["opens"], 1)
        self.assertEqual(snap["msmf:synth-0"]["fails"], 1)
        self.assertEqual(snap["msmf:synth-0"]["in_flight"], "")
        # A diagnostic failure arms no backoff (it is one spot check).
        self.assertEqual(snap["msmf:synth-0"]["level"], 0)


if __name__ == "__main__":
    unittest.main()
