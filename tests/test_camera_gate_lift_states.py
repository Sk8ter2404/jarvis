"""core/camera_gate.py - CameraGate.lift() says WHICH hold it lifted.

"use the Kinect again" (camera_unquarantine) lifts a culprit QUARANTINE and,
since v2.0.137 (R11), the slow DIES-ON-OPEN retry. The two need different
replies (the first is switched off again if the hub drops out; the second goes
back to the slow retry if it still dies on open), so the gate reports which
one it lifted. lift_quarantine() keeps its bool contract.

Real gate, frozen clock, synthetic keys (see tests/test_camera_gate_dies_on_
open.py for the rig).
"""
from __future__ import annotations

import unittest

from core import camera_gate as cg
from tests.test_camera_gate_dies_on_open import (
    COUNT, KINECT, LEFT, RIGHT, _Bridge, _Clock, _gate, _slow)


def _lift(g, key):
    fn = getattr(g, "lift", None)
    return fn(key) if callable(fn) else "no CameraGate.lift()"


class LiftStatesTests(unittest.TestCase):
    def _slow_kinect(self):
        clk = _Clock()
        g, _logs, _spoken = _gate(clk)
        b = _Bridge(g, clk)
        for _ in range(COUNT):
            self.assertTrue(b.dies_on_open())
        self.assertTrue(_slow(g, KINECT))
        return g

    def _quarantined_left(self, g=None, clk=None):
        clk = clk or _Clock()
        if g is None:
            g, _l, _s = _gate(clk, culprit_threshold=1,
                              storm_cooldown_s=60.0)
        g.begin(LEFT, "face-track")
        clk.advance(0.8)
        g.end(LEFT, "face-track", True)
        clk.advance(0.5)
        g.note_drop(LEFT, "face-track")
        g.note_drop(RIGHT, "face-track")
        self.assertTrue(g.quarantined(LEFT))
        return g

    def test_the_names_are_exported(self):
        self.assertEqual(getattr(cg, "LIFT_QUARANTINE", None), "quarantine")
        self.assertEqual(getattr(cg, "LIFT_SLOW_RETRY", None), "dies-on-open")
        self.assertIn("LIFT_QUARANTINE", cg.__all__)
        self.assertIn("LIFT_SLOW_RETRY", cg.__all__)

    def test_a_slow_retry_lift_says_so(self):
        g = self._slow_kinect()
        self.assertEqual(_lift(g, KINECT), ("dies-on-open",))
        self.assertEqual(_lift(g, KINECT), (), "nothing left to lift")

    def test_a_quarantine_lift_says_so(self):
        g = self._quarantined_left()
        self.assertEqual(_lift(g, LEFT), ("quarantine",))
        self.assertFalse(g.quarantined(LEFT))

    def test_both_holds_on_one_device(self):
        g = self._quarantined_left()
        rec = g._dev[LEFT]          # the slow retry, set by hand on top
        rec["doo_count"] = COUNT
        rec["doo_retry_s"] = 1800.0
        self.assertEqual(_lift(g, LEFT), ("quarantine", "dies-on-open"))

    def test_an_unknown_device_lifts_nothing(self):
        g = self._slow_kinect()
        self.assertEqual(_lift(g, "name:synth-nobody"), ())

    def test_lift_quarantine_keeps_its_bool_contract(self):
        g = self._slow_kinect()
        self.assertIs(g.lift_quarantine(KINECT), True)
        self.assertIs(g.lift_quarantine(KINECT), False)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
