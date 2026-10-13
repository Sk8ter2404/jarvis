"""core/camera_gate.py - per-camera FLAP counter (master list 315).

A camera gone from the device list and back within FLAP_WINDOW_S is a flap;
3 flaps in 10 minutes log ONE summary line (then at most one per 10 minutes).
Real gate, frozen clock, synthetic camera names.
"""
from __future__ import annotations

import unittest

from core import camera_gate as cg
from tests.test_camera_gate import _Clock, _gate

CAM_A = "name:cam_a"
CAM_B = "name:cam_b"


def _summaries(logs, key):
    return [ln for ln in logs if key in ln and "FLAPPING" in ln]


class FlapCounterTests(unittest.TestCase):
    def _rig(self):
        clk = _Clock()
        present = {CAM_A: True, CAM_B: True}
        g, logs, _s = _gate(clk, presence=lambda key: present.get(key))
        return clk, present, g, logs

    def _flap(self, clk, present, g, key, gone_s=20.0):
        """Vanish, stay gone gone_s, come back (the gate sees it on begin)."""
        present[key] = False
        g.note_drop(key, "face-track")
        clk.advance(gone_s)
        present[key] = True
        g.begin(key, "face-track")      # sees it back; settles
        clk.advance(cg.ABSENT_SETTLE_S + 1.0)
        d = g.begin(key, "face-track")  # settled: absent episode is over
        if d.allowed:
            g.end(key, "face-track", True)
        clk.advance(120.0)

    def test_three_short_drops_log_one_summary(self):
        clk, present, g, logs = self._rig()
        for _ in range(2):
            self._flap(clk, present, g, CAM_A)
        self.assertEqual(_summaries(logs, CAM_A), [])
        self._flap(clk, present, g, CAM_A)
        self.assertEqual(len(_summaries(logs, CAM_A)), 1)
        self.assertIn("3 short drops", _summaries(logs, CAM_A)[0])
        self.assertEqual(g.snapshot()["devices"][CAM_A]["flaps_10min"], 3)

    def test_a_long_outage_is_not_a_flap(self):
        clk, present, g, logs = self._rig()
        for _ in range(4):
            self._flap(clk, present, g, CAM_A, gone_s=cg.FLAP_WINDOW_S + 30.0)
        self.assertEqual(_summaries(logs, CAM_A), [])
        self.assertEqual(g.snapshot()["devices"][CAM_A]["flaps_10min"], 0)

    def test_six_flaps_are_rate_limited(self):
        clk, present, g, logs = self._rig()
        for _ in range(6):
            self._flap(clk, present, g, CAM_A)
        self.assertEqual(len(_summaries(logs, CAM_A)), 1)

    def test_summary_repeats_after_the_window_if_still_flapping(self):
        clk, present, g, logs = self._rig()
        for _ in range(3):
            self._flap(clk, present, g, CAM_A)
        clk.advance(cg.FLAP_SUMMARY_WINDOW_S)
        for _ in range(3):
            self._flap(clk, present, g, CAM_A)
        self.assertEqual(len(_summaries(logs, CAM_A)), 2)

    def test_flaps_older_than_ten_minutes_do_not_count(self):
        clk, present, g, logs = self._rig()
        for _ in range(2):
            self._flap(clk, present, g, CAM_A)
        clk.advance(cg.FLAP_SUMMARY_WINDOW_S)
        self._flap(clk, present, g, CAM_A)
        self.assertEqual(_summaries(logs, CAM_A), [])

    def test_cameras_are_counted_separately(self):
        clk, present, g, logs = self._rig()
        for _ in range(2):
            self._flap(clk, present, g, CAM_A)
        clk.advance(100.0)
        for _ in range(2):
            self._flap(clk, present, g, CAM_B)
        self.assertEqual(_summaries(logs, CAM_A), [])
        self.assertEqual(_summaries(logs, CAM_B), [])
        self._flap(clk, present, g, CAM_B)
        self.assertEqual(_summaries(logs, CAM_A), [])
        self.assertEqual(len(_summaries(logs, CAM_B)), 1)
        devs = g.snapshot()["devices"]
        self.assertLessEqual(devs[CAM_A]["flaps_10min"], 2)
        self.assertEqual(devs[CAM_B]["flaps_10min"], 3)


if __name__ == "__main__":
    unittest.main()
