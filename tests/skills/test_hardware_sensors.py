"""Logic tests for skills/hardware_sensors.py -- the HWiNFO sensor read-out
action, with the reader mocked (no live shared memory needed)."""
import sys
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _patch_hwinfo(fake):
    # The action does `from audio import hwinfo` lazily; route it to the fake.
    return mock.patch.dict(
        sys.modules,
        {"audio": mock.MagicMock(hwinfo=fake), "audio.hwinfo": fake},
    )


class HardwareSensorsTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("hardware_sensors")

    def _summary(self, **over):
        base = {"available": True, "count": 6,
                "cpu_temp_c": 55.0, "gpu_temp_c": 48.0,
                "cpu_load_pct": 10.0, "gpu_load_pct": None,
                "temps_c": [], "fans_rpm": [("CPU Fan", 1000.0), ("Rear", 800.0)],
                "power_w": [("CPU Package Power", 90.0)],
                "clocks_mhz": [], "voltages_v": []}
        base.update(over)
        return base

    def test_snapshot_reports_temps_fans_power(self):
        fake = mock.MagicMock()
        fake.summary.return_value = self._summary()
        with _patch_hwinfo(fake):
            out = self.actions["hardware_sensors"]("")
        self.assertIn("CPU 55", out)
        self.assertIn("GPU 48", out)
        self.assertIn("2 fans", out)
        self.assertIn("90 watts", out)

    def test_sm_off_gives_actionable_message(self):
        fake = mock.MagicMock()
        fake.summary.return_value = {"available": False}
        with _patch_hwinfo(fake):
            out = self.actions["hardware_sensors"]("")
        self.assertIn("Shared Memory Support", out)

    def test_fragment_query_finds_sensor(self):
        fake = mock.MagicMock()
        fake.find.return_value = ("GPU Hot Spot", 70.0, "C")
        with _patch_hwinfo(fake):
            out = self.actions["hardware_sensors"]("hot spot")
        self.assertIn("GPU Hot Spot", out)
        self.assertIn("70", out)

    def test_fragment_not_found_when_available(self):
        fake = mock.MagicMock()
        fake.find.return_value = None
        fake.available.return_value = True
        with _patch_hwinfo(fake):
            out = self.actions["hardware_sensors"]("nonsense")
        self.assertIn("couldn't find", out)

    def test_no_matching_labels_reports_sensor_count(self):
        fake = mock.MagicMock()
        fake.summary.return_value = self._summary(
            cpu_temp_c=None, gpu_temp_c=None, cpu_load_pct=None,
            fans_rpm=[], power_w=[], count=42)
        with _patch_hwinfo(fake):
            out = self.actions["hardware_sensors"]("")
        self.assertIn("42 sensors", out)


if __name__ == "__main__":
    unittest.main()
