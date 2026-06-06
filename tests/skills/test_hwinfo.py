"""Unit tests for audio/hwinfo.py — the HWiNFO SM2 reading parser (verified
against a synthesised block, since the live shared memory needs HWiNFO's
'Shared Memory Support' toggle on) and the graceful no-data path."""
import struct
import unittest
from unittest import mock

from audio import hwinfo


def _reading(label, value, unit):
    b = struct.pack("<III", 7, 0, 0)                    # tReading, sensorIndex, readingID
    b += label.encode("latin-1").ljust(128, b"\x00")   # szLabelOrig
    b += b"".ljust(128, b"\x00")                        # szLabelUser (empty)
    b += unit.encode("latin-1").ljust(16, b"\x00")      # szUnit
    b += struct.pack("<dddd", value, 0, 0, 0)           # Value/Min/Max/Avg
    return b


def _block(*readings):
    rsize = 316
    hdr = struct.pack("<IIIqIIIIII", 0x53695748,   # real HWiNFO dwSignature -> LE bytes b"HWiS"
                      2, 0, 0, 44, 0, 0, 44, rsize, len(readings))
    return hdr + b"".join(readings)


class HwinfoParseTests(unittest.TestCase):
    def test_parse_readings(self):
        raw = _block(_reading("CORSAIR VOID ELITE Wireless Battery", 72.0, "%"),
                     _reading("CPU Package", 55.0, "C"))
        parsed = hwinfo.parse_readings(raw)
        self.assertEqual(parsed[0], ("CORSAIR VOID ELITE Wireless Battery", 72.0, "%"))
        self.assertEqual(parsed[1][0], "CPU Package")

    def test_parse_bad_block_is_empty(self):
        self.assertEqual(hwinfo.parse_readings(b""), [])
        self.assertEqual(hwinfo.parse_readings(b"junk-not-SiWH"), [])

    def test_wrong_byte_order_signature_rejected(self):
        # Regression: the old bug checked for b"SiWH"; real HWiNFO memory leads
        # with the DWORD 0x53695748 (LE bytes b"HWiS"). A block carrying the
        # wrong-order signature must NOT parse (guards the byte-order fix).
        wrong = struct.pack("<IIIqIIIIII", int.from_bytes(b"SiWH", "little"),
                            2, 0, 0, 44, 0, 0, 44, 316, 0)
        self.assertEqual(hwinfo.parse_readings(wrong), [])

    def test_battery_and_find_via_mock(self):
        raw = _block(_reading("CORSAIR VOID ELITE Wireless Battery", 72.0, "%"))
        with mock.patch.object(hwinfo, "_read_raw", return_value=raw):
            self.assertTrue(hwinfo.available())
            self.assertEqual(hwinfo.battery("VOID"), 72.0)
            self.assertEqual(hwinfo.find("void", "battery")[1], 72.0)

    def test_battery_prefers_battery_label_over_volume(self):
        # The headset exposes a Volume reading in '%' too — battery() must pick
        # the battery-labelled reading, not the first '%' it sees.
        raw = _block(_reading("CORSAIR VOID ELITE Headphone Volume", 50.0, "%"),
                     _reading("CORSAIR VOID ELITE Wireless Battery", 60.0, "%"))
        with mock.patch.object(hwinfo, "_read_raw", return_value=raw):
            self.assertEqual(hwinfo.battery("VOID"), 60.0)
            self.assertEqual(hwinfo.find("void", "battery")[1], 60.0)

    def test_unavailable_is_graceful(self):
        with mock.patch.object(hwinfo, "_read_raw", return_value=None):
            self.assertFalse(hwinfo.available())
            self.assertIsNone(hwinfo.battery("VOID"))
            self.assertEqual(hwinfo.readings(), [])


class HwinfoSummaryTests(unittest.TestCase):
    def _full_block(self):
        return _block(
            _reading("CPU Package", 58.0, "C"),
            _reading("CPU (Tctl/Tdie)", 61.0, "C"),
            _reading("GPU Temperature", 49.0, "C"),
            _reading("GPU Hot Spot", 70.0, "C"),
            _reading("CPU Fan", 1100.0, "RPM"),
            _reading("Chassis Fan #2", 900.0, "RPM"),
            _reading("CPU Package Power", 95.0, "W"),
            _reading("Total CPU Usage", 12.0, "%"),
            _reading("GPU Utilization", 30.0, "%"),
            _reading("Vcore", 1.25, "V"),
            _reading("Core Clock", 5200.0, "MHz"),
        )

    def test_summary_groups_by_unit_and_label(self):
        with mock.patch.object(hwinfo, "_read_raw", return_value=self._full_block()):
            s = hwinfo.summary()
        self.assertTrue(s["available"])
        self.assertEqual(s["count"], 11)
        # CPU temp prefers the Package sensor; GPU temp prefers edge/Temperature.
        self.assertEqual(s["cpu_temp_c"], 58.0)
        self.assertEqual(s["gpu_temp_c"], 49.0)
        self.assertEqual(s["cpu_load_pct"], 12.0)
        self.assertEqual(s["gpu_load_pct"], 30.0)   # "Utilization" counts as load
        self.assertEqual(len(s["temps_c"]), 4)
        self.assertEqual(len(s["fans_rpm"]), 2)
        self.assertEqual(s["power_w"], [("CPU Package Power", 95.0)])
        self.assertEqual(s["voltages_v"], [("Vcore", 1.25)])
        self.assertEqual(s["clocks_mhz"], [("Core Clock", 5200.0)])

    def test_summary_handles_degree_symbol_unit(self):
        # HWiNFO may emit the unit as "°C" rather than "C"; both must count.
        raw = _block(_reading("CPU Package", 50.0, "°C"))
        with mock.patch.object(hwinfo, "_read_raw", return_value=raw):
            s = hwinfo.summary()
        self.assertEqual(s["cpu_temp_c"], 50.0)
        self.assertEqual(len(s["temps_c"]), 1)

    def test_summary_fahrenheit_units_converted_to_celsius(self):
        # If HWiNFO is set to display Fahrenheit globally, temps arrive as
        # "F"/"°F"/"FAH" — they must still register as temps AND be converted to
        # Celsius on ingest (c = (f - 32) / 1.8), keeping cpu/gpu_temp_c
        # Celsius-canonical. Regression: F units used to drop ALL temps silently.
        raw = _block(
            _reading("CPU Package", 140.0, "°F"),      # -> 60.0 C
            _reading("GPU Temperature", 122.0, "F"),   # -> 50.0 C
            _reading("CPU CCD1 (Tdie)", 149.0, "FAH"), # -> 65.0 C
        )
        with mock.patch.object(hwinfo, "_read_raw", return_value=raw):
            s = hwinfo.summary()
        self.assertEqual(len(s["temps_c"]), 3)
        self.assertAlmostEqual(s["cpu_temp_c"], 60.0)   # Package preferred, F->C
        self.assertAlmostEqual(s["gpu_temp_c"], 50.0)
        # All temps_c values are Celsius, not the raw Fahrenheit readings.
        self.assertNotIn(140.0, [v for _, v in s["temps_c"]])

    def test_summary_ghz_clock_normalised_to_mhz(self):
        raw = _block(_reading("CPU Clock", 5.2, "GHz"))
        with mock.patch.object(hwinfo, "_read_raw", return_value=raw):
            s = hwinfo.summary()
        self.assertEqual(s["clocks_mhz"], [("CPU Clock", 5200.0)])

    def test_summary_unavailable(self):
        with mock.patch.object(hwinfo, "_read_raw", return_value=None):
            s = hwinfo.summary()
        self.assertFalse(s["available"])
        self.assertEqual(s["count"], 0)
        self.assertIsNone(s["cpu_temp_c"])
        self.assertEqual(s["fans_rpm"], [])


if __name__ == "__main__":
    unittest.main()
