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


if __name__ == "__main__":
    unittest.main()
