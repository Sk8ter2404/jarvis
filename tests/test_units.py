"""Tests for core.units.meters_to_imperial_phrase — spoken imperial distances."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.units import meters_to_imperial_phrase  # noqa: E402


class MetersToImperialPhraseTests(unittest.TestCase):
    def test_close_range_is_feet(self):
        self.assertEqual(meters_to_imperial_phrase(0.6), "2 feet")   # ~1.97 ft
        self.assertEqual(meters_to_imperial_phrase(2.0), "7 feet")   # ~6.56 ft
        self.assertEqual(meters_to_imperial_phrase(2.5), "8 feet")   # ~8.2 ft

    def test_far_range_switches_to_yards(self):
        # >10 ft (>~3.05 m) reads in yards.
        self.assertEqual(meters_to_imperial_phrase(4.0), "4 yards")  # 13.1 ft → 4.4 yd
        self.assertEqual(meters_to_imperial_phrase(3.2), "3 yards")  # 10.5 ft → 3.5 yd

    def test_singular_grammar(self):
        # ~0.4 m → ~1.3 ft → the "about a foot" band (feet < 1.5).
        self.assertEqual(meters_to_imperial_phrase(0.4), "about a foot")
        # exactly ~0.914 m = 3 ft.
        self.assertEqual(meters_to_imperial_phrase(0.30), "about a foot")  # ~0.98 ft

    def test_never_says_metres(self):
        for m in (0.6, 1.5, 2.5, 3.5, 5.0, 10.0):
            self.assertNotIn("met", meters_to_imperial_phrase(m).lower())

    def test_missing_or_bad_value_is_empty(self):
        self.assertEqual(meters_to_imperial_phrase(None), "")
        self.assertEqual(meters_to_imperial_phrase(0), "")
        self.assertEqual(meters_to_imperial_phrase(-1), "")
        self.assertEqual(meters_to_imperial_phrase("nan-ish"), "")


if __name__ == "__main__":
    unittest.main()
