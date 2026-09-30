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



class UnitConversionDetectionTests(unittest.TestCase):
    """2026-09-29: a unit conversion is arithmetic, not a weather or hardware
    question — the prompt router and the monolith's preemptive weather
    injector both ask these two helpers before treating unit words as one."""

    def test_conversion_requests(self):
        from core.units import is_unit_conversion_request as conv
        for text in ("convert 100 degrees fahrenheit to celsius",
                     "Convert 20 C to F", "what is 30 celsius in fahrenheit",
                     "what's 5 feet in meters", "10 kg to pounds",
                     "how many cups in a quart",
                     "how many grams are in an ounce",
                     "how many degrees celsius is 100 fahrenheit",
                     "100 fahrenheit in celsius?"):
            with self.subTest(text=text):
                self.assertTrue(conv(text))

    def test_not_conversion_requests(self):
        from core.units import is_unit_conversion_request as conv
        for text in ("what's the weather in celsius",
                     "how hot is my gpu in celsius",
                     "set the thermostat to 70 degrees", "turn it 90 degrees",
                     "how cold is it outside", "how many degrees is it outside",
                     "will it be 90 degrees tomorrow",
                     "how many miles is it to the store", "", None):
            with self.subTest(text=text):
                self.assertFalse(conv(text))

    def test_a_weather_cue_is_never_a_conversion(self):
        # Review 2026-09-29: a zip code or a forecast figure plus "in
        # celsius" is the weather, which must keep its routes and injection.
        from core.units import is_unit_conversion_request as conv
        for text in ("what's the weather in 90210 in celsius",
                     "what's the weather for zip 12345 in celsius",
                     "is it going to hit 90 today in fahrenheit",
                     "what will it be at 5 pm tomorrow in fahrenheit",
                     "convert today's high to celsius"):
            with self.subTest(text=text):
                self.assertFalse(conv(text))

    def test_may_be_weather_request(self):
        from core.units import may_be_weather_request as wx
        for text in ("what's the temperature", "how cold is it",
                     "what's the weather", "will it rain tomorrow"):
            with self.subTest(text=text):
                self.assertTrue(wx(text))
        for text in ("convert 100 degrees fahrenheit to celsius",
                     "what is 30 celsius in fahrenheit", "", None):
            with self.subTest(text=text):
                self.assertFalse(wx(text))

    def test_reply_in_both_scales_is_a_conversion(self):
        from core.units import reply_is_temperature_conversion as both
        self.assertTrue(both("100 degrees Fahrenheit is about 37.8 degrees "
                             "Celsius, sir."))
        self.assertTrue(both("72°F is 22°C, sir."))
        self.assertFalse(both("That comes to 37.8 degrees Celsius, sir."))
        self.assertFalse(both("It's 64 degrees Fahrenheit, sir."))
        # Both scales WITH a weather cue is the weather, not a conversion.
        self.assertFalse(both("It's 64 degrees Fahrenheit outside, about 18 "
                              "Celsius."))
        self.assertFalse(both(None))


if __name__ == "__main__":
    unittest.main()
