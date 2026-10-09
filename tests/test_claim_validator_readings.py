"""core.claim_validator.find_ungrounded_reading / drop_ungrounded_readings.

Live 2026-10-09 17:53: "Jarvis, why can't I hear my YouTube video?" was
answered "I'm afraid the volume is currently set to 20%, sir. [ACTION:
system_pulse]" - system_pulse reports CPU / GPU / memory / windows, nothing
read the volume, and the invented number was spoken. A volume / brightness /
battery / temperature number is now held back unless an action result,
sensor line or context this turn (or the owner's own words) carried it; a
real reading quoted back still passes. Pure, light CI tier.

    python -m unittest tests.test_claim_validator_readings
"""
from __future__ import annotations

import unittest

from core import claim_validator as cv

LIVE_USER = "Jarvis, why can't I hear my YouTube video?"
LIVE_REPLY = ("I'm afraid the volume is currently set to 20%, sir. "
              "[ACTION: system_pulse]")
# A system_pulse-shaped result: numbers, one of them 20, but no volume.
PULSE = ("CPU 20% (8 cores) | GPU 31% at 52°C | RAM 58% of 32 GB | "
         "12 windows open")
AUDIO_READ = ("The system volume is at 20 percent, not muted, and sound is "
              "going to Speakers, sir.")


class LiveLineTests(unittest.TestCase):
    def test_live_line_is_flagged_with_nothing_read(self):
        self.assertEqual(
            cv.find_ungrounded_reading(LIVE_REPLY, user_text=LIVE_USER),
            "volume is currently set to 20%")

    def test_live_line_is_flagged_after_system_pulse(self):
        # system_pulse's 20 is a CPU number: it does not ground a volume.
        self.assertIsNotNone(cv.find_ungrounded_reading(
            LIVE_REPLY, grounding=[PULSE], user_text=LIVE_USER))

    def test_live_line_passes_when_the_volume_was_read(self):
        self.assertIsNone(cv.find_ungrounded_reading(
            LIVE_REPLY, grounding=[PULSE, AUDIO_READ], user_text=LIVE_USER))

    def test_drop_keeps_the_rest_of_the_reply(self):
        kept, flagged = cv.drop_ungrounded_readings(
            "I'm afraid the volume is currently set to 20%, sir. "
            "Shall I turn it up?", grounding=[PULSE], user_text=LIVE_USER)
        self.assertEqual(kept, "Shall I turn it up?")
        self.assertEqual(flagged, ["volume is currently set to 20%"])

    def test_drop_is_a_no_op_on_a_grounded_reply(self):
        reply = "The volume is at 20 percent, sir, and nothing is muted."
        self.assertEqual(
            cv.drop_ungrounded_readings(reply, grounding=[AUDIO_READ]),
            (reply, []))


class ReadingShapesTests(unittest.TestCase):
    def test_invented_readings_are_flagged(self):
        for reply in ("Your battery is at forty-five percent, sir.",
                      "Battery at 45%.",
                      "Volume's at twenty, sir.",
                      "The brightness is set to 70 percent.",
                      "The GPU temperature is 65 degrees.",
                      "The temperature outside is 72°F, sir.",
                      "You're at 45 percent battery, sir."):
            with self.subTest(reply=reply):
                self.assertIsNotNone(cv.find_ungrounded_reading(reply))

    def test_not_readings_are_left_alone(self):
        for reply in ("Shall I set the volume to 20%?",
                      "I'll set the volume to 20 percent.",
                      "If the volume is at 0%, nothing plays.",
                      "Ideally keep the battery above 20 percent.",
                      "20% volume should do it.",
                      "The volume is 2 notches up.",
                      "The moon's temperature swings by 250 degrees.",
                      "It is 2:53 PM, sir.",
                      "Your CPU is at 40 percent."):
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_ungrounded_reading(reply))

    def test_the_owners_own_number_grounds_it(self):
        self.assertIsNone(cv.find_ungrounded_reading(
            "The volume is now at 30 percent, sir.",
            user_text="set the volume to 30"))

    def test_a_conversion_is_arithmetic_not_a_reading(self):
        self.assertIsNone(cv.find_ungrounded_reading(
            "The temperature would be 37.8 degrees Celsius. "
            "So the temperature is 37.8 degrees.",
            user_text="convert 100 degrees fahrenheit to celsius"))

    def test_a_rounded_reading_is_grounded(self):
        self.assertIsNone(cv.find_ungrounded_reading(
            "The GPU temperature is 52 degrees, sir.",
            grounding=["GPU 31% at 51.6°C"]))

    def test_grounding_must_name_the_same_kind_of_reading(self):
        self.assertIsNotNone(cv.find_ungrounded_reading(
            "Your battery is at 58 percent.", grounding=[PULSE]))
        self.assertIsNone(cv.find_ungrounded_reading(
            "Your battery is at 58 percent.",
            grounding=["Battery 58%, plugged in"]))

    def test_never_raises(self):
        for bad in (None, "", 12, "volume is at"):
            self.assertIsNone(cv.find_ungrounded_reading(bad))
        self.assertEqual(cv.drop_ungrounded_readings(""), ("", []))


if __name__ == "__main__":
    unittest.main()
