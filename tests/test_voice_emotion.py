"""Tests for core.voice_emotion — the mood router extracted from the monolith.
Covers the excitement detector and the deterministic routing buckets. (The
casual/daytime path isn't asserted: detect_tone's late-night fallback reads the
real wall clock, so that bucket is clock-dependent by design.)"""
import datetime
import unittest
from unittest import mock

import core.voice_emotion as ve


class DetectExcitedTests(unittest.TestCase):
    def test_excitement_phrase(self):
        self.assertTrue(ve._detect_excited("this is awesome"))

    def test_exclamations_without_swearing(self):
        self.assertTrue(ve._detect_excited("yes!! finally!!"))

    def test_exclamations_with_swearing_not_excited(self):
        self.assertFalse(ve._detect_excited("fuck yes!!"))

    def test_plain_text(self):
        self.assertFalse(ve._detect_excited("open the calendar"))
        self.assertFalse(ve._detect_excited(""))

    def test_punctuation_only_is_not_excited(self):
        # Text that reduces to empty after stripping non-letters (so the
        # post-clean guard returns False) is not excited.
        self.assertFalse(ve._detect_excited("12345 ----"))


class RouteTests(unittest.TestCase):
    def test_swear_routes_to_stressed(self):
        self.assertEqual(ve.route_voice_emotion("what the fuck is going on")["mood"],
                         "stressed")

    def test_excited_routes_to_excited(self):
        r = ve.route_voice_emotion("this is amazing")
        self.assertEqual(r["mood"], "excited")
        self.assertIn("excited", r["addendum"].lower())

    def test_late_night_timestamp_forces_late_night(self):
        ts = datetime.datetime(2026, 1, 1, 2, 0).timestamp()
        self.assertEqual(ve.route_voice_emotion("open the notes", now=ts)["mood"],
                         "late_night")

    def test_cross_turn_repetition_routes_to_stressed(self):
        # 'frustrated' (from cross-turn restatement) folds into 'stressed'.
        r = ve.route_voice_emotion("turn off the lights",
                                   prev_user_text="turn off the lights now")
        self.assertEqual(r["mood"], "stressed")

    def test_returns_addendum_for_nonempty_mood(self):
        r = ve.route_voice_emotion("this is amazing")
        self.assertTrue(r["addendum"].startswith("\n\n[Per-turn voice tone]"))

    def test_daytime_neutral_routes_to_casual(self):
        # A neutral utterance at a daytime hour (no tone, not excited, not
        # late-night) falls through to 'casual' with an empty addendum. `now`
        # pins route's OWN clock check, but the nested detect_tone reads the real
        # wall clock with no arg — so pin tone_detector's late-night helper too,
        # else this flakes to 'late_night' at a late UTC hour on CI.
        ts = datetime.datetime(2026, 1, 1, 14, 0).timestamp()
        with mock.patch("core.tone_detector._is_late_night_hour", return_value=False):
            r = ve.route_voice_emotion("open the notes", now=ts)
        self.assertEqual(r["mood"], "casual")
        self.assertEqual(r["addendum"], "")

    def test_disabled_router_returns_casual(self):
        # When the feature flag is off the router short-circuits to casual
        # regardless of the text. Flag restored after the test.
        with mock.patch.object(ve, "VOICE_EMOTION_ROUTER_ENABLED", False):
            r = ve.route_voice_emotion("what the fuck is going on")
        self.assertEqual(r, {"mood": "casual", "addendum": ""})


if __name__ == "__main__":
    unittest.main()
