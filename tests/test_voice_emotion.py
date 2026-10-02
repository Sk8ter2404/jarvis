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
        # NIGHT_QUIET_ENABLED patched on explicitly: a gitignored
        # user_settings.json with it off must not flip this test.
        from core import config as cfg
        ts = datetime.datetime(2026, 1, 1, 2, 0).timestamp()
        with mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", True, create=True):
            mood = ve.route_voice_emotion("open the notes", now=ts)["mood"]
        self.assertEqual(mood, "late_night")

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

    def test_repeat_request_with_exclamations_is_not_excited(self):
        # Review TONE-1: detect_tone / classify_emotion call "Say that
        # again!!" neutral (he didn't hear), but _detect_excited still counted
        # its '!!', so the router said 'excited' (a quip, the faster TTS
        # preset). Before the batch it said 'stressed'. Same rule as the other
        # two classifiers now: a daytime repeat request is 'casual'.
        noon = datetime.datetime(2026, 9, 29, 12, 0).timestamp()
        with mock.patch("core.tone_detector._is_late_night_hour",
                        return_value=False), \
                mock.patch("core.tone_detector.TONE_DETECTION_ENABLED", True), \
                mock.patch.object(ve, "VOICE_EMOTION_ROUTER_ENABLED", True):
            for text in ("Say that again!!", "What?! Say that again!",
                         "say that again please!!", "Repeat that!!",
                         "Sorry, I missed that!!"):
                with self.subTest(text=text):
                    self.assertFalse(ve._detect_excited(text))
                    r = ve.route_voice_emotion(text, now=noon)
                    self.assertEqual(r["mood"], "casual")
                    self.assertEqual(r["addendum"], "")
            # Control: real excitement with '!!' still routes to excited.
            self.assertEqual(
                ve.route_voice_emotion("This is amazing!!", now=noon)["mood"],
                "excited")
            self.assertTrue(ve._detect_excited("yes!! finally!!"))

    def test_two_am_yes_finally_gets_the_energy_match_the_docstring_promises(self):
        # Audit A87: the docstring's own example. detect_tone called it
        # 'stressed' (the bare '!!' rule), so the router said 'stressed' even
        # though _detect_excited was True.
        from core import config as cfg
        two_am = datetime.datetime(2026, 1, 1, 2, 0).timestamp()
        with mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", True, create=True):
            r = ve.route_voice_emotion("yes!! finally!!", now=two_am)
        self.assertEqual(r["mood"], "excited")

    def test_disabled_router_returns_casual(self):
        # When the feature flag is off the router short-circuits to casual
        # regardless of the text. Flag restored after the test.
        with mock.patch.object(ve, "VOICE_EMOTION_ROUTER_ENABLED", False):
            r = ve.route_voice_emotion("what the fuck is going on")
        self.assertEqual(r, {"mood": "casual", "addendum": ""})


class ExclaimedUtteranceTableTests(unittest.TestCase):
    """Audit A87: a '!!' with positive words ("yes!! finally!!", "we did it!!
    it works!!") read as stressed, which also dims the lights and mutes
    nudges for 15 min. The fix must not tip an angry '!!' into excited, so
    the two sides are pinned together. Noon, router and detector enabled."""

    NOON = datetime.datetime(2026, 9, 29, 12, 0).timestamp()

    POSITIVE = (
        "yes!! finally!!",
        "Yes!! It works!!",
        "we did it!! it works!!",
        "it worked!! finally!!",
        "finally!! it works!!",
        "yeah!! we did it!!",
        "YES!! nailed it!!",
    )
    NEGATIVE = (
        "stop!! now!!",
        "turn it off!! right now!!",
        "no!! not that one!!",
        "finally!!",
        "finally!! took you long enough!!",
        "it doesn't work!! fix it!!",
        "it still doesn't work!! finally!!",
        "it's not working!! yes I checked!!",
        "yes it's broken!!",
        "yeah!! it didn't work!!",
        "no!! yes!! whatever!!",
        "yes!! no!! stop!!",
        "yeah right!! it works!!",
        "it worked?! why did it crash!!",
        "fuck yes!!",
        "yes!! I said yes!!",
    )

    def _route(self, text):
        with mock.patch("core.tone_detector._is_late_night_hour",
                        return_value=False), \
                mock.patch("core.tone_detector.TONE_DETECTION_ENABLED", True), \
                mock.patch.object(ve, "VOICE_EMOTION_ROUTER_ENABLED", True):
            return ve.detect_tone(text), ve.route_voice_emotion(
                text, now=self.NOON)["mood"]

    def test_positive_exclamations_read_as_excited(self):
        for text in self.POSITIVE:
            with self.subTest(text=text):
                self.assertEqual(self._route(text), ("excited", "excited"))

    def test_angry_exclamations_stay_stressed(self):
        for text in self.NEGATIVE:
            with self.subTest(text=text):
                tone, mood = self._route(text)
                self.assertIn(tone, ("stressed", "frustrated"))
                self.assertEqual(mood, "stressed")

    def test_positive_words_without_exclamations_are_not_excited(self):
        # The new markers count only when shouted: "yes, open it" is a
        # plain answer, not excitement.
        for text in ("yes, open it", "it works now, thanks", "yeah go ahead"):
            with self.subTest(text=text):
                self.assertNotEqual(self._route(text)[1], "excited")


if __name__ == "__main__":
    unittest.main()
