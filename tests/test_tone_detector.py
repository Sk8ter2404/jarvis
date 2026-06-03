"""Tests for core.tone_detector — the pre-LLM tone classifier extracted from
the monolith. Pins the label priority, the cross-turn repetition signal (now a
parameter, not a conversation_history reach-in), the late-night wrap-around, and
the addendum text. This logic ran on every utterance with zero test coverage
before the extraction."""
import datetime
import unittest
from unittest import mock

import core.tone_detector as td


class DetectToneTests(unittest.TestCase):
    def test_none_for_empty(self):
        self.assertIsNone(td.detect_tone(""))
        self.assertIsNone(td.detect_tone("   "))

    def test_none_when_text_is_only_non_letters(self):
        # A non-empty utterance that reduces to "" after the letter-only clean
        # (digits / punctuation only) returns None, not a tone.
        self.assertIsNone(td.detect_tone("12345 !!! ..."))

    def test_prev_user_text_str_failure_is_swallowed(self):
        # detect_tone guards the cross-turn similarity check: if coercing/parsing
        # prev_user_text raises, it degrades to "not similar" rather than
        # propagating. A prev whose __str__ blows up exercises that except.
        class Boom:
            def __str__(self):
                raise ValueError("cannot stringify")

        # Plain neutral current text → without the (failed) similarity signal it
        # classifies as None; the point is that it does not raise. Pin the clock
        # to daytime: detect_tone's late-night fallback reads the real wall clock
        # with no arg, so an un-pinned neutral result flakes to 'late_night' when
        # CI runs at a late UTC hour (matches LateNightTests' pattern).
        with mock.patch.object(td, "_is_late_night_hour", return_value=False):
            self.assertIsNone(td.detect_tone("open the notes", prev_user_text=Boom()))

    def test_frustrated_repetition_phrase(self):
        self.assertEqual(td.detect_tone("I said turn it off"), "frustrated")

    def test_frustrated_cross_turn_repetition(self):
        # Shares a majority of content words with the previous utterance →
        # the user is restating → frustrated, even with no "I said" marker.
        self.assertEqual(
            td.detect_tone("turn off the lights",
                           prev_user_text="turn off the lights now"),
            "frustrated",
        )

    def test_excited_beats_stressed(self):
        self.assertEqual(td.detect_tone("this is amazing"), "excited")

    def test_stressed_on_swear(self):
        self.assertEqual(td.detect_tone("what the fuck is going on"), "stressed")

    def test_rushed_on_urgency(self):
        self.assertEqual(td.detect_tone("do it now please"), "rushed")

    def test_tired(self):
        self.assertEqual(td.detect_tone("i'm exhausted"), "tired")

    def test_playful(self):
        self.assertEqual(td.detect_tone("haha nice one"), "playful")


class LateNightTests(unittest.TestCase):
    def test_late_band_true(self):
        self.assertTrue(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 23, 0)))
        self.assertTrue(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 2, 0)))
        self.assertTrue(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 4, 59)))

    def test_late_band_false(self):
        self.assertFalse(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 14, 0)))
        self.assertFalse(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 21, 59)))
        self.assertFalse(td._is_late_night_hour(datetime.datetime(2026, 1, 1, 5, 0)))

    def test_neutral_text_late_night_fallback(self):
        # A neutral utterance with no tone signal falls back to 'late_night'
        # ONLY when the clock is in the band. _is_late_night_hour() reads the
        # wall clock with no arg here, so patch it to make the branch deterministic.
        with mock.patch.object(td, "_is_late_night_hour", return_value=True):
            self.assertEqual(td.detect_tone("open the notes"), "late_night")

    def test_neutral_text_daytime_is_none(self):
        with mock.patch.object(td, "_is_late_night_hour", return_value=False):
            self.assertIsNone(td.detect_tone("open the notes"))


class AddendumTests(unittest.TestCase):
    def test_empty_for_none_or_unknown(self):
        self.assertEqual(td._tone_system_addendum(None), "")
        self.assertEqual(td._tone_system_addendum("not_a_real_tone"), "")

    def test_stressed_hint(self):
        out = td._tone_system_addendum("stressed")
        self.assertIn("USER_TONE: stressed", out)
        self.assertIn("[Per-turn tone hint]", out)


if __name__ == "__main__":
    unittest.main()
