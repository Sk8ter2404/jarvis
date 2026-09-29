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

    def test_frustrated_hint_asks_or_diagnoses_instead_of_guessing(self):
        out = td._tone_system_addendum("frustrated")
        self.assertIn("USER_TONE: frustrated", out)
        self.assertNotIn("Do NOT explain. Act.", out)
        self.assertIn("clarifying question", out)
        self.assertIn("diagnostic", out)


class ContextGatedStillTests(unittest.TestCase):
    """2026-09-29: 'still' was an urgency word, so a fault REPORT ("I'm still
    having USB issues") came out 'rushed', and 'frustrated' with any swear
    word attached. It now counts only after a failed turn or while the owner
    restates himself. The clock is pinned to daytime so a neutral result
    cannot flake to 'late_night' on a CI box running at a late UTC hour."""

    REPORT = "I'm still having USB issues"

    def setUp(self):
        p = mock.patch.object(td, "_is_late_night_hour", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_bare_still_report_is_neutral(self):
        self.assertIsNone(td.detect_tone(self.REPORT))

    def test_swear_plus_bare_still_is_not_frustrated(self):
        # Swearing alone is 'stressed'; 'still' no longer upgrades it.
        self.assertEqual(td.detect_tone("damn, I'm still having USB issues"),
                         "stressed")

    def test_still_after_a_failed_turn_is_frustrated(self):
        self.assertEqual(td.detect_tone(self.REPORT, prev_turn_failed=True),
                         "frustrated")

    def test_repeated_complaint_after_failed_turn_is_frustrated(self):
        self.assertEqual(
            td.detect_tone(self.REPORT, prev_user_text="I'm having USB issues",
                           prev_turn_failed=True),
            "frustrated")

    def test_other_triggers_are_unchanged(self):
        self.assertEqual(td.detect_tone("it's still not working"),
                         "frustrated")          # "still not" phrase, untouched
        self.assertEqual(td.detect_tone("I said turn it off"), "frustrated")
        self.assertEqual(td.detect_tone("do it now please"), "rushed")


class IsRestatementTests(unittest.TestCase):
    def test_majority_overlap_is_a_restatement(self):
        self.assertTrue(td.is_restatement("turn off the lights",
                                          "turn off the lights now"))

    def test_identical_line_is_not_a_restatement(self):
        # The caller's "previous" may be the current turn itself.
        self.assertFalse(td.is_restatement("turn off the lights",
                                           "Turn off the lights!"))

    def test_unrelated_or_missing_previous_is_not(self):
        self.assertFalse(td.is_restatement("open the notes",
                                           "what's the weather today"))
        self.assertFalse(td.is_restatement("open the notes", None))
        self.assertFalse(td.is_restatement("", "open the notes"))

    def test_unreadable_previous_is_not_and_does_not_raise(self):
        class Boom:
            def __str__(self):
                raise ValueError("cannot stringify")
        self.assertFalse(td.is_restatement("open the notes", Boom()))


if __name__ == "__main__":
    unittest.main()
