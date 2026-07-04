"""Logic tests for core.emotion_tracker — the per-utterance emotion classifier.

Pins the documented priority order (frustrated > stressed > excited > tired >
focused > late-night-tired fallback), the word-choice / sentence-length /
prosody branches that select each label, and the addendum / TTS-preset accessor
contracts. The classifier is pure (no LLM, no model load); `hour` is passed
explicitly via ProsodyHints so the time-of-day fallback is deterministic and
the suite never depends on wall-clock time.

stdlib unittest only.
"""
from __future__ import annotations

import unittest

from core import emotion_tracker as et
from core.emotion_tracker import ProsodyHints


# A daytime hour keeps the late-night tired fallback inert for the text-only
# cases, so a "no emotion" utterance reliably classifies as None.
DAY = ProsodyHints(hour=14)


class NormalizeTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation_keeps_apostrophe(self):
        self.assertEqual(et._normalize("Don't STOP, now!!"), "don't stop now")

    def test_collapses_whitespace(self):
        self.assertEqual(et._normalize("a   b\t c"), "a b c")

    def test_empty(self):
        self.assertEqual(et._normalize(""), "")
        self.assertEqual(et._normalize("!!!"), "")


class EmptyInputTests(unittest.TestCase):
    def test_none_and_blank_return_empty_result(self):
        for bad in ("", "   ", "!!!", None):
            r = et.classify_emotion(bad, DAY)
            self.assertIsNone(r.label)
            self.assertFalse(bool(r))          # EmotionResult.__bool__
            self.assertEqual(r.addendum, "")
            self.assertIsNone(r.tts_preset)


class FrustratedTests(unittest.TestCase):
    def test_repetition_phrase(self):
        r = et.classify_emotion("I said open the file", DAY)
        self.assertEqual(r.label, "frustrated")

    def test_blame_phrase(self):
        self.assertEqual(et.classify_emotion("you keep getting it wrong", DAY).label,
                         "frustrated")

    def test_swear_plus_clipped_imperative(self):
        # Real profanity + a short imperative escalates to frustrated.
        self.assertEqual(et.classify_emotion("damn stop", DAY).label, "frustrated")

    def test_frustrated_beats_stressed(self):
        # Contains both a frustration phrase and stress vocab; frustrated wins.
        self.assertEqual(et.classify_emotion("I told you to stop", DAY).label,
                         "frustrated")


class StressedTests(unittest.TestCase):
    def test_stress_intensifier_word(self):
        self.assertEqual(et.classify_emotion("Stop it now!", DAY).label, "stressed")

    def test_two_exclamations(self):
        self.assertEqual(et.classify_emotion("Hurry up!!", DAY).label, "stressed")

    def test_clipped_with_single_exclamation(self):
        self.assertEqual(et.classify_emotion("Now!", DAY).label, "stressed")

    def test_prosody_loud_spike_with_exclamation(self):
        # No stress vocabulary, but a loud RMS spike + an exclamation → stressed.
        p = ProsodyHints(rms=0.20, rms_baseline=0.02, hour=14)
        self.assertEqual(et.classify_emotion("get over here!", p).label, "stressed")

    def test_prosody_fast_rate_with_exclamation(self):
        p = ProsodyHints(speech_rate_wps=4.5, hour=14)
        self.assertEqual(et.classify_emotion("come on lets move!", p).label, "stressed")


class ExcitedTests(unittest.TestCase):
    def test_positive_phrase(self):
        self.assertEqual(et.classify_emotion("This is amazing", DAY).label, "excited")

    def test_positive_phrase_with_exclamation(self):
        # A positive-vocabulary phrase reaches the excited branch. (Note: two
        # bare exclamations WITHOUT a positive phrase are caught by the higher-
        # priority stressed `excl>=2` rule — see StressedTests — so excited is
        # reached here via the word-choice signal, not the punctuation.)
        self.assertEqual(et.classify_emotion("This is brilliant!", DAY).label,
                         "excited")

    def test_two_exclamations_with_positive_phrase_is_excited(self):
        # v1.82.0: 2+ '!' PLUS positive vocabulary now reaches the excited
        # excl-branch. Previously the unconditional stressed `excl>=2` rule fired
        # first and classified this as stressed, leaving the excited excl-branch
        # unreachable dead code.
        self.assertEqual(et.classify_emotion("This is amazing!!", DAY).label,
                         "excited")

    def test_two_exclamations_without_positive_phrase_stays_stressed(self):
        # The other side of the guard: 2+ '!' with NO positive vocabulary must
        # still resolve to stressed (the excited gate requires a positive phrase).
        self.assertEqual(et.classify_emotion("Hurry up!!", DAY).label, "stressed")

    def test_loud_spike_plus_positive_phrase(self):
        p = ProsodyHints(rms=0.20, rms_baseline=0.02, hour=14)
        self.assertEqual(et.classify_emotion("awesome", p).label, "excited")

    def test_positive_word_with_swearing_is_not_excited(self):
        # 'nice' is excitement vocab, but 'damn' (stress) blocks the excited path.
        self.assertNotEqual(et.classify_emotion("damn nice", DAY).label, "excited")


class TiredTests(unittest.TestCase):
    def test_explicit_fatigue_phrase(self):
        self.assertEqual(et.classify_emotion("I'm exhausted, going to bed", DAY).label,
                         "tired")

    def test_prosody_slow_and_quiet(self):
        p = ProsodyHints(speech_rate_wps=1.2, rms=0.005, rms_baseline=0.02, hour=14)
        self.assertEqual(et.classify_emotion("alright then", p).label, "tired")

    def test_late_night_fallback_short_declarative(self):
        # No textual/prosodic signal, but it's 02:00 and the line is short.
        p = ProsodyHints(hour=2)
        r = et.classify_emotion("what's the time", p)
        self.assertEqual(r.label, "tired")
        self.assertIn("late_night", r.reason)

    def test_late_night_does_not_downgrade_long_focused_ask(self):
        # A long engineering ask at 2 AM stays focused, not tired.
        p = ProsodyHints(hour=2)
        r = et.classify_emotion("let's refactor the dispatcher in the module", p)
        self.assertEqual(r.label, "focused")


class FocusedTests(unittest.TestCase):
    def test_engineering_vocab_declarative(self):
        r = et.classify_emotion("Let's refactor the dispatcher in the module", DAY)
        self.assertEqual(r.label, "focused")

    def test_focused_needs_min_words(self):
        # Engineering word but <4 words → not focused (falls through to None).
        self.assertIsNone(et.classify_emotion("debug", DAY).label)

    def test_exclamation_blocks_focused(self):
        # Same vocab but with an exclamation is not the focused register.
        self.assertNotEqual(
            et.classify_emotion("fix the bug in the click handler!", DAY).label,
            "focused")

    def test_focused_vocab_with_profanity_is_not_focused(self):
        # Negative signal dominates the focused path: engineering vocab plus
        # profanity never reads as 'focused'. (Non-clipped profanity lands in
        # the stressed branch — frustrated additionally requires a clipped
        # imperative, see FrustratedTests.test_swear_plus_clipped_imperative.)
        r = et.classify_emotion("fix the damn bug in the click handler", DAY)
        self.assertNotEqual(r.label, "focused")
        self.assertEqual(r.label, "stressed")


class NeutralTests(unittest.TestCase):
    def test_plain_question_is_none(self):
        self.assertIsNone(et.classify_emotion("What's the weather today", DAY).label)

    def test_greeting_is_none(self):
        self.assertIsNone(et.classify_emotion("Hello sir how are you doing today",
                                              DAY).label)


class AccessorAndResultTests(unittest.TestCase):
    def test_tts_preset_mapping(self):
        self.assertEqual(et.tts_preset_for("stressed"), "calm")
        self.assertEqual(et.tts_preset_for("frustrated"), "calm")
        self.assertEqual(et.tts_preset_for("excited"), "amused")
        self.assertEqual(et.tts_preset_for("focused"), "briefing")
        self.assertEqual(et.tts_preset_for("tired"), "concerned")

    def test_tts_preset_none_for_unknown_or_none(self):
        self.assertIsNone(et.tts_preset_for(None))
        self.assertIsNone(et.tts_preset_for("euphoric"))

    def test_addendum_present_for_each_label(self):
        for label in ("stressed", "frustrated", "excited", "focused", "tired"):
            add = et.system_prompt_addendum(label)
            self.assertIn("Per-turn emotion hint", add)
            self.assertIn("USER_EMOTION", add)

    def test_addendum_empty_for_none_or_unknown(self):
        self.assertEqual(et.system_prompt_addendum(None), "")
        self.assertEqual(et.system_prompt_addendum("euphoric"), "")

    def test_result_carries_matching_addendum_and_preset(self):
        r = et.classify_emotion("Stop it now!", DAY)
        self.assertEqual(r.label, "stressed")
        self.assertEqual(r.tts_preset, "calm")
        self.assertEqual(r.addendum, et.system_prompt_addendum("stressed"))
        self.assertTrue(bool(r))


class ProsodyHelperTests(unittest.TestCase):
    def test_loud_spike_threshold(self):
        self.assertTrue(et._is_loud_spike(ProsodyHints(rms=0.05, rms_baseline=0.02)))
        self.assertFalse(et._is_loud_spike(ProsodyHints(rms=0.025, rms_baseline=0.02)))

    def test_loud_spike_requires_rms(self):
        self.assertFalse(et._is_loud_spike(ProsodyHints(rms=None)))

    def test_quiet_drop_threshold(self):
        self.assertTrue(et._is_quiet_drop(ProsodyHints(rms=0.005, rms_baseline=0.02)))
        self.assertFalse(et._is_quiet_drop(ProsodyHints(rms=0.02, rms_baseline=0.02)))

    def test_zero_baseline_uses_floor(self):
        # base <= 0 is replaced with 0.02 so we don't divide-by-zero / over-fire.
        self.assertFalse(et._is_loud_spike(ProsodyHints(rms=0.02, rms_baseline=0.0)))


class LateNightBandTests(unittest.TestCase):
    def test_band_wraps_midnight(self):
        self.assertTrue(et._in_late_night_band(23))
        self.assertTrue(et._in_late_night_band(2))
        self.assertFalse(et._in_late_night_band(12))
        self.assertFalse(et._in_late_night_band(21))   # just before the 22 start


if __name__ == "__main__":
    unittest.main()
