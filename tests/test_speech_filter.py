"""Tests for core.speech_filter — the Whisper transcription gates extracted
from the monolith. These ran on every utterance with zero coverage before;
they pin the hallucination/confidence/length/always-accept gates and the
high-RMS confidence bypass."""
import unittest
from unittest import mock

import core.speech_filter as sf

NEUTRAL = {"no_speech_prob": 0.10, "avg_logprob": -0.30}
BAD = {"no_speech_prob": 0.95, "avg_logprob": -3.00}
# Whisper says "this IS speech" (low no_speech_prob) but is very unsure of the
# words (very negative avg_logprob) — isolates the avg_logprob gate.
LOW_LOGPROB = {"no_speech_prob": 0.10, "avg_logprob": -3.00}


class AmbientMusicTests(unittest.TestCase):
    def test_markers_detected(self):
        self.assertTrue(sf.is_ambient_music("[Music]"))
        self.assertTrue(sf.is_ambient_music("la la ♪ la"))
        self.assertTrue(sf.is_ambient_music("Music playing softly"))

    def test_plain_speech_not_music(self):
        self.assertFalse(sf.is_ambient_music("what time is it"))
        self.assertFalse(sf.is_ambient_music(""))


class ValidSpeechTests(unittest.TestCase):
    def test_empty_rejected(self):
        ok, reason = sf.is_valid_speech("", NEUTRAL)
        self.assertFalse(ok)
        self.assertEqual(reason, "empty")

    def test_always_accept_single_word(self):
        ok, _ = sf.is_valid_speech("yes", BAD)   # accepted before confidence gate
        self.assertTrue(ok)

    def test_hallucination_rejected(self):
        ok, reason = sf.is_valid_speech("thanks for watching", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("hallucination", reason)

    def test_too_short_chars(self):
        ok, reason = sf.is_valid_speech("a b", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("too short", reason)

    def test_valid_sentence(self):
        ok, _ = sf.is_valid_speech("what time is it", NEUTRAL)
        self.assertTrue(ok)

    def test_low_confidence_rejected(self):
        ok, reason = sf.is_valid_speech("some longer phrase here", BAD)
        self.assertFalse(ok)
        self.assertIn("no_speech_prob", reason)

    def test_high_rms_bypasses_confidence(self):
        # Loud, clearly-spoken audio is trusted even with bad Whisper scores.
        ok, _ = sf.is_valid_speech("turn it up", BAD, peak_rms=0.1)
        self.assertTrue(ok)

    def test_low_avg_logprob_rejected(self):
        # no_speech_prob passes its gate, but the avg_logprob confidence is
        # below WHISPER_MIN_AVG_LOGPROB → rejected with the logprob reason.
        ok, reason = sf.is_valid_speech("what time is it", LOW_LOGPROB)
        self.assertFalse(ok)
        self.assertIn("low confidence", reason)

    def test_single_long_word_too_few_words(self):
        # A single ≥4-char word that isn't an always-accept term clears the
        # char gate but trips the word-count gate (1 < WHISPER_MIN_WORDS) when
        # the audio wasn't loud enough to bypass it.
        ok, reason = sf.is_valid_speech("banana", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("1 words", reason)

    def test_missing_wake_word_rejected(self):
        # With a wake word configured, an utterance lacking it is dropped before
        # the confidence gates. Patched at module scope so the global is restored.
        with mock.patch.object(sf, "WAKE_WORD", "jarvis"):
            ok, reason = sf.is_valid_speech("what time is it", NEUTRAL)
        self.assertFalse(ok)
        self.assertIn("wake word", reason)

    def test_wake_word_present_passes(self):
        with mock.patch.object(sf, "WAKE_WORD", "jarvis"):
            ok, _ = sf.is_valid_speech("jarvis what time is it", NEUTRAL)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
