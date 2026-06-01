"""Tests for core.speech_filter — the Whisper transcription gates extracted
from the monolith. These ran on every utterance with zero coverage before;
they pin the hallucination/confidence/length/always-accept gates and the
high-RMS confidence bypass."""
import unittest

import core.speech_filter as sf

NEUTRAL = {"no_speech_prob": 0.10, "avg_logprob": -0.30}
BAD = {"no_speech_prob": 0.95, "avg_logprob": -3.00}


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


if __name__ == "__main__":
    unittest.main()
