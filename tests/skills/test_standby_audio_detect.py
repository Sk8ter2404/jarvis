"""Logic tests for skills/standby_audio_detect.py.

This skill adds music/lyric detection that finishes the auto-standby feature.
Tests cover the deterministic, numpy-backed core:
  • the pure-FFT spectral classifier (_classify_chunk) on synthesized tone vs
    noise vs near-silence,
  • the rhyme-density heuristic,
  • _looks_like_lyrics combining onset + rhyme + whisper markers,
  • should_refuse_wake gating (clear 'jarvis' vs lyric near-miss while music),
  • the feed_audio → sustained-music state machine and SUSTAINED_HOLD gate,
  • staleness reset,
  • music_state_summary phrasing,
  • the audio_music_status action.

All audio is synthesized in-memory; whisper / librosa / the background loop
are never invoked (the loop thread is neutered by the harness).
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


def _tone(freq=120.0, secs=1.0, sr=16000, amp=0.5):
    t = np.arange(int(secs * sr)) / sr
    # Bass tone + harmonic so flatness is low and bass-band energy is high.
    sig = amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t))
    return sig.astype(np.float32)


def _noise(secs=1.0, sr=16000, amp=0.5):
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(secs * sr))).astype(np.float32)


class SpectralClassifierTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    def test_classify_tone_as_musical(self):
        self.assertTrue(self.mod._classify_chunk(_tone(), 16000))

    def test_classify_white_noise_not_musical(self):
        # White noise is spectrally flat → not flagged as tonal/musical.
        self.assertFalse(self.mod._classify_chunk(_noise(), 16000))

    def test_classify_near_silence_not_musical(self):
        quiet = (np.zeros(16000) + 1e-4).astype(np.float32)
        self.assertFalse(self.mod._classify_chunk(quiet, 16000))

    def test_classify_too_short_not_musical(self):
        # < 0.25s of audio → bail out False.
        self.assertFalse(self.mod._classify_chunk(_tone(secs=0.1), 16000))

    def test_classify_handles_stereo(self):
        mono = _tone()
        stereo = np.stack([mono, mono], axis=1)
        self.assertTrue(self.mod._classify_chunk(stereo, 16000))


class LyricHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    # ── _rhyme_density ───────────────────────────────────────────────────
    def test_rhyme_density_high_for_rhyming_lines(self):
        # Each adjacent pair shares the last-2 suffix → density 1.0.
        text = "the cat sat splat that"
        self.assertGreater(self.mod._rhyme_density(text), 0.5)

    def test_rhyme_density_low_for_prose(self):
        text = "please schedule the quarterly budget review meeting"
        self.assertLess(self.mod._rhyme_density(text), 0.5)

    def test_rhyme_density_zero_for_short(self):
        self.assertEqual(self.mod._rhyme_density("hi there"), 0.0)
        self.assertEqual(self.mod._rhyme_density(""), 0.0)

    # ── _looks_like_lyrics ───────────────────────────────────────────────
    def test_looks_like_lyrics_whisper_marker(self):
        # An explicit [music] marker short-circuits to True.
        self.assertTrue(self.mod._looks_like_lyrics("[music] la la la", onset=0.0))

    def test_looks_like_lyrics_requires_onset_and_rhyme(self):
        # Rhyme-dense AND onset above the min → lyric.
        self.assertTrue(self.mod._looks_like_lyrics("cat sat splat that mat", onset=0.9))
        # Rhyme-dense but onset too low → not a lyric (could be a poem you typed).
        self.assertFalse(self.mod._looks_like_lyrics("cat sat splat that mat", onset=0.0))

    def test_looks_like_lyrics_empty_and_silent(self):
        self.assertFalse(self.mod._looks_like_lyrics("", onset=0.0))


class WakeGatingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    def test_no_refuse_when_no_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=False):
            self.assertFalse(self.mod.should_refuse_wake("some lyric words here"))

    def test_refuse_lyric_near_miss_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            # Long mid-sentence phrase while music plays → treated as lyric.
            self.assertTrue(self.mod.should_refuse_wake("and the radio kept playing on"))

    def test_allow_clear_wake_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            self.assertFalse(self.mod.should_refuse_wake("jarvis"))
            self.assertFalse(self.mod.should_refuse_wake("hey jarvis"))

    def test_refuse_empty_text_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            self.assertTrue(self.mod.should_refuse_wake(""))


class FeedAudioStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")
        # Reset rolling state between tests.
        self.mod._classifications.clear()
        self.mod._music_active[0] = False
        self.mod._music_since[0] = 0.0
        self.mod._last_feed_at[0] = 0.0
        self.mod._total_chunks_seen[0] = 0
        self.mod._total_music_chunks[0] = 0

    def test_sustained_music_activates_after_coverage(self):
        # Feed several musical chunks (each classifies True) past the 5s
        # coverage floor → _music_active flips on.
        tone = _tone(secs=1.0)
        for _ in range(7):
            self.mod.feed_audio(tone, 16000)
        self.assertTrue(self.mod._music_active[0])

    def test_is_music_playing_requires_sustained_hold(self):
        tone = _tone(secs=1.0)
        for _ in range(7):
            self.mod.feed_audio(tone, 16000)
        # Active, but _music_since is "now" → hold (15s) not yet satisfied.
        self.assertFalse(self.mod.is_music_currently_playing())
        # Backdate the activation past the hold window.
        self.mod._music_since[0] = self.mod._music_since[0] - (self.mod.SUSTAINED_HOLD_SECONDS + 1)
        self.assertTrue(self.mod.is_music_currently_playing())

    def test_noise_does_not_activate(self):
        for _ in range(7):
            self.mod.feed_audio(_noise(secs=1.0), 16000)
        self.assertFalse(self.mod._music_active[0])

    def test_stale_state_resets(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = 0.0  # ancient → > MUSIC_TIMEOUT_SECONDS ago
        self.assertFalse(self.mod.is_music_currently_playing())
        self.assertFalse(self.mod._music_active[0])

    def test_feed_none_is_safe(self):
        # None audio must not raise and must not change counters.
        before = self.mod._total_chunks_seen[0]
        self.mod.feed_audio(None, 16000)
        self.assertEqual(self.mod._total_chunks_seen[0], before)


class StatusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")
        self.mod._classifications.clear()
        self.mod._music_active[0] = False
        self.mod._total_chunks_seen[0] = 0
        self.mod._total_music_chunks[0] = 0

    def test_status_no_audio_yet(self):
        out = self.actions["audio_music_status"]("")
        self.assertIn("No audio analysed yet", out)

    def test_status_reports_chunk_ratio(self):
        self.mod._total_chunks_seen[0] = 10
        self.mod._total_music_chunks[0] = 3
        out = self.actions["audio_music_status"]("")
        self.assertIn("3 of 10", out)
        self.assertIn("30%", out)

    def test_status_active_music(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = self.mod.time.time()
        self.mod._music_since[0] = self.mod.time.time() - (self.mod.SUSTAINED_HOLD_SECONDS + 5)
        out = self.actions["audio_music_status"]("")
        self.assertIn("Wake-word filtering is active", out)


if __name__ == "__main__":
    unittest.main()
