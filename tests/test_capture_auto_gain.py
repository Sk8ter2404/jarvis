"""Unit tests for ``bobert_companion.apply_capture_auto_gain`` (TASK A).

The capture auto-gain helper is a PURE function over a float32 mono buffer and
its peak RMS: it CONSERVATIVELY boosts a too-quiet recording toward a usable
peak right before faster-whisper sees it, so a quiet mic can still wake JARVIS
with "JARVIS", WITHOUT degrading already-good/normal audio. These tests pin the
four contractual behaviours plus the guardrails:

  * quiet (noise_floor < peak < target)  → boosted, and the result stays
    hard-clipped inside [-1, 1];
  * normal/loud (peak >= target)         → returned UNCHANGED at gain 1.0;
  * silence / sub-noise-floor (peak <= floor) → UNCHANGED at gain 1.0 (never
    amplify room hiss into Whisper hallucinations);
  * auto-gain disabled                   → UNCHANGED at gain 1.0;
  * gain never exceeds CAPTURE_AUTO_GAIN_MAX;
  * a bad input never raises (returns the original at gain 1.0).

The helper lives on the ~15K-line monolith and only needs numpy + core.config,
so it runs in the LOCAL full-deps tier via the shared harness and @requires_-
monolith-skips on the light-deps CI runner.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import load_monolith, requires_monolith

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy absent => whole module skips
    np = None


@requires_monolith
class CaptureAutoGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()
        import core.config as _cfg
        cls.cfg = _cfg

    def _gain_fn(self):
        return self.bc.apply_capture_auto_gain

    # ── quiet input is boosted toward the target, and stays clipped ──────────
    def test_quiet_audio_is_boosted(self):
        # peak 0.02 is between the noise floor (0.005) and target (0.25):
        # expected gain = 0.25 / 0.02 = 12.5, capped at MAX_GAIN (10.0).
        peak = 0.02
        audio = np.full(2000, peak, dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertGreater(gain, 1.0)
        # Boost actually changed the samples.
        self.assertGreater(float(np.max(np.abs(out))), peak)
        # f32 preserved.
        self.assertEqual(out.dtype, np.float32)

    def test_boosted_audio_is_clipped_into_range(self):
        # A very quiet peak forces a large gain; samples that would exceed 1.0
        # after multiplication must be HARD-CLIPPED to [-1, 1].
        peak = 0.01
        # Mix of small and (relatively) large samples so the boost overshoots 1.0.
        audio = np.array([peak, -peak, 0.2, -0.2, 0.5, -0.5], dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertGreater(gain, 1.0)
        self.assertLessEqual(float(np.max(out)), 1.0)
        self.assertGreaterEqual(float(np.min(out)), -1.0)

    def test_gain_never_exceeds_max(self):
        # peak just above the floor would want a gigantic gain; it must cap at
        # CAPTURE_AUTO_GAIN_MAX.
        peak = 0.0051   # just above the 0.005 floor
        audio = np.full(100, peak, dtype=np.float32)
        _out, gain = self._gain_fn()(audio, peak)
        self.assertLessEqual(gain, float(self.cfg.CAPTURE_AUTO_GAIN_MAX) + 1e-6)

    # ── already-loud / normal audio is a NO-OP ──────────────────────────────
    def test_normal_audio_unchanged(self):
        # peak 0.3 >= target 0.25 → untouched, gain 1.0, same object semantics.
        peak = 0.3
        audio = np.array([0.3, -0.25, 0.1, -0.3], dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)   # returned unchanged (no copy/scale)

    def test_at_target_peak_unchanged(self):
        peak = float(self.cfg.CAPTURE_AUTO_GAIN_TARGET_PEAK)
        audio = np.full(50, peak, dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)

    # ── silence / sub-noise-floor is a NO-OP (no hiss amplification) ─────────
    def test_silence_below_noise_floor_unchanged(self):
        # peak below the noise floor (room hiss / silence) must NOT be amplified.
        peak = 0.001
        audio = np.full(100, peak, dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)

    def test_exact_noise_floor_unchanged(self):
        peak = float(self.cfg.CAPTURE_AUTO_GAIN_NOISE_FLOOR)
        audio = np.full(10, peak, dtype=np.float32)
        out, gain = self._gain_fn()(audio, peak)
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)

    # ── disabled switch is a NO-OP ──────────────────────────────────────────
    def test_disabled_is_noop(self):
        peak = 0.02   # would normally boost
        audio = np.full(100, peak, dtype=np.float32)
        with mock.patch.object(self.cfg, "CAPTURE_AUTO_GAIN_ENABLED", False):
            out, gain = self._gain_fn()(audio, peak)
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)

    # ── never raises ────────────────────────────────────────────────────────
    def test_none_audio_returns_none_no_raise(self):
        out, gain = self._gain_fn()(None, 0.02)
        self.assertIsNone(out)
        self.assertEqual(gain, 1.0)

    def test_bad_peak_does_not_raise(self):
        # A non-numeric peak must be swallowed → original audio at gain 1.0.
        audio = np.full(10, 0.02, dtype=np.float32)
        out, gain = self._gain_fn()(audio, "not-a-number")
        self.assertEqual(gain, 1.0)
        self.assertIs(out, audio)


if __name__ == "__main__":
    unittest.main()
