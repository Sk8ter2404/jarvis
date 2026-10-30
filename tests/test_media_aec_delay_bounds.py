"""MEDIA_AEC_DELAY_BOUNDS (v3.0.10): the media echo canceller's wall-clock
delay stays physical (core/audio_processor.MediaEchoCanceller).

Live 2026-10-29 (numbers only): the delay walked from the 40 ms prior to
-1485 ms in ~100 ms steps and the canceller sat in pass-through with ERLE
-0.6..4 dB. Each GCC search is relative to the CURRENT alignment, a lag "out
of the filter's reach" realigned at once on a single peak, and nothing
bounded the sum.

Synthetic and seeded (tests.test_media_aec's scene); no device. With the flag
off (delay_bounds=False) the old path runs exactly.

    python tools/run_tests.py media_aec_delay_bounds
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from core import audio_processor as ap
from core import config
from tests.test_media_aec import SR, FakeRef, make_scene, run


def _spurious(real, every=2, lag_s=-0.150, jitter_s=0.0, peak=0.06):
    """gcc_phat_lag that answers a spurious negative lag on every
    ``every``-th call (1 = always; a reflection / transient winning the PHAT
    peak), ``jitter_s`` apart from one spurious answer to the next."""
    calls = [0]

    def fake(x, d, max_lag, max_lead):
        calls[0] += 1
        if calls[0] % every == 0:
            j = jitter_s * (1 if calls[0] % 2 else -1)
            return float((lag_s + j) * SR), float(peak)
        return real(x, d, max_lag, max_lead)
    return fake


class DelayBoundsTests(unittest.TestCase):

    def _scene_run(self, bounds, **spur):
        ref, mic, echo, _near = make_scene(seconds=8.0, ppm=0.0,
                                           delay_ms=40.0, seed=11)
        aec = ap.MediaEchoCanceller(FakeRef(ref), delay_bounds=bounds)
        with mock.patch.object(ap, "gcc_phat_lag",
                               _spurious(ap.gcc_phat_lag, **spur)):
            lin, _sup = run(aec, mic)
        return aec.status()

    def test_single_spurious_negative_peaks_never_move_it(self):
        st = self._scene_run(True, every=2)
        self.assertAlmostEqual(st["delay_ms"], 40.0, delta=5.0)
        self.assertEqual(st["realigns"], 0)

    def test_flag_off_a_single_peak_moves_it_at_once(self):
        st = self._scene_run(False, every=2)
        self.assertGreater(st["realigns"], 3)
        self.assertNotIn("realign_rejected", st)

    def test_disagreeing_spurious_peaks_do_not_ratchet_it(self):
        # The live ratchet: peak after peak, each a few ms from the last.
        st = self._scene_run(True, every=1, jitter_s=0.005)
        self.assertAlmostEqual(st["delay_ms"], 40.0, delta=5.0)
        self.assertEqual(st["realigns"], 0)

    def test_flag_off_disagreeing_peaks_ratchet_it_far_negative(self):
        st = self._scene_run(False, every=1, jitter_s=0.005)
        self.assertLess(st["delay_ms"], -1000.0)

    def test_agreeing_spurious_peaks_stop_at_the_bound(self):
        st = self._scene_run(True, every=1)
        self.assertGreaterEqual(st["delay_ms"],
                                ap.MEDIA_AEC_DELAY_MIN_S * 1000.0)
        self.assertLessEqual(st["delay_ms"], ap.MEDIA_AEC_DELAY_MAX_S * 1000.0)
        self.assertGreaterEqual(st["realign_rejected"], 1)

    def test_an_out_of_bounds_realign_is_rejected_and_counted(self):
        aec = ap.MediaEchoCanceller(FakeRef(np.zeros(SR)), delay_bounds=True)
        before = aec.delay_wall
        aec._gcc_last = 123.0
        aec._realign(-0.5 * SR)
        self.assertEqual(aec.delay_wall, before)
        self.assertEqual(aec.stats["realign_rejected"], 1)
        self.assertEqual(aec.stats["realigns"], 0)
        self.assertIsNone(aec._gcc_last)
        aec._realign(0.1 * SR)                    # 140 ms: allowed
        self.assertAlmostEqual(aec.delay_wall, before + 0.1 * SR)
        self.assertEqual(aec.stats["realigns"], 1)

    def test_flag_off_realign_is_unbounded(self):
        aec = ap.MediaEchoCanceller(FakeRef(np.zeros(SR)), delay_bounds=False)
        before = aec.delay_wall
        aec._realign(-0.5 * SR)
        self.assertAlmostEqual(aec.delay_wall, before - 0.5 * SR)
        self.assertEqual(aec.stats["realigns"], 1)

    def _drift(self, bounds):
        aec = ap.MediaEchoCanceller(FakeRef(np.zeros(SR)), delay_bounds=bounds)
        aec.delay_wall = ap.MEDIA_AEC_DELAY_MAX_S * SR - 10.0
        aec._k = 100 * SR
        slope = -1.0e-4                            # +160 samples of delay
        aec._drift_pts = [(float(k), slope * k)
                          for k in range(0, 5 * SR, SR // 2)]
        aec._update_drift()
        self.assertEqual(aec.stats["drift_updates"], 1)
        return aec

    def test_a_drift_correction_is_clipped_into_the_bounds(self):
        aec = self._drift(True)
        self.assertAlmostEqual(aec.delay_wall, ap.MEDIA_AEC_DELAY_MAX_S * SR)
        self.assertEqual(aec.stats["delay_clamped"], 1)

    def test_flag_off_the_drift_correction_is_unclipped(self):
        aec = self._drift(False)
        self.assertGreater(aec.delay_wall, ap.MEDIA_AEC_DELAY_MAX_S * SR + 100)

    def test_the_normal_path_is_unchanged(self):
        # A 60 ms echo at 20 ppm never comes near the bounds: identical
        # output with the flag on and off.
        ref, mic, _echo, _near = make_scene(seconds=6.0, ppm=20.0,
                                            delay_ms=60.0, seed=7)
        outs = []
        for bounds in (True, False):
            aec = ap.MediaEchoCanceller(FakeRef(ref), delay_bounds=bounds)
            outs.append(run(aec, mic))
        self.assertTrue(np.array_equal(outs[0][0], outs[1][0]))
        self.assertTrue(np.array_equal(outs[0][1], outs[1][1]))

    def test_the_flag_is_read_from_config(self):
        with mock.patch.object(config, "MEDIA_AEC_DELAY_BOUNDS", True,
                               create=True):
            self.assertIsNotNone(ap.MediaEchoCanceller(None)._bounds)
        with mock.patch.object(config, "MEDIA_AEC_DELAY_BOUNDS", False,
                               create=True):
            self.assertIsNone(ap.MediaEchoCanceller(None)._bounds)


if __name__ == "__main__":
    unittest.main()
