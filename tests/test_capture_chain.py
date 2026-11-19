"""core/capture_chain.py - the capture chain's pure rules (v3.0.26).

The VAD floor's start threshold (never under VAD_THRESHOLD, never over its
cap, only once the floor was measured), the clip level the auto-gain reads,
the soft limiter, and the numbers-only "[vad-floor]" summary line.

    python tools/run_tests.py capture_chain
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy absent => whole module skips
    np = None

if np is not None:
    from core import capture_chain as cc


def _sine(rms, n=1024):
    t = np.arange(n, dtype=np.float32) / 16000.0
    return (np.sin(2 * np.pi * 220.0 * t) * rms * np.sqrt(2.0)).astype(
        np.float32)


@unittest.skipIf(np is None, "numpy not installed")
class FloorModeTests(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(cc.floor_mode("ON"), "on")
        self.assertEqual(cc.floor_mode(" shadow "), "shadow")
        self.assertEqual(cc.floor_mode("off"), "off")
        self.assertEqual(cc.floor_mode("turbo"), "off")
        self.assertEqual(cc.floor_mode(None, default="on"), "on")


@unittest.skipIf(np is None, "numpy not installed")
class PeakChunkRmsTests(unittest.TestCase):
    def test_loudest_chunk_not_the_whole_clip(self):
        clip = np.concatenate([_sine(0.01), _sine(0.04), _sine(0.01)])
        self.assertAlmostEqual(cc.peak_chunk_rms(clip), 0.04, delta=5e-4)
        whole = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        self.assertLess(whole, 0.03)

    def test_tail_shorter_than_a_chunk_counts(self):
        clip = np.concatenate([_sine(0.01), _sine(0.05, n=300)])
        self.assertAlmostEqual(cc.peak_chunk_rms(clip), 0.05, delta=3e-3)

    def test_short_and_empty_and_bad(self):
        self.assertAlmostEqual(cc.peak_chunk_rms(_sine(0.02, n=500)), 0.02,
                               delta=2e-3)
        self.assertEqual(cc.peak_chunk_rms(np.zeros(0, np.float32)), 0.0)
        self.assertEqual(cc.peak_chunk_rms("not audio"), 0.0)


@unittest.skipIf(np is None, "numpy not installed")
class SoftLimitTests(unittest.TestCase):
    def test_under_the_knee_is_untouched(self):
        x = np.array([0.0, 0.5, -0.89, 0.9], dtype=np.float32)
        before = x.copy()
        out = cc.soft_limit(x)
        self.assertIs(out, x)
        np.testing.assert_array_equal(out, before)

    def test_over_the_knee_is_squeezed_in_order_and_inside_one(self):
        x = np.array([0.95, 1.2, 3.0, -1.5, -0.95], dtype=np.float32)
        out = cc.soft_limit(x.copy())
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        self.assertTrue(0.9 < out[0] < out[1] < out[2] <= 1.0)
        self.assertTrue(-1.0 <= out[3] < out[4] < -0.9)
        # no corner at the knee: just past it, the slope is ~1
        y = cc.soft_limit(np.array([0.901], dtype=np.float32))
        self.assertAlmostEqual(float(y[0]), 0.901, places=4)

    def test_never_raises(self):
        self.assertEqual(cc.soft_limit("x"), "x")


@unittest.skipIf(np is None, "numpy not installed")
class NoiseFloorTests(unittest.TestCase):
    BASE = 0.008          # the shipped VAD_THRESHOLD
    FACTOR = 2.5          # the shipped VAD_FLOOR_FACTOR
    RATIO = 1.25          # the shipped VAD_FLOOR_MAX_RATIO

    def _thr(self, nf):
        return nf.start_threshold(self.BASE, self.FACTOR, self.RATIO)

    def test_shipped_config_matches_these_numbers(self):
        from core import config
        self.assertEqual(config.VAD_THRESHOLD, self.BASE)
        self.assertEqual(config.VAD_FLOOR_FACTOR, self.FACTOR)
        self.assertEqual(config.VAD_FLOOR_MAX_RATIO, self.RATIO)
        self.assertEqual(config.VAD_FLOOR_MODE, "on")
        self.assertEqual(config.VAD_START_CONFIRM_CHUNKS, 1)

    def test_unmeasured_floor_is_the_base(self):
        nf = cc.NoiseFloor()
        for _ in range(cc.MIN_SAMPLES - 1):
            nf.note_idle(0.007)
        self.assertIsNone(nf.floor())
        self.assertEqual(self._thr(nf), self.BASE)

    def test_a_quiet_room_never_lowers_the_base(self):
        nf = cc.NoiseFloor()
        for _ in range(cc.MIN_SAMPLES):
            nf.note_idle(0.001)
        self.assertAlmostEqual(nf.floor(), 0.001)
        self.assertEqual(self._thr(nf), self.BASE)    # 2.5 x 0.001 < base

    def test_a_noisy_room_rises_to_the_cap_and_no_further(self):
        # The 11-17 17:05 room: idle chunks around 0.0045-0.007.
        nf = cc.NoiseFloor()
        for i in range(cc.MIN_SAMPLES * 2):
            nf.note_idle(0.0045 + 0.0025 * ((i % 10) / 10.0))
        self.assertAlmostEqual(self._thr(nf), self.BASE * self.RATIO)
        self.assertAlmostEqual(self.BASE * self.RATIO, 0.010)
        for _ in range(cc.FLOOR_WINDOW):
            nf.note_idle(0.05)        # music: the cap still holds
        self.assertAlmostEqual(self._thr(nf), 0.010)

    def test_in_between_follows_the_floor(self):
        nf = cc.NoiseFloor()
        for _ in range(cc.MIN_SAMPLES):
            nf.note_idle(0.0036)
        self.assertAlmostEqual(self._thr(nf), 0.009, places=6)

    def test_percentile_ignores_his_own_onsets(self):
        # 70 % of idle chunks at the floor, 30 % loud (sub-threshold onsets,
        # a keyboard): the 20th percentile stays on the floor.
        nf = cc.NoiseFloor()
        for i in range(cc.MIN_SAMPLES * 2):
            nf.note_idle(0.003 if i % 10 < 7 else 0.0079)
        self.assertAlmostEqual(nf.floor(), 0.003)

    def test_window_forgets_an_old_room(self):
        nf = cc.NoiseFloor()
        for _ in range(cc.FLOOR_WINDOW):
            nf.note_idle(0.006)
        self.assertAlmostEqual(self._thr(nf), 0.010)
        for _ in range(cc.FLOOR_WINDOW):
            nf.note_idle(0.001)
        self.assertEqual(self._thr(nf), self.BASE)

    def test_bad_values_are_ignored_and_reset_clears(self):
        nf = cc.NoiseFloor()
        for v in (float("nan"), -1.0, "x", None):
            nf.note_idle(v)
        self.assertEqual(len(nf), 0)
        for _ in range(cc.MIN_SAMPLES):
            nf.note_idle(0.006)
        nf.reset()
        self.assertIsNone(nf.floor())
        self.assertEqual(len(nf), 0)

    def test_owner_base_above_the_cap_is_kept(self):
        nf = cc.NoiseFloor()
        for _ in range(cc.MIN_SAMPLES):
            nf.note_idle(0.02)
        self.assertEqual(nf.start_threshold(0.03, 2.5, 1.0), 0.03)


@unittest.skipIf(np is None, "numpy not installed")
class CaptureBookTests(unittest.TestCase):
    def _line(self, book, now, mode="on", force=False):
        return book.line(now, mode=mode, floor=0.0041, start_thr=0.0100,
                         base=0.008, cap=0.010, confirm=1, force=force)

    def test_no_line_before_its_window_then_one_with_numbers(self):
        book = cc.CaptureBook(every_s=600.0)
        book.note_capture({"single": True}, 0.0)
        book.note_held(1.0)
        book.note_turn({"peak": 0.024, "start_thr": 0.01, "single": False}, 2.0)
        book.note_gain(2.0, False, 3.0)
        book.note_gain(10.0, True, 3.0)
        book.note_rescue(True, 4.0)
        self.assertIsNone(self._line(book, 599.0))
        line = self._line(book, 601.0)
        self.assertIn("[vad-floor] on: floor 0.0041 -> start 0.0100", line)
        self.assertIn("1 captures, 1 turns (quietest peak 2.4x its start "
                      "threshold)", line)
        self.assertIn("1 noise episodes held under the floor", line)
        self.assertIn("1 one-chunk trips (0 of them turns)", line)
        self.assertIn("1 of 2 at the cap", line)
        self.assertIn("empty rescues 0 run, 1 skipped (no speech)", line)
        # the window restarted, empty
        self.assertIsNone(self._line(book, 1300.0))

    def test_shadow_counts_what_on_would_have_lost(self):
        book = cc.CaptureBook(every_s=600.0)
        meta = {"would_hold": True, "peak": 0.0095, "start_thr": 0.008}
        book.note_capture(meta, 0.0)
        book.note_turn(meta, 1.0)
        line = self._line(book, 700.0, mode="shadow")
        self.assertIn("1 would not have started (1 of them turns)", line)
        self.assertNotIn("held under the floor", line)

    def test_force_and_nothing_counted(self):
        book = cc.CaptureBook()
        self.assertIsNone(self._line(book, 10.0, force=True))
        book.note_capture({}, 0.0)
        self.assertIsNotNone(self._line(book, 10.0, force=True))
        book.reset()
        self.assertIsNone(self._line(book, 10_000.0, force=True))


if __name__ == "__main__":
    unittest.main()
