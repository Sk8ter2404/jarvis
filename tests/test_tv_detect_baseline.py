"""TV_DETECT_ADAPTIVE_BASELINE (v3.0.10): the uncalibrated (whole-frame) TV
detector learns the room's idle motion (audio.tv_detect).

Live 2026-10-28/29 (numbers only): about 236 TV ON / OFF lines in 25 h with
no TV rectangle calibrated; 58 of 118 ON spells ended within 30 s; the OFF
readings sat at motion 3.1-3.9 just under the fixed 4.0 floor and the ON
readings at 4.2-6 (19:20:04 ON at luma 85, motion 4.8): a person moving at
the desk. Pure decider tests, one reading a second, explicit timestamps.

    python tools/run_tests.py tv_detect_baseline
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

from audio import tv_detect as td
from core import config

LUMA = 70.0


def _room(seconds=600, seed=3):
    """The live room: idle motion alternating 3.3 / 3.6, with a 20-30 s burst
    of 4.2-6.0 (someone moving at the desk) every ~90 s."""
    rng = random.Random(seed)
    out, t = [], 0
    while t < seconds:
        idle = rng.randint(55, 70)
        out += [3.3 if i % 2 else 3.6 for i in range(idle)]
        burst = rng.randint(20, 30)
        out += [rng.uniform(4.2, 6.0) for _ in range(burst)]
        t += idle + burst
    return out[:seconds]


def _feed(decider, motions, t0=0.0, luma=LUMA):
    flips, last, t = 0, decider.is_on(now=t0), t0
    for m in motions:
        t += 1.0
        on = decider.observe(luma, m, ts=t)
        if on != last:
            flips += 1
            last = on
    return flips, t


def _decider(adaptive, calibrated=False):
    with mock.patch.object(config, "TV_DETECT_ADAPTIVE_BASELINE", adaptive,
                           create=True):
        return td.decider_for(calibrated=calibrated)


class AdaptiveBaselineTests(unittest.TestCase):

    def test_people_moving_in_the_room_never_turn_it_on(self):
        d = _decider(True)
        self.assertTrue(d.adaptive_baseline)
        flips, t = _feed(d, _room())
        self.assertEqual(flips, 0)
        self.assertFalse(d.is_on(now=t))
        st = d.debug_state(t)
        self.assertGreater(st["motion_floor"], td.TEMPORAL_DELTA_ON_MIN)
        self.assertLess(st["motion_floor"], 8.0)

    def test_flag_off_the_same_room_flaps(self):
        # The live failure on the old fixed floor.
        d = _decider(False)
        self.assertFalse(d.adaptive_baseline)
        flips, _t = _feed(d, _room())
        self.assertGreater(flips, 4)
        self.assertNotIn("motion_floor", d.debug_state(0.0))

    def test_a_playing_screen_still_turns_it_on_after_learning(self):
        d = _decider(True)
        _f, t = _feed(d, _room(seconds=200))
        _f, t = _feed(d, [8.0 + (i % 5) * 3.0 for i in range(20)], t0=t)
        self.assertTrue(d.is_on(now=t))

    def test_a_playing_screen_turns_it_on_from_a_cold_start(self):
        d = _decider(True)
        _f, t = _feed(d, [9.0] * (td.WHOLE_FRAME_ON_FRAMES + 1))
        self.assertTrue(d.is_on(now=t))

    def test_hysteresis_keeps_it_on_through_a_quieter_scene(self):
        d = _decider(True)
        _f, t = _feed(d, [3.4] * 60)                    # learn: floor 6.4
        _f, t = _feed(d, [12.0] * 15, t0=t)
        self.assertTrue(d.is_on(now=t))
        # 5.5: under the ON floor (6.4), over the hold floor (4.9).
        self.assertAlmostEqual(d.delta_threshold(), 3.4 + 1.5, places=6)
        _f, t = _feed(d, [5.5] * 60, t0=t)
        self.assertTrue(d.is_on(now=t))
        _f, t = _feed(d, [3.4] * 40, t0=t)
        self.assertFalse(d.is_on(now=t))

    def test_the_floor_waits_for_enough_idle_readings(self):
        d = _decider(True)
        _feed(d, [3.4] * (td.BASELINE_MIN_READINGS - 1))
        self.assertEqual(d.delta_threshold(), td.TEMPORAL_DELTA_ON_MIN)
        _feed(d, [3.4], t0=100.0)
        self.assertAlmostEqual(d.delta_threshold(), 3.4 + td.BASELINE_MARGIN)

    def test_the_idle_history_is_bounded_and_reset_clears_it(self):
        d = _decider(True)
        _feed(d, [1.0] * (td.BASELINE_READINGS * 3))
        self.assertEqual(len(d._idle_motion), td.BASELINE_READINGS)
        d.reset()
        self.assertEqual(d._idle_motion, [])
        self.assertEqual(d.delta_threshold(), td.TEMPORAL_DELTA_ON_MIN)

    def test_a_calibrated_region_never_adapts(self):
        d = _decider(True, calibrated=True)
        self.assertFalse(d.adaptive_baseline)
        self.assertEqual(d.window, td.DECISION_WINDOW)


if __name__ == "__main__":
    unittest.main()
