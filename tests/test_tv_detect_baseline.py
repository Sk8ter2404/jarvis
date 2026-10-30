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


_LOGGED_ON_MOTION = """
21.8 4.2 6.3 4.1 7.6 10.7 7.0 4.2 18.5 4.3 22.1 4.3
103.9 4.0 5.4 5.4 4.4 21.1 9.5 4.1 52.8 4.4 4.9 6.8
7.1 4.0 4.0 10.8 4.6 5.4 4.8 6.4 4.7 5.9 4.7 4.5
7.1 8.1 6.0 8.9 4.0 5.0 9.2 8.5 22.0 8.7 4.0 4.4
6.8 5.6 4.4 4.1 39.5 5.7 29.8 6.8 8.4 4.5 3.7 6.5
12.9 19.4 17.6 34.3 10.0 5.3 5.2 9.3 4.9 4.1 30.5 5.6
5.2 4.5 6.0 4.3 4.0 8.3 6.0 4.5 5.0 9.7 5.9 9.5
30.9 5.3 5.1 31.0 5.0 6.8 4.2 4.8 5.0 4.6 18.0 5.4
4.7 5.2 4.7 6.3 4.4 5.4 4.9 15.6 13.5 14.0 12.0 4.0
9.9 6.2 57.5 6.2 9.0 5.2 8.4 8.1 4.3 9.0 4.1 5.2
19.9 4.5
"""


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
        # Bursts of 4.2-6 only; harder bursts still flip it (see the replay
        # residual test below).
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

    def test_a_moderate_tv_still_turns_on_after_learning(self):
        # Review 10-29: learning every non-qualifying reading ratcheted the
        # floor to ~9.6 and a 4.5-9 TV never read ON.
        rng = random.Random(1)
        d = _decider(True)
        _f, t = _feed(d, [rng.uniform(3.1, 3.9) for _ in range(300)])
        on_s = 0
        for _ in range(1800):
            t += 1.0
            on_s += d.observe(LUMA, rng.uniform(4.5, 9.0), ts=t)
        self.assertGreater(on_s, 0.9 * 1800)
        self.assertLessEqual(d.delta_threshold(),
                             td.TEMPORAL_DELTA_ON_MIN + td.BASELINE_MARGIN)

    def test_a_tv_with_quiet_scenes_stays_on(self):
        rng = random.Random(1)
        d = _decider(True)
        _f, t = _feed(d, [rng.uniform(3.1, 3.9) for _ in range(300)])
        tv = []
        for _ in range(60):
            tv += [rng.uniform(4, 7) for _ in range(20)]
            tv += [rng.uniform(8, 25) for _ in range(10)]
        on_s = 0
        for m in tv:
            t += 1.0
            on_s += d.observe(LUMA, m, ts=t)
        # 89-98 % ON over seeds 1-7 (flag off ~99 %); the ratcheting floor
        # this replaces gave ~9 % and then OFF for good.
        self.assertGreater(on_s, 0.85 * len(tv))

    def test_the_floor_is_capped(self):
        d = _decider(True)
        d._idle_motion = [50.0] * td.BASELINE_READINGS
        self.assertEqual(d.delta_threshold(),
                         td.TEMPORAL_DELTA_ON_MIN + td.BASELINE_MARGIN)

    def test_replay_of_the_logged_on_motions_residual(self):
        # The 122 motion values logged with "TV ON" on 10-28 (numbers only),
        # each replayed as a 25 s burst (+-15 %) after 300 s of 3.1-3.9 idle.
        # Stated honestly: the adaptive floor cuts the flips about half
        # (205 -> 94 on this seed), NOT to zero - ~40 % of the live bursts
        # move harder than the learned floor. Calibrating the TV region is
        # the real cure.
        logged = [float(x) for x in _LOGGED_ON_MOTION.split()]
        self.assertEqual(len(logged), 122)
        rng = random.Random(1)
        seq = []
        for v in logged:
            seq += [rng.uniform(3.1, 3.9) for _ in range(300)]
            seq += [v * rng.uniform(0.85, 1.15) for _ in range(25)]
        off_flips, _t = _feed(_decider(False), seq)
        on_flips, _t = _feed(_decider(True), seq)
        self.assertGreater(off_flips, 150)
        self.assertLess(on_flips, 0.6 * off_flips)

    def test_a_calibrated_region_never_adapts(self):
        d = _decider(True, calibrated=True)
        self.assertFalse(d.adaptive_baseline)
        self.assertEqual(d.window, td.DECISION_WINDOW)


if __name__ == "__main__":
    unittest.main()
