"""TV detector flapping (2026-10-15).

The live log showed the UNCALIBRATED detector (whole webcam view) flipping
TV ON / TV OFF 513 times in 162 minutes, median 10 s apart, with luma flat at
~68 and motion swinging 3-16: a person moving at the desk, not a TV. These
tests pin the fix in audio.tv_detect: a sustained whole-frame profile until a
TV rectangle is calibrated, a verdict dwell for both profiles, and the
fail-safe drop to OFF when the camera stops feeding (never held by the dwell).

Pure decider tests pass explicit timestamps (one reading a second); the skill
test reuses the tv_detect skill harness. stdlib unittest + mock.
"""
from __future__ import annotations

import unittest

from audio import tv_detect as td

LIT, MOVING = 68.0, 9.0          # what the live log showed while it flapped
STILL = 2.0


def _feed(decider, pattern, t0=0.0):
    """Feed one reading a second; `pattern` is a list of bools (moving?).
    Returns the number of verdict flips and the final time."""
    flips, last, t = 0, decider.is_on(now=t0), t0
    for moving in pattern:
        t += 1.0
        on = decider.observe(LIT, MOVING if moving else STILL, ts=t)
        if on != last:
            flips += 1
            last = on
    return flips, t


# A person at the desk: a few seconds of movement, then a few still ones
# (the live flips were ~10 s apart).
_PERSON = ([True] * 4 + [False] * 6) * 60           # 10 minutes


class WholeFrameProfile(unittest.TestCase):
    def test_a_person_moving_at_the_desk_never_reads_as_a_tv(self):
        flips, _t = _feed(td.decider_for(calibrated=False), _PERSON)
        self.assertEqual(flips, 0)

    def test_the_old_window_flapped_on_the_same_pattern(self):
        # Documents the bug: the 3-of-5 window with no dwell flips constantly.
        flips, _t = _feed(td.TVDecider(), _PERSON)
        self.assertGreater(flips, 100)

    def test_a_steady_playing_screen_still_turns_it_on(self):
        d = td.decider_for(calibrated=False)
        _flips, t = _feed(d, [True] * (td.WHOLE_FRAME_ON_FRAMES + 1))
        self.assertTrue(d.is_on(now=t))

    def test_profiles_differ_only_where_intended(self):
        whole, cal = td.decider_for(False), td.decider_for(True)
        self.assertEqual((whole.window, whole.on_frames, whole.off_frames),
                         (td.WHOLE_FRAME_WINDOW, td.WHOLE_FRAME_ON_FRAMES,
                          td.WHOLE_FRAME_OFF_FRAMES))
        self.assertGreaterEqual(whole.max_age_s, whole.window)
        self.assertEqual((cal.window, cal.on_frames, cal.off_frames),
                         (td.DECISION_WINDOW, td.ON_QUALIFYING_FRAMES,
                          td.OFF_QUALIFYING_FRAMES))
        self.assertEqual(whole.min_dwell_s, td.VERDICT_MIN_DWELL_S)
        self.assertEqual(cal.min_dwell_s, td.VERDICT_MIN_DWELL_S)


class VerdictDwell(unittest.TestCase):
    def _on_at(self, dwell):
        d = td.TVDecider(min_dwell_s=dwell)
        _f, t = _feed(d, [True] * (td.ON_QUALIFYING_FRAMES + 1))
        self.assertTrue(d.is_on(now=t))
        return d, t

    def test_the_verdict_holds_for_the_dwell_then_follows_the_evidence(self):
        d, t = self._on_at(20.0)
        on_since = d._changed_at
        while t + 1.0 < on_since + 20.0:
            t += 1.0
            d.observe(LIT, STILL, ts=t)
            self.assertTrue(d.is_on(now=t), f"dropped at +{t - on_since:.0f} s")
        for _ in range(3):
            t += 1.0
            d.observe(LIT, STILL, ts=t)
        self.assertFalse(d.is_on(now=t))

    def test_no_dwell_keeps_the_old_behaviour(self):
        d, t = self._on_at(0.0)
        for _ in range(td.DECISION_WINDOW):
            t += 1.0
            d.observe(LIT, STILL, ts=t)
        self.assertFalse(d.is_on(now=t))

    def test_losing_the_camera_drops_to_off_at_once(self):
        d, t = self._on_at(20.0)
        # No readings arrive; once they age out the verdict fails safe to OFF
        # even though the dwell has not expired.
        self.assertFalse(d.is_on(now=t + td.READING_MAX_AGE_S + 1.0))

    def test_reset_clears_the_dwell(self):
        d, t = self._on_at(20.0)
        d.reset()
        _f, t2 = _feed(d, [True] * (td.ON_QUALIFYING_FRAMES + 1), t0=t)
        self.assertTrue(d.is_on(now=t2))
        self.assertEqual(d.debug_state(t2)["min_dwell_s"], 20.0)


try:
    from tests.test_tv_detect import _SkillBase
except Exception:                      # pragma: no cover - numpy missing
    _SkillBase = None


@unittest.skipIf(_SkillBase is None, "tv_detect skill harness unavailable")
class SkillPicksTheProfile(_SkillBase or unittest.TestCase):
    def test_uncalibrated_skill_uses_the_whole_frame_profile(self):
        mod = self._load()
        self.assertEqual(mod._get_decider().window, td.WHOLE_FRAME_WINDOW)

    def test_calibrating_switches_to_the_fast_profile(self):
        mod = self._load()
        self._patch_flag(True)
        from tests.test_tv_detect import _bright_noisy_frame
        self._fake_bc({0: _bright_noisy_frame(seed=0)})
        self.assertEqual(mod._get_decider().window, td.WHOLE_FRAME_WINDOW)
        out = mod.calibrate_tv_region("")
        self.assertIn("Calibrated", out)
        self.assertEqual(mod._get_decider().window, td.DECISION_WINDOW)


if __name__ == "__main__":
    unittest.main()
