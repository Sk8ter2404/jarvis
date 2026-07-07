#!/usr/bin/env python3
"""Unit tests for tools/calibrate_air_mouse.py — the air-mouse calibration wizard.

Exercises ONLY the PURE core (compute_calibration + merge_into_settings + the
small helpers) with synthetic samples and a temp settings file. NO Kinect, NO
display, NO stdin/stdout capture is needed: the live sensor path
(run_wizard / _hand_sample sampling) is import-guarded and never touched here, so
this runs headless on the Linux CI runner exactly as on the owner's box.

Asserted contract:
  * compute_calibration turns a resting-vs-raised height pair into engage/disengage
    margins with up > down (hysteresis) and BOTH margins between the resting and
    raised heights,
  * it CLAMPS absurd input (a raise below the rest, a giant/tiny/degenerate reach,
    an inverted reach) to safe physical ranges — never an inverted or broken gate,
  * it writes the EXACT persisted keys the live skill reads (KINECT_LIFT_* +
    KINECT_REACH_*),
  * merge_into_settings PRESERVES existing keys and adds the new ones, atomically,
  * a --dry-run wizard path writes NOTHING.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import calibrate_air_mouse as cal  # noqa: E402


class ComputeCalibrationTests(unittest.TestCase):
    """The pure fit: synthetic resting/raised heights + reach extremes in, a
    clamped margin/reach-box dict (keyed by the persisted setting names) out."""

    def test_margins_ordered_and_between_rest_and_raise(self):
        # Resting hand ~30 cm below the shoulder, raised ~15 cm above it.
        vals = cal.compute_calibration({
            "rest_lift": -0.30, "raise_lift": 0.15,
        })
        up = vals[cal.KEY_UP_MARGIN]
        down = vals[cal.KEY_DOWN_MARGIN]
        # Hysteresis: engage bar strictly above the disengage bar, by the floor.
        self.assertGreater(up, down)
        self.assertGreaterEqual(up - down, cal.MIN_HYSTERESIS - 1e-9)
        # Both margins fall BETWEEN the captured resting and raised heights, so a
        # real raise clears the up bar and a resting hand sits below the down bar.
        self.assertLess(-0.30, down)
        self.assertLess(down, up)
        self.assertLess(up, 0.15)

    def test_keys_match_what_the_skill_reads(self):
        # The height-gate keys MUST equal skills.kinect_air_mouse.SETTING_UP_MARGIN
        # / SETTING_DOWN_MARGIN so a calibration actually reaches the live gate.
        self.assertEqual(cal.KEY_UP_MARGIN, "KINECT_LIFT_UP_MARGIN")
        self.assertEqual(cal.KEY_DOWN_MARGIN, "KINECT_LIFT_DOWN_MARGIN")
        vals = cal.compute_calibration({"rest_lift": -0.25, "raise_lift": 0.10})
        for key in (cal.KEY_UP_MARGIN, cal.KEY_DOWN_MARGIN,
                    cal.KEY_REACH_HALF_W, cal.KEY_REACH_HALF_H,
                    cal.KEY_REACH_CENTER_X, cal.KEY_REACH_CENTER_Y):
            self.assertIn(key, vals)
            self.assertIsInstance(vals[key], float)

    def test_absurd_raise_below_rest_falls_back_to_safe_defaults(self):
        # A degenerate capture (the "raise" is LOWER than the rest) must NOT emit an
        # inverted margin — it falls back to the safe module defaults, still ordered.
        vals = cal.compute_calibration({"rest_lift": 0.20, "raise_lift": -0.40})
        up = vals[cal.KEY_UP_MARGIN]
        down = vals[cal.KEY_DOWN_MARGIN]
        self.assertGreater(up, down)
        self.assertGreaterEqual(up - down, cal.MIN_HYSTERESIS - 1e-9)
        self.assertEqual(round(up, 4), round(cal._DEFAULT_UP_MARGIN, 4))
        self.assertEqual(round(down, 4), round(cal._DEFAULT_DOWN_MARGIN, 4))

    def test_giant_reach_is_clamped(self):
        # A wild capture (arms flung far) must clamp to the physical ceiling, not
        # demand a room-wide swing.
        vals = cal.compute_calibration({
            "rest_lift": -0.30, "raise_lift": 0.15,
            "reach_min_x": -5.0, "reach_max_x": 5.0,   # 10 m span (absurd)
            "reach_min_y": -5.0, "reach_max_y": 5.0,
        })
        self.assertLessEqual(vals[cal.KEY_REACH_HALF_W], cal.REACH_HALF_W_MAX)
        self.assertLessEqual(vals[cal.KEY_REACH_HALF_H], cal.REACH_HALF_H_MAX)
        self.assertGreaterEqual(vals[cal.KEY_REACH_HALF_W], cal.REACH_HALF_W_MIN)

    def test_tiny_reach_falls_back_and_is_clamped(self):
        # A too-small span (barely moved) falls back to the default half-extent,
        # which is inside the clamp range.
        vals = cal.compute_calibration({
            "rest_lift": -0.30, "raise_lift": 0.15,
            "reach_min_x": 0.10, "reach_max_x": 0.11,   # 1 cm span (degenerate)
            "reach_min_y": 0.30, "reach_max_y": 0.305,
        })
        self.assertGreaterEqual(vals[cal.KEY_REACH_HALF_W], cal.REACH_HALF_W_MIN)
        self.assertLessEqual(vals[cal.KEY_REACH_HALF_W], cal.REACH_HALF_W_MAX)
        # Degenerate x span -> default half-width.
        self.assertEqual(round(vals[cal.KEY_REACH_HALF_W], 4),
                         round(cal._DEFAULT_REACH_HALF_W, 4))

    def test_reach_extremes_swapped_still_positive_extent(self):
        # If the two corners arrive swapped (max < min) the half-extent uses the
        # absolute span, so it's always positive and sensible.
        vals = cal.compute_calibration({
            "rest_lift": -0.30, "raise_lift": 0.15,
            "reach_min_x": 0.30, "reach_max_x": -0.20,   # swapped
            "reach_min_y": 0.10, "reach_max_y": 0.40,
        })
        self.assertGreater(vals[cal.KEY_REACH_HALF_W], 0.0)
        # centre is the midpoint regardless of order.
        self.assertAlmostEqual(vals[cal.KEY_REACH_CENTER_X], 0.05, places=3)

    def test_empty_samples_yields_defaults(self):
        # No captured samples at all -> all defaults, still a valid ordered gate.
        vals = cal.compute_calibration({})
        self.assertGreater(vals[cal.KEY_UP_MARGIN], vals[cal.KEY_DOWN_MARGIN])
        self.assertEqual(round(vals[cal.KEY_UP_MARGIN], 4),
                         round(cal._DEFAULT_UP_MARGIN, 4))
        self.assertEqual(round(vals[cal.KEY_REACH_CENTER_Y], 4),
                         round(cal._DEFAULT_REACH_CENTER_Y, 4))

    def test_none_argument_does_not_raise(self):
        vals = cal.compute_calibration(None)   # type: ignore[arg-type]
        self.assertIn(cal.KEY_UP_MARGIN, vals)


class MergeIntoSettingsTests(unittest.TestCase):
    """The atomic merge writer: preserve existing keys, add the new ones."""

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="calib_test_")
        self._path = os.path.join(self._dir, "user_settings.json")

    def tearDown(self):
        for name in os.listdir(self._dir):
            try:
                os.remove(os.path.join(self._dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self._dir)
        except OSError:
            pass

    def _read(self) -> dict:
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_preserves_existing_keys_and_adds_new(self):
        # Seed a settings file with unrelated owner keys.
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"SOME_OTHER_SETTING": 42,
                       "KINECT_AIR_MOUSE_ENABLED": True}, f)
        values = cal.compute_calibration({"rest_lift": -0.30, "raise_lift": 0.15})
        merged = cal.merge_into_settings(self._path, values)

        on_disk = self._read()
        # Pre-existing keys untouched.
        self.assertEqual(on_disk["SOME_OTHER_SETTING"], 42)
        self.assertEqual(on_disk["KINECT_AIR_MOUSE_ENABLED"], True)
        # New calibration keys present with the computed values.
        self.assertIn(cal.KEY_UP_MARGIN, on_disk)
        self.assertIn(cal.KEY_DOWN_MARGIN, on_disk)
        self.assertEqual(on_disk[cal.KEY_UP_MARGIN], values[cal.KEY_UP_MARGIN])
        # The returned dict matches what was written.
        self.assertEqual(merged[cal.KEY_UP_MARGIN], values[cal.KEY_UP_MARGIN])
        self.assertEqual(merged["SOME_OTHER_SETTING"], 42)

    def test_overwrites_only_calibration_keys(self):
        # A prior calibration is REPLACED (not duplicated) and other keys survive.
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({cal.KEY_UP_MARGIN: 0.99, "KEEP_ME": "yes"}, f)
        values = cal.compute_calibration({"rest_lift": -0.30, "raise_lift": 0.15})
        cal.merge_into_settings(self._path, values)
        on_disk = self._read()
        self.assertEqual(on_disk["KEEP_ME"], "yes")
        self.assertEqual(on_disk[cal.KEY_UP_MARGIN], values[cal.KEY_UP_MARGIN])
        self.assertNotEqual(on_disk[cal.KEY_UP_MARGIN], 0.99)

    def test_missing_file_is_created(self):
        self.assertFalse(os.path.exists(self._path))
        values = cal.compute_calibration({"rest_lift": -0.30, "raise_lift": 0.15})
        cal.merge_into_settings(self._path, values)
        self.assertTrue(os.path.exists(self._path))
        self.assertIn(cal.KEY_UP_MARGIN, self._read())

    def test_corrupt_file_is_treated_as_empty(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json ")
        values = cal.compute_calibration({"rest_lift": -0.30, "raise_lift": 0.15})
        # Must not raise; corrupt content is discarded and the new keys written.
        cal.merge_into_settings(self._path, values)
        on_disk = self._read()
        self.assertIn(cal.KEY_UP_MARGIN, on_disk)

    def test_no_temp_files_left_behind(self):
        values = cal.compute_calibration({"rest_lift": -0.30, "raise_lift": 0.15})
        cal.merge_into_settings(self._path, values)
        leftovers = [n for n in os.listdir(self._dir) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])


class DryRunTests(unittest.TestCase):
    """The --dry-run wizard path must compute + print but write NOTHING. Driven
    with a fake bridge + fake skill injected so no real Kinect is touched."""

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="calib_dry_")
        self._path = os.path.join(self._dir, "user_settings.json")

    def tearDown(self):
        for name in os.listdir(self._dir):
            try:
                os.remove(os.path.join(self._dir, name))
            except OSError:
                pass
        try:
            os.rmdir(self._dir)
        except OSError:
            pass

    def test_dry_run_writes_nothing(self):
        import io
        from unittest import mock

        # A fake bridge that reports ready, and a fake skill whose _hand_sample
        # yields a moving hand so every capture step gets frames.
        fake_kb = mock.Mock()
        fake_kb.get_enabled.return_value = True
        fake_kb.available.return_value = (True, "")

        class _Arm:
            def __init__(self, lift, hand):
                self.lift_m = lift
                self.hand = hand
            def reach_score(self):
                return self.lift_m if self.lift_m is not None else float("-inf")

        # Cycle a few positions so lift + xy both vary between steps.
        seq = [
            (_Arm(-0.30, (-0.20, 0.10, 1.0, 2)), None, "open", "open", True),
            (_Arm(0.15, (0.25, 0.40, 1.0, 2)), None, "open", "open", True),
        ]
        counter = {"i": 0}

        def _hand_sample(_bridge):
            item = seq[counter["i"] % len(seq)]
            counter["i"] += 1
            return item

        fake_skill = mock.Mock()
        fake_skill._hand_sample.side_effect = _hand_sample

        out = io.StringIO()
        # Instant sleep + a monotonically advancing clock so capture loops finish.
        clock = {"t": 0.0}

        def _now():
            clock["t"] += 0.5
            return clock["t"]

        with mock.patch.object(cal, "_bridge", return_value=fake_kb), \
             mock.patch.object(cal, "_skill", return_value=fake_skill):
            rc = cal.run_wizard(
                ["--dry-run", "--yes", "--path", self._path],
                inp=io.StringIO(""), out=out,
                sleep_fn=lambda _s: None, now_fn=_now)

        self.assertEqual(rc, 0)
        # The headline guarantee: --dry-run created NO settings file.
        self.assertFalse(os.path.exists(self._path),
                         "dry-run must not write the settings file")
        self.assertIn("dry-run", out.getvalue().lower())

    def test_show_path_prints_and_exits_without_sampling(self):
        import io
        from unittest import mock
        out = io.StringIO()
        with mock.patch.object(cal, "_bridge", return_value=None), \
             mock.patch.object(cal, "_skill", return_value=None):
            rc = cal.run_wizard(["--show-path", "--path", self._path],
                                inp=io.StringIO(""), out=out,
                                sleep_fn=lambda _s: None, now_fn=lambda: 0.0)
        self.assertEqual(rc, 0)
        self.assertIn(self._path, out.getvalue())


if __name__ == "__main__":
    unittest.main()
