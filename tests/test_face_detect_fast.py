"""core/face_detect_fast - the cheaper at-desk face detector path (2026-10-28).

The pure parts (mode parsing, the minSize arithmetic, the shadow counters,
the boot line) run on the light-deps runner. The detector parts need cv2 +
numpy and skip without them; the benchmark runs the REAL Haar cascades on a
synthetic noise frame (the worst case: nothing in it, so all four passes run)
and reports wall time and process CPU per frame for both paths. No image is
read from or written to disk anywhere in this file.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import face_detect_fast as fdf  # noqa: E402

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV2 = True
except Exception:  # pragma: no cover - light-deps runner
    HAVE_CV2 = False


class ModeTests(unittest.TestCase):
    def test_three_modes(self):
        for m in ("off", "shadow", "on"):
            self.assertEqual(fdf.normalise_mode(m), m)
        self.assertEqual(fdf.normalise_mode(" Shadow "), "shadow")

    def test_junk_is_today(self):
        for v in (None, "", "fast", 3, "maybe"):
            self.assertEqual(fdf.normalise_mode(v), "off", v)

    def test_true_means_on(self):
        self.assertEqual(fdf.normalise_mode(True), "on")


class MinSizeTests(unittest.TestCase):
    def _calls(self, scale, floor):
        cas = mock.Mock()
        cas.detectMultiScale.return_value = []
        cv = mock.Mock()
        gray = mock.Mock()
        gray.shape = (360, 640)
        fdf.cascade_passes(cv, gray, cas, cas, scale=scale, floor_px=floor)
        return [c.kwargs for c in cas.detectMultiScale.call_args_list]

    def test_full_resolution_calls_are_the_historic_ones(self):
        kws = self._calls(1.0, 0)
        self.assertEqual(
            [(k["scaleFactor"], k["minNeighbors"], k["minSize"]) for k in kws],
            [(1.05, 4, (40, 40)), (1.05, 3, (60, 60)),
             (1.05, 4, (40, 40)), (1.05, 4, (40, 40))])

    def test_half_scale_with_floor(self):
        kws = self._calls(0.5, 32)
        self.assertEqual([k["minSize"] for k in kws],
                         [(32, 32), (32, 32), (32, 32), (32, 32)])
        kws = self._calls(0.5, 0)
        self.assertEqual([k["minSize"] for k in kws],
                         [(20, 20), (30, 30), (20, 20), (20, 20)])


class ShadowStatsTests(unittest.TestCase):
    def test_due_one_in_n(self):
        st = fdf.ShadowStats()
        got = [st.due(4) for _ in range(9)]
        self.assertEqual(got, [True, False, False, False,
                               True, False, False, False, True])
        self.assertFalse(fdf.ShadowStats().due(0))
        self.assertFalse(fdf.ShadowStats().due("x"))

    def test_report_counts_and_resets_window(self):
        st = fdf.ShadowStats()
        st.record((True, True), (True, True), 90.0, 30.0)
        st.record((True, True), (False, False), 80.0, 20.0)
        st.record((False, False), (True, True), 100.0, 40.0)
        st.record((True, False), (False, False), 70.0, 10.0)
        st.note_error()
        self.assertEqual(st.window_size(), 4)
        line = st.report(640, 0.05)
        self.assertTrue(line.startswith("[face-detect shadow] last 4:"), line)
        self.assertIn("presence verdict same 2/4", line)
        self.assertIn("full-only 1, cheap-only 1", line)
        self.assertIn("any-face same 1/4", line)
        self.assertIn("median full 85.0 ms vs cheap@640px 25.0 ms", line)
        self.assertIn("errors 1", line)
        self.assertEqual(st.window_size(), 0)
        st.record((True, True), (True, True), 50.0, 5.0)
        line = st.report(640, 0.05)
        self.assertIn("last 1:", line)
        self.assertIn("total 5:", line)

    def test_timing_lists_stay_bounded(self):
        st = fdf.ShadowStats()
        for _ in range(5000):
            st.record((False, False), (False, False), 1.0, 1.0)
        self.assertLessEqual(len(st.total["full_ms"]), 2000)


class VerdictTests(unittest.TestCase):
    def test_presence_rule_is_the_gates(self):
        self.assertEqual(fdf.verdicts(None, None, (1280, 720)), (False, False))
        self.assertEqual(fdf.verdicts((0, 0, 300, 300), "frontal", (1280, 720)),
                         (True, True))
        # a relaxed-pass hit steers the eyes but is not presence
        self.assertEqual(
            fdf.verdicts((0, 0, 300, 300), "frontal_relaxed", (1280, 720)),
            (True, False))
        # a face across the room is not presence
        self.assertEqual(fdf.verdicts((0, 0, 60, 60), "frontal", (1280, 720)),
                         (True, False))


class PolicyLineTests(unittest.TestCase):
    def test_lines(self):
        self.assertIn("full resolution only",
                      fdf.policy_line("off", 640, 0.05, 20, 30))
        self.assertIn("CHEAP (640 px wide",
                      fdf.policy_line("on", 640, 0.05, 20, 30))
        s = fdf.policy_line("shadow", 640, 0.05, 20, 30)
        self.assertIn("1 in 20 detections", s)
        self.assertIn("every 30 samples", s)


@unittest.skipUnless(HAVE_CV2, "cv2 / numpy absent")
class DetectTests(unittest.TestCase):
    def _cascade(self, box_at_640=None):
        seen = []

        def _det(gray, **kw):
            seen.append((gray.shape, kw["minSize"]))
            if box_at_640 is None:
                return np.empty((0, 4))
            return np.array([box_at_640])
        cas = mock.Mock()
        cas.detectMultiScale.side_effect = _det
        return cas, seen

    def test_full_path_runs_at_full_resolution(self):
        cas, seen = self._cascade()
        frame = np.zeros((720, 1280, 3), np.uint8)
        box, by, size = fdf.detect(cv2, frame, cas, None)
        self.assertIsNone(box)
        self.assertEqual(size, (1280, 720))
        self.assertEqual(seen[0], ((720, 1280), (40, 40)))

    def test_cheap_path_shrinks_and_maps_back(self):
        cas, seen = self._cascade([100, 50, 150, 150])
        frame = np.zeros((720, 1280, 3), np.uint8)
        box, by, size = fdf.detect(cv2, frame, cas, None, width=640,
                                   min_frac=0.05)
        self.assertEqual(seen[0], ((360, 640), (32, 32)))
        self.assertEqual(by, "frontal")
        self.assertEqual(tuple(round(v) for v in box), (200, 100, 300, 300))

    def test_narrow_frame_is_not_upscaled(self):
        cas, seen = self._cascade()
        fdf.detect(cv2, np.zeros((480, 640, 3), np.uint8), cas, None,
                   width=640, min_frac=0.05)
        self.assertEqual(seen[0], ((480, 640), (40, 40)))


@unittest.skipUnless(HAVE_CV2, "cv2 / numpy absent")
class BenchmarkTests(unittest.TestCase):
    """CPU and time per frame, full vs cheap, real cascades, synthetic frame."""

    def test_cheap_path_costs_less(self):
        frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")
        if frontal.empty() or profile.empty():
            self.skipTest("bundled cascades missing")
        rng = np.random.default_rng(7)
        # Dim sensor noise: the night frame that makes every pass run.
        frame = rng.integers(0, 40, (720, 1280, 3), dtype=np.uint8)

        def bench(width, reps=3):
            fdf.detect(cv2, frame, frontal, profile, width=width,
                       min_frac=0.05)   # warm-up
            walls, cpus = [], []
            for _ in range(reps):
                t, c = time.perf_counter(), time.process_time()
                fdf.detect(cv2, frame, frontal, profile, width=width,
                           min_frac=0.05)
                walls.append((time.perf_counter() - t) * 1000.0)
                cpus.append((time.process_time() - c) * 1000.0)
            return sorted(walls)[reps // 2], sorted(cpus)[reps // 2]

        full_wall, full_cpu = bench(0)
        cheap_wall, cheap_cpu = bench(640)
        sys.stderr.write(
            f"\n[bench] 1280x720 noise frame: full {full_wall:.1f} ms wall / "
            f"{full_cpu:.0f} ms CPU; cheap@640 {cheap_wall:.1f} ms wall / "
            f"{cheap_cpu:.0f} ms CPU\n")
        self.assertLess(cheap_wall, full_wall)
        self.assertLess(cheap_cpu, full_cpu)


if __name__ == "__main__":
    unittest.main()
