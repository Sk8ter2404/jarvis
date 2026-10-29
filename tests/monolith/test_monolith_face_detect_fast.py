"""Monolith wiring for the cheaper at-desk face detector (2026-10-28).

Triage #401 (the camera feed lags: the face-track loop writes the HUD
preview between detections, and a 1280x720 detection with nobody in it
runs all four Haar passes, ~90 ms wall / ~1.2 CPU-s) and #414 (cut that
CPU). FACE_DETECT_FAST_MODE picks the frame _detect_face's passes run on:
"off" = full resolution (the old path), "shadow" = full resolution decides
and 1 in FACE_DETECT_SHADOW_EVERY_N detections also runs the downscaled
path, logging only agreement counts and timings, "on" = the downscaled
path decides. Default: shadow.

Every test drives the REAL _detect_face (and, once, the REAL face-track
producer). The cascades are faked (they record the grey frame size they
were handed); cv2's own image functions are real where the frame size
matters. No device is opened, no image is written. Patches use create=True
so a tree without the feature fails on the assertions, not a missing name.

    python -m unittest tests.monolith.test_monolith_face_detect_fast
"""
from __future__ import annotations

import contextlib
import io
import re
import threading
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


class _StubKinectBridge:
    """Never opens a sensor (same stub as the presence tests)."""

    class _Cap:
        _open_error = "kinect stubbed out for tests"

        def isOpened(self):   # noqa: N802
            return False

        def read(self):
            return False, None

        def get(self, _p):
            return 0.0

        def set(self, *_a, **_k):
            return False

        def release(self):
            pass

    def KinectCapture(self):   # noqa: N802
        return self._Cap()


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        kwargs.setdefault("create", True)
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        import numpy as np
        self.np = np
        bc = self.bc
        self._p(bc, "MIRROR_EYES_X", False)
        self._p(bc, "MIRROR_EYES_Y", False)
        self._p(bc, "FACE_DETECT_FAST_WIDTH", 640)
        self._p(bc, "FACE_DETECT_FAST_MIN_FRAC", 0.05)
        self._p(bc, "FACE_DETECT_SHADOW_EVERY_N", 1)
        self._p(bc, "FACE_DETECT_SHADOW_REPORT_N", 1000)
        # A fresh shadow counter per test (the module one is producer state).
        fast = getattr(bc, "_face_fast", None)
        if fast is not None:
            self._p(bc, "_face_detect_shadow", fast.ShadowStats())

    def _cascade(self, box_full=None, box_small=None, raise_small=False):
        """A fake frontal cascade. Records each call's grey (h, w); returns
        `box_full` on the 1280-wide frame and `box_small` on a narrower one."""
        np = self.np
        seen: list = []

        def _det(gray, **_kw):
            seen.append(tuple(gray.shape[:2]))
            small = gray.shape[1] < 1280
            if small and raise_small:
                raise RuntimeError("cheap path fault")
            box = box_small if small else box_full
            return np.empty((0, 4)) if box is None else np.array([box])
        cas = mock.Mock()
        cas.detectMultiScale.side_effect = _det
        return cas, seen

    def _frame(self):
        return self.np.zeros((720, 1280, 3), dtype=self.np.uint8)

    def _detect(self, cas, mode, frames=1):
        bc = self.bc
        self._p(bc, "FACE_DETECT_FAST_MODE", mode)
        self._p(bc, "_face_cascade", cas)
        self._p(bc, "_profile_cascade", None)
        buf = io.StringIO()
        outs = []
        with contextlib.redirect_stdout(buf):
            for _ in range(frames):
                outs.append(bc._detect_face(self._frame()))
        return outs, buf.getvalue()


class DetectFaceModeTests(_Base):
    def test_off_is_full_resolution_only(self):
        cas, seen = self._cascade(box_full=[400, 150, 300, 300])
        (out,), log = self._detect(cas, "off")
        self.assertEqual(set(seen), {(720, 1280)})
        self.assertAlmostEqual(out[0], 550 / 1280, places=4)
        self.assertNotIn("[face-detect shadow]", log)

    def test_on_runs_on_the_downscaled_frame_and_maps_back(self):
        cas, seen = self._cascade(box_small=[200, 75, 150, 150])
        (out,), _ = self._detect(cas, "on")
        self.assertEqual(set(seen), {(360, 640)},
                         "the 'on' path did not run on a 640-wide frame")
        fx, fy = out
        self.assertAlmostEqual(fx, (400 + 150) / 1280, places=4)
        self.assertAlmostEqual(fy, (150 + 150) / 720, places=4)
        info = self.bc._face_detect_last[0]
        self.assertEqual(info["pass"], "frontal")
        self.assertAlmostEqual(info["w_frac"], 300 / 1280, places=4)

    def test_on_search_floor_is_the_min_frac(self):
        cas, _ = self._cascade(box_small=[200, 75, 150, 150])
        self._detect(cas, "on")
        first = cas.detectMultiScale.call_args_list[0].kwargs
        self.assertEqual(first["minSize"], (32, 32))   # 5 % of 640

    def test_shadow_full_path_decides(self):
        # The cheap path finds nothing; the full path's face is what returns.
        cas, seen = self._cascade(box_full=[400, 150, 300, 300])
        (out,), _ = self._detect(cas, "shadow")
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out[0], 550 / 1280, places=4)
        self.assertIn((720, 1280), seen)
        self.assertIn((360, 640), seen, "shadow never ran the cheap path")
        self.assertEqual(self.bc._face_detect_last[0]["pass"], "frontal")

    def test_shadow_logs_agreement_numbers_only(self):
        self._p(self.bc, "FACE_DETECT_SHADOW_REPORT_N", 3)
        cas, _ = self._cascade(box_full=[400, 150, 300, 300])
        _, log = self._detect(cas, "shadow", frames=3)
        lines = [l for l in log.splitlines() if "[face-detect shadow]" in l]
        self.assertEqual(len(lines), 1, log)
        line = lines[0]
        self.assertIn("last 3: presence verdict same 0/3 (full-only 3, "
                      "cheap-only 0)", line)
        # Nothing but counts, milliseconds and the configured width / floor.
        self.assertNotRegex(line, r"\b(400|150|300)\b")
        self.assertNotIn("w_frac", line)
        self.assertRegex(line, r"median full [\d.]+ ms vs cheap@640px [\d.]+ ms")

    def test_shadow_samples_one_in_n(self):
        self._p(self.bc, "FACE_DETECT_SHADOW_EVERY_N", 4)
        cas, seen = self._cascade()
        self._detect(cas, "shadow", frames=8)
        small = [s for s in seen if s == (360, 640)]
        # 2 sampled frames x 2 frontal passes (strict + relaxed, no profile)
        self.assertEqual(len(small), 4, seen)
        self.assertEqual(self.bc._face_detect_shadow.window_size(), 2)

    def test_shadow_cheap_fault_never_reaches_the_result(self):
        cas, _ = self._cascade(box_full=[400, 150, 300, 300], raise_small=True)
        (out,), _ = self._detect(cas, "shadow")
        self.assertIsNotNone(out)
        self.assertEqual(self.bc._face_detect_shadow.total["errors"], 1)

    def test_unknown_mode_is_the_old_path(self):
        cas, seen = self._cascade()
        self._detect(cas, "turbo")
        self.assertEqual(set(seen), {(720, 1280)})

    def test_shipped_default_is_shadow_at_a_low_rate(self):
        from core import config
        self.assertEqual(getattr(config, "FACE_DETECT_FAST_MODE", None), "shadow")
        self.assertGreaterEqual(config.FACE_DETECT_SHADOW_EVERY_N, 10)
        self.assertLess(config.FACE_DETECT_FAST_WIDTH, 1280)
        self.assertLess(config.FACE_DETECT_FAST_MIN_FRAC, 0.07,
                        "the cheap floor must stay below the presence floor")

    def test_one_detector_body(self):
        # A stale second copy of the cascade passes in the monolith would
        # drift from the helper (both paths share core/face_detect_fast).
        import inspect
        src = inspect.getsource(self.bc)
        self.assertEqual(src.count(".detectMultiScale("), 0,
                         "cascade passes are duplicated in the monolith")


class FaceLoopShadowTests(_Base):
    """The REAL face-track producer in shadow mode prints its boot line and
    the agreement line; the presence it reports is the full path's."""

    CAM = {"index": 0, "label": "X", "name": "Test Webcam 0",
           "primary": True, "look_x": 0.5, "look_y": 0.5}

    def test_loop_logs_boot_and_shadow_lines(self):
        bc, np = self.bc, self.np
        import cv2 as real_cv2
        self._p(bc, "KINECT_AS_CAMERA", False)
        self._p(bc, "_kinect_bridge", _StubKinectBridge())
        self._p(bc, "_dshow_name_to_index", side_effect=lambda _n: 0)
        self._p(bc, "FACE_DETECT_FAST_MODE", "shadow")
        self._p(bc, "FACE_DETECT_SHADOW_REPORT_N", 3)
        for ev in (bc._face_track_stop, bc._face_track_pause,
                   bc._face_track_camera_off):
            ev.clear()
            self.addCleanup(ev.clear)
        cas, seen = self._cascade(box_full=[400, 150, 300, 300])
        a = np.zeros((720, 1280, 3), dtype=np.uint8)
        b = a.copy()
        b[::8, ::8] = 9
        frames = iter([a if i % 2 else b for i in range(40)])
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280
        cap.read.side_effect = lambda: (True, next(frames))
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        cv2.error = real_cv2.error
        for name in ("cvtColor", "resize", "equalizeHist", "flip"):
            setattr(cv2, name, getattr(real_cv2, name))
        for name in ("COLOR_BGR2GRAY", "INTER_AREA"):
            setattr(cv2, name, getattr(real_cv2, name))
        stop, n = bc._face_track_stop, [0]

        def _counted(*_a, **_k):
            n[0] += 1
            if n[0] >= 4:
                stop.set()

        err: dict = {}

        def _runner():
            try:
                bc._face_tracking_thread()
            except BaseException as exc:   # noqa: BLE001
                err["exc"] = exc

        buf = io.StringIO()
        with mock.patch.object(bc, "cv2", cv2), \
                mock.patch.object(bc, "CAMERAS", [dict(self.CAM)]), \
                mock.patch.object(bc, "_face_cascade", cas), \
                mock.patch.object(bc, "_profile_cascade", None), \
                mock.patch.object(bc, "_note_camera_read_attempt",
                                  side_effect=_counted), \
                mock.patch.object(bc, "_hud_camera_preview_enabled",
                                  return_value=False), \
                mock.patch.object(bc, "find_camera_locking_processes",
                                  return_value=[]), \
                mock.patch.object(bc, "send"), \
                mock.patch.object(bc.time, "sleep"), \
                contextlib.redirect_stdout(buf):
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(8.0)
            if t.is_alive():
                stop.set()
                t.join(10.0)
                self.fail("face-track loop did not stop")
        if "exc" in err:
            raise err["exc"]
        log = buf.getvalue()
        self.assertIn("[cpu] face-detect path: full resolution decides; "
                      "shadow runs the cheap path (640 px wide", log)
        self.assertRegex(log, re.escape("[face-detect shadow] last 3: "
                                        "presence verdict same 0/3"))
        self.assertIn((360, 640), seen)


if __name__ == "__main__":
    unittest.main()
