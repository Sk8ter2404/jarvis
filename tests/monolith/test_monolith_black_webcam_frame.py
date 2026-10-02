"""Audit P2-3 (2026-10-01): an all-black webcam frame is not a live frame.

When the Kinect saturates the USB controller a webcam can keep "succeeding" -
cap.read() returns True - while every frame it hands back is black. The
face-track producer cached that frame and stamped ``_camera_last_frame_at``
like any other, so camera_status said the webcam was live and look_around /
face enrolment (both behind ``_fresh_camera_frame``) used a black picture as
the current scene.

These tests drive the REAL producer body for one iteration (the harness from
test_monolith_camera_preview_keepalive.py: no real camera, no DirectShow
enumeration) and the REAL camera_status skill against what it wrote.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated
from tests.monolith import test_monolith_camera_preview_keepalive as _ka

_CAMS = [{"index": 0, "label": "Right webcam (top of right monitor)",
          "name": "usb 2.0 camera", "primary": True,
          "look_x": 0.85, "look_y": 0.5}]


class _ProducerHarness(MonolithGlobalsTestCase):
    """setUp + one REAL producer iteration; no tests of its own."""

    def setUp(self):
        import numpy as np
        self.np = np
        bc = self.bc
        # The frame caches are not in the harness's restore list: isolate them.
        for d in (bc._camera_latest_frame, bc._camera_last_frame_at,
                  bc._camera_last_read_error, bc._camera_last_read_error_at,
                  getattr(bc, "_camera_black_frame_at", {}),
                  getattr(bc, "_camera_latest_black_frame", {}),
                  getattr(bc, "_camera_black_warned_at", {}),
                  getattr(bc, "_camera_black_logged_run", {})):
            p = mock.patch.dict(d, clear=True)
            p.start()
            self.addCleanup(p.stop)
        bc._face_track_stop.clear()
        bc._face_track_pause.clear()
        bc._face_track_camera_off.clear()
        bc._standby_mode[0] = False
        self.black = np.zeros((72, 128, 3), dtype=np.uint8)
        self.lit = np.full((72, 128, 3), 90, dtype=np.uint8)

    def _run_one_iteration(self, frame):
        bc = self.bc
        gate = bc._make_camera_gate()
        gate.stagger_s = 0.0
        cap = _ka._LoopCap(frame)
        with mock.patch.object(bc, "CAMERAS", _CAMS), \
             mock.patch.object(bc, "_camera_gate", gate), \
             mock.patch.object(bc, "_face_track_stop", _ka._OneShotStop()), \
             mock.patch.object(bc, "_dshow_name_to_index", return_value=0), \
             mock.patch.object(bc, "_open_capture_bounded",
                               side_effect=lambda idx, opener, *a, **k: cap), \
             mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=False), \
             mock.patch.object(bc, "_detect_face", return_value=None), \
             mock.patch.object(bc, "send"):
            bc._face_tracking_thread_body()
        self.assertGreater(cap.reads, 0, "setup wrong: the camera was never read")


@requires_monolith
class BlackWebcamFrameTests(_ProducerHarness):

    def test_a_lit_frame_is_cached_as_live(self):
        """Control, so the black case below cannot pass for the wrong reason."""
        self._run_one_iteration(self.lit)
        self.assertIn(0, self.bc._camera_latest_frame)
        fr, ts = self.bc._fresh_camera_frame(0)
        self.assertIsNotNone(fr)
        self.assertNotIn(0, getattr(self.bc, "_camera_black_frame_at", {}))

    def test_a_black_frame_is_not_cached_or_stamped(self):
        self._run_one_iteration(self.black)
        bc = self.bc
        self.assertNotIn(0, bc._camera_latest_frame,
                         "a black frame was cached as the camera's live picture")
        self.assertNotIn(0, bc._camera_last_frame_at,
                         "a black frame stamped the camera as live")
        self.assertEqual(bc._fresh_camera_frame(0), (None, None))
        self.assertIn(0, bc._camera_black_frame_at)
        self.assertIn("black frames", bc._camera_last_read_error.get(0, ""))
        self.assertIn("black_frame_at", bc.get_camera_health()[0])

    def test_a_real_frame_after_black_ones_clears_the_state(self):
        self._run_one_iteration(self.black)
        self._run_one_iteration(self.lit)
        bc = self.bc
        self.assertIn(0, bc._camera_latest_frame)
        self.assertNotIn(0, bc._camera_black_frame_at)
        self.assertIsNone(bc._camera_last_read_error.get(0))

    def test_a_dim_room_with_a_lit_face_is_live(self):
        """A monitor-lit face on a dark wall: the WHOLE-frame mean is under the
        black level (about 6/255), yet the picture is perfectly usable. The
        P2-5 note on _frame_brightness_for_dark_check is about exactly this
        frame; the black test must not drop it (2026-10-02 review)."""
        frame = self.np.full((72, 128, 3), 2, dtype=self.np.uint8)
        frame[29:44, 52:77] = 120                       # the face, ~4% of it
        self.assertLess(float(frame.mean()), 10.0, "setup: not a dim frame")
        self._run_one_iteration(frame)
        bc = self.bc
        self.assertNotIn(0, bc._camera_black_frame_at,
                         "a dim but usable picture was called black")
        fr, _ts = bc._fresh_camera_frame(0)
        self.assertIsNotNone(fr)

    def test_a_bright_corner_alone_keeps_the_frame_live(self):
        # A lamp in the corner of an otherwise dark room: still a picture.
        frame = self.np.full((72, 128, 3), 1, dtype=self.np.uint8)
        frame[0:20, 0:40] = 90                          # ~9% of the frame
        self._run_one_iteration(frame)
        self.assertNotIn(0, self.bc._camera_black_frame_at)

    def test_an_unmeasurable_frame_is_not_black(self):
        """_webcam_frame_brightness promises None - never 'black' - when it
        cannot measure a frame; a helper that returns 0.0 on its own error must
        not turn that error into a black reading."""
        weird = self.np.array([[["x"] * 3] * 8] * 8, dtype=object)
        self.assertIsNone(self.bc._webcam_frame_brightness(weird))

    def test_camera_status_says_black_frames_not_live(self):
        """The spoken surface: what the producer wrote, read by the real skill."""
        self._run_one_iteration(self.black)
        mod, actions = load_skill_isolated("camera_system", register=True)
        mod._bc = lambda: self.bc
        mod._cfg_flag = lambda name, default=False: False   # Kinect off
        import core.config as _real_cfg
        with mock.patch.object(_real_cfg, "CAMERAS", _CAMS):
            out = actions["camera_status"]("")
        self.assertIn("delivering black frames", out)
        self.assertNotIn("is live", out)


@requires_monolith
class BlackRunLogTests(_ProducerHarness):
    """A dark room overnight logged one line a minute per camera, all night
    (about 1,000 lines with two webcams, 2026-10-02 review). The log says when
    a run starts and ends, with a long-interval reminder in between."""

    def _lines(self, fn):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        return [ln for ln in buf.getvalue().splitlines() if "black frames" in ln
                or "real frames again" in ln]

    def test_an_hour_of_black_frames_is_a_few_lines_not_sixty(self):
        bc = self.bc

        def _hour():
            for t in range(0, 3601):                 # one frame a second
                bc._warn_black_camera_frames("Right webcam", 0, 2.0,
                                             1_000_000.0 + t)
        lines = self._lines(_hour)
        self.assertGreaterEqual(len(lines), 1, "the run start was not logged")
        self.assertLessEqual(len(lines), 3, lines)

    def test_the_end_of_a_logged_run_is_logged_once(self):
        def _run():
            self._run_one_iteration(self.black)      # run starts: logged
            self._run_one_iteration(self.lit)        # run ends: logged
            self._run_one_iteration(self.lit)        # nothing new
        lines = self._lines(_run)
        self.assertEqual(len([ln for ln in lines if "black frames" in ln]), 1,
                         lines)
        self.assertEqual(len([ln for ln in lines if "real frames again" in ln]),
                         1, lines)

    def test_a_flickering_camera_cannot_flood_either(self):
        def _flicker():
            for _ in range(4):
                self._run_one_iteration(self.black)
                self._run_one_iteration(self.lit)
        lines = self._lines(_flicker)
        self.assertLessEqual(len(lines), 2, lines)


@requires_monolith
class SelfDiagnosticSeesTheBlackRunTests(_ProducerHarness):
    """The self-diagnostic's webcam probe judges the PRODUCER's frames when the
    producer owns the camera (skills/self_diagnostic._producer_latest_frame).
    Once black frames stopped being cached and stamped, it judged the last LIT
    frame for 10 s (PASS) and then found no fresh camera at all, so its
    "the webcam is producing only black frames" finding could never fire
    (2026-10-02 review). Driven by the REAL producer state, not a fake that
    holds a fresh black frame the producer can no longer produce."""

    def setUp(self):
        super().setUp()
        self.diag, _ = load_skill_isolated("self_diagnostic", register=False)

    def _probe(self):
        import sys
        import time
        import types
        bc = self.bc
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.data = types.SimpleNamespace(haarcascades="/cascades/")
        opened = []
        with mock.patch.dict(sys.modules, {"bobert_companion": bc,
                                           "cv2": cv2_stub}), \
             mock.patch.object(bc, "CAMERAS", _CAMS), \
             mock.patch.object(bc, "get_face_track_liveness",
                               return_value={"at": time.time(),
                                             "stage": "loop top"}), \
             mock.patch.object(self.diag, "_camera_backend_name",
                               return_value="dshow"), \
             mock.patch.object(self.diag, "_open_probe_capture",
                               side_effect=lambda idx, *a, **k:
                               opened.append(idx)), \
             mock.patch.object(self.diag, "_face_cascade_status",
                               return_value=(True, "loaded")), \
             mock.patch.object(self.diag, "_maybe_announce_once"), \
             mock.patch.object(self.diag.time, "sleep"):
            res = self.diag._probe_webcam()
        self.assertNotIn(0, opened, "opened the producer's own camera")
        return res

    def test_black_frames_after_a_lit_one_are_reported(self):
        # The lit frame is still in the cache and still "fresh": judging it
        # passed a camera that has been black ever since.
        self._run_one_iteration(self.lit)
        self._run_one_iteration(self.black)
        res = self._probe()
        self.assertFalse(res["ok"], "a black webcam passed the self-check")
        self.assertEqual(res["details"].get("failure_mode"),
                         "persistent_black_frame")

    def test_a_long_black_run_is_reported_not_unverified(self):
        # No lit frame inside the freshness window at all.
        self._run_one_iteration(self.black)
        res = self._probe()
        self.assertTrue(res["tested"], res)
        self.assertEqual(res["details"].get("failure_mode"),
                         "persistent_black_frame")
        self.assertEqual(res["details"].get("verified_via"),
                         "face-track producer telemetry")

    def test_a_dark_room_is_not_a_dead_sensor(self):
        # Too dark to treat as a live frame, yet not the probe's lens-cap black
        # (_BLACK_FRAME_MEAN_MIN): no finding, no "check the lens cover".
        self._run_one_iteration(self.np.full((72, 128, 3), 4, dtype=self.np.uint8))
        self.assertIn(0, self.bc._camera_black_frame_at, "setup: not a black run")
        res = self._probe()
        self.assertTrue(res["ok"], res.get("error"))
        self.assertNotIn("failure_mode", res["details"])

    def test_a_lit_frame_after_the_run_passes_again(self):
        self._run_one_iteration(self.black)
        self._run_one_iteration(self.lit)
        res = self._probe()
        self.assertTrue(res["ok"], res.get("error"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
