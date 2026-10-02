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


@requires_monolith
class BlackWebcamFrameTests(MonolithGlobalsTestCase):

    def setUp(self):
        import numpy as np
        self.np = np
        bc = self.bc
        # The frame caches are not in the harness's restore list: isolate them.
        for d in (bc._camera_latest_frame, bc._camera_last_frame_at,
                  bc._camera_last_read_error, bc._camera_last_read_error_at,
                  getattr(bc, "_camera_black_frame_at", {})):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
