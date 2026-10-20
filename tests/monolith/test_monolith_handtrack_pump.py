"""STEADY FRAMES (2026-10-20): the face-tracking producer and the hand
tracker's frame pump (core/handtrack_frames.FramePump) share ONE capture handle.

Live on the Dell (v2.0.226) the webcam hand cursor felt laggy: the tracker got
~18 frames/s but in clumps (the producer read ~0.15 s bursts between face
passes). While the tracker is alive the producer now takes the tracker
camera's frames from the pump, whose one thread reads that handle at the
camera's own rate. These tests drive the REAL producer body (the harness from
test_monolith_camera_preview_keepalive.py: no real camera, no DirectShow) and
pin the handle discipline:
  * no tracker -> the producer reads the handle itself, exactly as before, and
    no pump thread ever starts;
  * tracker alive -> the camera is opened ONCE, read by one thread at a time,
    every frame reaches the tracker evenly, and the producer's own nap stays;
  * every release goes through the pump first: a handle is never released
    while the pump is inside read() on it.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith import test_monolith_camera_preview_keepalive as _ka

_CAMS = [{"index": 0, "label": "Right webcam (top of right monitor)",
          "name": "usb 2.0 camera", "primary": True,
          "look_x": 0.85, "look_y": 0.5}]


class _PacedCap(_ka._LoopCap):
    """The producer-loop fake, but each read() blocks one frame interval and
    records which threads read and whether two were ever inside at once."""

    def __init__(self, frame, fps=30.0):
        super().__init__(frame)
        self.dt = 1.0 / fps
        self._lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.readers = set()
        self.release_while_reading = False

    def read(self):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            self.readers.add(threading.get_ident())
        try:
            time.sleep(self.dt)
            return super().read()
        finally:
            with self._lock:
                self.inside -= 1

    def release(self):
        with self._lock:
            if self.inside:
                self.release_while_reading = True
        super().release()


@requires_monolith
class FramePumpProducerTests(MonolithGlobalsTestCase):

    def setUp(self):
        import numpy as np
        bc = self.bc
        for d in (bc._camera_latest_frame, bc._camera_last_frame_at,
                  bc._camera_last_read_error, bc._camera_last_read_error_at):
            p = mock.patch.dict(d, clear=True)
            p.start()
            self.addCleanup(p.stop)
        bc._face_track_stop.clear()
        bc._face_track_pause.clear()
        bc._face_track_camera_off.clear()
        bc._standby_mode[0] = False
        self.frame = np.full((72, 128, 3), 90, dtype=np.uint8)
        self.ht = bc._handtrack_frames
        self.pump = self.ht.PUMP
        self.addCleanup(self.pump.detach)
        self.published = []

    def _pub(self, frame):
        # publish(None) is the idle-segment close tick, not a frame
        if frame is not None:
            self.published.append(time.monotonic())
        return frame is not None

    def _run(self, iterations, tracker_alive):
        bc = self.bc
        gate = bc._make_camera_gate()
        gate.stagger_s = 0.0
        cap = _PacedCap(self.frame)
        opens = []

        def _open(idx, opener, *a, **k):
            opens.append(idx)
            return cap

        starts0 = self.pump.starts
        producer = []
        with mock.patch.object(bc, "CAMERAS", _CAMS), \
             mock.patch.object(bc, "_camera_gate", gate), \
             mock.patch.object(bc, "_face_track_stop", _ka._OneShotStop(iterations)), \
             mock.patch.object(bc, "_dshow_name_to_index", return_value=0), \
             mock.patch.object(bc, "_open_capture_bounded", side_effect=_open), \
             mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=False), \
             mock.patch.object(bc, "_detect_face", return_value=None), \
             mock.patch.object(bc, "send"), \
             mock.patch.object(self.ht.WRITER, "active", return_value=tracker_alive), \
             mock.patch.object(self.ht.WRITER, "publish", side_effect=self._pub), \
             mock.patch.object(self.ht, "_setting", return_value="primary"):
            producer.append(threading.get_ident())
            bc._face_tracking_thread_body()
            # The supervisor's teardown, as it runs on the producer thread.
            bc._face_track_release_all(bc._face_track_caps[0])
        return cap, opens, self.pump.starts - starts0, producer[0]

    def test_no_tracker_reads_on_the_producer_and_starts_no_pump(self):
        cap, opens, starts, producer = self._run(4, tracker_alive=False)
        self.assertEqual(opens, [0])
        self.assertEqual(starts, 0)
        self.assertEqual(cap.readers, {producer})
        self.assertEqual(cap.reads, 4)                 # one read per pass, as before
        self.assertEqual(self.published, [])
        self.assertTrue(cap.released)

    def test_tracker_alive_one_open_one_reader_even_frames(self):
        cap, opens, starts, producer = self._run(6, tracker_alive=True)
        self.assertEqual(opens, [0], "the camera was opened a second time")
        self.assertEqual(starts, 1)
        self.assertEqual(cap.max_inside, 1, "two threads were inside read()")
        self.assertNotIn(producer, cap.readers,
                         "the producer read the handle the pump was reading")
        self.assertGreater(len(self.published), 6)      # more than one per pass
        gaps = sorted(b - a for a, b in zip(self.published, self.published[1:]))
        self.assertLess(gaps[len(gaps) // 2], 0.05, gaps)   # one camera frame
        self.assertLess(gaps[-1], 0.2, gaps)
        self.assertTrue(cap.released)
        self.assertFalse(cap.release_while_reading,
                         "released while the pump was inside read()")
        self.assertFalse(self.pump.bound())

    def test_release_detaches_the_pump_first(self):
        cap = _PacedCap(self.frame)
        with mock.patch.object(self.ht.WRITER, "active", return_value=True), \
             mock.patch.object(self.ht.WRITER, "publish"), \
             mock.patch.object(self.ht, "_setting", return_value="primary"):
            self.assertTrue(self.pump.serve(cap, _CAMS[0])[0])
            time.sleep(0.01)                           # the reader is in read()
            self.assertTrue(self.bc._release_capture_guarded(cap, 0, "cam"))
        self.assertTrue(cap.released)
        self.assertFalse(cap.release_while_reading)
        self.assertFalse(self.pump.bound())


@requires_monolith
class PumpPreviewHookTests(MonolithGlobalsTestCase):
    """The HUD preview follows the steady frames: the pump's hook writes the
    tracker camera's tiles at the pump cadence, gated exactly like the
    producer's own writes."""

    def setUp(self):
        import numpy as np
        self.frame = np.full((72, 128, 3), 90, dtype=np.uint8)
        self.bc._face_track_camera_off.clear()
        self.bc._standby_mode[0] = False

    def test_hooks_are_installed(self):
        ht = self.bc._handtrack_frames
        self.assertIs(ht.PUMP.on_frame, self.bc._ht_pump_preview)
        self.assertIs(ht.PUMP.accept, self.bc._ht_pump_frame_ok)

    def test_cadence_constants(self):
        bc = self.bc
        self.assertAlmostEqual(bc._HUD_CAM_PREVIEW_PUMP_GAP, 1 / 15, places=3)
        self.assertEqual(bc._HUD_CAM_PREVIEW_MIN_GAP, 0.15)   # no tracker: unchanged
        self.assertLess(bc._HUD_CAM_PREVIEW_PUMP_GAP, bc._HUD_CAM_PREVIEW_MIN_GAP)

    def test_primary_frame_writes_both_tiles_at_the_pump_gap(self):
        bc = self.bc
        with mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=True), \
             mock.patch.object(bc, "_hud_kinect_skeleton_overlay_enabled", return_value=False), \
             mock.patch.object(bc, "_hud_percam_preview_write") as per, \
             mock.patch.object(bc, "_hud_camera_preview_write") as shared:
            bc._ht_pump_preview(_CAMS[0], self.frame)
        self.assertEqual(per.call_args.kwargs["min_gap"], bc._HUD_CAM_PREVIEW_PUMP_GAP)
        self.assertEqual(shared.call_args.kwargs["min_gap"], bc._HUD_CAM_PREVIEW_PUMP_GAP)

    def test_composite_and_camera_off_are_respected(self):
        bc = self.bc
        with mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=True), \
             mock.patch.object(bc, "_hud_kinect_skeleton_overlay_enabled", return_value=True), \
             mock.patch.object(bc, "_hud_percam_preview_write") as per, \
             mock.patch.object(bc, "_hud_camera_preview_write") as shared:
            bc._ht_pump_preview(_CAMS[0], self.frame)
            self.assertEqual(per.call_count, 1)
            shared.assert_not_called()                 # the composite owns it
            bc._face_track_camera_off.set()
            try:
                bc._ht_pump_preview(_CAMS[0], self.frame)
            finally:
                bc._face_track_camera_off.clear()
            self.assertEqual(per.call_count, 1)        # camera off: nothing
        with mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=False), \
             mock.patch.object(bc, "_hud_percam_preview_write") as per:
            bc._ht_pump_preview(_CAMS[0], self.frame)
            per.assert_not_called()

    def test_min_gap_throttles_the_writer(self):
        bc = self.bc
        with mock.patch.object(bc, "_hud_cam_preview_last_write", [100.0]), \
             mock.patch.object(bc.cv2, "imencode", return_value=(False, None)) as enc:
            self.assertFalse(bc._hud_camera_preview_write(self.frame, 100.05,
                                                          min_gap=1 / 15))
            enc.assert_not_called()                    # throttled before encoding
            bc._hud_camera_preview_write(self.frame, 100.07, min_gap=1 / 15)
            self.assertEqual(enc.call_count, 1)
            bc._hud_camera_preview_write(self.frame, 100.10)  # default 0.15 gap
            self.assertEqual(enc.call_count, 1)

    def test_black_frames_are_not_handed_to_the_tracker(self):
        import numpy as np
        self.assertFalse(self.bc._ht_pump_frame_ok(np.zeros((72, 128, 3), np.uint8)))
        self.assertTrue(self.bc._ht_pump_frame_ok(self.frame))


if __name__ == "__main__":
    unittest.main()
