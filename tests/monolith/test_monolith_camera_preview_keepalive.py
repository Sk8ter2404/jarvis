"""The SHARED HUD camera preview must survive losing the PRIMARY camera.

THE DEFECT THIS FILE PINS (2026-09-05). The single HUD preview JPEG is written
from ONE branch in the face-tracking producer:

    if (cam["primary"] and not camera_off and _hud_camera_preview_enabled()):
        if not _hud_kinect_preview_write(now_loop):
            _hud_camera_preview_write(frame, now_loop)

That branch lives INSIDE ``for entry in caps:``, so it sits behind every
``continue`` above it - the quarantine bench, the reopen backoff, a failed
reopen, the soft-wake path's own bench. Losing the PRIMARY therefore did not
just darken the primary's tile, it stopped ALL shared-preview production, and
the only line the owner got promised the opposite ("its tile goes dark, every
OTHER camera keeps publishing"). On the live rig the primary is index 2, the
eMeet C960 - the camera Teams/Zoom is most likely to steal - so the HUD camera
tile went dark for 60-600 s at a time and pointed him at his webcam hardware.

WHY THESE TESTS DRIVE THE REAL LOOP. The sibling quarantine suite pins some
producer-body guarantees with ``inspect.getsource`` string assertions, which
cannot tell a working keep-alive from a commented-out one. These tests call
``_face_tracking_thread_body()`` itself for exactly one iteration, with a
one-shot stop event, and assert on what the producer actually published.

NOTHING HERE OPENS A REAL CAMERA. Every open in the body funnels through the
module-level ``_open_capture_bounded``, which is patched to hand back a fake;
``_dshow_name_to_index`` is patched so no DirectShow enumeration runs; and no
release path is exercised outside the monolith's own ``_camera_io_lock``.

The fixtures mirror the LIVE config shape on purpose: primary = index 2, and
``KINECT_ENABLED=False`` (so ``_hud_kinect_preview_write`` returns False and
the plain mirror is the only thing that can keep the shared tile alive).
"""
from __future__ import annotations

import logging
import time
import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, requires_monolith,
                                     load_monolith)


class _LoopCap:
    """Capture stand-in shaped for the PRODUCER LOOP (not the tile
    compositor): it opens, answers the loop's CAP_PROP_FRAME_* probe, and then
    either delivers one fixed frame forever (healthy) or never delivers one."""

    def __init__(self, frame=None):
        self._frame = frame
        self.released = False
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 1280.0

    def set(self, prop, val):
        return True

    def read(self):
        self.reads += 1
        if self._frame is None:
            return False, None
        return True, self._frame

    def release(self):
        self.released = True


class _OneShotStop:
    """Stop event that lets exactly ``iterations`` producer loops through."""

    def __init__(self, iterations=1):
        self._left = iterations

    def is_set(self):
        if self._left <= 0:
            return True
        self._left -= 1
        return False


@requires_monolith
class SharedPreviewKeepAliveTests(MonolithGlobalsTestCase):

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        import numpy as np
        self.np = np
        # The live CAMERAS shape from core/config.py: index 2 is PRIMARY.
        self.cams = [
            {"index": 2, "label": "Left webcam (left monitor)",
             "name": "emeet c960", "primary": True,
             "look_x": 0.5, "look_y": 0.5},
            {"index": 0, "label": "Right webcam (top of right monitor)",
             "name": "usb 2.0 camera", "primary": False,
             "look_x": 0.85, "look_y": 0.5},
        ]
        self.left = np.full((8, 8, 3), 11, dtype=np.uint8)
        self.right = np.full((8, 8, 3), 22, dtype=np.uint8)
        bc = self.bc
        bc._face_track_stop.clear()
        bc._face_track_pause.clear()
        bc._face_track_camera_off.clear()
        bc._standby_mode[0] = False
        bc._preview_failover_log[0] = 0.0
        bc._preview_failover_log[1] = ""

    def _bench(self, idx, label, now):
        """Drive `idx` into quarantine the way the producer does."""
        for i in range(self.bc._CAMERA_QUARANTINE_STRIKES):
            self.bc._camera_note_sick_cycle(idx, label, "read fail", now + i)

    def _run_one_iteration(self, caps):
        """Run the REAL producer body for exactly one iteration.

        `caps` maps camera index -> _LoopCap (a missing index means that camera
        fails to open, which is itself one of the paths that skipped the
        publisher). Returns (shared_preview_mock, percam_preview_mock)."""
        bc = self.bc
        shared = mock.Mock(return_value=True)
        percam = mock.Mock(return_value=True)

        # *args/**kwargs on purpose: _open_capture_bounded's keyword surface
        # (label=, timeout=, beat=, ...) is still being extended by the wider
        # resilience work, and this fixture must not break every time it grows.
        def _bounded(idx, opener, *args, **kwargs):
            return caps.get(idx)

        with mock.patch.object(bc, "CAMERAS", self.cams), \
             mock.patch.object(bc, "_face_track_stop", _OneShotStop()), \
             mock.patch.object(bc, "_dshow_name_to_index",
                               side_effect=lambda n: 2 if "emeet" in n else 0), \
             mock.patch.object(bc, "_open_capture_bounded", side_effect=_bounded), \
             mock.patch.object(bc, "_hud_camera_preview_enabled", return_value=True), \
             mock.patch.object(bc, "_hud_kinect_preview_write", return_value=False), \
             mock.patch.object(bc, "_hud_camera_preview_write", shared), \
             mock.patch.object(bc, "_hud_percam_preview_write", percam), \
             mock.patch.object(bc, "_detect_face", return_value=None), \
             mock.patch.object(bc, "send"):
            bc._face_tracking_thread_body()
        return shared, percam

    # ------------------------------------------------------------- control --

    def test_healthy_primary_publishes_its_own_frame(self):
        """Baseline, so a green result below cannot be green for the wrong
        reason: with everything healthy the PRIMARY's frame still reaches the
        shared preview, exactly as before."""
        shared, _percam = self._run_one_iteration(
            {2: _LoopCap(self.left), 0: _LoopCap(self.right)})
        self.assertEqual(shared.call_count, 1,
                         "the healthy path stopped publishing the shared preview")
        self.assertIs(shared.call_args.args[0], self.left,
                      "a healthy primary must still own the shared preview")

    # -------------------------------------------------------- the regression --

    def test_benched_primary_does_not_stop_the_shared_preview(self):
        """THE DEFECT. Bench index 2 (the PRIMARY) and run one iteration.

        Before the post-loop keep-alive, the quarantine gate's bare ``continue``
        skipped the publisher entirely and this call count was 0: the HUD camera
        tile went dark for the whole 60-600 s bench, silently."""
        now = time.time()
        self._bench(2, "Left webcam (left monitor)", now)
        self.assertTrue(self.bc._camera_is_quarantined(2),
                        "setup failed: the primary was not actually benched")

        caps = {2: _LoopCap(self.left), 0: _LoopCap(self.right)}
        shared, _percam = self._run_one_iteration(caps)

        self.assertGreaterEqual(
            shared.call_count, 1,
            "benching the PRIMARY stopped ALL shared-preview production - the "
            "HUD camera tile goes dark for the entire bench")
        self.assertIs(
            shared.call_args.args[0], self.right,
            "the shared preview must fail over to the healthy camera's frame")
        self.assertEqual(
            caps[2].reads, 0,
            "the benched primary was read anyway - the bench is not honoured")

    def test_the_failover_is_announced_not_silent(self):
        """Putting the RIGHT webcam into the tile the owner reads as the LEFT
        one, and saying nothing, would be the same lie in a new place. The
        substitution must reach the log, naming the camera now on screen."""
        now = time.time()
        self._bench(2, "Left webcam (left monitor)", now)
        with self.assertLogs(level=logging.WARNING) as cm:
            self._run_one_iteration(
                {2: _LoopCap(self.left), 0: _LoopCap(self.right)})
        blob = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("HUD camera tile is now showing", blob)
        self.assertIn("index 0", blob)
        self.assertIn("NOT the primary camera", blob)

    def test_a_primary_that_never_opened_also_keeps_the_tile_alive(self):
        """The bench is not the only ``continue`` above the publisher. A primary
        that fails to OPEN never enters `caps` at all, so the publisher branch
        is unreachable for it on every iteration - the same silent outage from a
        different door. Same keep-alive covers it."""
        shared, _percam = self._run_one_iteration({0: _LoopCap(self.right)})
        self.assertGreaterEqual(
            shared.call_count, 1,
            "a primary that could not be opened silently stopped the shared "
            "preview")
        self.assertIs(shared.call_args.args[0], self.right)

    def test_a_primary_that_opens_but_never_reads_keeps_the_tile_alive(self):
        """The failing camera OPENS FINE and then fails to READ - the owner's
        actual fault. The read-miss path ``continue``s too; the primary's own
        keep-alive covers the Kinect composite, and with the composite off
        (live config) the post-loop keep-alive must still publish something."""
        caps = {2: _LoopCap(None), 0: _LoopCap(self.right)}
        shared, _percam = self._run_one_iteration(caps)
        self.assertGreater(caps[2].reads, 0,
                           "setup wrong: the primary must have been read")
        self.assertGreaterEqual(
            shared.call_count, 1,
            "a primary that opens but delivers no frame took the shared "
            "preview down with it")
        self.assertIs(shared.call_args.args[0], self.right)

    def test_nothing_to_publish_is_reported_rather_than_left_dark(self):
        """Both cameras benched: there is genuinely nothing to publish, and
        substituting a stale frame would be a lie. The producer must SAY the
        tile is going stale and why, instead of letting the owner infer a
        hardware fault from a dark tile."""
        now = time.time()
        self._bench(2, "Left webcam (left monitor)", now)
        self._bench(0, "Right webcam (top of right monitor)", now)
        with self.assertLogs(level=logging.WARNING) as cm:
            shared, _percam = self._run_one_iteration(
                {2: _LoopCap(self.left), 0: _LoopCap(self.right)})
        blob = "\n".join(r.getMessage() for r in cm.records)
        self.assertEqual(shared.call_count, 0,
                         "nothing could honestly be published, so nothing must be")
        self.assertIn("going stale", blob)
        self.assertIn("NOT the HUD failing", blob)

    def test_camera_off_still_blanks_the_preview(self):
        """No regression on the deliberate blank: standby / low-memory
        camera_off must still stop the preview. The keep-alive is gated on the
        same camera_off + enabled checks as the branch it backs up, so it must
        NOT resurrect a tile the system intentionally turned off."""
        now = time.time()
        self._bench(2, "Left webcam (left monitor)", now)
        self.bc._standby_mode[0] = True
        try:
            shared, _percam = self._run_one_iteration(
                {2: _LoopCap(self.left), 0: _LoopCap(self.right)})
        finally:
            self.bc._standby_mode[0] = False
        self.assertEqual(shared.call_count, 0,
                         "the keep-alive published while the camera is OFF")

    # ------------------------------------------------- the misleading line --

    def test_benching_the_primary_no_longer_claims_the_others_publish(self):
        """The one line the owner actually got said 'its tile goes dark, every
        OTHER camera keeps publishing'. For the PRIMARY that was false, and it
        is what sent him looking at his webcam hardware."""
        now = time.time()
        with mock.patch.object(self.bc, "CAMERAS", self.cams), \
             self.assertLogs(level=logging.WARNING) as cm:
            self._bench(2, "Left webcam (left monitor)", now)
        blob = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("QUARANTINE", blob)
        self.assertIn("PRIMARY", blob)
        self.assertIn("fails over", blob)
        self.assertNotIn("every OTHER camera keeps publishing", blob)

    def test_benching_a_side_camera_still_reports_the_old_way(self):
        """No regression on the message for the case it was always true for."""
        now = time.time()
        with mock.patch.object(self.bc, "CAMERAS", self.cams), \
             self.assertLogs(level=logging.WARNING) as cm:
            self._bench(0, "Right webcam (top of right monitor)", now)
        blob = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("every OTHER camera keeps publishing", blob)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
