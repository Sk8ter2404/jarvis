"""B027 (2026-10-01): a STALE cached webcam frame must never pass for a live one.

The face-track producer writes ``_camera_latest_frame[idx]`` only on a good
read and never clears it, so after a camera dies, backs off, is benched or the
producer stalls, the cache keeps serving that camera's LAST frame. see_user,
tv_detect, the side tiles and camera_status judged the frame by its age; these
readers did not:

  * skills/face_id - whoami named the owner at an empty desk, enroll_face
    enrolled one frozen frame five times ("Captured 5 good views"), and
    face_id_status said "the webcam is live";
  * skills/guard_mode - an armed guard diffed the frozen frame against itself,
    never alerted, and still counted the camera ("I'll be watching N cameras");
  * skills/camera_system.look_around - described the old scene as current;
  * skills/face_tracker - the new-people greeting re-scanned the frozen frame.

Every test drives the REAL monolith helper (bobert_companion._fresh_camera_frame)
through the REAL skill code. Nothing opens a camera, the Kinect or a model: the
face-ID engine is a stand-in, and the frame cache is patched per test.
"""
from __future__ import annotations

import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated

_CAMS = [{"index": 0, "label": "Right webcam (top of right monitor)",
          "name": "synthcam right", "primary": True,
          "look_x": 0.85, "look_y": 0.5}]


def _engine():
    """audio.face_id stand-in: always sees the owner, records enrolments."""
    m = types.ModuleType("audio.face_id")
    m.is_available = lambda: (True, "")
    m.recognize = lambda frame: [{"name": "owner", "score": 0.9,
                                  "bbox": [100, 100, 200, 200]}]
    m.enrolled_frames = []

    def _enroll(name, frames):
        m.enrolled_frames.extend(frames)
        return len(frames)
    m.enroll = _enroll
    m.list_enrolled = lambda: []
    return m


@requires_monolith
class StaleFrameBase(MonolithGlobalsTestCase):

    def setUp(self):
        import numpy as np
        bc = self.bc
        self.frame = np.full((72, 128, 3), 40, dtype=np.uint8)
        for d in (bc._camera_latest_frame, bc._camera_last_frame_at):
            p = mock.patch.dict(d, clear=True)
            p.start()
            self.addCleanup(p.stop)
        import core.config as _cfg
        for p in (mock.patch.object(_cfg, "CAMERAS", [dict(c) for c in _CAMS]),
                  mock.patch.object(bc, "CAMERAS", [dict(c) for c in _CAMS])):
            p.start()
            self.addCleanup(p.stop)

    def _cache(self, age_s: float) -> None:
        """The producer last delivered a frame for index 0 ``age_s`` ago."""
        bc = self.bc
        with bc._camera_state_lock:
            bc._camera_latest_frame[0] = self.frame.copy()
            bc._camera_last_frame_at[0] = time.time() - age_s

    def _skill(self, name):
        mod, actions = load_skill_isolated(name, register=True)
        mod._bc = lambda: self.bc
        return mod, actions


class FaceIdTests(StaleFrameBase):

    def _face_id(self):
        mod, actions = self._skill("face_id")
        eng = _engine()
        mod._engine = lambda: eng
        mod._cfg_flag = lambda name, default=False: name == "FACE_ID_ENABLED"
        mod._is_staging = lambda: False
        mod._owner_name = lambda: "owner"
        return mod, actions, eng

    def test_control_a_live_frame_is_recognised(self):
        _mod, actions, _eng = self._face_id()
        self._cache(age_s=0.2)
        self.assertEqual(actions["whoami"](""), "That's you, sir.")

    def test_whoami_does_not_recognise_a_frozen_frame(self):
        _mod, actions, _eng = self._face_id()
        self._cache(age_s=30.0)
        self.assertIn("can't see", actions["whoami"]("").lower())

    def test_enroll_refuses_a_frozen_frame(self):
        _mod, actions, eng = self._face_id()
        self._cache(age_s=30.0)
        out = actions["enroll_face"]("")
        self.assertIn("can't see", out.lower())
        self.assertEqual(eng.enrolled_frames, [],
                         "a stale frame was enrolled into the owner's face")

    def test_status_calls_a_frozen_webcam_dark(self):
        _mod, actions, _eng = self._face_id()
        self._cache(age_s=30.0)
        out = actions["face_id_status"]("")
        self.assertIn("the webcam is dark", out)
        self.assertNotIn("the webcam is live", out)

    def test_one_frame_is_not_enrolled_five_times(self):
        # A RECENT frame the producer has not replaced since (one producer
        # timestamp): five grabs of it are one view, not five.
        mod, _actions, _eng = self._face_id()
        self._cache(age_s=0.2)
        frames = mod._grab_frames(0, n=5, gap_s=0.0)
        self.assertEqual(len(frames), 1)


class GuardModeTests(StaleFrameBase):

    def _guard(self):
        mod, _actions = self._skill("guard_mode")
        mod._cfg_flag = lambda name, default=False: False   # no Kinect
        return mod

    def test_control_a_live_camera_is_watched(self):
        mod = self._guard()
        self._cache(age_s=0.2)
        self.assertEqual(len(mod._collect_frames()), 1)

    def test_a_frozen_camera_is_not_watched_or_counted(self):
        mod = self._guard()
        self._cache(age_s=30.0)
        self.assertEqual(mod._collect_frames(), [])
        self.assertEqual(mod._available_camera_count(), 0)

    def test_a_camera_that_went_dark_re_baselines(self):
        mod = self._guard()
        mod._prev_frames["the right monitor camera"] = object()
        mod._motion_streak["the right monitor camera"] = 1
        self._cache(age_s=30.0)
        mod._collect_frames()
        self.assertNotIn("the right monitor camera", mod._prev_frames)
        self.assertNotIn("the right monitor camera", mod._motion_streak)


class LookAroundTests(StaleFrameBase):

    def test_look_around_does_not_describe_a_frozen_scene(self):
        mod, _actions = self._skill("camera_system")
        mod._kinect_health = lambda: {"available": False}
        self._cache(age_s=30.0)
        self.assertEqual(mod._collect_frames(), [])
        self._cache(age_s=0.2)
        self.assertEqual(len(mod._collect_frames()), 1)


class NewPeopleGrabTests(StaleFrameBase):

    def test_the_greeting_does_not_scan_a_frozen_frame(self):
        mod, _actions = self._skill("face_tracker")
        self._cache(age_s=30.0)
        self.assertIsNone(mod._grab_primary_frame(self.bc))
        self._cache(age_s=0.2)
        self.assertIsNotNone(mod._grab_primary_frame(self.bc))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
