"""camera_unquarantine lifts TWO holds, and says which (2026-09-30).

v2.0.137 (R11) let "use the Kinect again" (camera_unquarantine) also lift the
slow DIES-ON-OPEN retry, but the action's reply and its core/prompts.py line
still only spoke of the culprit QUARANTINE ("... if the hub drops out again
I'll switch it off again") - wrong for a device that was only on the slow
retry. And the face-track producer slept through the gate's whole reported
wait (up to 30 min on the slow retry) before asking again, so a lift for a
WEBCAM took effect only at its next scheduled ask; its re-ask is now capped
the way audio/kinect_bridge.py caps the Kinect's (_GATE_REASK_MAX_S).

The gate is driven by hand on a frozen clock with synthetic devices. Nothing
opens a camera.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_camera_storm import (
    _CAM_A, _CAM_B, _FakeBackend, _StormBase)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LEFT = "name:synthcam one"


@requires_monolith
class _LiftBase(_StormBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.g = bc._make_camera_gate(clock=self.clock.time)
        for p in (mock.patch.object(bc, "_camera_gate", self.g, create=True),
                  mock.patch.object(bc, "CAMERAS", [dict(_CAM_A), dict(_CAM_B)]),
                  mock.patch.object(bc, "_camera_backend",
                                    _FakeBackend(self.clock, names=lambda: (
                                        "SynthCam One", "SynthCam Two",
                                        "Synth Kinect Sensor"))),
                  mock.patch.object(bc, "proactive_announce",
                                    side_effect=lambda m, *a, **k:
                                    self.spoken.append(m) or True),
                  mock.patch("builtins.print")):
            p.start()
            self.addCleanup(p.stop)

    def _dies_on_open(self, key, component, times=3):
        g = self.g
        for _ in range(times):
            for _i in range(500):
                d = g.begin(key, component)
                if d.allowed:
                    break
                self.clock.advance(min(max(0.5, d.wait_s), 60.0))
            self.assertTrue(d.allowed, d)
            self.clock.advance(0.3)
            g.end(key, component, True)
            g.hold(key, component)
            self.clock.advance(4.0)
            g.unhold(key, component)
            g.note_drop(key, component)
        self.assertTrue(g.dies_on_open(key), f"{key} is not on the slow retry")

    def _quarantine_left(self):
        g = self.g
        g.culprit_threshold = 1
        g.begin(_LEFT, "face-track")
        g.end(_LEFT, "face-track", True)
        self.clock.advance(0.5)
        g.note_drop(_LEFT, "face-track")
        g.note_drop("name:synthcam two", "face-track")
        self.assertTrue(g.quarantined(_LEFT))

    def _say(self, arg):
        from skills import camera_system
        with mock.patch.object(camera_system, "_bc", return_value=self.bc):
            return camera_system.camera_unquarantine(arg)


class ReplyNamesTheHoldTests(_LiftBase):
    def test_slow_retry_lift_does_not_talk_about_the_hub(self):
        self._dies_on_open("kinect", "kinect-bridge")
        said = self._say("kinect")
        self.assertIn("the Kinect is back in use", said)
        self.assertIn("slow retry", said)
        self.assertIn("dropping off USB the moment it started streaming", said)
        self.assertIn("every thirty minutes", said)
        self.assertNotIn("hub drops out", said)
        self.assertFalse(self.g.dies_on_open("kinect"))

    def test_quarantine_lift_does_not_talk_about_the_slow_retry(self):
        self._quarantine_left()
        said = self._say("left")
        self.assertEqual(
            said, "Understood, sir - the left webcam is back in use. I'll "
                  "bring it up one at a time, and if the hub drops out again "
                  "I'll switch it off again.")

    def test_both_holds_on_two_cameras_are_named(self):
        self._quarantine_left()
        self.g.culprit_threshold = 99
        self._dies_on_open("kinect", "kinect-bridge")
        said = self._say("")
        self.assertIn("back in use", said)
        self.assertIn("I'll bring the left webcam up one at a time", said)
        self.assertIn("The Kinect was on the slow retry", said)
        self.assertFalse(self.g.quarantined(_LEFT))
        self.assertFalse(self.g.dies_on_open("kinect"))

    def test_nothing_lifted_names_a_camera_on_the_slow_retry(self):
        self._dies_on_open("kinect", "kinect-bridge")
        said = self._say("right")
        self.assertEqual(
            said, "The Kinect is the only camera on the slow retry, sir - "
                  "say 'use the Kinect again' to put it back.")
        self.assertTrue(self.g.dies_on_open("kinect"))

    def test_monolith_reports_which_hold_was_lifted(self):
        self._quarantine_left()
        self.g.culprit_threshold = 99
        self._dies_on_open("kinect", "kinect-bridge")
        got = dict(self.bc.camera_gate_lift(""))
        self.assertEqual(got, {"the left webcam": ("quarantine",),
                               "the Kinect": ("dies-on-open",)})
        # The labels-only API is unchanged for its other callers.
        self.assertEqual(self.bc.camera_gate_lift_quarantine(""), [])


class WebcamLiftTakesEffectWithinAMinuteTests(_LiftBase):
    def test_slow_retry_webcam_reask_is_capped(self):
        bc = self.bc
        self._dies_on_open(_LEFT, "face-track")
        entry = {"cam": dict(_CAM_A), "next_reopen_at": 0.0,
                 "contention_logged": False}
        now = self.clock.t
        backoff, _lk = bc._schedule_camera_reopen(entry, "Synth left",
                                                  _CAM_A["index"], now)
        self.assertGreater(backoff, 600.0, "the gate's own wait is long")
        self.assertLessEqual(entry["next_reopen_at"] - now, 60.0,
                             "the producer must ask again within a minute")
        # The early ask is still refused while the hold stands ...
        d = self.g.begin(_LEFT, "face-track", now=entry["next_reopen_at"])
        self.assertFalse(d.allowed)
        # ... and the owner's lift is seen at the very next ask.
        self.assertEqual(self._say("left").split(" - ")[0],
                         "Understood, sir")
        self.assertTrue(self.g.begin(_LEFT, "face-track",
                                     now=entry["next_reopen_at"]).allowed)

    def test_the_cap_matches_the_kinect_bridge(self):
        from audio import kinect_bridge
        self.assertEqual(self.bc._FACE_TRACK_GATE_REASK_MAX_S,
                         kinect_bridge._GATE_REASK_MAX_S)

    def test_short_waits_are_unchanged(self):
        bc = self.bc
        entry = {"cam": dict(_CAM_A), "next_reopen_at": 0.0,
                 "contention_logged": False}
        now = self.clock.t
        backoff, _lk = bc._schedule_camera_reopen(entry, "Synth left",
                                                  _CAM_A["index"], now)
        self.assertEqual(backoff, bc.CAMERA_REOPEN_BACKOFF_SEC)
        self.assertAlmostEqual(entry["next_reopen_at"], now + backoff)


class PromptDescribesBothHoldsTests(unittest.TestCase):
    def test_prompt_line_covers_the_slow_retry(self):
        with open(os.path.join(_ROOT, "core", "prompts.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("camera_unquarantine[, left|right|kinect]")
        block = src[i:src.index("[ACTION: camera_unquarantine, left]", i)]
        self.assertIn("USB hub offline", block)
        self.assertIn("slow retry", block)
        self.assertIn("drop", block)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
