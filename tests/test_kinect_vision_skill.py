"""Tests for skills/kinect_vision — the Kinect voice actions.

Loads the skill in isolation (no monolith boot) via the shared skill harness,
with a FAKE kinect_bridge injected into sys.modules so no real sensor is
touched. Covers each action in both the present and absent/off cases, and
verifies kinect_look routes the color PNG through the injected ask_vision.
stdlib unittest + mock.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated, make_fake_skill_utils


def _fake_bridge(*, enabled=True, available=(True, ""), presence=None,
                 color_png=b"\x89PNG-kinect", color_bgr=object(),
                 depth=object(), infrared=None):
    """Build a stand-in audio.kinect_bridge module."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_presence = lambda: (presence if presence is not None
                              else {"present": False, "count": 0,
                                    "nearest_m": None, "facing": None,
                                    "ts": 0.0})
    m.get_color_png = lambda: color_png
    m.get_color_bgr = lambda require_new=True: color_bgr
    m.get_depth = lambda: depth
    m.get_infrared_gray = lambda: infrared
    return m


class KinectSkillBase(unittest.TestCase):
    def _load(self, bridge, *, utils=None):
        """Inject the fake bridge, load the skill, return its actions dict."""
        old = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = bridge
        self.addCleanup(
            lambda: sys.modules.__setitem__("audio.kinect_bridge", old)
            if old is not None else sys.modules.pop("audio.kinect_bridge", None))
        _mod, actions = load_skill_isolated(
            "kinect_vision", utils=utils, register=True)
        return actions


# ─────────────────────────────────────────────────────────────────────────
# registration
# ─────────────────────────────────────────────────────────────────────────
class RegistrationTests(KinectSkillBase):
    def test_registers_all_actions(self):
        actions = self._load(_fake_bridge())
        for name in ("kinect_status", "who_is_here", "scan_room",
                     "kinect_look", "what_do_you_see_kinect"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


# ─────────────────────────────────────────────────────────────────────────
# kinect_status
# ─────────────────────────────────────────────────────────────────────────
class StatusTests(KinectSkillBase):
    def test_status_present_reports_streams_and_people(self):
        bridge = _fake_bridge(
            presence={"present": True, "count": 1, "nearest_m": 1.8,
                      "facing": True, "ts": 0.0})
        actions = self._load(bridge)
        out = actions["kinect_status"]("")
        self.assertIn("connected", out.lower())
        self.assertIn("color", out)        # color stream detected
        self.assertIn("depth", out)        # depth stream detected
        self.assertIn("one person", out)

    def test_status_off_when_disabled(self):
        actions = self._load(_fake_bridge(enabled=False))
        out = actions["kinect_status"]("")
        self.assertIn("off", out.lower())
        self.assertIn("privacy", out.lower())

    def test_status_unavailable_reports_reason(self):
        actions = self._load(_fake_bridge(available=(False, "no sensor")))
        out = actions["kinect_status"]("")
        self.assertIn("no sensor", out)


# ─────────────────────────────────────────────────────────────────────────
# who_is_here / scan_room
# ─────────────────────────────────────────────────────────────────────────
class WhoIsHereTests(KinectSkillBase):
    def test_one_person_with_distance(self):
        bridge = _fake_bridge(
            presence={"present": True, "count": 1, "nearest_m": 1.8,
                      "facing": True, "ts": 0.0})
        actions = self._load(bridge)
        out = actions["who_is_here"]("")
        self.assertIn("one person", out)
        self.assertIn("1.8", out)
        self.assertIn("facing", out)

    def test_no_one_present(self):
        actions = self._load(_fake_bridge())   # default presence = empty
        out = actions["who_is_here"]("")
        self.assertIn("anyone", out.lower())   # "don't see anyone"

    def test_multiple_people(self):
        bridge = _fake_bridge(
            presence={"present": True, "count": 3, "nearest_m": 2.4,
                      "facing": None, "ts": 0.0})
        actions = self._load(bridge)
        out = actions["who_is_here"]("")
        self.assertIn("3 people", out)
        self.assertIn("2.4", out)

    def test_scan_room_is_alias(self):
        bridge = _fake_bridge(
            presence={"present": True, "count": 1, "nearest_m": 1.5,
                      "facing": None, "ts": 0.0})
        actions = self._load(bridge)
        self.assertEqual(actions["scan_room"](""), actions["who_is_here"](""))

    def test_who_is_here_off_when_disabled(self):
        actions = self._load(_fake_bridge(enabled=False))
        out = actions["who_is_here"]("")
        self.assertIn("off", out.lower())


# ─────────────────────────────────────────────────────────────────────────
# kinect_look / what_do_you_see_kinect
# ─────────────────────────────────────────────────────────────────────────
class LookTests(KinectSkillBase):
    def test_look_routes_png_through_ask_vision(self):
        seen = {}

        def _ask(question, png):
            seen["question"] = question
            seen["png"] = png
            return "a tidy desk and one person"
        utils = make_fake_skill_utils(ask_vision=mock.MagicMock(side_effect=_ask))
        bridge = _fake_bridge(color_png=b"\x89PNG-XYZ")
        actions = self._load(bridge, utils=utils)
        out = actions["kinect_look"]("who is here")
        self.assertIn("Looking through the Kinect", out)
        self.assertIn("a tidy desk", out)
        self.assertEqual(seen["png"], b"\x89PNG-XYZ")
        self.assertEqual(seen["question"], "who is here")

    def test_look_default_question_when_blank(self):
        captured = {}
        utils = make_fake_skill_utils(
            ask_vision=mock.MagicMock(
                side_effect=lambda q, p: captured.setdefault("q", q) or "ok"))
        actions = self._load(_fake_bridge(), utils=utils)
        actions["kinect_look"]("")
        self.assertIn("room", captured["q"].lower())

    def test_look_off_when_disabled(self):
        actions = self._load(_fake_bridge(enabled=False))
        out = actions["kinect_look"]("what do you see")
        self.assertIn("off", out.lower())

    def test_look_unavailable_reports_reason(self):
        actions = self._load(_fake_bridge(available=(False, "sensor unplugged")))
        out = actions["kinect_look"]("hi")
        self.assertIn("sensor unplugged", out)

    def test_look_handles_no_frame(self):
        actions = self._load(_fake_bridge(color_png=None))
        out = actions["kinect_look"]("hi")
        self.assertIn("didn't hand me a frame", out.lower())

    def test_what_do_you_see_kinect_is_alias(self):
        utils = make_fake_skill_utils(
            ask_vision=mock.MagicMock(return_value="something"))
        actions = self._load(_fake_bridge(), utils=utils)
        out = actions["what_do_you_see_kinect"]("test")
        self.assertIn("Looking through the Kinect", out)


if __name__ == "__main__":
    unittest.main()
