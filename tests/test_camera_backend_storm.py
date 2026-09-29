"""core/camera_backend.py pieces added for the 2026-09-29 USB-storm fix.

  * webcam_users_now(): WHO IS USING a webcam, from Windows' camera privacy
    log - the evidence a camera is held. "Which camera apps are RUNNING" was
    wrong 23 of 23 times that day.
  * open_camera(): opens ONCE in the final format - the pixel format (MJPG)
    first, then width, then height, all before the first read.
  * open_camera(outcome=...): says WHY a failed open failed ("no-frame" is the
    shape of a held device on MSMF; "not-opened" the shape of a vanished one).

No registry, no camera: winreg and cv2 are fakes. App names are synthetic.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from core import camera_backend as cb


class _FakeReg:
    """The slice of winreg webcam_users_now() uses, over a nested dict:
    {subkey: {...} | values-dict}. A node is a key; its "_values" holds
    LastUsedTimeStart / LastUsedTimeStop."""

    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, tree):
        self.tree = tree

    def OpenKey(self, parent, sub):
        node = self.tree.get(sub) if parent == "HKCU" else parent.get(sub)
        if node is None:
            raise OSError("no such key")
        return node

    def EnumKey(self, key, i):
        names = [k for k in key if k != "_values"]
        if i >= len(names):
            raise OSError("no more")
        return names[i]

    def QueryValueEx(self, key, name):
        vals = key.get("_values", {})
        if name not in vals:
            raise OSError("no value")
        return vals[name], 11


def _app(start, stop):
    return {"_values": {"LastUsedTimeStart": start, "LastUsedTimeStop": stop}}


class WebcamUsersNowTests(unittest.TestCase):

    def _reg(self, apps, nonpackaged=None):
        webcam = dict(apps)
        if nonpackaged is not None:
            webcam["NonPackaged"] = nonpackaged
        return _FakeReg({cb._CONSENT_WEBCAM_KEY: webcam})

    def test_only_apps_using_a_webcam_right_now_are_listed(self):
        reg = self._reg(
            {"SynthMeet_8wekyb3d8bbwe": _app(1000, 0),      # streaming now
             "SynthChat_abc": _app(900, 950)},              # used, stopped
            {"C:#Tools#SynthCap#synthcap.exe": _app(1200, 0),
             "C:#Tools#Old#old.exe": _app(0, 0)})           # never used
        self.assertEqual(sorted(cb.webcam_users_now(reg=reg)),
                         ["SynthMeet", "synthcap.exe"])

    def test_jarvis_itself_is_not_another_app(self):
        here = os.path.join("C:\\", "SynthPython")
        reg = self._reg({}, {"C:#SynthPython#pythonw.exe": _app(1000, 0)})
        self.assertEqual(cb.webcam_users_now(exclude_dir=here, reg=reg), [])

    def test_nobody_using_a_webcam_is_an_empty_list_not_none(self):
        reg = self._reg({"SynthMeet_x": _app(10, 20)})
        self.assertEqual(cb.webcam_users_now(reg=reg), [])

    def test_an_unreadable_log_is_none(self):
        self.assertIsNone(cb.webcam_users_now(reg=_FakeReg({})))

    def test_off_windows_it_is_none(self):
        with mock.patch.object(cb.os, "name", "posix"):
            self.assertIsNone(cb.webcam_users_now())


class _Cap:
    def __init__(self, log, frames=True):
        self.log = log
        self._frames = frames

    def isOpened(self):
        return True

    def set(self, prop, val):
        self.log.append(("set", prop, val))
        return True

    def read(self):
        self.log.append(("read",))
        if self._frames:
            return True, type("F", (), {"size": 1})()
        return False, None

    def release(self):
        self.log.append(("release",))


class _Cv2:
    CAP_MSMF = 1400
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FOURCC = 6

    def __init__(self, frames=True, opened=True):
        self.log: list = []
        self._frames = frames
        self._opened = opened

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return "".join(chars)

    def VideoCapture(self, idx, api):
        cap = _Cap(self.log, self._frames)
        if not self._opened:
            cap.isOpened = lambda: False
        return cap


class OpenOnceInTheFinalFormatTests(unittest.TestCase):

    def test_mjpg_then_width_then_height_all_before_the_first_read(self):
        cv2 = _Cv2()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_CAMERA_FOURCC", None)
            cap = cb.open_camera(0, backend="msmf", width=1280, height=720,
                                 require_frame=0.5, cv2_mod=cv2)
        self.assertIsNotNone(cap)
        first_read = cv2.log.index(("read",))
        sets = [e for e in cv2.log[:first_read] if e[0] == "set"]
        self.assertEqual([e[1] for e in sets][:3],
                         [cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH,
                          cv2.CAP_PROP_FRAME_HEIGHT])
        self.assertEqual(sets[0][2], "MJPG")
        self.assertFalse(any(e[0] == "set" for e in cv2.log[first_read:]),
                         "a format change after the first read restarts the "
                         "stream")

    def test_the_escape_hatch_turns_the_format_request_off(self):
        cv2 = _Cv2()
        with mock.patch.dict(os.environ, {"JARVIS_CAMERA_FOURCC": "none"}):
            cb.open_camera(0, backend="msmf", width=640, height=480,
                           cv2_mod=cv2)
        self.assertNotIn(cv2.CAP_PROP_FOURCC,
                         [e[1] for e in cv2.log if e[0] == "set"])

    def test_no_size_no_format_request(self):
        cv2 = _Cv2()
        cb.open_camera(0, backend="msmf", cv2_mod=cv2)
        self.assertFalse([e for e in cv2.log if e[0] == "set"
                          and e[1] == cv2.CAP_PROP_FOURCC])


class OutcomeTests(unittest.TestCase):

    def test_opened(self):
        oc: dict = {}
        cb.open_camera(0, backend="msmf", cv2_mod=_Cv2(), outcome=oc)
        self.assertEqual(oc, {"result": "opened"})

    def test_opened_but_no_frame_is_the_held_shape(self):
        oc: dict = {}
        self.assertIsNone(cb.open_camera(0, backend="msmf", require_frame=0.05,
                                         cv2_mod=_Cv2(frames=False),
                                         outcome=oc))
        self.assertEqual(oc["result"], "no-frame")

    def test_not_opened_is_the_vanished_shape(self):
        oc: dict = {}
        self.assertIsNone(cb.open_camera(0, backend="msmf", retry_sleep=0.0,
                                         cv2_mod=_Cv2(opened=False),
                                         outcome=oc))
        self.assertEqual(oc["result"], "not-opened")


if __name__ == "__main__":
    unittest.main()
