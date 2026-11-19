"""core/camera_backend.py pieces added for the 2026-09-29 USB-storm fix.

  * webcam_users_now(): WHO IS USING a webcam, from Windows' camera privacy
    log - the evidence a camera is held. "Which camera apps are RUNNING" was
    wrong 23 of 23 times that day.
  * open_camera(): opens ONCE in the final format, all before the first read.
    MSMF: pixel format (MJPG), width, height. DirectShow (2026-11-17): width,
    height, THEN the pixel format, because a size change resets it to YUY2.
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

    def test_windows_paths_parse_the_same_on_a_posix_host(self):
        # CI 2026-09-29 (v2.0.131/132, Linux runner): the privacy log holds
        # WINDOWS paths, but they were split with os.path = posixpath there, so
        # basename() returned the whole path and the exclude-dir check never
        # matched. Force posixpath here so the Windows box catches it too.
        import posixpath
        with mock.patch.object(cb.os, "path", posixpath):
            reg = self._reg(
                {"SynthMeet_8wekyb3d8bbwe": _app(1000, 0)},
                {"C:#Tools#SynthCap#synthcap.exe": _app(1200, 0),
                 "C:#SynthPython#pythonw.exe": _app(1000, 0)})
            self.assertEqual(sorted(cb.webcam_users_now(exclude_dir=r"C:\SynthPython", reg=reg)),
                             ["SynthMeet", "synthcap.exe"])

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


_YUY2 = "YUY2"


class _RenegotiatingCap(_Cap):
    """A DirectShow capture that models the 2026-11-17 measurement: any
    width/height change renegotiates the media type and resets the format to
    uncompressed YUY2; setting the format afterwards sticks."""

    def __init__(self, log, frames=True):
        super().__init__(log, frames)
        self.fourcc = _YUY2
        self.w, self.h = 640, 480

    def set(self, prop, val):
        self.log.append(("set", prop, val))
        if prop == _Cv2.CAP_PROP_FRAME_WIDTH:
            self.w = int(val)
            self.fourcc = _YUY2
        elif prop == _Cv2.CAP_PROP_FRAME_HEIGHT:
            self.h = int(val)
            self.fourcc = _YUY2
        elif prop == _Cv2.CAP_PROP_FOURCC:
            self.fourcc = val
        return True

    def get(self, prop):
        if prop == _Cv2.CAP_PROP_FOURCC:
            v = 0
            for i, ch in enumerate(self.fourcc):
                v |= ord(ch) << (8 * i)
            return float(v)
        if prop == _Cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.w)
        if prop == _Cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.h)
        return 0.0


class _RenegotiatingCv2(_Cv2):
    def VideoCapture(self, idx, api):
        cap = _RenegotiatingCap(self.log, self._frames)
        self.caps = getattr(self, "caps", []) + [cap]
        return cap


class DirectShowKeepsTheFormatTests(unittest.TestCase):
    """The size change must not cost the requested format (measured 2026-11-17:
    YUY2 at 1280x720 = 3.6-3.8 fps, MJPG = 15 fps in the same light)."""

    def _open(self, backend, env=None):
        cv2 = _RenegotiatingCv2()
        logs: list = []
        with mock.patch.dict(os.environ, env or {}, clear=False):
            if not env:
                os.environ.pop("JARVIS_CAMERA_FOURCC", None)
            cap = cb.open_camera(0, backend=backend, width=1280, height=720,
                                 require_frame=0.5, cv2_mod=cv2,
                                 log=logs.append)
        self.assertIsNotNone(cap)
        return cv2, cap, logs

    def test_dshow_ends_on_mjpg_at_the_requested_size(self):
        cv2, cap, _ = self._open("dshow")
        self.assertEqual(cap.fourcc, "MJPG")
        self.assertEqual((cap.w, cap.h), (1280, 720))

    def test_dshow_sets_size_before_the_format(self):
        cv2, _cap, _ = self._open("dshow")
        props = [e[1] for e in cv2.log if e[0] == "set"]
        self.assertEqual(props[:3], [cv2.CAP_PROP_FRAME_WIDTH,
                                     cv2.CAP_PROP_FRAME_HEIGHT,
                                     cv2.CAP_PROP_FOURCC])
        self.assertLess(cv2.log.index(("set", cv2.CAP_PROP_FOURCC, "MJPG")),
                        cv2.log.index(("read",)))

    def test_msmf_order_is_unchanged(self):
        cv2, _cap, _ = self._open("msmf")
        props = [e[1] for e in cv2.log if e[0] == "set"]
        self.assertEqual(props[:3], [cv2.CAP_PROP_FOURCC,
                                     cv2.CAP_PROP_FRAME_WIDTH,
                                     cv2.CAP_PROP_FRAME_HEIGHT])

    def test_the_negotiated_format_is_logged_once_per_open(self):
        _cv2, _cap, logs = self._open("dshow")
        lines = [m for m in logs if "negotiated" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("MJPG 1280x720", lines[0])
        self.assertNotIn("NOT HONOURED", lines[0])

    def test_the_escape_hatch_leaves_the_driver_format(self):
        cv2, cap, logs = self._open("dshow", {"JARVIS_CAMERA_FOURCC": "none"})
        self.assertEqual(cap.fourcc, _YUY2)
        self.assertNotIn(cv2.CAP_PROP_FOURCC,
                         [e[1] for e in cv2.log if e[0] == "set"])
        self.assertNotIn("NOT HONOURED", " ".join(logs))

    def test_a_cap_without_get_still_opens(self):
        cv2 = _Cv2()              # the plain fake has no get()
        cap = cb.open_camera(0, backend="dshow", width=640, height=480,
                             cv2_mod=cv2, log=lambda m: None)
        self.assertIsNotNone(cap)


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
