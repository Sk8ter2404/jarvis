"""The webcam probe in skills/self_diagnostic.py must use ONE backend, and it
must be the non-leaking one.

────────────────────────────────────────────────────────────────────────────
THE BUG THIS FILE EXISTS FOR (measured live 2026-09-05, JARVIS stopped,
cameras idle, owner's rig)
────────────────────────────────────────────────────────────────────────────
_probe_webcam_locked() found a camera with a BACKEND-LESS ``cv2.VideoCapture(idx)``
— which resolves to MSMF on Windows; getBackendName() returned 'MSMF' — and
then handed the winning integer to _attempt_camera_wake(), which opened it with
``cv2.CAP_DSHOW``. The two backends do not enumerate the same device list:

    Media Foundation  0=Kinect V2 Video Sensor  1=USB 2.0 Camera  2=eMeet C960
    DirectShow        0=USB 2.0 Camera  1=Kinect V2 Video Sensor  2=eMeet C960
                      3=OBS Virtual Camera

So "index 0" meant the Kinect to the scan and the USB webcam to the wake. The
probe diagnosed one device and then tried to revive a different one.

It got worse, because the scan accepted the first index that merely
``isOpened()``. Under MSMF that is index 0, the Kinect's Media Foundation
interface, which opens at 512x424 and then fails every read
(``OnReadSample() ... error status: -2147024809``) — inherently, with nothing
else holding it. Reproduced directly:

    cv2.VideoCapture(0)  opened=True  backend=MSMF  0.23s
        -> first-two-reads ok=False frame=None  W=512.0 H=424.0

So the scan ALWAYS picked a camera that can never produce a frame, ALWAYS fell
through to the soft-wake branch, and the DirectShow open inside that branch is
what leaked +3 OS threads and ~+309 handles every 30 minutes — 0.101 of the
0.110 threads/min residual left after v2.0.101.

Asserted here: WHICH BACKEND, and that both halves agree. Not thread counts,
not this machine's device list — those are facts about a rig, not the code.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _Cap:
    """cv2.VideoCapture work-alike. ``frames`` == 0 is the shape of BOTH the
    Kinect's Media Foundation interface and any device another process holds:
    opens fine, delivers nothing."""

    def __init__(self, opened=True, frames=10 ** 9):
        self._opened = opened
        self._frames = frames
        self.released = 0

    def isOpened(self):
        return self._opened

    def set(self, *a):
        return True

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, types.SimpleNamespace(size=10, shape=(480, 640, 3))

    def release(self):
        self.released += 1


def _stub_cv2():
    """A cv2 stand-in carrying only the constants this code names.

    _attempt_camera_wake starts with ``import cv2`` as an availability check
    and bails with "opencv unavailable" if it fails, so on the light-deps CI
    runner (which blocks cv2) every wake test would assert against that message
    instead of the behaviour under test. The device itself is always mocked
    here; only the import has to succeed."""
    mod = types.ModuleType("cv2")
    mod.CAP_DSHOW = 700
    mod.CAP_MSMF = 1400
    mod.CAP_PROP_FRAME_WIDTH = 3
    mod.CAP_PROP_FRAME_HEIGHT = 4
    mod.CAP_PROP_BUFFERSIZE = 38
    return mod


class SelfDiagnosticCameraBackendTests(unittest.TestCase):

    def setUp(self):
        self.mod, _ = load_skill_isolated("self_diagnostic", register=False)
        patcher = mock.patch.dict(sys.modules, {"cv2": _stub_cv2()})
        patcher.start()
        self.addCleanup(patcher.stop)

    # ── the backend the probe uses ──────────────────────────────────────
    def test_probe_backend_is_msmf_by_default(self):
        self.assertEqual(self.mod._camera_backend_name(), "msmf")

    def test_probe_backend_follows_the_shared_config(self):
        with mock.patch("core.camera_backend.configured_backend",
                        return_value="dshow"):
            self.assertEqual(self.mod._camera_backend_name(), "dshow")

    def test_probe_open_goes_through_the_shared_opener(self):
        cap = _Cap()
        with mock.patch("core.camera_backend.open_camera",
                        return_value=cap) as oc:
            got = self.mod._open_probe_capture(2, "msmf")
        self.assertIs(got, cap)
        oc.assert_called_once()
        self.assertEqual(oc.call_args.kwargs["backend"], "msmf")

    def test_the_scan_proves_a_frame_but_the_wake_does_not_have_to(self):
        """Where the frame proof belongs, and where it would be waste.

        The SCAN must prove one: it picks the first index that works, and under
        MSMF index 0 on this rig is the Kinect's Media Foundation interface,
        which opens and then fails every read. Without the proof the scan picks
        that device every time and never reaches a real webcam.

        The WAKE must not: it reads the device itself and that read is its
        verdict, so proving a frame inside the open would consume one to learn
        the same thing twice."""
        seen = []
        with mock.patch.object(self.mod, "_open_probe_capture",
                               lambda idx, backend, require_frame=0.0:
                               seen.append((idx, require_frame)) or _Cap()):
            self.mod._attempt_camera_wake(1, backend="msmf")
        self.assertEqual(seen[-1][1], 0.0, "the wake should not double-prove")

        import inspect
        scan = inspect.getsource(self.mod._probe_webcam_locked)
        self.assertIn("require_frame=_CAMERA_PROBE_WARMUP_S", scan)
        self.assertGreater(self.mod._CAMERA_PROBE_WARMUP_S, 0)

    def test_the_scan_budget_cannot_starve_the_face_tracker(self):
        """The scan runs holding bobert_companion._camera_io_lock, and the
        face-track producer waits on that lock. Worst case is all three indices
        opening and none delivering: 3 x (open + budget). An MSMF open with
        1280x720 set on it measured ~0.5 s, so the whole hold must stay well
        under the producer's 30 s stall warning — a probe that trips that
        warning makes a healthy producer look wedged, which is the exact class
        of false diagnosis this file is about."""
        worst_open_s = 0.5
        worst_hold = 3 * (worst_open_s + self.mod._CAMERA_PROBE_WARMUP_S)
        self.assertLess(worst_hold, 10.0,
                        "the scan can hold the camera lock for %.1fs" % worst_hold)

    # ── the regression: one index space, not two ────────────────────────
    def test_wake_uses_the_backend_it_is_given(self):
        seen = {}

        def fake_open(idx, backend, require_frame=0.0):
            seen["idx"], seen["backend"] = idx, backend
            return _Cap()

        with mock.patch.object(self.mod, "_open_probe_capture", fake_open):
            ok, note = self.mod._attempt_camera_wake(1, backend="msmf")
        self.assertTrue(ok)
        self.assertEqual(seen, {"idx": 1, "backend": "msmf"})
        self.assertIn("produced a frame", note)

    def test_wake_defaults_to_the_configured_backend_not_directshow(self):
        # The old signature hard-wired cv2.CAP_DSHOW here. Defaulting to
        # DirectShow IS the bug: it is the half that disagreed with the scan.
        seen = {}

        def fake_open(idx, backend, require_frame=0.0):
            seen["backend"] = backend
            return _Cap()

        with mock.patch.object(self.mod, "_open_probe_capture", fake_open):
            self.mod._attempt_camera_wake(0)
        self.assertEqual(seen["backend"], "msmf")

    def test_wake_reports_failure_when_the_open_is_refused(self):
        with mock.patch.object(self.mod, "_open_probe_capture",
                               lambda idx, backend, require_frame=0.0: None):
            ok, note = self.mod._attempt_camera_wake(0, backend="msmf")
        self.assertFalse(ok)
        self.assertIn("refused open", note.lower())
        # The backend is named, because on MSMF a refusal means something
        # different from a DirectShow refusal and the note has to survive being
        # read months later.
        self.assertIn("msmf", note.lower())

    def test_wake_fails_when_the_device_opens_but_delivers_nothing(self):
        # The MSMF busy-device shape reaches the wake as an OPEN that succeeds
        # and reads that do not. isOpened() cannot see the difference; the read
        # can, and it is the read that decides.
        with mock.patch.object(self.mod, "_open_probe_capture",
                               lambda idx, backend, require_frame=0.0:
                               _Cap(frames=0)):
            ok, note = self.mod._attempt_camera_wake(0, backend="msmf")
        self.assertFalse(ok)
        self.assertIn("no frame", note.lower())

    def test_wake_never_opens_with_directshow_when_msmf_is_configured(self):
        # The end-to-end shape of the regression: drive the real
        # _open_probe_capture with a fake cv2 and assert the API constant.
        calls = []

        class FakeCv2:
            CAP_DSHOW = 700
            CAP_MSMF = 1400

            def VideoCapture(self, idx, api=None):
                calls.append((idx, api))
                return _Cap()

        # Force the core-less fallback path so the assertion is about the
        # skill's own choice of constant, not core/camera_backend's.
        with mock.patch.dict(sys.modules,
                             {"core.camera_backend": None}), \
                mock.patch.dict(sys.modules, {"cv2": FakeCv2()}):
            self.mod._open_probe_capture(0, "msmf")
        self.assertEqual(calls, [(0, 1400)])
        self.assertNotIn(700, [c[1] for c in calls])


if __name__ == "__main__":
    unittest.main()
