"""core/camera_backend.py — the backend choice and the index translation.

WHAT THIS FILE IS DEFENDING

v2.0.101 removed the DirectShow *enumeration* leak by gating it. The *open*
leaks too, and no gate can help there because the face-track loop has to open
the camera. Measured on the owner's rig 2026-09-05, 25 open/read/release cycles
per figure, each in a fresh process, 1280x720 requested, 30 reads per cycle:

    eMeet C960   CAP_DSHOW  +4.58 OS threads  +479 handles  per cycle
    eMeet C960   CAP_MSMF   -0.21 OS threads    +0.97       per cycle
    USB 2.0 Cam  CAP_DSHOW  +4.55 OS threads  +479 handles  per cycle
    USB 2.0 Cam  CAP_MSMF   -0.18 OS threads    +1.06       per cycle

So the code moved to Media Foundation. The DANGEROUS half of that move is not
the backend constant, it is the INDEX: the two backends enumerate different
device lists, of different lengths.

    DirectShow  0=USB 2.0 Camera  1=Kinect  2=eMeet C960  3=OBS Virtual Camera
    MediaFdn.   0=Kinect          1=USB 2.0 Camera  2=eMeet C960

A naive `CAP_DSHOW -> CAP_MSMF` edit repoints index 0 from the USB webcam to
the Kinect AND SUCCEEDS — the worst possible failure, because nothing errors.
The tests below pin that this cannot happen silently.

DELIBERATELY NOT ASSERTED: thread counts, handle counts, or this machine's
actual device list. Those are facts about a rig, not about the code, and a test
that asserts them is flaky by construction. What is asserted is the DECISION —
which index and which backend the code chooses given a device layout, and what
it does when it cannot decide honestly.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from core import camera_backend as cbk


# The owner's rig, as measured 2026-09-05. Used as the layout fixture because a
# regression that only shows up on a machine where the two lists disagree is
# exactly the regression worth pinning.
DSHOW_NAMES = ["USB 2.0 Camera", "Kinect V2 Video Sensor",
               "HD Webcam eMeet C960", "OBS Virtual Camera"]
MSMF_NAMES = ("Kinect V2 Video Sensor", "USB 2.0 Camera",
              "HD Webcam eMeet C960")


class _FakeCapture:
    """Minimal cv2.VideoCapture work-alike.

    ``frames`` is how many reads succeed; 0 reproduces the measured MSMF
    busy-device shape (opens, accepts set(), never delivers)."""

    def __init__(self, opened=True, frames=10 ** 9):
        self._opened = opened
        self._frames = frames
        self.released = 0
        self.sets = []

    def isOpened(self):
        return self._opened

    def set(self, prop, val):
        self.sets.append((prop, val))
        return True

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, _FakeFrame()

    def release(self):
        self.released += 1


class _FakeFrame:
    size = 100


class _FakeCv2:
    CAP_MSMF = 1400
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, *captures):
        self._queue = list(captures)
        self.calls = []

    def VideoCapture(self, index, api):
        self.calls.append((index, api))
        return self._queue.pop(0) if self._queue else _FakeCapture()


def _patch_msmf(names=MSMF_NAMES):
    return mock.patch.object(cbk, "msmf_device_names",
                             side_effect=lambda *a, **k: names)


class TranslationTests(unittest.TestCase):
    """The index a backend is handed must name the camera the caller meant."""

    def test_name_match_wins_and_needs_no_dshow_enumeration(self):
        # The whole point of matching by name: the DirectShow enumeration is
        # the leaky primitive, and a named camera never has to touch it.
        with _patch_msmf():
            idx, backend, why = cbk.resolve_capture_target(
                2, name="emeet c960", dshow_names=None, backend="msmf")
        self.assertEqual((idx, backend), (2, "msmf"))
        self.assertIn("emeet c960", why)

    def test_name_match_translates_across_a_disagreeing_index(self):
        # CAMERAS says DirectShow index 0 for the USB webcam. Media Foundation
        # calls that device index 1. Handing 0 to MSMF would open the KINECT.
        with _patch_msmf():
            idx, backend, _ = cbk.resolve_capture_target(
                0, name="usb 2.0 camera", dshow_names=None, backend="msmf")
        self.assertEqual((idx, backend), (1, "msmf"))

    def test_unnamed_index_translates_via_the_dshow_name_list(self):
        with _patch_msmf():
            self.assertEqual(
                cbk.resolve_capture_target(0, dshow_names=DSHOW_NAMES,
                                           backend="msmf")[:2],
                (1, "msmf"))
            self.assertEqual(
                cbk.resolve_capture_target(1, dshow_names=DSHOW_NAMES,
                                           backend="msmf")[:2],
                (0, "msmf"))
            self.assertEqual(
                cbk.resolve_capture_target(2, dshow_names=DSHOW_NAMES,
                                           backend="msmf")[:2],
                (2, "msmf"))

    def test_directshow_only_device_falls_back_to_directshow(self):
        # OBS Virtual Camera exists at DirectShow index 3 and has no Media
        # Foundation presence at all. Opening it on MSMF at index 3 would fail;
        # opening SOME OTHER index would be worse. Fall back to DirectShow at
        # the original index — leaky, but correct.
        with _patch_msmf():
            idx, backend, why = cbk.resolve_capture_target(
                3, dshow_names=DSHOW_NAMES, backend="msmf")
        self.assertEqual((idx, backend), (3, "dshow"))
        self.assertIn("DirectShow-only", why)

    def test_no_dshow_names_and_no_name_fails_closed_to_directshow(self):
        # Nothing to translate FROM. Guessing here is what would open the
        # wrong camera, so the answer is the original index on the old backend.
        with _patch_msmf():
            self.assertEqual(
                cbk.resolve_capture_target(0, dshow_names=None,
                                           backend="msmf")[:2],
                (0, "dshow"))

    def test_media_foundation_unavailable_falls_back_to_directshow(self):
        with mock.patch.object(cbk, "msmf_device_names",
                               side_effect=lambda *a, **k: None):
            idx, backend, why = cbk.resolve_capture_target(
                2, name="emeet c960", dshow_names=DSHOW_NAMES, backend="msmf")
        self.assertEqual((idx, backend), (2, "dshow"))
        self.assertIn("unavailable", why)

    def test_pinned_dshow_backend_never_translates(self):
        # If someone pins the old backend, the index must stay untouched —
        # translating for a backend we are not using is its own wrong-camera bug.
        with _patch_msmf():
            self.assertEqual(
                cbk.resolve_capture_target(0, name="usb 2.0 camera",
                                           dshow_names=DSHOW_NAMES,
                                           backend="dshow")[:2],
                (0, "dshow"))

    def test_index_translation_is_pure_and_never_enumerates_directshow(self):
        # This module must never acquire the leaky list itself; the monolith
        # owns the one gated call site. Passing None must fail closed rather
        # than reach for it.
        with _patch_msmf():
            self.assertIsNone(cbk.msmf_index_for_dshow_index(0, None))
            self.assertIsNone(cbk.msmf_index_for_dshow_index(99, DSHOW_NAMES))
            self.assertIsNone(cbk.msmf_index_for_dshow_index(-1, DSHOW_NAMES))
            self.assertIsNone(cbk.msmf_index_for_dshow_index("x", DSHOW_NAMES))

    def test_unknown_dshow_device_does_not_match_by_accident(self):
        with _patch_msmf():
            self.assertIsNone(
                cbk.msmf_index_for_dshow_index(0, ["Some Other Camera"]))

    def test_a_near_miss_name_still_translates(self):
        # Exact match is the normal case; a suffix difference between the two
        # enumerations must not defeat translation while it is unambiguous.
        with _patch_msmf(("Kinect V2 Video Sensor", "USB 2.0 Camera")):
            self.assertEqual(
                cbk.msmf_index_for_dshow_index(0, ["USB 2.0 Camera (VID_1BCF)"]),
                1)

    def test_an_ambiguous_containment_refuses_to_guess(self):
        # "Cam" is contained in BOTH Media Foundation entries and matches
        # neither exactly. Picking the first would hand back the WRONG PHYSICAL
        # CAMERA and succeed — nothing raises, nothing logs, the tracker just
        # watches the other device. Fail closed; the caller then falls back to
        # DirectShow at the original index, which is leaky but correct.
        with _patch_msmf(("Cam A", "Cam B")):
            self.assertIsNone(cbk.msmf_index_for_dshow_index(0, ["Cam"]))

    def test_an_exact_match_wins_over_an_ambiguous_containment(self):
        # Exactness is checked first, so a device whose name is a prefix of
        # another's still resolves.
        with _patch_msmf(("USB Camera 2", "USB Camera")):
            self.assertEqual(
                cbk.msmf_index_for_dshow_index(0, ["USB Camera"]), 1)

    def test_two_identical_cameras_match_by_occurrence(self):
        """Two of the same webcam produce two identical friendly names, and
        "the first device called X" is then a coin flip. Both enumerations
        list devices in a stable per-name order, so the Nth on one side is the
        Nth on the other."""
        dshow = ["USB 2.0 Camera", "Kinect V2 Video Sensor", "USB 2.0 Camera"]
        msmf = ("Kinect V2 Video Sensor", "USB 2.0 Camera", "USB 2.0 Camera")
        with _patch_msmf(msmf):
            # first "USB 2.0 Camera" on each side
            self.assertEqual(cbk.msmf_index_for_dshow_index(0, dshow), 1)
            # SECOND one — a plain first-match would wrongly return 1 again
            self.assertEqual(cbk.msmf_index_for_dshow_index(2, dshow), 2)
            self.assertEqual(cbk.msmf_index_for_dshow_index(1, dshow), 0)

    def test_more_duplicates_on_dshow_than_msmf_fails_closed(self):
        dshow = ["USB 2.0 Camera", "USB 2.0 Camera"]
        with _patch_msmf(("USB 2.0 Camera",)):
            self.assertEqual(cbk.msmf_index_for_dshow_index(0, dshow), 0)
            # There is no second one to map to; guessing would reuse the first.
            self.assertIsNone(cbk.msmf_index_for_dshow_index(1, dshow))


class BackendSelectionTests(unittest.TestCase):

    def test_default_is_msmf(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_CAMERA_BACKEND", None)
            self.assertEqual(cbk.configured_backend(), "msmf")

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_CAMERA_BACKEND": "dshow"}):
            self.assertEqual(cbk.configured_backend(), "dshow")

    def test_garbage_env_is_ignored_not_obeyed(self):
        with mock.patch.dict(os.environ,
                             {"JARVIS_CAMERA_BACKEND": "ffmpeg"}):
            self.assertIn(cbk.configured_backend(), ("msmf", "dshow"))

    def test_config_default_ships_as_msmf(self):
        # Pins the shipped value, so a silent revert to DirectShow in
        # core/config.py fails a test instead of quietly costing 4.6 OS threads
        # per camera open again.
        from core import config as cfg
        self.assertEqual(getattr(cfg, "CAMERA_BACKEND", None), "msmf")

    def test_backend_name_maps_the_cv2_constants(self):
        self.assertEqual(cbk.backend_name(700), "dshow")
        self.assertEqual(cbk.backend_name(1400), "msmf")


class OpenCameraTests(unittest.TestCase):
    """The two measured MSMF hazards, encoded once."""

    def test_uses_the_requested_backend_constant(self):
        fake = _FakeCv2(_FakeCapture())
        cbk.open_camera(2, backend="msmf", cv2_mod=fake)
        self.assertEqual(fake.calls, [(2, _FakeCv2.CAP_MSMF)])

        fake = _FakeCv2(_FakeCapture())
        cbk.open_camera(2, backend="dshow", cv2_mod=fake)
        self.assertEqual(fake.calls, [(2, _FakeCv2.CAP_DSHOW)])

    def test_transient_open_failure_is_retried_once(self):
        # Measured: MSMF fails ~3-7% of back-to-back reopens of an idle camera,
        # in ~40 ms, and one retry recovered 120 out of 120.
        good = _FakeCapture()
        fake = _FakeCv2(_FakeCapture(opened=False), good)
        with mock.patch.object(cbk.time, "sleep"):
            cap = cbk.open_camera(2, backend="msmf", cv2_mod=fake, retries=1)
        self.assertIs(cap, good)
        self.assertEqual(len(fake.calls), 2)

    def test_a_persistent_open_failure_still_gives_up(self):
        fake = _FakeCv2(_FakeCapture(opened=False), _FakeCapture(opened=False))
        with mock.patch.object(cbk.time, "sleep"):
            self.assertIsNone(
                cbk.open_camera(2, backend="msmf", cv2_mod=fake, retries=1))
        self.assertEqual(len(fake.calls), 2)

    def test_retries_zero_means_one_attempt(self):
        fake = _FakeCv2(_FakeCapture(opened=False))
        self.assertIsNone(
            cbk.open_camera(2, backend="msmf", cv2_mod=fake, retries=0))
        self.assertEqual(len(fake.calls), 1)

    def test_opened_but_frameless_is_rejected_and_released(self):
        # THE BUSY-DEVICE SHAPE. Measured with one process holding the eMeet:
        # a DirectShow contender got isOpened() == False, but an MSMF contender
        # got isOpened() True, set() True, get() reading back 1280x720 — and
        # 0 frames out of 20. Trusting isOpened() there hands the face-track
        # loop a handle that can never produce an image.
        dead = _FakeCapture(frames=0)
        fake = _FakeCv2(dead)
        with mock.patch.object(cbk.time, "sleep"):
            cap = cbk.open_camera(2, backend="msmf", cv2_mod=fake,
                                  require_frame=0.05)
        self.assertIsNone(cap)
        self.assertEqual(dead.released, 1)

    def test_a_frame_within_the_budget_is_accepted(self):
        live = _FakeCapture(frames=1)
        fake = _FakeCv2(live)
        cap = cbk.open_camera(2, backend="msmf", cv2_mod=fake,
                              require_frame=1.0)
        self.assertIs(cap, live)
        self.assertEqual(live.released, 0)

    def test_require_frame_zero_does_not_read_at_all(self):
        # _probe_camera_index has its own read-until-deadline loop whose budget
        # scales with its timeout; a second nested budget would only confuse
        # the timing, so it must be possible to opt out.
        live = _FakeCapture(frames=0)
        fake = _FakeCv2(live)
        self.assertIs(cbk.open_camera(2, backend="msmf", cv2_mod=fake,
                                      require_frame=0.0), live)

    def test_requested_resolution_is_applied(self):
        live = _FakeCapture()
        fake = _FakeCv2(live)
        cbk.open_camera(2, backend="msmf", cv2_mod=fake,
                        width=1280, height=720)
        self.assertIn((_FakeCv2.CAP_PROP_FRAME_WIDTH, 1280), live.sets)
        self.assertIn((_FakeCv2.CAP_PROP_FRAME_HEIGHT, 720), live.sets)

    def test_failure_paths_release_through_the_hook(self):
        """The monolith runs this on a throwaway worker whose camera I/O lock
        can be RETIRED under it, so a bare cap.release() on a failure path
        would overlap the live threads' camera I/O — the heap corruption the
        locking scheme exists to prevent. Every failure path must go through
        the hook."""
        released = []
        dead = _FakeCapture(frames=0)
        fake = _FakeCv2(dead)
        with mock.patch.object(cbk.time, "sleep"):
            cbk.open_camera(2, backend="msmf", cv2_mod=fake,
                            require_frame=0.05,
                            release_hook=released.append)
        self.assertEqual(released, [dead])
        self.assertEqual(dead.released, 0, "bypassed the hook")

        released.clear()
        refused = _FakeCapture(opened=False)
        fake = _FakeCv2(refused)
        with mock.patch.object(cbk.time, "sleep"):
            cbk.open_camera(2, backend="dshow", cv2_mod=fake,
                            release_hook=released.append)
        self.assertEqual(released, [refused])
        self.assertEqual(refused.released, 0, "bypassed the hook")

    def test_a_raising_hook_cannot_turn_a_failed_open_into_an_exception(self):
        def boom(_cap):
            raise RuntimeError("retired lock is gone")

        fake = _FakeCv2(_FakeCapture(opened=False))
        with mock.patch.object(cbk.time, "sleep"):
            self.assertIsNone(
                cbk.open_camera(2, backend="dshow", cv2_mod=fake,
                                release_hook=boom))

    def test_never_raises(self):
        class Exploding:
            CAP_MSMF = 1400
            CAP_DSHOW = 700

            def VideoCapture(self, *a):
                raise RuntimeError("boom")

        with mock.patch.object(cbk.time, "sleep"):
            self.assertIsNone(cbk.open_camera(0, backend="msmf",
                                              cv2_mod=Exploding()))


class EnumerationTests(unittest.TestCase):

    def test_names_are_ttl_cached(self):
        calls = []

        def fake_raw():
            calls.append(1)
            return list(MSMF_NAMES), ["a", "b", "c"]

        cbk._enum_cache[0] = None
        with mock.patch.object(cbk, "_raw_msmf_devices", fake_raw):
            self.assertEqual(cbk.msmf_device_names(now=1000.0), MSMF_NAMES)
            self.assertEqual(cbk.msmf_device_names(now=1000.0 + 1.0),
                             MSMF_NAMES)
            self.assertEqual(len(calls), 1)
            cbk.msmf_device_names(now=1000.0 + cbk.MSMF_ENUM_TTL_SEC + 1)
            self.assertEqual(len(calls), 2)
        cbk._enum_cache[0] = None

    def test_unavailable_enumeration_is_none_not_empty(self):
        # None means "cannot translate, fall back". An empty tuple would read
        # as "no cameras exist" and silently defeat every name match.
        cbk._enum_cache[0] = None
        with mock.patch.object(cbk, "_raw_msmf_devices", lambda: None):
            self.assertIsNone(cbk.msmf_device_names(now=2000.0))
        cbk._enum_cache[0] = None

    def test_name_lookup_is_case_insensitive_and_substring(self):
        with _patch_msmf():
            self.assertEqual(cbk.msmf_index_for_name("EMEET"), 2)
            self.assertEqual(cbk.msmf_index_for_name("usb 2.0"), 1)
            self.assertIsNone(cbk.msmf_index_for_name("no such camera"))
            self.assertIsNone(cbk.msmf_index_for_name(""))


if __name__ == "__main__":
    unittest.main()
