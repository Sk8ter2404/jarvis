"""The self-diagnostic webcam probe must never open a camera the face-tracking
producer is streaming.

────────────────────────────────────────────────────────────────────────────
THE BUG THIS FILE EXISTS FOR (in-process, measured on the owner's rig
2026-09-06 with JARVIS stopped; CONTROL and PROBE arms identical but for one
extra open, same camera, 30 s windows either side)
────────────────────────────────────────────────────────────────────────────
    CONTROL  producer alone on MSMF index 1 at 1280x720
        before  828 reads / 0 fail / 30.0 fps    during 361 / 0 / 30.1 fps
        after   539 reads / 0 fail / 30.0 fps
    PROBE    same, plus ONE 3.1 s _probe_webcam_locked-shaped scan of 0,1,2
        before  826 reads / 0 fail / 30.0 fps
        during  3,704,096 FAILED reads /  4.4 fps
        after   5,411,405 FAILED reads /  0.0 fps  — and it never recovered

The probe won index 1: the producer's OWN right webcam. Not a slowdown — the
end of the stream, for a device that was perfectly healthy.

Two things made it reachable, and both look like fixes:

  * _camera_io_lock. Taking it proves no other open/release is in flight. It
    does NOT mean the device is free — the producer holds its handles across
    that lock and reads outside it, deliberately (holding it across a blocking
    read is what wedged the subsystem for two hours in v2.0.100). The probe
    treated "I hold the lock" as "the cameras are mine".
  * require_frame. Before it, the scan stopped at MSMF index 0 — the Kinect —
    which opens and never delivers, so it never REACHED a producer-held
    camera. All 70+ diagnostics from 2026-08-20 to 2026-09-05T20:25 recorded
    index 0, and not one is followed by a face-track read failure inside the
    8-17 s window that every post-migration sweep is (those sessions do
    contain read failures; they just fall nowhere near a sweep). Making the
    scan demand a real frame walked it straight onto index 1: every record
    from 2026-09-05T22:39 on says {'index': 1, 'backend': 'msmf'}, and the
    logs show a "[face-track] Right webcam … read failure #25" 8-17 s after
    EVERY one — NINE for nine across three sessions. Once (04:32:56) the scan
    landed on index 2 instead and it was the LEFT webcam that failed, so no
    camera was safe.

require_frame protects the CONTENDER from a useless handle. Nothing protects
the HOLDER, and the harm is done by the open and the reads that precede the
verdict. The only fix is to ask who owns the device first.

WHY THE MIGRATION'S OWN EVIDENCE MISSED IT: core/camera_backend.py records
"Both backends leave the HOLDER undisturbed … so a partial migration is safe".
That was measured ACROSS PROCESSES, where MSMF refuses the contender honestly.
In-process it does not refuse — and in-process is the only configuration JARVIS
runs in.

Asserted here: WHO the probe is allowed to open, and what it says when the
answer is "nobody". Not frame rates, not this rig's device list — those are
facts about a machine, not about the code.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _Cap:
    """cv2.VideoCapture work-alike that opens and delivers."""

    def __init__(self, frames=10 ** 9):
        self._frames = frames
        self.released = 0

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, types.SimpleNamespace(size=10, shape=(480, 640, 3),
                                           mean=lambda: 120.0)

    def release(self):
        self.released += 1


class _Cascade:
    def __init__(self, empty=False):
        self._empty = empty

    def empty(self):
        return self._empty


def _stub_cv2(cascade_empty=False):
    mod = types.ModuleType("cv2")
    mod.CAP_DSHOW = 700
    mod.CAP_MSMF = 1400
    mod.CAP_PROP_FRAME_WIDTH = 3
    mod.CAP_PROP_FRAME_HEIGHT = 4
    mod.CAP_PROP_BUFFERSIZE = 38
    mod.data = types.SimpleNamespace(haarcascades="/cascades/")
    mod.CascadeClassifier = lambda path: _Cascade(cascade_empty)
    return mod


# The owner's real configuration, which is what produced the failure:
#   CAMERAS index 2 = eMeet C960  (left),  index 0 = USB 2.0 Camera (right)
#   Media Foundation order        0=Kinect  1=USB 2.0 Camera  2=eMeet C960
_CAMERAS = [
    {"index": 2, "label": "Left webcam (left monitor)",
     "name": "emeet c960", "primary": True},
    {"index": 0, "label": "Right webcam (top of right monitor)",
     "name": "usb 2.0 camera", "primary": False},
]
_MSMF_FOR_NAME = {"emeet c960": 2, "usb 2.0 camera": 1}


class _FakeBC(types.ModuleType):
    """Stand-in for the bobert_companion monolith, exposing exactly the surface
    the probe is allowed to read: two public accessors and the producer's
    published capture list."""

    def __init__(self, *, stage="loop top", beat_at=1000.0, held=(),
                 frame_ages=None, cameras=None, quarantined=(),
                 frame_means=None):
        super().__init__("bobert_companion")
        import threading
        self._camera_io_lock = threading.RLock()
        self._camera_state_lock = threading.Lock()
        self._camera_latest_frame = {
            idx: types.SimpleNamespace(shape=(720, 1280, 3),
                                       mean=(lambda m=m: m))
            for idx, m in (frame_means or {}).items()}
        self.CAMERAS = list(_CAMERAS if cameras is None else cameras)
        self._stage = stage
        self._beat_at = beat_at
        self._frame_ages = dict(frame_ages or {})
        # dict {config_index: seconds_left} or a bare iterable of indices
        self._quarantined = (dict(quarantined)
                             if isinstance(quarantined, dict)
                             else {i: 0.0 for i in quarantined})
        self._face_track_caps = [
            [{"cam": cam, "cap": (_Cap() if cam["index"] in held else None)}
             for cam in self.CAMERAS]
        ]

    def get_face_track_liveness(self, now=None):
        return {"at": self._beat_at, "stage": self._stage, "iters": 42,
                "age_s": 0.2, "stalled": False}

    def get_camera_health(self):
        import time
        now = time.time()
        return {cam["index"]: {
            "last_frame_at": (0.0 if self._frame_ages.get(cam["index"]) is None
                              else now - self._frame_ages[cam["index"]]),
            "last_read_error": None,
            "quarantined": cam["index"] in self._quarantined,
            "quarantine_until": (now + self._quarantined[cam["index"]]
                                 if cam["index"] in self._quarantined
                                 else 0.0),
        } for cam in self.CAMERAS}


class ProducerOwnershipTests(unittest.TestCase):

    def setUp(self):
        self.mod, _ = load_skill_isolated("self_diagnostic", register=False)
        p = mock.patch.dict(sys.modules, {"cv2": _stub_cv2()})
        p.start()
        self.addCleanup(p.stop)
        p2 = mock.patch("core.camera_backend.configured_backend",
                        return_value="msmf")
        p2.start()
        self.addCleanup(p2.stop)
        p3 = mock.patch("core.camera_backend.msmf_index_for_name",
                        side_effect=lambda n, now=None:
                        _MSMF_FOR_NAME.get((n or "").strip().lower()))
        p3.start()
        self.addCleanup(p3.stop)

    # ── plumbing ────────────────────────────────────────────────────────
    def _install(self, bc):
        p = mock.patch.dict(sys.modules, {"bobert_companion": bc})
        p.start()
        self.addCleanup(p.stop)
        return bc

    def _run_probe(self, answers=(1, 2)):
        """Run the whole probe, recording every index it tried to OPEN.

        ``answers`` are the MSMF indices that would hand back a working
        capture. Index 0 is never among them: on this rig it is the Kinect,
        which opens and then fails every read, so require_frame refuses it."""
        opened = []

        def fake_open(idx, backend, require_frame=0.0):
            opened.append(idx)
            return _Cap() if idx in answers else None

        with mock.patch.object(self.mod, "_open_probe_capture", fake_open):
            res = self.mod._probe_webcam()
        return res, opened

    # ── THE REGRESSION ──────────────────────────────────────────────────
    def test_the_probe_never_opens_a_camera_the_producer_is_streaming(self):
        """The whole point. MSMF index 1 is the producer's right webcam; a
        second handle on it ended the stream at 0 fps and it never came back.

        Before this fix the scan walked 0 (Kinect, refused) then 1 and took
        it — which is exactly what data/self_diagnostic.json recorded, every
        30 minutes, from 2026-09-05T22:39 onward."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 300.0, 2: 300.0}))
        res, opened = self._run_probe()
        self.assertNotIn(1, opened,
                         "opened MSMF index 1 — the producer's right webcam")
        self.assertNotIn(2, opened,
                         "opened MSMF index 2 — the producer's left webcam")

    def test_a_stale_producer_camera_is_reported_unverified_not_opened(self):
        """Owned but silent is the sick case. It must NOT be probed (opening it
        is what kills the stream) and it must NOT be reported as a failure —
        we never tested the device. The producer's own read-failure signal
        already carries the sick case into _run_autoqueue_pass."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 300.0, 2: 300.0}))
        res, opened = self._run_probe()
        self.assertFalse(res["tested"], "claimed to have tested a device it "
                                        "never opened")
        self.assertFalse(res["ok"])
        self.assertIn("UNVERIFIED", res["error"])
        self.assertEqual(res["details"]["unverified_short_cause"],
                         self.mod._UNVERIFIED_SHORT_CAUSES["camera_busy"])
        # Only the Kinect may be touched; the producer's own devices may not.
        self.assertEqual(opened, [0])

    # ── the verdict that costs nothing ──────────────────────────────────
    def test_a_streaming_producer_camera_passes_with_no_device_io_at_all(self):
        """A camera the producer pulled a frame from moments ago is PROVEN
        working — that is a continuous open-and-read test of the exact device
        (measured live 2026-09-06 at 6.4 fps median, 0.347 s worst period),
        which is strictly stronger than one spot check every half hour. So:
        pass, and open nothing."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 0.3, 2: 0.5}))
        res, opened = self._run_probe()
        self.assertTrue(res["ok"], res["error"])
        self.assertTrue(res["tested"])
        self.assertEqual(opened, [], "opened a device it did not need to")
        self.assertIs(res["details"]["device_opened"], False)
        self.assertEqual(res["details"]["verified_via"],
                         "face-track producer telemetry")
        self.assertEqual(res["details"]["cascade"], "loaded")

    def test_the_telemetry_verdict_still_checks_the_face_cascade(self):
        """face_tracker needs the cascade as much as it needs the camera, and
        that check touches no device — so skipping the open must not skip it."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 0.3, 2: 0.5}))
        with mock.patch.dict(sys.modules,
                             {"cv2": _stub_cv2(cascade_empty=True)}):
            res, opened = self._run_probe()
        self.assertFalse(res["ok"])
        self.assertTrue(res["tested"], "a broken cascade is a real failure")
        self.assertIn("cascade", res["error"])
        self.assertEqual(opened, [])

    # ── the checks that must survive not opening the device ─────────────
    def test_the_black_frame_check_survives_the_missing_open(self):
        """A camera that streams perfectly and sees nothing — lens cap, privacy
        shutter, unpowered sensor — is a finding this probe has always made.
        Paying for the ownership fence with a blind spot would be trading one
        silent failure for another, so the verdict judges the producer's OWN
        cached frame instead of one it opened the device to get."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 0.3, 2: 0.2},
                              frame_means={2: 0.0}))
        with mock.patch.object(self.mod.time, "sleep", lambda s: None):
            res, opened = self._run_probe()
        self.assertEqual(opened, [], "opened a device to make this check")
        self.assertFalse(res["ok"])
        self.assertTrue(res["tested"], "a black sensor is a real finding")
        self.assertEqual(res["details"]["failure_mode"],
                         "persistent_black_frame")
        self.assertIs(res["details"]["auto_repairable"], False)
        self.assertEqual(res["severity"], self.mod.SEVERITY_LOW)

    def test_a_bright_producer_frame_passes_and_is_reported(self):
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 0.3, 2: 0.2},
                              frame_means={2: 118.0}))
        res, opened = self._run_probe()
        self.assertTrue(res["ok"], res["error"])
        self.assertEqual(opened, [])
        self.assertEqual(res["details"]["frame_mean"], 118.0)
        self.assertEqual(res["details"]["frame_shape"], [720, 1280, 3])

    def test_no_cached_frame_is_not_a_black_frame(self):
        """Absence of evidence. The producer may not have cached anything yet;
        that must not be reported as a dead sensor."""
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 0.3, 2: 0.2},
                              frame_means={}))
        res, opened = self._run_probe()
        self.assertTrue(res["ok"], res["error"])
        self.assertEqual(opened, [])
        self.assertNotIn("frame_mean", res["details"])

    def test_both_black_frame_verdicts_share_one_threshold(self):
        """Two places now judge 'black'. A threshold fixed in one copy while
        the other rots is this codebase's most expensive bug shape, so there is
        exactly one constant and both read it."""
        import inspect
        body = inspect.getsource(self.mod._probe_webcam_locked)
        # The telemetry verdict and the opened-device verdict each compare
        # twice (first sample, then the retry) and each judges once more at the
        # end, so every branch has to be reading the shared name.
        self.assertGreaterEqual(body.count("_BLACK_FRAME_MEAN_MIN"), 6)
        self.assertGreaterEqual(body.count("_BLACK_FRAME_RETRIES"), 2)
        # and no local literal or shadow constant may survive alongside it
        self.assertNotIn("< 1.0", body)
        self.assertNotIn(">= 1.0", body)
        self.assertNotIn("BLACK_FRAME_RETRIES = ", body)

    def test_it_reports_the_freshest_owned_camera(self):
        self._install(_FakeBC(held=(0, 2), frame_ages={0: 4.0, 2: 0.2}))
        res, _ = self._run_probe()
        self.assertTrue(res["ok"], res["error"])
        self.assertEqual(res["details"]["camera"], "Left webcam (left monitor)")
        self.assertEqual(res["details"]["index"], 2)

    def test_the_reported_index_is_in_the_backend_space_it_names(self):
        """``index`` and ``backend`` have to agree, or a record read months
        later names the wrong physical camera. The owner's right webcam is
        CAMERAS/DirectShow 0 and Media Foundation 1 — pick the eMeet and the
        two numbers coincide, which is precisely why this asserts on the one
        camera where they do NOT."""
        right = [dict(_CAMERAS[1])]
        self._install(_FakeBC(held=(0,), frame_ages={0: 0.2}, cameras=right))
        res, opened = self._run_probe()
        self.assertTrue(res["ok"], res["error"])
        self.assertEqual(opened, [])
        self.assertEqual(res["details"]["backend"], "msmf")
        self.assertEqual(res["details"]["index"], 1, "not an MSMF index")
        self.assertEqual(res["details"]["config_index"], 0)

    # ── a free camera is still a real, device-verified check ────────────
    def test_a_camera_the_producer_does_not_own_is_still_opened(self):
        """The fix is a fence around the producer's devices, not a blanket
        refusal. With only the right webcam configured, MSMF index 2 (the
        eMeet) belongs to nobody and must still be probed for real."""
        one = [dict(_CAMERAS[1])]
        self._install(_FakeBC(held=(0,), frame_ages={0: 300.0}, cameras=one))
        res, opened = self._run_probe()
        self.assertNotIn(1, opened, "opened the producer's own camera")
        self.assertIn(2, opened, "left a free camera unchecked")
        self.assertTrue(res["ok"], res["error"])
        self.assertTrue(res["tested"])
        self.assertEqual(res["details"]["index"], 2)

    # ── a benched camera belongs to nobody ──────────────────────────────
    def test_a_benched_camera_is_still_diagnosed(self):
        """The producer's quarantine gate releases the handle and never opens
        the device again until the bench expires — so for that window the
        camera is genuinely free, and it is the one the diagnostic most wants
        to look at. Fencing it off would trade a collision for a permanently
        blind probe on a single-camera rig whose one camera is sick."""
        one = [dict(_CAMERAS[1])]           # right webcam only, MSMF index 1
        self._install(_FakeBC(held=(), frame_ages={0: 300.0}, cameras=one,
                              quarantined={0: 600.0}))
        res, opened = self._run_probe(answers=(1,))
        self.assertIn(1, opened, "left a benched camera undiagnosed")
        self.assertTrue(res["tested"])

    def test_a_bench_about_to_expire_is_not_treated_as_free(self):
        """If the bench lifts mid-probe the producer's reopen and this open are
        racing again, so the margin has to exceed the scan's worst-case hold."""
        one = [dict(_CAMERAS[1])]
        self._install(_FakeBC(held=(), frame_ages={0: 300.0}, cameras=one,
                              quarantined={0: 5.0}))
        res, opened = self._run_probe(answers=(1,))
        self.assertNotIn(1, opened, "opened a camera whose bench was expiring")
        self.assertFalse(res["tested"])

    def test_the_bench_margin_clears_the_worst_case_scan_hold(self):
        worst_hold = 3 * (0.5 + self.mod._CAMERA_PROBE_WARMUP_S)
        self.assertGreater(self.mod._QUARANTINE_SAFE_MARGIN_S, worst_hold)

    # ── fail closed when the index cannot be placed ─────────────────────
    def test_an_unplaceable_producer_camera_stops_the_sweep(self):
        """If we cannot say WHICH scan index is the producer's camera, every
        index might be it. Guessing wrong costs the owner his primary vision
        until JARVIS restarts; skipping one sweep costs a data point."""
        nameless = [{"index": 0, "label": "Right webcam", "name": ""}]
        self._install(_FakeBC(held=(0,), frame_ages={0: 300.0},
                              cameras=nameless))
        res, opened = self._run_probe()
        self.assertEqual(opened, [], "swept indices it could not rule out")
        self.assertFalse(res["tested"])
        self.assertIn("UNVERIFIED", res["error"])

    def test_an_unknown_backend_is_not_a_licence_to_guess(self):
        self.assertIsNone(self.mod._translate_camera_index(0, "usb 2.0 camera",
                                                           "ffmpeg"))

    # ── ownership is the union of two signals ───────────────────────────
    def test_a_wedged_producer_still_owns_its_cameras(self):
        """A producer that has stopped beating has NOT released anything. The
        heartbeat is a liveness signal, not an ownership one."""
        bc = self._install(_FakeBC(stage="camera 0", held=(0, 2),
                                   frame_ages={0: 300.0, 2: 300.0}))
        bc.get_face_track_liveness = lambda now=None: {
            "at": 1000.0, "stage": "camera 0", "iters": 7,
            "age_s": 900.0, "stalled": True}
        res, opened = self._run_probe()
        self.assertNotIn(1, opened)
        self.assertNotIn(2, opened)

    def test_a_recorded_handle_owns_even_if_the_heartbeat_says_stopped(self):
        """Belt and braces: the two signals are UNIONED, not chosen between."""
        self._install(_FakeBC(stage="stopped", held=(0, 2),
                              frame_ages={0: 300.0, 2: 300.0}))
        own = self.mod._producer_camera_ownership("msmf")
        self.assertTrue(own["owns"])
        self.assertEqual(own["indices"], {1, 2})

    def test_a_producer_between_open_and_record_still_owns_the_device(self):
        """The window where the open has returned but the handle is not yet in
        the published list. The heartbeat covers it — which is why the stage
        signal exists at all."""
        self._install(_FakeBC(stage="reopen camera 0", held=(),
                              frame_ages={0: 300.0, 2: 300.0}))
        own = self.mod._producer_camera_ownership("msmf")
        self.assertTrue(own["owns"])
        self.assertEqual(own["indices"], {1, 2})

    # ── nothing changes when there is no producer ───────────────────────
    def test_with_no_producer_the_scan_is_exactly_what_it_was(self):
        """CI, bare-skill imports, face tracking switched off, and a JARVIS
        whose producer never started all land here. The fence must not cost
        them their check."""
        self._install(_FakeBC(stage="stopped", beat_at=0.0, held=(),
                              frame_ages={0: None, 2: None}))
        res, opened = self._run_probe()
        self.assertEqual(opened, [0, 1], "the sweep changed shape")
        self.assertTrue(res["ok"], res["error"])
        self.assertEqual(res["details"]["index"], 1)

    def test_without_the_monolith_at_all_the_scan_is_unchanged(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("bobert_companion", None)
            res, opened = self._run_probe()
        self.assertEqual(opened, [0, 1])
        self.assertTrue(res["ok"], res["error"])

    def test_ownership_never_raises_on_a_hostile_monolith(self):
        """Every failure here must degrade to 'we know nothing', which leaves
        the old scan behaviour intact. A diagnostic that raises is worse than
        one that shrugs."""
        broken = types.ModuleType("bobert_companion")
        broken.get_face_track_liveness = lambda: 1 / 0
        broken.get_camera_health = lambda: 1 / 0
        broken.CAMERAS = "not a list at all"
        self._install(broken)
        own = self.mod._producer_camera_ownership("msmf")
        self.assertFalse(own["owns"])
        self.assertEqual(own["indices"], set())

    # ── the translation itself ──────────────────────────────────────────
    def test_translation_is_by_name_not_arithmetic_on_the_index(self):
        """CAMERAS holds DirectShow indices. The owner's right webcam is
        DirectShow 0 and Media Foundation 1 — the two lists are different
        lengths and different orders, so only the NAME survives."""
        self.assertEqual(
            self.mod._translate_camera_index(0, "usb 2.0 camera", "msmf"), 1)
        self.assertEqual(
            self.mod._translate_camera_index(2, "emeet c960", "msmf"), 2)

    def test_on_dshow_the_configured_index_is_already_the_right_one(self):
        self.assertEqual(
            self.mod._translate_camera_index(0, "usb 2.0 camera", "dshow"), 0)

    def test_an_unmatched_name_does_not_fall_back_to_the_raw_index(self):
        """Falling back would hand the fence the WRONG index and quietly
        re-open the hole this file exists to close."""
        self.assertIsNone(
            self.mod._translate_camera_index(0, "some other camera", "msmf"))

    # ── the freshness window is a threshold, not a guess ────────────────
    def test_the_freshness_window_cannot_be_tripped_by_one_slow_iteration(self):
        """The producer's worst measured period is ~2.1 s; the dashboard calls
        a tile dead at 5 s and the producer's watchdog calls the loop stalled
        at 30 s. The window has to sit between the two."""
        self.assertGreater(self.mod._PRODUCER_FRAME_FRESH_S, 5.0)
        self.assertLess(self.mod._PRODUCER_FRAME_FRESH_S, 30.0)


if __name__ == "__main__":
    unittest.main()
