"""The camera open gate, wired into the monolith (2026-09-29).

LIVE EVIDENCE THIS FILE PINS. Windows logged the top hub of the owner's chain
of three cascaded powered USB hubs "surprise removed ... Count of devices
removed: 20" (Kernel-PnP event 1010) about once a minute for half an hour,
and ONLY while JARVIS was reopening cameras: zero resets while the
camera-preview producer sat wedged (so nothing opened a camera), zero in the
11+ minutes after JARVIS stopped. The session log for that window showed the
producer re-opening a failing webcam every few seconds, a lock hint blaming a
meeting app, an MSMF -> DirectShow fallback on a device that had just left the
bus, a boot probe wedging, and "the HUD camera tile is going stale" 16 times
in ten minutes.

Each test drives the REAL code path - the producer loop, _camera_open,
_probe_camera_index, _open_tile_capture, the audio device accessors - with
fake captures, a fake capture backend and a frozen clock. Nothing opens a
camera, the Kinect or an audio device. Device and app names are synthetic.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import time as _real_time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeTime:
    """Stands in for the monolith's `time` module ONLY (patched as bc.time):
    a frozen wall clock the test advances, a sleep that never blocks the
    producer under test, and everything else from the real module."""

    def __init__(self, t0: float = 10_000.0):
        self.t = t0

    def time(self):
        return self.t

    def sleep(self, _s):
        return None

    def advance(self, s: float) -> None:
        self.t += s

    def __getattr__(self, name):
        return getattr(_real_time, name)


class _Cap:
    """cv2.VideoCapture work-alike for the producer loop. Delivers frames while
    ``alive(now)`` is true, then fails every read."""

    def __init__(self, clock, alive=lambda now: True):
        self._clock = clock
        self._alive = alive
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, _prop):
        return 1280.0

    def set(self, *_a):
        return True

    def read(self):
        import numpy as np
        self.reads += 1
        if self._alive(self._clock.t):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


class _IterStop:
    """A stop event that lets ``n`` producer iterations through."""

    def __init__(self, n: int):
        self.left = n

    def is_set(self):
        if self.left <= 0:
            return True
        self.left -= 1
        return False


class _FakeBackend:
    """core.camera_backend stand-in: a named camera resolves on Media
    Foundation, every open is scripted by call number, ``names()`` is the
    Media Foundation device list (what the gate's presence check reads) and
    ``users()`` is Windows' camera privacy log (who is using a webcam now)."""

    def __init__(self, clock, script=None, resolve=None, names=None,
                 users=None):
        self._clock = clock
        self._script = script or (lambda n: (_ for _ in ()).throw(
            AssertionError("this test must not reach a real open")))
        self._resolve = resolve
        self._names = names or (lambda: ("SynthCam One", "SynthCam Two"))
        self._users = users or (lambda: [])
        self.calls: list = []

    def configured_backend(self):
        return "msmf"

    def msmf_device_names(self, *a, **k):
        return tuple(self._names())

    def webcam_users_now(self, exclude_dir=None, **k):
        return list(self._users())

    def resolve_capture_target(self, idx, name=None, dshow_names=None, **k):
        if self._resolve is not None:
            return self._resolve(len(self.calls) + 1, idx, name)
        return 0, "msmf", "matched CAMERAS name %r at msmf index 0" % (name,)

    def open_camera(self, idx, *, backend="msmf", outcome=None, **kw):
        self.calls.append((self._clock.t, backend))
        result, cap = self._script(len(self.calls))
        if isinstance(outcome, dict):
            outcome["result"] = result
        return cap


_CAM_A = {"index": 41, "label": "Synth left", "name": "synthcam one",
          "primary": True, "look_x": 0.5, "look_y": 0.5}
_CAM_B = {"index": 42, "label": "Synth right", "name": "synthcam two",
          "primary": False, "look_x": 0.85, "look_y": 0.5}


@requires_monolith
class _StormBase(MonolithGlobalsTestCase):

    _STATE = ("_camera_latest_frame", "_camera_last_frame_at",
              "_camera_last_read_error", "_camera_last_read_error_at",
              "_camera_wake_attempts", "_camera_recoveries", "_camera_read_ms",
              "_camera_seen_at")

    def setUp(self):
        bc = self.bc
        saved = {n: dict(getattr(bc, n)) for n in self._STATE
                 if isinstance(getattr(bc, n, None), dict)}

        def _restore():
            for n, v in saved.items():
                d = getattr(bc, n)
                d.clear()
                d.update(v)
        self.addCleanup(_restore)
        for n in saved:
            getattr(bc, n).clear()
        cache = list(bc._camera_locker_cache)
        self.addCleanup(lambda: bc._camera_locker_cache.__setitem__(
            slice(None), cache))
        bc._camera_locker_cache[:] = [0.0, []]
        for ev in (bc._face_track_pause, bc._face_track_camera_off):
            ev.clear()
        bc._standby_mode[0] = False
        self.clock = _FakeTime()
        # getattr, not attribute access: on a tree WITHOUT the gate these tests
        # must still run - and fail on what the code DOES, not on a missing
        # name (that is how the fix is proven against the old code).
        _mk = getattr(bc, "_make_camera_gate", None)
        self.gate = _mk(clock=self.clock.time) if _mk else None
        self.spoken: list = []
        self.lockers = lambda: []
        # A storm trip SPEAKS: the gate's announce is _usb_storm_announce ->
        # proactive_announce, which writes the LIVE project-root
        # pending_speech.json (it is bound to the monolith's __file__, so no
        # env redirect reaches it). Tests that assert on the line patch
        # proactive_announce themselves, inside this default; any other trip
        # lands here instead of in the owner's queue (found 2026-09-30 by a
        # write audit: three storm tests queued real announcements).
        self.stray_announcements: list = []
        _pa = mock.patch.object(
            bc, "proactive_announce",
            side_effect=lambda m, *a, **k:
            self.stray_announcements.append(m) or True)
        _pa.start()
        self.addCleanup(_pa.stop)

    def _producer(self, cams, *, iterations, opener=None, step=None,
                  bounded_real=False, backend=None):
        """Run the REAL producer body. Returns captured stdout."""
        bc = self.bc
        clock = self.clock
        step = step or (lambda: 0.1)

        def _beat_iteration(now=None):
            clock.advance(step())

        def _bounded(idx, opener_fn, *a, **k):
            return opener(idx)

        buf = io.StringIO()
        patches = [
            mock.patch.object(bc, "CAMERAS", [dict(c) for c in cams]),
            mock.patch.object(bc, "_face_track_stop", _IterStop(iterations)),
            mock.patch.object(bc, "time", clock),
            mock.patch.object(bc, "_camera_gate", self.gate, create=True),
            mock.patch.object(bc, "_face_track_beat_iteration", _beat_iteration),
            mock.patch.object(bc, "_dshow_name_to_index",
                              side_effect=lambda n: next(
                                  c["index"] for c in cams if c["name"] == n)),
            mock.patch.object(bc, "_hud_camera_preview_enabled",
                              return_value=False),
            mock.patch.object(bc, "_hud_camera_preview_remove"),
            mock.patch.object(bc, "_release_side_tile_webcams_if_open"),
            mock.patch.object(bc, "_detect_face", return_value=None),
            mock.patch.object(bc, "send"),
            mock.patch.object(bc, "find_camera_locking_processes",
                              side_effect=lambda: list(self.lockers())),
            mock.patch.object(bc, "proactive_announce",
                              side_effect=lambda m, *a, **k:
                              self.spoken.append((clock.t, m)) or True),
        ]
        if not bounded_real:
            patches.append(mock.patch.object(bc, "_open_capture_bounded",
                                             side_effect=_bounded))
        # ALWAYS a fake device list: the gate's presence check must not read
        # this machine's real one, which has never heard of a synthetic camera
        # and would call every one of them ABSENT.
        backend = backend or _FakeBackend(clock)
        patches.append(mock.patch.object(bc, "_camera_backend", backend))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(buf))
            bc._face_tracking_thread_body()
        return buf.getvalue()

    def _held_step(self):
        """0.1 s per iteration while the producer holds a handle (reads are
        what advance a failure episode), 5 s while it holds none."""
        caps = self.bc._face_track_caps[0] or []
        held = any(isinstance(e, dict) and e.get("cap") is not None
                   for e in caps)
        return 0.1 if held else 5.0


class ReopenStormIsGoneTests(_StormBase):
    """A camera that keeps failing is retried 30 -> 60 -> 120 -> 300 s apart,
    not every 2 s."""

    def test_a_failing_camera_backs_off_instead_of_hammering_the_bus(self):
        t0 = self.clock.t
        opens: list = []

        def _open(idx):
            opens.append(self.clock.t)
            if len(opens) == 1:          # streams one second, then dies
                return _Cap(self.clock, alive=lambda now: now < t0 + 1.0)
            return None                  # every reopen fails
        self._producer([_CAM_A], iterations=260, opener=_open,
                       step=self._held_step)
        self.assertGreater(self.clock.t - t0, 560.0, "the run was too short")
        self.assertLessEqual(
            len(opens), 6,
            f"{len(opens)} opens of a dead camera in "
            f"{self.clock.t - t0:.0f}s - that is the reopen storm")
        gaps = [b - a for a, b in zip(opens[2:], opens[3:])]
        for got, want in zip(gaps, (60.0, 120.0, 300.0)):
            self.assertGreaterEqual(got, want - 0.01, gaps)
        self.assertGreaterEqual(opens[2] - opens[1], 30.0 - 0.01)


class UsbStormBreakerTests(_StormBase):
    """A hub reset takes every camera at once: the breaker opens, NOTHING is
    reopened for the cool-down, the owner hears about it once, and after it
    the cameras come back one at a time."""

    def test_two_cameras_dropping_together_stop_every_open_for_ten_minutes(self):
        t0 = self.clock.t
        reset_at = t0 + 20.0
        opens: list = []

        def _open(idx):
            opens.append((self.clock.t, idx))
            n_for_idx = sum(1 for _t, i in opens if i == idx)
            if n_for_idx == 1:
                return _Cap(self.clock, alive=lambda now: now < reset_at)
            return _Cap(self.clock)      # the bus is fine again later
        out = self._producer([_CAM_A, _CAM_B], iterations=700, opener=_open,
                             step=self._held_step)
        storm_lines = [ln for ln in out.splitlines() if "[usb-storm]" in ln
                       and "treating it as a USB bus event" in ln]
        self.assertEqual(len(storm_lines), 1, out[-3000:])
        self.assertIn("2 cameras dropped", storm_lines[0])
        self.assertEqual(len(self.spoken), 1, self.spoken)
        trip_t, said = self.spoken[0]
        self.assertEqual(said, "Sir, the USB bus looks unstable, so I'm "
                               "leaving the cameras alone for ten minutes.")
        during = [(t, i) for t, i in opens if trip_t <= t < trip_t + 600.0]
        self.assertEqual(during, [], "a camera was opened during the cool-down")
        after = [(t, i) for t, i in opens if t >= trip_t + 600.0]
        self.assertEqual(sorted(i for _t, i in after), [41, 42],
                         "the cameras did not come back after the cool-down")
        self.assertGreaterEqual(abs(after[1][0] - after[0][0]), 3.0 - 0.01,
                                "the post-storm reopens were not staggered")
        self.assertEqual(sum(1 for ln in out.splitlines()
                             if "cool-down over" in ln), 1)


class LockedCameraTests(_StormBase):
    """A camera ANOTHER APP IS USING - per Windows' camera privacy log, not per
    "is that app running" - is not retried on a timer: it is reopened once the
    app stops using a webcam (or after 10 min at most)."""

    def test_no_timer_retries_while_another_app_uses_the_camera(self):
        t0 = self.clock.t
        grab_at, done_at = t0 + 10.0, t0 + 200.0
        # The privacy log: a meeting app streams from grab_at to done_at.
        users = lambda: (["SynthMeet"] if grab_at <= self.clock.t < done_at
                         else [])

        def _script(n):
            if n == 1:
                return "opened", _Cap(self.clock,
                                      alive=lambda now: now < grab_at)
            if self.clock.t < done_at:
                return "no-frame", None      # MSMF: held by the other app
            return "opened", _Cap(self.clock)
        backend = _FakeBackend(self.clock, _script, users=users)
        out = self._producer([_CAM_A], iterations=240, backend=backend,
                             bounded_real=True, step=self._held_step)
        self.assertGreater(self.clock.t, done_at, "the run was too short")
        times = [t for t, _b in backend.calls]
        self.assertEqual(len(times), 3,
                         f"opens at {[round(t - t0, 1) for t in times]}: a "
                         f"held camera was retried on a timer")
        self.assertGreaterEqual(times[2], done_at)
        self.assertLess(times[2], done_at + 15.0,
                        "the camera was not reopened promptly once the other "
                        "app had stopped using it")
        self.assertEqual(out.count("appears to be IN USE"), 1)
        self.assertIn("NOT retrying it on a timer", out)
        self.assertIn("SynthMeet is no longer using a webcam", out)

    def test_a_merely_running_meeting_app_is_not_a_lock(self):
        """THE 23-OF-23 CASE (2026-09-29). The old hint said "appears LOCKED by
        <meeting app>, <chat app>" 23 times in one day because both were
        RUNNING; the privacy log showed neither had used a camera. Here the
        open fails exactly like a held device ("opened, no frame"), the
        meeting app is running - and nobody is using a webcam. That is NOT a
        lock: the camera stays on the ordinary 30 -> 60 -> ... s ladder."""
        t0 = self.clock.t
        self.lockers = lambda: ["SynthMeet.exe", "SynthChat.exe"]   # running

        def _script(n):
            if n == 1:
                return "opened", _Cap(self.clock,
                                      alive=lambda now: now < t0 + 5.0)
            return "no-frame", None
        backend = _FakeBackend(self.clock, _script, users=lambda: [])
        out = self._producer([_CAM_A], iterations=170, backend=backend,
                             bounded_real=True, step=self._held_step)
        self.assertNotIn("LOCKED", out)
        self.assertNotIn("IN USE", out)
        self.assertEqual(self.gate.locked_by("name:synthcam one"), [])
        times = [t for t, _b in backend.calls]
        self.assertGreaterEqual(len(times), 3,
                                "the camera was parked instead of backing off")

    def test_a_plain_refused_open_is_not_blamed_on_a_running_chat_app(self):
        """A refused open ("did not open") is not a lock, whatever runs."""
        t0 = self.clock.t
        self.lockers = lambda: ["SynthChat.exe"]

        def _script(n):
            if n == 1:
                return "opened", _Cap(self.clock,
                                      alive=lambda now: now < t0 + 5.0)
            return "not-opened", None
        backend = _FakeBackend(self.clock, _script)
        out = self._producer([_CAM_A], iterations=60, backend=backend,
                             bounded_real=True, step=self._held_step)
        self.assertNotIn("appears LOCKED", out)
        self.assertEqual(self.gate.locked_by("name:synthcam one"), [])


class AbsentCameraTests(_StormBase):
    """A camera that VANISHED from the device list is not reopened on a timer:
    JARVIS waits for it to come back, then settles before the one open."""

    def test_a_vanished_camera_waits_for_its_return_then_settles(self):
        t0 = self.clock.t
        gone_at, back_at = t0 + 5.0, t0 + 300.0
        names = lambda: (() if gone_at <= self.clock.t < back_at
                         else ("SynthCam One",))
        opens: list = []

        def _open(idx):
            opens.append(self.clock.t)
            if len(opens) == 1:
                return _Cap(self.clock, alive=lambda now: now < gone_at)
            return _Cap(self.clock)
        backend = _FakeBackend(self.clock, names=names)
        out = self._producer([_CAM_A], iterations=200, opener=_open,
                             backend=backend, step=self._held_step)
        self.assertGreater(self.clock.t, back_at + 15.0, "run too short")
        during = [t for t in opens if gone_at < t < back_at]
        self.assertEqual(during, [],
                         f"reopened {len(during)}x while the camera was gone "
                         f"from the bus - that is a timer retry")
        after = [t for t in opens if t >= back_at]
        self.assertEqual(len(after), 1, f"opens after return: {after}")
        settle = self.bc._camera_gate_mod.ABSENT_SETTLE_S
        self.assertGreaterEqual(after[0] - back_at, settle - 0.01,
                                "reopened the moment the device came back - "
                                "no settle")
        self.assertLess(after[0] - back_at, settle + 10.0)
        self.assertIn("gone from the device list", out)
        self.assertIn("back on the device list", out)


class WedgedOpenHoldsEveryOpenTests(_StormBase):
    """While an open is stuck inside the camera driver, nothing else opens."""

    def test_a_stuck_open_holds_every_other_device_until_it_returns(self):
        import threading
        bc = self.bc
        lock = bc._CameraIOLock()
        release = threading.Event()
        self.addCleanup(release.set)

        def _stuck_opener():
            with bc._camera_io_lock:
                release.wait(20.0)
            return None
        with mock.patch.object(bc, "_camera_io_lock", lock), \
             mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            got = bc._open_capture_bounded(41, _stuck_opener, label="synth",
                                           timeout=0.3,
                                           gate_key="name:synthcam one")
            self.assertIsNone(got)
            self.assertTrue(self.gate.wedged(), "the stuck open was not reported")
            for key, comp in (("name:synthcam two", "side-tile"),
                              ("kinect", "kinect-bridge"),
                              ("name:synthcam one", "face-track")):
                d = self.gate.begin(key, comp)
                self.assertFalse(d.allowed, key)
                self.assertEqual(d.reason, "wedged")
            release.set()
            for _ in range(100):
                if not self.gate.wedged():
                    break
                _real_time.sleep(0.02)
        self.assertFalse(self.gate.wedged(), "the returned call did not clear it")
        self.assertIn("stuck inside the camera driver", out.getvalue())
        self.assertIn("the stuck open returned", out.getvalue())


class DirectShowLookupIsGatedTests(_StormBase):
    """The DirectShow name lookup walks the same kernel-streaming devices: it
    is refused during a storm or a wedge, and has a hard timeout."""

    def test_no_lookup_during_a_storm(self):
        bc = self.bc
        self.gate.note_drop("name:synth-x", "face-track")
        self.gate.note_drop("name:synth-y", "face-track")
        raw = mock.Mock(return_value=["Synth"])
        with mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(bc, "_enumerate_dshow_input_devices_raw", raw), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(bc._enumerate_dshow_input_devices())
        raw.assert_not_called()

    def test_a_hung_lookup_is_abandoned(self):
        import threading
        bc = self.bc
        stop = threading.Event()
        self.addCleanup(stop.set)

        def _hang():
            stop.wait(10.0)
            return ["late"]
        t0 = _real_time.monotonic()
        buf = io.StringIO()
        with mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(bc, "_enumerate_dshow_input_devices_raw", _hang), \
             mock.patch.object(bc, "_DSHOW_ENUM_TIMEOUT_S", 0.3), \
             mock.patch.object(bc, "_dshow_enum_timeout_noted", [0.0]), \
             contextlib.redirect_stdout(buf):
            self.assertIsNone(bc._enumerate_dshow_input_devices())
        self.assertLess(_real_time.monotonic() - t0, 2.0)
        self.assertIn("did not answer", buf.getvalue())

    def test_a_healthy_lookup_still_answers(self):
        bc = self.bc
        with mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(bc, "_enumerate_dshow_input_devices_raw",
                               return_value=["Synth A", "Synth B"]):
            self.assertEqual(bc._enumerate_dshow_input_devices(),
                             ["Synth A", "Synth B"])


class BootDoesNotStreamTestAListedCameraTests(_StormBase):
    """Boot keeps a NAMED camera that is on the device list without starting
    its stream: the face tracker's one open is the test (a stream START is
    what tripped the hub resets)."""

    def _cams(self):
        return [dict(_CAM_A), dict(_CAM_B, name="synthcam gone")]

    def test_preflight_probes_only_the_camera_it_cannot_see(self):
        bc = self.bc
        probe = mock.Mock(return_value=True)
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "CAMERAS", self._cams()), \
             mock.patch.object(bc, "_camera_backend", _FakeBackend(self.clock)), \
             mock.patch.object(bc, "_probe_camera_index", probe), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            bc._preflight_cameras(timeout_sec=0.1)
        self.assertEqual([c.args[0] for c in probe.call_args_list], [42],
                         "the listed camera was stream-tested at boot")
        self.assertIn("kept without a boot stream test", out.getvalue())

    def test_the_boot_probe_does_the_same(self):
        bc = self.bc
        probe = mock.Mock(return_value=True)
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "CAMERAS", self._cams()), \
             mock.patch.object(bc, "_camera_backend", _FakeBackend(self.clock)), \
             mock.patch.object(bc, "_probe_camera_index", probe), \
             contextlib.redirect_stdout(io.StringIO()):
            working, failed = bc.probe_cameras_and_update_config()
        self.assertEqual([c.args[0] for c in probe.call_args_list], [42])
        self.assertCountEqual(working, [41, 42])

    # B088 (2026-10-01): the summary lines claimed a stream test that never
    # ran - "kept without a boot stream test" was followed by "opens cleanly"
    # / "working" for the SAME camera.
    def test_preflight_summary_does_not_claim_an_open_it_never_made(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "CAMERAS", self._cams()), \
             mock.patch.object(bc, "_camera_backend", _FakeBackend(self.clock)), \
             mock.patch.object(bc, "_probe_camera_index",
                               mock.Mock(return_value=True)), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            bc._preflight_cameras(timeout_sec=0.1)
        text = out.getvalue()
        self.assertNotIn("camera index 41: opens cleanly", text)
        self.assertIn("camera index 41: on the device list, not stream-tested",
                      text)
        # The camera that WAS probed keeps the old wording.
        self.assertIn("camera index 42: opens cleanly", text)

    def test_cam_probe_summary_does_not_claim_an_open_it_never_made(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "CAMERAS", self._cams()), \
             mock.patch.object(bc, "_camera_backend", _FakeBackend(self.clock)), \
             mock.patch.object(bc, "_probe_camera_index",
                               mock.Mock(return_value=True)), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            bc.probe_cameras_and_update_config()
        text = out.getvalue()
        self.assertNotIn("index 41: working", text)
        self.assertIn("index 41: listed (not stream-tested)", text)
        self.assertIn("index 42: working", text)


class KinectServiceLineTests(_StormBase):
    """While the Kinect runtime service is stopped, the preview does not log a
    'color frame was None' line per frame gap (1,253 of them on 2026-09-29):
    the bridge said it once."""

    def test_no_color_none_spam_while_the_service_is_down(self):
        bc = self.bc
        buf = io.StringIO()
        with mock.patch.object(bc._kinect_bridge, "service_down",
                               return_value=True), \
             mock.patch.object(bc, "_kinect_preview_color_none_log_last", [0.0]), \
             contextlib.redirect_stdout(buf):
            for _ in range(5):
                bc._kinect_preview_color_none_logged()
        self.assertNotIn("color frame was None", buf.getvalue())

    def test_no_color_none_spam_while_the_camera_gate_holds_the_kinect(self):
        # 2026-10-01 regression: through each dies-on-open hold (30-60 min) the
        # line printed every 10 s - 365 of them in one session. Drive the REAL
        # bridge state: no runtime, a gate refusal remembered for a minute.
        bc = self.bc
        kb = bc._kinect_bridge
        buf = io.StringIO()
        held = (f"{kb._GATE_ERR_PREFIX} (backoff: its last 3 opens each died "
                f"within 15s); asking again in 60s")
        with mock.patch.object(kb, "service_down", return_value=False), \
             mock.patch.object(kb, "_runtime", [None]), \
             mock.patch.object(kb, "_open_error", [held]), \
             mock.patch.object(kb, "_gate_hold_until",
                               [_real_time.monotonic() + 60.0]), \
             mock.patch.object(bc, "_kinect_preview_color_none_log_last", [0.0]), \
             contextlib.redirect_stdout(buf):
            for _ in range(5):
                bc._kinect_preview_color_none_logged()
        self.assertNotIn("color frame was None", buf.getvalue())

    def test_an_open_but_stale_runtime_still_logs(self):
        bc = self.bc
        kb = bc._kinect_bridge
        buf = io.StringIO()
        with mock.patch.object(kb, "service_down", return_value=False), \
             mock.patch.object(kb, "_runtime", [object()]), \
             mock.patch.object(kb, "_open_error", [kb._GATE_ERR_PREFIX]), \
             mock.patch.object(kb, "_gate_hold_until",
                               [_real_time.monotonic() + 60.0]), \
             mock.patch.object(bc, "_kinect_preview_color_none_log_last", [0.0]), \
             contextlib.redirect_stdout(buf):
            bc._kinect_preview_color_none_logged()
        self.assertIn("color frame was None", buf.getvalue())

    def test_the_line_still_appears_when_the_service_runs(self):
        bc = self.bc
        buf = io.StringIO()
        with mock.patch.object(bc._kinect_bridge, "service_down",
                               return_value=False), \
             mock.patch.object(bc, "_kinect_preview_color_none_log_last", [0.0]), \
             contextlib.redirect_stdout(buf):
            bc._kinect_preview_color_none_logged()
        self.assertIn("color frame was None", buf.getvalue())


class MillisecondStampTests(_StormBase):
    """Camera open / close / failure lines carry milliseconds: 5 of 16 'open
    -> hub reset' pairs could not be ordered at the log's whole seconds."""

    _MS = r" @\d\d:\d\d:\d\d\.\d{3}$"

    def test_open_and_release_lines_carry_milliseconds(self):
        t0 = self.clock.t
        opens: list = []

        def _open(idx):
            opens.append(idx)
            return _Cap(self.clock, alive=lambda now: now < t0 + 1.0)
        out = self._producer([_CAM_A], iterations=120, opener=_open,
                             step=self._held_step)
        lines = out.splitlines()
        opened = [ln for ln in lines if "Opened Synth left" in ln]
        released = [ln for ln in lines if "[camera] released" in ln]
        failures = [ln for ln in lines if "read failure #" in ln]
        self.assertTrue(opened and released and failures, out[-2000:])
        for ln in opened + released + failures:
            self.assertRegex(ln, self._MS)


class PowerPlanSurvivesSelfRestartTests(_StormBase):
    """A self-restart used to leave the PC on High Performance: the successor
    booted with it active and recorded it as the plan to restore."""

    _OWNER_PLAN = "11111111-2222-3333-4444-555555555555"

    def test_the_successor_adopts_the_plan_handed_down(self):
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        with mock.patch.dict(os.environ,
                             {"JARVIS_PRIOR_POWER_PLAN": self._OWNER_PLAN}), \
             mock.patch.object(bc, "_get_active_power_plan_guid",
                               return_value=bc._HIGH_PERF_GUID), \
             mock.patch.object(bc, "_set_power_plan",
                               side_effect=AssertionError("must not switch")), \
             contextlib.redirect_stdout(io.StringIO()):
            bc._activate_high_performance_plan()
        self.assertEqual(bc._prior_power_plan_guid, self._OWNER_PLAN)
        with mock.patch.object(bc, "_set_power_plan", return_value=True) as setp, \
             contextlib.redirect_stdout(io.StringIO()):
            bc._restore_prior_power_plan()
        setp.assert_called_once_with(self._OWNER_PLAN)

    def test_the_restart_hands_the_plan_down(self):
        from core import actions
        bc = self.bc
        orig = bc._prior_power_plan_guid
        self.addCleanup(setattr, bc, "_prior_power_plan_guid", orig)
        bc._prior_power_plan_guid = self._OWNER_PLAN
        env = actions._successor_env(bc)
        self.assertEqual(env["JARVIS_PRIOR_POWER_PLAN"], self._OWNER_PLAN)
        bc._prior_power_plan_guid = bc._HIGH_PERF_GUID
        self.assertIsNone(actions._successor_env(bc))
        src = ast.get_source_segment(
            open(os.path.join(_ROOT, "core", "actions.py"),
                 encoding="utf-8").read(),
            next(n for n in ast.parse(open(os.path.join(
                _ROOT, "core", "actions.py"), encoding="utf-8").read()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "_act_restart"))
        self.assertIn("env=_successor_env(bc)", src)


class NoDirectShowFallbackTests(_StormBase):
    """A camera that was a Media Foundation device is never handed to
    DirectShow because it has dropped off the MF list."""

    def _open(self, backend):
        with mock.patch.object(self.bc, "_camera_backend", backend), \
             mock.patch.object(self.bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(self.bc, "CAMERAS", [dict(_CAM_A)]):
            return self.bc._camera_open(41, name="synthcam one",
                                        label="synth")

    def test_a_vanished_msmf_camera_does_not_fall_back_to_directshow(self):
        def _resolve(n, idx, name):
            if n == 1:
                return 0, "msmf", "matched CAMERAS name"
            return idx, "dshow", ("no media-foundation device matches dshow "
                                  "index %s (device is DirectShow-only)" % idx)
        backend = _FakeBackend(self.clock,
                               lambda n: ("opened", _Cap(self.clock)),
                               resolve=_resolve)
        self.assertIsNotNone(self._open(backend))
        self.assertIsNone(self._open(backend))
        self.assertEqual([b for _t, b in backend.calls], ["msmf"],
                         "a DirectShow open was attempted for a camera that "
                         "had just dropped off Media Foundation")
        notes = list(self.bc._camera_backend_pending)
        self.assertTrue(any("NOT falling back to DirectShow" in n
                            for n in notes), notes)

    def test_a_directshow_only_device_still_opens(self):
        # DirectShow-only: absent from the Media Foundation list, present (by
        # its name) on the DirectShow one. Since 2026-10-01 a named camera MF
        # does not list is opened only when DirectShow lists THAT name.
        backend = _FakeBackend(
            self.clock, lambda n: ("opened", _Cap(self.clock)),
            resolve=lambda n, idx, name: (idx, "dshow",
                                          "device is DirectShow-only"),
            names=lambda: ("SynthCam Two",))
        with mock.patch.object(self.bc, "_dshow_input_devices_gated",
                               return_value=["SynthCam One"]):
            self.assertIsNotNone(self._open(backend))
            self.assertIsNotNone(self._open(backend))
        self.assertEqual([b for _t, b in backend.calls], ["dshow", "dshow"])

    def test_no_directshow_fallback_during_a_storm(self):
        self.gate.note_drop("name:synth-x", "face-track")
        self.gate.note_drop("name:synth-y", "face-track")
        backend = _FakeBackend(
            self.clock, lambda n: ("opened", _Cap(self.clock)),
            resolve=lambda n, idx, name: (idx, "dshow",
                                          "device is DirectShow-only"))
        self.assertIsNone(self._open(backend))
        self.assertEqual(backend.calls, [])


class _RosterBackend:
    """core.camera_backend stand-in with a REAL device roster: the configured
    webcam is OFF the bus, so DirectShow index 0 and Media Foundation index 0
    are both the depth sensor's video interface - the rig's 2026-09-29 state.
    resolve_capture_target follows the real rules (name on the MF list, else
    the DirectShow name at that index translated to MF, else DirectShow at
    the index). Every open is recorded; none delivers."""

    MSMF = ("Synth Kinect Sensor", "SynthCam Spare")
    DSHOW = ["Synth Kinect Sensor", "SynthCam Spare", "Synth Virtual Cam"]

    def __init__(self):
        self.opened: list = []

    def configured_backend(self):
        return "msmf"

    def msmf_device_names(self, *a, **k):
        return self.MSMF

    def webcam_users_now(self, *a, **k):
        return []

    def resolve_capture_target(self, idx, name=None, dshow_names=None, **k):
        low = [n.lower() for n in self.MSMF]
        if name:
            for i, n in enumerate(low):
                if str(name).lower() in n:
                    return i, "msmf", "matched name"
        if dshow_names and 0 <= int(idx) < len(dshow_names):
            dn = str(dshow_names[int(idx)]).lower()
            for i, n in enumerate(low):
                if n == dn:
                    return i, "msmf", f"translated dshow {idx} -> msmf {i}"
        return int(idx), "dshow", "device is DirectShow-only"

    def open_camera(self, idx, *, backend="msmf", outcome=None, **kw):
        self.opened.append((idx, backend))
        if isinstance(outcome, dict):
            outcome["result"] = "not-opened"
        return None


_DESK = {"index": 0, "label": "Right webcam (top of right monitor)",
         "name": "synthcam desk", "primary": True, "look_x": 0.85,
         "look_y": 0.5}


class WrongDeviceProbeTests(_StormBase):
    """B025 (2026-10-01): with the configured webcam off the bus at boot, the
    probe opened a DIFFERENT camera (the depth sensor) under the webcam's gate
    key, and the boot probes then dropped the webcam - or replaced it with
    unnamed 'Probed webcam' entries - for the whole session."""

    def _patches(self, backend, cams):
        bc = self.bc
        return [mock.patch.object(bc, "_camera_backend", backend),
                mock.patch.object(bc, "_camera_gate", self.gate, create=True),
                mock.patch.object(bc, "CAMERAS", [dict(c) for c in cams]),
                mock.patch.object(bc, "_camera_msmf_seen", set()),
                mock.patch.object(bc, "_camera_open_last_result", {}),
                # The open path's DirectShow cache is WARM: the face tracker's
                # _open_capture resolves the name (_dshow_name_to_index)
                # before it opens, and the boot probe asks where the camera is
                # listed before it probes.
                mock.patch.object(bc, "_dshow_open_devices_cache",
                                  [list(_RosterBackend.DSHOW), "fp", 0.0]),
                mock.patch.object(bc, "_dshow_input_devices_gated",
                                  return_value=list(_RosterBackend.DSHOW))]

    def _run(self, backend, cams, fn):
        with contextlib.ExitStack() as stack:
            for p in self._patches(backend, cams):
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return fn(), list(self.bc.CAMERAS), dict(
                self.bc._camera_open_last_result)

    def test_the_probe_does_not_open_another_device(self):
        backend = _RosterBackend()
        ok, _cams, _res = self._run(
            backend, [_DESK],
            lambda: self.bc._probe_camera_index(0, timeout_sec=0.3))
        self.assertFalse(ok)
        self.assertEqual(backend.opened, [],
                         "the probe stream-started a device that is not the "
                         "configured webcam")

    def test_the_first_open_after_a_restart_does_not_fall_back(self):
        # Nothing has been seen on Media Foundation yet this process: the old
        # rule fell back to DirectShow at the static index.
        backend = _RosterBackend()
        cap, _cams, res = self._run(
            backend, [_DESK],
            lambda: self.bc._camera_open(0, name="synthcam desk",
                                         label="synth"))
        self.assertIsNone(cap)
        self.assertEqual(backend.opened, [])
        self.assertEqual(res.get("name:synthcam desk"), "absent")

    def test_a_directshow_only_camera_opens_at_its_own_index(self):
        backend = _RosterBackend()
        virt = dict(_DESK, name="synth virtual cam")
        self._run(backend, [virt],
                  lambda: self.bc._camera_open(0, name="synth virtual cam",
                                               label="synth"))
        self.assertEqual(backend.opened, [(2, "dshow")])

    def _two_cams(self):
        return [dict(_DESK), dict(_CAM_B)]

    def test_preflight_keeps_a_named_camera_that_did_not_answer(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "_camera_boot_presence", return_value=None), \
             mock.patch.object(bc, "_camera_rescued_by_name", return_value=False), \
             mock.patch.object(bc, "_probe_camera_index",
                               side_effect=lambda i, *a, **k: i == 42):
            _r, cams, _res = self._run(
                _RosterBackend(), self._two_cams(),
                lambda: bc._preflight_cameras(timeout_sec=0.05))
        self.assertIn("synthcam desk", [c.get("name") for c in cams])

    def test_the_boot_probe_keeps_a_named_camera_that_did_not_answer(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "_camera_boot_presence", return_value=None), \
             mock.patch.object(bc, "_camera_rescued_by_name", return_value=False), \
             mock.patch.object(bc, "_probe_camera_index",
                               side_effect=lambda i, *a, **k: i == 42):
            _r, cams, _res = self._run(
                _RosterBackend(), self._two_cams(),
                bc.probe_cameras_and_update_config)
        self.assertIn("synthcam desk", [c.get("name") for c in cams])

    def test_no_index_sweep_replaces_a_named_camera(self):
        bc = self.bc
        probe = mock.Mock(side_effect=lambda i, *a, **k: i != 0)
        with mock.patch.object(bc, "CAMERA_PROBE_ENABLED", True), \
             mock.patch.object(bc, "_camera_boot_presence", return_value=None), \
             mock.patch.object(bc, "_camera_rescued_by_name", return_value=False), \
             mock.patch.object(bc, "camera_users_now", return_value=None), \
             mock.patch.object(bc, "find_camera_locking_processes",
                               return_value=[]), \
             mock.patch.object(bc, "_probe_camera_index", probe):
            (working, _failed), cams, _res = self._run(
                _RosterBackend(), [_DESK],
                bc.probe_cameras_and_update_config)
        self.assertEqual([c.get("name") for c in cams], ["synthcam desk"],
                         "an index sweep replaced the configured webcam")
        self.assertEqual([c.args[0] for c in probe.call_args_list], [0],
                         "other indices were stream-started")
        self.assertEqual(working, [])


class OneDeviceTwoComponentsTests(_StormBase):
    """Two JARVIS components never open the same device back-to-back."""

    def setUp(self):
        super().setUp()
        # A fake device list that LISTS the synthetic cameras: the boot probe
        # asks where a named camera is listed before it probes it (B025,
        # 2026-10-01), and this machine's real lists have never heard of them.
        p = mock.patch.object(self.bc, "_camera_backend",
                              _FakeBackend(self.clock))
        p.start()
        self.addCleanup(p.stop)

    def _fake_camera_open(self, calls):
        def _open(idx, **kw):
            calls.append((self.clock.t, idx, kw.get("label")))
            return _Cap(self.clock)
        return _open

    def test_the_side_tile_waits_out_the_boot_probe(self):
        calls: list = []
        with mock.patch.object(self.bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(self.bc, "CAMERAS", [dict(_CAM_A)]), \
             mock.patch.object(self.bc, "_camera_open",
                               self._fake_camera_open(calls)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc._probe_camera_index(41, timeout_sec=0.5))
            self.clock.advance(3.0)
            self.assertIsNone(
                self.bc._open_tile_capture(41, name="synthcam one"),
                "the tile opened a device the boot probe opened 3 s ago")
            self.assertEqual(len(calls), 1)
            self.clock.advance(7.1)
            cap = self.bc._open_tile_capture(41, name="synthcam one")
            self.assertIsNotNone(cap)
        self.assertEqual(len(calls), 2)

    def test_the_boot_probe_reuses_its_own_fresh_verdict(self):
        calls: list = []
        buf = io.StringIO()
        with mock.patch.object(self.bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(self.bc, "CAMERAS", [dict(_CAM_A)]), \
             mock.patch.object(self.bc, "_camera_open",
                               self._fake_camera_open(calls)), \
             contextlib.redirect_stdout(buf):
            self.assertTrue(self.bc._probe_camera_index(41, timeout_sec=0.5))
            self.clock.advance(30.0)
            self.assertTrue(self.bc._probe_camera_index(41, timeout_sec=0.5))
        self.assertEqual(len(calls), 1,
                         "boot opened a healthy camera twice to learn one fact")
        self.assertIn("proven working", buf.getvalue())

    def test_a_probe_does_not_touch_a_camera_in_a_storm(self):
        calls: list = []
        self.gate.note_drop("name:synth-x", "face-track")
        self.gate.note_drop("name:synth-y", "face-track")
        buf = io.StringIO()
        with mock.patch.object(self.bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(self.bc, "CAMERAS", [dict(_CAM_A)]), \
             mock.patch.object(self.bc, "_camera_open",
                               self._fake_camera_open(calls)), \
             contextlib.redirect_stdout(buf):
            self.assertFalse(self.bc._probe_camera_index(41, timeout_sec=0.5))
        self.assertEqual(calls, [])
        self.assertIn("NOT probed", buf.getvalue())


class TileWaitsForTheProducersFirstOpenTests(_StormBase):
    """While the gate defers the producer's FIRST open of a camera, the side
    tile must not grab that camera - the producer's open moments later would be
    the in-process second handle that wrecks a stream."""

    def _tiles(self, entries):
        bc = self.bc
        opened = mock.Mock(return_value=None)
        saved = bc._face_track_caps[0]
        self.addCleanup(lambda: bc._face_track_caps.__setitem__(0, saved))
        bc._face_track_caps[0] = entries
        for slot in ("left", "right"):
            bc._kinect_tile_caps[slot] = None
            bc._kinect_tile_frames[slot] = None
            bc._kinect_tile_last_read[slot] = 0.0
        with mock.patch.object(bc, "CAMERAS", [dict(_CAM_A), dict(_CAM_B)]), \
             mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(bc, "_resolve_webcam_indices_by_name",
                               return_value={"left": 41, "right": 42}), \
             mock.patch.object(bc, "_open_tile_capture", opened):
            bc._read_side_tile_webcams(123_456.0)
        return opened

    def test_a_pending_first_open_keeps_the_tile_off_the_camera(self):
        opened = self._tiles([{"cam": dict(_CAM_B), "cap": None,
                               "never_opened": True}])
        self.assertNotIn(42, [c.args[0] for c in opened.call_args_list],
                         "the tile opened a camera the producer was about to "
                         "open for the first time")

    def test_a_camera_the_producer_let_go_of_is_still_the_tiles(self):
        # No regression: once the producer HAS opened a camera and let it go
        # (backoff, dead after N reads), the tile may show it itself.
        opened = self._tiles([{"cam": dict(_CAM_B), "cap": None}])
        self.assertIn(42, [c.args[0] for c in opened.call_args_list])


class BootStaggerTests(_StormBase):
    """The cameras and the Kinect are not opened in the same second."""

    def test_the_second_camera_and_the_kinect_are_spaced_out(self):
        # The Kinect bridge's import-time pump opened the sensor 1 s ago.
        if self.gate is not None:
            self.gate.begin("kinect", "kinect-bridge")
            self.gate.end("kinect", "kinect-bridge", True)
        self.clock.advance(1.0)
        t0 = self.clock.t
        opens: list = []

        def _open(idx):
            opens.append((self.clock.t, idx))
            return _Cap(self.clock)
        out = self._producer([_CAM_A, _CAM_B], iterations=40, opener=_open,
                             step=lambda: 0.25)
        self.assertEqual([i for _t, i in opens], [41, 42], out)
        self.assertGreaterEqual(opens[0][0] - (t0 - 1.0), 3.0 - 0.01,
                                "the first webcam opened within 3 s of the "
                                "Kinect")
        self.assertGreaterEqual(opens[1][0] - opens[0][0], 3.0 - 0.01,
                                "both webcams opened in the same second")
        self.assertIn("first open staggered", out)
        self.assertIn("deferred first open", out)
        self.assertNotIn("after recovery", out)


class StaleTileLineTests(_StormBase):
    """'The HUD camera tile is going stale' at most once per 5 minutes per
    tile - it printed 16 times in ~10 minutes during the storm."""

    def test_the_stale_line_is_rate_limited_even_when_failover_flips(self):
        bc = self.bc
        buf = io.StringIO()
        with mock.patch.object(bc, "CAMERAS", [dict(_CAM_A), dict(_CAM_B)]), \
             contextlib.redirect_stdout(buf):
            now = 50_000.0
            for i in range(60):                  # ten minutes, every 10 s
                bc._note_preview_starved(now)
                bc._note_preview_failover(dict(_CAM_B), now + 5.0)
                now += 10.0
        n = buf.getvalue().count("going stale")
        self.assertLessEqual(n, 2, f"'going stale' printed {n} times in 10 min")
        self.assertGreaterEqual(n, 1)


class AudioDropFeedsTheBreakerTests(_StormBase):
    """A camera and an audio device dropping within ~10 s is the hub chain."""

    def test_a_vanished_speaker_plus_a_camera_drop_trips_the_breaker(self):
        bc = self.bc
        self.gate.note_drop("name:synthcam one", "face-track")
        with mock.patch.object(bc, "_camera_gate", self.gate, create=True), \
             mock.patch.object(bc, "_refresh_devices"), \
             mock.patch.object(bc.sd, "query_devices",
                               side_effect=RuntimeError("device gone")), \
             mock.patch.object(bc, "proactive_announce",
                               side_effect=lambda m, *a, **k:
                               self.spoken.append(m) or True), \
             contextlib.redirect_stdout(io.StringIO()):
            bc._device_cache["out"] = 7
            self.assertIsNone(bc.get_output_device())
        self.assertTrue(self.gate.storm_active())
        self.assertEqual(len(self.spoken), 1)


class KnobTests(_StormBase):
    """CAMERA_REOPEN_MAX_BACKOFF_S, USB_STORM_COOLDOWN_S, CAMERA_OPEN_MIN_GAP_S:
    shipped as floats, mirrored in the Settings schema and the template, and
    READ by the gate the monolith builds."""

    _KNOBS = {"CAMERA_REOPEN_MAX_BACKOFF_S": 600.0,
              "USB_STORM_COOLDOWN_S": 600.0,
              "CAMERA_OPEN_MIN_GAP_S": 10.0}

    def test_config_ships_them_as_float_literals(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        for k, v in self._KNOBS.items():
            self.assertIsInstance(lits.get(k), float, k)
            self.assertEqual(lits[k], v, k)

    def test_settings_window_and_template_mirror_them(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "jarvis_settings_window_storm",
            os.path.join(_ROOT, "tools", "settings_window.py"))
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            example = json.load(fh)
        for k, v in self._KNOBS.items():
            self.assertEqual(sw.SCHEMA[k]["type"], "float", k)
            self.assertEqual(sw.SCHEMA[k]["default"], v, k)
            self.assertEqual(example[k], v, k)

    def test_the_gate_the_monolith_builds_reads_them(self):
        bc = self.bc
        with mock.patch.object(bc, "CAMERA_REOPEN_MAX_BACKOFF_S", 120.0), \
             mock.patch.object(bc, "USB_STORM_COOLDOWN_S", 900.0), \
             mock.patch.object(bc, "CAMERA_OPEN_MIN_GAP_S", 7.5):
            g = bc._make_camera_gate()
        self.assertEqual(g.max_backoff_s, 120.0)
        self.assertEqual(g.storm_cooldown_s, 900.0)
        self.assertEqual(g.min_gap_s, 7.5)
        self.assertEqual(bc._camera_gate.min_gap_s, bc.CAMERA_OPEN_MIN_GAP_S)
        # The Kinect bridge got the SAME gate at import.
        self.assertIs(bc._kinect_bridge._open_gate[0], bc._camera_gate)


@requires_monolith
class EveryOpenerAsksTheGateTests(MonolithGlobalsTestCase):
    """Structural: a camera open that bypasses the gate is how five private
    retry clocks fed the storm. Walks the AST, not the text."""

    def _calls(self, tree, fn_name):
        out = {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {c.func.id for c in ast.walk(node)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            out[node.name] = names
        return {n for n, names in out.items() if fn_name in names}, out

    def test_every_monolith_camera_open_asks_first(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        openers, calls = self._calls(tree, "_camera_open")
        self.assertTrue(openers, "the AST scan found no opener at all")
        # list_cameras() is the --list-cameras CLI: a separate process the
        # owner runs by hand, which an in-memory gate cannot span.
        ungated = sorted(n for n in openers - {"list_cameras"}
                         if "camera_gate_begin" not in calls[n])
        self.assertEqual(ungated, [],
                         f"these open a camera without asking the gate: {ungated}")
        self.assertEqual(openers, {"_face_tracking_thread_body",
                                   "_open_tile_capture", "_probe_camera_index",
                                   "list_cameras"},
                         "the set of camera openers changed - gate the new one")

    def test_the_raw_directshow_enumerator_is_only_reached_through_the_gate(self):
        with open(os.path.join(_ROOT, "bobert_companion.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        callers, _calls = self._calls(tree, "_enumerate_dshow_input_devices_raw")
        # (the bounded worker is a closure inside the gated wrapper)
        self.assertEqual(callers, {"_enumerate_dshow_input_devices"})

    def test_every_self_diagnostic_open_asks_first(self):
        with open(os.path.join(_ROOT, "skills", "self_diagnostic.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        openers, calls = self._calls(tree, "_open_probe_capture")
        self.assertEqual(openers, {"_attempt_camera_wake",
                                   "_probe_webcam_locked"})
        for n in openers:
            self.assertIn("_camera_gate_begin", calls[n], n)

    def test_the_kinect_is_opened_in_one_gated_place(self):
        with open(os.path.join(_ROOT, "audio", "kinect_bridge.py"),
                  encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        sites = set()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for c in ast.walk(node):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "PyKinectRuntime"):
                    sites.add(node.name)
        self.assertEqual(sites, {"_open_runtime_locked"})
        src = ast.get_source_segment(
            open(os.path.join(_ROOT, "audio", "kinect_bridge.py"),
                 encoding="utf-8").read(),
            next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_open_runtime_locked"))
        self.assertIn('_gate_call("begin"', src)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
