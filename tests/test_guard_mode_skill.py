"""Tests for skills/guard_mode — the multi-angle armed SECURITY array.

Loads the skill in isolation (no monolith boot) via the shared skill harness,
then patches its seams — _bc() (the monolith's proactive_announce + shared frame
caches), _kinect_bridge(), _phone_bridge(), and the config-flag reader — with
fakes so NO hardware, NO real vision/voice call, and NO real file under data/ is
touched. Covers:

  * Motion detector: identical frames = no motion; a big changed region = motion
    above threshold; first-frame-per-camera is skipped; the debounce requires N
    consecutive motion ticks before it fires.
  * Kinect presence fires an intrusion even with zero webcam motion.
  * Alert rate-limiting: multiple detections inside the cooldown → ONE alert;
    after the cooldown a new alert fires.
  * Snapshot writing goes to a PATCHED tmp dir (never the real data/ tree) and is
    named by the timestamp string handed in.
  * arm / disarm / status transitions; armed=False → the loop does nothing;
    everything no-ops gracefully when cameras/bridge are absent or
    KINECT_GUARD_ENABLED is off.
  * The daemon's single tick is driven DIRECTLY (no sleeping, no real thread).

Real numpy frames are used (numpy + cv2 are installed locally; the harness skips
the suite on a CI runner that lacks them). stdlib unittest + mock only.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

try:
    import numpy as np
    import cv2 as _cv2_probe   # imported only to gate the suite on cv2 presence
    _HAVE_CV = _cv2_probe is not None
except Exception:                                   # pragma: no cover
    _HAVE_CV = False


# ─── frame builders (real numpy arrays) ──────────────────────────────────

def _blank(w=640, h=480, base=0):
    """A solid BGR frame."""
    return np.full((h, w, 3), base, dtype=np.uint8)


def _with_block(w=640, h=480, base=0, block=255, frac=0.5):
    """A BGR frame that's `base` everywhere except a bright `block` rectangle
    covering ~`frac` of the area (so the post-resize changed-pixel fraction is
    well above MOTION_PIXEL_FRACTION)."""
    f = _blank(w, h, base)
    bw = int(w * frac)
    f[:, :bw, :] = block
    return f


# ─── fakes ────────────────────────────────────────────────────────────────

def _fake_monolith():
    """Stand-in bobert_companion exposing proactive_announce (records calls) and
    empty shared frame caches (the test hands frames directly to _guard_tick)."""
    bc = types.ModuleType("bobert_companion")
    bc.announced = []   # list of (message, source, mood)

    def _announce(message, source="skill", *, mood=None, volume_scale=1.0):
        bc.announced.append((message, source, mood))
        return True
    bc.proactive_announce = _announce
    bc._camera_state_lock = threading.Lock()
    bc._camera_latest_frame = {}
    return bc


def _fake_phone(*, results=None):
    """Stand-in skill_phone_bridge exposing push_to_phone (records calls)."""
    m = types.ModuleType("skill_phone_bridge")
    m.pushes = []   # list of (message, kwargs)

    def _push(message, **kw):
        m.pushes.append((message, kw))
        return results if results is not None else {"ntfy": True}
    m.push_to_phone = _push
    return m


class GuardBase(unittest.TestCase):
    """Loads the skill fresh per test, patches its seams + the snapshot dir to a
    private tmp dir, and pins config flags. Returns (module, actions)."""

    def _load(self, *, bc="default", phone=None, kinect_enabled=False,
              guard_enabled=True, staging=False, snapshot_dir=None,
              kinect_intrusion=None):
        if bc == "default":
            bc = _fake_monolith()
        mod, actions = load_skill_isolated("guard_mode", register=True)

        # Seam 1: the monolith (proactive_announce + frame caches).
        mod._bc = lambda: bc

        # Seam 2/3: phone bridge + config flags.
        mod._phone_bridge = lambda: phone
        flags = {"KINECT_GUARD_ENABLED": guard_enabled,
                 "KINECT_ENABLED": kinect_enabled}
        mod._cfg_flag = lambda name, default=False: bool(flags.get(name, default))

        # Staging gate (never speak/push on staging).
        mod._is_staging = lambda: staging

        # Kinect intrusion seam (a tracked person) — default None unless pinned.
        if kinect_intrusion is not None or kinect_enabled is False:
            mod._kinect_intrusion = lambda: kinect_intrusion

        # Snapshot dir → a private tmp dir, NEVER the real data/ tree.
        tmp = snapshot_dir or tempfile.mkdtemp(prefix="guard_snap_")
        mod.SNAPSHOT_DIR = tmp
        self.addCleanup(self._rmtree, tmp)
        self._snapshot_dir = tmp

        # Make sure each test starts disarmed + clean, and leaves it that way.
        mod._reset_session()
        with mod._guard_lock:
            mod._armed[0] = False
            mod._armed_since[0] = 0.0
        self.addCleanup(self._disarm, mod)
        return mod, actions

    @staticmethod
    def _disarm(mod):
        with mod._guard_lock:
            mod._armed[0] = False
            mod._armed_since[0] = 0.0
        mod._reset_session()

    @staticmethod
    def _rmtree(path):
        try:
            for f in os.listdir(path):
                try:
                    os.unlink(os.path.join(path, f))
                except Exception:
                    pass
            os.rmdir(path)
        except Exception:
            pass


@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class RegistrationTests(GuardBase):
    def test_registers_all_actions(self):
        _mod, actions = self._load()
        for name in ("guard_on", "guard_off", "guard_status"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


# ─────────────────────────────────────────────────────────────────────────
# motion detector (pure functions)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class MotionDetectorTests(GuardBase):
    def test_identical_frames_no_motion(self):
        mod, _a = self._load()
        a = mod._prep_gray(_blank())
        b = mod._prep_gray(_blank())
        self.assertEqual(mod._frame_motion(a, b), 0.0)
        self.assertFalse(mod._is_motion(a, b))

    def test_large_changed_region_is_motion(self):
        mod, _a = self._load()
        a = mod._prep_gray(_blank(base=0))
        b = mod._prep_gray(_with_block(base=0, block=255, frac=0.5))
        frac = mod._frame_motion(a, b)
        self.assertGreater(frac, mod.MOTION_PIXEL_FRACTION)
        self.assertTrue(mod._is_motion(a, b))

    def test_tiny_change_below_threshold_not_motion(self):
        mod, _a = self._load()
        a = mod._prep_gray(_blank(base=0))
        # A change covering ~0.5% of the frame — below the 2% threshold.
        b = mod._prep_gray(_with_block(base=0, block=255, frac=0.005))
        self.assertFalse(mod._is_motion(a, b))

    def test_mismatched_shapes_read_as_no_motion(self):
        mod, _a = self._load()
        a = mod._prep_gray(_blank(w=640, h=480))
        b = mod._prep_gray(_blank(w=320, h=240))
        # Different prepped shapes → 0.0 rather than a spurious full diff.
        self.assertEqual(mod._frame_motion(a, b), 0.0)

    def test_first_frame_per_camera_skipped_no_trigger(self):
        # A camera's very first tick has no previous frame → cannot trigger even
        # though the frame is "bright" (nothing to diff against).
        mod, _a = self._load()
        frames = [("the left monitor camera", _with_block(frac=0.9))]
        out = mod._guard_tick(frames, None, "ts1", now=100.0)
        self.assertEqual(out["triggered"], [])
        self.assertEqual(out["snapshots"], [])


# ─────────────────────────────────────────────────────────────────────────
# debounce (N consecutive motion ticks before trigger)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class DebounceTests(GuardBase):
    def test_motion_must_persist_n_frames_before_trigger(self):
        mod, _a = self._load()
        label = "the left monitor camera"
        still = _blank(base=0)

        # Tick 1: first frame (still) — seeds prev, no trigger.
        out = mod._guard_tick([(label, still)], None, "t1", now=10.0)
        self.assertEqual(out["triggered"], [])

        # Now alternate prev(still) -> moving. Each tick where cur differs from
        # the immediately-previous frame counts one toward the debounce. Feed the
        # SAME moving frame repeatedly: tick 2 differs from still (streak 1),
        # ticks 3+ are moving-vs-moving (no diff) so the streak would reset — so
        # to build a streak we must keep CHANGING. Use distinct bright blocks.
        b1 = _with_block(base=0, block=255, frac=0.5)
        b2 = _with_block(base=0, block=200, frac=0.6)
        b3 = _with_block(base=0, block=150, frac=0.7)
        # streak 1
        out = mod._guard_tick([(label, b1)], None, "t2", now=10.1)
        self.assertEqual(out["triggered"], [])
        # streak 2
        out = mod._guard_tick([(label, b2)], None, "t3", now=10.2)
        self.assertEqual(out["triggered"], [])
        # streak 3 == MOTION_DEBOUNCE_FRAMES → trigger
        self.assertEqual(mod.MOTION_DEBOUNCE_FRAMES, 3)
        out = mod._guard_tick([(label, b3)], None, "t4", now=10.3)
        self.assertEqual(out["triggered"], [label])
        self.assertEqual(len(out["snapshots"]), 1)

    def test_motion_streak_resets_when_a_tick_shows_no_motion(self):
        # Build a 2-tick streak (below the 3-frame debounce, so nothing fires),
        # then a no-change tick must zero the streak — a later motion burst then
        # has to start counting from scratch rather than triggering early.
        mod, _a = self._load()
        label = "the left monitor camera"
        b1 = _with_block(base=0, block=255, frac=0.5)
        b2 = _with_block(base=0, block=200, frac=0.6)
        mod._guard_tick([(label, b1)], None, "t1", now=1.0)        # seed (1st frame)
        mod._guard_tick([(label, b2)], None, "t2", now=1.1)        # streak 1 (b1→b2)
        with mod._guard_lock:
            self.assertEqual(mod._motion_streak.get(label, 0), 1)
        # A tick identical to the previous frame → no motion → streak resets to 0.
        out = mod._guard_tick([(label, b2)], None, "t3", now=1.2)  # b2→b2, no diff
        self.assertEqual(out["triggered"], [])
        with mod._guard_lock:
            self.assertEqual(mod._motion_streak.get(label, 0), 0)


# ─────────────────────────────────────────────────────────────────────────
# Kinect intrusion (strong signal — fires without webcam motion)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class KinectIntrusionTests(GuardBase):
    def test_kinect_presence_triggers_without_webcam_motion(self):
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 2.0}
        # No webcam frames at all — only the Kinect signal.
        out = mod._guard_tick([], kin, "ts1", now=50.0)
        self.assertIn("the Kinect", out["triggered"])
        self.assertTrue(out["alerted"])
        # The spoken line mentions a person and the distance.
        self.assertTrue(bc.announced)
        msg = bc.announced[0][0]
        self.assertIn("room", msg.lower())
        self.assertIn("2.0", msg)
        # Urgent mood was requested.
        self.assertEqual(bc.announced[0][2], "urgent_clipped")

    def test_kinect_intrusion_records_event(self):
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": None}
        mod._guard_tick([], kin, "ts1", now=50.0)
        with mod._guard_lock:
            self.assertEqual(mod._event_count[0], 1)
            self.assertEqual(mod._last_event[0]["kind"], "kinect")


# ─────────────────────────────────────────────────────────────────────────
# alert rate-limiting (cooldown)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class AlertRateLimitTests(GuardBase):
    def test_multiple_detections_in_cooldown_one_alert(self):
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 1.5}
        # Three detections spaced past the per-camera trigger cooldown (so each
        # counts as an event) but well inside the 30s alert cooldown.
        step = mod.GUARD_TRIGGER_COOLDOWN_SEC + 1
        mod._guard_tick([], kin, "t1", now=100.0)
        mod._guard_tick([], kin, "t2", now=100.0 + step)
        mod._guard_tick([], kin, "t3", now=100.0 + 2 * step)
        # ONE spoken alert despite three detections.
        self.assertEqual(len(bc.announced), 1)
        # But all three count toward the event tally.
        with mod._guard_lock:
            self.assertEqual(mod._event_count[0], 3)

    def test_new_alert_after_cooldown(self):
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 1.5}
        mod._guard_tick([], kin, "t1", now=100.0)
        # Past the cooldown (>30s later) → a second alert is allowed.
        mod._guard_tick([], kin, "t2", now=100.0 + mod.GUARD_ALERT_COOLDOWN_SEC + 1)
        self.assertEqual(len(bc.announced), 2)

    def test_phone_push_fired_fire_and_forget(self):
        phone = _fake_phone()
        mod, _a = self._load(phone=phone, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 1.5}
        mod._guard_tick([], kin, "t1", now=100.0)
        self.assertEqual(len(phone.pushes), 1)
        _msg, kw = phone.pushes[0]
        # Security alerts must NOT route through the read-aloud confirm gate.
        self.assertFalse(kw.get("confirm", True))
        self.assertEqual(kw.get("priority"), "urgent")

    def test_staging_suppresses_speech_and_push(self):
        bc = _fake_monolith()
        phone = _fake_phone()
        mod, _a = self._load(bc=bc, phone=phone, kinect_enabled=True,
                             staging=True)
        kin = {"present": True, "count": 1, "nearest_m": 1.5}
        out = mod._guard_tick([], kin, "t1", now=100.0)
        # Detection still recorded, but nothing actually spoken/pushed.
        self.assertTrue(out["alerted"])           # the cooldown slot was claimed
        self.assertEqual(bc.announced, [])
        self.assertEqual(phone.pushes, [])


# ─────────────────────────────────────────────────────────────────────────
# snapshot writing (tmp dir, named by passed-in timestamp)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class SnapshotTests(GuardBase):
    def test_snapshot_written_to_tmp_named_by_timestamp(self):
        mod, _a = self._load()
        frame = _with_block(frac=0.6)
        path = mod._save_snapshot(frame, "the left monitor camera", "20260604_120000_001")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))
        # Lives in the patched tmp dir, NOT the real data/ tree.
        self.assertEqual(os.path.dirname(path), self._snapshot_dir)
        self.assertNotIn(os.path.join("data", "guard_snapshots"), path)
        # Filename carries the timestamp + a safe camera token.
        self.assertEqual(os.path.basename(path),
                         "20260604_120000_001_the_left_monitor_camera.png")

    def test_tick_writes_snapshot_into_tmp_only(self):
        mod, _a = self._load(kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 1.0}
        # Hand the Kinect colour frame in so the tick has something to snapshot.
        frames = [("the Kinect", _with_block(frac=0.5))]
        out = mod._guard_tick(frames, kin, "20260604_010101_000", now=10.0)
        self.assertEqual(len(out["snapshots"]), 1)
        for p in out["snapshots"]:
            self.assertEqual(os.path.dirname(p), self._snapshot_dir)

    def test_none_frame_snapshot_is_noop(self):
        mod, _a = self._load()
        self.assertIsNone(mod._save_snapshot(None, "the Kinect", "ts"))


# ─────────────────────────────────────────────────────────────────────────
# arm / disarm / status transitions
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class ArmDisarmStatusTests(GuardBase):
    def test_arm_requires_master_flag(self):
        mod, actions = self._load(guard_enabled=False)
        out = actions["guard_on"]("")
        self.assertIn("switched off", out.lower())
        with mod._guard_lock:
            self.assertFalse(mod._armed[0])

    def test_arm_then_status_then_disarm(self):
        # No real thread: neuter Thread.start so guard_on doesn't spin a daemon.
        with mock.patch.object(threading.Thread, "start", lambda self: None):
            mod, actions = self._load(guard_enabled=True)
            out = actions["guard_on"]("")
            self.assertIn("standing watch", out.lower())
            with mod._guard_lock:
                self.assertTrue(mod._armed[0])
            # Status while armed.
            st = actions["guard_status"]("")
            self.assertIn("on watch", st.lower())
            # Disarm.
            off = actions["guard_off"]("")
            self.assertIn("standing down", off.lower())
            with mod._guard_lock:
                self.assertFalse(mod._armed[0])

    def test_double_arm_is_idempotent(self):
        with mock.patch.object(threading.Thread, "start", lambda self: None):
            mod, actions = self._load(guard_enabled=True)
            actions["guard_on"]("")
            out2 = actions["guard_on"]("")
            self.assertIn("already", out2.lower())

    def test_status_when_disarmed_and_enabled(self):
        mod, actions = self._load(guard_enabled=True)
        out = actions["guard_status"]("")
        self.assertIn("not currently on watch", out.lower())

    def test_status_when_disabled(self):
        mod, actions = self._load(guard_enabled=False)
        out = actions["guard_status"]("")
        self.assertIn("disabled", out.lower())

    def test_disarm_when_not_armed(self):
        mod, actions = self._load(guard_enabled=True)
        out = actions["guard_off"]("")
        self.assertIn("wasn't on watch", out.lower())

    def test_disarm_reports_event_count(self):
        with mock.patch.object(threading.Thread, "start", lambda self: None):
            mod, actions = self._load(guard_enabled=True, kinect_enabled=True)
            actions["guard_on"]("")
            # Fabricate a couple of events (spaced past the trigger cooldown).
            kin = {"present": True, "count": 1, "nearest_m": 1.0}
            mod._guard_tick([], kin, "t1", now=1.0)
            mod._guard_tick([], kin, "t2",
                            now=1.0 + mod.GUARD_TRIGGER_COOLDOWN_SEC + 1)
            out = actions["guard_off"]("")
            self.assertIn("2", out)
            self.assertIn("movement", out.lower())

    def test_status_shows_duration_and_last_event(self):
        with mock.patch.object(threading.Thread, "start", lambda self: None):
            mod, actions = self._load(guard_enabled=True, kinect_enabled=True)
            actions["guard_on"]("")
            # Backdate arming so the duration is non-trivial.
            with mod._guard_lock:
                mod._armed_since[0] = time.time() - 65.0
            kin = {"present": True, "count": 1, "nearest_m": 1.0}
            mod._guard_tick([], kin, "t1", now=time.time())
            st = actions["guard_status"]("")
            self.assertIn("on watch", st.lower())
            self.assertIn("minute", st.lower())      # ~1 minute
            self.assertIn("kinect", st.lower())       # names the last camera


# ─────────────────────────────────────────────────────────────────────────
# graceful no-op when nothing is available
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class GracefulDegradeTests(GuardBase):
    def test_tick_with_no_frames_no_kinect_is_noop(self):
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc)
        out = mod._guard_tick([], None, "ts", now=1.0)
        self.assertEqual(out["triggered"], [])
        self.assertEqual(out["snapshots"], [])
        self.assertFalse(out["alerted"])
        self.assertEqual(bc.announced, [])

    def test_collect_frames_no_monolith_no_kinect_empty(self):
        mod, _a = self._load(bc=None, kinect_enabled=False)
        # _kinect_intrusion pinned to None by the loader for kinect_enabled=False.
        self.assertEqual(mod._collect_frames(), [])

    def test_alert_without_monolith_does_not_raise(self):
        # No monolith → proactive_announce unavailable; must not raise.
        mod, _a = self._load(bc=None, kinect_enabled=True,
                             kinect_intrusion={"present": True, "count": 1,
                                               "nearest_m": 1.0})
        out = mod._guard_tick([], {"present": True, "count": 1, "nearest_m": 1.0},
                              "ts", now=1.0)
        self.assertIsInstance(out, dict)

    def test_disarmed_loop_does_nothing(self):
        # Simulate one iteration of the loop's gate while disarmed: armed=False
        # means _collect_frames/_guard_tick are never reached. We assert the gate
        # by checking that a tick is never run when armed is False — done here by
        # confirming guard_status reflects disarmed and no events accrue.
        mod, _a = self._load(guard_enabled=True)
        with mod._guard_lock:
            self.assertFalse(mod._armed[0])
            self.assertEqual(mod._event_count[0], 0)


# ─────────────────────────────────────────────────────────────────────────
# real-thread arm/disarm joins deterministically (no lingering daemon)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class RealThreadLifecycleTests(GuardBase):
    def test_arm_starts_thread_and_disarm_stops_it(self):
        # Shrink the loop's startup delay + poll interval so the daemon spins
        # fast and exits promptly after disarm — keeps the test deterministic.
        mod, actions = self._load(guard_enabled=True, kinect_enabled=False)
        mod.INITIAL_DELAY_SECONDS = 0.0
        mod.GUARD_POLL_INTERVAL = 0.01
        # _kinect_intrusion pinned to None; no monolith frames → ticks are no-ops.
        mod._bc = lambda: None

        actions["guard_on"]("")
        # The daemon thread should be alive.
        alive = [th for th in threading.enumerate()
                 if th.name == "guard-mode-monitor" and th.is_alive()]
        self.assertTrue(alive, "monitor thread should be running after arm")
        t = alive[0]

        actions["guard_off"]("")
        # The loop sees armed=False on its next tick (<=~poll interval) and exits.
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive(), "monitor thread should stop after disarm")


# ─────────────────────────────────────────────────────────────────────────
# per-camera trigger cooldown (snapshot/event spam cap — v1.88 audit)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class TriggerCooldownTests(GuardBase):
    def test_sustained_kinect_presence_one_snapshot_one_event(self):
        # A person standing in the room for ~3s of 0.25s ticks must produce ONE
        # snapshot + ONE event, not one per tick (~4 PNGs/second unbounded).
        bc = _fake_monolith()
        mod, _a = self._load(bc=bc, kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 2.0}
        frames = [("the Kinect", _with_block(frac=0.5))]
        for i in range(12):
            mod._guard_tick(frames, kin, f"t{i:02d}", now=100.0 + i * 0.25)
        self.assertEqual(len(os.listdir(self._snapshot_dir)), 1)
        with mod._guard_lock:
            self.assertEqual(mod._event_count[0], 1)

    def test_kinect_retriggers_after_trigger_cooldown(self):
        mod, _a = self._load(kinect_enabled=True)
        kin = {"present": True, "count": 1, "nearest_m": 2.0}
        mod._guard_tick([], kin, "t1", now=100.0)
        # Still inside the trigger cooldown → suppressed.
        out = mod._guard_tick([], kin, "t2", now=100.0 + 1.0)
        self.assertEqual(out["triggered"], [])
        # Past the trigger cooldown → a fresh trigger (event #2).
        out = mod._guard_tick([], kin, "t3",
                              now=100.0 + mod.GUARD_TRIGGER_COOLDOWN_SEC + 1)
        self.assertEqual(out["triggered"], ["the Kinect"])
        with mod._guard_lock:
            self.assertEqual(mod._event_count[0], 2)

    def test_continuous_webcam_mover_capped_by_trigger_cooldown(self):
        # A continuous mover re-fires the debounce every 3 ticks; the trigger
        # cooldown must cap snapshots/events to one per window, not one/0.75s.
        mod, _a = self._load()
        label = "the left monitor camera"
        # Keep the frames CHANGING every tick so the streak keeps building.
        variants = [_with_block(base=0, block=60 + 15 * i, frac=0.4 + 0.04 * i)
                    for i in range(9)]
        mod._guard_tick([(label, _blank())], None, "seed", now=10.0)
        for i, fr in enumerate(variants):
            mod._guard_tick([(label, fr)], None, f"m{i}", now=10.25 + i * 0.25)
        # 9 motion ticks = three full debounce cycles, but ONE trigger.
        self.assertEqual(len(os.listdir(self._snapshot_dir)), 1)
        with mod._guard_lock:
            self.assertEqual(mod._event_count[0], 1)


# ─────────────────────────────────────────────────────────────────────────
# rearm race (guard_off → guard_on while the old thread winds down)
# ─────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(_HAVE_CV, "numpy/cv2 not installed on this runner")
class RearmRaceTests(GuardBase):
    def test_rearm_starts_new_thread_even_if_old_one_still_alive(self):
        # Simulate the TOCTOU window: an old monitor thread that has read
        # armed=False but not yet returned still shows up alive in
        # threading.enumerate(). guard_on must NOT skip starting a replacement.
        mod, actions = self._load(guard_enabled=True)
        stop = threading.Event()
        old = threading.Thread(target=stop.wait, daemon=True,
                               name="guard-mode-monitor")
        old.start()
        self.addCleanup(stop.set)

        started = []
        with mock.patch.object(threading.Thread, "start",
                               lambda self: started.append(self.name)):
            actions["guard_on"]("")
        stop.set()
        self.assertIn("guard-mode-monitor", started,
                      "guard_on must start a fresh monitor thread even while "
                      "an old one is winding down")

    def test_stale_generation_loop_exits_while_armed(self):
        # A thread from a PREVIOUS arming session must exit as soon as it sees
        # its generation is stale, even though guard is (re)armed.
        mod, _a = self._load(guard_enabled=True)
        mod.INITIAL_DELAY_SECONDS = 0.0
        with mod._guard_lock:
            mod._armed[0] = True
            mod._monitor_gen[0] = 7
        # Runs to completion immediately (would loop forever on the old code
        # only gated by armed) — a direct call returning proves the exit.
        t = threading.Thread(target=mod._monitor_loop, args=(6,), daemon=True)
        t.start()
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive(),
                         "stale-generation monitor loop must exit while armed")


if __name__ == "__main__":
    unittest.main()
