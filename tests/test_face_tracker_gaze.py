"""Tests for the Kinect head-direction GAZE layer in skills/face_tracker.

Covers ONLY the new "which monitor am I looking at, from Kinect head yaw" work
(feat/kinect-gaze):
  * the yaw → monitor mapping (default geometry + learned calibration + the
    forward dead-zone + hysteresis)
  * the merge of a presence dict's head_yaw_deg into _state, behind
    KINECT_GAZE_ENABLED
  * the Kinect being the PRIMARY which-monitor signal, with the 2-webcam look_x
    heuristic as the FALLBACK when there's no fresh Kinect yaw
  * which-monitor still resolving with ZERO webcams configured (CAMERAS == [])
  * the 'calibrate gaze' voice action sampling + storing a per-monitor yaw band
    to a THROWAWAY calibration file (never the real one)

The skill is loaded fresh in isolation (its own globals) per test with a fake
kinect_bridge + a fake bobert_companion ('bc') + the live-config flags patched,
and JARVIS_GAZE_CALIBRATION_PATH pointed at a temp file so no test touches the
real data/kinect_gaze_calibration.json. No real sensor, no monolith boot.
stdlib unittest + mock.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

# A standard 4-monitor desk layout (matches core.config.MONITORS defaults).
_MONITORS = {
    "left":   (-2560, 0,     2560, 1440),
    "middle": (0,     0,     2560, 1440),
    "right":  (2560,  0,     2560, 1440),
    "top":    (0,     -1440, 2560, 1440),
}


def _fake_bridge(*, yaw=None, present=True, count=1, available=(True, "")):
    """A kinect_bridge stand-in exposing the surface the gaze layer reads:
    available(), get_presence() (carrying head_yaw_deg), get_head_yaw()."""
    m = types.ModuleType("audio.kinect_bridge")
    m.available = lambda: available
    m.get_presence = lambda: {
        "present": present, "count": (count if present else 0),
        "nearest_m": (1.5 if present else None),
        "facing": (True if present else None),
        "head_yaw_deg": yaw, "ts": 0.0,
    }
    m.get_head_yaw = lambda: yaw
    m.get_enabled = lambda: True
    return m


def _fake_bc(cameras):
    """Minimal bobert_companion: MONITORS + CAMERAS + the camera-state plumbing
    _classify_sides reads."""
    import threading
    bc = types.ModuleType("bobert_companion")
    bc.MONITORS = dict(_MONITORS)
    bc.CAMERAS = list(cameras)
    bc._camera_state_lock = threading.Lock()
    bc._camera_last_seen = {}
    bc._camera_latest_frame = {}
    return bc


class _GazeBase(unittest.TestCase):
    def setUp(self):
        # Point the calibration store at a throwaway file (never the real one).
        d = tempfile.mkdtemp(prefix="jarvis_gaze_cal_")
        self._cal_path = os.path.join(d, "kinect_gaze_calibration.json")
        p = mock.patch.dict(os.environ,
                            {"JARVIS_GAZE_CALIBRATION_PATH": self._cal_path})
        p.start()
        self.addCleanup(p.stop)

    def _load(self):
        mod, _actions = load_skill_isolated("face_tracker", register=False)
        return mod

    def _patch_config(self, *, gaze=True, presence=False):
        from core import config as cfg
        for name, val in (("KINECT_GAZE_ENABLED", gaze),
                          ("KINECT_PRESENCE_ENABLED", presence),
                          ("KINECT_PRESENCE_STANDBY", False),
                          ("KINECT_PRESENCE_WAKE", False)):
            q = mock.patch.object(cfg, name, val, create=True)
            q.start()
            self.addCleanup(q.stop)

    def _inject(self, name, module):
        old = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, old) if old is not None
            else sys.modules.pop(name, None))


# ─────────────────────────────────────────────────────────────────────────
# yaw → monitor mapping (pure)
# ─────────────────────────────────────────────────────────────────────────
class YawMappingTests(_GazeBase):
    def test_default_left_right_forward(self):
        ft = self._load()
        bc = _fake_bc([])
        self.assertEqual(ft._default_yaw_to_monitor(bc, -25.0), "left")
        self.assertEqual(ft._default_yaw_to_monitor(bc, 25.0), "right")
        # inside the dead-zone → the forward monitor (middle present)
        self.assertEqual(ft._default_yaw_to_monitor(bc, 3.0), "middle")
        self.assertEqual(ft._default_yaw_to_monitor(bc, -3.0), "middle")

    def test_forward_is_top_when_no_middle(self):
        ft = self._load()
        bc = _fake_bc([])
        bc.MONITORS = {"left": _MONITORS["left"], "right": _MONITORS["right"],
                       "top": _MONITORS["top"]}
        self.assertEqual(ft._default_yaw_to_monitor(bc, 0.0), "top")

    def test_side_falls_back_to_forward_when_absent(self):
        # No 'left' monitor configured → a left turn maps to the forward monitor
        # rather than naming a screen that isn't there.
        ft = self._load()
        bc = _fake_bc([])
        bc.MONITORS = {"middle": _MONITORS["middle"], "right": _MONITORS["right"]}
        self.assertEqual(ft._default_yaw_to_monitor(bc, -30.0), "middle")

    def test_calibrated_band_wins_over_default(self):
        ft = self._load()
        bc = _fake_bc([])
        # Learn: yaw 5..30 is the RIGHT monitor (a narrower turner's range).
        self.assertTrue(ft._save_gaze_calibration({"right": [5.0, 30.0]}))
        # yaw=8 is below the default side threshold (12) → default says middle,
        # but the calibration says right; calibration must win.
        self.assertEqual(ft._yaw_to_monitor(bc, 8.0, None), "right")

    def test_calibration_overlap_resolves_to_nearest_centre(self):
        ft = self._load()
        bands = {"left": [-30.0, -5.0], "middle": [-8.0, 8.0]}
        # yaw=-6 falls in BOTH; middle's centre (0) is nearer than left's (-17.5)
        self.assertEqual(ft._calibrated_yaw_to_monitor(-6.0, bands), "middle")

    def test_hysteresis_holds_near_boundary(self):
        ft = self._load()
        bc = _fake_bc([])
        # Currently on the left monitor; yaw drifts to -10 (inside default
        # side=12). Without hysteresis -10 → middle; hysteresis should HOLD left
        # until the yaw pulls back inside (side - hysteresis = 8).
        self.assertEqual(ft._yaw_to_monitor(bc, -10.0, "left"), "left")
        # Pulled well inside → recentres to middle.
        self.assertEqual(ft._yaw_to_monitor(bc, -2.0, "left"), "middle")


# ─────────────────────────────────────────────────────────────────────────
# merge presence head_yaw_deg → _state (behind KINECT_GAZE_ENABLED)
# ─────────────────────────────────────────────────────────────────────────
class MergeYawTests(_GazeBase):
    def test_merge_sets_kinect_monitor_when_enabled(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])
        now = 1000.0
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "head_yaw_deg": -25.0}, now, bc)
        self.assertEqual(ft._state["kinect_head_yaw"], -25.0)
        self.assertEqual(ft._state["kinect_monitor"], "left")
        self.assertEqual(ft._state["kinect_yaw_at"], now)

    def test_merge_ignores_yaw_when_gaze_disabled(self):
        ft = self._load()
        self._patch_config(gaze=False)
        bc = _fake_bc([])
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "head_yaw_deg": -25.0}, 1000.0, bc)
        self.assertIsNone(ft._state["kinect_monitor"])
        self.assertIsNone(ft._state["kinect_head_yaw"])

    def test_kinect_gaze_monitor_freshness(self):
        ft = self._load()
        self._patch_config(gaze=True)
        with ft._state_lock:
            ft._state["kinect_monitor"] = "right"
            ft._state["kinect_yaw_at"] = 5000.0
        # Fresh read → returns the monitor.
        self.assertEqual(ft._kinect_gaze_monitor(5000.0 + 1.0), "right")
        # Stale (past GAZE_YAW_FRESH_SECONDS) → None.
        self.assertIsNone(
            ft._kinect_gaze_monitor(5000.0 + ft.GAZE_YAW_FRESH_SECONDS + 1.0))

    def test_kinect_gaze_monitor_none_when_disabled(self):
        ft = self._load()
        self._patch_config(gaze=False)
        with ft._state_lock:
            ft._state["kinect_monitor"] = "right"
            ft._state["kinect_yaw_at"] = 5000.0
        self.assertIsNone(ft._kinect_gaze_monitor(5000.0 + 1.0))


# ─────────────────────────────────────────────────────────────────────────
# PRIMARY (Kinect) / FALLBACK (camera) selection in the poller
# ─────────────────────────────────────────────────────────────────────────
class PollSelectionTests(_GazeBase):
    def _run_poll(self, ft, bc, n=2):
        # Drive enough ticks to satisfy the hysteresis-commit window.
        for _ in range(max(n, ft.HYSTERESIS_SAMPLES)):
            ft._poll_once(bc)

    def test_kinect_is_primary_with_webcams_off(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])   # ZERO webcams configured
        self._inject("audio.kinect_bridge", _fake_bridge(yaw=22.0))  # right
        self._run_poll(ft, bc)
        self.assertEqual(ft._snapshot_state()["current_monitor"], "right")

    def test_fallback_to_camera_when_no_kinect_yaw(self):
        ft = self._load()
        self._patch_config(gaze=True)
        # Webcams present; LEFT camera (look_x < .5) sees the face.
        import time as _t
        bc = _fake_bc([{"index": 1, "look_x": 0.2}, {"index": 0, "look_x": 0.85}])
        bc._camera_last_seen = {1: _t.time()}
        # Kinect available but NO body → head_yaw_deg None.
        self._inject("audio.kinect_bridge",
                     _fake_bridge(yaw=None, present=False))
        self._run_poll(ft, bc)
        # No Kinect yaw → camera heuristic wins → 'left'.
        self.assertEqual(ft._snapshot_state()["current_monitor"], "left")

    def test_camera_only_when_gaze_disabled(self):
        ft = self._load()
        self._patch_config(gaze=False)
        import time as _t
        bc = _fake_bc([{"index": 1, "look_x": 0.2}, {"index": 0, "look_x": 0.85}])
        bc._camera_last_seen = {0: _t.time()}   # RIGHT camera sees the face
        # Even though a bridge with a yaw exists, gaze off → it's never consulted.
        self._inject("audio.kinect_bridge", _fake_bridge(yaw=-25.0))
        self._run_poll(ft, bc)
        self.assertEqual(ft._snapshot_state()["current_monitor"], "right")


# ─────────────────────────────────────────────────────────────────────────
# which-monitor action resolves webcam-free (the headline requirement)
# ─────────────────────────────────────────────────────────────────────────
class WhichMonitorWebcamFreeTests(_GazeBase):
    def test_which_monitor_wrapper_answers_from_kinect_with_zero_webcams(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])   # ZERO webcams
        self._inject("bobert_companion", bc)
        self._inject("audio.kinect_bridge", _fake_bridge(yaw=-25.0))  # left
        # The original webcam action would fail with no cameras:
        original = lambda _a="": "user is not visible to any camera"
        actions = {"which_monitor": original}
        ft.register(actions)
        # Seed a fresh Kinect monitor via the merge (as the poller would).
        import time as _t
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "head_yaw_deg": -25.0},
                _t.time(), bc)
        out = actions["which_monitor"]("")
        self.assertIn("LEFT", out)
        self.assertIn("Kinect", out)
        self.assertNotIn("not visible", out)

    def test_core_act_which_monitor_uses_kinect_first(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])
        self._inject("bobert_companion", bc)
        self._inject("skill_face_tracker", ft)   # core.actions resolves via this
        import time as _t
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "head_yaw_deg": 25.0},  # right
                _t.time(), bc)
        from core import actions as core_actions
        out = core_actions._act_which_monitor("")
        self.assertIn("RIGHT", out)
        self.assertIn("Kinect", out)

    def test_core_kinect_shortcircuit_none_when_no_skill(self):
        # With the face_tracker skill not loaded, the Kinect short-circuit
        # helper returns None so _act_which_monitor falls through to its camera
        # path. We target the helper directly (not the full action) because the
        # camera fallback imports cv2, which is blocked under the CI sim — the
        # point here is purely that the short-circuit declines gracefully.
        self._patch_config(gaze=True)
        self._inject("skill_face_tracker", None)
        from core import actions as core_actions
        self.assertIsNone(core_actions._kinect_gaze_which_monitor())


# ─────────────────────────────────────────────────────────────────────────
# calibrate gaze voice action
# ─────────────────────────────────────────────────────────────────────────
class CalibrateGazeTests(_GazeBase):
    def test_calibrate_stores_band_to_throwaway_file(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])
        self._inject("bobert_companion", bc)
        # Bridge that reports a steady yaw band around the right monitor.
        self._inject("audio.kinect_bridge", _fake_bridge(yaw=20.0))
        # Make sampling deterministic + instant: 12 frames of ~20° then stop.
        seq = iter([20.0, 21.0, 19.0, 20.5, 20.0, 19.5,
                    20.0, 21.0, 19.0, 20.5, 20.0, 19.5])
        with mock.patch.object(ft, "_sample_yaw",
                               return_value=list(seq)):
            out = ft.calibrate_gaze("right")
        self.assertIn("right", out.lower())
        # Persisted to the throwaway file, NOT the real one.
        self.assertTrue(os.path.exists(self._cal_path))
        bands = ft._load_gaze_calibration()
        self.assertIn("right", bands)
        lo, hi = bands["right"]
        self.assertLessEqual(lo, 20.0)
        self.assertGreaterEqual(hi, 20.0)

    def test_calibrate_rejects_unknown_monitor(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])
        self._inject("bobert_companion", bc)
        out = ft.calibrate_gaze("banana")
        self.assertIn("don't have", out.lower())
        self.assertFalse(os.path.exists(self._cal_path))

    def test_calibrate_refuses_when_gaze_off(self):
        ft = self._load()
        self._patch_config(gaze=False)
        bc = _fake_bc([])
        self._inject("bobert_companion", bc)
        out = ft.calibrate_gaze("left")
        self.assertIn("off", out.lower())

    def test_calibrate_honest_when_too_few_samples(self):
        ft = self._load()
        self._patch_config(gaze=True)
        bc = _fake_bc([])
        self._inject("bobert_companion", bc)
        self._inject("audio.kinect_bridge", _fake_bridge(yaw=20.0))
        with mock.patch.object(ft, "_sample_yaw", return_value=[20.0, 21.0]):
            out = ft.calibrate_gaze("right")
        self.assertIn("steady", out.lower())
        self.assertFalse(os.path.exists(self._cal_path))

    def test_calibration_status_and_forget(self):
        ft = self._load()
        self._patch_config(gaze=True)
        self.assertIn("No gaze calibration", ft.gaze_calibration_status(""))
        ft._save_gaze_calibration({"left": [-30.0, -8.0]})
        self.assertIn("left", ft.gaze_calibration_status("").lower())
        out = ft.forget_gaze_calibration("")
        self.assertIn("Cleared", out)
        self.assertEqual(ft._load_gaze_calibration(), {})


# ─────────────────────────────────────────────────────────────────────────
# enable toggle
# ─────────────────────────────────────────────────────────────────────────
class ToggleTests(_GazeBase):
    def test_gaze_on_off_flip_live_flag(self):
        ft = self._load()
        from core import config as cfg
        q = mock.patch.object(cfg, "KINECT_GAZE_ENABLED", False, create=True)
        q.start()
        self.addCleanup(q.stop)
        # Persisting hits the Settings writer; stub it so no file is touched.
        with mock.patch.object(ft, "_persist_setting", return_value=True):
            self._inject("audio.kinect_bridge", _fake_bridge(yaw=10.0))
            on = ft.gaze_tracking_on("")
            self.assertTrue(cfg.KINECT_GAZE_ENABLED)
            self.assertIn("on", on.lower())
            off = ft.gaze_tracking_off("")
            self.assertFalse(cfg.KINECT_GAZE_ENABLED)
            self.assertIn("off", off.lower())


# ─────────────────────────────────────────────────────────────────────────
# gaze_stats reads _dwell_total UNDER _state_lock (P2 race regression guard)
#
# The background poller inserts into _dwell_total via _commit_state while
# holding _state_lock. gaze_stats copies that dict to "close out" the current
# run; if the copy is UNLOCKED, a concurrent insert during the dict() copy
# raises "RuntimeError: dictionary changed size during iteration" on the
# user-facing gaze_stats action.
# ─────────────────────────────────────────────────────────────────────────
class GazeStatsLockTests(_GazeBase):
    def test_gaze_stats_copies_dwell_total_under_state_lock(self):
        """Direct proof: at the instant dict(_dwell_total) runs, _state_lock is
        held. A tripwire mapping samples the lock when copied — dict(mapping)
        drives the generic keys() protocol (unlike a dict subclass, which takes
        a C fast path that skips keys()). _state_lock is a non-reentrant
        threading.Lock, so a same-thread acquire(blocking=False) returns False
        *iff* gaze_stats already holds it. Fails against the pre-fix (unlocked)
        copy; passes after."""
        from collections.abc import MutableMapping
        ft = self._load()
        seen = {}

        class _TripwireMapping(MutableMapping):
            def __init__(self, d):
                self._d = dict(d)

            def _sample(self):
                got = ft._state_lock.acquire(blocking=False)
                if got:
                    ft._state_lock.release()
                seen["lock_held_during_copy"] = not got

            def keys(self):           # dict(self) copies via keys()
                self._sample()
                return self._d.keys()

            def __iter__(self):       # belt-and-braces if copy iterates instead
                self._sample()
                return iter(self._d)

            def __getitem__(self, k): return self._d[k]
            def __setitem__(self, k, v): self._d[k] = v
            def __delitem__(self, k): del self._d[k]
            def __len__(self): return len(self._d)

        with ft._state_lock:
            ft._dwell_total = _TripwireMapping({"left": 12.0, "right": 7.0})
        # No current run open → gaze_stats only copies + formats the totals.
        out = ft.gaze_stats("")
        self.assertTrue(
            seen.get("lock_held_during_copy"),
            "gaze_stats copied _dwell_total WITHOUT holding _state_lock")
        # And it still produced the expected breakdown (behaviour preserved).
        self.assertIn("left", out)
        self.assertIn("right", out)

    def test_gaze_stats_survives_concurrent_poller_inserts(self):
        """The actual race: a writer thread hammers _dwell_total[...] = ...
        under _state_lock (exactly as _commit_state does) while we call
        gaze_stats repeatedly. Unlocked, the dict() copy hits "dictionary
        changed size during iteration"; locked, it never does."""
        ft = self._load()
        stop = threading.Event()
        errors: list[BaseException] = []

        def _writer():
            i = 0
            while not stop.is_set():
                # Mutate the SIZE of the dict under the lock, like the poller.
                with ft._state_lock:
                    key = f"m{i % 64}"
                    if key in ft._dwell_total:
                        del ft._dwell_total[key]
                    else:
                        ft._dwell_total[key] = float(i)
                i += 1

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        try:
            for _ in range(400):
                try:
                    ft.gaze_stats("")
                except BaseException as exc:   # noqa: BLE001 — capture & fail
                    errors.append(exc)
                    break
        finally:
            stop.set()
            t.join(timeout=5.0)
        self.assertEqual(
            errors, [],
            f"gaze_stats raced a concurrent insert: {errors!r}")


if __name__ == "__main__":
    unittest.main()
