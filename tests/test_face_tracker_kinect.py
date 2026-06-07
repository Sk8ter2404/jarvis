"""Tests for the Kinect presence integration in skills/face_tracker.

Covers ONLY the new Kinect-presence behaviour added on the feat/kinect-v2
branch:
  * _read_kinect_presence respects KINECT_PRESENCE_ENABLED + bridge availability
  * presence merges into _state and surfaces in gaze_status
  * empty-room → standby fires ONLY when KINECT_PRESENCE_STANDBY is on, and only
    after the sustained window with the hysteresis latch
  * person-returns → clears standby ONLY when KINECT_PRESENCE_WAKE is on

The skill is loaded fresh in isolation (its own globals) per test, with a fake
kinect_bridge + a fake bobert_companion ('bc') + the live-config flags patched.
No real sensor, no monolith boot. stdlib unittest + mock.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bridge(presence, *, available=(True, "")):
    m = types.ModuleType("audio.kinect_bridge")
    m.available = lambda: available
    m.get_presence = lambda: presence
    m.get_enabled = lambda: True
    return m


def _fake_bc():
    """A minimal bobert_companion stand-in carrying the standby flags + the
    bridge functions the presence-action path reaches for."""
    bc = types.ModuleType("bobert_companion")
    bc._standby_mode = [False]
    bc._sleep_mode = [False]
    bc._standby_auto_engage_lock = __import__("threading").Lock()
    bc._engage_calls = []

    def _engage(reason="music"):
        bc._engage_calls.append(reason)
        # mimic the real one: set the flags, return True if it changed state
        if bc._standby_mode[0] or bc._sleep_mode[0]:
            return False
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        return True
    bc._standby_auto_engage = _engage
    bc._write_hud_state = lambda **kw: None
    return bc


class _FaceTrackerKinectBase(unittest.TestCase):
    def _load(self):
        mod, _actions = load_skill_isolated("face_tracker", register=False)
        return mod

    def _patch_config(self, *, enabled=True, standby=False, wake=False):
        """Patch the Kinect flags on the REAL core.config module.

        ``_cfg_flag`` does ``from core import config`` — once the ``core``
        package is imported (which the whole suite does), that binds the real
        submodule, NOT a ``sys.modules['core.config']`` swap. So we must set
        the attributes on the live module. mock.patch.object(create=True)
        restores them afterward whether or not they pre-existed."""
        from core import config as cfg
        for name, val in (("KINECT_PRESENCE_ENABLED", enabled),
                          ("KINECT_PRESENCE_STANDBY", standby),
                          ("KINECT_PRESENCE_WAKE", wake)):
            p = mock.patch.object(cfg, name, val, create=True)
            p.start()
            self.addCleanup(p.stop)

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
# _read_kinect_presence gating
# ─────────────────────────────────────────────────────────────────────────
class ReadPresenceTests(_FaceTrackerKinectBase):
    def test_disabled_flag_returns_none(self):
        ft = self._load()
        self._inject("audio.kinect_bridge",
                     _fake_bridge({"present": True, "count": 1}))
        self._patch_config(enabled=False)
        self.assertIsNone(ft._read_kinect_presence())

    def test_unavailable_bridge_returns_none(self):
        ft = self._load()
        self._inject("audio.kinect_bridge",
                     _fake_bridge({"present": True, "count": 1},
                                  available=(False, "no sensor")))
        self._patch_config(enabled=True)
        self.assertIsNone(ft._read_kinect_presence())

    def test_enabled_and_available_returns_presence(self):
        ft = self._load()
        pres = {"present": True, "count": 2, "nearest_m": 1.5,
                "facing": True, "ts": 0.0}
        self._inject("audio.kinect_bridge", _fake_bridge(pres))
        self._patch_config(enabled=True)
        self.assertEqual(ft._read_kinect_presence(), pres)


# ─────────────────────────────────────────────────────────────────────────
# merge into _state + surfacing in gaze_status
# ─────────────────────────────────────────────────────────────────────────
class MergeTests(_FaceTrackerKinectBase):
    def test_merge_populates_state(self):
        ft = self._load()
        now = 1000.0
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 2, "nearest_m": 1.8,
                 "facing": True, "ts": 0.0}, now)
        self.assertIs(ft._state["kinect_present"], True)
        self.assertEqual(ft._state["kinect_count"], 2)
        self.assertEqual(ft._state["kinect_nearest_m"], 1.8)
        self.assertEqual(ft._state["last_face_at"], now)   # counts as a sighting

    def test_gaze_status_mentions_kinect_when_webcams_blind(self):
        ft = self._load()
        # Simulate: webcams see nothing (monitor 'away'), Kinect sees a body.
        with ft._state_lock:
            ft._state["last_sample_at"] = 5.0
            ft._state["current_monitor"] = "away"
            ft._state["kinect_present"] = True
            ft._state["kinect_count"] = 1
            ft._state["kinect_nearest_m"] = 1.8
            ft._state["kinect_at"] = __import__("time").monotonic()
        out = ft.gaze_status("")
        self.assertIn("Kinect", out)
        self.assertIn("one person", out)


# ─────────────────────────────────────────────────────────────────────────
# empty-room → standby (opt-in)
# ─────────────────────────────────────────────────────────────────────────
class StandbyOnEmptyTests(_FaceTrackerKinectBase):
    def test_no_standby_when_flag_off(self):
        ft = self._load()
        bc = _fake_bc()
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=False, wake=False)
        # Even with the room empty for a long time, nothing fires.
        ft._kinect_empty_since[0] = 0.0
        ft._apply_kinect_presence_actions(present=False, now=10_000.0)
        ft._apply_kinect_presence_actions(present=False, now=10_999.0)
        self.assertEqual(bc._engage_calls, [])

    def test_standby_fires_after_sustained_empty(self):
        ft = self._load()
        bc = _fake_bc()
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=True, wake=False)
        # First empty read arms the timer; nothing yet.
        ft._apply_kinect_presence_actions(present=False, now=1000.0)
        self.assertEqual(bc._engage_calls, [])
        # Still inside the window → nothing.
        ft._apply_kinect_presence_actions(
            present=False, now=1000.0 + ft.KINECT_EMPTY_STANDBY_SECONDS - 1)
        self.assertEqual(bc._engage_calls, [])
        # Past the window → standby engages exactly once (latched).
        ft._apply_kinect_presence_actions(
            present=False, now=1000.0 + ft.KINECT_EMPTY_STANDBY_SECONDS + 1)
        ft._apply_kinect_presence_actions(
            present=False, now=1000.0 + ft.KINECT_EMPTY_STANDBY_SECONDS + 2)
        self.assertEqual(bc._engage_calls, ["room_empty"])   # only once

    def test_presence_resets_empty_timer(self):
        ft = self._load()
        bc = _fake_bc()
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=True, wake=False)
        ft._apply_kinect_presence_actions(present=False, now=1000.0)  # arm
        ft._apply_kinect_presence_actions(present=True, now=1100.0)   # reset
        self.assertEqual(ft._kinect_empty_since[0], 0.0)
        # A fresh empty run must start the clock over, not fire immediately.
        ft._apply_kinect_presence_actions(present=False, now=1200.0)
        ft._apply_kinect_presence_actions(
            present=False, now=1200.0 + ft.KINECT_EMPTY_STANDBY_SECONDS - 5)
        self.assertEqual(bc._engage_calls, [])


# ─────────────────────────────────────────────────────────────────────────
# person-returns → wake (opt-in)
# ─────────────────────────────────────────────────────────────────────────
class WakeOnPresenceTests(_FaceTrackerKinectBase):
    def test_no_wake_when_flag_off(self):
        ft = self._load()
        bc = _fake_bc()
        bc._standby_mode[0] = True
        bc._sleep_mode[0] = True
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=False, wake=False)
        ft._apply_kinect_presence_actions(present=True, now=1000.0)
        self.assertTrue(bc._standby_mode[0])   # still asleep
        self.assertTrue(bc._sleep_mode[0])

    def test_wake_clears_standby_when_flag_on(self):
        ft = self._load()
        bc = _fake_bc()
        bc._standby_mode[0] = True
        bc._sleep_mode[0] = True
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=False, wake=True)
        ft._apply_kinect_presence_actions(present=True, now=1000.0)
        self.assertFalse(bc._standby_mode[0])   # woken
        self.assertFalse(bc._sleep_mode[0])

    def test_wake_noop_when_not_in_standby(self):
        ft = self._load()
        bc = _fake_bc()   # not in standby
        self._inject("bobert_companion", bc)
        self._inject("__main__", bc)
        self._patch_config(standby=False, wake=True)
        # Should not raise and should leave flags clear.
        ft._apply_kinect_presence_actions(present=True, now=1000.0)
        self.assertFalse(bc._standby_mode[0])
        self.assertFalse(bc._sleep_mode[0])


# ─────────────────────────────────────────────────────────────────────────
# kinect_at staleness clock (P2 regression)
# ─────────────────────────────────────────────────────────────────────────
class KinectStalenessClockTests(_FaceTrackerKinectBase):
    """Regression for P2: _kinect_presence_note compares kinect_at against
    time.monotonic(), so _merge_kinect_presence MUST stamp kinect_at on the
    monotonic clock — not the wall-clock `now` it's handed (which legitimately
    drives the human-facing *_at fields). When it stamped wall-clock, the diff
    was hugely negative, never exceeded 5.0, and 'the Kinect sees N people'
    persisted forever after the room emptied."""

    def test_merge_stamps_kinect_at_on_monotonic_clock(self):
        ft = self._load()
        # A small wall-clock `now` that could never be confused with a real
        # time.monotonic() reading (seconds since boot is far larger).
        wall_now = 1000.0
        mono_before = __import__("time").monotonic()
        with ft._state_lock:
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "nearest_m": 1.5,
                 "facing": True, "ts": 0.0}, wall_now)
        mono_after = __import__("time").monotonic()
        kinect_at = ft._state["kinect_at"]
        # On the monotonic clock, bracketed by readings around the merge …
        self.assertGreaterEqual(kinect_at, mono_before)
        self.assertLessEqual(kinect_at, mono_after)
        # … and emphatically NOT the wall-clock value it was handed.
        self.assertNotEqual(kinect_at, wall_now)
        # The wall-clock `now` still drives the human-facing sighting fields.
        self.assertEqual(ft._state["last_face_at"], wall_now)

    def test_fresh_merge_surfaces_note_then_goes_stale(self):
        ft = self._load()
        # Drive the REAL merge path (not a hand-set kinect_at): a body is seen.
        with ft._state_lock:
            ft._state["last_sample_at"] = 5.0
            ft._state["current_monitor"] = "away"
            ft._merge_kinect_presence(
                {"present": True, "count": 1, "nearest_m": 1.8,
                 "facing": True, "ts": 0.0}, 2000.0)
        # Fresh reading → the note surfaces (under the old wall-clock bug this
        # path could only ever surface, never expire).
        fresh = ft.gaze_status("")
        self.assertIn("Kinect", fresh)
        self.assertIn("one person", fresh)
        # Now age the monotonic stamp past the 5.0s staleness window. With the
        # fix this expires; with the bug it never would.
        with ft._state_lock:
            ft._state["kinect_at"] = __import__("time").monotonic() - 30.0
        stale = ft.gaze_status("")
        self.assertNotIn("Kinect", stale)


if __name__ == "__main__":
    unittest.main()
