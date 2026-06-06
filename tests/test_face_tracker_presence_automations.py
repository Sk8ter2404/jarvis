"""Tests for the presence AUTOMATIONS added to skills/face_tracker on the
feat/kinect-v2 branch: auto-greet on entry and the posture/stand nudge.

Both are opt-in (KINECT_GREET_ON_ENTRY / KINECT_POSTURE_NUDGE, default False)
and route spoken output through bobert_companion.proactive_announce. We load the
skill fresh per test (its own globals) with a fake bc + fake kinect_bridge, patch
the live-config flags, and drive the _apply_* helpers directly with hand-rolled
timestamps — no sensor, no threads, no monolith boot.

Asserts:
  * greeting fires ONCE on empty→present after the min-empty window, is hard
    rate-limited, is skipped while JARVIS is busy, and never fires when the flag
    is off,
  * posture nudge fires once after a sustained hunch and once after long seated
    time, then cools down; never fires when the flag is off.

stdlib unittest + mock.
"""
from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


TRACKED = 2


def _fake_bc():
    bc = types.ModuleType("bobert_companion")
    bc._sleep_mode = [False]
    bc._standby_mode = [False]
    bc._tts_playback_active = [False]
    bc._record_speech_active = [False]
    bc._announced = []   # list of (source, message)
    bc.proactive_announce = lambda msg, source="skill", **k: (
        bc._announced.append((source, msg)) or True)
    return bc


def _spine_body(*, lean_forward_m=0.0, distance_m=2.0):
    """A body whose spine_base→spine_shoulder vector leans forward by
    `lean_forward_m` metres in z over ~0.5 m of height. lean_forward_m=0 is
    perfectly upright."""
    base = (0.0, -0.3, 2.0, TRACKED)
    # spine_shoulder is 0.5 m above the base; push it forward (−z toward sensor)
    # by lean_forward_m to simulate a hunch.
    top = (0.0, 0.2, 2.0 - lean_forward_m, TRACKED)
    return {"id": 0, "joints": {"spine_base": base, "spine_shoulder": top},
            "head": None, "distance_m": distance_m, "facing": None}


def _fake_bridge(bodies):
    m = types.ModuleType("audio.kinect_bridge")
    m.available = lambda: (True, "")
    m.get_enabled = lambda: True
    m.get_bodies = lambda: bodies
    m.get_presence = lambda: {"present": bool(bodies), "count": len(bodies),
                              "nearest_m": 2.0, "facing": None, "ts": 0.0}
    return m


class _Base(unittest.TestCase):
    def _load(self):
        mod, _ = load_skill_isolated("face_tracker", register=False)
        return mod

    def _inject(self, name, module):
        old = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, old) if old is not None
            else sys.modules.pop(name, None))

    def _patch_config(self, **flags):
        from core import config as cfg
        for name, val in flags.items():
            p = mock.patch.object(cfg, name, val, create=True)
            p.start()
            self.addCleanup(p.stop)


# ─── auto-greet on entry ────────────────────────────────────────────────────
class GreetTests(_Base):
    def test_no_greet_when_flag_off(self):
        ft = self._load()
        bc = _fake_bc()
        self._patch_config(KINECT_GREET_ON_ENTRY=False)
        # Long empty, then present — but flag off → nothing.
        ft._apply_greet_on_entry(present=False, now=1000.0, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=1000.0 + ft.GREET_MIN_EMPTY_SECONDS + 5, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=1000.0 + ft.GREET_MIN_EMPTY_SECONDS + 7, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_greet_fires_once_on_entry(self):
        ft = self._load()
        bc = _fake_bc()
        self._patch_config(KINECT_GREET_ON_ENTRY=True)
        t0 = 1000.0
        # Room empty, armed.
        ft._apply_greet_on_entry(present=False, now=t0, bc=bc)
        # Re-enters after the min-empty window; presence must hold for the
        # confirm window, so the first present tick alone shouldn't greet.
        enter = t0 + ft.GREET_MIN_EMPTY_SECONDS + 5
        ft._apply_greet_on_entry(present=True, now=enter, bc=bc)
        self.assertEqual(bc._announced, [])     # confirm window not yet elapsed
        # Hold presence past the confirm window → greet once.
        ft._apply_greet_on_entry(
            present=True, now=enter + ft.GREET_PRESENT_CONFIRM_SECONDS + 0.1,
            bc=bc)
        self.assertEqual(len(bc._announced), 1)
        self.assertEqual(bc._announced[0][0], "greet")

    def test_brief_absence_does_not_greet(self):
        ft = self._load()
        bc = _fake_bc()
        self._patch_config(KINECT_GREET_ON_ENTRY=True)
        t0 = 1000.0
        ft._apply_greet_on_entry(present=False, now=t0, bc=bc)
        # Returns BEFORE the min-empty window — a momentary step out of frame.
        back = t0 + (ft.GREET_MIN_EMPTY_SECONDS / 2.0)
        ft._apply_greet_on_entry(present=True, now=back, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=back + ft.GREET_PRESENT_CONFIRM_SECONDS + 1, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_greet_rate_limited(self):
        ft = self._load()
        bc = _fake_bc()
        self._patch_config(KINECT_GREET_ON_ENTRY=True)
        t0 = 1000.0
        # First entry → greet.
        ft._apply_greet_on_entry(present=False, now=t0, bc=bc)
        enter = t0 + ft.GREET_MIN_EMPTY_SECONDS + 2
        ft._apply_greet_on_entry(present=True, now=enter, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=enter + ft.GREET_PRESENT_CONFIRM_SECONDS + 0.1, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        # Leave briefly, come back within the rate-limit window → NO 2nd greet.
        t1 = enter + 5
        ft._apply_greet_on_entry(present=False, now=t1, bc=bc)
        re_enter = t1 + ft.GREET_MIN_EMPTY_SECONDS + 2   # still < 60s since 1st
        ft._apply_greet_on_entry(present=True, now=re_enter, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=re_enter + ft.GREET_PRESENT_CONFIRM_SECONDS + 0.1,
            bc=bc)
        self.assertEqual(len(bc._announced), 1)          # still just the one

    def test_greet_skipped_when_busy(self):
        ft = self._load()
        bc = _fake_bc()
        bc._tts_playback_active = [True]   # JARVIS is speaking
        self._patch_config(KINECT_GREET_ON_ENTRY=True)
        t0 = 1000.0
        ft._apply_greet_on_entry(present=False, now=t0, bc=bc)
        enter = t0 + ft.GREET_MIN_EMPTY_SECONDS + 2
        ft._apply_greet_on_entry(present=True, now=enter, bc=bc)
        ft._apply_greet_on_entry(
            present=True, now=enter + ft.GREET_PRESENT_CONFIRM_SECONDS + 0.1, bc=bc)
        self.assertEqual(bc._announced, [])


# ─── posture / stand nudge ──────────────────────────────────────────────────
class PostureTests(_Base):
    def test_no_nudge_when_flag_off(self):
        ft = self._load()
        bc = _fake_bc()
        self._inject("audio.kinect_bridge", _fake_bridge([_spine_body(lean_forward_m=0.6)]))
        self._patch_config(KINECT_POSTURE_NUDGE=False)
        t0 = 1000.0
        ft._apply_posture_nudge(present=True, now=t0, bc=bc)
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_HUNCH_SECONDS + 60, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_hunch_nudge_fires_once_then_cools_down(self):
        ft = self._load()
        bc = _fake_bc()
        # A strongly forward-leaning spine (0.6 m forward over 0.5 m rise →
        # well past POSTURE_LEAN_DEG).
        self._inject("audio.kinect_bridge",
                     _fake_bridge([_spine_body(lean_forward_m=0.6)]))
        self._patch_config(KINECT_POSTURE_NUDGE=True)
        t0 = 1000.0
        # Start the hunch run.
        ft._apply_posture_nudge(present=True, now=t0, bc=bc)
        self.assertEqual(bc._announced, [])
        # Just before the threshold → still nothing.
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_HUNCH_SECONDS - 5, bc=bc)
        self.assertEqual(bc._announced, [])
        # Past the threshold → one nudge.
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_HUNCH_SECONDS + 5, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        self.assertEqual(bc._announced[0][0], "posture")
        # Immediately after → cooldown suppresses any further nudge.
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_HUNCH_SECONDS + 10, bc=bc)
        self.assertEqual(len(bc._announced), 1)

    def test_upright_does_not_nudge(self):
        ft = self._load()
        bc = _fake_bc()
        # Perfectly upright spine — never crosses the lean threshold; and seated
        # time stays below the (much larger) stand threshold here.
        self._inject("audio.kinect_bridge",
                     _fake_bridge([_spine_body(lean_forward_m=0.0)]))
        self._patch_config(KINECT_POSTURE_NUDGE=True)
        t0 = 1000.0
        ft._apply_posture_nudge(present=True, now=t0, bc=bc)
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_HUNCH_SECONDS + 60, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_stand_nudge_fires_after_long_seated(self):
        ft = self._load()
        bc = _fake_bc()
        # Upright (no hunch) but seated continuously past the stand threshold.
        self._inject("audio.kinect_bridge",
                     _fake_bridge([_spine_body(lean_forward_m=0.0)]))
        self._patch_config(KINECT_POSTURE_NUDGE=True)
        t0 = 1000.0
        ft._apply_posture_nudge(present=True, now=t0, bc=bc)
        ft._apply_posture_nudge(
            present=True, now=t0 + ft.POSTURE_SEATED_SECONDS + 5, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        self.assertEqual(bc._announced[0][0], "posture")

    def test_leaving_resets_seated_run(self):
        ft = self._load()
        bc = _fake_bc()
        self._inject("audio.kinect_bridge",
                     _fake_bridge([_spine_body(lean_forward_m=0.0)]))
        self._patch_config(KINECT_POSTURE_NUDGE=True)
        t0 = 1000.0
        ft._apply_posture_nudge(present=True, now=t0, bc=bc)
        # Leave for longer than POSTURE_ABSENT_RESET_SECONDS → seated run resets.
        ft._apply_posture_nudge(present=False, now=t0 + 10, bc=bc)
        ft._apply_posture_nudge(
            present=False, now=t0 + 10 + ft.POSTURE_ABSENT_RESET_SECONDS + 5,
            bc=bc)
        self.assertEqual(ft._posture_seated_since[0], 0.0)
        # Come back and sit just under the threshold from the NEW start → no
        # nudge (the old time didn't carry over).
        back = t0 + 10 + ft.POSTURE_ABSENT_RESET_SECONDS + 10
        ft._apply_posture_nudge(present=True, now=back, bc=bc)
        ft._apply_posture_nudge(
            present=True, now=back + ft.POSTURE_SEATED_SECONDS - 60, bc=bc)
        self.assertEqual(bc._announced, [])


# ─── new-people greeting (GREET_NEW_PEOPLE_ENABLED) ─────────────────────────
class NewPeopleTests(_Base):
    """The proactive 'who are all these new people?' greeting. We drive the
    firing logic by stubbing the module's _count_unknown_faces (so no cv2 /
    models / real frames are needed); a final integration test instead injects a
    fake audio.face_id engine + frame cache to prove the count flows through the
    real recognise seam. present=True throughout (a body is in the room)."""

    def _stub_count(self, ft, count):
        """Make _count_unknown_faces always report `count` unknown faces."""
        p = mock.patch.object(ft, "_count_unknown_faces", lambda bc: count)
        p.start()
        self.addCleanup(p.stop)

    # Spread timestamps past the scan throttle + confirm window for clarity.
    def _confirm(self, ft):
        return ft.GREET_NEW_PEOPLE_CONFIRM_SECONDS

    def test_no_greet_when_flag_off(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 3)                      # plenty of strangers
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=False)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 5, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_greet_fires_once_on_two_unknowns(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 2)                      # exactly the threshold
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        # First sighting arms the sustained-crowd timer; confirm window not yet
        # elapsed → no greeting on the first tick.
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        self.assertEqual(bc._announced, [])
        # Hold the crowd past the confirm window → exactly one greeting.
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 0.5, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        self.assertEqual(bc._announced[0][0], "new_people")
        # The offer-to-enrol clause rides along on the same utterance.
        self.assertIn("remember their face", bc._announced[0][1].lower())

    def test_no_greet_for_single_unknown(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 1)                      # only ONE stranger
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 5, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_no_greet_when_only_owner_present(self):
        ft = self._load()
        bc = _fake_bc()
        # Owner recognised → ZERO unknown faces (the recognise pass names them).
        self._stub_count(ft, 0)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 5, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_no_greet_before_confirm_window(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 3)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        # Just BEFORE the confirm window elapses → still nothing.
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) - 0.5, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_greet_rate_limited_within_window(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 2)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 0.5, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        # The crowd stays in the room well past the confirm window but still
        # inside the rate-limit window → the hard rate limit suppresses any
        # second greeting (one announcement per gathering).
        for extra in (10, 60, 120, ft.GREET_NEW_PEOPLE_RATE_LIMIT_SECONDS - 30):
            ft._apply_greet_new_people(
                present=True, now=t0 + self._confirm(ft) + extra, bc=bc)
        self.assertEqual(len(bc._announced), 1)     # still just the one

    def test_greet_again_after_rate_limit_window(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 2)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 0.5, bc=bc)
        self.assertEqual(len(bc._announced), 1)
        # Long after the rate-limit window, a fresh sustained crowd greets again.
        later = t0 + ft.GREET_NEW_PEOPLE_RATE_LIMIT_SECONDS + 100
        ft._apply_greet_new_people(present=True, now=later, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=later + self._confirm(ft) + 0.5, bc=bc)
        self.assertEqual(len(bc._announced), 2)

    def test_greet_skipped_when_busy(self):
        ft = self._load()
        bc = _fake_bc()
        bc._record_speech_active = [True]            # mid-conversation
        self._stub_count(ft, 3)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        ft._apply_greet_new_people(
            present=True, now=t0 + self._confirm(ft) + 0.5, bc=bc)
        self.assertEqual(bc._announced, [])

    def test_not_present_disarms(self):
        ft = self._load()
        bc = _fake_bc()
        self._stub_count(ft, 3)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)
        t0 = 1000.0
        # A crowd appears and the timer arms...
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        self.assertNotEqual(ft._new_people_present_since[0], 0.0)
        # ...then the room empties (presence false) before the confirm window →
        # the sustained-crowd timer resets so no greeting fires on return.
        ft._apply_greet_new_people(present=False, now=t0 + 1, bc=bc)
        self.assertEqual(ft._new_people_present_since[0], 0.0)
        self.assertEqual(bc._announced, [])

    def test_integration_counts_unknowns_via_engine(self):
        """End-to-end through the real _count_unknown_faces seam: a fake
        audio.face_id engine returns two 'unknown' + one owner result, and a
        fake bc serves a webcam frame — proving the owner is excluded and the
        two strangers trigger the greeting."""
        ft = self._load()
        bc = _fake_bc()
        # Webcam frame cache + primary camera, like the monolith exposes. The
        # frame needs a .copy() (the grab helper copies it under the lock).
        bc._camera_state_lock = threading.Lock()
        bc.CAMERAS = [{"index": 0, "primary": True, "look_x": 0.85}]
        frame = mock.MagicMock()
        frame.copy.return_value = frame
        bc._camera_latest_frame = {0: frame}

        eng = types.ModuleType("audio.face_id")
        eng.is_available = lambda: (True, "")
        eng.recognize = lambda fr: [
            {"name": "unknown", "score": 0.1, "bbox": [0, 0, 9, 9]},
            {"name": "owner",   "score": 0.8, "bbox": [9, 0, 9, 9]},
            {"name": "unknown", "score": 0.0, "bbox": [18, 0, 9, 9]},
        ]
        self._inject("audio.face_id", eng)
        self._patch_config(GREET_NEW_PEOPLE_ENABLED=True)

        t0 = 1000.0
        ft._apply_greet_new_people(present=True, now=t0, bc=bc)
        self.assertEqual(bc._announced, [])          # confirm window pending
        ft._apply_greet_new_people(
            present=True, now=t0 + ft.GREET_NEW_PEOPLE_CONFIRM_SECONDS + 0.5,
            bc=bc)
        self.assertEqual(len(bc._announced), 1)
        self.assertEqual(bc._announced[0][0], "new_people")


if __name__ == "__main__":
    unittest.main()
