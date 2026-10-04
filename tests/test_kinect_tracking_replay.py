"""REPLAY tests: numeric body traces through the REAL Kinect hand pipeline -
bridge parser → shared stabiliser → two-hand / air-mouse / gesture pollers
(tests/_kinect_replay.py) - for the owner's 2026-10-04 "hand tracking is
unstable".

Traces are synthetic geometry; where noted they carry the ANONYMISED streams
recorded at the owner's desk (tests/_kinect_replay_desk_noise.csv: jitter
residuals, joint TrackingStates, SDK grip states - numbers only, no time of
day, no coordinates). Every knob is at its live default.

The tree decides the pipeline: on a bridge without the shared stabiliser
(origin/main d5931da) the same traces run through the legacy per-poll path, and
the flap / phantom-click / phantom-swipe assertions FAIL there (measured:
two-hand mode 14 and 37 engages in 20 s, a click from a flicker, 13 cursor
drops in 30 s, 3 false swipes in 6 s).

    python -B -m unittest tests.test_kinect_tracking_replay
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from audio import kinect_bridge as kb
from tests import _kinect_replay as R

SHARED = hasattr(kb, "publish_body_frame")
T0 = 1000.0


def setUpModule():
    # Belt and braces: nothing here should touch a settings file, but never
    # let a stray read/write reach the real one.
    global _SAVED, _TMP
    _SAVED = os.environ.get("JARVIS_SETTINGS_PATH")
    _TMP = tempfile.mkdtemp(prefix="jarvis_kreplay_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(_TMP, "s.json")


def tearDownModule():
    if _SAVED is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED
    shutil.rmtree(_TMP, True)


class _Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = R.load_fixture()
        cls.mods = R.load_modules()
        cls.am = cls.mods[0]

    @classmethod
    def tearDownClass(cls):
        R.unload_modules()

    def replay(self, frames, shared=SHARED):
        return R.run(frames, shared=shared, modules=self.mods)

    @staticmethod
    def releases(res):
        return [(b["t"], b["why"]) for a, b in zip(res.polls, res.polls[1:])
                if a["engaged"] and not b["engaged"]]


class RecordedDeskTraceTests(_Replay):
    def test_fixture_is_the_live_recording_shape(self):
        # 30 s of real frame steps with drops, both hands toggling between
        # Tracked and Inferred - a fixture that went flat would pass anything.
        self.assertEqual(len(self.rows), 900)
        self.assertGreater(sum(1 for r in self.rows if r["dtf"] > 1), 20)
        flips = sum(1 for a, b in zip(self.rows, self.rows[1:])
                    if a["ts_HandRight"] != b["ts_HandRight"])
        self.assertGreater(flips, 100)

    def test_hands_resting_at_the_desk_do_nothing(self):
        res = self.replay(R.scenario_desk(self.rows))
        self.assertEqual(res.edges("engaged"), [])
        self.assertEqual(res.presses(), [])
        self.assertEqual(res.edges("two"), [])
        self.assertEqual(res.gestures, [])
        # The shared layer saw every frame and never changed a grip or named an
        # active hand, although the raw SDK hand state changed again and again.
        self.assertEqual(len(res.snaps), len(self.rows))
        raw = sum(1 for a, b in zip(self.rows, self.rows[1:])
                  if a["hr_state"] != b["hr_state"])
        self.assertGreater(raw, 10)
        for side in ("left", "right"):
            grips = {s["hands"][side]["grip"] for _, s in res.snaps}
            self.assertEqual(grips, {"open"}, side)
        self.assertEqual({s["active"] for _, s in res.snaps}, {None})


class TwoHandFlapTests(_Replay):
    """The 2026-10-04 log: 51 two-hand engages in 28 min, 48 gone within the
    second, each one standing the cursor down."""

    def test_second_hand_hovering_at_the_line_does_not_flap(self):
        res = self.replay(R.scenario_second_hand_hover())
        self.assertLessEqual(len(res.edges("two")), 1)
        self.assertLessEqual(len(res.edges("engaged")), 1)
        self.assertEqual({w for _, w in self.releases(res)} - {"two-hand"}, set())

    def test_second_hand_tracking_flicker_does_not_flap(self):
        # The second hand is clearly up (+0.12 m) but its joint toggles
        # Tracked/Inferred exactly as the recording's right hand did.
        res = self.replay(R.scenario_second_hand_hover(
            hover_lift=0.12, amp=0.0, rows=self.rows))
        edges = res.edges("two")
        self.assertLessEqual(len(edges), 1)
        self.assertLessEqual(len(res.edges("engaged")), 1)


class DrivingHandFlickerTests(_Replay):
    def test_cursor_survives_the_recorded_tracking_flicker(self):
        res = self.replay(R.scenario_raise_and_hold(
            self.rows, seconds=30, tracking_from="right"))
        self.assertEqual(self.releases(res), [])
        after = [p for p in res.polls if p["t"] > T0 + 2.0]
        self.assertTrue(all(p["engaged"] for p in after))
        self.assertEqual(len({p["hand"] for p in after}), 1)


class ClickTests(_Replay):
    def test_grip_flickers_never_click(self):
        # 1-3 frame High-confidence and 2-6 frame Low-confidence closed
        # flickers on an engaged, open hand.
        res = self.replay(R.scenario_grip_flickers())
        self.assertTrue(res.edges("engaged"))
        self.assertEqual(res.presses(), [])

    def test_a_real_grip_still_clicks_promptly(self):
        res = self.replay(R.scenario_grip_flickers(flickers=(), real_grip_at=3.0))
        downs = [t for t, a, b in res.buttons if a == "down"]
        ups = [t for t, a, b in res.buttons if a == "up"]
        self.assertEqual(len(downs), 1)
        self.assertEqual(len(ups), 1)
        first_closed = T0 + 91 * R.FRAME_S          # frame 90 of the trace
        first_open = T0 + (91 + 18) * R.FRAME_S
        self.assertLessEqual(downs[0] - first_closed, 0.20)
        self.assertLessEqual(ups[0] - first_open, 0.10)


class DeliberateGestureTests(_Replay):
    def test_raised_open_hand_takes_the_cursor(self):
        res = self.replay(R.scenario_raise_and_hold(seconds=3))
        eng = res.edges("engaged")
        self.assertEqual(len(eng), 1)
        self.assertLessEqual(eng[0] - T0, 0.65)

    def test_two_hand_raise_engages_once_and_grabs(self):
        frames = R.scenario_two_hand_raise()
        res = self.replay(frames)
        cross = next(T0 + (k + 1) * R.FRAME_S for k, fr in enumerate(frames)
                     if fr.left[1] - R.SPINE_SHOULDER[1] >= 0.07)
        on = res.edges("two")
        self.assertEqual(len(on), 1)
        self.assertLessEqual(on[0] - cross, 0.45)
        off = [b["t"] for a, b in zip(res.polls, res.polls[1:])
               if a["two"] and not b["two"]]
        self.assertEqual(len(off), 1)
        grab = next(p["t"] for p in res.polls if p["phase"] == "grabbed")
        fists = T0 + 1.8 + R.FRAME_S
        self.assertLessEqual(grab - fists, 0.75)
        self.assertTrue(res.windows)

    def test_a_second_hand_going_high_does_not_yank_the_cursor(self):
        # Until two-hand mode takes over (and stands the cursor down) the
        # cursor stays on the hand that was driving it - no 0.1 s jump across.
        res = self.replay(R.scenario_second_hand_goes_high())
        driving = {p["hand"] for p in res.polls if p["engaged"]}
        self.assertEqual(len(driving), 1, driving)
        self.assertEqual([w for _, w in self.releases(res)], ["two-hand"])
        self.assertEqual(len(res.edges("two")), 1)

    def test_hand_switch_hands_the_cursor_over_without_dropping_it(self):
        res = self.replay(R.scenario_hand_switch())
        want = "right" if self.am._hand_mirror_enabled() else "left"
        sw = next(p["t"] for p in res.polls
                  if p["t"] > T0 + 3.0 and p["hand"] == want)
        self.assertLessEqual(sw - (T0 + 3.0), 0.60)
        during = [p for p in res.polls if T0 + 3.0 < p["t"] <= sw]
        self.assertTrue(all(p["engaged"] for p in during))


class OwnerBodyTests(_Replay):
    def test_a_nearer_passer_by_does_not_take_or_drop_the_cursor(self):
        res = self.replay(R.scenario_passer_by())
        self.assertEqual(len(res.edges("engaged")), 1)
        self.assertEqual(self.releases(res), [])


class SharedSnapshotConsumerTests(_Replay):
    """Every consumer reads the SAME snapshot: the air-mouse (mirrored), the
    gesture recognizer (owner body only) and point-to-control (owner body)."""

    def _snap(self, **over):
        owner = {"id": 7, "joints": {}, "distance_m": 1.2, "facing": True}
        hv = {"pos": (0.15, 0.55, 0.9), "lift": 0.25, "raised": True,
              "grip": "closed", "source": "wrist", "conf": "high",
              "ext": {"forward_reach_m": 0.3, "straightness": 0.95,
                      "reach_ratio": 0.8}}
        snap = {"seq": 3, "t": 5.0, "age": 0.01, "stale": False,
                "owner_id": 7, "owner": owner, "fresh": True, "tracked": True,
                "active": "right", "two_hand": True,
                "hands": {"left": {"pos": None, "lift": None, "grip": "open"},
                          "right": hv}}
        snap.update(over)
        return snap

    def _bridge(self, snap):
        import types
        return types.SimpleNamespace(
            get_enabled=lambda: True, available=lambda: (True, ""),
            get_bodies=lambda: [{"id": 99, "joints": {}}],
            get_tracked_frame=lambda: snap)

    def test_air_mouse_reads_the_snapshot_mirrored(self):
        am = self.am
        from unittest import mock
        with mock.patch.object(am, "_hand_mirror_enabled", lambda: True):
            le, re_, lg, rg, tracked = am._hand_sample(self._bridge(self._snap()))
        self.assertTrue(tracked)
        self.assertEqual(lg, "closed")              # SDK right → owner's left
        self.assertAlmostEqual(le.lift_m, 0.25)
        self.assertIsNone(re_.hand)
        self.assertEqual(am._last_active_side[0], "left")
        self.assertIs(am._last_two_hand[0], True)
        self.assertEqual(am._last_body_id[0], 7)

    def test_a_stale_snapshot_reads_as_not_tracked(self):
        le, re_, lg, rg, tracked = self.am._hand_sample(
            self._bridge(self._snap(stale=True)))
        self.assertFalse(tracked)
        self.assertIs(self.am._last_two_hand[0], False)

    def test_gestures_and_pointing_read_the_owner_only(self):
        gs = self.mods[2]
        import sys
        from tests._skill_harness import load_skill_isolated
        prev = sys.modules.get("skill_kinect_pointing")
        self.addCleanup(lambda: sys.modules.__setitem__("skill_kinect_pointing", prev)
                        if prev is not None
                        else sys.modules.pop("skill_kinect_pointing", None))
        kp, _ = load_skill_isolated("kinect_pointing", register=False)
        snap = self._snap()
        b = self._bridge(snap)
        self.assertEqual(gs._owner_bodies(b), [snap["owner"]])
        self.assertIs(kp._shared_owner_body(b), snap["owner"])
        held = self._bridge(self._snap(fresh=False))
        self.assertEqual(gs._owner_bodies(held), [])
        self.assertIsNone(kp._shared_owner_body(held))
        stale = self._bridge(self._snap(stale=True))
        self.assertEqual(gs._owner_bodies(stale), [{"id": 99, "joints": {}}])
        self.assertIsNone(kp._shared_owner_body(stale))


class GestureSideLockTests(_Replay):
    def test_alternating_hands_fire_no_phantom_swipe(self):
        for period in (1, 4):
            res = self.replay(R.scenario_both_hands_alternating(
                period_frames=period))
            self.assertEqual(res.gestures, [], f"period {period}")


class CursorQualityTests(_Replay):
    """The shared pipeline against the legacy per-poll path on the SAME code
    and the SAME trace (on a tree without the shared layer these error: there
    is no shared pipeline to measure)."""

    @staticmethod
    def _jitter(res, t_from):
        xs = [(x, y) for t, x, y in res.moves if t >= t_from]
        steps = sorted(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                       for a, b in zip(xs, xs[1:]))
        return steps[int(0.95 * len(steps))], steps[len(steps) // 2]

    def test_rest_jitter_is_at_least_halved(self):
        frames = R.scenario_raise_and_hold(self.rows, seconds=20, lift=0.12)
        new95, new50 = self._jitter(self.replay(frames, shared=True), T0 + 2.0)
        old95, old50 = self._jitter(self.replay(frames, shared=False), T0 + 2.0)
        self.assertLess(new95, 0.5 * old95)
        self.assertLessEqual(new50, old50)

    @staticmethod
    def _crossing(series, level, rising):
        for (t0, x0), (t1, x1) in zip(series, series[1:]):
            if (x0 < level <= x1) if rising else (x0 > level >= x1):
                return t0 + (t1 - t0) * (level - x0) / (x1 - x0)
        return None

    def _lag_ms(self, distance, move_s, shared, reps=4, hold=1.0):
        frames = R.scenario_reach(hold=hold, repeats=reps, distance=distance,
                                  move_s=move_s)
        res = self.replay(frames, shared=shared)
        box = self.am.ReachBox(R.DESKTOP[2], R.DESKTOP[3])
        ideal = [(T0 + (k + 1) * R.FRAME_S, box.map(fr.right[0], R.lift_y(0.2))[0])
                 for k, fr in enumerate(frames)]
        cur = [(t, x) for t, x, y in res.moves]
        lags = []
        for rep in range(reps):
            a = T0 + hold + rep * (move_s + hold)
            si = [p for p in ideal if a - 0.05 <= p[0] <= a + move_s + 0.6]
            sc = [p for p in cur if a - 0.05 <= p[0] <= a + move_s + 1.0]
            lo, hi = si[0][1], si[-1][1]
            for frac in (0.25, 0.5, 0.75):
                lvl = lo + frac * (hi - lo)
                ti = self._crossing(si, lvl, hi > lo)
                tc = self._crossing(sc, lvl, hi > lo)
                if ti is not None and tc is not None:
                    lags.append((tc - ti) * 1000.0)
        self.assertGreaterEqual(len(lags), 3 * reps - 1)
        lags.sort()
        return lags[len(lags) // 2]

    def test_cursor_latency_at_normal_speeds_is_not_worse(self):
        for distance, move_s in ((0.30, 0.6), (0.15, 0.6)):   # 0.94 / 0.47 m/s peak
            new = self._lag_ms(distance, move_s, True)
            old = self._lag_ms(distance, move_s, False)
            self.assertLessEqual(new, old + 5.0, (distance, new, old))

    def test_slow_moves_cost_at_most_40ms_more(self):
        new = self._lag_ms(0.05, 0.5, True)                    # 0.19 m/s peak
        old = self._lag_ms(0.05, 0.5, False)
        self.assertLessEqual(new, old + 40.0, (new, old))


if __name__ == "__main__":
    unittest.main()
