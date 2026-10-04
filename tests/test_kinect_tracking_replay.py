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

    @staticmethod
    def _p(name, default):
        """A shared-stabiliser default; `default` (the shipped value) on a tree
        without that knob, so a before-run fails on BEHAVIOUR, not a KeyError."""
        try:
            from audio import kinect_stabilizer as ks
            return ks.DEFAULTS.get(name, default)
        except ImportError:
            return default


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
        # Behaviour only - holds on any tree (origin/main does nothing here
        # either; this is a regression guard, not a before/after claim).
        res = self.replay(R.scenario_desk(self.rows))
        self.assertEqual(res.edges("engaged"), [])
        self.assertEqual(res.presses(), [])
        self.assertEqual(res.edges("two"), [])
        self.assertEqual(res.gestures, [])

    def test_a_replay_leaves_no_bodies_behind(self):
        """Review 2: reset_tracking() kept the bridge's body cache, stamped in
        SIMULATED time (~1000 s) - on a machine up for less than that,
        get_bodies() served the fake body as fresh to whatever ran next."""
        self.replay(R.scenario_raise_and_hold(seconds=1))
        self.assertIsNone(kb._body_cache[0])
        self.assertIsNone(kb._get_cached_bodies(now=300.0))

    @unittest.skipUnless(SHARED, "shared snapshot only")
    def test_the_shared_layer_sees_every_desk_frame_and_stays_quiet(self):
        res = self.replay(R.scenario_desk(self.rows))
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

    def test_rest_jitter_filter_alone_with_equal_noise_on_hand_and_wrist(self):
        # FILTER ONLY: the same i.i.d. 3 mm jitter on the hand AND the wrist, so
        # reading the wrist first buys nothing - what is left is the One Euro
        # filter vs the old per-poll EMA. (Measured: p95 step 19 vs 59 px.)
        # lift 0.12 keeps the hand inside the reach box, so y is measured too.
        frames = R.scenario_raise_and_hold(seconds=20, lift=0.12,
                                           noise=R.iid_noise(600, 0.003))
        new95, new50 = self._jitter(self.replay(frames, shared=True), T0 + 2.0)
        old95, old50 = self._jitter(self.replay(frames, shared=False), T0 + 2.0)
        self.assertLess(new95, 0.5 * old95)
        self.assertLessEqual(new50, old50)

    def test_rest_jitter_on_the_recorded_noise_with_its_own_tracking_states(self):
        # The recorded at-rest residuals, each frame labelled with the hand /
        # wrist TrackingState it really had (the hand's widest excursions are
        # Inferred frames - the old fixture called them Tracked). Here the gain
        # is source selection (the Tracked wrist + offset) AND the filter.
        # (Measured: p95 step 109 vs 462 px.)
        frames = R.scenario_raise_and_hold(self.rows, seconds=20, lift=0.12,
                                           noise_states=True)
        new95, _ = self._jitter(self.replay(frames, shared=True), T0 + 2.0)
        old95, _ = self._jitter(self.replay(frames, shared=False), T0 + 2.0)
        self.assertLess(new95, 0.5 * old95)

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


# ══════════════════════════════════════════════════════════════════════════
#  2026-10-04 REVIEW FIXES, end to end. Each test reproduces one reviewer
#  finding through the real pipeline; "fails on" names the trees it failed
#  on (abf81ef = the first stabiliser commit, main = origin/main d5931da).
# ══════════════════════════════════════════════════════════════════════════
def _in(res, a, b):
    return [p for p in res.polls if T0 + a <= p["t"] < T0 + b]


def _frac(polls, key):
    return sum(1 for p in polls if p[key]) / max(1, len(polls))


class ReviewGripSafetyTests(_Replay):
    def test_a_fist_resting_on_the_desk_never_clicks(self):
        """HIGH, fails on abf81ef AND main: the NON-driving hand resting on the
        desk closing (Tracked, High confidence) right-clicked while the other
        hand drove the cursor."""
        res = self.replay(R.scenario_offhand_fist_at_desk())
        self.assertTrue(all(p["engaged"] for p in _in(res, 2.0, 2.5)))
        self.assertEqual(res.buttons, [])

    def test_a_drag_lets_go_when_the_hand_opens_at_low_confidence(self):
        """HIGH, fails on abf81ef: the SDK read the opened hand at Low
        confidence and the drag stayed down 2.4 s (1,477 px dragged)."""
        res = self.replay(R.scenario_drag_then_open(after=2, after_conf=0))
        downs = [t for t, a, _ in res.buttons if a == "down"]
        ups = [t for t, a, _ in res.buttons if a == "up"]
        self.assertEqual(len(downs), 1)
        self.assertEqual(len(ups), 1)
        self.assertLessEqual(ups[0] - (T0 + 2.5), 0.15)

    def test_a_drag_with_no_grip_evidence_is_bounded(self):
        """HIGH, fails on abf81ef AND main: after the press the SDK says only
        Unknown while the hand moves - the button stayed down until a
        confident open (2.4 s). Now it lets go after the hold bound."""
        res = self.replay(R.scenario_drag_then_open(after=0, after_conf=0))
        ups = [t for t, a, _ in res.buttons if a == "up"]
        self.assertEqual(len(ups), 1)
        cap = self._p("KINECT_GRIP_CLOSED_HOLD_MAX_SEC", 1.0)
        self.assertLessEqual(ups[0] - (T0 + 2.5), cap + 0.15)

    def test_pointing_or_a_low_confidence_fist_never_takes_the_cursor(self):
        """HIGH, fails on abf81ef: AIR_MOUSE_REQUIRE_OPEN_PALM (the owner's
        setting) was bypassed by a pointing hand (Lasso) or a fist read Low -
        engaged 89% of a 3.5 s hold. An open palm still engages."""
        for state, conf in ((4, 1), (4, 0), (3, 0)):
            res = self.replay(R.scenario_reach_with_hand_state(state=state,
                                                               conf=conf))
            self.assertEqual(res.edges("engaged"), [], (state, conf))
            self.assertEqual(res.moves, [], (state, conf))
        res = self.replay(R.scenario_reach_with_hand_state(state=2, conf=0))
        self.assertEqual(len(res.edges("engaged")), 1)

    def test_switching_hands_mid_drag_lets_go_before_the_cursor_moves(self):
        """HIGH, fails on abf81ef AND main: the hand that started a drag was
        lowered and the OTHER hand took the cursor - the button stayed down
        and the other hand swept it 5,900 px across the desktop."""
        res = self.replay(R.scenario_switch_mid_drag())
        downs = [t for t, a, _ in res.buttons if a == "down"]
        ups = [t for t, a, _ in res.buttons if a == "up"]
        self.assertEqual(len(downs), 1)
        self.assertEqual(len(ups), 1)
        # Not one cursor move between the press and the release lands on the
        # new hand: the cursor never left the dragging hand's last spot.
        held = [(x, y) for t, x, y in res.moves if downs[0] <= t <= ups[0]]
        before = [(x, y) for t, x, y in res.moves if t < ups[0]]
        self.assertTrue(held)
        hand_at_up = before[-1]
        self.assertTrue(all(abs(x - hand_at_up[0]) < 60 for x, _ in held[-3:]))
        self.assertLess(ups[0], T0 + 4.0)      # before the new hand sweeps

    def test_a_still_hand_cursor_never_moves_when_the_hand_joint_drops_out(self):
        """MEDIUM, fails on abf81ef: with the hand joint Inferred > 2 s (the
        wrist Tracked) the stand-in fell back to the raw wrist - the cursor of
        a perfectly still hand jumped 540 px, dragging a held button with it."""
        for grip_from in (None, 1.5):
            res = self.replay(R.scenario_hand_joint_inferred(grip_from=grip_from))
            ys = [y for t, x, y in res.moves if t >= T0 + 1.5]
            self.assertLessEqual(max(ys) - min(ys), 5, grip_from)
        self.assertEqual([a for _, a, _ in res.buttons], ["down"])


class ReviewTwoHandTests(_Replay):
    def test_a_second_hand_resting_under_the_line_gives_the_cursor_back(self):
        """MEDIUM, fails on abf81ef: after the second hand crossed the line
        once, two-hand mode held while it rested anywhere above -0.10 m -
        cursor 0%, two-hand 100% for 4.5 s. Time-in-state, not edges."""
        for rest in (0.0, -0.05):
            res = self.replay(R.scenario_second_hand_crosses_then_rests(
                rest_lift=rest))
            after = _in(res, 3.5, 8.0)
            self.assertGreaterEqual(_frac(after, "engaged"), 0.95, rest)
            self.assertEqual(_frac(after, "two"), 0.0, rest)

    def test_no_window_is_grabbed_with_a_hand_under_the_line(self):
        """MEDIUM, fails on abf81ef: with the second fist resting at chin
        height (lift 0.0) the window was grabbed and moved (9 SetWindowPos)
        and the driving hand's click never happened."""
        res = self.replay(R.scenario_second_hand_crosses_then_rests(
            rest_lift=0.0, fists_at=4.0))
        self.assertEqual(res.windows, [])
        self.assertNotIn("grabbed", {p["phase"] for p in res.polls})
        self.assertIn("left" if self.am._hand_mirror_enabled() else "right",
                      {b for _, a, b in res.buttons if a == "down"})

    def test_a_dip_mid_resize_can_be_regrabbed_and_the_mode_stays_on(self):
        """MEDIUM, fails on abf81ef: a 0.6 s dip under the engage line ended
        the grab, then the two-hand controller sat IDLE while the shared
        verdict kept the cursor stood down - nothing responded (0/119 polls)
        and RAISE_HAND fired 3 times. Now: it re-grabs, and the mode never
        blinks off while both hands stay up."""
        res = self.replay(R.scenario_two_hand_dip())
        grabs = [b["t"] for a, b in zip(res.polls, res.polls[1:])
                 if a["phase"] != "grabbed" and b["phase"] == "grabbed"]
        self.assertEqual(len(grabs), 2)
        self.assertGreater(grabs[1], T0 + 3.6)
        on = [p for p in res.polls if p["t"] >= T0 + 1.0]
        self.assertTrue(all(p["two"] for p in on))
        self.assertEqual(res.gestures, [])

    def test_no_grab_while_a_hand_sits_in_the_two_hand_band(self):
        """The two-hand MODE holds down to 4 cm under the line, but a window is
        only GRABBED with both hands at or over it: fists closed with one hand
        just under the line hold the pose without grabbing; raising it grabs."""
        res = self.replay(R.scenario_two_hand_band_fists())
        early = _in(res, 2.6, 4.0)
        self.assertTrue(early and all(p["two"] for p in early))
        self.assertNotIn("grabbed", {p["phase"] for p in early})
        grab = next(p["t"] for p in res.polls if p["phase"] == "grabbed")
        self.assertGreater(grab, T0 + 4.0)
        self.assertLessEqual(grab - (T0 + 4.0), 0.9)

    def test_three_resizes_in_a_row_each_grab(self):
        """Mutant 'two-hand re-grab block never cleared' survived: one grab,
        then two-hand mode never engaged again for the session."""
        res = self.replay(R.scenario_two_hand_cycles(cycles=3))
        grabs = [b["t"] for a, b in zip(res.polls, res.polls[1:])
                 if a["phase"] != "grabbed" and b["phase"] == "grabbed"]
        self.assertEqual(len(grabs), 3)
        self.assertEqual(len(res.edges("two")), 3)

    def test_a_hovering_second_hand_is_one_steady_mode_not_a_dead_zone(self):
        """Review 2: the flap tests counted edges only, so 'no flapping'
        could not be told from 'nothing works'. A second hand hovering AT the
        line (+0.07 +/- 3.5 cm) now holds ONE steady mode - two-hand - and at
        every poll after it settles either the cursor or two-hand mode is
        live. (Owner knob: KINECT_TWO_HAND_ENTER_ABOVE_M raises the two-hand
        entry bar if he wants the cursor there instead.)"""
        res = self.replay(R.scenario_second_hand_hover())
        live = [p for p in res.polls if p["t"] >= T0 + 1.0]
        covered = sum(1 for p in live if p["engaged"] or p["two"])
        self.assertGreaterEqual(covered / len(live), 0.98)
        self.assertLessEqual(len(res.edges("two")), 1)


class ReviewOwnerTests(_Replay):
    def test_the_owner_gets_the_cursor_back_after_a_dropout(self):
        """HIGH (review 2), fails on abf81ef: with a passive person 0.2 m
        farther in view, a 0.5 s dropout handed them the owner and the real
        owner never re-engaged (main: back 0.4 s after returning)."""
        for new_id in (False, True):
            res = self.replay(R.scenario_owner_dropout(new_id=new_id))
            back = [t for t in res.edges("engaged") if t > T0 + 3.5]
            self.assertTrue(back, new_id)
            self.assertLessEqual(back[0] - (T0 + 3.5), 1.0, new_id)

    def test_the_owner_wins_over_a_passive_person_seen_first(self):
        """HIGH (review 2), fails on abf81ef: a passive person seen first
        (0.1-0.2 m farther) kept the owner - 0% engaged over 7 s."""
        for dz in (0.10, 0.20):
            res = self.replay(R.scenario_passive_first(other_dz=dz))
            eng = res.edges("engaged")
            self.assertTrue(eng, dz)
            self.assertLessEqual(eng[0] - (T0 + 1.0), 1.0, dz)


@unittest.skipUnless(SHARED, "the shared snapshot's staleness is what's tested")
class ReviewPumpStallTests(_Replay):
    def test_a_short_pump_stall_keeps_a_drag(self):
        """MEDIUM (review 2), fails on abf81ef: a body-pump stall of 0.35 s
        or more read as tracking lost and dropped the drag (main held 0.6 s);
        a phantom RAISE_HAND followed."""
        for stall in (0.35, 0.45, 0.55):
            res = self.replay(R.scenario_drag_with_stall(stall=stall))
            self.assertEqual([a for _, a, _ in res.buttons], ["down", "up"],
                             stall)
            up = [t for t, a, _ in res.buttons if a == "up"][0]
            self.assertGreater(up, T0 + 5.95, stall)    # the hand opens at 6.0
            self.assertEqual(self.releases(res), [], stall)
            self.assertEqual(res.gestures, [], stall)

    def test_a_long_stall_still_lets_go_promptly(self):
        """Mutant 'shared controller grace 0 -> 0.30' survived: once the
        snapshot is stale the shared path must release at once (its one
        grace is the snapshot's), not 0.3 s later."""
        res = self.replay(R.scenario_drag_with_stall(stall=1.0))
        up = [t for t, a, _ in res.buttons if a == "up"][0]
        stale = self._p("KINECT_SNAPSHOT_STALE_SEC", 0.60)
        last_frame = T0 + 4.0
        self.assertLessEqual(up - (last_frame + stale), 0.10)
        self.assertEqual([w for _, w in self.releases(res)], ["tracking-lost"])


class ReviewTelemetryTests(_Replay):
    def test_an_owner_who_leaves_is_released_as_tracking_lost(self):
        """Mutant 'untracked snapshot read as tracked' survived (the release
        was logged as 'lowered')."""
        if not SHARED:
            self.skipTest("shared snapshot only")
        frames = R.scenario_raise_and_hold(seconds=3)
        for fr in frames[60:]:
            fr.body_id = None
        res = self.replay(frames)
        self.assertEqual([w for _, w in self.releases(res)], ["tracking-lost"])

    def test_the_telemetry_line_names_the_shared_active_hand_and_fps(self):
        am = self.am
        from unittest import mock
        snap = {"fps": 29.6, "hands": {"left": {"source": "wrist",
                                                "conf": "high"},
                                       "right": {"source": "hand",
                                                 "conf": "low"}}}
        lo = am.ArmExtension("left", 0.3, 0.9, (0, 0, 1, 2), lift_m=0.30)
        hi = am.ArmExtension("right", 0.3, 0.9, (0, 0, 1, 2), lift_m=0.40)
        ctrl = am.AirMouseController(am.ReachBox(100, 100))
        with mock.patch.object(am, "_last_active_side", ["left"]), \
                mock.patch.object(am, "_last_tracked_frame", [snap]), \
                mock.patch.object(am, "_last_two_hand", [False]), \
                mock.patch.object(am, "_hand_mirror_enabled", lambda: False):
            line = am._format_reach_debug(lo, hi, True, ctrl, yielding=False)
        self.assertIn("hand=left", line)      # the sticky hand, not the higher
        self.assertIn("src=wrist", line)
        self.assertIn("fps=30", line)


if __name__ == "__main__":
    unittest.main()
