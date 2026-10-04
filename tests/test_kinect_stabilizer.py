"""Unit tests for audio/kinect_stabilizer.py - the ONE shared per-frame hand
stabiliser (2026-10-04, "hand tracking is unstable") - and its bridge wiring
(audio.kinect_bridge.publish_body_frame / get_tracked_frame).

Every guard is driven at its LIVE default (kinect_stabilizer.DEFAULTS, pinned
equal to core/config.py below) with real frame times - no unrealistic knobs.
Synthetic numeric traces only. The end-to-end replay through the real
consumers lives in tests/test_kinect_tracking_replay.py.

    python -B -m unittest tests.test_kinect_stabilizer
"""
from __future__ import annotations

import ast
import math
import os
import random
import unittest

from audio import kinect_bridge as kb
from audio import kinect_stabilizer as ks

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 1.0 / 30.0
P = dict(ks.DEFAULTS)


def _config_literals() -> dict:
    with open(os.path.join(_PROJECT, "core", "config.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


# ─── synthetic bodies (generic geometry, camera space metres) ───────────────
SS = (0.0, 0.30, 1.20)          # spine_shoulder


def _joints(left=None, right=None, *, lstate=2, rstate=2, lwrist=2, rwrist=2):
    """A body's joints with each hand at the given (x, y, z) (None = at the
    desk). Wrist sits 6 cm below / 2 cm behind the hand."""
    j = {"spine_shoulder": SS + (2,), "spine_mid": (0.0, 0.05, 1.22, 2),
         "head": (0.0, 0.52, 1.19, 2), "neck": (0.0, 0.37, 1.20, 2),
         "shoulder_left": (-0.18, 0.28, 1.20, 2),
         "shoulder_right": (0.18, 0.28, 1.20, 2)}
    for side, hand, hs, ws in (("left", left, lstate, lwrist),
                               ("right", right, rstate, rwrist)):
        if hand is None:
            hand = (-0.2 if side == "left" else 0.2, 0.0, 1.05)
        j[f"hand_{side}"] = tuple(hand) + (hs,)
        j[f"wrist_{side}"] = (hand[0], hand[1] - 0.06, hand[2] + 0.02, ws)
        j[f"elbow_{side}"] = (hand[0], hand[1] - 0.25, hand[2] + 0.1, 2)
    return j


def _up(side, lift, x=None):
    return ((-0.15 if side == "left" else 0.15) if x is None else x,
            SS[1] + lift, 0.90)


def _body(joints, *, bid=1, dist=1.2, hl="open", hr="open", hlc="high",
          hrc="high"):
    return {"id": bid, "joints": joints, "distance_m": dist,
            "hand_left": hl, "hand_right": hr,
            "hand_left_conf": hlc, "hand_right_conf": hrc}


def _stab(**over):
    p = dict(P)
    p.update(over)
    return ks.HandStabilizer(arm_extension_fn=kb.arm_extension,
                             joint_reliable_fn=kb._joint_reliable,
                             params_fn=lambda: p)


class ConfigParityTests(unittest.TestCase):
    """The stabiliser's defaults and core/config.py are ONE value each (the
    stale-duplicate bug class): same names, same literals, each documented."""

    def test_every_default_is_the_config_literal(self):
        lits = _config_literals()
        for name, default in ks.DEFAULTS.items():
            self.assertIn(name, lits, f"{name} missing from core/config.py")
            self.assertEqual(lits[name], default, name)
            self.assertIs(type(lits[name]), type(default), name)

    def test_every_setting_has_a_comment(self):
        with open(os.path.join(_PROJECT, "core", "config.py"),
                  encoding="utf-8") as f:
            text = f.read()
        for name in list(ks.DEFAULTS) + ["KINECT_GESTURE_RELEASE_MUTE_SEC"]:
            self.assertIn(f"# {name} —", text, f"{name} has no comment")

    def test_lift_margins_match_the_air_mouse_gate(self):
        from tests._skill_harness import load_skill_isolated
        am, _ = load_skill_isolated("kinect_air_mouse", register=False)
        self.assertEqual(ks.DEFAULTS["KINECT_LIFT_UP_MARGIN"],
                         am.AIR_MOUSE_ENGAGE_UP_MARGIN_M)
        self.assertEqual(ks.DEFAULTS["KINECT_LIFT_DOWN_MARGIN"],
                         am.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)

    def test_gesture_mute_default_matches_config(self):
        from tests._skill_harness import load_skill_isolated
        gs, _ = load_skill_isolated("kinect_gestures", register=False)
        self.assertEqual(gs._RELEASE_MUTE_SEC_DEFAULT,
                         _config_literals()["KINECT_GESTURE_RELEASE_MUTE_SEC"])

    def test_load_params_coerces_and_falls_back(self):
        vals = {"KINECT_GRIP_CLOSE_MIN_VOTES": "5", "KINECT_GRIP_CLOSE_SEC": "nan",
                "KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE": "false",
                "KINECT_HAND_FILTER_BETA": "oops"}
        p = ks.load_params(lambda n, d: vals.get(n, d))
        self.assertEqual(p["KINECT_GRIP_CLOSE_MIN_VOTES"], 5)
        self.assertEqual(p["KINECT_GRIP_CLOSE_SEC"], ks.DEFAULTS["KINECT_GRIP_CLOSE_SEC"])
        self.assertIs(p["KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE"], False)
        self.assertEqual(p["KINECT_HAND_FILTER_BETA"], ks.DEFAULTS["KINECT_HAND_FILTER_BETA"])


class OneEuroTests(unittest.TestCase):
    def _f(self):
        return ks.OneEuro3(P["KINECT_HAND_FILTER_MIN_CUTOFF_HZ"],
                           P["KINECT_HAND_FILTER_BETA"],
                           P["KINECT_HAND_FILTER_D_CUTOFF_HZ"])

    @staticmethod
    def _ramp_lag_ms(filt, v):
        x = 0.0
        y = None
        for i in range(120):
            x = v * i * T
            y = filt(x)
        return (x - y) / v * 1000.0

    def test_added_lag_under_40ms_at_normal_hand_speeds(self):
        for v in (0.25, 0.5, 1.0):
            f = self._f()
            lag = self._ramp_lag_ms(lambda x: f.filter((x, 0, 0), T)[0], v)
            self.assertLess(lag, 40.0, f"{v} m/s: {lag:.1f} ms")

    def test_faster_than_the_old_ema_from_half_a_metre_per_second(self):
        old = {"v": None}

        def ema(x):
            old["v"] = x if old["v"] is None else 0.55 * x + 0.45 * old["v"]
            return old["v"]
        old_lag = self._ramp_lag_ms(ema, 0.5)
        f = self._f()
        new_lag = self._ramp_lag_ms(lambda x: f.filter((x, 0, 0), T)[0], 0.5)
        self.assertLess(new_lag, old_lag)

    def test_rest_jitter_far_below_the_old_ema(self):
        rnd = random.Random(7)
        noise = [rnd.gauss(0.0, 0.004) for _ in range(900)]   # 4 mm, as measured
        f = self._f()
        new = [f.filter((n, 0, 0), T)[0] for n in noise]
        v = None
        old = []
        for n in noise:
            v = n if v is None else 0.55 * n + 0.45 * v
            old.append(v)

        def sd(xs):
            xs = xs[60:]
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        self.assertLess(sd(new), 0.6 * sd(old))


class GripFilterTests(unittest.TestCase):
    def _run(self, seq, conf="high", tracked=True, **over):
        """Feed one vote per 30 Hz frame; return the stable grip per frame."""
        p = dict(P)
        p.update(over)
        g = ks.GripFilter()
        out = []
        for i, s in enumerate(seq):
            c = conf[i] if isinstance(conf, (list, tuple)) else conf
            out.append(g.update(ks.GripFilter.vote_for(s, c, tracked, p),
                                100.0 + i * T, p))
        return out

    def test_flickers_of_three_frames_or_less_never_press(self):
        for n in (1, 2, 3):
            seq = ["open"] * 10 + ["closed"] * n + ["open"] * 10
            self.assertNotIn("closed", self._run(seq), f"{n}-frame flicker")

    def test_a_real_grip_presses_on_its_fourth_frame(self):
        out = self._run(["open"] * 5 + ["closed"] * 8)
        self.assertEqual(out.index("closed"), 5 + 3)

    def test_low_confidence_never_presses(self):
        out = self._run(["open"] * 5 + ["closed"] * 60, conf="low")
        self.assertNotIn("closed", out)

    def test_an_untracked_hand_never_presses(self):
        out = self._run(["open"] * 5 + ["closed"] * 60, tracked=False)
        self.assertNotIn("closed", out)

    def test_lasso_is_no_vote_by_default_and_configurable(self):
        seq = ["open"] * 5 + ["lasso"] * 20
        self.assertNotIn("closed", self._run(seq))
        self.assertIn("closed", self._run(seq, KINECT_GRIP_LASSO_AS="closed"))

    def test_one_open_frame_mid_drag_does_not_release(self):
        seq = ["closed"] * 6 + ["open"] + ["closed"] * 6
        out = self._run(seq)
        self.assertEqual(out[-1], "closed")
        self.assertEqual(out[5:], ["closed"] * 8)

    def test_two_open_frames_release(self):
        out = self._run(["closed"] * 6 + ["open"] * 2)
        self.assertEqual(out[-1], "open")

    def test_low_confidence_contrary_frames_break_nothing(self):
        # Closed High x2, then Low frames, then closed High x2: the Low frames
        # are no-votes inside the gap, so the run continues to 4 votes.
        states = ["closed"] * 6
        confs = ["high", "high", "low", "high", "high", "high"]
        out = self._run(states, conf=confs)
        self.assertEqual(out[4], "closed")

    def test_a_long_vote_gap_restarts_the_run(self):
        states = ["closed", "closed", "unknown", "unknown", "unknown", "unknown",
                  "closed", "closed"]
        self.assertNotIn("closed", self._run(states))

    def test_a_late_timestamp_cannot_turn_a_3_frame_flicker_into_a_click(self):
        # The pump starved (CPU pinned at 90% on 2026-10-04 13:38): the third
        # closed frame of a flicker is parsed 40 ms late. Its span now passes
        # the time dwell - the 4-frame minimum still refuses the press.
        g = ks.GripFilter()
        for t in (100.0, 100.0 + T, 100.0 + 2 * T + 0.040):
            g.update("closed", t, P)
        self.assertEqual(g.stable, "open")

    def test_a_burst_of_frames_cannot_shortcut_the_close_dwell(self):
        # Four closed frames delivered 10 ms apart (a pump catching up) are
        # four votes but only 30 ms of evidence: no press.
        g = ks.GripFilter()
        for i in range(4):
            g.update("closed", 100.0 + i * 0.010, P)
        self.assertEqual(g.stable, "open")
        g.update("closed", 100.0 + 0.095, P)
        self.assertEqual(g.stable, "closed")

    def test_a_stall_between_votes_restarts_the_run(self):
        # Two closed frames, then the pump stalls 300 ms (no frames at all),
        # then two more: two runs of 2, never one run of 4 - no press.
        g = ks.GripFilter()
        for t in (100.0, 100.0 + T, 100.3 + 2 * T, 100.3 + 3 * T):
            g.update("closed", t, P)
        self.assertEqual(g.stable, "open")

    def test_the_release_vote_minimum_holds_even_with_no_time_dwell(self):
        # Belt and braces: with KINECT_GRIP_OPEN_SEC set to 0 the 2-frame
        # minimum alone still keeps one open frame from dropping a drag.
        out = self._run(["closed"] * 6 + ["open"] + ["closed"] * 3,
                        KINECT_GRIP_OPEN_SEC=0.0)
        self.assertEqual(out[6:], ["closed"] * 4)

    def test_the_same_frame_fed_twice_counts_once(self):
        g = ks.GripFilter()
        for i in range(2):
            for _ in range(3):           # three polls of the same frame
                g.update("closed", 100.0 + i * T, P)
        self.assertEqual(g.stable, "open")
        self.assertEqual(g._votes, 2)


class HandTrackTests(unittest.TestCase):
    def _feed(self, st, frames, t0=100.0):
        snaps = []
        for i, (left, right, kw) in enumerate(frames):
            snaps.append(st.process([_body(_joints(left, right, **kw))],
                                    t0 + i * T))
        return snaps

    def test_one_inferred_frame_keeps_lift_and_raised(self):
        st = _stab()
        up = _up("right", 0.25)
        frames = [(None, up, {})] * 20
        frames += [(None, up, {"rstate": 1, "rwrist": 1})]
        frames += [(None, up, {})] * 5
        snaps = self._feed(st, frames)
        h = snaps[20]["hands"]["right"]
        self.assertEqual(h["source"], "inferred")     # a guess: position only
        self.assertFalse(h["lift_measured"])
        self.assertIsNotNone(h["lift"])               # ...the lift is HELD
        self.assertTrue(h["raised"])
        self.assertEqual(snaps[20]["active"], "right")

    def test_loss_grace_then_gone(self):
        st = _stab()
        up = _up("right", 0.25)
        frames = [(None, up, {})] * 20 + [(None, up, {"rstate": 0, "rwrist": 0})] * 15
        snaps = self._feed(st, frames)
        lost = [s["hands"]["right"] for s in snaps[20:]]
        # Held for the grace (0.30 s = 9 frames), then gone.
        held = [h for h in lost if h["pos"] is not None]
        self.assertEqual(len(held), 9)
        self.assertTrue(all(h["raised"] for h in held))
        self.assertIsNone(lost[-1]["pos"])
        self.assertIsNone(lost[-1]["lift"])
        self.assertFalse(lost[-1]["raised"])
        self.assertIsNone(snaps[-1]["active"])

    def test_single_frame_jump_is_rejected(self):
        st = _stab()
        up = _up("right", 0.20)
        far = (up[0] + 0.6, up[1], up[2])
        snaps = self._feed(st, [(None, up, {})] * 10 + [(None, far, {})]
                           + [(None, up, {})] * 3)
        p10, p9 = snaps[10]["hands"]["right"]["pos"], snaps[9]["hands"]["right"]["pos"]
        self.assertLess(abs(p10[0] - p9[0]), 0.01)

    def test_a_jump_that_stays_is_accepted(self):
        st = _stab()
        up = _up("right", 0.20)
        far = (up[0] + 0.6, up[1], up[2])
        snaps = self._feed(st, [(None, up, {})] * 10 + [(None, far, {})] * 4)
        self.assertAlmostEqual(snaps[-1]["hands"]["right"]["pos"][0], far[0],
                               delta=0.01)

    def test_wrist_stand_in_has_no_step(self):
        st = _stab()
        up = _up("right", 0.20)
        snaps = self._feed(st, [(None, up, {})] * 30
                           + [(None, up, {"rstate": 1})] * 10)
        a = snaps[29]["hands"]["right"]
        b = snaps[35]["hands"]["right"]
        self.assertEqual(b["source"], "wrist")
        self.assertTrue(b["lift_measured"])
        self.assertLess(abs(b["pos"][1] - a["pos"][1]), 0.005)

    def test_inferred_only_never_measures_a_raise(self):
        st = _stab()
        up = _up("right", 0.30)
        snaps = self._feed(st, [(None, up, {"rstate": 1, "rwrist": 1})] * 30)
        self.assertFalse(any(s["hands"]["right"]["raised"] for s in snaps))
        self.assertTrue(all(s["hands"]["right"]["lift"] is None for s in snaps))

    def test_raise_needs_the_enter_dwell(self):
        st = _stab()
        frames = ([(None, None, {})] * 5 + [(None, _up("right", 0.20), {})] * 2
                  + [(None, None, {})] * 10)
        snaps = self._feed(st, frames)
        self.assertFalse(any(s["hands"]["right"]["raised"] for s in snaps))
        st = _stab()
        snaps = self._feed(st, [(None, _up("right", 0.20), {})] * 8)
        first = next(i for i, s in enumerate(snaps) if s["hands"]["right"]["raised"])
        self.assertLessEqual(first * T, P["KINECT_RAISE_ENTER_SEC"] + 2 * T)

    def test_a_two_frame_spike_over_the_line_is_not_a_raise(self):
        # Hand just under the engage line, then two frames 10 cm higher (a
        # depth glitch / a twitch), then back: never "raised".
        st = _stab()
        frames = ([(None, _up("right", 0.04), {})] * 15
                  + [(None, _up("right", 0.14), {})] * 2
                  + [(None, _up("right", 0.04), {})] * 15)
        snaps = self._feed(st, frames)
        self.assertFalse(any(s["hands"]["right"]["raised"] for s in snaps))

    def test_hysteresis_holds_a_hand_at_the_line(self):
        st = _stab()
        frames = [(None, _up("right", 0.20), {})] * 15
        for i in range(60):          # wobble around the UP margin, never < DOWN
            frames.append((None, _up("right", 0.07 + 0.05 * math.sin(i)), {}))
        snaps = self._feed(st, frames)
        self.assertTrue(all(s["hands"]["right"]["raised"] for s in snaps[15:]))


class ActiveHandTests(unittest.TestCase):
    def _feed(self, st, seq, t0=100.0):
        return [st.process([_body(_joints(l, r))], t0 + i * T)
                for i, (l, r) in enumerate(seq)]

    def test_holder_kept_against_a_small_or_brief_lead(self):
        st = _stab()
        seq = [(None, _up("right", 0.20))] * 15
        seq += [(_up("left", 0.30), _up("right", 0.20))] * 30     # lead 0.10
        seq += [(_up("left", 0.45), _up("right", 0.20))] * 8      # lead 0.25, 0.27 s
        seq += [(_up("left", 0.30), _up("right", 0.20))] * 10
        snaps = self._feed(st, seq)
        self.assertEqual({s["active"] for s in snaps[15:]}, {"right"})

    def test_clear_sustained_lead_switches_after_the_dwell(self):
        st = _stab()
        seq = [(None, _up("right", 0.20))] * 15
        seq += [(_up("left", 0.45), _up("right", 0.20))] * 30
        snaps = self._feed(st, seq)
        sw = next(i for i, s in enumerate(snaps) if s["active"] == "left")
        self.assertGreaterEqual((sw - 15) * T, P["KINECT_ACTIVE_HAND_SWITCH_SEC"])
        self.assertLessEqual((sw - 15) * T, P["KINECT_ACTIVE_HAND_SWITCH_SEC"]
                             + P["KINECT_RAISE_ENTER_SEC"] + 0.15)

    def test_lowering_the_holder_hands_over_at_once(self):
        st = _stab()
        seq = [(_up("left", 0.15), _up("right", 0.20))] * 20
        seq += [(_up("left", 0.15), None)] * 12
        snaps = self._feed(st, seq)
        self.assertEqual(snaps[19]["active"], "right")
        sw = next(i for i, s in enumerate(snaps) if s["active"] == "left")
        # The lowered hand exits after its exit dwell (+ filter settling).
        self.assertLessEqual((sw - 20) * T, 0.30)


class TwoHandGateTests(unittest.TestCase):
    def test_single_frame_coincidence_never_engages(self):
        g = ks.TwoHandGate()
        for i in range(100):
            g.update(i % 5 == 0, 100.0 + i * T, P)
            self.assertFalse(g.active)

    def test_engage_edges_are_at_least_a_second_apart_whatever_the_input(self):
        min_gap = (P["KINECT_TWO_HAND_EXIT_SEC"] + P["KINECT_TWO_HAND_REARM_SEC"]
                   + P["KINECT_TWO_HAND_ENTER_SEC"])
        self.assertGreater(min_gap, 1.0)
        for seed in range(25):
            rnd = random.Random(seed)
            g = ks.TwoHandGate()
            edges, prev, state = [], False, False
            for i in range(1800):
                if rnd.random() < 0.15:      # random flapping input
                    state = not state
                t = 100.0 + i * T
                g.update(state, t, P)
                if g.active and not prev:
                    edges.append(t)
                prev = g.active
            gaps = [b - a for a, b in zip(edges, edges[1:])]
            self.assertTrue(all(gap >= min_gap - 1e-6 for gap in gaps),
                            (seed, gaps[:5]))

    def test_deliberate_both_hands_engage_and_release(self):
        g = ks.TwoHandGate()
        on = [g.update(True, 100.0 + i * T, P) for i in range(15)]
        first = on.index(True)
        self.assertAlmostEqual(first * T, P["KINECT_TWO_HAND_ENTER_SEC"], delta=T)
        off = [g.update(False, 100.5 + i * T, P) for i in range(15)]
        self.assertAlmostEqual(off.index(False) * T, P["KINECT_TWO_HAND_EXIT_SEC"],
                               delta=T)


class OwnerTests(unittest.TestCase):
    def _two(self, st, frames, near_dist):
        snaps = []
        for i in range(frames):
            bodies = [_body(_joints(), bid=1, dist=1.2)]
            if near_dist is not None:
                bodies.append(_body(_joints(), bid=2, dist=near_dist))
            snaps.append(st.process(bodies, 100.0 + i * T))
        return snaps

    def test_a_passer_by_nearer_for_under_a_second_does_not_steal(self):
        st = _stab()
        self._two(st, 10, None)
        snaps = self._two(st, 25, 0.8)      # 0.4 m nearer for 0.83 s
        self.assertEqual({s["owner_id"] for s in snaps}, {1})

    def test_a_clearly_nearer_body_takes_over_after_the_dwell(self):
        st = _stab()
        self._two(st, 10, None)
        snaps = self._two(st, 40, 0.8)
        self.assertEqual(snaps[-1]["owner_id"], 2)

    def test_owner_held_through_a_short_dropout(self):
        st = _stab()
        st.process([_body(_joints(None, _up("right", 0.2)))], 100.0)
        for i in range(1, 6):
            st.process([_body(_joints(None, _up("right", 0.2)))], 100.0 + i * T)
        s = st.process([], 100.0 + 6 * T)
        self.assertTrue(s["tracked"])
        self.assertFalse(s["fresh"])
        self.assertEqual(s["owner_id"], 1)
        s = st.process([], 100.0 + 6 * T + 0.35)
        self.assertFalse(s["tracked"])
        self.assertIsNone(s["owner_id"])


class BridgeWiringTests(unittest.TestCase):
    def setUp(self):
        kb.reset_tracking()
        self.addCleanup(kb.reset_tracking)
        # publish_body_frame fills the bridge's body cache with SIMULATED
        # stamps; never leave it for a later test (2026-10-04 review).
        self.addCleanup(_clear_body_cache)

    def test_publish_advances_once_per_frame_and_snapshot_ages(self):
        b = [_body(_joints(None, _up("right", 0.2)))]
        kb.publish_body_frame(b, now=50.0)
        s1 = kb.get_tracked_frame(now=50.0)
        kb.publish_body_frame(b, now=50.0 + T)
        s2 = kb.get_tracked_frame(now=50.0 + T)
        self.assertEqual(s2["seq"], s1["seq"] + 1)
        self.assertFalse(s2["stale"])
        # A pump stall SHORTER than KINECT_SNAPSHOT_STALE_SEC holds (a drag
        # survives it); a longer one reads as stale (not tracked).
        stale_after = ks.DEFAULTS["KINECT_SNAPSHOT_STALE_SEC"]
        self.assertGreater(stale_after, ks.DEFAULTS["KINECT_OWNER_LOSS_GRACE_SEC"])
        self.assertFalse(kb.get_tracked_frame(now=50.0 + T + 0.45)["stale"])
        self.assertTrue(kb.get_tracked_frame(
            now=50.0 + T + stale_after + 0.01)["stale"])
        # get_bodies' cache got the same frame.
        self.assertEqual(kb._body_cache[0], b)

    def test_pushed_lift_margins_reach_the_shared_gate(self):
        kb.set_lift_margins(0.20, 0.05)
        b = [_body(_joints(None, _up("right", 0.15)))]
        for i in range(10):
            kb.publish_body_frame(b, now=60.0 + i * T)
        self.assertFalse(kb.get_tracked_frame(now=60.3)["hands"]["right"]["raised"])
        kb.set_lift_margins(0.30, 0.50)          # inverted pair: ignored
        self.assertEqual(kb._lift_margin_override[0], (0.20, 0.05))

    def test_stop_pump_clears_the_snapshot(self):
        kb.publish_body_frame([_body(_joints())], now=70.0)
        self.assertIsNotNone(kb.get_tracked_frame(now=70.0))
        kb.stop_body_pump()
        self.assertIsNone(kb.get_tracked_frame(now=70.0))

    def test_a_broken_frame_never_raises(self):
        kb.publish_body_frame([{"id": 1, "joints": {"hand_left": ("x",)}}], now=80.0)
        snap = kb.get_tracked_frame(now=80.0)
        self.assertIsNotNone(snap)


# ══════════════════════════════════════════════════════════════════════════
#  2026-10-04 REVIEW FIXES. A docstring that names a review finding marks a
#  test that FAILED on abf81ef (the first stabiliser commit); one that names a
#  "Mutant" pins a rule a guard mutation showed was unprotected; the rest are
#  the guard rails of a fix (what it must NOT start doing).
# ══════════════════════════════════════════════════════════════════════════
def _grip_trace(st, states, *, side="right", confs="high", jstates=2,
                lift=0.25, t0=100.0):
    """Feed one frame per (state, conf, joint-state) with `side` raised;
    returns the snapshots. states/confs/jstates: a list or one value."""
    snaps = []
    for i, s in enumerate(states):
        c = confs[i] if isinstance(confs, (list, tuple)) else confs
        js = jstates[i] if isinstance(jstates, (list, tuple)) else jstates
        if side == "right":
            j = _joints(None, _up("right", lift), rstate=js)
        else:
            j = _joints(_up("left", lift), None, lstate=js)
        b = _body(j, **({"hr": s, "hrc": c} if side == "right"
                        else {"hl": s, "hlc": c}))
        snaps.append(st.process([b], t0 + i * T))
    return snaps


class GripEvidenceTests(unittest.TestCase):
    """The click grip: confident presses, any-confidence releases, and a hold
    that can't outlive the evidence for it."""

    def grips(self, states, **kw):
        return [s["hands"]["right"]["grip"]
                for s in _grip_trace(_stab(), states, **kw)]

    def test_a_low_confidence_open_releases_a_drag(self):
        """HIGH (both reviewers): a Low-confidence Open counted as no vote, so
        a drag stayed down until a High open came (2.4 s, 1,477 px dragged)."""
        g = self.grips(["closed"] * 10 + ["open"] * 4,
                       confs=["high"] * 10 + ["low"] * 4)
        self.assertEqual(g[9], "closed")
        self.assertEqual(g[-1], "open")
        self.assertLessEqual(g.index("open", 10) - 10, 1)   # 2 frames = 33 ms

    def test_low_confidence_still_never_presses(self):
        self.assertNotIn("closed", self.grips(["open"] * 5 + ["closed"] * 60,
                                              confs="low"))

    def test_a_low_confidence_open_breaks_a_run_toward_a_press(self):
        # closed High x3, ONE Low open, closed High x3: never 4 in a row.
        g = self.grips(["open"] * 5 + ["closed"] * 3 + ["open"] + ["closed"] * 3,
                       confs=["high"] * 8 + ["low"] + ["high"] * 3)
        self.assertNotIn("closed", g)

    def test_alternating_closed_open_flicker_never_presses(self):
        """Mutant 'an agreeing vote does not clear the candidate' survived the
        old tests (they only used contiguous flicker runs)."""
        g = self.grips(["open"] * 5 + ["closed", "open"] * 20)
        self.assertNotIn("closed", g)

    def test_a_hold_with_no_evidence_lets_go(self):
        """A pressed hand the SDK then calls Unknown for good: the button is
        let go after KINECT_GRIP_CLOSED_HOLD_MAX_SEC instead of never."""
        cap = P["KINECT_GRIP_CLOSED_HOLD_MAX_SEC"]
        n = int(cap / T) + 6
        g = self.grips(["closed"] * 6 + ["unknown"] * n)
        self.assertEqual(g[5], "closed")
        first_open = g.index("open", 6)
        self.assertAlmostEqual((first_open - 5) * T, cap, delta=2 * T)

    def test_a_closed_reading_on_an_inferred_joint_keeps_the_hold(self):
        # The SDK still says Closed while it only Infers the hand joint: no
        # vote (it can't press or release), but evidence to keep the drag.
        n = int(2.0 / T)
        g = self.grips(["closed"] * (6 + n), jstates=[2] * 6 + [1] * n)
        self.assertEqual(set(g[5:]), {"closed"})

    def test_an_open_reading_on_an_inferred_joint_never_releases_early(self):
        n = int(0.5 / T)
        g = self.grips(["closed"] * 6 + ["open"] * n, jstates=[2] * 6 + [1] * n)
        self.assertEqual(set(g[5:]), {"closed"})

    def test_the_grip_resets_when_the_hand_is_lost(self):
        """Mutant 'grip reset when hand lost' survived: a fist lost past the
        grace must not come back 'closed' (a phantom press on re-engage)."""
        st = _stab()
        _grip_trace(st, ["closed"] * 8)
        gone = [_body(_joints(None, _up("right", 0.25), rstate=0, rwrist=0),
                      hr="unknown")]
        for i in range(15):
            st.process(gone, 100.0 + (8 + i) * T)
        s = st.process([_body(_joints(None, _up("right", 0.25)), hr="unknown")],
                       100.0 + 23 * T)
        self.assertEqual(s["hands"]["right"]["grip"], "open")


class PalmSignalTests(unittest.TestCase):
    """HIGH (review 1): AIR_MOUSE_REQUIRE_OPEN_PALM read the CLICK grip, which
    defaults to 'open' on no confident evidence - so a pointing hand (Lasso) or
    a fist read at Low confidence passed the open-palm gate."""

    def palms(self, states, confs="high"):
        return [s["hands"]["right"]["palm"]
                for s in _grip_trace(_stab(), states, confs=confs)]

    def test_lasso_and_a_low_confidence_fist_are_not_an_open_palm(self):
        self.assertEqual(self.palms(["lasso"] * 10)[-1], "closed")
        self.assertEqual(self.palms(["closed"] * 10, confs="low")[-1], "closed")

    def test_an_open_hand_at_any_confidence_is_an_open_palm(self):
        p = self.palms(["lasso"] * 10 + ["open"] * 4, confs="low")
        self.assertEqual(p[-1], "open")

    def test_unknown_carries_no_evidence(self):
        # Legacy-equal: a hand the classifier never reads stays "open".
        self.assertEqual(set(self.palms(["unknown"] * 20)), {"open"})

    def test_a_single_lasso_frame_does_not_flip_it(self):
        p = self.palms(["open"] * 5 + ["lasso"] + ["open"] * 5)
        self.assertEqual(set(p), {"open"})


class WristOffsetTests(unittest.TestCase):
    """MEDIUM (review 1): after KINECT_WRIST_OFFSET_MAX_AGE_SEC the stand-in
    fell back to the RAW wrist - a still hand's cursor stepped 540 px, and
    stepped back when the hand joint was Tracked again."""

    def test_a_still_hand_does_not_move_through_a_long_inferred_stretch(self):
        st = _stab()
        up = _up("right", 0.10)
        frames = [(None, up, {})] * 30 + [(None, up, {"rstate": 1})] * 120 \
            + [(None, up, {})] * 30
        ys = [st.process([_body(_joints(l, r, **kw))], 100.0 + i * T)
              ["hands"]["right"]["pos"][1] for i, (l, r, kw) in enumerate(frames)]
        self.assertLess(max(ys[29:]) - min(ys[29:]), 0.002)

    def test_a_relearned_offset_never_steps_the_hand(self):
        # The hand joint comes back Tracked 3 cm off the stale offset: the hand
        # glides there (low-pass), no single-frame jump.
        st = _stab()
        up = _up("right", 0.10)
        shifted = (up[0] + 0.03, up[1], up[2])
        frames = [(None, up, {})] * 30 + [(None, up, {"rstate": 1})] * 90 \
            + [(None, shifted, {})] * 60
        xs = [st.process([_body(_joints(l, r, **kw))], 100.0 + i * T)
              ["hands"]["right"]["pos"][0] for i, (l, r, kw) in enumerate(frames)]
        steps = [abs(b - a) for a, b in zip(xs[119:], xs[120:])]
        self.assertLess(max(steps), 0.012)
        self.assertAlmostEqual(xs[-1], shifted[0], delta=0.004)


class HandTrackEdgeTests(unittest.TestCase):
    def _feed(self, st, frames, t0=100.0, dts=None):
        out, t = [], t0
        for i, (l, r, kw) in enumerate(frames):
            if dts and i in dts:
                t += dts[i]
            else:
                t += T
            out.append(st.process([_body(_joints(l, r, **kw))], t))
        return out

    def test_a_short_dip_under_the_stay_line_does_not_lower(self):
        """Mutant 'raise EXIT dwell' survived: a 2-frame dip below DOWN
        (a depth glitch) must not lower a raised hand."""
        st = _stab()
        # Raised, resting in the band at lift 0.0; a 2-frame dip to -0.22 (a
        # 0.22 m step - under the jump-reject bar, so it IS measured).
        frames = ([(None, _up("right", 0.20), {})] * 15
                  + [(None, _up("right", 0.0), {})] * 30
                  + [(None, _up("right", -0.22), {})] * 2
                  + [(None, _up("right", 0.0), {})] * 10)
        snaps = self._feed(st, frames)
        dip = [s["hands"]["right"]["lift"] for s in snaps[45:47]]
        self.assertLess(min(dip), P["KINECT_LIFT_DOWN_MARGIN"])   # it did cross
        self.assertTrue(all(s["hands"]["right"]["raised"] for s in snaps[10:]))

    def test_a_real_move_after_a_frame_gap_is_not_rejected_as_a_jump(self):
        """Mutant 'jump threshold not scaled by elapsed frames' survived: after
        a 0.3 s pump gap the hand legitimately moved 0.4 m - accept it at
        once instead of dropping frames as glitches."""
        st = _stab()
        a = _up("right", 0.20)
        b = (a[0] - 0.40, a[1], a[2])
        frames = [(None, a, {})] * 10 + [(None, b, {})] * 3
        snaps = self._feed(st, frames, dts={10: 0.30})
        self.assertTrue(snaps[10]["hands"]["right"]["measured"])
        self.assertAlmostEqual(snaps[10]["hands"]["right"]["pos"][0], b[0],
                               delta=0.05)


class ActiveHandChallengeTests(unittest.TestCase):
    def test_an_interrupted_challenge_does_not_accumulate(self):
        """Mutant 'challenge accumulates across interruptions' survived: a
        lead that keeps breaking off must restart the dwell every time."""
        st = _stab()
        seq = [(None, _up("right", 0.20))] * 15
        for _ in range(6):     # 0.30 s leading, 0.10 s not - never 0.40 s
            seq += [(_up("left", 0.45), _up("right", 0.20))] * 9
            seq += [(_up("left", 0.25), _up("right", 0.20))] * 3
        snaps = [st.process([_body(_joints(l, r))], 100.0 + i * T)
                 for i, (l, r) in enumerate(seq)]
        self.assertEqual({s["active"] for s in snaps[15:]}, {"right"})


class TwoHandBandTests(unittest.TestCase):
    """MEDIUM (review 1): two-hand mode ended only when a hand fell below the
    single-hand STAY line (-0.10 m), so a second hand resting at chin / chest
    height after crossing the line once kept the cursor locked out."""

    def _run(self, left_lifts):
        st = _stab()
        snaps = []
        for i, ll in enumerate(left_lifts):
            snaps.append(st.process(
                [_body(_joints(_up("left", ll), _up("right", 0.25)))],
                100.0 + i * T))
        return snaps

    def test_a_second_hand_resting_under_the_line_ends_two_hand_mode(self):
        for rest in (0.0, -0.05):
            snaps = self._run([0.25] * 20 + [rest] * 30)
            self.assertTrue(snaps[19]["two_hand"])
            self.assertFalse(snaps[-1]["two_hand"], rest)

    def test_two_hand_mode_holds_inside_its_own_band(self):
        up, below = P["KINECT_LIFT_UP_MARGIN"], P["KINECT_TWO_HAND_EXIT_BELOW_M"]
        snaps = self._run([0.25] * 20 + [up - below / 2] * 30)
        self.assertTrue(all(s["two_hand"] for s in snaps[19:]))

    def test_entry_needs_both_hands_over_the_line(self):
        up = P["KINECT_LIFT_UP_MARGIN"]
        # Raised once (so 'raised' holds), then parked just under the line.
        snaps = self._run([0.25] * 4 + [up - 0.02] * 40)
        self.assertFalse(any(s["two_hand"] for s in snaps))


class OwnerClaimTests(unittest.TestCase):
    """HIGH (review 2): the sticky owner could lock the real owner out for
    good - whoever was seen first, or whoever a 0.3 s dropout fell to, kept the
    cursor unless the real owner sat 0.25 m nearer for 1 s."""

    @staticmethod
    def _frames(st, n, bodies_fn, t0):
        return [st.process(bodies_fn(i), t0 + i * T) for i in range(n)]

    def test_the_real_owner_claims_it_back_by_raising_a_hand(self):
        st = _stab()

        def passive():
            return _body(_joints(), bid=2, dist=1.4)

        def owner():
            return _body(_joints(None, _up("right", 0.25)), bid=1)
        # A passive person (hands down) is seen FIRST and becomes the owner.
        self._frames(st, 10, lambda i: [passive()], 100.0)
        # The real owner arrives 0.2 m NEARER with a hand raised.
        snaps = self._frames(st, 30, lambda i: [passive(), owner()], 100.0 + 10 * T)
        claim = next(i for i, s in enumerate(snaps) if s["owner_id"] == 1)
        self.assertLessEqual(claim * T, P["KINECT_OWNER_CLAIM_SEC"] + 2 * T)
        self.assertEqual(snaps[-1]["owner_id"], 1)

    def test_a_farther_body_can_never_claim(self):
        st = _stab()
        self._frames(st, 10, lambda i: [_body(_joints(), bid=1, dist=1.2)], 100.0)
        snaps = self._frames(st, 60, lambda i: [
            _body(_joints(), bid=1, dist=1.2),
            _body(_joints(None, _up("right", 0.30)), bid=2, dist=1.3)],
            100.0 + 10 * T)
        self.assertEqual({s["owner_id"] for s in snaps}, {1})

    def test_nobody_claims_from_an_owner_using_a_hand(self):
        st = _stab()

        def own():
            return _body(_joints(None, _up("right", 0.25)), bid=1, dist=1.2)
        self._frames(st, 15, lambda i: [own()], 100.0)
        snaps = self._frames(st, 60, lambda i: [
            own(), _body(_joints(_up("left", 0.30), None), bid=2, dist=1.0)],
            100.0 + 15 * T)
        self.assertEqual({s["owner_id"] for s in snaps}, {1})

    def test_a_brief_raise_does_not_claim(self):
        st = _stab()
        self._frames(st, 10, lambda i: [_body(_joints(), bid=2, dist=1.4)], 100.0)
        n = int(P["KINECT_OWNER_CLAIM_SEC"] / T) - 2
        snaps = self._frames(st, 30, lambda i: [
            _body(_joints(), bid=2, dist=1.4),
            _body(_joints(None, _up("right", 0.25) if i < n else None), bid=1)],
            100.0 + 10 * T)
        self.assertEqual({s["owner_id"] for s in snaps}, {2})


class OwnerTakeoverGuardTests(unittest.TestCase):
    def test_a_slightly_nearer_body_never_takes_over(self):
        """Mutant 'owner nearer margin -> 0' survived: a body only 0.1 m nearer
        (hands down, no claim) must never take the owner, however long."""
        st = _stab()
        for i in range(10):
            st.process([_body(_joints(), bid=1, dist=1.2)], 100.0 + i * T)
        snaps = [st.process([_body(_joints(), bid=1, dist=1.2),
                             _body(_joints(), bid=2, dist=1.1)],
                            100.0 + (10 + i) * T) for i in range(90)]
        self.assertEqual({s["owner_id"] for s in snaps}, {1})

    def test_an_intermittently_nearer_body_does_not_accumulate(self):
        """Mutant 'owner challenger reset when not nearer' survived: nearer for
        0.8 s, then not, then nearer again - the 1 s dwell restarts."""
        st = _stab()
        for i in range(10):
            st.process([_body(_joints(), bid=1, dist=1.2)], 100.0 + i * T)
        snaps, t = [], 100.0 + 10 * T
        for _ in range(3):
            for near in (True,) * 24 + (False,) * 6:
                snaps.append(st.process(
                    [_body(_joints(), bid=1, dist=1.2),
                     _body(_joints(), bid=2, dist=0.8 if near else 1.5)], t))
                t += T
        self.assertEqual({s["owner_id"] for s in snaps}, {1})


class BridgeFailClosedTests(unittest.TestCase):
    """LOW (review 1): a stabiliser that can't be built was retried on EVERY
    frame, silently, and the docstring promised a fallback the air-mouse never
    made. Now: fail closed, say so once, retry at most every 30 s."""

    def setUp(self):
        kb.reset_tracking()
        self.addCleanup(kb.reset_tracking)
        self.addCleanup(_clear_body_cache)

    def test_an_unbuildable_stabiliser_fails_closed_loudly_and_backs_off(self):
        import contextlib
        import io
        from unittest import mock
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise ImportError("simulated")
        out = io.StringIO()
        b = [_body(_joints(None, _up("right", 0.25)))]
        with mock.patch.object(kb, "_stabilizer_module", boom), \
                contextlib.redirect_stdout(out):
            for i in range(60):
                kb.publish_body_frame(b, now=200.0 + i * T)
        self.assertEqual(calls["n"], 1)
        self.assertIsNone(kb.get_tracked_frame(now=202.0))
        self.assertIn("simulated", kb.tracking_error() or "")
        self.assertEqual(out.getvalue().count("hand stabiliser UNAVAILABLE"), 1)
        # The body pipe itself kept working.
        self.assertEqual(kb._body_cache[0], b)
        # After the back-off it retries, and recovers once the module loads.
        with contextlib.redirect_stdout(io.StringIO()):
            kb.publish_body_frame(b, now=200.0 + kb._TRACKER_RETRY_SEC + 1.0)
        self.assertIsNotNone(kb.get_tracked_frame(
            now=200.0 + kb._TRACKER_RETRY_SEC + 1.0))
        self.assertIsNone(kb.tracking_error())


def _clear_body_cache():
    with kb._body_cache_lock:
        kb._body_cache[0] = None
        kb._body_cache_at[0] = 0.0


if __name__ == "__main__":
    unittest.main()
