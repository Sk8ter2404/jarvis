"""tools/handtrack_gestures.py: hand states and gesture events from SYNTHETIC
landmarks (no mediapipe, no camera). Each gesture fires once, with
hysteresis; a wave is not a palm hold; a clap and an explode are different
things."""
import math
import os
import random
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import handtrack_gestures as hg  # noqa: E402

W, H = 640, 480
FPS = 30.0
DT = 1.0 / FPS

# ── a synthetic hand ───────────────────────────────────────────────────
# Canonical RIGHT hand, palm facing the (mirrored) camera, fingers up, in
# hand-size units (wrist at 0,0; middle MCP 1.0 above it). Screen y is down.
_MCP = {5: (-0.30, -0.95), 9: (0.0, -1.0), 13: (0.28, -0.95), 17: (0.52, -0.82)}
_FINGER_DEG = {5: -10.0, 9: 0.0, 13: 8.0, 17: 20.0}
_LEN = {5: (0.50, 0.30, 0.25), 9: (0.55, 0.33, 0.27),
        13: (0.50, 0.30, 0.25), 17: (0.40, 0.22, 0.20)}


def _dir(deg):
    r = math.radians(deg)
    return (math.sin(r), -math.cos(r))


def make_hand(kind="open", side="Right", cx=320.0, cy=240.0, hs=100.0,
              angle=0.0, spread=1.0, back=False, thumb_gap=None, score=0.95):
    """Landmarks (normalised) of a synthetic hand. kind: open | fist | point |
    pinch. ``angle``: clockwise degrees (0 = fingers up). ``back``: show the
    back of the hand. ``thumb_gap``: thumb tip to index tip distance in hs."""
    pts = [(0.0, 0.0)] * 21
    pts = list(pts)
    curled = {"open": (), "pinch": (), "fist": (5, 9, 13, 17),
              "point": (9, 13, 17)}[kind]
    for mcp, (mx, my) in _MCP.items():
        deg = _FINGER_DEG[mcp] * spread
        d = _dir(deg)
        l1, l2, l3 = _LEN[mcp]
        pts[mcp] = (mx, my)
        if mcp in curled:
            pip = (mx + 0.6 * l1 * d[0], my + 0.6 * l1 * d[1])
            dip = (pip[0] + 0.2 * d[0], pip[1] + 0.2 * d[1])
            tip = (mx - 0.15 * d[0], my - 0.15 * d[1])
        else:
            pip = (mx + l1 * d[0], my + l1 * d[1])
            dip = (pip[0] + l2 * d[0], pip[1] + l2 * d[1])
            tip = (dip[0] + l3 * d[0], dip[1] + l3 * d[1])
        pts[mcp + 1], pts[mcp + 2], pts[mcp + 3] = pip, dip, tip
    pts[0] = (0.0, 0.0)
    pts[1], pts[2], pts[3] = (-0.25, -0.20), (-0.50, -0.45), (-0.70, -0.65)
    if kind == "fist":
        pts[4] = (-0.05, -0.60)
    else:
        pts[4] = (-0.88, -0.82)
    if kind == "pinch" or thumb_gap is not None:
        gap = 0.05 if thumb_gap is None else thumb_gap
        tip = pts[8]
        pts[4] = (tip[0] + gap, tip[1])
        pts[3] = (pts[4][0] - 0.1, pts[4][1] + 0.2)
        pts[2] = (pts[3][0] - 0.1, pts[3][1] + 0.2)
    if side == "Left":
        pts = [(-x, y) for x, y in pts]
    if back:
        pts = [(-x, y) for x, y in pts]
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rot = [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]
    mx = sum(rot[i][0] for i in hg.PALM_POINTS) / 5.0
    my = sum(rot[i][1] for i in hg.PALM_POINTS) / 5.0
    lm = [[((x - mx) * hs + cx) / W, ((y - my) * hs + cy) / H, 0.0]
          for x, y in rot]
    return {"side": side, "score": score, "lm": lm}


def run(engine, seq):
    """seq: iterable of (t, [hands]) -> (all events, last infos)."""
    events, infos = [], []
    for t, hands in seq:
        infos, ev = engine.update(t, hands, W, H)
        events.extend(ev)
    return events, infos


def names(events):
    return [e["name"] for e in events]


def frames(seconds, fn, t0=0.0):
    n = int(round(seconds * FPS))
    return [(t0 + i * DT, fn(i, t0 + i * DT)) for i in range(n)]


class StateTests(unittest.TestCase):
    def one(self, hand):
        eng = hg.GestureEngine()
        _, infos = run(eng, [(0.0, [hand]), (0.05, [hand]), (0.1, [hand]),
                             (0.15, [hand])])
        return infos[0]

    def test_open_palm_facing_right_hand(self):
        i = self.one(make_hand("open"))
        self.assertEqual(i["state"], "open")
        self.assertTrue(i["palm_facing"])
        self.assertEqual(i["side"], "Right")

    def test_left_hand_facing(self):
        i = self.one(make_hand("open", side="Left"))
        self.assertEqual(i["state"], "open")
        self.assertTrue(i["palm_facing"])

    def test_back_of_hand_is_not_facing(self):
        self.assertFalse(self.one(make_hand("open", back=True))["palm_facing"])
        self.assertFalse(self.one(make_hand("open", side="Left", back=True))["palm_facing"])

    def test_rotation_does_not_change_facing(self):
        for ang in (-60, -20, 25, 70, 120):
            self.assertTrue(self.one(make_hand("open", angle=ang))["palm_facing"], ang)

    def test_fist_point_pinch(self):
        self.assertEqual(self.one(make_hand("fist"))["state"], "fist")
        self.assertEqual(self.one(make_hand("point"))["state"], "point")
        i = self.one(make_hand("pinch"))
        self.assertEqual(i["state"], "pinch")
        self.assertGreaterEqual(i["pinch"], 0.9)

    def test_pinch_strength_scales(self):
        self.assertLess(self.one(make_hand("open"))["pinch"], 0.1)
        mid = self.one(make_hand("open", thumb_gap=0.4))
        self.assertTrue(0.2 < mid["pinch"] < 0.7, mid["pinch"])

    def test_palm_position_is_normalised_and_mirrored_input_kept(self):
        i = self.one(make_hand("open", cx=480.0, cy=120.0))
        self.assertAlmostEqual(i["palm"][0], 0.75, places=2)
        self.assertAlmostEqual(i["palm"][1], 0.25, places=2)

    def test_invert_palm_flag(self):
        eng = hg.GestureEngine(invert_palm=True)
        _, infos = run(eng, [(0.0, [make_hand("open")])])
        self.assertFalse(infos[0]["palm_facing"])


class PalmHoldTests(unittest.TestCase):
    def still(self, seconds, **kw):
        rnd = random.Random(3)
        return frames(seconds, lambda i, t: [make_hand(
            "open", cx=320 + rnd.uniform(-1.5, 1.5), cy=240 + rnd.uniform(-1.5, 1.5),
            angle=rnd.uniform(-1.5, 1.5), **kw)])

    def test_fires_once_after_the_hold(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, self.still(1.3))
        self.assertEqual(names(ev), [])
        eng = hg.GestureEngine()
        ev, _ = run(eng, self.still(5.0))
        self.assertEqual(names(ev), ["palm_hold"])          # once, not every frame
        self.assertEqual(ev[0]["side"], "Right")
        self.assertAlmostEqual(ev[0]["pos"][0], 0.5, delta=0.03)
        self.assertGreaterEqual(ev[0]["value"], hg.PALM_HOLD_S)

    def test_rearms_after_the_hand_leaves(self):
        eng = hg.GestureEngine()
        seq = self.still(2.0)
        seq += [(2.0 + i * DT, []) for i in range(int(1.0 * FPS))]      # hand away
        seq += [(3.0 + t, h) for t, h in self.still(2.0)]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["palm_hold", "palm_hold"])

    def test_no_palm_hold_during_a_wave(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, frames(6.0, lambda i, t: [make_hand(
            "open", cx=320 + 25 * math.sin(2 * math.pi * 2.0 * t), cy=240,
            angle=30 * math.sin(2 * math.pi * 2.0 * t))]))
        self.assertEqual([e for e in ev if e["name"] == "palm_hold"], [])

    def test_no_palm_hold_for_a_slow_drift(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, frames(6.0, lambda i, t: [make_hand(
            "open", cx=100 + 130 * t, cy=240)]))
        self.assertEqual([e for e in ev if e["name"] == "palm_hold"], [])

    def test_needs_the_palm_toward_the_camera(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, self.still(4.0, back=True))
        self.assertEqual(names(ev), [])

    def test_needs_spread_fingers(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, self.still(4.0, spread=0.3))
        self.assertEqual(names(ev), [])

    def test_a_fist_is_not_a_palm_hold(self):
        eng = hg.GestureEngine()
        ev, _ = run(eng, frames(4.0, lambda i, t: [make_hand("fist")]))
        self.assertEqual([e for e in ev if e["name"] == "palm_hold"], [])

    def test_one_dropped_frame_does_not_restart(self):
        eng = hg.GestureEngine()
        seq = self.still(2.0)
        del seq[30:32]                                        # two missing frames
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["palm_hold"])


class PinchFistTests(unittest.TestCase):
    def test_pinch_start_hold_end(self):
        eng = hg.GestureEngine()
        seq = frames(0.5, lambda i, t: [make_hand("open")])
        seq += [(0.5 + t, h) for t, h in frames(1.5, lambda i, t: [make_hand("pinch")])]
        seq += [(2.0 + t, h) for t, h in frames(0.5, lambda i, t: [make_hand("open")])]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["pinch_start", "pinch_hold", "pinch_end"])
        self.assertAlmostEqual(ev[1]["value"], hg.PINCH_HOLD_S, delta=0.2)

    def test_short_pinch_has_no_hold(self):
        eng = hg.GestureEngine()
        seq = frames(0.3, lambda i, t: [make_hand("open")])
        seq += [(0.3 + t, h) for t, h in frames(0.35, lambda i, t: [make_hand("pinch")])]
        seq += [(0.65 + t, h) for t, h in frames(0.4, lambda i, t: [make_hand("open")])]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["pinch_start", "pinch_end"])

    def test_pinch_hysteresis_holds_through_a_wobble(self):
        eng = hg.GestureEngine()
        gaps = [0.05] * 10 + [0.30, 0.05, 0.28, 0.05] * 3 + [0.05] * 10
        seq = [(i * DT, [make_hand("open", thumb_gap=g)]) for i, g in enumerate(gaps)]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev).count("pinch_start"), 1)
        self.assertEqual(names(ev).count("pinch_end"), 0)

    def test_fist_grab_and_release(self):
        eng = hg.GestureEngine()
        seq = frames(0.5, lambda i, t: [make_hand("open")])
        seq += [(0.5 + t, h) for t, h in frames(1.0, lambda i, t: [make_hand("fist")])]
        seq += [(1.5 + t, h) for t, h in frames(0.5, lambda i, t: [make_hand("open")])]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["fist_grab", "fist_release"])

    def test_fist_release_when_the_hand_vanishes(self):
        eng = hg.GestureEngine()
        seq = frames(1.0, lambda i, t: [make_hand("fist")])
        seq += [(1.0 + i * DT, []) for i in range(15)]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["fist_grab", "fist_release"])

    def test_one_frame_flicker_is_not_a_grab(self):
        eng = hg.GestureEngine()
        seq = [(i * DT, [make_hand("fist" if i == 10 else "open")]) for i in range(30)]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), [])


class FlickTests(unittest.TestCase):
    def sweep(self, total_deg, seconds, start=-20.0):
        eng = hg.GestureEngine()
        n = max(2, int(round(seconds * FPS)))
        seq = frames(0.5, lambda i, t: [make_hand("point", angle=start)])
        t0 = 0.5
        for i in range(n):
            a = start + total_deg * (i + 1) / n
            seq.append((t0 + i * DT, [make_hand("point", angle=a)]))
        t1 = t0 + n * DT
        seq += [(t1 + i * DT, [make_hand("point", angle=start + total_deg)])
                for i in range(int(0.5 * FPS))]
        return run(eng, seq)[0]

    def test_fast_flick_right(self):
        ev = self.sweep(50.0, 0.15)
        self.assertEqual(names(ev), ["point_flick"])
        self.assertGreater(ev[0]["value"], hg.FLICK_OMEGA_DEG_S)

    def test_fast_flick_left_is_negative(self):
        ev = self.sweep(-50.0, 0.15)
        self.assertEqual(names(ev), ["point_flick"])
        self.assertLess(ev[0]["value"], -hg.FLICK_OMEGA_DEG_S)

    def test_slow_sweep_is_not_a_flick(self):
        self.assertEqual(self.sweep(50.0, 1.5), [])

    def test_small_twitch_is_not_a_flick(self):
        self.assertEqual(self.sweep(12.0, 0.1), [])

    def test_open_hand_swing_is_not_a_flick(self):
        eng = hg.GestureEngine()
        seq = frames(0.5, lambda i, t: [make_hand("open", angle=-20)])
        seq += [(0.5 + i * DT, [make_hand("open", angle=-20 + 10 * i)]) for i in range(5)]
        ev, _ = run(eng, seq)
        self.assertEqual([e for e in ev if e["name"] == "point_flick"], [])

    def test_translation_alone_is_not_a_flick(self):
        eng = hg.GestureEngine()
        seq = frames(0.5, lambda i, t: [make_hand("point", cx=200)])
        seq += [(0.5 + i * DT, [make_hand("point", cx=200 + 60 * (i + 1))]) for i in range(5)]
        ev, _ = run(eng, seq)
        self.assertEqual([e for e in ev if e["name"] == "point_flick"], [])

    def test_cooldown(self):
        eng = hg.GestureEngine()
        seq = frames(0.4, lambda i, t: [make_hand("point", angle=-25)])
        t = 0.4
        for k in range(2):
            for i in range(5):
                seq.append((t, [make_hand("point", angle=-25 + 10 * (i + 1))]))
                t += DT
            for i in range(5):
                seq.append((t, [make_hand("point", angle=25 - 10 * (i + 1))]))
                t += DT
        ev, _ = run(eng, seq)
        self.assertLessEqual(names(ev).count("point_flick"), 2)
        self.assertGreaterEqual(names(ev).count("point_flick"), 1)


class LowFrameRateFlickTests(unittest.TestCase):
    """Live 2026-10-14: the owner's webcam gave ~12 fps in room light, with
    ~0.1 s gaps while JARVIS's face pass runs. A real flick must still fire
    there; jitter and slow swings must not."""

    def seq_at(self, fps, angles, settle_s=0.6, start=-20.0, gap_after=None, gap_s=0.0):
        dt = 1.0 / fps
        seq, t = [], 0.0
        for _ in range(int(settle_s * fps)):
            seq.append((t, [make_hand("point", angle=start)]))
            t += dt
        for i, a in enumerate(angles):
            seq.append((t, [make_hand("point", angle=a)]))
            t += dt + (gap_s if gap_after is not None and i == gap_after else 0.0)
        for _ in range(int(0.5 * fps)):
            seq.append((t, [make_hand("point", angle=angles[-1])]))
            t += dt
        return run(hg.GestureEngine(), seq)[0]

    def test_a_flick_between_two_frames_at_10_fps_fires_once(self):
        # 0.1 s frames: a 0.18 s window holds 2 samples, so "3 samples" never fired
        ev = self.seq_at(10, [40.0])                        # 60 deg in one frame
        self.assertEqual(names(ev), ["point_flick"])
        self.assertGreater(ev[0]["value"], hg.FLICK_OMEGA_DEG_S)

    def test_a_flick_across_a_face_pass_gap_fires(self):
        # 15 fps bursts with a 0.15 s hole where JARVIS's face pass runs
        ev = self.seq_at(15, [0.0, 60.0], gap_after=0, gap_s=0.15)
        self.assertEqual(names(ev), ["point_flick"])

    def test_jitter_at_12_fps_is_not_a_flick(self):
        wobble = [-20 + (8 if i % 2 else -8) for i in range(12)]
        self.assertEqual(self.seq_at(12, wobble), [])

    def test_a_slow_swing_at_12_fps_is_not_a_flick(self):
        slow = [-20 + 6 * (i + 1) for i in range(12)]      # 72 deg over 1 s
        self.assertEqual(self.seq_at(12, slow), [])

    def test_30_fps_keeps_the_three_sample_rule(self):
        eng = hg.GestureEngine()
        seq = [(i / 30.0, [make_hand("point", angle=-20)]) for i in range(20)]
        ev, _ = run(eng, seq)
        tr = eng._tracks[0]
        self.assertLess(tr.dt_ema, hg.FLICK_SLOW_FRAME_S)


class TwoHandTests(unittest.TestCase):
    @staticmethod
    def pair(kind, d_hs, t_list=None, hs=100.0, k2=None):
        cxl = 320 - d_hs * hs / 2
        cxr = 320 + d_hs * hs / 2
        return [make_hand(kind, side="Left", cx=cxl, hs=hs),
                make_hand(k2 or kind, side="Right", cx=cxr, hs=hs)]

    def seq_d(self, kind, ds, t0=0.0, k2=None):
        return [(t0 + i * DT, self.pair(kind, d, k2=k2)) for i, d in enumerate(ds)]

    @staticmethod
    def ramp(a, b, n):
        return [a + (b - a) * (i + 1) / n for i in range(n)]

    def test_clap_fires_once_and_is_not_an_explode(self):
        eng = hg.GestureEngine()
        ds = [3.0] * 10 + self.ramp(3.0, 0.8, 6) + [0.8] + self.ramp(0.8, 3.0, 6) + [3.0] * 10
        ev, _ = run(eng, self.seq_d("open", ds))
        self.assertEqual(names(ev), ["clap"])
        self.assertEqual(ev[0]["side"], "both")
        self.assertGreater(ev[0]["value"], hg.CLAP_RATE_HS_S)

    def test_clap_when_a_hand_is_lost_on_contact(self):
        eng = hg.GestureEngine()
        ds = [3.0] * 10 + self.ramp(3.0, 1.5, 4)
        seq = self.seq_d("open", ds)
        t = seq[-1][0]
        seq += [(t + (i + 1) * DT, []) for i in range(10)]
        ev, _ = run(eng, seq)
        self.assertEqual(names(ev), ["clap"])

    def test_slow_hands_meeting_is_no_clap(self):
        eng = hg.GestureEngine()
        ds = [3.0] * 5 + self.ramp(3.0, 0.8, 40) + [0.8] * 10
        ev, _ = run(eng, self.seq_d("open", ds))
        self.assertEqual(names(ev).count("clap"), 0)
        self.assertEqual(names(ev).count("explode"), 0)

    def test_fists_meeting_is_no_clap(self):
        eng = hg.GestureEngine()
        ds = [3.0] * 10 + self.ramp(3.0, 0.8, 6) + [0.8] * 5
        ev, _ = run(eng, self.seq_d("fist", ds))
        self.assertEqual([e for e in ev if e["name"] == "clap"], [])

    def test_explode_two_fists_flying_apart(self):
        eng = hg.GestureEngine()
        ds = [1.5] * 12 + self.ramp(1.5, 5.0, 7) + [5.0] * 8
        ev, _ = run(eng, self.seq_d("fist", ds))
        self.assertEqual([e for e in ev if e["name"] != "fist_grab"], [
            e for e in ev if e["name"] == "explode"])
        self.assertEqual(names(ev).count("explode"), 1)
        self.assertEqual(names(ev).count("clap"), 0)

    def test_explode_palms_together_then_apart(self):
        eng = hg.GestureEngine()
        ds = [3.0] * 5 + self.ramp(3.0, 0.9, 40) + [0.9] * 20 + self.ramp(0.9, 4.5, 8) + [4.5] * 8
        ev, _ = run(eng, self.seq_d("open", ds))
        self.assertEqual(names(ev).count("explode"), 1)
        self.assertEqual(names(ev).count("clap"), 0)

    def test_slow_parting_is_no_explode(self):
        eng = hg.GestureEngine()
        ds = [0.9] * 20 + self.ramp(0.9, 4.5, 60)
        ev, _ = run(eng, self.seq_d("open", ds))
        self.assertEqual(names(ev).count("explode"), 0)

    def test_clap_rebound_is_not_an_explode(self):
        # hands bounce straight back after contact: no held-together phase
        eng = hg.GestureEngine()
        ds = [3.0] * 10 + self.ramp(3.0, 0.8, 5) + self.ramp(0.8, 4.0, 5) + [4.0] * 10
        ev, _ = run(eng, self.seq_d("open", ds))
        self.assertEqual(names(ev).count("explode"), 0)

    def test_hand_ids_are_stable_through_a_dropout(self):
        eng = hg.GestureEngine()
        seq = []
        for i in range(30):
            hands = self.pair("open", 3.0)
            if 10 <= i < 13:
                hands = hands[1:]                      # left hand missing 3 frames
            seq.append((i * DT, hands))
        ids = {}
        for t, hands in seq:
            infos, _ = eng.update(t, hands, W, H)
            for hd, inf in zip(hands, infos):
                ids.setdefault(hd["side"], set()).add(inf["id"])
        self.assertEqual(ids["Left"], {ids["Left"].copy().pop()})
        self.assertEqual(len(ids["Left"]), 1)
        self.assertEqual(len(ids["Right"]), 1)
        self.assertNotEqual(ids["Left"], ids["Right"])


class RecognitionTests(unittest.TestCase):
    """2026-10-20 (owner, in a classroom: "I don't want false triggers"): a
    small / far hand, an unsure detection, or a hand that has only just come
    into view never fires palm_hold, pinch_hold or point_flick."""

    def _palm(self, seconds, **kw):
        rnd = random.Random(5)
        return frames(seconds, lambda i, t: [make_hand(
            "open", cx=320 + rnd.uniform(-1.5, 1.5), cy=240 + rnd.uniform(-1.5, 1.5),
            **kw)])

    def test_a_small_far_hand_fires_nothing(self):
        # hs 30 px of a 480 px frame = 0.06: a classmate a few metres back
        ev, infos = run(hg.GestureEngine(), self._palm(4.0, hs=30.0))
        self.assertEqual([e for e in ev if e["name"] == "palm_hold"], [])
        self.assertFalse(infos[0]["ok"])
        seq = frames(2.0, lambda i, t: [make_hand("pinch", hs=30.0)])
        ev, _ = run(hg.GestureEngine(), seq)
        self.assertNotIn("pinch_hold", names(ev))
        self.assertIn("pinch_start", names(ev))         # states still reported

    def test_an_unsure_detection_fires_nothing(self):
        ev, _ = run(hg.GestureEngine(), self._palm(4.0, score=0.7))
        self.assertEqual(names(ev), [])

    def test_size_and_score_hysteresis(self):
        eng = hg.GestureEngine()
        seq = frames(1.0, lambda i, t: [make_hand("open", hs=100.0)])
        _, infos = run(eng, seq)
        self.assertTrue(infos[0]["ok"])
        # shrinking to 0.085 (between OFF and ON) keeps it ok ...
        _, infos = run(eng, [(1.0 + t, h) for t, h in frames(
            0.5, lambda i, t: [make_hand("open", hs=41.0, score=0.65)])])
        self.assertTrue(infos[0]["ok"])
        # ... below OFF it is not
        _, infos = run(eng, [(1.5 + t, h) for t, h in frames(
            0.6, lambda i, t: [make_hand("open", hs=30.0)])])
        self.assertFalse(infos[0]["ok"])
        self.assertAlmostEqual(infos[0]["size"], 30.0 / H, delta=0.01)

    def test_a_hand_that_just_arrived_does_not_flick(self):
        # pointing and flicking from the first frame (a hand swinging into view)
        seq = [(i * DT, [make_hand("point", angle=-20 + 10 * i)]) for i in range(6)]
        ev, _ = run(hg.GestureEngine(), seq)
        self.assertNotIn("point_flick", names(ev))

    def test_a_settled_hand_still_flicks_and_pinch_holds(self):
        self.assertEqual(names(FlickTests().sweep(50.0, 0.15)), ["point_flick"])
        seq = frames(1.0, lambda i, t: [make_hand("pinch")])
        ev, _ = run(hg.GestureEngine(), seq)
        self.assertEqual(names(ev), ["pinch_start", "pinch_hold"])


class PalmHoldAtWebcamRatesTests(unittest.TestCase):
    """TASK 4 (2026-10-20): the open-palm hand menu never fired in the owner's
    Dell test. With the steady frames (core/handtrack_frames.FramePump) the
    tracker runs at 12-20 fps; the hold-still thresholds must be reachable
    there with real tracking noise, on a 16:9 frame (640x360) and with the old
    clumpy delivery too. (They are: no threshold needed changing.)"""

    def _hold(self, fps, jitter_px=3.0, angle_jitter=4.0, h=480, hs=80.0,
              clumpy=False, seconds=4.0):
        global H
        rnd = random.Random(11)
        old_h, H = H, h                         # make_hand normalises by H
        try:
            eng = hg.GestureEngine()
            ev, t, i = [], 0.0, 0
            while t < seconds:
                hand = make_hand("open", cx=320 + rnd.gauss(0, jitter_px),
                                 cy=h / 2 + rnd.gauss(0, jitter_px),
                                 hs=hs * (1 + rnd.gauss(0, 0.03)),
                                 angle=rnd.gauss(0, angle_jitter))
                ev.extend(eng.update(t, [hand], W, h)[1])
                i += 1
                t += ((1 / 30) if i % 5 else 0.17) if clumpy else 1.0 / fps
        finally:
            H = old_h
        return [e for e in ev if e["name"] == "palm_hold"]

    def test_fires_at_12_15_18_20_fps(self):
        for fps in (12, 15, 18, 20):
            self.assertEqual(len(self._hold(fps)), 1, fps)

    def test_fires_on_a_16_9_frame(self):
        for fps in (15, 20):
            self.assertEqual(len(self._hold(fps, h=360, hs=70.0)), 1, fps)

    def test_fires_with_the_old_clumpy_delivery(self):
        self.assertEqual(len(self._hold(0, clumpy=True)), 1)

    def test_still_needs_about_one_and_a_half_seconds(self):
        self.assertEqual(self._hold(15, seconds=1.4), [])


if __name__ == "__main__":
    unittest.main()
