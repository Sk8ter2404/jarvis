"""Unit tests for audio.kinect_gestures.GestureRecognizer — the pure,
hardware-free gesture recognizer.

No sensor, no pykinect2, no threads. We fabricate get_bodies()-shaped frames
(joint dicts with (x, y, z, tracking_state) tuples) and feed them through the
recognizer on a DETERMINISTIC injected clock, asserting:

  * wave / raise-hand / swipe each fire on a matching synthetic sequence,
  * non-matching sequences do NOT fire,
  * the cooldown debounce blocks an identical second sequence inside the window,
    and the gesture re-fires once the cooldown lapses,
  * malformed / empty / untracked frames degrade to None (never raise).

stdlib unittest only.
"""
from __future__ import annotations

import unittest

from audio import kinect_gestures as kg


# ─── fake clock + frame builders ──────────────────────────────────────────
class _Clock:
    """A monotonic-like clock the test advances by hand."""
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


TRACKED = 2   # PyKinect TrackingState: fully tracked


def _body(hand_right=None, elbow_right=None, head=None, spine_base=None,
          spine_shoulder=None, hand_left=None, elbow_left=None,
          distance_m=2.0, body_id=0):
    """Build one get_bodies() entry. Joints passed as (x, y, z) get the TRACKED
    state appended; pass a full 4-tuple to set an explicit state. Only the
    joints provided are present (the recognizer must tolerate the rest missing)."""
    def _j(v):
        if v is None:
            return None
        return tuple(v) if len(v) == 4 else (v[0], v[1], v[2], TRACKED)

    joints = {}
    for name, val in (
        ("hand_right", hand_right), ("elbow_right", elbow_right),
        ("head", head), ("spine_base", spine_base),
        ("spine_shoulder", spine_shoulder),
        ("hand_left", hand_left), ("elbow_left", elbow_left),
    ):
        jj = _j(val)
        if jj is not None:
            joints[name] = jj
    return {"id": body_id, "joints": joints,
            "head": (head[0], head[1], head[2]) if head else None,
            "distance_m": distance_m, "facing": None}


def _frame(body):
    """get_bodies() returns a LIST of bodies."""
    return [body] if body is not None else []


# ─── WAVE ──────────────────────────────────────────────────────────────────
class WaveTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)

    def _wave_sequence(self, reversals=3, amp=0.15, dt=0.08):
        """Yield frames of a hand (above its elbow, below head) oscillating in x.
        `reversals` swing direction changes; amplitude `amp` per swing exceeds
        WAVE_MIN_AMPLITUDE_M. Hand y is well above elbow y."""
        head = (0.0, 0.6, 2.0)
        elbow = (0.0, 0.0, 2.0)
        # Build an x path that reverses `reversals` times: e.g. 0,+a,-a,+a,...
        xs = [0.0]
        sign = 1.0
        for _ in range(reversals + 1):
            xs.append(round(xs[-1] + sign * amp, 4))
            sign *= -1.0
        fired = None
        for x in xs:
            hand = (x, 0.3, 2.0)   # y=0.3 > elbow 0.0, < head 0.6
            body = _body(hand_right=hand, elbow_right=elbow, head=head,
                         spine_base=(0.0, -0.4, 2.0))
            got = self.rec.update(_frame(body))
            fired = fired or got
            self.clk.advance(dt)
        return fired

    def test_wave_fires(self):
        self.assertEqual(self._wave_sequence(reversals=3), kg.WAVE)

    def test_single_swing_does_not_fire(self):
        # One direction change is below WAVE_MIN_REVERSALS (2).
        self.assertIsNone(self._wave_sequence(reversals=1))

    def test_hand_below_elbow_does_not_wave(self):
        # Same oscillation but the hand is BELOW the elbow → not a wave pose.
        head = (0.0, 0.6, 2.0)
        elbow = (0.0, 0.5, 2.0)        # elbow high
        fired = None
        sign = 1.0
        x = 0.0
        for _ in range(6):
            x = round(x + sign * 0.15, 4)
            sign *= -1.0
            hand = (x, 0.2, 2.0)       # hand y 0.2 < elbow 0.5
            body = _body(hand_right=hand, elbow_right=elbow, head=head)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(0.08)
        self.assertIsNone(fired)

    def test_micro_jitter_does_not_wave(self):
        # Tiny x wobble below WAVE_MIN_AMPLITUDE_M must not count as swings.
        head = (0.0, 0.6, 2.0)
        elbow = (0.0, 0.0, 2.0)
        fired = None
        sign = 1.0
        x = 0.0
        for _ in range(8):
            x = round(x + sign * 0.01, 4)   # 1 cm, below the 6 cm gate
            sign *= -1.0
            body = _body(hand_right=(x, 0.3, 2.0), elbow_right=elbow, head=head)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(0.08)
        self.assertIsNone(fired)


# ─── RAISE_HAND ─────────────────────────────────────────────────────────────
class RaiseHandTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)

    def _hold_hand_above_head(self, total_seconds, dt=0.1, hand_y=0.9):
        head = (0.0, 0.6, 2.0)
        fired = None
        elapsed = 0.0
        while elapsed <= total_seconds + 1e-9:
            body = _body(hand_right=(0.2, hand_y, 2.0),
                         elbow_right=(0.2, 0.2, 2.0), head=head)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(dt)
            elapsed += dt
        return fired

    def test_raise_hand_fires_after_dwell(self):
        # Hold for longer than RAISE_HAND_DWELL_SECONDS.
        self.assertEqual(
            self._hold_hand_above_head(kg.RAISE_HAND_DWELL_SECONDS + 0.4),
            kg.RAISE_HAND)

    def test_brief_raise_does_not_fire(self):
        # Up for clearly less than the dwell → no fire.
        self.assertIsNone(
            self._hold_hand_above_head(kg.RAISE_HAND_DWELL_SECONDS / 2.0))

    def test_hand_below_head_does_not_fire(self):
        # Hand sustained but BELOW head height.
        self.assertIsNone(
            self._hold_hand_above_head(kg.RAISE_HAND_DWELL_SECONDS + 0.5,
                                       hand_y=0.4))


# ─── SWIPE ──────────────────────────────────────────────────────────────────
class SwipeTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)

    def _swipe(self, x_start, x_end, steps=4, dt=0.08, y=0.3,
               spine_base=(0.0, -0.4, 2.0)):
        """Move the hand from x_start to x_end over a short window with minimal
        vertical change. spine_base defaults to a tracked waist well below the
        hand; pass None (absent) or a 4-tuple with a sub-TRACKED state to
        exercise the untracked-waist path."""
        fired = None
        for i in range(steps + 1):
            x = x_start + (x_end - x_start) * (i / steps)
            body = _body(hand_right=(round(x, 4), y, 2.0),
                         elbow_right=(0.0, 0.1, 2.0),
                         head=(0.0, 0.6, 2.0), spine_base=spine_base)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(dt)
        return fired

    def test_swipe_right_fires(self):
        # +x travel > SWIPE_MIN_DX_M → swipe right (sensor frame).
        self.assertEqual(self._swipe(-0.3, +0.3), kg.SWIPE_RIGHT)

    def test_swipe_left_fires(self):
        self.assertEqual(self._swipe(+0.3, -0.3), kg.SWIPE_LEFT)

    def test_short_lateral_move_does_not_swipe(self):
        # Travel below SWIPE_MIN_DX_M.
        self.assertIsNone(self._swipe(-0.1, +0.1))

    def test_swipe_requires_tracked_spine_base(self):
        # REGRESSION (2026-07-21 audit): with spine_base untracked the waist gate
        # failed OPEN — the `is None or` predicate admitted EVERY sample, so a
        # flat desk-level reach (the desk occludes the lower torso, spine_base
        # reports Inferred) fired SWIPE and cancelled speech + a pending
        # confirmation. The gate must fail CLOSED like raise-hand/wave: the exact
        # sweep test_swipe_right_fires accepts must NOT fire when the waist
        # reference is absent on every sample.
        self.assertIsNone(self._swipe(-0.3, +0.3, spine_base=None))

    def test_swipe_inferred_spine_base_fails_closed(self):
        # Same fail-closed contract for the SDK's actual seated-desk report:
        # spine_base present but Inferred (state 1 < MIN_TRACKING_STATE) →
        # _sample_hand records spine_base_y None → no fire.
        self.assertIsNone(
            self._swipe(-0.3, +0.3, spine_base=(0.0, -0.4, 2.0, 1)))

    def test_swipe_below_tracked_waist_does_not_fire(self):
        # Locks the fail-safe DIRECTION: a tracked waist with the hand sweeping
        # BELOW it (hand_y -0.6 < spine_base_y -0.4) is filtered by the height
        # gate — the lateral travel alone must not fire.
        self.assertIsNone(self._swipe(-0.3, +0.3, y=-0.6))

    def test_diagonal_move_does_not_swipe(self):
        # Big horizontal AND big vertical change → fails the flatness gate
        # (and shouldn't be a wave: monotonic x, no reversals).
        spine_base = (0.0, -0.4, 2.0)
        fired = None
        steps = 4
        for i in range(steps + 1):
            x = -0.3 + 0.6 * (i / steps)
            y = 0.1 + 0.6 * (i / steps)   # 0.6 m vertical >> SWIPE_MAX_DY_M
            body = _body(hand_right=(round(x, 4), round(y, 4), 2.0),
                         elbow_right=(0.0, 0.0, 2.0),
                         head=(0.0, 1.0, 2.0), spine_base=spine_base)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(0.08)
        self.assertIsNone(fired)


# ─── debounce / cooldown ────────────────────────────────────────────────────
class CooldownTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)

    def _do_swipe(self):
        """Fire one swipe-right; return whether it fired."""
        spine_base = (0.0, -0.4, 2.0)
        fired = None
        for i in range(5):
            x = -0.3 + 0.6 * (i / 4)
            body = _body(hand_right=(round(x, 4), 0.3, 2.0),
                         elbow_right=(0.0, 0.1, 2.0),
                         head=(0.0, 0.6, 2.0), spine_base=spine_base)
            fired = fired or self.rec.update(_frame(body))
            self.clk.advance(0.08)
        return fired

    def test_identical_sequence_within_cooldown_does_not_refire(self):
        self.assertEqual(self._do_swipe(), kg.SWIPE_RIGHT)
        # Immediately repeat — still inside COOLDOWN_SECONDS → no second fire.
        self.assertIsNone(self._do_swipe())

    def test_refires_after_cooldown_lapses(self):
        self.assertEqual(self._do_swipe(), kg.SWIPE_RIGHT)
        # Jump the clock past the cooldown, then repeat → fires again.
        self.clk.advance(kg.COOLDOWN_SECONDS + 0.5)
        self.assertEqual(self._do_swipe(), kg.SWIPE_RIGHT)

    def test_in_cooldown_predicate(self):
        self.assertFalse(self.rec.in_cooldown())
        self._do_swipe()
        self.assertTrue(self.rec.in_cooldown())
        self.clk.advance(kg.COOLDOWN_SECONDS + 0.1)
        self.assertFalse(self.rec.in_cooldown())


# ─── robustness: malformed / empty frames never raise ──────────────────────
class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)

    def test_empty_list_returns_none(self):
        self.assertIsNone(self.rec.update([]))

    def test_none_returns_none(self):
        self.assertIsNone(self.rec.update(None))

    def test_body_without_joints_returns_none(self):
        self.assertIsNone(self.rec.update([{"id": 0, "joints": {}}]))

    def test_untracked_joint_ignored(self):
        # Hand present but TrackingState 0 (not tracked) → no usable sample.
        body = _body(hand_right=(0.0, 0.3, 2.0, 0),
                     elbow_right=(0.0, 0.0, 2.0, 0), head=(0.0, 0.6, 2.0))
        self.assertIsNone(self.rec.update(_frame(body)))

    def test_garbage_shapes_do_not_raise(self):
        for junk in (42, "nope", [{"joints": None}], [{"no_joints": 1}],
                     [{"joints": {"hand_right": (0.0,)}}]):
            # Should return None, never raise.
            self.assertIsNone(self.rec.update(junk))

    def test_nearest_body_selected(self):
        # Two bodies; only the NEAR one is waving. The recognizer tracks the
        # nearest, so it should still detect the wave.
        head = (0.0, 0.6, 2.0)
        elbow = (0.0, 0.0, 2.0)
        sign = 1.0
        x = 0.0
        fired = None
        for _ in range(6):
            x = round(x + sign * 0.15, 4)
            sign *= -1.0
            near = _body(hand_right=(x, 0.3, 1.5), elbow_right=(0.0, 0.0, 1.5),
                         head=(0.0, 0.6, 1.5), distance_m=1.5, body_id=0)
            far = _body(hand_right=(0.0, -0.5, 3.5), elbow_right=elbow,
                        head=head, distance_m=3.5, body_id=1)
            fired = fired or self.rec.update([far, near])
            self.clk.advance(0.08)
        self.assertEqual(fired, kg.WAVE)


class NearestBodyFacingTests(unittest.TestCase):
    """2026-07-15 ghost fix: _nearest_body prefers a FACING body (the owner at the
    desk) over a nearer non-facing one (a reflection), so a reflection can't starve
    the owner or drive a gesture off itself."""

    def test_facing_owner_beats_nearer_reflection(self):
        reflection = _body(hand_right=(0.0, 0.3, 2.0), distance_m=1.2, body_id=1)
        reflection["facing"] = False
        owner = _body(hand_right=(0.0, 0.3, 2.0), distance_m=2.5, body_id=2)
        owner["facing"] = True
        self.assertEqual(kg._nearest_body([reflection, owner])["id"], 2)

    def test_falls_back_to_distance_when_facing_unknown(self):
        near = _body(hand_right=(0.0, 0.3, 2.0), distance_m=1.2, body_id=1)
        far = _body(hand_right=(0.0, 0.3, 2.0), distance_m=2.5, body_id=2)
        # facing None on both (the default) → pure distance decides → nearer wins.
        self.assertEqual(kg._nearest_body([near, far])["id"], 1)


if __name__ == "__main__":
    unittest.main()
