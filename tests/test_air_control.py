"""Tests for core/air_control — the PURE movie-style AIR CONTROL engine.

Drives AirControlEngine.update() directly with hand-rolled bodies frames in the
EXACT shape audio.kinect_bridge.get_bodies() emits (the bridge is the contract:
top-level "hand_right"/"hand_left" state strings, "joints" of
(x, y, z, tracking_state) tuples in camera-space metres — x sensor-right, y UP,
z depth AWAY from the sensor) — so NO sensor, NO mouse, NO threads. A fake
monotonic clock is injected (now_fn) so the debounce / click-vs-drag timing is
exercised deterministically, without a single real sleep.

Asserts the owner-facing contract:
  * ENGAGEMENT gate: a hand at the side / not reaching forward emits IDLE; a
    hand pushed ≥ AIR_ENGAGE_FORWARD_M past the shoulder AND above the waist
    ENGAGES and the cursor follows it; retracting past the hysteresis band
    (or losing the body) DISENGAGES — and a held drag is RELEASED (OP_UP), so
    a dropped hand can never strand a grabbed window.
  * MAPPING: the body-relative box maps NON-mirrored (hand right → cursor
    right), Y-INVERTED (hand up → cursor up), spans the FULL virtual desktop
    (incl. a negative multi-monitor origin), and CLAMPS at the edges.
  * FIST state machine: closed (after the AIR_GRAB_DEBOUNCE_MS debounce) →
    OP_DOWN grab; held closed → OP_MOVE drag; open after a long/travelled hold
    → OP_UP; a QUICK close→open with little travel → OP_CLICK at the grab spot.
  * LASSO scroll: first lasso frame arms (amount 0); vertical hand motion emits
    signed scroll clicks (hand up = positive); jitter inside
    AIR_SCROLL_DEADZONE_M emits 0.
  * EMA smoothing: a step change in hand position moves the cursor only
    AIR_SMOOTHING of the way per frame.
  * ROBUSTNESS: None / malformed frames, missing joints, untracked joints all
    degrade to IDLE — update() NEVER raises.

stdlib unittest only; CI-sim clean (no win32/pyautogui/Kinect).
"""
from __future__ import annotations

import unittest

from core.air_control import (
    AIR_BOX_CENTER_DROP_M, AIR_BOX_WIDTH_M, AIR_BOX_HEIGHT_M,
    AIR_CLICK_MS, AIR_ENGAGE_FORWARD_M, AIR_GRAB_DEBOUNCE_MS,
    AIR_SCROLL_DEADZONE_M, AIR_SCROLL_GAIN, AIR_SMOOTHING,
    AirControlEngine, OP_CLICK, OP_DOWN, OP_IDLE, OP_MOVE, OP_SCROLL, OP_UP,
)

# A comfortable single-monitor desktop for most cases; multi-monitor (negative
# origin) is exercised explicitly in test_negative_virtual_origin.
DESK = (0, 0, 1920, 1080)

# The reference skeleton (camera-space metres): body ~2 m from the sensor,
# spine_shoulder (the box anchor) at y=0.45, spine_mid (the waist) at y=0.0.
_SHOULDER_Z = 2.0
_ANCHOR = (0.0, 0.45, _SHOULDER_Z, 2)
# The hand position that lands EXACTLY at the interaction-box centre → screen
# centre: x = anchor x, y = anchor y − AIR_BOX_CENTER_DROP_M, z pushed forward
# well past the engage threshold.
_CENTER_HAND_Y = 0.45 - AIR_BOX_CENTER_DROP_M
_ENGAGED_Z = _SHOULDER_Z - (AIR_ENGAGE_FORWARD_M + 0.07)


def _body(hand=(0.0, _CENTER_HAND_Y, _ENGAGED_Z), state="open",
          side="right", distance=2.0):
    """One get_bodies() entry in the bridge's REAL shape (see module doc)."""
    joints = {
        "spine_mid": (0.0, 0.0, _SHOULDER_Z, 2),
        "spine_base": (0.0, -0.2, _SHOULDER_Z, 2),
        "spine_shoulder": _ANCHOR,
        f"shoulder_{side}": (0.2 if side == "right" else -0.2, 0.4,
                             _SHOULDER_Z, 2),
        f"hand_{side}": (hand[0], hand[1], hand[2], 2),
    }
    return {"id": 1, "joints": joints, "head": None, "distance_m": distance,
            "facing": True, "facing_yaw_deg": 0.0,
            "hand_right": state if side == "right" else "unknown",
            "hand_left": state if side == "left" else "unknown"}


class _Clock:
    """Deterministic monotonic clock for now_fn injection."""
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def tick(self, sec):
        self.t += sec


def _engine():
    clk = _Clock()
    return AirControlEngine(now_fn=clk), clk


def _settle_grab(eng, clk, state, hand=(0.0, _CENTER_HAND_Y, _ENGAGED_Z)):
    """Feed the same frame until the grab debounce accepts `state`; return the
    op emitted on the frame the flip landed."""
    eng.update([_body(hand=hand, state=state)], DESK)   # candidate frame
    clk.tick((AIR_GRAB_DEBOUNCE_MS / 1000.0) + 0.005)
    return eng.update([_body(hand=hand, state=state)], DESK)


class TestEngagement(unittest.TestCase):
    def test_no_bodies_is_idle(self):
        eng, _ = _engine()
        for frame in (None, [], [None], ["junk"], [{}], [{"joints": {}}]):
            op = eng.update(frame, DESK)
            self.assertEqual(op.kind, OP_IDLE)
            self.assertFalse(op.engaged)

    def test_hand_at_side_does_not_engage(self):
        eng, _ = _engine()
        # Hand hanging at the hip: NOT forward (z ≈ shoulder z), BELOW waist.
        op = eng.update([_body(hand=(0.0, -0.4, _SHOULDER_Z - 0.02))], DESK)
        self.assertEqual(op.kind, OP_IDLE)
        self.assertFalse(op.engaged)

    def test_forward_raised_hand_engages_and_moves(self):
        eng, _ = _engine()
        op = eng.update([_body()], DESK)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertTrue(op.engaged)
        # Box-centre hand → screen centre (first frame seeds the EMA directly).
        self.assertAlmostEqual(op.x, 960, delta=2)
        self.assertAlmostEqual(op.y, 540, delta=2)

    def test_retracting_disengages(self):
        eng, _ = _engine()
        self.assertEqual(eng.update([_body()], DESK).kind, OP_MOVE)
        # Pull the hand back to the shoulder plane → inside the disengage band.
        op = eng.update([_body(hand=(0.0, _CENTER_HAND_Y,
                                     _SHOULDER_Z - 0.02))], DESK)
        self.assertEqual(op.kind, OP_IDLE)
        self.assertFalse(eng.engaged)

    def test_untracked_hand_joint_is_idle(self):
        eng, _ = _engine()
        b = _body()
        j = b["joints"]["hand_right"]
        b["joints"]["hand_right"] = (j[0], j[1], j[2], 1)   # inferred, not tracked
        self.assertEqual(eng.update([b], DESK).kind, OP_IDLE)


class TestMapping(unittest.TestCase):
    def test_hand_right_moves_cursor_right_non_mirrored(self):
        eng, _ = _engine()
        centre = eng.update([_body()], DESK)
        eng2, _ = _engine()
        right = eng2.update(
            [_body(hand=(0.15, _CENTER_HAND_Y, _ENGAGED_Z))], DESK)
        self.assertGreater(right.x, centre.x)

    def test_hand_up_moves_cursor_up_inverted_y(self):
        eng, _ = _engine()
        centre = eng.update([_body()], DESK)
        eng2, _ = _engine()
        up = eng2.update(
            [_body(hand=(0.0, _CENTER_HAND_Y + 0.1, _ENGAGED_Z))], DESK)
        self.assertLess(up.y, centre.y)

    def test_clamped_at_desktop_edges(self):
        eng, _ = _engine()
        # Hand way past the right/top of the box → pinned inside the desktop.
        op = eng.update(
            [_body(hand=(AIR_BOX_WIDTH_M, _CENTER_HAND_Y + AIR_BOX_HEIGHT_M,
                         _ENGAGED_Z))], DESK)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertLessEqual(op.x, DESK[2] - 1)
        self.assertLessEqual(op.y, DESK[3] - 1)
        self.assertGreaterEqual(op.y, 0)

    def test_negative_virtual_origin(self):
        # Monitor arranged LEFT of the primary: origin (-1920, 0), 2 monitors.
        desk = (-1920, 0, 3840, 1080)
        eng, _ = _engine()
        # Hand hard LEFT → cursor near the LEFT monitor's left edge (negative x).
        op = eng.update(
            [_body(hand=(-AIR_BOX_WIDTH_M, _CENTER_HAND_Y, _ENGAGED_Z))], desk)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertLess(op.x, 0)
        self.assertGreaterEqual(op.x, -1920)

    def test_ema_smoothing_step_response(self):
        eng, _ = _engine()
        first = eng.update([_body()], DESK)          # seeds the EMA at centre
        # Step the hand to the box's right edge; one frame should close only
        # AIR_SMOOTHING of the gap to the raw target (the right desktop edge).
        target_hand = (AIR_BOX_WIDTH_M / 2.0, _CENTER_HAND_Y, _ENGAGED_Z)
        second = eng.update([_body(hand=target_hand)], DESK)
        raw_target = DESK[2] - 1                     # clamped right edge
        expected = first.x + AIR_SMOOTHING * (raw_target - first.x)
        self.assertAlmostEqual(second.x, expected, delta=2)
        self.assertLess(second.x, raw_target)        # not a snap


class TestGrabClickDrag(unittest.TestCase):
    def test_fist_grabs_after_debounce(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)                  # engage, open
        # First closed frame: still debouncing → move, no grab yet.
        op1 = eng.update([_body(state="closed")], DESK)
        self.assertEqual(op1.kind, OP_MOVE)
        self.assertFalse(eng.button_down)
        clk.tick((AIR_GRAB_DEBOUNCE_MS / 1000.0) + 0.005)
        op2 = eng.update([_body(state="closed")], DESK)
        self.assertEqual(op2.kind, OP_DOWN)
        self.assertTrue(eng.button_down)

    def test_hold_and_move_is_drag_then_release_up(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        _settle_grab(eng, clk, "closed")             # OP_DOWN happened
        # Hold the fist well past the click window, dragging the hand.
        clk.tick((AIR_CLICK_MS / 1000.0) + 0.2)
        drag = eng.update(
            [_body(hand=(0.2, _CENTER_HAND_Y, _ENGAGED_Z), state="closed")],
            DESK)
        self.assertEqual(drag.kind, OP_MOVE)
        self.assertIn("drag", drag.reason)
        # Open → release (a DRAG, not a click: the hold exceeded AIR_CLICK_MS).
        op = _settle_grab(eng, clk, "open",
                          hand=(0.2, _CENTER_HAND_Y, _ENGAGED_Z))
        self.assertEqual(op.kind, OP_UP)
        self.assertFalse(eng.button_down)

    def test_quick_close_open_is_click_at_grab_spot(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        down = _settle_grab(eng, clk, "closed")
        self.assertEqual(down.kind, OP_DOWN)
        gx, gy = down.x, down.y
        # Reopen quickly (debounce + a hair — well inside AIR_CLICK_MS) with
        # the hand held still (no travel).
        op = _settle_grab(eng, clk, "open")
        self.assertEqual(op.kind, OP_CLICK)
        self.assertEqual((op.x, op.y), (gx, gy))
        self.assertFalse(eng.button_down)

    def test_single_flicker_frame_never_grabs(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        # ONE mis-classified closed frame, back to open before the debounce.
        eng.update([_body(state="closed")], DESK)
        clk.tick(0.02)                               # < AIR_GRAB_DEBOUNCE_MS
        op = eng.update([_body(state="open")], DESK)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertFalse(eng.button_down)

    def test_disengage_mid_drag_releases(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        _settle_grab(eng, clk, "closed")
        self.assertTrue(eng.button_down)
        # Body vanishes mid-drag → exactly one OP_UP, then IDLE.
        op = eng.update([], DESK)
        self.assertEqual(op.kind, OP_UP)
        self.assertFalse(eng.button_down)
        self.assertEqual(eng.update([], DESK).kind, OP_IDLE)

    def test_release_helper_mid_drag(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        _settle_grab(eng, clk, "closed")
        op = eng.release()
        self.assertIsNotNone(op)
        self.assertEqual(op.kind, OP_UP)
        self.assertIsNone(eng.release())             # idempotent: nothing held


class TestScroll(unittest.TestCase):
    def test_lasso_arms_then_scrolls_signed(self):
        eng, _ = _engine()
        eng.update([_body()], DESK)
        armed = eng.update([_body(state="lasso")], DESK)
        self.assertEqual(armed.kind, OP_SCROLL)
        self.assertEqual(armed.scroll_amount, 0)
        # Hand UP 5 cm → positive scroll of dy * AIR_SCROLL_GAIN clicks.
        up = eng.update(
            [_body(hand=(0.0, _CENTER_HAND_Y + 0.05, _ENGAGED_Z),
                   state="lasso")], DESK)
        self.assertEqual(up.kind, OP_SCROLL)
        self.assertEqual(up.scroll_amount, int(round(0.05 * AIR_SCROLL_GAIN)))
        # Hand back DOWN → negative.
        down = eng.update([_body(state="lasso")], DESK)
        self.assertLess(down.scroll_amount, 0)

    def test_scroll_deadzone_emits_zero(self):
        eng, _ = _engine()
        eng.update([_body()], DESK)
        eng.update([_body(state="lasso")], DESK)     # arm
        jitter = AIR_SCROLL_DEADZONE_M * 0.5
        op = eng.update(
            [_body(hand=(0.0, _CENTER_HAND_Y + jitter, _ENGAGED_Z),
                   state="lasso")], DESK)
        self.assertEqual(op.kind, OP_SCROLL)
        self.assertEqual(op.scroll_amount, 0)


class TestBridgeContract(unittest.TestCase):
    """Pin the engine to the ACTUAL bridge keys (the 2026-07-07 mismatch fix:
    an earlier draft read 'hand_right_state', which the bridge never emits)."""

    def test_reads_bridge_hand_key(self):
        eng, clk = _engine()
        eng.update([_body()], DESK)
        # _body() writes the REAL bridge key "hand_right" — a closed fist there
        # must reach the grab machine (proves we don't read '*_state' only).
        op = _settle_grab(eng, clk, "closed")
        self.assertEqual(op.kind, OP_DOWN)

    def test_unknown_hand_state_never_grabs(self):
        eng, _ = _engine()
        # The bridge collapses NotTracked → "unknown": must move, never grab.
        op = eng.update([_body(state="unknown")], DESK)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertFalse(eng.button_down)

    def test_nearest_body_wins(self):
        eng, _ = _engine()
        far = _body(hand=(0.15, _CENTER_HAND_Y, _ENGAGED_Z), distance=3.5)
        near = _body(distance=1.8)                   # centre hand
        op = eng.update([far, near], DESK)
        self.assertEqual(op.kind, OP_MOVE)
        self.assertAlmostEqual(op.x, 960, delta=2)   # mapped from NEAR body


if __name__ == "__main__":
    unittest.main()
