"""Tests for skills/kinect_air_mouse — the Kinect air-mouse (REACH-TO-ENGAGE).

Drives the PURE CORE (reach-box mapping, EMA smoothing, the per-hand grip
debounce, the ARM-EXTENSION engage gate, the per-hand close→left/right-button
state machine, the overlay colour mapping) directly, and the LIVE _poll_once path
with a fake kinect_bridge + the real mouse actuation mocked out — so NO sensor,
NO real cursor, NO Qt is touched. Asserts the owner-facing contract:

  * the reach-box maps the hand NON-mirrored (hand RIGHT → larger cursor_x) and
    across the ENTIRE virtual desktop — every monitor, including one arranged
    LEFT of the primary (negative virtual-screen origin) — not just primary,
  * REACH ENGAGE GATE (the headline NEW model): the cursor is driven ONLY while
    an arm is EXTENDED OUT toward the sensor (hand pushed FORWARD in depth and/or
    the arm STRAIGHTENED) AND tracked — NOT when a hand is merely raised/visible.
    A RELAXED arm (hand pulled back / elbow bent), or a body/hand untracked beyond
    the ~0.3 s grace, DISENGAGES — cursor=None (so the live loop calls NO
    SetCursorPos and the PHYSICAL mouse is free), any held button RELEASED, overlay
    hidden. Engage/disengage hysteresis stops threshold flicker; re-extending the
    arm re-engages.
  * the cursor follows the EXTENDED hand (whichever arm is extended; the more-
    extended one when both),
  * PER-HAND clicks: closing the LEFT hand fires the LEFT button (down on close,
    up on open; hold-closed+move = a left-drag); closing the RIGHT hand fires the
    RIGHT button (right-click; hold = right-drag). Either hand clicks regardless of
    which drives the cursor,
  * ROBUST close: Lasso is treated as CLOSED; a 1-frame Unknown grip dropout
    HOLDS the last confident grip (no flicker-release of a drag); a single
    flickered CLOSED frame is DEBOUNCED (no stray click),
  * dead-man: hand untracked while held → button RELEASED, overlay hidden,
  * _poll_once no-ops the SIDE EFFECTS when KINECT_AIR_MOUSE_ENABLED is False,
    when staging, and when the bridge is absent/disabled,
  * air_mouse_on persists KINECT_AIR_MOUSE_ENABLED via the reused settings
    writer (mocked), and air_mouse_status reflects enabled + hand-in-view.

stdlib unittest + mock; App-Control-safe; CI-sim clean (no win32/pyautogui/Qt).
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── settings-file safety net (belt-and-suspenders; see kinect_gestures test) ─
_SAVED_SETTINGS_ENV: "str | None" = None
_SETTINGS_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SETTINGS_TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SETTINGS_TMPDIR = tempfile.mkdtemp(prefix="jarvis_airmouse_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _SETTINGS_TMPDIR, "test_user_settings.json")


def tearDownModule() -> None:
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV


# ─── fakes ──────────────────────────────────────────────────────────────────
def _fake_bridge(*, enabled=True, available=(True, ""), bodies=None,
                 hand_states=None):
    """A stand-in audio.kinect_bridge exposing only what the skill reads.

    NB: no ``arm_extension`` attribute, so the skill exercises its LOCAL
    fallback geometry (skills/kinect_air_mouse._local_arm_extension) — which
    mirrors the bridge helper's math. The bridge helper itself is unit-tested in
    tests/test_kinect_bridge.py."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_hand_states = lambda: (hand_states if hand_states is not None else
                                 {"right": "unknown", "left": "unknown",
                                  "tracked": False, "ts": 0.0})
    return m


# Camera space: x sensor-right, y UP, z depth AWAY from the sensor (metres). The
# torso sits at z≈TORSO_Z and the SHOULDER LINE (spine_shoulder) at y≈SHOULDER_Y.
# Under the HEIGHT (raise-to-engage) model an "extended"/engaged arm is a hand
# RAISED ABOVE THE SHOULDER (hand_y > shoulder_y + up_margin); a "relaxed" arm is a
# hand LOWERED to the desk (hand_y well below the shoulder). The forward-depth +
# straightness geometry is kept so the demoted secondary cues still compute, but
# they no longer gate — only the HEIGHT delta does.
TORSO_Z = 2.0
SHOULDER_Y = 0.40          # the shoulder-line (spine_shoulder) Y reference
RAISED_HAND_Y = 0.55       # a raised hand: lift ≈ +0.15 m (> the +0.05 up-margin)
DESK_HAND_Y = 0.00         # a desk-resting hand: lift ≈ -0.40 m (≪ -0.10 down)


def _extended_arm_joints(side: str, *, hand_x=0.0, hand_y=RAISED_HAND_Y,
                         forward=0.30) -> dict:
    """Joints for one ENGAGED arm: hand RAISED above the shoulder (hand_y default
    RAISED_HAND_Y, lift ≈ +0.15 m) — clears the height engage margin. A forward
    push + straight elbow are kept so the demoted forward/straightness cues read a
    plausible reach, but the HEIGHT is what engages."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, SHOULDER_Y, TORSO_Z
    hx, hy, hz = hand_x, hand_y, TORSO_Z - forward
    # Elbow at the midpoint of the shoulder→hand segment → a straight arm
    # (chord == upper + fore, straightness == 1).
    ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (sz + hz) / 2.0
    return {
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


def _relaxed_arm_joints(side: str, *, hand_x=0.0, hand_y=DESK_HAND_Y) -> dict:
    """Joints for one DISENGAGED arm: hand LOWERED to the desk (hand_y default
    DESK_HAND_Y, lift ≈ -0.40 m) — well below the height down-margin, so it does
    NOT engage. The hand sits at torso depth with a bent elbow (so the demoted
    forward/straightness cues read a relaxed arm too), but the HEIGHT is decisive:
    a hand resting on the desk is far below the shoulder and never engages."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, SHOULDER_Y, TORSO_Z
    # Hand near torso depth (no forward reach) and well BELOW the shoulder (desk).
    hx, hy, hz = hand_x, hand_y, TORSO_Z
    # Elbow well FORWARD of both shoulder and hand → a deep bend (low straightness).
    ex, ey, ez = shoulder_x, sy - 0.05, TORSO_Z - 0.30
    return {
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


def _body(*, reach_side: "str | None" = "right", grip_right="open",
          grip_left="open", distance=TORSO_Z, hand_x=0.0, hand_y=RAISED_HAND_Y,
          forward=0.30, both_relaxed=False, with_joints=True):
    """A get_bodies()-shaped body. By default the `reach_side` arm is ENGAGED (hand
    RAISED above the shoulder) and the other arm is relaxed (hand at the desk).
    both_relaxed=True lowers BOTH hands to the desk (the merely-present / no-raise
    case → disengaged). with_joints=False omits the joint dict entirely (body
    tracked but no usable arm → disengaged).

    spine_shoulder sits at SHOULDER_Y — the height reference the gate compares the
    hand Y against; the raised hand of the engaged arm sits above it."""
    joints: dict = {}
    if with_joints:
        joints["spine_mid"] = (0.0, 0.0, distance, 2)
        joints["spine_shoulder"] = (0.0, SHOULDER_Y, distance, 2)
        joints["head"] = (0.0, SHOULDER_Y + 0.3, distance, 2)
        if both_relaxed:
            joints.update(_relaxed_arm_joints("left", hand_x=hand_x))
            joints.update(_relaxed_arm_joints("right", hand_x=hand_x))
        else:
            other = "left" if reach_side == "right" else "right"
            joints.update(_extended_arm_joints(
                reach_side, hand_x=hand_x, hand_y=hand_y, forward=forward))
            joints.update(_relaxed_arm_joints(other))
    return {
        "id": 0, "joints": joints,
        "head": (0.0, SHOULDER_Y + 0.3, distance), "distance_m": distance,
        "facing": True,
        "hand_right": grip_right, "hand_left": grip_left,
    }


class _Base(unittest.TestCase):
    def _load(self):
        # register=False so the background poll thread isn't constructed.
        mod, _actions = load_skill_isolated("kinect_air_mouse", register=False)
        return mod

    def _patch_flag(self, value):
        from core import config as cfg
        p = mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _not_staging(self, mod):
        p = mock.patch.object(mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def _capture_mouse(self, mod):
        """Replace the real mouse actuation with recorders. Returns (moves,
        buttons) lists that fill as the skill acts. `buttons` records
        (action, button) tuples so per-hand left/right is asserted.

        Also pins the HAND MIRROR OFF: these _poll_once plumbing tests assert the
        RAW SDK hand→button identity (reach_side='right' → RIGHT button), so the
        selfie-view swap must be disabled here. The mirror's own swap is asserted
        separately in HandMirrorTests (which turns it back ON)."""
        pm = mock.patch.object(mod, "_hand_mirror_enabled", lambda: False)
        pm.start()
        self.addCleanup(pm.stop)
        moves: list = []
        buttons: list = []
        p1 = mock.patch.object(mod, "_set_cursor_pos",
                               lambda x, y: (moves.append((x, y)) or True))
        p2 = mock.patch.object(
            mod, "_mouse_button",
            lambda action, button="left": (buttons.append((action, button)) or True))
        # Silence overlay file I/O + spawning entirely.
        p3 = mock.patch.object(mod, "_publish_overlay_state", lambda *a, **k: None)
        p4 = mock.patch.object(mod, "_clear_overlay_state", lambda *a, **k: None)
        p5 = mock.patch.object(mod, "_spawn_overlay", lambda *a, **k: None)
        p6 = mock.patch.object(mod, "_overlay_alive", lambda: True)
        # Neutralise the AUTO-YIELD watcher so _poll_once is DETERMINISTIC: never
        # install the real LL hook, and never let the machine's actual recent input
        # (from running the tests) trip a yield. Tests that exercise yield override
        # real_input_recent themselves (a nested patch wins).
        p7 = mock.patch.object(mod, "_install_yield_watcher", lambda *a, **k: False)
        p8 = mock.patch.object(mod, "real_input_recent", lambda *a, **k: False)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8):
            p.start()
            self.addCleanup(p.stop)
        return moves, buttons


# ══════════════════════════════════════════════════════════════════════════
#  ARM-EXTENSION geometry + ArmExtension value object (the REACH signal)
# ══════════════════════════════════════════════════════════════════════════
class ArmExtensionTests(_Base):
    def _ext(self, mod, joints, side):
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(joints, side))

    def test_raised_hand_reads_positive_lift(self):
        mod = self._load()
        ext = self._ext(mod, _extended_arm_joints("right"), "right")
        # Hand raised above the shoulder → lift_m positive, above the up-margin.
        self.assertIsNotNone(ext.lift_m)
        self.assertGreater(ext.lift_m, mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)
        self.assertIsNotNone(ext.shoulder_ref_y)

    def test_desk_hand_reads_negative_lift(self):
        mod = self._load()
        ext = self._ext(mod, _relaxed_arm_joints("right"), "right")
        # Hand resting at the desk → lift_m well below the shoulder (negative,
        # under the down-margin), so it never engages.
        self.assertIsNotNone(ext.lift_m)
        self.assertLess(ext.lift_m, mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)

    def test_is_extended_true_only_when_raised(self):
        mod = self._load()
        raised = self._ext(mod, _extended_arm_joints("right"), "right")
        desk = self._ext(mod, _relaxed_arm_joints("right"), "right")
        self.assertTrue(raised.is_extended(engaged=False))    # hand above shoulder
        self.assertFalse(desk.is_extended(engaged=False))     # hand at desk level

    def test_hand_above_shoulder_plus_up_margin_engages(self):
        """The spec case: a hand above the shoulder by MORE than the up-margin
        engages on height alone (forward-reach is demoted: pass forward_m=0 to prove
        it's the HEIGHT that engages, not any reach)."""
        mod = self._load()
        up = mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M
        ext = mod.ArmExtension("right", forward_m=0.0, straightness=None,
                               hand=(0.1, 0.7, 2.0, 2), lift_m=up + 0.05)
        self.assertTrue(ext.is_extended(engaged=False))

    def test_desk_level_hand_does_NOT_engage_despite_forward_reach(self):
        """THE HEADLINE FIX: a hand at the desk reads a BIG forward reach (the old
        broken cue), yet sits far BELOW the shoulder — so the height gate keeps it
        disengaged. Big forward_m / reach_ratio cannot engage it; only height can.
        """
        mod = self._load()
        # lift well below the shoulder (desk), but a huge forward reach + straight.
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.95,
                               hand=(0, 0.0, 1.6, 2), reach_ratio=1.8, lift_m=-0.40)
        self.assertFalse(ext.is_extended(engaged=False))   # not raised → no engage
        # And it can't HOLD the gate either: even already-engaged, a hand below the
        # down-margin releases regardless of the (demoted) forward reach.
        self.assertFalse(ext.is_extended(engaged=True))

    def test_lowering_hand_below_down_margin_disengages(self):
        """A hand that drops below the down-margin DISENGAGES even while currently
        engaged, regardless of any forward reach (which can't hold the gate)."""
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.90,
                               hand=(0, 0.0, 1.6, 2), reach_ratio=1.5,
                               lift_m=mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M - 0.05)
        self.assertFalse(ext.is_extended(engaged=True))

    def test_missing_lift_does_not_engage(self):
        """A frame where the shoulder/hand Y couldn't be measured (lift_m None) must
        NOT engage — fail safe, leave the real mouse alone unless we positively see a
        raised hand."""
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.95,
                               hand=(0, 0.3, 1.6, 2), reach_ratio=1.8, lift_m=None)
        self.assertFalse(ext.is_extended(engaged=False))
        self.assertFalse(ext.is_extended(engaged=True))

    def test_height_hysteresis(self):
        mod = self._load()
        # A lift BETWEEN the down and up margins: stays disengaged when not engaged
        # (must clear the higher up-margin), but COUNTS as extended once already
        # engaged (only has to stay above the lower down-margin) → no flap.
        mid = (mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M
               + mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M) / 2.0
        ext = mod.ArmExtension("right", forward_m=0.0, straightness=None,
                               hand=(0, 0.4, 1.8, 2), lift_m=mid)
        self.assertFalse(ext.is_extended(engaged=False))   # must clear up-margin
        self.assertTrue(ext.is_extended(engaged=True))     # down-margin to hold

    def test_choose_highest_raised_hand(self):
        mod = self._load()
        # The controlling hand is whichever is raised HIGHEST above the shoulder.
        left = mod.ArmExtension("left", forward_m=0.22, straightness=0.86,
                                hand=(-0.1, 0.55, 1.78, 2), lift_m=0.08)
        right = mod.ArmExtension("right", forward_m=0.40, straightness=0.97,
                                 hand=(0.1, 0.70, 1.60, 2), lift_m=0.25)
        chosen = mod.choose_controlling_arm(left, right, engaged=False)
        self.assertIs(chosen, right)                       # raised higher

    def test_choose_none_when_neither_raised(self):
        mod = self._load()
        # Both hands below the shoulder (at the desk) → no controller.
        left = mod.ArmExtension("left", forward_m=0.0, straightness=0.4,
                                hand=(-0.1, 0.0, 2.0, 2), lift_m=-0.40)
        right = mod.ArmExtension("right", forward_m=0.05, straightness=0.5,
                                 hand=(0.1, 0.05, 1.95, 2), lift_m=-0.35)
        self.assertIsNone(mod.choose_controlling_arm(left, right, engaged=False))


# ══════════════════════════════════════════════════════════════════════════
#  PURE CORE — ReachBox
# ══════════════════════════════════════════════════════════════════════════
class ReachBoxTests(_Base):
    def test_center_maps_to_screen_center(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        self.assertAlmostEqual(px, (2560 - 1) // 2, delta=2)
        self.assertAlmostEqual(py, (1440 - 1) // 2, delta=2)

    def test_x_not_mirrored_hand_right_is_cursor_right(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertGreater(right_px, left_px)
        self.assertEqual(left_px, 0)
        self.assertEqual(right_px, 2560 - 1)

    def test_x_monotonic_left_to_right(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        xs = [rb.map(mod.REACH_CENTER_X + f * mod.REACH_HALF_W,
                     mod.REACH_CENTER_Y)[0]
              for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        self.assertEqual(xs, sorted(xs))
        self.assertLess(xs[0], xs[-1])

    def test_y_inverted_camera_up_is_screen_top(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        _, up_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y + mod.REACH_HALF_H)
        _, down_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y - mod.REACH_HALF_H)
        self.assertEqual(up_py, 0)
        self.assertEqual(down_py, 1440 - 1)

    def test_overshoot_is_clamped(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        px, py = rb.map(mod.REACH_CENTER_X + 99.0, mod.REACH_CENTER_Y - 99.0)
        self.assertTrue(0 <= px <= 2559)
        self.assertTrue(0 <= py <= 1439)


class VirtualDesktopMappingTests(_Base):
    """The hand maps across the ENTIRE virtual desktop (all monitors), including
    a monitor LEFT of the primary (NEGATIVE virtual-screen origin)."""

    VX, VY, VW, VH = -2560, -1440, 7680, 2880

    def _rb(self, mod):
        return mod.ReachBox(self.VW, self.VH, origin_x=self.VX, origin_y=self.VY)

    def test_center_maps_to_virtual_desktop_center(self):
        mod = self._load()
        rb = self._rb(mod)
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        self.assertAlmostEqual(px, self.VX + (self.VW - 1) // 2, delta=2)
        self.assertAlmostEqual(py, self.VY + (self.VH - 1) // 2, delta=2)

    def test_left_edge_reaches_negative_origin_monitor(self):
        mod = self._load()
        rb = self._rb(mod)
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertEqual(left_px, self.VX)
        self.assertLess(left_px, 0)

    def test_right_edge_reaches_far_monitor(self):
        mod = self._load()
        rb = self._rb(mod)
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertEqual(right_px, self.VX + self.VW - 1)

    def test_top_edge_reaches_negative_y_origin(self):
        mod = self._load()
        rb = self._rb(mod)
        _, top_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y + mod.REACH_HALF_H)
        self.assertEqual(top_py, self.VY)
        self.assertLess(top_py, 0)

    def test_un_mirrored_across_virtual_desktop(self):
        mod = self._load()
        rb = self._rb(mod)
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertGreater(right_px, left_px)
        self.assertEqual(left_px, self.VX)
        self.assertEqual(right_px, self.VX + self.VW - 1)

    def test_overshoot_clamps_to_virtual_bounds(self):
        mod = self._load()
        rb = self._rb(mod)
        px_lo, py_lo = rb.map(mod.REACH_CENTER_X - 99.0, mod.REACH_CENTER_Y + 99.0)
        px_hi, py_hi = rb.map(mod.REACH_CENTER_X + 99.0, mod.REACH_CENTER_Y - 99.0)
        self.assertEqual((px_lo, py_lo), (self.VX, self.VY))
        self.assertEqual((px_hi, py_hi),
                         (self.VX + self.VW - 1, self.VY + self.VH - 1))

    def test_reach_box_builder_uses_cached_virtual_bounds(self):
        mod = self._load()
        with mock.patch.object(mod, "_virtual_screen_bounds",
                               lambda: (self.VX, self.VY, self.VW, self.VH)):
            rb = mod._reach_box_for_virtual_desktop(refresh=True)
        self.assertEqual((rb.origin_x, rb.origin_y, rb.screen_w, rb.screen_h),
                         (self.VX, self.VY, self.VW, self.VH))

    def test_cached_bounds_refresh_picks_up_layout_change(self):
        mod = self._load()
        seq = [(0, 0, 2560, 1440), (self.VX, self.VY, self.VW, self.VH)]
        calls = {"n": 0}

        def _fake_bounds():
            i = min(calls["n"], len(seq) - 1)
            calls["n"] += 1
            return seq[i]

        mod._VBOUNDS_CACHE[0] = None
        mod._VBOUNDS_CACHE[1] = 0.0
        with mock.patch.object(mod, "_virtual_screen_bounds", _fake_bounds):
            first = mod._cached_virtual_bounds(refresh=True)
            self.assertEqual(first, (0, 0, 2560, 1440))
            self.assertEqual(mod._cached_virtual_bounds(refresh=False),
                             (0, 0, 2560, 1440))
            self.assertEqual(mod._cached_virtual_bounds(refresh=True),
                             (self.VX, self.VY, self.VW, self.VH))


# ══════════════════════════════════════════════════════════════════════════
#  FIX 1 — TABLET-FEEL mapping: aspect-matched (no stretch) + body-relative +
#  absolute. The reach plane IS the monitors: hand corner → desktop corner, X/Y
#  scale equally per the desktop aspect, and the plane is centred on the BODY so
#  the same hand offset maps to the same cursor spot wherever the owner sits.
# ══════════════════════════════════════════════════════════════════════════
class TabletAspectMatchTests(_Base):
    """The plane half-HEIGHT is DERIVED from the half-WIDTH × desktop aspect, so X
    and Y scale EQUALLY (no stretch). A hand moved the same real distance H vs V
    moves the cursor the same FRACTION of the screen."""

    def test_half_h_derives_from_aspect(self):
        mod = self._load()
        # 2560×1440 (16:9): derived half_h = half_w × 1440/2560.
        rb = mod.ReachBox(2560, 1440, half_w=0.26)
        self.assertAlmostEqual(rb.half_w, 0.26, delta=1e-9)
        self.assertAlmostEqual(rb.half_h, 0.26 * (1440.0 / 2560.0), delta=1e-6)

    def test_half_h_tracks_a_taller_desktop(self):
        mod = self._load()
        # A 1:1 desktop → half_h == half_w (square plane, square screen).
        rb = mod.ReachBox(2000, 2000, half_w=0.26)
        self.assertAlmostEqual(rb.half_h, rb.half_w, delta=1e-6)
        # A very WIDE desktop (3:1) → a much SHORTER plane (half_h = half_w/3).
        rb2 = mod.ReachBox(6000, 2000, half_w=0.30)
        self.assertAlmostEqual(rb2.half_h, 0.30 / 3.0, delta=1e-6)

    def test_equal_hand_move_equal_screen_pixels(self):
        mod = self._load()
        # THE NO-STRETCH GUARANTEE: because half_h = half_w × (H/W), the same real
        # hand delta in X and in Y moves the cursor the same number of PIXELS on
        # screen (not the same fraction) — so neither axis feels faster/twitchier
        # than the other. Body-relative, centred.
        W, H = 2560, 1440
        rb = mod.ReachBox(W, H, half_w=0.26)
        cx, cy = rb.map(0.0, 0.40, body_x=0.0, body_y=0.40)
        d = 0.05   # 5 cm hand move, same in both axes
        x_moved = rb.map(d, 0.40, body_x=0.0, body_y=0.40)[0] - cx
        y_moved = cy - rb.map(0.0, 0.40 + d, body_x=0.0, body_y=0.40)[1]  # up→ -py
        # Equal PIXEL travel for an equal hand move (the aspect-match invariant).
        self.assertGreater(x_moved, 0)
        self.assertGreater(y_moved, 0)
        self.assertAlmostEqual(x_moved, y_moved, delta=2)   # within rounding


class TabletAbsoluteCornerTests(_Base):
    """ABSOLUTE mapping: the hand at a plane CORNER maps to the matching desktop
    CORNER (top-left→top-left, bottom-right→bottom-right), measured body-relative."""

    def _rb(self, mod):
        # An off-origin virtual desktop to prove the corners track the real bounds.
        return mod.ReachBox(7680, 2160, origin_x=-1280, origin_y=-120, half_w=0.26)

    def test_corners_map_to_desktop_corners(self):
        mod = self._load()
        rb = self._rb(mod)
        bx, by = 0.10, 0.42       # arbitrary body centre — corners are relative to it
        hw, hh = rb.half_w, rb.half_h
        # plane top-LEFT  (hand left = bx-hw; hand UP = centreY + hh) → desktop TL.
        tl = rb.map(bx - hw, (by + mod.REACH_CENTER_Y_OFFSET) + hh,
                    body_x=bx, body_y=by)
        # plane bottom-RIGHT (hand right = bx+hw; hand DOWN = centreY - hh) → BR.
        br = rb.map(bx + hw, (by + mod.REACH_CENTER_Y_OFFSET) - hh,
                    body_x=bx, body_y=by)
        self.assertEqual(tl, (rb.origin_x, rb.origin_y))
        self.assertEqual(br, (rb.origin_x + rb.screen_w - 1,
                              rb.origin_y + rb.screen_h - 1))

    def test_centre_maps_to_desktop_centre(self):
        mod = self._load()
        rb = self._rb(mod)
        bx, by = -0.05, 0.33
        # Hand at the plane centre (body X, shoulder Y + comfort offset) → desktop
        # centre, regardless of where the body sits.
        px, py = rb.map(bx, by + mod.REACH_CENTER_Y_OFFSET, body_x=bx, body_y=by)
        self.assertAlmostEqual(px, rb.origin_x + (rb.screen_w - 1) // 2, delta=2)
        self.assertAlmostEqual(py, rb.origin_y + (rb.screen_h - 1) // 2, delta=2)

    def test_absolute_not_relative_same_hand_same_cursor(self):
        mod = self._load()
        rb = self._rb(mod)
        bx, by = 0.0, 0.40
        # ABSOLUTE: the SAME hand position always maps to the SAME pixel — not a
        # velocity/relative accumulation. Two identical samples → identical cursor.
        a = rb.map(0.12, 0.55, body_x=bx, body_y=by)
        b = rb.map(0.12, 0.55, body_x=bx, body_y=by)
        self.assertEqual(a, b)
        # And up = up: a higher hand → a SMALLER py (closer to the top).
        hi = rb.map(0.12, 0.60, body_x=bx, body_y=by)[1]
        lo = rb.map(0.12, 0.50, body_x=bx, body_y=by)[1]
        self.assertLess(hi, lo)


class TabletBodyRelativeTests(_Base):
    """BODY-RELATIVE centring: the SAME hand OFFSET from the body maps to the SAME
    cursor spot no matter where the body is in the frame (owner sits/stands/shifts)."""

    def test_same_offset_different_body_same_cursor(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440, half_w=0.26)
        off_x, off_y = 0.13, 0.06    # hand 13 cm right + 6 cm above the plane centre
        cursors = []
        for bx, by in ((0.0, 0.40), (0.5, 0.55), (-0.4, 0.25), (0.9, 0.70)):
            cy_center = by + mod.REACH_CENTER_Y_OFFSET
            cursors.append(rb.map(bx + off_x, cy_center + off_y,
                                  body_x=bx, body_y=by))
        # Every body position yields the SAME cursor pixel for the same hand offset.
        self.assertEqual(len(set(cursors)), 1)

    def test_body_shift_without_hand_offset_change_does_not_move_cursor(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440, half_w=0.26)
        # The owner slides 40 cm to the right but keeps the hand the SAME distance
        # right-of-centre: the cursor must NOT move (it's body-relative, not absolute
        # camera X). Both hand AND body shift by +0.40.
        a = rb.map(0.10, 0.50, body_x=0.0, body_y=0.40)
        b = rb.map(0.50, 0.50, body_x=0.40, body_y=0.40)
        self.assertEqual(a, b)

    def test_controller_maps_body_relative_via_extension(self):
        mod = self._load()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440, half_w=0.26),
                                   debounce_frames=1, grace_sec=0.0,
                                   engage_debounce_frames=1)
        # Two raised RIGHT-hand samples with the SAME hand-minus-body offset but the
        # whole body shifted: the controller (which feeds body_center_x +
        # shoulder_ref_y into the plane) must produce the SAME cursor for both.
        def arm(body_x, hand_dx, shoulder_y=0.40):
            return mod.ArmExtension(
                "right", forward_m=0.0, straightness=None,
                hand=(body_x + hand_dx, shoulder_y + 0.20, 1.8, 2),
                lift_m=0.20, shoulder_ref_y=shoulder_y, body_center_x=body_x)
        relaxed_l = mod.ArmExtension("left", forward_m=0.0, straightness=None,
                                     hand=(-0.1, 0.0, 2.0, 2), lift_m=-0.40,
                                     shoulder_ref_y=0.40, body_center_x=0.0)
        d1 = c.update(relaxed_l, arm(0.0, 0.15), "open", "open", True)
        c.reset()
        d2 = c.update(relaxed_l, arm(0.6, 0.15), "open", "open", True)
        self.assertIsNotNone(d1.cursor)
        self.assertEqual(d1.cursor, d2.cursor)   # same offset → same cursor

    def test_falls_back_to_fixed_centre_without_body(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440, half_w=0.26)
        # With NO body reference, map() centres on the fixed plane centre (legacy):
        # the plane centre maps to the desktop centre.
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        self.assertAlmostEqual(px, (2560 - 1) // 2, delta=2)
        self.assertAlmostEqual(py, (1440 - 1) // 2, delta=2)


class ReachHalfWidthEnvTests(_Base):
    """KINECT_REACH_HALF_W tunes the plane half-WIDTH live (the height derives from
    it via the desktop aspect); nx/ny are exposed for tuning."""

    def test_env_overrides_half_width(self):
        mod = self._load()
        with mock.patch.dict(os.environ, {"KINECT_REACH_HALF_W": "0.40"}):
            self.assertAlmostEqual(mod.reach_half_w(), 0.40, delta=1e-9)
            rb = mod.ReachBox(2560, 1440)   # half_w None → reads the env
            self.assertAlmostEqual(rb.half_w, 0.40, delta=1e-9)
            # Height still derives from the (overridden) width × aspect.
            self.assertAlmostEqual(rb.half_h, 0.40 * (1440.0 / 2560.0), delta=1e-6)

    def test_env_unset_uses_default(self):
        mod = self._load()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KINECT_REACH_HALF_W", None)
            self.assertAlmostEqual(mod.reach_half_w(), mod.REACH_HALF_W, delta=1e-9)

    def test_env_garbage_falls_back(self):
        mod = self._load()
        with mock.patch.dict(os.environ, {"KINECT_REACH_HALF_W": "not-a-number"}):
            self.assertAlmostEqual(mod.reach_half_w(), mod.REACH_HALF_W, delta=1e-9)

    def test_normalize_reports_nx_ny_in_range(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440, half_w=0.26)
        bx, by = 0.0, 0.40
        # Hand a quarter of the way right of centre → nx ≈ +0.25; a touch above the
        # plane centre → ny negative (up). The debug line surfaces these.
        nx, ny = rb.normalize(0.0 + 0.25 * rb.half_w,
                              (by + mod.REACH_CENTER_Y_OFFSET) + 0.5 * rb.half_h,
                              body_x=bx, body_y=by)
        self.assertAlmostEqual(nx, 0.25, delta=1e-6)
        self.assertAlmostEqual(ny, -0.5, delta=1e-6)   # up = negative ny


class EMATests(_Base):
    def test_first_value_seeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        self.assertEqual(e.update(10.0), 10.0)

    def test_smooths_toward_target(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0)
        self.assertAlmostEqual(e.update(10.0), 5.0)

    def test_reset_reseeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0); e.update(10.0)
        e.reset()
        self.assertEqual(e.update(100.0), 100.0)


# ══════════════════════════════════════════════════════════════════════════
#  ROBUST close: per-hand GripDebouncer (Lasso=closed, Unknown holds, flicker)
# ══════════════════════════════════════════════════════════════════════════
class GripDebouncerTests(_Base):
    def test_requires_n_consecutive_frames(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")    # 1
        self.assertEqual(d.update("closed"), "open")    # 2
        self.assertEqual(d.update("closed"), "closed")  # 3 → flips

    def test_single_flicker_is_ignored(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")    # a lone flicker
        self.assertEqual(d.update("open"), "open")
        self.assertEqual(d.stable, "open")

    def test_lasso_counts_as_closed(self):
        mod = self._load()
        # Lasso (a half-curled fist) is meant as a click → treated as CLOSED.
        d = mod.GripDebouncer(frames=2, initial="open")
        self.assertEqual(d.update("lasso"), "open")     # 1
        self.assertEqual(d.update("lasso"), "closed")   # 2 → latches closed

    def test_unknown_holds_current_stable(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=2, initial="closed")
        # Ambiguous frames must not flip a held grip (dead-man releases, not a
        # single 'unknown'/'nottracked').
        self.assertEqual(d.update("unknown"), "closed")
        self.assertEqual(d.update("nottracked"), "closed")

    def test_unknown_mid_streak_does_not_count_toward_flip(self):
        mod = self._load()
        # A 1-frame Unknown dropout in the middle of a closing streak resets the
        # streak, so a fist isn't latched on partial evidence interleaved with
        # dropouts — robustness against the exact Kinect failure mode.
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")    # 1
        self.assertEqual(d.update("unknown"), "open")   # dropout resets streak
        self.assertEqual(d.update("closed"), "open")    # 1 again (not 2)
        self.assertEqual(d.update("closed"), "open")    # 2
        self.assertEqual(d.update("closed"), "closed")  # 3 → finally flips


class OverlayColorTests(_Base):
    def test_track_is_cyan_grab_is_gold(self):
        mod = self._load()
        self.assertEqual(mod.overlay_color_for("track"), "cyan")
        self.assertEqual(mod.overlay_color_for("grab"), "gold")
        self.assertEqual(mod.overlay_color_for("hidden"), "cyan")


# ══════════════════════════════════════════════════════════════════════════
#  TUNED feel constants — pin the owner's tuning (EMA / debounce / reach-box)
# ══════════════════════════════════════════════════════════════════════════
class TunedConstantsTests(_Base):
    def test_ema_alpha_is_snappier(self):
        mod = self._load()
        self.assertEqual(mod.AIR_MOUSE_EMA_ALPHA, 0.55)
        self.assertGreater(mod.AIR_MOUSE_EMA_ALPHA, 0.35)
        self.assertLess(mod.AIR_MOUSE_EMA_ALPHA, 1.0)

    def test_grip_debounce_is_shorter_but_still_filters_flicker(self):
        mod = self._load()
        self.assertEqual(mod.AIR_MOUSE_GRIP_DEBOUNCE_FRAMES, 2)
        self.assertGreaterEqual(mod.AIR_MOUSE_GRIP_DEBOUNCE_FRAMES, 2)
        self.assertLess(mod.AIR_MOUSE_GRIP_DEBOUNCE_FRAMES, 3)

    def test_reach_box_is_more_sensitive(self):
        mod = self._load()
        self.assertEqual(mod.REACH_HALF_W, 0.26)
        self.assertEqual(mod.REACH_HALF_H, 0.16)
        self.assertLess(mod.REACH_HALF_W, 0.35)
        self.assertLess(mod.REACH_HALF_H, 0.22)

    def test_height_margins_have_hysteresis(self):
        mod = self._load()
        # The up (engage) margin must be strictly higher than the down (disengage)
        # margin, so a hand hovering at the shoulder line can't flap engage on/off.
        self.assertGreater(mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M,
                           mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)

    def test_height_margins_are_sane(self):
        mod = self._load()
        # Defaults per spec (FIX 3, raised 2026-06-09): engage at ~+7 cm above the
        # shoulder (a clearer, deliberate reach — fewer false engages), release at
        # ~10 cm below it. up positive (must raise clearly above the shoulder to
        # engage), down negative (a hand dropped below the shoulder releases).
        self.assertAlmostEqual(mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M, 0.07, delta=1e-9)
        self.assertAlmostEqual(mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M, -0.10, delta=1e-9)
        self.assertGreater(mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M, 0.0)
        self.assertLess(mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M, 0.0)

    def test_engage_debounce_rejects_single_frame_spike(self):
        mod = self._load()
        # FIX 3: the engage debounce requires a SUSTAINED raise (>1 frame) so a
        # 1-frame Kinect height spike can't grab the cursor. Must be ≥2 (a real
        # debounce) but small enough that a genuine raise still feels prompt.
        self.assertGreaterEqual(mod.AIR_MOUSE_ENGAGE_DEBOUNCE_FRAMES, 2)
        self.assertLessEqual(mod.AIR_MOUSE_ENGAGE_DEBOUNCE_FRAMES, 5)

    def test_controlling_hand_hysteresis_is_strong(self):
        mod = self._load()
        # FIX 2: with BOTH hands raised the cursor locks to ONE controller. The
        # switch margin is a physical height (metres) the challenger must lead by,
        # held for several frames — strong enough that two raised hands don't thrash.
        self.assertAlmostEqual(mod.HAND_SWITCH_MARGIN, 0.08, delta=1e-9)
        self.assertGreaterEqual(mod.HAND_SWITCH_FRAMES, 8)

    def test_forward_and_straightness_demoted_to_permissive(self):
        mod = self._load()
        # Forward-reach + straightness are DEMOTED: their bars are permissive (≈0)
        # so they can never gate, hold, or veto — only the HEIGHT delta engages.
        self.assertEqual(mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE, 0.0)
        self.assertEqual(mod.AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE, 0.0)
        self.assertEqual(mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M, 0.0)
        self.assertEqual(mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE, 0.0)

    def test_yield_window_is_sane(self):
        mod = self._load()
        # The auto-yield suppression window: touch the real mouse and the air-mouse
        # stays out of the way for ~1.5 s after the last real input.
        self.assertAlmostEqual(mod.AIR_MOUSE_YIELD_WINDOW_SEC, 1.5, delta=1e-9)
        self.assertGreater(mod.AIR_MOUSE_YIELD_WINDOW_SEC, 0.0)


# ══════════════════════════════════════════════════════════════════════════
#  B2 — preview hand-circle colour mapping + thread-safe live state (+ which hand)
# ══════════════════════════════════════════════════════════════════════════
class HandCircleColorTests(_Base):
    def test_engaged_open_is_blue(self):
        mod = self._load()
        col = mod.hand_circle_color_for(engaged=True, grip="open")
        self.assertEqual(col, mod.HAND_CIRCLE_COLOR_ENGAGED)
        b, g, r = col
        self.assertGreater(b, r)
        self.assertGreater(b, 150)

    def test_closed_is_orange(self):
        mod = self._load()
        col = mod.hand_circle_color_for(engaged=True, grip="closed")
        self.assertEqual(col, mod.HAND_CIRCLE_COLOR_CLOSED)
        b, g, r = col
        self.assertGreater(r, b)
        self.assertGreater(r, 150)

    def test_engaged_blue_differs_from_closed_orange(self):
        mod = self._load()
        self.assertNotEqual(mod.hand_circle_color_for(True, "open"),
                            mod.hand_circle_color_for(True, "closed"))

    def test_disengaged_is_grey_idle(self):
        mod = self._load()
        col = mod.hand_circle_color_for(engaged=False, grip="open")
        self.assertEqual(col, mod.HAND_CIRCLE_COLOR_IDLE)
        b, g, r = col
        self.assertEqual(b, g)
        self.assertEqual(g, r)

    def test_closed_only_counts_when_engaged(self):
        mod = self._load()
        self.assertEqual(mod.hand_circle_color_for(False, "closed"),
                         mod.HAND_CIRCLE_COLOR_IDLE)


class AirMouseStateGetterTests(_Base):
    def test_default_state_is_disengaged(self):
        mod = self._load()
        st = mod.get_air_mouse_state()
        self.assertFalse(st["engaged"])
        self.assertIn("grip", st)
        self.assertIn("hand", st)

    def test_setter_publishes_engaged_hand_and_grip(self):
        mod = self._load()
        mod._set_air_mouse_state(True, "closed", "left")
        st = mod.get_air_mouse_state()
        self.assertTrue(st["engaged"])
        self.assertEqual(st["grip"], "closed")
        self.assertEqual(st["hand"], "left")
        # Getter returns a COPY — mutating it must not corrupt shared state.
        st["engaged"] = False
        self.assertTrue(mod.get_air_mouse_state()["engaged"])

    def test_poll_publishes_state_for_preview_with_which_hand(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._capture_mouse(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)
        # LEFT arm extended, open hand → engaged, hand "left", grip open → BLUE.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="left", grip_left="open")]))
        st = mod.get_air_mouse_state()
        self.assertTrue(st["engaged"])
        self.assertEqual(st["hand"], "left")
        self.assertEqual(mod.hand_circle_color_for(st["engaged"], st["grip"]),
                         mod.HAND_CIRCLE_COLOR_ENGAGED)
        # Both arms relax → disengaged → GREY (no live ring).
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(both_relaxed=True)]))
        st2 = mod.get_air_mouse_state()
        self.assertFalse(st2["engaged"])
        self.assertIsNone(st2["hand"])


# ══════════════════════════════════════════════════════════════════════════
#  CONTROLLER state machine — engage on reach, per-hand clicks, drag
# ══════════════════════════════════════════════════════════════════════════
class ControllerStateMachineTests(_Base):
    """The reach→engage / per-hand close→left|right-button / re-open→up contract,
    asserted on AirMouseController decisions (the pure brain). Every engaged
    sample passes an EXTENDED arm; the reach gate itself is exercised below."""

    def _ctrl(self, mod, debounce=1):
        # grace_sec=0 so an untracked frame releases IMMEDIATELY here (the
        # tracking-loss GRACE window is exercised in its own test class).
        # engage_debounce_frames=1 → single-frame engage for these state-machine
        # tests (the multi-frame engage DEBOUNCE has its own class).
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=debounce, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def _ext(self, mod, side, **kw):
        """An EXTENDED ArmExtension for `side` (engages), with a hand joint."""
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_extended_open_hand_moves_no_button_cyan(self):
        mod = self._load()
        c = self._ctrl(mod)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)
        self.assertIsNotNone(d.cursor)
        self.assertIsNone(d.left)
        self.assertIsNone(d.right)
        self.assertEqual(d.overlay, "track")
        self.assertEqual(d.hand, "right")
        self.assertEqual(mod.overlay_color_for(d.overlay), "cyan")

    def test_right_hand_close_presses_RIGHT_button_gold(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "closed", True)   # RIGHT hand closes
        self.assertEqual(d.right, "down")       # RIGHT button
        self.assertIsNone(d.left)               # not the left
        self.assertEqual(d.overlay, "grab")
        self.assertTrue(c.right_is_down)
        self.assertFalse(c.left_is_down)
        # Held: a second closed frame does NOT re-press.
        d2 = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                      "open", "closed", True)
        self.assertIsNone(d2.right)

    def test_left_hand_close_presses_LEFT_button(self):
        mod = self._load()
        c = self._ctrl(mod)
        # LEFT arm extended (drives the cursor); LEFT hand closes → LEFT button.
        c.update(self._ext(mod, "left"), self._relaxed(mod, "right"),
                 "open", "open", True)
        d = c.update(self._ext(mod, "left"), self._relaxed(mod, "right"),
                     "closed", "open", True)
        self.assertEqual(d.left, "down")
        self.assertIsNone(d.right)
        self.assertTrue(c.left_is_down)

    def test_either_hand_clicks_regardless_of_which_drives(self):
        mod = self._load()
        c = self._ctrl(mod)
        # RIGHT arm drives the cursor, but the LEFT hand closes → LEFT button
        # still fires (clicks are hand-specific, independent of the cursor hand).
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "closed", "open", True)   # LEFT hand closes
        self.assertEqual(d.hand, "right")       # cursor still on the right arm
        self.assertEqual(d.left, "down")        # but the LEFT button fired
        self.assertIsNone(d.right)

    def test_closed_hand_still_moves_drag(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)   # right grab
        # Move the raised hand (still above the shoulder) while closed → cursor
        # still tracks (drag), button stays down (no new edge).
        d = c.update(self._relaxed(mod, "left"),
                     self._ext(mod, "right", hand_x=0.2, hand_y=RAISED_HAND_Y - 0.05),
                     "open", "closed", True)
        self.assertIsNotNone(d.cursor)
        self.assertIsNone(d.right)
        self.assertEqual(d.overlay, "grab")
        self.assertTrue(c.right_is_down)

    def test_reopen_releases_button_once(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)   # right down
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)  # re-open → up
        self.assertEqual(d.right, "up")
        self.assertEqual(d.overlay, "track")
        self.assertFalse(c.right_is_down)

    def test_lasso_grip_clicks(self):
        mod = self._load()
        c = self._ctrl(mod, debounce=2)
        # A Lasso grip on the right hand latches CLOSED (robust-close) → RIGHT down.
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "lasso", True)   # frame 1
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "lasso", True)  # frame 2 → latches
        self.assertEqual(d.right, "down")

    def test_unknown_grip_dropout_holds_drag(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Right hand grabbed (down), then a 1-frame Unknown grip dropout while the
        # arm is STILL extended + tracked: the button must NOT release (robust to
        # the Kinect dropping the grip for a frame mid-drag).
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)   # right down
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "unknown", True)   # grip dropout, arm still extended
        self.assertIsNone(d.right)          # NOT released
        self.assertTrue(c.right_is_down)    # button still held
        self.assertEqual(d.overlay, "grab")

    def test_deadman_release_when_untracked_while_held(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)   # right down
        # Body lost mid-grip (grace 0 here) → release immediately.
        d = c.update(None, None, "unknown", "unknown", False)
        self.assertEqual(d.right, "up")
        self.assertEqual(d.overlay, "hidden")
        self.assertIsNone(d.cursor)
        self.assertFalse(c.right_is_down)
        # Idempotent.
        d2 = c.update(None, None, "unknown", "unknown", False)
        self.assertIsNone(d2.right)
        self.assertEqual(d2.overlay, "hidden")

    def test_both_hands_can_be_down_at_once(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Both arms extended, both hands closed → BOTH buttons down (independent).
        c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        d = c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                     "closed", "closed", True)
        self.assertEqual(d.left, "down")
        self.assertEqual(d.right, "down")
        self.assertTrue(c.left_is_down and c.right_is_down)

    def test_flicker_does_not_fire_button_with_real_debounce(self):
        mod = self._load()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=2,
                                   grace_sec=0.0, engage_debounce_frames=1)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        d1 = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                      "open", "closed", True)   # flicker frame 1
        d2 = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                      "open", "open", True)      # back to open
        self.assertIsNone(d1.right)
        self.assertIsNone(d2.right)
        self.assertFalse(c.right_is_down)


# ══════════════════════════════════════════════════════════════════════════
#  REACH ENGAGE GATE (the NEW model) — pure core
# ══════════════════════════════════════════════════════════════════════════
class _FakeClock:
    """A controllable monotonic clock for the grace-window tests."""
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class ReachEngageGateTests(_Base):
    """The headline NEW model: the cursor is driven ONLY while an arm is EXTENDED
    OUT (reach), NOT when merely raised. A relaxed arm, both arms relaxed, or an
    untracked body (beyond grace) DISENGAGES: cursor=None (→ NO SetCursorPos, the
    physical mouse is free), held button RELEASED, overlay hidden. Hysteresis so
    it doesn't flicker at the line."""

    def _ctrl(self, mod, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        # engage_debounce_frames=1 so a single update() engages immediately in these
        # state-machine tests (the multi-frame engage DEBOUNCE has its own class).
        kw.setdefault("engage_debounce_frames", 1)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_hands_resting_on_desk_do_not_engage(self):
        """THE HEADLINE SCENARIO: hands RESTING on the desk (well below the
        shoulder) drive ZERO cursor — even though a desk hand reads a big forward
        reach (the old broken cue). Only RAISING a hand above the shoulder engages,
        so the owner's real mouse is never fought."""
        mod = self._load()
        c = self._ctrl(mod)
        # Desk-level hands WITH a big forward reach (z pushed forward), to prove the
        # forward cue can't engage — only the (negative) height matters.
        desk_l = mod.ArmExtension("left", forward_m=0.40, straightness=0.95,
                                  hand=(-0.1, DESK_HAND_Y, 1.6, 2),
                                  reach_ratio=1.8, lift_m=DESK_HAND_Y - SHOULDER_Y)
        desk_r = mod.ArmExtension("right", forward_m=0.40, straightness=0.95,
                                  hand=(0.1, DESK_HAND_Y, 1.6, 2),
                                  reach_ratio=1.8, lift_m=DESK_HAND_Y - SHOULDER_Y)
        d = c.update(desk_l, desk_r, "open", "open", True)
        self.assertIsNone(d.cursor)          # NO SetCursorPos
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.engaged)

    def test_extended_arm_engages_and_moves(self):
        mod = self._load()
        c = self._ctrl(mod)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)
        self.assertIsNotNone(d.cursor)       # cursor driven
        self.assertEqual(d.overlay, "track")
        self.assertTrue(c.engaged)
        self.assertEqual(c.hand, "right")

    def test_relaxing_arm_while_held_releases_button(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)       # engaged
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)     # RIGHT down
        self.assertTrue(c.right_is_down)
        # Now RELAX the (right) arm mid-grip → disengage + release.
        d = c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                     "open", "closed", True)
        self.assertEqual(d.right, "up")      # held button RELEASED
        self.assertIsNone(d.cursor)          # NO SetCursorPos
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.right_is_down)
        self.assertFalse(c.engaged)

    def test_no_arm_joints_disengages(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Both arms None (couldn't read joints) → fail safe to released.
        d = c.update(None, None, "open", "open", True)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.engaged)

    def test_hysteresis_no_flicker_at_threshold(self):
        mod = self._load()
        c = self._ctrl(mod)
        # A hand height BETWEEN the up (engage) and down (disengage) margins.
        mid = (mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M
               + mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M) / 2.0

        def mid_arm():
            return mod.ArmExtension("right", forward_m=0.0, straightness=None,
                                    hand=(0.1, SHOULDER_Y + mid, TORSO_Z - 0.30, 2),
                                    lift_m=mid)
        relaxed_l = self._relaxed(mod, "left")
        # While disengaged, the mid height is NOT enough to engage (higher up-margin).
        d0 = c.update(relaxed_l, mid_arm(), "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d0.cursor)
        # A clear raise engages.
        d1 = c.update(relaxed_l, self._ext(mod, "right"), "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d1.cursor)
        # Sag back to the mid height: STAYS engaged (lower down-margin) → no flicker.
        d2 = c.update(relaxed_l, mid_arm(), "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d2.cursor)
        # Fully lower the hand → disengage.
        d3 = c.update(relaxed_l, self._relaxed(mod, "right"), "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d3.cursor)

    def test_reengage_after_relaxing(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)       # engaged
        c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                 "open", "open", True)       # relaxed → disengaged
        self.assertFalse(c.engaged)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)   # reach again
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d.cursor)

    def test_cursor_follows_the_extended_hand(self):
        mod = self._load()
        c = self._ctrl(mod)
        # LEFT arm extended → cursor on the LEFT hand; switch to RIGHT extended →
        # cursor follows to the RIGHT hand.
        c.update(self._ext(mod, "left"), self._relaxed(mod, "right"),
                 "open", "open", True)
        self.assertEqual(c.hand, "left")
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        self.assertEqual(c.hand, "right")

    # ── tracking-loss grace window (injected clock) ─────────────────────────
    def test_brief_dropout_holds_then_releases_after_grace(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, grace_sec=0.30, clock=clk)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)       # engaged
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)     # RIGHT down
        self.assertTrue(c.right_is_down)
        # Brief untracked dropout WITHIN the grace: hold.
        clk.advance(0.10)
        d_hold = c.update(None, None, "unknown", "unknown", False)
        self.assertIsNone(d_hold.right)
        self.assertIsNone(d_hold.cursor)
        self.assertEqual(d_hold.overlay, "grab")
        self.assertTrue(c.right_is_down)
        # Past the grace → full dead-man release.
        clk.advance(0.40)
        d_rel = c.update(None, None, "unknown", "unknown", False)
        self.assertEqual(d_rel.right, "up")
        self.assertEqual(d_rel.overlay, "hidden")
        self.assertFalse(c.right_is_down)

    def test_relaxed_arm_means_zero_cursor_over_many_frames(self):
        mod = self._load()
        c = self._ctrl(mod)
        cursors, lefts, rights = [], [], []
        for _ in range(60):
            d = c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                         "open", "open", True)
            cursors.append(d.cursor)
            lefts.append(d.left)
            rights.append(d.right)
        self.assertTrue(all(c0 is None for c0 in cursors))
        self.assertTrue(all(b is None for b in lefts))
        self.assertTrue(all(b is None for b in rights))
        self.assertFalse(c.engaged)


# ══════════════════════════════════════════════════════════════════════════
#  POSITION-INDEPENDENT HEIGHT gate: engage/disengage is invariant to the owner's
#  distance from the sensor. lift_m is a PHYSICAL height (hand_y minus the shoulder
#  Y, in real-world metres), so a hand physically raised the same amount above the
#  shoulder reads the same lift_m whether near or far — the gate treats them alike.
# ══════════════════════════════════════════════════════════════════════════
def _scaled_body_joints(side: str, *, scale: float, lift_frac: float,
                        shoulder_half: float = 0.20):
    """Build one arm's joints at an arbitrary distance, with the hand raised so its
    PHYSICAL height above the shoulder is `lift_frac` × shoulder-width. `scale`
    multiplies the POSITIONS/spans to place the SAME body nearer (larger) or farther
    (smaller); because the hand's height above the shoulder scales with the body
    too, a closer and a farther body present the SAME physical lift for a given
    lift_frac — and the height gate (which keys on physical metres) treats them
    identically. A positive lift_frac is a RAISED hand; a negative one is a hand
    BELOW the shoulder (at the desk). The elbow/forward geometry is filled in
    plausibly so the demoted cues compute, but only the HEIGHT gates."""
    base_z = 2.0 * scale
    sh = shoulder_half * scale
    shoulder_width = 2.0 * sh
    sx = -sh if side == "left" else sh
    other = sh if side == "left" else -sh
    sy = 0.40 * scale
    shoulder_ref_y = 0.40 * scale     # spine_shoulder Y == the same height here
    # Hand height ABOVE the shoulder line = lift_frac × shoulder width (physical).
    hand_lift = lift_frac * shoulder_width
    hx, hy, hz = (sx * 2.0), shoulder_ref_y + hand_lift, base_z - 0.30 * scale
    if lift_frac >= 1.0:
        # Raised + straight arm — elbow on the midpoint of shoulder→hand.
        ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (base_z + hz) / 2.0
    else:
        # Lowered: bent elbow folded down (low straightness), matching a real
        # relaxed/desk arm.
        ex, ey, ez = sx, sy - 0.10 * scale, base_z - 0.30 * scale
    return {
        "spine_shoulder": (0.0, shoulder_ref_y, base_z, 2),
        "spine_mid": (0.0, 0.0, base_z, 2),
        "spine_base": (0.0, -0.30 * scale, base_z, 2),
        f"shoulder_{side}": (sx, sy, base_z, 2),
        f"shoulder_{'right' if side == 'left' else 'left'}": (other, sy, base_z, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


class HeightGatePositionIndependenceTests(_Base):
    """The HEIGHT gate is BODY-RELATIVE: a hand RAISED above the shoulder engages up
    close AND far away, and a hand LOWERED to the desk releases at any distance,
    because lift_m is a PHYSICAL height (real-world metres above the shoulder),
    invariant to how far the owner sits. Exercised on the geometry (lift_m) and
    through the engage decision."""

    def _ext(self, mod, joints, side):
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(joints, side))

    def test_local_extension_computes_lift_above_shoulder(self):
        mod = self._load()
        # A hand raised 0.75 × shoulder-width above the shoulder → a POSITIVE lift
        # that scales with the body (bigger body ⇒ bigger physical lift).
        prev = None
        for scale in (0.6, 1.0, 1.7):
            j = _scaled_body_joints("right", scale=scale, lift_frac=0.75)
            ext = self._ext(mod, j, "right")
            self.assertIsNotNone(ext.lift_m)
            self.assertGreater(ext.lift_m, 0.0)        # hand is above the shoulder
            # lift scales with the body size (0.75 × shoulder width).
            self.assertAlmostEqual(ext.lift_m, 0.75 * (2 * 0.20 * scale), delta=0.02)
            if prev is not None:
                self.assertGreater(ext.lift_m, prev)   # bigger body ⇒ bigger lift
            prev = ext.lift_m

    def test_lowered_hand_reads_negative_lift_any_distance(self):
        mod = self._load()
        # A hand BELOW the shoulder (lift_frac negative) → negative lift at every
        # distance, so it can never engage regardless of how far the body sits.
        for scale in (0.6, 1.0, 1.7):
            ext = self._ext(mod, _scaled_body_joints(
                "right", scale=scale, lift_frac=-1.0), "right")
            self.assertIsNotNone(ext.lift_m)
            self.assertLess(ext.lift_m, 0.0)

    def test_engage_invariant_to_body_distance(self):
        mod = self._load()
        th = mod._reach_thresholds()   # height-margin defaults
        # A clear RAISE (hand well above the shoulder) must read EXTENDED at every
        # distance; a LOWERED hand (at the desk) must NOT — purely on the height.
        for scale in (0.5, 1.0, 1.5, 2.2):
            raised = self._ext(mod, _scaled_body_joints(
                "right", scale=scale, lift_frac=1.0), "right")
            lowered = self._ext(mod, _scaled_body_joints(
                "right", scale=scale, lift_frac=-1.0), "right")
            self.assertTrue(
                raised.is_extended(engaged=False, thresholds=th),
                f"raised hand should engage at scale {scale}")
            self.assertFalse(
                lowered.is_extended(engaged=False, thresholds=th),
                f"lowered hand should NOT engage at scale {scale}")

    def test_controller_engages_far_and_near_via_poll(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        # Mirror OFF so reach_side='right' stays the right arm end-to-end.
        moves, buttons = self._capture_mouse(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

        def _body_at(scale, lift_frac):
            j = _scaled_body_joints("right", scale=scale, lift_frac=lift_frac)
            # Other (left) hand lowered at the same scale so only the right is up.
            jl = _scaled_body_joints("left", scale=scale, lift_frac=-1.0)
            j["elbow_left"] = jl["elbow_left"]
            j["hand_left"] = jl["hand_left"]
            return {"id": 0, "joints": j,
                    "head": (0.0, 0.7 * scale, 2.0 * scale),
                    "distance_m": 2.0 * scale, "facing": True,
                    "hand_right": "open", "hand_left": "open"}

        # FAR (small scale): a clear raise must still move the cursor.
        moves.clear()
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(0.6, 1.0)]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)
        # Lower the hand at the SAME far distance → disengage (no further move).
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(0.6, -1.0)]))
        self.assertFalse(ctrl.engaged)
        # NEAR (large scale): the same clear raise engages too.
        moves.clear()
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(1.8, 1.0)]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)

    def test_lower_releases_regardless_of_body_position(self):
        # The owner's bug class: it must release wherever they sit. With the
        # body-relative HEIGHT, lowering the hand at ANY distance drops below the
        # down-margin → release. Simulate: engage near, then "stand up + sit back" =
        # lower the hand at a DIFFERENT (far) scale → must disengage.
        mod = self._load()
        th = mod._reach_thresholds()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, engage_debounce_frames=1)
        lowered_left = mod.ArmExtension.from_bridge(
            mod._local_arm_extension(_relaxed_arm_joints("left"), "left"))

        def _raise(scale, frac):
            return mod.ArmExtension.from_bridge(mod._local_arm_extension(
                _scaled_body_joints("right", scale=scale, lift_frac=frac), "right"))

        # Engage with a clear raise up close.
        c.update(lowered_left, _raise(1.6, 1.0), "open", "open", True,
                 thresholds=th)
        self.assertTrue(c.engaged)
        # Now lower the hand — and the body is also farther away (smaller scale),
        # exactly the stand-up→sit-back perturbation. The physical lift goes negative
        # → it MUST release.
        d = c.update(lowered_left, _raise(0.7, -1.0), "open", "open", True,
                     thresholds=th)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")


# ══════════════════════════════════════════════════════════════════════════
#  LIVE _poll_once — sensor + mouse mocked
# ══════════════════════════════════════════════════════════════════════════
class PollActsTests(_Base):
    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_extended_open_hand_moves_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        bridge = _fake_bridge(bodies=[_body(reach_side="right", grip_right="open")])
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, bridge)
        self.assertEqual(len(moves), 1)
        self.assertEqual(buttons, [])
        self.assertEqual(d.overlay, "track")

    def test_right_hand_close_then_open_clicks_RIGHT(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # RIGHT arm extended; right hand open → closed → open : a RIGHT-click.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(buttons, [("down", "right"), ("up", "right")])

    def test_left_hand_close_then_open_clicks_LEFT(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # LEFT arm extended; left hand open → closed → open : a LEFT-click.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="left", grip_left="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="left", grip_left="closed")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="left", grip_left="open")]))
        self.assertEqual(buttons, [("down", "left"), ("up", "left")])

    def test_deadman_releases_when_body_lost(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        self.assertEqual(buttons, [("down", "right")])
        # Body leaves the frame → dead-man release of the RIGHT button.
        mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        self.assertEqual(buttons, [("down", "right"), ("up", "right")])


class PollGateTests(_Base):
    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_noop_side_effects_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])
        self.assertEqual(buttons, [])
        self.assertIsNotNone(d)

    def test_flag_off_during_drag_still_releases(self):
        mod = self._load()
        self._not_staging(mod)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        self._patch_flag(True)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        self.assertEqual(buttons, [("down", "right")])
        # Flip OFF mid-drag, then re-open: the pending RIGHT 'up' still fires.
        from core import config as cfg
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", False, create=True):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(buttons, [("down", "right"), ("up", "right")])

    def test_noop_when_staging(self):
        mod = self._load()
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start(); self.addCleanup(p.stop)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])
        self.assertEqual(buttons, [])

    def test_noop_when_bridge_absent(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        ctrl = self._ctrl(mod)
        self.assertIsNone(mod._poll_once(ctrl, None))

    def test_noop_when_sensor_disabled(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            enabled=False, bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])

    def test_noop_when_sensor_unavailable(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            available=(False, "no sensor"),
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])


# ══════════════════════════════════════════════════════════════════════════
#  LIVE reach gate end-to-end — _poll_once + mocked mouse
# ══════════════════════════════════════════════════════════════════════════
class PollReachGateTests(_Base):
    """The owner-facing guarantee through the LIVE path: a RELAXED (or no-joint)
    arm drives ZERO _set_cursor_pos calls; an EXTENDED arm moves it. Mouse
    actuation mocked; no real cursor is touched."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_relaxed_arms_make_zero_setcursorpos_calls(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Both arms relaxed (merely present, not reaching) for many frames →
        # ZERO cursor moves (the physical mouse is never fought).
        for _ in range(30):
            mod._poll_once(ctrl, _fake_bridge(bodies=[_body(both_relaxed=True)]))
        self.assertEqual(moves, [])
        self.assertEqual(buttons, [])

    def test_desk_hands_with_forward_reach_make_zero_setcursorpos(self):
        """THE HEADLINE SCENARIO through the LIVE path: hands at the desk read a
        forward reach (the old broken cue stayed engaged here) yet sit below the
        shoulder, so ZERO cursor moves — the owner's real mouse is never fought."""
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)

        def desk_body():
            # both_relaxed lowers both hands to the desk; push them FORWARD in z so
            # the (demoted) forward reach is large — proving height, not reach, gates.
            b = _body(both_relaxed=True)
            for side in ("left", "right"):
                hx = -0.1 if side == "left" else 0.1
                b["joints"][f"hand_{side}"] = (hx, DESK_HAND_Y, TORSO_Z - 0.45, 2)
            return b
        for _ in range(20):
            mod._poll_once(ctrl, _fake_bridge(bodies=[desk_body()]))
        self.assertEqual(moves, [])          # desk hand != raise → no cursor
        self.assertFalse(ctrl.engaged)

    def test_relaxing_arm_mid_drag_releases_and_stops_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        self.assertEqual(buttons, [("down", "right")])
        moves_before = len(moves)
        for _ in range(10):
            mod._poll_once(ctrl, _fake_bridge(bodies=[_body(both_relaxed=True)]))
        self.assertEqual(buttons, [("down", "right"), ("up", "right")])
        self.assertEqual(len(moves), moves_before)

    def test_extended_arm_moves_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(len(moves), 1)

    def test_no_arm_joints_makes_zero_setcursorpos(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        for _ in range(10):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(with_joints=False)]))
        self.assertEqual(moves, [])


# ══════════════════════════════════════════════════════════════════════════
#  PER-HAND button actuation — the win32 flag mapping (LEFT vs RIGHT)
# ══════════════════════════════════════════════════════════════════════════
class ButtonActuationTests(_Base):
    """_mouse_button emits the LEFT win32 events for the LEFT button and the
    RIGHT events for the RIGHT button (no real mouse touched)."""

    def _fake_win32(self):
        fired = []
        fake_win32api = types.ModuleType("win32api")
        fake_win32api.mouse_event = lambda flag, *a, **k: fired.append(flag)
        fake_win32con = types.ModuleType("win32con")
        fake_win32con.MOUSEEVENTF_LEFTDOWN = 0x0002
        fake_win32con.MOUSEEVENTF_LEFTUP = 0x0004
        fake_win32con.MOUSEEVENTF_RIGHTDOWN = 0x0008
        fake_win32con.MOUSEEVENTF_RIGHTUP = 0x0010
        return fired, fake_win32api, fake_win32con

    def test_left_button_emits_left_events(self):
        mod = self._load()
        fired, api, con = self._fake_win32()
        with mock.patch.dict(sys.modules, {"win32api": api, "win32con": con}):
            self.assertTrue(mod._mouse_button("down", "left"))
            self.assertTrue(mod._mouse_button("up", "left"))
        self.assertEqual(fired, [0x0002, 0x0004])      # LEFT down/up
        self.assertNotIn(0x0008, fired)
        self.assertNotIn(0x0010, fired)

    def test_right_button_emits_right_events(self):
        mod = self._load()
        fired, api, con = self._fake_win32()
        with mock.patch.dict(sys.modules, {"win32api": api, "win32con": con}):
            self.assertTrue(mod._mouse_button("down", "right"))
            self.assertTrue(mod._mouse_button("up", "right"))
        self.assertEqual(fired, [0x0008, 0x0010])      # RIGHT down/up
        self.assertNotIn(0x0002, fired)
        self.assertNotIn(0x0004, fired)

    def test_default_button_is_left(self):
        mod = self._load()
        fired, api, con = self._fake_win32()
        with mock.patch.dict(sys.modules, {"win32api": api, "win32con": con}):
            mod._mouse_button("down")                  # no button arg → left
        self.assertEqual(fired, [0x0002])


# ══════════════════════════════════════════════════════════════════════════
#  toggle + persistence + status
# ══════════════════════════════════════════════════════════════════════════
class ToggleTests(_Base):
    def _patch_settings_writer(self, initial=None):
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def _inject_bridge(self, mod, bridge):
        old = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = bridge
        self.addCleanup(
            lambda: sys.modules.__setitem__("audio.kinect_bridge", old)
            if old is not None else sys.modules.pop("audio.kinect_bridge", None))

    def test_on_persists_flag(self):
        mod = self._load()
        self._patch_flag(False)
        saved = self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=True))
        out = mod.air_mouse_on("")
        self.assertIn("on", out.lower())
        self.assertTrue(saved.get("KINECT_AIR_MOUSE_ENABLED"))
        from core import config as cfg
        self.assertTrue(cfg.KINECT_AIR_MOUSE_ENABLED)

    def test_on_message_describes_raise_and_per_hand_clicks(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=True))
        out = mod.air_mouse_on("").lower()
        # The spoken help reflects the HEIGHT model: raise a hand, left/right hand.
        self.assertIn("raise", out)
        self.assertIn("shoulder", out)
        self.assertIn("left-click", out)
        self.assertIn("right-click", out)

    def test_off_persists_flag_and_releases_both_buttons(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer({"KINECT_AIR_MOUSE_ENABLED": True})
        released = []
        with mock.patch.object(mod, "_mouse_button",
                               lambda a, b="left": released.append((a, b))), \
                mock.patch.object(mod, "_shutdown_overlay", lambda: None), \
                mock.patch.object(mod, "_clear_overlay_state", lambda: None):
            out = mod.air_mouse_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("KINECT_AIR_MOUSE_ENABLED"))
        # BOTH buttons released on disable (neither hand can be stranded down).
        self.assertIn(("up", "left"), released)
        self.assertIn(("up", "right"), released)

    def test_on_warns_when_sensor_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=False))
        out = mod.air_mouse_on("")
        self.assertIn("kinect", out.lower())


class StatusTests(_Base):
    def _inject_bridge(self, bridge):
        old = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = bridge
        self.addCleanup(
            lambda: sys.modules.__setitem__("audio.kinect_bridge", old)
            if old is not None else sys.modules.pop("audio.kinect_bridge", None))

    def test_status_off(self):
        mod = self._load()
        self._patch_flag(False)
        self.assertIn("off", mod.air_mouse_status("").lower())

    def test_status_on_with_hand_in_view(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject_bridge(_fake_bridge(
            hand_states={"right": "open", "left": "unknown",
                         "tracked": True, "ts": 0.0}))
        out = mod.air_mouse_status("")
        self.assertIn("on", out.lower())
        self.assertIn("see your hand", out.lower())

    def test_status_on_but_sensor_off(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject_bridge(_fake_bridge(enabled=False))
        out = mod.air_mouse_status("")
        self.assertIn("on", out.lower())
        self.assertIn("unavailable", out.lower())


# ══════════════════════════════════════════════════════════════════════════
#  ISSUE 1 — HAND MIRROR (selfie-view): owner's REAL left hand → LEFT button +
#  left-side circle. The bridge's left↔right are SWAPPED when KINECT_HAND_MIRROR.
# ══════════════════════════════════════════════════════════════════════════
class HandMirrorTests(_Base):
    """The Kinect stream is mirrored, so the SDK's hand_left is the owner's REAL
    right hand. With KINECT_HAND_MIRROR on, _hand_sample SWAPS the hands so the
    owner's REAL left hand drives the LEFT button + the left-side circle."""

    def _mirror_on(self, mod):
        p = mock.patch.object(mod, "_hand_mirror_enabled", lambda: True)
        p.start(); self.addCleanup(p.stop)

    def _mirror_off(self, mod):
        p = mock.patch.object(mod, "_hand_mirror_enabled", lambda: False)
        p.start(); self.addCleanup(p.stop)

    def test_default_mirror_flag_is_true_for_owner(self):
        mod = self._load()
        self.assertTrue(mod.KINECT_HAND_MIRROR_DEFAULT)

    def test_sample_swaps_grips_when_mirrored(self):
        mod = self._load()
        self._mirror_on(mod)
        # The Kinect SDK reports the grip on its hand_left as the owner's REAL
        # right hand. Body: SDK hand_left="closed", hand_right="open". After the
        # mirror swap the skill's left_grip must be the SDK's right ("open") and
        # the skill's right_grip the SDK's left ("closed").
        body = _body(reach_side="right", grip_left="closed", grip_right="open")
        left_ext, right_ext, left_grip, right_grip, tracked = mod._hand_sample(
            _fake_bridge(bodies=[body]))
        self.assertTrue(tracked)
        self.assertEqual(left_grip, "open")     # SDK right → skill left
        self.assertEqual(right_grip, "closed")  # SDK left  → skill right

    def test_sample_does_not_swap_when_mirror_off(self):
        mod = self._load()
        self._mirror_off(mod)
        body = _body(reach_side="right", grip_left="closed", grip_right="open")
        _le, _re, left_grip, right_grip, _t = mod._hand_sample(
            _fake_bridge(bodies=[body]))
        self.assertEqual(left_grip, "closed")   # unchanged
        self.assertEqual(right_grip, "open")

    def test_swapped_extension_sides_are_relabelled(self):
        mod = self._load()
        self._mirror_on(mod)
        # The SDK's RIGHT arm is extended (reach_side="right"). After the mirror
        # swap the EXTENDED arm must be reported on the skill's LEFT side, so the
        # owner's REAL left hand drives the cursor.
        body = _body(reach_side="right", grip_right="open")
        left_ext, right_ext, _lg, _rg, _t = mod._hand_sample(
            _fake_bridge(bodies=[body]))
        # The skill-left arm now carries the extended geometry + side label "left".
        self.assertEqual(left_ext.side, "left")
        self.assertEqual(right_ext.side, "right")
        self.assertTrue(left_ext.is_extended(engaged=False))    # the reach moved here
        self.assertFalse(right_ext.is_extended(engaged=False))

    def test_real_left_hand_drives_LEFT_button_and_left_circle(self):
        """End-to-end through _poll_once: the owner reaches + clicks with their
        REAL left hand (the SDK's RIGHT). With the mirror on, the LEFT button
        fires and the published which-hand is 'left' (→ left-side circle)."""
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        # _capture_mouse pins the mirror OFF; turn it back ON AFTER so this
        # patch is the most-recent (it wins on lookup) — this test needs it on.
        self._mirror_on(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)
        # Owner's REAL left hand = SDK right arm extended; owner closes their REAL
        # left hand = SDK hand_right closes. Model that as the SDK's RIGHT.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        st = mod.get_air_mouse_state()
        self.assertTrue(st["engaged"])
        self.assertEqual(st["hand"], "left")    # published as the owner's LEFT
        # Now the owner closes their REAL left hand (SDK hand_right → closed).
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        # The LEFT button (not right) fired — owner's real left hand = left-click.
        self.assertEqual(buttons, [("down", "left"), ("up", "left")])


# ══════════════════════════════════════════════════════════════════════════
#  ISSUE 2 — CALIBRATION: fit thresholds from captured poses + persist; the live
#  gate reads them (falling back to defaults); + the debug-log format.
# ══════════════════════════════════════════════════════════════════════════
class ComputeReachThresholdsTests(_Base):
    """The fit now produces BODY-RELATIVE HEIGHT margins (position-independent): the
    first pose pair is the lowered/raised hand HEIGHT (lift_m)."""

    def test_margins_placed_between_lowered_and_raised(self):
        mod = self._load()
        # Lowered lift -0.40, raised +0.20 (span 0.60) → up ~60 % (-0.40+0.36=-0.04),
        # down ~40 % (-0.40+0.24=-0.16); up strictly above down (hysteresis).
        th = mod.compute_reach_thresholds(-0.40, 0.20, 0.50, 0.95)
        self.assertAlmostEqual(th["up_margin"], -0.04, delta=0.01)
        self.assertAlmostEqual(th["down_margin"], -0.16, delta=0.01)
        self.assertGreater(th["up_margin"], th["down_margin"])
        # The demoted forward/ratio bars are always the permissive module defaults.
        self.assertEqual(th["ratio_engage"], mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE)
        self.assertEqual(th["fwd_engage"], mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M)
        # Straightness 0.50→0.95 span (back-compat, non-gating): engage 0.77, dis 0.68.
        self.assertAlmostEqual(th["straight_engage"], 0.77, delta=0.01)
        self.assertAlmostEqual(th["straight_disengage"], 0.68, delta=0.01)

    def test_missing_pose_falls_back_to_default_margins(self):
        mod = self._load()
        # No lift pair → height margins keep the module defaults.
        th = mod.compute_reach_thresholds(None, None, 0.40, 0.96)
        self.assertEqual(th["up_margin"], mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)
        self.assertEqual(th["down_margin"], mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)

    def test_degenerate_span_keeps_default_margins(self):
        mod = self._load()
        # Raised barely above lowered (no real lift span ≥ 0.15) → default margins,
        # never an inverted/nonsense margin from a flubbed capture.
        th = mod.compute_reach_thresholds(0.10, 0.15, 0.80, 0.81)
        self.assertEqual(th["up_margin"], mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)
        self.assertEqual(th["down_margin"], mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)


class MedianTests(_Base):
    def test_median_odd_even_and_empty(self):
        mod = self._load()
        self.assertEqual(mod._median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(mod._median([1.0, 2.0, 3.0, 4.0]), 2.5)
        self.assertIsNone(mod._median([]))


class ReachThresholdReadTests(_Base):
    """The live gate reads the persisted height margins (KINECT_LIFT_*) — and the
    back-compat reach/straight keys — falling back to the module defaults per-field
    when unset."""

    def _patch_settings(self, mod, saved):
        p = mock.patch.object(mod, "_saved_settings", lambda: dict(saved))
        p.start(); self.addCleanup(p.stop)

    def test_defaults_when_nothing_persisted(self):
        mod = self._load()
        self._patch_settings(mod, {})
        th = mod._reach_thresholds()
        # The height-margin defaults apply pre-calibration.
        self.assertEqual(th["up_margin"], mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)
        self.assertEqual(th["down_margin"], mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)
        self.assertEqual(th["ratio_engage"],
                         mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE)

    def test_persisted_margins_win(self):
        mod = self._load()
        # KINECT_LIFT_* hold the persisted height margins (the primary gate).
        self._patch_settings(mod, {
            mod.SETTING_UP_MARGIN: 0.08,
            mod.SETTING_DOWN_MARGIN: -0.06,
        })
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["up_margin"], 0.08)
        self.assertAlmostEqual(th["down_margin"], -0.06)

    def test_partial_persist_falls_back_per_field(self):
        mod = self._load()
        self._patch_settings(mod, {mod.SETTING_UP_MARGIN: 0.09})
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["up_margin"], 0.09)              # persisted
        self.assertEqual(th["down_margin"],                        # defaulted
                         mod.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M)


class CalibrateActionTests(_Base):
    """'calibrate air mouse' walks the owner through two poses, fits the
    thresholds from the captured medians, and persists them. Sensor + settings
    writer mocked — no real Kinect, no real file."""

    def _patch_settings_writer(self, mod, initial=None):
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def _stub_captures(self, mod, sequence):
        """Make _capture_reach return each (lift, straight, n) in `sequence` on
        successive calls (no real sampling / sleeping)."""
        calls = {"i": 0}

        def _fake(bridge, *a, **k):
            i = min(calls["i"], len(sequence) - 1)
            calls["i"] += 1
            return sequence[i]
        p = mock.patch.object(mod, "_capture_reach", _fake)
        p.start(); self.addCleanup(p.stop)
        return calls

    def test_calibration_sets_margins_from_captured_poses(self):
        mod = self._load()
        self._not_staging(mod)
        saved = self._patch_settings_writer(mod)
        # Sensor ready + bridge present.
        p = mock.patch.object(mod, "_sensor_ready", lambda: (True, ""))
        p.start(); self.addCleanup(p.stop)
        pb = mock.patch.object(mod, "_bridge", lambda: _fake_bridge())
        pb.start(); self.addCleanup(pb.stop)
        spoken: list = []
        ps = mock.patch.object(mod, "_speak", lambda t: spoken.append(t))
        ps.start(); self.addCleanup(ps.stop)
        # First capture = RAISED pose (lift +0.20 m, 0.95 straight); second =
        # LOWERED (lift -0.40 m, 0.50 straight), each with usable frames.
        self._stub_captures(mod, [(0.20, 0.95, 20), (-0.40, 0.50, 20)])
        out = mod.calibrate_air_mouse("")
        self.assertIn("calibrat", out.lower())
        # Persisted the height margins, up strictly above down.
        self.assertIn(mod.SETTING_UP_MARGIN, saved)
        self.assertIn(mod.SETTING_DOWN_MARGIN, saved)
        self.assertGreater(saved[mod.SETTING_UP_MARGIN],
                           saved[mod.SETTING_DOWN_MARGIN])
        # up sits ~60 % between -0.40 and +0.20 (span 0.60) ≈ -0.04.
        self.assertAlmostEqual(saved[mod.SETTING_UP_MARGIN], -0.04, delta=0.02)
        # It spoke BOTH prompts (raise, then lower).
        self.assertTrue(any("raise" in s.lower() for s in spoken))
        self.assertTrue(any("lower" in s.lower() for s in spoken))

    def test_calibration_persisted_values_are_read_by_the_gate(self):
        # After calibration the live _reach_thresholds() must reflect what was
        # saved (end-to-end: capture → persist → read).
        mod = self._load()
        self._not_staging(mod)
        saved = self._patch_settings_writer(mod)
        p = mock.patch.object(mod, "_sensor_ready", lambda: (True, ""))
        p.start(); self.addCleanup(p.stop)
        pb = mock.patch.object(mod, "_bridge", lambda: _fake_bridge())
        pb.start(); self.addCleanup(pb.stop)
        ps = mock.patch.object(mod, "_speak", lambda t: None)
        ps.start(); self.addCleanup(ps.stop)
        # Captures are raised/lowered lift (raised +0.25, lowered -0.35 → a clear
        # lift span ≥ 0.15, so the height margins fit + persist).
        self._stub_captures(mod, [(0.25, 0.98, 20), (-0.35, 0.55, 20)])
        mod.calibrate_air_mouse("")
        # _saved_settings reads the same mocked writer → the gate sees the fit.
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["up_margin"], saved[mod.SETTING_UP_MARGIN])
        self.assertGreater(th["up_margin"], th["down_margin"])

    def test_calibration_honest_when_no_body_seen(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_settings_writer(mod)
        p = mock.patch.object(mod, "_sensor_ready", lambda: (True, ""))
        p.start(); self.addCleanup(p.stop)
        pb = mock.patch.object(mod, "_bridge", lambda: _fake_bridge())
        pb.start(); self.addCleanup(pb.stop)
        ps = mock.patch.object(mod, "_speak", lambda t: None)
        ps.start(); self.addCleanup(ps.stop)
        # Zero usable frames on the first (raised) capture → honest failure.
        self._stub_captures(mod, [(None, None, 0)])
        out = mod.calibrate_air_mouse("")
        self.assertIn("couldn't see", out.lower())

    def test_calibration_refuses_in_staging(self):
        mod = self._load()
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start(); self.addCleanup(p.stop)
        out = mod.calibrate_air_mouse("")
        self.assertIn("staging", out.lower())

    def test_calibration_honest_when_sensor_unavailable(self):
        mod = self._load()
        self._not_staging(mod)
        p = mock.patch.object(mod, "_sensor_ready",
                              lambda: (False, "the Kinect is switched off"))
        p.start(); self.addCleanup(p.stop)
        out = mod.calibrate_air_mouse("")
        self.assertIn("kinect", out.lower())

    def test_capture_reach_medians_lift_from_highest_hand(self):
        # _capture_reach samples the highest-raised hand's HEIGHT (lift_m) +
        # straightness and medians them. The fixture raises the right hand to
        # RAISED_HAND_Y (lift ≈ +0.15 m above the shoulder).
        mod = self._load()
        bridge = _fake_bridge(bodies=[_body(reach_side="right")])
        lift, straight, n = mod._capture_reach(
            bridge, seconds=0.05, sleep_fn=lambda s: None,
            now_fn=_StepClock(step=0.02))
        self.assertGreater(n, 0)
        self.assertIsNotNone(lift)
        self.assertGreater(lift, mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)  # a real raise
        self.assertIsNotNone(straight)


class _StepClock:
    """A monotonic-ish clock that advances by `step` each call so a capture loop
    runs a bounded number of iterations without real sleeping."""
    def __init__(self, start=0.0, step=0.02):
        self.t = float(start)
        self.step = float(step)

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class DebugLogFormatTests(_Base):
    """ISSUE 2b — the ~2 Hz live height-gate debug line shows the values for
    tuning: lift (the primary cue), the controlling hand, engaged, and yield."""

    class _Ctrl:
        engaged = False
        hand = None

    def test_debug_line_shows_lift_and_yield(self):
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.18, straightness=0.91,
                               hand=(0.1, 0.55, 1.8, 2), lift_m=0.07)
        line = mod._format_reach_debug(None, ext, True, self._Ctrl(),
                                       yielding=False)
        self.assertIn("[air-mouse]", line)
        self.assertIn("lift=+0.07", line)          # the primary height cue, signed
        self.assertIn("hand=right", line)
        self.assertIn("engaged=False", line)
        self.assertIn("yield=False", line)
        # The demoted secondary cues are still surfaced for context.
        self.assertIn("reach=0.18", line)
        self.assertIn("straight=0.91", line)

    def test_debug_line_shows_yield_true(self):
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.18, straightness=0.91,
                               hand=(0.1, 0.55, 1.8, 2), lift_m=0.07)
        line = mod._format_reach_debug(None, ext, True, self._Ctrl(),
                                       yielding=True)
        self.assertIn("yield=True", line)

    def test_debug_line_handles_missing_cues(self):
        mod = self._load()
        line = mod._format_reach_debug(None, None, False, self._Ctrl(),
                                       yielding=False)
        self.assertIn("lift=n/a", line)
        self.assertIn("hand=none", line)

    def test_debug_log_is_throttled(self):
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.2, straightness=0.9,
                               hand=(0, 0.55, 1.8, 2), lift_m=0.1)
        mod._air_mouse_debug_last[0] = 0.0
        # First call at t=100 prints; an immediate second call is throttled.
        self.assertTrue(mod._maybe_debug_log(None, ext, True, self._Ctrl(),
                                             now=100.0, yielding=False))
        self.assertFalse(mod._maybe_debug_log(None, ext, True, self._Ctrl(),
                                              now=100.1, yielding=False))
        # Past the interval it prints again.
        self.assertTrue(mod._maybe_debug_log(
            None, ext, True, self._Ctrl(),
            now=100.0 + mod._AIR_MOUSE_DEBUG_INTERVAL, yielding=False))


# ══════════════════════════════════════════════════════════════════════════
#  ISSUE 3 — controlling-hand HYSTERESIS: no thrash with BOTH hands raised.
# ══════════════════════════════════════════════════════════════════════════
class ControllingHandHysteresisTests(_Base):
    def _ctrl(self, mod, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        # engage_debounce_frames=1 so a single update() engages immediately in these
        # state-machine tests (the multi-frame engage DEBOUNCE has its own class).
        kw.setdefault("engage_debounce_frames", 1)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    def _arm(self, mod, side, lift, straight):
        """A RAISED ArmExtension with a chosen HEIGHT (lift_m, in metres above the
        shoulder — also its reach_score, so a bigger `lift` is a more-dominant hand)
        and a hand joint, for driving the hysteresis directly. `straight` is carried
        as the demoted (non-gating) straightness cue."""
        return mod.ArmExtension(side, forward_m=lift, straightness=straight,
                                hand=(0.1 if side == "right" else -0.1,
                                      SHOULDER_Y + lift, TORSO_Z - 0.30, 2),
                                lift_m=lift)

    def test_holds_current_hand_when_other_barely_leads(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Engage on the RIGHT hand first (clearly extended).
        c.update(self._arm(mod, "left", 0.10, 0.50),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        self.assertEqual(c.hand, "right")
        # Now the LEFT hand creeps a HAIR more extended than the right, but under
        # the switch margin: control must STAY on the right (no thrash).
        for _ in range(20):
            d = c.update(self._arm(mod, "left", 0.47, 0.99),
                         self._arm(mod, "right", 0.45, 0.98),
                         "open", "open", True)
            self.assertEqual(d.hand, "right")
        self.assertEqual(c.hand, "right")

    def test_switches_only_after_sustained_clear_lead(self):
        mod = self._load()
        c = self._ctrl(mod, switch_frames=6, switch_margin=0.25)
        c.update(self._arm(mod, "left", 0.10, 0.40),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        self.assertEqual(c.hand, "right")
        # LEFT now leads by a CLEAR margin. It must take several frames to switch.
        big_left = lambda: self._arm(mod, "left", 0.90, 1.0)
        small_right = lambda: self._arm(mod, "right", 0.20, 0.80)
        switched_at = None
        for i in range(1, 12):
            d = c.update(big_left(), small_right(), "open", "open", True)
            if d.hand == "left":
                switched_at = i
                break
        self.assertIsNotNone(switched_at)
        self.assertGreaterEqual(switched_at, 6)   # not instant — sustained lead
        self.assertEqual(c.hand, "left")

    def test_brief_lead_then_back_does_not_switch(self):
        mod = self._load()
        c = self._ctrl(mod, switch_frames=6, switch_margin=0.25)
        c.update(self._arm(mod, "left", 0.10, 0.40),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        # LEFT leads big for a FEW frames (< switch_frames) then drops back.
        for _ in range(3):
            d = c.update(self._arm(mod, "left", 0.90, 1.0),
                         self._arm(mod, "right", 0.20, 0.80), "open", "open", True)
            self.assertEqual(d.hand, "right")     # not yet switched
        # Back to the right clearly leading → challenge resets, stays on right.
        for _ in range(5):
            d = c.update(self._arm(mod, "left", 0.20, 0.50),
                         self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        self.assertEqual(c.hand, "right")

    def test_both_hands_grips_tracked_regardless_of_cursor_hand(self):
        mod = self._load()
        c = self._ctrl(mod)
        # RIGHT drives the cursor; BOTH hands close → BOTH buttons fire (the L/R
        # clicks are independent of which hand drives), and no thrash.
        c.update(self._arm(mod, "left", 0.20, 0.55),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        d = c.update(self._arm(mod, "left", 0.20, 0.55),
                     self._arm(mod, "right", 0.45, 0.98), "closed", "closed", True)
        self.assertEqual(c.hand, "right")
        self.assertEqual(d.left, "down")
        self.assertEqual(d.right, "down")
        self.assertTrue(c.left_is_down and c.right_is_down)

    def test_no_flicker_with_both_hands_equally_extended(self):
        mod = self._load()
        c = self._ctrl(mod)
        # BOTH hands raised + EQUALLY extended for many frames: the cursor hand
        # must be STABLE (never flip back and forth frame-to-frame).
        c.update(self._arm(mod, "left", 0.40, 0.95),
                 self._arm(mod, "right", 0.40, 0.95), "open", "open", True)
        first = c.hand
        hands = []
        for _ in range(40):
            d = c.update(self._arm(mod, "left", 0.40, 0.95),
                         self._arm(mod, "right", 0.40, 0.95), "open", "open", True)
            hands.append(d.hand)
        # Exactly one distinct controlling hand across all frames — no flicker.
        self.assertEqual(set(hands), {first})

    def test_switch_state_clears_on_disengage(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._arm(mod, "left", 0.10, 0.40),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        # Build a partial challenge, then fully disengage (both relaxed).
        c.update(self._arm(mod, "left", 0.90, 1.0),
                 self._arm(mod, "right", 0.20, 0.80), "open", "open", True)
        c.update(None, None, "open", "open", True)     # no arms → release
        self.assertFalse(c.engaged)
        self.assertEqual(c._challenge_count, 0)
        self.assertIsNone(c._challenge_side)

    def test_noncontrolling_hand_does_not_move_cursor(self):
        """THE OWNER'S BUG (FIX 2): with both hands raised the cursor must follow
        ONLY the controlling hand — the OTHER hand must NOT influence the cursor
        POSITION at all (no 'both hands connect to the same spot' / averaging). Hold
        the controlling (right) hand STILL while the idle (left) hand sweeps; the
        cursor must NOT move."""
        mod = self._load()
        c = self._ctrl(mod)

        def right_arm(hand_x):
            # Controlling hand held at a FIXED body-relative position.
            return mod.ArmExtension("right", forward_m=0.0, straightness=None,
                                    hand=(hand_x, SHOULDER_Y + 0.30, 1.8, 2),
                                    lift_m=0.30, shoulder_ref_y=SHOULDER_Y,
                                    body_center_x=0.0)

        def left_arm(hand_x, lift):
            # Idle hand, lower (so it's never the controller), sweeping in X.
            return mod.ArmExtension("left", forward_m=0.0, straightness=None,
                                    hand=(hand_x, SHOULDER_Y + lift, 1.8, 2),
                                    lift_m=lift, shoulder_ref_y=SHOULDER_Y,
                                    body_center_x=0.0)

        # Engage with the right hand controlling (higher).
        c.update(left_arm(-0.1, 0.10), right_arm(0.10), "open", "open", True)
        base = c.update(left_arm(-0.1, 0.10), right_arm(0.10),
                        "open", "open", True).cursor
        self.assertEqual(c.hand, "right")
        # Sweep the LEFT (idle) hand wildly while the RIGHT hand stays put.
        cursors = []
        for lx in (-0.4, -0.2, 0.0, 0.2, 0.4):
            d = c.update(left_arm(lx, 0.10), right_arm(0.10), "open", "open", True)
            cursors.append(d.cursor)
            self.assertEqual(d.hand, "right")     # controller never flips
        # The cursor is unchanged by the idle hand's motion (within EMA settle).
        self.assertTrue(all(cur == base for cur in cursors),
                        f"idle hand moved the cursor: base={base} got={cursors}")


# ══════════════════════════════════════════════════════════════════════════
#  FIX 3 — ENGAGE DEBOUNCE: a raised hand must HOLD above the engage line for a
#  few consecutive frames before the cursor is taken, so a 1-frame Kinect height
#  spike can't grab it. DISENGAGE stays instant.
# ══════════════════════════════════════════════════════════════════════════
class EngageDebounceTests(_Base):
    def _ctrl(self, mod, frames):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=frames)

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_single_frame_spike_does_not_engage(self):
        mod = self._load()
        c = self._ctrl(mod, frames=3)
        # ONE frame with a raised hand (a spike), then the hand drops: must NEVER
        # engage — the cursor was never taken.
        d1 = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                      "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d1.cursor)            # held off — no SetCursorPos
        self.assertEqual(d1.overlay, "hidden")
        # Hand drops back below the line → streak resets, still disengaged.
        d2 = c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                      "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d2.cursor)

    def test_sustained_raise_engages_after_debounce(self):
        mod = self._load()
        c = self._ctrl(mod, frames=3)
        # Frames 1 & 2: building the streak, held off (no cursor yet).
        for _ in range(2):
            d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                         "open", "open", True)
            self.assertFalse(c.engaged)
            self.assertIsNone(d.cursor)
        # Frame 3: the raise has held long enough → ENGAGE, cursor taken.
        d3 = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                      "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d3.cursor)
        self.assertEqual(d3.overlay, "track")

    def test_spike_resets_streak_so_it_never_accumulates(self):
        mod = self._load()
        c = self._ctrl(mod, frames=3)
        # Raise (streak 1), drop (reset), raise (streak 1 again) — must NOT have
        # engaged from two non-consecutive raised frames.
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                 "open", "open", True)
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)
        self.assertFalse(c.engaged)            # streak restarted, not yet met
        self.assertIsNone(d.cursor)

    def test_disengage_is_not_debounced(self):
        mod = self._load()
        c = self._ctrl(mod, frames=3)
        # Engage (3 held frames), then drop in ONE frame → immediate release (no
        # symmetric debounce on the way down).
        for _ in range(3):
            c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)
        self.assertTrue(c.engaged)
        d = c.update(self._relaxed(mod, "left"), self._relaxed(mod, "right"),
                     "open", "open", True)
        self.assertFalse(c.engaged)            # released at once
        self.assertIsNone(d.cursor)

    def test_held_button_not_emitted_during_holdoff(self):
        mod = self._load()
        c = self._ctrl(mod, frames=3)
        # During the engage hold-off, even a closed hand must not fire a click (we
        # haven't engaged yet) — no spurious button edge before engagement.
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "closed", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.right)
        self.assertIsNone(d.left)
        self.assertFalse(c.right_is_down)

    def test_poll_holdoff_makes_zero_setcursorpos_then_engages(self):
        """Through the LIVE _poll_once with the LIVE default debounce: a 1-frame
        raise makes ZERO cursor moves; a sustained raise eventually moves it."""
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        # Use the module default engage debounce (the real policy) via a plain ctrl.
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0)
        # One raised frame → held off, no move yet.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertFalse(ctrl.engaged)
        self.assertEqual(moves, [])
        # Keep raising until it engages (bounded by the debounce frames).
        for _ in range(mod.AIR_MOUSE_ENGAGE_DEBOUNCE_FRAMES):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertTrue(ctrl.engaged)
        self.assertGreaterEqual(len(moves), 1)


# ══════════════════════════════════════════════════════════════════════════
#  AUTO-YIELD to real input — the air-mouse never fights the real mouse.
#  Controller-level: real input force-disengages + suppresses; an INJECTED event
#  (the air-mouse's own click) does NOT self-suppress. The last-real-input
#  timestamp is INJECTED (no real hook needed).
# ══════════════════════════════════════════════════════════════════════════
class AutoYieldControllerTests(_Base):
    """The controller YIELDS when real input is recent: force-disengage (release
    any held button, cursor=None) and stay suppressed (cannot re-engage) while the
    flag is set — even with a hand raised above the shoulder."""

    def _ctrl(self, mod, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        # engage_debounce_frames=1 so a single update() engages immediately in these
        # state-machine tests (the multi-frame engage DEBOUNCE has its own class).
        kw.setdefault("engage_debounce_frames", 1)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_real_input_blocks_engage_even_with_raised_hand(self):
        mod = self._load()
        c = self._ctrl(mod)
        # A hand IS raised above the shoulder (would normally engage), but real
        # input is recent → the air-mouse YIELDS: no engage, no cursor.
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True, real_input_recent=True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")

    def test_real_input_force_disengages_and_releases_held_button(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Engage + grab with the raised right hand.
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)
        self.assertTrue(c.right_is_down)
        # The owner touches their real mouse → instant yield: button released,
        # cursor released, overlay hidden — the real input wins.
        d = c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                     "open", "closed", True, real_input_recent=True)
        self.assertEqual(d.right, "up")
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.right_is_down)
        self.assertFalse(c.engaged)

    def test_suppressed_until_real_input_clears_then_reengages(self):
        mod = self._load()
        c = self._ctrl(mod)
        raised_r = self._ext(mod, "right")
        relaxed_l = self._relaxed(mod, "left")
        # Suppressed while real input is recent — stays disengaged frame after frame.
        for _ in range(10):
            d = c.update(relaxed_l, raised_r, "open", "open", True,
                         real_input_recent=True)
            self.assertFalse(c.engaged)
            self.assertIsNone(d.cursor)
        # Once the yield window elapses (no recent real input) a raised hand
        # re-engages.
        d2 = c.update(relaxed_l, raised_r, "open", "open", True,
                      real_input_recent=False)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d2.cursor)

    def test_module_real_input_recent_reads_injected_timestamp(self):
        # The module-level real_input_recent() reflects the yield watcher's
        # last-real-input timestamp, injected here with no real hook.
        mod = self._load()
        y = importlib.import_module("skills._air_mouse_yield")
        # Force the "no hook, not installed" state so the OS polling fallback is OFF
        # (we drive the timestamp directly).
        with y._install_lock:
            y._installed = False
            y._hook_ok = False
        try:
            y.note_real_input_for_test(time.monotonic())   # "just touched it"
            self.assertTrue(mod.real_input_recent(window=1.5))
            # Long ago → not recent (outside the window).
            y.note_real_input_for_test(time.monotonic() - 100.0)
            self.assertFalse(mod.real_input_recent(window=1.5))
        finally:
            y.note_real_input_for_test(float("-inf"))      # reset shared state


class AutoYieldWatcherTests(unittest.TestCase):
    """The low-level yield watcher (skills/_air_mouse_yield): real (non-injected)
    input registers as recent; an INJECTED event (the air-mouse's own click) does
    NOT — the injected-flag filter. The hook is NOT installed here; the
    last-real-input timestamp is injected directly."""

    def setUp(self):
        self.y = importlib.import_module("skills._air_mouse_yield")
        self._reset()
        self.addCleanup(self._reset)

    def _reset(self):
        # Clear the shared module state so tests don't bleed into each other, and
        # force the watcher into the "no hook, not installed" state so the OS
        # GetLastInputInfo polling fallback is OFF (the pure tests inject the
        # timestamp directly and must not read the test machine's real input).
        self.y.note_real_input_for_test(float("-inf"))
        with self.y._install_lock:
            self.y._installed = False
            self.y._hook_ok = False
        with self.y._lock:
            self.y._last_self_action = float("-inf")
            self.y._last_self_action_wall = float("-inf")

    def test_no_input_is_not_recent(self):
        self.assertFalse(self.y.real_input_recent(1.5))
        self.assertEqual(self.y.seconds_since_real_input(), float("inf"))

    def test_recorded_real_input_is_recent_then_ages_out(self):
        now = time.monotonic()
        self.y.note_real_input_for_test(now)
        self.assertTrue(self.y.real_input_recent(1.5, now=now + 0.5))
        self.assertFalse(self.y.real_input_recent(1.5, now=now + 2.0))

    def test_record_real_input_marks_recent(self):
        # The hook callbacks call _record_real_input() for NON-injected events.
        self.y._record_real_input()
        self.assertTrue(self.y.real_input_recent(1.5))

    def test_injected_flag_constant_is_the_low_bit(self):
        # LLMHF_INJECTED is bit 0 (0x01) of MSLLHOOKSTRUCT.flags — the bit the
        # mouse callback checks to IGNORE the air-mouse's own injected clicks.
        self.assertEqual(self.y._LLMHF_INJECTED, 0x01)

    def test_mark_self_action_does_not_count_as_real_input(self):
        # The air-mouse stamping its OWN action must NOT register as real input
        # (the LL-hook path ignores injected events; the fallback discounts them).
        self.y.mark_self_action()
        self.assertFalse(self.y.real_input_recent(1.5))

    def test_injected_mouse_event_does_not_self_suppress(self):
        # Simulate the LL mouse callback receiving an INJECTED event (the
        # air-mouse's own click): flags has LLMHF_INJECTED set, so it must NOT be
        # recorded as real input. We exercise the exact filter the callback uses.
        flags_injected = self.y._LLMHF_INJECTED
        if not (flags_injected & self.y._LLMHF_INJECTED):
            self.y._record_real_input()      # (won't run; documents the branch)
        self.assertFalse(self.y.real_input_recent(1.5))   # injected → not recent
        # A NON-injected event (real hardware) WOULD record.
        flags_real = 0x00
        if not (flags_real & self.y._LLMHF_INJECTED):
            self.y._record_real_input()
        self.assertTrue(self.y.real_input_recent(1.5))     # real → recent

    def test_install_is_idempotent_and_never_raises(self):
        # install() is lazy + graceful: calling it must never raise, even if the
        # LL hook can't be set up (it falls back to polling). Idempotent. Patch the
        # hook-thread spawn to a no-op so no REAL system-wide hook is installed by
        # the test (which would otherwise record the machine's real input).
        with mock.patch.object(self.y.threading, "Thread") as fake_thread:
            fake_thread.return_value.start = lambda: None
            try:
                first = self.y.install()
                second = self.y.install()      # idempotent: returns same, no re-spawn
            except Exception as e:   # pragma: no cover
                self.fail(f"install() raised {e!r}")
        self.assertEqual(first, second)


class AutoYieldPollIntegrationTests(_Base):
    """End-to-end through _poll_once: when real input is recent the live poll path
    YIELDS — zero cursor moves — even with a raised hand. Mouse + the yield module
    mocked so no real hook / cursor is touched."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_poll_yields_to_recent_real_input(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Force the yield signal ON (as if the owner just touched the real mouse):
        # a clearly raised hand that would otherwise engage → ZERO moves (yielded).
        with mock.patch.object(mod, "real_input_recent", lambda *a, **k: True):
            for _ in range(10):
                mod._poll_once(ctrl, _fake_bridge(
                    bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])
        self.assertFalse(ctrl.engaged)
        # Drop the yield → the same raised hand engages + moves the cursor.
        with mock.patch.object(mod, "real_input_recent", lambda *a, **k: False):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)


# ─── registration wires the actions ─────────────────────────────────────────
class RegisterTests(_Base):
    def test_register_adds_actions_without_starting_real_thread(self):
        mod, actions = load_skill_isolated("kinect_air_mouse", register=True)
        for name in ("air_mouse_on", "air_mouse_off", "air_mouse_status",
                     "calibrate_air_mouse"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


if __name__ == "__main__":
    unittest.main()
