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

    def test_reach_box_uses_persisted_calibration_geometry(self):
        # The calibration wizard writes KINECT_REACH_* to user_settings; the
        # reach-box must pick them up so the map fits the owner's arm span.
        mod = self._load()
        saved = {mod.SETTING_REACH_CENTER_X: 0.05,
                 mod.SETTING_REACH_CENTER_Y: 0.42,
                 mod.SETTING_REACH_HALF_W: 0.31,
                 mod.SETTING_REACH_HALF_H: 0.19}
        with mock.patch.object(mod, "_virtual_screen_bounds",
                               lambda: (self.VX, self.VY, self.VW, self.VH)), \
             mock.patch.object(mod, "_saved_settings", lambda: dict(saved)):
            rb = mod._reach_box_for_virtual_desktop(refresh=True)
        self.assertAlmostEqual(rb.center_x, 0.05)
        self.assertAlmostEqual(rb.center_y, 0.42)
        self.assertAlmostEqual(rb.half_w, 0.31)
        self.assertAlmostEqual(rb.half_h, 0.19)

    def test_reach_box_falls_back_per_field_on_partial_or_bad_calibration(self):
        # A partial (only one key) and a degenerate (half_w <= 0) calibration
        # must fall back to the module defaults per-field, never collapse the map.
        mod = self._load()
        saved = {mod.SETTING_REACH_CENTER_Y: 0.50,   # only one good key
                 mod.SETTING_REACH_HALF_W: 0.0}       # degenerate → default
        with mock.patch.object(mod, "_virtual_screen_bounds",
                               lambda: (self.VX, self.VY, self.VW, self.VH)), \
             mock.patch.object(mod, "_saved_settings", lambda: dict(saved)):
            rb = mod._reach_box_for_virtual_desktop(refresh=True)
        self.assertAlmostEqual(rb.center_y, 0.50)             # persisted
        self.assertAlmostEqual(rb.center_x, mod.REACH_CENTER_X)  # default
        self.assertAlmostEqual(rb.half_w, mod.REACH_HALF_W)      # default (was 0)
        self.assertAlmostEqual(rb.half_h, mod.REACH_HALF_H)      # default (absent)

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
#  REVERT (2026-06-09): the v1.73.0 "tablet-feel" mapping (aspect-match +
#  body-relative + absolute remap + KINECT_REACH_HALF_W env knob + nx/ny) was
#  JITTERY and unwanted. The single-hand cursor mapping is restored to v1.72.0
#  (fixed-centre reach-box, plain 2-arg ReachBox.map). The four tablet test
#  classes that pinned that behaviour are intentionally GONE; ReachBoxTests +
#  VirtualDesktopMappingTests above assert the restored v1.72.0 mapping, and the
#  SingleHandRevertedMappingTests below assert the controller drives the cursor
#  through the plain fixed-centre box (NO body args).
# ══════════════════════════════════════════════════════════════════════════
class SingleHandRevertedMappingTests(_Base):
    """The reverted v1.72.0 single-hand mapping: the controller feeds the smoothed
    hand (x, y) straight into ReachBox.map(sx, sy) with NO body centring / aspect
    derivation. A raised hand engages and the cursor lands where the FIXED-centre
    box maps it; sliding the body (without changing the absolute hand X) MOVES the
    cursor (it is sensor-absolute again, not body-relative)."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_cursor_matches_fixed_centre_box_map(self):
        mod = self._load()
        c = self._ctrl(mod)
        rb = mod.ReachBox(2560, 1440)
        relaxed_l = mod.ArmExtension("left", forward_m=0.0, straightness=None,
                                     hand=(-0.1, 0.0, 2.0, 2), lift_m=-0.40,
                                     shoulder_ref_y=0.40)
        # A raised RIGHT hand at a known camera (x, y). The cursor must equal the
        # PLAIN fixed-centre box map of that same (x, y) — body refs are ignored.
        hx, hy = 0.10, 0.55
        arm = mod.ArmExtension("right", forward_m=0.0, straightness=None,
                               hand=(hx, hy, 1.8, 2), lift_m=0.15,
                               shoulder_ref_y=0.40)
        d = c.update(relaxed_l, arm, "open", "open", True)
        self.assertIsNotNone(d.cursor)
        # EMA seeds to the first sample, so the mapped pixel is the box map of (hx, hy).
        self.assertEqual(d.cursor, rb.map(hx, hy))

    def test_body_shift_moves_cursor_absolute_again(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # Two raised RIGHT-hand samples with the SAME hand-minus-body offset but the
        # whole body shifted. Under the REVERTED (sensor-absolute) mapping the cursor
        # follows the ABSOLUTE hand X, so the two map to DIFFERENT pixels (the v1.73.0
        # body-relative behaviour, where both mapped to the same pixel, is gone).
        relaxed_l = mod.ArmExtension("left", forward_m=0.0, straightness=None,
                                     hand=(-0.1, 0.0, 2.0, 2), lift_m=-0.40,
                                     shoulder_ref_y=0.40)

        def arm(body_x, hand_dx, shoulder_y=0.40):
            return mod.ArmExtension(
                "right", forward_m=0.0, straightness=None,
                hand=(body_x + hand_dx, shoulder_y + 0.20, 1.8, 2),
                lift_m=0.20, shoulder_ref_y=shoulder_y)
        c1 = self._ctrl(mod)
        d1 = c1.update(relaxed_l, arm(0.0, 0.15), "open", "open", True)
        c2 = self._ctrl(mod)
        d2 = c2.update(relaxed_l, arm(0.6, 0.15), "open", "open", True)
        self.assertIsNotNone(d1.cursor)
        self.assertNotEqual(d1.cursor, d2.cursor)   # absolute hand X → different cursor
        # And each equals the plain box map of its absolute hand position.
        self.assertEqual(d1.cursor, rb.map(0.0 + 0.15, 0.40 + 0.20))
        self.assertEqual(d2.cursor, rb.map(0.6 + 0.15, 0.40 + 0.20))

    def test_map_takes_no_body_kwargs(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # The reverted ReachBox.map is the plain 2-arg form — passing body_x/body_y
        # (the v1.73.0 signature) must raise TypeError, proving the remap is gone.
        with self.assertRaises(TypeError):
            rb.map(0.0, 0.40, body_x=0.0, body_y=0.40)
        # And the v1.73.0 helpers/attrs no longer exist.
        self.assertFalse(hasattr(mod, "reach_half_w"))
        self.assertFalse(hasattr(rb, "normalize"))
        self.assertFalse(hasattr(rb, "center_y_offset"))


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

    def test_controlling_hand_hysteresis_reverted_to_v172(self):
        mod = self._load()
        # REVERTED 2026-06-09: the v1.73.0 "both-hands lock" (0.08 m / 8 frames,
        # which pinned two raised hands to ONE cursor) is undone — both hands raised
        # now enters TWO-HAND mode instead of a locked single cursor. The single-hand
        # hysteresis is back to the v1.72.0 values (0.25 / 6).
        self.assertAlmostEqual(mod.HAND_SWITCH_MARGIN, 0.25, delta=1e-9)
        self.assertEqual(mod.HAND_SWITCH_FRAMES, 6)

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

    def test_both_hands_raised_stands_down_for_two_hand_mode(self):
        """FILTER 8 (supersedes the old 'both buttons down at once'): when BOTH
        hands are raised above the engage line the single-hand controller STANDS
        DOWN locally (two-hand pinch-to-resize mode owns both hands), instead of
        grabbing one hand / firing both buttons. cursor=None, overlay hidden,
        disengaged — even with both hands closed. (Per-hand click independence with
        ONE hand raised is covered by test_either_hand_clicks_regardless_of_which_
        drives.)"""
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        d = c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                     "closed", "closed", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")
        self.assertIsNone(d.left)
        self.assertIsNone(d.right)
        self.assertFalse(c.left_is_down or c.right_is_down)

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

    def _relaxed_low(self, mod, side):
        """A LOWERED ArmExtension (hand below the engage line) so engaging on the
        OTHER hand doesn't trip the FILTER-8 both-hands-raised stand-down."""
        return mod.ArmExtension(side, forward_m=0.0, straightness=0.4,
                                hand=(0.1 if side == "right" else -0.1,
                                      SHOULDER_Y - 0.40, TORSO_Z, 2),
                                lift_m=-0.40)

    # NOTE (FILTER 8): with BOTH hands clearly raised above the engage line the
    # single-hand controller now STANDS DOWN (two-hand mode owns both hands), so the
    # controlling-hand switch HYSTERESIS can no longer be exercised through update()
    # with two simultaneously-raised hands — update() pre-empts to a release first.
    # The hysteresis machinery (_select_controlling_arm's per-frame margin + the
    # multi-frame switch streak) is therefore tested DIRECTLY below (it is still live
    # code: it picks the one raised hand, and would switch hands across frames where
    # the hands come up one at a time). The pure per-frame tie-break is also covered
    # by ArmExtensionTests.test_choose_highest_raised_hand / choose_controlling_arm.

    def test_holds_current_hand_when_other_barely_leads(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Engage on the RIGHT hand, then drive _select_controlling_arm DIRECTLY (so
        # the FILTER-8 both-hands-raised stand-down in update() doesn't pre-empt the
        # selection): the LEFT hand creeps a HAIR ahead but under the switch margin —
        # control must STAY on the right (no thrash).
        c.update(self._relaxed_low(mod, "left"),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        self.assertEqual(c.hand, "right")
        for _ in range(20):
            arm = c._select_controlling_arm(
                self._arm(mod, "left", 0.47, 0.99),
                self._arm(mod, "right", 0.45, 0.98), None)
            self.assertEqual(arm.side, "right")

    def test_switches_only_after_sustained_clear_lead(self):
        mod = self._load()
        c = self._ctrl(mod, switch_frames=6, switch_margin=0.25)
        c.update(self._relaxed_low(mod, "left"),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        self.assertEqual(c.hand, "right")
        # LEFT now leads by a CLEAR margin. Driving _select_controlling_arm directly
        # (bypassing the FILTER-8 pre-empt), it must take several frames to switch.
        switched_at = None
        for i in range(1, 12):
            arm = c._select_controlling_arm(self._arm(mod, "left", 0.90, 1.0),
                                            self._arm(mod, "right", 0.20, 0.80), None)
            c._hand = arm.side
            if arm.side == "left":
                switched_at = i
                break
        self.assertIsNotNone(switched_at)
        self.assertGreaterEqual(switched_at, 6)   # not instant — sustained lead

    def test_brief_lead_then_back_does_not_switch(self):
        mod = self._load()
        c = self._ctrl(mod, switch_frames=6, switch_margin=0.25)
        c.update(self._relaxed_low(mod, "left"),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        # LEFT leads big for a FEW frames (< switch_frames) then drops back. Direct
        # selection (no FILTER-8 pre-empt): control must never leave the right.
        for _ in range(3):
            arm = c._select_controlling_arm(self._arm(mod, "left", 0.90, 1.0),
                                            self._arm(mod, "right", 0.20, 0.80), None)
            self.assertEqual(arm.side, "right")   # not yet switched
        # Back to the right clearly leading → challenge resets, stays on right.
        for _ in range(5):
            arm = c._select_controlling_arm(self._arm(mod, "left", 0.20, 0.50),
                                            self._arm(mod, "right", 0.45, 0.98), None)
        self.assertEqual(arm.side, "right")

    def test_both_hands_raised_clicks_stand_down(self):
        """FILTER 8: BOTH hands raised → the single-hand controller stands down (no
        cursor, no buttons), superseding the old 'both grips tracked, both buttons
        fire' behaviour — two-hand pinch-to-resize mode owns both raised hands."""
        mod = self._load()
        c = self._ctrl(mod)
        c.update(self._arm(mod, "left", 0.20, 0.55),
                 self._arm(mod, "right", 0.45, 0.98), "open", "open", True)
        d = c.update(self._arm(mod, "left", 0.20, 0.55),
                     self._arm(mod, "right", 0.45, 0.98), "closed", "closed", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.left)
        self.assertIsNone(d.right)
        self.assertFalse(c.left_is_down or c.right_is_down)

    def test_both_hands_equally_raised_stands_down_no_flicker(self):
        """FILTER 8 (supersedes the old equal-extension tie-break): BOTH hands raised
        + EQUALLY extended now STANDS DOWN every frame (two-hand mode owns them) — so
        the cursor hand is None throughout (no flicker, and no single-cursor grab on
        either hand). Stable disengaged, not a thrashing pick between the two."""
        mod = self._load()
        c = self._ctrl(mod)
        hands = []
        for _ in range(40):
            d = c.update(self._arm(mod, "left", 0.40, 0.95),
                         self._arm(mod, "right", 0.40, 0.95), "open", "open", True)
            hands.append(d.hand)
            self.assertIsNone(d.cursor)
        self.assertEqual(set(hands), {None})   # stood down every frame
        self.assertFalse(c.engaged)

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
        """With ONE hand raised the cursor must follow ONLY the controlling hand —
        the OTHER (LOWERED, below the engage line) hand must NOT influence the cursor
        POSITION at all (no 'both hands connect to the same spot' / averaging). Hold
        the controlling (right) hand STILL while the idle (left, low) hand sweeps;
        the cursor must NOT move. The idle hand is kept BELOW the engage line so it
        is neither the controller nor a FILTER-8 both-hands-raised trigger."""
        mod = self._load()
        c = self._ctrl(mod)

        def right_arm(hand_x):
            # Controlling hand held at a FIXED position, clearly raised.
            return mod.ArmExtension("right", forward_m=0.0, straightness=None,
                                    hand=(hand_x, SHOULDER_Y + 0.30, 1.8, 2),
                                    lift_m=0.30, shoulder_ref_y=SHOULDER_Y)

        def left_arm(hand_x):
            # Idle hand LOWERED (below the engage line) so it's never the controller
            # and never trips the both-hands-raised pre-empt, sweeping in X.
            return mod.ArmExtension("left", forward_m=0.0, straightness=None,
                                    hand=(hand_x, SHOULDER_Y - 0.40, 1.8, 2),
                                    lift_m=-0.40, shoulder_ref_y=SHOULDER_Y)

        # Engage with the right hand controlling (the only raised hand).
        c.update(left_arm(-0.1), right_arm(0.10), "open", "open", True)
        base = c.update(left_arm(-0.1), right_arm(0.10),
                        "open", "open", True).cursor
        self.assertEqual(c.hand, "right")
        # Sweep the LEFT (idle, low) hand wildly while the RIGHT hand stays put.
        cursors = []
        for lx in (-0.4, -0.2, 0.0, 0.2, 0.4):
            d = c.update(left_arm(lx), right_arm(0.10), "open", "open", True)
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


# ══════════════════════════════════════════════════════════════════════════
#  ROBUSTNESS FILTERS (feat/kinect-harden-skills) — tracking-state floor, click
#  tracking-state, body-id pin, grace cap, two-hand rising-edge pre-empt.
# ══════════════════════════════════════════════════════════════════════════
def _inferred_arm_joints(side: str, *, hand_x=0.0, hand_y=RAISED_HAND_Y,
                         forward=0.30, hand_state=1, ref_state=2) -> dict:
    """Joints for one RAISED arm but with the HAND joint INFERRED (TrackingState 1
    by default) — the noisy guess the SDK emits for a hand it can't actually see.
    `ref_state` sets the spine_shoulder reference's state. Geometry matches
    _extended_arm_joints so only the tracking STATE differs."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, SHOULDER_Y, TORSO_Z
    hx, hy, hz = hand_x, hand_y, TORSO_Z - forward
    ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (sz + hz) / 2.0
    return {
        "spine_shoulder": (0.0, SHOULDER_Y, TORSO_Z, ref_state),
        "spine_mid": (0.0, 0.0, TORSO_Z, 2),
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, hand_state),   # INFERRED hand
    }


class JointWellTrackedTests(_Base):
    """The pure TrackingState floor helper (FILTERS 1/2/4): only a sensor-TRACKED,
    finite, non-zero joint passes."""

    def test_tracked_joint_passes(self):
        mod = self._load()
        self.assertTrue(mod.joint_well_tracked((0.1, 0.5, 1.8, 2)))
        self.assertTrue(mod.joint_well_tracked((0.1, 0.5, 1.8, 3)))   # >=2

    def test_inferred_and_nottracked_fail(self):
        mod = self._load()
        self.assertFalse(mod.joint_well_tracked((0.1, 0.5, 1.8, 1)))  # inferred
        self.assertFalse(mod.joint_well_tracked((0.1, 0.5, 1.8, 0)))  # not tracked

    def test_zero_origin_sentinel_fails(self):
        mod = self._load()
        # The (0,0,0) origin sentinel an unseen joint reads — rejected even at state 2.
        self.assertFalse(mod.joint_well_tracked((0.0, 0.0, 0.0, 2)))

    def test_non_finite_coords_fail(self):
        mod = self._load()
        inf = float("inf")
        nan = float("nan")
        self.assertFalse(mod.joint_well_tracked((inf, 0.5, 1.8, 2)))
        self.assertFalse(mod.joint_well_tracked((0.1, nan, 1.8, 2)))

    def test_malformed_joint_fails_safely(self):
        mod = self._load()
        self.assertFalse(mod.joint_well_tracked(None))
        self.assertFalse(mod.joint_well_tracked((0.1, 0.5, 1.8)))   # no state slot
        self.assertFalse(mod.joint_well_tracked(()))


class TrackingStateFloorTests(_Base):
    """FILTER 1: _local_arm_extension computes lift_m ONLY when BOTH the hand joint
    AND the shoulder-ref are sensor-tracked; an inferred hand leaves lift_m None →
    the gate can't engage on a hand the Kinect doesn't actually see."""

    def _ext(self, mod, joints, side):
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(joints, side))

    def test_inferred_hand_leaves_lift_none(self):
        mod = self._load()
        ext = self._ext(mod, _inferred_arm_joints("right", hand_state=1), "right")
        self.assertIsNone(ext.lift_m)            # not computed on an inferred hand
        self.assertFalse(ext.is_extended(engaged=False))
        self.assertFalse(ext.is_extended(engaged=True))

    def test_tracked_hand_still_computes_lift(self):
        mod = self._load()
        ext = self._ext(mod, _inferred_arm_joints("right", hand_state=2), "right")
        self.assertIsNotNone(ext.lift_m)         # fully-tracked hand → lift computed
        self.assertGreater(ext.lift_m, mod.AIR_MOUSE_ENGAGE_UP_MARGIN_M)

    def test_inferred_shoulder_ref_falls_back_then_floors(self):
        mod = self._load()
        # spine_shoulder INFERRED and NO same-side shoulder present → no usable ref
        # → lift_m None (can't confirm a raise).
        j = _inferred_arm_joints("right", hand_state=2, ref_state=1)
        j.pop("shoulder_right", None)
        ext = self._ext(mod, j, "right")
        self.assertIsNone(ext.lift_m)

    def test_zero_hand_joint_leaves_lift_none(self):
        mod = self._load()
        j = _extended_arm_joints("right")
        j["hand_right"] = (0.0, 0.0, 0.0, 2)     # origin sentinel (unseen)
        ext = self._ext(mod, j, "right")
        self.assertIsNone(ext.lift_m)

    def test_inferred_hand_makes_zero_setcursorpos_via_poll(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

        def inferred_body():
            j = _inferred_arm_joints("right", hand_state=1)
            jl = _relaxed_arm_joints("left")
            j["elbow_left"] = jl["elbow_left"]
            j["hand_left"] = jl["hand_left"]
            j["shoulder_left"] = jl["shoulder_left"]
            return {"id": 0, "joints": j, "head": (0.0, 0.7, TORSO_Z),
                    "distance_m": TORSO_Z, "facing": True,
                    "hand_right": "open", "hand_left": "open"}
        for _ in range(10):
            mod._poll_once(ctrl, _fake_bridge(bodies=[inferred_body()]))
        self.assertEqual(moves, [])              # inferred hand never drives
        self.assertFalse(ctrl.engaged)


class ClickTrackingStateTests(_Base):
    """FILTER 4: a hand whose JOINT isn't tracked is fed 'unknown' to its debouncer,
    so it can never press/hold a button — only a hand the Kinect SEES can click."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                      grace_sec=0.0, engage_debounce_frames=1)

    def _ext(self, mod, side, *, hand_state=2, lift=0.20):
        hx = 0.1 if side == "right" else -0.1
        return mod.ArmExtension(side, forward_m=0.0, straightness=None,
                                hand=(hx, SHOULDER_Y + lift, TORSO_Z - 0.30,
                                      hand_state),
                                lift_m=lift, shoulder_ref_y=SHOULDER_Y)

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_grip_if_tracked_passes_tracked_hand(self):
        mod = self._load()
        ext = self._ext(mod, "right", hand_state=2)
        self.assertEqual(
            mod.AirMouseController._grip_if_tracked(ext, "closed"), "closed")

    def test_grip_if_tracked_suppresses_untracked_hand(self):
        mod = self._load()
        ext = self._ext(mod, "right", hand_state=1)   # inferred
        self.assertEqual(
            mod.AirMouseController._grip_if_tracked(ext, "closed"), "unknown")
        # A None ext (arm unreadable) is likewise suppressed.
        self.assertEqual(
            mod.AirMouseController._grip_if_tracked(None, "closed"), "unknown")

    def test_inferred_clicking_hand_does_not_press(self):
        mod = self._load()
        c = self._ctrl(mod)
        # The cursor-driving hand is the LEFT (tracked, raised). The RIGHT hand is
        # kept BELOW the engage line (so it isn't a FILTER-8 both-hands-raised
        # trigger) and is INFERRED — even reading "closed" it must NOT press the
        # right button (clicks are evaluated for both hands, but an untracked joint
        # is fed 'unknown').
        left = self._ext(mod, "left", hand_state=2, lift=0.30)
        right_inf = self._ext(mod, "right", hand_state=1, lift=-0.40)
        c.update(left, right_inf, "open", "open", True)
        d = c.update(left, right_inf, "open", "closed", True)
        self.assertEqual(c.hand, "left")          # left drives
        self.assertIsNone(d.right)                # inferred right CANNOT press
        self.assertFalse(c.right_is_down)
        # The SAME low right hand, but TRACKED, does fire the click (proving it's the
        # tracking STATE — not the height — that suppressed it above).
        c2 = self._ctrl(mod)
        right_ok = self._ext(mod, "right", hand_state=2, lift=-0.40)
        c2.update(left, right_ok, "open", "open", True)
        d2 = c2.update(left, right_ok, "open", "closed", True)
        self.assertEqual(d2.right, "down")


class BodyIdPinTests(_Base):
    """FILTER 6: the controlling body id is latched on engage; if the nearest-body
    id changes (a closer 2nd person) it's a tracking-loss (release + EMA reset), not
    a seamless retarget."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                      grace_sec=0.0, engage_debounce_frames=1)

    def _ext(self, mod, side):
        j = _extended_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_body_id_change_releases_mid_drag(self):
        mod = self._load()
        c = self._ctrl(mod)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        # Engage + grab on body 7.
        c.update(left, right, "open", "open", True, body_id=7)
        c.update(left, right, "open", "closed", True, body_id=7)
        self.assertTrue(c.engaged and c.right_is_down)
        self.assertEqual(c._locked_body_id, 7)
        # A CLOSER 2nd person (id 9) becomes the nearest body → release, NOT retarget.
        d = c.update(left, right, "open", "closed", True, body_id=9)
        self.assertFalse(c.engaged)
        self.assertEqual(d.right, "up")           # held button released
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")

    def test_same_body_id_keeps_driving(self):
        mod = self._load()
        c = self._ctrl(mod)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        c.update(left, right, "open", "open", True, body_id=3)
        d = c.update(left, right, "open", "open", True, body_id=3)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d.cursor)            # same body → uninterrupted

    def test_none_body_id_disables_pin(self):
        mod = self._load()
        c = self._ctrl(mod)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        # No ids supplied (back-compat) → never releases on the (absent) id.
        c.update(left, right, "open", "open", True)
        d = c.update(left, right, "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d.cursor)

    def test_poll_passes_body_id_and_swap_releases(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                      grace_sec=0.0, engage_debounce_frames=1)
        # Body id 1 engages + grabs (close the right hand for a drag).
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="closed")]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(buttons, [("down", "right")])
        # A DIFFERENT body id (a closer person) under the same raised-hand sample →
        # the pin releases the drag instead of retargeting onto the interloper.
        other = _body(reach_side="right", grip_right="closed")
        other["id"] = 99
        mod._poll_once(ctrl, _fake_bridge(bodies=[other]))
        self.assertFalse(ctrl.engaged)
        self.assertEqual(buttons, [("down", "right"), ("up", "right")])


class GraceCapTests(_Base):
    """FILTER 7: cumulative untracked time per engagement is CAPPED, so a flickering
    body (one Tracked frame renews the per-dropout grace) can't hold a drag forever.
    A solid run of Tracked frames re-arms (zeroes) the accumulator."""

    def _ctrl(self, mod, clk):
        # Grace 0.30 s, ceiling 0.50 s, re-arm after 3 solid Tracked frames.
        return mod.AirMouseController(
            mod.ReachBox(2560, 1440), debounce_frames=1, grace_sec=0.30,
            engage_debounce_frames=1, untracked_ceiling_sec=0.50,
            retrack_frames=3, clock=clk)

    def _ext(self, mod, side):
        j = _extended_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_flicker_cannot_renew_grace_past_ceiling(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        c.update(left, right, "open", "open", True)          # engage
        c.update(left, right, "open", "closed", True)        # RIGHT down (drag)
        self.assertTrue(c.right_is_down)
        # FLICKER: each cycle is one untracked frame (0.20 s, within the 0.30 s
        # per-dropout grace) then one lone Tracked frame that renews the grace but is
        # NOT a solid re-acquisition (< retrack_frames in a row). The cumulative
        # untracked time climbs and must eventually trip the 0.50 s ceiling.
        released = False
        for _ in range(8):
            clk.advance(0.20)
            d = c.update(None, None, "unknown", "unknown", False)   # untracked
            if d.right == "up" or not c.right_is_down:
                released = True
                break
            # one lone Tracked frame (renews per-dropout grace, breaks no streak cap)
            clk.advance(0.01)
            c.update(left, right, "open", "closed", True)
        self.assertTrue(released, "grace cap never force-released a flickering body")
        self.assertFalse(c.right_is_down)

    def test_solid_retrack_rearms_the_budget(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        # SMART-ENGAGE: engage with an OPEN palm first (the passive gate requires
        # it), THEN close the right hand for the drag.
        c.update(left, right, "open", "open", True)          # engage (open palm)
        c.update(left, right, "open", "closed", True)        # RIGHT down (drag)
        # Spend some untracked budget (under the ceiling), then RE-ACQUIRE solidly
        # for several consecutive Tracked frames → the accumulator zeroes.
        clk.advance(0.20)
        c.update(None, None, "unknown", "unknown", False)
        self.assertGreater(c._untracked_accum, 0.0)
        for _ in range(4):                                    # > retrack_frames
            clk.advance(0.01)
            c.update(left, right, "open", "closed", True)
        self.assertEqual(c._untracked_accum, 0.0)            # re-armed
        self.assertTrue(c.right_is_down)                     # drag survived

    def test_single_brief_dropout_still_holds(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk)
        right = self._ext(mod, "right")
        left = self._relaxed(mod, "left")
        # SMART-ENGAGE: engage with an OPEN palm first, THEN close for the drag.
        c.update(left, right, "open", "open", True)          # engage (open palm)
        c.update(left, right, "open", "closed", True)        # RIGHT down (drag)
        # A single short dropout within both the grace AND the ceiling → still holds.
        clk.advance(0.10)
        d = c.update(None, None, "unknown", "unknown", False)
        self.assertIsNone(d.right)                           # NOT released
        self.assertTrue(c.right_is_down)
        self.assertEqual(d.overlay, "grab")


class TwoHandPreemptTests(_Base):
    """FILTER 8: the controller stands down the instant it LOCALLY sees BOTH hands
    raised (a two-hand-mode pre-empt) — closing the 1-frame entry twitch before the
    cross-process two-hand heartbeat arrives."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                      grace_sec=0.0, engage_debounce_frames=1)

    def _ext(self, mod, side, *, hand_state=2):
        j = _extended_arm_joints(side)
        if hand_state != 2:
            hx, hy, hz, _ = j[f"hand_{side}"]
            j[f"hand_{side}"] = (hx, hy, hz, hand_state)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_helper_true_only_when_both_tracked_and_raised(self):
        mod = self._load()
        both_up = mod.AirMouseController._both_hands_raised(
            self._ext(mod, "left"), self._ext(mod, "right"), None)
        self.assertTrue(both_up)
        # One hand lowered → not both raised.
        self.assertFalse(mod.AirMouseController._both_hands_raised(
            self._relaxed(mod, "left"), self._ext(mod, "right"), None))
        # Both raised but one hand INFERRED → not a pre-empt (FILTER 2/8 tracking).
        self.assertFalse(mod.AirMouseController._both_hands_raised(
            self._ext(mod, "left", hand_state=1), self._ext(mod, "right"), None))

    def test_rising_second_hand_preempts_engage(self):
        mod = self._load()
        c = self._ctrl(mod)
        # One hand raised → engages.
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        self.assertTrue(c.engaged)
        # The SECOND hand comes up the very next frame → stand down at once (no twitch
        # waiting for the two-hand heartbeat).
        d = c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                     "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")

    def test_preempt_releases_held_button(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Engage + grab with the right hand, then raise the left → release the drag.
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "open", True)
        c.update(self._relaxed(mod, "left"), self._ext(mod, "right"),
                 "open", "closed", True)
        self.assertTrue(c.right_is_down)
        d = c.update(self._ext(mod, "left"), self._ext(mod, "right"),
                     "open", "closed", True)
        self.assertEqual(d.right, "up")           # the held button is let go
        self.assertFalse(c.engaged)


# ─── overlay-state file ownership: two-hand stand-down + disabled edge gate ─
class OverlayStateWriteGateTests(_Base):
    """AIR_CURSOR_STATE_FILE has TWO writers (this skill + kinect_two_hand's
    poller). These tests pin the write discipline that stops them fighting:

      * while two_hand_active() the enabled poll path must NOT publish its
        hidden frames (the two-hand poller owns the file; interleaved 30 Hz
        hidden writes strobed the dual reticles), and
      * the DISABLED poll path clears the file / preview state ONCE per
        disable edge, not on every ~33 ms tick (SSD churn + the same fight)."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0,
                                      engage_debounce_frames=1)

    def test_two_hand_mode_suppresses_overlay_publish(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._capture_mouse(mod)
        publishes = []
        p = mock.patch.object(mod, "_publish_overlay_state",
                              lambda d, v, **k: publishes.append((d.overlay, v)))
        p.start()
        self.addCleanup(p.stop)
        ctrl = self._ctrl(mod)
        body = _body(reach_side="right", grip_right="open")
        # Two-hand mode active → the air-mouse yields AND stays off the file.
        with mock.patch.object(mod, "two_hand_active", lambda: True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[body]))
            mod._poll_once(ctrl, _fake_bridge(bodies=[body]))
        self.assertEqual(publishes, [])
        # Two-hand mode ends → publishing resumes.
        mod._poll_once(ctrl, _fake_bridge(bodies=[body]))
        self.assertEqual(len(publishes), 1)

    def test_disabled_poller_clears_overlay_state_once_not_every_tick(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        self._capture_mouse(mod)
        clears, states = [], []
        p1 = mock.patch.object(mod, "_clear_overlay_state",
                               lambda: clears.append(True))
        p2 = mock.patch.object(mod, "_set_air_mouse_state",
                               lambda *a, **k: states.append(a))
        for p in (p1, p2):
            p.start()
            self.addCleanup(p.stop)
        ctrl = self._ctrl(mod)
        for _ in range(5):
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        # ONE clear + ONE disengaged-preview write on the disable edge, then quiet.
        self.assertEqual(len(clears), 1)
        self.assertEqual(len(states), 1)

    def test_disabled_clear_rearms_after_an_enabled_tick(self):
        mod = self._load()
        self._not_staging(mod)
        self._capture_mouse(mod)
        clears = []
        p = mock.patch.object(mod, "_clear_overlay_state",
                              lambda: clears.append(True))
        p.start()
        self.addCleanup(p.stop)
        ctrl = self._ctrl(mod)
        from core import config as cfg
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", False, create=True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        self.assertEqual(len(clears), 1)
        # Enabled tick resets the edge gate…
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", True, create=True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        # …so the next disable clears exactly once more.
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", False, create=True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
            mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        self.assertEqual(len(clears), 2)


# ══════════════════════════════════════════════════════════════════════════
#  SMART-ENGAGE (2026-07, feat/smart-engage) — the HYBRID engage model.
#  A. the PURE engage_decision() gate (no sensor/clock): armed relaxes to
#     height-only; passive needs open+facing+still+dwell; a fast reach (dwell not
#     met) never engages; closed/low/look-away don't engage passively; missing
#     facing doesn't block; prime progresses 0→1 over the dwell.
#  B. the controller wiring (dwell timing via injected clock, stillness, fist
#     release), the per-app disable, facing extraction, arm/disarm actions, and the
#     overlay `prime` publish.
# ══════════════════════════════════════════════════════════════════════════
class EngageDecisionPureTests(_Base):
    """The PURE smart-engage gate engage_decision() — every knob is an argument so
    no sensor / clock / config is touched."""

    def _dec(self, mod, **kw):
        # Sensible passive defaults for the pure gate; each test overrides.
        kw.setdefault("require_open_palm", True)
        kw.setdefault("facing_max_deg", 40.0)
        kw.setdefault("dwell_sec", 0.30)
        kw.setdefault("arm_debounce_sec", 0.15)
        kw.setdefault("arm_relaxes_gate", True)
        return mod.engage_decision(**kw)

    # ── ARMED relaxes to height-only + a short hold ─────────────────────────
    def test_armed_engages_on_height_after_short_hold(self):
        mod = self._load()
        # Armed: a raised hand held past the arm debounce engages — even with a
        # CLOSED grip, no facing, and moving (grip/facing/still not required).
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=True,
                      grip="closed", facing_deg=170.0, hand_still=False,
                      arm_debounce_elapsed=0.20)
        self.assertTrue(v.engaged)
        self.assertEqual(v.prime, 0.0)          # no priming ring in armed mode

    def test_armed_holds_off_until_short_debounce(self):
        mod = self._load()
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=True,
                      arm_debounce_elapsed=0.05)   # under 0.15 s
        self.assertFalse(v.engaged)

    def test_armed_no_lift_does_not_engage(self):
        mod = self._load()
        v = self._dec(mod, lift_ok=False, currently_engaged=False, armed=True,
                      arm_debounce_elapsed=1.0)
        self.assertFalse(v.engaged)

    # ── PASSIVE needs open + facing + still + dwell ─────────────────────────
    def test_passive_full_pose_engages_after_dwell(self):
        mod = self._load()
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="open", facing_deg=10.0, hand_still=True,
                      dwell_elapsed=0.30)
        self.assertTrue(v.engaged)

    def test_passive_closed_hand_never_engages(self):
        mod = self._load()
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="closed", facing_deg=0.0, hand_still=True,
                      dwell_elapsed=1.0)
        self.assertFalse(v.engaged)
        self.assertFalse(v.priming)

    def test_passive_look_away_never_engages(self):
        mod = self._load()
        # Facing 60° > the 40° bar → not facing → never engages.
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="open", facing_deg=60.0, hand_still=True,
                      dwell_elapsed=1.0)
        self.assertFalse(v.engaged)
        self.assertFalse(v.priming)

    def test_passive_low_hand_never_engages(self):
        mod = self._load()
        # lift_ok False (hand below the shoulder) → never engages regardless.
        v = self._dec(mod, lift_ok=False, currently_engaged=False, armed=False,
                      grip="open", facing_deg=0.0, hand_still=True,
                      dwell_elapsed=1.0)
        self.assertFalse(v.engaged)

    def test_passive_moving_hand_never_engages(self):
        mod = self._load()
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="open", facing_deg=0.0, hand_still=False,
                      dwell_elapsed=1.0)
        self.assertFalse(v.engaged)
        self.assertFalse(v.priming)

    def test_fast_reach_never_engages_dwell_not_met(self):
        mod = self._load()
        # A natural fast reach: a valid pose but only a fraction of the dwell held
        # → priming, not engaged (it would leave the zone before the dwell).
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="open", facing_deg=0.0, hand_still=True,
                      dwell_elapsed=0.10)          # < 0.30 s
        self.assertFalse(v.engaged)
        self.assertTrue(v.priming)

    def test_missing_facing_does_not_block(self):
        mod = self._load()
        # facing_deg None (bridge didn't provide it) → treated as facing OK.
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="open", facing_deg=None, hand_still=True,
                      dwell_elapsed=0.30)
        self.assertTrue(v.engaged)

    def test_prime_progresses_zero_to_one_over_dwell(self):
        mod = self._load()
        primes = []
        for t in (0.0, 0.075, 0.15, 0.225, 0.29):
            v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                          grip="open", facing_deg=0.0, hand_still=True,
                          dwell_elapsed=t)
            self.assertTrue(v.priming)
            primes.append(v.prime)
        # Monotonic 0→~1 as the dwell fills.
        self.assertEqual(primes, sorted(primes))
        self.assertAlmostEqual(primes[0], 0.0, delta=0.01)
        self.assertGreater(primes[-1], 0.9)
        # At/after the dwell → engaged, prime resets to 0 (nothing to prime).
        v_done = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                           grip="open", facing_deg=0.0, hand_still=True,
                           dwell_elapsed=0.30)
        self.assertTrue(v_done.engaged)
        self.assertEqual(v_done.prime, 0.0)

    def test_staying_engaged_needs_only_height(self):
        mod = self._load()
        # Already engaged: height alone keeps it (pose/dwell only guard ACQUIRING).
        # A closed, look-away, moving hand STAYS engaged as long as lift holds.
        v = self._dec(mod, lift_ok=True, currently_engaged=True, armed=False,
                      grip="closed", facing_deg=170.0, hand_still=False)
        self.assertTrue(v.engaged)
        # Lift lost → not engaged (the controller then releases).
        v2 = self._dec(mod, lift_ok=False, currently_engaged=True, armed=False,
                       grip="open", facing_deg=0.0, hand_still=True)
        self.assertFalse(v2.engaged)

    def test_open_palm_not_required_when_knob_off(self):
        mod = self._load()
        # With require_open_palm False a closed hand can pass the passive pose.
        v = self._dec(mod, lift_ok=True, currently_engaged=False, armed=False,
                      grip="closed", facing_deg=0.0, hand_still=True,
                      dwell_elapsed=0.30, require_open_palm=False)
        self.assertTrue(v.engaged)


class SmartEngageControllerTests(_Base):
    """The controller wiring: the PASSIVE dwell (timed via the injected clock), the
    stillness accumulator, and the ARMED relaxed gate — end-to-end through
    update()."""

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_passive_open_palm_hold_engages_after_dwell(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        # A real dwell-timed controller (NOT the frame-debounce shortcut): dwell
        # 0.30 s, a large engage_debounce_frames so only the CLOCK crosses it.
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk, dwell_sec=0.30,
                                   engage_debounce_frames=999, require_open_palm=True,
                                   facing_max_deg=40.0, arm_relaxes_gate=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        # Open palm raised, still, facing → primes but doesn't engage until dwell.
        d0 = c.update(left, right, "open", "open", True, facing_deg=0.0, armed=False)
        self.assertFalse(c.engaged)
        self.assertTrue(d0.prime > 0.0)          # priming ring filling
        # Hold past the dwell → engage.
        clk.advance(0.31)
        d1 = c.update(left, right, "open", "open", True, facing_deg=0.0, armed=False)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d1.cursor)
        self.assertEqual(d1.prime, 0.0)          # engaged → nothing to prime

    def test_passive_closed_hand_never_primes(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk, dwell_sec=0.30,
                                   engage_debounce_frames=999, require_open_palm=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        for _ in range(10):
            clk.advance(0.05)
            d = c.update(left, right, "open", "closed", True, facing_deg=0.0)
            self.assertFalse(c.engaged)
            self.assertEqual(d.prime, 0.0)       # closed hand → no priming

    def test_passive_look_away_never_engages(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk, dwell_sec=0.30,
                                   engage_debounce_frames=999, facing_max_deg=40.0)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        for _ in range(10):
            clk.advance(0.05)
            c.update(left, right, "open", "open", True, facing_deg=80.0)  # turned away
        self.assertFalse(c.engaged)

    def test_invalid_pose_does_not_precharge_dwell_at_live_default(self):
        # 2026-07-07 review (MED-HIGH): at the LIVE default engage_debounce_frames
        # =3 (every other controller test pins 999, which masks the frame-credit),
        # a raised OPEN palm held while LOOKING AWAY (an invalid passive pose) used
        # to pre-charge the engage streak via the frame-credit — so the first frame
        # the owner FACED the sensor engaged INSTANTLY with zero dwell and no ring,
        # re-opening the false-trigger the rewrite killed. The streak must count
        # only VALID frames; the first valid frame after an invalid hold must PRIME,
        # not engage.
        mod = self._load()
        clk = _FakeClock(100.0)
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk, dwell_sec=0.30,
                                   engage_debounce_frames=3, require_open_palm=True,
                                   facing_max_deg=40.0, arm_relaxes_gate=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        # Open palm raised + still, but LOOKING AWAY for ~1 s → invalid pose the
        # whole time (would pre-charge the streak to ~20 with the old bug).
        for _ in range(20):
            clk.advance(0.05)
            c.update(left, right, "open", "open", True, facing_deg=170.0, armed=False)
            self.assertFalse(c.engaged)
        # First frame FACING the sensor → must PRIME, never engage instantly.
        clk.advance(0.05)
        d = c.update(left, right, "open", "open", True, facing_deg=0.0, armed=False)
        self.assertFalse(c.engaged,
                         "invalid-pose pre-charge engaged on the first valid frame")
        self.assertLess(d.prime, 1.0)
        # A genuinely-held valid pose still engages after the real dwell.
        clk.advance(0.31)
        c.update(left, right, "open", "open", True, facing_deg=0.0, armed=False)
        self.assertTrue(c.engaged)

    def test_moving_hand_breaks_the_dwell(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk, dwell_sec=0.30,
                                   engage_debounce_frames=999, still_m=0.06)
        left = self._relaxed(mod, "left")
        # Sweep the hand far in X each frame → travel exceeds the still bar → the
        # dwell keeps resetting / never engages.
        for i in range(12):
            clk.advance(0.05)
            right = self._ext(mod, "right", hand_x=0.02 * i)   # 2 cm per frame
            c.update(left, right, "open", "open", True, facing_deg=0.0)
        self.assertFalse(c.engaged)

    def test_armed_engages_fast_height_only(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, clock=clk,
                                   arm_debounce_sec=0.15, engage_debounce_frames=999,
                                   require_open_palm=True, arm_relaxes_gate=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        # ARMED: even a CLOSED-grip, look-away raised hand engages after the short
        # arm debounce (the relaxed gate ignores grip/facing/still/long-dwell).
        c.update(left, right, "open", "closed", True, facing_deg=170.0, armed=True)
        self.assertFalse(c.engaged)              # under the 0.15 s hold
        clk.advance(0.20)
        c.update(left, right, "open", "closed", True, facing_deg=170.0, armed=True)
        self.assertTrue(c.engaged)

    def test_per_app_disabled_stands_down(self):
        mod = self._load()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, engage_debounce_frames=1,
                                   arm_relaxes_gate=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        # Armed (would engage immediately) BUT the foreground app is disabled →
        # stand down: no engage, cursor None, overlay hidden.
        d = c.update(left, right, "open", "open", True, armed=True,
                     per_app_disabled=True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")

    def test_per_app_disable_force_releases_engaged(self):
        mod = self._load()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, engage_debounce_frames=1,
                                   arm_relaxes_gate=True, arm_debounce_sec=0.0)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)   # engage
        c.update(left, right, "open", "closed", True, armed=True)  # right down
        self.assertTrue(c.right_is_down)
        # A disabled app comes to the foreground → release the held button + stand down.
        d = c.update(left, right, "open", "closed", True, armed=True,
                     per_app_disabled=True)
        self.assertEqual(d.right, "up")
        self.assertFalse(c.engaged)


class FistReleaseTests(_Base):
    """A SUSTAINED closed fist while engaged force-disengages (AIR_MOUSE_FIST_-
    RELEASES); a normal click / short drag never trips it."""

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _ctrl(self, mod, clk, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        kw.setdefault("engage_debounce_frames", 1)
        kw.setdefault("arm_relaxes_gate", True)
        kw.setdefault("arm_debounce_sec", 0.0)   # engage on the first armed frame
        return mod.AirMouseController(mod.ReachBox(2560, 1440), clock=clk, **kw)

    def test_sustained_fist_releases(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)   # engage
        c.update(left, right, "open", "closed", True, armed=True)  # right down
        self.assertTrue(c.engaged and c.right_is_down)
        # Hold the fist closed past the release window → force-disengage + release.
        clk.advance(0.70)
        d = c.update(left, right, "open", "closed", True, armed=True)
        self.assertFalse(c.engaged)
        self.assertEqual(d.right, "up")
        self.assertIsNone(d.cursor)

    def test_short_close_does_not_release(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)
        c.update(left, right, "open", "closed", True, armed=True)  # right down
        clk.advance(0.20)                                          # brief close
        d = c.update(left, right, "open", "closed", True, armed=True)
        self.assertTrue(c.engaged)          # not released — a short close is a drag
        self.assertIsNone(d.right)          # no button edge

    def test_fist_release_off_when_knob_false(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=False)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)
        c.update(left, right, "open", "closed", True, armed=True)
        clk.advance(1.0)
        c.update(left, right, "open", "closed", True, armed=True)
        self.assertTrue(c.engaged)          # knob off → a long fist stays a drag

    def test_fist_release_latches_no_reengage_oscillation(self):
        """The 2026-07-07 owner report ("jittery when hand closed — turns on and
        off"): in ARMED mode (height-only re-engage), a sustained fist would
        release, then the still-raised fist would instantly re-grab, re-release,
        and OSCILLATE. The fist-release LATCH must hold re-engagement OFF while the
        hand stays closed — stable disengaged, no flicker."""
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)    # engage
        c.update(left, right, "open", "closed", True, armed=True)  # fist down
        self.assertTrue(c.engaged)
        clk.advance(0.70)
        c.update(left, right, "open", "closed", True, armed=True)  # fist-release fires
        self.assertFalse(c.engaged)
        self.assertTrue(c.fist_release_latched)
        # Hold the fist CLOSED + raised for many frames — it must NOT re-engage.
        for _ in range(12):
            clk.advance(0.05)
            d = c.update(left, right, "open", "closed", True, armed=True)
            self.assertFalse(c.engaged)          # stays released (no on/off)
            self.assertIsNone(d.cursor)
        self.assertTrue(c.fist_release_latched)   # never opened → still latched

    def test_open_hand_clears_fist_latch_and_reengages(self):
        """Opening the hand is the ONLY thing that clears the fist-release latch —
        then a normal armed engage resumes."""
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)
        c.update(left, right, "open", "closed", True, armed=True)
        clk.advance(0.70)
        c.update(left, right, "open", "closed", True, armed=True)   # latch
        self.assertTrue(c.fist_release_latched and not c.engaged)
        # Open the raised hand → latch clears (debounce_frames=1) → re-engages.
        clk.advance(0.05)
        c.update(left, right, "open", "open", True, armed=True)
        self.assertFalse(c.fist_release_latched)
        self.assertTrue(c.engaged)

    def test_short_fist_does_not_latch(self):
        """A quick close→open (a click / short drag) is UNDER the release window, so
        it never latches — normal clicking is unaffected."""
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, clk, fist_release_sec=0.60, fist_releases=True)
        left = self._relaxed(mod, "left")
        right = self._ext(mod, "right")
        c.update(left, right, "open", "open", True, armed=True)
        c.update(left, right, "open", "closed", True, armed=True)   # down
        clk.advance(0.20)                                           # brief
        c.update(left, right, "open", "open", True, armed=True)     # opened (click)
        self.assertTrue(c.engaged)
        self.assertFalse(c.fist_release_latched)


class AppDisableMatchTests(_Base):
    """The PURE per-app disable matcher app_is_disabled() + _body_facing_deg."""

    def test_matches_title_substring(self):
        mod = self._load()
        self.assertTrue(mod.app_is_disabled("Netflix — Watching", "Chrome_WidgetWin",
                                            hints=["netflix"]))

    def test_matches_class_substring(self):
        mod = self._load()
        self.assertTrue(mod.app_is_disabled("Some Game", "UnrealWindow",
                                            hints=["unrealwindow"]))

    def test_no_match_returns_false(self):
        mod = self._load()
        self.assertFalse(mod.app_is_disabled("Notepad", "Notepad",
                                             hints=["netflix", "vlc"]))

    def test_empty_window_is_not_disabled(self):
        mod = self._load()
        self.assertFalse(mod.app_is_disabled("", "", hints=["netflix"]))

    def test_case_insensitive(self):
        mod = self._load()
        self.assertTrue(mod.app_is_disabled("YOUTUBE - Foo", "", hints=["youtube - "]))

    def test_default_hints_used_when_none(self):
        mod = self._load()
        # None hints → the live AIR_MOUSE_DISABLED_APP_HINTS (which include vlc).
        self.assertTrue(mod.app_is_disabled("VLC media player", ""))

    def test_body_facing_deg_prefers_yaw(self):
        mod = self._load()
        self.assertEqual(mod._body_facing_deg({"facing_yaw_deg": -25.0}), 25.0)
        self.assertEqual(mod._body_facing_deg({"facing_yaw_deg": 10.0}), 10.0)

    def test_body_facing_deg_bool_fallback(self):
        mod = self._load()
        self.assertEqual(mod._body_facing_deg({"facing": True}), 0.0)
        self.assertEqual(mod._body_facing_deg({"facing": False}), 180.0)

    def test_body_facing_deg_missing_is_none(self):
        mod = self._load()
        self.assertIsNone(mod._body_facing_deg({}))


class ArmStateTests(_Base):
    """The module ARMED flag + arm/disarm actions."""

    def setUp(self):
        # Reset the shared armed flag around each test so they don't bleed.
        self._mod = self._load()
        self._mod.air_mouse_disarm()
        self.addCleanup(self._mod.air_mouse_disarm)

    def test_arm_and_disarm_flip_the_flag(self):
        mod = self._mod
        self.assertFalse(mod.air_mouse_is_armed())
        mod.air_mouse_arm()
        self.assertTrue(mod.air_mouse_is_armed())
        mod.air_mouse_disarm()
        self.assertFalse(mod.air_mouse_is_armed())

    def test_update_reads_module_armed_flag_when_arg_none(self):
        mod = self._mod
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0, engage_debounce_frames=1,
                                   arm_relaxes_gate=True, require_open_palm=True,
                                   arm_debounce_sec=0.0)   # engage on first armed frame
        j = _extended_arm_joints("right")
        right = mod.ArmExtension.from_bridge(mod._local_arm_extension(j, "right"))
        jl = _relaxed_arm_joints("left")
        left = mod.ArmExtension.from_bridge(mod._local_arm_extension(jl, "left"))
        # Not armed + closed hand → passive gate rejects (armed=None reads the flag).
        c.update(left, right, "open", "closed", True)
        self.assertFalse(c.engaged)
        # Arm via the module flag → the same closed-hand raise engages (relaxed).
        mod.air_mouse_arm()
        c.update(left, right, "open", "closed", True)
        self.assertTrue(c.engaged)

    def test_arm_action_arms_and_enables(self):
        mod = self._mod
        self._patch_flag(False)
        from tools import settings_window as sw
        saved = {}
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p3 = mock.patch.object(mod, "_sensor_ready", lambda: (True, ""))
        for p in (p1, p2, p3):
            p.start(); self.addCleanup(p.stop)
        out = mod.air_mouse_arm_action("")
        self.assertTrue(mod.air_mouse_is_armed())
        self.assertIn("armed", out.lower())
        self.assertTrue(saved.get("KINECT_AIR_MOUSE_ENABLED"))   # enabled it too

    def test_disarm_action_disarms_and_releases(self):
        mod = self._mod
        mod.air_mouse_arm()
        released = []
        with mock.patch.object(mod, "_mouse_button",
                               lambda a, b="left": released.append((a, b))), \
                mock.patch.object(mod, "_clear_overlay_state", lambda: None):
            out = mod.air_mouse_disarm_action("")
        self.assertFalse(mod.air_mouse_is_armed())
        self.assertIn("released", out.lower())
        # Both buttons let go so disarm can't strand one down.
        self.assertIn(("up", "left"), released)
        self.assertIn(("up", "right"), released)


class OverlayPrimePublishTests(_Base):
    """The overlay `prime` key (shared HUD contract): published in the state file,
    exposed via get_air_mouse_state()/get_air_mouse_prime()."""

    def test_publish_writes_prime_key(self):
        mod = self._load()
        import json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "air_cursor_state.json")
            with mock.patch.object(mod, "AIR_CURSOR_STATE_FILE", path):
                # A priming decision (cursor None, prime>0) → state "prime", visible,
                # positioned at prime_xy, prime carried through.
                dec = mod.AirMouseDecision(cursor=None, left=None, right=None,
                                           overlay="hidden", hand=None, grip="open",
                                           prime=0.5)
                mod._publish_overlay_state(dec, visible=True, prime_xy=(640, 360))
                data = json.load(open(path, encoding="utf-8"))
        self.assertIn("prime", data)
        self.assertAlmostEqual(data["prime"], 0.5)
        self.assertEqual(data["state"], "prime")
        self.assertTrue(data["visible"])
        self.assertEqual((data["x"], data["y"]), (640, 360))

    def test_engaged_publish_has_zero_prime(self):
        mod = self._load()
        import json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "air_cursor_state.json")
            with mock.patch.object(mod, "AIR_CURSOR_STATE_FILE", path):
                dec = mod.AirMouseDecision(cursor=(100, 200), left=None, right=None,
                                           overlay="track", hand="right", grip="open")
                mod._publish_overlay_state(dec, visible=True)
                data = json.load(open(path, encoding="utf-8"))
        self.assertEqual(data["prime"], 0.0)
        self.assertEqual(data["state"], "track")

    def test_state_getter_exposes_prime(self):
        mod = self._load()
        mod._set_air_mouse_state(False, "open", None, prime=0.7)
        st = mod.get_air_mouse_state()
        self.assertAlmostEqual(st["prime"], 0.7)
        self.assertAlmostEqual(mod.get_air_mouse_prime(), 0.7)
        # Engaged clears prime.
        mod._set_air_mouse_state(True, "open", "right", prime=0.0)
        self.assertEqual(mod.get_air_mouse_prime(), 0.0)


class PollSmartEngageTests(_Base):
    """End-to-end through _poll_once: passive dwell primes then engages; the poll
    passes facing + armed + per-app-disable into the controller."""

    def _ctrl(self, mod, clk, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        kw.setdefault("engage_debounce_frames", 999)   # clock-timed only
        kw.setdefault("dwell_sec", 0.30)
        kw.setdefault("require_open_palm", True)
        kw.setdefault("arm_relaxes_gate", True)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), clock=clk, **kw)

    def test_passive_poll_primes_then_engages(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        mod.air_mouse_disarm()
        self.addCleanup(mod.air_mouse_disarm)
        moves, buttons = self._capture_mouse(mod)
        clk = _FakeClock(100.0)
        ctrl = self._ctrl(mod, clk)
        body = _body(reach_side="right", grip_right="open")   # facing True in fixture
        # Frame 1: valid pose but dwell not met → priming, no cursor move.
        mod._poll_once(ctrl, _fake_bridge(bodies=[body]))
        self.assertFalse(ctrl.engaged)
        self.assertEqual(moves, [])
        self.assertGreater(mod.get_air_mouse_prime(), 0.0)     # ring filling
        # Hold past the dwell → engage + move.
        clk.advance(0.35)
        mod._poll_once(ctrl, _fake_bridge(bodies=[body]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)
        self.assertEqual(mod.get_air_mouse_prime(), 0.0)       # engaged → no ring

    def test_armed_poll_engages_immediately_via_module_flag(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        clk = _FakeClock(100.0)
        ctrl = self._ctrl(mod, clk, arm_debounce_sec=0.0)   # no armed hold
        mod.air_mouse_arm()
        self.addCleanup(mod.air_mouse_disarm)
        # Armed + a raised hand → engages on the first poll (relaxed gate).
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)

    def test_poll_stands_down_over_disabled_app(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        clk = _FakeClock(100.0)
        ctrl = self._ctrl(mod, clk, arm_debounce_sec=0.0)
        mod.air_mouse_arm()
        self.addCleanup(mod.air_mouse_disarm)
        # Foreground app is disabled → the poll passes per_app_disabled True → no move.
        with mock.patch.object(mod, "_per_app_disabled", lambda: True):
            for _ in range(5):
                mod._poll_once(ctrl, _fake_bridge(
                    bodies=[_body(reach_side="right", grip_right="open")]))
        self.assertEqual(moves, [])
        self.assertFalse(ctrl.engaged)


# ─── registration wires the actions ─────────────────────────────────────────
class RegisterTests(_Base):
    def test_register_adds_actions_without_starting_real_thread(self):
        mod, actions = load_skill_isolated("kinect_air_mouse", register=True)
        for name in ("air_mouse_on", "air_mouse_off", "air_mouse_status",
                     "calibrate_air_mouse", "air_mouse_arm", "air_mouse_disarm"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_register_wires_arm_disarm_aliases(self):
        mod, actions = load_skill_isolated("kinect_air_mouse", register=True)
        for alias in ("mouse_control_on", "take_the_cursor", "give_me_the_cursor",
                      "hand_mouse_on"):
            self.assertIs(actions[alias], actions["air_mouse_arm"])
        for alias in ("mouse_control_off", "release_the_cursor", "hand_mouse_off"):
            self.assertIs(actions[alias], actions["air_mouse_disarm"])


if __name__ == "__main__":
    unittest.main()
