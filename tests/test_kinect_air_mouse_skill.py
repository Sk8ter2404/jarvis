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

import os
import sys
import tempfile
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


# Camera space: x sensor-right, y up, z depth AWAY from the sensor (metres). The
# torso sits at z≈TORSO_Z; an EXTENDED arm pushes the hand FORWARD (smaller z)
# AND straightens it; a RELAXED arm keeps the hand near the torso depth with a
# bent elbow (forearm folded back → low straightness).
TORSO_Z = 2.0


def _extended_arm_joints(side: str, *, hand_x=0.0, hand_y=0.30,
                         forward=0.30) -> dict:
    """Joints for one EXTENDED arm: hand pushed `forward` m toward the sensor
    (hand z = TORSO_Z - forward) with the elbow on the straight shoulder→hand
    line, so BOTH cues (forward-depth + straightness≈1) read as a clear reach."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, 0.40, TORSO_Z
    hx, hy, hz = hand_x, hand_y, TORSO_Z - forward
    # Elbow at the midpoint of the shoulder→hand segment → a straight arm
    # (chord == upper + fore, straightness == 1).
    ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (sz + hz) / 2.0
    return {
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


def _relaxed_arm_joints(side: str, *, hand_x=0.0, hand_y=0.30) -> dict:
    """Joints for one RELAXED arm: hand at the torso depth (no forward reach) with
    a sharply BENT elbow — the forearm folds back so the shoulder→hand chord is
    well short of the summed bone length (low straightness). Neither cue clears
    its engage bar → this arm does NOT engage."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, 0.40, TORSO_Z
    # Hand near torso depth (no forward reach) and only a touch below the shoulder.
    hx, hy, hz = hand_x, hand_y, TORSO_Z
    # Elbow well FORWARD of both shoulder and hand → a deep bend: the chord
    # (shoulder→hand, ~vertical/short) is far less than upper+fore, so
    # straightness is low (~0.5-0.6), comfortably under the 0.85 engage bar.
    ex, ey, ez = shoulder_x, sy - 0.05, TORSO_Z - 0.30
    return {
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


def _body(*, reach_side: "str | None" = "right", grip_right="open",
          grip_left="open", distance=TORSO_Z, hand_x=0.0, hand_y=0.30,
          forward=0.30, both_relaxed=False, with_joints=True):
    """A get_bodies()-shaped body. By default the `reach_side` arm is EXTENDED
    (engaged) and the other arm is RELAXED. both_relaxed=True makes BOTH arms
    relaxed (the merely-present / no-reach case → disengaged). with_joints=False
    omits the joint dict entirely (body tracked but no usable arm → disengaged).

    The torso reference joints (spine_mid / spine_shoulder) sit at the body depth
    so forward-reach is measured against them; the hand of the extended arm sits
    `forward` m in front of that."""
    joints: dict = {}
    if with_joints:
        joints["spine_mid"] = (0.0, 0.0, distance, 2)
        joints["spine_shoulder"] = (0.0, 0.15, distance, 2)
        joints["head"] = (0.0, 0.6, distance, 2)
        if both_relaxed:
            joints.update(_relaxed_arm_joints("left", hand_x=hand_x, hand_y=hand_y))
            joints.update(_relaxed_arm_joints("right", hand_x=hand_x, hand_y=hand_y))
        else:
            other = "left" if reach_side == "right" else "right"
            joints.update(_extended_arm_joints(
                reach_side, hand_x=hand_x, hand_y=hand_y, forward=forward))
            joints.update(_relaxed_arm_joints(other))
    return {
        "id": 0, "joints": joints,
        "head": (0.0, 0.6, distance), "distance_m": distance,
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
        for p in (p1, p2, p3, p4, p5, p6):
            p.start()
            self.addCleanup(p.stop)
        return moves, buttons


# ══════════════════════════════════════════════════════════════════════════
#  ARM-EXTENSION geometry + ArmExtension value object (the REACH signal)
# ══════════════════════════════════════════════════════════════════════════
class ArmExtensionTests(_Base):
    def _ext(self, mod, joints, side):
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(joints, side))

    def test_extended_arm_reads_forward_and_straight(self):
        mod = self._load()
        ext = self._ext(mod, _extended_arm_joints("right", forward=0.30), "right")
        # Hand pushed ~0.30 m forward of the torso, arm near-straight (~1.0).
        self.assertIsNotNone(ext.forward_m)
        self.assertGreater(ext.forward_m, 0.20)
        self.assertIsNotNone(ext.straightness)
        self.assertGreater(ext.straightness, 0.9)

    def test_relaxed_arm_reads_not_forward_and_bent(self):
        mod = self._load()
        ext = self._ext(mod, _relaxed_arm_joints("right"), "right")
        # No forward reach (hand at torso depth → ~0) and a bent arm (< engage).
        self.assertLess(ext.forward_m or 0.0, mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M)
        self.assertLess(ext.straightness or 1.0, mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE)

    def test_is_extended_true_only_when_reaching(self):
        mod = self._load()
        reaching = self._ext(mod, _extended_arm_joints("right"), "right")
        relaxed = self._ext(mod, _relaxed_arm_joints("right"), "right")
        self.assertTrue(reaching.is_extended(engaged=False))
        self.assertFalse(relaxed.is_extended(engaged=False))

    def test_forward_cue_alone_engages(self):
        mod = self._load()
        # The forward-reach cue present (no straightness reading at all), well past
        # the bar → extended. A missing straightness must NOT veto the engage (we
        # don't strand a clear reach when the elbow joint couldn't be measured).
        ext = mod.ArmExtension("right", forward_m=0.30, straightness=None,
                               hand=(0, 0.3, 1.7, 2))
        self.assertTrue(ext.is_extended(engaged=False))

    def test_straightness_alone_does_NOT_engage(self):
        """THE HEADLINE FIX: a straight arm with NO forward reach must NEVER engage.
        The OLD gate (fwd_ok OR straight_ok) let a straight-but-not-reaching arm
        latch the cursor — the 'stuck in a dark room, couldn't disengage' bug. The
        forward-reach ratio is now PRIMARY + NECESSARY, so straightness alone fails.
        """
        mod = self._load()
        # Straight (0.95) but NO forward reach: ratio path → ratio 0.0 < engage bar;
        # metres-fallback path → forward_m below the bar. Either way: NOT extended.
        ratio_arm = mod.ArmExtension("right", forward_m=0.0, straightness=0.95,
                                     hand=(0, 0.3, 2.0, 2), reach_ratio=0.0)
        self.assertFalse(ratio_arm.is_extended(engaged=False))
        # And it can't HOLD the gate either: even already-engaged, a collapsed ratio
        # with a perfectly straight arm releases (straightness can't latch).
        self.assertFalse(ratio_arm.is_extended(engaged=True))
        # Same with the absolute-metres fallback (no body scale → reach_ratio None).
        metres_arm = mod.ArmExtension("right", forward_m=0.0, straightness=0.95,
                                      hand=(0, 0.3, 2.0, 2))
        self.assertFalse(metres_arm.is_extended(engaged=False))
        self.assertFalse(metres_arm.is_extended(engaged=True))

    def test_straight_but_low_reach_arm_disengages_log_bug(self):
        """The EXACT live-log failure: every frame reach_ratio is BELOW the
        disengage bar while straightness ~0.90 is high. The OLD OR-latch kept it
        engaged=True forever; the new ratio-primary gate DISENGAGES (returns
        not-extended) the instant the ratio drops under the disengage bar,
        regardless of straightness."""
        mod = self._load()
        # ratio well under the 0.40 disengage bar, straightness high (the latch).
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.90,
                               hand=(0, 0.3, 1.6, 2), reach_ratio=0.30)
        # Currently ENGAGED, yet the low ratio releases it (straightness can't hold).
        self.assertFalse(ext.is_extended(engaged=True))

    def test_forward_but_bent_arm_vetoed_on_engage(self):
        """A hand shoved forward (clear reach ratio) with a SHARPLY BENT elbow
        (low straightness) must NOT engage: straightness is a secondary veto on the
        rising edge so a bent forward arm is rejected."""
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.45,
                               hand=(0, 0.3, 1.6, 2), reach_ratio=0.85)
        self.assertFalse(ext.is_extended(engaged=False))   # veto rejects the bend
        # But the SAME bent arm, once already engaged, is NOT re-vetoed on
        # straightness — only the ratio keeps/releases it (a momentary straightness
        # wobble must not drop a live reach). Ratio 0.85 ≥ disengage → stays.
        self.assertTrue(ext.is_extended(engaged=True))

    def test_genuine_reach_engages_at_new_bars(self):
        """A genuine reach at the owner's REAL range (ratio ~0.85, well under the
        old unreachable 1.6 bar) engages at the new 0.65 engage bar."""
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.40, straightness=0.95,
                               hand=(0, 0.3, 1.6, 2), reach_ratio=0.85)
        self.assertTrue(ext.is_extended(engaged=False))

    def test_relaxed_ratio_zero_disengages(self):
        """A fully relaxed arm gives forward_reach→~0 → ratio→~0, which is well
        below the 0.40 disengage bar → DISENGAGED even while currently engaged."""
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.0, straightness=0.88,
                               hand=(0, 0.3, 2.0, 2), reach_ratio=0.0)
        self.assertFalse(ext.is_extended(engaged=True))

    def test_extension_hysteresis(self):
        mod = self._load()
        # A reach between the disengage and engage bars: stays disengaged when
        # not engaged, but COUNTS as extended once already engaged (so it doesn't
        # flap at the line).
        mid = (mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M
               + mod.AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M) / 2.0
        ext = mod.ArmExtension("right", forward_m=mid, straightness=0.0,
                               hand=(0, 0.3, 1.8, 2))
        self.assertFalse(ext.is_extended(engaged=False))   # must clear higher bar
        self.assertTrue(ext.is_extended(engaged=True))     # lower bar to hold

    def test_choose_more_extended_arm(self):
        mod = self._load()
        left = mod.ArmExtension("left", forward_m=0.22, straightness=0.86,
                                hand=(-0.1, 0.3, 1.78, 2))
        right = mod.ArmExtension("right", forward_m=0.40, straightness=0.97,
                                 hand=(0.1, 0.3, 1.60, 2))
        chosen = mod.choose_controlling_arm(left, right, engaged=False)
        self.assertIs(chosen, right)                       # more extended

    def test_choose_none_when_neither_extended(self):
        mod = self._load()
        left = mod.ArmExtension("left", forward_m=0.0, straightness=0.4,
                                hand=(-0.1, 0.3, 2.0, 2))
        right = mod.ArmExtension("right", forward_m=0.05, straightness=0.5,
                                 hand=(0.1, 0.3, 1.95, 2))
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

    def test_extension_thresholds_have_hysteresis(self):
        mod = self._load()
        # The engage bar must be strictly higher than the disengage bar on ALL
        # cues, so an arm at the line can't flap engage on/off.
        self.assertGreater(mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE,
                           mod.AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE)
        self.assertGreater(mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
                           mod.AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M)
        self.assertGreater(mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
                           mod.AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE)

    def test_reach_ratio_defaults_are_sane_position_independent(self):
        mod = self._load()
        # Body-relative bars TUNED TO THE OWNER'S REAL RANGE: the Kinect sits UNDER
        # the monitor, so reaching at the screen tops out around ratio ~0.8-0.9 —
        # the OLD 1.6 engage bar was unreachable (only the removed straightness latch
        # could clear it). engage 0.65 (a genuine reach ~0.8-0.9 clears it), holding
        # to 0.40. Sane + usable WITHOUT calibration.
        self.assertEqual(mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE, 0.65)
        self.assertEqual(mod.AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE, 0.40)
        # A relaxed hand (ratio ~0) is well below the disengage bar → always
        # releases; a genuine reach (~0.8-0.9 > 0.65) engages.
        self.assertGreater(mod.AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE, 0.0)
        self.assertLess(mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE, 0.90)

    def test_straightness_is_only_a_modest_engage_veto(self):
        mod = self._load()
        # Straightness is no longer an independent engage cue: the engage-veto floor
        # is modest (~0.6) so a normal reach (straightness ~0.9-1.0) clears it, and
        # it can never latch the gate on its own (asserted in ArmExtension tests).
        self.assertEqual(mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE, 0.60)
        self.assertLessEqual(mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE, 0.7)


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
                                      debounce_frames=1, grace_sec=0.0)
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
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=debounce, grace_sec=0.0)

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
        # Move the extended hand while closed → cursor still tracks (drag),
        # button stays down (no new edge).
        d = c.update(self._relaxed(mod, "left"),
                     self._ext(mod, "right", hand_x=0.2, hand_y=0.1),
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
                                   grace_sec=0.0)
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
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    def _ext(self, mod, side, **kw):
        j = _extended_arm_joints(side, **kw)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def _relaxed(self, mod, side):
        j = _relaxed_arm_joints(side)
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(j, side))

    def test_merely_raised_hand_does_not_engage(self):
        mod = self._load()
        c = self._ctrl(mod)
        # A hand HELD HIGH (hand_y well up) but NOT reaching — relaxed arm geometry
        # (elbow bent, no forward push). The OLD model engaged here; the NEW one
        # must NOT: merely raising the hand drives ZERO cursor.
        raised_relaxed_l = mod.ArmExtension.from_bridge(
            mod._local_arm_extension(_relaxed_arm_joints("left", hand_y=0.7), "left"))
        raised_relaxed_r = mod.ArmExtension.from_bridge(
            mod._local_arm_extension(_relaxed_arm_joints("right", hand_y=0.7), "right"))
        d = c.update(raised_relaxed_l, raised_relaxed_r, "open", "open", True)
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
        # A reach BETWEEN the engage and disengage forward bars.
        mid = (mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M
               + mod.AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M) / 2.0

        def mid_arm():
            return mod.ArmExtension("right", forward_m=mid, straightness=0.0,
                                    hand=(0.1, 0.30, TORSO_Z - mid, 2))
        relaxed_l = self._relaxed(mod, "left")
        # While disengaged, the mid reach is NOT enough to engage (higher bar).
        d0 = c.update(relaxed_l, mid_arm(), "open", "open", True)
        self.assertFalse(c.engaged)
        self.assertIsNone(d0.cursor)
        # A clear reach engages.
        d1 = c.update(relaxed_l, self._ext(mod, "right"), "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d1.cursor)
        # Sag back to the mid reach: STAYS engaged (lower disengage bar) → no flicker.
        d2 = c.update(relaxed_l, mid_arm(), "open", "open", True)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d2.cursor)
        # Fully relax → disengage.
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
#  POSITION-INDEPENDENT REACH (the headline fix): engage/disengage is invariant
#  to the owner's distance from the sensor — gated on a BODY-RELATIVE reach RATIO
#  (forward reach / shoulder width), not absolute metres.
# ══════════════════════════════════════════════════════════════════════════
def _scaled_body_joints(side: str, *, scale: float, reach_frac: float,
                        shoulder_half: float = 0.20):
    """Build one arm's joints at an arbitrary BODY SCALE, with the hand reaching
    `reach_frac` × shoulder-width forward. `scale` multiplies ALL absolute metres
    (positions + depth offsets) to simulate the SAME body at a different distance:
    a closer body reads larger spans + bigger absolute reach, a farther one
    smaller — but the reach/shoulder-width RATIO is identical (= reach_frac). The
    position-independent gate must treat them the same. Shoulders span
    2*shoulder_half*scale; the forward reach is reach_frac*(shoulder width).

    The elbow tracks the reach REALISTICALLY: a big reach (high reach_frac) gives a
    near-straight arm (elbow on the shoulder→hand line); a small reach (relaxed)
    folds the elbow back so straightness is LOW too — so a relaxed pose fails BOTH
    the ratio and straightness cues, the same as a real relaxed arm."""
    base_z = 2.0 * scale
    sh = shoulder_half * scale
    shoulder_width = 2.0 * sh
    forward = reach_frac * shoulder_width          # reach as a fraction of span
    sx = -sh if side == "left" else sh
    other = sh if side == "left" else -sh
    sy = 0.40 * scale
    hx, hy, hz = (sx * 2.0), 0.30 * scale, base_z - forward
    if reach_frac >= 1.0:
        # Extended: straight arm — elbow on the midpoint of shoulder→hand.
        ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (base_z + hz) / 2.0
    else:
        # Relaxed: BENT elbow folded forward so the chord is well short of the
        # summed bone length (low straightness), matching a real relaxed arm.
        ex, ey, ez = sx, sy - 0.10 * scale, base_z - 0.30 * scale
    return {
        "spine_shoulder": (0.0, 0.15 * scale, base_z, 2),
        "spine_mid": (0.0, 0.0, base_z, 2),
        "spine_base": (0.0, -0.30 * scale, base_z, 2),
        f"shoulder_{side}": (sx, sy, base_z, 2),
        f"shoulder_{'right' if side == 'left' else 'left'}": (other, sy, base_z, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


class ReachRatioPositionIndependenceTests(_Base):
    """The reach gate is BODY-RELATIVE, so a reach that engages up close also
    engages far away (and a relax releases at any distance). Exercised both on the
    geometry (reach_ratio) and through the engage decision."""

    def _ext(self, mod, joints, side):
        return mod.ArmExtension.from_bridge(mod._local_arm_extension(joints, side))

    def test_local_extension_computes_body_relative_ratio(self):
        mod = self._load()
        # A 0.75-of-shoulder-width reach → reach_ratio ≈ 0.75, regardless of scale.
        for scale in (0.6, 1.0, 1.7):
            j = _scaled_body_joints("right", scale=scale, reach_frac=0.75)
            ext = self._ext(mod, j, "right")
            self.assertIsNotNone(ext.reach_ratio)
            self.assertAlmostEqual(ext.reach_ratio, 0.75, delta=0.06)
            # The ABSOLUTE forward metres DO scale with distance (proving the ratio
            # is doing real work) — bigger scale, bigger absolute reach.
            self.assertIsNotNone(ext.forward_m)

    def test_absolute_metres_differ_but_ratio_constant_across_distance(self):
        mod = self._load()
        near = self._ext(mod, _scaled_body_joints("right", scale=1.6, reach_frac=1.8),
                         "right")
        far = self._ext(mod, _scaled_body_joints("right", scale=0.6, reach_frac=1.8),
                        "right")
        # Same gesture (1.8× shoulder width) at two distances: absolute metres very
        # different, the body-relative ratio the SAME.
        self.assertNotAlmostEqual(near.forward_m, far.forward_m, delta=0.05)
        self.assertAlmostEqual(near.reach_ratio, far.reach_ratio, delta=0.05)

    def test_engage_invariant_to_body_distance(self):
        mod = self._load()
        th = mod._reach_thresholds()   # position-independent ratio defaults
        # A clear reach (ratio ~1.8 > the 0.65 engage bar) must read EXTENDED at
        # every distance; a relaxed reach (ratio ~0.3 < 0.40 disengage) must NOT.
        for scale in (0.5, 1.0, 1.5, 2.2):
            reach = self._ext(mod, _scaled_body_joints(
                "right", scale=scale, reach_frac=1.8), "right")
            relax = self._ext(mod, _scaled_body_joints(
                "right", scale=scale, reach_frac=0.3), "right")
            # Straightness is geometry-driven here; force the test onto the RATIO
            # cue by checking is_extended with straightness neutralised.
            reach_only_ratio = mod.ArmExtension(
                "right", forward_m=reach.forward_m, straightness=None,
                hand=reach.hand, reach_ratio=reach.reach_ratio)
            relax_only_ratio = mod.ArmExtension(
                "right", forward_m=relax.forward_m, straightness=None,
                hand=relax.hand, reach_ratio=relax.reach_ratio)
            self.assertTrue(
                reach_only_ratio.is_extended(engaged=False, thresholds=th),
                f"clear reach should engage at scale {scale}")
            self.assertFalse(
                relax_only_ratio.is_extended(engaged=False, thresholds=th),
                f"relaxed arm should NOT engage at scale {scale}")

    def test_controller_engages_far_and_near_via_poll(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        # Mirror OFF so reach_side='right' stays the right arm end-to-end.
        moves, buttons = self._capture_mouse(mod)
        ctrl = mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0)

        def _body_at(scale, reach_frac):
            j = _scaled_body_joints("right", scale=scale, reach_frac=reach_frac)
            # Other (left) arm relaxed at the same scale so only the right reaches.
            jl = _scaled_body_joints("left", scale=scale, reach_frac=0.2)
            j["elbow_left"] = jl["elbow_left"]
            j["hand_left"] = jl["hand_left"]
            return {"id": 0, "joints": j,
                    "head": (0.0, 0.6 * scale, 2.0 * scale),
                    "distance_m": 2.0 * scale, "facing": True,
                    "hand_right": "open", "hand_left": "open"}

        # FAR (small scale): a clear reach must still move the cursor.
        moves.clear()
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(0.6, 1.9)]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)
        # Relax at the SAME far distance → disengage (no further move).
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(0.6, 0.25)]))
        self.assertFalse(ctrl.engaged)
        # NEAR (large scale): the same clear reach engages too.
        moves.clear()
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body_at(1.8, 1.9)]))
        self.assertTrue(ctrl.engaged)
        self.assertEqual(len(moves), 1)

    def test_relax_releases_regardless_of_body_position(self):
        # The owner's exact bug: worked, then on standing up + sitting back it would
        # NOT disengage. With the body-relative ratio, a relax at ANY distance drops
        # below the disengage bar → release. Simulate: engage near, then "stand up +
        # sit back" = relax at a DIFFERENT (far) scale → must disengage.
        mod = self._load()
        th = mod._reach_thresholds()
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1,
                                   grace_sec=0.0)
        relaxed_left = mod.ArmExtension.from_bridge(
            mod._local_arm_extension(_relaxed_arm_joints("left"), "left"))

        def _reach(scale, frac):
            return mod.ArmExtension.from_bridge(mod._local_arm_extension(
                _scaled_body_joints("right", scale=scale, reach_frac=frac), "right"))

        # Engage with a clear reach up close.
        c.update(relaxed_left, _reach(1.6, 1.9), "open", "open", True,
                 thresholds=th)
        self.assertTrue(c.engaged)
        # Now relax — but the body is also farther away (smaller scale), exactly the
        # stand-up→sit-back perturbation. Absolute forward metres shrink, yet the
        # RATIO drops below the disengage bar → it MUST release.
        d = c.update(relaxed_left, _reach(0.7, 0.25), "open", "open", True,
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
                                      debounce_frames=1, grace_sec=0.0)

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
                                      debounce_frames=1, grace_sec=0.0)

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
                                      debounce_frames=1, grace_sec=0.0)

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

    def test_merely_raised_hand_makes_zero_setcursorpos(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)

        def raised_body():
            b = _body(both_relaxed=True)
            # Lift BOTH (relaxed-geometry) hands high — raised but not reaching.
            for side in ("left", "right"):
                b["joints"].update(_relaxed_arm_joints(side, hand_y=0.75))
            return b
        for _ in range(20):
            mod._poll_once(ctrl, _fake_bridge(bodies=[raised_body()]))
        self.assertEqual(moves, [])          # raised != reach → no cursor

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

    def test_on_message_describes_reach_and_per_hand_clicks(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=True))
        out = mod.air_mouse_on("").lower()
        # The spoken help reflects the NEW model: reach out, left/right hand.
        self.assertIn("reach", out)
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
                                      debounce_frames=1, grace_sec=0.0)
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
    """The fit now produces BODY-RELATIVE reach-RATIO bars (position-independent),
    not absolute metres: the first pose pair is the relaxed/extended RATIO."""

    def test_bars_placed_between_relaxed_and_extended(self):
        mod = self._load()
        # Relaxed ratio 0.00, extended 2.00 → engage ~60 % (1.20), disengage ~40 %
        # (0.80); engage strictly above disengage (hysteresis).
        th = mod.compute_reach_thresholds(0.0, 2.0, 0.50, 0.95)
        self.assertAlmostEqual(th["ratio_engage"], 1.20, delta=0.02)
        self.assertAlmostEqual(th["ratio_disengage"], 0.80, delta=0.02)
        self.assertGreater(th["ratio_engage"], th["ratio_disengage"])
        # The absolute fwd_* fallback bars are always the module defaults.
        self.assertEqual(th["fwd_engage"], mod.AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M)
        self.assertEqual(th["fwd_disengage"],
                         mod.AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M)
        # Straightness 0.50→0.95 span: engage 0.50+0.6*0.45=0.77, disengage 0.68.
        self.assertAlmostEqual(th["straight_engage"], 0.77, delta=0.01)
        self.assertAlmostEqual(th["straight_disengage"], 0.68, delta=0.01)
        self.assertGreater(th["straight_engage"], th["straight_disengage"])

    def test_missing_pose_falls_back_to_defaults(self):
        mod = self._load()
        # No ratio pair → ratio bars keep the module defaults; straightness pair
        # present → its bars are fitted.
        th = mod.compute_reach_thresholds(None, None, 0.40, 0.96)
        self.assertEqual(th["ratio_engage"],
                         mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE)
        self.assertEqual(th["ratio_disengage"],
                         mod.AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE)
        self.assertNotEqual(th["straight_engage"],
                            mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE)

    def test_degenerate_span_keeps_defaults(self):
        mod = self._load()
        # Extended barely above relaxed (no real RATIO span ≥ 0.30) → defaults,
        # never an inverted/nonsense threshold from a flubbed capture.
        th = mod.compute_reach_thresholds(0.90, 1.00, 0.80, 0.81)
        self.assertEqual(th["ratio_engage"],
                         mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE)
        self.assertEqual(th["straight_engage"],
                         mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE)


class MedianTests(_Base):
    def test_median_odd_even_and_empty(self):
        mod = self._load()
        self.assertEqual(mod._median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(mod._median([1.0, 2.0, 3.0, 4.0]), 2.5)
        self.assertIsNone(mod._median([]))


class ReachThresholdReadTests(_Base):
    """The live gate reads the persisted KINECT_REACH_* (calibration), falling
    back to the module defaults per-field when unset."""

    def _patch_settings(self, mod, saved):
        p = mock.patch.object(mod, "_saved_settings", lambda: dict(saved))
        p.start(); self.addCleanup(p.stop)

    def test_defaults_when_nothing_persisted(self):
        mod = self._load()
        self._patch_settings(mod, {})
        th = mod._reach_thresholds()
        # The position-independent ratio defaults apply pre-calibration.
        self.assertEqual(th["ratio_engage"],
                         mod.AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE)
        self.assertEqual(th["straight_disengage"],
                         mod.AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE)

    def test_persisted_values_win(self):
        mod = self._load()
        # KINECT_REACH_* now hold the body-relative REACH RATIO bars.
        self._patch_settings(mod, {
            mod.SETTING_REACH_ENGAGE: 1.7,
            mod.SETTING_REACH_DISENGAGE: 1.1,
            mod.SETTING_STRAIGHT_ENGAGE: 0.82,
            mod.SETTING_STRAIGHT_DISENGAGE: 0.70,
        })
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["ratio_engage"], 1.7)
        self.assertAlmostEqual(th["ratio_disengage"], 1.1)
        self.assertAlmostEqual(th["straight_engage"], 0.82)
        self.assertAlmostEqual(th["straight_disengage"], 0.70)

    def test_partial_persist_falls_back_per_field(self):
        mod = self._load()
        self._patch_settings(mod, {mod.SETTING_REACH_ENGAGE: 1.9})
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["ratio_engage"], 1.9)            # persisted
        self.assertEqual(th["straight_engage"],                    # defaulted
                         mod.AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE)


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
        """Make _capture_reach return each (fwd, straight, n) in `sequence` on
        successive calls (no real sampling / sleeping)."""
        calls = {"i": 0}

        def _fake(bridge, *a, **k):
            i = min(calls["i"], len(sequence) - 1)
            calls["i"] += 1
            return sequence[i]
        p = mock.patch.object(mod, "_capture_reach", _fake)
        p.start(); self.addCleanup(p.stop)
        return calls

    def test_calibration_sets_thresholds_from_captured_poses(self):
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
        # First capture = EXTENDED pose (0.40 m, 0.95 straight); second = RELAXED
        # (0.00 m, 0.50 straight), each with usable frames.
        self._stub_captures(mod, [(0.40, 0.95, 20), (0.00, 0.50, 20)])
        out = mod.calibrate_air_mouse("")
        self.assertIn("calibrat", out.lower())
        # Persisted the four fitted thresholds, engage above disengage.
        self.assertIn(mod.SETTING_REACH_ENGAGE, saved)
        self.assertIn(mod.SETTING_REACH_DISENGAGE, saved)
        self.assertGreater(saved[mod.SETTING_REACH_ENGAGE],
                           saved[mod.SETTING_REACH_DISENGAGE])
        # The fitted forward-engage sits ~60 % between 0.00 and 0.40 (≈0.24).
        self.assertAlmostEqual(saved[mod.SETTING_REACH_ENGAGE], 0.24, delta=0.02)
        # It spoke BOTH prompts (extend, then relax).
        self.assertTrue(any("extend" in s.lower() for s in spoken))
        self.assertTrue(any("relax" in s.lower() for s in spoken))

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
        # Captures are now relaxed/extended RATIO (extended 0.50, relaxed 0.10 →
        # a clear ratio span ≥ 0.30, so the ratio bars fit + persist).
        self._stub_captures(mod, [(0.50, 0.98, 20), (0.10, 0.55, 20)])
        mod.calibrate_air_mouse("")
        # _saved_settings reads the same mocked writer → the gate sees the fit. The
        # KINECT_REACH_ENGAGE key holds the body-relative ratio engage bar.
        th = mod._reach_thresholds()
        self.assertAlmostEqual(th["ratio_engage"], saved[mod.SETTING_REACH_ENGAGE])
        self.assertGreater(th["ratio_engage"], th["ratio_disengage"])

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
        # Zero usable frames on the first (extended) capture → honest failure.
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

    def test_capture_reach_medians_from_more_extended_arm(self):
        # _capture_reach samples the more-extended arm's body-relative reach RATIO
        # + straightness and medians them. The fixture's shoulders span ~0.4 m, so
        # a 0.30 m forward reach gives a ratio ≈ 0.75 (position-independent).
        mod = self._load()
        bridge = _fake_bridge(bodies=[_body(reach_side="right", forward=0.30)])
        ratio, straight, n = mod._capture_reach(
            bridge, seconds=0.05, sleep_fn=lambda s: None,
            now_fn=_StepClock(step=0.02))
        self.assertGreater(n, 0)
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 0.4)          # the extended arm's reach ratio
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
    """ISSUE 2b — the ~2 Hz live reach debug line shows the values for tuning."""

    class _Ctrl:
        engaged = False
        hand = None

    def test_debug_line_shape(self):
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.18, straightness=0.91,
                               hand=(0.1, 0.3, 1.8, 2))
        line = mod._format_reach_debug(None, ext, True, self._Ctrl())
        self.assertIn("[air-mouse]", line)
        self.assertIn("reach=0.18", line)
        self.assertIn("straight=0.91", line)
        self.assertIn("hand=right", line)
        self.assertIn("engaged=False", line)

    def test_debug_line_handles_missing_cues(self):
        mod = self._load()
        line = mod._format_reach_debug(None, None, False, self._Ctrl())
        self.assertIn("reach=n/a", line)
        self.assertIn("hand=none", line)

    def test_debug_log_is_throttled(self):
        mod = self._load()
        ext = mod.ArmExtension("right", forward_m=0.2, straightness=0.9,
                               hand=(0, 0.3, 1.8, 2))
        mod._air_mouse_debug_last[0] = 0.0
        # First call at t=100 prints; an immediate second call is throttled.
        self.assertTrue(mod._maybe_debug_log(None, ext, True, self._Ctrl(),
                                             now=100.0))
        self.assertFalse(mod._maybe_debug_log(None, ext, True, self._Ctrl(),
                                              now=100.1))
        # Past the interval it prints again.
        self.assertTrue(mod._maybe_debug_log(None, ext, True, self._Ctrl(),
                                             now=100.0 + mod._AIR_MOUSE_DEBUG_INTERVAL))


# ══════════════════════════════════════════════════════════════════════════
#  ISSUE 3 — controlling-hand HYSTERESIS: no thrash with BOTH hands raised.
# ══════════════════════════════════════════════════════════════════════════
class ControllingHandHysteresisTests(_Base):
    def _ctrl(self, mod, **kw):
        kw.setdefault("debounce_frames", 1)
        kw.setdefault("grace_sec", 0.0)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    def _arm(self, mod, side, forward, straight):
        """An EXTENDED ArmExtension with a chosen reach (forward + straightness)
        and a hand joint, for driving the hysteresis directly."""
        return mod.ArmExtension(side, forward_m=forward, straightness=straight,
                                hand=(0.1 if side == "right" else -0.1, 0.3,
                                      TORSO_Z - forward, 2))

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
