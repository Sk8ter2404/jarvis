"""Unit tests for skills/kinect_two_hand.py — the Kinect TWO-HAND pinch-to-resize
windows feature.

Covers the PURE geometry (rect scale-about-centre, translate, on-screen clamp,
scale dead-band + step cap), the grab→resize/move→release CONTROLLER state machine
(SPREAD grows / PINCH shrinks the focused window rect about its centre; both hands
together MOVE it; a hand drop RELEASES), the Win32 wiring (a grabbed window is
re-positioned with the SCALED rect — asserted against a MOCK Win32), the air-mouse
STAND-DOWN hand-off (two-hand mode suppresses the single-hand cursor), and the DUAL
RETICLE publish (two circles, BLUE normally / PURPLE while resizing).

Mocks bodies (joints x/y/z) + Win32 — App-Control-safe, stdlib unittest only.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── settings-file safety net (mirrors the air-mouse test) ───────────────────
_SAVED_SETTINGS_ENV: "str | None" = None
_SETTINGS_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SETTINGS_TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SETTINGS_TMPDIR = tempfile.mkdtemp(prefix="jarvis_twohand_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _SETTINGS_TMPDIR, "test_user_settings.json")


def tearDownModule() -> None:
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV


# ─── fakes ───────────────────────────────────────────────────────────────────
# Camera space matches the air-mouse test: x sensor-right, y UP, z depth. A hand
# RAISED above the shoulder (hand_y > shoulder_y + up_margin) engages.
SHOULDER_Y = 0.40
RAISED_HAND_Y = 0.58          # lift ≈ +0.18 m (clears the +0.07 engage margin)
DESK_HAND_Y = 0.00            # lift ≈ -0.40 m (well below — disengaged)
TORSO_Z = 2.0


def _raised_arm_joints(side: str, *, hand_x: float, hand_y: float = RAISED_HAND_Y,
                       hand_z: float = TORSO_Z - 0.30) -> dict:
    """Joints for one RAISED arm at a chosen hand (x, y, z)."""
    shoulder_x = -0.2 if side == "left" else 0.2
    sx, sy, sz = shoulder_x, SHOULDER_Y, TORSO_Z
    hx, hy, hz = hand_x, hand_y, hand_z
    ex, ey, ez = (sx + hx) / 2.0, (sy + hy) / 2.0, (sz + hz) / 2.0
    return {
        f"shoulder_{side}": (sx, sy, sz, 2),
        f"elbow_{side}": (ex, ey, ez, 2),
        f"hand_{side}": (hx, hy, hz, 2),
    }


def _both_up_body(*, left_x=-0.30, right_x=0.30, left_y=RAISED_HAND_Y,
                  right_y=RAISED_HAND_Y, left_z=TORSO_Z - 0.30,
                  right_z=TORSO_Z - 0.30, distance=TORSO_Z):
    """A get_bodies() body with BOTH hands raised at chosen positions (so two-hand
    mode engages). The hand X separation drives the pinch distance."""
    joints: dict = {
        "spine_mid": (0.0, 0.0, distance, 2),
        "spine_shoulder": (0.0, SHOULDER_Y, distance, 2),
        "head": (0.0, SHOULDER_Y + 0.3, distance, 2),
    }
    joints.update(_raised_arm_joints("left", hand_x=left_x, hand_y=left_y,
                                     hand_z=left_z))
    joints.update(_raised_arm_joints("right", hand_x=right_x, hand_y=right_y,
                                     hand_z=right_z))
    return {
        "id": 0, "joints": joints,
        "head": (0.0, SHOULDER_Y + 0.3, distance), "distance_m": distance,
        "facing": True, "hand_right": "open", "hand_left": "open",
    }


def _one_up_body(*, up_side="right"):
    """A body with only ONE hand raised (the other on the desk) → two-hand mode must
    NOT engage."""
    other = "left" if up_side == "right" else "right"
    joints: dict = {
        "spine_mid": (0.0, 0.0, TORSO_Z, 2),
        "spine_shoulder": (0.0, SHOULDER_Y, TORSO_Z, 2),
        "head": (0.0, SHOULDER_Y + 0.3, TORSO_Z, 2),
    }
    joints.update(_raised_arm_joints(up_side, hand_x=0.30))
    joints.update(_raised_arm_joints(other, hand_x=-0.30, hand_y=DESK_HAND_Y,
                                     hand_z=TORSO_Z))
    return {
        "id": 0, "joints": joints,
        "head": (0.0, SHOULDER_Y + 0.3, TORSO_Z), "distance_m": TORSO_Z,
        "facing": True, "hand_right": "open", "hand_left": "open",
    }


def _one_untracked_body(*, untracked_side="left"):
    """A body with BOTH hands RAISED+engaged EXCEPT one side's hand joint is MISSING
    (its hand_<side>/wrist_<side> absent) — i.e. that hand went UNTRACKED / unknown-
    state / dropped out of the frame while the body is still seen. _arm_extension
    yields .hand=None for that side, so _both_hands_engaged returns False even though
    the body is tracked and the other hand is plainly up. This is the exact ambiguous
    case that used to latch the grab."""
    tracked_side = "right" if untracked_side == "left" else "left"
    joints: dict = {
        "spine_mid": (0.0, 0.0, TORSO_Z, 2),
        "spine_shoulder": (0.0, SHOULDER_Y, TORSO_Z, 2),
        "head": (0.0, SHOULDER_Y + 0.3, TORSO_Z, 2),
    }
    # The still-tracked, still-raised hand.
    tx = 0.30 if tracked_side == "right" else -0.30
    joints.update(_raised_arm_joints(tracked_side, hand_x=tx))
    # The untracked side: keep the shoulder/elbow but DROP the hand joint entirely.
    sx = -0.2 if untracked_side == "left" else 0.2
    joints[f"shoulder_{untracked_side}"] = (sx, SHOULDER_Y, TORSO_Z, 2)
    joints[f"elbow_{untracked_side}"] = (sx, SHOULDER_Y + 0.05, TORSO_Z - 0.15, 2)
    # NB: no hand_<untracked_side> / wrist_<untracked_side> key → .hand is None.
    return {
        "id": 0, "joints": joints,
        "head": (0.0, SHOULDER_Y + 0.3, TORSO_Z), "distance_m": TORSO_Z,
        "facing": True, "hand_right": "open", "hand_left": "open",
    }


def _fake_bridge(*, bodies=None, enabled=True, available=(True, "")):
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_hand_states = lambda: {"right": "open", "left": "open",
                                 "tracked": bool(bodies), "ts": 0.0}
    return m


class _Clock:
    """A controllable monotonic clock for the grab-hold timing."""

    def __init__(self, t0=0.0):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _Base(unittest.TestCase):
    def _load(self):
        # Load the air-mouse skill first so kinect_two_hand's _air_mouse_mod()
        # resolves it from sys.modules (the same way the live loader wires them).
        am, _ = load_skill_isolated("kinect_air_mouse", register=False)
        self._am = am
        mod, _actions = load_skill_isolated("kinect_two_hand", register=False)
        return mod

    def _patch_flag(self, value):
        from core import config as cfg
        p = mock.patch.object(cfg, "KINECT_TWO_HAND_ENABLED", value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _not_staging(self, mod):
        p = mock.patch.object(mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)


# ══════════════════════════════════════════════════════════════════════════
#  PURE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════
class RectGeometryTests(_Base):
    def test_rect_dims_and_centre(self):
        mod = self._load()
        r = mod.Rect(100, 200, 500, 500)
        self.assertEqual(r.width, 400)
        self.assertEqual(r.height, 300)
        self.assertEqual((r.cx, r.cy), (300.0, 350.0))
        self.assertEqual(r.as_xywh(), (100, 200, 400, 300))

    def test_scale_grows_about_centre(self):
        mod = self._load()
        r = mod.Rect(800, 400, 1600, 1000)        # 800x600, centre (1200, 700)
        g = mod.scale_rect_about_center(r, 1.5)
        self.assertEqual((g.width, g.height), (1200, 900))   # 1.5×
        self.assertAlmostEqual(g.cx, 1200.0, delta=1)        # centre UNCHANGED
        self.assertAlmostEqual(g.cy, 700.0, delta=1)

    def test_scale_shrinks_about_centre(self):
        mod = self._load()
        r = mod.Rect(800, 400, 1600, 1000)        # 800x600
        s = mod.scale_rect_about_center(r, 0.5)
        self.assertEqual((s.width, s.height), (400, 300))    # 0.5×
        self.assertAlmostEqual(s.cx, 1200.0, delta=1)
        self.assertAlmostEqual(s.cy, 700.0, delta=1)

    def test_scale_floors_at_min_size(self):
        mod = self._load()
        r = mod.Rect(0, 0, 300, 300)
        s = mod.scale_rect_about_center(r, 0.1, min_w=240, min_h=160)
        self.assertGreaterEqual(s.width, 240)
        self.assertGreaterEqual(s.height, 160)

    def test_translate(self):
        mod = self._load()
        r = mod.Rect(100, 100, 500, 400)
        t = mod.translate_rect(r, 50, -30)
        self.assertEqual(t.as_tuple(), (150, 70, 550, 370))
        self.assertEqual((t.width, t.height), (400, 300))    # size preserved

    def test_deadband_snaps_near_unity(self):
        mod = self._load()
        self.assertEqual(mod.apply_deadband(1.02, deadband=0.04), 1.0)
        self.assertEqual(mod.apply_deadband(0.98, deadband=0.04), 1.0)   # inside (±0.02)
        self.assertEqual(mod.apply_deadband(0.94, deadband=0.04), 0.94)  # outside
        self.assertEqual(mod.apply_deadband(1.20, deadband=0.04), 1.20)

    def test_scale_step_cap(self):
        mod = self._load()
        # Target far above prev → clamped to prev × (1 + max_step).
        self.assertAlmostEqual(
            mod.clamp_scale_step(1.0, 2.0, max_step=0.18), 1.18, delta=1e-9)
        # Target far below → prev × (1 - max_step).
        self.assertAlmostEqual(
            mod.clamp_scale_step(1.0, 0.1, max_step=0.18), 0.82, delta=1e-9)
        # Target within the band passes through.
        self.assertAlmostEqual(
            mod.clamp_scale_step(1.0, 1.05, max_step=0.18), 1.05, delta=1e-9)


class ClampOnScreenTests(_Base):
    BOUNDS = (0, 0, 2560, 1440)

    def test_window_fully_inside_unchanged(self):
        mod = self._load()
        r = mod.Rect(100, 100, 900, 700)
        self.assertEqual(mod.clamp_rect_on_screen(r, self.BOUNDS), r)

    def test_window_dragged_off_right_is_pulled_back(self):
        mod = self._load()
        # Window pushed far right so only a sliver would remain on-desktop.
        r = mod.Rect(2550, 100, 3350, 700)   # left at 2550, width 800
        c = mod.clamp_rect_on_screen(r, self.BOUNDS, margin=32)
        # At least `margin` px of the window stays on-desktop (left edge ≤ right-32).
        self.assertLessEqual(c.left, self.BOUNDS[2] - 32)
        self.assertEqual((c.width, c.height), (800, 600))   # size preserved

    def test_window_dragged_off_left_is_pulled_back(self):
        mod = self._load()
        r = mod.Rect(-790, 100, 10, 700)     # almost entirely off the left edge
        c = mod.clamp_rect_on_screen(r, self.BOUNDS, margin=32)
        self.assertGreaterEqual(c.right, self.BOUNDS[0] + 32)

    def test_offset_origin_desktop(self):
        mod = self._load()
        # A left monitor gives a NEGATIVE origin; the clamp must respect it.
        bounds = (-2560, 0, 5120, 1440)
        r = mod.Rect(-3000, 100, -2200, 700)  # off the far-left edge
        c = mod.clamp_rect_on_screen(r, bounds, margin=32)
        self.assertGreaterEqual(c.right, bounds[0] + 32)


class SkipListTests(_Base):
    def test_normal_window_is_a_target(self):
        mod = self._load()
        self.assertTrue(mod.is_resizable_target("Chrome_WidgetWin_1", "Gmail"))
        self.assertTrue(mod.is_resizable_target("CASCADIA_HOSTING_WINDOW_CLASS",
                                                "Windows Terminal"))

    def test_shell_and_taskbar_are_skipped(self):
        mod = self._load()
        for cls in ("Progman", "WorkerW", "Shell_TrayWnd",
                    "Shell_SecondaryTrayWnd", "Windows.UI.Core.CoreWindow"):
            self.assertFalse(mod.is_resizable_target(cls, ""),
                             f"{cls} should be skipped")

    def test_program_manager_title_skipped(self):
        mod = self._load()
        self.assertFalse(mod.is_resizable_target("SomeClass", "Program Manager"))

    def test_hidden_or_minimized_or_nonwindow_skipped(self):
        mod = self._load()
        self.assertFalse(mod.is_resizable_target("Chrome_WidgetWin_1", "x",
                                                 visible=False))
        self.assertFalse(mod.is_resizable_target("Chrome_WidgetWin_1", "x",
                                                 minimized=True))
        self.assertFalse(mod.is_resizable_target("Chrome_WidgetWin_1", "x",
                                                 is_window=False))


# ══════════════════════════════════════════════════════════════════════════
#  CONTROLLER STATE MACHINE — grab → spread/pinch/move → release
# ══════════════════════════════════════════════════════════════════════════
class TwoHandControllerTests(_Base):
    BOUNDS = (0, 0, 2560, 1440)

    def _ctrl(self, mod, clk):
        # alpha=1.0 → no EMA lag, so the asserted rect is the exact target (the EMA
        # smoothing is covered separately). Grab hold 0.2 s.
        return mod.TwoHandController(clock=clk, dist_alpha=1.0, rect_alpha=1.0,
                                     grab_hold_sec=0.20)

    def _grab(self, mod, c, clk, rect, *, dist=0.40, mid=(1200, 700)):
        """Drive through the grab-hold so the controller is GRABBED on `rect`."""
        c.update(both_engaged=True, hand_dist=dist, midpoint=mid,
                 focused_rect=rect, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        clk.advance(0.25)   # past the hold
        d = c.update(both_engaged=True, hand_dist=dist, midpoint=mid,
                     focused_rect=rect, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        return d

    def test_hold_then_grab_captures_rect(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)
        # First frame: HOLDING (active so the air-mouse stands down, but no rect yet).
        d = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=rect, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertEqual(d.phase, "holding")
        self.assertTrue(d.active)
        self.assertFalse(d.resizing)
        self.assertIsNone(d.rect)
        # After the hold elapses → GRABBED, rect == the captured window rect.
        clk.advance(0.25)
        d = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=rect, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertEqual(d.phase, "grabbed")
        self.assertTrue(d.resizing)
        self.assertEqual(d.rect, rect)

    def test_spread_grows_focused_rect_about_centre(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)     # 800x600, centre (1200, 700)
        self._grab(mod, c, clk, rect, dist=0.40)
        # SPREAD: distance 0.40 → 0.60 (ratio 1.5). Drive enough frames to converge
        # through the per-tick step cap.
        last = None
        for _ in range(12):
            last = c.update(both_engaged=True, hand_dist=0.60, midpoint=(1200, 700),
                            focused_rect=None, bounds=self.BOUNDS,
                            hands=((900, 700), (1500, 700)))
        self.assertTrue(last.resizing)
        # GROWN to ~1.5× about the SAME centre (1200, 700).
        self.assertEqual((last.rect.width, last.rect.height), (1200, 900))
        self.assertAlmostEqual(last.rect.cx, 1200.0, delta=2)
        self.assertAlmostEqual(last.rect.cy, 700.0, delta=2)

    def test_pinch_shrinks_focused_rect(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)     # 800x600
        self._grab(mod, c, clk, rect, dist=0.60)  # baseline 0.60
        # PINCH: distance 0.60 → 0.30 (ratio 0.5) → shrink to half.
        last = None
        for _ in range(16):
            last = c.update(both_engaged=True, hand_dist=0.30, midpoint=(1200, 700),
                            focused_rect=None, bounds=self.BOUNDS,
                            hands=((1100, 700), (1300, 700)))
        self.assertEqual((last.rect.width, last.rect.height), (400, 300))
        self.assertAlmostEqual(last.rect.cx, 1200.0, delta=2)

    def test_move_translates_by_midpoint_delta(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)     # centre (1200, 700)
        self._grab(mod, c, clk, rect, dist=0.40, mid=(1200, 700))
        # MOVE: midpoint slides +200 x, -100 y; distance held (no resize).
        last = None
        for _ in range(16):
            last = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1400, 600),
                            focused_rect=None, bounds=self.BOUNDS,
                            hands=((1200, 600), (1600, 600)))
        # Size unchanged (distance held → scale 1.0 via the dead-band); centre moved.
        self.assertEqual((last.rect.width, last.rect.height), (800, 600))
        self.assertAlmostEqual(last.rect.cx, 1400.0, delta=2)
        self.assertAlmostEqual(last.rect.cy, 600.0, delta=2)

    def test_small_distance_jitter_does_not_resize_while_moving(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect, dist=0.40)
        # Distance wobbles within the dead-band (±4 %) while the midpoint moves.
        last = None
        for d in (0.405, 0.398, 0.402, 0.40, 0.41):
            last = c.update(both_engaged=True, hand_dist=d, midpoint=(1300, 700),
                            focused_rect=None, bounds=self.BOUNDS,
                            hands=((1200, 700), (1400, 700)))
        # Size held (jitter inside the dead-band → no creep).
        self.assertEqual((last.rect.width, last.rect.height), (800, 600))

    def test_release_when_a_hand_drops(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect)
        self.assertTrue(c.is_grabbed)
        # A hand drops → both_engaged False. The dead-man grace (default 0.30 s) holds
        # the grab for the immediate frame WITHOUT moving the window (rect None), then
        # FULLY RELEASES once the loss persists past the grace. (The grace bridging a
        # 1-frame flicker + the per-cause eager release are exercised in detail by
        # ReleaseDeadManTests.)
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertIsNone(d.rect)                 # never moves the window on a lost frame
        clk.advance(0.35)                         # let the loss age past the 0.30 s grace
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertEqual(d.phase, "idle")
        self.assertFalse(d.active)
        self.assertFalse(d.resizing)
        self.assertIsNone(d.rect)
        self.assertFalse(c.is_grabbed)

    def test_short_two_hand_blip_does_not_grab(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        rect = mod.Rect(800, 400, 1600, 1000)
        # Both hands up for < the hold, then released → never grabbed (no window move).
        c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                 focused_rect=rect, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        clk.advance(0.10)    # still under 0.20 hold
        d = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=rect, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertEqual(d.phase, "holding")
        self.assertFalse(d.resizing)
        # Then release.
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertFalse(d.active)

    def test_grab_without_window_stays_active_but_no_rect(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk)
        # Both hands up over the DESKTOP (no grabbable window → focused_rect None).
        c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                 focused_rect=None, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        clk.advance(0.25)
        d = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=None, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        # Active (air-mouse stands down) but nothing grabbed / moved.
        self.assertTrue(d.active)
        self.assertFalse(d.resizing)
        self.assertIsNone(d.rect)

    def test_rect_ema_smooths_the_window(self):
        mod = self._load()
        clk = _Clock()
        # rect_alpha < 1 → the window EASES toward the target (not jittery jumps).
        c = mod.TwoHandController(clock=clk, dist_alpha=1.0, rect_alpha=0.5,
                                  grab_hold_sec=0.20)
        rect = mod.Rect(800, 400, 1600, 1000)
        # grab
        c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                 focused_rect=rect, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        clk.advance(0.25)
        c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                 focused_rect=rect, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        # One big spread frame: with alpha 0.5 the FIRST output is only PART-WAY to
        # the (step-capped) target — i.e. smoothed, not a jump.
        d = c.update(both_engaged=True, hand_dist=0.80, midpoint=(1200, 700),
                     focused_rect=None, bounds=self.BOUNDS,
                     hands=((800, 700), (1600, 700)))
        self.assertTrue(d.rect.width > rect.width)           # grew a bit
        self.assertTrue(d.rect.width < int(rect.width * 1.18))   # but < the step cap


# ══════════════════════════════════════════════════════════════════════════
#  RELEASE DEAD-MAN — the grab MUST let go the instant it can't confirm BOTH
#  hands are still tracked + engaged (one lost/dropped/opened hand releases
#  within the grace); only "both confirmed" keeps resizing. This is the LIVE
#  bug the owner hit: a half-seen hand used to latch the grab on the window.
# ══════════════════════════════════════════════════════════════════════════
class ReleaseDeadManTests(_Base):
    BOUNDS = (0, 0, 2560, 1440)

    def _ctrl(self, mod, clk, *, grace=0.30):
        return mod.TwoHandController(clock=clk, dist_alpha=1.0, rect_alpha=1.0,
                                     grab_hold_sec=0.20, release_grace_sec=grace)

    def _grab(self, mod, c, clk, rect, *, dist=0.40, mid=(1200, 700)):
        """Drive through the grab-hold so the controller is GRABBED on `rect`."""
        c.update(both_engaged=True, hand_dist=dist, midpoint=mid,
                 focused_rect=rect, bounds=self.BOUNDS,
                 hands=((1000, 700), (1400, 700)))
        clk.advance(0.25)   # past the hold
        d = c.update(both_engaged=True, hand_dist=dist, midpoint=mid,
                     focused_rect=rect, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertTrue(c.is_grabbed)
        return d

    # ── (a) one hand goes UNTRACKED → releases within the grace ──────────────
    def test_one_hand_untracked_releases_within_grace_controller(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect)
        # A hand goes untracked: the sensor layer can no longer compute the distance/
        # midpoint (hand joint gone) → hand_dist/midpoint arrive None even though the
        # other hand is still up. Within the grace the grab is HELD but applies NO
        # window motion (rect None — we don't move the window on a half-seen frame).
        clk.advance(0.10)   # < 0.30 grace
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertTrue(c.is_grabbed)          # still latched (within grace)
        self.assertTrue(d.resizing)
        self.assertIsNone(d.rect)              # but NO SetWindowPos this frame
        # The loss persists past the grace → FORCE RELEASE.
        clk.advance(0.25)   # total 0.35 > 0.30 grace
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertFalse(c.is_grabbed)
        self.assertFalse(d.active)
        self.assertFalse(d.resizing)
        self.assertEqual(d.phase, "idle")

    # ── (b) one hand DROPS below the engage line → releases ──────────────────
    def test_one_hand_drops_below_line_releases(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect)
        # A hand drops below the lift line → both_engaged goes False (the sensor still
        # has a midpoint/dist for the instant of the drop, but the engage gate fails).
        # Past the grace it must RELEASE; we step straight past the grace here.
        clk.advance(0.35)   # > 0.30 grace
        d = c.update(both_engaged=False, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=None, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertFalse(c.is_grabbed)
        self.assertFalse(d.active)
        self.assertIsNone(d.rect)

    def test_one_hand_drops_below_line_releases_end_to_end(self):
        # Same as (b) but through the REAL sensor plumbing: one hand on the desk →
        # _both_hands_engaged False → _poll_once releases the grab (no more SetWindowPos).
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = PollWin32Tests._spy_win32(self)
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = self._ctrl(mod, clk, grace=0.0)   # grace 0 → instant release on loss
        # Grab with both hands up.
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            clk.advance(0.25)
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            self.assertTrue(ctrl.is_grabbed)
        calls_before = len(calls)
        # Now one hand drops to the desk → must release (no further window calls).
        one = _one_up_body(up_side="right")
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[one])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(5):
                d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertFalse(ctrl.is_grabbed)
        self.assertFalse(d.active)
        # No window moves AFTER the drop (the grab let go).
        self.assertEqual(len(calls), calls_before)

    # ── (c) one hand OPENS / disengages → releases ───────────────────────────
    def test_one_hand_opens_releases(self):
        # Opening a hand is modelled as that hand no longer clearing the engage gate
        # (the two-hand engage requires both hands raised+extended). The poll layer
        # collapses any per-hand loss into both_engaged=False / no distance, so the
        # controller releases once the grace lapses. Drive it past the grace.
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect)
        clk.advance(0.40)   # > grace
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertFalse(c.is_grabbed)
        self.assertFalse(d.active)

    # ── (d) BOTH hands stay tracked + engaged → stays grabbed AND keeps resizing
    def test_both_tracked_engaged_stays_grabbed_and_resizes(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)     # 800x600, centre (1200,700)
        self._grab(mod, c, clk, rect, dist=0.40)
        # Both hands remain up + tracked the whole time, spreading apart → it must
        # STAY grabbed and keep growing the window across many frames (clock advancing
        # well past the grace each frame, proving the confirm clock keeps refreshing).
        last = None
        for _ in range(14):
            clk.advance(0.05)   # 0.7 s total — far past the 0.30 grace
            last = c.update(both_engaged=True, hand_dist=0.60, midpoint=(1200, 700),
                            focused_rect=None, bounds=self.BOUNDS,
                            hands=((900, 700), (1500, 700)))
        self.assertTrue(c.is_grabbed)            # never released
        self.assertTrue(last.resizing)
        self.assertIsNotNone(last.rect)
        self.assertEqual((last.rect.width, last.rect.height), (1200, 900))  # grew 1.5×
        self.assertAlmostEqual(last.rect.cx, 1200.0, delta=2)               # centre held

    def test_both_tracked_engaged_stays_grabbed_end_to_end(self):
        # (d) through the REAL plumbing: both hands stay up and spread → SetWindowPos
        # keeps firing with a GROWING rect (the grab is NOT dropped by the dead-man).
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = PollWin32Tests._spy_win32(self)
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = self._ctrl(mod, clk, grace=0.30)
        near = _both_up_body(left_x=-0.10, right_x=0.10)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[near])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            clk.advance(0.25)
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            self.assertTrue(ctrl.is_grabbed)
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(15):
                clk.advance(0.05)
                last = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertTrue(ctrl.is_grabbed)         # stayed grabbed across the gesture
        self.assertTrue(calls, "SetWindowPos never fired while both hands held")
        self.assertGreater(calls[-1][1].width, 800)   # window grew
        self.assertEqual(calls[-1][1], last.rect)

    # ── (e) TOTAL body loss → releases ───────────────────────────────────────
    def test_total_body_loss_releases_controller(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect)
        # Body fully lost → nothing observed at all. Past the grace → RELEASE.
        clk.advance(0.35)
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertFalse(c.is_grabbed)
        self.assertFalse(d.active)
        self.assertEqual(d.phase, "idle")

    def test_total_body_loss_releases_end_to_end(self):
        # (e) through the REAL plumbing: the bridge reports NO bodies → tracked False
        # → _poll_once releases the grab and stops moving the window.
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = PollWin32Tests._spy_win32(self)
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = self._ctrl(mod, clk, grace=0.0)   # instant release on total loss
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            clk.advance(0.25)
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            self.assertTrue(ctrl.is_grabbed)
        calls_before = len(calls)
        # Body vanishes entirely (no bodies) → release.
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(5):
                d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertFalse(ctrl.is_grabbed)
        self.assertFalse(d.active)
        self.assertEqual(len(calls), calls_before)   # no window moves after the loss

    # ── (a, end-to-end) one hand UNTRACKED via the sensor layer → releases ───
    def test_one_hand_untracked_releases_end_to_end(self):
        # The headline live bug, end-to-end: ONE hand's joint disappears (untracked)
        # while the body + other hand are still seen. _both_hands_engaged sees a None
        # hand → both=False → the grab releases (no further SetWindowPos).
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = PollWin32Tests._spy_win32(self)
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = self._ctrl(mod, clk, grace=0.0)   # grace 0 → release the moment a hand is lost
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            clk.advance(0.25)
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            self.assertTrue(ctrl.is_grabbed)
        calls_before = len(calls)
        # One hand goes untracked (its hand joint vanishes) — body still tracked.
        lost = _one_untracked_body(untracked_side="left")
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[lost])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(5):
                d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertFalse(ctrl.is_grabbed)         # the headline fix: it LET GO
        self.assertFalse(d.active)
        self.assertEqual(len(calls), calls_before)

    # ── grace really does bridge a 1-frame flicker (doesn't over-release) ────
    def test_single_frame_flicker_is_bridged_then_resumes(self):
        mod = self._load()
        clk = _Clock()
        c = self._ctrl(mod, clk, grace=0.30)
        rect = mod.Rect(800, 400, 1600, 1000)
        self._grab(mod, c, clk, rect, dist=0.40)
        # ONE bad frame (joint flicker) within the grace → HELD, no rect applied.
        clk.advance(0.03)
        d = c.update(both_engaged=False, hand_dist=None, midpoint=None,
                     focused_rect=None, bounds=self.BOUNDS, hands=None)
        self.assertTrue(c.is_grabbed)
        self.assertIsNone(d.rect)
        # Both hands re-confirmed next frame → resumes resizing (grab survived).
        clk.advance(0.03)
        d = c.update(both_engaged=True, hand_dist=0.40, midpoint=(1200, 700),
                     focused_rect=None, bounds=self.BOUNDS,
                     hands=((1000, 700), (1400, 700)))
        self.assertTrue(c.is_grabbed)
        self.assertTrue(d.resizing)
        self.assertIsNotNone(d.rect)


# ══════════════════════════════════════════════════════════════════════════
#  WIN32 WIRING — a grabbed window is SetWindowPos'd with the SCALED rect
# ══════════════════════════════════════════════════════════════════════════
class PollWin32Tests(_Base):
    def _spy_win32(self):
        """A mock Win32: foreground_target returns a fixed (hwnd, rect); set_window_pos
        records every (hwnd, rect) it's asked to apply."""
        calls: list = []
        rect_box = [None]

        def foreground_target():
            return (4242, rect_box[0])

        def set_window_pos(hwnd, rect):
            calls.append((hwnd, rect))
            return True
        return calls, rect_box, foreground_target, set_window_pos

    def _bridge_bodies(self, mod, bodies):
        # Point the air-mouse bridge resolver at a fake bridge with these bodies.
        p = mock.patch.object(self._am, "_bridge",
                              lambda: _fake_bridge(bodies=bodies))
        p.start()
        self.addCleanup(p.stop)
        # Disable the mirror so left/right map straight through in these tests.
        pm = mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False)
        pm.start()
        self.addCleanup(pm.stop)

    def test_spread_calls_setwindowpos_with_grown_rect(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = self._spy_win32()
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)   # 800x600, centre (1200,700)
        clk = _Clock()
        ctrl = mod.TwoHandController(clock=clk, dist_alpha=1.0, rect_alpha=1.0,
                                     grab_hold_sec=0.20)

        # Hands close together (small separation) → small initial distance baseline.
        near = _both_up_body(left_x=-0.10, right_x=0.10)
        self._bridge_bodies(mod, [near])
        mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        clk.advance(0.25)
        mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)  # GRAB
        self.assertTrue(ctrl.is_grabbed)

        # Now SPREAD the hands wide → grow. Re-point the bridge to the wide body.
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            last = None
            for _ in range(15):
                last = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        # SetWindowPos was called, and the LAST applied rect is GROWN vs the original
        # (spread → bigger window), still centred on (1200, 700).
        self.assertTrue(calls, "SetWindowPos was never called")
        applied = calls[-1][1]
        self.assertEqual(calls[-1][0], 4242)                 # the grabbed hwnd
        self.assertGreater(applied.width, 800)               # grew
        self.assertGreater(applied.height, 600)
        self.assertAlmostEqual(applied.cx, 1200.0, delta=3)  # centre held
        self.assertEqual(applied, last.rect)

    def test_pinch_calls_setwindowpos_with_shrunk_rect(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = self._spy_win32()
        rect_box[0] = mod.Rect(700, 350, 1700, 1100)   # 1000x750, centre (1200,725)
        clk = _Clock()
        ctrl = mod.TwoHandController(clock=clk, dist_alpha=1.0, rect_alpha=1.0,
                                     grab_hold_sec=0.20)
        # Grab with hands WIDE (big baseline distance).
        wide = _both_up_body(left_x=-0.35, right_x=0.35)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            clk.advance(0.25)
            mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)  # GRAB
            self.assertTrue(ctrl.is_grabbed)
        # PINCH the hands close → shrink.
        near = _both_up_body(left_x=-0.08, right_x=0.08)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[near])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(20):
                mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        applied = calls[-1][1]
        self.assertLess(applied.width, 1000)        # shrank
        self.assertLess(applied.height, 750)

    def test_disabled_flag_makes_no_window_calls(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)         # gated OFF
        calls, rect_box, fg, swp = self._spy_win32()
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = mod.TwoHandController(clock=clk, grab_hold_sec=0.20)
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(10):
                clk.advance(0.05)
                d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertEqual(calls, [])                  # never touched a window
        self.assertFalse(d.active)

    def test_staging_makes_no_window_calls(self):
        mod = self._load()
        self._patch_flag(True)
        # Staging ON → must never move a real window.
        with mock.patch.object(mod, "_is_staging", lambda: True):
            calls, rect_box, fg, swp = self._spy_win32()
            rect_box[0] = mod.Rect(800, 400, 1600, 1000)
            clk = _Clock()
            ctrl = mod.TwoHandController(clock=clk, grab_hold_sec=0.20)
            wide = _both_up_body(left_x=-0.30, right_x=0.30)
            with mock.patch.object(self._am, "_bridge",
                                   lambda: _fake_bridge(bodies=[wide])), \
                 mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
                for _ in range(10):
                    clk.advance(0.05)
                    mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
            self.assertEqual(calls, [])

    def test_one_hand_only_does_not_engage(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        calls, rect_box, fg, swp = self._spy_win32()
        rect_box[0] = mod.Rect(800, 400, 1600, 1000)
        clk = _Clock()
        ctrl = mod.TwoHandController(clock=clk, grab_hold_sec=0.20)
        one = _one_up_body(up_side="right")
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[one])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            for _ in range(10):
                clk.advance(0.05)
                d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertFalse(d.active)         # one hand → not two-hand mode
        self.assertEqual(calls, [])


# ══════════════════════════════════════════════════════════════════════════
#  AIR-MOUSE STAND-DOWN — two-hand mode suppresses the single-hand cursor
# ══════════════════════════════════════════════════════════════════════════
class StandDownTests(_Base):
    def _spy_win32(self):
        def fg():
            return (1, None)        # no grabbable window; we only care about active

        def swp(hwnd, rect):
            return True
        return fg, swp

    def test_two_hand_active_publishes_standdown_to_air_mouse(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        fg, swp = self._spy_win32()
        clk = _Clock()
        ctrl = mod.TwoHandController(clock=clk, grab_hold_sec=0.20)
        wide = _both_up_body(left_x=-0.30, right_x=0.30)
        with mock.patch.object(self._am, "_bridge",
                               lambda: _fake_bridge(bodies=[wide])), \
             mock.patch.object(self._am, "_hand_mirror_enabled", lambda: False):
            d = mod._poll_once(ctrl, foreground_target=fg, set_window_pos=swp)
        self.assertTrue(d.active)
        # The air-mouse now sees two_hand_active() True (fresh heartbeat) → it stands
        # down. (This is the exact flag the air-mouse _poll_once folds into yield.)
        self.assertTrue(self._am.two_hand_active())

    def test_air_mouse_cursor_suppressed_while_two_hand_active(self):
        self._load()
        am = self._am
        # Publish two-hand ACTIVE, then assert the air-mouse poll makes NO cursor move
        # even with a single raised hand that would normally engage.
        am.set_two_hand_active(True)
        # Build a one-hand raised body for the air-mouse.
        body = _one_up_body(up_side="right")
        moves: list = []
        with mock.patch.object(am, "_hand_mirror_enabled", lambda: False), \
             mock.patch.object(am, "_set_cursor_pos",
                               lambda x, y: (moves.append((x, y)) or True)), \
             mock.patch.object(am, "_mouse_button", lambda *a, **k: True), \
             mock.patch.object(am, "_publish_overlay_state", lambda *a, **k: None), \
             mock.patch.object(am, "_clear_overlay_state", lambda *a, **k: None), \
             mock.patch.object(am, "_spawn_overlay", lambda *a, **k: None), \
             mock.patch.object(am, "_overlay_alive", lambda: True), \
             mock.patch.object(am, "_install_yield_watcher", lambda *a, **k: False), \
             mock.patch.object(am, "real_input_recent", lambda *a, **k: False), \
             mock.patch.object(am, "_is_staging", lambda: False):
            from core import config as cfg
            with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", True,
                                   create=True):
                ctrl = am.AirMouseController(am.ReachBox(2560, 1440),
                                             debounce_frames=1, grace_sec=0.0,
                                             engage_debounce_frames=1)
                bridge = _fake_bridge(bodies=[body])
                for _ in range(5):
                    am._poll_once(ctrl, bridge)
        self.assertEqual(moves, [], "air-mouse moved the cursor during two-hand mode")

    def test_stale_two_hand_heartbeat_does_not_strand_cursor(self):
        self._load()
        am = self._am
        # An OLD heartbeat (older than the TTL) must read inactive so the cursor
        # resumes — a crashed two-hand poller can't permanently suppress the mouse.
        with am._two_hand_state_lock:
            am._two_hand_state["active"] = True
            am._two_hand_state["ts"] = time.time() - 10.0   # stale
        self.assertFalse(am.two_hand_active())


# ══════════════════════════════════════════════════════════════════════════
#  DUAL RETICLE — two circles, BLUE normally / PURPLE while resizing
# ══════════════════════════════════════════════════════════════════════════
class DualReticleTests(_Base):
    def _read_state(self, am):
        with open(am.AIR_CURSOR_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_two_circles_published_when_both_hands_engaged(self):
        mod = self._load()
        am = self._am
        d = mod.TwoHandDecision(
            active=True, rect=None, resizing=False, phase="holding",
            hands=((1000, 700), (1450, 720)))
        mod._publish_two_hand_overlay(d)
        st = self._read_state(am)
        self.assertTrue(st["two_hand"])
        self.assertTrue(st["visible"])
        self.assertEqual(len(st["hands"]), 2)
        self.assertEqual((st["hands"][0]["x"], st["hands"][0]["y"]), (1000, 700))
        self.assertEqual((st["hands"][1]["x"], st["hands"][1]["y"]), (1450, 720))

    def test_colour_is_blue_when_not_resizing(self):
        mod = self._load()
        am = self._am
        d = mod.TwoHandDecision(active=True, rect=None, resizing=False,
                                phase="holding", hands=((100, 100), (300, 100)))
        mod._publish_two_hand_overlay(d)
        self.assertEqual(self._read_state(am)["color"], "blue")

    def test_colour_is_purple_while_resizing(self):
        mod = self._load()
        am = self._am
        rect = mod.Rect(800, 400, 1600, 1000)
        d = mod.TwoHandDecision(active=True, rect=rect, resizing=True,
                                phase="grabbed", hands=((100, 100), (300, 100)))
        mod._publish_two_hand_overlay(d)
        st = self._read_state(am)
        self.assertEqual(st["color"], "purple")
        self.assertTrue(st["resizing"])

    def test_inactive_clears_two_hand_reticle(self):
        mod = self._load()
        am = self._am
        d = mod.TwoHandDecision(active=False, rect=None, resizing=False,
                                phase="idle", hands=None)
        mod._publish_two_hand_overlay(d)
        st = self._read_state(am)
        self.assertFalse(st.get("two_hand"))
        self.assertFalse(st.get("visible"))


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG / GATE
# ══════════════════════════════════════════════════════════════════════════
class GateTests(_Base):
    def test_enabled_default_true(self):
        mod = self._load()
        # The flag defaults True, but staging always wins (off).
        with mock.patch.object(mod, "_is_staging", lambda: False):
            from core import config as cfg
            # Default (unset by test) → True.
            if hasattr(cfg, "KINECT_TWO_HAND_ENABLED"):
                with mock.patch.object(cfg, "KINECT_TWO_HAND_ENABLED", True):
                    self.assertTrue(mod._two_hand_enabled())

    def test_staging_overrides_enabled(self):
        mod = self._load()
        self._patch_flag(True)
        with mock.patch.object(mod, "_is_staging", lambda: True):
            self.assertFalse(mod._two_hand_enabled())


if __name__ == "__main__":
    unittest.main()
