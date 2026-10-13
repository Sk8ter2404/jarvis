"""Tests for the 2026-10-12 Kinect gesture set: TWO_HAND_X (close the focused
window), POINT (pick a monitor) and swipe = previous / next track.

Synthetic skeleton frames only - no sensor, no Win32, no real keys: every window,
media action and cursor call is a fake. Covers:

  * the pure geometry (two_hand_x_pose / point_aim / monitor_for_aim),
  * the recognizer's hold / once-per-pose / re-arm behaviour on an injected clock,
  * the skill's per-gesture settings, the swipe modes and the close / point
    safety rules,
  * a [gesture] log line for every recognised gesture.

stdlib unittest + mock.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

from audio import kinect_gestures as kg

TRACKED = 2
STEP = 1.0 / 18.0           # the gesture poller's cadence


# ─── frame builders ─────────────────────────────────────────────────────────
class _Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


def _j(x, y, z, state=TRACKED):
    return (float(x), float(y), float(z), state)


SH_Y = 0.40         # shoulder height (camera y)
SH_Z = 2.00         # shoulder depth (z grows away from the sensor)
SH_HALF = 0.20      # half the shoulder span; the owner's RIGHT is +x


def _body(*, hand_left=None, hand_right=None, grip_left="open", grip_right="open",
          mirrored=False, drop=()):
    """A whole-body frame. Hands default to hanging at the sides."""
    sign = -1.0 if mirrored else 1.0
    joints = {
        "head": _j(0.0, 0.70, SH_Z),
        "spine_shoulder": _j(0.0, SH_Y, SH_Z),
        "spine_mid": _j(0.0, 0.10, SH_Z),
        "spine_base": _j(0.0, -0.20, SH_Z),
        "shoulder_left": _j(-sign * SH_HALF, SH_Y, SH_Z),
        "shoulder_right": _j(sign * SH_HALF, SH_Y, SH_Z),
        "elbow_left": _j(-sign * 0.25, 0.10, SH_Z),
        "elbow_right": _j(sign * 0.25, 0.10, SH_Z),
        "hand_left": hand_left or _j(-sign * 0.28, -0.10, SH_Z),
        "hand_right": hand_right or _j(sign * 0.28, -0.10, SH_Z),
    }
    for name in drop:
        joints.pop(name, None)
    return {"id": 1, "joints": joints, "head": (0.0, 0.7, SH_Z),
            "distance_m": SH_Z, "facing": True,
            "hand_left": grip_left, "hand_right": grip_right}


def _x_body(*, lift=-0.05, forward=0.30, cross=0.05, mirrored=False, drop=()):
    """Both wrists crossed in front of the upper chest: the right hand is `cross`
    past the midline on the left side and vice-versa."""
    sign = -1.0 if mirrored else 1.0
    y = SH_Y + lift
    z = SH_Z - forward
    return _body(hand_right=_j(-sign * cross, y, z), hand_left=_j(sign * cross, y, z),
                 mirrored=mirrored, drop=drop)


def _aim_hand(az_deg, el_deg=0.0, reach=0.60, side="right"):
    """The hand joint of an arm aimed at (azimuth, elevation) from the shoulder."""
    sx = SH_HALF if side == "right" else -SH_HALF
    az, el = math.radians(az_deg), math.radians(el_deg)
    return _j(sx + reach * math.sin(az) * math.cos(el),
              SH_Y + reach * math.sin(el),
              SH_Z - reach * math.cos(az) * math.cos(el))


def _point_body(az_deg, el_deg=0.0, reach=0.60, grip="lasso", side="right"):
    kw = {f"hand_{side}": _aim_hand(az_deg, el_deg, reach, side),
          f"grip_{side}": grip}
    return _body(**kw)


# ─── pure geometry ──────────────────────────────────────────────────────────
class TwoHandXPoseTests(unittest.TestCase):
    def test_crossed_wrists_at_the_upper_chest_count(self):
        self.assertTrue(kg.two_hand_x_pose(_x_body()))

    def test_orientation_agnostic_when_the_stream_is_mirrored(self):
        self.assertTrue(kg.two_hand_x_pose(_x_body(mirrored=True)))

    def test_uncrossed_hands_do_not_count(self):
        self.assertFalse(kg.two_hand_x_pose(_x_body(cross=-0.05)))

    def test_barely_crossed_does_not_count(self):
        self.assertFalse(kg.two_hand_x_pose(_x_body(cross=0.01)))

    def test_folded_arms_do_not_count(self):
        """Arms folded across the stomach: crossed, but low and hugging the torso."""
        self.assertFalse(kg.two_hand_x_pose(_x_body(lift=-0.32, forward=0.08)))

    def test_hands_tight_to_the_chest_do_not_count(self):
        self.assertFalse(kg.two_hand_x_pose(_x_body(forward=0.05)))

    def test_hands_above_the_shoulder_are_the_air_mouse_not_this(self):
        self.assertFalse(kg.two_hand_x_pose(_x_body(lift=0.20)))

    def test_missing_shoulder_fails_closed(self):
        self.assertFalse(kg.two_hand_x_pose(_x_body(drop=("shoulder_left",))))

    def test_untracked_hand_fails_closed(self):
        b = _x_body()
        b["joints"]["hand_left"] = (0.05, 0.35, 1.7, 1)       # inferred only
        self.assertFalse(kg.two_hand_x_pose(b))

    def test_garbage_never_raises(self):
        for bad in (None, {}, {"joints": None}, {"joints": {"hand_left": "x"}}, 5):
            self.assertFalse(kg.two_hand_x_pose(bad))


class PointAimTests(unittest.TestCase):
    def test_aim_reports_azimuth_and_elevation(self):
        aim = kg.point_aim(_point_body(20.0, 10.0))
        self.assertIsNotNone(aim)
        self.assertEqual(aim["side"], "right")
        self.assertAlmostEqual(aim["azimuth_deg"], 20.0, places=3)
        self.assertAlmostEqual(aim["elevation_deg"], 10.0, places=3)

    def test_left_arm_aims_too(self):
        aim = kg.point_aim(_point_body(-15.0, side="left"))
        self.assertEqual(aim["side"], "left")
        self.assertAlmostEqual(aim["azimuth_deg"], -15.0, places=3)

    def test_needs_the_lasso_hand_state_by_default(self):
        self.assertIsNone(kg.point_aim(_point_body(0.0, grip="open")))
        self.assertIsNone(kg.point_aim(_point_body(0.0, grip="closed")))
        self.assertIsNotNone(kg.point_aim(_point_body(0.0, grip="open"),
                                          require_lasso=False))

    def test_a_folded_arm_is_not_a_point(self):
        self.assertIsNone(kg.point_aim(_point_body(0.0, reach=0.30)))

    def test_an_arm_aimed_at_the_desk_is_not_a_point(self):
        self.assertIsNone(kg.point_aim(_point_body(0.0, el_deg=-45.0)))

    def test_hand_behind_the_shoulder_is_not_a_point(self):
        self.assertIsNone(kg.point_aim(_point_body(170.0)))

    def test_garbage_never_raises(self):
        for bad in (None, {}, {"joints": 3}, 7):
            self.assertIsNone(kg.point_aim(bad))


class MonitorForAimTests(unittest.TestCase):
    LAYOUT = {"left": (-2560, 0, 2560, 1440), "middle": (0, 0, 2560, 1440),
              "right": (2560, 0, 2560, 1440), "top": (0, -1440, 2560, 1440)}

    def test_row_splits_into_azimuth_bins(self):
        f = kg.monitor_for_aim
        self.assertEqual(f(-25.0, 0.0, self.LAYOUT), "left")
        self.assertEqual(f(-5.0, 0.0, self.LAYOUT), "middle")
        self.assertEqual(f(5.0, 0.0, self.LAYOUT), "middle")
        self.assertEqual(f(25.0, 0.0, self.LAYOUT), "right")

    def test_outer_bins_take_everything_beyond(self):
        self.assertEqual(kg.monitor_for_aim(-80.0, 0.0, self.LAYOUT), "left")
        self.assertEqual(kg.monitor_for_aim(80.0, 0.0, self.LAYOUT), "right")

    def test_aimed_high_picks_the_monitor_above_the_row(self):
        self.assertEqual(kg.monitor_for_aim(0.0, 35.0, self.LAYOUT), "top")
        self.assertEqual(kg.monitor_for_aim(0.0, 10.0, self.LAYOUT), "middle")

    def test_aimed_high_without_a_top_monitor_stays_in_the_row(self):
        row = {k: v for k, v in self.LAYOUT.items() if k != "top"}
        self.assertEqual(kg.monitor_for_aim(0.0, 40.0, row), "middle")

    def test_two_monitors_split_at_straight_ahead(self):
        two = {"a": (0, 0, 1920, 1080), "b": (1920, 0, 1920, 1080)}
        self.assertEqual(kg.monitor_for_aim(-3.0, 0.0, two), "a")
        self.assertEqual(kg.monitor_for_aim(3.0, 0.0, two), "b")

    def test_bin_width_is_tunable(self):
        self.assertEqual(kg.monitor_for_aim(15.0, 0.0, self.LAYOUT, bin_deg=24.0),
                         "right")
        self.assertEqual(kg.monitor_for_aim(15.0, 0.0, self.LAYOUT, bin_deg=40.0),
                         "middle")

    def test_empty_layout_is_none(self):
        self.assertIsNone(kg.monitor_for_aim(0.0, 0.0, {}))
        self.assertIsNone(kg.monitor_for_aim(0.0, 0.0, None))


# ─── recognizer ─────────────────────────────────────────────────────────────
class _RecBase(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.rec = kg.GestureRecognizer(now_fn=self.clk)
        self.details = []          # last_detail captured at each fire

    def feed(self, body, seconds, step=STEP):
        """Feed one body for `seconds`; return every gesture fired, in order."""
        fired = []
        for _ in range(max(1, int(round(seconds / step)))):
            self.clk.advance(step)
            g = self.rec.update([body] if body is not None else [])
            if g:
                fired.append(g)
                self.details.append(dict(self.rec.last_detail))
        return fired


class TwoHandXRecognizerTests(_RecBase):
    def test_fires_once_after_the_hold(self):
        self.feed(_body(), 0.7)                         # arms down: arms the detector
        fired = self.feed(_x_body(), kg.X_HOLD_SECONDS + 0.5)
        self.assertEqual(fired, [kg.TWO_HAND_X])

    def test_a_short_hold_does_not_fire(self):
        self.feed(_body(), 0.7)
        self.assertEqual(self.feed(_x_body(), kg.X_HOLD_SECONDS - 0.3), [])

    def test_holding_the_sign_never_fires_a_second_time(self):
        self.feed(_body(), 0.7)
        fired = self.feed(_x_body(), 6.0)
        self.assertEqual(fired, [kg.TWO_HAND_X])

    def test_fires_again_after_the_arms_come_apart_and_the_cooldown(self):
        self.feed(_body(), 0.7)
        self.assertEqual(self.feed(_x_body(), 1.5), [kg.TWO_HAND_X])
        self.feed(_body(), kg.X_REARM_ABSENT_SECONDS + 0.2 + kg.COOLDOWN_SECONDS)
        self.assertEqual(self.feed(_x_body(), 1.5), [kg.TWO_HAND_X])

    def test_a_pose_already_held_at_start_never_fires(self):
        """No lead-in of 'arms down': the detector is disarmed until it has seen
        the pose lapse (e.g. just after the air-mouse let go)."""
        self.assertEqual(self.feed(_x_body(), 3.0), [])

    def test_reset_disarms_it(self):
        self.feed(_body(), 0.7)
        self.rec.reset()
        self.assertEqual(self.feed(_x_body(), 3.0), [])

    def test_a_one_frame_tracking_blip_keeps_the_hold(self):
        self.feed(_body(), 0.7)
        fired = self.feed(_x_body(), 0.4)
        fired += self.feed(_x_body(drop=("hand_left",)), STEP)       # blip
        fired += self.feed(_x_body(), 0.8)
        self.assertEqual(fired, [kg.TWO_HAND_X])

    def test_a_long_break_restarts_the_hold(self):
        self.feed(_body(), 0.7)
        fired = self.feed(_x_body(), 0.5)
        fired += self.feed(_body(), 0.4)
        fired += self.feed(_x_body(), 0.5)
        self.assertEqual(fired, [])

    def test_folded_arms_never_fire(self):
        self.feed(_body(), 0.7)
        self.assertEqual(self.feed(_x_body(lift=-0.32, forward=0.08), 5.0), [])

    def test_disabled_gesture_does_not_fire_or_burn_the_cooldown(self):
        self.rec.set_enabled({kg.WAVE})
        self.feed(_body(), 0.7)
        self.assertEqual(self.feed(_x_body(), 3.0), [])
        self.assertFalse(self.rec.in_cooldown())

    def test_travel_into_the_sign_is_not_read_as_a_swipe(self):
        """The right hand sweeps 0.5 m across the chest into the X: no SWIPE."""
        self.feed(_body(), 0.7)
        fired = []
        for i in range(8):                                   # 0.45 s of travel
            x = 0.30 - i * 0.05
            self.clk.advance(STEP)
            g = self.rec.update([_body(hand_right=_j(x, SH_Y - 0.05, SH_Z - 0.30),
                                       hand_left=_j(0.05, SH_Y - 0.05, SH_Z - 0.30))])
            if g:
                fired.append(g)
        fired += self.feed(_x_body(), 1.2)
        self.assertNotIn(kg.SWIPE_LEFT, fired)
        self.assertNotIn(kg.SWIPE_RIGHT, fired)
        self.assertIn(kg.TWO_HAND_X, fired)

    def test_no_body_resets_the_hold(self):
        self.feed(_body(), 0.7)
        fired = self.feed(_x_body(), 0.5)
        fired += self.feed(None, 0.4)
        fired += self.feed(_x_body(), 0.5)
        self.assertEqual(fired, [])


class PointRecognizerTests(_RecBase):
    def test_steady_lasso_aim_fires_with_the_aim(self):
        fired = self.feed(_point_body(20.0, 5.0), 1.2)
        self.assertEqual(fired, [kg.POINT])
        self.assertAlmostEqual(self.details[0]["azimuth_deg"], 20.0, places=2)
        self.assertAlmostEqual(self.details[0]["elevation_deg"], 5.0, places=2)

    def test_last_detail_clears_on_the_next_frame(self):
        self.feed(_point_body(20.0), 1.2)
        self.assertTrue(self.details[0])
        self.feed(_body(), STEP)
        self.assertEqual(self.rec.last_detail, {})

    def test_open_hand_is_not_a_point_but_is_when_lasso_is_not_required(self):
        self.assertEqual(self.feed(_point_body(0.0, grip="open"), 2.0), [])
        self.rec.reset()
        self.rec.set_require_lasso(False)
        self.assertEqual(self.feed(_point_body(0.0, grip="open"), 1.2), [kg.POINT])

    def test_a_wobbling_aim_does_not_fire(self):
        fired = []
        for i in range(36):                              # 2 s, +/-12 deg every frame
            self.clk.advance(STEP)
            g = self.rec.update([_point_body(12.0 if i % 2 else -12.0)])
            if g:
                fired.append(g)
        self.assertNotIn(kg.POINT, fired)     # (a real wave may fire; not a point)

    def test_a_short_point_does_not_fire(self):
        self.assertEqual(self.feed(_point_body(0.0), 0.4), [])

    def test_one_fire_per_point_even_if_held(self):
        self.assertEqual(self.feed(_point_body(0.0), 6.0), [kg.POINT])

    def test_swinging_to_another_monitor_fires_again(self):
        fired = self.feed(_point_body(-25.0), 1.2)
        fired += self.feed(_point_body(25.0), 3.5)       # past the cooldown
        self.assertEqual(fired, [kg.POINT, kg.POINT])
        self.assertLess(self.details[0]["azimuth_deg"], -20.0)
        self.assertGreater(self.details[1]["azimuth_deg"], 20.0)

    def test_lowering_the_arm_rearms_the_same_aim(self):
        fired = self.feed(_point_body(0.0), 1.2)
        fired += self.feed(_body(), kg.POINT_REARM_ABSENT_SECONDS + kg.COOLDOWN_SECONDS)
        fired += self.feed(_point_body(0.0), 1.2)
        self.assertEqual(fired, [kg.POINT, kg.POINT])

    def test_a_lasso_flicker_does_not_restart_the_hold(self):
        fired = self.feed(_point_body(0.0), 0.5)
        fired += self.feed(_point_body(0.0, grip="open"), STEP)
        fired += self.feed(_point_body(0.0), 0.7)
        self.assertEqual(fired, [kg.POINT])

    def test_switching_arms_restarts_the_hold(self):
        fired = []
        for i in range(30):
            self.clk.advance(STEP)
            side = "right" if i % 2 else "left"
            g = self.rec.update([_point_body(0.0, side=side)])
            if g:
                fired.append(g)
        self.assertNotIn(kg.POINT, fired)

    def test_disabled_point_does_not_fire(self):
        self.rec.set_enabled({kg.WAVE, kg.SWIPE_LEFT, kg.SWIPE_RIGHT})
        self.assertEqual(self.feed(_point_body(0.0), 3.0), [])


class ExistingGesturesStillWorkTests(_RecBase):
    """The new detectors must not disturb wave / raise / swipe."""

    def test_a_swipe_still_fires(self):
        fired = []
        for i in range(8):                                # 0.45 s, 0.56 m sideways
            self.clk.advance(0.065)
            hand = _j(-0.3 + i * 0.08, 0.20, SH_Z - 0.2)
            g = self.rec.update([_body(hand_right=hand)])
            if g:
                fired.append(g)
        self.assertEqual(fired, [kg.SWIPE_RIGHT])

    def test_disabling_swipes_stops_them(self):
        self.rec.set_enabled({kg.WAVE, kg.RAISE_HAND})
        for i in range(8):
            self.clk.advance(0.065)
            hand = _j(-0.3 + i * 0.08, 0.20, SH_Z - 0.2)
            self.assertIsNone(self.rec.update([_body(hand_right=hand)]))

    def test_all_gestures_lists_the_new_ones(self):
        self.assertIn(kg.TWO_HAND_X, kg.ALL_GESTURES)
        self.assertIn(kg.POINT, kg.ALL_GESTURES)


# ─── skill wiring ───────────────────────────────────────────────────────────
class _Win:
    """A stand-in pygetwindow window."""
    def __init__(self, title="Notes - Editor", left=100, top=100, width=800,
                 height=600, minimized=False):
        self.title, self.left, self.top = title, left, top
        self.width, self.height = width, height
        self.isMinimized = minimized
        self.activated = 0

    def activate(self):
        self.activated += 1


def _fake_bc(*, standby=False, speaking=False, pending=None):
    bc = types.ModuleType("bobert_companion")
    bc._standby_mode = [bool(standby)]
    bc._sleep_mode = [False]
    bc._pending_confirmation = list(pending or [])
    bc.FORBIDDEN_TARGETS = ("claude code",)
    bc._spoken = []
    bc._speak = lambda *a, **k: bc._spoken.append(a[0] if a else "")
    bc.media = []
    bc._act_media_next = lambda arg="": (bc.media.append("next") or "media next pressed")
    bc._act_media_prev = lambda arg="": (bc.media.append("prev") or "media previous pressed")
    bc.hotkeys = []
    pag = types.SimpleNamespace(hotkey=lambda *k: bc.hotkeys.append(k))
    bc._get_pyautogui = lambda: pag
    bc._interrupts = []

    def _interrupt(source="x", acoustic=True):
        if not speaking:
            return False
        bc._interrupts.append(source)
        return True
    bc.request_tts_interrupt = _interrupt
    return bc


class _SkillBase(unittest.TestCase):
    def setUp(self):
        mod, _ = load_skill_isolated("kinect_gestures", register=False)
        self.mod = mod
        # Never let a test see real input state or a real window.
        self._patch(mod, "_real_input_recent", lambda: False)
        self._patch(mod, "_per_app_blocked", lambda: False)
        from core import config as cfg
        self.cfg = cfg
        self._cfg(KINECT_SWIPE_MODE="media", KINECT_TWO_HAND_CLOSE_ENABLED=True,
                  KINECT_POINT_MONITOR_ENABLED=True, KINECT_POINT_MONITOR_ACTION="cursor",
                  KINECT_POINT_REQUIRE_LASSO=True, KINECT_POINT_MONITOR_BIN_DEG=24.0,
                  MONITORS=dict(MonitorForAimTests.LAYOUT))

    def _patch(self, obj, name, value):
        p = mock.patch.object(obj, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _cfg(self, **kw):
        for k, v in kw.items():
            self._patch(self.cfg, k, v)

    def run_logged(self, fn, *a, **k):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
        return out, buf.getvalue()


class EnabledGesturesTests(_SkillBase):
    def test_defaults_enable_everything(self):
        on = self.mod._enabled_gestures()
        for g in kg.ALL_GESTURES:
            self.assertIn(g, on)

    def test_each_gesture_has_its_own_switch(self):
        self._cfg(KINECT_TWO_HAND_CLOSE_ENABLED=False)
        self.assertNotIn(kg.TWO_HAND_X, self.mod._enabled_gestures())
        self.assertIn(kg.POINT, self.mod._enabled_gestures())
        self._cfg(KINECT_POINT_MONITOR_ENABLED=False)
        self.assertNotIn(kg.POINT, self.mod._enabled_gestures())
        self._cfg(KINECT_SWIPE_MODE="off")
        on = self.mod._enabled_gestures()
        self.assertNotIn(kg.SWIPE_LEFT, on)
        self.assertNotIn(kg.SWIPE_RIGHT, on)
        self.assertIn(kg.WAVE, on)

    def test_unknown_swipe_mode_falls_back_to_media(self):
        self._cfg(KINECT_SWIPE_MODE="banana")
        self.assertEqual(self.mod._swipe_mode(), "media")


class SwipeActionTests(_SkillBase):
    def test_swipe_right_is_next_and_left_is_previous_by_default(self):
        bc = _fake_bc()
        _, log1 = self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        _, log2 = self.run_logged(self.mod._dispatch, bc, kg.SWIPE_LEFT)
        self.assertEqual(bc.media, ["next", "prev"])
        self.assertIn("[gesture] swipe_right", log1)
        self.assertIn("[gesture] swipe_left", log2)

    def test_desktop_mode_switches_virtual_desktops(self):
        self._cfg(KINECT_SWIPE_MODE="desktop")
        bc = _fake_bc()
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_LEFT)
        self.assertEqual(bc.hotkeys, [("ctrl", "win", "right"), ("ctrl", "win", "left")])
        self.assertEqual(bc.media, [])

    def test_dismiss_mode_never_touches_media(self):
        self._cfg(KINECT_SWIPE_MODE="dismiss")
        bc = _fake_bc()
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        self.assertEqual(bc.media, [])
        self.assertEqual(bc.hotkeys, [])

    def test_off_mode_does_nothing_even_with_speech(self):
        self._cfg(KINECT_SWIPE_MODE="off")
        bc = _fake_bc(speaking=True)
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        self.assertEqual(bc._interrupts, [])
        self.assertEqual(bc.media, [])

    def test_a_swipe_while_speaking_is_the_never_mind_not_a_skip(self):
        bc = _fake_bc(speaking=True)
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        self.assertEqual(bc._interrupts, ["kinect-swipe"])
        self.assertEqual(bc.media, [])

    def test_a_swipe_with_a_pending_confirmation_cancels_it_only(self):
        bc = _fake_bc(pending=[("shutdown_pc", "")])
        self.run_logged(self.mod._dispatch, bc, kg.SWIPE_LEFT)
        self.assertEqual(bc._pending_confirmation, [])
        self.assertEqual(bc.media, [])

    def test_missing_media_action_is_reported_not_raised(self):
        bc = _fake_bc()
        del bc._act_media_next
        _, log = self.run_logged(self.mod._dispatch, bc, kg.SWIPE_RIGHT)
        self.assertIn("no media action available", log)


class CloseFocusedWindowTests(_SkillBase):
    def setUp(self):
        super().setUp()
        from core import actions as act
        self.act = act
        self.win = _Win()
        self.closed = []
        self._patch(self.mod, "_focused_window", lambda: self.win)
        self._patch(act._window_scope, "user_windows", lambda ws: list(ws))
        self._patch(act, "_is_terminal_window", lambda w: False)
        self._patch(act, "_browser_page_title", lambda bc, t: None)
        self._patch(act, "_send_close",
                    lambda w: (self.closed.append(w.title) or "closed"))

    def close(self, bc=None):
        bc = bc or _fake_bc()
        out, log = self.run_logged(self.mod._do_close_focused, bc)
        return bc, out, log

    def test_closes_the_focused_window_with_the_x_button_close(self):
        bc, out, log = self.close()
        self.assertEqual(self.closed, ["Notes - Editor"])
        self.assertIn("closed", out)
        self.assertIn("[two_hand_x]".replace("[", "").replace("]", ""), log)
        self.assertTrue(any("Closed Editor" in s for s in bc._spoken))

    def test_dispatch_routes_the_gesture_and_logs_it(self):
        bc = _fake_bc()
        _, log = self.run_logged(self.mod._dispatch, bc, kg.TWO_HAND_X)
        self.assertEqual(self.closed, ["Notes - Editor"])
        self.assertIn("[gesture] two_hand_x recognised", log)

    def test_its_own_setting_turns_it_off(self):
        self._cfg(KINECT_TWO_HAND_CLOSE_ENABLED=False)
        self.run_logged(self.mod._dispatch, _fake_bc(), kg.TWO_HAND_X)
        self.assertEqual(self.closed, [])

    def test_a_terminal_is_never_closed(self):
        self._patch(self.act, "_is_terminal_window", lambda w: True)
        bc, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("terminal", out)
        self.assertTrue(any("terminal" in s for s in bc._spoken))

    def test_a_window_that_may_host_jarvis_is_never_closed(self):
        self.win.title = "Claude Code - session"
        _, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("refused", out)

    def test_the_desktop_and_jarvis_windows_are_never_closed(self):
        self._patch(self.act._window_scope, "user_windows", lambda ws: [])
        _, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("refused", out)

    def test_a_fullscreen_window_is_never_closed(self):
        self.win.left, self.win.top, self.win.width, self.win.height = 0, 0, 2560, 1440
        bc, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("fullscreen", out)

    def test_a_fullscreen_game_the_air_mouse_stands_down_over_is_left_alone(self):
        self._patch(self.mod, "_per_app_blocked", lambda: True)
        _, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("fullscreen", out)

    def test_nothing_closes_while_the_real_mouse_or_keyboard_is_in_use(self):
        self._patch(self.mod, "_real_input_recent", lambda: True)
        _, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("real mouse/keyboard", out)

    def test_nothing_closes_while_jarvis_is_dormant(self):
        _, out, _ = self.close(_fake_bc(standby=True))
        self.assertEqual(self.closed, [])
        self.assertIn("dormant", out)

    def test_no_focused_window(self):
        self._patch(self.mod, "_focused_window", lambda: None)
        _, out, _ = self.close()
        self.assertEqual(self.closed, [])
        self.assertIn("no focused window", out)

    def test_a_browser_window_closes_only_its_active_tab(self):
        self.win.title = "Some Page - Google Chrome"
        tabs = []
        self._patch(self.act, "_browser_page_title", lambda bc, t: "Some Page")
        self._patch(self.act, "_close_browser_tab",
                    lambda bc, w: (tabs.append(w.title) or True))
        bc, out, _ = self.close()
        self.assertEqual(tabs, ["Some Page - Google Chrome"])
        self.assertEqual(self.closed, [])          # the window itself is untouched
        self.assertIn("tab", out)

    def test_an_elevated_window_gets_an_honest_refusal(self):
        self._patch(self.act, "_send_close", lambda w: "denied")
        bc, out, _ = self.close()
        self.assertIn("administrator", out)
        self.assertTrue(any("administrator" in s for s in bc._spoken))

    def test_a_close_that_errors_is_reported_not_raised(self):
        self._patch(self.act, "_send_close", lambda w: "error")
        _, out, _ = self.close()
        self.assertIn("close failed", out)


class RealScopeTests(_SkillBase):
    def test_the_real_shell_title_is_filtered_by_the_real_scope(self):
        """No scope patching here: core.window_scope itself must refuse the desktop."""
        from core import actions as act
        closed = []
        w = _Win(title="Program Manager")
        self._patch(self.mod, "_focused_window", lambda: w)
        self._patch(self.mod, "_in_standby", lambda bc: False)
        self._patch(act, "_send_close", lambda x: closed.append(x.title) or "closed")
        _, log = self.run_logged(self.mod._do_close_focused, _fake_bc())
        self.assertEqual(closed, [])
        self.assertIn("refused", log)


class PointMonitorTests(_SkillBase):
    def setUp(self):
        super().setUp()
        self.cursor = []
        self._patch(self.mod, "_set_cursor_pos",
                    lambda x, y: (self.cursor.append((x, y)) or True))

    def point(self, az, el=0.0, bc=None):
        detail = {"azimuth_deg": az, "elevation_deg": el}
        return self.run_logged(self.mod._do_point, bc or _fake_bc(), detail)

    def test_moves_the_cursor_to_the_middle_of_the_pointed_monitor(self):
        out, log = self.point(25.0)
        self.assertEqual(out, "right")
        self.assertEqual(self.cursor, [(2560 + 1280, 720)])
        self.assertIn("[gesture] point", log)
        self.assertEqual(self.mod.get_last_point_monitor()["monitor"], "right")

    def test_negative_origin_monitor(self):
        self.point(-25.0)
        self.assertEqual(self.cursor, [(-2560 + 1280, 720)])

    def test_aiming_high_picks_the_monitor_above(self):
        out, _ = self.point(0.0, 35.0)
        self.assertEqual(out, "top")
        self.assertEqual(self.cursor, [(1280, -1440 + 720)])

    def test_focus_action_activates_the_top_window_on_that_monitor(self):
        try:
            import pygetwindow  # noqa: F401  (patched below)
        except ImportError:
            self.skipTest("pygetwindow is Windows-only (not on the Linux CI)")
        self._cfg(KINECT_POINT_MONITOR_ACTION="focus")
        from core import actions as act
        left_win = _Win(title="Left Win", left=-2000, top=200, width=800, height=600)
        mid_win = _Win(title="Mid Win", left=500, top=200, width=800, height=600)
        gw = types.SimpleNamespace(getAllWindows=lambda: [mid_win, left_win])
        self._patch(act._window_scope, "user_windows", lambda ws: list(ws))
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            out, log = self.point(-25.0)
        self.assertEqual(out, "left")
        self.assertEqual((left_win.activated, mid_win.activated), (1, 0))
        self.assertEqual(self.cursor, [])
        self.assertIn("focused 'Left Win'", log)

    def test_both_actions(self):
        self._cfg(KINECT_POINT_MONITOR_ACTION="both")
        from core import actions as act
        w = _Win(title="Mid Win", left=500, top=200)
        gw = types.SimpleNamespace(getAllWindows=lambda: [w])
        self._patch(act._window_scope, "user_windows", lambda ws: list(ws))
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.point(0.0)
        self.assertEqual(self.cursor, [(1280, 720)])
        self.assertEqual(w.activated, 1)

    def test_nothing_moves_while_the_real_mouse_is_in_use(self):
        self._patch(self.mod, "_real_input_recent", lambda: True)
        out, log = self.point(0.0)
        self.assertEqual(out, "input")
        self.assertEqual(self.cursor, [])
        self.assertIn("real mouse/keyboard", log)

    def test_nothing_moves_while_jarvis_is_dormant(self):
        out, _ = self.point(0.0, bc=_fake_bc(standby=True))
        self.assertEqual(out, "dormant")
        self.assertEqual(self.cursor, [])

    def test_no_monitor_layout_is_reported(self):
        self._cfg(MONITORS={})
        out, log = self.point(0.0)
        self.assertEqual(out, "no monitor")
        self.assertIn("no monitor layout", log)

    def test_its_own_setting_turns_it_off(self):
        self._cfg(KINECT_POINT_MONITOR_ENABLED=False)
        self.run_logged(self.mod._dispatch, _fake_bc(), kg.POINT,
                        {"azimuth_deg": 0.0, "elevation_deg": 0.0})
        self.assertEqual(self.cursor, [])

    def test_dispatch_passes_the_aim_through_and_logs(self):
        _, log = self.run_logged(self.mod._dispatch, _fake_bc(), kg.POINT,
                                 {"azimuth_deg": 25.0, "elevation_deg": 0.0})
        self.assertEqual(self.cursor, [(3840, 720)])
        self.assertIn("[gesture] point recognised", log)

    def test_bad_detail_never_raises(self):
        out, _ = self.run_logged(self.mod._do_point, _fake_bc(), {"azimuth_deg": "x"})
        self.assertEqual(out, "no aim")


class PollWiringTests(_SkillBase):
    """_poll_once narrows the recognizer to the enabled gestures and hands the
    POINT aim to the dispatcher."""

    class _Rec:
        def __init__(self, gesture, detail=None):
            self.gesture, self.last_detail = gesture, detail or {}
            self.enabled = self.lasso = None

        def update(self, bodies):
            return self.gesture

        def reset(self):
            pass

        def set_enabled(self, g):
            self.enabled = g

        def set_require_lasso(self, on):
            self.lasso = on

    def setUp(self):
        super().setUp()
        self._patch(self.mod, "_is_staging", lambda: False)
        self._patch(self.mod, "_air_mouse_engaged", lambda: False)
        self._patch(self.mod, "_two_hand_active", lambda: False)
        self._patch(self.mod, "_shared_two_hand_on", lambda kb: False)
        self._cfg(KINECT_GESTURES_ENABLED=True)
        kb = types.ModuleType("audio.kinect_bridge")
        kb.get_enabled = lambda: True
        kb.available = lambda: (True, "")
        kb.get_bodies = lambda: [{"id": 1}]
        p = mock.patch.dict(sys.modules, {"audio.kinect_bridge": kb})
        p.start()
        self.addCleanup(p.stop)

    def test_recognizer_is_narrowed_each_tick(self):
        self._cfg(KINECT_POINT_MONITOR_ENABLED=False, KINECT_POINT_REQUIRE_LASSO=False)
        rec = self._Rec(None)
        self.mod._poll_once(rec, _fake_bc())
        self.assertNotIn(kg.POINT, rec.enabled)
        self.assertIn(kg.SWIPE_LEFT, rec.enabled)
        self.assertFalse(rec.lasso)

    def test_point_aim_reaches_the_handler(self):
        cursor = []
        self._patch(self.mod, "_set_cursor_pos", lambda x, y: cursor.append((x, y)) or True)
        rec = self._Rec(kg.POINT, {"azimuth_deg": -25.0, "elevation_deg": 0.0})
        got, log = self.run_logged(self.mod._poll_once, rec, _fake_bc())
        self.assertEqual(got, kg.POINT)
        self.assertEqual(cursor, [(-1280, 720)])
        self.assertEqual(self.mod.get_last_gesture()["gesture"], kg.POINT)

    def test_labels_exist_for_the_badge(self):
        self.assertEqual(self.mod.gesture_label_for(kg.TWO_HAND_X), "CLOSE")
        self.assertEqual(self.mod.gesture_label_for(kg.POINT), "POINT")


class EndToEndSyntheticFrameTests(_SkillBase):
    """Real recognizer + real skill poll tick on a clock, synthetic frames."""

    def test_crossed_wrists_close_the_focused_window(self):
        from core import actions as act
        closed = []
        w = _Win()
        self._patch(self.mod, "_focused_window", lambda: w)
        self._patch(act._window_scope, "user_windows", lambda ws: list(ws))
        self._patch(act, "_is_terminal_window", lambda x: False)
        self._patch(act, "_browser_page_title", lambda bc, t: None)
        self._patch(act, "_send_close", lambda x: closed.append(x.title) or "closed")
        self._patch(self.mod, "_is_staging", lambda: False)
        self._patch(self.mod, "_air_mouse_engaged", lambda: False)
        self._patch(self.mod, "_two_hand_active", lambda: False)
        self._patch(self.mod, "_shared_two_hand_on", lambda kb: False)
        self._cfg(KINECT_GESTURES_ENABLED=True)

        clk = _Clock(5000.0)
        rec = kg.GestureRecognizer(now_fn=clk)
        frame = {"b": _body()}
        kb = types.ModuleType("audio.kinect_bridge")
        kb.get_enabled = lambda: True
        kb.available = lambda: (True, "")
        kb.get_bodies = lambda: [frame["b"]]
        bc = _fake_bc()
        with mock.patch.dict(sys.modules, {"audio.kinect_bridge": kb}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                for body, secs in ((_body(), 0.7), (_x_body(), 1.6)):
                    frame["b"] = body
                    for _ in range(int(secs / STEP)):
                        clk.advance(STEP)
                        self.mod._poll_once(rec, bc)
        self.assertEqual(closed, ["Notes - Editor"])
        self.assertEqual(buf.getvalue().count("[gesture] two_hand_x recognised"), 1)


if __name__ == "__main__":
    unittest.main()
