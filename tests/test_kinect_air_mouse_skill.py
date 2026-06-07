"""Tests for skills/kinect_air_mouse — the Kinect air-mouse.

Drives the PURE CORE (reach-box mapping, EMA smoothing, grip debounce, the
open→move / closed→right-button / re-open→release state machine, the overlay
colour mapping) directly, and the LIVE _poll_once path with a fake kinect_bridge
+ the real mouse actuation mocked out — so NO sensor, NO real cursor, NO Qt is
touched. Asserts the owner-facing contract:

  * OPEN hand            → the cursor MOVES (no button), overlay "track"/cyan,
  * OPEN → CLOSED        → the RIGHT button goes DOWN once, overlay "grab"/gold,
  * CLOSED held + move   → cursor still moves (a right-DRAG), button stays down,
  * CLOSED → OPEN        → the RIGHT button goes UP once (close→open = a click),
  * a single flickered CLOSED frame is DEBOUNCED (no stray button),
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
    """A stand-in audio.kinect_bridge exposing only what the skill reads."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_hand_states = lambda: (hand_states if hand_states is not None else
                                 {"right": "unknown", "left": "unknown",
                                  "tracked": False, "ts": 0.0})
    return m


def _body(*, hand_x=0.0, hand_y=0.30, grip_right="open", grip_left="unknown",
          distance=1.8, side="right"):
    """A get_bodies()-shaped body dict with a hand joint + grip on `side`."""
    joints = {
        "head": (0.0, 0.6, distance, 2),
        "spine_shoulder": (0.0, 0.0, distance, 2),
        f"hand_{side}": (hand_x, hand_y, distance, 2),
    }
    return {
        "id": 0, "joints": joints,
        "head": (0.0, 0.6, distance), "distance_m": distance, "facing": True,
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
        buttons) lists that fill as the skill acts."""
        moves: list = []
        buttons: list = []
        p1 = mock.patch.object(mod, "_set_cursor_pos",
                               lambda x, y: (moves.append((x, y)) or True))
        p2 = mock.patch.object(mod, "_mouse_button",
                               lambda action: (buttons.append(action) or True))
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
#  PURE CORE
# ══════════════════════════════════════════════════════════════════════════
class ReachBoxTests(_Base):
    def test_center_maps_to_screen_center(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        # Centre of the box → centre of the screen (±1 px rounding).
        self.assertAlmostEqual(px, (2560 - 1) // 2, delta=2)
        self.assertAlmostEqual(py, (1440 - 1) // 2, delta=2)

    def test_x_is_mirrored(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # Hand to the sensor's RIGHT (+x) should map to the LEFT of the screen
        # (mirror, so it tracks like a real mirror from the user's seat).
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertLess(right_px, left_px)
        self.assertEqual(right_px, 0)
        self.assertEqual(left_px, 2560 - 1)

    def test_y_inverted_camera_up_is_screen_top(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # Hand UP (+y in camera space) → TOP of the screen (py small).
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


class EMATests(_Base):
    def test_first_value_seeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        self.assertEqual(e.update(10.0), 10.0)   # first sample is the seed

    def test_smooths_toward_target(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0)
        v = e.update(10.0)
        self.assertAlmostEqual(v, 5.0)           # halfway with alpha 0.5

    def test_reset_reseeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0); e.update(10.0)
        e.reset()
        self.assertEqual(e.update(100.0), 100.0)


class GripDebouncerTests(_Base):
    def test_requires_n_consecutive_frames(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")   # 1
        self.assertEqual(d.update("closed"), "open")   # 2
        self.assertEqual(d.update("closed"), "closed")  # 3 → flips

    def test_single_flicker_is_ignored(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")   # a lone flicker
        self.assertEqual(d.update("open"), "open")     # back to open
        # Streak was broken → stable never moved off "open".
        self.assertEqual(d.stable, "open")

    def test_unknown_holds_current_stable(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=2, initial="closed")
        # An ambiguous frame must not flip a held grip (dead-man releases, not
        # a single 'unknown').
        self.assertEqual(d.update("unknown"), "closed")
        self.assertEqual(d.update("lasso"), "closed")


class OverlayColorTests(_Base):
    def test_track_is_cyan_grab_is_gold(self):
        mod = self._load()
        self.assertEqual(mod.overlay_color_for("track"), "cyan")
        self.assertEqual(mod.overlay_color_for("grab"), "gold")
        self.assertEqual(mod.overlay_color_for("hidden"), "cyan")


class ControllerStateMachineTests(_Base):
    """The open→move / closed→RMB-down / re-open→RMB-up contract, asserted on
    AirMouseController decisions (the pure brain the live loop applies)."""

    def _ctrl(self, mod, debounce=1):
        # debounce=1 makes each grip change take effect immediately so the state
        # machine is easy to assert; the debounce itself is tested separately.
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=debounce)

    def test_open_hand_moves_no_button_cyan(self):
        mod = self._load()
        c = self._ctrl(mod)
        d = c.update((0.1, 0.30), "open", True)
        self.assertIsNotNone(d.cursor)
        self.assertIsNone(d.button)
        self.assertEqual(d.overlay, "track")
        self.assertEqual(mod.overlay_color_for(d.overlay), "cyan")

    def test_close_presses_right_down_once_gold(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True)
        d = c.update((0.0, 0.30), "closed", True)
        self.assertEqual(d.button, "down")
        self.assertEqual(d.overlay, "grab")
        self.assertEqual(mod.overlay_color_for(d.overlay), "gold")
        # Held: a second closed frame does NOT re-press.
        d2 = c.update((0.0, 0.30), "closed", True)
        self.assertIsNone(d2.button)
        self.assertTrue(c.button_is_down)

    def test_closed_hand_still_moves_drag(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True)
        c.update((0.0, 0.30), "closed", True)
        d = c.update((0.2, 0.10), "closed", True)   # move while closed
        self.assertIsNotNone(d.cursor)              # cursor still tracks → drag
        self.assertIsNone(d.button)                 # button stays down (no edge)
        self.assertEqual(d.overlay, "grab")

    def test_reopen_releases_right_up_once(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True)
        c.update((0.0, 0.30), "closed", True)       # down
        d = c.update((0.0, 0.30), "open", True)     # re-open → up
        self.assertEqual(d.button, "up")
        self.assertEqual(d.overlay, "track")
        self.assertFalse(c.button_is_down)

    def test_deadman_release_when_untracked_while_held(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True)
        c.update((0.0, 0.30), "closed", True)       # button down
        d = c.update(None, "unknown", False)        # hand lost mid-grip
        self.assertEqual(d.button, "up")            # dead-man releases
        self.assertEqual(d.overlay, "hidden")
        self.assertIsNone(d.cursor)
        self.assertFalse(c.button_is_down)
        # Idempotent: a second lost frame keeps it hidden with no new edge.
        d2 = c.update(None, "unknown", False)
        self.assertIsNone(d2.button)
        self.assertEqual(d2.overlay, "hidden")

    def test_flicker_does_not_fire_button_with_real_debounce(self):
        mod = self._load()
        # Real debounce (3 frames): a single closed flicker must NOT press.
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=3)
        c.update((0.0, 0.30), "open", True)
        d1 = c.update((0.0, 0.30), "closed", True)  # flicker frame 1
        d2 = c.update((0.0, 0.30), "open", True)    # back to open
        self.assertIsNone(d1.button)
        self.assertIsNone(d2.button)
        self.assertFalse(c.button_is_down)


# ══════════════════════════════════════════════════════════════════════════
#  LIVE _poll_once — sensor + mouse mocked
# ══════════════════════════════════════════════════════════════════════════
class PollActsTests(_Base):
    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1)

    def test_open_hand_moves_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        bridge = _fake_bridge(bodies=[_body(grip_right="open")])
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, bridge)
        self.assertEqual(len(moves), 1)             # cursor moved
        self.assertEqual(buttons, [])               # no button
        self.assertEqual(d.overlay, "track")

    def test_close_then_open_clicks_right(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # open → closed → open : a right-click (down then up).
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(buttons, ["down", "up"])

    def test_deadman_releases_when_hand_lost(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        self.assertEqual(buttons, ["down"])
        # Hand leaves the frame (no bodies) → dead-man release.
        mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        self.assertEqual(buttons, ["down", "up"])


class PollGateTests(_Base):
    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=1)

    def test_noop_side_effects_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)                     # air-mouse OFF
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])                 # cursor NOT moved
        self.assertEqual(buttons, [])               # no button
        self.assertIsNotNone(d)                     # controller still advanced

    def test_flag_off_during_drag_still_releases(self):
        mod = self._load()
        self._not_staging(mod)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Grab while enabled.
        self._patch_flag(True)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        self.assertEqual(buttons, ["down"])
        # Now flip OFF mid-drag, then re-open: the pending 'up' must still fire
        # so the button isn't stranded down.
        from core import config as cfg
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", False, create=True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(buttons, ["down", "up"])

    def test_noop_when_staging(self):
        mod = self._load()
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start(); self.addCleanup(p.stop)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
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
        mod._poll_once(ctrl, _fake_bridge(enabled=False,
                                          bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])                 # dead-man: not tracked

    def test_noop_when_sensor_unavailable(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(available=(False, "no sensor"),
                                          bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])


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

    def test_off_persists_flag_and_releases(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer({"KINECT_AIR_MOUSE_ENABLED": True})
        # off() releases the button + clears the overlay — make those inert.
        released = []
        with mock.patch.object(mod, "_mouse_button",
                               lambda a: released.append(a)), \
                mock.patch.object(mod, "_shutdown_overlay", lambda: None), \
                mock.patch.object(mod, "_clear_overlay_state", lambda: None):
            out = mod.air_mouse_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("KINECT_AIR_MOUSE_ENABLED"))
        self.assertIn("up", released)               # safety release on disable

    def test_on_warns_when_sensor_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=False))
        out = mod.air_mouse_on("")
        self.assertIn("kinect", out.lower())        # mentions the sensor is off


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


# ─── registration wires the three actions ──────────────────────────────────
class RegisterTests(_Base):
    def test_register_adds_actions_without_starting_real_thread(self):
        # neuter_threads=True (harness default) makes Thread.start a no-op, so
        # register() wires the actions but never spins the live poller.
        mod, actions = load_skill_isolated("kinect_air_mouse", register=True)
        for name in ("air_mouse_on", "air_mouse_off", "air_mouse_status"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


if __name__ == "__main__":
    unittest.main()
