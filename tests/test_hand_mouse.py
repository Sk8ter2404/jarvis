"""The hand mouse (air mouse v2): core/hand_mouse.py (state machine, One Euro
filter, mapping) plus its wiring in skills/hand_tracking.py and the
AIR_MOUSE_ENGINE routing in skills/kinect_air_mouse.py.

FAKES ONLY. The OS input layer (core.hand_mouse.WindowsMouse) is never
constructed here: every test drives a FakeMouse, so the real cursor cannot move
and no real input can be sent while the suite runs.
"""
import importlib.util
import json
import math
import os
import sys
import types
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import config  # noqa: E402
from core import hand_mouse as hm  # noqa: E402
from tests._skill_harness import load_skill_isolated  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "skill_hand_tracking_mouse_under_test",
    os.path.join(_ROOT, "skills", "hand_tracking.py"))
ht = importlib.util.module_from_spec(_spec)
sys.modules["skill_hand_tracking_mouse_under_test"] = ht
_spec.loader.exec_module(ht)

SCREEN = (0, 0, 1920, 1080)
_REAL_PER_APP_BLOCKED = ht._per_app_blocked     # before any test patches it


# ── fakes and builders ─────────────────────────────────────────────────────
class FakeMouse(hm.MouseOutput):
    def __init__(self):
        self.events = []             # ("move", x, y) | ("down",) | ("up",)
        self.pos = (500, 500)
        self.override = None         # set to simulate the real mouse being moved

    def move(self, x, y):
        self.events.append(("move", x, y))
        self.pos = (x, y)

    def button(self, down):
        self.events.append(("down",) if down else ("up",))

    def wheel(self, delta):
        self.events.append(("wheel", delta))

    def idle_s(self):
        return self.idle

    idle = None                      # seconds since real input (None = unknown)

    def position(self):
        return self.override if self.override is not None else self.pos

    def screen_rect(self):
        return SCREEN

    @property
    def moves(self):
        return [e[1:] for e in self.events if e[0] == "move"]

    @property
    def buttons(self):
        return [e[0] for e in self.events if e[0] in ("down", "up")]


def hand(side="Right", state="unknown", pt=(0.5, 0.45), palm=None,
         facing=False, score=0.9, size=0.16, hid=0):
    """A tracker hand. ``size`` = hand size over the frame height (the wrist
    sits ``size`` below the middle-finger MCP, as in a 640x480 frame)."""
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    lm[hm.MIDDLE_MCP] = [pt[0], pt[1] + 0.04, 0.0]
    lm[hm.WRIST] = [pt[0], pt[1] + 0.04 + size, 0.0]
    lm[hm.INDEX_MCP] = [pt[0] - 0.02, pt[1] + 0.045, 0.0]
    lm[hm.PINKY_MCP] = [pt[0] + 0.02, pt[1] + 0.05, 0.0]
    lm[hm.THUMB_TIP] = [pt[0] - 0.01, pt[1], 0.0]
    lm[hm.INDEX_TIP] = [pt[0] + 0.01, pt[1], 0.0]
    return {"id": hid, "side": side, "score": score, "lm": lm, "state": state,
            "pinch": 1.0 if state == "pinch" else 0.0,
            "palm": list(palm or pt), "palm_facing": facing}


def msg(*hands):
    return {"type": "hands", "hands": list(hands)}


class Rig:
    """A HandMouse on a FakeMouse with a simulated clock and frame rate."""

    def __init__(self, fps=15.0, params=None):
        self.out = FakeMouse()
        self.cues = []
        self.block = ""
        self.m = hm.HandMouse(self.out, blocker=lambda: self.block,
                              cue=lambda k, r: self.cues.append((k, r)),
                              params=params, log=lambda s: None)
        self.m.set_enabled(True)
        self.fps = fps
        self.t = 1000.0

    def frame(self, **kw):
        self.t += 1.0 / self.fps
        self.m.on_hands(msg(hand(**kw)), self.t)

    def empty(self):
        self.t += 1.0 / self.fps
        self.m.on_hands(msg(), self.t)

    def run(self, secs, **kw):
        for _ in range(max(1, round(secs * self.fps))):
            self.frame(**kw)

    def hold_event(self, side="Right"):
        return self.m.on_gesture({"type": "gesture", "name": "pinch_hold",
                                  "side": side, "pos": [0.5, 0.45], "value": 0.6},
                                 self.t)

    def engage(self, pt=(0.5, 0.45)):
        """pinch ~0.65 s, the engine's pinch_hold event, a bit more pinch, then
        release into the follow pose. Leaves the machine ENGAGED and FREE."""
        self.run(0.65, state="pinch", pt=pt)
        assert self.hold_event() == "engaged"
        self.run(0.3, state="pinch", pt=pt)
        self.run(0.25, state="unknown", pt=pt)
        assert self.m.phase == hm.ENGAGED, self.m.status()


# ── off by default, no engage without the hold ─────────────────────────────
class DefaultsTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertIs(config.HAND_MOUSE_ENABLED, False)
        self.assertEqual(config.AIR_MOUSE_ENGINE, "kinect")
        m = hm.HandMouse(FakeMouse(), log=lambda s: None)
        self.assertFalse(m.enabled)
        self.assertEqual(m.phase, hm.IDLE)

    def test_disabled_machine_ignores_everything(self):
        r = Rig()
        r.m.set_enabled(False)
        r.run(1.0, state="pinch")
        self.assertEqual(r.hold_event(), "")
        self.assertEqual(r.out.events, [])
        self.assertEqual(r.m.phase, hm.IDLE)

    def test_pinching_without_the_hold_event_never_engages(self):
        r = Rig()
        r.run(3.0, state="pinch")             # a long pinch, but no pinch_hold
        r.run(1.0, state="unknown")
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertEqual(r.out.events, [])

    def test_other_gestures_do_not_engage(self):
        r = Rig()
        r.run(0.5, state="pinch")
        for name in ("pinch_start", "pinch_end", "palm_hold", "fist_grab", "clap"):
            r.m.on_gesture({"type": "gesture", "name": name, "side": "Right"}, r.t)
        self.assertEqual(r.m.phase, hm.IDLE)

    def test_hold_event_without_a_pinching_hand_is_ignored(self):
        r = Rig()
        r.run(0.5, state="open")
        self.assertTrue(r.hold_event().startswith("ignored"))
        self.assertEqual(r.m.phase, hm.IDLE)
        r2 = Rig()                                  # no frame at all
        self.assertTrue(r2.hold_event().startswith("ignored"))
        r3 = Rig()                                  # a stale frame
        r3.run(0.3, state="pinch")
        r3.t += 2.0
        self.assertTrue(r3.hold_event().startswith("ignored"))

    def test_engage_gives_a_cue_and_moves_nothing(self):
        r = Rig()
        r.run(0.65, state="pinch")
        self.assertEqual(r.hold_event(), "engaged")
        r.run(1.0, state="pinch", pt=(0.6, 0.5))
        self.assertEqual(r.cues, [("engage", "")])
        self.assertEqual(r.out.events, [])           # engaging never moves or clicks

    def test_engaging_pinch_is_consumed_then_the_cursor_follows(self):
        r = Rig()
        r.run(0.65, state="pinch")
        r.hold_event()
        r.run(2.0, state="pinch")                    # still pinching: nothing
        self.assertEqual(r.out.events, [])
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        self.assertTrue(r.out.moves)
        self.assertEqual(r.out.buttons, [])
        x, y = r.out.moves[0]
        self.assertAlmostEqual(x, 959.5, delta=2)    # centre of the box -> centre of screen
        self.assertAlmostEqual(y, 539.5, delta=2)


# ── stillness and smoothing ────────────────────────────────────────────────
def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _noise(i, amp):
    # deterministic, zero-mean, not periodic with the frame rate
    return amp * math.sin(i * 2.399963) * (1.0 if i % 3 else -0.8)


class SmoothingTests(unittest.TestCase):
    def _still(self, fps):
        r = Rig(fps)
        r.engage()
        n0 = len(r.out.moves)
        for i in range(int(4 * fps)):
            r.frame(state="unknown",
                    pt=(0.5 + _noise(i, 0.0015), 0.45 + _noise(i + 7, 0.0015)))
        return len(r.out.moves) - n0

    def test_still_hand_still_cursor_12_fps(self):
        self.assertLessEqual(self._still(12), 1)

    def test_still_hand_still_cursor_30_fps(self):
        self.assertLessEqual(self._still(30), 1)

    def _sweep(self, fps, span, secs=2.0):
        """The hand sweeps left to right (span = fraction of the frame, over
        `secs`) with tracking noise. Returns (raw mapped x, cursor x)."""
        r = Rig(fps)
        r.engage(pt=(0.30, 0.45))
        raw, cur = [], []
        n = int(fps * secs)
        for i in range(n + 1):
            x = 0.30 + span * i / n + _noise(i, 0.002)
            r.frame(state="unknown", pt=(x, 0.45))
            raw.append((x - 0.2) / 0.6 * 1919)
            cur.append(r.out.pos[0])
        return raw, cur

    @staticmethod
    def _step_sd(v):
        d = [v[i + 1] - v[i] for i in range(4, len(v) - 1)]
        m = sum(d) / len(d)
        return math.sqrt(sum((x - m) ** 2 for x in d) / len(d)), m

    def _check_smooth(self, fps):
        for span in (0.08, 0.20):
            raw, cur = self._sweep(fps, span)
            sd_raw, _ = self._step_sd(raw)
            sd_cur, mean = self._step_sd(cur)
            self.assertLess(sd_cur, 0.8 * sd_raw, (fps, span))     # smoother ...
            want = span / 0.6 * 1919 / (2.0 * fps)                  # ... but it keeps up
            self.assertAlmostEqual(mean, want, delta=0.15 * want, msg=(fps, span))

    def test_smoothing_at_12_fps(self):
        self._check_smooth(12)

    def test_smoothing_at_30_fps(self):
        self._check_smooth(30)

    def test_a_fast_sweep_has_little_lag_at_either_rate(self):
        for fps in (12, 30):
            raw, cur = self._sweep(fps, 0.40, secs=1.0)
            self.assertLess(abs(raw[-1] - cur[-1]), 60, fps)

    def test_one_euro_is_dt_aware(self):
        # the same step response, sampled at 12 fps and at 30 fps, reaches about
        # the same value after the same wall-clock time
        def settle(fps):
            f = hm.OneEuro(1.0, 0.0, 1.0)
            f(0.0, 0.0)
            v = 0.0
            for i in range(1, int(fps * 0.5) + 1):
                v = f(100.0, i / fps)
            return v
        a, b = settle(12), settle(30)
        self.assertGreater(a, 60)
        self.assertGreater(b, 60)
        self.assertAlmostEqual(a, b, delta=12)

    def test_cursor_is_clamped_to_the_screen(self):
        r = Rig()
        r.engage()
        r.run(1.0, state="unknown", pt=(0.99, 0.99))
        x, y = r.out.pos
        self.assertLessEqual(x, 1919)
        self.assertLessEqual(y, 1079)

    def test_screen_rect_override_maps_to_another_monitor(self):
        r = Rig(params=hm.Params(screen_rect=(-1920, 0, 1920, 1080)))
        r.engage(pt=(0.2, 0.15))                       # top-left of the box
        self.assertLessEqual(r.out.pos[0], -1900)
        self.assertLessEqual(r.out.pos[1], 20)


# ── click and drag ─────────────────────────────────────────────────────────
class ClickDragTests(unittest.TestCase):
    def test_tap_is_exactly_one_left_click_at_the_frozen_spot(self):
        r = Rig()
        r.engage()
        r.run(0.5, state="unknown", pt=(0.5, 0.45))
        before = r.out.pos
        n_events = len(r.out.events)
        # a quick pinch (0.2 s) while the pinch point drifts as the fingers close
        for i in range(3):
            r.frame(state="pinch", pt=(0.5 + 0.01 * i, 0.45 + 0.01 * i))
        r.run(0.5, state="unknown", pt=(0.5, 0.45))
        new = r.out.events[n_events:]
        self.assertEqual([e[0] for e in new if e[0] != "move"], ["down", "up"])
        self.assertEqual(r.m.clicks, 1)
        self.assertEqual(r.out.pos, before)            # the click did not nudge it
        # nothing moved between the down and the up
        di = [e[0] for e in new].index("down")
        self.assertEqual([e[0] for e in new][di:di + 2], ["down", "up"])

    def test_two_taps_are_two_clicks(self):
        r = Rig()
        r.engage()
        for _ in range(2):
            r.run(0.4, state="unknown")
            r.run(0.2, state="pinch")
        r.run(0.4, state="unknown")
        self.assertEqual(r.out.buttons, ["down", "up", "down", "up"])

    def test_the_engaging_pinch_never_drags(self):
        r = Rig()
        r.run(0.65, state="pinch")
        r.hold_event()
        r.run(5.0, state="pinch", pt=(0.6, 0.5))       # held and moved for 5 s
        self.assertNotIn("down", r.out.buttons)
        r.run(0.3, state="unknown")
        self.assertEqual(r.out.buttons, [])

    def test_drag_needs_a_second_pinch_held(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.4, 0.4))
        r.out.events.clear()
        # second pinch: not yet a drag at 0.3 s ...
        r.run(0.3, state="pinch", pt=(0.4, 0.4))
        self.assertNotIn("down", r.out.buttons)
        # ... pressed once held past the threshold ...
        r.run(0.3, state="pinch", pt=(0.4, 0.4))
        self.assertEqual(r.out.buttons, ["down"])
        self.assertTrue(r.m.button_down())
        # ... the cursor follows the hand while it is held ...
        n = len(r.out.moves)
        for i in range(1, 16):
            r.frame(state="pinch", pt=(0.4 + 0.015 * i, 0.4 + 0.01 * i))
        self.assertGreater(len(r.out.moves), n)
        self.assertGreater(r.out.pos[0], 700)
        # ... and releasing the pinch releases the button (no extra click).
        r.run(0.3, state="unknown", pt=(0.6, 0.55))
        self.assertEqual(r.out.buttons, ["down", "up"])
        self.assertFalse(r.m.button_down())
        self.assertEqual(r.m.clicks, 0)

    def test_the_cursor_is_frozen_during_a_pending_pinch(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        n = len(r.out.moves)
        r.run(0.3, state="pinch", pt=(0.7, 0.6))        # hand jumps while pinched
        self.assertEqual(len(r.out.moves), n)


# ── every way out releases the button ──────────────────────────────────────
class DisengageTests(unittest.TestCase):
    def _dragging(self):
        r = Rig()
        r.engage()
        r.run(0.2, state="unknown")
        r.run(0.7, state="pinch")
        self.assertTrue(r.m.button_down())
        return r

    def _assert_released(self, r, reason_part):
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertFalse(r.m.button_down())
        self.assertEqual(r.out.buttons, ["down", "up"])
        self.assertIn(reason_part, r.m.last_reason)
        self.assertEqual(r.cues[-1][0], "disengage")

    def test_open_palm_held_disengages(self):
        r = Rig()
        r.engage()
        r.run(0.2, state="unknown")
        r.run(0.2, state="open", facing=True, palm=(0.5, 0.5))
        self.assertEqual(r.m.phase, hm.ENGAGED)          # not yet 0.4 s
        r.run(0.4, state="open", facing=True, palm=(0.5, 0.5))
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertIn("open palm", r.m.last_reason)

    def test_a_moving_open_hand_or_one_not_facing_does_not_disengage(self):
        r = Rig()
        r.engage()
        for i in range(30):                              # 2 s of open hand sweeping
            r.frame(state="open", facing=True, palm=(0.3 + 0.012 * i, 0.5))
        self.assertEqual(r.m.phase, hm.ENGAGED)
        r.run(2.0, state="open", facing=False, palm=(0.5, 0.5))
        self.assertEqual(r.m.phase, hm.ENGAGED)

    def test_open_palm_release_can_be_turned_off(self):
        r = Rig(params=hm.Params(palm_release_s=0.0))
        r.engage()
        r.run(2.0, state="open", facing=True, palm=(0.5, 0.5))
        self.assertEqual(r.m.phase, hm.ENGAGED)

    def test_palm_disengage_releases_nothing_it_does_not_hold(self):
        r = Rig()
        r.engage()
        r.run(1.0, state="open", facing=True, palm=(0.5, 0.5))
        self.assertEqual(r.out.buttons, [])

    def test_hand_gone_for_a_second_disengages_and_releases(self):
        r = self._dragging()
        for _ in range(int(0.5 * r.fps)):
            r.empty()
        self.assertEqual(r.m.phase, hm.ENGAGED)          # 0.5 s: still engaged
        for _ in range(int(0.7 * r.fps)):
            r.empty()
        self._assert_released(r, "left the view")

    def test_hand_gone_with_no_frames_at_all_is_caught_by_tick(self):
        r = self._dragging()
        r.m.tick(r.t + 0.5)
        self.assertEqual(r.m.phase, hm.ENGAGED)
        r.m.tick(r.t + 1.2)
        self._assert_released(r, "left the view")

    def test_only_the_engaged_hand_counts(self):
        r = Rig()
        r.engage()
        r.t += 1.0 / r.fps
        r.m.on_hands(msg(hand(side="Left", state="pinch")), r.t)   # other hand only
        self.assertEqual(r.m.phase, hm.ENGAGED)

    def test_tracker_stop_disengages_and_releases(self):
        r = self._dragging()
        r.m.on_tracker_stopped(r.t)
        self._assert_released(r, "tracker stopped")

    def test_hand_mouse_off_disengages_and_releases(self):
        r = self._dragging()
        r.m.set_enabled(False, r.t)
        self._assert_released(r, "hand mouse off")
        self.assertFalse(r.m.enabled)
        r.run(0.5, state="pinch")
        self.assertEqual(r.out.buttons, ["down", "up"])  # and it stays quiet

    def test_a_blocker_turning_on_disengages_and_releases(self):
        r = self._dragging()
        r.block = "game mode is on"
        r.run(1.0, state="pinch")
        self._assert_released(r, "game mode")

    def test_touching_the_real_mouse_yields(self):
        r = Rig()
        r.engage()
        r.out.override = (50, 50)                        # the user grabbed the mouse
        r.run(0.5, state="unknown", pt=(0.6, 0.5))
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertIn("moved the mouse", r.m.last_reason)

    def test_disengage_is_idempotent(self):
        r = Rig()
        self.assertFalse(r.m.disengage("x"))
        r.engage()
        self.assertTrue(r.m.disengage("x"))
        self.assertFalse(r.m.disengage("x"))
        self.assertEqual(len([c for c in r.cues if c[0] == "disengage"]), 1)

    def test_it_can_engage_again_after_disengaging(self):
        r = Rig()
        r.engage()
        r.m.disengage("test")
        r.engage()
        self.assertEqual(r.m.engagements, 2)


# ── blockers block engaging ────────────────────────────────────────────────
class BlockerTests(unittest.TestCase):
    def test_a_blocker_stops_engaging(self):
        r = Rig()
        r.block = "you are in a call"
        r.run(0.65, state="pinch")
        self.assertEqual(r.hold_event(), "ignored: you are in a call")
        r.run(1.0, state="pinch")
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertEqual(r.out.events, [])

    def test_a_raising_blocker_does_not_break_it(self):
        r = Rig()
        r.m.blocker = mock.Mock(side_effect=RuntimeError("boom"))
        r.run(0.65, state="pinch")
        self.assertEqual(r.hold_event(), "engaged")


# ── the skill: wiring, listener, voice actions ─────────────────────────────
class FakeLauncher:
    def __init__(self):
        self._running = False
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1
        self._running = True
        return True, "started"

    def stop(self):
        self.stopped += 1
        self._running = False

    def running(self):
        return self._running

    def wanted(self):
        return self._running

    def gave_up(self):
        return False

    def heartbeat(self):
        return None


class _SkillBase(unittest.TestCase):
    def setUp(self):
        self.la = FakeLauncher()
        self.fake_out = FakeMouse()
        self.built = []

        def factory():
            self.built.append(1)
            return self.fake_out
        ht.set_mouse_output_factory(factory)            # NEVER the real Windows layer
        self.addCleanup(ht.set_mouse_output_factory, None)
        for target, name, val in (
                (ht, "_launcher", lambda: self.la),
                (ht, "_ensure_listener", lambda: None),
                (ht, "_game_mode_active", lambda: False),
                (ht, "_fullscreen_app", lambda: False),
                (ht, "_in_call", lambda: False),
                (ht, "_per_app_blocked", lambda: False),
                (ht, "_staging", lambda: False)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        for key, val in (("HAND_MOUSE_ENABLED", False), ("HAND_MOUSE_CUE_SOUND", False),
                         ("HAND_TRACKING_ENABLED", False)):
            p = mock.patch.object(config, key, val)
            p.start()
            self.addCleanup(p.stop)
        ht._holo["owned"] = False

    def feed_pinch_and_engage(self):
        m = ht._mouse[0]
        t = 5000.0
        for i in range(10):
            m.on_hands(msg(hand(state="pinch")), t + i * 0.07)
        res = m.on_gesture({"type": "gesture", "name": "pinch_hold", "side": "Right"},
                           t + 0.7)
        return m, t + 0.7, res


class SkillActionTests(_SkillBase):
    def test_register_adds_the_three_actions_and_stays_off(self):
        acts = {}
        with mock.patch.object(ht, "start_at_boot") as boot:
            ht.register(acts)
        for a in ("hand_mouse_on", "hand_mouse_off", "hand_mouse_status"):
            self.assertTrue(callable(acts[a]))
            self.assertIn(a, ht.SPEAK_VERBATIM_ACTIONS)
        boot.assert_not_called()
        self.assertEqual(self.built, [])               # nothing built while off

    def test_boot_setting_starts_the_tracker(self):
        with mock.patch.object(config, "HAND_MOUSE_ENABLED", True), \
                mock.patch.object(ht, "start_at_boot") as boot:
            ht.register({})
        boot.assert_called_once()

    def test_start_with_the_tracker_enables_it(self):
        with mock.patch.object(config, "HAND_MOUSE_ENABLED", True):
            ok, _ = ht.start_tracking()
            self.assertTrue(ok)
            self.assertTrue(ht._mouse_enabled())

    def test_on_starts_the_tracker_and_enables_it(self):
        out = ht._hand_mouse_on("")
        self.assertIn("Hand mouse on, sir", out)
        self.assertEqual(self.la.started, 1)
        self.assertTrue(config.HAND_MOUSE_ENABLED)
        self.assertTrue(ht._mouse_enabled())
        self.assertEqual(self.fake_out.events, [])     # turning it on moves nothing

    def test_on_does_not_restart_a_running_tracker(self):
        self.la._running = True
        ht._hand_mouse_on("")
        self.assertEqual(self.la.started, 0)

    def test_on_twice_says_already_on(self):
        ht._hand_mouse_on("")
        self.assertIn("already on", ht._hand_mouse_on(""))

    def test_on_is_honest_when_the_tracker_will_not_start(self):
        self.la.start = lambda: (False, "the model is missing")
        out = ht._hand_mouse_on("")
        self.assertIn("couldn't start the hand mouse", out)
        self.assertIn("the model is missing", out)
        self.assertFalse(config.HAND_MOUSE_ENABLED)

    def test_on_says_when_a_blocker_is_up(self):
        with mock.patch.object(ht, "_game_mode_active", lambda: True):
            self.assertIn("won't take the cursor while game mode is on",
                          ht._hand_mouse_on(""))

    def test_off_and_status(self):
        self.assertIn("off", ht._hand_mouse_status(""))
        self.assertIn("already off", ht._hand_mouse_off(""))
        ht._hand_mouse_on("")
        self.assertIn("Hold a pinch still", ht._hand_mouse_status(""))
        out = ht._hand_mouse_off("")
        self.assertIn("Hand mouse off, sir", out)
        self.assertIn("still on", out)                  # the tracker stays
        self.assertEqual(self.la.stopped, 0)
        self.assertFalse(ht._mouse_enabled())
        self.assertIn("off", ht._hand_mouse_status(""))

    def test_status_while_engaged_and_dragging(self):
        ht._hand_mouse_on("")
        m, t, res = self.feed_pinch_and_engage()
        self.assertEqual(res, "engaged")
        self.assertIn("Let go of the pinch", ht._hand_mouse_status(""))
        for i in range(1, 6):
            m.on_hands(msg(hand(state="unknown")), t + 0.1 * i)
        self.assertIn("has the cursor", ht._hand_mouse_status(""))

    def test_the_hand_mouse_owns_the_trackers_lifetime_when_asked(self):
        ht._holo["owned"] = True
        ht._hand_mouse_on("")
        self.assertFalse(ht._holo["owned"])            # the hologram will not stop it

    def test_hand_tracking_off_stops_the_hand_mouse(self):
        ht._hand_mouse_on("")
        m, t, _ = self.feed_pinch_and_engage()
        for i in range(1, 4):
            m.on_hands(msg(hand(state="unknown")), t + 0.1 * i)
        for i in range(1, 10):
            m.on_hands(msg(hand(state="pinch")), t + 0.5 + 0.1 * i)
        self.assertTrue(m.button_down())
        out = ht._hand_tracking_off("")
        self.assertIn("hand mouse is off too", out)
        self.assertFalse(m.engaged())
        self.assertFalse(m.button_down())
        self.assertEqual(self.fake_out.buttons[-1], "up")
        self.assertFalse(config.HAND_MOUSE_ENABLED)
        self.assertFalse(ht._mouse_enabled())
        self.assertEqual(self.la.stopped, 1)

    def test_hand_tracking_off_with_only_the_mouse_on_still_reports(self):
        ht._hand_mouse_on("")
        self.la._running = False
        self.la.wanted = lambda: False
        self.assertIn("hand mouse is off", ht._hand_tracking_off(""))

    def test_the_listener_stopping_lets_go_of_the_mouse(self):
        ht._hand_mouse_on("")
        m, _, _ = self.feed_pinch_and_engage()
        self.assertTrue(m.engaged())
        ht._stop_listener()
        self.assertFalse(m.engaged())


class SkillBlockTests(_SkillBase):
    def _try(self, **patches):
        ht._hand_mouse_on("")
        with mock.patch.multiple(ht, **patches):
            return self.feed_pinch_and_engage()

    def test_game_mode_blocks_engaging(self):
        m, _, res = self._try(_game_mode_active=lambda: True)
        self.assertEqual(res, "ignored: game mode is on")
        self.assertFalse(m.engaged())

    def test_a_fullscreen_app_blocks_engaging(self):
        m, _, res = self._try(_fullscreen_app=lambda: True)
        self.assertEqual(res, "ignored: a fullscreen app is in front")
        self.assertFalse(m.engaged())

    def test_a_call_blocks_engaging(self):
        m, _, res = self._try(_in_call=lambda: True)
        self.assertEqual(res, "ignored: you are in a call")
        self.assertFalse(m.engaged())

    def test_a_blocker_coming_on_disengages(self):
        ht._hand_mouse_on("")
        m, t, res = self.feed_pinch_and_engage()
        self.assertEqual(res, "engaged")
        with mock.patch.object(ht, "_in_call", lambda: True):
            for i in range(1, 12):
                m.on_hands(msg(hand(state="unknown")), t + 0.1 * i)
        self.assertFalse(m.engaged())
        self.assertIn("call", m.last_reason)

    def test_handle_gesture_forwards_pinch_hold_only_while_enabled(self):
        # off: nothing is even constructed
        ev = {"type": "gesture", "name": "pinch_hold", "side": "Right"}
        self.assertEqual(ht.handle_gesture(ev, 1.0), "ignored: not a palm hold")
        self.assertEqual(self.built, [])
        # on: the event reaches the state machine
        ht._hand_mouse_on("")
        m = ht._mouse[0]
        for i in range(10):
            m.on_hands(msg(hand(state="pinch")), 10.0 + i * 0.07)
        ht.handle_gesture(ev, 10.7)
        self.assertTrue(m.engaged())
        # and the palm-hold menu trigger is unaffected
        self.assertEqual(ht.handle_gesture(
            {"type": "gesture", "name": "palm_hold"}, 11.0),
            "ignored: hand menu is off")


    def test_an_app_on_the_air_mouse_disable_list_blocks_engaging(self):
        m, _, res = self._try(_per_app_blocked=lambda: True)
        self.assertEqual(res, "ignored: an app on the air-mouse disable list is in front")
        self.assertFalse(m.engaged())

    def test_the_palm_menu_does_not_use_the_per_app_list(self):
        with mock.patch.object(ht, "_per_app_blocked", lambda: True):
            self.assertEqual(ht.blocked_reason(), "")
            self.assertIn("disable list", ht.mouse_blocked_reason())

    def test_per_app_reads_the_air_mouses_own_check(self):
        fake = types.SimpleNamespace(_per_app_disabled=lambda: True)
        with mock.patch.object(ht, "_air_mouse_mod", return_value=fake):
            self.assertTrue(_REAL_PER_APP_BLOCKED())
        with mock.patch.object(ht, "_air_mouse_mod", return_value=None):
            self.assertFalse(_REAL_PER_APP_BLOCKED())


class FakeSink:
    def __init__(self):
        self.calls = []

    def show(self, ret):
        self.calls.append(("show", ret["state"], ret["x"], ret["y"]))

    def clear(self):
        self.calls.append(("clear",))

    def shutdown(self):
        self.calls.append(("shutdown",))


class ReticleSkillTests(_SkillBase):
    """TASK 2: the webcam hand mouse shows the Kinect air mouse's reticle."""

    def setUp(self):
        super().setUp()
        self.sink = FakeSink()
        ht.set_reticle_sink(self.sink)
        self.addCleanup(ht.set_reticle_sink, None)

    def test_no_real_overlay_with_a_fake_mouse_or_in_staging(self):
        ht.set_reticle_sink(None)
        self.assertIsNone(ht._get_reticle_sink())             # fake output = a test
        with mock.patch.object(ht, "_mouse_output_factory", [None]):
            self.assertIsInstance(ht._get_reticle_sink(), ht._OverlaySink)
            with mock.patch.object(ht, "_staging", lambda: True):
                self.assertIsNone(ht._get_reticle_sink())
            with mock.patch.object(config, "HAND_MOUSE_RETICLE", False, create=True):
                self.assertIsNone(ht._get_reticle_sink())

    def test_engage_follow_grab_and_hide(self):
        ht._hand_mouse_on("")
        m = ht._mouse[0]
        t = 5000.0
        for i in range(10):
            m.on_hands(msg(hand(state="pinch")), t + i * 0.07)
            ht._publish_reticle(t + i * 0.07)
        self.assertEqual(self.sink.calls, [])                 # idle: nothing drawn
        m.on_gesture({"name": "pinch_hold", "side": "Right"}, t + 0.7)
        ht._publish_reticle(t + 0.7)
        self.assertEqual(self.sink.calls[-1][:2], ("show", "prime"))
        for i in range(1, 8):
            m.on_hands(msg(hand(state="unknown", pt=(0.5 + 0.02 * i, 0.45))),
                       t + 0.7 + 0.07 * i)
            ht._publish_reticle(t + 0.7 + 0.07 * i)
        last = self.sink.calls[-1]
        self.assertEqual(last[:2], ("show", "track"))
        self.assertEqual((last[2], last[3]), self.fake_out.pos)   # at the cursor
        n = len(self.sink.calls)
        ht._publish_reticle(t + 1.25)                         # unchanged, < keep-alive
        self.assertEqual(len(self.sink.calls), n)
        ht._publish_reticle(t + 1.6)                          # keep-alive rewrite
        self.assertEqual(len(self.sink.calls), n + 1)
        m.disengage("test")
        ht._publish_reticle(t + 1.7)
        ht._publish_reticle(t + 1.8)
        self.assertEqual(self.sink.calls[-1], ("clear",))
        self.assertEqual(self.sink.calls.count(("clear",)), 1)   # hidden once

    def test_hand_mouse_off_shuts_the_overlay(self):
        ht._hand_mouse_on("")
        ht._hand_mouse_off("")
        self.assertEqual(self.sink.calls[-1], ("shutdown",))

    def test_overlay_sink_reuses_the_air_mouse_helpers(self):
        am = mock.Mock()
        am.AirMouseDecision = lambda *a: ("D",) + a
        am._overlay_alive.return_value = False
        with mock.patch.object(ht, "_air_mouse_mod", return_value=am):
            ht._OverlaySink().show({"state": "track", "x": 10, "y": 20})
            d, visible = am._publish_overlay_state.call_args.args
            self.assertEqual((d[1], d[4], visible), ((10, 20), "track", True))
            am._spawn_overlay.assert_called_once()
            ht._OverlaySink().show({"state": "prime", "x": 10, "y": 20})
            self.assertEqual(am._publish_overlay_state.call_args.kwargs["prime_xy"],
                             (10, 20))
            with mock.patch.object(config, "KINECT_AIR_MOUSE_ENABLED", True):
                ht._OverlaySink().shutdown()
                am._shutdown_overlay.assert_not_called()   # the Kinect uses it
            ht._OverlaySink().shutdown()
            am._shutdown_overlay.assert_called_once()

    def test_quiet_while_two_hand_owns_the_overlay(self):
        ht._hand_mouse_on("")
        m, t, _ = self.feed_pinch_and_engage()
        with mock.patch.object(ht, "_two_hand_active", lambda: True):
            ht._publish_reticle(t)
        self.assertEqual(self.sink.calls, [])


class ListenerTests(_SkillBase):
    HANDS = json.dumps(msg(hand()), separators=(",", ":"))

    def test_hands_are_dropped_unread_when_the_mouse_is_off(self):
        got = []
        ls = ht.GestureListener("ws://x", on_hands=got.append, wants_hands=lambda: False)
        with mock.patch.object(ht.json, "loads", side_effect=AssertionError("parsed")):
            ls.feed(self.HANDS)
        self.assertEqual(got, [])

    def test_hands_are_dropped_with_no_hook_at_all(self):
        ls = ht.GestureListener("ws://x")
        with mock.patch.object(ht.json, "loads", side_effect=AssertionError("parsed")):
            ls.feed(self.HANDS)

    def test_hands_reach_the_mouse_only_while_it_is_on(self):
        got = []
        on = [False]
        ls = ht.GestureListener("ws://x", on_hands=got.append, wants_hands=lambda: on[0])
        ls.feed(self.HANDS)
        self.assertEqual(got, [])
        on[0] = True
        ls.feed(self.HANDS)
        ls.feed('{"type":"hands", broken')
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["type"], "hands")

    def test_gestures_still_flow(self):
        ev = []
        ls = ht.GestureListener("ws://x", on_event=ev.append, on_hands=lambda d: None,
                                wants_hands=lambda: True)
        ls.feed('{"type":"gesture","name":"pinch_hold","side":"Right","pos":[0.5,0.5]}')
        self.assertEqual(ev[0]["name"], "pinch_hold")

    def test_idle_and_disconnect_hooks(self):
        calls = []

        class WS:
            def __init__(self):
                self.n = 0

            def recv(self, timeout=None):
                self.n += 1
                if self.n == 1:
                    raise TimeoutError()
                ls.stop()
                return "x"
        ls = ht.GestureListener("ws://x", on_idle=lambda: calls.append("idle"))
        ls.pump(WS())
        self.assertEqual(calls, ["idle"])

    def test_the_real_listener_is_wired_to_the_mouse(self):
        src = _read("skills", "hand_tracking.py")
        self.assertIn("wants_hands=_wants_hands", src)
        # hands are parsed for the hand mouse or the two-hand grab, nothing else
        with mock.patch.object(ht, "_mouse_enabled", return_value=False), \
             mock.patch.object(ht, "_two_hand_enabled", return_value=False):
            self.assertFalse(ht._wants_hands())
        with mock.patch.object(ht, "_mouse_enabled", return_value=False), \
             mock.patch.object(ht, "_two_hand_enabled", return_value=True):
            self.assertTrue(ht._wants_hands())


# ── AIR_MOUSE_ENGINE routing in the Kinect skill ───────────────────────────
class EngineRoutingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("kinect_air_mouse", register=False)
        self.calls = []
        self.kinect = []
        reg = {n: (lambda a, n=n: self.calls.append(n) or f"hand:{n}")
               for n in ("hand_mouse_on", "hand_mouse_off", "hand_mouse_status")}
        fake = types.SimpleNamespace(ACTIONS=reg)
        p = mock.patch.dict(sys.modules, {"bobert_companion": fake})
        p.start()
        self.addCleanup(p.stop)
        for name, val in (
                ("_set_enabled", lambda on: self.kinect.append(("set_enabled", on)) or True),
                ("_mouse_button", lambda *a, **k: self.kinect.append(("button", a)) or True),
                ("_sensor_ready", lambda: (True, "")),
                ("_shutdown_overlay", lambda *a, **k: None),
                ("_clear_overlay_state", lambda *a, **k: None),
                ("_hand_in_view", lambda: True)):
            q = mock.patch.object(self.mod, name, val)
            q.start()
            self.addCleanup(q.stop)
        q = mock.patch.object(config, "KINECT_AIR_MOUSE_ENABLED", False, create=True)
        q.start()
        self.addCleanup(q.stop)

    def test_default_engine_leaves_the_kinect_alone_and_never_routes(self):
        with mock.patch.object(config, "AIR_MOUSE_ENGINE", "kinect"):
            out = self.mod.air_mouse_on("")
            self.assertIn("Air-mouse on", out)
            self.assertIn(("set_enabled", True), self.kinect)
            self.mod.air_mouse_status("")
            self.mod.air_mouse_off("")
        self.assertEqual(self.calls, [])

    def test_hands_engine_routes_on_off_status(self):
        with mock.patch.object(config, "AIR_MOUSE_ENGINE", "hands"):
            self.assertEqual(self.mod.air_mouse_on(""), "hand:hand_mouse_on")
            self.assertEqual(self.mod.air_mouse_off(""), "hand:hand_mouse_off")
            self.assertEqual(self.mod.air_mouse_status(""), "hand:hand_mouse_status")
            self.assertEqual(self.mod.air_mouse_arm_action(""), "hand:hand_mouse_on")
            self.assertEqual(self.mod.air_mouse_disarm_action(""), "hand:hand_mouse_off")
        self.assertEqual(self.kinect, [])               # the Kinect was never touched

    def test_hands_engine_with_no_hand_mouse_loaded_says_so_and_does_not_fall_back(self):
        with mock.patch.object(config, "AIR_MOUSE_ENGINE", "hands"), \
                mock.patch.dict(sys.modules, {"bobert_companion": types.SimpleNamespace(
                    ACTIONS={})}):
            out = self.mod.air_mouse_on("")
        self.assertIn("isn't loaded", out)
        self.assertEqual(self.kinect, [])

    def test_a_bad_engine_value_is_the_kinect(self):
        with mock.patch.object(config, "AIR_MOUSE_ENGINE", "banana"):
            self.mod.air_mouse_on("")
        self.assertEqual(self.calls, [])
        self.assertIn(("set_enabled", True), self.kinect)

    def test_one_registration_per_action_name(self):
        # the Kinect skill used to alias hand_mouse_on/off to arm/disarm: gone now
        _m, kinect_actions = load_skill_isolated("kinect_air_mouse", register=False)
        self.assertEqual(kinect_actions, {})
        self.assertNotIn('"hand_mouse_on"', _read("skills", "kinect_air_mouse.py").split(
            "def register(actions):")[1])
        acts = {}
        ht.register(acts)
        self.assertEqual(sorted(a for a in acts if a.startswith("hand_mouse")),
                         ["hand_mouse_off", "hand_mouse_on", "hand_mouse_status"])


# ── safety: the real OS layer is isolated ──────────────────────────────────
# ── recognition: no false triggers (2026-10-20) ────────────────────────────
class RecognitionTests(unittest.TestCase):
    """Owner, testing in a classroom: "I don't want false triggers". Typing,
    talking with his hands, a classmate's hand in frame must never take the
    cursor, click, drag or scroll."""

    def test_a_small_far_hand_never_engages(self):
        r = Rig()
        r.run(1.0, state="pinch", size=0.06)            # a classmate 2 m back
        self.assertTrue(r.hold_event().startswith("ignored: hand too small"))
        self.assertEqual(r.m.phase, hm.IDLE)

    def test_a_low_confidence_hand_never_engages(self):
        r = Rig()
        r.run(1.0, state="pinch", score=0.7)
        self.assertTrue(r.hold_event().startswith("ignored: hand too small"))

    def test_size_and_score_have_hysteresis(self):
        r = Rig()
        r.engage()
        r.run(0.5, state="unknown", size=0.085, score=0.65)   # under ON, over OFF
        self.assertEqual(r.m.phase, hm.ENGAGED)
        moves = len(r.out.moves)
        r.run(0.3, state="unknown", pt=(0.6, 0.5), size=0.085, score=0.65)
        self.assertGreater(len(r.out.moves), moves)            # still driving
        r.run(1.2, state="unknown", pt=(0.4, 0.5), size=0.05)  # under OFF
        self.assertEqual(r.m.phase, hm.IDLE)                   # = gone
        # ... and coming back at the in-between size is not enough to engage
        r2 = Rig()
        r2.run(1.0, state="pinch", size=0.085)
        self.assertTrue(r2.hold_event().startswith("ignored"))

    def test_a_hand_that_just_arrived_is_ignored(self):
        r = Rig()
        r.run(0.2, state="pinch")                      # in view for 0.2 s only
        self.assertTrue(r.hold_event().startswith("ignored: hand too small, unsure or just arrived"))
        r.run(0.5, state="pinch")
        self.assertEqual(r.hold_event(), "engaged")

    def test_a_flickered_pinch_frame_does_not_engage(self):
        r = Rig()
        r.run(1.0, state="unknown")
        r.frame(state="pinch")                         # one stray pinch frame
        self.assertTrue(r.hold_event().startswith("ignored"))

    def test_engage_needs_the_pinch_confirmed_here_too(self):
        r = Rig()
        r.run(1.0, state="unknown")
        r.run(0.2, state="pinch")                      # confirmed only ~0.07 s ago
        self.assertEqual(r.hold_event(), "ignored: pinch not held long enough")

    def test_typing_or_mouse_use_blocks_engaging(self):
        r = Rig()
        r.out.idle = 0.3                               # a key 0.3 s ago
        r.run(0.7, state="pinch")
        self.assertEqual(r.hold_event(), "ignored: you are using the keyboard or mouse")
        r.out.idle = 2.0
        self.assertEqual(r.hold_event(), "engaged")

    def test_moving_the_real_mouse_suppresses_re_engaging(self):
        r = Rig()
        r.engage()
        r.out.override = (50, 50)
        r.run(0.3, state="unknown", pt=(0.6, 0.5))
        self.assertIn("moved the mouse", r.m.last_reason)
        r.out.override = None
        r.run(0.7, state="pinch", pt=(0.6, 0.5))
        self.assertEqual(r.hold_event(), "ignored: you are using the keyboard or mouse")
        r.run(0.6, state="pinch", pt=(0.6, 0.5))       # > input_quiet_s later
        self.assertEqual(r.hold_event(), "engaged")

    def test_a_one_frame_pinch_is_no_click(self):
        r = Rig()
        r.engage()
        r.run(0.4, state="unknown")
        r.frame(state="pinch")
        r.run(0.5, state="unknown")
        self.assertEqual(r.out.buttons, [])
        self.assertEqual(r.m.clicks, 0)

    def test_debounce_scales_with_the_frame_rate(self):
        for fps, frames_needed in ((15, 3), (30, 4)):
            c = hm._Confirm()
            t = 0.0
            c.update("unknown", t, 0.1, 2)
            n = 0
            while c.stable != "pinch":
                t += 1.0 / fps
                n += 1
                c.update("pinch", t, 0.1, 2)
            self.assertEqual(n, frames_needed, fps)

    def test_another_hand_never_takes_over(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        n = len(r.out.moves)
        # the owner's hand vanishes; a NEW right hand (new track id) appears
        # across the frame: it is not the owner's, so nothing follows it
        r.run(0.6, state="unknown", pt=(0.25, 0.7), hid=1)
        self.assertEqual(len(r.out.moves), n)
        r.run(0.6, state="unknown", pt=(0.25, 0.7), hid=1)
        self.assertEqual(r.m.phase, hm.IDLE)            # gone > hand_gone_s
        self.assertIn("left the view", r.m.last_reason)

    def test_the_owners_hand_coming_back_where_it_left_is_adopted(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        for _ in range(6):                              # 0.4 s dropout
            r.empty()
        r.run(0.5, state="unknown", pt=(0.52, 0.46), hid=1)
        self.assertEqual(r.m.phase, hm.ENGAGED)
        n = len(r.out.moves)
        r.run(0.4, state="unknown", pt=(0.65, 0.5), hid=1)
        self.assertGreater(len(r.out.moves), n)

    def test_params_from_config_recognition_knobs(self):
        p = hm.Params.from_config(lambda n, d: {
            "HAND_MOUSE_MIN_SCORE": 0.9, "HAND_MOUSE_MIN_HAND_SIZE": 0.08,
            "HAND_MOUSE_SETTLE_S": 0.6, "HAND_MOUSE_INPUT_QUIET_S": 2.0,
            "HAND_MOUSE_SCROLL_ENABLED": False}.get(n, d))
        self.assertEqual((p.min_score_on, p.min_size_on, p.settle_s,
                          p.input_quiet_s, p.scroll_enabled),
                         (0.9, 0.08, 0.6, 2.0, False))
        self.assertLess(p.min_size_off, p.min_size_on)  # exit stays looser
        self.assertLess(p.min_score_off, p.min_score_on)
        d = hm.Params()
        self.assertLess(d.min_score_off, d.min_score_on)
        self.assertLess(d.min_size_off, d.min_size_on)


# ── scroll (2026-10-20) ────────────────────────────────────────────────────
class ScrollTests(unittest.TestCase):
    def _fist_at(self, r, secs, y, **kw):
        r.run(secs, state="fist", palm=(0.5, y), pt=(0.5, y), **kw)

    def wheels(self, r):
        return [e[1] for e in r.out.events if e[0] == "wheel"]

    def test_fist_moved_down_scrolls_up_like_grabbing_the_page(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        self._fist_at(r, 0.5, 0.45)                     # close + hold: nothing yet
        self.assertEqual(self.wheels(r), [])
        self.assertEqual(r.m.sub, hm.SCROLL)
        n = len(r.out.moves)
        for i in range(1, 16):                          # move the fist down 0.6 hs
            self._fist_at(r, 1 / 15, 0.45 + 0.0064 * i)
        w = self.wheels(r)
        self.assertTrue(w)
        self.assertTrue(all(d > 0 for d in w))          # hand down = wheel up (natural)
        want = (0.6 - hm.Params().scroll_deadzone_hs) * hm.Params().scroll_gain
        self.assertAlmostEqual(sum(w), want, delta=0.25 * want)
        self.assertEqual(len(r.out.moves), n)           # the cursor stayed put
        self.assertEqual(r.out.buttons, [])             # never a click

    def test_up_scrolls_down_and_the_rate_is_capped(self):
        r = Rig()
        r.engage()
        self._fist_at(r, 0.5, 0.45)
        self._fist_at(r, 0.4, 0.15)                     # a big fast jump up
        w = self.wheels(r)
        self.assertTrue(all(d < 0 for d in w))
        cap = hm.Params().scroll_max_rate
        self.assertLessEqual(-sum(w), cap * 0.4 + 50)   # <= max rate x time

    def test_dead_zone_and_opening_the_hand_stops(self):
        r = Rig()
        r.engage()
        self._fist_at(r, 0.5, 0.45)
        self._fist_at(r, 0.5, 0.47)                     # 0.12 hs: inside the dead zone
        self.assertEqual(self.wheels(r), [])
        r.run(0.3, state="unknown", pt=(0.5, 0.47))
        self.assertEqual(r.m.sub, hm.FREE)
        self._fist_at(r, 0.1, 0.8)                      # a fist flash: no scroll
        self.assertEqual(self.wheels(r), [])

    def test_a_quick_fist_is_not_a_scroll(self):
        r = Rig()
        r.engage()
        self._fist_at(r, 0.2, 0.45)                     # confirmed, but < hold
        self.assertNotEqual(r.m.sub, hm.SCROLL)
        r.run(0.3, state="unknown")
        self.assertEqual(r.m.scrolls, 0)

    def test_never_while_idle_or_with_two_hands(self):
        r = Rig()
        self._fist_at(r, 1.0, 0.45)
        self._fist_at(r, 0.5, 0.75)
        self.assertEqual(self.wheels(r), [])            # not engaged: nothing
        r.engage()
        for _ in range(int(0.8 * r.fps)):               # a second usable hand
            r.t += 1.0 / r.fps
            r.m.on_hands(msg(hand(state="fist", palm=(0.5, 0.45)),
                             hand(side="Left", state="fist", pt=(0.3, 0.45))), r.t)
        self.assertEqual(r.m.scrolls, 0)

    def test_scroll_can_be_turned_off(self):
        r = Rig(params=hm.Params(scroll_enabled=False))
        r.engage()
        self._fist_at(r, 0.5, 0.45)
        self._fist_at(r, 0.5, 0.75)
        self.assertEqual(self.wheels(r), [])

    def test_after_scrolling_the_cursor_does_not_jump(self):
        r = Rig()
        r.engage()
        r.run(0.3, state="unknown", pt=(0.5, 0.45))
        pos = r.out.pos
        self._fist_at(r, 0.5, 0.45)
        self._fist_at(r, 0.5, 0.65)
        r.run(0.4, state="unknown", pt=(0.5, 0.65))
        self.assertLess(abs(r.out.pos[1] - pos[1]), 40)


# ── reticle state and the two-hand stand-down ──────────────────────────────
class ReticleTests(unittest.TestCase):
    def test_states_follow_the_phase(self):
        r = Rig()
        self.assertEqual(r.m.reticle()["state"], "hidden")
        self.assertFalse(r.m.reticle()["visible"])
        r.run(0.65, state="pinch")
        r.hold_event()
        ret = r.m.reticle()
        self.assertEqual((ret["state"], ret["visible"]), ("prime", True))
        self.assertEqual((ret["x"], ret["y"]), r.out.pos)    # where the cursor is
        r.run(0.3, state="pinch")
        r.run(0.3, state="unknown", pt=(0.6, 0.5))
        ret = r.m.reticle()
        self.assertEqual(ret["state"], "track")
        self.assertEqual((ret["x"], ret["y"]), r.out.pos)
        r.run(0.3, state="pinch", pt=(0.6, 0.5))
        self.assertEqual(r.m.reticle()["state"], "grab")
        r.m.disengage("x")
        self.assertEqual(r.m.reticle()["state"], "hidden")


class TwoHandStandDownTests(unittest.TestCase):
    def test_two_hand_mode_disengages_and_blocks_engaging(self):
        on = [False]
        r = Rig()
        r.m.two_hand = lambda: on[0]
        r.engage()
        r.run(0.2, state="unknown")
        r.run(0.7, state="pinch")                       # dragging
        self.assertTrue(r.m.button_down())
        on[0] = True
        r.frame(state="pinch")
        self.assertEqual(r.m.phase, hm.IDLE)
        self.assertFalse(r.m.button_down())
        self.assertEqual(r.out.buttons, ["down", "up"])
        self.assertIn("two-hand", r.m.last_reason)
        r.run(0.7, state="pinch")
        self.assertEqual(r.hold_event(), "ignored: two-hand mode")


class SafetyTests(unittest.TestCase):
    def test_importing_the_module_touches_no_windows_api(self):
        src = _read("core", "hand_mouse.py")
        head = src.split("class WindowsMouse")[0]
        self.assertNotIn("windll", head)
        self.assertNotIn("import ctypes", head)

    def test_windows_layer_only_in_the_one_class(self):
        for rel in ("skills/hand_tracking.py", "tools/handtrack_gestures.py"):
            text = _read(*rel.split("/"))
            for bad in ("SetCursorPos", "SendInput", "mouse_event", "pyautogui"):
                self.assertNotIn(bad, text, rel)

    def test_fake_output_satisfies_the_interface(self):
        for name in ("move", "button", "position", "screen_rect"):
            self.assertTrue(callable(getattr(FakeMouse(), name)))
            self.assertTrue(callable(getattr(hm.WindowsMouse, name)))

    def test_params_from_config(self):
        p = hm.Params.from_config(lambda n, d: {
            "HAND_MOUSE_BOX": (0.1, 0.1, 0.9, 0.9),
            "HAND_MOUSE_SCREEN_RECT": (-1920, 0, 1920, 1080),
            "HAND_MOUSE_PALM_RELEASE_S": 0.0}.get(n, d))
        self.assertEqual(p.box, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(p.screen_rect, (-1920, 0, 1920, 1080))
        self.assertEqual(p.palm_release_s, 0.0)
        bad = hm.Params.from_config(lambda n, d: {"HAND_MOUSE_BOX": (1, 1, 1, 1),
                                                  "HAND_MOUSE_SCREEN_RECT": "x"}.get(n, d))
        self.assertEqual(bad.box, hm.Params().box)
        self.assertIsNone(bad.screen_rect)

    def test_garbage_input_never_raises(self):
        r = Rig()
        r.engage()
        for bad in ({}, {"hands": None}, {"hands": [None, 5, {}]},
                    {"hands": [{"side": "Right", "lm": []}]},
                    {"hands": [{"side": "Right", "state": "pinch", "lm": [[0, 0]]}]}):
            r.m.on_hands(bad, r.t + 0.1)
        r.m.on_gesture({}, r.t)
        r.m.on_gesture({"name": "pinch_hold"}, r.t)


if __name__ == "__main__":
    unittest.main()
