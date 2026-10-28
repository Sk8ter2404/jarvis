"""core/hand_two_hand.py: the WEBCAM two-hand window grab (2026-10-20) - two
fists grab the window between them, move it, spread / close to resize, open
either hand to let go - plus its wiring in skills/hand_tracking.py (the hand
mouse stands down while it is active).

FAKES ONLY: the window layer is a FakeWin; nothing here can move a real window
or touch the overlay. The state machine is the REAL Kinect TwoHandController
(skills/kinect_two_hand.py), on a simulated clock.
"""
import importlib
import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import config  # noqa: E402
from core import hand_two_hand as h2  # noqa: E402
from tests.test_hand_mouse import FakeLauncher, FakeMouse, ht, hand, msg  # noqa: E402

kt = importlib.import_module("skills.kinect_two_hand")


class FakeWin:
    def __init__(self, clock, rect=None, hwnd=42):
        self.clock = clock
        self.rect = rect or kt.Rect(400, 300, 1200, 900)
        self.hwnd = hwnd
        self.lookups, self.moves, self.published = [], [], []

    def controller(self):
        return kt.TwoHandController(clock=self.clock)

    def window_at(self, x, y):
        self.lookups.append((x, y))
        return self.hwnd, self.rect

    def set_window_pos(self, hwnd, rect):
        self.moves.append((hwnd, rect))
        return True

    def screen_rect(self):
        return (0, 0, 1920, 1080)

    def bounds(self):
        return (0, 0, 1920, 1080)

    def publish(self, decision, edge):
        self.published.append(None if decision is None
                              else (decision.active, decision.resizing))


class Rig:
    def __init__(self, fps=15.0, blocker=None):
        self.t = 100.0
        self.fps = fps
        self.win = FakeWin(lambda: self.t)
        self.g = h2.TwoHandGrab(self.win, blocker=blocker, log=lambda s: None)

    def frame(self, left="fist", right="fist", lx=0.40, rx=0.60, y=0.45, **kw):
        self.t += 1.0 / self.fps
        hands = []
        if left is not None:
            hands.append(hand(side="Left", state=left, pt=(lx, y), palm=(lx, y),
                              hid=1, **kw))
        if right is not None:
            hands.append(hand(side="Right", state=right, pt=(rx, y), palm=(rx, y),
                              hid=0, **kw))
        return self.g.on_hands(msg(*hands), self.t)

    def run(self, secs, **kw):
        d = None
        for _ in range(max(1, round(secs * self.fps))):
            d = self.frame(**kw)
        return d

    def grab(self):
        d = self.run(1.2)
        assert d.resizing, d.phase
        return d


class GrabTests(unittest.TestCase):
    def test_two_fists_held_grab_the_window_under_the_midpoint(self):
        r = Rig()
        r.run(0.6)                                     # settling + debounce
        self.assertFalse(r.win.moves)
        d = r.run(0.7)
        self.assertTrue(d.resizing)
        self.assertEqual(r.g.grabs, 1)
        x, y = r.win.lookups[-1]
        self.assertAlmostEqual(x, 959.5, delta=3)      # between the hands
        self.assertAlmostEqual(y, 0.5 * 1079, delta=3)
        self.assertTrue(all(h == 42 for h, _ in r.win.moves))

    def test_moving_both_hands_moves_the_window(self):
        r = Rig()
        r.grab()
        for i in range(1, 16):
            r.frame(lx=0.40 + 0.006 * i, rx=0.60 + 0.006 * i)
        rect = r.win.moves[-1][1]
        self.assertGreater(rect.left, 400 + 150)       # ~0.09 of the box -> ~290 px
        self.assertAlmostEqual(rect.width, 800, delta=40)   # same size

    def test_spreading_resizes(self):
        r = Rig()
        r.grab()
        for i in range(1, 16):
            r.frame(lx=0.40 - 0.006 * i, rx=0.60 + 0.006 * i)
        self.assertGreater(r.win.moves[-1][1].width, 900)
        r2 = Rig()
        r2.grab()
        for i in range(1, 16):
            r2.frame(lx=0.40 + 0.004 * i, rx=0.60 - 0.004 * i)
        self.assertLess(r2.win.moves[-1][1].width, 700)

    def test_opening_either_hand_lets_go(self):
        for which in ("left", "right"):
            r = Rig()
            r.grab()
            kw = {which: "open"}
            r.run(0.6, **kw)
            self.assertFalse(r.g.ctrl.is_grabbed, which)
            n = len(r.win.moves)
            r.run(0.5, lx=0.2, rx=0.4, **kw)
            self.assertEqual(len(r.win.moves), n, which)

    def test_losing_a_hand_lets_go_at_once(self):
        r = Rig()
        r.grab()
        r.frame(left=None)
        self.assertFalse(r.g.ctrl.is_grabbed)
        self.assertFalse(r.g.active(r.t))


class FalseTriggerTests(unittest.TestCase):
    def test_one_fist_and_one_open_hand_never_grabs(self):
        r = Rig()
        r.run(3.0, right="open")
        self.assertEqual((r.win.lookups, r.win.moves, r.g.grabs), ([], [], 0))
        self.assertFalse(r.g.active(r.t))

    def test_pinches_are_not_fists(self):
        r = Rig()
        r.run(3.0, left="pinch", right="pinch")
        self.assertEqual(r.g.grabs, 0)

    def test_small_far_hands_never_grab(self):
        r = Rig()
        r.run(3.0, size=0.06)
        self.assertEqual(r.g.grabs, 0)
        self.assertFalse(r.g.active(r.t))

    def test_hands_that_just_arrived_must_settle(self):
        r = Rig()
        d = r.run(0.3)
        self.assertEqual(d.phase, "idle")
        self.assertFalse(r.g.active(r.t))

    def test_a_blocker_stops_the_grab_but_keeps_the_cursor_down(self):
        r = Rig(blocker=lambda: "game mode is on")
        r.run(2.0)
        self.assertEqual(r.g.grabs, 0)
        self.assertTrue(r.g.active(r.t))               # the hand mouse still stands down

    def test_a_flickered_fist_frame_is_nothing(self):
        r = Rig()
        r.run(1.0, left="open", right="open")
        r.frame()
        r.run(1.0, left="open", right="open")
        self.assertEqual(r.g.grabs, 0)


class StateTests(unittest.TestCase):
    def test_active_ages_out(self):
        r = Rig()
        r.run(0.8)
        self.assertTrue(r.g.active(r.t))
        self.assertFalse(r.g.active(r.t + h2.ACTIVE_TTL_S + 0.1))

    def test_stop_lets_go_and_clears_the_reticles(self):
        r = Rig()
        r.grab()
        r.g.stop()
        self.assertFalse(r.g.ctrl.is_grabbed)
        self.assertFalse(r.g.active(r.t))
        self.assertEqual(r.win.published[-1], None)

    def test_dual_reticles_are_published_with_our_own_edge_flag(self):
        r = Rig()
        r.grab()
        self.assertIn((True, True), r.win.published)
        self.assertIs(r.g._edge, r.g._edge)
        self.assertIsNot(r.g._edge, kt._two_hand_overlay_was_active)

    def test_garbage_never_raises(self):
        r = Rig()
        for bad in ({}, {"hands": None}, {"hands": [None, 3, {}]},
                    {"hands": [{"side": "Left", "lm": []}, {"side": "Right"}]}):
            r.g.on_hands(bad, r.t)

    def test_windows_backend_reuses_the_kinect_helpers(self):
        fake = mock.Mock()
        b = h2.WindowsBackend(kt=fake)
        b.window_at(1, 2)
        fake._target_rect_at.assert_called_once_with(1, 2)
        b.set_window_pos(5, "R")
        fake._set_window_pos.assert_called_once_with(5, "R")
        d = mock.Mock(active=True)
        b.publish(d, [False])
        fake._publish_two_hand_overlay.assert_called_once()
        fake._ensure_overlay_alive.assert_called_once()
        b.publish(None, [True])
        fake.TwoHandDecision.assert_called_once_with(False, None, False, "idle")


class SkillWiringTests(unittest.TestCase):
    """skills/hand_tracking.py: the two-hand grab runs only with the setting on
    AND AIR_MOUSE_ENGINE = "hands"; while active, the hand mouse stands down."""

    def setUp(self):
        self.t = 5000.0
        self.out = FakeMouse()
        ht.set_mouse_output_factory(lambda: self.out)
        self.addCleanup(ht.set_mouse_output_factory, None)
        self.win = FakeWin(lambda: self.t)
        ht.set_two_hand_backend_factory(lambda: self.win)
        self.addCleanup(ht.set_two_hand_backend_factory, None)
        la = FakeLauncher()
        for target, name, val in ((ht, "_launcher", lambda: la),
                                  (ht, "_ensure_listener", lambda: None),
                                  (ht, "blocked_reason", lambda: ""),
                                  (ht, "_per_app_blocked", lambda: False),
                                  (ht, "_staging", lambda: False),
                                  (ht, "_get_reticle_sink", lambda: None)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        for key, val in (("HAND_TWO_HAND_ENABLED", True), ("AIR_MOUSE_ENGINE", "hands"),
                         ("HAND_MOUSE_CUE_SOUND", False), ("HAND_MOUSE_ENABLED", False)):
            p = mock.patch.object(config, key, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: ht._mouse.__setitem__(0, None))

    def feed(self, *hands_):
        self.t += 1 / 15.0
        with mock.patch.object(ht.time, "monotonic", lambda: self.t):
            ht._mouse_hands(msg(*hands_))

    def test_enabled_only_with_the_setting_and_the_hands_engine(self):
        self.assertTrue(ht._two_hand_enabled())
        with mock.patch.object(config, "AIR_MOUSE_ENGINE", "kinect"):
            self.assertFalse(ht._two_hand_enabled())
        with mock.patch.object(config, "HAND_TWO_HAND_ENABLED", False):
            self.assertFalse(ht._two_hand_enabled())
        with mock.patch.object(ht, "_staging", lambda: True):
            self.assertFalse(ht._two_hand_enabled())
        self.assertIs(type(config.HAND_TWO_HAND_ENABLED), bool)

    def test_default_is_off(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cfg_defaults_probe", os.path.join(_ROOT, "core", "config.py"))
        src = open(spec.origin, encoding="utf-8").read()
        self.assertIn("HAND_TWO_HAND_ENABLED = False", src)

    def test_no_real_windows_with_a_fake_mouse(self):
        ht.set_two_hand_backend_factory(None)
        self.assertIsNone(ht._get_two_hand())

    def test_two_fists_grab_and_the_hand_mouse_stands_down(self):
        ht._hand_mouse_on("")
        m = ht._mouse[0]
        for _ in range(10):
            self.feed(hand(state="pinch"))
        self.assertEqual(m.on_gesture({"name": "pinch_hold", "side": "Right"},
                                      self.t), "engaged")
        for _ in range(20):
            self.feed(hand(side="Left", state="fist", pt=(0.4, 0.45), hid=1),
                      hand(side="Right", state="fist", pt=(0.6, 0.45)))
        self.assertFalse(m.engaged())
        self.assertIn("two-hand", m.last_reason)
        self.assertEqual(ht._two_hand[0].grabs, 1)
        self.assertTrue(self.win.moves)
        self.assertEqual(self.out.buttons, [])         # never a click on the way
        self.assertEqual([e for e in self.out.events if e[0] == "wheel"], [])

    def test_turning_it_off_lets_go(self):
        for _ in range(20):
            self.feed(hand(side="Left", state="fist", pt=(0.4, 0.45), hid=1),
                      hand(side="Right", state="fist", pt=(0.6, 0.45)))
        g = ht._two_hand[0]
        self.assertTrue(g.ctrl.is_grabbed)
        with mock.patch.object(config, "HAND_TWO_HAND_ENABLED", False):
            self.feed(hand(side="Left", state="fist", pt=(0.4, 0.45), hid=1),
                      hand(side="Right", state="fist", pt=(0.6, 0.45)))
        self.assertFalse(g.ctrl.is_grabbed)

    def test_tracker_gone_lets_go(self):
        for _ in range(20):
            self.feed(hand(side="Left", state="fist", pt=(0.4, 0.45), hid=1),
                      hand(side="Right", state="fist", pt=(0.6, 0.45)))
        ht._mouse_tracker_gone()
        self.assertFalse(ht._two_hand[0].ctrl.is_grabbed)


class KinectHelperTests(unittest.TestCase):
    def test_window_at_point_skips_the_overlay_and_the_shell(self):
        try:
            importlib.import_module("ctypes.wintypes")  # _target_rect_at needs it
        except ImportError:
            # ci-sim hides ctypes.wintypes (it models a Linux runner); the
            # helper degrades to (0, None) there, so there is nothing to pin.
            self.skipTest("ctypes.wintypes is not importable on this runner")
        with mock.patch.object(kt, "_window_title", return_value="JARVIS Air Cursor"):
            fake_user32 = mock.Mock()
            with mock.patch("ctypes.windll", create=True) as wd:
                wd.user32 = fake_user32
                fake_user32.WindowFromPoint.return_value = 7
                fake_user32.GetAncestor.return_value = 9
                self.assertEqual(kt._target_rect_at(10, 20), (9, None))
        self.assertFalse(kt.is_resizable_target("Shell_TrayWnd", ""))
        self.assertFalse(kt.is_resizable_target("Progman", "Program Manager"))

    def test_the_kinect_poller_keeps_its_own_edge_flag(self):
        import inspect
        sig = inspect.signature(kt._publish_two_hand_overlay)
        self.assertIn("edge", sig.parameters)
        self.assertIsNone(sig.parameters["edge"].default)


if __name__ == "__main__":
    unittest.main()
