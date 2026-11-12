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
        # The same dead-man as WindowsBackend.controller.
        return kt.TwoHandController(clock=self.clock,
                                    deadman_sec=h2.LOSS_GRACE_S)

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

    def test_one_missing_frame_keeps_the_grab_and_freezes_the_window(self):
        # 2026-11-12: the tracker drops a real hand for single frames; each
        # one used to end the grab on the spot.
        r = Rig()
        r.grab()
        r.frame(lx=0.42, rx=0.62)
        frozen = r.win.moves[-1][1]
        r.frame(left=None)
        self.assertTrue(r.g.ctrl.is_grabbed)
        self.assertEqual(r.win.moves[-1][1], frozen)
        r.frame(lx=0.45, rx=0.65)
        self.assertTrue(r.g.ctrl.is_grabbed)
        self.assertNotEqual(r.win.moves[-1][1], frozen, "it follows again")

    def test_a_hand_that_stays_lost_lets_go_after_the_grace(self):
        r = Rig()
        r.grab()
        r.run(h2.LOSS_GRACE_S - 0.2, left=None)
        self.assertTrue(r.g.ctrl.is_grabbed, "still inside the grace")
        r.run(0.5, left=None)
        self.assertFalse(r.g.ctrl.is_grabbed)
        self.assertFalse(r.g.active(r.t))

    def test_a_seen_open_hand_lets_go_well_inside_the_grace(self):
        for which in ("left", "right"):
            r = Rig()
            r.grab()
            r.run(0.3, **{which: "open"})   # the state confirm, not the grace
            self.assertFalse(r.g.ctrl.is_grabbed, which)

    def test_a_phantom_hand_cannot_stand_in_mid_grab(self):
        r = Rig()
        r.grab()
        r.frame(lx=0.42, rx=0.62)
        before = r.win.moves[-1][1]
        # The real left hand vanishes and a tiny "Left" shows up far away.
        r.t += 1.0 / r.fps
        phantom = hand(side="Left", state="fist", pt=(0.95, 0.9),
                       palm=(0.95, 0.9), hid=9, size=0.02)
        real = hand(side="Right", state="fist", pt=(0.62, 0.45),
                    palm=(0.62, 0.45), hid=0)
        r.g.on_hands(msg(phantom, real), r.t)
        self.assertTrue(r.g.ctrl.is_grabbed)
        self.assertEqual(r.win.moves[-1][1], before)


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


class WebcamMisreadTests(unittest.TestCase):
    """Live on the Dell 2026-11-06: the tracker's misreads dropped held grabs."""

    def feed(self, r, *hands):
        r.t += 1.0 / r.fps
        return r.g.on_hands(msg(*hands), now=r.t)

    def fists(self, lid=1, rid=2, lx=0.40, rx=0.60):
        return (hand(side="Left", state="fist", pt=(lx, 0.45), palm=(lx, 0.45), hid=lid),
                hand(side="Right", state="fist", pt=(rx, 0.45), palm=(rx, 0.45), hid=rid))

    def grabbed_rig(self):
        r = Rig()
        for _ in range(30):
            self.feed(r, *self.fists())
        self.assertEqual(r.g.grabs, 1)
        self.assertTrue(r.g.last.resizing)
        return r

    def test_a_phantom_hand_on_one_side_does_not_stop_the_grab(self):
        r = Rig()
        phantom = hand(side="Left", state="pinch", pt=(0.8, 0.2), palm=(0.8, 0.2),
                       score=1.0, size=0.03, hid=7)
        for _ in range(90):
            left, right = self.fists()
            self.feed(r, left, right, phantom)
        self.assertEqual(r.g.grabs, 1)
        self.assertTrue(r.g.last.resizing)

    def test_a_fist_read_as_a_pinch_keeps_the_grab(self):
        r = self.grabbed_rig()
        for _ in range(45):
            left, _right = self.fists()
            self.feed(r, left, hand(side="Right", state="pinch", pt=(0.6, 0.45),
                                    palm=(0.6, 0.45), hid=2))
        self.assertTrue(r.g.last.resizing)
        self.assertEqual(r.g.grabs, 1)

    def test_a_hand_on_its_way_to_open_lets_go_before_it_gets_there(self):
        for state in ("point", "unknown"):
            with self.subTest(state=state):
                r = self.grabbed_rig()
                for _ in range(15):      # 1 s, well past the 0.30 s dead-man
                    left, _right = self.fists()
                    self.feed(r, left, hand(side="Right", state=state, pt=(0.6, 0.45),
                                            palm=(0.6, 0.45), hid=2))
                self.assertFalse(r.g.last.resizing)

    def test_one_misread_frame_is_not_a_release(self):
        r = self.grabbed_rig()
        for i in range(45):
            left, _right = self.fists()
            state = "unknown" if i % 15 == 0 else "fist"
            self.feed(r, left, hand(side="Right", state=state, pt=(0.6, 0.45),
                                    palm=(0.6, 0.45), hid=2))
        self.assertTrue(r.g.last.resizing)
        self.assertEqual(r.g.grabs, 1)

    def test_an_open_hand_still_lets_go_after_a_misread(self):
        r = self.grabbed_rig()
        for state in ("pinch", "open"):
            for _ in range(15):
                left, _right = self.fists()
                self.feed(r, left, hand(side="Right", state=state, pt=(0.6, 0.45),
                                        palm=(0.6, 0.45), hid=2))
        self.assertFalse(r.g.last.resizing)

    def test_losing_a_hand_still_lets_go_after_a_misread(self):
        r = self.grabbed_rig()
        for _ in range(15):
            left, _right = self.fists()
            self.feed(r, left)
        self.assertFalse(r.g.last.resizing)

    def test_a_track_id_change_keeps_the_grab(self):
        r = self.grabbed_rig()
        for _ in range(45):
            self.feed(r, *self.fists(lid=2, rid=1))
        self.assertTrue(r.g.last.resizing)
        self.assertEqual(r.g.grabs, 1)

    def test_a_fist_read_as_a_pinch_still_cannot_start_a_grab(self):
        r = Rig()
        for _ in range(45):
            left, _right = self.fists()
            self.feed(r, left, hand(side="Right", state="pinch", pt=(0.6, 0.45),
                                    palm=(0.6, 0.45), hid=2))
        self.assertEqual(r.g.grabs, 0)

    def test_two_real_hands_with_one_label_are_placed_by_position(self):
        r = Rig()
        for _ in range(30):
            self.feed(r,
                      hand(side="Left", state="fist", pt=(0.7, 0.45), palm=(0.7, 0.45), hid=1),
                      hand(side="Left", state="fist", pt=(0.3, 0.45), palm=(0.3, 0.45), hid=2))
        self.assertEqual(r.g.grabs, 1)
        self.assertTrue(r.g.last.resizing)


class HandsBySideTests(unittest.TestCase):
    P = h2.hm.Params()

    def pick(self, *hands):
        return h2.hm.hands_by_side(list(hands), 640, 480, self.P)

    def test_a_real_hand_beats_a_higher_scoring_phantom(self):
        real = hand(side="Left", score=0.8, hid=1)
        phantom = hand(side="Left", score=1.0, size=0.03, hid=2)
        self.assertIs(self.pick(phantom, real)["Left"], real)
        self.assertIs(self.pick(real, phantom)["Left"], real)

    def test_the_score_decides_between_real_hands_when_both_sides_are_taken(self):
        low, high = hand(side="Left", score=0.8, hid=1), hand(side="Left", score=0.9, hid=2)
        right = hand(side="Right", hid=3)
        got = self.pick(low, high, right)
        self.assertIs(got["Left"], high)
        self.assertIs(got["Right"], right)

    def test_position_places_two_real_hands_on_one_label(self):
        a = hand(side="Right", palm=(0.8, 0.4), hid=1)
        b = hand(side="Right", palm=(0.2, 0.4), hid=2)
        got = self.pick(a, b)
        self.assertEqual((got["Left"]["id"], got["Right"]["id"]), (2, 1))
        self.assertEqual((got["Left"]["side"], got["Right"]["side"]), ("Left", "Right"))
        self.assertEqual(a["side"], "Right")     # the tracker's own dict is untouched

    def test_only_a_small_hand_is_still_returned_for_the_usable_rule(self):
        small = hand(side="Left", size=0.03, hid=1)
        self.assertIs(self.pick(small)["Left"], small)

    def test_garbage_is_ignored(self):
        self.assertEqual(h2.hm.hands_by_side(None, 640, 480, self.P), {})
        self.assertEqual(h2.hm.hands_by_side(["x", 3, None], 640, 480, self.P), {})


if __name__ == "__main__":
    unittest.main()
