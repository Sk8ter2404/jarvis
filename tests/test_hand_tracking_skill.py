"""skills/hand_tracking.py: the palm-hold -> desktop holo window trigger (menu
setting, 10 s rate limit, game-mode / fullscreen / call blocks, no cursor or
OS input), the listener's message filter, and the three voice actions. A fake
launcher and fake websocket; nothing real runs."""
import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "skill_hand_tracking_under_test", os.path.join(_ROOT, "skills", "hand_tracking.py"))
ht = importlib.util.module_from_spec(_spec)
sys.modules["skill_hand_tracking_under_test"] = ht
_spec.loader.exec_module(ht)

def _read(rel):
    with open(os.path.join(_ROOT, rel), encoding="utf-8") as f:
        return f.read()


PALM = {"type": "gesture", "name": "palm_hold", "side": "Right", "pos": [0.5, 0.5]}


class FakeLauncher:
    def __init__(self, start=(True, "started")):
        self._start = start
        self._running = False
        self._wanted = False
        self.gone = False
        self.hb = None
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1
        if self._start[0]:
            self._running = self._wanted = True
        return self._start

    def stop(self):
        self.stopped += 1
        self._running = self._wanted = False

    def running(self):
        return self._running

    def wanted(self):
        return self._wanted

    def gave_up(self):
        return self.gone

    def heartbeat(self):
        return self.hb


class TriggerTests(unittest.TestCase):
    def setUp(self):
        ht._state.update({"last_open": -1e9, "opens": 0, "warned_missing": False,
                          "last_block": ""})
        self.opened = []
        ht.set_open_hook(lambda: self.opened.append(1))
        self.addCleanup(ht.set_open_hook, None)
        p = mock.patch.multiple(config, HAND_MENU_ENABLED=True)
        p.start()
        self.addCleanup(p.stop)
        for name in ("_game_mode_active", "_fullscreen_app", "_in_call"):
            q = mock.patch.object(ht, name, return_value=False)
            q.start()
            self.addCleanup(q.stop)

    def test_palm_hold_opens_the_window(self):
        self.assertEqual(ht.handle_gesture(PALM, now=100.0), "opened")
        self.assertEqual(self.opened, [1])

    def test_off_by_default_and_when_the_menu_is_disabled(self):
        with mock.patch.object(config, "HAND_MENU_ENABLED", False):
            self.assertIn("menu is off", ht.handle_gesture(PALM, now=100.0))
        self.assertEqual(self.opened, [])

    def test_other_gestures_do_not_open_it(self):
        for name in ("pinch_hold", "fist_grab", "clap", "explode", "point_flick"):
            ht.handle_gesture({"type": "gesture", "name": name}, now=100.0)
        self.assertEqual(self.opened, [])

    def test_at_most_once_per_ten_seconds(self):
        self.assertEqual(ht.handle_gesture(PALM, now=100.0), "opened")
        self.assertIn("a moment ago", ht.handle_gesture(PALM, now=105.0))
        self.assertIn("a moment ago", ht.handle_gesture(PALM, now=109.9))
        self.assertEqual(ht.handle_gesture(PALM, now=110.1), "opened")
        self.assertEqual(len(self.opened), 2)

    def test_blocked_attempts_do_not_start_the_cooldown(self):
        with mock.patch.object(ht, "_game_mode_active", return_value=True):
            self.assertIn("game mode", ht.handle_gesture(PALM, now=100.0))
        self.assertEqual(self.opened, [])
        self.assertEqual(ht.handle_gesture(PALM, now=101.0), "opened")

    def test_never_in_game_mode_fullscreen_or_a_call(self):
        for name, why in (("_game_mode_active", "game mode"),
                          ("_fullscreen_app", "fullscreen"),
                          ("_in_call", "call")):
            with mock.patch.object(ht, name, return_value=True):
                out = ht.handle_gesture(PALM, now=1000.0 + len(self.opened))
                self.assertIn(why, out)
        self.assertEqual(self.opened, [])

    def test_holo_action_is_used_when_present(self):
        ht.set_open_hook(None)
        calls = []
        bc = mock.Mock(ACTIONS={"holo_mode_on": lambda a="": calls.append(a)})
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.assertEqual(ht.handle_gesture(PALM, now=100.0), "opened")
        self.assertEqual(calls, ["desktop"])

    def test_missing_holo_action_is_logged_once_and_does_nothing(self):
        ht.set_open_hook(None)
        bc = mock.Mock(ACTIONS={})
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
                mock.patch("builtins.print") as pr:
            self.assertIn("no holo opener", ht.handle_gesture(PALM, now=100.0))
            self.assertIn("no holo opener", ht.handle_gesture(PALM, now=200.0))
        warned = [c for c in pr.call_args_list if "holo_mode_on" in str(c)]
        self.assertEqual(len(warned), 1)
        self.assertEqual(ht._state["opens"], 0)

    def test_a_failing_opener_is_contained(self):
        def boom():
            raise RuntimeError("window failed")
        ht.set_open_hook(boom)
        self.assertIn("would not open", ht.handle_gesture(PALM, now=100.0))

    def test_malformed_event_never_raises(self):
        self.assertTrue(ht.handle_gesture({}, now=1.0).startswith("ignored"))
        self.assertTrue(ht.handle_gesture(None).startswith("ignored"))


class BlockerTests(unittest.TestCase):
    def test_game_mode_is_read_from_the_game_mode_skill(self):
        st = mock.Mock(active=True)
        with mock.patch.dict(sys.modules, {"skill_game_mode": mock.Mock(_st=st)}):
            self.assertTrue(ht._game_mode_active())
            self.assertEqual(ht.blocked_reason(), "game mode is on")
        st.active = False
        with mock.patch.dict(sys.modules, {"skill_game_mode": mock.Mock(_st=st)}):
            self.assertFalse(ht._game_mode_active())

    def test_fullscreen_foreground_app(self):
        gm = mock.Mock(_st=mock.Mock(active=False),
                       _fullscreen_foreground=lambda: (123, "app.exe", "C:/app.exe"))
        with mock.patch.dict(sys.modules, {"skill_game_mode": gm}):
            self.assertIn("fullscreen", ht.blocked_reason())
        gm._fullscreen_foreground = lambda: (None, None, None)
        with mock.patch.dict(sys.modules, {"skill_game_mode": gm, "skill_banter": mock.Mock(
                _is_in_call=lambda: False)}):
            self.assertEqual(ht.blocked_reason(), "")

    def test_in_a_call(self):
        gm = mock.Mock(_st=mock.Mock(active=False),
                       _fullscreen_foreground=lambda: (None, None, None))
        with mock.patch.dict(sys.modules, {"skill_game_mode": gm, "skill_banter": mock.Mock(
                _is_in_call=lambda: True)}):
            self.assertIn("call", ht.blocked_reason())

    def test_unknown_state_does_not_block(self):
        with mock.patch.dict(sys.modules, {"skill_game_mode": None, "skill_banter": None,
                                           "skill_anticipation_engine": None,
                                           "skill_weekly_digest_briefing": None}):
            self.assertEqual(ht.blocked_reason(), "")


class SafetyTests(unittest.TestCase):
    def test_the_skill_never_touches_the_cursor_or_os_input(self):
        src = _read("skills/hand_tracking.py")
        for bad in ("pyautogui", "SetCursorPos", "mouse_event", "SendInput",
                    "keybd_event", "pynput"):
            self.assertNotIn(bad, src)
        for path in ("core/handtrack_frames.py", "core/hand_tracker_launcher.py",
                     "tools/hand_tracker_service.py", "tools/handtrack_gestures.py"):
            text = _read(path)
            for bad in ("pyautogui", "SetCursorPos", "mouse_event", "SendInput",
                        "keybd_event", "pynput"):
                self.assertNotIn(bad, text, path)

    def test_defaults_are_off(self):
        self.assertFalse(config.HAND_TRACKING_ENABLED)
        self.assertFalse(config.HAND_MENU_ENABLED)
        self.assertEqual(config.HAND_TRACK_CAMERA, "primary")

    def test_register_does_not_start_anything_by_default(self):
        with mock.patch.object(ht, "start_at_boot") as boot:
            actions = {}
            ht.register(actions)
        boot.assert_not_called()
        self.assertEqual(sorted(actions), ["hand_tracking_off", "hand_tracking_on",
                                           "hand_tracking_status"])

    def test_register_starts_the_tracker_when_the_setting_is_on(self):
        with mock.patch.object(config, "HAND_TRACKING_ENABLED", True), \
                mock.patch.object(ht, "start_at_boot") as boot:
            ht.register({})
        boot.assert_called_once()

    def test_verbatim_speak_set_matches_the_actions(self):
        self.assertEqual(set(ht.SPEAK_VERBATIM_ACTIONS),
                         {"hand_tracking_on", "hand_tracking_off",
                          "hand_tracking_status"})


class ListenerTests(unittest.TestCase):
    def test_only_gesture_messages_are_parsed(self):
        got = []
        ls = ht.GestureListener("ws://x", on_event=got.append)
        ls.feed(json.dumps({"type": "hands", "hands": []}, separators=(",", ":")))
        ls.feed(b"\xff\xd8binary")
        ls.feed("not json at all")
        ls.feed('{"type":"gesture","name":"palm_hold"')        # truncated
        ls.feed(json.dumps(PALM, separators=(",", ":")))
        self.assertEqual(got, [PALM])

    def test_pump_reads_until_stopped(self):
        got = []
        ls = ht.GestureListener("ws://x", on_event=got.append)
        msgs = [json.dumps(PALM, separators=(",", ":")), TimeoutError(), "{}"]

        class WS:
            def recv(self, timeout=None):
                m = msgs.pop(0)
                if isinstance(m, Exception):
                    raise m
                if not msgs:
                    ls.stop()
                return m
        ls.pump(WS())
        self.assertEqual(got, [PALM])

    def test_url_uses_the_loopback_port(self):
        with mock.patch.object(config, "HAND_TRACK_PORT", 8793):
            self.assertEqual(ht._ws_url(), "ws://127.0.0.1:8793")


class _ActionBase(unittest.TestCase):
    def setUp(self):
        self.la = FakeLauncher()
        for target, kw in ((ht, {"_launcher": lambda: self.la}),):
            for k, v in kw.items():
                p = mock.patch.object(target, k, v)
                p.start()
                self.addCleanup(p.stop)
        p = mock.patch.object(ht, "_ensure_listener")
        self.ensure = p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(ht, "_stop_listener")
        self.stop_l = p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(ht, "_staging", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.multiple(config, HAND_TRACKING_ENABLED=False,
                                HAND_MENU_ENABLED=False, HAND_TRACK_CAMERA="primary")
        q.start()
        self.addCleanup(q.stop)


class ActionTests(_ActionBase):
    def test_on_starts_the_tracker_and_the_listener(self):
        out = ht._hand_tracking_on()
        self.assertIn("Hand tracking is on", out)
        self.assertIn("never touches your cursor", out)
        self.assertEqual(self.la.started, 1)
        self.ensure.assert_called_once()
        self.assertTrue(config.HAND_TRACKING_ENABLED)

    def test_on_twice_says_already_on(self):
        ht._hand_tracking_on()
        self.la._start = (True, "already running")
        self.assertIn("already on", ht._hand_tracking_on())

    def test_on_is_honest_when_it_cannot_start(self):
        self.la._start = (False, "the hand model file isn't downloaded (HAND_TRACK_MODEL)")
        out = ht._hand_tracking_on()
        self.assertIn("couldn't start", out)
        self.assertIn("model file", out)
        self.assertFalse(config.HAND_TRACKING_ENABLED)
        self.ensure.assert_not_called()

    def test_off_stops_everything(self):
        ht._hand_tracking_on()
        out = ht._hand_tracking_off()
        self.assertIn("Hand tracking is off", out)
        self.assertEqual(self.la.stopped, 1)
        self.stop_l.assert_called()
        self.assertFalse(config.HAND_TRACKING_ENABLED)
        self.assertIn("already off", ht._hand_tracking_off())

    def test_off_mentions_the_boot_setting(self):
        ht._hand_tracking_on()
        with mock.patch.object(config, "HAND_TRACKING_ENABLED", True):
            # the setting is read after the runtime flag is cleared, so pin it
            with mock.patch.object(ht, "_set_runtime"):
                out = ht._hand_tracking_off()
        self.assertIn("start at boot", out)

    def test_status_wording(self):
        self.assertEqual(ht._hand_tracking_status(), "Hand tracking is off, sir.")
        ht._hand_tracking_on()
        self.assertIn("loading", ht._hand_tracking_status())
        self.la.hb = {"age_s": 0.5, "attached": False}
        self.assertIn("isn't getting frames", ht._hand_tracking_status())
        self.la.hb = {"age_s": 0.5, "attached": True, "hands": 0, "fps": 29.6}
        s = ht._hand_tracking_status()
        self.assertIn("can't see any hands", s)
        self.assertIn("30 frames", s)
        self.la.hb["hands"] = 2
        self.assertIn("both hands", ht._hand_tracking_status())
        self.la.hb["age_s"] = 20
        self.assertIn("loading", ht._hand_tracking_status())
        self.la.gone = True
        self.assertIn("gave up", ht._hand_tracking_status())

    def test_staging_never_runs_the_tracker(self):
        with mock.patch.object(ht, "_staging", return_value=True):
            ok, msg = ht.start_tracking()
        self.assertFalse(ok)
        self.assertEqual(self.la.started, 0)


if __name__ == "__main__":
    unittest.main()


class HologramOwnershipTests(_ActionBase):
    """The hologram starts the tracker when it isn't running and stops only a
    tracker IT started (by voice close, or its window gone 45 s); a tracker
    the owner turned on is never stopped by the hologram."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(ht, "_start_holo_watch")
        self.watch = p.start()
        self.addCleanup(p.stop)
        ht._holo.update(owned=False, idle_since=None)
        self.addCleanup(lambda: ht._holo.update(owned=False, idle_since=None))

    def test_hologram_starts_the_tracker_and_owns_it(self):
        ok, msg = ht.ensure_for_holo()
        self.assertTrue(ok)
        self.assertEqual(self.la.started, 1)
        self.assertTrue(ht._holo["owned"])
        self.watch.assert_called_once()
        self.assertTrue(ht.release_for_holo())
        self.assertEqual(self.la.stopped, 1)
        self.assertFalse(config.HAND_TRACKING_ENABLED)

    def test_hologram_never_stops_the_owners_tracker(self):
        ht._hand_tracking_on()
        ok, msg = ht.ensure_for_holo()
        self.assertEqual((ok, msg), (True, "already running"))
        self.assertFalse(ht._holo["owned"])
        self.assertFalse(ht.release_for_holo())
        self.assertEqual(self.la.stopped, 0)

    def test_owner_turning_it_on_takes_it_over(self):
        ht.ensure_for_holo()
        ht._hand_tracking_on()
        self.assertFalse(ht._holo["owned"])
        self.assertFalse(ht.release_for_holo())
        self.assertEqual(self.la.stopped, 0)

    def test_a_failed_start_owns_nothing(self):
        self.la._start = (False, "the tracker's Python isn't installed")
        ok, msg = ht.ensure_for_holo()
        self.assertFalse(ok)
        self.assertIn("Python", msg)
        self.assertFalse(ht._holo["owned"])
        self.watch.assert_not_called()

    def test_window_gone_45_s_stops_it_and_a_live_window_resets_the_clock(self):
        ht.ensure_for_holo()
        ht._holo["idle_since"] = None
        self.assertFalse(ht._holo_watch_step(100.0, False))      # starts the clock
        self.assertFalse(ht._holo_watch_step(140.0, False))      # 40 s: not yet
        self.assertFalse(ht._holo_watch_step(141.0, True))       # window back
        self.assertIsNone(ht._holo["idle_since"])
        self.assertFalse(ht._holo_watch_step(150.0, False))
        self.assertFalse(ht._holo_watch_step(190.0, False))
        self.assertTrue(ht._holo_watch_step(195.0, False))       # 45 s gone
        self.assertEqual(self.la.stopped, 1)
        self.assertFalse(ht._holo_watch_step(300.0, False))      # nothing left

