"""Monolith wiring for the 2026-10-01 diagnostic-batch fixes (claude/diag-fixes).

Each class pins one defect from the 09-05 heavy live diagnostic / the 10-01
audit, at the monolith level (the light-tier halves live in tests/test_*.py).
Generic fixtures only.

    python -m unittest tests.monolith.test_monolith_diag_fixes
"""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _ShortcutBase(MonolithGlobalsTestCase):
    """_run_voice_shortcuts with every earlier shortcut standing aside and
    the LLM booby-trapped (the test_monolith_fast_paths pattern)."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken = []
        self._p(bc, "_speak", side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "set_state")
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self.llm = self._p(bc, "_call_llm",
                           side_effect=AssertionError("LLM called"))
        self._p(bc, "maybe_replay_last_action", return_value=None)
        router = types.ModuleType("core.mode_router")
        router.maybe_handle_mode_toggle = lambda _t: None
        router.controlled_dispatch = lambda _t, _a: None
        router.is_in_controlled_mode = lambda: False
        disp = types.ModuleType("core.dispatcher")
        disp.resolve_and_dispatch = lambda _t, _a: None
        voice = types.SimpleNamespace(maybe_switch_backend=lambda _t: None)
        patcher = mock.patch.dict(bc.sys.modules, {
            "core.mode_router": router, "core.dispatcher": disp,
            "skill_custom_voice": voice})
        patcher.start()
        self.addCleanup(patcher.stop)
        bc.conversation_history.clear()

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _turn(self, text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            handled = self.bc._run_voice_shortcuts(text)
        return handled, buf.getvalue()


# ── item 6: "are you ok" / "run a system check" run the real self-check ────
class SelfCheckShortcutTests(_ShortcutBase):
    SUMMARY = ("Sir, nothing is reporting a failure, but I am not able to call "
               "the system nominal: the microphone did not run. "
               "(2.1s sweep, 20 probes.)")

    def setUp(self):
        super().setUp()
        self.diag = mock.Mock(return_value=self.SUMMARY)
        patcher = mock.patch.dict(self.bc.ACTIONS, {"are_you_ok": self.diag})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_are_you_ok_runs_the_self_diagnostic_and_speaks_it(self):
        for text in ("are you ok", "Jarvis, are you okay?", "run a system check"):
            with self.subTest(text=text):
                self.diag.reset_mock()
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, f"{text!r} fell through to the LLM")
                self.diag.assert_called_once()
                self.assertEqual(self.spoken[-1], self.SUMMARY)
                self.assertIn("[fast-path] self-check", log)
                self.assertIn(f"JARVIS: {self.SUMMARY}", log)
        self.llm.assert_not_called()

    def test_the_turn_is_recorded(self):
        self._turn("are you ok")
        self.assertEqual(self.bc.conversation_history[-2:], [
            {"role": "user", "content": "are you ok"},
            {"role": "assistant", "content": self.SUMMARY}])

    def test_a_diagnostic_that_raises_is_reported_honestly(self):
        self.diag.side_effect = RuntimeError("probe table missing")
        handled, _log = self._turn("are you ok")
        self.assertTrue(handled)
        self.assertIn("self-check", self.spoken[-1].lower())
        self.assertNotIn("nominal", self.spoken[-1].lower())

    def test_no_diagnostic_skill_falls_through(self):
        # Nothing to run: the shortcut stands aside rather than invent a pass.
        with mock.patch.dict(self.bc.ACTIONS, clear=False) as acts:
            for name in ("are_you_ok", "run_diagnostic", "self_diagnostic",
                         "system_check"):
                acts.pop(name, None)
            with mock.patch.object(self.bc, "_run_fast_paths",
                                   return_value=False):
                handled, _log = self._turn("are you ok")
        self.assertFalse(handled)

    def test_disabled_with_the_fast_paths(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        handled, _log = self._turn("are you ok")
        self.assertFalse(handled)
        self.diag.assert_not_called()

    def test_ordinary_turns_never_run_it(self):
        with mock.patch.object(self.bc, "_run_fast_paths", return_value=False):
            for text in ("are you ok with that", "check the system",
                         "what's the weather"):
                with self.subTest(text=text):
                    handled, _ = self._turn(text)
                    self.assertFalse(handled)
        self.diag.assert_not_called()



# ── item 4: list_timers never makes things up ───────────────────────────────
class TimerListShortcutTests(_ShortcutBase):
    """A whole-utterance timer question is answered from the timer store
    with no LLM, so the model never gets the chance to invent a timer."""

    LINE = "One timer is running, sir: number 1, 'tea', in 4 minutes."

    def setUp(self):
        super().setUp()
        self.lister = mock.Mock(return_value=self.LINE)
        patcher = mock.patch.dict(self.bc.ACTIONS, {"list_timers": self.lister})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_timer_questions_are_answered_from_the_store(self):
        for text in ("what timers do I have", "list my timers",
                     "Jarvis, any timers running?", "do I have any reminders",
                     "how much time is left on my timer"):
            with self.subTest(text=text):
                self.lister.reset_mock()
                self.spoken.clear()
                handled, log = self._turn(text)
                self.assertTrue(handled, f"{text!r} reached the LLM")
                self.lister.assert_called_once()
                self.assertEqual(self.spoken, [self.LINE])
                self.assertIn("[fast-path] timers", log)
        self.llm.assert_not_called()

    def test_setting_or_cancelling_a_timer_is_not_a_listing(self):
        with mock.patch.object(self.bc, "_run_fast_paths", return_value=False):
            for text in ("set a timer for 5 minutes", "cancel my timer",
                         "what time is it"):
                with self.subTest(text=text):
                    handled, _ = self._turn(text)
                    self.assertFalse(handled)
        self.lister.assert_not_called()

    def test_no_timer_skill_falls_through(self):
        with mock.patch.dict(self.bc.ACTIONS) as acts:
            acts.pop("list_timers", None)
            with mock.patch.object(self.bc, "_run_fast_paths",
                                   return_value=False):
                handled, _ = self._turn("what timers do I have")
        self.assertFalse(handled)


@requires_monolith
class TimerClaimTests(MonolithGlobalsTestCase):
    """How the invention happened: the local model answered "what timers do
    I have" in its OWN words — with no token at all (no guard knew a timer
    claim), or as prose in front of [ACTION: list_timers] (answer-first
    drops only pure acknowledgements, so the invented sentence was spoken
    before the real list). A timer-state claim now injects list_timers, and
    whenever list_timers runs the model's own timer claims are dropped: the
    store's line is the only answer voiced."""

    REAL = "No timers are running, sir."

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.lister = mock.Mock(return_value=self.REAL)
        acts = dict(bc.ACTIONS)
        acts["list_timers"] = self.lister
        for name, value in (("ACTIONS", acts),
                            ("_speak", lambda *a, **k: None),
                            ("_write_hud_state", lambda **k: None),
                            ("record_session_action", lambda *a, **k: None),
                            ("record_action_history", lambda *a, **k: None),
                            ("_cmd_autocorrect", None),
                            ("PC_CONTROL_ENABLED", True),
                            ("_needs_confirmation", lambda n, a: False),
                            ("_jarvis_pushback", lambda n, a: None)):
            p = mock.patch.object(bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _run(self, reply):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cleaned, results = self.bc.parse_and_run_actions(reply)
        return cleaned, results, buf.getvalue()

    def test_invented_prose_next_to_the_token_is_dropped(self):
        cleaned, results, log = self._run(
            "One moment, sir. You have two timers running, sir: tea in 4 "
            "minutes and the laundry in 20. [ACTION: list_timers]")
        self.lister.assert_called_once()
        self.assertEqual([r[0] for r in results], ["list_timers"])
        self.assertNotIn("two timers", cleaned)
        self.assertNotIn("laundry", cleaned)
        self.assertIn("One moment, sir.", cleaned)
        self.assertIn("[timers]", log)

    def test_a_tokenless_timer_claim_runs_the_real_list(self):
        for reply in ("You have a 5-minute tea timer running, sir.",
                      "There are no timers running right now, sir.",
                      "Your tea timer has 4 minutes left, sir.",
                      "It's 5 minutes left on your timer, sir."):
            with self.subTest(reply=reply):
                self.lister.reset_mock()
                verdict = self.bc._detect_preemptive_hallucination(reply)
                self.assertEqual(verdict[:2], ("inject", "list_timers"))
                cleaned, _results, _log = self._run(reply)
                self.lister.assert_called_once()
                self.assertEqual(cleaned, "",
                                 "the invented timer line was still voiced")

    def test_timer_prose_that_claims_no_state_is_left_alone(self):
        for reply in ("Timer set for 5 minutes, sir.",
                      "I'll remind you in 5 minutes, sir.",
                      "You have 3 unread emails, sir.",
                      "Your timer is up, sir.",
                      "Setting a timer is easy, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(self.bc._TIMER_STATE_CLAIM_RE.search(reply))

    def test_other_actions_keep_their_prose(self):
        other = mock.Mock(return_value="done")
        self.bc.ACTIONS["noop_x"] = other
        cleaned, _results, _log = self._run(
            "You have two timers running, sir. [ACTION: noop_x]")
        self.assertIn("two timers", cleaned)



# ── item 7: ordinary Chrome windows end up visible + maximized ──────────────
def _fake_win32(rect, work=(0, 0, 2560, 1440), is_window=True, zoomed=False):
    w32 = mock.MagicMock(name="win32gui")
    w32.IsWindow.return_value = is_window
    w32.IsZoomed.return_value = zoomed
    w32.GetWindowRect.return_value = rect
    con = mock.MagicMock(name="win32con")
    con.SW_RESTORE = 9
    con.SW_MAXIMIZE = 3
    api = mock.MagicMock(name="win32api")
    api.MonitorFromWindow.return_value = 111
    api.GetMonitorInfo.return_value = {"Work": work, "Monitor": work}
    return w32, con, api


def _win(hwnd, title, w=1200, h=800):
    return types.SimpleNamespace(_hWnd=hwnd, title=title, width=w, height=h)


@requires_monolith
class OrdinaryBrowserWindowPlacementTests(MonolithGlobalsTestCase):
    """The streaming paths pull the window JARVIS opened on-screen and
    maximize it (_adopt_media_window -> _ensure_window_visible_maximized);
    the ordinary paths (_open_url_in_browser's 'chrome --new-window',
    _open_url_new_window) never did, so a Chrome window restored to a
    remembered spot above a negative-origin monitor kept its title bar off
    the top of the screen (2026-07-07 report, still open on 10-01)."""

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.popen = self._p(bc.subprocess, "Popen")
        self._p(bc, "_find_chrome", return_value="chrome.exe")
        self._p(bc, "_window_handles_snapshot", return_value={1, 2})
        self._p(bc.webbrowser, "get", side_effect=RuntimeError("no controller"))
        self.place = self._p(bc, "_place_new_browser_window_async")
        self._p(bc, "_focus_steal_guard_active", return_value=False)

    def test_open_url_in_browser_places_the_new_window(self):
        how = self.bc._open_url_in_browser("example.com")
        self.assertEqual(how, "chrome")
        self.popen.assert_called_once()
        self.place.assert_called_once()
        args, kw = self.place.call_args
        self.assertEqual(args[0], {1, 2})
        self.assertIsNone(kw.get("monitor"))
        self.assertTrue(kw.get("activate"))

    def test_the_monitor_the_request_named_is_passed_through(self):
        self.bc._open_url_in_browser("example.com", monitor="left")
        self.assertEqual(self.place.call_args.kwargs.get("monitor"), "left")
        self.place.reset_mock()
        self.assertTrue(self.bc._open_url_new_window("example.com",
                                                     monitor="right"))
        self.assertEqual(self.place.call_args.kwargs.get("monitor"), "right")

    def test_open_url_new_window_places_the_new_window(self):
        self.assertTrue(self.bc._open_url_new_window("example.com"))
        self.place.assert_called_once()

    def test_media_mode_leaves_placement_to_the_adopter(self):
        self._p(self.bc, "_close_browser_windows_matching", return_value=0)
        self.bc._open_url_in_browser("example.com", close_matching=["x"])
        self.place.assert_not_called()

    def test_a_game_in_front_means_no_focus_is_taken(self):
        self._p(self.bc, "_focus_steal_guard_active", return_value=True)
        self.bc._open_url_in_browser("example.com")
        self.assertFalse(self.place.call_args.kwargs.get("activate"))


@requires_monolith
class NewBrowserWindowFinderTests(MonolithGlobalsTestCase):
    def test_only_a_new_browser_window_is_placed(self):
        bc = self.bc
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [
            _win(1, "A stream - Google Chrome"),             # pre-existing
            _win(5, "Untitled - Notepad"),                    # not a browser
            _win(6, "tooltip - Google Chrome", w=50, h=20),  # too small
            _win(7, "New Tab - Google Chrome")]               # THE new one
        with mock.patch.dict(bc.sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(bc, "_ensure_window_visible_maximized",
                                  return_value=True) as ensure:
            got = bc._place_new_browser_window({1, 2}, monitor="left",
                                               activate=False, timeout=1.0,
                                               poll=0.01)
        self.assertEqual(got, 7)
        ensure.assert_called_once_with(7, monitor="left", activate=False)

    def test_no_new_window_times_out_quietly(self):
        bc = self.bc
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [_win(1, "Old - Google Chrome")]
        with mock.patch.dict(bc.sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(bc, "_ensure_window_visible_maximized") as ensure:
            got = bc._place_new_browser_window({1}, timeout=0.05, poll=0.01)
        self.assertIsNone(got)
        ensure.assert_not_called()

    def test_the_async_placer_is_a_bounded_daemon(self):
        bc = self.bc
        with mock.patch.object(bc, "_place_new_browser_window",
                               return_value=None) as worker:
            t = bc._place_new_browser_window_async({1}, monitor=None,
                                                   activate=True)
            t.join(5)
        self.assertTrue(t.daemon)
        self.assertFalse(t.is_alive())
        worker.assert_called_once()


@requires_monolith
class EnsureVisibleOnMonitorTests(MonolithGlobalsTestCase):
    def test_a_named_monitor_is_the_target(self):
        bc = self.bc
        # A window on the middle monitor; the request named "left".
        w32, con, api = _fake_win32((100, 100, 1300, 900))
        with mock.patch.dict(bc.sys.modules, {"win32gui": w32,
                                              "win32con": con,
                                              "win32api": api}), \
                mock.patch.dict(bc.MONITORS, {"left": (-2560, 0, 2560, 1440)}):
            self.assertTrue(bc._ensure_window_visible_maximized(
                4242, monitor="left"))
        x, y = w32.SetWindowPos.call_args.args[2:4]
        self.assertTrue(-2560 <= x < 0, x)
        modes = [c.args[1] for c in w32.ShowWindow.call_args_list]
        self.assertEqual(modes[-1], con.SW_MAXIMIZE)

    def test_without_activation_nothing_takes_the_foreground(self):
        bc = self.bc
        w32, con, api = _fake_win32((100, -80, 1300, 720))   # off the top
        with mock.patch.dict(bc.sys.modules, {"win32gui": w32,
                                              "win32con": con,
                                              "win32api": api}):
            self.assertTrue(bc._ensure_window_visible_maximized(
                4242, activate=False))
        # ShowWindow(SW_RESTORE / SW_MAXIMIZE) activates the window: never.
        w32.ShowWindow.assert_not_called()
        args = w32.SetWindowPos.call_args.args
        self.assertEqual(args[2:6], (0, 0, 2560, 1440))      # fills the work area
        self.assertTrue(args[6] & 0x0010)                     # SWP_NOACTIVATE
        self.assertTrue(args[6] & 0x0004)                     # SWP_NOZORDER

    def test_streaming_callers_are_unchanged(self):
        # The default (activate=True, no monitor) is the old behaviour.
        bc = self.bc
        w32, con, api = _fake_win32((100, 100, 1300, 900))
        with mock.patch.dict(bc.sys.modules, {"win32gui": w32,
                                              "win32con": con,
                                              "win32api": api}):
            self.assertTrue(bc._ensure_window_visible_maximized(4242))
        w32.SetWindowPos.assert_not_called()
        modes = [c.args[1] for c in w32.ShowWindow.call_args_list]
        self.assertEqual(modes, [con.SW_RESTORE, con.SW_MAXIMIZE])


@requires_monolith
class FocusStealGuardTests(MonolithGlobalsTestCase):
    def _guard(self, fg_rect, mon_rect, cls="UnrealWindow", fg=99):
        bc = self.bc
        w32 = mock.MagicMock()
        w32.GetForegroundWindow.return_value = fg
        w32.GetClassName.return_value = cls
        w32.GetWindowRect.return_value = fg_rect
        w32.IsIconic.return_value = False
        api = mock.MagicMock()
        api.GetMonitorInfo.return_value = {"Monitor": mon_rect,
                                           "Work": mon_rect}
        with mock.patch.dict(bc.sys.modules, {"win32gui": w32,
                                              "win32api": api}), \
                mock.patch.object(bc, "_game_mode_active", return_value=False):
            return bc._focus_steal_guard_active()

    def test_game_mode_is_a_guard(self):
        with mock.patch.object(self.bc, "_game_mode_active", return_value=True):
            self.assertTrue(self.bc._focus_steal_guard_active())

    def test_a_fullscreen_game_in_front_is_a_guard(self):
        full = (0, 0, 2560, 1440)
        self.assertTrue(self._guard(full, full))

    def test_a_normal_window_or_the_desktop_is_not(self):
        full = (0, 0, 2560, 1440)
        self.assertFalse(self._guard((100, 100, 900, 700), full))
        self.assertFalse(self._guard(full, full, cls="WorkerW"))
        self.assertFalse(self._guard(full, full, cls="Chrome_WidgetWin_1"))
        self.assertFalse(self._guard(full, full, fg=0))



# ── item 8: the web wake-word switch flips REQUIRE_WAKE_MODE live ──────────
@requires_monolith
class WakeWordModeTrayTests(MonolithGlobalsTestCase):
    """The dashboard sends wake_word_mode_on / _off through tray_commands.json;
    the drainer must run the same live flip as the voice command and publish
    the state the dashboard's switch shows."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        import core.config as cfg
        saved_rt, saved_cfg = bc._require_wake_runtime, cfg.REQUIRE_WAKE_MODE

        def restore():
            bc._require_wake_runtime = saved_rt
            cfg.REQUIRE_WAKE_MODE = saved_cfg
        self.addCleanup(restore)
        self.cfg = cfg
        self.hud = []
        for target, name, value in (
                (bc, "_write_hud_state",
                 lambda **k: self.hud.append(k)),
                (bc, "_publish_tray_result", mock.Mock())):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        from tools import settings_window as sw
        p_save = mock.patch.object(sw, "save_settings")
        self.save = p_save.start()
        self.addCleanup(p_save.stop)
        p_load = mock.patch.object(sw, "load_settings", return_value={})
        p_load.start()
        self.addCleanup(p_load.stop)

    def _tray(self, cmd):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._dispatch_tray_command(cmd, {"cmd": cmd, "rid": "r1"})
        return buf.getvalue()

    def test_on_and_off_apply_live_and_publish(self):
        bc = self.bc
        log = self._tray("wake_word_mode_on")
        self.assertIs(bc._require_wake_runtime, True)
        self.assertIs(self.cfg.REQUIRE_WAKE_MODE, True)
        self.assertIn({"require_wake_mode": True}, self.hud)
        self.assertEqual(self.save.call_args.kwargs.get("changed"),
                         ("REQUIRE_WAKE_MODE",))
        self.assertIn("Wake-word mode on", log)
        bc._publish_tray_result.assert_called_with(
            "r1", "wake_word_mode_on", mock.ANY)
        self._tray("wake_word_mode_off")
        self.assertIs(bc._require_wake_runtime, False)
        self.assertIn({"require_wake_mode": False}, self.hud)

    def test_the_voice_command_publishes_too(self):
        self.bc._act_wake_word_mode_set(True)
        self.assertIn({"require_wake_mode": True}, self.hud)

    def test_boot_publishes_the_configured_value(self):
        import inspect
        main = inspect.getsource(self.bc.main)
        at = main.index("_restore_tray_toggle_state()")
        self.assertIn("_publish_wake_mode_state()", main[at:at + 400])


if __name__ == "__main__":
    unittest.main()
