"""open_on_monitor and slow-starting apps (2026-10-02 live).

Live 14:47:52-14:48:07: "open Teams on the top monitor" launched Teams (closed
since the morning) but no Teams window appeared inside the action's 15 s wait,
so it returned "launched ..., but couldn't find new window to move it - if it
reused a window that was already open, ask me ...". That wording made the
follow-up round tell the owner Teams "appears to be already open" and offer to
move "the existing window"; at 14:48:38 move_window_to_monitor found no Teams
window at all - the new one simply had not appeared yet.

Now:
  * an app whose window has not appeared when the wait ends is watched for up
    to ~30 s more on a background thread (never the voice turn), and its window
    is moved when it appears, with one log line;
  * the immediate result says exactly that, carries no failure wording, and
    never calls the app already open;
  * only a window of THAT app (process-name match, the v2.0.176 close_window
    helper) found BEFORE the launch counts as "already open" - and it is moved
    at once. A window that merely has the vendor word in its title ("Microsoft
    Edge" for "Microsoft Teams") is not the app.

Fake clocks, fake window lists, fake process names; no real window, process,
thread or app is touched (the one threading test runs a stub watcher).

    python -m unittest tests.test_open_on_monitor_slow_app
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import time
import types
import unittest
from unittest import mock

import core.actions as A
from core.failure_markers import FAILURE_MARKERS
from tests.test_actions_sec3 import _FakeWindow, _base_bc, _clock, _patch_bc

MONS = {"left": (0, 0, 1920, 1080), "top": (0, -1080, 1920, 1080)}


def _win(title, hwnd, width=1200, height=800):
    w = _FakeWindow(title, width=width, height=height)
    w._hWnd = hwnd
    w.moves = 0
    real_max = w.maximize

    def _maximize():
        w.moves += 1
        real_max()
    w.maximize = _maximize
    return w


def _gw_frames(frames):
    """Fake pygetwindow whose getAllWindows() walks ``frames`` (the last one
    repeats). Frame 0 is the handler's pre-launch snapshot."""
    mod = types.ModuleType("pygetwindow")
    state = {"i": 0}

    def _all():
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1
        return list(frames[i])

    mod.getAllWindows = _all
    mod.calls = state
    return mod


def _bc():
    bc = _base_bc()
    bc._open_url_new_window.return_value = True
    bc._strip_bidi_and_nbsp = lambda s: s
    bc._BROWSER_CHROME_SUFFIXES = (" - google chrome", " - microsoft edge",
                                   " - mozilla firefox")
    return bc


def _is_failure(text) -> bool:
    low = str(text).lower()
    return any(m in low for m in FAILURE_MARKERS)


class _Base(unittest.TestCase):
    def setUp(self):
        self.procs = {}
        p = mock.patch.object(
            A, "_window_process_name",
            side_effect=lambda w: self.procs.get(getattr(w, "_hWnd", None)))
        p.start()
        self.addCleanup(p.stop)


# ════════════════════════════════════════════════════════════════════════════
#  The action's immediate result
# ════════════════════════════════════════════════════════════════════════════
class _OpenBase(_Base):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(A, "_start_open_watch", return_value=object())
        self.watch = p.start()
        self.addCleanup(p.stop)

    def _open(self, frames, args="top | Microsoft Teams", launch="launched"):
        bc = _bc()
        gw = _gw_frames(frames)
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A, "_act_launch_app",
                                  return_value=launch) as la, \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock(step=0.5)):
            out = A._act_open_on_monitor(args)
        return out, la, gw


class SlowAppResultTests(_OpenBase):
    def test_the_live_case_says_it_will_move_teams_when_it_appears(self):
        # Teams closed, nothing of it on screen; no window within the wait.
        other = _win("Inbox - Mail", 0x10)
        self.procs[0x10] = "olk.exe"
        out, la, _gw = self._open([[other], [other]])
        la.assert_called_once_with("Microsoft Teams")
        self.assertEqual(out, "launched Microsoft Teams; I'll move it to the "
                              "top monitor when its window appears")
        self.assertFalse(_is_failure(out), out)
        self.assertNotIn("already open", out)
        self.assertNotIn("reused", out)
        self.watch.assert_called_once()
        args, kwargs = self.watch.call_args
        gw_arg, _bc_arg, target, words, before, monitor, rect = args
        self.assertEqual((target, words, monitor), ("Microsoft Teams",
                                                    ["teams"], "top"))
        self.assertEqual(rect, MONS["top"])
        self.assertEqual(before, {0x10})
        self.assertGreater(kwargs["waited_s"], 0)

    def test_a_vendor_word_in_another_apps_title_is_not_teams(self):
        # "Microsoft" in an Edge title matched the old any-token test; it is
        # not Teams, so it is neither moved nor called "already open".
        edge = _win("New tab - Microsoft Edge", 0x20)
        self.procs[0x20] = "msedge.exe"
        out, _la, _gw = self._open([[edge], [edge]])
        self.assertFalse(edge.maximized)
        self.assertIsNone(edge.moved_to)
        self.assertNotIn("already open", out)
        self.assertIn("when its window appears", out)
        self.watch.assert_called_once()

    def test_teams_open_before_the_launch_is_moved_now(self):
        teams = _win("Chat | Microsoft Teams", 0x30)
        self.procs[0x30] = "ms-teams.exe"
        out, _la, _gw = self._open([[teams], [teams]])
        self.assertTrue(teams.maximized)
        self.assertEqual(teams.moves, 1)
        self.assertEqual(teams.moved_to, (50, -1030))
        self.assertIn("Microsoft Teams was already open, so I moved its "
                      "'Chat | Microsoft Teams' window to the top monitor", out)
        self.assertFalse(_is_failure(out), out)
        self.watch.assert_not_called()

    def test_a_minimized_teams_window_counts_as_open(self):
        # Windows reports a minimized window as a tiny icon rect.
        teams = _win("Chat | Microsoft Teams", 0x31, width=160, height=28)
        teams.isMinimized = True
        self.procs[0x31] = "ms-teams.exe"
        out, _la, _gw = self._open([[teams], [teams]])
        self.assertTrue(teams.maximized)
        self.assertIn("was already open", out)

    def test_the_new_window_inside_the_wait_is_moved_as_before(self):
        new = _win("Microsoft Teams", 0x40)
        self.procs[0x40] = "ms-teams.exe"
        out, _la, _gw = self._open([[], [], [new]])
        self.assertTrue(new.maximized)
        self.assertIn("opened 'Microsoft Teams' on top monitor", out)
        self.watch.assert_not_called()

    def test_a_new_window_found_by_its_process_alone_is_moved(self):
        # A starting app's first window title may not name it yet.
        new = _win("Loading", 0x41)
        self.procs[0x41] = "ms-teams.exe"
        out, _la, _gw = self._open([[], [new]])
        self.assertTrue(new.maximized)
        self.assertIn("on top monitor", out)

    def test_a_failed_launch_is_reported_and_nothing_is_watched(self):
        out, _la, gw = self._open([[]], launch="could not launch Microsoft "
                                               "Teams: not found")
        self.assertEqual(out, "could not launch Microsoft Teams: not found")
        self.watch.assert_not_called()
        self.assertEqual(gw.calls["i"], 1, "polled after a failed launch")

    def test_a_url_never_starts_a_watch(self):
        out, _la, _gw = self._open([[]], args="top | example.com")
        self.assertIn("couldn't find new window to move it", out)
        self.assertNotIn("already open", out)
        self.watch.assert_not_called()

    def test_no_watch_thread_means_an_honest_failure(self):
        self.watch.return_value = None
        out, _la, _gw = self._open([[]])
        self.assertTrue(_is_failure(out), out)
        self.assertNotIn("already open", out)


# ════════════════════════════════════════════════════════════════════════════
#  The background watch
# ════════════════════════════════════════════════════════════════════════════
class _FakeClock:
    """A clock that only moves when the watch sleeps."""

    def __init__(self, t=500.0):
        self.t = t
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class WatchTests(_Base):
    def _watch(self, frames, before=(), timeout_s=30.0, poll_s=0.5,
               cancel=None):
        clock = _FakeClock()
        gw = _gw_frames(frames)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = A._watch_for_app_window(
                gw, _bc(), "Microsoft Teams", ["teams"], set(before), "top",
                MONS["top"], waited_s=15.0, timeout_s=timeout_s, poll_s=poll_s,
                clock=clock, sleep=clock.sleep, cancel=cancel)
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        return res, lines, clock, gw

    def test_the_window_that_appears_later_is_moved_once(self):
        teams = _win("Calendar | Microsoft Teams", 0x50)
        self.procs[0x50] = "ms-teams.exe"
        # 20 empty polls (10 s), then Teams.
        res, lines, clock, _gw = self._watch([[]] * 21 + [[teams]])
        self.assertEqual(res, "moved")
        self.assertEqual(teams.moves, 1)
        self.assertEqual(teams.moved_to, (50, -1030))
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("moved it to the top monitor", lines[0])
        self.assertIn("26 s after the launch", lines[0])

    def test_no_window_ends_at_the_bound_with_one_line(self):
        res, lines, clock, gw = self._watch([[]])
        self.assertEqual(res, "timeout")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("no window within 45 s of the launch", lines[0])
        self.assertAlmostEqual(sum(clock.sleeps), 30.0)
        self.assertLessEqual(gw.calls["i"], 61)

    def test_other_windows_are_never_moved(self):
        # A new Edge window, a pre-existing Teams-titled page and a tiny
        # Teams stub all appear during the watch: none is Teams' own window.
        edge = _win("Teams - Microsoft Edge", 0x60)
        self.procs[0x60] = "msedge.exe"
        old = _win("Chat | Microsoft Teams", 0x61)
        self.procs[0x61] = "ms-teams.exe"
        stub = _win("Microsoft Teams", 0x62, width=120, height=40)
        self.procs[0x62] = "ms-teams.exe"
        res, lines, _clock_, _gw = self._watch([[edge, old, stub]],
                                               before={0x61})
        self.assertEqual(res, "timeout")
        for w in (edge, old, stub):
            self.assertFalse(w.maximized, w.title)
            self.assertIsNone(w.moved_to, w.title)

    def test_a_newer_request_cancels_the_watch(self):
        cancel = threading.Event()
        cancel.set()
        res, lines, _clock_, gw = self._watch([[]], cancel=cancel)
        self.assertEqual(res, "cancelled")
        self.assertEqual(len(lines), 1, lines)

    def test_a_move_that_fails_is_logged_and_stops(self):
        teams = _win("Microsoft Teams", 0x70)
        self.procs[0x70] = "ms-teams.exe"
        teams.maximize = mock.Mock(side_effect=RuntimeError("denied"))
        res, lines, _clock_, _gw = self._watch([[], [teams]])
        self.assertEqual(res, "move-failed")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("could not be moved: denied", lines[0])

    def test_a_packaged_app_is_judged_by_its_title(self):
        # Calculator's window belongs to ApplicationFrameHost.exe.
        calc = _win("Calculator", 0x80)
        self.procs[0x80] = "ApplicationFrameHost.exe"
        clock = _FakeClock()
        with contextlib.redirect_stdout(io.StringIO()):
            res = A._watch_for_app_window(
                _gw_frames([[], [calc]]), _bc(), "calculator", ["calculator"],
                set(), "top", MONS["top"], clock=clock, sleep=clock.sleep)
        self.assertEqual(res, "moved")
        self.assertTrue(calc.maximized)


class StartWatchTests(_Base):
    def test_the_watch_runs_off_the_calling_thread_and_returns_at_once(self):
        release = threading.Event()
        threads, cancels = [], []
        both = threading.Semaphore(0)

        def fake_watch(*a, **k):
            threads.append(threading.current_thread())
            cancels.append(k.get("cancel"))
            both.release()
            release.wait(5.0)
            return "moved"

        with mock.patch.object(A, "_watch_for_app_window", fake_watch):
            t0 = time.monotonic()
            t = A._start_open_watch(object(), _bc(), "Microsoft Teams",
                                    ["teams"], set(), "top", MONS["top"], 15.0)
            self.assertLess(time.monotonic() - t0, 1.0)
            self.assertIsNotNone(t)
            self.assertTrue(both.acquire(timeout=5.0))
            self.assertTrue(t.daemon)
            self.assertIsNot(threads[0], threading.current_thread())
            # A second request for the same app takes over: the first watch
            # is told to stop, the second is not.
            t2 = A._start_open_watch(object(), _bc(), "Microsoft Teams",
                                     ["teams"], set(), "top", MONS["top"], 15.0)
            self.assertTrue(both.acquire(timeout=5.0))
            self.assertTrue(cancels[0].is_set())
            self.assertFalse(cancels[1].is_set())
            release.set()
            t.join(5.0)
            t2.join(5.0)
        self.assertFalse(t.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(A._OPEN_WATCHES, {})


class AppWindowMatchTests(_Base):
    def test_process_name_decides_when_it_can_be_read(self):
        bc = _bc()
        w = _win("Chat | Microsoft Teams", 0x90)
        self.procs[0x90] = "ms-teams.exe"
        self.assertTrue(A._is_app_window(bc, w, A._app_words("Microsoft Teams")))
        self.procs[0x90] = "msedge.exe"
        self.assertFalse(A._is_app_window(bc, w,
                                          A._app_words("Microsoft Teams")))

    def test_title_decides_only_without_a_process_name(self):
        bc = _bc()
        self.assertTrue(A._is_app_window(
            bc, _win("Chat | Microsoft Teams", 0x91), ["teams"]))
        # A page in a browser is not the app (unless the app IS the browser).
        self.assertFalse(A._is_app_window(
            bc, _win("Teams - Google Chrome", 0x92), ["teams"]))
        self.assertTrue(A._is_app_window(
            bc, _win("Teams - Google Chrome", 0x93), ["chrome"]))

    def test_app_words_drop_the_vendor(self):
        self.assertEqual(A._app_words("Microsoft Teams"), ["teams"])
        self.assertEqual(A._app_words("Google Chrome"), ["chrome"])
        self.assertEqual(A._app_words("Microsoft"), ["microsoft"])


# ════════════════════════════════════════════════════════════════════════════
#  Review 2026-10-02: the wrong app's window, a newer request, a browser tab
# ════════════════════════════════════════════════════════════════════════════
class WrongAppTests(_OpenBase):
    """ANY word of a multi-word name inside an exe name made another app's
    window "the app": "Visual Studio Code" matched bambustudio.exe through
    "studio", so a slow VS Code start moved the owner's Bambu Studio window
    and called VS Code "already open"."""

    def test_another_studio_app_is_not_vs_code_and_is_not_moved(self):
        bambu = _win("Bambu Studio", 0xA0)
        self.procs[0xA0] = "bambu-studio.exe"
        out, _la, _gw = self._open([[bambu], [bambu]],
                                   args="top | Visual Studio Code")
        self.assertFalse(bambu.maximized)
        self.assertIsNone(bambu.moved_to)
        self.assertNotIn("already open", out)
        self.assertIn("when its window appears", out)
        self.watch.assert_called_once()

    def test_the_shell_desktop_window_is_never_file_explorer(self):
        # "Program Manager" is explorer.exe too, and screen-sized.
        desk = _win("Program Manager", 0xA1, width=1920, height=1080)
        self.procs[0xA1] = "explorer.exe"
        out, _la, _gw = self._open([[desk], [desk]],
                                   args="top | File Explorer")
        self.assertFalse(desk.maximized)
        self.assertNotIn("already open", out)

    def test_the_watch_ignores_a_new_window_of_another_studio_app(self):
        atmel = _win("Solution1 - Atmel Studio", 0xA2)
        self.procs[0xA2] = "AtmelStudio.exe"
        clock = _FakeClock()
        with contextlib.redirect_stdout(io.StringIO()):
            res = A._watch_for_app_window(
                _gw_frames([[], [atmel]]), _bc(), "Visual Studio Code",
                A._app_words("Visual Studio Code"), set(), "top",
                MONS["top"], timeout_s=3.0, clock=clock, sleep=clock.sleep)
        self.assertEqual(res, "timeout")
        self.assertFalse(atmel.maximized)


class AppMatchTableTests(_Base):
    """_is_app_window on real exe / title shapes: the app's own windows
    match, other apps' windows sharing a word do not."""

    CASES = (
        # target, process, title, is that app's window
        ("Microsoft Teams", "ms-teams.exe", "Chat | Microsoft Teams", True),
        ("Microsoft Teams", "Teams.exe", "Microsoft Teams", True),
        ("Microsoft Teams", "msedge.exe", "Chat | Microsoft Teams", False),
        ("Microsoft Teams", "explorer.exe", "Teams - File Explorer", False),
        ("Microsoft Teams", "notepad.exe", "teams notes.txt - Notepad",
         False),
        ("Visual Studio Code", "Code.exe", "app.py - proj - Visual Studio "
                                           "Code", True),
        ("Visual Studio Code", "bambu-studio.exe", "Bambu Studio", False),
        ("Visual Studio Code", "AtmelStudio.exe", "x - Atmel Studio", False),
        ("Task Manager", "Taskmgr.exe", "Task Manager", True),
        ("OBS Studio", "obs64.exe", "OBS 30.2 - Profile: A - Scenes: B",
         True),
        ("Stream Deck", "StreamDeck.exe", "Stream Deck", True),
        ("Stream Deck", "streamlabs obs.exe", "Streamlabs Desktop", False),
        ("File Explorer", "explorer.exe", "Downloads - File Explorer", True),
        ("File Explorer", "explorer.exe", "Program Manager", False),
        # The exe name doesn't carry the app's name: its title's last part.
        ("Outlook", "olk.exe", "Inbox - someone - Outlook", True),
        ("PowerPoint", "POWERPNT.EXE", "Deck1 - PowerPoint", True),
        ("Spotify", "Spotify.exe", "Spotify Premium", True),
        ("Spotify", "chrome.exe", "Spotify - Web Player - Google Chrome",
         False),
        ("Google Chrome", "chrome.exe", "New Tab - Google Chrome", True),
    )

    def test_the_table(self):
        bc = _bc()
        for i, (target, proc, title, want) in enumerate(self.CASES):
            with self.subTest(target=target, proc=proc, title=title):
                w = _win(title, 0xB00 + i)
                self.procs[0xB00 + i] = proc
                self.assertEqual(
                    A._is_app_window(bc, w, A._app_words(target)), want)


class NewerRequestTests(_Base):
    """A newer request for the same app takes over AT ONCE: the old watch
    must not move the window after the owner's newer request put it
    somewhere else."""

    def setUp(self):
        super().setUp()
        self.cancel = threading.Event()
        A._OPEN_WATCHES.clear()
        A._OPEN_WATCHES["teams"] = self.cancel
        self.addCleanup(A._OPEN_WATCHES.clear)

    def test_a_new_open_on_monitor_cancels_the_watch_before_it_waits(self):
        seen = []
        bc = _bc()
        gw = _gw_frames([[]])

        def launch(_name):
            seen.append(self.cancel.is_set())
            return "launched"

        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", MONS), \
                mock.patch.dict(sys.modules, {"pygetwindow": gw}), \
                mock.patch.object(A, "_act_launch_app", side_effect=launch), \
                mock.patch.object(A, "_start_open_watch",
                                  return_value=object()), \
                mock.patch.object(A.time, "sleep"), \
                mock.patch.object(A.time, "time", _clock(step=0.5)):
            A._act_open_on_monitor("left | Teams")
        self.assertEqual(seen, [True], "the old watch still ran during the "
                                       "new request's wait")

    def test_moving_the_app_by_hand_cancels_its_watch(self):
        teams = _win("Chat | Microsoft Teams", 0xC0)
        self.procs[0xC0] = "ms-teams.exe"
        bc = _bc()
        bc._find_windows_by_title.return_value = [teams]
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": None, "win32con": None}), \
                mock.patch.object(A.time, "sleep"):
            teams.resizeTo = mock.Mock()
            out = A._act_move_window_to_monitor("Microsoft Teams | left")
        self.assertIn("moved", out)
        self.assertTrue(self.cancel.is_set())
        self.assertNotIn("teams", A._OPEN_WATCHES)

    def test_moving_another_app_leaves_the_watch_alone(self):
        pad = _win("notes.txt - Notepad", 0xC1)
        self.procs[0xC1] = "notepad.exe"
        bc = _bc()
        bc._find_windows_by_title.return_value = [pad]
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", MONS), \
                mock.patch.dict(sys.modules,
                                {"win32gui": None, "win32con": None}), \
                mock.patch.object(A.time, "sleep"):
            pad.resizeTo = mock.Mock()
            A._act_move_window_to_monitor("Notepad | left")
        self.assertFalse(self.cancel.is_set())

    def test_a_cancel_during_the_poll_stops_the_move(self):
        teams = _win("Microsoft Teams", 0xC2)
        self.procs[0xC2] = "ms-teams.exe"
        cancel = threading.Event()
        clock = _FakeClock()

        def sleep(s):
            clock.sleep(s)
            cancel.set()          # the newer request lands mid-sleep

        with contextlib.redirect_stdout(io.StringIO()):
            res = A._watch_for_app_window(
                _gw_frames([[teams]]), _bc(), "Microsoft Teams", ["teams"],
                set(), "top", MONS["top"], clock=clock, sleep=sleep,
                cancel=cancel)
        self.assertEqual(res, "cancelled")
        self.assertFalse(teams.maximized)


class BrowserHostedAppTests(_OpenBase):
    """Apple Music is the web player in the owner's browser: its launch
    either finds it already open or opens a TAB. No window of an app ever
    appears, so "I'll move it when its window appears" was a promise
    nothing could keep (main offered the move instead)."""

    def test_already_open_in_the_browser_says_so_at_once(self):
        chrome = _win("Song - Apple Music - Google Chrome", 0xD0)
        self.procs[0xD0] = "chrome.exe"
        out, _la, gw = self._open(
            [[chrome], [chrome]], args="top | Apple Music",
            launch="Apple Music is already open in the browser, sir.")
        self.watch.assert_not_called()
        self.assertNotIn("when its window appears", out)
        self.assertIn("already open", out)
        self.assertFalse(chrome.maximized)       # B092: never moved
        self.assertEqual(gw.calls["i"], 1, "waited for a window that the "
                                           "launch said already exists")

    def test_the_real_launcher_line_is_the_one_recognised(self):
        # The check above keys on the launcher's own words: pin them, so a
        # rewording there can't silently bring the false promise back.
        bc = _bc()
        with _patch_bc(bc), \
                mock.patch.object(A, "_web_player_titles",
                                  return_value=("Apple Music - Web",)):
            line = A._act_open_apple_music()
        self.assertIn("already open", line.lower())
        self.assertFalse(_is_failure(line), line)

    def test_a_new_tab_in_an_open_browser_window_is_offered(self):
        before = _win("Inbox - Google Chrome", 0xD1)
        after = _win("Apple Music - Google Chrome", 0xD1)
        self.procs[0xD1] = "chrome.exe"
        out, _la, _gw = self._open([[before], [after]],
                                   args="top | Apple Music",
                                   launch="opened Apple Music in the browser,"
                                          " sir")
        self.watch.assert_not_called()
        self.assertIn("reused your existing 'Apple Music - Google Chrome' "
                      "window", out)
        self.assertFalse(after.maximized)
        self.assertNotIn("when its window appears", out)

    def test_a_tab_that_named_the_app_before_the_launch_is_not_reuse(self):
        # Teams on the web in Chrome while the desktop app starts slowly:
        # that page is not where the launch went.
        web = _win("Chat | Microsoft Teams - Google Chrome", 0xD2)
        self.procs[0xD2] = "chrome.exe"
        out, _la, _gw = self._open([[web], [web]])
        self.assertNotIn("reused", out)
        self.assertIn("when its window appears", out)
        self.assertFalse(web.maximized)


if __name__ == "__main__":
    unittest.main()
