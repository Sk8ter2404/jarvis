"""close_all_windows_except / minimize_all_windows_except (2026-10-03).

Live 17:22 the owner asked JARVIS to close every window but the Claude app.
There was no action for it: the brain ran list_windows and then
minimize_window six times - one of his folders, JARVIS's own HUD and Reticle,
a media player, "Program Manager" and "Windows Input Experience" - and
closed nothing.

Now ONE action does it:
  * every one of the owner's windows except the named ones is closed through
    the close path (WM_CLOSE via pygetwindow's close(): an app may ask to
    save; never a kill) - or minimized, for the minimize variant;
  * a name keeps a window by title OR by process ("Claude" keeps claude.exe);
  * JARVIS's own windows and the shell's are never touched (core.window_scope);
  * one summary: "Closed 2 windows, sir; kept Claude.";
  * an elevated window is reported ONCE, with close_window's terminal line;
  * a name that matches nothing closes NOTHING (a terminal line), because
    "everything except <nothing>" is everything.

Every window, Win32 read and process lookup is faked; nothing real is
closed, minimized or queried. Light tier (core.actions with a mocked
monolith).

    python -m unittest tests.test_close_all_windows_except
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import core.actions as A
from core import failure_markers as fm
from core import window_scope as ws

_HUD_PID = 9001


class _Win:
    """A pygetwindow-like window recording close() / minimize()."""

    def __init__(self, title, hwnd, *, raises=None, minimized=False):
        self.title = title
        self._hWnd = hwnd
        self.width, self.height = 900, 700
        self.isMinimized = minimized
        self._raises = raises
        self.closed = self.minimized = False

    def close(self):
        if self._raises is not None:
            raise self._raises
        self.closed = True

    def minimize(self):
        self.minimized = True


def _denied():
    return Exception("Error code from Windows: 5 - Access is denied.")


class _Base(unittest.TestCase):
    def setUp(self):
        self.bc = mock.Mock()
        self.bc.FORBIDDEN_TARGETS = ["powershell", "python", "terminal",
                                     "bobert"]
        self.bc._read_focused_window.return_value = (None, "", None)
        p = mock.patch.object(A, "_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)
        self.windows: list = []
        self.procs: dict = {}      # hwnd -> exe name
        self.pids: dict = {}       # hwnd -> pid
        self.classes: dict = {}    # hwnd -> window class
        self.cloaked: set = set()
        self.elevated: set = set()
        fake_gw = types.SimpleNamespace(getAllWindows=lambda: list(self.windows))
        for target, name, fn in (
                (A, "_window_process_name",
                 lambda w: self.procs.get(getattr(w, "_hWnd", None))),
                (A, "_window_is_elevated",
                 lambda w: getattr(w, "_hWnd", None) in self.elevated),
                (ws, "probe", lambda w: ws.WindowFacts(
                    w._hWnd, self.pids.get(w._hWnd),
                    self.classes.get(w._hWnd, ""), w._hWnd in self.cloaked)),
                (ws, "own_pids", lambda: frozenset({4000, _HUD_PID})),
                (ws, "own_window_handles", lambda: frozenset())):
            q = mock.patch.object(target, name, side_effect=fn)
            q.start()
            self.addCleanup(q.stop)
        q = mock.patch.dict(sys.modules, {"pygetwindow": fake_gw})
        q.start()
        self.addCleanup(q.stop)
        # A bulk close must never kill anything.
        for target, name in ((os, "kill"), (subprocess, "Popen"),
                             (subprocess, "run")):
            q = mock.patch.object(target, name,
                                  side_effect=AssertionError(
                                      f"{name} called: a close must be "
                                      "WM_CLOSE, never a kill"))
            q.start()
            self.addCleanup(q.stop)

    def add(self, title, hwnd, exe=None, **kw):
        w = _Win(title, hwnd, **kw)
        self.windows.append(w)
        if exe:
            self.procs[hwnd] = exe
        return w

    def live_desktop(self):
        """The live turn's windows (titles generic)."""
        claude = self.add("Claude", 0x10, "claude.exe")
        files = self.add("Downloads - File Explorer", 0x11, "explorer.exe")
        hud = self.add("JARVIS HUD", 0x12, "python.exe")
        self.pids[0x12] = _HUD_PID
        ret = self.add("JARVIS Reticle", 0x13, "pythonw.exe")
        self.pids[0x13] = _HUD_PID
        media = self.add("Media Player", 0x14, "ApplicationFrameHost.exe")
        desk = self.add("Program Manager", 0x15, "explorer.exe")
        self.classes[0x15] = "progman"
        wie = self.add("Windows Input Experience", 0x16, "TextInputHost.exe")
        self.cloaked.add(0x16)
        return claude, files, hud, ret, media, desk, wie


class LiveTurnTests(_Base):
    def test_closes_the_owners_windows_and_keeps_claude(self):
        claude, files, hud, ret, media, desk, wie = self.live_desktop()
        out = A._act_close_all_windows_except("Claude")
        self.assertEqual(out, "Closed 2 windows, sir; kept Claude.")
        self.assertTrue(files.closed and media.closed)
        self.assertFalse(claude.closed)
        for w in (hud, ret, desk, wie):
            self.assertFalse(w.closed or w.minimized, w.title)
        # Closing is closing: nothing was minimized instead.
        self.assertFalse(any(w.minimized for w in self.windows))

    def test_the_summary_is_spoken_whole(self):
        self.live_desktop()
        out = A._act_close_all_windows_except("Claude")
        low = out.lower()
        self.assertFalse(any(m in low for m in fm.FAILURE_MARKERS), out)
        self.assertEqual(fm.terminal_failure_text(out), "")

    def test_the_minimize_variant_minimizes_the_same_set(self):
        claude, files, hud, ret, media, desk, wie = self.live_desktop()
        out = A._act_minimize_all_windows_except("Claude")
        self.assertEqual(out, "Minimized 2 windows, sir; kept Claude.")
        self.assertTrue(files.minimized and media.minimized)
        for w in (claude, hud, ret, desk, wie):
            self.assertFalse(w.minimized, w.title)
        self.assertFalse(any(w.closed for w in self.windows))


class KeepMatchingTests(_Base):
    def test_claude_is_kept_by_process_name_whatever_the_title(self):
        app = self.add("New conversation", 0x20, "claude.exe")
        pad = self.add("Untitled - Notepad", 0x21, "notepad.exe")
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(app.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude.")

    def test_claude_is_kept_by_title_in_another_process(self):
        term = self.add("Claude Code", 0x22, "WindowsTerminal.exe")
        web = self.add("Claude - Google Chrome", 0x23, "chrome.exe")
        pad = self.add("notes.txt - Notepad", 0x24, "notepad.exe")
        out = A._act_close_all_windows_except("the Claude app")
        self.assertFalse(term.closed or web.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept the Claude app.")

    def test_several_names(self):
        claude = self.add("Claude", 0x25, "claude.exe")
        music = self.add("Spotify Premium", 0x26, "Spotify.exe")
        pad = self.add("notes.txt - Notepad", 0x27, "notepad.exe")
        out = A._act_close_all_windows_except("Claude and Spotify")
        self.assertFalse(claude.closed or music.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude and Spotify.")

    def test_a_name_that_is_not_open_is_said_once(self):
        self.add("Claude", 0x28, "claude.exe")
        pad = self.add("notes.txt - Notepad", 0x29, "notepad.exe")
        out = A._act_close_all_windows_except("Claude, the Spotify app")
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude. I saw no "
                              "Spotify window.")

    def test_this_one_keeps_the_window_in_front(self):
        front = self.add("Budget.xlsx - Excel", 0x2A, "EXCEL.EXE")
        pad = self.add("notes.txt - Notepad", 0x2B, "notepad.exe")
        self.bc._read_focused_window.return_value = (0x2A, front.title, None)
        out = A._act_close_all_windows_except("this one")
        self.assertFalse(front.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Excel.")

    def test_a_folder_whose_path_mentions_the_app_is_closed(self):
        # Review 2026-10-03: the live desktop's File Explorer window was a
        # folder under a path containing "Claude" (paraphrased here). A
        # title-substring keep kept it, so the live request would have
        # closed only the media player. The Claude app is open, so "Claude"
        # names the app: its windows (process, or a title that ends with
        # the name) are what is kept.
        claude = self.add("Claude", 0x50, "claude.exe")
        folder = self.add("C:\\Work\\Claude Projects\\drafts - File Explorer",
                          0x51, "explorer.exe")
        media = self.add("Media Player", 0x52, "ApplicationFrameHost.exe")
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(claude.closed)
        self.assertTrue(folder.closed)
        self.assertTrue(media.closed)
        self.assertEqual(out, "Closed 2 windows, sir; kept Claude.")

    def test_a_document_named_in_the_keep_is_kept_when_no_app_is(self):
        # No open app is called "budget": the name then keeps any window
        # whose title mentions it, so the owner's document stays open.
        self.add("Claude", 0x53, "claude.exe")
        sheet = self.add("Budget 2026.xlsx - Excel", 0x54, "EXCEL.EXE")
        pad = self.add("notes.txt - Notepad", 0x55, "notepad.exe")
        out = A._act_close_all_windows_except("Claude and the budget")
        self.assertFalse(sheet.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude and the "
                              "budget.")

    def test_keep_names_parse(self):
        self.assertEqual(A._keep_names("Claude"), ["Claude"])
        # Names are kept as said (the summary repeats them) and matched by
        # the app: "the Claude app" is matched as "Claude".
        self.assertEqual(A._keep_names("for the Claude app open"),
                         ["the Claude app"])
        self.assertEqual(A._keep_key("the Claude app"), "Claude")
        self.assertEqual(A._keep_names("Claude, Spotify and Chrome window"),
                         ["Claude", "Spotify", "Chrome window"])
        self.assertEqual(A._keep_names("Claude and the claude app"),
                         ["Claude"])
        for front in ("the current window", "this one", "this window",
                      "the one I'm using", "what I'm working on"):
            with self.subTest(front=front):
                self.assertEqual(A._keep_names(front), [A._KEEP_FRONT])
        self.assertEqual(A._keep_names("  "), [])


class NothingToAnchorOnTests(_Base):
    def test_a_name_that_matches_nothing_closes_nothing(self):
        pad = self.add("notes.txt - Notepad", 0x30, "notepad.exe")
        files = self.add("Downloads - File Explorer", 0x31, "explorer.exe")
        out = A._act_close_all_windows_except("Clod")
        self.assertFalse(pad.closed or files.closed)
        line = fm.terminal_failure_text(out)
        self.assertIn("I don't see a Clod window to keep", line)
        self.assertIn("I've closed nothing", line)

    def test_an_unreadable_front_window_closes_nothing(self):
        pad = self.add("notes.txt - Notepad", 0x32, "notepad.exe")
        out = A._act_close_all_windows_except("this one")
        self.assertFalse(pad.closed)
        self.assertIn("which window is in front",
                      fm.terminal_failure_text(out))

    def test_an_empty_argument_is_the_usage_line(self):
        self.assertIn("format: close_all_windows_except",
                      A._act_close_all_windows_except(""))
        self.assertIn("format: minimize_all_windows_except",
                      A._act_minimize_all_windows_except("   "))

    def test_keeping_a_jarvis_window_is_not_a_miss(self):
        # JARVIS's own windows are always kept, so "except the HUD" keeps it
        # and closes the owner's windows.
        _c, files, hud, *_rest = self.live_desktop()
        out = A._act_close_all_windows_except("the HUD")
        self.assertTrue(files.closed)
        self.assertFalse(hud.closed or hud.minimized)
        self.assertTrue(out.startswith("Closed 3 windows, sir; kept the "
                                       "HUD."), out)


class SafetyTests(_Base):
    def test_a_window_that_may_host_jarvis_is_left_open(self):
        self.add("Claude", 0x40, "claude.exe")
        shell = self.add("Windows PowerShell", 0x41, "powershell.exe")
        pad = self.add("notes.txt - Notepad", 0x42, "notepad.exe")
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(shell.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude. I left "
                              "Windows PowerShell open; it may be running me.")

    def test_a_browser_tab_that_mentions_python_is_not_a_host(self):
        self.add("Claude", 0x4D, "claude.exe")
        tab = self.add("Learn Python - YouTube - Google Chrome", 0x4E,
                       "chrome.exe")
        ide = self.add("bobert_companion.py - Visual Studio Code", 0x4F,
                       "Code.exe")
        out = A._act_close_all_windows_except("Claude")
        self.assertTrue(tab.closed)
        self.assertFalse(ide.closed)
        self.assertIn("I left Visual Studio Code open", out)

    def test_terminals_are_left_open_whatever_their_title(self):
        # Review 2026-10-03: WM_CLOSE on a console or terminal window ends
        # every program running in it with no chance to save - a kill, not
        # the X-button close this action promises - and the title-only host
        # rule missed a terminal titled by what runs in it (a coding
        # session, a build, an ssh login).
        self.add("Claude", 0x60, "claude.exe")
        term = self.add("* Fix the build", 0x61, "WindowsTerminal.exe")
        self.classes[0x61] = "cascadia_hosting_window_class"
        con = self.add("ssh build-box", 0x62, "ssh.exe")
        self.classes[0x62] = "consolewindowclass"
        pad = self.add("notes.txt - Notepad", 0x63, "notepad.exe")
        # The "Close N windows?" count never includes them either.
        self.assertEqual(A._close_all_windows_except_preview("Claude"),
                         ["notes.txt - Notepad"])
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(term.closed or con.closed)
        self.assertTrue(pad.closed)
        self.assertEqual(out, "Closed 1 window, sir; kept Claude. I left "
                              "Windows Terminal and ssh build-box open; they "
                              "may be running me.")

    def test_a_terminal_named_in_the_keep_is_simply_kept(self):
        self.add("Claude", 0x64, "claude.exe")
        term = self.add("Claude Code", 0x65, "WindowsTerminal.exe")
        self.classes[0x65] = "cascadia_hosting_window_class"
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(term.closed)
        self.assertEqual(out, "Nothing else to close, sir; kept Claude.")

    def test_elevated_windows_are_reported_once_with_the_terminal_line(self):
        self.add("Claude", 0x43, "claude.exe")
        pad = self.add("notes.txt - Notepad", 0x44, "notepad.exe")
        tm = self.add("Task Manager", 0x45, "Taskmgr.exe", raises=_denied())
        reg = self.add("Registry Editor", 0x46, "regedit.exe",
                       raises=_denied())
        out = A._act_close_all_windows_except("Claude")
        self.assertTrue(pad.closed)
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX), out)
        line = fm.terminal_failure_text(out)
        self.assertEqual(line.count("runs as administrator"), 1, line)
        self.assertIn("I closed the other window, sir, but Task Manager runs "
                      "as administrator", line)
        self.assertIn("I kept Claude.", line)
        self.assertFalse(tm.closed or reg.closed)

    def test_an_ordinary_close_error_is_terminal_too(self):
        self.add("Claude", 0x47, "claude.exe")
        self.add("Stuck App", 0x48, "stuck.exe", raises=Exception("1400"))
        out = A._act_close_all_windows_except("Claude")
        line = fm.terminal_failure_text(out)
        self.assertIn("Stuck App wouldn't close.", line)

    def test_the_minimize_variant_skips_already_minimized_windows(self):
        self.add("Claude", 0x49, "claude.exe")
        low = self.add("notes.txt - Notepad", 0x4A, "notepad.exe",
                       minimized=True)
        up = self.add("Downloads - File Explorer", 0x4B, "explorer.exe")
        out = A._act_minimize_all_windows_except("Claude")
        self.assertFalse(low.minimized)
        self.assertTrue(up.minimized)
        self.assertEqual(out, "Minimized 1 window, sir; kept Claude.")

    def test_nothing_else_open(self):
        self.add("Claude", 0x4C, "claude.exe")
        self.assertEqual(A._act_close_all_windows_except("Claude"),
                         "Nothing else to close, sir; kept Claude.")


class PreviewTests(_Base):
    def test_the_preview_is_what_would_close(self):
        self.live_desktop()
        self.assertEqual(sorted(A._close_all_windows_except_preview("Claude")),
                         ["Downloads - File Explorer", "Media Player"])
        self.assertFalse(any(w.closed for w in self.windows))

    def test_the_preview_of_a_refusal_is_empty(self):
        self.live_desktop()
        self.assertEqual(A._close_all_windows_except_preview("Clod"), [])
        self.assertEqual(A._close_all_windows_except_preview(""), [])


if __name__ == "__main__":
    unittest.main()
