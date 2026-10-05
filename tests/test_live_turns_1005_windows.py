"""The 2026-10-05 window turns, against core.actions with a fake desktop.

Live (session_2026-10-05_00-21-55.log, paraphrased; titles generic):
  00:25:02  "close everything except for Claw" (Parakeet's "Claude") - kept
            nothing, closed nothing, offered nothing.
  00:25:12  "close everything except for Claude" - "Closed 5 windows, sir;
            kept Claude." - and a Chrome window stayed open: its tab was a
            Claude page, and the keep rule kept any window whose title named
            Claude although the Claude APP was running.
  00:25:27  "you forgot Google Chrome" - read as one more name to keep.
  00:24:21+ "close File Explorer" / "close Google Chrome" - a named close
            (routing: tests/test_named_close_route.py and the monolith
            replay tests/monolith/test_monolith_live_turns_1005.py).

Pinned here: the running app's own windows win the keep; a window kept only
by its title is reported; a misheard keep / close name gets ONE "did you
mean"; close_window finds the app the owner names by its process; focus acts
on a misheard name and says so; the last bulk close is remembered for "you
forgot X". Nothing real is listed, closed or focused.

    python -m unittest tests.test_live_turns_1005_windows
"""
from __future__ import annotations

from unittest import mock

import core.actions as A
from core import failure_markers as fm
from core import window_scope as ws
from tests.test_close_all_windows_except import _Base, _Win


class _FWin(_Win):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.activated = False

    def activate(self):
        self.activated = True


class _Desk(_Base):
    """The light _Base desktop with the monolith's window seams wired to
    the REAL core.actions / core.window_scope lookups over it."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        bc._find_windows_by_title.side_effect = (
            lambda q: ws.matching_windows(self.windows, q, ""))
        # (getattr: on a tree without the 2026-10-05 fix the tests fail on
        # behaviour or on the missing helper they call, not in set-up.)
        for seam in ("_find_app_windows", "_window_name_suggestion",
                     "_close_all_windows_except_preview",
                     "_close_all_windows_except_suggestion"):
            getattr(bc, seam).side_effect = getattr(A, seam, None)
        bc._strip_bidi_and_nbsp.side_effect = lambda s: s
        bc._BROWSER_CHROME_SUFFIXES = (" - google chrome",)
        bulk = getattr(A, "_LAST_BULK_CLOSE", None)
        if bulk is not None:
            bulk[0] = None
            self.addCleanup(bulk.__setitem__, 0, None)

    def add(self, title, hwnd, exe=None, **kw):
        w = _FWin(title, hwnd, **kw)
        self.windows.append(w)
        if exe:
            self.procs[hwnd] = exe
        return w

    def live_desktop(self):
        """The 00:25:12 desktop: the Claude app, a Chrome window whose tab
        is a Claude page, and five more of the owner's windows."""
        self.claude = self.add("Claude", 0x80, "claude.exe")
        self.web = self.add("Claude - Google Chrome", 0x81, "chrome.exe")
        self.files = self.add("Downloads - File Explorer", 0x82,
                              "explorer.exe")
        self.media = self.add("Media Player", 0x83,
                              "ApplicationFrameHost.exe")
        self.pad = self.add("notes.txt - Notepad", 0x84, "notepad.exe")
        self.sheet = self.add("Budget.xlsx - Excel", 0x85, "EXCEL.EXE")


class KeepTheRunningAppTests(_Desk):
    def test_the_live_close_all_closes_the_chrome_window_too(self):
        self.live_desktop()
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(self.claude.closed)
        for w in (self.web, self.files, self.media, self.pad, self.sheet):
            self.assertTrue(w.closed, w.title)
        self.assertEqual(out, "Closed 5 windows, sir; kept Claude.")

    def test_with_no_claude_app_the_titled_window_is_kept_and_said(self):
        self.live_desktop()
        self.windows.remove(self.claude)
        out = A._act_close_all_windows_except("Claude")
        self.assertFalse(self.web.closed)
        self.assertTrue(self.files.closed and self.pad.closed)
        self.assertEqual(out, "Closed 4 windows, sir; kept Claude. Google "
                              "Chrome stays open for its Claude page.")

    def test_the_minimize_variant_follows_the_same_keep(self):
        self.live_desktop()
        A._act_minimize_all_windows_except("Claude")
        self.assertTrue(self.web.minimized)
        self.assertFalse(self.claude.minimized)

    def test_the_last_bulk_close_is_remembered_briefly(self):
        self.live_desktop()
        A._act_close_all_windows_except("Claude")
        rec = A._last_bulk_close()
        self.assertIsNotNone(rec)
        self.assertEqual(rec.keep, ("Claude",))
        self.assertIn("Claude - Google Chrome", rec.closed)
        with mock.patch.object(A.time, "monotonic",
                               return_value=rec.at + A.BULK_CLOSE_FOLLOWUP_S
                               + 1.0):
            self.assertIsNone(A._last_bulk_close())

    def test_minimizing_is_not_a_bulk_close_to_follow_up(self):
        self.live_desktop()
        A._act_minimize_all_windows_except("Claude")
        self.assertIsNone(A._last_bulk_close())


class MisheardKeepTests(_Desk):
    def test_the_live_claw_closes_nothing_and_offers_claude(self):
        self.live_desktop()
        out = A._act_close_all_windows_except("Claw")
        self.assertFalse(any(w.closed for w in self.windows))
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX))
        self.assertEqual(fm.terminal_failure_text(out),
                         "I don't see a Claw window to keep, sir, so I've "
                         "closed nothing. Did you mean Claude?")

    def test_the_corrected_command_and_its_count(self):
        self.live_desktop()
        fix = A._close_all_windows_except_suggestion("Claw")
        self.assertIsNotNone(fix)
        fixed_arg, closing, heard, sugg = fix
        self.assertEqual((fixed_arg, heard, sugg), ("Claude", "Claw",
                                                    "Claude"))
        self.assertEqual(len(closing), 5)
        self.assertIn("Claude - Google Chrome", closing)
        self.assertFalse(any(w.closed for w in self.windows))

    def test_the_question_queues_the_corrected_command(self):
        self.live_desktop()
        got = A._close_name_question("close_all_windows_except", "Claw")
        self.assertEqual(got[0], "I don't see a Claw window, sir. Did you "
                                 "mean Claude? Say yes and I'll close the "
                                 "other 5 windows.")
        self.assertEqual(got[2], "Claude")

    def test_a_name_that_matches_asks_nothing(self):
        self.live_desktop()
        self.assertIsNone(A._close_name_question("close_all_windows_except",
                                                 "Claude"))

    def test_a_name_like_nothing_open_asks_nothing(self):
        self.live_desktop()
        self.assertIsNone(A._close_all_windows_except_suggestion("Zebra"))
        self.assertIsNone(A._close_name_question("close_all_windows_except",
                                                 "Zebra"))
        out = A._act_close_all_windows_except("Zebra")
        self.assertNotIn("Did you mean", out)


class NamedCloseWindowTests(_Desk):
    def test_a_named_app_is_found_by_its_process(self):
        app = self.add("New conversation", 0x90, "claude.exe")
        pad = self.add("notes.txt - Notepad", 0x91, "notepad.exe")
        out = A._act_close_window("Claude")
        self.assertTrue(app.closed)
        self.assertFalse(pad.closed)
        self.assertEqual(out, "closed: New conversation")

    def test_file_explorer_and_chrome_by_name(self):
        self.live_desktop()
        self.assertEqual(A._act_close_window("File Explorer"),
                         "closed: Downloads - File Explorer")
        self.assertTrue(self.files.closed)
        self.assertEqual(A._act_close_window("Google Chrome"),
                         "closed: Claude - Google Chrome")
        self.assertTrue(self.web.closed)
        self.assertFalse(self.claude.closed)

    def test_a_terminal_closes_by_name_only_when_it_is_the_name(self):
        # Review 2026-10-05: a terminal is closed by name only when the
        # owner named the terminal itself - never one a title word swept in.
        term = self.add("build log", 0x92, "WindowsTerminal.exe")
        self.classes[0x92] = "cascadia_hosting_window_class"
        self.assertEqual(A._find_app_windows("Windows Terminal"), [term])
        # A word of its title is not its name: left open, and said.
        out = A._act_close_window("build")
        self.assertFalse(term.closed)
        self.assertEqual(fm.terminal_failure_text(out),
                         "build log is a terminal, sir; closing it would end "
                         "whatever runs in it, so I've left it open.")
        self.assertFalse(A._names_open_window("build"))
        # Its whole title is.
        self.assertEqual(A._act_close_window("build log"),
                         "closed: build log")
        self.assertTrue(term.closed)

    def test_the_route_bar_is_a_window_named_as_itself(self):
        # _names_open_window decides whether "close <name>" is routed to
        # close_window without the brain: a running app, or a title that
        # names the window itself - never a word inside a page or document.
        self.live_desktop()
        self.add("Doorbell camera - Google Chrome", 0x94, "chrome.exe")
        self.add("Home - YouTube - Google Chrome", 0x95, "chrome.exe")
        for name in ("File Explorer", "file explorer", "Google Chrome",
                     "Chrome", "Claude", "Notepad", "YouTube",
                     "Media Player", "the Claude app"):
            with self.subTest(name=name):
                self.assertTrue(A._names_open_window(name))
        for name in ("door", "garage door", "budget", "Downloads x",
                     "Spotify", ""):
            with self.subTest(name=name):
                self.assertFalse(A._names_open_window(name))
        self.assertFalse(any(w.closed for w in self.windows))

    def test_a_misheard_name_closes_nothing_and_says_what_it_heard(self):
        self.live_desktop()
        out = A._act_close_window("Claw")
        self.assertFalse(any(w.closed for w in self.windows))
        self.assertTrue(out.startswith("no window matching 'Claw'"), out)
        self.assertIn("did you mean Claude?", out)

    def test_the_pushback_question_for_a_single_close(self):
        self.live_desktop()
        got = A._close_name_question("close_window", "Claw")
        self.assertEqual(got[0], "I don't see a Claw window, sir. Did you "
                                 "mean Claude?")
        self.assertEqual(got[2], "Claude")
        self.assertIsNone(A._close_name_question("close_window", "Notepad"))
        self.assertIsNone(A._close_name_question("close_window",
                                                 "taskmgr.exe"))

    def test_a_failing_lookup_asks_nothing(self):
        self.live_desktop()
        self.bc._find_windows_by_title.side_effect = RuntimeError("down")
        self.assertIsNone(A._close_name_question("close_window", "Claw"))


class FocusTests(_Desk):
    def test_focus_acts_on_a_misheard_name_and_says_so(self):
        self.live_desktop()
        out = A._act_focus_window("Claw")
        self.assertTrue(self.claude.activated)
        self.assertEqual(out, "focused 'Claude' (I heard 'Claw')")

    def test_focus_finds_an_app_by_its_process(self):
        app = self.add("New conversation", 0x93, "claude.exe")
        out = A._act_focus_window("Claude")
        self.assertTrue(app.activated)
        self.assertEqual(out, "focused 'New conversation'")

    def test_nothing_like_it_is_still_no_match(self):
        self.live_desktop()
        self.assertEqual(A._act_focus_window("Zebra"),
                         "no window matching 'Zebra'")


if __name__ == "__main__":   # pragma: no cover
    import unittest
    unittest.main()
