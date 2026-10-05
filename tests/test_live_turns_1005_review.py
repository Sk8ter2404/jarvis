"""Review of claude/live-turn-fixes-1005, light tier: core.actions against a
fake desktop (tests/test_live_turns_1005_windows._Desk). Nothing real is
listed, closed or focused. The monolith replays of the same findings are in
tests/monolith/test_monolith_live_turns_1005_review.py.

  * a close BY NAME closes the windows that ARE that name
    (_resolve_named_close): the app by its executable, else a title that
    names it as itself; never a terminal or a folder a title word swept in;
    an explorer.exe window that is not a folder window is not "File
    Explorer"; the pushback counts what the close closes;
  * the "did you mean" also asks when another keep name matched;
  * "Google Chrome stays open for its Claude page" - only for a browser
    window kept by its page, never for an app's own title (PowerPoint, the
    new Outlook);
  * "you forgot X" spares what the bulk close kept by name or spared, and
    closes a browser window it kept only for a page when nothing else
    answers.

    python -m unittest tests.test_live_turns_1005_review
"""
from __future__ import annotations

from unittest import mock

import core.actions as A
from core import failure_markers as fm
from tests.test_live_turns_1005_windows import _Desk


class NamedCloseResolutionTests(_Desk):
    def _claude_desk(self):
        self.live_desktop()
        self.term = self.add("✳ Claude Code", 0x90, "WindowsTerminal.exe")
        self.folder = self.add("Claude Code - File Explorer", 0x91,
                               "explorer.exe")

    def test_close_claude_is_the_app_alone(self):
        self._claude_desk()
        out = A._act_close_window("Claude")
        self.assertEqual(out, "closed: Claude")
        self.assertFalse(self.term.closed or self.folder.closed
                         or self.web.closed)

    def test_close_code_is_vs_code_alone(self):
        self._claude_desk()
        vsc = self.add("app.py - proj - Visual Studio Code", 0x92, "Code.exe")
        A._act_close_window("Code")
        self.assertTrue(vsc.closed)
        self.assertFalse(self.term.closed or self.folder.closed)

    def test_the_route_bar_and_the_close_resolve_the_same_windows(self):
        self._claude_desk()
        self.windows.remove(self.claude)
        self.windows.remove(self.web)
        # Only the terminal names "Claude" now - by a title word.
        self.assertFalse(A._names_open_window("Claude"))
        self.assertEqual(A._named_close_state("Claude"), "named")
        out = A._act_close_window("Claude")
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX))
        self.assertFalse(self.term.closed or self.folder.closed)

    def test_file_explorer_is_every_folder_window(self):
        self._claude_desk()
        self.assertEqual(A._act_close_window("File Explorer"),
                         "closed: Downloads - File Explorer, Claude Code - "
                         "File Explorer")

    def test_a_non_folder_explorer_window_is_not_file_explorer(self):
        props = self.add("notes.txt Properties", 0x93, "explorer.exe")
        self.classes[0x93] = "#32770"
        folder = self.add("Pictures", 0x94, "explorer.exe")
        self.classes[0x94] = "cabinetwclass"
        A._act_close_window("File Explorer")
        self.assertTrue(folder.closed)
        self.assertFalse(props.closed)

    def test_the_pushback_count_includes_process_matches(self):
        for i in range(3):
            self.add(f"Chat {i}", 0xA0 + i, "claude.exe")
        self.assertEqual(sorted(A._close_window_preview("Claude")),
                         ["Chat 0", "Chat 1", "Chat 2"])

    def test_a_title_query_still_closes_by_title(self):
        self.live_desktop()
        self.assertEqual(A._act_close_window("notes.txt - Notepad"),
                         "closed: notes.txt - Notepad")
        self.assertIsNone(A._resolve_named_close(self.bc, "notes.txt - "
                                                          "Notepad"))

    def test_a_word_inside_titles_only_is_loose(self):
        self.live_desktop()
        self.assertEqual(A._named_close_state("Budget"), "loose")
        self.assertEqual(A._named_close_state("Zebra"), "none")
        self.assertEqual(A._named_close_state("Notepad"), "named")


class PartlyMisheardKeepTests(_Desk):
    def test_one_name_matched_one_misheard_asks(self):
        self.live_desktop()
        got = A._close_name_question("close_all_windows_except",
                                     "Excel, Claw")
        self.assertIsNotNone(got)
        self.assertEqual(got[0], "I don't see a Claw window, sir. Did you "
                                 "mean Claude? Say yes and I'll close the "
                                 "other 4 windows.")
        self.assertEqual(got[2], "Excel, Claude")
        self.assertFalse(any(w.closed for w in self.windows))

    def test_an_unmatched_name_with_no_suggestion_asks_nothing(self):
        self.live_desktop()
        self.assertIsNone(A._close_name_question("close_all_windows_except",
                                                 "Excel, Zebra"))


class TitleOnlyNoteTests(_Desk):
    def _pad(self):
        return self.add("notes.txt - Notepad", 0xB9, "notepad.exe")

    def test_an_apps_own_title_gets_no_note(self):
        for title, exe, keep in (("Deck1 - PowerPoint", "POWERPNT.EXE",
                                  "PowerPoint"),
                                 ("Inbox - someone - Outlook", "olk.exe",
                                  "Outlook")):
            with self.subTest(keep=keep):
                self.windows.clear()
                self.add(title, 0xB0, exe)
                pad = self._pad()
                out = A._act_close_all_windows_except(keep)
                self.assertTrue(pad.closed)
                self.assertEqual(out, f"Closed 1 window, sir; kept {keep}.")

    def test_a_web_page_keep_names_its_browser(self):
        self.add("Lofi mix - YouTube - Google Chrome", 0xB2, "chrome.exe")
        self._pad()
        out = A._act_close_all_windows_except("YouTube")
        self.assertEqual(out, "Closed 1 window, sir; kept YouTube. Google "
                              "Chrome stays open for its YouTube page.")
        self.assertNotIn("only because", out)


class ForgotAfterBulkTests(_Desk):
    def _forgot_turn(self, said):
        self.bc._turn_user_text.return_value = said

    def test_forgot_chrome_spares_the_window_kept_for_youtube(self):
        yt = self.add("Lo-fi beats - YouTube - Google Chrome", 0xC0,
                      "chrome.exe")
        mail = self.add("Inbox - Mail - Google Chrome", 0xC1, "chrome.exe")
        self.add("Claude", 0xC2, "claude.exe")
        with mock.patch.object(mail, "close", lambda: None):
            A._act_close_all_windows_except("Claude, YouTube")
        self._forgot_turn("Jarvis, you forgot about Chrome.")
        self.assertTrue(A._names_open_window("Chrome",
                                             A._last_bulk_close()))
        A._act_close_window("Chrome")
        self.assertTrue(mail.closed)
        self.assertFalse(yt.closed)

    def test_forgot_the_browser_kept_only_for_a_page_closes_it(self):
        web = self.add("Claude - Google Chrome", 0xC3, "chrome.exe")
        self.add("notes.txt - Notepad", 0xC4, "notepad.exe")
        out = A._act_close_all_windows_except("Claude")
        self.assertIn("Google Chrome stays open for its Claude page.", out)
        self._forgot_turn("Jarvis, you forgot about Google Chrome.")
        self.assertTrue(A._names_open_window("Google Chrome",
                                             A._last_bulk_close()))
        A._act_close_window("Google Chrome")
        self.assertTrue(web.closed)

    def test_forgot_a_spared_terminal_is_said_and_left(self):
        term = self.add("npm run dev", 0xC5, "mintty.exe")
        self.add("Claude", 0xC6, "claude.exe")
        self.add("notes.txt - Notepad", 0xC7, "notepad.exe")
        A._act_close_all_windows_except("Claude")
        self.assertFalse(term.closed)
        self._forgot_turn("Jarvis, you forgot about npm run dev.")
        self.assertFalse(A._names_open_window("npm run dev",
                                              A._last_bulk_close()))
        out = A._act_close_window("npm run dev")
        self.assertFalse(term.closed)
        self.assertEqual(fm.terminal_failure_text(out),
                         "I left npm run dev open on purpose, sir: closing "
                         "it could end whatever runs in it - me included.")

    def test_forgot_the_kept_app_is_said_and_left(self):
        app = self.add("Claude", 0xC8, "claude.exe")
        self.add("notes.txt - Notepad", 0xC9, "notepad.exe")
        A._act_close_all_windows_except("Claude")
        self._forgot_turn("Jarvis, you forgot about Claude.")
        out = A._act_close_window("Claude")
        self.assertFalse(app.closed)
        self.assertEqual(fm.terminal_failure_text(out),
                         "You asked me to keep Claude open, sir, so I've "
                         "left it.")

    def test_outside_a_forgot_turn_nothing_is_spared(self):
        app = self.add("Claude", 0xCA, "claude.exe")
        self.add("notes.txt - Notepad", 0xCB, "notepad.exe")
        A._act_close_all_windows_except("Claude")
        self.bc._turn_user_text.return_value = "Jarvis, close Claude."
        A._act_close_window("Claude")
        self.assertTrue(app.closed)


if __name__ == "__main__":   # pragma: no cover
    import unittest
    unittest.main()
