"""Review of claude/live-turn-fixes-1005: replays of each confirmed finding
through the REAL monolith dispatch, on the same fake desktop as
tests/monolith/test_monolith_live_turns_1005.py (pygetwindow is a fake
module, process names come from a table, every window is a stub; nothing
real is listed, closed, clicked or typed; no LLM, audio or network).

  R1  a close BY NAME closed every title that mentioned the name: "close
      Claude" took the Claude Code terminal and a "Claude Code" folder, and
      "close settings" took JARVIS's own Settings window.
  R2  "you forgot X" closed windows the bulk close had just KEPT, and a
      terminal it had deliberately left open.
  R3  the close_last_opened rewrite broke "close what you opened" and
      closed an unrelated "Song lyrics draft" for "close the song".
  R4  "close everything except Excel and Claw" closed Claude without asking.
  R5-R8  the sign-in guard: natural rewordings bypassed the owner-asked
      check; typing + Enter, and a refused click + Enter, still signed in;
      find_on_screen + a coordinate click got through; ordinary clicks were
      refused with a false "sign-in page" line.
  R9  the live 00:24:21-00:25:42 sequence end to end, the bulk close in it.

The owner's words are paraphrased; the account is a placeholder.

    python -m unittest tests.monolith.test_monolith_live_turns_1005_review
"""
from __future__ import annotations

import contextlib
import io
import time

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base
from tests.monolith.test_monolith_live_turns_1005 import (
    ACCOUNT_ENTRY, CHOOSER_LOOK, LIVE_CLOSE_THAT, PAGE, _Desk, _Win,
    grounded_stand_in)

_PROCS = {0x70: "claude.exe", 0x71: "chrome.exe", 0x72: "explorer.exe",
          0x73: "ApplicationFrameHost.exe", 0x74: "notepad.exe",
          0x75: "EXCEL.EXE", 0x76: "pythonw.exe"}


class _Desk2(_Desk):
    """_Desk with a process table tests can add to."""

    def setUp(self):
        # The real pushback, before _Base stubs it for every test.
        self._pushback = self.bc._jarvis_pushback
        super().setUp()
        self.procs = dict(_PROCS)
        self.A._window_process_name.side_effect = (
            lambda w: self.procs.get(w._hWnd))

    def add(self, title, hwnd, proc):
        w = _Win(title, hwnd)
        self.wins.append(w)
        self.procs[hwnd] = proc
        return w

    def real_pushback(self, enabled=True):
        bc = self.bc
        self._p(bc, "_jarvis_pushback", self._pushback)
        self._p(bc, "PUSHBACK_ENABLED", enabled)
        self._p(bc, "PUSHBACK_MAX_CLOSE_WINDOWS", 5)
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_confirmation_at", [0.0])

    def yes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc.handle_confirmation_response("Jarvis, yes.")


# ════════════════════════════════════════════════════════════════════════
#  R1 - a close by name closes what IS that name
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class NamedCloseIsTheNameTests(_Desk2):
    def setUp(self):
        super().setUp()
        self.term = self.add("✳ Claude Code", 0x90, "WindowsTerminal.exe")
        self.folder = self.add("Claude Code - File Explorer", 0x91,
                               "explorer.exe")

    def test_close_claude_closes_the_app_alone(self):
        printed = self._turn("Jarvis, close Claude.", LIVE_CLOSE_THAT)
        self.llm.assert_not_called()
        self.assertIn("[route] named close -> close_window", printed)
        self.assertTrue(self.claude.closed)
        for w in (self.term, self.folder, self.web, self.files, self.hud):
            self.assertFalse(w.closed, w.title)

    def test_close_code_closes_vs_code_alone(self):
        vsc = self.add("app.py - proj - Visual Studio Code", 0x92, "Code.exe")
        self._turn("Jarvis, close Code.", LIVE_CLOSE_THAT)
        self.llm.assert_not_called()
        self.assertTrue(vsc.closed)
        self.assertFalse(self.term.closed or self.folder.closed
                         or self.claude.closed)

    def test_the_brains_close_window_resolves_the_same_way(self):
        # Not routed (a stubbed route table is irrelevant here): the brain
        # writes close_window itself, and the action resolves by name.
        self._turn("Jarvis, shut Claude down for me.",
                   "[intent:confirmation] Very good, sir. "
                   "[ACTION: close_window, Claude]")
        self.assertTrue(self.claude.closed)
        self.assertFalse(self.term.closed or self.folder.closed
                         or self.web.closed)

    def test_a_terminal_named_only_by_a_title_word_is_left_and_said(self):
        # No Claude app: the Chrome page and the terminal name Claude by
        # title; the page's tab may close, the terminal is left and named.
        self.wins.remove(self.claude)
        self._turn("Jarvis, close Claude Code.",
                             "[intent:confirmation] Very good, sir. "
                             "[ACTION: close_window, Claude Code]")
        self.assertTrue(self.term.closed, "named exactly: it is the name")
        self.term.closed = False
        self._turn("Jarvis, close Claude.",
                   "[intent:confirmation] Very good, sir. "
                   "[ACTION: close_window, Claude]")
        self.assertFalse(self.term.closed)
        self.assertFalse(self.folder.closed)
        out = self.bc.ACTIONS["close_window"]("Claude")
        self.assertIn("left 'Claude Code' open: a terminal", out)

    def test_only_a_swept_in_terminal_is_an_honest_terminal_line(self):
        self.wins.remove(self.claude)
        self.wins.remove(self.web)
        self._turn("Jarvis, close Claude.",
                   "[intent:confirmation] Very good, sir. "
                   "[ACTION: close_window, Claude]")
        self.assertFalse(self.term.closed or self.folder.closed)
        self.assertTrue(any(
            "Claude Code is a terminal, sir; closing it would end whatever "
            "runs in it, so I've left it open." in s for s in self.spoken),
            self.spoken)

    def test_close_settings_is_windows_settings_not_jarvis_settings(self):
        from core import window_scope as ws
        st = self.add("Settings", 0x93, "ApplicationFrameHost.exe")
        js = self.add("JARVIS Settings", 0x94, "pythonw.exe")
        ws.probe.side_effect = lambda w: ws.WindowFacts(
            w._hWnd, {0x76: 9101, 0x94: 9101}.get(w._hWnd), "", False)
        self._turn("Jarvis, close settings.", LIVE_CLOSE_THAT)
        self.llm.assert_not_called()
        self.assertTrue(st.closed)
        self.assertFalse(js.closed)

    def test_the_pushback_counts_what_the_close_closes(self):
        self.real_pushback()
        for i in range(6):
            self.add(f"file{i}.py - proj - Visual Studio Code", 0xA0 + i,
                     "Code.exe")
        got = self.bc._jarvis_pushback("close_window", "Code")
        self.assertIsNotNone(got)
        self.assertIn("close 6 windows", got[0])


# ════════════════════════════════════════════════════════════════════════
#  R2 - "you forgot X" never takes what the bulk close kept or spared
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class ForgotTests(_Desk2):
    def test_you_forgot_chrome_spares_the_kept_youtube_window(self):
        yt = self.add("Lo-fi beats - YouTube - Google Chrome", 0x95,
                      "chrome.exe")
        self.wins.remove(self.web)
        mail = self.add("Inbox - Mail - Google Chrome", 0x96, "chrome.exe")
        mail.close = lambda: None     # a page that holds its window open
        self._turn("Jarvis, close every window except Claude and YouTube.",
                   "[ACTION: close_all_windows_except, Claude, YouTube]")
        self.assertFalse(yt.closed or self.claude.closed)
        mail.close = lambda: setattr(mail, "closed", True)
        printed = self._turn("Jarvis, you forgot about Chrome.",
                             "[intent:confirmation] Which one, sir?")
        self.assertIn("[route] you forgot X after a bulk close", printed)
        self.assertTrue(mail.closed)
        self.assertFalse(yt.closed, "the kept YouTube window was closed")

    def test_you_forgot_the_spared_terminal_says_why_it_stays(self):
        term = self.add("npm run dev", 0x97, "mintty.exe")
        self._turn("Jarvis, close every window except for Claude.", "")
        self.assertFalse(term.closed)
        self.spoken.clear()
        printed = self._turn("Jarvis, you forgot about npm run dev.",
                             "[intent:confirmation] Very good, sir. "
                             "[ACTION: close_window, npm run dev]")
        self.llm.assert_called_once()       # not routed: it was spared
        self.assertNotIn("[route] you forgot X", printed)
        self.assertFalse(term.closed)
        self.assertTrue(any("I left npm run dev open on purpose" in s
                            for s in self.spoken), self.spoken)

    def test_you_forgot_the_kept_app_is_the_brains_and_closes_nothing(self):
        # Mutation gap M31: "you forgot Claude" right after "except Claude".
        self._turn("Jarvis, close every window except for Claude.", "")
        self.spoken.clear()
        self._turn("Jarvis, you forgot about Claude.",
                   "[intent:confirmation] Very good, sir. "
                   "[ACTION: close_window, Claude]")
        self.llm.assert_called_once()
        self.assertFalse(self.claude.closed)
        self.assertIn("You asked me to keep Claude open, sir, so I've left "
                      "it.", self.spoken)


# ════════════════════════════════════════════════════════════════════════
#  R3 - close_last_opened stays for what JARVIS opened
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class NamelessCloseTests(_Desk2):
    def setUp(self):
        super().setUp()
        from core import opened_ledger as ol
        ol.reset()
        self.addCleanup(ol.reset)
        ol.note_opened("open_on_monitor", "notepad", hwnd=0x74,
                       kind="window")

    def test_close_what_you_opened_closes_what_jarvis_opened(self):
        for said in ("Jarvis, close what you just opened.",
                     "Jarvis, close the window you just opened.",
                     "Jarvis, close the thing you opened."):
            with self.subTest(said=said):
                self.pad.closed = False
                self.close_that_calls.clear()
                from core import opened_ledger as ol
                ol.reset()
                ol.note_opened("open_on_monitor", "notepad", hwnd=0x74,
                               kind="window")
                printed = self._turn(said, LIVE_CLOSE_THAT,
                                     ["[intent:bad_news] Hm."])
                self.assertNotIn("[named-close]", printed)
                self.assertEqual(self.close_that_calls, [""])
                self.assertTrue(self.pad.closed)

    def test_close_the_song_never_closes_a_document_about_songs(self):
        song = self.add("Song lyrics draft.txt - Notepad", 0x7B,
                        "notepad.exe")
        self._turn("Jarvis, close the song.", LIVE_CLOSE_THAT,
                   ["[intent:bad_news] Hm."])
        self.assertFalse(song.closed)
        self.assertEqual(self.close_that_calls, [""])

    def test_a_word_inside_a_title_leaves_close_last_opened(self):
        doc = self.add("Budget notes.docx - Word", 0x7C, "WINWORD.EXE")
        printed = self._turn("Jarvis, close budget.", LIVE_CLOSE_THAT,
                             ["[intent:bad_news] Hm."])
        self.assertIn("only appears inside a window title", printed)
        self.assertFalse(doc.closed)
        self.assertEqual(self.close_that_calls, [""])


# ════════════════════════════════════════════════════════════════════════
#  R4 - a partly misheard bulk close asks first
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class PartlyMisheardTests(_Desk2):
    def test_except_excel_and_claw_asks_and_the_yes_keeps_both(self):
        self.real_pushback()
        self._turn("Jarvis, close every window except Excel and Claw.",
                   "[ACTION: close_all_windows_except, Excel, Claw]")
        self.assertFalse(any(w.closed for w in self.wins), "closed on a "
                         "guess")
        self.assertIn("I don't see a Claw window, sir. Did you mean Claude? "
                      "Say yes and I'll close the other 4 windows.",
                      " ".join(self.spoken))
        self.assertEqual(list(self.bc._pending_confirmation),
                         [("close_all_windows_except", "Excel, Claude")])
        self.spoken.clear()
        self.yes()
        self.assertFalse(self.claude.closed or self.sheet.closed)
        for w in (self.web, self.files, self.media, self.pad):
            self.assertTrue(w.closed, w.title)

    def test_it_asks_with_pushback_off_too(self):
        self.real_pushback(enabled=False)
        got = self.bc._jarvis_pushback("close_all_windows_except",
                                       "Excel, Claw")
        self.assertIsNotNone(got)
        self.assertEqual(got[2], "Excel, Claude")


# ════════════════════════════════════════════════════════════════════════
#  R5-R8 - the sign-in guard
# ════════════════════════════════════════════════════════════════════════
class _SignIn(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._stub("open_url", f"opened {PAGE} — use see_screen to read what "
                               "loaded")
        self.find = self._p(bc, "find_click_target", return_value=(40, 50))
        self.click = self._p(bc, "ui_click")
        from core import grounded_click as G
        self._p(G, "run_bounded", side_effect=grounded_stand_in(bc))
        self.typed = self._p(bc, "ui_type")
        self.pressed = self._p(bc, "ui_press")
        self._p(bc, "_looks_like_shell_command", return_value=False)
        self._p(bc, "_read_focused_window",
                return_value=(None, "Claude Console - Google Chrome", None))
        from core import opened_ledger as ol
        ol.reset()
        self.addCleanup(ol.reset)

    def _go(self, said, *followups, look=CHOOSER_LOOK):
        self._stub("see_screen", look)
        return self._dispatch(said, f"[ACTION: open_url, {PAGE}]",
                              [f"[ACTION: see_screen, {PAGE}]"]
                              + list(followups))


@requires_monolith
class OwnerWordsTests(_SignIn):
    def test_rewordings_of_the_live_request_click_nothing(self):
        for said in ("Jarvis, bring that page up so I can pick my account.",
                     "Jarvis, bring that page up, don't click anything, "
                     "I'll choose my account."):
            with self.subTest(said=said):
                self.click.reset_mock()
                self._go(said, f"[ACTION: click, {ACCOUNT_ENTRY}] I've "
                               "selected your account.")
                self.click.assert_not_called()

    def test_an_exact_request_still_clicks(self):
        self._go("Jarvis, open the console and pick my account.",
                 f"[ACTION: click, {ACCOUNT_ENTRY}]")
        self.click.assert_called_once_with(40, 50)


@requires_monolith
class KeysTests(_SignIn):
    def test_typing_the_address_and_enter_on_the_chooser_page(self):
        self._go("Jarvis, bring that page up so I can sign in.",
                 "[ACTION: type, pat.example@example.com] "
                 "[ACTION: press, enter]")
        self.typed.assert_not_called()
        self.pressed.assert_not_called()

    def test_a_refused_click_then_enter_in_the_same_reply(self):
        # The look says nothing a sign-in page would: only the refused
        # account click marks the turn.
        self._go("Jarvis, bring that page up so I can sign in.",
                 f"[ACTION: click, {ACCOUNT_ENTRY}] [ACTION: press, enter]",
                 look="[local-vision] A web page with a small panel.")
        self.click.assert_not_called()
        self.pressed.assert_not_called()

    def test_enter_on_an_ordinary_page_still_works(self):
        self._dispatch("Jarvis, press enter.", "[ACTION: press, enter]")
        self.pressed.assert_called_once_with("enter")


@requires_monolith
class FindThenClickTests(_SignIn):
    def test_find_on_screen_then_its_coordinates(self):
        self._stub("find_on_screen", "found at 500,300")
        # A look that names no sign-in page: only what find_on_screen
        # looked for makes the coordinates the account entry.
        self._go("Jarvis, bring that page up so I can sign in.",
                 f"[ACTION: find_on_screen, {ACCOUNT_ENTRY}]",
                 "[ACTION: click, 500, 300]",
                 look="[local-vision] The Claude Console with a small panel "
                      "in the corner.")
        self.click.assert_not_called()

    def test_find_on_screen_of_an_ordinary_target_then_click(self):
        # The screen-vision merge routes a whole "click the X" utterance to
        # click_on_screen before the brain; this test is about the brain's
        # find_on_screen + coordinate click, so that route is off here.
        self._p(self.bc, "CLICK_ROUTE_ENABLED", False)
        self._stub("find_on_screen", "found at 500,300")
        self._dispatch("Jarvis, click the play button.",
                       "[ACTION: find_on_screen, the play button]",
                       ["[ACTION: click, 500, 300]"])
        self.click.assert_called_once_with(500, 300, "left")

    def test_an_alias_of_a_look_feeds_the_guard(self):
        # recall_screen / last_screen are one handler with previous_screen.
        self.assertTrue(self.bc._is_screen_look_action("last_screen"))
        self.assertTrue(self.bc._is_screen_look_action("recall_screen"))
        self.assertTrue(self.bc._is_screen_look_action("previous_screen"))
        self.assertFalse(self.bc._is_screen_look_action("list_windows"))


@requires_monolith
class PageByTitleTests(_SignIn):
    def test_a_google_sign_in_window_title_refuses_the_click(self):
        # The likelier live signal: the window in front IS the Google
        # sign-in page, whatever the vision model says about it.
        self._p(self.bc, "_read_focused_window", return_value=(
            None, "Sign in - Google Accounts - Google Chrome", None))
        self._go("Jarvis, bring that page up so I can sign in.",
                 "[ACTION: click, the blue Continue button]",
                 look="[local-vision] The browser window is not showing the "
                      "console page.")
        self.find.assert_not_called()
        self.click.assert_not_called()
        from core.auth_guard import READY_LINE
        self.assertIn(READY_LINE, self.spoken)


@requires_monolith
class OverBlockTests(_SignIn):
    def test_a_cookie_consent_click_goes_ahead(self):
        self._dispatch("Jarvis, open YouTube and accept the cookies.",
                       "[ACTION: click, Accept all button on the cookie "
                       "consent dialog]")
        self.click.assert_called_once_with(40, 50)

    def test_a_stale_login_address_blocks_nothing(self):
        from core import opened_ledger as ol
        ol.note_opened("open_url", "https://github.com/login",
                       now=time.time() - 240)
        self._p(self.bc, "_read_focused_window",
                return_value=(None, "Lo-fi - YouTube - Google Chrome", None))
        self._dispatch("Jarvis, play the first video.",
                       "[ACTION: click, first video thumbnail]")
        self.click.assert_called_once_with(40, 50)

    def test_a_sign_in_button_on_an_ordinary_page_says_so_honestly(self):
        from core.auth_guard import CONTROL_LINE, READY_LINE
        self._p(self.bc, "_read_focused_window",
                return_value=(None, "Home - YouTube - Google Chrome", None))
        self._dispatch("Jarvis, play something.",
                       "[ACTION: click, the Sign in button]")
        self.click.assert_not_called()
        self.assertIn(CONTROL_LINE, self.spoken)
        self.assertNotIn(READY_LINE, self.spoken)


# ════════════════════════════════════════════════════════════════════════
#  R9 - the live 00:24:21-00:25:42 sequence, the bulk close in it
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class LiveSequenceTests(_Desk2):
    def test_the_whole_sequence(self):
        self.real_pushback()
        more = self.add("Pictures - File Explorer", 0x77, "explorer.exe")
        # 00:24:21 - 00:24:46: File Explorer, three times. The first close
        # takes both folder windows; the repeats find none and say so.
        for said in ("Jarvis, please close out File Explorer 2.",
                     "Jarvis, close the file explorer.",
                     "Jarvis Close File Explorer now."):
            self._turn(said, LIVE_CLOSE_THAT,
                       ["[intent:bad_news] No File Explorer window is open, "
                        "sir."])
        self.assertTrue(self.files.closed and more.closed)
        self.assertEqual(self.close_that_calls, [])
        # 00:25:02: "Claw" - asked, nothing closed; he answers by saying the
        # whole command again instead of "yes".
        self.spoken.clear()
        self._turn("Jarvis close every window except for Claw.", "")
        self.assertTrue(any("Did you mean Claude?" in s for s in self.spoken))
        self.assertFalse(self.web.closed or self.pad.closed)
        # 00:25:12: "except Claude" - the Claude app runs, so the Chrome
        # window with the Claude page closes too (4 left to close <= 5: no
        # count question).
        self.spoken.clear()
        self._turn("Jarvis close every window except for Claude.", "")
        self.assertTrue(self.web.closed and self.pad.closed
                        and self.media.closed and self.sheet.closed)
        self.assertFalse(self.claude.closed or self.hud.closed)
        self.assertEqual(self.spoken, ["Closed 4 windows, sir; kept Claude."])
        # 00:25:26: "you forgot Google Chrome" - nothing of Chrome is left,
        # so it goes to the brain, whose keep-more reply is harmless now.
        self.spoken.clear()
        self._turn("Jarvis, you forgot about Google Chrome.",
                   "[intent:confirmation] Certainly, sir. "
                   "[ACTION: close_window, Google Chrome]",
                   ["[intent:bad_news] Chrome is already closed, sir."])
        self.assertFalse(self.claude.closed)
        # 00:25:42: "close Google Chrome" - the brain's close_last_opened is
        # rewritten; the honest answer is that no Chrome window is open.
        printed = self._turn("Jarvis close Google Chrome for me.",
                             LIVE_CLOSE_THAT,
                             ["[intent:bad_news] No Chrome window is open, "
                              "sir."])
        self.assertIn("[named-close] the owner named the window", printed)
        self.assertEqual(self.close_that_calls, [])
        for s in self.spoken:
            self.assertNotIn("no record", s)
        self.assertFalse(self.claude.closed or self.hud.closed)


if __name__ == "__main__":   # pragma: no cover
    import unittest
    unittest.main()
