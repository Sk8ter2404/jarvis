"""Replays of the 2026-10-05 live turns through the REAL monolith dispatch.

Logs: session_2026-10-05_00-21-55.log and session_2026-10-04_23-13-00.log.
The owner's words are paraphrased; window titles are generic; the account in
the sign-in replay is a placeholder. The brain's replies are the live ones
(canned), and parse_and_run_actions, the routes, the pushback, the follow-up
loop, close_window / close_all_windows_except / click and core.window_scope
all run for real against a FAKE desktop: pygetwindow is a fake module, every
window is a stub, process names come from a table, the mouse and vision are
mocks that fail the test if touched. Nothing real is listed, closed or
clicked; no LLM, audio or network.

  N  00:24:21-00:25:43  "close out File Explorer 2", "close file explorer",
     "close Google Chrome" -> the brain said [ACTION: close_last_opened]
     every time (only what JARVIS opened) and each failed. Now a named close
     is routed to close_window, and a close_last_opened written for one is
     rewritten.
  K  00:25:02-00:25:27  "close everything except for Claw" kept and closed
     nothing; "... except for Claude" kept a Chrome window whose tab was a
     Claude page; "you forgot Google Chrome" became close_all_windows_except
     (Claude, Chrome). Now "Claw" asks "Did you mean Claude?" and the yes
     closes everything but the Claude app; "you forgot X" closes X.
  S  00:14:07-00:14:21  "pull up that page so I can sign in" -> open_url,
     see_screen, then an unasked click on the owner's account entry. Now the
     click is refused at the action layer and the turn ends on "the page is
     ready for you".

    python -m unittest tests.monolith.test_monolith_live_turns_1005
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
from unittest import mock

from core import failure_markers as fm
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

# The brain's live reply to every named close (00:24:22 ... 00:25:42).
LIVE_CLOSE_THAT = "[intent:confirmation] As you wish, sir. " \
                  "[ACTION: close_last_opened]"
CLOSE_ALL = "close_all_windows_except"


class _Win:
    """A pygetwindow-like window recording close() / minimize()."""

    def __init__(self, title, hwnd):
        self.title = title
        self._hWnd = hwnd
        self.left, self.top, self.width, self.height = 0, 0, 900, 700
        self.isMinimized = False
        self.closed = self.minimized = False

    def close(self):
        self.closed = True

    def minimize(self):
        self.minimized = True

    def activate(self):
        pass


class _Desk(_Base):
    """The live desktop of 00:24-00:25, faked."""

    def setUp(self):
        super().setUp()
        import core.actions as A
        from core import window_scope as ws
        self.A = A
        self.claude = _Win("Claude", 0x70)
        self.web = _Win("Claude - Google Chrome", 0x71)
        self.files = _Win("Downloads - File Explorer", 0x72)
        self.media = _Win("Media Player", 0x73)
        self.pad = _Win("notes.txt - Notepad", 0x74)
        self.sheet = _Win("Budget.xlsx - Excel", 0x75)
        self.hud = _Win("JARVIS HUD", 0x76)
        self.wins = [self.claude, self.web, self.files, self.media, self.pad,
                     self.sheet, self.hud]
        procs = {0x70: "claude.exe", 0x71: "chrome.exe", 0x72: "explorer.exe",
                 0x73: "ApplicationFrameHost.exe", 0x74: "notepad.exe",
                 0x75: "EXCEL.EXE", 0x76: "pythonw.exe"}
        pids = {0x76: 9101}
        self._p(ws, "probe", side_effect=lambda w: ws.WindowFacts(
            w._hWnd, pids.get(w._hWnd), "", False))
        self._p(ws, "own_pids", return_value=frozenset({9100, 9101}))
        self._p(ws, "own_window_handles", return_value=frozenset())
        self._p(A, "_window_process_name",
                side_effect=lambda w: procs.get(w._hWnd))
        self._p(A, "_window_is_elevated", return_value=False)
        fake = types.SimpleNamespace(getAllWindows=lambda: [
            w for w in self.wins if not w.closed],
            getActiveWindow=lambda: None)
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)
        self._p(self.bc, "_UTTERANCE_ROUTES", [])
        # Never a real key, click or foreground read: a browser-tab close
        # checks the window in front before its Ctrl+W.
        for ui in ("ui_hotkey", "ui_press", "ui_type", "ui_click"):
            self._p(self.bc, ui)
        self._p(self.bc, "_read_focused_window",
                return_value=(None, "", None))
        # (getattr: on a tree without the 2026-10-05 fix these replays fail
        # on what JARVIS DOES, not on a missing name in their set-up.)
        bulk = getattr(A, "_LAST_BULK_CLOSE", None)
        if bulk is not None:
            bulk[0] = None
            self.addCleanup(bulk.__setitem__, 0, None)
        # close_last_opened must never be what a named close runs.
        self.close_that_calls: list = []
        real = self._actions["close_last_opened"]

        def _close_that(arg=""):
            self.close_that_calls.append(arg)
            return real(arg)
        self._actions["close_last_opened"] = _close_that

    def _turn(self, said, reply, followups=()):
        self.llm = self._p(self.bc, "get_response_with_animation",
                           return_value=reply)
        self.gfr = self._p(self.bc, "get_followup_response",
                           side_effect=list(followups) + [None] * 8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.bc._run_llm_dispatch(said)
        return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════
#  N - a named close is close_window
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class NamedCloseReplayTests(_Desk):
    def test_close_out_file_explorer_2_is_routed_to_close_window(self):
        printed = self._turn("Jarvis, please close out File Explorer "
                             "2.", LIVE_CLOSE_THAT)
        self.llm.assert_not_called()
        self.assertIn("[route] named close -> close_window", printed)
        self.assertTrue(self.files.closed)
        self.assertEqual(self.close_that_calls, [])
        for w in (self.claude, self.web, self.media, self.pad, self.sheet,
                  self.hud):
            self.assertFalse(w.closed, w.title)
        # One short acknowledgement, as the brain gives a close it got right.
        self.assertEqual(self.spoken, ["[intent:confirmation] Very good, sir."])

    def test_close_google_chrome_closes_the_chrome_window(self):
        self._turn("Jarvis close Google Chrome for me.", LIVE_CLOSE_THAT)
        self.llm.assert_not_called()
        self.assertTrue(self.web.closed)
        self.assertFalse(self.claude.closed)
        self.assertEqual(self.close_that_calls, [])

    def test_the_four_live_turns_in_order(self):
        # The live sequence, turn by turn. The first close takes every File
        # Explorer window (close_window closes each title match), so the two
        # repeats find none: the brain answers them, and its nameless close
        # is rewritten to the named one, which says honestly that nothing of
        # that name is open. close_last_opened never runs.
        more = _Win("Pictures - File Explorer", 0x77)
        self.wins.append(more)
        procs = {0x70: "claude.exe", 0x71: "chrome.exe",
                 0x72: "explorer.exe", 0x73: "ApplicationFrameHost.exe",
                 0x74: "notepad.exe", 0x75: "EXCEL.EXE",
                 0x76: "pythonw.exe", 0x77: "explorer.exe"}
        self.A._window_process_name.side_effect = (
            lambda w: procs.get(w._hWnd))
        routed = []
        for said in ("Jarvis, please close out File Explorer 2.",
                     "Jarvis, close the file explorer.",
                     "Jarvis Close File Explorer now.",
                     "Jarvis close Google Chrome for me."):
            printed = self._turn(said, LIVE_CLOSE_THAT,
                                 ["[intent:bad_news] No File Explorer window "
                                  "is open, sir."])
            routed.append(not self.llm.called)
            if not routed[-1]:
                self.assertIn("[named-close] the owner named the window",
                              printed)
        self.assertEqual(routed, [True, False, False, True])
        self.assertTrue(self.files.closed and more.closed)
        self.assertTrue(self.web.closed)
        self.assertEqual(self.close_that_calls, [])
        self.assertFalse(self.claude.closed or self.hud.closed
                         or self.pad.closed or self.media.closed)
        for s in self.spoken:
            self.assertNotIn("no record", s)

    def test_a_named_close_the_brain_still_answers_is_rewritten(self):
        # Nothing to route (no window of that name is open now): the brain
        # answers, and its close_last_opened becomes close_window <name>.
        self.files.closed = True
        printed = self._turn("Jarvis, close the file explorer.", LIVE_CLOSE_THAT,
                             ["[intent:bad_news] There's no File Explorer "
                              "window open, sir."])
        self.llm.assert_called_once()
        self.assertIn("[named-close] the owner named the window", printed)
        self.assertEqual(self.close_that_calls, [])
        self.assertIn("[action] close_window: no window matching 'file "
                      "explorer'", printed)

    def test_the_rewrite_runs_whatever_close_window_is_registered(self):
        stub = self._stub("close_window", "closed: Claude - Google Chrome")
        self.assertIsNotNone(stub)
        self._turn("Jarvis close Google Chrome for me.", LIVE_CLOSE_THAT)
        # A replaced handler is not pre-resolved by the route ...
        self.llm.assert_called_once()
        # ... and the brain's nameless close still becomes the named one.
        self.assertEqual(self.calls["close_window"], ["Google Chrome"])
        self.assertEqual(self.close_that_calls, [])

    def test_close_that_is_still_close_last_opened(self):
        self._turn("Jarvis, close that.", LIVE_CLOSE_THAT,
                   ["[intent:bad_news] Which window, sir?"])
        self.llm.assert_called_once()
        self.assertEqual(self.close_that_calls, [""])
        self.assertFalse(any(w.closed for w in self.wins))

    def test_a_name_that_is_no_window_goes_to_the_brain(self):
        # Not even when a page title merely contains the word.
        bell = _Win("Doorbell camera - Google Chrome", 0x78)
        self.wins.append(bell)
        self.A._window_process_name.side_effect = (
            lambda w: "chrome.exe" if w._hWnd in (0x71, 0x78) else None)
        self._turn("Jarvis, close the garage door.",
                   "[intent:confirmation] I've no garage door control, sir.")
        self.llm.assert_called_once()
        self.assertFalse(any(w.closed for w in self.wins))


# ════════════════════════════════════════════════════════════════════════
#  K - close everything except Claude: the keep, "Claw", "you forgot X"
# ════════════════════════════════════════════════════════════════════════
@requires_monolith
class CloseAllReplayTests(_Desk):
    def setUp(self):
        real = self.bc._jarvis_pushback
        super().setUp()
        bc = self.bc
        self._p(bc, "_jarvis_pushback", real)
        self._p(bc, "PUSHBACK_ENABLED", True)
        self._p(bc, "PUSHBACK_MAX_CLOSE_WINDOWS", 5)
        self._p(bc, "_pending_confirmation", [])
        self._p(bc, "_pending_confirmation_at", [0.0])

    def _owner_windows_closed(self):
        return [w.title for w in (self.web, self.files, self.media, self.pad,
                                  self.sheet) if w.closed]

    def test_claw_asks_did_you_mean_claude_and_the_yes_closes(self):
        # 00:25:02, the live words.
        self._turn("Jarvis close every window except for Claw.",
                   "[intent:confirmation] Very good, sir.")
        self.llm.assert_not_called()
        self.assertFalse(any(w.closed for w in self.wins), "closed on a "
                         "guess")
        self.assertIn("I don't see a Claw window, sir. Did you mean Claude? "
                      "Say yes and I'll close the other 5 windows.",
                      " ".join(self.spoken))
        self.assertEqual(list(self.bc._pending_confirmation),
                         [(CLOSE_ALL, "Claude")])
        self.spoken.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.bc.handle_confirmation_response(
                "Jarvis, yes."))
        self.assertEqual(len(self._owner_windows_closed()), 5)
        self.assertFalse(self.claude.closed or self.hud.closed)
        self.assertEqual(self.spoken, ["Closed 5 windows, sir; kept Claude."])

    def test_a_no_to_the_question_closes_nothing(self):
        self._turn("Jarvis close every window except for Claw.", "")
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.handle_confirmation_response("No.")
        self.assertFalse(any(w.closed for w in self.wins))

    def test_the_question_asks_with_pushback_off_too(self):
        self._p(self.bc, "PUSHBACK_ENABLED", False)
        got = self.bc._jarvis_pushback(CLOSE_ALL, "Claw")
        self.assertIsNotNone(got)
        self.assertEqual(got[2], "Claude")

    def test_except_claude_closes_the_chrome_window_with_a_claude_tab(self):
        # 00:25:12: the Claude APP runs, so it is what "Claude" keeps.
        self._turn("Jarvis close every window except for Claude.", "")
        self.llm.assert_not_called()
        self.assertTrue(self.web.closed)
        self.assertEqual(len(self._owner_windows_closed()), 5)
        self.assertFalse(self.claude.closed or self.hud.closed)
        self.assertEqual(self.spoken, ["Closed 5 windows, sir; kept Claude."])

    def test_you_forgot_chrome_closes_chrome(self):
        # No Claude app this time: the Chrome window with the Claude page is
        # what "Claude" keeps - and the owner hears why it stayed. Then the
        # live follow-up closes it; the brain's live reply (keep Chrome too)
        # never runs.
        self.wins.remove(self.claude)
        self._turn("Jarvis close every window except for Claude.", "")
        self.assertFalse(self.web.closed)
        self.assertEqual(self.spoken, [
            "Closed 4 windows, sir; kept Claude. Google Chrome stays open "
            "for its Claude page."])
        self.spoken.clear()
        printed = self._turn(
            "Jarvis, you forgot about Google Chrome.",
            "[intent:confirmation] Certainly, sir. "
            "[ACTION: close_all_windows_except, Claude, Chrome]")
        self.llm.assert_not_called()
        self.assertIn("[route] you forgot X after a bulk close -> "
                      "close_window", printed)
        self.assertTrue(self.web.closed)
        self.assertEqual(self.spoken, ["[intent:confirmation] Very good, sir."])

    def test_you_forgot_without_a_bulk_close_is_the_brains(self):
        self._turn("Jarvis, you forgot about Google Chrome.",
                   "[intent:confirmation] Forgot what about it, sir?")
        self.llm.assert_called_once()
        self.assertFalse(self.web.closed)


# ════════════════════════════════════════════════════════════════════════
#  S - "pull up that page so I can sign in": never an unasked sign-in click
# ════════════════════════════════════════════════════════════════════════
OWNER_ASK = "Jarvis, bring that page up so I can sign in for you."
PAGE = "https://console.example.com/"
CHOOSER_LOOK = ("[local-vision] The browser window is not showing the console "
                "page; it shows a Google 'Choose an account' screen listing "
                "one account and 'Use another account'.")
ACCOUNT_ENTRY = "Pat Example (pat.example@example.com)"


@requires_monolith
class SignInReplayTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._stub("open_url", f"opened {PAGE} — use see_screen to read what "
                               "loaded")
        self._stub("see_screen", CHOOSER_LOOK)
        self.find = self._p(bc, "find_click_target", side_effect=AssertionError(
            "looked for the account entry"))
        self.click = self._p(bc, "ui_click", side_effect=AssertionError(
            "the mouse moved"))
        self._p(bc, "_read_focused_window", return_value=(None, "Claude",
                                                          None))
        from core import opened_ledger as ol
        ol.reset()
        self.addCleanup(ol.reset)

    def _replay(self, click_reply):
        return self._dispatch(
            OWNER_ASK,
            f"[intent:confirmation] Right away, sir. [ACTION: open_url, "
            f"{PAGE}]",
            [f"[intent:briefing] One moment, sir. [ACTION: see_screen, "
             f"{PAGE}]",
             click_reply,
             f"[intent:bad_news] Let me look again. [ACTION: see_screen, "
             f"{PAGE}]"])

    def test_the_live_account_click_is_refused_and_the_turn_ends(self):
        self._replay("[intent:confirmation] Certainly, sir. "
                     f"[ACTION: click, {ACCOUNT_ENTRY}] I've selected your "
                     "account; just one more step to get through the gate.")
        self.find.assert_not_called()
        self.click.assert_not_called()
        from core.auth_guard import READY_LINE
        self.assertEqual(self.gfr.call_count, 2)
        self.assertEqual(self.calls["see_screen"], [PAGE])
        self.assertIn(READY_LINE, self.spoken)
        for s in self.spoken:
            self.assertNotIn("selected your account", s)
            self.assertNotIn("example.com", s.replace(PAGE, ""))

    def test_any_click_on_the_chooser_page_is_refused(self):
        self._replay("[ACTION: click, the blue Continue button]")
        self.find.assert_not_called()
        self.click.assert_not_called()
        from core.auth_guard import READY_LINE
        self.assertIn(READY_LINE, self.spoken)

    def test_the_look_is_kept_for_the_turn(self):
        seen = {}

        def _click(arg=""):
            seen["screens"] = self.bc._turn_screen_texts()
            return "clicked"
        self._actions["click"] = _click
        self._replay("[ACTION: click, somewhere]")
        self.assertEqual(seen["screens"], [CHOOSER_LOOK])
        self.assertEqual(self.bc._turn_screen_texts(), [])

    def test_an_exact_click_request_still_clicks(self):
        self.click.side_effect = None
        self.find.side_effect = None
        self.find.return_value = (40, 50)
        self._dispatch("Jarvis, click Continue with Google.",
                       "[ACTION: click, Continue with Google]")
        self.click.assert_called_once_with(40, 50)


@requires_monolith
class TerminalLineTests(_Base):
    def test_the_refusal_is_a_terminal_line_spoken_verbatim(self):
        from core.auth_guard import READY_LINE
        line = fm.TERMINAL_FAILURE_PREFIX + READY_LINE
        self.assertEqual(self.bc._verbatim_result_text("click", line),
                         READY_LINE)


if __name__ == "__main__":   # pragma: no cover
    import unittest
    unittest.main()
