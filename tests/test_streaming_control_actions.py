"""core/actions.py - the streaming-control fixes of 2026-10-02, through the
real handlers with the monolith faked (CI-light: core.actions._bc is
patched, nothing real is opened, closed, clicked or captured).

Live 16:12-16:14 (paraphrased; no owner words):
  S1  "close that and open <service> instead" ran only the open;
  S2  the brain opened a guessed hbomax.com/search URL (a 404);
  S3  see_screen got a bare URL, then the owner's "continue", as its question;
  S4  a click meant for the page on the MIDDLE monitor landed on the LEFT one;
  S5  the page said "Sign In" / "Oops ... isn't working" and the turn kept
      clicking.

    python -m unittest tests.test_streaming_control_actions
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

import core.actions as A
from core import opened_ledger as L
from core.failure_markers import TERMINAL_FAILURE_PREFIX, terminal_failure_text

MONS = {
    "left":   (-2560, 0, 2560, 1440),
    "middle": (0, 0, 2560, 1440),
    "right":  (2560, 0, 2560, 1440),
    "top":    (0, -1440, 2560, 1440),
}
SUFFIX = " - Google Chrome"


class _Win:
    def __init__(self, title, hwnd, box=(0, 0, 2560, 1400)):
        self.title = title
        self._hWnd = hwnd
        self.left, self.top, self.width, self.height = box
        self.closed = False

    def close(self):
        self.closed = True


def _gw(windows):
    mod = types.ModuleType("pygetwindow")
    mod.getAllWindows = lambda: [w for w in windows if not w.closed]
    return mod


def _clock(start=100.0, step=0.5):
    """An auto-incrementing time.time for open_on_monitor's polling loop."""
    state = {"t": start - step}

    def _now():
        state["t"] += step
        return state["t"]
    return _now


def _fake_bc():
    bc = mock.Mock()
    bc.FORBIDDEN_TARGETS = ("powershell", "bobert")
    bc._strip_bidi_and_nbsp = lambda s: s or ""
    bc._BROWSER_CHROME_SUFFIXES = (SUFFIX.lower(),)
    bc.screenshot_privacy_block_reason.return_value = None
    bc._turn_user_text.return_value = ""
    bc._parse_monitor_prefix.side_effect = lambda q: (None, q)
    bc._see_screen_budget_state = types.SimpleNamespace(used=0)
    bc.SEE_SCREEN_BUDGET_PER_INTENT = 3
    bc._is_self_close_attempt.return_value = False
    bc._VIDEO_QUERY_HINTS = ()
    return bc


class _Base(unittest.TestCase):
    def setUp(self):
        L.reset()
        self.addCleanup(L.reset)
        self.bc = _fake_bc()
        self._enter(mock.patch.object(A, "_bc", return_value=self.bc))
        self._enter(mock.patch("core.config.MONITORS", MONS))
        self._enter(mock.patch.object(A.time, "sleep"))
        self._enter(contextlib.redirect_stdout(io.StringIO()))
        self.windows: list = []
        self._enter(mock.patch.dict(sys.modules, {"pygetwindow": _gw(self.windows)}))

    def _enter(self, cm):
        v = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return v


# ════════════════════════════════════════════════════════════════════════════
#  S2 - open_url / open_on_monitor never open a guessed streaming search link
# ════════════════════════════════════════════════════════════════════════════
class OpenUrlGuardTests(_Base):
    def test_the_guessed_hbo_max_search_opens_the_verified_one(self):
        with mock.patch.object(A.webbrowser, "open") as wb:
            out = A._act_open_url("https://www.hbomax.com/search?q=Some+Show")
        wb.assert_called_once_with("https://play.hbomax.com/search?q=Some%20Show")
        self.assertIn("not a real HBO Max link", out)
        self.assertIn("see_screen", out)

    def test_no_verified_pattern_opens_the_home_page_and_says_so(self):
        with mock.patch.object(A.webbrowser, "open") as wb:
            out = A._act_open_url("https://www.disneyplus.com/search?q=Some+Show")
        wb.assert_called_once_with("https://www.disneyplus.com")
        self.assertIn("no verified Disney+ search link", out)

    def test_a_bare_service_name_opens_its_home(self):
        with mock.patch.object(A.webbrowser, "open") as wb:
            A._act_open_url("HBO Max")
        wb.assert_called_once_with("https://play.hbomax.com")

    def test_other_sites_are_untouched(self):
        with mock.patch.object(A.webbrowser, "open") as wb:
            out = A._act_open_url("https://example.com/search?q=x")
        wb.assert_called_once_with("https://example.com/search?q=x")
        self.assertEqual(out, "opened https://example.com/search?q=x — use "
                              "see_screen to read what loaded")

    def test_open_on_monitor_fixes_a_guessed_url_but_launches_a_bare_name(self):
        new = _Win("Some Show - HBO Max" + SUFFIX, 0x300)
        state = {"n": 0}

        def _all():
            state["n"] += 1
            return [] if state["n"] == 1 else [new]
        sys.modules["pygetwindow"].getAllWindows = _all
        self.bc._open_url_new_window.return_value = True
        new.restore = new.moveTo = new.maximize = lambda *a: None
        with mock.patch.object(A.time, "time", _clock()):
            out = A._act_open_on_monitor("main | www.hbomax.com/search?q=Some+Show")
        self.bc._open_url_new_window.assert_called_once_with(
            "https://play.hbomax.com/search?q=Some%20Show", monitor="middle")
        self.assertIn("not a real HBO Max link", out)
        with mock.patch.object(A, "_act_launch_app", return_value="ok") as la, \
                mock.patch.object(A.time, "time", _clock()):
            state["n"] = 0
            A._act_open_on_monitor("main | netflix")
        la.assert_called_once_with("netflix")


# ════════════════════════════════════════════════════════════════════════════
#  S1 - the ledger is fed by the openers, and close_last_opened uses it
# ════════════════════════════════════════════════════════════════════════════
class LedgerFeedTests(_Base):
    def test_open_on_monitor_records_the_window_it_made(self):
        new = _Win("results - YouTube" + SUFFIX, 0x200)
        new.restore = new.moveTo = new.maximize = lambda *a: None
        state = {"n": 0}

        def _all():
            state["n"] += 1
            return [] if state["n"] == 1 else [new]
        sys.modules["pygetwindow"].getAllWindows = _all
        self.bc._open_url_new_window.return_value = True
        with mock.patch.object(A.time, "time", _clock()):
            A._act_open_on_monitor("main | youtube a show")
        e = L.last_opened(max_age_s=1e12)
        self.assertEqual((e.via, e.hwnd, e.kind, e.monitor),
                         ("open_on_monitor", 0x200, "window", "middle"))

    def test_open_url_records_the_tab_that_came_to_the_front(self):
        before = (0x10, "Inbox - Mail", (0, 0, 800, 600))
        after = (0x11, "Some Page" + SUFFIX, (-2560, 0, 2560, 1400))
        self.bc._read_focused_window.side_effect = [before, after]
        with mock.patch.object(A, "_loaded_bc", return_value=self.bc), \
                mock.patch.object(A.webbrowser, "open"):
            A._act_open_url("https://example.com/page")
        e = L.last_opened()
        self.assertEqual((e.via, e.hwnd, e.kind, e.monitor, e.title),
                         ("open_url", 0x11, "tab", "left", "Some Page" + SUFFIX))

    def test_nothing_new_in_front_records_nothing(self):
        same = (0x11, "His Page" + SUFFIX, (0, 0, 2560, 1400))
        self.bc._read_focused_window.side_effect = [same, same]
        with mock.patch.object(A, "_loaded_bc", return_value=self.bc), \
                mock.patch.object(A.webbrowser, "open"):
            A._act_web_search("anything")
        self.assertIsNone(L.last_opened())

    def test_a_non_browser_in_front_records_nothing(self):
        self.bc._read_focused_window.side_effect = [
            (0x10, "a", None), (0x12, "Untitled - Notepad", (0, 0, 9, 9))]
        with mock.patch.object(A, "_loaded_bc", return_value=self.bc), \
                mock.patch.object(A.webbrowser, "open"):
            A._act_open_url("https://example.com")
        self.assertIsNone(L.last_opened())


class CloseLastOpenedTests(_Base):
    def test_closes_the_window_jarvis_made_and_nothing_else(self):
        his = _Win("His Stream" + SUFFIX, 0x100)
        mine = _Win("results - YouTube" + SUFFIX, 0x200)
        self.windows[:] = [his, mine]
        L.note_opened("open_on_monitor", "https://www.youtube.com/results?search_query=x",
                      hwnd=0x200, kind="window", monitor="middle")
        out = A._act_close_last_opened("")
        self.assertTrue(mine.closed)
        self.assertFalse(his.closed)
        self.assertEqual(out, "closed the YouTube page I opened")
        self.assertIsNone(L.last_opened())

    def test_a_tab_closes_only_while_it_is_still_in_front(self):
        win = _Win("Some Page" + SUFFIX, 0x11)
        self.windows[:] = [win]
        L.note_opened("open_url", "https://example.com/page", hwnd=0x11,
                      kind="tab", title="Some Page" + SUFFIX)
        with mock.patch.object(A, "_close_browser_tab", return_value=True) as ct:
            out = A._act_close_last_opened("")
        ct.assert_called_once_with(self.bc, win)
        self.assertIn("just that tab", out)
        self.assertFalse(win.closed)          # the WINDOW stays

    def test_his_tab_in_front_is_left_alone(self):
        win = _Win("His Own Page" + SUFFIX, 0x11)
        self.windows[:] = [win]
        L.note_opened("open_url", "https://example.com/page", hwnd=0x11,
                      kind="tab", title="Some Page" + SUFFIX)
        with mock.patch.object(A, "_close_browser_tab") as ct:
            out = A._act_close_last_opened("")
        ct.assert_not_called()
        self.assertFalse(win.closed)
        self.assertTrue(out.startswith("didn't close it"))

    def test_no_record_closes_nothing_and_says_so(self):
        his = _Win("His Stream" + SUFFIX, 0x100)
        self.windows[:] = [his]
        out = A._act_close_last_opened("")
        self.assertFalse(his.closed)
        self.assertTrue(out.startswith("couldn't close it"))   # a failure result

    def test_an_already_closed_window(self):
        L.note_opened("open_on_monitor", "notepad", hwnd=0x999)
        self.assertEqual(A._act_close_last_opened(""),
                         "the notepad window I opened is already closed")


# ════════════════════════════════════════════════════════════════════════════
#  S4 - a description click is aimed at the page's monitor
# ════════════════════════════════════════════════════════════════════════════
class ClickPinTests(_Base):
    def setUp(self):
        super().setUp()
        self.bc.find_click_target.return_value = (10, 20)

    def _page_on(self, box):
        self.windows[:] = [_Win("Search - HBO Max" + SUFFIX, 0x200, box)]
        L.note_opened("open_on_monitor", "https://play.hbomax.com/search?q=x",
                      hwnd=0x200, monitor="middle")

    def test_aimed_at_the_monitor_the_page_is_on(self):
        self._page_on((-8, -8, 2576, 1456))
        A._act_click('"Some Show" result')
        self.bc.find_click_target.assert_called_once_with(
            '"Some Show" result', monitor="middle")

    def test_the_monitor_it_is_on_now_wins(self):
        self._page_on((2560, 0, 2560, 1400))     # he moved it to the right
        A._act_click("the first result")
        self.bc.find_click_target.assert_called_once_with(
            "the first result", monitor="right")

    def test_a_monitor_in_his_words_wins(self):
        self._page_on((-8, -8, 2576, 1456))
        self.bc._turn_user_text.return_value = "click the first one on the left monitor"
        A._act_click("the first result")
        self.bc.find_click_target.assert_called_once_with(
            "the first result", monitor="left")

    def test_an_explicit_prefix_wins(self):
        self._page_on((-8, -8, 2576, 1456))
        self.bc._parse_monitor_prefix.side_effect = None
        self.bc._parse_monitor_prefix.return_value = ("top", "the button")
        A._act_click("monitor:top|the button")
        self.bc.find_click_target.assert_called_once_with("the button", monitor="top")

    def test_no_live_page_is_the_whole_desktop_as_before(self):
        L.note_opened("open_on_monitor", "x", hwnd=0x777, monitor="middle")
        A._act_click("the button")
        self.bc.find_click_target.assert_called_once_with("the button", monitor=None)


# ════════════════════════════════════════════════════════════════════════════
#  S3 + S5 - see_screen's question, its capture and the sign-in wall
# ════════════════════════════════════════════════════════════════════════════
class SeeScreenPageTests(_Base):
    PAGE = "https://play.hbomax.com/search?q=Some%20Show"

    def setUp(self):
        super().setUp()
        self.bc.take_screenshot.return_value = b"PNG"
        self.bc.take_all_monitor_screenshots.return_value = {"middle": b"PNG"}
        self.bc.ask_vision.return_value = "Search results for the show."
        self.bc.ask_vision_multi.return_value = "a desktop"

    def _opened(self, target=PAGE):
        self.windows[:] = [_Win("Search - HBO Max" + SUFFIX, 0x200,
                                (-8, -8, 2576, 1456))]
        L.note_opened("open_url", target, hwnd=0x200, kind="tab",
                      title="Search - HBO Max" + SUFFIX)

    def test_a_bare_url_is_turned_into_a_question_about_that_page(self):
        self._opened()
        A._act_see_screen("https://www.hbomax.com/search?q=Some+Show")
        self.bc.take_screenshot.assert_called_once_with(monitor="middle")
        self.bc.take_all_monitor_screenshots.assert_not_called()
        q = self.bc.ask_vision.call_args[0][0]
        self.assertTrue(q.startswith(
            "What is on the screen in the browser window showing "
            "https://www.hbomax.com/search?q=Some+Show?"))
        self.assertIn("error messages", q)
        self.assertIn("Ignore chat and assistant windows", q)

    def test_a_bare_url_with_no_opened_page_still_asks_about_the_page(self):
        A._act_see_screen("www.example.com")
        q = self.bc.ask_vision_multi.call_args[0][0]
        self.assertIn("browser window showing www.example.com", q)

    def test_continue_is_never_the_question(self):
        self._opened()
        for said in ("Jarvis, continue.", "Jarvis continued.", "go on",
                     "try again please", "okay"):
            with self.subTest(said=said):
                self.bc.ask_vision.reset_mock()
                self.bc._see_screen_budget_state.used = 0
                self.bc._turn_user_text.return_value = said
                A._act_see_screen("")
                q = self.bc.ask_vision.call_args[0][0]
                self.assertIn(f"browser window showing {self.PAGE}", q)
                self.assertNotIn("contin", q.lower())
                self.assertNotIn("The owner asked", q)

    def test_continue_with_nothing_opened_is_the_generic_look(self):
        self.bc._turn_user_text.return_value = "Jarvis, continue."
        A._act_see_screen("")
        q = self.bc.ask_vision_multi.call_args[0][0]
        self.assertIn("Describe in detail", q)

    def test_a_real_request_is_still_his_words(self):
        self._opened()
        self.bc._turn_user_text.return_value = "what does the error say"
        A._act_see_screen("")
        self.assertIn('The owner asked: "what does the error say"',
                      self.bc.ask_vision_multi.call_args[0][0])

    def test_a_sign_in_wall_ends_the_turn_with_one_plain_line(self):
        self._opened()
        self.bc.ask_vision.return_value = (
            "[local-vision] The page says 'Oops! Looks like this link isn't "
            "working.' and shows a Sign In button.")
        out = A._act_see_screen(self.PAGE)
        self.assertTrue(out.startswith(TERMINAL_FAILURE_PREFIX))
        self.assertEqual(terminal_failure_text(out),
                         "HBO Max isn't signed in on this browser, sir - sign "
                         "in once and I can take it from there.")

    def test_an_ordinary_page_answer_passes_through(self):
        self._opened()
        out = A._act_see_screen(self.PAGE)
        self.assertEqual(out, "Search results for the show.")

    def test_a_sign_in_button_on_a_non_streaming_page_is_not_a_wall(self):
        self._opened("https://example.com/page")
        self.bc.ask_vision.return_value = "A news page with a Sign In link."
        out = A._act_see_screen("https://example.com/page")
        self.assertEqual(out, "A news page with a Sign In link.")


class StreamingSearchActionTests(_Base):
    def test_routes_to_the_verified_opener(self):
        self.bc._streaming_open_search.return_value = "ok"
        self.assertEqual(A._act_streaming_search("HBO Max | Some Show"), "ok")
        self.bc._streaming_open_search.assert_called_once_with("max", "Some Show")

    def test_bad_input(self):
        self.assertTrue(A._act_streaming_search("no separator").startswith("format:"))
        self.assertIn("unknown streaming service",
                      A._act_streaming_search("Peacock | x"))


if __name__ == "__main__":
    unittest.main()
