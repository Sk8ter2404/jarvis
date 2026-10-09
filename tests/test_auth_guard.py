"""core.auth_guard + the click actions - JARVIS never signs in for him.

Live 2026-10-05 00:14:07-00:14:21 (session_2026-10-04_23-13-00.log,
paraphrased): the owner asked JARVIS to pull up the console page so HE could
sign in. JARVIS opened it, looked at the screen (an account chooser) and then,
unasked, clicked the owner's own Google account entry (his name and e-mail
address) - "I've selected your account; just one more step". The click
missed. Now a sign-in click the owner did not ask for in that turn is refused
at the ACTION layer with one terminal line: the page is ready for him.

Placeholders only: no real name or address appears here. Light tier: the
monolith is a Mock; vision, the mouse and every window are faked.

    python -m unittest tests.test_auth_guard
"""
from __future__ import annotations

import unittest
from unittest import mock

import core.actions as A
from core import auth_guard as ag
from core import failure_markers as fm
from core import opened_ledger as ol

# The live click's target, with a placeholder identity.
ACCOUNT_ENTRY = "Pat Example (pat.example@example.com)"
# What the live look at the screen said, paraphrased.
CHOOSER_LOOK = ("[local-vision] The browser window is not showing the console "
                "page; it shows a Google 'Choose an account' screen with one "
                "account listed and a 'Use another account' option.")
OWNER_ASK = "Jarvis, bring that page up so I can sign in for you."
READY = ag.READY_LINE


class AuthControlTests(unittest.TestCase):
    def test_an_account_entry_is_a_sign_in_control(self):
        self.assertTrue(ag.auth_control(ACCOUNT_ENTRY))

    def test_sign_in_controls(self):
        for d in ("Sign in", "the Log in button", "Continue with Google",
                  "Use another account", "Allow access", "Authorize app",
                  "select your Google account", "Login",
                  "Sign into your account", "Continue as Pat",
                  "Continue with Microsoft"):
            with self.subTest(d=d):
                self.assertTrue(ag.auth_control(d))

    def test_ordinary_controls_are_not(self):
        for d in ("the play button", "Saved Songs", "Settings", "Billing",
                  "the first search result", "Next episode", ""):
            with self.subTest(d=d):
                self.assertEqual(ag.auth_control(d), "")

    def test_look_alikes_are_not_sign_in_controls(self):
        # Review 2026-10-05: each of these was refused with "the sign-in
        # page is up and ready for you". On a REAL sign-in page the page
        # evidence still refuses them.
        for d in ("Continue with the download button",
                  "Continue with free version", "Continue as guest",
                  "Accept all cookies (consent)",
                  "Reject all on the cookie consent dialog",
                  "Show password eye icon", "the password field", "Allow",
                  "Login flow redesign.docx",
                  # An inbox row carries an address but is no account entry.
                  "Prof. Example <prof@school.example.edu> - Exam schedule"):
            with self.subTest(d=d):
                self.assertEqual(ag.auth_control(d), "")


class AuthPageTests(unittest.TestCase):
    def test_sign_in_urls(self):
        for u in ("https://accounts.google.com/v3/signin/accountchooser?x=1",
                  "https://login.microsoftonline.com/common/oauth2",
                  "https://example.com/login", "https://example.com/u/signin?"
                  "next=/", "https://github.com/login"):
            with self.subTest(u=u):
                self.assertTrue(ag.auth_page(urls=[u]))

    def test_ordinary_urls(self):
        for u in ("https://console.example.com/", "https://www.youtube.com/"
                  "watch?v=x", "https://example.com/blog/how-to-login-faster"
                  "-tips"):
            with self.subTest(u=u):
                self.assertEqual(ag.auth_page(urls=[u]), "")

    def test_sign_in_titles(self):
        for t in ("Sign in - Google Accounts - Google Chrome",
                  "Log in to Example - Google Chrome", "Login | Example",
                  "Choose an account"):
            with self.subTest(t=t):
                self.assertTrue(ag.auth_page(titles=[t]))

    def test_ordinary_titles(self):
        for t in ("Claude", "Logins and passwords - Settings",
                  "Home - YouTube - Google Chrome", "",
                  # Review 2026-10-05: the title check is anchored at the
                  # title's start - these merely mention signing in.
                  "How to log in to Fortnite on PC - YouTube - Google Chrome",
                  "How to Sign in to Roblox - YouTube - Google Chrome",
                  "Fixing sign in with Google bug - Claude",
                  "Login flow redesign.docx - Word"):
            with self.subTest(t=t):
                self.assertEqual(ag.auth_page(titles=[t]), "")

    def test_what_a_look_said(self):
        self.assertTrue(ag.auth_page(screen_texts=[CHOOSER_LOOK]))
        self.assertTrue(ag.auth_page(screen_texts=[
            "The console is requesting a login before it shows anything."]))
        # A page with a "Sign in" link in its header is not a sign-in page,
        # and a page that says he is signed in is not either.
        self.assertEqual(ag.auth_page(screen_texts=[
            "YouTube home page with a Sign in button at the top right."]), "")
        self.assertEqual(ag.auth_page(screen_texts=[
            "The dashboard; you are already signed in as the owner."]), "")
        # A "Sign in with Google" pop-up over an ordinary page does not make
        # it a sign-in page (a click ON the pop-up is still a sign-in
        # control).
        one_tap = ("A news article page with a 'Sign in with Google' pop-up "
                   "in the corner offering to continue with Google.")
        self.assertEqual(ag.auth_page(screen_texts=[one_tap]), "")
        self.assertEqual(ag.click_refusal("the article headline",
                                          "open the first story",
                                          screen_texts=[one_tap]), "")
        self.assertTrue(ag.click_refusal("Continue with Google",
                                         "open the first story",
                                         screen_texts=[one_tap]))
        # ... but a click with no target (coordinates) may land on the
        # pop-up, so it is refused (review 2026-10-05).
        self.assertTrue(ag.auth_overlay([one_tap]))
        self.assertTrue(ag.click_refusal("", "open the first story",
                                         screen_texts=[one_tap]))

    def test_more_ways_a_look_describes_a_sign_in_page(self):
        # Review 2026-10-05: wordings the first version missed.
        for look in ("I can see a login prompt for the console.",
                     "The console is asking for an email address to "
                     "continue.",
                     "The Claude Console with a Google sign-in pop-up "
                     "listing one account.",
                     "A Google account chooser popup is open."):
            with self.subTest(look=look):
                self.assertTrue(ag.auth_page(screen_texts=[look]))
        # "to continue to" alone is not one (a video ad, a checkout).
        for look in ("A YouTube ad is playing; skip to continue to the "
                     "video.", "A checkout form; press Next to continue to "
                     "payment."):
            with self.subTest(look=look):
                self.assertEqual(ag.auth_page(screen_texts=[look]), "")
                self.assertEqual(ag.auth_overlay([look]), "")


class OwnerAskedTests(unittest.TestCase):
    def test_so_i_can_sign_in_asks_for_no_click(self):
        self.assertFalse(ag.owner_asked_for_click(OWNER_ASK, ACCOUNT_ENTRY))
        self.assertFalse(ag.owner_asked_for_click("sign me in", "Sign in"))

    def test_an_exact_click_request(self):
        self.assertTrue(ag.owner_asked_for_click(
            "Jarvis, click Continue with Google", "Continue with Google"))
        self.assertTrue(ag.owner_asked_for_click("press the sign in button",
                                                 "Sign in"))
        self.assertTrue(ag.owner_asked_for_click("pick my account",
                                                 ACCOUNT_ENTRY))

    def test_a_different_click_is_not_that_click(self):
        self.assertFalse(ag.owner_asked_for_click("click sign in",
                                                  "Continue with Google"))

    def test_a_coordinate_click_needs_a_sign_in_word(self):
        self.assertTrue(ag.owner_asked_for_click("click log in", ""))
        self.assertFalse(ag.owner_asked_for_click("click there", ""))

    def test_rewordings_of_the_live_request_ask_for_no_click(self):
        # Review 2026-10-05: a bag of words let each of these through.
        for said, target in (
                ("Jarvis, bring that page up so I can pick my account.",
                 ACCOUNT_ENTRY),
                ("Jarvis, bring that page up, don't click anything, I'll "
                 "choose my account.", ACCOUNT_ENTRY),
                ("Open the console, I'll choose my Google account myself.",
                 "Continue with Google"),
                ("Open the console so I can sign in and pick a plan.", ""),
                ("Do not select my account, just open the page.",
                 ACCOUNT_ENTRY)):
            with self.subTest(said=said):
                self.assertFalse(ag.owner_asked_for_click(said, target))

    def test_a_command_to_jarvis_in_its_own_clause(self):
        for said, target in (
                ("Jarvis, can you pick my account?", ACCOUNT_ENTRY),
                ("open the console and click continue with Google",
                 "Continue with Google"),
                ("Okay Jarvis, select the Pat Example account",
                 ACCOUNT_ENTRY)):
            with self.subTest(said=said):
                self.assertTrue(ag.owner_asked_for_click(said, target))
        # The target's words must be in the CLICK clause, not anywhere.
        self.assertFalse(ag.owner_asked_for_click(
            "click the first video, I'll handle my Google account",
            "Continue with Google"))


class RefusalTests(unittest.TestCase):
    def test_the_live_click_is_refused_with_the_ready_line(self):
        out = ag.click_refusal(ACCOUNT_ENTRY, OWNER_ASK)
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX))
        self.assertEqual(fm.terminal_failure_text(out), READY)
        # The owner's details never end up in the line.
        self.assertNotIn("example.com", out)
        self.assertNotIn("Pat", out)

    def test_anything_on_a_sign_in_page_is_refused(self):
        out = ag.click_refusal("the blue button", OWNER_ASK,
                               screen_texts=[CHOOSER_LOOK])
        self.assertEqual(fm.terminal_failure_text(out), READY)

    def test_an_asked_for_click_goes_ahead(self):
        self.assertEqual(ag.click_refusal(
            "Continue with Google", "Jarvis, click continue with Google",
            screen_texts=[CHOOSER_LOOK]), "")

    def test_an_ordinary_click_goes_ahead(self):
        self.assertEqual(ag.click_refusal("the play button", "play my songs",
                                          titles=["Music - Google Chrome"]),
                         "")

    def test_a_sign_in_button_on_an_ordinary_page_is_not_a_page_claim(self):
        # Review 2026-10-05: the refusal said "the sign-in page is up" for a
        # word match alone.
        out = ag.click_refusal("the Sign in button", "play my mix",
                               titles=["Home - YouTube - Google Chrome"])
        self.assertEqual(fm.terminal_failure_text(out), ag.CONTROL_LINE)

    def test_after_a_refusal_the_rest_of_the_turn_is_refused(self):
        out = ag.click_refusal("the blue button", OWNER_ASK,
                               refused_before=True)
        self.assertEqual(fm.terminal_failure_text(out), READY)

    def test_a_coordinate_click_after_finding_the_account(self):
        out = ag.click_refusal("", OWNER_ASK, looked_for=[ACCOUNT_ENTRY])
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.assertEqual(ag.click_refusal("", OWNER_ASK,
                                          looked_for=["the play button"]), "")

    def test_never_raises(self):
        self.assertEqual(ag.click_refusal(object(), object(), [object()],
                                          [None], [3]), "")
        self.assertEqual(ag.input_refusal(object(), object(), object()), "")


class InputRefusalTests(unittest.TestCase):
    """Typing and submit keys (review 2026-10-05: the click guard alone left
    "[ACTION: type, <address>] [ACTION: press, enter]" free to sign in)."""

    def test_typing_on_a_sign_in_page_is_refused(self):
        out = ag.input_refusal("type", "pat.example@example.com", OWNER_ASK,
                               screen_texts=[CHOOSER_LOOK])
        self.assertEqual(fm.terminal_failure_text(out), READY)

    def test_submit_keys_on_a_sign_in_page_are_refused(self):
        for key in ("enter", "Return", "tab", "space", "ctrl+enter"):
            with self.subTest(key=key):
                self.assertTrue(ag.input_refusal(
                    "press", key, OWNER_ASK,
                    titles=["Sign in - Google Accounts - Google Chrome"]))

    def test_other_keys_and_ordinary_pages_go_ahead(self):
        self.assertEqual(ag.input_refusal("press", "volumeup", OWNER_ASK,
                                          screen_texts=[CHOOSER_LOOK]), "")
        self.assertEqual(ag.input_refusal("press", "enter", "send it",
                                          titles=["Chat - Teams"]), "")
        self.assertEqual(ag.input_refusal("type", "hello", "say hello",
                                          titles=["notes.txt - Notepad"]), "")

    def test_the_owner_may_ask_for_exactly_that_input(self):
        self.assertEqual(ag.input_refusal(
            "press", "enter", "Jarvis, press enter",
            screen_texts=[CHOOSER_LOOK]), "")
        self.assertEqual(ag.input_refusal(
            "type", "pat.example@example.com", "type my email address in",
            screen_texts=[CHOOSER_LOOK]), "")
        self.assertTrue(ag.input_refusal(
            "press", "enter", "press tab", screen_texts=[CHOOSER_LOOK]))

    def test_a_sign_in_pop_up_refuses_typing_too(self):
        pop = "A shop page with a 'Sign in with Google' pop-up."
        self.assertTrue(ag.input_refusal("type", "x", "", screen_texts=[pop]))


def _grounded_stand_in(bc):
    """core.grounded_click.run_bounded for a desk with no real windows: find
    the target through ``bc.find_click_target`` and click it through
    ``bc.ui_click`` - the executor's line is the old click line."""
    from core import grounded_click as G

    def run_bounded(arg, said="", mode="click", backend=None, budget_s=None):
        pt = bc.find_click_target(arg)
        if pt is None:
            return G.Result(f"could not locate '{arg}' on screen",
                            G.NOT_FOUND)
        bc.ui_click(pt[0], pt[1])
        return G.Result(f"clicked '{arg}' at {tuple(pt)}", G.VERIFIED,
                        label=arg)
    return run_bounded


class _ClickBase(unittest.TestCase):
    def setUp(self):
        self.bc = mock.Mock()
        self.bc._parse_monitor_prefix.side_effect = lambda a: (None, a)
        self.bc._is_self_close_attempt.return_value = False
        self.bc._turn_user_text.return_value = OWNER_ASK
        self.bc._turn_screen_texts.return_value = [CHOOSER_LOOK]
        self.bc._read_focused_window.return_value = (None, "Claude", None)
        self.bc.find_click_target.return_value = (100, 200)
        self.bc.ui_click.side_effect = AssertionError(
            "the mouse moved: a refused click must never reach ui_click")
        p = mock.patch.object(A, "_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)
        # A description click is executed by core.grounded_click (screen
        # vision, merged after v2.0.181) instead of find_click_target +
        # ui_click. This stand-in executor locates through the same
        # find_click_target stub and clicks through ui_click, so what the
        # sign-in guard lets through - or stops before the executor - stays
        # observable exactly as before.
        p = mock.patch("core.grounded_click.run_bounded",
                       side_effect=_grounded_stand_in(self.bc))
        self.grounded = p.start()
        self.addCleanup(p.stop)
        ol.reset()
        self.addCleanup(ol.reset)


class ClickActionTests(_ClickBase):
    def test_the_live_click_never_happens(self):
        out = A._act_click(ACCOUNT_ENTRY)
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.find_click_target.assert_not_called()
        self.bc.ui_click.assert_not_called()

    def test_a_coordinate_click_on_the_sign_in_page_is_refused_too(self):
        out = A._act_click("812, 433")
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.ui_click.assert_not_called()

    def test_the_page_jarvis_opened_is_read_by_its_window_title(self):
        self.bc._turn_screen_texts.return_value = []
        ol.note_opened("open_url", "https://console.example.com/",
                       hwnd=0x5150, kind="tab")
        win = mock.Mock(_hWnd=0x5150,
                        title="Sign in - Google Accounts - Google Chrome")
        fake_gw = mock.Mock(getAllWindows=mock.Mock(return_value=[win]))
        with mock.patch.dict("sys.modules", {"pygetwindow": fake_gw}):
            out = A._act_click("the blue button")
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.ui_click.assert_not_called()

    def test_an_asked_for_click_runs(self):
        self.bc._turn_user_text.return_value = ("Jarvis, click Continue with "
                                                "Google")
        self.bc.ui_click.side_effect = None
        out = A._act_click("Continue with Google")
        self.assertEqual(out, "clicked 'Continue with Google' at (100, 200)")
        self.bc.ui_click.assert_called_once_with(100, 200)

    def test_an_ordinary_click_runs_as_before(self):
        self.bc._turn_screen_texts.return_value = [
            "A music app with a sidebar and a Play button."]
        self.bc._turn_user_text.return_value = "play my saved songs"
        self.bc.ui_click.side_effect = None
        out = A._act_click("Saved Songs")
        self.assertEqual(out, "clicked 'Saved Songs' at (100, 200)")


class ClickOnScreenTests(_ClickBase):
    """click_on_screen (screen vision) gets the same pre-check as click: the
    monolith hands every description click to it (_click_alias), and the
    screen route sends "click that X" to it."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(A, "_loaded_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)

    def test_the_live_click_never_reaches_the_executor(self):
        out = A._act_click_on_screen(ACCOUNT_ENTRY)
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.grounded.assert_not_called()
        self.bc._turn_note_auth_refused.assert_called_once_with()

    def test_a_monitor_prefix_is_judged_by_its_target(self):
        out = A._act_click_on_screen(f"monitor:middle|{ACCOUNT_ENTRY}")
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.grounded.assert_not_called()

    def test_an_asked_for_click_reaches_the_executor(self):
        self.bc._turn_user_text.return_value = ("Jarvis, click Continue with "
                                                "Google")
        self.bc.ui_click.side_effect = None
        out = A._act_click_on_screen("Continue with Google")
        self.assertEqual(out, "clicked 'Continue with Google' at (100, 200)")
        self.grounded.assert_called_once()

    def test_a_pick_answer_is_left_to_the_executor_s_own_guard(self):
        # "pick:2" names no target: grounded_click judges the option it
        # resolves to, with core.auth_guard, by that window's page.
        self.bc.ui_click.side_effect = None
        A._act_click_on_screen("pick:2")
        self.grounded.assert_called_once()


class ReviewClickActionTests(_ClickBase):
    """Review 2026-10-05 findings at the action layer."""

    def _ledger_window(self, title_then, title_now, hwnd=0x5151,
                       url="https://example.com/login"):
        ol.note_opened("open_url", url, hwnd=hwnd, kind="tab",
                       title=title_then)
        win = mock.Mock(_hWnd=hwnd, title=title_now)
        fake_gw = mock.Mock(getAllWindows=mock.Mock(return_value=[win]))
        p = mock.patch.dict("sys.modules", {"pygetwindow": fake_gw})
        p.start()
        self.addCleanup(p.stop)

    def test_a_login_address_stops_counting_once_the_page_moves_on(self):
        self.bc._turn_screen_texts.return_value = []
        self.bc._turn_user_text.return_value = "play the first video"
        self.bc.ui_click.side_effect = None
        self._ledger_window("Login - Example - Google Chrome",
                            "Dashboard - Example - Google Chrome")
        self.assertEqual(A._act_click("the first video"),
                         "clicked 'the first video' at (100, 200)")

    def test_a_login_address_counts_while_its_page_is_in_front(self):
        self.bc._turn_screen_texts.return_value = []
        self.bc._turn_user_text.return_value = "play the first video"
        self._ledger_window("Example - Google Chrome",
                            "Example - Google Chrome")
        self.bc._read_focused_window.return_value = (
            0x5151, "Example - Google Chrome", None)
        out = A._act_click("the first video")
        self.assertEqual(fm.terminal_failure_text(out), READY)

    def test_the_opened_page_counts_only_while_it_is_in_front(self):
        self.bc._turn_screen_texts.return_value = []
        self.bc._turn_user_text.return_value = "play the first video"
        self.bc.ui_click.side_effect = None
        self._ledger_window("Sign in - Google Accounts - Google Chrome",
                            "Sign in - Google Accounts - Google Chrome")
        self.bc._read_focused_window.return_value = (
            0x7777, "Lo-fi - YouTube - Google Chrome", None)
        self.assertEqual(A._act_click("the first video"),
                         "clicked 'the first video' at (100, 200)")

    def test_a_coordinate_click_on_what_find_on_screen_located(self):
        self.bc._turn_screen_texts.return_value = [
            "[local-vision] The console with a small panel in the corner."]
        self.bc._turn_click_targets.return_value = [ACCOUNT_ENTRY]
        out = A._act_click("500, 300")
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.ui_click.assert_not_called()

    def test_a_refusal_marks_the_turn(self):
        A._act_click(ACCOUNT_ENTRY)
        self.bc._turn_note_auth_refused.assert_called_once_with()


class KeyActionTests(_ClickBase):
    def setUp(self):
        super().setUp()
        self.bc.ui_type.side_effect = AssertionError("typed")
        self.bc.ui_press.side_effect = AssertionError("pressed")
        self.bc.ui_hotkey.side_effect = AssertionError("hotkey")
        self.bc._looks_like_shell_command.return_value = False
        self.bc._normalize_key.side_effect = lambda k: k.strip().lower()

    def test_typing_the_address_on_the_chooser_page_is_refused(self):
        out = A._act_type("pat.example@example.com")
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.ui_type.assert_not_called()

    def test_enter_on_the_chooser_page_is_refused(self):
        self.assertEqual(fm.terminal_failure_text(A._act_press("enter")),
                         READY)
        self.assertEqual(fm.terminal_failure_text(A._act_hotkey("enter")),
                         READY)
        self.bc.ui_press.assert_not_called()
        self.bc.ui_hotkey.assert_not_called()

    def test_after_a_refused_click_enter_is_refused_anywhere(self):
        self.bc._turn_screen_texts.return_value = []
        self.bc._turn_auth_refused.return_value = True
        self.assertEqual(fm.terminal_failure_text(A._act_press("enter")),
                         READY)

    def test_ordinary_keys_and_pages_are_untouched(self):
        self.bc.ui_press.side_effect = None
        self.bc._turn_screen_texts.return_value = []
        self.assertEqual(A._act_press("enter"), "pressed enter")
        self.bc._turn_screen_texts.return_value = [CHOOSER_LOOK]
        self.assertEqual(A._act_press("volumeup"), "pressed volumeup")


class LocalVisionClickTests(_ClickBase):
    def test_the_local_eye_refuses_the_same_click(self):
        from skills import local_vision as lv
        with mock.patch.object(lv, "_bobert", return_value=self.bc), \
             mock.patch.object(lv, "_find_click_target_local",
                               side_effect=AssertionError("looked")):
            out = lv.local_click_target_by_description(ACCOUNT_ENTRY)
        self.assertEqual(fm.terminal_failure_text(out), READY)
        self.bc.ui_click.assert_not_called()


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
