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
OWNER_ASK = "Jarvis, pull up that page so I can sign in for you."
READY = ag.READY_LINE


class AuthControlTests(unittest.TestCase):
    def test_an_account_entry_is_a_sign_in_control(self):
        self.assertTrue(ag.auth_control(ACCOUNT_ENTRY))

    def test_sign_in_controls(self):
        for d in ("Sign in", "the Log in button", "Continue with Google",
                  "Use another account", "the password field", "Next - "
                  "enter password", "Allow access", "Authorize app",
                  "select your Google account", "Login"):
            with self.subTest(d=d):
                self.assertTrue(ag.auth_control(d))

    def test_ordinary_controls_are_not(self):
        for d in ("the play button", "Saved Songs", "Settings", "Billing",
                  "the first search result", "Next episode", ""):
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
                  "Home - YouTube - Google Chrome", ""):
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

    def test_never_raises(self):
        self.assertEqual(ag.click_refusal(object(), object(), [object()],
                                          [None], [3]), "")


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
