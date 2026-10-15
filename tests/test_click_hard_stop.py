"""core/send_guard.py + the click path - a hard stop before send, submit,
post, publish, pay, buy, transfer, delete, accept terms, sign (2026-10-15).

JARVIS's screen control must STOP before pressing a control whose purpose is
irreversible or outward-facing, ask in one honest sentence, and act only on
the owner's explicit spoken yes for that one action - one yes is one click.
Typed Enter in a chat / mail compose is held the same way.

Everything here is faked: the synthetic research pages of tests/_screen_fakes
(no real window, UIA, OCR, pointer, keyboard or model), a Mock monolith for
the action layer.

    python -m unittest tests.test_click_hard_stop
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import core.actions as A
from core import config as cfg
from core import dispatcher as D
from core import failure_markers as fm
from core import grounded_click as G
from core import onscreen_refs as OR
from core import send_guard as sg
from core.screen_text import El
from tests import _screen_fakes as F

# ── vocabulary ──────────────────────────────────────────────────────────────
BLOCKED = {
    sg.SEND: ["Send", "Send message", "Send email", "&Send (Alt+S)",
              "Send reply", "Send invite", "Save & Send", "Send feedback"],
    sg.SUBMIT: ["Submit", "Submit order", "Submit application",
                "Submit answers"],
    sg.POST: ["Post", "Post reply", "Post comment", "Post now", "Tweet"],
    sg.PUBLISH: ["Publish", "Publish now", "Publish site"],
    sg.PAY: ["Pay", "Pay now", "Pay $24.00", "Pay invoice",
             "Complete payment", "Confirm payment", "Make a payment"],
    sg.ORDER: ["Buy", "Buy now", "Purchase", "Place order",
               "Place your order", "Order now", "Checkout", "Check out",
               "Proceed to checkout", "Secure checkout", "Donate"],
    sg.TRANSFER: ["Transfer", "Transfer funds", "Send money", "Withdraw",
                  "Wire transfer"],
    sg.DELETE: ["Delete", "Delete forever", "Delete account", "Delete all",
                "Remove permanently", "Remove account", "Remove",
                "Empty recycle bin", "Erase", "Discard draft", "Uninstall",
                "Format drive", "Factory reset", "Cancel subscription"],
    sg.TERMS: ["Accept", "Accept terms", "I accept", "Agree",
               "I agree", "Agree and continue", "Sign", "Sign here",
               "Sign document", "Sign up", "Create account"],
    sg.CONFIRM: ["Confirm", "Confirm and continue"],
    sg.ACCOUNT: ["Sign out", "Log out", "Unsubscribe"],
}
PASSES = ["Open", "Next page", "Search", "Back", "Forward", "Reload",
          "Home", "Settings", "Play", "Save", "Cancel", "Close", "Reply",
          "Compose", "New post", "Add to cart", "View cart",
          "Submittal folder", "Sender", "Postal codes", "Post office",
          "Payment methods", "Purchase history", "Order history",
          "Transfer files", "Send/Receive", "Send to back",
          "Send to front", "Send backward", "Send to", "Unpublish",
          "Accept changes", "Remove from playlist", "Remove filter",
          "Sign in", "Log in", "Continue with Google", "Open the post",
          "Show password", "Delivery options",
          "Check out our blog", "Subscribe", "Share", "Forward email"]


class VocabularyTests(unittest.TestCase):
    def test_every_class_blocks(self):
        for kind, labels in BLOCKED.items():
            for label in labels:
                with self.subTest(kind=kind, label=label):
                    self.assertEqual(sg.classify(label, "Button"), kind)

    def test_ordinary_controls_pass(self):
        for label in PASSES:
            for role in ("Button", "MenuItem", "Hyperlink", "ocr", ""):
                with self.subTest(label=label, role=role):
                    self.assertEqual(sg.classify(label, role), "")

    def test_send_to_back_is_layering_not_outward_facing(self):
        # Decided: "Send to back / front / backward / forward" move a shape
        # in a drawing program and reach nobody, "Send to" is Explorer's
        # submenu; "Send feedback" and "Send to John" do reach somebody.
        for label in ("Send to back", "Send to front", "Send backward",
                      "Send forward", "Send to"):
            self.assertEqual(sg.classify(label, "MenuItem"), "", label)
        for label in ("Send feedback", "Send to John", "Send message"):
            self.assertEqual(sg.classify(label, "MenuItem"), sg.SEND, label)

    def test_only_the_first_verb_counts(self):
        # a trigger word LATER in the label is not the control's verb
        for label in ("Open the send dialog", "Next page of results",
                      "Search posts", "Back to orders", "Go back",
                      "Pause payments", "Show deleted items"):
            self.assertEqual(sg.classify(label, "Button"), "", label)

    def test_whole_words_only(self):
        for label in ("Submittal", "Sendmail setup", "Postal", "Payroll",
                      "Buyers guide", "Deleted items", "Publisher",
                      "Transferred", "Signature", "Signal"):
            self.assertEqual(sg.classify(label, "Button"), "", label)

    def test_links_and_text_must_be_short_and_not_a_sentence(self):
        # a link / list item / OCR line that is a sentence leads somewhere
        for label in ("Send us a message", "Post a job", "Pay your bill",
                      "Delete the old files from your account today"):
            for role in ("Hyperlink", "ocr", "ListItem"):
                self.assertEqual(sg.classify(label, role), "", (label, role))
        # ...but the bare verb as a link / OCR text is the control
        for label in ("Send", "Delete account", "Place order", "Buy now"):
            for role in ("Hyperlink", "ocr"):
                self.assertTrue(sg.classify(label, role), (label, role))
        # a button is judged on the whole label
        self.assertEqual(sg.classify("Send us a message", "Button"), sg.SEND)

    def test_automation_id_names_the_control(self):
        for aid, kind in (("btnSendMessage", sg.SEND), ("submit-button",
                          sg.SUBMIT), ("PlaceOrder", sg.ORDER),
                          ("send_button", sg.SEND), ("btnPayNow", sg.PAY)):
            self.assertEqual(sg.classify("", "Button", aid), kind, aid)
        for aid in ("SendToBack", "searchBox", "btnNext", "PostOfficeLink",
                    "navHome"):
            self.assertEqual(sg.classify("", "Button", aid), "", aid)

    def test_a_generic_ok_is_judged_by_the_words_around_it(self):
        self.assertEqual(sg.classify("OK", "Button", nearby=[
            "Delete File", "Are you sure you want to permanently delete "
            "this file?"]), sg.DELETE)
        self.assertEqual(sg.classify("Continue", "Button", nearby=[
            "Place your order"]), sg.ORDER)
        self.assertEqual(sg.classify("Yes", "Button", nearby=[
            "This cannot be undone"]), sg.DELETE)
        self.assertEqual(sg.classify("OK", "Button", nearby=[
            "Settings saved"]), "")
        self.assertEqual(sg.classify("Next", "Button", nearby=[
            "Place your order"]), "")
        self.assertEqual(sg.classify("Open", "Button", nearby=[
            "Delete this file", "cannot be undone"]), "")

    def test_the_line_is_one_short_honest_sentence(self):
        line = sg.held_text(sg.hold_line(sg.SEND))
        self.assertEqual(line, "That button sends the message, sir - say "
                               "'yes, send it' and I'll click it.")
        for kind in BLOCKED:
            text = sg.held_text(sg.hold_line(kind))
            self.assertEqual(text.count("."), 1, text)
            self.assertIn("sir", text)
            self.assertIn("I'll click it", text)
        self.assertTrue(sg.hold_line(sg.PAY).startswith(
            fm.TERMINAL_FAILURE_PREFIX))

    def test_enter_that_would_send(self):
        w = sg.enter_would_send
        self.assertEqual(w("enter", ["#general | Friends - Discord"]), sg.SEND)
        self.assertEqual(w("return", ["Inbox (3) - WhatsApp"]), sg.SEND)
        self.assertEqual(w("ctrl+enter", ["New Message - Mail"]), sg.SEND)
        self.assertEqual(w("enter", ["Re: lunch - Message (HTML)"]), sg.SEND)
        self.assertEqual(w("enter", ["Chat - Notes"], [
            "the screen shows a message box with a cursor"]), sg.SEND)
        # a new line, another key, another window: untouched
        self.assertEqual(w("shift+enter", ["Discord"]), "")
        self.assertEqual(w("alt+enter", ["Discord"]), "")
        self.assertEqual(w("tab", ["Discord"]), "")
        self.assertEqual(w("volumeup", ["Discord"]), "")
        self.assertEqual(w("enter", ["Untitled - Notepad"]), "")
        self.assertEqual(w("enter", ["Signal processing - Wikipedia"]), "")
        self.assertEqual(w("enter", ["Search - Google Chrome"]), "")

    def test_only_an_explicit_yes_answers(self):
        for ok in ("yes", "Yes.", "yeah go ahead", "yes, send it",
                   "yes send it", "Yes, pay it, sir", "go ahead", "do it",
                   "okay, delete it", "yes please"):
            self.assertTrue(sg.is_yes_answer(ok), ok)
        for no in ("", "no", "please", "right", "correct", "that one",
                   "one", "the first one", "yes send it to dana instead",
                   "yes but not now", "send it", "maybe"):
            self.assertFalse(sg.is_yes_answer(no), no)


# ── the click path ──────────────────────────────────────────────────────────
def _patch_cfg(test, **values):
    for k, v in values.items():
        p = mock.patch.object(cfg, k, v, create=True)
        p.start()
        test.addCleanup(p.stop)


class _Desk(unittest.TestCase):
    def setUp(self):
        G.reset_state()
        self.addCleanup(G.reset_state)
        self.td = tempfile.mkdtemp(prefix="hardstop_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        _patch_cfg(self, MONITORS=F.MONITORS, CLICK_VERIFY_TIMEOUT_S=0.3,
                   VISION_TRACE="off", SCREEN_UIA_ENABLED=True,
                   VISION_GROUNDING_FORMAT="box2d",
                   SCREENSHOT_PRIVACY_BLOCKLIST=["bankingsite"])
        from core import opened_ledger as L
        L.reset()
        self.addCleanup(L.reset)
        out = mock.patch("builtins.print")
        out.start()
        self.addCleanup(out.stop)

    def desk(self, labels, on_click="nothing", **kw):
        """A fake desktop whose one page holds buttons named ``labels``,
        each (name, role) or just a name (a Button), side by side."""
        win = F.FakeWindow(101, "console_light", "middle")
        base = F.page_elements("console_light", 0, 0)[0][0]
        els = []
        for i, item in enumerate(labels):
            name, role = (item, "Button") if isinstance(item, str) else item
            els.append(base._replace(
                name=name, ctype=role, href="", invokable=True,
                in_document=True, offscreen=False, occluded=False,
                rect=(1000.0 + 190.0 * i, 300.0, 160.0, 32.0),
                ref=("hs", i)))
        win.elements_override = els
        return F.FakeBackend([win], on_click=on_click, **kw)


class HardStopClickTests(_Desk):
    def test_every_class_stops_asks_and_clicks_nothing(self):
        for kind, labels in BLOCKED.items():
            for label in labels:
                if label.startswith("&"):
                    continue
                G.reset_state()
                b = self.desk([label])
                r = G.run(label, said=f"click {label}", backend=b)
                with self.subTest(kind=kind, label=label):
                    self.assertEqual(r.outcome, G.ASKED, r.text)
                    self.assertEqual(b.clicks, [])
                    self.assertEqual(b.invokes, [])
                    self.assertEqual(b.hotkeys, [])
                    self.assertTrue(r.text.startswith(
                        fm.TERMINAL_FAILURE_PREFIX))
                    self.assertIn("I'll click it", r.text)
                    p = G.pending_choice()
                    self.assertTrue(p["hard"])
                    self.assertEqual(p["hold"], kind)
                    self.assertTrue(p["allow_yes"])

    def test_the_headline_sentence(self):
        b = self.desk(["Send"])
        r = G.run("send", said="send it", backend=b)
        self.assertEqual(sg.held_text(r.text),
                         "That button sends the message, sir - say "
                         "'yes, send it' and I'll click it.")

    def test_ordinary_buttons_are_clicked(self):
        for label in ("Refresh", "Next page", "Search", "Back", "Settings"):
            G.reset_state()
            b = self.desk([label])
            r = G.run(label, said=f"click {label}", backend=b)
            with self.subTest(label=label):
                self.assertNotEqual(r.outcome, G.ASKED, r.text)
                self.assertIsNone(G.pending_choice())
                self.assertEqual(len(b.clicks), 1)

    def test_send_to_back_menu_item_is_clicked(self):
        b = self.desk([("Send to back", "MenuItem")])
        r = G.run("send to back", said="click send to back", backend=b)
        self.assertNotEqual(r.outcome, G.ASKED, r.text)
        self.assertEqual(len(b.clicks), 1)

    def test_a_yes_approves_exactly_one_click(self):
        b = self.desk(["Place order"])
        r = G.run("place order", said="click place order", backend=b)
        self.assertEqual(r.outcome, G.ASKED)
        self.assertEqual(b.clicks, [])
        r2 = G.pick(1, said="yes, place it", backend=b)
        self.assertNotEqual(r2.outcome, G.ASKED, r2.text)
        self.assertEqual(len(b.clicks), 1)
        self.assertIsNone(G.pending_choice())
        # the same yes again clicks nothing: there is no question open
        r3 = G.pick(1, said="yes", backend=b)
        self.assertEqual(len(b.clicks), 1)
        self.assertIn("not waiting", r3.text)

    def test_a_second_click_needs_a_new_yes(self):
        b = self.desk(["Delete account"])
        G.run("delete account", said="click delete account", backend=b)
        G.pick(1, said="yes", backend=b)
        self.assertEqual(len(b.clicks), 1)
        again = G.run("delete account", said="click delete account again",
                      backend=b)
        self.assertEqual(again.outcome, G.ASKED, again.text)
        self.assertEqual(len(b.clicks), 1)          # held again, not clicked
        G.pick(1, said="yes", backend=b)
        self.assertEqual(len(b.clicks), 2)

    def test_his_own_words_do_not_waive_the_stop(self):
        b = self.desk(["Send"])
        r = G.run("send", said="Jarvis, click send, I want to send it now",
                  backend=b)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertEqual(b.clicks, [])

    def test_a_yes_for_a_covered_control_carries_one_action(self):
        # the control is behind another app: bringing it forward is part of
        # the SAME confirmed click, and says so - not a second hold
        win = F.FakeWindow(101, "console_light", "middle")
        note = F.FakeWindow(300, None, "middle", title="Untitled - Notepad",
                            process="notepad.exe", rect=(900, 280, 800, 120))
        b = self.desk(["Pay now"])
        b.wins.insert(0, note)
        del win
        r = G.run("pay now", said="click pay now", backend=b)
        self.assertEqual(r.outcome, G.ASKED)
        self.assertEqual(G.pending_choice()["kind"], "confirm")
        r2 = G.pick(1, said="yes", backend=b)
        self.assertIn("bring it forward and press it", r2.text)
        self.assertEqual(b.clicks, [])
        self.assertTrue(G.pending_choice()["confirmed"])

    def test_the_automation_id_alone_holds_a_control(self):
        b = self.desk(["Proceed"])
        b.wins[0].elements_override = [
            e._replace(auto_id="btnSendMessage")
            for e in b.wins[0].elements_override]
        r = G.run("proceed", said="click proceed", backend=b)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertEqual(b.clicks, [])

    def test_a_generic_ok_next_to_a_delete_warning_is_held(self):
        b = self.desk(["OK", "Are you sure you want to permanently delete "
                       "this file?"])
        b.wins[0].elements_override[1] = b.wins[0].elements_override[1]._replace(
            ctype="Text", invokable=False)
        r = G.run("ok", said="click ok", backend=b)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertEqual(b.clicks, [])

    def test_the_auth_page_guard_still_works(self):
        # a sign-in control is refused by core.auth_guard first (the owner
        # never signed in by JARVIS), and is not a hard-stop question
        b = self.desk(["Sign in"])
        r = G.run("sign in", said="pull the page up so I can sign in",
                  backend=b)
        self.assertEqual(r.outcome, G.REFUSED_AUTH, r.text)
        self.assertEqual(b.clicks, [])
        self.assertIsNone(G.pending_choice())
        # and "Send" on a sign-in page is refused by the page evidence
        chooser = F.FakeWindow(101, "chooser_light", "middle")
        b2 = F.FakeBackend([chooser])
        r2 = G.run("the test user account",
                   said="pull up the console so I can sign in", backend=b2)
        self.assertEqual(r2.outcome, G.REFUSED_AUTH, r2.text)
        self.assertEqual(b2.clicks, [])

    def test_find_mode_never_holds_or_clicks(self):
        b = self.desk(["Send"])
        r = G.run("send", said="find send", mode="find", backend=b)
        self.assertEqual(r.outcome, G.FOUND, r.text)
        self.assertEqual(b.clicks, [])


class YesRoutingTests(_Desk):
    def _route(self, said, hard=True):
        b = self.desk(["Send"])
        G.run("send", said="click send", backend=b)
        p = G.pending_choice()
        if not hard:
            p = dict(p, hard=False)
        return D.screen_route(said, {"pending": p, "allow_yes": True})

    def test_an_explicit_yes_picks_the_one_option(self):
        for said in ("yes", "yes, send it", "go ahead", "do it", "Yes sir"):
            self.assertEqual(self._route(said),
                             "[ACTION: click_on_screen, pick:1]", said)

    def test_a_weak_or_other_answer_does_not(self):
        for said in ("please", "right", "that one", "one", "the first one",
                     "no", "maybe later", "play some jazz"):
            self.assertNotEqual(self._route(said),
                                "[ACTION: click_on_screen, pick:1]", said)

    def test_an_ordinary_question_keeps_its_old_answers(self):
        self.assertEqual(self._route("please", hard=False),
                         "[ACTION: click_on_screen, pick:1]")
        self.assertEqual(
            OR.pending_choice_answer("one", [{"label": "A"}], allow_yes=True),
            0)
        self.assertIsNone(OR.pending_choice_answer(
            "one", [{"label": "A"}], allow_yes=True, hard_yes=True))


class PointClickTests(_Desk):
    def test_a_coordinate_click_on_a_send_button_is_held(self):
        b = self.desk(["Send message"])
        r = G.guard_point(1100, 316, said="click there", backend=b)
        self.assertIsNotNone(r)
        self.assertEqual(r.outcome, G.ASKED)
        self.assertEqual(b.clicks, [])
        r2 = G.pick(1, said="yes, send it", backend=b)
        self.assertEqual(b.clicks, [(1100, 316)])
        self.assertEqual(r2.outcome, G.VERIFIED, r2.text)
        r3 = G.pick(1, said="yes", backend=b)
        self.assertEqual(len(b.clicks), 1)
        self.assertIn("not waiting", r3.text)

    def test_a_coordinate_click_on_an_ordinary_control_passes(self):
        b = self.desk(["Open"])
        self.assertIsNone(G.guard_point(1100, 316, backend=b))
        self.assertIsNone(G.pending_choice())
        # nothing under the point: nothing can be told
        self.assertIsNone(G.guard_point(10, 10, backend=F.FakeBackend([])))

    def test_the_control_moving_away_cancels_the_click(self):
        b = self.desk(["Send message"])
        G.guard_point(1100, 316, backend=b)
        b.wins[0].elements_override = [
            e._replace(name="Open") for e in b.wins[0].elements_override]
        r = G.pick(1, said="yes", backend=b)
        self.assertEqual(b.clicks, [])
        self.assertIn("any more", r.text)


class KeyHoldTests(_Desk):
    def test_the_hold_presses_nothing_then_one_yes_presses_once(self):
        b = F.FakeBackend([F.FakeWindow(202, None, "middle",
                                        title="#general - Discord",
                                        process="discord.exe")], fg=202)
        r = G.hold_key("enter", said="send it", backend=b)
        self.assertEqual(r.outcome, G.ASKED)
        self.assertIn("Enter would send that message, sir", r.text)
        self.assertEqual(b.hotkeys, [])
        r2 = G.pick(1, said="yes, send it", backend=b)
        self.assertEqual(b.hotkeys, [("enter",)])
        self.assertEqual(r2.outcome, G.VERIFIED, r2.text)
        r3 = G.pick(1, said="yes", backend=b)
        self.assertEqual(b.hotkeys, [("enter",)])
        self.assertIn("not waiting", r3.text)

    def test_the_window_changing_cancels_the_press(self):
        b = F.FakeBackend([F.FakeWindow(202, None, "middle",
                                        title="#general - Discord"),
                           F.FakeWindow(203, None, "middle",
                                        title="Notes")], fg=202)
        G.hold_key("enter", backend=b)
        b.fg = 203
        r = G.pick(1, said="yes", backend=b)
        self.assertEqual(b.hotkeys, [])
        self.assertIn("has changed", r.text)


class _ActionBase(_Desk):
    def setUp(self):
        super().setUp()
        self.bc = mock.Mock()
        self.bc._parse_monitor_prefix.side_effect = lambda a: (None, a)
        self.bc._is_self_close_attempt.return_value = False
        self.bc._turn_user_text.return_value = "send it"
        self.bc._turn_screen_texts.return_value = []
        self.bc._turn_click_targets.return_value = []
        self.bc._turn_auth_refused.return_value = False
        self.bc._read_focused_window.return_value = (
            202, "#general - Discord", None)
        self.bc._looks_like_shell_command.return_value = False
        self.bc._normalize_key.side_effect = lambda k: k.strip().lower()
        self.bc.ui_press.side_effect = AssertionError("pressed")
        self.bc.ui_hotkey.side_effect = AssertionError("hotkey")
        self.bc.ui_click.side_effect = AssertionError("clicked")
        self.bc.ui_type.side_effect = AssertionError("typed")
        p = mock.patch.object(A, "_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)
        self.fake = F.FakeBackend([F.FakeWindow(
            202, None, "middle", title="#general - Discord",
            process="discord.exe")], fg=202)
        G.set_default_backend(self.fake)
        self.addCleanup(G.set_default_backend, None)


class KeyActionTests(_ActionBase):
    def test_enter_in_a_chat_asks_instead_of_sending(self):
        out = A._act_press("enter")
        self.assertIn("Enter would send that message, sir", out)
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX))
        self.bc.ui_press.assert_not_called()
        self.assertEqual(self.fake.hotkeys, [])
        self.assertEqual(G.pending_choice()["kind"], "press_key")

    def test_ctrl_enter_hotkey_asks_too(self):
        out = A._act_hotkey("ctrl+enter")
        self.assertIn("Enter would send", out)
        self.bc.ui_hotkey.assert_not_called()

    def test_shift_enter_and_other_keys_and_windows_go_through(self):
        self.bc.ui_hotkey.side_effect = None
        self.assertEqual(A._act_hotkey("shift+enter"), "pressed shift+enter")
        self.bc.ui_press.side_effect = None
        self.assertEqual(A._act_press("volumeup"), "pressed volumeup")
        self.bc._read_focused_window.return_value = (1, "Search - Chrome", None)
        self.assertEqual(A._act_press("enter"), "pressed enter")

    def test_typing_is_not_held(self):
        self.bc.ui_type.side_effect = None
        self.assertEqual(A._act_type("hello there"), "typed: hello there")

    def test_the_yes_presses_once_through_the_screen_route(self):
        A._act_press("enter")
        tok = D.screen_route("yes, send it", {
            "pending": G.pending_choice(), "allow_yes": True})
        self.assertEqual(tok, "[ACTION: click_on_screen, pick:1]")
        out = A._act_click_on_screen("pick:1")
        self.assertIn("pressed Enter", out)
        self.assertEqual(self.fake.hotkeys, [("enter",)])
        # nothing left to say yes to
        self.assertIsNone(D.screen_route("yes", {
            "pending": G.pending_choice(), "allow_yes": True}))

    def test_a_screen_look_that_saw_a_message_box_holds_enter(self):
        self.bc._read_focused_window.return_value = (1, "Chrome", None)
        self.bc._turn_screen_texts.return_value = [
            "the page has a message box with the cursor in it"]
        self.assertIn("Enter would send", A._act_press("enter"))


class ClickActionTests(_ActionBase):
    def test_a_coordinate_click_on_a_send_control_is_held(self):
        self.fake.wins[0].page = "console_light"
        base = F.page_elements("console_light", 0, 0)[0][0]
        self.fake.wins[0].elements_override = [base._replace(
            name="Send", ctype="Button", href="", invokable=True,
            rect=(1000.0, 300.0, 160.0, 32.0), ref=("hs", 0))]
        out = A._act_click("1050, 316")
        self.assertIn("That button sends the message, sir", out)
        self.bc.ui_click.assert_not_called()
        self.assertEqual(self.fake.clicks, [])

    def test_a_right_click_and_a_blind_coordinate_click_go_through(self):
        self.bc.ui_click.side_effect = None
        self.assertEqual(A._act_click("50, 60"), "clicked left at (50,60)")
        self.assertEqual(A._act_click("50, 60, right"),
                         "clicked right at (50,60)")

    def test_the_cloud_eye_pixel_fallback_does_not_click_a_send_button(self):
        from skills import local_vision as lv
        with mock.patch.object(lv, "_bobert", return_value=self.bc), \
             mock.patch("core.grounded_click.run_bounded",
                        side_effect=RuntimeError("down")), \
             mock.patch.object(lv, "_find_click_target_local",
                               side_effect=AssertionError("looked")):
            out = lv.local_click_target_by_description("the Send button")
        self.assertIn("I haven't clicked", out)
        self.bc.ui_click.assert_not_called()


class UiaAutomationIdTests(unittest.TestCase):
    def test_el_carries_an_automation_id_default_empty(self):
        el = El(name="x", ctype="Button", rect=(0, 0, 1, 1))
        self.assertEqual(el.auto_id, "")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
