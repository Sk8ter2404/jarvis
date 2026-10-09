"""The 2026-10-05 review of claude/screen-vision, finding by finding, against
FAKE desktops (tests/_screen_fakes: synthetic pages in fake windows; no real
window, UIA, OCR, pointer, model or screen). Each test failed on the
reviewed head (7d52f87) and passes with the fix.

    python -m unittest tests.test_screen_review_fixes
"""
from __future__ import annotations

import difflib
import os
import random
import threading
import time
import types
import unittest
from unittest import mock

from core import config as cfg
from core import grounded_click as G
from core import onscreen_refs as OR
from core import screen_privacy as P
from core import screen_resolve as R
from core.screen_text import El
from tests import _screen_fakes as F
from tests.test_grounded_click import _Base, _patch_cfg

_BANK_URL = "https://secure.chase.com/web/auth/dashboard#/dashboard/overview"
_BANK_TITLE = "Accounts Overview - Google Chrome"


def _trace_entries(td):
    from core import vision_trace as VT
    VT.flush(5)
    return VT.read_index()


def _trace_images(td):
    out = []
    for root, _d, files in os.walk(os.path.join(td, "vision_trace")):
        out += [f for f in files if f.endswith((".webp", ".png", ".jpg"))]
    return out


# ── 1 + 2: pages private by their ADDRESS ───────────────────────────────
class AddressPrivacyClickTests(_Base):
    def setUp(self):
        super().setUp()
        from core import vision_trace as VT
        _patch_cfg(self, VISION_TRACE="on")
        VT._reset_for_tests()
        self.addCleanup(VT._reset_for_tests)

    def test_a_bank_known_only_by_its_address_is_never_ocrd_spoken_or_traced(
            self):
        # snapshot None (no page fixture) -> the OCR fallback; the bank is
        # private only by its address.
        bank = F.FakeWindow(104, None, "middle", title=_BANK_TITLE,
                            url=_BANK_URL)
        b = self.desk(bank, fg=104, ocr_lines=[
            {"t": "Total balance 4,102.83", "rect": [10, 200, 300, 20]}])
        r = G.run("the transfer button", said="click the transfer button",
                  backend=b)
        self.assertNotIn("4,102.83", r.text)
        self.assertEqual(b.captures, 0)
        self.assertEqual(b.clicks, [])
        self.assertEqual(_trace_images(self.td), [])
        for e in _trace_entries(self.td):
            self.assertNotIn("4,102.83", str(e))

    def test_an_unreadable_address_is_judged_by_the_address_bar_ocr(self):
        # UIA gave no address; OCR reads the address bar (y 60 < toolbar 88)
        # - a bank host there makes the whole window private.
        class NoUrl(F.FakeBackend):
            def read_url(self, hwnd):
                return None
        bank = F.FakeWindow(104, None, "middle", title=_BANK_TITLE, url="")
        b = NoUrl([bank], fg=104, ocr_lines=[
            {"t": "secure.chase.com/web/auth/dashboard",
             "rect": [200, 50, 400, 20]},
            {"t": "Total balance 4,102.83", "rect": [10, 300, 300, 20]}])
        r = G.run("the transfer button", said="click the transfer button",
                  backend=b)
        self.assertNotIn("4,102.83", r.text)
        self.assertEqual(b.clicks, [])
        self.assertEqual(_trace_images(self.td), [])

    def test_a_capture_masks_a_bank_tab_above_its_target(self):
        target = F.FakeWindow(101, "home_dark", "middle").win(1)
        bank = F.FakeWindow(104, None, "middle", title=_BANK_TITLE,
                            rect=(100, 100, 800, 600)).win(0)
        infos = G._capture_infos([bank, target], (0, 0, 2560, 1440), 101,
                                 lambda h: _BANK_URL if h == 104 else "")
        by = {i["hwnd"]: i["private"] for i in infos}
        self.assertTrue(by[104])
        self.assertFalse(by[101])
        # an unreadable address ABOVE the target fails closed
        infos = G._capture_infos([bank, target], (0, 0, 2560, 1440), 101,
                                 lambda h: "")
        self.assertTrue({i["hwnd"]: i["private"] for i in infos}[104])


class AddressPrivacyDigestTests(_Base):
    def test_the_digest_never_ocrs_a_bank_known_only_by_its_address(self):
        from core import screen_digest as SD
        bank = F.FakeWindow(104, None, "middle", title=_BANK_TITLE,
                            url=_BANK_URL)
        b = self.desk(bank, fg=104, ocr_lines=[
            {"t": "Total balance 4,102.83", "rect": [10, 200, 300, 20]}])
        out = SD.digest("window", hwnd=104, backend=b)
        self.assertNotIn("4,102.83", out["text"])
        self.assertEqual(out["private"], 1)
        self.assertEqual(b.captures, 0)

    def test_the_window_list_hides_a_bank_tab_by_its_address(self):
        from core import screen_digest as SD
        bank = F.FakeWindow(104, None, "middle", title=_BANK_TITLE,
                            url=_BANK_URL)
        home = F.FakeWindow(101, "home_dark", "top")
        b = self.desk(bank, home, fg=101)
        out = SD.digest("overview", backend=b)
        self.assertNotIn("Accounts Overview", out["text"])
        self.assertIn("(a private window)", out["text"])

    def test_a_whole_screen_look_names_the_private_window(self):
        bank = F.FakeWindow(104, None, "right", title=_BANK_TITLE).win(0)
        home = F.FakeWindow(101, "home_dark", "middle").win(1)
        self.assertTrue(P.visible_private(
            [home, bank], url_of=lambda h: _BANK_URL if h == 104 else ""))
        self.assertIsNone(P.visible_private(
            [home], url_of=lambda h: "https://videosite.example/"))


class AddressPrivacyMemoryTests(unittest.TestCase):
    """Screen memory (core.screen_memory) with a bank tab whose title says
    nothing - the review's repro."""

    def _run(self, env_cls):
        from tests.test_screen_memory import _Base as MB, FakeEnv, _thumb
        outer = self

        class T(MB):
            def runTest(self):
                bank = F.FakeWindow(101, "home_dark", "middle",
                                    title=_BANK_TITLE, url=_BANK_URL)
                env = env_cls(FakeEnv)([bank])
                w = self.watcher(env)
                self.tick(w)
                env.input_ms += 1
                env.thumbs["middle"] = _thumb(block=(20, 10, 90, 60))
                self.tick(w)
                outer.rows = self.tl.query()
                outer.snaps = list(env.snapshots)
                outer.captures = env.captures
        res = unittest.TestResult()
        T().run(res)
        self.assertEqual(res.errors + res.failures, [])

    def test_a_bank_known_only_by_its_address_is_never_stored(self):
        self._run(lambda base: base)
        self.assertEqual(self.snaps, [])
        self.assertEqual(self.captures, 0)
        for r in self.rows:
            self.assertNotIn("chase", str(r).lower())
            self.assertNotIn("Accounts Overview", str(r))

    def test_a_browser_whose_address_cannot_be_read_is_skipped(self):
        def no_url(base):
            class E(base):
                def read_url(self, hwnd):
                    return ""
            return E
        self._run(no_url)
        self.assertEqual(self.snaps, [])
        self.assertEqual(self.rows, [])


# ── 3: "already playing" ────────────────────────────────────────────────
class AlreadyPlayingTests(_Base):
    def _layout(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        playing = F.FakeWindow(102, None, "top", title=(
            "MrBeast Finally Did It - YouTube - Google Chrome"))
        return home, playing

    def test_a_matching_card_on_the_page_beats_a_playing_title(self):
        home, playing = self._layout()
        b = self.desk(playing, home, fg=101)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        self.assertEqual(r.monitor, "middle")
        self.assertEqual(len(b.clicks), 1)

    def test_the_named_monitor_filters_what_is_playing(self):
        home, playing = self._layout()
        b = self.desk(playing, home, fg=101)
        r = G.run("that MrBeast video",
                  said="click that MrBeast video on the middle monitor",
                  backend=b)
        self.assertNotIn("top monitor", r.text)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)

    def test_the_title_match_names_only_the_named_monitors_player(self):
        playing = [("MrBeast Finally Did It", "top")]
        self.assertIsNone(G._already_playing("that MrBeast video", playing,
                                             hard_monitor="middle"))
        self.assertEqual(G._already_playing("the MrBeast Finally Did It "
                                            "video", playing),
                         ("MrBeast Finally Did It", "top"))


# ── 4: redirects; 5: other apps' windows ─────────────────────────────────
class _Redirect(F.FakeBackend):
    landing = "/latest/"

    def _effect(self, w, el):
        if el is None:
            return
        w.history.append((w.url, w.title, w.page))
        w.url = el.href.rstrip("/") + self.landing
        # a title that does NOT name the link: only the address can tell
        w.title = "Getting Started - Docs - Google Chrome"


class VerifyTests(_Base):
    def _docs(self, backend_cls, href="https://example.org/docs/install"):
        home = F.FakeWindow(101, None, "middle",
                            title="Project Home - Google Chrome",
                            url="https://example.org/")
        x, y = home.rect[0] + 200, home.rect[1] + 300
        home.page = "home_dark"
        home.elements_override = [El(
            name="Installation guide", ctype="Hyperlink",
            rect=(x, y, 220, 24), href=href, invokable=True,
            in_document=True, ref=("x", 1))]
        return home, backend_cls([home], fg=101)

    def test_a_redirect_inside_the_site_is_the_right_page(self):
        home, b = self._docs(_Redirect)
        r = G.run("the installation guide link",
                  said="click the installation guide link", backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        self.assertEqual(b.backs, [])
        self.assertTrue(home.url.endswith("/docs/install/latest/"))

    def test_a_hop_to_another_site_is_reported_not_undone(self):
        class Shortener(F.FakeBackend):
            def _effect(self, w, el):
                w.history.append((w.url, w.title, w.page))
                w.url = "https://cdn.other.example/landing"
                w.title = "Landing - Google Chrome"
        home, b = self._docs(Shortener)
        r = G.run("the installation guide link",
                  said="click the installation guide link", backend=b)
        self.assertEqual(r.outcome, G.NAVIGATED, r.text)
        self.assertEqual(b.backs, [])

    def test_the_same_sites_other_page_is_still_wrong(self):
        self.assertEqual(G._landing("example.org/blog", "example.org/docs"),
                         "wrong")
        self.assertEqual(G._landing("example.org/docs/install/latest",
                                    "example.org/docs/install"), "redirect")
        self.assertEqual(G._landing("example.org/en-us/docs/install",
                                    "example.org/docs/install"), "redirect")

    def test_another_apps_new_window_is_not_success_and_never_closed(self):
        class Popup(F.FakeBackend):
            # the click does nothing; another app's window appears meanwhile
            def _popup(self):
                if not any(w.hwnd == 777 for w in self.wins):
                    self.wins.insert(0, F.FakeWindow(
                        777, None, "right", title="Mom - Messenger",
                        process="Messenger.exe"))

            def click(self, x, y):
                self.clicks.append((int(x), int(y)))
                self._popup()

            def invoke(self, el):
                self.invokes.append(el.name if el is not None else None)
                self._popup()
                return True
        home = F.FakeWindow(101, "home_dark", "middle")
        b = Popup([home], fg=101, on_click="nothing")
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertNotEqual(r.outcome, G.VERIFIED, r.text)
        G.undo(False, "go back", backend=b)
        self.assertNotIn(777, b.closed)


    def test_another_apps_window_after_a_button_click_is_not_success(self):
        # Not a video, so any new window of the CLICKED app would count -
        # a Messenger window appearing meanwhile must not.
        class Popup(F.FakeBackend):
            def _popup(self):
                if not any(w.hwnd == 777 for w in self.wins):
                    self.wins.insert(0, F.FakeWindow(
                        777, None, "right", title="Mom - Messenger",
                        process="Messenger.exe"))

            def click(self, x, y):
                self.clicks.append((int(x), int(y)))
                self._popup()

            def invoke(self, el):
                self.invokes.append(el.name if el is not None else None)
                self._popup()
                return True
        home = F.FakeWindow(101, "home_dark", "middle")
        x, y = home.rect[0] + 300, home.rect[1] + 300
        home.elements_override = [El(
            name="Open settings", ctype="Button", rect=(x, y, 160, 30),
            invokable=True, in_document=True, ref=("home_dark", 9998))]
        b = Popup([home], fg=101, on_click="nothing")
        r = G.run("the open settings button",
                  said="click the open settings button", backend=b)
        self.assertNotEqual(r.outcome, G.VERIFIED, r.text)
        G.undo(False, "go back", backend=b)
        self.assertNotIn(777, b.closed)

    def test_undo_closes_only_the_clicked_apps_own_windows(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        mine = F.FakeWindow(901, None, "middle", title="Video - Chrome")
        other = F.FakeWindow(777, None, "right", title="Mom - Messenger",
                             process="Messenger.exe")
        b = self.desk(mine, other, home, fg=101)
        G.note_ui_action({"kind": "click", "hwnd": 101, "monitor": "middle",
                          "label": "x", "referent": "x", "url_before": "",
                          "url_after": "", "title_before": home.title,
                          "tabs_before": None, "tops_before": [101],
                          "new_hwnds": [901, 777], "tabs_after": None,
                          "outcome": G.VERIFIED, "options": [], "said": "",
                          "process": "chrome.exe"})
        G.undo(False, "go back", backend=b)
        self.assertIn(901, b.closed)
        self.assertNotIn(777, b.closed)


# ── 6: "go back" after the owner moved on ───────────────────────────────
class GoBackTests(_Base):
    def _opened(self, title_now):
        from core import opened_ledger as L
        yt = F.FakeWindow(101, "home_dark", "middle",
                          url="https://www.youtube.com/watch?v=abc",
                          title=title_now)
        L.note_opened("open_url", "https://www.youtube.com", hwnd=101,
                      kind="window", monitor="middle",
                      title="YouTube - Google Chrome",
                      now=time.time() - 40)
        return yt, self.desk(yt, fg=101, ledger=(
            101, "middle", "https://www.youtube.com", "open_url",
            "YouTube - Google Chrome"))

    def test_go_back_leaves_a_page_he_has_navigated_himself(self):
        from core.dispatcher import screen_route
        yt, b = self._opened("Some Video - YouTube - Google Chrome")
        self.assertFalse(G.undoable(b))
        self.assertIsNone(screen_route("go back",
                                       {"recent_ui": G.undoable(b)}))
        G.undo(False, "go back", backend=b)
        self.assertEqual(b.closed, [])
        self.assertTrue(yt.alive)

    def test_a_second_go_back_after_an_undone_click_leaves_the_page(self):
        yt, b = self._opened("YouTube - Google Chrome")
        G.note_ui_action({"kind": "click", "hwnd": 101, "monitor": "middle",
                          "label": "x", "referent": "x",
                          "url_before": "https://www.youtube.com/",
                          "url_after": "https://www.youtube.com/watch?v=abc",
                          "title_before": "YouTube - Google Chrome",
                          "tabs_before": None, "tops_before": [101],
                          "new_hwnds": [], "tabs_after": None,
                          "outcome": G.VERIFIED, "options": [], "said": "",
                          "process": "chrome.exe"})
        G.undo(False, "go back", backend=b)       # Back, inside the page
        G.undo(False, "go back", backend=b)       # nothing more of his
        self.assertEqual(b.closed, [])
        self.assertTrue(yt.alive)

    def test_go_back_right_after_an_open_still_closes_it(self):
        yt, b = self._opened("YouTube - Google Chrome")
        self.assertTrue(G.undoable(b))
        r = G.undo(False, "go back", backend=b)
        self.assertEqual(b.closed, [101], r.text)


# ── 7: the sign-in guard ────────────────────────────────────────────────
class SignInGuardTests(_Base):
    def test_a_sign_in_page_opened_elsewhere_does_not_refuse_this_click(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        login = F.FakeWindow(103, None, "top",
                             title="Claude Console - Google Chrome",
                             url="https://console.anthropic.com/login")
        b = self.desk(home, login, fg=101, ledger=(
            103, "top", "https://console.anthropic.com/login", "open_url"))
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)

    def test_a_turn_that_refused_an_input_refuses_its_next_click(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, fg=101)
        b.adopt_frame({"user_text": "pull up the console",
                       "auth_refused": True})
        self.addCleanup(b.adopt_frame, None)
        r = G.run("that MrBeast video", said="pull up the console",
                  backend=b)
        self.assertEqual(r.outcome, G.REFUSED_AUTH, r.text)
        self.assertEqual(b.clicks, [])

    def test_the_turn_rule_reaches_an_auth_guard_that_takes_it(self):
        seen = {}

        def new_sig(description="", owner_text="", urls=(), titles=(),
                    screen_texts=(), looked_for=(), refused_before=False):
            seen.update(looked_for=list(looked_for),
                        refused_before=refused_before)
            return ""

        calls = []

        def broken(description="", owner_text="", urls=(), titles=(),
                   screen_texts=(), looked_for=(), refused_before=False):
            calls.append(1)
            raise TypeError("an error INSIDE the guard")

        mod = types.ModuleType("core.auth_guard")
        mod.click_refusal = new_sig
        with mock.patch.dict("sys.modules", {"core.auth_guard": mod}):
            G._auth_refusal("x", "click x", looked_for=["Sign in"],
                            refused_before=True)
            self.assertEqual(seen, {"looked_for": ["Sign in"],
                                    "refused_before": True})
            # an error inside the guard is not retried as "an old signature"
            mod.click_refusal = broken
            G._auth_refusal("x", "click x", refused_before=True)
            self.assertEqual(calls, [1])

    def test_a_sign_in_page_answers_only_a_request_that_could_be_on_it(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        chooser = F.FakeWindow(
            103, "chooser_light", "top",
            url="https://accounts.google.com/accountchooser",
            title="Sign in - Google Accounts - Google Chrome")
        b = self.desk(chooser, home, fg=103)
        r = G.run("the dragon video", said="click the dragon video",
                  backend=b)
        self.assertEqual(r.outcome, G.NOT_FOUND, r.text)
        self.assertNotIn("sign-in", r.text)
        r = G.run("the test user account",
                  said="pull up the console so I can sign in", backend=b)
        self.assertEqual(r.outcome, G.REFUSED_AUTH, r.text)
        self.assertEqual(b.clicks, [])


# ── 8: find_on_screen failures read as failures ─────────────────────────
class FindFailureTextTests(_Base):
    def test_not_found_in_find_mode_carries_a_failure_marker(self):
        from core.failure_markers import FAILURE_MARKERS
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, fg=101)
        r = G.run("the dragon picture", said="find the dragon picture",
                  mode="find", backend=b)
        self.assertEqual(r.outcome, G.NOT_FOUND)
        self.assertTrue(any(m in r.text.lower() for m in FAILURE_MARKERS),
                        r.text)


# ── 9: the worker thread runs inside the owner's turn ───────────────────
class TurnFrameTests(_Base):
    def test_the_vision_cap_counts_on_the_click_worker(self):
        _patch_cfg(self, VISION_GROUNDING_FORMAT="box2d")
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, fg=101, vision_ok=True,
                      vision_answers=['{"box_2d": [10, 10, 20, 20]}'] * 4)
        frame = {"user_text": "click the dragon video",
                 "vision_calls": G.MAX_VISION_PER_TURN}
        b.adopt_frame(frame)
        self.addCleanup(b.adopt_frame, None)
        G.run_bounded("the dragon video", said="click the dragon video",
                      backend=b, budget_s=6.0)
        self.assertEqual(b.vision_calls, [])

    def test_the_production_backend_hands_the_frame_to_the_worker(self):
        fake_bc = types.SimpleNamespace(_turn_grounding=threading.local())
        fake_bc._turn_grounding.frame = {"vision_calls": 3,
                                         "screen": ["Sign in to continue"]}
        b = G.Backend()
        seen = {}
        with mock.patch.object(G.Backend, "_bc", return_value=fake_bc):
            frame = b.turn_frame()

            def worker():
                prev = b.adopt_frame(frame)
                seen["v"] = b.turn_vision(1)
                seen["s"] = b.screen_texts()
                b.adopt_frame(prev)
            t = threading.Thread(target=worker)
            t.start()
            t.join(5)
        self.assertEqual(seen, {"v": 4, "s": ["Sign in to continue"]})
        self.assertEqual(frame["vision_calls"], 4)


# ── 10: a worker that outlived its caller ───────────────────────────────
class LateWorkerTests(_Base):
    def test_a_timed_out_worker_leaves_no_question_a_yes_could_act_on(self):
        from core.dispatcher import screen_route

        class Slow(F.FakeBackend):
            def snapshot(self, win, budget_ms=600):
                time.sleep(2.2)              # past run_bounded's join (1.7 s)
                return super().snapshot(win, budget_ms)
        home = F.FakeWindow(101, "home_dark", "middle")
        x, y = home.rect[0] + 300, home.rect[1] + 300
        home.elements_override = [El(
            name="Delete account", ctype="Button", rect=(x, y, 160, 30),
            invokable=True, in_document=True, ref=("home_dark", 9999))]
        b = Slow([home], fg=101, on_click="nothing")
        r = G.run_bounded("the account button",
                          said="click the account button", backend=b,
                          budget_s=0.2)
        self.assertIn("took too long", r.text)
        time.sleep(1.6)                       # the worker finishes late
        p = G.pending_choice()
        self.assertIsNone(p)
        self.assertIsNone(screen_route("yes", {"pending": p,
                                               "allow_yes": True}))
        self.assertEqual(b.clicks, [])
        self.assertEqual(b.invokes, [])

    def test_a_press_already_sent_is_never_reported_as_nothing_clicked(self):
        class SlowVerify(F.FakeBackend):
            def window_title(self, hwnd):
                if self.clicks or self.invokes:
                    time.sleep(2.4)          # confirming outlives the join
                return super().window_title(hwnd)
        home = F.FakeWindow(101, "home_dark", "middle")
        b = SlowVerify([home], fg=101)
        r = G.run_bounded("that MrBeast video", said="click that MrBeast "
                          "video", backend=b, budget_s=0.6)
        self.assertTrue(b.clicks or b.invokes)
        self.assertNotIn("before clicking anything", r.text)
        self.assertIn("I pressed", r.text)
        time.sleep(2.0)                      # let the worker finish
        self.assertEqual(b.backs, [])


# ── 11-13, 15: routing ──────────────────────────────────────────────────
class RoutingTests(unittest.TestCase):
    def test_keys_and_devices_are_not_screen_clicks(self):
        for s in ("press the enter key", "press the space bar",
                  "press the escape key", "press the mute button",
                  "select the USB desk mic",
                  "select the headset as output", "click the F5 key"):
            self.assertIsNone(OR.onscreen_click_target(s), s)
        for s, want in (("click the return policy link",
                         "the return policy link"),
                        ("click the next arrow", "the next arrow"),
                        ("press the subscribe button",
                         "the subscribe button"),
                        ("click that MrBeast video", "that MrBeast video")):
            self.assertEqual(OR.onscreen_click_target(s), want, s)

    def test_stop_watching_the_room_is_not_a_screen_exclusion(self):
        from core.dispatcher import screen_route
        never = lambda name: False                       # noqa: E731
        always = lambda name: True                       # noqa: E731
        for st in ({}, {"watching": False, "app_known": always},
                   {"watching": True, "app_known": never}):
            self.assertIsNone(screen_route("stop watching the room", st), st)
        self.assertEqual(
            screen_route("don't watch Discord",
                         {"watching": True, "app_known": always}),
            "[ACTION: screen_memory, exclude discord]")

    def test_dont_watch_gmail_names_a_browser_tab(self):
        P.clear_exclusions()
        self.addCleanup(P.clear_exclusions)
        P.exclude_app("gmail")
        P.exclude_app("youtube")
        P.exclude_app("bank")
        self.assertTrue(P.excluded({"process": "chrome.exe", "title":
                                    "Inbox (3) - me@x.example - Gmail - "
                                    "Google Chrome"}))
        self.assertTrue(P.excluded({"process": "chrome.exe", "title":
                                    "Some Video - YouTube - Google Chrome"}))
        self.assertTrue(P.excluded({"process": "chrome.exe", "title":
                                    "Home - Google Chrome",
                                    "url": "https://mybank.example.com/"}))
        self.assertIsNone(P.excluded({"process": "chrome.exe", "title":
                                      "Banking basics for kids - Google "
                                      "Chrome",
                                      "url": "https://learn.example.org/"}))
        self.assertTrue(P.app_open("Gmail", [{"process": "chrome.exe",
                                              "title": "Inbox - Gmail - "
                                                       "Google Chrome"}]))
        self.assertFalse(P.app_open("the room", [{"process": "chrome.exe",
                                                  "title": "Inbox - Gmail - "
                                                           "Google Chrome"}]))

    def test_a_search_request_is_never_a_rewrite_referent(self):
        self.assertIsNone(OR.rewrite_referent(
            "search google for how to click a link in python"))
        self.assertEqual(OR.rewrite_referent(
            "click that MrBeast video and then go back into wake word mode"),
            OR.referent_phrase("click that MrBeast video and then go back "
                               "into wake word mode"))

    def test_play_that_video_on_youtube_keeps_its_search(self):
        self.assertEqual(OR.youtube_play_query(
            "play that MrBeast video on YouTube"), "MrBeast")
        self.assertIsNone(OR.youtube_play_query("click that MrBeast video"))

    def test_ask_claude_for_a_task_is_not_a_developer_note(self):
        self.assertIsNone(OR.claude_note("ask Claude to make me a workout "
                                         "plan"))
        self.assertIsNone(OR.claude_note("have Claude review my essay"))
        self.assertTrue(OR.claude_note("ask Claude to fix your clicking"))
        self.assertTrue(OR.claude_note("tell Claude to research the screen "
                                       "vision"))


# ── 14: the UI AUTOMATION section ships only for screen turns ───────────
class RouterTests(unittest.TestCase):
    def _has_ui(self, text):
        from core import prompt_router as PR
        from core.prompts import PC_CONTROL_PROMPT
        _core, sections = PR.split_pc_control(PC_CONTROL_PROMPT)
        inc, _d = PR.select_sections(text, sections)
        return any(n.upper().startswith("UI AUTOMATION") for n in inc)

    def test_unrelated_turns_do_not_get_it(self):
        for s in ("go back to sleep", "go back to the previous song",
                  "go back into wake word mode",
                  "how many clicks did my post get",
                  "start the one hour timer", "play the one by Drake"):
            self.assertFalse(self._has_ui(s), s)

    def test_screen_turns_still_get_it(self):
        for s in ("click that MrBeast video", "go back", "not that one",
                  "play that MrBeast video", "click on the save button"):
            self.assertTrue(self._has_ui(s), s)


# ── 16: UI Automation switches ──────────────────────────────────────────
class UiaSwitchTests(_Base):
    def test_switched_off_means_no_ui_automation_at_all(self):
        from core import uia_host as U
        _patch_cfg(self, SCREEN_UIA_ENABLED=False)
        ran = []
        ok, why = U.call(lambda uia: ran.append(1), timeout_s=0.2)
        self.assertEqual((ok, why), (False, U.SWITCHED_OFF))
        self.assertFalse(U.submit(lambda uia: ran.append(1)))
        self.assertEqual(ran, [])

    def test_other_apps_are_read_only_as_the_setting_allows(self):
        code = F.FakeWindow(105, "home_dark", "left",
                            title="main.py - project - Visual Studio Code",
                            process="Code.exe")
        home = F.FakeWindow(101, "home_dark", "middle")
        for mode, said, fg, want in (
                ("off", "click the run button", 105, {101}),
                ("on_demand", "click the run button", 101, {101}),
                ("on_demand", "click the run button", 105, {101, 105}),
                ("on_demand", "click the run button in code", 101,
                 {101, 105})):
            G.reset_state()
            _patch_cfg(self, SCREEN_UIA_NONBROWSER=mode)
            read = []

            class B(F.FakeBackend):
                def snapshot(self, win, budget_ms=600):
                    read.append(win.hwnd)
                    return super().snapshot(win, budget_ms)
            G.freeze_scene(said, backend=B([code, home], fg=fg))
            self.assertEqual(set(read), want, (mode, said, fg))


# ── 17: cost ────────────────────────────────────────────────────────────
def _old_group_cards(cands):
    """The reviewed head's group_cards, verbatim - the reference."""
    cs = sorted([c for c in cands if c.get("rect")],
                key=lambda c: (c["rect"][1], c["rect"][0]))
    cards = []
    for c in cs:
        x, y, w, h = c["rect"]
        home = None
        for card in cards:
            lx, ly, lw, lh = card[-1]["rect"]
            ov = min(x + w, lx + lw) - max(x, lx)
            if (ov > 0.5 * min(w, lw) and -4 <= y - (ly + lh) <= 18
                    and abs(x - card[0]["rect"][0]) < 60):
                home = card
                break
        if home is None:
            cards.append([c])
        else:
            home.append(c)
    return cards


class CostTests(_Base):
    def test_card_grouping_is_unchanged(self):
        rng = random.Random(1005)
        for _trial in range(60):
            cands = []
            for i in range(rng.randint(5, 220)):
                cands.append({"text": f"t{i}", "rect": [
                    rng.choice([0, 20, 300, 320, 640, 900, 1500]) +
                    rng.randint(-30, 30), rng.randint(0, 2400),
                    rng.randint(40, 400), rng.randint(8, 200)]})
            new = [[id(c) for c in card] for card in R.group_cards(cands)]
            old = [[id(c) for c in card] for card in _old_group_cards(cands)]
            self.assertEqual(new, old)

    def test_token_similarity_is_unchanged(self):
        def old_tok_sim(q, c):              # the reviewed head's, verbatim
            if q == c:
                return 1.0
            if len(q) >= 4 and len(c) >= 4:
                if R._stem(q) == R._stem(c):
                    return 0.9
                r = difflib.SequenceMatcher(None, q, c).ratio()
                if r >= 0.8 and (abs(len(q) - len(c)) <= 1 or r >= 0.9):
                    return r
            if len(c) >= 5 and c in q and len(q) - len(c) >= 3:
                return 0.8
            if len(q) >= 5 and q in c:
                return 0.8
            return 0.0
        rng = random.Random(7)
        letters = "aeinorst"
        R.tok_sim.cache_clear()
        for _ in range(4000):
            q = "".join(rng.choice(letters) for _ in range(rng.randint(1, 11)))
            c = (q[:rng.randint(0, len(q))]
                 + "".join(rng.choice(letters)
                           for _ in range(rng.randint(0, 6))))
            self.assertEqual(R.tok_sim(q, c), old_tok_sim(q, c), (q, c))

    def test_the_scene_look_back_stops_with_the_budget(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, fg=101)
        G.freeze_scene("click the dragon video", backend=b)
        dl = G.Deadline(0.0)
        st = G._vt.NULL_STEP
        with mock.patch.object(G._R, "resolve",
                               wraps=G._R.resolve) as res:
            G._not_found("the dragon video", [home.win()], [], "", b,
                         "click", st, dl)
        self.assertEqual(res.call_count, 0)

    def test_a_link_address_is_one_cross_process_call(self):
        from core import screen_text as ST

        class E:
            calls = []

            def GetCurrentPropertyValue(self, pid):
                self.calls.append(("prop", pid))
                return "https://example.org/x"

            def GetCurrentPattern(self, pid):
                self.calls.append(("pattern", pid))
                raise AssertionError("two calls")
        e = E()
        self.assertEqual(ST._value_of(None, e, single_call=True),
                         "https://example.org/x")
        self.assertEqual([c[0] for c in e.calls], ["prop"])


# ── 18: the production backend's hit test fallback ───────────────────────
class HitTestTests(unittest.TestCase):
    def test_the_production_backend_answers_root_at(self):
        with mock.patch("core.screen_text._root_at", return_value=105) as ra:
            self.assertEqual(G.Backend().root_at(10, 20), 105)
        ra.assert_called_once_with(10, 20)


# ── 19: adverts are not videos ──────────────────────────────────────────
class AdTests(unittest.TestCase):
    @staticmethod
    def _card(x, y, title, extra=()):
        out = [{"text": "", "rect": [x, y, 360, 200], "type": "Hyperlink"},
               {"text": title, "rect": [x, y + 210, 360, 40],
                "type": "Hyperlink"}]
        yy = y + 254
        for t in extra:
            out.append({"text": t, "rect": [x, yy, 300, 18], "type": "Text"})
            yy += 20
        return out

    def test_the_first_video_skips_a_sponsored_card(self):
        cands = (self._card(0, 0, "Feastables MrBeast Bar - Shop Now",
                            ("Sponsored", "feastables.example"))
                 + self._card(400, 0, "I Survived 7 Days In An Abandoned "
                              "City", ("MrBeast", "120M views"))
                 + self._card(800, 0, "Why Every Bridge Has These Weird "
                              "Gaps", ("Veritasium", "9M views")))
        res = R.resolve("the first video", cands)
        self.assertEqual(res["status"], "ok", res)
        self.assertIn("I Survived", R.label_of(res["target"]))
        self.assertTrue(all(not R.is_ad_card(cd)
                            for cd in R.video_cards(R.prepare(cands))))

    def test_a_download_button_under_a_heading_is_not_an_advert(self):
        cands = [{"text": "Python 3.13.0", "rect": [0, 0, 300, 30],
                  "type": "Text"},
                 {"text": "Download", "rect": [0, 40, 120, 30],
                  "type": "Button"},
                 {"text": "Learn more", "rect": [0, 80, 120, 20],
                  "type": "Hyperlink"}]
        res = R.resolve("click the download button", cands)
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(R.label_of(res["target"]), "Download")
        self.assertFalse(R.is_ad_card(cands))


# ── 21 + 22: the tray checkmark; "stop watching" stops every record ─────
class ScreenMemoryStateTests(unittest.TestCase):
    def setUp(self):
        from core import screen_memory as SW
        from tests.test_screen_memory import FakeEnv
        self.SW = SW
        self.w = SW.Watcher(env=FakeEnv([]), clock=time.time)
        self.w.thread = types.SimpleNamespace(is_alive=lambda: True)
        p = mock.patch.dict(SW._singleton, {"w": self.w})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(SW.set_state_publisher, None)
        out = mock.patch("builtins.print")
        out.start()
        self.addCleanup(out.stop)

    def test_every_start_and_stop_reaches_the_tray(self):
        seen = []
        self.SW.set_state_publisher(seen.append)
        self.SW.start()
        self.SW.stop()
        self.assertEqual(seen, [False, True, False])

    def test_stop_watching_stops_the_timeline_and_the_trace_too(self):
        import shutil
        import tempfile
        from core import screen_timeline as T
        from core import vision_trace as VT
        td = tempfile.mkdtemp(prefix="svpause_")
        self.addCleanup(shutil.rmtree, td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": td})
        env.start()
        self.addCleanup(env.stop)
        p = mock.patch.object(cfg, "VISION_TRACE", "on", create=True)
        p.start()
        self.addCleanup(p.stop)
        VT._reset_for_tests()
        self.addCleanup(VT._reset_for_tests)
        self.w.owner_pause_until = time.time() + 600
        tl = mock.Mock()
        tl.add.return_value = True
        with mock.patch.object(T, "get", return_value=tl), \
                mock.patch.dict(T._singleton, {"enabled": True}):
            self.assertFalse(T.add(ts=time.time(), monitor="middle",
                                   title="x", source="click", text="clicked"))
        tl.add.assert_not_called()
        with VT.step("click", utterance="click that video") as st:
            st.model_call("which one?", [], "the second")
            st.finish("verified")
        VT.flush(5)
        e = VT.read_index()[-1]
        self.assertEqual(e.get("privacy"), VT.PAUSED_SKIP)
        self.assertNotIn("utterance", e)


if __name__ == "__main__":
    unittest.main()
