"""core/grounded_click.py - the click executor, against a FAKE desktop.

tests/_screen_fakes.FakeBackend puts the synthetic research pages (real UIA
trees of pages rendered in a throw-away Chrome profile) into fake browser
windows on a synthetic 4-monitor layout; a click changes the fake URL /
title / windows the way a browser would, so verify and undo run for real.
Nothing here touches a real window, UIA, OCR, the pointer or a model.

    python -m unittest tests.test_grounded_click
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from core import config as cfg
from core import grounded_click as G
from tests import _screen_fakes as F


def _patch_cfg(test, **values):
    for k, v in values.items():
        p = mock.patch.object(cfg, k, v, create=True)
        p.start()
        test.addCleanup(p.stop)


class _Base(unittest.TestCase):
    def setUp(self):
        G.reset_state()
        self.addCleanup(G.reset_state)
        self.td = tempfile.mkdtemp(prefix="gclick_")
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

    def desk(self, *windows, **kw):
        return F.FakeBackend(list(windows), **kw)


class ScopeTests(_Base):
    def test_click_that_mrbeast_video_lands_on_the_middle_card_verified(self):
        # The live 00:28:23 layout: YouTube home on the MIDDLE monitor, a
        # video playing on TOP. The card's title link is clicked - not the
        # channel - and the URL proves it.
        home = F.FakeWindow(101, "home_dark", "middle")
        watch = F.FakeWindow(102, "watch_dark", "top")
        b = self.desk(watch, home, fg=102)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        self.assertEqual(r.monitor, "middle")
        self.assertEqual(home.url, "https://videosite.example/watch?v=v01")
        self.assertIn("Playing 'I Survived 7 Days In An Abandoned City' on "
                      "the middle monitor", r.text)
        self.assertEqual(len(b.clicks), 1)

    def test_a_monitor_in_the_owners_words_is_a_hard_filter(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        watch = F.FakeWindow(102, "watch_dark", "top")
        b = self.desk(watch, home, fg=101)
        r = G.run("that MrBeast video",
                  said="click that MrBeast video on the top monitor", backend=b)
        # Nothing on the TOP page is a MrBeast video: no click elsewhere.
        self.assertEqual(r.outcome, G.NOT_FOUND, r.text)
        self.assertEqual(b.clicks, [])
        self.assertIn("top monitor", r.text)

    def test_the_models_monitor_prefix_is_only_a_prior(self):
        # Live 00:30:05: the model said monitor:top, the page was on middle.
        home = F.FakeWindow(101, "home_dark", "middle")
        watch = F.FakeWindow(102, "watch_dark", "top")
        b = self.desk(watch, home, fg=102)
        r = G.run("monitor:top|the burger ranking video",
                  said="click the burger ranking video", backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        self.assertEqual(r.monitor, "middle")

    def test_jarvis_windows_are_never_in_scope(self):
        mine = F.FakeWindow(150, "home_dark", "middle", jarvis=True,
                            title="JARVIS Console")
        b = self.desk(mine)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(b.clicks, [])
        self.assertNotEqual(r.outcome, G.VERIFIED)

    def test_never_the_whole_desktop(self):
        _patch_cfg(self, VISION_GROUNDING_FORMAT="pixel")
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, vision_ok=True)
        calls = []
        b.legacy_find = lambda d, m: calls.append(m) or None
        G.run("the dragon video", said="click the dragon video", backend=b)
        self.assertTrue(calls)
        self.assertNotIn(None, calls)

    def test_at_most_four_snapshots(self):
        wins = [F.FakeWindow(200 + i, "home_dark", m, rect=(
            F.MONITORS[m][0] + i, F.MONITORS[m][1], 1200, 1400))
            for i, m in enumerate(("left", "middle", "right", "top", "left",
                                   "middle"))]
        b = self.desk(*wins)
        G.run("the dragon video", said="click the dragon video on the left "
              "monitor", backend=b)
        self.assertLessEqual(b.snapshots, G.MAX_SNAPSHOTS)


class GuardTests(_Base):
    def test_a_target_behind_another_app_is_not_clicked(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        note = F.FakeWindow(300, None, "middle", title="Untitled - Notepad",
                            process="notepad.exe", rect=(300, 400, 400, 200))
        b = self.desk(note, home)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertIn("behind", r.text)
        self.assertEqual(b.clicks, [])
        # "yes" brings it forward and clicks it
        r2 = G.pick(1, said="yes", backend=b)
        self.assertIn(101, b.focused)
        self.assertEqual(r2.outcome, G.ASKED)   # still covered in the fake

    def test_a_jarvis_window_over_it_means_invoke(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        hud = F.FakeWindow(301, None, "middle", title="JARVIS HUD",
                           process="pythonw.exe", jarvis=True,
                           rect=(300, 400, 400, 200))
        b = self.desk(hud, home)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(b.clicks, [])
        self.assertTrue(b.invokes)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)

    def test_a_private_window_is_refused_and_never_read(self):
        bank = F.FakeWindow(101, "home_dark", "middle",
                            title="Accounts - BankingSite - Google Chrome")
        b = self.desk(bank)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(b.clicks, [])
        self.assertEqual(b.snapshots, 0)
        self.assertNotEqual(r.outcome, G.VERIFIED)

    def test_sign_in_guard_on_the_resolved_label(self):
        # Live 00:14:07: "pull up the console so I can sign in" - JARVIS
        # clicked his account entry on its own. The RESOLVED label (an
        # e-mail address) is what the guard judges.
        calls = []
        stub = types.ModuleType("core.auth_guard")

        def click_refusal(description="", owner_text="", urls=(), titles=(),
                          screen_texts=()):
            calls.append((description, owner_text, list(urls), list(titles)))
            return ("failed (final): The sign-in page is up and ready for "
                    "you, sir") if "@" in description else ""
        stub.click_refusal = click_refusal
        chooser = F.FakeWindow(101, "chooser_light", "middle")
        b = self.desk(chooser)
        with mock.patch.dict(sys.modules, {"core.auth_guard": stub}):
            r = G.run("the test user account",
                      said="pull up the console so I can sign in", backend=b)
        self.assertEqual(r.outcome, G.REFUSED_AUTH, r.text)
        self.assertEqual(b.clicks, [])
        self.assertIn("@", calls[0][0])
        self.assertEqual(calls[0][1], "pull up the console so I can sign in")

    def test_a_sign_in_page_is_never_read_or_clicked(self):
        # Without core.auth_guard (it lands with claude/live-turn-fixes-1005)
        # core.screen_privacy's own sign-in check makes the page private: it
        # is not read, nothing is clicked, and the owner hears the page is
        # his to sign in on.
        chooser = F.FakeWindow(101, "chooser_light", "middle",
                               url="https://accounts.google.com/accountchooser")
        b = self.desk(chooser)
        with mock.patch.dict(sys.modules, {"core.auth_guard": None}):
            r = G.run("the test user account",
                      said="pull up the console so I can sign in", backend=b)
        self.assertEqual(r.outcome, G.REFUSED_AUTH, r.text)
        self.assertIn("leave the signing in to you", r.text)
        self.assertEqual(b.clicks, [])
        self.assertEqual(b.snapshots, 0)

    def test_destructive_label_is_held_for_a_yes(self):
        console = F.FakeWindow(101, "console_light", "middle")
        b = self.desk(console)
        els = [e for e in F.page_elements("console_light", 0, 0)[0]]
        target = next(e for e in els if e.name.lower().startswith("create"))
        console.elements_override = els + [target._replace(
            name="Delete key", rect=(1500.0, 300.0, 120.0, 30.0),
            ref=("x", 999))]
        r = G.run("the delete key button", said="click the red one at the end",
                  backend=b)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertIn("shall I press it", r.text)
        self.assertEqual(b.clicks, [])
        self.assertTrue(G.pending_choice()["allow_yes"])


class VerifyTests(_Base):
    def test_nothing_changed_is_said_honestly_after_one_invoke(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, on_click="nothing")
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.NO_CHANGE)
        self.assertIn("nothing changed", r.text)
        self.assertEqual(len(b.clicks), 1)
        self.assertEqual(len(b.invokes), 1)

    def test_the_wrong_page_is_taken_back(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, on_click="wrong")
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.CHANGED_WRONG, r.text)
        self.assertIn("I've gone back", r.text)
        self.assertEqual(home.url, "https://videosite.example/")
        self.assertIn(101, b.backs)

    def test_a_new_window_is_verified_and_undone(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home, on_click="new_window")
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        new = [w for w in b.wins if w.hwnd != 101]
        self.assertEqual(len(new), 1)
        u = G.undo(said="go back", backend=b)
        self.assertEqual(u.outcome, G.VERIFIED, u.text)
        self.assertFalse(new[0].alive)

    def test_same_tab_navigation_is_undone_with_back(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        G.run("the pizza video", said="click the pizza video", backend=b)
        self.assertIn("watch?v=v02", home.url)
        u = G.undo(said="go back", backend=b)
        self.assertEqual(u.outcome, G.VERIFIED, u.text)
        self.assertEqual(home.url, "https://videosite.example/")
        self.assertIn("Back on", u.text)

    def test_undo_with_nothing_recent(self):
        b = self.desk(F.FakeWindow(101, "home_dark", "middle"))
        self.assertEqual(G.undo(said="go back", backend=b).outcome,
                         G.NOT_FOUND)

    def test_a_slow_back_press_is_not_doubled_with_alt_left(self):
        # Live bench 2026-10-05: Chrome answered the Back press after ~2 s,
        # past the wait. "Unconfirmed" (None) must NOT be followed by
        # Alt+Left - that went back TWO pages.
        import threading
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        G.run("the pizza video", said="click the pizza video", backend=b)
        real_back = b.back

        def slow_back(hwnd):
            threading.Timer(0.3, real_back, args=(hwnd,)).start()
            return None
        b.back = slow_back
        u = G.undo(said="go back", backend=b)
        self.assertEqual(b.hotkeys, [])
        self.assertEqual(u.outcome, G.VERIFIED, u.text)
        self.assertEqual(home.url, "https://videosite.example/")

    def test_a_failed_back_press_falls_back_to_alt_left(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        G.run("the pizza video", said="click the pizza video", backend=b)
        b.back = lambda hwnd: False
        G.undo(said="go back", backend=b)
        self.assertEqual(b.hotkeys, [("alt", "left")])

    def test_an_unconfirmed_invoke_is_verified_not_failed(self):
        # A JARVIS window over the target means invoke; an invoke that is
        # still running when the wait ends may land - verify, never "could
        # not press it".
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        real = b.invoke

        def slow(el):
            real(el)
            return None
        b.invoke = slow
        with mock.patch.object(G, "_hit_test", return_value=("jarvis", None)):
            r = G.run("the pizza video", said="click the pizza video",
                      backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        self.assertEqual(b.clicks, [])


class AlreadyPlayingTests(_Base):
    def test_the_video_already_playing_is_not_clicked(self):
        # Live 00:30:05-00:30:24: two clicks hunted for the video that was
        # already playing on the top monitor.
        playing = F.FakeWindow(102, None, "top", title=(
            "Eat Everything In A Grocery Store, Win $1,000,000 - YouTube - "
            "Google Chrome"))
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(playing, home)
        r = G.run('the video "Eat Everything In A Grocery Store, Win '
                  '$1,000,000"', said="click that video", backend=b)
        self.assertEqual(r.outcome, G.ALREADY_PLAYING, r.text)
        self.assertIn("already playing on the top monitor", r.text)
        self.assertEqual(b.clicks, [])

    def test_a_channel_name_is_not_already_playing(self):
        playing = F.FakeWindow(102, None, "top", title=(
            "Eat Everything In A Grocery Store - YouTube - Google Chrome"))
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(playing, home, now_playing={"title": "Eat Everything In "
                                                  "A Grocery Store",
                                                  "artist": "MrBeast"})
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)


class AskAndPickTests(_Base):
    def test_ambiguous_asks_with_real_options_then_pick(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        r = G.run("the every video", said="click the every video", backend=b)
        self.assertEqual(r.outcome, G.AMBIGUOUS, r.text)
        self.assertIn("Which one, sir?", r.text)
        self.assertEqual(b.clicks, [])
        p = G.pending_choice()
        self.assertGreaterEqual(len(p["options"]), 2)
        second = p["options"][1]["label"]
        r2 = G.run("pick:2", said="the second one", backend=b)
        self.assertEqual(r2.outcome, G.VERIFIED, r2.text)
        self.assertEqual(r2.label, second)
        self.assertIsNone(G.pending_choice())

    def test_find_mode_returns_facts_and_never_clicks(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        r = G.run("subscriptions", said="find subscriptions", mode="find",
                  backend=b)
        self.assertEqual(r.outcome, G.FOUND)
        self.assertIn("found 'Subscriptions' on the middle monitor", r.text)
        self.assertEqual(b.clicks, [])

    def test_not_found_names_what_is_there(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        r = G.run("the dragon video", said="click the dragon video", backend=b)
        self.assertEqual(r.outcome, G.NOT_FOUND)
        self.assertIn("I don't see 'the dragon video'", r.text)
        self.assertIn("On screen:", r.text)
        self.assertEqual(b.clicks, [])


class SceneTests(_Base):
    def test_the_one_that_was_on_screen_comes_from_the_frozen_scene(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        self.assertTrue(G.freeze_scene("click that MrBeast video", backend=b))
        # Something else navigated the page away (youtube_play, a search).
        home.history.append((home.url, home.title, home.page))
        home.url, home.page = "https://videosite.example/watch?v=zz", None
        r = G.run("scene:previous",
                  said="I wanted the one that was on screen at the time",
                  backend=b)
        self.assertEqual(r.outcome, G.NOT_FOUND, r.text)
        self.assertIn("I last saw 'I Survived 7 Days In An Abandoned City'",
                      r.text)
        self.assertIn("shall I open it", r.text)
        self.assertNotIn("Kai Cenat", r.text)
        self.assertEqual(G.pending_choice()["kind"], "open_href")

    def test_scene_back_clicks_it_when_still_visible(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        G.freeze_scene("click that MrBeast video", backend=b)
        r = G.run("scene:previous", said="the one that was on screen",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)

    def test_undo_other_asks_from_the_scene_before_the_action(self):
        # "That's not the right video" after a youtube_play open: close the
        # window JARVIS opened, then ask with the options that were on screen
        # BEFORE that action.
        home = F.FakeWindow(101, "home_dark", "middle")
        opened = F.FakeWindow(102, None, "top", title="MrBeast - YouTube - "
                              "Google Chrome")
        from core import opened_ledger as L
        b = self.desk(opened, home, ledger=(102, "top", "https://www.youtube."
                                            "com/watch?v=zz", "play_streaming"))
        G.freeze_scene("click that MrBeast video and then wake word mode",
                       backend=b)
        time.sleep(0.01)
        L.note_opened("play_streaming", "https://www.youtube.com/watch?v=zz",
                      hwnd=102, kind="window", monitor="top",
                      title=opened.title)
        r = G.undo(other=True, said="that's not the right video", backend=b)
        self.assertEqual(b.closed_last_opened, 1)
        self.assertIn("I Survived 7 Days In An Abandoned City", r.text)
        self.assertNotIn("Kai Cenat", r.text)
        self.assertIsNotNone(G.pending_choice())


class VisionTierTests(_Base):
    def _thin(self, b, win):
        win.elements_override = []
        return b

    def test_set_of_mark_answer_is_the_candidates_own_rect(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        els = F.page_elements("home_dark", 0, 0)[0]
        imgs = [e for e in els if e.ctype == "Hyperlink" and e.rect[3] > 120]
        home.elements_override = imgs[:5]       # thumbnails only, no titles
        b = self.desk(home, vision_ok=True,
                      vision_answers=["[local-vision] 3"])
        r = G.run("the thumbnail with the red car", said="click the thumbnail "
                  "with the red car", backend=b)
        self.assertEqual(len(b.vision_calls), 1)
        self.assertEqual(r.outcome, G.VERIFIED, r.text)
        x, y = b.clicks[0]
        rx, ry, rw, rh = imgs[2].rect
        self.assertTrue(rx <= x <= rx + rw and ry <= y <= ry + rh)

    def test_box2d_two_stages_must_agree(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        home.elements_override = []
        b = self.desk(home, vision_ok=True, vision_answers=[
            '[local-vision] {"box_2d": [300, 100, 340, 200]}',
            '[local-vision] {"box_2d": [900, 900, 950, 990]}'])
        r = G.run("the gear icon", said="click the gear icon", backend=b)
        self.assertEqual(len(b.vision_calls), 2)
        self.assertEqual(r.outcome, G.ASKED, r.text)
        self.assertEqual(b.clicks, [])

    def test_model_says_none(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        home.elements_override = []
        b = self.desk(home, vision_ok=True,
                      vision_answers=["[local-vision] NONE"] * 2)
        r = G.run("the gear icon", said="click the gear icon", backend=b)
        self.assertEqual(r.outcome, G.NOT_FOUND)
        self.assertEqual(b.clicks, [])

    def test_at_most_two_model_calls_per_click(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        els = F.page_elements("home_dark", 0, 0)[0]
        home.elements_override = [e for e in els if e.rect[3] > 120][:4]
        b = self.desk(home, vision_ok=True, vision_answers=[
            "[local-vision] I am not sure", '{"box_2d": [1, 2, 3, 4]}',
            "x", "y"])
        G.run("the logo of a cat", said="click the logo of a cat", backend=b)
        self.assertLessEqual(len(b.vision_calls), G.MAX_VISION_CALLS)

    def test_ocr_tier_reads_a_window_uia_cannot(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        home.elements_override = []
        # core.screen_ocr.ocr_image's shape: lines with their rect (image px
        # = screen px here: the window is at the middle monitor's origin).
        lines = [{"t": ln["t"], "rect": tuple(ln["rect"]), "words": []}
                 for ln in F.PAGES["ocr"]["home_dark"]["lines"]]
        b = self.desk(home, ocr_lines=lines)
        r = G.run("the pizza video", said="click the pizza video", backend=b)
        self.assertEqual(r.tier, "ocr", r.text)
        self.assertEqual(len(b.clicks), 1)


class BudgetTests(_Base):
    def test_a_slow_read_never_clicks_late(self):
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        real = b.snapshot

        def slow(win, budget_ms=600):
            time.sleep(1.2)
            return real(win, budget_ms)
        b.snapshot = slow
        r = G.run_bounded("that MrBeast video", said="click that MrBeast video",
                          backend=b, budget_s=0.3)
        self.assertIn("took too long", r.text)
        time.sleep(2.0)                      # the worker finishes ...
        self.assertEqual(b.clicks, [])       # ... and never clicks

    def test_never_raises(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError("boom")
        r = G.run("anything", said="click anything", backend=Broken())
        self.assertEqual(r.outcome, G.FAILED)


class TraceAndTimelineTests(_Base):
    def test_a_click_is_traced_and_recorded(self):
        _patch_cfg(self, VISION_TRACE="on")
        from core import vision_trace as VT
        from core import screen_timeline as TL
        VT._reset_for_tests()
        home = F.FakeWindow(101, "home_dark", "middle")
        b = self.desk(home)
        r = G.run("that MrBeast video", said="click that MrBeast video",
                  backend=b)
        self.assertEqual(r.outcome, G.VERIFIED)
        VT.flush(5)
        entries = VT.read_index()
        self.assertTrue(any(e.get("step") == "click"
                            and e.get("outcome") == "verified"
                            for e in entries), entries)
        tl = TL.get()
        tl.flush(5)
        rows = tl.query(text="Abandoned City")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["source"], "click")

    def test_a_private_window_is_a_text_only_skip(self):
        _patch_cfg(self, VISION_TRACE="on")
        from core import vision_trace as VT
        VT._reset_for_tests()
        bank = F.FakeWindow(101, "home_dark", "middle",
                            title="Accounts - BankingSite - Google Chrome")
        b = self.desk(bank)
        G.run("that MrBeast video", said="click that MrBeast video", backend=b)
        VT.flush(5)
        for e in VT.read_index():
            self.assertNotIn("BankingSite", str(e))



class NormUrlTests(unittest.TestCase):
    """The address bar shows a local page as "C:/dir/page.html" while its
    link's href is "file:///C:/dir/page.html" - found building the live UIA bench
    (tools/vision_bench/live_uia_bench.py): without this a correct click on
    a local page read as "opened the wrong page" and was undone."""

    def test_file_urls_and_drive_paths_agree(self):
        from core.grounded_click import _norm_url
        a = _norm_url("file:///C:/Pages/My%20Site/watch_f04.html")
        self.assertEqual(a, _norm_url("C:/Pages/My Site/watch_f04.html"))
        self.assertEqual(a, _norm_url("C:\\Pages\\My Site\\watch_f04.html"))
        self.assertNotEqual(a, _norm_url("C:/Pages/My Site/home.html"))

    def test_web_urls_unchanged(self):
        from core.grounded_click import _norm_url
        self.assertEqual(_norm_url("https://www.youtube.com/watch?v=abc&t=3"),
                         "yt:abc")
        self.assertEqual(_norm_url("example.com/a/"), "example.com/a")


if __name__ == "__main__":
    unittest.main()
