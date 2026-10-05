"""Replays of the 2026-10-05 screen-vision turns (session 00:21-00:33)
through the REAL dispatch: the built-in screen routes, the router, the
rewrite guard, the click alias, the verbatim speech and the follow-up loop
run for real; the model's replies are canned and the desktop is FAKE
(tests/_screen_fakes: synthetic research pages in fake browser windows on a
synthetic 4-monitor layout). The owner's words are paraphrased.

  1. "click that Mr. Beast video and then go back into wake word mode" -
     the brain answered youtube_play + wake_word_mode_on; now the middle
     card is clicked (verified) and wake-word mode still turns on;
  2. "that's not the right video" after a youtube_play open - the window it
     opened is closed and he is asked, naming what was on screen BEFORE;
  3. "the one that was on screen at the time" - resolved from the frozen
     scene; "Kai Cenat" (the live invention) never appears;
  4. "pull up the console so I can sign in" with an account chooser - no
     click;
  5. "tell Claude to research the screen vision" - a developer note, no
     web_search, no claimed relay;
  6. see_screen on YouTube quotes titles from the page text, no vision call;
  7. clicking the video already playing - "already playing", no click, no
     loop.

    python -m unittest tests.monolith.test_monolith_screen_vision
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from unittest import mock

from tests import _screen_fakes as F
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base


@requires_monolith
class ScreenVisionReplayTests(_Base):
    def setUp(self):
        super().setUp()
        import core.actions as A
        from core import config as cfg
        from core import grounded_click as G
        from core import opened_ledger as L
        self.G, self.L, self.A = G, L, A
        self.td = tempfile.mkdtemp(prefix="svmono_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        for k, v in (("MONITORS", F.MONITORS), ("CLICK_VERIFY_TIMEOUT_S", 0.3),
                     ("VISION_TRACE", "off"), ("SCREEN_UIA_ENABLED", True),
                     ("VISION_GROUNDING_FORMAT", "box2d"),
                     ("CLICK_ROUTE_ENABLED", True)):
            self._p(cfg, k, v, create=True)
        self._p(self.bc, "MONITORS", F.MONITORS)
        G.reset_state()
        self.addCleanup(G.reset_state)
        L.reset()
        self.addCleanup(L.reset)
        self.home = F.FakeWindow(101, "home_dark", "middle",
                                 url="https://www.youtube.com/",
                                 title="Home - YouTube - Google Chrome")
        self.desk = F.FakeBackend([self.home], fg=101)
        G.set_default_backend(self.desk)
        self.addCleanup(G.set_default_backend, None)
        self._p(A, "_screen_digest_backend", [self.desk])
        self._stub("youtube_play", "playing 'MrBeast' on YouTube")
        self._stub("wake_word_mode_on", "Wake-word mode on, sir.")
        self._stub("web_search", "opened Google search for 'x'")
        self._stub("open_url", "opened https://console.example.com")
        self.vision = self._p(self.bc, "ask_vision", return_value="(vision)")
        self.vision_multi = self._p(self.bc, "ask_vision_multi",
                                    return_value="(vision)")
        self._p(self.bc, "screenshot_privacy_block_reason", return_value=None)

    def _all_spoken(self):
        return " | ".join(self.spoken)

    # 1 ──────────────────────────────────────────────────────────────────
    def test_compound_click_and_wake_word_mode(self):
        self._dispatch(
            "Jarvis, go ahead and click that Mr. Beast video and then go back "
            "into, what's it called, wake word mode.",
            "[intent:confirmation] On it, sir. [ACTION: youtube_play, "
            "MrBeast][ACTION: wake_word_mode_on]")
        self.assertEqual(self.calls["youtube_play"], [])
        self.assertEqual(self.calls["wake_word_mode_on"], [""])
        self.assertEqual(len(self.desk.clicks), 1)
        self.assertEqual(self.home.url,
                         "https://videosite.example/watch?v=v01")
        self.assertIn("Playing 'I Survived 7 Days In An Abandoned City' on "
                      "the middle monitor", self._all_spoken())

    def test_a_whole_click_request_needs_no_brain(self):
        resp = self._p(self.bc, "get_response_with_animation")
        self._quiet(self.bc._run_llm_dispatch, "click that Mr. Beast video")
        resp.assert_not_called()
        self.assertEqual(len(self.desk.clicks), 1)
        self.assertIn("Playing", self._all_spoken())

    def test_a_search_with_nothing_on_screen_still_searches(self):
        self._dispatch("play a lofi mix on YouTube",
                       "[intent:confirmation] [ACTION: youtube_play, lofi mix]")
        self.assertEqual(len(self.calls["youtube_play"]), 1)
        self.assertIn("lofi mix", self.calls["youtube_play"][0])
        self.assertEqual(self.desk.clicks, [])

    def test_a_weak_on_screen_match_keeps_the_brains_search(self):
        # The rewrite guard needs a STRONG match (screen_resolve.REWRITE_MIN
        # 0.75): "that phone review" only half-matches 'The Phone That Fixes
        # Everything (Almost)' (0.49), so the brain's youtube_play stands.
        self._stub("volume_up", "Volume up, sir.")
        self._dispatch("play that phone review video and then turn it up",
                       "[intent:confirmation] [ACTION: youtube_play, phone "
                       "review] [ACTION: volume_up]")
        self.assertEqual(len(self.calls["youtube_play"]), 1)
        self.assertEqual(self.calls["volume_up"], [""])
        self.assertEqual(self.desk.clicks, [])

    # 2 + 3 ──────────────────────────────────────────────────────────────
    def _after_a_youtube_play_open(self):
        self.G.freeze_scene("click that Mr. Beast video and then wake word "
                            "mode", backend=self.desk)
        time.sleep(0.01)
        opened = F.FakeWindow(102, None, "top",
                              title="Eat Everything - YouTube - Google Chrome")
        self.desk.wins.insert(0, opened)
        self.desk.ledger_entry = (102, "top",
                                  "https://www.youtube.com/watch?v=zz",
                                  "play_streaming")
        self.L.note_opened("play_streaming",
                           "https://www.youtube.com/watch?v=zz", hwnd=102,
                           kind="window", monitor="top", title=opened.title)
        return opened

    def test_thats_not_the_right_video_closes_it_and_asks(self):
        opened = self._after_a_youtube_play_open()
        resp = self._p(self.bc, "get_response_with_animation")
        self._quiet(self.bc._run_llm_dispatch,
                    "Jarvis, that's not the right video.")
        resp.assert_not_called()
        self.assertFalse(opened.alive)
        said = self._all_spoken()
        self.assertIn("I Survived 7 Days In An Abandoned City", said)
        self.assertNotIn("Kai Cenat", said)
        self.assertIsNotNone(self.G.pending_choice())

    def test_the_one_that_was_on_screen_at_the_time(self):
        self._after_a_youtube_play_open()
        self._quiet(self.bc._run_llm_dispatch,
                    "Jarvis, that's not the right video.")
        self.spoken.clear()
        resp = self._p(self.bc, "get_response_with_animation")
        self._quiet(self.bc._run_llm_dispatch,
                    "Jarvis, I wanted the one that was on screen at the time.")
        resp.assert_not_called()
        said = self._all_spoken()
        self.assertNotIn("Kai Cenat", said)
        self.assertEqual(len(self.desk.clicks), 1)
        self.assertEqual(self.home.url,
                         "https://videosite.example/watch?v=v01")
        self.vision.assert_not_called()
        self.vision_multi.assert_not_called()

    # 4 ──────────────────────────────────────────────────────────────────
    def test_sign_in_page_gets_no_click(self):
        chooser = F.FakeWindow(103, "chooser_light", "top",
                               url="https://accounts.google.com/accountchooser",
                               title="Sign in - Google Accounts - Google Chrome")
        self.desk.wins.insert(0, chooser)
        self.desk.fg = 103
        self._dispatch(
            "pull up the console so I can sign in",
            "[intent:confirmation] Right away, sir. [ACTION: open_url, "
            "console.example.com] [ACTION: click, the test user account]")
        self.assertEqual(self.desk.clicks, [])
        self.assertIn("leave the signing in to you", self._all_spoken())

    # 5 ──────────────────────────────────────────────────────────────────
    def test_tell_claude_is_a_note_not_a_search(self):
        from core import dev_notes as DN
        resp = self._p(self.bc, "get_response_with_animation")
        self._quiet(self.bc._run_llm_dispatch,
                    "Jarvis, also tell Claude to go ahead and do some research "
                    "on the screen vision, it's not working properly.")
        resp.assert_not_called()
        self.assertEqual(self.calls["web_search"], [])
        self.assertIn(DN.SPOKEN_LINE, self.spoken)
        self.assertNotIn("relay", self._all_spoken().lower())
        self.assertIn("research", DN.read_notes()[-1]["note"])

    # 6 ──────────────────────────────────────────────────────────────────
    def test_see_screen_quotes_titles_without_a_vision_call(self):
        self._dispatch("Jarvis, what videos are on my screen?",
                       "[ACTION: see_screen, what videos are on screen?]",
                       ["The middle monitor shows 'I Survived 7 Days In An "
                        "Abandoned City' by MrBeast, sir."])
        self.vision.assert_not_called()
        self.vision_multi.assert_not_called()
        results = self.gfr.call_args_list[0].args[0]
        self.assertTrue(any("I Survived 7 Days In An Abandoned City" in r
                            for _n, r in results), results)

    # 7 ──────────────────────────────────────────────────────────────────
    def test_the_video_already_playing_is_not_clicked_and_no_loop(self):
        playing = F.FakeWindow(102, None, "top", title=(
            "Eat Everything In A Grocery Store, Win $1,000,000 - YouTube - "
            "Google Chrome"))
        self.desk.wins.insert(0, playing)
        self._dispatch(
            "click the one playing on the top monitor",
            '[intent:confirmation] I\'ve found it, sir. [ACTION: click, '
            'monitor:top|the video "Eat Everything In A Grocery Store, Win '
            '$1,000,000"]',
            ["[ACTION: click, monitor:top|the video Eat Everything]"] * 4)
        self.assertEqual(self.desk.clicks, [])
        self.assertIn("already playing on the top monitor", self._all_spoken())
        self.assertEqual(self.gfr.call_count, 0)

    # the failure family ─────────────────────────────────────────────────
    def test_find_then_click_failures_stop_at_the_second(self):
        self._stub("find_on_screen", "could not find 'x' on screen")
        self._stub("local_click_target_by_description",
                   "could not find 'x' on screen via local vision")
        self._dispatch("find the dragon thing",
                       "[ACTION: find_on_screen, the dragon thing]",
                       ["[ACTION: local_click_target_by_description, dragon]",
                        "[ACTION: find_on_screen, dragon again]"])
        self.assertEqual(self.gfr.call_count, 1)
