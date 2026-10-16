"""Playing music, live 2026-10-15 21:20-21:22 ("Skrillex", five tries, nothing
played). Pins two monolith fixes:

  * the resolved Apple Music TRACK page's first play step falls back from the
    highlighted song row to the page's Play button, then (fresh page, and only
    when playback can be confirmed) SPACE - the old plan held the row and
    "recheck" only, so a row vision could not see meant nothing was pressed;
  * the model's YouTube SEARCH action runs as youtube_play when the turn's
    own words asked to play ("play it on YouTube instead"), and stays a
    search otherwise.

The pure intent check (core.dispatcher.youtube_search_means_play) is covered
here too. stdlib unittest + mock; no UI, network or screen is touched.
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import dispatcher as D
from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


class YoutubeSearchMeansPlay(unittest.TestCase):
    def test_the_live_request_is_a_play(self):
        self.assertTrue(D.youtube_search_means_play(
            "Jarvis, play it on YouTube instead. It's broken on Apple Music."))

    def test_play_requests(self):
        for s in ("put Skrillex on YouTube", "play some lo-fi",
                  "shuffle Skrillex on youtube", "I want to listen to Skrillex"):
            self.assertTrue(D.youtube_search_means_play(s), s)

    def test_searches_and_screen_references_are_not(self):
        for s in ("search YouTube for Skrillex", "find Skrillex videos",
                  "show me Skrillex on YouTube", "look up a lofi mix",
                  "play that video", "play the one on the left", "Skrillex",
                  "", None, 42):
            self.assertFalse(D.youtube_search_means_play(s), repr(s))


@requires_monolith
class YoutubePlayAlias(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _alias(self, said, actions=None):
        bc = self.bc
        acts = dict(actions if actions is not None else
                    {"youtube": bc._act_youtube, "youtube_play": object()})
        with mock.patch.object(bc, "ACTIONS", acts), \
                mock.patch.object(bc, "_last_user_text", [said]):
            return bc._youtube_play_alias("youtube", "Skrillex")

    def test_a_play_request_runs_youtube_play(self):
        self.assertEqual(self._alias("play it on YouTube instead"),
                         "youtube_play")

    def test_a_search_request_stays_a_search(self):
        self.assertEqual(self._alias("search YouTube for Skrillex"), "youtube")

    def test_no_youtube_play_registered_keeps_the_search(self):
        acts = {"youtube": self.bc._act_youtube}
        self.assertEqual(self._alias("play it on YouTube", acts), "youtube")

    def test_a_replaced_youtube_action_is_respected(self):
        acts = {"youtube": lambda a: a, "youtube_play": object()}
        self.assertEqual(self._alias("play it on YouTube", acts), "youtube")

    def test_other_actions_pass_through(self):
        with mock.patch.object(self.bc, "_last_user_text", ["play it"]):
            self.assertEqual(self.bc._youtube_play_alias("web_search", "x"),
                             "web_search")


@requires_monolith
class TrackRowOrPlay(MonolithGlobalsTestCase):
    """_streaming_apply_play_strategy('track_row_or_play', ...)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run(self, finds, confirm_possible=True):
        bc = self.bc
        with mock.patch.object(bc, "_streaming_find_with_retry",
                               side_effect=list(finds)), \
                mock.patch.object(bc, "_streaming_confirmation_possible",
                                  return_value=confirm_possible), \
                mock.patch.object(bc, "ui_double_click") as dc, \
                mock.patch.object(bc, "ui_click") as click, \
                mock.patch.object(bc, "ui_press") as press, \
                mock.patch.object(bc, "_focus_music_window", return_value=None), \
                mock.patch.object(bc.time, "sleep"):
            out = bc._streaming_apply_play_strategy(
                "track_row_or_play", {"track_play_hint": "row", "play_hint": "btn"},
                None)
        return out, dc, click, press

    def test_the_song_row_first(self):
        (ok, desc), dc, click, press = self._run([(5, 6)])
        self.assertTrue(ok)
        dc.assert_called_once_with(5, 6)
        click.assert_not_called()
        press.assert_not_called()

    def test_no_row_clicks_the_play_button(self):
        (ok, desc), dc, click, press = self._run([None, (7, 8)])
        self.assertTrue(ok)
        dc.assert_not_called()
        click.assert_called_once_with(7, 8)
        press.assert_not_called()
        self.assertIn("row wasn't visible", desc)

    def test_nothing_visible_presses_space_on_the_fresh_page(self):
        (ok, desc), dc, click, press = self._run([None, None])
        self.assertTrue(ok)
        press.assert_called_once_with("space")
        dc.assert_not_called()
        click.assert_not_called()

    def test_never_toggles_blind(self):
        (ok, desc), dc, click, press = self._run([None, None],
                                                 confirm_possible=False)
        self.assertFalse(ok)
        press.assert_not_called()

    def test_the_resolved_track_plan_starts_with_it(self):
        src = self.bc._streaming_auto_play.__code__.co_consts
        plans = [c for c in src if isinstance(c, tuple)
                 and "track_row_or_play" in c]
        # The list literal compiles to a tuple constant on CPython.
        self.assertTrue(plans or "track_row_or_play" in repr(src))


class CloseFrontTabRoute(unittest.TestCase):
    """Live 22:06:25 "Jarvis, delete that website." became a guarded click on
    a 'delete' button. The whole request means the browser tab in front."""

    def test_the_live_request_and_its_kin(self):
        for s in ("Jarvis, delete that website.", "close this tab",
                  "get rid of that page please", "remove the website",
                  "can you close that site", "Jarvis close out of this tab"):
            self.assertTrue(D.close_front_tab_route(s), s)

    def test_anything_more_stays_with_the_model(self):
        for s in ("delete that website from my bookmarks", "delete that file",
                  "close Chrome", "close that", "delete it", "open that website",
                  "", None):
            self.assertFalse(D.close_front_tab_route(s), repr(s))


@requires_monolith
class CloseFrontTabMonolith(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _front(self, title):
        with mock.patch.object(self.bc, "_read_focused_window",
                               return_value=(0x10, title, None)):
            return self.bc._front_browser_page()

    def test_the_page_part_of_a_browser_title(self):
        self.assertEqual(
            self._front("(906) lofi hip hop radio - YouTube - Google Chrome"),
            "(906) lofi hip hop radio - YouTube")

    def test_brackets_and_pipes_never_break_the_action_token(self):
        self.assertEqual(self._front("[LIVE] Mix | Chill - YouTube - Google Chrome"),
                         "LIVE Mix")

    def test_not_a_browser(self):
        self.assertIsNone(self._front("Untitled - Notepad"))
        self.assertIsNone(self._front(""))

    def test_the_route_closes_the_tab_in_front(self):
        bc = self.bc
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(0x10, "Omnitrix - YouTube - Google Chrome", None)), \
                mock.patch.object(bc, "_screen_route_reply", return_value=None), \
                mock.patch.dict(bc.ACTIONS, {"close_window": lambda a: a}):
            out = bc._utterance_route_reply("Jarvis, delete that website.")
        self.assertEqual(out, "[ACTION: close_window, Omnitrix - YouTube]")

    def test_no_browser_in_front_leaves_it_to_the_model(self):
        bc = self.bc
        with mock.patch.object(bc, "_read_focused_window",
                               return_value=(0x10, "Untitled - Notepad", None)), \
                mock.patch.object(bc, "_screen_route_reply", return_value=None), \
                mock.patch.dict(bc.ACTIONS, {"close_window": lambda a: a}):
            out = bc._utterance_route_reply("Jarvis, delete that website.")
        self.assertFalse(out and "close_window" in out)


if __name__ == "__main__":
    unittest.main()
