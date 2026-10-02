"""Apple Music playlists play by direct link or keyboard, vision last.

Owner request (jarvis_todo, 2026-06-03): screen clicks did not land on Apple
Music in Chrome; use keyboard navigation or direct playlist links, and make
the playlist actually start (Shuffle / Play), not stop at a list.

Before 2026-10-02 _apple_music_play_playlist always opened Library >
Playlists, found the playlist TILE by screen vision, clicked it, then found
the Play/Shuffle button by vision too; with vision off it gave up at once.
Now:

  * APPLE_MUSIC_PLAYLIST_LINKS {name: music.apple.com link} opens the
    playlist's own page directly (no tile to find);
  * otherwise the playlist is opened by KEYBOARD from the Library page
    (Chrome's find bar: Ctrl+F, the name, Esc focuses the link holding the
    match, Enter opens it), proven by the media window's title changing to
    one that names the playlist;
  * the play step presses "Shuffle" the same way before any vision click;
  * keys only ever go to the window JARVIS opened, re-checked before each.

Every UI / window call is mocked: nothing here reaches a real window.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_HWND = 4242
_LIB_TITLE = "‎Apple Music - Web Player - Google Chrome"
_PL_TITLE = "‎Road Trip - Apple Music - Google Chrome"


class _Win:
    def __init__(self, hwnd: int, title: str):
        self._hWnd = hwnd
        self.title = title

    def activate(self):
        pass


@requires_monolith
class _PlaylistBase(MonolithGlobalsTestCase):
    """Common boundary mocks: the browser open, the new-window adoption, the
    focus checks and every key / click. `self.keys` records the key calls."""

    links: dict = {}
    vision = True
    focused = _HWND

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.keys = []
        self.opened = []

        def _open(url, **_kw):
            self.opened.append(url)
            return "chrome"

        patches = [
            mock.patch.object(bc, "_open_url_in_browser", side_effect=_open),
            mock.patch.object(bc, "_window_handles_snapshot", return_value=set()),
            mock.patch.object(bc, "_find_browser_window_matching",
                              return_value=_Win(_HWND, _LIB_TITLE)),
            mock.patch.object(bc, "_ensure_window_visible_maximized",
                              return_value=True),
            mock.patch.object(bc, "_monitor_name_for_window", return_value="left"),
            mock.patch.object(bc.time, "sleep"),
            mock.patch.object(bc, "SCREEN_VISION_ENABLED", self.vision),
            mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True),
            mock.patch.object(bc, "_vision_click_backend_available",
                              return_value=self.vision),
            mock.patch.object(bc, "APPLE_MUSIC_PLAYLIST_LINKS", dict(self.links),
                              create=True),
            mock.patch.object(bc, "_focus_window_hwnd", return_value=True),
            mock.patch.object(bc, "_read_focused_window",
                              side_effect=lambda: (self.focused, "", None)),
            mock.patch.object(bc, "ui_hotkey",
                              side_effect=lambda *k: self.keys.append(("hotkey",) + k)),
            mock.patch.object(bc, "ui_type",
                              side_effect=lambda t: self.keys.append(("type", t))),
            mock.patch.object(bc, "ui_press",
                              side_effect=lambda k: self.keys.append(("press", k))),
            mock.patch.object(bc, "ui_click"),
            mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.find = self._start(mock.patch.object(
            bc, "_streaming_find_with_retry", return_value=None))
        self.fct = self._start(mock.patch.object(
            bc, "find_click_target", return_value=None))
        self.pv = self._start(mock.patch.object(
            bc, "_streaming_play_and_verify",
            return_value="playing 'road trip' on Apple Music"))

    def _start(self, patcher):
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def _titles(self, *titles):
        """The media window's title on each read (the last one repeats)."""
        seq = list(titles)

        def _read(_hwnd):
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return mock.patch.object(self.bc, "_window_title_for_hwnd",
                                 side_effect=_read, create=True)


class DirectLinkTests(_PlaylistBase):
    links = {"Road Trip": "https://music.apple.com/library/playlist/p.AbC123"}

    def test_a_named_link_opens_the_playlist_page_without_any_vision(self):
        out = self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(self.opened,
                         ["https://music.apple.com/library/playlist/p.AbC123"])
        self.find.assert_not_called()
        self.fct.assert_not_called()
        self.assertEqual(out, "playing 'road trip' on Apple Music")
        cfg = self.pv.call_args.args[0]
        self.assertEqual(cfg["play_strategies"][0], "find_text_play")
        self.assertEqual(cfg["keyboard_play_text"], "Shuffle")

    def test_apostrophes_and_case_do_not_matter(self):
        with mock.patch.object(self.bc, "APPLE_MUSIC_PLAYLIST_LINKS",
                               {"Taylor’s Mix": "https://music.apple.com/x"}):
            self.assertEqual(self.bc._apple_music_playlist_link("taylors mix"),
                             "https://music.apple.com/x")

    def test_a_link_that_is_not_apple_music_is_never_opened(self):
        with mock.patch.object(self.bc, "APPLE_MUSIC_PLAYLIST_LINKS",
                               {"road trip": "https://example.com/p"}), \
                self._titles(_LIB_TITLE):
            self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(self.opened, ["https://music.apple.com/library/playlists"])


class KeyboardOpenTests(_PlaylistBase):
    def test_the_playlist_opens_by_keyboard_and_vision_is_never_asked(self):
        with self._titles(_LIB_TITLE, _PL_TITLE):
            out = self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(self.keys, [("hotkey", "ctrl", "f"),
                                     ("type", "road trip"),
                                     ("press", "esc"),
                                     ("press", "enter")])
        self.find.assert_not_called()
        self.fct.assert_not_called()
        self.pv.assert_called_once()
        self.assertEqual(out, "playing 'road trip' on Apple Music")

    def test_a_playlist_whose_name_holds_the_one_asked_for_is_not_it(self):
        """Chrome's find matches substrings, so asking for "Mix" can land on
        "Taylor's Mix" earlier on the page and Enter opens it. Its title holds
        "mix", which used to confirm it: the wrong playlist played and JARVIS
        said "playing 'Mix'" (2026-10-02 review). Vision gets the next try."""
        with self._titles(_LIB_TITLE,
                          "\u200eTaylor\u2019s Mix - Apple Music - Google Chrome"):
            out = self.bc._apple_music_play_playlist("mix")
        self.pv.assert_not_called()
        self.assertTrue(self.find.called, "the vision tile search never ran")
        self.assertIn("couldn't find a playlist named 'mix'", out)

    def test_the_title_must_lead_with_the_playlist_name(self):
        names = self.bc._title_names_playlist
        for title, name in (
                ("\u200eMix - Apple Music - Google Chrome", "mix"),
                ("Taylor\u2019s Mix - Apple Music", "taylors mix"),
                ("Chill - Summer - Apple Music - Google Chrome", "chill - summer"),
                ("Road Trip by A Curator - Apple Music", "Road Trip"),
                ("Road Trip", "road trip")):
            self.assertTrue(names(title, name), (title, name))
        for title, name in (
                ("Taylor\u2019s Mix - Apple Music - Google Chrome", "mix"),
                ("Mixtape - Apple Music - Google Chrome", "mix"),
                ("Apple Music - Web Player - Google Chrome", "mix"),
                ("", "mix"),
                ("Mix - Apple Music", "")):
            self.assertFalse(names(title, name), (title, name))

    def test_an_unconfirmed_keyboard_open_falls_back_to_vision(self):
        # The title never changes: the find matched nothing it could open.
        with self._titles(_LIB_TITLE):
            out = self.bc._apple_music_play_playlist("road trip")
        self.assertTrue(self.find.called)        # the vision tile search ran
        self.assertIn("couldn't find a playlist named 'road trip'", out)

    def test_keys_stop_the_moment_the_media_window_loses_focus(self):
        # The owner clicks into another window after Ctrl+F: the name and the
        # Enter must never reach it.
        def _press_ctrl_f(*k):
            self.keys.append(("hotkey",) + k)
            self.focused = 999
        with mock.patch.object(self.bc, "ui_hotkey", side_effect=_press_ctrl_f), \
                self._titles(_LIB_TITLE):
            self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(self.keys, [("hotkey", "ctrl", "f")])

    def test_nothing_is_typed_when_the_window_is_not_in_front(self):
        self.focused = 999
        with self._titles(_LIB_TITLE):
            self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(self.keys, [])

    def test_a_name_the_keyboard_cannot_type_is_not_typed(self):
        with self._titles(_LIB_TITLE):
            self.bc._apple_music_play_playlist("café del mar")
        self.assertEqual(self.keys, [])
        self.assertTrue(self.find.called)


class KeyboardWithoutVisionTests(_PlaylistBase):
    vision = False

    def test_vision_off_no_longer_blocks_the_keyboard_route(self):
        # It used to return "auto-click needs SCREEN_VISION_ENABLED ..." before
        # trying anything.
        with self._titles(_LIB_TITLE, _PL_TITLE):
            out = self.bc._apple_music_play_playlist("road trip")
        self.assertEqual(out, "playing 'road trip' on Apple Music")
        cfg = self.pv.call_args.args[0]
        self.assertEqual(cfg["play_strategies"], ["find_text_play", "recheck"])


@requires_monolith
class FindTextPlayStrategyTests(MonolithGlobalsTestCase):
    def test_shuffle_is_pressed_by_keyboard_in_the_media_window(self):
        bc = self.bc
        keys = []
        cfg = {"service_key": "apple_music", "keyboard_play_text": "Shuffle"}
        with mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"apple_music": _HWND}, clear=True), \
                mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True), \
                mock.patch.object(bc, "_focus_window_hwnd", return_value=True), \
                mock.patch.object(bc, "_read_focused_window",
                                  return_value=(_HWND, "", None)), \
                mock.patch.object(bc.time, "sleep"), \
                mock.patch.object(bc, "ui_hotkey",
                                  side_effect=lambda *k: keys.append(k)), \
                mock.patch.object(bc, "ui_type",
                                  side_effect=lambda t: keys.append(t)), \
                mock.patch.object(bc, "ui_press",
                                  side_effect=lambda k: keys.append(k)), \
                mock.patch.object(bc, "_streaming_find_with_retry") as vis:
            attempted, desc = bc._streaming_apply_play_strategy(
                "find_text_play", cfg, None)
        self.assertTrue(attempted, desc)
        self.assertEqual(keys, [("ctrl", "f"), "Shuffle", "esc", "enter"])
        vis.assert_not_called()

    def test_no_recorded_window_means_no_keys(self):
        bc = self.bc
        cfg = {"service_key": "apple_music", "keyboard_play_text": "Shuffle"}
        with mock.patch.dict(bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True), \
                mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True), \
                mock.patch.object(bc, "ui_hotkey") as hk, \
                mock.patch.object(bc, "ui_type") as ty:
            attempted, _ = bc._streaming_apply_play_strategy(
                "find_text_play", cfg, None)
        self.assertFalse(attempted)
        hk.assert_not_called()
        ty.assert_not_called()


if __name__ == "__main__":
    unittest.main()
