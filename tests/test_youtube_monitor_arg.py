"""youtube_play's monitor argument (v3.0.6 B6; core/actions.py).

Live 2026-10-28 17:28:11 (session_2026-10-28_17-21-18.log): "Jarvis play
vision quest on the left monitor." -> [ACTION: youtube_play, vision quest |
monitor:left] -> "opened YouTube search for 'vision quest | monitor:left'"
and the player pinned to the MIDDLE monitor. The monitor never goes into the
search now; with YOUTUBE_MONITOR_ARG 'honor' the window JARVIS opened (its
recorded handle) is moved there once the video plays.

    python -m unittest tests.test_youtube_monitor_arg
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from core import actions as A
from core import config as cfg

LIVE_ARG = "vision quest | monitor:left"
PLAYING = "playing 'vision quest' on YouTube"


class MonitorArgParseTests(unittest.TestCase):
    def test_the_live_argument(self):
        self.assertEqual(A.youtube_monitor_arg(LIVE_ARG),
                         ("vision quest", "left"))

    def test_monitor_shapes(self):
        for arg, want in (
                ("vision quest|left", ("vision quest", "left")),
                ("monitor:left | vision quest", ("vision quest", "left")),
                ("vision quest | monitor: left", ("vision quest", "left")),
                ("vision quest | the left monitor", ("vision quest", "left")),
                ("vision quest on the left monitor", ("vision quest", "left")),
                ("lofi beats | middle screen", ("lofi beats", "middle")),
                ("x | on the top screen", ("x", "top")),
                # an unknown monitor is still never searched
                ("vision quest | monitor:garage", ("vision quest", None))):
            with self.subTest(arg=arg):
                self.assertEqual(A.youtube_monitor_arg(arg), want)

    def test_searches_without_a_monitor_are_untouched(self):
        for arg in ("vision quest", "monitor lizard documentary",
                    "left 4 dead trailer", "the screen actors guild awards",
                    "on the left monitor", ""):
            with self.subTest(arg=arg):
                self.assertEqual(A.youtube_monitor_arg(arg), (arg, None))

    def test_a_bare_monitor_word_in_a_title_is_not_the_monitor(self):
        # v3.0.6 final review: a bare piece counts only when it is the last
        # one and a search is left; an explicit "monitor:x" wins over it.
        for arg, want in (
                ("The Middle | Jimmy Eat World",
                 ("The Middle | Jimmy Eat World", None)),
                ("left | monitor:right", ("left", "right")),
                ("monitor lizard documentary | left",
                 ("monitor lizard documentary", "left"))):
            with self.subTest(arg=arg):
                self.assertEqual(A.youtube_monitor_arg(arg), want)

    def test_never_raises(self):
        self.assertEqual(A.youtube_monitor_arg(None), ("", None))


class YoutubePlayHandlerTests(unittest.TestCase):
    def _bc(self, result=PLAYING, hwnd=1234):
        bc = mock.Mock()
        bc._streaming_auto_play.return_value = result
        bc._JARVIS_MEDIA_WINDOW_HWND = {"youtube": hwnd} if hwnd else {}
        return bc

    def _win32(self, rect=(0, 0, 2560, 1440)):
        g = mock.MagicMock()
        g.GetWindowRect.return_value = rect
        c = mock.MagicMock(SW_RESTORE=9, SW_MAXIMIZE=3)
        return g, mock.patch.dict(sys.modules, {"win32gui": g, "win32con": c})

    def test_live_argument_searches_the_title_and_moves_the_player(self):
        bc = self._bc()
        g, mods = self._win32()      # opened on the middle monitor
        with mock.patch.object(A, "_bc", return_value=bc), mods:
            out = A._act_youtube_play(LIVE_ARG)
        bc._streaming_auto_play.assert_called_once_with("youtube",
                                                        "vision quest")
        g.SetWindowPos.assert_called_once()
        self.assertEqual(g.SetWindowPos.call_args[0][:6],
                         (1234, 0) + cfg.MONITORS["left"])
        self.assertIn("left monitor", out)

    def test_already_on_that_monitor_is_not_moved(self):
        bc = self._bc()
        g, mods = self._win32(rect=(-2560, 0, 0, 1440))
        with mock.patch.object(A, "_bc", return_value=bc), mods:
            A._act_youtube_play(LIVE_ARG)
        g.SetWindowPos.assert_not_called()

    def test_a_failed_play_is_returned_as_is_and_nothing_moves(self):
        bc = self._bc(result="opened YouTube search for 'vision quest', but "
                             "I couldn't start playback")
        g, mods = self._win32()
        with mock.patch.object(A, "_bc", return_value=bc), mods:
            out = A._act_youtube_play(LIVE_ARG)
        g.SetWindowPos.assert_not_called()
        self.assertNotIn("monitor", out)

    def test_no_window_of_its_own_is_said_and_nothing_moves(self):
        bc = self._bc(hwnd=None)
        g, mods = self._win32()
        with mock.patch.object(A, "_bc", return_value=bc), mods:
            out = A._act_youtube_play(LIVE_ARG)
        g.SetWindowPos.assert_not_called()
        self.assertIn("couldn't move it to the left monitor", out)

    def test_strip_mode_cleans_the_search_only(self):
        bc = self._bc()
        g, mods = self._win32()
        with mock.patch.object(A, "_bc", return_value=bc), mods, \
                mock.patch.object(cfg, "YOUTUBE_MONITOR_ARG", "strip"):
            out = A._act_youtube_play(LIVE_ARG)
        bc._streaming_auto_play.assert_called_once_with("youtube",
                                                        "vision quest")
        g.SetWindowPos.assert_not_called()
        self.assertEqual(out, PLAYING)

    def test_off_mode_is_the_old_behaviour(self):
        bc = self._bc()
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(cfg, "YOUTUBE_MONITOR_ARG", "off"):
            A._act_youtube_play(LIVE_ARG)
        bc._streaming_auto_play.assert_called_once_with("youtube", LIVE_ARG)

    def test_play_streaming_youtube_takes_the_same_path(self):
        for arg in ("youtube | vision quest | monitor:left",
                    "youtube | vision quest on the left monitor"):
            with self.subTest(arg=arg):
                bc = self._bc(hwnd=None)
                bc._normalize_service.return_value = "youtube"
                bc._STREAMING_SERVICES = {"youtube": {}}
                with mock.patch.object(A, "_bc", return_value=bc),                         mock.patch.object(cfg, "YOUTUBE_MONITOR_ARG", "strip"):
                    A._act_play_streaming(arg)
                bc._streaming_auto_play.assert_called_once_with(
                    "youtube", "vision quest")

    def test_a_plain_search_is_passed_through_unchanged(self):
        bc = self._bc()
        with mock.patch.object(A, "_bc", return_value=bc):
            self.assertEqual(A._act_youtube_play("q"), PLAYING)
        bc._streaming_auto_play.assert_called_once_with("youtube", "q")


if __name__ == "__main__":
    unittest.main()
