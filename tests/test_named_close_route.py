"""core.dispatcher.named_close_target / forgot_close_target (2026-10-05).

Live 00:24:21-00:25:43 (session_2026-10-05_00-21-55.log, paraphrased): the
owner NAMED the window four times - "go ahead and close out File Explorer 2"
(Parakeet's "2" for "too"), "close file explorer" twice, "close Google
Chrome" - and every one was answered [ACTION: close_last_opened], which only
closes what JARVIS itself opened; all four failed. And at 00:25:27, right
after a bulk close, "you forgot Google Chrome" was read as one more window to
KEEP. These parsers read the name out of both shapes; the monolith routes
them to close_window when the name is an open window
(tests/monolith/test_monolith_live_turns_1005.py).

Light tier, pure functions.

    python -m unittest tests.test_named_close_route
"""
from __future__ import annotations

import unittest

from core.dispatcher import (forgot_close_target, named_close_target,
                             window_keep_route)

# The live utterances, paraphrased: the load-bearing words kept ("close
# out", Parakeet's "2" for "too", the missing comma after the wake word).
LIVE_NAMED = (
    ("Jarvis, please close out File Explorer 2.", "File Explorer"),
    ("Jarvis, close the file explorer.", "file explorer"),
    ("Jarvis Close File Explorer now.", "File Explorer"),
    ("Jarvis close Google Chrome for me.", "Google Chrome"),
)


class NamedCloseTargetTests(unittest.TestCase):
    def test_the_live_named_closes(self):
        for said, name in LIVE_NAMED:
            with self.subTest(said=said):
                self.assertEqual(named_close_target(said), name)

    def test_other_named_shapes(self):
        for said, name in (
                ("close Spotify", "Spotify"),
                ("Jarvis closed notepad.", "notepad"),
                ("close my notes", "notes"),
                ("close the Claude app", "Claude"),
                ("close the Downloads folder", "Downloads"),
                # An app whose name ends in a common word is still a name.
                ("close Apple Music", "Apple Music"),
                ("close Epic Games", "Epic Games"),
                ("Jarvis, quit Spotify please", "Spotify"),
                ("close out of chrome", "chrome"),
                ("close down Spotify for me", "Spotify"),
                ("close that notepad", "notepad")):
            with self.subTest(said=said):
                self.assertEqual(named_close_target(said), name)

    def test_a_close_that_names_nothing_is_not_named(self):
        # These stay close_last_opened's ("close that" = what JARVIS opened).
        for said in ("close that", "close it", "Jarvis, close it out",
                     "close this one", "close the window", "close the tab",
                     "close the last one", "close them", "close those",
                     "close them all",
                     # Review 2026-10-05: these point back at what JARVIS
                     # opened / played, and were rewritten away from
                     # close_last_opened when read as names.
                     "close what you opened", "close what you just opened",
                     "close the window you just opened",
                     "close the thing you opened", "close the tab you opened",
                     "close the page you just opened",
                     "close the video you put on",
                     "close that YouTube video", "close the YouTube tab",
                     "close the Netflix show", "close the song",
                     "close the music", "close the browser",
                     "close the search results"):
            with self.subTest(said=said):
                self.assertIsNone(named_close_target(said))

    def test_bulk_compound_and_monitor_shapes_are_left_alone(self):
        for said in ("Jarvis close every window except for Claude.",
                     "close all windows but Spotify",
                     "close that and open Netflix",
                     "close Spotify and open Netflix",
                     "close paint and the calculator",
                     "close the task manager on the left",
                     "close Chrome on the main monitor",
                     "close Chrome, then open Edge",
                     # The live turn the brain got right on its own: a
                     # pointing-back close with the name riding at the end.
                     "Jarvis, close that window for me, though, media "
                     "player."):
            with self.subTest(said=said):
                self.assertIsNone(named_close_target(said))
        # The bulk close keeps its own route.
        self.assertEqual(
            window_keep_route("Jarvis close every window except for Claude."),
            "[ACTION: close_all_windows_except, Claude]")

    def test_not_a_close(self):
        for said in ("open File Explorer", "what is close to me",
                     "how close is the storm", "", None, 42):
            with self.subTest(said=said):
                self.assertIsNone(named_close_target(said))


class ForgotCloseTargetTests(unittest.TestCase):
    def test_the_live_follow_up(self):
        self.assertEqual(forgot_close_target("Jarvis, you forgot about "
                                             "Google Chrome."), "Google Chrome")

    def test_other_shapes(self):
        for said, name in (("Chrome is still open", "Chrome"),
                           ("Chrome's still open, Jarvis", "Chrome"),
                           ("you left Chrome open", "Chrome"),
                           ("you left out Spotify", "Spotify"),
                           ("you missed the media player", "media player"),
                           ("but you didn't close File Explorer",
                            "File Explorer")):
            with self.subTest(said=said):
                self.assertEqual(forgot_close_target(said), name)

    def test_not_a_window(self):
        for said in ("you forgot to turn off the lights", "you forgot it",
                     "you missed one", "you forgot everything",
                     "you forgot Chrome and Spotify", "", None):
            with self.subTest(said=said):
                self.assertIsNone(forgot_close_target(said))


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
