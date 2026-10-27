"""core.media_recall - "play my song from earlier" replays the last media
request (live 2026-10-26 20:08: the brain played a song literally titled
"Earlier" instead of Famous Last Words, asked for three minutes before).

    python -m unittest tests.test_media_recall
"""
from __future__ import annotations

import unittest

from core import media_recall as M

NOW = 1_800_000_000.0


class RecordTests(unittest.TestCase):
    def setUp(self):
        M.reset()
        self.addCleanup(M.reset)

    def test_the_live_youtube_search_on_a_monitor_is_remembered_as_a_play(self):
        M.record("open_on_monitor", 'left | youtube "famous last words"', now=NOW)
        self.assertEqual(M.recall_route("Jarvis, just play my damn song from earlier.",
                                        now=NOW + 180),
                         "[ACTION: youtube_play, famous last words] "
                         "[ACTION: move_window_to_monitor, youtube | left]")

    def test_play_actions_and_youtube_searches(self):
        for action, arg, want in (
                ("youtube_play", "famous last words", "[ACTION: youtube_play, famous last words]"),
                ("play_music", "song:/Famous Last Words", "[ACTION: play_music, Famous Last Words]"),
                ("youtube", "lofi beats", "[ACTION: youtube_play, lofi beats]"),
                ("spotify", "chill mix", "[ACTION: spotify, chill mix]")):
            with self.subTest(action=action):
                M.record(action, arg, now=NOW)
                self.assertEqual(M.recall_route("play that song again", now=NOW + 5), want)

    def test_other_actions_do_not_replace_the_request(self):
        M.record("youtube_play", "famous last words", now=NOW)
        for action, arg in (("screenshot", ""), ("web_search", "how to log hours"),
                            ("click_on_screen", "YouTube video result"),
                            ("open_on_monitor", "left | chrome")):
            M.record(action, arg, now=NOW + 1)
        self.assertEqual(M.recall_route("play it again", now=NOW + 60),
                         "[ACTION: youtube_play, famous last words]")

    def test_an_old_or_missing_request_leaves_it_to_the_model(self):
        self.assertIsNone(M.recall_route("play my song from earlier", now=NOW))
        M.record("youtube_play", "famous last words", now=NOW)
        self.assertIsNone(M.recall_route("play my song from earlier",
                                         now=NOW + M.RECALL_MAX_AGE_S + 1))


class PhrasingTests(unittest.TestCase):
    def test_recall_phrasings(self):
        for u in ("Jarvis, just play my damn song from earlier.",
                  "play my song from earlier", "play the song from before",
                  "play that song again", "play the video again",
                  "play it again", "Jarvis play that again",
                  "can you play the song I asked for", "put on my last song"):
            with self.subTest(u=u):
                self.assertTrue(M.is_recall_request(u))

    def test_new_requests_are_not_recalls(self):
        for u in ("play famous last words on youtube", "play some jazz",
                  "play the song earlier by Jiří Horák on apple music",
                  "what was the song from earlier", "play", "", None, 42):
            with self.subTest(u=u):
                self.assertFalse(M.is_recall_request(u))

    def test_never_raises(self):
        for junk in (None, 12, object()):
            self.assertIsNone(M.recall_route(junk, now=NOW))
            self.assertIsNone(M.request_from(junk, junk))


if __name__ == "__main__":
    unittest.main()
