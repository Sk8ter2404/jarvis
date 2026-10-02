"""'play X on YouTube' plays it (NEW #7, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_21-48-36.log, 21:43:17): the owner
asked for a named artist mix on YouTube; Whisper wrote "Jarvis plays ...
essentials on YouTube". One command never reaches the chain rules (they need
two or more segments), so the turn went to the model, which emitted
[ACTION: youtube] - the SEARCH action. The results page opened, nothing
played, and JARVIS said "Right away, sir." The BASIC prompt line listed
`youtube` as "YouTube search" with no pointer to `youtube_play`.

  * core.dispatcher.youtube_play_route claims a whole "play / plays / put on
    <X> on YouTube" request as [ACTION: youtube_play, <X>].
  * a search request, a vague object ("play it on YouTube"), another service,
    or extra clauses are NOT claimed - the model keeps those turns.
  * the BASIC `youtube` line says it plays nothing and names youtube_play.

Light tier: core.dispatcher and core.prompts import without the monolith.
Synthetic titles only.

    python -m unittest tests.test_youtube_play_route
"""
from __future__ import annotations

import re
import unittest

from core import dispatcher
from core import prompts


def _route(text):
    fn = getattr(dispatcher, "youtube_play_route", None)
    return fn(text) if fn is not None else None


class YouTubePlayRouteTests(unittest.TestCase):

    def test_the_live_sentence_routes_to_youtube_play(self):
        # Whisper's "plays" for "play", with the wake word still on the front.
        self.assertEqual(
            _route("Jarvis plays Artist Name essentials on YouTube."),
            "[ACTION: youtube_play, Artist Name essentials]")

    def test_common_phrasings_route(self):
        cases = {
            "Jarvis, play the Synthwave Mix on YouTube":
                "[ACTION: youtube_play, the Synthwave Mix]",
            "jarvis. play lofi beats on youtube":
                "[ACTION: youtube_play, lofi beats]",
            "Hey Jarvis, can you play some rain sounds on YouTube, please?":
                "[ACTION: youtube_play, rain sounds]",
            "please put on the cooking show on you tube":
                "[ACTION: youtube_play, the cooking show]",
            "Play Song Title by Some Band from YouTube":
                "[ACTION: youtube_play, Song Title by Some Band]",
            "play me Track Nine on YT now":
                "[ACTION: youtube_play, Track Nine]",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_route(text), want)

    def test_requests_the_model_should_keep(self):
        for text in (
                "search YouTube for lofi beats",          # a search, not play
                "jarvis what's playing on youtube",        # a question
                "play it on youtube",                      # vague object
                "play that on YouTube",
                "play something on youtube",
                "play lofi beats",                         # no service named
                "play lofi beats on Spotify",              # another service
                "play lofi beats on YouTube Music",        # another service
                "play lofi beats on youtube on the left monitor",  # extra clause
                "open youtube",
                "",
                None):
            with self.subTest(text=text):
                self.assertIsNone(_route(text))

    def test_route_token_is_well_formed_and_bounded(self):
        token = _route("play " + "x" * 400 + " on youtube")
        # Too long for an action argument: left to the model, never a token
        # the monolith's route validator would have to reject.
        self.assertIsNone(token)
        token = _route("play a [weird] title on youtube")
        self.assertIsNone(token, "a ']' would break the action token")

    def test_never_raises(self):
        for junk in (123, object(), b"play x on youtube"):
            with self.subTest(junk=type(junk).__name__):
                try:
                    _route(junk)
                except Exception as e:   # pragma: no cover - the failure path
                    self.fail(f"route raised {type(e).__name__}: {e}")


class BasicYouTubeLineTests(unittest.TestCase):
    """The BASIC catalogue line the model read at 21:43 said only 'YouTube
    search'. It must say the action plays nothing and point at youtube_play."""

    def _basic_youtube_block(self) -> str:
        text = prompts.PC_CONTROL_PROMPT
        start = text.index("BASIC actions:")
        end = text.index("SCREEN VISION", start)
        block = text[start:end]
        m = re.search(r"  youtube, <query>.*?(?=\n  [a-z_]+[ ,])", block, re.S)
        self.assertIsNotNone(m, "the BASIC youtube line is gone")
        return m.group(0)

    def test_basic_line_points_play_requests_at_youtube_play(self):
        line = self._basic_youtube_block()
        self.assertIn("youtube_play", line,
                      "the BASIC youtube line never mentions youtube_play")
        self.assertRegex(line.lower(), r"plays nothing|does not play|never plays")

    def test_basic_line_keeps_its_catalogue_shape(self):
        # tests/test_audit_action_reachability.py's fixtures quote this
        # prefix verbatim; the reachability rule needs the name set off.
        self.assertIn("  youtube, <query>             — YouTube search",
                      prompts.PC_CONTROL_PROMPT)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
