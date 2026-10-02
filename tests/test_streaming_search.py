"""core/streaming_search.py - the VERIFIED streaming search links (S2), the
"find / play <title> on <service>" route and the sign-in wall (S5).

Live 2026-10-02 (paraphrased): asked to find a show on HBO Max, the brain
wrote open_url with https://www.hbomax.com/search?q=<show>, a 404 page
("Oops! Looks like this link isn't working."), with "Sign In" at the top -
the browser was not signed in. Pure functions; no network, no browser.

    python -m unittest tests.test_streaming_search
"""
from __future__ import annotations

import re
import unittest
import urllib.parse

from core import dispatcher
from core import streaming_search as S

# The exact shape the brain guessed live (the show title is a placeholder).
GUESSED = "https://www.hbomax.com/search?q=Some+Show"


class VerifiedTableTests(unittest.TestCase):
    def test_every_required_service_is_in_the_table(self):
        for key in ("max", "netflix", "hulu", "disney_plus", "prime_video",
                    "youtube", "apple_tv"):
            with self.subTest(key=key):
                self.assertIn(key, S.SERVICES)

    def test_the_check_date_is_recorded(self):
        self.assertRegex(S.VERIFIED_CHECKED, r"^\d{4}-\d{2}-\d{2}$")
        with open(S.__file__, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn(f"Verified {S.VERIFIED_CHECKED}", src)

    def test_each_verified_pattern_lives_on_its_own_host(self):
        for key, svc in S.SERVICES.items():
            if not svc.search:
                continue
            with self.subTest(key=key):
                p = urllib.parse.urlsplit(svc.search.format(q="x"))
                self.assertEqual(p.hostname, svc.search_host)
                self.assertTrue(p.path.startswith(svc.search_path))
                self.assertIn(svc.search_param,
                              urllib.parse.parse_qs(p.query))
                self.assertEqual(S.service_for_url(svc.search), key)

    def test_hbo_max_searches_on_the_player_host_not_the_404_guess(self):
        self.assertEqual(S.search_url("max", "Some Show"),
                         "https://play.hbomax.com/search?q=Some%20Show")
        self.assertNotIn("www.hbomax.com/search", S.SERVICES["max"].search)

    def test_disney_plus_has_no_verified_search(self):
        self.assertIsNone(S.SERVICES["disney_plus"].search)
        self.assertIsNone(S.search_url("disney_plus", "Some Show"))
        self.assertEqual(S.home_url("disney_plus"), "https://www.disneyplus.com")

    def test_query_is_encoded(self):
        url = S.search_url("netflix", "A & B / C?")
        self.assertEqual(url, "https://www.netflix.com/search?q=A%20%26%20B%20%2F%20C%3F")
        self.assertIsNone(S.search_url("netflix", "   "))

    def test_spoken_names(self):
        for name, key in (("HBO Max", "max"), ("hbo", "max"), ("Max", "max"),
                          ("Disney Plus", "disney_plus"), ("disney +", "disney_plus"),
                          ("Disney+", "disney_plus"), ("Amazon Prime Video", "prime_video"),
                          ("prime", "prime_video"), ("Apple TV+", "apple_tv"),
                          ("apple tv", "apple_tv"), ("You Tube", "youtube"),
                          ("the Netflix app", "netflix"), ("Hulu", "hulu"),
                          ("spotify", None), ("", None), (None, None)):
            with self.subTest(name=name):
                self.assertEqual(S.canon_service(name), key)

    def test_the_dispatcher_uses_the_same_names(self):
        # One canonicaliser (the stale-duplicate rule): the chain resolver's
        # service names come from the verified table now.
        for spoken in ("hbo max", "max", "disney plus", "disney +", "prime",
                       "amazon prime video", "you tube", "netflix", "hulu"):
            with self.subTest(spoken=spoken):
                self.assertEqual(dispatcher._canon_streaming_service(spoken),
                                 S.canon_service(spoken))
        self.assertEqual(dispatcher._canon_streaming_service("Spotify"), "spotify")


class FixSearchUrlTests(unittest.TestCase):
    def test_the_live_guess_becomes_the_verified_search(self):
        fix = S.fix_search_url(GUESSED)
        self.assertEqual(fix.url, "https://play.hbomax.com/search?q=Some%20Show")
        self.assertEqual(fix.service, "max")
        self.assertIn("not a real HBO Max link", fix.note)

    def test_a_service_without_a_pattern_opens_its_home_and_says_so(self):
        fix = S.fix_search_url("https://www.disneyplus.com/search?q=Some+Show")
        self.assertEqual(fix.url, "https://www.disneyplus.com")
        self.assertIn("no verified Disney+ search link", fix.note)
        self.assertIn("Some Show", fix.note)

    def test_verified_and_non_search_links_are_left_alone(self):
        for url in (S.search_url("max", "x"), S.search_url("youtube", "x"),
                    "https://tv.apple.com/us/search?term=x",
                    "https://www.primevideo.com/search?phrase=x",
                    "https://www.netflix.com/title/123", "https://www.hulu.com",
                    "max.com", "https://example.com/search?q=x",
                    "https://www.google.com/search?q=x"):
            with self.subTest(url=url):
                fix = S.fix_search_url(url)
                self.assertEqual(fix.url, url)
                self.assertEqual(fix.note, "")

    def test_other_guess_shapes(self):
        for url, want in (
                ("https://play.max.com/search?q=x", S.search_url("max", "x")),
                ("hbomax.com/search?query=x", S.search_url("max", "x")),
                ("https://www.hulu.com/search/x%20y", S.search_url("hulu", "x y")),
                ("https://www.netflix.com/browse?q=x", S.search_url("netflix", "x"))):
            with self.subTest(url=url):
                self.assertEqual(S.fix_search_url(url).url, want)

    def test_a_bare_service_name_is_its_home_page(self):
        self.assertEqual(S.fix_search_url("HBO Max").url, "https://play.hbomax.com")
        self.assertEqual(S.fix_search_url("open the pod bay").url, "open the pod bay")


class StreamingRouteTests(unittest.TestCase):
    def test_the_live_find_and_play_shape_plays(self):
        # Paraphrase of the live turn: a control sentence, then the command.
        self.assertEqual(
            S.streaming_route("Jarvis, try again. Find Some Show on HBO Max "
                              "and start and resume playing."),
            "[ACTION: play_streaming, max|Some Show]")

    def test_find_alone_searches(self):
        for text, want in (
                ("find Severance on Apple TV", "apple_tv|Severance"),
                ("can you pull up Bluey on Disney plus please", "disney_plus|Bluey"),
                ("look up lofi beats on YouTube", "youtube|lofi beats"),
                ("Jarvis, search for The Bear on Hulu.", "hulu|The Bear")):
            with self.subTest(text=text):
                self.assertEqual(S.streaming_route(text),
                                 f"[ACTION: streaming_search, {want}]")

    def test_play_verbs_play(self):
        for text, want in (
                ("play The Bear on Hulu", "hulu|The Bear"),
                ("watch Reacher on Amazon Prime Video", "prime_video|Reacher"),
                ("put on Succession on Max", "max|Succession"),
                ("find Dark on Netflix and play it for me", "netflix|Dark")):
            with self.subTest(text=text):
                self.assertEqual(S.streaming_route(text),
                                 f"[ACTION: play_streaming, {want}]")

    def test_left_to_the_brain(self):
        for text in (
                "Jarvis, close that and open up HBO Max instead. The show is there.",
                "play it on Netflix", "find something on Hulu",
                "play The Bear on Hulu on the left monitor",
                "find Dark on Netflix and turn off the lights",
                "play lofi beats on YouTube",      # youtube_play_route owns it
                "play a song on Spotify",          # music has its own routes
                "find Dark on Netflix. Then dim the lights.",
                "", None, 42):
            with self.subTest(text=text):
                self.assertIsNone(S.streaming_route(text))

    def test_switches(self):
        self.assertIsNone(S.streaming_route("play Dark on Netflix", allow_play=False))
        self.assertIsNone(S.streaming_route("find Dark on Netflix", allow_find=False))

    def test_every_token_is_a_well_formed_route_token(self):
        rx = re.compile(r"^\[ACTION:\s*([A-Za-z0-9_]+)\s*(?:,\s*([^\]\n]{0,200}?))?\s*\]$")
        tok = S.streaming_route("find Some Show on HBO Max")
        self.assertRegex(tok, rx)


class SignInWallTests(unittest.TestCase):
    def test_the_live_page_is_a_sign_in_wall(self):
        live = ("The browser shows 'Oops! Looks like this link isn't working.' "
                "on HBO Max, with a Sign In button at the top right.")
        self.assertEqual(S.wall_kind(live), "sign_in")

    def test_kinds(self):
        for text, want in (
                ("HBO Max is not signed in on this browser.", "sign_in"),
                ("A Log In form fills the page.", "sign_in"),
                ("The page says page not found.", "error"),
                ("Something went wrong. Please try again.", "error"),
                ("Search results for the show; the viewer is signed in as Guest.", None),
                ("There is no sign-in prompt; three seasons are listed.", None),
                ("The page is not asking to sign in. It lists episodes.", None),
                ("A grid of show posters.", None), ("", None), (None, None)):
            with self.subTest(text=text):
                self.assertEqual(S.wall_kind(text), want)

    def test_the_line_is_plain_and_names_the_service(self):
        self.assertEqual(
            S.wall_line("max", "sign_in"),
            "HBO Max isn't signed in on this browser, sir - sign in once and "
            "I can take it from there.")
        self.assertIn("Netflix showed an error page", S.wall_line("netflix", "error"))
        self.assertEqual(S.wall_line("max", None), "")

    def test_verdict_parser(self):
        for answer, want in (("[local-vision] SIGNIN - a Sign In button", "sign_in"),
                             ("SIGN-IN.", "sign_in"), ("LOGIN", "sign_in"),
                             ("ERROR: Oops page", "error"), ("OK, results shown", "ok"),
                             ("Yes it is", None), ("", None), (None, None)):
            with self.subTest(answer=answer):
                self.assertEqual(S.parse_wall_verdict(answer), want)

    def test_the_question_names_the_service_and_the_words(self):
        q = S.wall_question("max")
        self.assertIn("HBO Max", q)
        for word in ("SIGNIN", "ERROR", "OK", "chat"):
            self.assertIn(word, q)


if __name__ == "__main__":
    unittest.main()
