"""v3.0.9 C: a transport word is not a song (PLAY_MUSIC_TRANSPORT_WORDS).

Live 2026-10-29 19:12 "Jarvis Resume movie." while a movie sat paused in
Chrome: the brain wrote [ACTION: play_music, resume], play_music searched
Apple Music for "resume", opened a song called "Resume", spent 45 s failing
to confirm it, and the movie stayed paused.

Pinned here (light tier: core.dispatcher and core.actions only, every media
handler stubbed, nothing is played or paused for real):
  * core.dispatcher.transport_arg_op - which play_music ARGUMENTS are only a
    transport request, and that real queries are not;
  * core.dispatcher.video_transport_route - the whole "resume the movie"
    utterance route;
  * core.actions._act_play_music - a transport argument runs resume_music /
    pause_music and speaks that handler's own line, a real query still goes
    to the Apple Music search, and the flag off is the old path exactly.

    python tools/run_tests.py play_music_transport
"""
from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest import mock

import core.actions as A
import core.config as cfg
from core import dispatcher as D

# Every transport form of the brief, as a play_music argument.
RESUME_ARGS = (
    "resume", "Resume", "RESUME", "resumé", "Résumé", "resume.",
    "continue", "unpause", "un-pause", "un pause", "play", "start again",
    "keep playing", "carry on", "resume playing", "continue playing",
    "carry on with the movie", "continue with the show",
    "resume the movie", "resume movie", "continue the video",
    "unpause the show", "resume the episode", "resume the film",
    "resume the music", "play the song", "resume the track",
    "resume playback", "play it", "play that", "resume the media",
    "resume audio", "resume the stream", "resume my movie",
    "resume this video", "please resume", "resume please",
    "resume, sir", "Jarvis, resume the movie.", "resume the movie please sir",
    "hey jarvis resume", "  resume  ",
    # Review 2026-10-29: more transport forms the brain writes.
    "resume again", "resume the movie in chrome", "resume the youtube video",
    "resume the paused movie", "resume the tv show", "resume netflix",
    "resume youtube", "resume chrome", "continue the video on youtube",
    '"resume"', "'resume'", '"Resume"', "(resume)", "“resume”",
    "it", "that", "It.",
)
PAUSE_ARGS = (
    "pause", "Pause", "stop", "pause the movie", "pause movie",
    "pause it", "stop the music", "pause playback", "please pause",
    "pause, sir", "stop the video please",
    "stop playing", "pause playing", "hold it", '"pause"',
    "pause the movie in chrome",
)
NEXT_ARGS = ("next", "skip", "next song", "next track", "skip this song",
             "skip it", "Next.", '"next"')
# Real queries: a song search, exactly as before.
QUERY_ARGS = (
    "Resume by Jenevieve", "song:Resume", "song: Resume", "artist:Continue",
    "track:Play", "album:Resume", "play Earth Song", "Earth Song",
    "continue by Some Artist", "resume by", "Smooth Criminal",
    "the movie", "play the movie Inception", "resume the movie soundtrack",
    "pause by the band", "stop this train", "Don't Stop Believin'",
    "Unpause Me", "library:resume",
    "Start Me Up", "Unstoppable", "The Movie Song", "Play That Song",
    "Play It Again", "Stop the Music - Ella", "Hold On", "the movie",
    "Next to Me",
)


class TransportArgOpTests(unittest.TestCase):
    def test_every_resume_form(self):
        for arg in RESUME_ARGS:
            with self.subTest(arg=arg):
                self.assertEqual(D.transport_arg_op(arg), "resume")

    def test_every_pause_form(self):
        for arg in PAUSE_ARGS:
            with self.subTest(arg=arg):
                self.assertEqual(D.transport_arg_op(arg), "pause")

    def test_every_next_form(self):
        for arg in NEXT_ARGS:
            with self.subTest(arg=arg):
                self.assertEqual(D.transport_arg_op(arg), "next")

    def test_real_queries_are_not_transport(self):
        for arg in QUERY_ARGS:
            with self.subTest(arg=arg):
                self.assertIsNone(D.transport_arg_op(arg))

    def test_junk_never_raises(self):
        for junk in (None, 3, "", "   ", ["resume"]):
            with self.subTest(junk=junk):
                self.assertIsNone(D.transport_arg_op(junk))


class VideoTransportRouteTests(unittest.TestCase):
    RESUME = "[ACTION: resume_music]"
    PAUSE = "[ACTION: pause_music]"

    def test_the_live_utterance(self):
        self.assertEqual(D.video_transport_route("Jarvis Resume movie."),
                         self.RESUME)

    def test_resume_forms(self):
        for said in ("resume the movie", "Jarvis, resume the video.",
                     "resume the show", "resume the episode",
                     "resume the film", "unpause the movie",
                     "un-pause the video", "continue the show",
                     "continue with the movie", "carry on with the film",
                     "keep playing the video", "can you resume the movie",
                     "please resume the movie", "resume my show please",
                     "hey jarvis, résumé the movie", "resume this video now"):
            with self.subTest(said=said):
                self.assertEqual(D.video_transport_route(said), self.RESUME)

    def test_pause_forms(self):
        for said in ("Jarvis Pause movie.", "pause the video",
                     "can you pause the show", "pause the episode please"):
            with self.subTest(said=said):
                self.assertEqual(D.video_transport_route(said), self.PAUSE)

    def test_everything_else_stays_with_the_brain(self):
        for said in ("resume", "continue", "pause", "resume the music",
                     "play the movie", "stop the movie", "play the movie Up",
                     "resume the movie on the left monitor",
                     "resume the movie and turn the volume up",
                     "what movie is this", "resume by Jenevieve",
                     "continue the conversation", "resume my resume",
                     "is the movie paused?", "", None, 7):
            with self.subTest(said=said):
                self.assertIsNone(D.video_transport_route(said))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class PlayMusicTransportTests(unittest.TestCase):
    def setUp(self):
        self._patch(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", True, create=True)
        self.resume = self._patch(A, "_act_resume_music",
                                  return_value="Resumed Chrome, sir.")
        self.pause = self._patch(A, "_act_pause_music",
                                 return_value="Paused Chrome, sir.")
        self.next = self._patch(A, "_act_next_song",
                                return_value="Skipped, sir.")
        self.search = self._patch(A, "_act_apple_music",
                                  return_value="streamed via apple music")
        self.said = self._patch(A, "_turn_said", return_value="")
        self.offer = self._patch(A, "_turn_taken_offer", return_value="")

    def _patch(self, *a, **k):
        p = mock.patch.object(*a, **k)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def test_the_live_argument_resumes_and_never_searches(self):
        out = _quiet(A._act_play_music, "resume")
        self.assertEqual(out, "Resumed Chrome, sir.")
        self.resume.assert_called_once_with()
        self.search.assert_not_called()
        self.pause.assert_not_called()

    def test_every_resume_form_runs_resume_music(self):
        for arg in RESUME_ARGS:
            with self.subTest(arg=arg):
                self.resume.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "Resumed Chrome, sir.")
                self.resume.assert_called_once_with()
        self.search.assert_not_called()
        self.pause.assert_not_called()

    def test_every_pause_form_runs_pause_music(self):
        for arg in PAUSE_ARGS:
            with self.subTest(arg=arg):
                self.pause.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "Paused Chrome, sir.")
                self.pause.assert_called_once_with()
        self.search.assert_not_called()
        self.resume.assert_not_called()

    def test_every_next_form_runs_next_song(self):
        for arg in NEXT_ARGS:
            with self.subTest(arg=arg):
                self.next.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "Skipped, sir.")
                self.next.assert_called_once_with()
        self.search.assert_not_called()

    def test_a_title_asked_for_by_name_is_still_a_song(self):
        # Review 2026-10-29: "Carry On" / "Stop" / "Play" are real songs.
        for said, arg in (("Jarvis play Carry On.", "Carry On"),
                          ("play Stop by the Spice Girls", "Stop"),
                          ("play Carry On and set a 5 minute timer",
                           "Carry On"),
                          ("put on Start Again", "Start Again"),
                          ("queue up Pause from Pitbull", "Pause"),
                          ("play Resume", '"Resume"'),
                          ("Jarvis, play Play.", "Play")):
            with self.subTest(said=said):
                self.said.return_value = said
                self.search.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "streamed via apple music")
                self.search.assert_called_once()
        self.resume.assert_not_called()
        self.pause.assert_not_called()
        self.next.assert_not_called()

    def test_the_turn_words_keep_the_live_case_a_resume(self):
        for said, arg in (("Jarvis Resume movie.", "resume"),
                          ("", "resume"),
                          ("Jarvis play", "play"),
                          ("play it", "it"),
                          ("play the song after resume", "resume")):
            with self.subTest(said=said, arg=arg):
                self.said.return_value = said
                self.resume.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "Resumed Chrome, sir.")
                self.resume.assert_called_once_with()
        self.search.assert_not_called()

    def test_an_accepted_song_offer_is_still_a_song(self):
        # Cross-lane review 2026-10-29: "Yes." to "Shall I play Carry On for
        # you, sir?" asks for that song; the offer he took counts as his
        # words.
        for said, offer, arg in (
                ("Yes.", "Shall I play Carry On for you, sir?", "Carry On"),
                ("Yes, go ahead.", "Shall I play Resume, sir?", "Resume"),
                ("All right.", "Shall I put on Start Again now?",
                 "Start Again"),
                ("Yes.", "Shall I play Stop by the Spice Girls?", "Stop")):
            with self.subTest(offer=offer):
                self.said.return_value = said
                self.offer.return_value = offer
                self.search.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "streamed via apple music")
                self.search.assert_called_once_with(arg)
        self.resume.assert_not_called()
        self.pause.assert_not_called()

    def test_an_accepted_transport_offer_is_still_a_transport(self):
        for said, offer, arg in (
                ("Yes.", "Shall I resume the movie?", "resume"),
                ("Yes.", "Shall I play it for you, sir?", "it"),
                ("Yes.", "Shall I play Carry On, the song you liked?",
                 "Carry On"),
                ("Yes.", "Shall I play something else, or resume?",
                 "resume")):
            with self.subTest(offer=offer):
                self.said.return_value = said
                self.offer.return_value = offer
                self.resume.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "Resumed Chrome, sir.")
                self.resume.assert_called_once_with()
        self.search.assert_not_called()

    def test_a_raising_offer_reader_keeps_the_transport(self):
        self.offer.side_effect = RuntimeError("boom")
        self.assertEqual(_quiet(A._act_play_music, "resume"),
                         "Resumed Chrome, sir.")
        self.search.assert_not_called()

    def test_real_queries_still_search_apple_music(self):
        cases = {
            "Resume by Jenevieve": "Resume by Jenevieve",
            "song:Resume": "Resume",
            "artist:Continue": "Continue",
            "play Earth Song": "play Earth Song",
            "continue by Some Artist": "continue by Some Artist",
            "Smooth Criminal": "Smooth Criminal",
        }
        for arg, query in cases.items():
            with self.subTest(arg=arg):
                self.search.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "streamed via apple music")
                self.search.assert_called_once_with(query)
        self.resume.assert_not_called()
        self.pause.assert_not_called()

    def test_flag_off_is_the_old_path_exactly(self):
        self._patch(cfg, "PLAY_MUSIC_TRANSPORT_WORDS", False)
        for arg, query in (("resume", "resume"), ("pause", "pause"),
                           ("continue the movie", "continue the movie"),
                           ("next", "next"), ("it", "it"),
                           ('"resume"', '"resume"'),
                           ("song:Resume", "Resume")):
            with self.subTest(arg=arg):
                self.search.reset_mock()
                self.assertEqual(_quiet(A._act_play_music, arg),
                                 "streamed via apple music")
                self.search.assert_called_once_with(query)
        self.resume.assert_not_called()
        self.pause.assert_not_called()
        self.next.assert_not_called()

    def test_a_raising_classifier_falls_back_to_the_search(self):
        self._patch(D, "transport_arg_op", side_effect=RuntimeError("boom"))
        self.assertEqual(_quiet(A._act_play_music, "resume"),
                         "streamed via apple music")
        self.search.assert_called_once_with("resume")
        self.resume.assert_not_called()

    def test_the_flag_ships_on(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "core", "config.py")
        with open(path, encoding="utf-8") as f:
            self.assertIn("\nPLAY_MUSIC_TRANSPORT_WORDS = True\n", f.read())


class TakenOfferReaderTests(unittest.TestCase):
    def test_the_monolith_reader_and_its_failures(self):
        bc = mock.Mock()
        bc._turn_taken_offer_text.return_value = "Shall I play Resume?"
        self.assertEqual(A._turn_taken_offer(bc), "Shall I play Resume?")
        bc._turn_taken_offer_text.side_effect = RuntimeError("boom")
        self.assertEqual(A._turn_taken_offer(bc), "")
        bc._turn_taken_offer_text.side_effect = None
        bc._turn_taken_offer_text.return_value = None
        self.assertEqual(A._turn_taken_offer(bc), "")
        with mock.patch.object(A, "_loaded_bc", return_value=None):
            self.assertEqual(A._turn_taken_offer(), "")

    def test_the_offer_tail(self):
        for offer, arg, want in (
                ("Shall I play Carry On for you, sir?", "Carry On", True),
                ("Shall I play Resume, sir?", "Resume", True),
                ("Shall I play Resume now?", "Resume", True),
                ("Shall I play Stop by the Spice Girls?", "Stop", True),
                ("Shall I resume the movie?", "resume", False),
                ("Shall I play it?", "it", False),
                ("Shall I play Carry On, the song you liked?", "Carry On",
                 False)):
            with self.subTest(offer=offer):
                self.assertIs(A._said_play_title(offer, arg, offer=True),
                              want)
        # The owner's own words keep the old, stricter tail.
        self.assertFalse(A._said_play_title(
            "Shall I play Carry On for you, sir?", "Carry On"))

    def test_a_quoted_title_and_courtesy_before_or(self):
        # Adversarial review 2026-10-29: the brain quotes titles in its
        # offers, and puts "for you, sir" before "or ...".
        for offer, arg, want in (
                ('Shall I play "Carry On" for you, sir?', "Carry On", True),
                ("Shall I play “Carry On” by fun., sir?",
                 "Carry On", True),
                ("Shall I play 'Carry On'?", "Carry On", True),
                ("Shall I play ‘Carry On’ again?", "Carry On",
                 True),
                ('Shall I play "Resume"?', '"Resume"', True),
                ("Shall I play Carry On for you, sir, or something calmer?",
                 "Carry On", True),
                ("I could play Carry On if you like, sir.", "Carry On",
                 True),
                ("Shall I play Don't Stop Me Now?", "Don't Stop Me Now",
                 True),
                ('Shall I play "it"?', "it", False),
                ("Shall I play something else, or resume?", "resume",
                 False),
                ("Shall I play Carry On Wayward Son?", "Carry On", False)):
            with self.subTest(offer=offer):
                self.assertIs(A._said_play_title(offer, arg, offer=True),
                              want)
        # The owner's own words are unchanged (quotes still count).
        self.assertFalse(A._said_play_title('play "Carry On"', "Carry On"))


class NothingToResumeTests(unittest.TestCase):
    """The REAL resume / pause handlers with the media session reporting
    nothing: their honest line is the answer; Apple Music is never searched
    for the transport word."""

    def test_nothing_to_resume_is_the_handlers_own_line(self):
        for arg, op in (("resume", "play"), ("continue the movie", "play"),
                        ("pause", "pause")):
            with self.subTest(arg=arg):
                with mock.patch.object(cfg, "PLAY_MUSIC_TRANSPORT_WORDS",
                                       True, create=True):
                    with mock.patch.object(A, "_act_apple_music",
                                           return_value="WRONG") as search:
                        with mock.patch.object(
                                A, "_smtc_transport_reply",
                                return_value=A._NOTHING_PLAYING_MSG) as smtc:
                            out = _quiet(A._act_play_music, arg)
                self.assertEqual(out, A._NOTHING_PLAYING_MSG)
                smtc.assert_called_once_with(op)
                search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
