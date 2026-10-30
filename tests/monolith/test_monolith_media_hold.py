"""The media hold for queued proactive speech (v3.0.10, MEDIA_HOLD_ENABLED).

THE LIVE EVIDENCE (session_2026-10-28_20-18-06.log, 2026-10-29): he watched
a movie in Chrome from 18:45; "[ambient-learn] skipped: media/music playing"
repeated the whole time. At 18:46:00 a queued anticipation offer was held for
room talk, the room-talk hold lapsed, and at 18:48:33 the offer was spoken
over the film; he paused the movie by voice at 18:53:21. A credits alert
(10:21:53) and a memory alert (15:24:37) were spoken over playing media too.
_presence_hold_reason knew only "gaming", "room talk" and "owner away".

  * a video playing (the media session, not his music; or the camera TV
    detector) -> "media playing": only his reminders, the guard alert and
    urgent lines speak; the rest waits;
  * the video stops -> the held lines are spoken, a stale one folded into
    the "Earlier, sir" recap; MEDIA_HOLD_MAX_S caps one hold;
  * his own music (the Apple Music web player, the Apple Music / Spotify
    apps) never holds; MEDIA_HOLD_ENABLED False = the 3.0.9 drain.

Drives the REAL _speak_pending with the presence-hold test base (voice,
input watcher and clock faked; the queue in a temp dir). Synthetic lines.

    python tools/run_tests.py monolith_media_hold
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests.monolith.test_monolith_presence_hold import _Base, TIMER

OFFER = "You typically check Claude around now, sir. Shall I bring it up?"
CREDITS = "Sir, the Claude credit balance is down to four dollars."
H2D = "Sir, the H2D reports a print failure."


class _MediaBase(_Base):
    def setUp(self):
        super().setUp()
        self._owner_spoke(60)              # he is here, on the couch
        self.video = [True]
        self._p(self.bc, "_media_hold_signal",
                side_effect=lambda: self.video[0])
        self._p(self.bc, "MEDIA_HOLD_ENABLED", True, create=True)
        self._p(self.bc, "MEDIA_HOLD_LINGER_S", 30, create=True)
        self._p(self.bc, "MEDIA_HOLD_MAX_S", 1800, create=True)
        self._cell("_media_hold_seen_at", 0.0)
        self._cell("_media_hold_since", 0.0)

    def _stop_video(self, after_s=None):
        self.video[0] = False
        self.mono[0] += (self.bc.MEDIA_HOLD_LINGER_S + 1
                         if after_s is None else after_s)
        self._owner_spoke(60)


class MovieTests(_MediaBase):
    def test_live_18_48_offer_waits_for_the_movie_then_speaks(self):
        self._write([self._e(OFFER, "anticipation")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [], "spoken over the movie")
        self.assertIn("holding queued speech: media playing", log)
        self._stop_video()
        self._drain()
        self.assertEqual(self.spoke, [OFFER])

    def test_his_timer_and_an_urgent_line_still_speak(self):
        self._write([self._e(CREDITS, "credits"),
                     self._e(TIMER, "timer"),
                     dict(self._e(H2D, "bambu"), urgent=True)])
        self._drain()
        self.assertEqual(self.spoke, [TIMER, H2D])
        self.assertEqual([e["source"] for e in self._read()], ["credits"])

    def test_a_stale_status_line_comes_out_as_the_earlier_recap(self):
        self._write([self._e(CREDITS, "credits", age_s=40 * 60)])
        self._drain()
        self.assertEqual(self.spoke, [])
        self._stop_video()
        self._drain()
        self.assertEqual(len(self.spoke), 1, self.spoke)
        self.assertTrue(self.spoke[0].startswith("Earlier, sir:"),
                        self.spoke[0])

    def test_a_flicker_of_the_signal_does_not_release_the_hold(self):
        self._write([self._e(OFFER, "anticipation")])
        self._drain()
        # The camera detector flips OFF for a few seconds mid-film.
        self.video[0] = False
        self.mono[0] += 10
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_the_hold_is_capped_and_re_arms_after_the_video_stops(self):
        self._write([self._e(OFFER, "anticipation")])
        self._drain()
        self.assertEqual(self.spoke, [])
        self.mono[0] += self.bc.MEDIA_HOLD_MAX_S + 1
        self._owner_spoke(60)
        self._drain()
        self.assertEqual(self.spoke, [OFFER], "the cap released the line")
        # Still the same show: a new line is spoken (the hold is spent)...
        self._write([self._e(CREDITS, "credits")])
        self.mono[0] += 5
        self._drain()
        self.assertEqual(self.spoke, [OFFER, CREDITS])
        # ...and once it has stopped, the next video holds again.
        self._stop_video()
        self._drain()
        self.video[0] = True
        self._write([self._e(OFFER, "anticipation")])
        self.mono[0] += 1
        self._drain()
        self.assertEqual(self.spoke, [OFFER, CREDITS])

    def test_kill_switch_is_the_3_0_9_drain(self):
        self._p(self.bc, "MEDIA_HOLD_ENABLED", False)
        self._write([self._e(OFFER, "anticipation")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [OFFER])
        self.assertNotIn("media playing", log)

    def test_away_wins_over_media(self):
        # Nobody here and the TV on: the hold is "owner away" (the recap then
        # says "While you were away").
        self.bc._last_owner_voice_at[0] = 0.0
        self._write([self._e(OFFER, "anticipation")])
        _, log = self._drain()
        self.assertIn("holding queued speech: owner away", log)

    def test_a_broken_signal_fails_open(self):
        self._p(self.bc, "_media_hold_signal", side_effect=RuntimeError("x"))
        self._write([self._e(OFFER, "anticipation")])
        self._drain()
        self.assertEqual(self.spoke, [OFFER])


@requires_monolith
class SignalTests(MonolithGlobalsTestCase):
    """_media_hold_signal: a video, not his music (the real helper; the media
    session and the TV detector faked)."""

    def setUp(self):
        self._p(self.bc, "_apple_music_web_player_titles", return_value=[])
        self.snap = [None]
        import core.media_now_playing as mnp
        self._p(mnp, "get_now_playing", side_effect=lambda: self.snap[0])
        mods = mock.patch.dict(sys.modules)
        mods.start()
        self.addCleanup(mods.stop)
        sys.modules.pop("skill_tv_detect", None)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _sig(self):
        return self.bc._media_hold_signal()

    def test_a_chrome_video_is_a_video(self):
        self.snap[0] = {"app": "Chrome", "title": "Movie", "artist": "",
                        "album": "", "playing": True}
        self.assertTrue(self._sig())

    def test_a_paused_video_is_not(self):
        self.snap[0] = {"app": "Chrome", "title": "Movie", "playing": False}
        self.assertFalse(self._sig())

    def test_the_apple_music_web_player_is_music(self):
        self.snap[0] = {"app": "Chrome", "title": "Song", "artist": "Band",
                        "album": "Record", "playing": True}
        self.assertFalse(self._sig())

    def test_the_music_apps_are_music(self):
        for app in ("Apple Music", "Spotify", "iTunes"):
            self.snap[0] = {"app": app, "title": "Song", "playing": True}
            self.assertFalse(self._sig(), app)

    def test_the_camera_tv_detector_counts(self):
        tv = mock.MagicMock()
        tv.is_tv_on.return_value = True
        sys.modules["skill_tv_detect"] = tv
        self.assertTrue(self._sig())

    def test_a_broken_probe_is_not_a_video(self):
        import core.media_now_playing as mnp
        self._p(mnp, "get_now_playing", side_effect=RuntimeError("smtc"))
        tv = mock.MagicMock()
        tv.is_tv_on.side_effect = RuntimeError("cam")
        sys.modules["skill_tv_detect"] = tv
        self.assertFalse(self._sig())


if __name__ == "__main__":
    unittest.main()
