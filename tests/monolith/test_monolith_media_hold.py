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
        self._p(self.bc, "MEDIA_HOLD_SIGNAL_CACHE_S", 0, create=True)
        self._p(self.bc, "MEDIA_HOLD_PASS_ASKED", True, create=True)
        self._cell("_media_hold_seen_at", 0.0)
        self._cell("_media_hold_since", 0.0)
        self._cell("_media_hold_signal_cache", 0.0)

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
        # The drain asks on every pass while a line is held (v3.0.10 review:
        # a gap with no asks starts a new hold).
        step = 25
        for _ in range(self.bc.MEDIA_HOLD_MAX_S // step + 1):
            self.mono[0] += step
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

    def test_a_later_video_after_a_gap_with_no_checks_is_held(self):
        # v3.0.10 review (sim_media.py): video A held a line, the queue then
        # emptied (he walked away, a cap release), so no drain asked for an
        # hour; video B's line was never held because the hold's start was
        # still video A's.
        self._write([self._e(OFFER, "anticipation")])
        self._drain()
        self.assertEqual(self.spoke, [])
        self._write([])
        self.mono[0] += 3600
        self._owner_spoke(60)
        self._write([self._e(CREDITS, "credits")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [], "spoken over video B")

    def test_a_result_he_asked_for_speaks_through_the_hold(self):
        # v3.0.10 review: "find the cheapest flight" ran in the background,
        # he started a video; the browser result used to wait 30 min and
        # come out as a recap fragment.
        asked = dict(self._e("I found three flights, sir.", "browser"),
                     history_text="I found three flights: ...")
        self._write([self._e(CREDITS, "credits"), asked])
        self._drain()
        self.assertEqual(self.spoke, ["I found three flights, sir."])
        self.assertEqual([e["source"] for e in self._read()], ["credits"])

    def test_pass_asked_off_holds_the_result_too(self):
        self._p(self.bc, "MEDIA_HOLD_PASS_ASKED", False)
        asked = dict(self._e("I found three flights, sir.", "browser"),
                     history_text="I found three flights: ...")
        self._write([asked])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_the_signal_is_cached_between_drain_passes(self):
        # v3.0.10 review: each read can be a synchronous media-session
        # refresh; a held line made every drain pass read it.
        self._p(self.bc, "MEDIA_HOLD_SIGNAL_CACHE_S", 5)
        sig = self._p(self.bc, "_media_hold_signal", return_value=True)
        self._write([self._e(OFFER, "anticipation")])
        self._drain()
        self.mono[0] += 2
        self._drain()
        self.assertEqual(sig.call_count, 1)
        self.mono[0] += 5
        self._drain()
        self.assertEqual(sig.call_count, 2)
        self.assertEqual(self.spoke, [])

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


class FinishLineThroughTheHoldTests(_MediaBase):
    """v3.0.10 cross-lane review (lane D's media hold x lane F's
    FINISH_LINE_KEEP): a print that finishes during his film. The finish
    line and the companion's offer wait, whole and in order, and are both
    spoken once the film stops - the offer never alone, the head never lost.
    The REAL bambu_monitor is the live skill."""

    HEAD = ("Print complete, sir — 'cube' is finished. "
            "Shall I notify you when the bed has cooled?")
    TAIL = "Shall I dim the workshop lights for you, sir?"

    def setUp(self):
        super().setUp()
        from tests.skills.test_bambu_monitor import _load_bambu
        self.bm, _ = _load_bambu()
        p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": self.bm})
        p.start()
        self.addCleanup(p.stop)
        self.bm._finish_line_passed_at[0] = 0.0
        from core import config
        self._p(config, "FINISH_LINE_KEEP", True, create=True)
        with self.bm._state_lock:
            self.bm._state.update(gcode_state="FINISH", filename="cube.3mf",
                                  task_id="7")

    def test_held_through_the_film_then_head_and_offer_in_order(self):
        self._write([
            dict(self._e(self.HEAD, "bambu", age_s=20 * 60),
                 still_valid="bambu_finish:cube|job=7"),
            dict(self._e(self.TAIL, "print_companion", age_s=20 * 60),
                 still_valid="bambu_finish_offer:"),
        ])
        self._drain()
        self.assertEqual(self.spoke, [], "spoken over the film")
        self._stop_video()
        _, log = self._drain()
        self.assertEqual(self.spoke, [self.HEAD, self.TAIL])
        self.assertNotIn("finish offer dropped", log)


@requires_monolith
class ArrivalQuietTests(MonolithGlobalsTestCase):
    """v3.0.10 review (ARRIVAL_QUIET_ALL_MOMENTS): inside the quiet period
    after the arrival briefing, the other moment-lines are not queued and
    no spontaneous remark starts. The real proactive_announce, its queue
    file in a temp dir."""

    def setUp(self):
        import os
        import tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        self.path = os.path.join(tmp, "pending_speech.json")
        self._p(self.bc, "__file__", os.path.join(tmp, "bobert_companion.py"))
        self._p(self.bc, "focus_mode_active", return_value=False)
        self._p(self.bc, "ARRIVAL_QUIET_ALL_MOMENTS", True, create=True)
        import core.arrival_quiet as aq
        self.quiet = [True]
        self._p(aq, "arrival_quiet_active",
                side_effect=lambda now=None: self.quiet[0])

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _queued(self):
        import json
        import os
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [e["source"] for e in json.load(f)]

    def test_a_wellness_nudge_is_not_queued_behind_the_briefing(self):
        self.assertFalse(self.bc.proactive_announce(
            "Hydration, perhaps, sir?", source="wellness"))
        self.assertEqual(self._queued(), [])

    def test_status_and_urgent_lines_are_still_queued(self):
        self.assertTrue(self.bc.proactive_announce(CREDITS, source="credits"))
        self.assertTrue(self.bc.proactive_announce(
            "Posture, sir.", source="posture", urgent=True))
        self.assertEqual(self._queued(), ["credits", "posture"])

    def test_after_the_quiet_it_is_queued(self):
        self.quiet[0] = False
        self.assertTrue(self.bc.proactive_announce(
            "Hydration, perhaps, sir?", source="wellness"))
        self.assertEqual(self._queued(), ["wellness"])

    def test_switch_off_is_the_first_rule(self):
        self._p(self.bc, "ARRIVAL_QUIET_ALL_MOMENTS", False)
        self.assertTrue(self.bc.proactive_announce(
            "Hydration, perhaps, sir?", source="wellness"))
        self.assertEqual(self._queued(), ["wellness"])

    def test_no_spontaneous_remark_inside_the_quiet(self):
        self._p(self.bc, "PROACTIVE_ENABLED", True)
        self._p(self.bc, "_game_quiet_active", return_value=False)
        self.assertFalse(self.bc.should_be_proactive())


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

    def _tv(self, calibrated):
        tv = mock.MagicMock()
        tv.is_tv_on.return_value = True
        if calibrated is None:
            tv._store.return_value = None
        else:
            tv._store.return_value.is_calibrated.return_value = calibrated
        sys.modules["skill_tv_detect"] = tv

    def test_an_uncalibrated_tv_detector_does_not_count(self):
        # v3.0.10 review: no data/tv_region.json live; the whole-view
        # detector turned ON 123 times while he worked at the desk.
        self._p(self.bc, "MEDIA_HOLD_TV_NEEDS_REGION", True, create=True)
        self._tv(False)
        self.assertFalse(self._sig())
        self._tv(None)
        self.assertFalse(self._sig())
        self._tv(True)
        self.assertTrue(self._sig())

    def test_tv_needs_region_off_counts_it_uncalibrated(self):
        self._p(self.bc, "MEDIA_HOLD_TV_NEEDS_REGION", False, create=True)
        self._tv(False)
        self.assertTrue(self._sig())

    def test_a_song_in_any_player_is_music(self):
        # v3.0.10 review: Groove / Amazon Music read as "Media Player".
        self._p(self.bc, "MEDIA_HOLD_ALBUM_IS_MUSIC", True, create=True)
        for app in ("Media Player", "VLC", "TIDAL"):
            self.snap[0] = {"app": app, "title": "Song", "artist": "Band",
                            "album": "Record", "playing": True}
            self.assertFalse(self._sig(), app)
        self.snap[0] = {"app": "VLC", "title": "Film", "artist": "",
                        "album": "", "playing": True}
        self.assertTrue(self._sig())

    def test_album_is_music_off_is_the_first_rule(self):
        self._p(self.bc, "MEDIA_HOLD_ALBUM_IS_MUSIC", False, create=True)
        self.snap[0] = {"app": "Media Player", "title": "Song",
                        "artist": "Band", "album": "Record", "playing": True}
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
