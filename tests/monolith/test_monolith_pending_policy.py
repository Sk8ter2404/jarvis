"""Monolith wiring of core/pending_policy in _speak_pending, of the clap's wider
media check, and of the rescue measure. Replays the live morning of 2026-11-05:
a queue left on disk for a week by the run before."""
from __future__ import annotations

import inspect
import time
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_presence_hold import _Base

DAY = 86400.0
WEEK = 7 * DAY
ARRIVAL = ("[intent:briefing] Good morning, sir. "
           "I couldn't get the news just now.")
OLD_RECAP = ("[intent:briefing] Sir, today played 1 tracks. Shall I queue the "
             "same morning briefing for tomorrow?")
OLD_EVENING = "[intent:briefing] Good evening, sir. A quiet day on the voice channel."
ACK = "You rang, sir?"
MIC = "Switched to the Blue Snowball, sir."
RAM = "Sir, memory usage is at 92 percent. Chrome has 9 gigabytes."
DISK = "Sir, the system drive is nearly full."


def _at_hour(hour):
    return mock.patch("core.pending_policy._local_hour", return_value=hour)


@requires_monolith
class LiveMorningTests(_Base):
    def setUp(self):
        super().setUp()
        self.input_age[0] = 3.0      # he just sat down
        self._p(self.bc, "PENDING_SOURCE_POLICY", True, create=True)
        self._p(self.bc, "PENDING_GREETING_FIRST", True, create=True)

    def test_the_week_old_queue_is_not_read_out_only_the_greeting_is(self):
        self._write([self._e(OLD_RECAP, "recap", WEEK),
                     self._e(OLD_EVENING, "evening", WEEK),
                     self._e(ACK, "clap", WEEK),
                     self._e(MIC, "audio", WEEK),
                     self._e(ARRIVAL, "arrival_v2", 1)])
        with _at_hour(10):
            _, log = self._drain()
        self.assertEqual(self.spoke, [ARRIVAL])
        self.assertIn("[pending] too old to say (evening;", log)
        self.assertIn("[pending] too old to say (clap;", log)
        self.assertNotIn("folded into the recap", log)
        self.assertEqual(self._read(), [])

    def test_only_old_lines_leave_nothing_said_and_an_empty_queue(self):
        self._write([self._e(OLD_EVENING, "evening", WEEK)])
        with _at_hour(10):
            out, _log = self._drain()
        self.assertFalse(out)
        self.assertEqual(self.spoke, [])
        self.assertEqual(self._read(), [])

    def test_the_flag_off_keeps_every_line_as_before(self):
        self._p(self.bc, "PENDING_SOURCE_POLICY", False, create=True)
        self._write([self._e(OLD_EVENING, "evening", WEEK)])
        self._drain()
        self.assertEqual(self.spoke, [OLD_EVENING])

    def test_an_evening_line_that_waited_is_not_said_before_five(self):
        self._write([self._e(OLD_EVENING, "evening", 40 * 60)])
        with _at_hour(16):
            self._drain()
        self.assertEqual(self.spoke, [])

    def test_an_evening_line_that_waited_is_said_at_night(self):
        self._write([self._e(OLD_EVENING, "evening", 40 * 60)])
        with _at_hour(22):
            self._drain()
        self.assertEqual(self.spoke, [OLD_EVENING])

    def test_an_evening_briefing_he_just_asked_for_speaks(self):
        self._write([self._e(OLD_EVENING, "evening", 2)])
        with _at_hour(11):
            self._drain()
        self.assertEqual(self.spoke, [OLD_EVENING])

    def test_the_greeting_goes_before_the_recap(self):
        self._write([self._e(DISK, "disk", 2 * 3600),
                     self._e(ARRIVAL, "arrival_v2", 1)])
        with _at_hour(10):
            self._drain()
        self.assertEqual(len(self.spoke), 2, self.spoke)
        self.assertEqual(self.spoke[0], ARRIVAL)
        self.assertTrue(self.spoke[1].endswith("system drive is nearly full."),
                        self.spoke[1])

    def test_greeting_first_off_keeps_queue_order(self):
        self._p(self.bc, "PENDING_GREETING_FIRST", False, create=True)
        self._write([self._e(DISK, "disk", 2 * 3600),
                     self._e(ARRIVAL, "arrival_v2", 1)])
        with _at_hour(10):
            self._drain()
        self.assertEqual(self.spoke[1], ARRIVAL)

    def test_a_clap_ack_is_said_at_once_or_not_at_all(self):
        self._write([self._e(ACK, "clap", 2)])
        self._drain()
        self.assertEqual(self.spoke, [ACK])
        self.spoke.clear()
        self.bc._recent_spoken_messages.clear()
        self._write([self._e(ACK, "clap", 96)])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_a_clap_ack_that_waited_out_room_talk_is_dropped(self):
        # Live 11:05:19 -> 11:06:55. Held, then expired, never queued behind.
        self._write([self._e(ACK, "clap", 2)])
        with mock.patch.object(self.bc, "_presence_hold_reason",
                               return_value="room talk"):
            self._drain()
        self.assertEqual(self.spoke, [])
        self.assertEqual(len(self._read()), 1, "a held line is left on disk")
        rows = self._read()
        rows[0]["ts"] = time.time() - 96
        self._write(rows)
        self._drain()
        self.assertEqual(self.spoke, [])
        self.assertEqual(self._read(), [])

    def test_a_mic_change_is_not_folded_and_still_said_when_young(self):
        self._write([self._e(MIC, "audio", 60)])
        self._drain()
        self.assertEqual(self.spoke, [MIC])

    def test_stale_ram_is_not_in_the_recap_but_other_status_is(self):
        # Live 12:55:14 and 14:44:19.
        self._write([self._e(RAM, "system_monitor", 15 * 60),
                     self._e(DISK, "system_monitor", 15 * 60)])
        _, log = self._drain()
        self.assertEqual(len(self.spoke), 1, self.spoke)
        self.assertNotIn("memory", self.spoke[0])
        self.assertIn("system drive", self.spoke[0])
        self.assertIn("ram:", log)

    def test_fresh_ram_is_spoken_whole(self):
        self._write([self._e(RAM, "system_monitor", 30)])
        self._drain()
        self.assertEqual(self.spoke, [RAM])

    def test_his_timer_survives_any_wait(self):
        msg = "Reminder, sir: your timer is up."
        self._write([self._e(msg, "timer", WEEK)])
        self._drain()
        self.assertEqual(self.spoke, [msg])


@requires_monolith
class ClapMediaRecentTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.bc, "CLAP_MEDIA_LINGER_S", 30.0, create=True)
        self._p(self.bc, "MEDIA_HOLD_ENABLED", True, create=True)
        self.state = dict(self.bc._music_state)
        self._p(self.bc, "_music_state", self.state)
        self.hold = self._p(self.bc, "_media_hold_active", return_value=False)

    def test_nothing_playing_lets_a_real_clap_through(self):
        self.state.update(at=None, playing=False, playing_at=None)
        self.assertFalse(self.bc._clap_media_recent())

    def test_the_shared_reading_says_playing(self):
        self.state.update(at=time.monotonic(), playing=True)
        self.assertTrue(self.bc._clap_media_recent())

    def test_a_gap_within_the_linger_still_counts(self):
        # Live 11:04:54 "music ended", clap at 11:05:19.
        self.state.update(at=time.monotonic(), playing=False,
                          playing_at=time.monotonic() - 25.0)
        self.assertTrue(self.bc._clap_media_recent())

    def test_past_the_linger_it_does_not(self):
        self.state.update(at=time.monotonic(), playing=False,
                          playing_at=time.monotonic() - 45.0)
        self.assertFalse(self.bc._clap_media_recent())

    def test_the_media_hold_counts(self):
        self.state.update(at=None, playing=False, playing_at=None)
        self.hold.return_value = True
        self.assertTrue(self.bc._clap_media_recent())

    def test_a_failure_does_not_block_a_clap(self):
        self.hold.side_effect = RuntimeError("boom")
        self.state.update(at=None, playing=False, playing_at=None)
        self.assertFalse(self.bc._clap_media_recent())

    def test_the_refresh_stamps_playing_at(self):
        self.state.update(at=None, playing=False, playing_at=None)
        with mock.patch.object(self.bc, "_music_read_state",
                               return_value={"music": False, "peak": 0.5,
                                             "room": False, "playing": True}):
            self.bc._music_state_refresh()
        self.assertIsNotNone(self.state.get("playing_at"))


@requires_monolith
class RescueMeasureTests(_Base):
    def test_usable_means_the_wake_gates_would_let_it_in(self):
        primary = None
        for v in vars(self.bc).values():
            if type(v).__name__ == "Primary":
                primary = v
                break
        self.assertIsNotNone(primary)
        with mock.patch.object(self.bc, "_parakeet_wake_lost",
                               return_value=False):
            self.assertTrue(primary._useful("Jarvis, lights on"))
            self.assertFalse(primary._useful("   "))
        with mock.patch.object(self.bc, "_parakeet_wake_lost",
                               return_value=True):
            self.assertFalse(primary._useful("the telly talking"))


@requires_monolith
class BrainWarmupWiringTests(_Base):
    def test_the_warmup_thread_reports_its_start_and_end(self):
        src = inspect.getsource(self.bc._warm_up_local_llm_async)
        self.assertIn("_brain_warmup_mod.note_start()", src)
        self.assertIn("finally:", src)
        self.assertIn("_brain_warmup_mod.note_done()", src)


if __name__ == "__main__":
    unittest.main()
