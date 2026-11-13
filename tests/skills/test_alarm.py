"""skills/alarm.py - a wake-up alarm that rings until someone answers it.

Live 2026-11-12 11:32-11:34 four asks for "an actual alarm that will wake me
up" each became set_timer: one spoken line, then silence.

Nothing real is reached: the tone, the volume, the input tick, the speaker
switch and the speech queue are the module's hooks, replaced here.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SAVED_DATA_DIR_ENV = None
_DATA_TMPDIR = None


def setUpModule() -> None:
    global _SAVED_DATA_DIR_ENV, _DATA_TMPDIR
    _SAVED_DATA_DIR_ENV = os.environ.get("JARVIS_DATA_DIR")
    _DATA_TMPDIR = tempfile.mkdtemp(prefix="jarvis_alarm_test_")
    os.environ["JARVIS_DATA_DIR"] = _DATA_TMPDIR


def tearDownModule() -> None:
    if _SAVED_DATA_DIR_ENV is None:
        os.environ.pop("JARVIS_DATA_DIR", None)
    else:
        os.environ["JARVIS_DATA_DIR"] = _SAVED_DATA_DIR_ENV
    if _DATA_TMPDIR:
        shutil.rmtree(_DATA_TMPDIR, ignore_errors=True)


class _Rig:
    """The machine, faked: what the alarm did to it, in order."""

    def __init__(self):
        self.events = []
        self.volume = (0.30, True)       # quiet and muted, as at night
        self.tick = 1000
        self.said = []

    def install(self, mod, case):
        # Set for good, never un-patched: the module is this test's private
        # copy, and a ring thread still on its way out after a FAILED test
        # used to find the real hooks back in place (the real volume write,
        # the real PlaySound).
        def patch(name, fn):
            setattr(mod, name, fn)

        patch("_sound_start", lambda path: self.events.append(("start", os.path.basename(path))))
        patch("_sound_stop", lambda: self.events.append(("stop",)))
        patch("_volume_read", lambda: self.volume)
        patch("_volume_write", self._write)
        patch("_input_tick", lambda: self.tick)
        patch("_to_speakers", lambda: self.events.append(("speakers",)))
        patch("_say", lambda message, key=None: self.said.append(message))
        # Never the real PC-speaker beep, never real COM, no real waiting.
        patch("_beep", self._beep)
        patch("_com_enter", lambda: self.events.append(("com",)))
        patch("ALARM_TONE_RETRY_S", 0.0)

    def _beep(self):
        self.events.append(("beep",))
        time.sleep(0.01)

    def _write(self, level, muted):
        self.volume = (float(level), bool(muted))
        self.events.append(("volume", round(float(level), 2), bool(muted)))
        return True


class _AlarmCase(unittest.TestCase):
    def setUp(self):
        for name in ("alarms.json",):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(_DATA_TMPDIR, name))
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod, self.actions = load_skill_isolated("alarm")
        self.rig = _Rig()
        self.rig.install(self.mod, self)
        self.addCleanup(self._cancel_all)

    def _cancel_all(self):
        self.mod._stop_ringing("test over")
        with self.mod._lock:
            rows = list(self.mod._alarms.values())
            self.mod._alarms.clear()
        for a in rows:
            a["timer"].cancel()

    def _quiet(self, fn, *a, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **kw)

    def _ring_in_thread(self, label=""):
        t = threading.Thread(target=lambda: self._quiet(self.mod.ring, 7, label),
                             daemon=True)
        self.addCleanup(t.join, 5.0)       # after _cancel_all has stopped it
        t.start()
        deadline = time.monotonic() + 5.0
        # Ringing AND the tone started (the first ring writes the tone file).
        while time.monotonic() < deadline and not (
                self.mod.is_ringing()
                and any(e[0] == "start" for e in self.rig.events)):
            time.sleep(0.01)
        self.assertTrue(self.mod.is_ringing())
        return t


class ParseTests(_AlarmCase):
    def test_clock_times_and_durations(self):
        p = self.mod._parse_when
        self.assertEqual(p("20 minutes"), 1200)
        self.assertEqual(p("in an hour"), 3600)
        for text in ("7:00 am", "at 6:30 pm", "07:15"):
            with self.subTest(text=text):
                secs = p(text)
                self.assertIsNotNone(secs)
                self.assertTrue(0 < secs <= 86400)

    def test_a_bare_hour_is_the_next_time_the_clock_reads_it(self):
        now = datetime(2026, 11, 12, 22, 30).timestamp()       # 10:30 PM
        with mock.patch.object(self.mod.time, "time", return_value=now):
            self.assertEqual(self.mod._parse_when("at 7"), int(8.5 * 3600))
            self.assertEqual(self.mod._parse_when("at 11"), 1800)

    def test_a_clock_time_as_speech_to_text_writes_it(self):
        # Live 2026-11-12 16:21:15: "alarm for 4:22" arrived as "422" and was
        # set 422 minutes ahead (11:23 PM) instead of one minute ahead.
        now = datetime(2026, 11, 12, 16, 21).timestamp()        # 4:21 PM
        p = self.mod._parse_when
        with mock.patch.object(self.mod.time, "time", return_value=now):
            for text in ("422", "for 422", "4 22", "4:22", "4.22", "at 4 22",
                         "422 pm", "4 22 p.m."):
                with self.subTest(text=text):
                    self.assertEqual(p(text), 60)
            self.assertEqual(p("1030"), int(6.15 * 3600))        # 10:30 PM
            self.assertEqual(p("730"), int(3.15 * 3600))         # 7:30 PM
            self.assertEqual(p("7:30 am"), int(15.15 * 3600))
            self.assertEqual(p("0715"), int(14.9 * 3600))        # 24-hour
            self.assertEqual(p("1630"), 9 * 60)
            # One that has just passed is the next one, twelve hours on.
            self.assertEqual(p("4 20"), 12 * 3600 - 60)

    def test_the_half_of_the_day_in_words(self):
        # Live 2026-11-12 16:22:30: "5.30 in the morning".
        now = datetime(2026, 11, 12, 16, 21).timestamp()        # 4:21 PM
        p = self.mod._parse_when
        H = 3600
        with mock.patch.object(self.mod.time, "time", return_value=now):
            self.assertEqual(p("5.30 in the morning"), int(13.15 * H))
            self.assertEqual(p("530 tomorrow morning"), int(13.15 * H))
            self.assertEqual(p("7 in the morning"), int(14.65 * H))
            self.assertEqual(p("at 7 in the morning"), int(14.65 * H))
            self.assertEqual(p("6 in the evening"), int(1.65 * H))
            self.assertEqual(p("9 tonight"), int(4.65 * H))
            self.assertEqual(p("9:15 tonight"), int(4.9 * H))
            self.assertEqual(p("2 at night"), int(9.65 * H))
            self.assertEqual(p("12 tonight"), int(7.65 * H))
            self.assertEqual(p("midnight"), int(7.65 * H))
            self.assertEqual(p("noon"), int(19.65 * H))
            # Still durations.
            self.assertEqual(p("in 20 minutes tonight"), 1200)
            self.assertEqual(p("in 5 tonight"), 300)

    def test_times_said_in_words(self):
        now = datetime(2026, 11, 12, 16, 21).timestamp()        # Thu 4:21 PM
        p = self.mod._parse_when
        with mock.patch.object(self.mod.time, "time", return_value=now):
            for said, same_as in (
                    # They were minutes ("seven am" = 7 minutes) or nothing.
                    ("seven am", "7 am"), ("eleven pm", "11 pm"),
                    ("seven thirty", "7:30"), ("six thirty in the morning", "6:30 am"),
                    ("seven fifteen", "7:15"), ("five forty-five", "5:45"),
                    ("seven oh five", "7:05"), ("six-thirty", "6:30"),
                    ("half past six", "6:30"), ("half past 6 in the morning", "6:30 am"),
                    ("quarter to seven", "6:45"), ("a quarter after six", "6:15"),
                    ("ten to seven", "6:50"), ("20 minutes before 7", "6:40"),
                    ("at seven", "at 7"), ("six o'clock", "at 6"), ("7 o'clock", "at 7"),
                    ("7ish", "at 7"), ("around 7", "at 7"), ("by 7", "at 7"),
                    ("6 in the am", "6 am"), ("6 every morning", "6 am"),
                    ("morning alarm at 6", "6 am"), ("at one", "at 1"),
                    ("one o'clock", "at 1"), ("7:00:00 AM", "7 am"),
                    ("2026-11-13 07:00", "07:00"), ("7 oh 5 am", "7:05 am")):
                with self.subTest(said=said):
                    self.assertIsNotNone(p(same_as))
                    self.assertEqual(p(said), p(same_as))

    def test_lengths_of_time_keep_every_part(self):
        p = self.mod._parse_when
        for said, secs in (
                ("7 and a half hours", 27000), ("twenty five minutes", 1500),
                ("forty-five minutes", 2700), ("1 hour 30", 5400),
                ("8 hours 15", 29700), ("in a couple of hours", 7200),
                ("for 10 more minutes", 600), ("for 2 and a half hours", 9000),
                ("for 10 m", 600), ("for 8 h", 28800), ("five to ten minutes", 600),
                ("in ten", 600), ("one minute", 60), ("three minutes", 180),
                ("in one hour", 3600), ("in 20 minutes to take pills", 1200),
                ("in 1 hour after lunch", 3600)):
            with self.subTest(said=said):
                self.assertEqual(p(said), secs)

    def test_tomorrow_is_never_later_today(self):
        H = 3600
        p = self.mod._parse_when
        for hh, mm, said, want in (
                (16, 21, "tomorrow at 5 pm", 24 * H + 39 * 60),
                (16, 21, "tomorrow at 4:30 pm", 24 * H + 9 * 60),
                (16, 21, "at 7 tomorrow", int(14.65 * H)),
                (16, 21, "7 tomorrow", int(14.65 * H)),
                (16, 21, "8 pm tomorrow", int(27.65 * H)),
                (16, 21, "tomorrow at noon", int(19.65 * H)),
                (6, 10, "7 am tomorrow", 24 * H + 50 * 60),
                (6, 10, "tomorrow morning at 7", 24 * H + 50 * 60),
                # Before 4 AM "tomorrow" is the morning that is coming.
                (1, 0, "7 am tomorrow", 6 * H),
                # About something else: nothing moves.
                (6, 10, "wake me at 7 am, i have a test tomorrow", 50 * 60),
                (16, 21, "alarm at 9 tonight to remind me about the morning meeting",
                 int(4.65 * H))):
            now = datetime(2026, 11, 12, hh, mm).timestamp()
            with self.subTest(said=said, at=(hh, mm)), \
                    mock.patch.object(self.mod.time, "time", return_value=now):
                self.assertEqual(p(said), want)

    def test_the_first_time_in_the_ask_is_the_alarm(self):
        # v3.0.19 let a later number win: "7 am for the 9 30 lecture" rang
        # at 9:30 PM.
        now = datetime(2026, 11, 12, 16, 21).timestamp()
        p = self.mod._parse_when
        with mock.patch.object(self.mod.time, "time", return_value=now):
            seven_am, six_am = p("7 am"), p("6 am")
            for said, want in (
                    ("7 am for the 9 30 lecture", seven_am),
                    ("set an alarm for 7 am, i have a meeting at 9:30", seven_am),
                    ("i have a class at 9, wake me at 7 in the morning", seven_am),
                    ("wake me at 6 in the morning because i have class at 8 am", six_am),
                    ("i've got a 9 am class so wake me at 7 30", p("7:30")),
                    # A length of time said first is that length of time.
                    ("alarm in 30 minutes, my 9 am class", 1800),
                    ("wake me in 20 minutes for my 9 am class", 1200),
                    ("jarvis setting an alarm for 422. that's really good.", 60)):
                with self.subTest(said=said):
                    self.assertEqual(p(said), want)

    def test_words_about_other_things_do_not_move_the_alarm(self):
        H = 3600
        p = self.mod._parse_when
        for hh, mm, said, want in (
                # "tomorrow" / "tonight" in another clause moved it a day or
                # half a day.
                (16, 21, "wake me at 7 pm, i have a class tomorrow at 9 am", int(2.65 * H)),
                (16, 21, "alarm for 6 pm, meeting tomorrow at 8", int(1.65 * H)),
                (6, 10, "wake me at 7 am, i have class tomorrow at 9", 50 * 60),
                (23, 30, "wake me up at 6 i have a game tonight", int(6.5 * H)),
                # A time in the talk before the ask.
                (16, 21, "i have class at 9. set one for 7:30 am", int(15.15 * H)),
                (16, 21, "i have a meeting at 9 set it for 7 am", int(14.65 * H)),
                (16, 21, "class is at 9, make it 7 am", int(14.65 * H)),
                (16, 21, "meeting at 2 pm. alarm 7 am", int(14.65 * H)),
                # A length of time first, with or without "in".
                (16, 21, "20 minutes, i have class at 9", 1200),
                (16, 21, "20 minutes for my two o'clock", 1200),
                (16, 21, "wake me 20 minutes before my 8 am class", int(15.65 * H) - 1200),
                (16, 21, "wake me an hour before my 9 am class", int(15.65 * H)),
                # "tomorrow" with a comma or "about" before the time.
                (16, 21, "tomorrow, 5 pm", 24 * H + 39 * 60),
                (16, 21, "tomorrow at about 5 pm", 24 * H + 39 * 60),
                (16, 21, "tomorrow at 12", int(19.65 * H)),
                (16, 21, "early tomorrow at 5", int(12.65 * H)),
                (16, 21, "7 in the a.m.", int(14.65 * H)),
                (6, 10, "set an alarm for 6 30 a.m. tomorrow", 24 * H + 20 * 60),
                (16, 21, "good morning, set an alarm for 2 30", int(10.15 * H)),
                (16, 21, "0700 hours", int(14.65 * H))):
            now = datetime(2026, 11, 12, hh, mm).timestamp()
            with self.subTest(said=said, at=(hh, mm)), \
                    mock.patch.object(self.mod.time, "time", return_value=now):
                self.assertEqual(p(said), want)

    def test_tomorrow_keeps_the_hour_on_a_daylight_saving_night(self):
        for y, mo, d in ((2027, 3, 13), (2026, 10, 31), (2026, 11, 12)):
            now = datetime(y, mo, d, 6, 10).timestamp()
            with self.subTest(day=(y, mo, d)), \
                    mock.patch.object(self.mod.time, "time", return_value=now):
                secs = self.mod._parse_when("7 am tomorrow")
            ring = datetime.fromtimestamp(now + secs)
            self.assertEqual((ring.hour, ring.minute), (7, 0))
            self.assertEqual((ring.date() - datetime(y, mo, d).date()).days, 1)

    def test_number_words_in_lengths_of_time(self):
        p = self.mod._parse_when
        for said, secs in (("twenty one minutes", 1260), ("thirty-one minutes", 1860),
                           ("sixty five minutes", 3900), ("ninety minutes", 5400),
                           ("seventy minutes", 4200), ("one and a half hours", 5400),
                           ("quarter of an hour", 900),
                           ("three quarters of an hour", 2700)):
            with self.subTest(said=said):
                self.assertEqual(p(said), secs)

    def test_a_label_is_never_read_as_the_time(self):
        now = datetime(2026, 11, 12, 16, 21).timestamp()
        with mock.patch.object(self.mod.time, "time", return_value=now):
            sp = self.mod._split_args
            self.assertEqual(sp("7 am | take 2 pills"), (52740, "take 2 pills"))
            self.assertEqual(sp("6:30 am | 5k run at 8"), (50940, "5k run at 8"))
            self.assertEqual(sp("gym | 6:30"), (7740, "gym"))
            self.assertEqual(sp("| gym"), (None, "gym"))
            # The swap is for a label half that is nothing but a time.
            self.assertEqual(sp("lunch | take 2 pills"), (None, "take 2 pills"))
            self.assertEqual(sp("tomorrow | gym at 7"), (None, "gym at 7"))
            self.assertEqual(sp("gym | at 6 30 in the morning"), (50940, "gym"))

    def test_a_number_with_a_unit_or_after_in_stays_a_duration(self):
        p = self.mod._parse_when
        self.assertEqual(p("100 seconds"), 100)
        self.assertEqual(p("120 minutes"), 7200)
        self.assertEqual(p("in 422"), 422 * 60)
        self.assertEqual(p("1 hour 30 minutes"), 5400)
        self.assertEqual(p("45"), 45 * 60)
        self.assertEqual(p("90m"), 5400)

    def test_no_time_is_none_and_sets_nothing(self):
        self.assertIsNone(self.mod._parse_when("when I get tired"))
        out = self.actions["set_alarm"]("loud please")
        self.assertIn("What time should it ring", out)
        # No failure word: the line is spoken as written, not reworded.
        for word in ("couldn't", "can't", "unable", "failed", "sorry"):
            self.assertNotIn(word, out.lower())
        self.assertEqual(self.mod._snapshot(), [])


class SetListCancelTests(_AlarmCase):
    def test_set_alarm_arms_saves_and_says_when(self):
        out = self.actions["set_alarm"]("20 minutes")
        self.assertIn("Alarm #1 set", out)
        self.assertIn("20 minutes from now", out)
        rows = self.mod._snapshot()
        self.assertEqual(len(rows), 1)
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved[0]["id"], 1)
        self.assertAlmostEqual(saved[0]["at"], time.time() + 1200, delta=5)
        self.assertIn("One alarm", self.actions["list_alarms"](""))

    def test_the_same_ask_twice_is_one_alarm(self):
        self.actions["set_alarm"]("20 minutes | work")
        out = self.actions["set_alarm"]("20 minutes | work")
        self.assertIn("already set", out)
        self.assertEqual(len(self.mod._snapshot()), 1)
        self.assertIn("Alarm #", self.actions["set_alarm"]("20 minutes | gym"))

    def test_the_time_confirmed_is_the_time_asked(self):
        # Seconds-until was rounded down, so now + that landed in the minute
        # before: "7:00 am" was confirmed as "6:59 AM".
        for frac in (0.0, 0.4, 0.999):
            now = datetime(2026, 11, 12, 16, 21).timestamp() + frac
            with self.subTest(frac=frac), \
                    mock.patch.object(self.mod.time, "time", return_value=now):
                with mock.patch.object(self.mod, "_arm", return_value=1), \
                        mock.patch.object(self.mod, "_persist"):
                    self.assertIn("set for 4:22 PM,", self.actions["set_alarm"]("at 4:22"))
                    self.assertIn("set for 7:00 AM tomorrow,",
                                  self.actions["set_alarm"]("7:00 am"))
                    self.assertIn("set for 12:00 PM tomorrow,",
                                  self.actions["set_alarm"]("noon"))

    def test_the_hour_asked_on_a_daylight_saving_night(self):
        for y, mo, d in ((2027, 3, 13), (2027, 11, 6), (2026, 11, 12)):
            now = datetime(y, mo, d, 23, 0).timestamp()
            with self.subTest(day=(y, mo, d)), \
                    mock.patch.object(self.mod.time, "time", return_value=now):
                secs = self.mod._parse_when("7 am")
            self.assertEqual(datetime.fromtimestamp(now + secs).hour, 7)

    def test_cancel_by_clock_time_and_what_is_left(self):
        now = datetime(2026, 11, 12, 16, 21).timestamp()
        with mock.patch.object(self.mod.time, "time", return_value=now):
            self.actions["set_alarm"]("7 am")
            self.actions["set_alarm"]("9 am")
            self.actions["set_alarm"]("6:30 pm")
            # It used to read "7 am" as alarm #7.
            self.assertEqual(self.actions["cancel_alarm"]("the 8 am alarm"),
                             "There is no alarm at that time, sir.")
            self.assertEqual(self.actions["cancel_alarm"]("the 7 am alarm"),
                             "Cancelled the 7:00 AM tomorrow alarm, sir. "
                             "2 others are still set.")
            self.assertEqual(self.actions["cancel_alarm"]("6 30"),
                             "Cancelled the 6:30 PM alarm, sir. "
                             "The 9:00 AM tomorrow one is still set.")
            self.assertEqual(self.actions["cancel_alarm"]("#9"),
                             "There is no alarm #9, sir.")
            self.assertEqual(self.actions["cancel_alarm"]("2"),
                             "Cancelled the 9:00 AM tomorrow alarm, sir.")

    def test_cancel_never_guesses(self):
        now = datetime(2026, 11, 12, 16, 21).timestamp()
        with mock.patch.object(self.mod.time, "time", return_value=now):
            self.actions["set_alarm"]("6:30 am | gym")
            self.actions["set_alarm"]("7 am")
            self.actions["set_alarm"]("7 pm | all hands")
            cancel = self.actions["cancel_alarm"]
            # Odd words from the brain used to cancel the next one due, and
            # a label with "all" in it cancelled everything.
            for arg in ("the morning one", "banana", "the loud one"):
                with self.subTest(arg=arg):
                    self.assertEqual(cancel(arg),
                                     "Which alarm, sir? Say its time or its number.")
            self.assertEqual(len(self.mod._snapshot()), 3)
            self.assertIn("Cancelled the 7:00 PM alarm", cancel("the all hands one"))
            self.assertEqual(len(self.mod._snapshot()), 2)
            self.actions["set_alarm"]("7 pm")
            # "7:00" with a 7 AM and a 7 PM set: the next one due, not both.
            self.assertIn("Cancelled the 7:00 PM alarm", cancel("7:00"))
            self.assertIn("Cancelled the 6:30 AM tomorrow alarm", cancel("gym"))
            self.assertIn("Cancelled the 7:00 AM tomorrow alarm", cancel("seven am"))
            self.assertEqual(self.mod._snapshot(), [])

    def test_cancel_by_time_or_number_reaches_the_one_ringing(self):
        at = datetime(2026, 11, 13, 7, 0).timestamp()
        for arg in ("7 am", "7"):
            t = threading.Thread(
                target=lambda: self._quiet(self.mod.ring, 7, "", at), daemon=True)
            t.start()
            self.addCleanup(t.join, 5.0)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not self.mod.is_ringing():
                time.sleep(0.01)
            with self.subTest(arg=arg):
                self.assertEqual(self.actions["cancel_alarm"](arg),
                                 "Alarm stopped, sir.")
            t.join(5.0)
            self.assertFalse(self.mod.is_ringing())

    def test_the_list_names_the_one_ringing(self):
        t = self._ring_in_thread()
        self.assertEqual(self.actions["list_alarms"](""),
                         "Alarm #7 is ringing now, sir. Nothing else is set.")
        self.mod._stop_ringing("voice")
        t.join(5.0)

    def test_a_day_that_cannot_be_set_is_said_not_guessed(self):
        for arg in ("7 am on saturday", "monday at 7 am", "in 2 days at 7 am",
                    "the day after tomorrow at 7 am"):
            with self.subTest(arg=arg):
                self.assertIsNone(self.mod._parse_when(arg))
                self.assertIn("only set an alarm for today or tomorrow",
                              self.actions["set_alarm"](arg))
        self.assertFalse(self.mod.asked_when_recently())
        self.assertIn("What time should it ring", self.actions["set_alarm"]("loud"))
        self.assertTrue(self.mod.asked_when_recently())
        self.actions["set_alarm"]("20 minutes")
        self.assertFalse(self.mod.asked_when_recently())

    def test_cancel_both_is_all_and_cancel_while_ringing_is_the_one_ringing(self):
        self.actions["set_alarm"]("20 minutes")
        self.actions["set_alarm"]("40 minutes")
        self.assertEqual(self.actions["cancel_alarm"]("both of them"),
                         "Cancelled 2 alarms, sir.")
        self.actions["set_alarm"]("20 minutes")
        t = self._ring_in_thread()
        self.assertEqual(self.actions["cancel_alarm"](""), "Alarm stopped, sir.")
        t.join(5.0)
        self.assertEqual(len(self.mod._snapshot()), 1, "the pending one is kept")

    def test_cancel_alarm_drops_it_and_the_saved_copy(self):
        self.actions["set_alarm"]("20 minutes")
        self.assertIn("Cancelled", self.actions["cancel_alarm"](""))
        self.assertEqual(self.mod._snapshot(), [])
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])
        self.assertEqual(self.actions["cancel_alarm"](""), "No alarms are set, sir.")


class RingTests(_AlarmCase):
    def test_it_rings_loud_on_the_speakers_until_stopped_then_restores(self):
        t = self._ring_in_thread()
        # COM first (the volume and the speaker switch are COM calls and a
        # timer thread has none), then the speakers, then the volume.
        self.assertEqual(self.rig.events[:2], [("com",), ("speakers",)])
        self.assertIn(("volume", self.mod.ALARM_VOLUME, False), self.rig.events)
        self.assertIn(("start", "alarm_tone.wav"), self.rig.events)
        self.assertNotIn(("stop",), self.rig.events, "still ringing")
        self.assertEqual(self.actions["stop_alarm"](""), "Alarm off, sir.")
        t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.events[-2:], [("stop",), ("volume", 0.3, True)],
                         "the tone stops and his own volume and mute come back")

    def test_a_louder_volume_is_never_lowered(self):
        self.rig.volume = (0.95, False)
        p = mock.patch.object(self.mod, "ALARM_VOLUME", 0.5)
        p.start()
        self.addCleanup(p.stop)
        t = self._ring_in_thread()
        self.assertIn(("volume", 0.95, False), self.rig.events)
        self.mod._stop_ringing("voice")
        t.join(5.0)

    def test_a_key_or_the_mouse_stops_it_after_the_grace(self):
        with mock.patch.object(self.mod, "ALARM_INPUT_GRACE_S", 0.0):
            t = self._ring_in_thread()
            time.sleep(0.6)                 # the loop reads its baseline tick
            self.assertTrue(self.mod.is_ringing())
            self.rig.tick += 50
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertIn("Alarm off, sir.", self.rig.said)

    def test_it_gives_up_after_the_limit_and_says_so(self):
        with mock.patch.object(self.mod, "ALARM_MAX_RING_S", 0.3):
            t = self._ring_in_thread()
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.events[-2:], [("stop",), ("volume", 0.3, True)])
        self.assertTrue(any("no answer" in s for s in self.rig.said), self.rig.said)

    def test_he_talks_over_the_tone_and_keeps_at_it(self):
        with mock.patch.object(self.mod, "ALARM_SAY_FIRST_S", 0.0),                 mock.patch.object(self.mod, "ALARM_SAY_EVERY_S", 0.3):
            t = self._ring_in_thread("class")
            deadline = time.monotonic() + 5.0
            while len(self.rig.said) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.mod._stop_ringing("voice")
            t.join(5.0)
        self.assertGreaterEqual(len(self.rig.said), 3, self.rig.said)
        self.assertIn("Time to wake up.", self.rig.said[0])
        self.assertIn("You set this for class.", self.rig.said[0])
        self.assertEqual(self.rig.said[1], "Sir, you need to wake up.")

    def test_no_wake_line_can_answer_its_own_alarm(self):
        # The mic hears the speakers: a line that read as "stop" or "snooze"
        # would end the alarm it belongs to.
        t = self._ring_in_thread()
        try:
            for n in range(3 * len(self.mod.WAKE_LINES)):
                line = self.mod.wake_line(n, 60.0 * n, "work")
                with self.subTest(line=line):
                    self.assertEqual(self.mod.ringing_command(line), "")
                    self.assertNotIn("{", line)
        finally:
            self.mod._stop_ringing("voice")
            t.join(5.0)

    def test_the_lines_cycle_without_repeating_the_clock_line(self):
        first = self.mod.wake_line(0, 2.0, now=datetime(2026, 11, 13, 7, 0).timestamp())
        self.assertEqual(first, "Sir, it is 7:00 AM. Time to wake up.")
        n = len(self.mod.WAKE_LINES)
        self.assertEqual(self.mod.wake_line(n, 200.0), self.mod.wake_line(1, 20.0))
        self.assertIn("4 minutes", self.mod.wake_line(5, 240.0))

    def test_snooze_stops_it_and_arms_another(self):
        t = self._ring_in_thread("work")
        out = self.actions["snooze_alarm"]("")
        t.join(5.0)
        self.assertIn(f"{self.mod.ALARM_SNOOZE_MIN} minutes", out)
        rows = self.mod._snapshot()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "work")
        self.assertAlmostEqual(rows[0]["at"],
                               time.time() + self.mod.ALARM_SNOOZE_MIN * 60, delta=5)

    def test_a_tone_that_will_not_play_rings_with_beeps(self):
        # It used to be one spoken sentence at normal volume, then nothing.
        def boom(path):
            raise OSError("no audio device")
        with mock.patch.object(self.mod, "_sound_start", boom):
            t = threading.Thread(target=lambda: self._quiet(self.mod.ring, 3, ""),
                                 daemon=True)
            t.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and \
                    self.rig.events.count(("beep",)) < 2:
                time.sleep(0.02)
            self.assertTrue(self.mod.is_ringing())
            self.assertGreaterEqual(self.rig.events.count(("beep",)), 2)
            self.assertEqual(self.rig.volume, (self.mod.ALARM_VOLUME, False),
                             "still loud while it beeps")
            self.mod._stop_ringing("voice")
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.volume, (0.30, True))
        self.assertFalse(any("could not ring" in s for s in self.rig.said))

    def test_no_tone_and_no_beep_keeps_talking_and_tries_the_tone_again(self):
        # It ended after one spoken line, with the alarm gone for good.
        calls = []

        def tone(path):
            calls.append(path)
            if len(calls) < 3:
                raise OSError("no audio device")
            self.rig.events.append(("start", os.path.basename(path)))

        def boom(*a):
            raise OSError("no audio device")
        with mock.patch.object(self.mod, "_sound_start", tone), \
                mock.patch.object(self.mod, "_beep", boom), \
                mock.patch.object(self.mod, "ALARM_TONE_AGAIN_S", 0.3), \
                mock.patch.object(self.mod, "ALARM_SAY_FIRST_S", 0.0):
            t = threading.Thread(target=lambda: self._quiet(self.mod.ring, 3, ""),
                                 daemon=True)
            t.start()
            self.addCleanup(t.join, 5.0)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not any(
                    e[0] == "start" for e in self.rig.events):
                time.sleep(0.02)
            self.assertTrue(self.mod.is_ringing())
            self.assertIn(("start", "alarm_tone.wav"), self.rig.events)
            self.assertTrue(self.rig.said, "he is still being spoken to")
            self.mod._stop_ringing("voice")
            t.join(5.0)
        self.assertEqual(self.rig.volume, (0.30, True))
        self.assertFalse(any("could not ring" in s for s in self.rig.said))

    def test_a_ring_that_breaks_is_tried_again_not_dropped(self):
        def boom():
            raise RuntimeError("input read broke")
        with mock.patch.object(self.mod, "_input_tick", boom):
            self._quiet(self.mod.ring, 3, "work", 123.0)
            with self.mod._lock:
                rows = [(a["label"], a["at"], a["tries"])
                        for a in self.mod._alarms.values()]
            self.assertEqual(rows, [("work", 123.0, 1)])
            self.assertEqual(self.rig.said, [])
            self.assertEqual(self.rig.volume, (0.30, True))
            self._cancel_all()
            self._quiet(self.mod.ring, 4, "work", 123.0, self.mod.ALARM_MAX_RETRIES)
        self.assertEqual(self.mod._snapshot(), [])
        self.assertTrue(any("could not ring" in s for s in self.rig.said))

    def test_a_volume_write_that_fails_is_tried_again(self):
        writes = []

        def write(level, muted):
            writes.append(level)
            if len(writes) < 3:
                return False
            return self.rig._write(level, muted)
        with mock.patch.object(self.mod, "_volume_write", write), \
                mock.patch.object(self.mod, "ALARM_VOLUME_RETRY_S", 0.2):
            t = self._ring_in_thread()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and \
                    self.rig.volume != (self.mod.ALARM_VOLUME, False):
                time.sleep(0.02)
            self.assertEqual(self.rig.volume, (self.mod.ALARM_VOLUME, False))
            self.mod._stop_ringing("voice")
            t.join(5.0)

    def test_a_hung_speaker_switch_does_not_keep_it_silent(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        with mock.patch.object(self.mod, "_to_speakers", lambda: gate.wait(30)), \
                mock.patch.object(self.mod, "ALARM_PREP_MAX_S", 0.2):
            t = self._ring_in_thread()
            self.assertIn(("start", "alarm_tone.wav"), self.rig.events)
            self.mod._stop_ringing("voice")
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())

    def test_a_ring_that_never_ended_does_not_block_the_next_alarm(self):
        dead = {"id": 1, "label": "", "at": None, "stop": threading.Event(),
                "why": "", "since": time.monotonic() - self.mod.ALARM_MAX_RING_S - 300}
        self.mod._ringing[0] = dead
        t = self._ring_in_thread()
        self.assertIn(("start", "alarm_tone.wav"), self.rig.events)
        self.assertTrue(dead["stop"].is_set())
        self.mod._stop_ringing("voice")
        t.join(5.0)
        self.assertFalse(self.mod.is_ringing())

    def test_the_volume_is_raised_even_when_it_cannot_be_read(self):
        with mock.patch.object(self.mod, "_volume_read", lambda: None):
            t = self._ring_in_thread()
            self.assertIn(("volume", self.mod.ALARM_VOLUME, False), self.rig.events)
            n = len(self.rig.events)
            self.mod._stop_ringing("voice")
            t.join(5.0)
        self.assertNotIn("volume", [e[0] for e in self.rig.events[n:]],
                         "an unknown level is not 'restored'")

    def test_no_answer_rings_again_later_then_gives_up_for_good(self):
        with mock.patch.object(self.mod, "ALARM_MAX_RING_S", 0.3):
            self._quiet(self.mod.ring, 7, "work")
            rows = self.mod._snapshot()
            self.assertEqual([r["label"] for r in rows], ["work"])
            self.assertAlmostEqual(rows[0]["at"],
                                   time.time() + self.mod.ALARM_RETRY_GAP_S, delta=5)
            with self.mod._lock:
                self.assertEqual([a["tries"] for a in self.mod._alarms.values()], [1])
            self.assertIn("ring again", self.rig.said[-1])
            self._cancel_all()
            self._quiet(self.mod.ring, 8, "work", None, self.mod.ALARM_MAX_RETRIES)
        self.assertEqual(self.mod._snapshot(), [])
        self.assertIn("I have stopped it", self.rig.said[-1])

    def test_one_that_comes_due_during_a_ring_is_not_lost(self):
        now = time.time()
        t = threading.Thread(
            target=lambda: self._quiet(self.mod.ring, 1, "first", now), daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.mod.is_ringing():
            time.sleep(0.01)
        # A backup five minutes later: kept, tried again shortly.
        self._quiet(self.mod.ring, 2, "backup", now + 300)
        rows = self.mod._snapshot()
        self.assertEqual([(r["label"], r["at"]) for r in rows], [("backup", now + 300)])
        # The same row fired twice (same label, same moment): one alarm.
        self._quiet(self.mod.ring, 3, "first", now)
        self.assertEqual(len(self.mod._snapshot()), 1)
        # A backup one minute later with the same (empty) label is NOT a
        # duplicate: it was dropped inside a 90 s window.
        self._quiet(self.mod.ring, 4, "first", now + 61)
        self.assertEqual(len(self.mod._snapshot()), 2)
        self.assertEqual(self.rig.events.count(("start", "alarm_tone.wav")), 1)
        self.mod._stop_ringing("voice")
        t.join(5.0)

    def test_the_one_ringing_stays_saved_until_it_is_answered(self):
        # A crash or a restart mid-ring used to lose it: the file was already
        # empty.
        path = os.path.join(_DATA_TMPDIR, "alarms.json")
        at = time.time()
        t = threading.Thread(
            target=lambda: self._quiet(self.mod.ring, 5, "work", at), daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(
                e[0] == "start" for e in self.rig.events):
            time.sleep(0.01)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"id": 5, "at": at, "label": "work"}])
        self.assertEqual(self.mod._snapshot(), [], "not listed as pending")
        self.mod._stop_ringing("voice")
        t.join(5.0)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])

    def test_a_ring_that_raises_frees_the_slot_and_tries_again(self):
        self._quiet(self.actions["set_alarm"], "20 minutes | work")
        aid = self.mod._snapshot()[0]["id"]

        def boom(*a, **k):
            with self.mod._lock:
                self.mod._ringing[0] = {"id": aid, "label": "work", "at": None,
                                        "stop": threading.Event(), "why": ""}
            raise RuntimeError("boom")
        with mock.patch.object(self.mod, "ring", boom):
            self._quiet(self.mod._fire, aid)
        self.assertFalse(self.mod.is_ringing())
        rows = self.mod._snapshot()
        self.assertEqual([r["label"] for r in rows], ["work"])

    def test_a_label_that_reads_as_an_answer_is_not_said_aloud(self):
        self.assertNotIn("stop smoking", self.mod.wake_line(0, 2.0, "stop smoking"))
        self.assertIn("for the gym", self.mod.wake_line(0, 2.0, "the gym"))


class HeardWhileRingingTests(_AlarmCase):
    def test_nothing_rings_nothing_is_a_command(self):
        for text in ("stop", "snooze", "I'm up"):
            self.assertEqual(self.mod.ringing_command(text), "")
        self.assertIsNone(self.mod.handle_ringing("stop"))

    def test_stop_and_snooze_phrases_while_ringing(self):
        t = self._ring_in_thread()
        cmd = self.mod.ringing_command
        for text in ("Stop.", "okay okay", "I'm up, I'm up", "turn it off",
                     "Jarvis, shut up", "I am awake", "that's enough"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "stop")
        for text in ("snooze", "Jarvis snooze it", "five more minutes"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "snooze")
        for text in ("what time is it", "is this for medical or dental",
                     # Measured 2026-11-12: talk in the room and a video ended
                     # a ring. A lone word counts only in a short line.
                     "the tv is off", "Thanks for watching, and don't stop",
                     "I told him that was enough of that for one day",
                     "set an alarm to go off at 7 am",
                     "we should cancel the meeting on friday"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "")
        for text in ("Jarvis, that is enough of that noise for today",
                     "okay I heard you now please stop the alarm already",
                     "alarm off", "cancel", "turn that off",
                     "for the love of god I'm up",
                     # As speech-to-text writes them: punctuated, repeated.
                     "Okay, okay.", "OK, OK.", "ok ok",
                     "Stop, stop, stop, stop, stop, stop!",
                     "Stop! Stop it! Stop it right now, stop!",
                     "would you please just stop already",
                     "I'm getting up.", "Im up"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "stop")
        for text in ("5 more minutes", "give me 5 more minutes"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "snooze")
        self.assertEqual(self.mod.handle_ringing("I'm up"), "Alarm off, sir.")
        t.join(5.0)
        self.assertFalse(self.mod.is_ringing())


class RestoreTests(_AlarmCase):
    def _save(self, rows):
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)

    def test_a_pending_alarm_is_re_armed_after_a_restart(self):
        at = time.time() + 3600
        self._save([{"id": 4, "at": at, "label": "class"}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 1)
        rows = self.mod._snapshot()
        self.assertEqual((rows[0]["label"], rows[0]["at"]), ("class", at))

    def test_he_is_told_a_saved_alarm_is_still_set(self):
        at = time.time() + 3600
        self._save([{"id": 4, "at": at, "label": ""}])
        self._quiet(self.mod.restore_persisted_alarms)
        self.assertEqual(len(self.rig.said), 1)
        self.assertIn("alarm is still set, sir.", self.rig.said[0])

    def test_one_already_due_waits_for_the_speakers_to_exist(self):
        # The skills are still loading when this runs: live 2026-11-12 the
        # speaker switch registered 8 s after this skill.
        at = time.time() - 600
        self._save([{"id": 4, "at": at, "label": "work"}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 1)
        with self.mod._lock:
            (row,) = self.mod._alarms.values()
        self.assertEqual(row["at"], at)
        self.assertAlmostEqual(row["timer"].interval,
                               self.mod.ALARM_RESTORE_LATE_S, delta=0.01)
        self.assertEqual(self.rig.said, [], "nothing said: it is about to ring")

    def test_a_damaged_row_never_rings(self):
        self._save([{"id": 1, "at": float("nan"), "label": "n"},
                    {"id": 2, "at": -5, "label": ""},
                    {"id": 3, "at": time.time() + 9e9, "label": ""}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 0)
        self.assertEqual(self.mod._snapshot(), [])

    def test_one_missed_by_hours_is_reported_not_rung(self):
        self._save([{"id": 4, "at": time.time() - 5 * 3600, "label": ""}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 0)
        self.assertEqual(self.mod._snapshot(), [])
        self.assertTrue(any("never rang" in s for s in self.rig.said))

    def test_the_tone_file_is_a_loopable_wav(self):
        import wave
        path = self.mod._tone_path()
        with wave.open(path, "rb") as w:
            secs = w.getnframes() / w.getframerate()
            self.assertEqual((w.getnchannels(), w.getsampwidth()), (1, 2))
        self.assertTrue(1.5 < secs < 3.5, secs)


if __name__ == "__main__":
    unittest.main()
