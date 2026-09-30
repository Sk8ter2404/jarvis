"""core/date_math.py: deterministic relative-date answers (2026-09-29).

Live evidence this pins (v2.0.115, typed turns, Tue 2026-09-29 US Central):
  * "what's the date tomorrow" -> the LLM read back TODAY's date;
  * "how long until Friday" -> "Roughly 1 day and 14 hours" (it is 3 days);
  * "how many days until Christmas" -> "86 days" (it is 87).
And v2.0.140 (22:15, same Tuesday): "what's the date next Monday" -> "It will
be September 29th, sir." (today; it is Monday, October 5).

Every test passes an explicit, frozen ``now``: the module never reads the
clock, so these are timezone-independent (CI runs in UTC). Stdlib unittest,
CI-safe.

    python -m unittest tests.test_date_math
"""
from __future__ import annotations

import datetime as dt
import unittest

from core import date_math as dm

# The live-evidence moment: Tuesday 2026-09-29, 14:56 local.
TUE = dt.datetime(2026, 9, 29, 14, 56)


def _at(y, m, d, hh=9, mm=0):
    return dt.datetime(y, m, d, hh, mm)


# (now, utterance, kind, exact spoken reply)
_TABLE = (
    # ── the three live bugs ─────────────────────────────────────────────
    (TUE, "what's the date tomorrow", "date",
     "Tomorrow is Wednesday, September 30, 2026, sir."),
    (TUE, "how long until Friday", "days-until",
     "Friday is 3 days away, on October 2, sir."),
    (TUE, "how many days until Christmas", "days-until",
     "Christmas is 87 days away, on Friday, December 25, sir."),
    # ── today / tomorrow / yesterday ────────────────────────────────────
    (TUE, "what day is it", "date",
     "Today is Tuesday, September 29, 2026, sir."),
    (TUE, "What's today's date?", "date",
     "Today is Tuesday, September 29, 2026, sir."),
    (TUE, "what was yesterday's date", "date",
     "Yesterday was Monday, September 28, 2026, sir."),
    (TUE, "what day of the week is tomorrow", "date",
     "Tomorrow is Wednesday, September 30, 2026, sir."),
    (TUE, "what's the day after tomorrow", "date",
     "The day after tomorrow is Thursday, October 1, 2026, sir."),
    (TUE, "what day was the day before yesterday", "date",
     "The day before yesterday was Sunday, September 27, 2026, sir."),
    (TUE, "do you know what day it is", "date",
     "Today is Tuesday, September 29, 2026, sir."),
    # ── month and year rollover ─────────────────────────────────────────
    (_at(2026, 9, 30, 23, 59), "what's the date tomorrow", "date",
     "Tomorrow is Thursday, October 1, 2026, sir."),
    (_at(2026, 12, 31), "what's tomorrow's date", "date",
     "Tomorrow is Friday, January 1, 2027, sir."),
    (_at(2027, 1, 1, 0, 5), "what was the date yesterday", "date",
     "Yesterday was Thursday, December 31, 2026, sir."),
    (_at(2026, 12, 31), "how many days until new year's", "days-until",
     "New Year's Day is tomorrow, sir."),
    (_at(2027, 1, 1, 0, 5), "how many days until new year's day",
     "days-until", "New Year's Day is today, sir."),
    (_at(2026, 12, 30), "what's the date in 3 days", "date-offset",
     "In 3 days it will be Saturday, January 2, 2027, sir."),
    (_at(2026, 12, 29), "how long until friday", "days-until",
     "Friday is 3 days away, on January 1, 2027, sir."),
    # ── the leap day 2028-02-29 ─────────────────────────────────────────
    (_at(2028, 2, 28), "what's the date tomorrow", "date",
     "Tomorrow is Tuesday, February 29, 2028, sir."),
    (_at(2028, 2, 29), "what's the date tomorrow", "date",
     "Tomorrow is Wednesday, March 1, 2028, sir."),
    (_at(2028, 2, 29), "what day is it", "date",
     "Today is Tuesday, February 29, 2028, sir."),
    (TUE, "what day of the week is february 29 2028", "date-of",
     "February 29, 2028 is a Tuesday, sir."),
    # A yearless Feb 29 rolls to the next year that has one.
    (TUE, "how many days until february 29", "days-until",
     "February 29, 2028 is 518 days away, on a Tuesday, sir."),
    # ── Christmas around the day itself ─────────────────────────────────
    (_at(2026, 12, 24), "how many days until christmas", "days-until",
     "Christmas is tomorrow, sir."),
    (_at(2026, 12, 25), "how many days until christmas", "days-until",
     "Christmas is today, sir."),
    # Asked on Dec 26: next year's Christmas.
    (_at(2026, 12, 26), "how many days until christmas", "days-until",
     "Christmas is 364 days away, on Saturday, December 25, 2027, sir."),
    (_at(2026, 12, 26), "what day is christmas", "date-of",
     "Christmas is on Saturday, December 25, 2027, 364 days from now, sir."),
    (_at(2026, 12, 26), "what day was christmas this year", "date-of",
     "Christmas was on Friday, December 25, 2026, sir."),
    (_at(2026, 12, 25), "how long until next christmas", "days-until",
     "Christmas is 365 days away, on Saturday, December 25, 2027, sir."),
    # ── weekdays, including the same weekday ────────────────────────────
    (TUE, "how long until tuesday", "days-until",
     "Today is Tuesday, so next Tuesday is 7 days away, on October 6, sir."),
    (TUE, "what date is tuesday", "date-of",
     "Today is Tuesday; next Tuesday is October 6, 2026, sir."),
    (_at(2026, 10, 2), "how long until friday", "days-until",
     "Today is Friday, so next Friday is 7 days away, on October 9, sir."),
    (TUE, "how long until wednesday", "days-until",
     "Wednesday is tomorrow, September 30, sir."),
    (TUE, "how many days until monday", "days-until",
     "Monday is 6 days away, on October 5, sir."),
    (TUE, "what's the date on friday", "date-of",
     "Friday is October 2, 2026, sir."),
    (TUE, "how long till it's friday", "days-until",
     "Friday is 3 days away, on October 2, sir."),
    # ── "next <weekday>" (2026-09-29, v2.0.140 live: "what's the date next
    #    Monday" went to the LLM, which said "September 29th" = TODAY) ──────
    # The coming Monday (Oct 5) is already in NEXT Mon-Sun week, so "the
    # coming one" and "next week's" agree: answer it, in every frame asked.
    (TUE, "what's the date next Monday", "date-of",
     "Next Monday is October 5, 2026, sir."),
    (TUE, "what date is next Monday", "date-of",
     "Next Monday is October 5, 2026, sir."),
    (TUE, "what date is it next Monday", "date-of",
     "Next Monday is October 5, 2026, sir."),
    (TUE, "when is next Monday", "date-of",
     "Next Monday is October 5, 2026, sir."),
    (TUE, "what's next Monday's date", "date-of",
     "Next Monday is October 5, 2026, sir."),
    (TUE, "how many days until next monday", "days-until",
     "Next Monday is 6 days away, on October 5, sir."),
    # Asked ON a Monday: the coming Monday is 7 days out, next week.
    (_at(2026, 9, 28), "what's the date next monday", "date-of",
     "Today is Monday; next Monday is October 5, 2026, sir."),
    (_at(2026, 9, 28), "how long until next monday", "days-until",
     "Today is Monday, so next Monday is 7 days away, on October 5, sir."),
    # Asked on a Sunday. These two rows used to pin ONE date ("Next Monday
    # is tomorrow"), measuring weeks Monday-to-Sunday only. On a US calendar
    # the week STARTS on Sunday, so Mon..Sat are still "this week" and "next
    # Monday" usually means the one 8 days out: ambiguous under either
    # convention, so both dates (review DATE-2). Only "next Sunday" agrees.
    (_at(2026, 10, 4), "what date is next friday", "date-of",
     "This Friday is October 9, and the Friday after is October 16, sir."),
    (_at(2026, 10, 4), "how long until next monday", "days-until",
     "This Monday is tomorrow, October 5, and the Monday after is 8 days "
     "away, on October 12, sir."),
    (_at(2026, 10, 4), "what's the date next monday", "date-of",
     "This Monday is tomorrow, October 5, and the Monday after is October "
     "12, sir."),
    (_at(2026, 10, 4), "when is next saturday", "date-of",
     "This Saturday is October 10, and the Saturday after is October 17, "
     "sir."),
    (_at(2026, 10, 4), "what date is next sunday", "date-of",
     "Today is Sunday; next Sunday is October 11, 2026, sir."),
    # The coming Friday (Oct 2) is still THIS week: "next Friday" may mean it
    # or Oct 9. These two used to be pinned to None (never guess); the reply
    # now gives both readings, which is still never a guess.
    (TUE, "what date is next friday", "date-of",
     "This Friday is October 2, and the Friday after is October 9, sir."),
    (TUE, "what's next Friday's date", "date-of",
     "This Friday is October 2, and the Friday after is October 9, sir."),
    (TUE, "how long until next friday", "days-until",
     "This Friday is 3 days away, on October 2, and the Friday after is "
     "10 days away, on October 9, sir."),
    (TUE, "what date is next wednesday", "date-of",
     "This Wednesday is tomorrow, September 30, and the Wednesday after is "
     "October 7, sir."),
    (_at(2026, 12, 29), "what date is next friday", "date-of",
     "This Friday is January 1, 2027, and the Friday after is January 8, "
     "2027, sir."),
    (TUE, "what's friday's date", "date-of",
     "Friday is October 2, 2026, sir."),
    # ── named holidays ──────────────────────────────────────────────────
    (TUE, "when is thanksgiving", "date-of",
     "Thanksgiving is on Thursday, November 26, 2026, 58 days from now, sir."),
    (_at(2026, 11, 26), "when is thanksgiving", "date-of",
     "Thanksgiving is today, Thursday, November 26, 2026, sir."),
    # 4th Thursday of November 2027 = Nov 25.
    (_at(2026, 11, 27), "how many days until thanksgiving", "days-until",
     "Thanksgiving is 363 days away, on Thursday, November 25, 2027, sir."),
    (TUE, "how far away is thanksgiving", "days-until",
     "Thanksgiving is 58 days away, on Thursday, November 26, sir."),
    (TUE, "how long until halloween", "days-until",
     "Halloween is 32 days away, on Saturday, October 31, sir."),
    (TUE, "how many days until christmas eve", "days-until",
     "Christmas Eve is 86 days away, on Thursday, December 24, sir."),
    (TUE, "how many days until new year's eve", "days-until",
     "New Year's Eve is 93 days away, on Thursday, December 31, sir."),
    (TUE, "how many days until independence day", "days-until",
     "Independence Day is 278 days away, on Sunday, July 4, 2027, sir."),
    (TUE, "how many days until the fourth of july", "days-until",
     "Independence Day is 278 days away, on Sunday, July 4, 2027, sir."),
    (TUE, "how many days until the 4th of july", "days-until",
     "Independence Day is 278 days away, on Sunday, July 4, 2027, sir."),
    (TUE, "how many days until valentine's day", "days-until",
     "Valentine's Day is 138 days away, on Sunday, February 14, 2027, sir."),
    (TUE, "what day does christmas fall on next year", "date-of",
     "Christmas is on Saturday, December 25, 2027, 452 days from now, sir."),
    (TUE, "days until halloween", "days-until",
     "Halloween is 32 days away, on Saturday, October 31, sir."),
    # ── "in N days / weeks" and "ago" ───────────────────────────────────
    (TUE, "what's the date in 3 days", "date-offset",
     "In 3 days it will be Friday, October 2, 2026, sir."),
    (TUE, "what day will it be in two weeks", "date-offset",
     "In 2 weeks it will be Tuesday, October 13, 2026, sir."),
    (TUE, "what's the date a week from today", "date-offset",
     "In a week it will be Tuesday, October 6, 2026, sir."),
    (TUE, "what day was it 5 days ago", "date-offset",
     "5 days ago it was Thursday, September 24, 2026, sir."),
    # ── calendar dates ──────────────────────────────────────────────────
    (TUE, "what day of the week is december 25", "date-of",
     "December 25, 2026 is a Friday, sir."),
    (TUE, "what day of the week was july 4 1776", "date-of",
     "July 4, 1776 was a Thursday, sir."),
    (TUE, "how many days until 12/25", "days-until",
     "December 25 is 87 days away, on a Friday, sir."),
    (TUE, "how many days until the 25th of december", "days-until",
     "December 25 is 87 days away, on a Friday, sir."),
    (TUE, "how many days until december 25 2025", "days-until",
     "December 25, 2025 was 278 days ago, on a Thursday, sir."),
    # ── wake word / politeness are ignored ──────────────────────────────
    (TUE, "Jarvis, can you tell me how many days until Christmas, please?",
     "days-until", "Christmas is 87 days away, on Friday, December 25, sir."),
)

# Commands and other domains that must fall through to the normal turn.
_NEGATIVE = (
    "remind me tomorrow to call the office",
    "remind me in 3 days to renew the permit",
    "what's the weather tomorrow",
    "what's the forecast for friday",
    "what's on my calendar tomorrow",
    "what are my plans for tomorrow",
    "do i have anything on friday",
    "set a timer for 3 days",
    "set a timer for ten minutes",
    "schedule a meeting for friday",
    "schedule a call tomorrow",
    "wake me up tomorrow",
    "wake me up tomorrow at 7",
    "play music until friday",
    "pause until tomorrow",
    "snooze until tomorrow",
    "mute notifications until monday",
    "turn off the lights until friday",
    "cancel my alarm for tomorrow",
    "add milk to the list for tomorrow",
    "set a countdown until christmas",
    "what time is it",
    "what's tomorrow",
    "what's the date and time",
    "what's the date tomorrow and what's the weather",
    "how many days until christmas and set a timer",
    "how long until my timer ends",
    "how many days until my birthday",
    "how long until christmas break",
    "how long until 5 pm",
    "how long until tomorrow",
    "how long until the weekend",
    "how many weeks until christmas",
    # "how long until next friday" / "what date is next friday" used to be
    # here (ambiguous: this week's or next?). They now answer BOTH readings
    # (see the table), which is not a guess. Still None: a "next" that is
    # not a weekday or holiday, and "next <weekday>" pinned to a year.
    "when is next week",
    "what's the date next week",
    "how long until next december 25",
    "what's the date next monday this year",
    "what's next week's date",
    "how many days until february 29 this year",   # no such day in 2026
    "how many days until february 30",
    "how many days until 13/45",
    "what day is my dentist appointment",
    "what day is the party",
    "what day is it in tokyo",
    "what day is it in 3 months",
    "what's the date of the meeting tomorrow",
    "what is the date tomorrow in london",
    "when is my flight",
    "when is the next bus",
    "how long is the movie",
    "how long is the drive to chicago",
    "how far is chicago",
    "what did we do yesterday",
    "what did I just ask you",
    "what's my name",
    "tell me a joke",
    "",
    "   ",
)


class DateMathTableTests(unittest.TestCase):
    def test_table(self):
        for now, text, kind, reply in _TABLE:
            with self.subTest(now=now.isoformat(), text=text):
                got = dm.answer(text, now)
                self.assertIsNotNone(got)
                self.assertEqual(got.kind, kind)
                self.assertEqual(got.reply, reply)

    def test_every_reply_ends_with_sir(self):
        for now, text, _kind, _reply in _TABLE:
            with self.subTest(text=text):
                self.assertTrue(dm.answer(text, now).reply.endswith(", sir."))

    def test_time_of_day_never_matters(self):
        # Calendar days: 00:01 and 23:59 on the same day give the same answer.
        for text in ("how long until Friday", "how many days until Christmas",
                     "what's the date tomorrow"):
            with self.subTest(text=text):
                early = dm.answer(text, _at(2026, 9, 29, 0, 1))
                late = dm.answer(text, _at(2026, 9, 29, 23, 59))
                self.assertEqual(early, late)
                self.assertEqual(early, dm.answer(text, TUE))

    def test_accepts_a_date_as_now(self):
        self.assertEqual(dm.answer("what day is it", dt.date(2026, 9, 29)),
                         dm.answer("what day is it", TUE))

    def test_christmas_from_every_day_of_december(self):
        # Counts down to 0 on the 25th, then rolls to next year's.
        for day in range(1, 32):
            now = _at(2026, 12, day)
            got = dm.answer("how many days until christmas", now)
            with self.subTest(day=day):
                target = dt.date(2026 if day <= 25 else 2027, 12, 25)
                n = (target - now.date()).days
                if n == 0:
                    self.assertEqual(got.reply, "Christmas is today, sir.")
                elif n == 1:
                    self.assertEqual(got.reply, "Christmas is tomorrow, sir.")
                else:
                    self.assertIn(f" {n} days away", got.reply)

    def test_weekday_answer_is_1_to_7_calendar_days_for_every_pair(self):
        monday = dt.date(2026, 9, 28)
        for offset in range(7):
            today = monday + dt.timedelta(days=offset)
            for wd, name in enumerate(dm.WEEKDAYS):
                expected = (wd - today.weekday()) % 7 or 7
                got = dm.answer(f"how long until {name}", today)
                with self.subTest(today=today.isoformat(), target=name):
                    if expected == 1:
                        self.assertIn(" is tomorrow, ", got.reply)
                    else:
                        self.assertIn(f" {expected} days away", got.reply)
                    if expected == 7:
                        self.assertTrue(got.reply.startswith(
                            f"Today is {name.title()}, so next "))


    def test_next_weekday_for_every_pair(self):
        # "next X" is answered as ONE date exactly when the coming X falls in
        # next week under BOTH week conventions (both readings agree);
        # otherwise both are given. The week ends on the ISO Sunday — except
        # when asked ON a Sunday, which starts a US Sunday-to-Saturday week
        # that runs to the coming Saturday. (This used to use the ISO Sunday
        # for every day, which pinned a single guess on Sundays: DATE-2.)
        monday = dt.date(2026, 9, 28)
        for offset in range(7):
            today = monday + dt.timedelta(days=offset)
            week_end = monday + dt.timedelta(days=6)
            if today.weekday() == 6:
                week_end = today + dt.timedelta(days=6)
            for wd, name in enumerate(dm.WEEKDAYS):
                coming = today + dt.timedelta(
                    days=(wd - today.weekday()) % 7 or 7)
                got = dm.answer(f"what date is next {name}", today)
                with self.subTest(today=today.isoformat(), target=name):
                    self.assertIsNotNone(got)
                    md = f"{coming:%B} {coming.day}"
                    if coming > week_end:
                        self.assertNotIn(" after is ", got.reply)
                        self.assertIn(f"{md}, {coming.year}", got.reply)
                    else:
                        later = coming + dt.timedelta(days=7)
                        self.assertTrue(got.reply.startswith(
                            f"This {name.title()} is "), got.reply)
                        self.assertIn(md, got.reply)
                        self.assertIn(f"after is {later:%B} {later.day}",
                                      got.reply)


class PastTenseTests(unittest.TestCase):
    """Review DATE-1: the "what's Monday's date" frame accepted "was", so a
    past-tense question got the COMING weekday, a week off from the one
    asked about ("What was Monday's date?" asked Tue Sep 29 -> "Monday is
    October 5"). The older "what date was it on Monday" frames had the same
    flaw. A past-tense question is never answered with a date this module
    rolled forward to: None, so the LLM answers."""

    def test_past_tense_weekday_is_never_the_coming_one(self):
        for text, now in (
                ("What was Monday's date?", _at(2026, 9, 29, 22, 15)),
                ("What was Friday's date?", _at(2026, 10, 3, 10)),
                ("what was tuesday's date", _at(2026, 9, 30, 9)),
                ("what was next monday's date", TUE),
                ("what was the date on monday", TUE),
                ("what date was it on monday", TUE),
                ("what date was monday", TUE),
                ("what day was friday", TUE),
                ("when was friday", TUE)):
            with self.subTest(text=text):
                self.assertIsNone(dm.answer(text, now))

    def test_past_tense_holiday_or_date_not_yet_this_year(self):
        # "what day was Christmas" in September means last Christmas, not the
        # one 87 days away.
        for text in ("what day was christmas", "when was thanksgiving",
                     "what day was december 25", "what day did christmas "
                     "fall on"):
            with self.subTest(text=text):
                self.assertIsNone(dm.answer(text, TUE))

    def test_past_tense_that_is_well_defined_still_answers(self):
        for text, now, reply in (
                ("what was yesterday's date", TUE,
                 "Yesterday was Monday, September 28, 2026, sir."),
                ("what day was christmas this year", _at(2026, 12, 26),
                 "Christmas was on Friday, December 25, 2026, sir."),
                ("what day was christmas", _at(2026, 12, 25),
                 "Christmas is today, Friday, December 25, 2026, sir."),
                ("what day of the week was july 4 1776", TUE,
                 "July 4, 1776 was a Thursday, sir."),
                ("what day was it 5 days ago", TUE,
                 "5 days ago it was Thursday, September 24, 2026, sir."),
                # Present tense is unchanged.
                ("what's monday's date", TUE,
                 "Monday is October 5, 2026, sir."),
                ("what date is monday", TUE,
                 "Monday is October 5, 2026, sir.")):
            with self.subTest(text=text):
                got = dm.answer(text, now)
                self.assertIsNotNone(got)
                self.assertEqual(got.reply, reply)


class DateMathNegativeTests(unittest.TestCase):
    def test_commands_and_other_domains_return_none(self):
        for text in _NEGATIVE:
            with self.subTest(text=text):
                self.assertIsNone(dm.answer(text, TUE))

    def test_bad_inputs_return_none(self):
        self.assertIsNone(dm.answer(None, TUE))
        self.assertIsNone(dm.answer(42, TUE))
        self.assertIsNone(dm.answer("what day is it", None))
        self.assertIsNone(dm.answer("what day is it", "2026-09-29"))
        self.assertIsNone(dm.answer("what's the date " + "x " * 200, TUE))

    def test_huge_offset_returns_none_instead_of_raising(self):
        self.assertIsNone(dm.answer("what's the date in 99999 days", TUE))
        self.assertIsNone(dm.answer("what day was it 99999 weeks ago", TUE))

    def test_never_raises_on_the_extreme_calendar_edges(self):
        for now in (dt.datetime(1, 1, 1), dt.datetime(9999, 12, 31)):
            for text in ("what's the date tomorrow", "what was yesterday's date",
                         "how many days until christmas",
                         "what's the date in 3 days"):
                with self.subTest(now=now.isoformat(), text=text):
                    got = dm.answer(text, now)   # None or an answer, no raise
                    self.assertTrue(got is None or got.reply.endswith("sir."))


class NormalizeTests(unittest.TestCase):
    def test_wake_word_politeness_and_contractions(self):
        self.assertEqual(
            dm.normalize("Hey JARVIS, what's the date tomorrow, please?"),
            "what is the date tomorrow")
        self.assertEqual(dm.normalize("what day is it now"), "what day is it")
        self.assertEqual(dm.normalize("how long until christmas from now"),
                         "how long until christmas from now")
        self.assertEqual(dm.normalize("the 25th of December"),
                         "the 25 of december")
        self.assertEqual(dm.normalize(None), "")


if __name__ == "__main__":
    unittest.main()
