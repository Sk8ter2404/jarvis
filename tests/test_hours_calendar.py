"""Pay-period calendar for the Monday hours reminder (core/hours_calendar.py).

Anchor: one period runs Sep 28 - Oct 11, 2026, so the next starts Oct 12 and
the hours for the period that just ended are due on that Monday.
"""
from __future__ import annotations

import unittest
from datetime import date, datetime

from core import hours_calendar as hc

ANCHOR = "2026-09-28"


class PeriodMathTests(unittest.TestCase):
    def test_the_known_period(self):
        p = hc.period_containing(date(2026, 10, 5), ANCHOR)
        self.assertEqual(p, hc.Period(date(2026, 9, 28), date(2026, 10, 11)))
        self.assertEqual(p.start.weekday(), 0)
        self.assertEqual(p.end.weekday(), 6)

    def test_next_period_starts_oct_12(self):
        p = hc.period_containing(date(2026, 10, 12), ANCHOR)
        self.assertEqual(p.start, date(2026, 10, 12))
        self.assertEqual(p.end, date(2026, 10, 25))

    def test_periods_before_the_anchor(self):
        p = hc.period_containing(date(2026, 9, 27), ANCHOR)
        self.assertEqual(p, hc.Period(date(2026, 9, 14), date(2026, 9, 27)))

    def test_hours_days_are_every_other_monday(self):
        for d in (date(2026, 9, 28), date(2026, 10, 12), date(2026, 10, 26),
                  date(2026, 11, 9), date(2027, 1, 4)):
            self.assertTrue(hc.is_hours_day(d, ANCHOR), d)
        for d in (date(2026, 10, 5), date(2026, 10, 19), date(2026, 11, 2),
                  date(2026, 10, 13)):
            self.assertFalse(hc.is_hours_day(d, ANCHOR), d)

    def test_due_period_is_the_one_that_just_ended(self):
        due = hc.period_due(date(2026, 10, 12))
        self.assertEqual(due, hc.Period(date(2026, 9, 28), date(2026, 10, 11)))

    def test_hours_week_is_the_week_of_an_hours_day(self):
        self.assertEqual(hc.hours_week_monday(date(2026, 10, 14), ANCHOR),
                         date(2026, 10, 12))
        self.assertEqual(hc.hours_week_monday(date(2026, 10, 18), ANCHOR),
                         date(2026, 10, 12))
        self.assertIsNone(hc.hours_week_monday(date(2026, 10, 19), ANCHOR))
        self.assertIsNone(hc.hours_week_monday(date(2026, 10, 25), ANCHOR))

    def test_next_hours_day(self):
        self.assertEqual(hc.next_hours_day(date(2026, 10, 12), ANCHOR),
                         date(2026, 10, 12))
        self.assertEqual(
            hc.next_hours_day(date(2026, 10, 12), ANCHOR, include_today=False),
            date(2026, 10, 26))
        self.assertEqual(hc.next_hours_day(date(2026, 10, 13), ANCHOR),
                         date(2026, 10, 26))
        self.assertEqual(hc.next_hours_day(date(2026, 9, 1), ANCHOR),
                         date(2026, 9, 14))

    def test_year_rollover_keeps_the_rhythm(self):
        # Oct 12 + 14k: ... Dec 21, Jan 4. ISO week parity would break here.
        self.assertTrue(hc.is_hours_day(date(2026, 12, 21), ANCHOR))
        self.assertTrue(hc.is_hours_day(date(2027, 1, 4), ANCHOR))
        self.assertFalse(hc.is_hours_day(date(2026, 12, 28), ANCHOR))

    def test_datetime_is_accepted(self):
        self.assertTrue(hc.is_hours_day(datetime(2026, 10, 12, 9, 0), ANCHOR))


class AnchorSettingTests(unittest.TestCase):
    def test_non_monday_anchor_counts_as_its_monday(self):
        self.assertEqual(hc.parse_anchor("2026-10-01"), date(2026, 9, 28))
        self.assertTrue(hc.is_hours_day(date(2026, 10, 12), "2026-10-01"))

    def test_bad_anchor_falls_back_to_the_default(self):
        for bad in ("", "soon", None, "2026-13-45", 42):
            self.assertEqual(hc.parse_anchor(bad), date(2026, 9, 28), bad)

    def test_other_anchor_moves_the_rhythm(self):
        self.assertTrue(hc.is_hours_day(date(2026, 10, 5), "2026-09-21"))
        self.assertFalse(hc.is_hours_day(date(2026, 10, 12), "2026-09-21"))

    def test_clock_setting(self):
        self.assertEqual(hc.parse_clock("09:00"), (9, 0))
        self.assertEqual(hc.parse_clock("7:45"), (7, 45))
        self.assertEqual(hc.parse_clock("25:00"), (9, 0))
        self.assertEqual(hc.parse_clock("nine"), (9, 0))
        self.assertEqual(hc.parse_clock(None), (9, 0))


class SpokenLineTests(unittest.TestCase):
    def test_reminder_line_names_both_dates(self):
        line = hc.reminder_line(date(2026, 10, 12))
        self.assertEqual(
            line, "Time to put in your hours, sir, for Monday, September 28 "
                  "to Sunday, October 11.")

    def test_status_on_the_hours_day(self):
        line = hc.status_line(date(2026, 10, 12), ANCHOR)
        self.assertTrue(line.startswith("Yes, sir, it's hours week."))
        self.assertIn("Monday, September 28 to Sunday, October 11", line)
        self.assertIn("Today's the day", line)

    def test_status_later_in_the_hours_week(self):
        line = hc.status_line(date(2026, 10, 15), ANCHOR)
        self.assertTrue(line.startswith("Yes, sir, it's hours week."))
        self.assertIn("due now", line)

    def test_status_in_the_off_week_names_the_next_one(self):
        line = hc.status_line(date(2026, 10, 20), ANCHOR)
        self.assertTrue(line.startswith("Not this week, sir."))
        self.assertIn("Monday, October 26", line)
        self.assertIn("Monday, October 12 to Sunday, October 25", line)

    def test_status_on_the_sunday_before_an_hours_day(self):
        line = hc.status_line(date(2026, 10, 11), ANCHOR)
        self.assertIn("Monday, October 12", line)
        self.assertTrue(line.startswith("Not this week, sir."))


class QuestionGrammarTests(unittest.TestCase):
    YES = (
        "is it hours week", "Is it hours week?", "jarvis is it hours week",
        "is this an hours week", "is it my hours week", "is it timesheet week",
        "when do I put in my hours", "When do I put in my hours?",
        "when should I put in my hours", "when do I enter my hours",
        "when are my hours due", "when's my next hours day",
        "is it time to put in my hours", "are my hours due",
        "do I need to put in my hours today", "when is hours day",
        "what's the next hours week",
    )
    NO = (
        "put in my hours", "how many hours of battery are left",
        "what hours are you open", "remind me to put in my hours",
        "set a timer for two hours", "is it a good time", "",
        "when do I get paid", "how many hours did I work",
    )

    def test_questions_match(self):
        for t in self.YES:
            with self.subTest(t=t):
                self.assertTrue(hc.is_hours_question(t), t)

    def test_non_questions_do_not(self):
        for t in self.NO:
            with self.subTest(t=t):
                self.assertFalse(hc.is_hours_question(t), t)

    def test_non_strings_never_raise(self):
        for t in (None, 5, [], b"is it hours week"):
            self.assertFalse(hc.is_hours_question(t))


if __name__ == "__main__":
    unittest.main()
