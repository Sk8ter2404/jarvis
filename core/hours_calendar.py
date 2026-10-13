"""Pay-period calendar for the timesheet-hours reminder (2026-10-12).

Pay periods run Monday through Sunday, two weeks at a time, and the hours for
a period can only be entered once its days have passed. So the day to put the
hours in is the MONDAY a period ends: on that Monday the sheet to fill is the
fortnight that finished the day before.

Everything here is pure (no I/O, no clock of its own, never raises on a bad
setting), so the Monday job, the "is it hours week?" answer and the tests all
read ONE calendar. The anchor is the first day of any one period
(core.config.HOURS_REMINDER_ANCHOR); every period is a whole number of 14-day
steps from it, in both directions.

Vocabulary used throughout:
  period start   the Monday a two-week pay period begins (anchor + 14 k days).
  hours day      a period-start Monday: the previous period just ended and its
                 hours are due now.
  hours week     the Monday-Sunday week that starts on an hours day.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import NamedTuple, Optional

DEFAULT_ANCHOR = "2026-09-28"
DEFAULT_TIME = "09:00"

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


class Period(NamedTuple):
    start: date   # a Monday
    end: date     # the Sunday thirteen days later


def _as_date(value) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def parse_anchor(value) -> date:
    """The anchor as a Monday date. A bad or blank setting falls back to the
    shipped default rather than raising, and a non-Monday date counts as the
    Monday of its own week."""
    d = _as_date(value)
    if d is None:
        try:
            d = datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
        except Exception:
            d = datetime.strptime(DEFAULT_ANCHOR, "%Y-%m-%d").date()
    return d - timedelta(days=d.weekday())


def parse_clock(value) -> tuple[int, int]:
    """(hour, minute) from "HH:MM"; a bad setting gives the default time."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if m and int(m.group(1)) <= 23 and int(m.group(2)) <= 59:
        return int(m.group(1)), int(m.group(2))
    h, mi = DEFAULT_TIME.split(":")
    return int(h), int(mi)


def period_containing(day, anchor) -> Period:
    """The two-week period that ``day`` falls in."""
    d = _as_date(day)
    a = parse_anchor(anchor)
    steps = (d - a).days // 14
    start = a + timedelta(days=14 * steps)
    return Period(start, start + timedelta(days=13))


def is_hours_day(day, anchor) -> bool:
    """True on a period-start Monday: the previous period just ended."""
    d = _as_date(day)
    return d.weekday() == 0 and (d - parse_anchor(anchor)).days % 14 == 0


def hours_week_monday(day, anchor) -> Optional[date]:
    """The hours-day Monday of the week ``day`` is in, or None when this is
    not an hours week."""
    d = _as_date(day)
    monday = d - timedelta(days=d.weekday())
    return monday if is_hours_day(monday, anchor) else None


def period_due(hours_day) -> Period:
    """The period whose hours are due on ``hours_day`` (an hours-day Monday):
    the fortnight that ended the day before."""
    d = _as_date(hours_day)
    return Period(d - timedelta(days=14), d - timedelta(days=1))


def next_hours_day(day, anchor, *, include_today: bool = True) -> date:
    """The first hours day on or after ``day`` (strictly after it when
    ``include_today`` is False)."""
    d = _as_date(day)
    a = parse_anchor(anchor)
    if not include_today:
        d = d + timedelta(days=1)
    steps = -((a - d).days // 14)   # ceil((d - a) / 14)
    return a + timedelta(days=14 * steps)


def date_words(d) -> str:
    """"Monday, September 28" (no leading zero, no platform strftime flags)."""
    d = _as_date(d)
    return f"{_WEEKDAYS[d.weekday()]}, {_MONTHS[d.month - 1]} {d.day}"


def span_words(period: Period) -> str:
    """"Monday, September 28 to Sunday, October 11"."""
    return f"{date_words(period.start)} to {date_words(period.end)}"


def reminder_line(hours_day) -> str:
    """The spoken Monday reminder for ``hours_day``."""
    return (f"Time to put in your hours, sir, for "
            f"{span_words(period_due(hours_day))}.")


def status_line(today, anchor) -> str:
    """The spoken answer to "is it hours week?" / "when do I put in my
    hours" as of ``today``."""
    t = _as_date(today)
    monday = hours_week_monday(t, anchor)
    if monday is not None:
        due = span_words(period_due(monday))
        if t == monday:
            return f"Yes, sir, it's hours week. Today's the day: {due}."
        return (f"Yes, sir, it's hours week. Your hours for {due} are due "
                f"now.")
    nxt = next_hours_day(t, anchor, include_today=False)
    return (f"Not this week, sir. Your next hours day is "
            f"{date_words(nxt)}, for {span_words(period_due(nxt))}.")


# ── "is it hours week?" / "when do I put in my hours" ──────────────────────
# Whole-utterance grammars only (the core.fast_paths contract): a sentence
# that merely mentions hours ("how many hours of battery") never matches, and
# "put in my hours" as a COMMAND never does either.
_H = r"(?:hours|timesheets?|time sheets?)"
_HOURS_Q_RES = tuple(re.compile(p) for p in (
    rf"^(?:is it|is this|is today|is that) (?:an? )?(?:my )?{_H} "
    rf"(?:week|day)$",
    rf"^(?:is it|is this) (?:time|the day) (?:to|for me to) "
    rf"(?:put in|enter|submit|do|fill out|fill in|log) "
    rf"(?:my |the )?{_H}$",
    rf"^(?:is|are) (?:my |the )?{_H} due(?: (?:today|this week|now))?$",
    rf"^do i (?:have|need) to (?:put in|enter|submit|do|log) "
    rf"(?:my |the )?{_H}(?: (?:today|this week|now))?$",
    rf"^when (?:do i|should i|do i have to|am i supposed to|is it time to) "
    rf"(?:put in|enter|submit|do|fill out|fill in|log|turn in) "
    rf"(?:my |the )?{_H}$",
    rf"^when (?:are|is) (?:my |the )?{_H} (?:due|next|supposed to be "
    rf"(?:put in|entered|submitted))$",
    rf"^when(?:s| is) (?:my |the )?next {_H} "
    rf"(?:week|day|reminder|deadline)$",
    rf"^what(?:s| is) the next {_H} (?:week|day)$",
    rf"^when(?:s| is) {_H} (?:week|day)$",
))


def is_hours_question(text) -> bool:
    """True for a whole-utterance "is it hours week" / "when do I put in my
    hours" question. Never raises."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        from core.date_math import normalize
        t = re.sub(r"\s+", " ", normalize(text)).strip()
        return any(rx.match(t) for rx in _HOURS_Q_RES)
    except Exception:
        return False
