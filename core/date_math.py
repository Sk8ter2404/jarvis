"""Deterministic answers to relative-date questions, with no LLM involved.

The local model is unreliable at calendar arithmetic, and slow. Live on
2026-09-29 it read back TODAY's date for "what's the date tomorrow", answered
"1 day and 14 hours" for a Friday three calendar days away, and counted 86 days
to Christmas instead of 87. Everything here is computed from a supplied ``now``
so the caller (the monolith's fast path) owns the clock and tests freeze it.

Entry point: ``answer(text, now) -> DateAnswer | None``.

What it understands (after ``normalize``: case, punctuation, contractions, a
leading wake word / "can you tell me" and a trailing "please" / "sir" are
ignored):

  * today / tomorrow / yesterday / the day after tomorrow / the day before
    yesterday: "what's the date tomorrow", "what day is it", "what was
    yesterday's date", "what day of the week is tomorrow";
  * a day or week offset: "what's the date in 3 days", "what day will it be in
    two weeks", "what's the date a week from today", "what day was it 5 days
    ago";
  * days until a target: "how long until Friday", "how many days until
    Christmas", "how far away is Thanksgiving", "days until December 25";
  * the weekday / date of a target: "what day of the week is December 25",
    "what day does Christmas fall on this year", "when is Thanksgiving".

A target is a weekday, a named holiday (Christmas, Christmas Eve, New Year's
Day, New Year's Eve, Halloween, Thanksgiving = the 4th Thursday of November,
Independence Day / the Fourth of July, Valentine's Day) or a calendar date
("December 25", "the 25th of December", "Dec 25 2027", "12/25", "2027-12-25";
numeric dates are US month/day).

Counting rules (the contract tests/test_date_math.py pins):

  * Days are CALENDAR days between the two dates; the time of day never
    matters (Tue Sep 29 -> Fri Dec 25 = 87; Tue -> Fri = 3).
  * A bare weekday always means the NEXT one, never today: asked on a
    Tuesday, "how long until Tuesday" is 7 days and the reply says today is
    Tuesday. "next Friday" is ambiguous in speech (this week's or next
    week's?) and returns None.
  * A holiday or a date without a year means its next occurrence, today
    included: Christmas asked on Dec 25 is "today", asked on Dec 26 it is next
    year's. "next Christmas" skips today. A trailing "this year" / "next year"
    pins the year and may give a past date ("was N days ago"). A yearless date
    that does not exist this year (Feb 29) rolls to the next year it exists.
  * An explicit year is taken literally, past or future.

It returns None for anything it does not FULLY understand: the utterance must
match one of the question frames below end to end AND the "when" part must
parse completely. So commands and other domains ("remind me tomorrow to ...",
"what's the weather tomorrow", "what's on my calendar tomorrow", "set a timer
...", "schedule ...", "wake me up tomorrow", "play music until Friday", "pause
until tomorrow") never match and fall through to the normal turn.

Replies are short and end with ", sir." like the rest of JARVIS. Stdlib only,
no I/O, never raises: a bug returns None (the LLM answers), never a wrong turn.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import NamedTuple, Optional


class DateAnswer(NamedTuple):
    kind: str    # "date" | "date-offset" | "days-until" | "date-of"
    reply: str   # the spoken line, ending ", sir."


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday")
_WEEKDAY_TITLE = tuple(w.title() for w in WEEKDAYS)
_MONTH_TITLE = ("January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December")
_MONTHS = {name.lower(): i + 1 for i, name in enumerate(_MONTH_TITLE)}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
                "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11,
                "dec": 12})

_UNITS = {w: i for i, w in enumerate(
    ("zero one two three four five six seven eight nine ten eleven twelve "
     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_ORD_UNITS = {w: i + 1 for i, w in enumerate(
    ("first second third fourth fifth sixth seventh eighth ninth tenth "
     "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
     "eighteenth nineteenth").split())}
_ORD_TENS = {"twentieth": 20, "thirtieth": 30}

# The furthest offset answered ("in 36500 days"); anything bigger is None.
_MAX_OFFSET_DAYS = 36500


# ── normalisation ──────────────────────────────────────────────────────────

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"',
                         "”": '"'})
_CONTRACTIONS = (
    (re.compile(r"\bwhat'?s\b"), "what is"),
    (re.compile(r"\bwhat'll\b"), "what will"),
    (re.compile(r"\bwhen'?s\b"), "when is"),
    (re.compile(r"\bit's\b"), "it is"),
)
# Politeness and the wake word, at either end. Deliberately NOT "remind me":
# "remind me tomorrow to ..." is a command and must stay one.
_LEAD_RE = re.compile(
    r"^(?:(?:hey|hi|ok|okay|yo|so|and|um|uh|er|well|oh|please|alright|"
    r"all right|jarvis|sir|quick question)\b\s*)+")
_TRAIL_RE = re.compile(
    r"(?:\s+(?:please|jarvis|sir|thanks|thank you|for me|then|again|"
    r"right now|exactly))+$")
# "can you tell me what day it is" -> "what day it is" (date grammar only).
_ASK_LEAD_RE = re.compile(
    r"^(?:(?:can|could|would|will) you(?: please)? tell me|"
    r"(?:please )?tell me|do you (?:happen to )?know|"
    r"i (?:want|need|would like|d like) to know|i wonder|i was wondering|"
    r"any idea)\b\s*")


def normalize(text) -> str:
    """Lower-case, expand the few contractions the grammars use, drop
    apostrophes and sentence punctuation (keeping the / and - of numeric
    dates), strip ordinal suffixes from numbers ("25th" -> "25") and peel a
    leading wake word / politeness and a trailing "please" / "sir" / "now"
    (but not the "now" of "from now"). Non-strings normalise to ""."""
    if not isinstance(text, str):
        return ""
    t = text.translate(_QUOTES).lower().strip()
    for pat, rep in _CONTRACTIONS:
        t = pat.sub(rep, t)
    t = t.replace("'", "")
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    t = re.sub(r"[\"?!.,;:()\[\]]+", " ", t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = _LEAD_RE.sub("", t).strip()
        t = _TRAIL_RE.sub("", t).strip()
        if t.endswith(" now") and not t.endswith(" from now"):
            t = t[:-4].strip()
    return t


# ── numbers ────────────────────────────────────────────────────────────────

def _small_number(toks: list, ordinal: bool = False) -> Optional[int]:
    """1..99 from number words ("three", "twenty one", "twenty first")."""
    units = _ORD_UNITS if ordinal else _UNITS
    if len(toks) == 1:
        t = toks[0]
        if ordinal:
            return _ORD_UNITS.get(t) or _ORD_TENS.get(t)
        return _UNITS.get(t) if t in _UNITS else _TENS.get(t)
    if len(toks) == 2 and toks[0] in _TENS:
        u = units.get(toks[1])
        if u is not None and 1 <= u <= 9:
            return _TENS[toks[0]] + u
    return None


def _cardinal(s: str) -> Optional[int]:
    """A count: digits, "a"/"an", "a couple (of)", number words up to 999."""
    s = s.strip()
    if re.fullmatch(r"\d{1,5}", s):
        return int(s)
    if s in ("a", "an"):
        return 1
    if re.fullmatch(r"(?:a )?couple(?: of)?", s):
        return 2
    toks = s.split()
    if "hundred" in toks:
        i = toks.index("hundred")
        head, tail = toks[:i], toks[i + 1:]
        if head in ([], ["a"], ["one"]):
            hundreds = 1
        else:
            hundreds = _small_number(head)
            if hundreds is None or not 1 <= hundreds <= 9:
                return None
        if tail and tail[0] == "and":
            tail = tail[1:]
        rest = _small_number(tail) if tail else 0
        if rest is None:
            return None
        return hundreds * 100 + rest
    return _small_number(toks)


def _day_of_month(s: str) -> Optional[int]:
    s = s.strip()
    if re.fullmatch(r"\d{1,2}", s):
        return int(s)
    toks = s.split()
    return _small_number(toks, ordinal=True) or _small_number(toks)


# ── targets: weekdays, holidays, dates ─────────────────────────────────────

def _thanksgiving(year: int) -> _dt.date:
    """US Thanksgiving: the 4th Thursday of November."""
    first = _dt.date(year, 11, 1)
    return first + _dt.timedelta(days=(3 - first.weekday()) % 7 + 21)


_HOLIDAYS = {
    "christmas": ("Christmas", lambda y: _dt.date(y, 12, 25)),
    "christmas eve": ("Christmas Eve", lambda y: _dt.date(y, 12, 24)),
    "new years day": ("New Year's Day", lambda y: _dt.date(y, 1, 1)),
    "new years eve": ("New Year's Eve", lambda y: _dt.date(y, 12, 31)),
    "halloween": ("Halloween", lambda y: _dt.date(y, 10, 31)),
    "thanksgiving": ("Thanksgiving", _thanksgiving),
    "independence day": ("Independence Day", lambda y: _dt.date(y, 7, 4)),
    "valentines day": ("Valentine's Day", lambda y: _dt.date(y, 2, 14)),
}
# Spoken forms after normalize() (apostrophes gone, "4th" -> "4").
_HOLIDAY_ALIASES = {
    "christmas": "christmas", "christmas day": "christmas",
    "xmas": "christmas", "xmas day": "christmas",
    "christmas eve": "christmas eve", "xmas eve": "christmas eve",
    "new years": "new years day", "new year": "new years day",
    "new years day": "new years day", "new year day": "new years day",
    "the new year": "new years day",
    "new years eve": "new years eve", "new year eve": "new years eve",
    "halloween": "halloween",
    "thanksgiving": "thanksgiving", "thanksgiving day": "thanksgiving",
    "independence day": "independence day",
    "fourth of july": "independence day", "4 of july": "independence day",
    "july fourth": "independence day", "july 4": "independence day",
    "valentines day": "valentines day", "valentines": "valentines day",
    "valentine day": "valentines day", "valentine": "valentines day",
    "saint valentines day": "valentines day",
    "st valentines day": "valentines day",
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_TOK = r"(?P<d>\d{1,2}|[a-z]+(?: [a-z]+)?)"
_DATE_RES = (
    re.compile(rf"(?P<m>{_MONTH_ALT}) (?:the )?{_DAY_TOK}"
               rf"(?: (?:of )?(?P<y>\d{{4}}))?"),
    re.compile(rf"(?:the )?{_DAY_TOK}(?: of)? (?P<m>{_MONTH_ALT})"
               rf"(?: (?P<y>\d{{4}}))?"),
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
    re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})(?:/(?P<y>\d{4}|\d{2}))?"),
)


def _parse_date(t: str):
    """(month, day, year | None) for a calendar-date phrase, else None."""
    for rx in _DATE_RES:
        m = rx.fullmatch(t)
        if not m:
            continue
        month_s = m.group("m")
        month = _MONTHS.get(month_s) if not month_s.isdigit() else int(month_s)
        day = _day_of_month(m.group("d"))
        year_s = m.group("y")
        year = None
        if year_s:
            year = int(year_s) + (2000 if len(year_s) == 2 else 0)
        if not month or not day or not 1 <= month <= 12 or not 1 <= day <= 31:
            return None
        if year is not None and not 1 <= year <= 9999:
            return None
        return month, day, year
    return None


class _Target(NamedTuple):
    kind: str            # "weekday" | "holiday" | "date"
    date: _dt.date
    label: str           # "Friday", "Christmas", ""
    explicit_year: bool  # the user named the year ("this year" counts)


def _parse_target(w: str, today: _dt.date, qual: Optional[str]):
    t = re.sub(r"^(?:it is|its|it will be|on) ", "", w)
    t = re.sub(r" on$", "", t)
    nxt = False
    m = re.match(r"(?:this coming|the coming|coming|this|next) ", t)
    if m:
        nxt = m.group(0).strip() == "next"
        t = t[m.end():]
    if t in WEEKDAYS:
        if nxt or qual:
            return None
        wd = WEEKDAYS.index(t)
        ahead = (wd - today.weekday()) % 7 or 7
        return _Target("weekday", today + _dt.timedelta(days=ahead),
                       t.title(), False)
    key = _HOLIDAY_ALIASES.get(t)
    if key is None and t.startswith("the "):
        key = _HOLIDAY_ALIASES.get(t[4:])
    if key is not None:
        label, on = _HOLIDAYS[key]
        if qual:
            d = on(today.year + (1 if qual == "next" else 0))
        else:
            d = on(today.year)
            if d < today or (nxt and d == today):
                d = on(today.year + 1)
        return _Target("holiday", d, label, qual is not None)
    if nxt:
        return None     # "next December 25": rare, and as ambiguous as a weekday
    parsed = _parse_date(t)
    if parsed is None:
        return None
    month, day, year = parsed
    if year is not None:
        if qual:
            return None
        try:
            return _Target("date", _dt.date(year, month, day), "", True)
        except ValueError:
            return None
    if qual:
        try:
            d = _dt.date(today.year + (1 if qual == "next" else 0), month, day)
        except ValueError:
            return None
        return _Target("date", d, "", True)
    for year in range(today.year, today.year + 9):
        try:
            d = _dt.date(year, month, day)
        except ValueError:
            continue
        if d >= today:
            return _Target("date", d, "", False)
    return None


# ── the "when" part of a date question ─────────────────────────────────────

_REL = {"today": 0, "tomorrow": 1, "yesterday": -1,
        "the day after tomorrow": 2, "day after tomorrow": 2,
        "after tomorrow": 2,
        "the day before yesterday": -2, "day before yesterday": -2,
        "before yesterday": -2}
_REL_WORDS = {0: ("Today", "is"), 1: ("Tomorrow", "is"),
              -1: ("Yesterday", "was"), 2: ("The day after tomorrow", "is"),
              -2: ("The day before yesterday", "was")}
_POSS = {"todays": 0, "tomorrows": 1, "yesterdays": -1}

_OFFSET_RES = (
    (re.compile(r"in (?P<n>.+?) (?P<u>days?|weeks?)"
                r"(?: time| from (?:now|today))?"), False),
    (re.compile(r"(?P<n>.+?) (?P<u>days?|weeks?) from (?:now|today)"), False),
    (re.compile(r"(?P<n>.+?) (?P<u>days?|weeks?) ago"), True),
)


def _parse_offset(w: str):
    """(count, "day" | "week", past) for "in 3 days" / "a week from today" /
    "5 days ago", else None."""
    for rx, past in _OFFSET_RES:
        m = rx.fullmatch(w)
        if not m:
            continue
        n = _cardinal(m.group("n"))
        unit = "week" if m.group("u").startswith("week") else "day"
        if not n or n * (7 if unit == "week" else 1) > _MAX_OFFSET_DAYS:
            return None
        return n, unit, past
    return None


# ── spoken forms ───────────────────────────────────────────────────────────

def _full(d: _dt.date, with_year: bool = True) -> str:
    """ "Friday, December 25, 2026" (the year optional). """
    out = f"{_WEEKDAY_TITLE[d.weekday()]}, {_MONTH_TITLE[d.month - 1]} {d.day}"
    return out + (f", {d.year}" if with_year else "")


def _month_day(d: _dt.date, today: _dt.date, force_year: bool = False) -> str:
    """ "December 25", plus ", 2027" when the year is not this one. """
    out = f"{_MONTH_TITLE[d.month - 1]} {d.day}"
    if force_year or d.year != today.year:
        out += f", {d.year}"
    return out


def _count(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


def _rel_answer(k: int, today: _dt.date) -> DateAnswer:
    word, verb = _REL_WORDS[k]
    d = today + _dt.timedelta(days=k)
    return DateAnswer("date", f"{word} {verb} {_full(d)}, sir.")


def _offset_answer(n: int, unit: str, past: bool,
                   today: _dt.date) -> DateAnswer:
    days = n * (7 if unit == "week" else 1)
    d = today + _dt.timedelta(days=-days if past else days)
    span = f"a {unit}" if n == 1 else _count(n, unit)
    if past:
        return DateAnswer("date-offset", f"{span[0].upper()}{span[1:]} ago it "
                                         f"was {_full(d)}, sir.")
    return DateAnswer("date-offset", f"In {span} it will be {_full(d)}, sir.")


def _until_answer(tgt: _Target, today: _dt.date) -> DateAnswer:
    n = (tgt.date - today).days
    if tgt.kind == "weekday":
        on = _month_day(tgt.date, today)
        if n == 7:
            reply = (f"Today is {tgt.label}, so next {tgt.label} is 7 days "
                     f"away, on {on}, sir.")
        elif n == 1:
            reply = f"{tgt.label} is tomorrow, {on}, sir."
        else:
            reply = f"{tgt.label} is {n} days away, on {on}, sir."
        return DateAnswer("days-until", reply)
    if tgt.kind == "holiday":
        label = tgt.label
        where = f", on {_full(tgt.date, with_year=tgt.date.year != today.year)}"
    else:
        label = _month_day(tgt.date, today, force_year=tgt.explicit_year)
        where = f", on a {_WEEKDAY_TITLE[tgt.date.weekday()]}"
    if n == 0:
        reply = f"{label} is today, sir."
    elif n == 1:
        reply = f"{label} is tomorrow, sir."
    elif n == -1:
        reply = f"{label} was yesterday, sir."
    elif n > 1:
        reply = f"{label} is {n} days away{where}, sir."
    else:
        reply = f"{label} was {-n} days ago{where}, sir."
    return DateAnswer("days-until", reply)


def _date_of_answer(tgt: _Target, today: _dt.date) -> DateAnswer:
    n = (tgt.date - today).days
    weekday = _WEEKDAY_TITLE[tgt.date.weekday()]
    md_y = _month_day(tgt.date, today, force_year=True)
    if tgt.kind == "weekday":
        if n == 7:
            reply = (f"Today is {tgt.label}; next {tgt.label} is {md_y}, "
                     f"sir.")
        else:
            reply = f"{tgt.label} is {md_y}, sir."
    elif tgt.kind == "holiday":
        full = _full(tgt.date)
        if n == 0:
            reply = f"{tgt.label} is today, {full}, sir."
        elif n == 1:
            reply = f"{tgt.label} is tomorrow, {full}, sir."
        elif n > 1:
            reply = f"{tgt.label} is on {full}, {n} days from now, sir."
        else:
            reply = f"{tgt.label} was on {full}, sir."
    else:
        if n == 0:
            reply = f"{md_y} is today, a {weekday}, sir."
        elif n > 0:
            reply = f"{md_y} is a {weekday}, sir."
        else:
            reply = f"{md_y} was a {weekday}, sir."
    return DateAnswer("date-of", reply)


# ── question frames ────────────────────────────────────────────────────────
# Each frame must match the WHOLE (normalised) utterance; its <w> part must
# then parse completely, or the frame is skipped. The first frame whose <w>
# resolves wins.

_NOUN = r"(?:date|day(?: of the (?:week|month))?|day of week)"
_Q = r"(?:what|which)"
_BE = r"(?:is|was|will be|would be)"
_W = r"(?P<w>.+?)"

_DATE_FRAMES = tuple(re.compile(p) for p in (
    rf"{_Q} {_BE} (?:the )?{_NOUN}(?: (?:for|on|of))?(?: {_W})?",
    rf"{_Q} (?:is|was) (?:the )?{_NOUN} (?:going to be|gonna be) {_W}",
    rf"{_Q} {_NOUN} (?:is|was)(?: it)?(?: going to be)?(?: {_W})?",
    rf"{_Q} {_NOUN} will(?: it)? be(?: {_W})?",
    rf"{_Q} {_NOUN} (?:will|does|did|is|was) {_W} "
    rf"(?:be|fall|land|fall on|land on|be on|on)",
    rf"{_Q} (?:the )?{_NOUN}(?: it)? {_BE}(?: {_W})?",
    rf"{_Q} {_NOUN} {_W} (?:is|was|will be|is on|was on|falls on|fell on|"
    rf"lands on|will fall on)",
))
# Only after "can you tell me" / "do you know": "tell me the date tomorrow".
_ASKED_FRAMES = tuple(re.compile(p) for p in (
    rf"the {_NOUN}(?: (?:for|on|of))?(?: {_W})?",
))
_WHEN_FRAMES = tuple(re.compile(p) for p in (
    rf"when (?:is|was|will be) {_W}",
    rf"when (?:does|did|will) {_W} (?:fall|land|come|be)(?: on)?",
))
_POSS_FRAMES = tuple(re.compile(p) for p in (
    rf"{_Q} {_BE} (?P<poss>todays|tomorrows|yesterdays) (?:date|day)",
    r"(?P<poss>todays|tomorrows|yesterdays) date",
))
_FILL = (r"(?:is it|is there|are there|are there left|are left|is left|left|"
         r"remain|remaining|more|do i have|do we have|do i have left|"
         r"do we have left|have i got|have we got|do i have to wait|"
         r"do we have to wait)")
_UNTIL_FRAMES = tuple(re.compile(p) for p in (
    rf"how (?:long|many (?:more )?days)(?: {_FILL})? "
    rf"(?:until|till|til|to|before) {_W}(?: from (?:now|today))?",
    rf"how (?:far(?: away| off| out)?|many days(?: away| off| out)?|"
    rf"long away) is {_W}(?: away| off| from (?:now|today))?",
    rf"(?:the )?(?:number of )?days (?:left )?(?:until|till|til|to) {_W}",
))


def _resolve_date_question(w: Optional[str], today: _dt.date,
                           qual: Optional[str],
                           targets_only: bool = False) -> Optional[DateAnswer]:
    if w is None:
        if targets_only or qual:
            return None
        return _rel_answer(0, today)
    w = re.sub(r"^on ", "", w)
    if not targets_only and not qual:
        if w in _REL:
            return _rel_answer(_REL[w], today)
        off = _parse_offset(w)
        if off is not None:
            return _offset_answer(*off, today)
    tgt = _parse_target(w, today, qual)
    return _date_of_answer(tgt, today) if tgt is not None else None


def _answer(text, now) -> Optional[DateAnswer]:
    if isinstance(now, _dt.datetime):
        today = now.date()
    elif isinstance(now, _dt.date):
        today = now
    else:
        return None
    t = normalize(text)
    if not t or len(t) > 120:
        return None
    asked = _ASK_LEAD_RE.match(t)
    if asked:
        t = t[asked.end():].strip()
    qual = None
    m = re.fullmatch(r"(.+?) (this|next) year", t)
    if m:
        t, qual = m.group(1), m.group(2)

    for rx in _UNTIL_FRAMES:
        m = rx.fullmatch(t)
        if m:
            tgt = _parse_target(m.group("w"), today, qual)
            if tgt is not None:
                return _until_answer(tgt, today)
    for rx in _POSS_FRAMES:
        m = rx.fullmatch(t)
        if m and not qual:
            return _rel_answer(_POSS[m.group("poss")], today)
    frames = _DATE_FRAMES + (_ASKED_FRAMES if asked else ())
    for rx in frames:
        m = rx.fullmatch(t)
        if m:
            got = _resolve_date_question(m.group("w"), today, qual)
            if got is not None:
                return got
    for rx in _WHEN_FRAMES:
        m = rx.fullmatch(t)
        if m:
            got = _resolve_date_question(m.group("w"), today, qual,
                                         targets_only=True)
            if got is not None:
                return got
    return None


def answer(text, now) -> Optional[DateAnswer]:
    """The spoken answer to a relative-date question asked at ``now`` (a
    datetime or date), or None when ``text`` is not one this module fully
    understands. Never raises."""
    try:
        return _answer(text, now)
    except Exception:
        return None
