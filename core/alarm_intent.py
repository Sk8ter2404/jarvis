"""core/alarm_intent.py - "set an alarm" recognised without the brain.

Live 2026-11-12: the same line, "set an alarm for 422", was passed through
to set_alarm by the brain at 16:21:18 and answered with a question about the
number at 17:03:22. A plain ask for an alarm at a time should not depend on
how the model reads the time.

match(text) -> the argument for set_alarm ("at 4:22", "6:30 am", "in 20
minutes"), or None when the line is not a plain ask for an alarm at a time
the grammar reads. Whole line only: the ask must START the line (after the
wake word and politeness) and nothing but a short reaction may follow the
time. Anything else is left to the brain. Pure, stdlib only, never raises.
"""
from __future__ import annotations

import re
from typing import Optional

_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "ninety": 90}
_NUM_WORD_RE = re.compile(
    r"\b(?:(?P<oh>oh|o) (?P<ohu>" + "|".join(_UNITS) + r")"
    r"|(?P<tens>" + "|".join(_TENS) + r")(?: (?P<tu>" + "|".join(_UNITS) + r"))?"
    r"|(?P<teen>" + "|".join(_TEENS) + r")"
    r"|(?P<unit>" + "|".join(_UNITS) + r"))\b")


def _digits(t: str) -> str:
    """Number words as digits: "six thirty" -> "6 30", "six oh five" ->
    "6 05", "forty five minutes" -> "45 minutes"."""
    def rep(m):
        if m.group("oh"):
            return f"0{_UNITS[m.group('ohu')]}"
        if m.group("tens"):
            return str(_TENS[m.group("tens")] + _UNITS.get(m.group("tu") or "", 0))
        if m.group("teen"):
            return str(_TEENS[m.group("teen")])
        return str(_UNITS[m.group("unit")])
    return _NUM_WORD_RE.sub(rep, t)


# The wake word, fillers and politeness in front of the ask.
_LEAD_RE = re.compile(
    r"^(?:(?:hey|hi|ok|okay|yo|um|uh|so|and|well|oh|alright|all right|yeah|"
    r"yes|yep|right|now|also|please|sir|jarvis)\b[\s,.!]*"
    r"|(?:can|could|would) you (?:please )?"
    r"|go ahead and |just |quickly "
    r"|i(?: am|'m) going to need you to |i need you to |i want you to "
    r"|i(?: would|'d) like you to "
    r"|i need to |i want to |i wanna |i(?: would|'d) like to |let'?s )")

_VERB = (r"(?:set(?:ting)?(?: up)?|make|create|start|add|schedule|put on|do|"
         r"give me|i need|i want|i(?: would|'d) like|we need)(?: me)?")
_ADJ = r"(?:(?:new|wake ?up|morning|loud|quick|test|second) )*"
_PRE = (r"(?:tomorrow morning|in the morning|this morning|tomorrow afternoon|"
        r"this afternoon|tomorrow evening|this evening|tonight|tomorrow night|"
        r"tomorrow)")
# "set an alarm for ...", "alarm at ...", "an alarm to go off in ..."
_HEAD_ALARM_RE = re.compile(
    rf"^(?:(?P<verb>{_VERB}) )?(?:(?P<art>an?|another|1 more|the|my) )?{_ADJ}"
    r"alarm(?: clock)?(?: set| going)?"
    r"(?: to (?:go off|ring|sound|wake me(?: up)?))?,?"
    rf"(?: (?:for )?(?P<pre>{_PRE}))?"
    r"(?: (?P<prep>for|at|in|by|around))?(?: (?:like|about|around|exactly))? "
    r"(?=\d|an? |half |noon|midnight)")
# "wake me up at ...", "get me up by ..."
_HEAD_WAKE_RE = re.compile(
    rf"^(?:wake|get) me(?: up)?(?: (?P<pre>{_PRE}))?"
    r"(?: (?P<prep>at|in|by|around|for))?(?: (?:like|about|around|exactly))? "
    r"(?=\d|an? |half |noon|midnight)")
# "set a 20 minute alarm", "do a test 1 minute alarm", "set a 6 30 am alarm"
_HEAD_TIME_FIRST_RE = re.compile(
    rf"^(?:(?P<verb>{_VERB}) )?(?:(?P<art>an?|another|me an?) )?{_ADJ}"
    r"(?:(?P<n>\d{1,3})[- ](?P<unit>second|minute|hour)"
    r"|(?P<clock>\d{1,2}(?:[:. ]?\d{2})?(?: ?(?:a\.?\s?m|p\.?\s?m)\b)?))"
    rf" {_ADJ}alarm(?: clock)?(?=$|[ ,.!?])")

_AMPM = r"(?: ?(?P<ap>a\.?\s?m|p\.?\s?m)\b)"
_HINT = (r"(?:in the morning|this morning|tomorrow morning|in the afternoon|"
         r"this afternoon|tomorrow afternoon|in the evening|this evening|"
         r"tomorrow evening|tonight|at night|tomorrow night|tomorrow)")
_CLOCK_RE = re.compile(
    r"^(?P<h>\d{1,2})(?:[:. ]?(?P<m>\d{2}))?(?!\d)(?: ?o'?clock)?"
    rf"(?:{_AMPM}(?:\.(?= {_HINT}))?)?(?: (?P<hint>{_HINT}))?")
_WORD_CLOCK_RE = re.compile(rf"^(?P<w>noon|midnight)\b(?: (?P<hint>{_HINT}))?")
_U = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)"
_DUR_RE = re.compile(
    r"^(?:half an hour|an hour and a half"
    rf"|(?:an?|\d{{1,3}}) (?:hours?|hrs?)(?: and)? (?:a half|\d{{1,2}} (?:minutes?|mins?))"
    rf"|(?:an?|\d{{1,3}}) {_U})(?: from now)?(?![a-z0-9])")
_BARE_MIN_RE = re.compile(r"^(\d{1,3})(?![\d:.]?\d)(?![a-z])")

_POLITE_TAIL_RE = re.compile(
    r"^(?:[, ]*\b(?:please|sir|jarvis|thanks|thank you|for me|then|okay|ok)\b)*")
# A short reaction after the ask is let through ("... for 422. That's really
# good."). An allowlist, not a denylist: "Just kidding", "Hold on", "Different
# time" after the time must reach the brain, and no list of bad words can
# name them all.
_REMARK_OK_RE = re.compile(
    r"^(?:(?:that's|thats|that|is|really|very|so|good|great|nice|cool|"
    r"thanks|thank|you|okay|ok|perfect|awesome|all|sir|jarvis|please|wow|"
    r"alright)\b[\s,.!]*)+$")
_REMARK_MAX_WORDS = 8


def _clean(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").lower()
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = _LEAD_RE.sub("", t, count=1).strip()
    return _digits(t)


def _clock_arg(h_txt, m_txt, ap, hint) -> Optional[str]:
    h = int(h_txt)
    m = int(m_txt) if m_txt is not None else None
    if m is not None and m > 59:
        return None
    tail = f" {hint}" if hint else ""
    if hint == "tomorrow" and not ap:
        return None                  # "7 tomorrow": morning or evening?
    if ap:
        if not 1 <= h <= 12:
            return None
        return f"{h}:{m or 0:02d} {'am' if ap[0] == 'a' else 'pm'}{tail}"
    if m is None:
        # A bare hour: 1-12 only ("for 20" could be minutes or 20:00).
        return f"at {h}{tail}" if 1 <= h <= 12 else None
    if h > 23:
        return None
    if h == 0 or h > 12 or (len(h_txt) == 2 and h_txt[0] == "0"):
        # "0630", "13 30", "0 30": 24-hour, and a half-of-day word is at odds.
        return None if hint else f"{h:02d}:{m:02d}"
    return f"at {h}:{m:02d}{tail}"


def _time_arg(rest: str, prep: str):
    """(set_alarm argument, characters of ``rest`` used) or (None, 0)."""
    if prep in ("", "for", "in"):
        m = _DUR_RE.match(rest)
        if m and not re.match(r"0+ ", m.group(0)):
            return "in " + m.group(0).replace(" from now", ""), m.end()
    if prep == "in":
        m = _BARE_MIN_RE.match(rest)
        if m and int(m.group(1)) >= 1:
            return f"in {int(m.group(1))} minutes", m.end()
        return None, 0
    m = _WORD_CLOCK_RE.match(rest)
    if m:
        if m.group("hint"):
            return None, 0
        return m.group("w"), m.end()
    m = _CLOCK_RE.match(rest)
    if m:
        return _clock_arg(m.group("h"), m.group("m"), m.group("ap"),
                          m.group("hint")), m.end()
    return None, 0


def _tail_ok(tail: str, verbless: bool) -> bool:
    tail = _POLITE_TAIL_RE.sub("", tail, count=1)
    if not tail.strip(" ,.!"):
        return True
    if tail.strip() == "?":
        return not verbless          # "alarm at 7?" asks; "can you set ...?" tells
    m = re.match(r"^\s*[.!]+\s+(.*)$", tail)
    if not m:
        return False                 # words run on from the time: not ours
    remark = m.group(1).strip()
    return (len(remark.split()) <= _REMARK_MAX_WORDS
            and bool(_REMARK_OK_RE.match(remark)))


_PRE_CLASH_RE = re.compile(_HINT + r"$|^\d\d:|^noon|^midnight|^in ")


def match(text) -> Optional[str]:
    """The set_alarm argument for a plain "set an alarm / wake me" line, else
    None. Never raises."""
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        t = _clean(text)
        m = _HEAD_TIME_FIRST_RE.match(t)
        if m and (m.group("n") or (m.group("verb") and m.group("art"))):
            arg = None
            if m.group("n"):
                if int(m.group("n")) >= 1:
                    arg = f"in {int(m.group('n'))} {m.group('unit')}s"
            else:
                c = _CLOCK_RE.match(m.group("clock"))
                if c and c.end() == len(m.group("clock")):
                    arg = _clock_arg(c.group("h"), c.group("m"),
                                     c.group("ap"), None)
            if arg and _tail_ok(t[m.end():], not m.group("verb")):
                return arg
            return None
        verbless = False
        m = _HEAD_WAKE_RE.match(t)
        if not m:
            m = _HEAD_ALARM_RE.match(t)
            if not m:
                return None
            verbless = not m.group("verb")
            if verbless and m.group("art") in ("the", "my"):
                return None          # "my alarm at 7 ..." is talk about one
        arg, used = _time_arg(t[m.end():], m.group("prep") or "")
        if not arg:
            return None
        pre = m.group("pre")
        if pre:
            # Two half-of-day cues, or one on a duration / 24-hour time.
            has_ap = bool(re.search(r" [ap]m$", arg))
            if _PRE_CLASH_RE.search(arg) or has_ap != (pre == "tomorrow"):
                return None
            arg = f"{arg} {pre}"
        if not _tail_ok(t[m.end() + used:], verbless):
            return None
        return arg
    except Exception:
        return None
