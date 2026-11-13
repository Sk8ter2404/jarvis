"""alarm skill - a wake-up alarm that rings until someone answers it.

Live 2026-11-12 11:32-11:34 the owner asked four times for "an actual alarm
that will wake me up" and got set_timer each time: one spoken line at speech
volume, then silence. A timer tells; an alarm rings.

  set_alarm, <clock time | duration> [| label]   "wake me at 7", "alarm in 20
                                                 minutes"
  list_alarms                                    what is set
  cancel_alarm, [id|all]                         drop a pending alarm
  stop_alarm                                     silence the one ringing
  snooze_alarm, [minutes]                        ring again later (9 min)

When an alarm is due it:
  * moves the default output to the desk speakers when the audio autoswitch
    skill offers that (a headset on the desk wakes nobody),
  * unmutes and raises the master volume to ALARM_VOLUME (full; never lowers it),
  * loops a tone through Windows' own player (winsound), not PortAudio, so a
    device re-enumeration or a busy TTS stream cannot stop or crash it,
  * says a wake-up line over the tone every ALARM_SAY_EVERY_S ("Sir, you
    need to wake up."), naming what the alarm is for when he gave a label,
  * rings until "stop" / "I'm up" / "snooze" is heard, a key or the mouse is
    touched, or ALARM_MAX_RING_S passes, then puts the volume back.

Pending alarms are saved (data/alarms.json) and re-armed at boot; one that
came due while JARVIS was down rings at once if it is under
ALARM_LATE_RING_S late, otherwise it is reported as missed. It cannot ring
while the PC or JARVIS is off: it is not a substitute for a phone alarm on a
morning that matters.

Everything that touches the machine goes through the module-level hooks
(_sound_start, _sound_stop, _volume_read, _volume_write, _input_tick,
_to_speakers) so tests replace them and nothing real is reached.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import threading
import time
import wave
from datetime import datetime, timedelta

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time guard
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# Every reply is a finished sentence about the alarm store: voiced as written.
SPEAK_VERBATIM_ACTIONS = (
    "set_alarm", "list_alarms", "cancel_alarm", "stop_alarm", "snooze_alarm",
)

ALARM_VOLUME = 1.0           # master volume while ringing: he is a heavy sleeper
ALARM_MAX_RING_S = 600.0     # give up after ten minutes with no answer
ALARM_SNOOZE_MIN = 9         # default snooze
ALARM_INPUT_GRACE_S = 3.0    # input in the first seconds does not dismiss
ALARM_LATE_RING_S = 2 * 3600.0
ALARM_SAY_FIRST_S = 2.0      # first spoken line, seconds into the ring
ALARM_SAY_EVERY_S = 18.0     # then one every this many seconds
ALARM_RETRY_GAP_S = 300.0    # no answer for ALARM_MAX_RING_S: quiet this long,
ALARM_MAX_RETRIES = 6        # then ring again, up to this many more times
ALARM_BUSY_RETRY_S = 30.0    # came due while another rings: try again after
ALARM_DUP_S = 2.0            # same label, same moment as the one ringing: a duplicate
ALARM_PREP_MAX_S = 8.0       # speakers + volume get this long; the tone starts anyway
ALARM_VOLUME_RETRY_S = 2.0   # the volume would not go up: try again this often
ALARM_ASKED_WHEN_S = 90.0    # "what time should it ring?" is live this long
ALARM_RESTORE_LATE_S = 20.0  # one already due at start-up waits for the speakers
ALARM_TONE_RETRY_S = 1.0     # the tone would not start: one more try after this
ALARM_TONE_AGAIN_S = 10.0    # still no tone mid-ring: try it again this often
ALARM_STOP_MAX_WORDS = 5     # a bare "stop" counts only in a line this short
# What he says over the tone, in order, then round again from the second.
# None of them may read as an answer to the alarm (ringing_command): the mic
# hears the speakers, and a line with "stop" in it could end its own alarm.
WAKE_LINES = (
    "Sir, it is {clock}. Time to wake up.",
    "Sir, you need to wake up.",
    "I must insist, sir. You asked me to wake you.",
    "Sir. Wake up.",
    "On your feet, sir. The day is not going to wait.",
    "Still asleep, sir? This has been ringing for {minutes}.",
    "Sir, I can keep this up far longer than you can sleep through it.",
)
_ALARMS_FILE_NAME = "alarms.json"
_TONE_FILE_NAME = "alarm_tone.wav"

_lock = threading.Lock()
_persist_lock = threading.Lock()
# id -> {"at": epoch seconds, "label": str, "timer": threading.Timer,
#        "tries": rings already given up on}
_alarms: dict = {}
_next_id = [1]
# The alarm ringing now: {"id", "label", "at", "stop": Event, "why": str} or
# None.
_ringing: list = [None]
_actions_ref: list = [None]
# When set_alarm last asked "What time should it ring" (monotonic), so the
# next bare time can be read as the answer.
_asked_when_at: list = [0.0]


# ── machine hooks (tests replace these) ─────────────────────────────────
def _data_file(name: str) -> str:
    from core import paths as _paths
    return _paths.data_file(name)


def _sound_start(path: str) -> None:
    import winsound
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC
                       | winsound.SND_LOOP | winsound.SND_NODEFAULT)


def _sound_stop() -> None:
    # A None sound stops whatever this process is playing. SND_PURGE is not
    # honoured on current Windows, so it is not what this relies on.
    import winsound
    winsound.PlaySound(None, 0)


def _beep() -> None:
    """One round of the tone through the PC-speaker call: what rings when the
    tone file will not play. Blocks for its length (well under a second)."""
    import winsound
    for freq in (1900, 2400, 1900, 2400):
        winsound.Beep(freq, 160)


def _com_enter() -> None:
    """COM for the ring's own thread. The volume and the speaker switch are
    COM calls, and a timer thread has never initialised it: measured
    2026-11-12, they fail there with "CoInitialize has not been called"
    unless some other thread happens to hold a multithreaded apartment. Not
    undone: the thread ends with the ring."""
    try:
        import comtypes
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except Exception as e:
        print(f"  [alarm] COM not initialised on the ring thread ({e})")


def _endpoint():
    from core import actions as _ca
    return _ca._endpoint_volume()


def _volume_read():
    """(level 0..1, muted) of the default output, or None when unreadable."""
    try:
        vol = _endpoint()
        return (float(vol.GetMasterVolumeLevelScalar()), bool(vol.GetMute()))
    except Exception:
        return None


def _volume_write(level, muted) -> bool:
    try:
        vol = _endpoint()
        vol.SetMute(1 if muted else 0, None)
        vol.SetMasterVolumeLevelScalar(max(0.0, min(1.0, float(level))), None)
        return True
    except Exception:
        return False


def _input_tick():
    """Windows' last-input tick (keyboard or mouse), or None."""
    try:
        import ctypes

        class _LII(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        lii = _LII()
        lii.cbSize = ctypes.sizeof(_LII)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return int(lii.dwTime)
    except Exception:
        pass
    return None


def _to_speakers() -> None:
    """Ask the audio autoswitch skill for the desk speakers, if it is loaded."""
    try:
        fn = (_actions_ref[0] or {}).get("switch_to_speakers")
        if callable(fn):
            print(f"  [alarm] speakers: {fn('')}")
        else:
            print("  [alarm] no speaker switch loaded: ringing on the "
                  "current output")
    except Exception as e:
        print(f"  [alarm] could not move audio to the speakers ({e})")


def _say(message: str, key: str | None = None) -> None:
    try:
        from skills import timer as _timer
        _timer._enqueue_speech(message, key=key)
    except Exception as e:
        print(f"  [alarm] could not queue speech ({e}): {message}")


# ── the tone ─────────────────────────────────────────────────────────────
def _tone_path() -> str:
    """The looped alarm tone, written once: four rising beeps and a gap, so
    a spoken "stop" has quiet to land in."""
    path = _data_file(_TONE_FILE_NAME)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    rate = 22050
    frames = bytearray()

    def beep(freq: float, secs: float) -> None:
        n = int(rate * secs)
        for i in range(n):
            # 8 ms fade at both ends: no click, still sharp.
            edge = min(1.0, i / (rate * 0.008), (n - i) / (rate * 0.008))
            v = 0.9 * edge * math.sin(2.0 * math.pi * freq * i / rate)
            frames.extend(struct.pack("<h", int(v * 32767)))

    def gap(secs: float) -> None:
        frames.extend(b"\x00\x00" * int(rate * secs))

    for freq in (1900.0, 2400.0, 1900.0, 2400.0):
        beep(freq, 0.16)
        gap(0.09)
    gap(1.1)
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    os.replace(tmp, path)
    return path


# ── parsing ──────────────────────────────────────────────────────────────
# Whole unit words only: "6 day shift" and "daily" are not six days.
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b"
# A clock time the way speech-to-text writes it: "422", "4 22", "4.22",
# "1030". Live 2026-11-12 16:21:15: "an alarm for 4:22" arrived as "422", fell
# through to the bare-number-is-minutes rule and was set 422 minutes ahead
# (11:23 PM). Never a number that carries a unit ("100 seconds").
_SPOKEN_CLOCK_RE = re.compile(
    r"(?<![\d:.])(\d{1,2})[ .]?(\d{2})(?![\d:])"
    r"(?!\s*" + _UNIT + r"|\s*[smhd]\b)")
# "4:22" with no am/pm and no leading zero ("07:15" stays 24-hour).
_CLOCK_NO_HALF_RE = re.compile(
    r"(?<![\d:])([1-9]|1[0-2]):([0-5]\d)\b(?!\s*[ap]\.?\s*m)")
_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b")
_H24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
# "at 7", "for 7 o'clock", "by 7", "around 7ish", "7 o'clock": an hour, unless
# a unit follows ("for 10 more minutes", "for 2 and a half hours").
_BARE_HOUR_RE = re.compile(
    r"\b(?:at|for|by|around)\s+(\d{1,2})(?:ish)?(?:\s*o'?clock)?\b"
    r"(?![:.]\d|\s*(?:more\s+|and\s+a\s+half\s+|or\s+so\s+)?" + _UNIT
    + r"|\s*[smhd]\b)"
    r"|(?<![\d:.])(\d{1,2})\s*(?:o'?clock|ish)\b")
# Where the ask itself starts: a time before it is talk ("i have a class at
# 9, wake me at 7").
_ANCHOR_RE = re.compile(
    r"\b(?:wake\s+(?:me\s+)?(?:up\s+)?|alarms?\s+|get\s+me\s+up\s+"
    r"|(?:ring|go\s+off|call)\s+(?:me\s+)?)(?:at|for|by|around)\b"
    r"|\b(?:set|make|give|put)\s+(?:me\s+)?(?:it|one|that|another)\s+"
    r"(?:for|at)\b|\bmake\s+it\b|\balarms?\s+(?=\d)")
# A length of time said before any clock: "in 30 minutes", "for an hour",
# "20 minutes".
_DURATION_FIRST_RE = re.compile(
    r"\b(?:in\s+|for\s+)?(?:about\s+)?(?:(?:an?|half\s+an?|\d+(?:\.\d+)?)\s*)"
    + _UNIT)
# Only today and tomorrow can be named. "on saturday", "in 2 days at 7" used
# to be set for the next time the clock read that hour, the wrong day.
_FAR_DAY_RE = re.compile(
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|\bday after tomorrow\b"
    r"|\bnext week\b|\bin\s+\d+\s+days?\b")

# The half of the day in words, for a time said with no am/pm: "5:30 in the
# morning" (heard live 2026-11-12 16:22:30), "7 tonight".
_MORNING_RE = re.compile(
    r"\b(?:in the|this|tomorrow|every|each|early) morning\b|\bin the a\.?\s?m\b"
    r"|\bmorning alarm\b")
_AFTERNOON_RE = re.compile(r"\b(?:in the|this|tomorrow) (?:afternoon|evening)\b")
_NIGHT_RE = re.compile(r"\b(?:tonight|at night|tomorrow night)\b")
# The same words when they come BEFORE the time ("tomorrow morning at 7",
# "morning alarm at 6"), read at the end of what precedes it.
_MORNING_BEFORE_RE = re.compile(
    r"(?:\b(?:in the|this|tomorrow|every|each|early) morning"
    r"|\bin the a\.?\s?m\.?|(?<!good )\bmorning)$")
_AFTERNOON_BEFORE_RE = re.compile(
    r"\b(?:in the|this|tomorrow) (?:afternoon|evening)$")
_NIGHT_BEFORE_RE = re.compile(r"\b(?:tonight|at night|tomorrow night)$")
_TOMORROW_BEFORE_RE = re.compile(
    r"\btomorrow(?: (?:morning|afternoon|evening|night))?$")
# What may stand between such a word and the time it belongs to.
_LEAD_IN_RE = re.compile(
    r"(?:[,\s]+|\b(?:at|around|by|about|for|alarms?|an?|set|wake|me|up|get)\b)*$")
# Where the words about the time end and other talk begins: "wake me at 7
# pm, i have a class tomorrow at 9" is not tomorrow, "wake me at 6 i have a
# game tonight" is not 6 PM (both measured wrong 2026-11-12).
_CLAUSE_RE = re.compile(
    r"[,.;!?]|\b(?:i|i've|i'm|im|because|so|since|when|before|after|my|we|and"
    r"|then|to|for)\b")
_HINTED_HOUR_RE = re.compile(
    r"(?<![\d:.])(\d{1,2})(?![\d:.])(?!\s*" + _UNIT + r"|\s*[smhd]\b)")

# Number words, as the speech-to-text writes small numbers ("one minute",
# "three minutes" in the 2026-11-12 logs).
_ONES = "one|two|three|four|five|six|seven|eight|nine"
_NUM = {w: i + 1 for i, w in enumerate(_ONES.split("|"))}
_NUM.update({w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen".split())})
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_WORDNUM_RE = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[ -](" + _ONES + r"))?"
    r"|(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|two|three|four|five|six|seven|eight|nine))\b")
# "one" is also a pronoun ("set one for 7"): a number only next to a clock
# or unit word.
_ONE_RE = re.compile(
    r"\b(at|for|past|to|till|after|in)\s+one\b"
    r"|\bone(?=\s+(?:o'?clock|[ap]\.?\s?m\b|oh\b|and\s+a\s+half\b|\d|"
    + _UNIT + r"))")
# "half past 6", "quarter to 7", "10 to 7", "20 minutes before 7".
_RELATIVE_RE = re.compile(
    r"\b(?:(?:a\s+)?(quarter|half)|(5|10|15|20|25)(?:\s+minutes?)?)\s+"
    r"(past|after|to|till|til|before)\s+(\d{1,2})\b(?![:.]?\d|\s*" + _UNIT + ")")


def _normalise(t: str) -> str:
    """The TIME half of the argument (never a label) with number words,
    "half past" forms and machine formats rewritten to what the rules below
    read. Durations the shared timer parser drops a part of are completed."""
    t = re.sub(r"\b\d{4}-\d{2}-\d{2}[t ]?", "", t)                # an ISO date
    # "a.m." -> "am": its full stops are not the end of a clause.
    t = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", t)
    t = re.sub(r"\b(\d{1,2}:\d{2}):\d{2}\b", r"\1", t)            # seconds
    t = re.sub(r"\bthree\s+quarters\s+of\s+an\s+hour\b", "45 minutes", t)
    t = re.sub(r"\b(?:a\s+)?quarter\s+of\s+an\s+hour\b", "15 minutes", t)
    # Tens first: "twenty one minutes" is 21, not "twenty" and "1 minutes".
    t = _WORDNUM_RE.sub(lambda m: str(
        _TENS[m.group(1)] + _NUM.get(m.group(2), 0) if m.group(1)
        else _NUM[m.group(3)]), t)
    t = _ONE_RE.sub(lambda m: (m.group(1) + " 1") if m.group(1) else "1", t)
    t = re.sub(r"\b(\d{4})\s*(?:hours|hrs)\b", r"\1", t)         # "0700 hours"
    t = re.sub(r"(?<![\d-])(\d{1,2})-(\d{2})(?![\d-])", r"\1 \2", t)   # "6-30"
    t = re.sub(r"\b(\d{1,2})\s+(?:oh|o)\s+(\d)\b", r"\1:0\2", t)       # "7 oh 5"

    def rel(m):
        mins = {"quarter": 15, "half": 30}.get(m.group(1)) or int(m.group(2))
        hour = int(m.group(4))
        if not 1 <= hour <= 12:
            return m.group(0)
        if m.group(3) in ("past", "after"):
            return f"{hour}:{mins:02d}"
        return f"{(hour - 2) % 12 + 1}:{60 - mins:02d}"
    t = _RELATIVE_RE.sub(rel, t)
    t = re.sub(r"\b(\d+)\s+and\s+a\s+half\s+(?=" + _UNIT + ")", r"\1.5 ", t)
    t = re.sub(r"\ba\s+couple\s+(?:of\s+)?(?=" + _UNIT + ")", "2 ", t)
    t = re.sub(r"\b(hours?|hrs?)\s+(?:and\s+)?(\d{1,2})\b(?![:.]?\d|\s*"
               + _UNIT + ")", r"\1 \2 minutes", t)
    return t


def _hints_in(s: str) -> tuple:
    """(half of the day, tomorrow, today) named anywhere in ``s``. The half
    is 'am', 'pm', 'night' or ''."""
    half = ("am" if _MORNING_RE.search(s) else
            "pm" if _AFTERNOON_RE.search(s) else
            "night" if _NIGHT_RE.search(s) else "")
    return (half, bool(re.search(r"\btomorrow\b", s)),
            bool(re.search(r"\btoday\b", s)))


def _hint_near(t: str, start: int, end: int) -> tuple:
    """The same three, but only from the words that belong to the time at
    ``t[start:end]``: what follows it up to the next clause, and a hint that
    leads straight into it."""
    after = t[end:end + 40]
    cut = _CLAUSE_RE.search(after)
    half, tomorrow, today = _hints_in(after[:cut.start()] if cut else after)
    head = t[:start]
    head = head[:_LEAD_IN_RE.search(head).start()]
    if not half:
        half = ("am" if _MORNING_BEFORE_RE.search(head) else
                "pm" if _AFTERNOON_BEFORE_RE.search(head) else
                "night" if _NIGHT_BEFORE_RE.search(head) else "")
    tomorrow = tomorrow or bool(_TOMORROW_BEFORE_RE.search(head))
    today = today or bool(re.search(r"\btoday$", head))
    return half, tomorrow, today


def _hours(hour: int, half: str = "", tomorrow: bool = False) -> tuple:
    """The 24-hour hours a 12-hour ``hour`` may mean: one when the half of
    the day is known, else both (the sooner one is used)."""
    h = hour % 12
    if half == "am":
        return (h,)
    if half == "pm":
        return (h + 12,)
    if half == "night":
        # "12 at night", "2 at night" are after midnight; "9 tonight" is not.
        return (h,) if hour in (12, 1, 2, 3, 4) else (h + 12,)
    if tomorrow:
        # "7 tomorrow", for an alarm, is the morning; "12 tomorrow" is noon.
        return (12,) if hour == 12 else (h,)
    return (h, h + 12)


def _half_hours(t: str, hour: int) -> tuple:
    """_hours for a text that is nothing but one time ("the 7 am one")."""
    half, tomorrow, _today = _hints_in(t)
    return _hours(hour, half, tomorrow)


def _spoken_clock(t: str) -> str:
    """``t`` with a spoken clock time rewritten as H:MM ("422" -> "4:22").
    Unchanged when the number follows "in" (a duration) or is no clock time."""
    t = re.sub(r"\bmidnight\b", "12:00 am", t)
    t = re.sub(r"\bnoon\b", "12:00 pm", t)
    m = _SPOKEN_CLOCK_RE.search(t)
    if not m or re.search(r"\bin\s+$", t[:m.start()]):
        return t
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return t
    return f"{t[:m.start()]}{m.group(1)}:{minute:02d}{t[m.end():]}"


def _until_clock(minute: int, *hours: int) -> int:
    """Seconds until the clock next reads any of ``hours``:``minute``. Real
    timestamps, so a daylight-saving night still rings at the hour asked;
    rounded UP, so now + this never lands in the minute before (a 7:00 alarm
    was confirmed as "6:59")."""
    now_ts = time.time()
    now = datetime.fromtimestamp(now_ts)
    best = None
    for h in hours:
        target = now.replace(hour=h, minute=minute, second=0, microsecond=0)
        delta = target.timestamp() - now_ts
        if delta <= 0:
            delta = (target + timedelta(days=1)).timestamp() - now_ts
        best = delta if best is None else min(best, delta)
    return int(math.ceil(best - 1e-6))


def _tomorrow_shift(secs: int) -> int:
    """"7 am tomorrow" said at 6 AM is not fifty minutes away: when the time
    found is still today, it is the same clock time a day later. Before 4 AM
    "tomorrow" is the morning that is coming, so nothing moves. Wall-clock
    arithmetic, so a daylight-saving night keeps the hour."""
    now_ts = time.time()
    now = datetime.fromtimestamp(now_ts)
    target = datetime.fromtimestamp(now_ts + secs)
    if now.hour >= 4 and target.date() == now.date():
        secs = int(math.ceil((target + timedelta(days=1)).timestamp()
                             - now_ts - 1e-6))
    return secs


def _parse_when(text: str):
    """Seconds from now for 'at 7', '7:30 am', '422', 'seven thirty', 'half
    past six', 'in 20 minutes', '6 tomorrow'. None when no time is named, or
    when a day other than today or tomorrow is.

    The FIRST thing in the ask that reads as a time wins ("7 am for the 9 30
    lecture" is 7 AM), starting at "wake me at" / "alarm for" when the line
    has them. A length of time said first ("in 30 minutes, before my 9 am
    class") is that length of time, and "20 minutes before my 8 am class" is
    7:40. "in the morning", "tonight" and "tomorrow" count only next to the
    time they belong to."""
    from skills import timer as _timer
    t = _spoken_clock(_normalise((text or "").strip().lower()))
    if not t or _FAR_DAY_RE.search(t):
        return None
    a = _ANCHOR_RE.search(t)
    pos = a.start() if a else 0
    cands = []            # (where its hour starts, rule, minute, hours, tomorrow)

    def add(start, end, rule, minute, hour, fixed=None):
        half, tomorrow, _today = _hint_near(t, start, end)
        hours = fixed if fixed is not None else _hours(hour, half, tomorrow)
        if all(0 <= h <= 23 for h in hours) and minute <= 59:
            cands.append((start, rule, minute, hours, tomorrow))

    m = _AMPM_RE.search(t, pos)
    if m:
        add(m.start(1), m.end(), 0, int(m.group(2) or 0), 0,
            (int(m.group(1)) % 12 + (12 if m.group(3) == "p" else 0),))
    m = _CLOCK_NO_HALF_RE.search(t, pos)
    if m:
        add(m.start(1), m.end(), 1, int(m.group(2)), int(m.group(1)))
    m = _H24_RE.search(t, pos)
    if m:
        add(m.start(1), m.end(), 2, int(m.group(2)), 0, (int(m.group(1)),))
    m = _BARE_HOUR_RE.search(t, pos)
    if m:
        g = 1 if m.group(1) else 2
        if 1 <= int(m.group(g)) <= 12:
            add(m.start(g), m.end(), 3, 0, int(m.group(g)))
    else:
        # "7 in the morning", "9 tonight", "7 tomorrow": the hint beside it
        # makes a bare number an hour, unless it follows "in" (a duration).
        for m in _HINTED_HOUR_RE.finditer(t, pos):
            if re.search(r"\bin\s+$", t[:m.start()]) or \
                    not 1 <= int(m.group(1)) <= 12:
                continue
            if any(_hint_near(t, m.start(1), m.end())):
                add(m.start(1), m.end(), 3, 0, int(m.group(1)))
                break
    first = min(cands) if cands else None
    d = _DURATION_FIRST_RE.search(t, pos)
    if d and (first is None or d.start() < first[0]):
        seg = t[d.start():first[0]] if first else t[d.start():]
        rel = re.search(r"\s+(before|after)\s+(?:(?:my|the|our|that)\s+)?$", seg)
        if first and rel:
            # "20 minutes before my 8 am class": the clock, less the span.
            span = _timer._parse_duration(seg[:rel.start()])
            clock = _until_clock(first[2], *first[3])
            if span is not None and span > 0:
                secs = clock - span if rel.group(1) == "before" else clock + span
                return int(secs if secs > 0 else secs + 86400)
        else:
            secs = _timer._parse_duration(seg)
            if secs is not None and secs > 0:
                return int(secs)
    if first:
        secs = _until_clock(first[2], *first[3])
        return _tomorrow_shift(secs) if first[4] else secs
    secs = _timer._parse_duration(t)
    if secs is not None and secs > 0:
        return int(secs)
    return None


# Words a time is made of. A label half that is nothing else is the time,
# said the other way round ("gym | 6:30").
_TIME_ONLY_RE = re.compile(
    r"\b(?:at|in|for|by|around|about|am|pm|a\.?m\.?|p\.?m\.?|o'?clock|minutes?"
    r"|mins?|hours?|hrs?|seconds?|secs?|tonight|tomorrow|today|morning"
    r"|afternoon|evening|night|noon|midnight|half|quarter|past|to|an?|the"
    r"|and|this)\b|[\d:.\s]")


def _split_args(args: str):
    """(seconds, label) from "<time>" or "<time> | <label>". Only the time
    half is ever rewritten. "gym | 6:30" is read the other way round, but a
    digit inside a real label ("lunch | take 2 pills") is never the time."""
    raw = (args or "").strip()
    when, _, label = raw.partition("|")
    if not label.strip():
        return _parse_when(when), ""
    secs = _parse_when(when)
    if secs is None and not re.search(r"\d", when) and \
            not _TIME_ONLY_RE.sub("", label.lower()).strip():
        other = _parse_when(label)
        if other is not None:
            return other, when.strip()
    return secs, label.strip()


def _clock(epoch: float) -> str:
    d = datetime.fromtimestamp(epoch)
    hour = d.hour % 12 or 12
    return f"{hour}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'}"


def _clock_day(epoch: float) -> str:
    """The clock time, with the day when it is not today: "7:00 AM
    tomorrow", "7:00 AM on Saturday"."""
    days = (datetime.fromtimestamp(epoch).date()
            - datetime.fromtimestamp(time.time()).date()).days
    if days == 1:
        return f"{_clock(epoch)} tomorrow"
    if days > 1:
        return f"{_clock(epoch)} on {datetime.fromtimestamp(epoch):%A}"
    return _clock(epoch)


def _span(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 90:
        return f"{secs} seconds"
    mins = round(secs / 60)
    if mins < 90:
        return f"{mins} minutes"
    hours, rem = divmod(mins, 60)
    return (f"{hours} hour{'' if hours == 1 else 's'}"
            + (f" {rem} minutes" if rem else ""))


# ── persistence ──────────────────────────────────────────────────────────
def _snapshot() -> list:
    with _lock:
        return [{"id": aid, "at": a["at"], "label": a["label"]}
                for aid, a in sorted(_alarms.items())]


def _persist() -> None:
    """Save the pending alarms and the one ringing now. An alarm used to
    leave the file the moment it came due, so a crash or a restart mid-ring
    lost it; now it stays until it is answered and rings again at the next
    start. The rows are read inside the file lock, so two threads saving at
    once cannot write the older picture last."""
    try:
        with _persist_lock:
            with _lock:
                rows = [{"id": aid, "at": a["at"], "label": a["label"]}
                        for aid, a in sorted(_alarms.items())]
                me = _ringing[0]
                if me is not None and me.get("at") is not None \
                        and not me["why"]:
                    rows.append({"id": me["id"], "at": me["at"],
                                 "label": me["label"]})
            _atomic_write_json(_data_file(_ALARMS_FILE_NAME), rows)
    except Exception as e:
        print(f"  [alarm] could not save alarms ({e})")


def restore_persisted_alarms() -> int:
    """Re-arm the alarms the previous process saved. Returns how many."""
    try:
        with open(_data_file(_ALARMS_FILE_NAME), "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0
    except Exception as e:
        print(f"  [alarm] could not read saved alarms ({e})")
        return 0
    n = 0
    still = []
    now = time.time()
    for row in data if isinstance(data, list) else []:
        try:
            at = float(row["at"])
            label = str(row.get("label") or "")
        except Exception:
            continue
        if not math.isfinite(at) or at < 0 or at > now + 366 * 86400.0:
            continue                      # a damaged row never rings
        late = now - at
        if late > ALARM_LATE_RING_S:
            print(f"  [alarm] missed while I was off: {_clock(at)} {label}".rstrip())
            _say(f"Sir, I was not running at {_clock(at)}, so that alarm "
                 "never rang.", key=f"alarm-missed#{int(at)}")
            continue
        wait = at - now
        if wait < ALARM_RESTORE_LATE_S:
            # Already due. This runs while the skills are still loading: the
            # speaker switch and the speech queue are not there yet (live
            # 2026-11-12 the switch registered 8 s after this skill).
            wait = ALARM_RESTORE_LATE_S
        else:
            still.append(at)
        _arm(wait, label, at=at)
        n += 1
    _persist()
    if still:
        first = min(still)
        _say(f"Your {_clock_day(first)} alarm is still set, sir."
             if len(still) == 1 else
             f"Your {len(still)} alarms are still set, sir. The next is "
             f"{_clock_day(first)}.", key=f"alarm-kept#{int(first)}")
    return n


# ── arming and ringing ───────────────────────────────────────────────────
def _arm(secs: float, label: str, at: float | None = None,
         tries: int = 0) -> int:
    with _lock:
        aid = _next_id[0]
        _next_id[0] += 1
    fire_at = float(at) if at is not None else time.time() + float(secs)
    t = threading.Timer(max(0.0, float(secs)), _fire, args=(aid,))
    t.daemon = True
    with _lock:
        _alarms[aid] = {"at": fire_at, "label": label, "timer": t,
                        "tries": int(tries)}
    t.start()
    return aid


def _fire(aid: int) -> None:
    with _lock:
        a = _alarms.pop(aid, None)
    if a is None:
        return
    tries = int(a.get("tries") or 0)
    try:
        ring(aid, a["label"], at=a["at"], tries=tries)
    except Exception as e:
        # ring() is written never to raise. If it does, the alarm must not
        # vanish and must not leave "ringing" set (which drops every later
        # alarm): free the slot and try again shortly, a bounded number of
        # times.
        print(f"  [alarm] #{aid} could not ring ({type(e).__name__}: {e})")
        with _lock:
            if _ringing[0] is not None and _ringing[0]["id"] == aid:
                _ringing[0] = None
        if tries < ALARM_MAX_RETRIES:
            _arm(ALARM_BUSY_RETRY_S, a["label"], at=a["at"], tries=tries + 1)
        _persist()


def is_ringing() -> bool:
    return _ringing[0] is not None


def _ring_while_busy(aid: int, label: str, at, tries: int, cur: dict) -> None:
    """An alarm came due while another rings. It used to be dropped without
    a word, so a 7:05 backup behind a still-ringing 7:00 never rang. A true
    duplicate (same label, same minute) is one alarm; anything else waits
    and tries again."""
    if (label == cur.get("label") and at is not None
            and cur.get("at") is not None
            and abs(float(at) - float(cur["at"])) < ALARM_DUP_S):
        print(f"  [alarm] #{aid} is a duplicate of #{cur['id']}, which is "
              "ringing")
    else:
        print(f"  [alarm] #{aid} came due while #{cur['id']} rings; trying "
              f"again in {int(ALARM_BUSY_RETRY_S)} s")
        _arm(ALARM_BUSY_RETRY_S, label, at=at, tries=tries)
    _persist()


def _start_tone() -> bool:
    """Start the looping tone. One more try after a failure; False when it
    will not play at all (the caller then beeps instead)."""
    for attempt in (0, 1):
        try:
            _sound_start(_tone_path())
            return True
        except Exception as e:
            print(f"  [alarm] the tone would not start ({e})")
            if attempt == 0:
                time.sleep(ALARM_TONE_RETRY_S)
    return False


def _prepare(box: dict) -> None:
    """Speakers and volume for a ring, on a thread of its own (see ring)."""
    try:
        _com_enter()
        _to_speakers()
        box["before"] = _volume_read()
        level = ALARM_VOLUME if box["before"] is None \
            else max(box["before"][0], ALARM_VOLUME)
        # Raised even when the level could not be read: a quiet or muted PC
        # is the one thing an alarm cannot have.
        box["raised"] = bool(_volume_write(level, False))
        print(f"  [alarm] volume {box['before']} -> {level:.2f}, unmuted "
              f"({'done' if box['raised'] else 'COULD NOT be set'})")
    except Exception as e:
        print(f"  [alarm] speakers / volume step failed ({e})")
    finally:
        box["done"] = True


def ring(aid: int, label: str = "", at: float | None = None,
         tries: int = 0) -> None:
    """Ring until answered. Blocks its own thread; never raises."""
    stop = threading.Event()
    me = {"id": aid, "label": label, "at": at, "stop": stop, "why": "",
          "since": time.monotonic()}
    with _lock:
        cur = _ringing[0]
        if cur is not None and (time.monotonic() - cur.get("since", 0.0)
                                > ALARM_MAX_RING_S + 120.0):
            # A ring that never ended (a hung audio call): it must not keep
            # every later alarm from ringing.
            print(f"  [alarm] #{cur['id']} never finished; #{aid} takes over")
            cur["stop"].set()
            cur = None
        if cur is None:
            _ringing[0] = me
    if cur is not None:
        _ring_while_busy(aid, label, at, tries, cur)
        return
    box = {"before": None, "raised": False, "done": False}
    try:
        print(f"  [alarm] \u23f0 ringing #{aid} {label}".rstrip())
        _persist()
        # The speaker switch and the volume are calls into the audio service
        # with no time limit of their own. They get a few seconds on their
        # own thread; the tone starts whether or not they have come back.
        prep = threading.Thread(target=_prepare, args=(box,), daemon=True)
        prep.start()
        prep.join(ALARM_PREP_MAX_S)
        if not box["done"]:
            print("  [alarm] speakers / volume still not done; ringing anyway")
        _com_enter()
        tone = _start_tone()
        if not tone:
            print("  [alarm] ringing with beeps instead of the tone")
        started = time.monotonic()
        tick0 = None
        said = 0
        say_at = ALARM_SAY_FIRST_S
        vol_at = ALARM_VOLUME_RETRY_S
        tone_at = ALARM_TONE_AGAIN_S
        beeps = True
        while not stop.wait(0.25):
            age = time.monotonic() - started
            if age >= ALARM_MAX_RING_S:
                with _lock:
                    me["why"] = me["why"] or "timeout"
                break
            if age >= say_at:
                _say(wake_line(said, age, label), key=f"alarm-wake#{aid}.{said}")
                said += 1
                say_at = age + ALARM_SAY_EVERY_S
            if box["done"] and not box["raised"] and age >= vol_at:
                # One failed write must not mean ten silent minutes.
                if box["before"] is None:
                    box["before"] = _volume_read()
                box["raised"] = bool(_volume_write(
                    ALARM_VOLUME if box["before"] is None
                    else max(box["before"][0], ALARM_VOLUME), False))
                vol_at = age + ALARM_VOLUME_RETRY_S
            tick = _input_tick()
            if age < ALARM_INPUT_GRACE_S or tick0 is None:
                tick0 = tick
            elif tick is not None and tick != tick0:
                with _lock:
                    me["why"] = me["why"] or "input"
                break
            if not tone:
                # No sound at all is not a reason to stop: the spoken lines
                # go out another way, and the tone is tried again.
                if beeps:
                    try:
                        _beep()
                    except Exception as e:
                        beeps = False
                        print(f"  [alarm] the beep failed too ({e})")
                if age >= tone_at:
                    tone_at = age + ALARM_TONE_AGAIN_S
                    try:
                        _sound_start(_tone_path())
                        tone = True
                        print("  [alarm] the tone started on a later try")
                    except Exception:
                        beeps = True
    except Exception as e:
        with _lock:
            me["why"] = me["why"] or f"error: {type(e).__name__}: {e}"
        print(f"  [alarm] ring failed ({e})")
    finally:
        try:
            _sound_stop()
        except Exception as e:
            print(f"  [alarm] could not stop the tone ({e})")
        if box["before"] is not None:
            _volume_write(box["before"][0], box["before"][1])
        with _lock:
            if _ringing[0] is me:
                _ringing[0] = None
    try:
        _after_ring(aid, label, me["why"] or "voice", tries, at)
    except Exception as e:
        print(f"  [alarm] after-ring step failed ({e})")
    _persist()


def _after_ring(aid: int, label: str, why: str, tries: int,
                at: float | None = None) -> None:
    print(f"  [alarm] #{aid} stopped ({why})")
    mins = int(ALARM_MAX_RING_S // 60)
    if why == "timeout" and tries < ALARM_MAX_RETRIES:
        # A heavy sleeper who slept through it gets another go, not silence
        # for the rest of the morning.
        gap = max(1, int(round(ALARM_RETRY_GAP_S / 60.0)))
        _arm(ALARM_RETRY_GAP_S, label, tries=tries + 1)
        _say(f"Sir, no answer after {mins} minutes. I will ring again in "
             f"{gap} minute{'' if gap == 1 else 's'}.",
             key=f"alarm-timeout#{aid}")
    elif why == "timeout":
        _say(f"Sir, your alarm rang {tries + 1} times for {mins} minutes "
             "with no answer, so I have stopped it.",
             key=f"alarm-timeout#{aid}")
    elif why == "input":
        _say("Alarm off, sir.", key=f"alarm-off#{aid}")
    elif why.startswith("error") and tries < ALARM_MAX_RETRIES:
        # Neither the tone nor the beeps would sound (the audio stack was not
        # ready). One spoken line needs that same audio: try the ring again.
        _arm(ALARM_BUSY_RETRY_S, label, at=at, tries=tries + 1)
    elif why.startswith("error"):
        _say("Sir, your alarm is due, and I could not ring it.",
             key=f"alarm-error#{aid}")


def wake_line(n: int, age_s: float, label: str = "", now: float | None = None) -> str:
    """The n-th line (0-based) spoken over the tone, ``age_s`` into the ring."""
    if n < len(WAKE_LINES):
        line = WAKE_LINES[n]
    else:
        line = WAKE_LINES[1 + (n - 1) % (len(WAKE_LINES) - 1)]
    mins = max(1, int(round(age_s / 60.0)))
    line = line.format(clock=_clock(time.time() if now is None else now),
                       minutes="a minute" if mins == 1 else f"{mins} minutes")
    # A label that reads as an answer ("stop smoking") is not said aloud: the
    # mic hears the speakers, and it would end its own alarm.
    if n == 0 and label and not (_STOP_RE.search(label.lower())
                                 or _reads_as_answer(f"this for {label}")):
        line += f" You set this for {label}."
    return line


def _stop_ringing(why: str) -> bool:
    with _lock:
        me = _ringing[0]
        if me is None:
            return False
        me["why"] = why
    me["stop"].set()
    return True


# ── what he says while it rings ──────────────────────────────────────────
_SNOOZE_RE = re.compile(
    r"\bsnooze\b|\b(?:five|ten|a few|\d{1,2}) more minutes\b")
# Any word that could read as an answer (what a spoken label must not hold).
_STOP_RE = re.compile(
    r"\b(?:stop|off|quiet|enough|shut up|silence|cancel|dismiss)\b"
    r"|\bi(?:'m| am|m) (?:up|awake|getting up)\b|\bok(?:ay)? ok(?:ay)?\b")
# Plainly about the alarm or about being awake: an answer at any length.
_STOP_PLAIN_RE = re.compile(
    r"\bi(?:'m| am|m) (?:up|awake|getting up)\b|\bok(?:ay)? ok(?:ay)?\b"
    r"|\bshut up\b"
    r"|\b(?:stop|cancel|silence|dismiss|kill|turn off|shut off) "
    r"(?:the |that |this |my )?alarm\b"
    r"|\balarm off\b|\b(?:turn|shut|switch) (?:it|that|the alarm) off\b")
# One word is an answer only in a short line, or one that names him or the
# alarm. The gate lets an answer through with no wake word, and measured
# 2026-11-12 "the tv is off" and a video's "... and don't stop" both ended a
# ring.
_STOP_WORD_RE = re.compile(r"\b(?:stop|quiet|enough)\b")
_STOP_SHORT_RE = re.compile(r"\b(?:off|cancel|silence|dismiss)\b")


def _reads_as_answer(text: str) -> str:
    """'stop' / 'snooze' / '' for ``text``, were an alarm ringing."""
    # Punctuation goes: speech-to-text writes "Okay, okay." and "Stop! Stop!".
    t = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    # "set an alarm to go off at 7" is not "off".
    t = re.sub(r"\bgo(?:es|ing)? off\b", " ", t)
    # "... and don't stop" (a video's sign-off) is not "stop".
    t = " ".join(re.sub(
        r"\b(?:don't|dont|do not|never|not|can't|won't) (?:ever )?stop\b",
        " ", t).split())
    if _SNOOZE_RE.search(t):
        return "snooze"
    if _STOP_PLAIN_RE.search(t):
        return "stop"
    named = bool(re.search(r"\b(?:jarvis|alarm)\b", t))
    words = t.split()
    # "stop, stop, stop, stop!" is short however often he says it.
    others = len([w for w in words if not _STOP_WORD_RE.fullmatch(w)])
    if _STOP_WORD_RE.search(t) and (named or others <= ALARM_STOP_MAX_WORDS):
        return "stop"
    if _STOP_SHORT_RE.search(t) and (named or len(words) <= 3):
        return "stop"
    return ""


def ringing_command(text: str) -> str:
    """'stop' / 'snooze' / '' for a line heard while an alarm rings."""
    if not is_ringing():
        return ""
    return _reads_as_answer(text)


def handle_ringing(text: str):
    """Act on a line heard while ringing. The spoken reply, or None when the
    line is not about the alarm (or nothing rings)."""
    cmd = ringing_command(text)
    if cmd == "snooze":
        return snooze("")
    if cmd == "stop":
        return stop_alarm("")
    return None


# ── actions ──────────────────────────────────────────────────────────────
def asked_when_recently() -> bool:
    """True while set_alarm's own "What time should it ring" is unanswered
    and fresh: the next bare time is the answer."""
    return 0.0 < time.monotonic() - _asked_when_at[0] < ALARM_ASKED_WHEN_S


def set_alarm(args: str = "") -> str:
    secs, label = _split_args(args)
    if secs is None or secs <= 0:
        if _FAR_DAY_RE.search((args or "").lower()):
            return ("I can only set an alarm for today or tomorrow, sir. "
                    "Nothing is set.")
        _asked_when_at[0] = time.monotonic()
        # A question, with no failure word in it, so it is said as written.
        return ("What time should it ring, sir? Say 'at 7 AM' or "
                "'in 20 minutes'.")
    _asked_when_at[0] = 0.0
    at = time.time() + secs
    with _lock:
        same = next((a for a in _alarms.values()
                     if a["label"] == label and abs(a["at"] - at) < 60.0), None)
    if same is not None:
        # The same ask heard twice is one alarm.
        return f"That alarm is already set for {_clock_day(same['at'])}, sir."
    aid = _arm(secs, label)
    _persist()
    tail = f" for {label}" if label else ""
    return (f"Alarm #{aid} set for {_clock_day(at)}{tail}, {_span(secs)} "
            "from now, sir. It rings until you say stop or touch the "
            "keyboard.")


def list_alarms(_: str = "") -> str:
    rows = _snapshot()
    with _lock:
        me = _ringing[0]
    now = (f"Alarm #{me['id']} is ringing now, sir. "
           if me is not None and not me["why"] else "")
    if not rows:
        return (now + "Nothing else is set.") if now else "No alarms are set, sir."
    parts = [f"#{r['id']} at {_clock_day(r['at'])}"
             + (f" for {r['label']}" if r["label"] else "") for r in rows]
    return now + ("One alarm, sir: " if len(parts) == 1
                  else f"{len(parts)} alarms, sir: ") + ", ".join(parts) + "."


# "all", "both of them", "every one", "all my alarms" - and nothing else: a
# label with "all" in it ("the all hands one") is not everything.
_CANCEL_ALL_RE = re.compile(
    r"(?:all|both|every(?:thing| one)?)(?: of)?(?: my| the| them| those)?"
    r"(?: alarms?)?")
_CANCEL_ID_RE = re.compile(r"(?:the\s+)?(?:alarm\s*)?(?:#|number\s*)?(\d{1,3})")
_CANCEL_NEXT_RE = re.compile(
    r"(?:(?:the|my|that|this)\s+)?(?:next\s+)?(?:alarm|one|it|that)?")


def _clock_targets(low: str):
    """The (hour, minute) pairs a clock time in ``low`` may mean, or None
    when it names none. "7 am" is one pair, "6:30" two (either half)."""
    t = _spoken_clock(_normalise(low))
    m = _AMPM_RE.search(t)
    if m:
        return {(int(m.group(1)) % 12 + (12 if m.group(3) == "p" else 0),
                 int(m.group(2) or 0))}
    m = _CLOCK_NO_HALF_RE.search(t)
    if m:
        return {(h, int(m.group(2))) for h in _half_hours(t, int(m.group(1)))}
    m = _H24_RE.search(t)
    if m:
        return {(int(m.group(1)), int(m.group(2)))}
    m = re.search(r"\b(\d{1,2})\s*o'?clock\b", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return {(h, 0) for h in _half_hours(t, int(m.group(1)))}
    return None


def _hm(epoch) -> tuple:
    d = datetime.fromtimestamp(epoch)
    return (d.hour, d.minute)


def cancel_alarm(args: str = "") -> str:
    """No argument: the next one due (or, while one rings, the one ringing).
    "all" / "both": every one. A bare number: that alarm's #. A clock time
    ("7 am", "the 6:30 one"): the alarm set for it - it used to be read as
    an id, so "cancel the 7 am alarm" answered "There is no alarm #7". A
    label ("gym"): the alarm with it. Anything else cancels NOTHING and asks
    which: a wake-up alarm must not go because the brain passed odd words."""
    low = (args or "").strip().lower().rstrip(".!?")
    norm = _normalise(low)
    want_all = bool(_CANCEL_ALL_RE.fullmatch(low))
    by_id = None if want_all else _CANCEL_ID_RE.fullmatch(norm)
    targets = None if want_all else _clock_targets(low)
    plain = bool(_CANCEL_NEXT_RE.fullmatch(low))
    with _lock:
        me = _ringing[0]
        if by_id and int(by_id.group(1)) in _alarms:
            targets = None                 # a live id is an id
        elif by_id and targets is None:
            if me is not None and me["id"] == int(by_id.group(1)):
                pass                       # the one ringing, by number
            elif _alarms:
                return f"There is no alarm #{int(by_id.group(1))}, sir."
        if me is not None and not want_all and (
                plain or (targets and me.get("at") is not None
                          and _hm(me["at"]) in targets)
                or (by_id and me["id"] == int(by_id.group(1)))):
            ring_only = True
        else:
            ring_only = False
        ids = []
        if ring_only or not _alarms:
            pass
        elif want_all:
            ids = list(_alarms)
        elif targets:
            hits = [i for i, a in _alarms.items() if _hm(a["at"]) in targets]
            if not hits:
                return "There is no alarm at that time, sir."
            # "7:00" with a 7 AM and a 7 PM set: the next one, not both.
            ids = [min(hits, key=lambda i: _alarms[i]["at"])]
        elif by_id:
            ids = [int(by_id.group(1))]
        elif plain:
            ids = [min(_alarms, key=lambda k: _alarms[k]["at"])]
        else:
            key = " ".join(re.sub(r"\b(?:the|my|that|this|one|alarm)\b", " ",
                                  low).split())
            ids = [i for i, a in _alarms.items() if key and a["label"] and (
                key in a["label"].lower() or a["label"].lower() in key)][:1]
            if not ids:
                return "Which alarm, sir? Say its time or its number."
        gone = [(i, _alarms.pop(i)) for i in ids]
        left = sorted(a["at"] for a in _alarms.values())
    for _i, a in gone:
        a["timer"].cancel()
    stopped = (want_all or ring_only) and _stop_ringing("cancelled")
    if not gone:
        return "Alarm stopped, sir." if stopped else "No alarms are set, sir."
    _persist()
    if len(gone) > 1:
        return f"Cancelled {len(gone)} alarms, sir."
    msg = f"Cancelled the {_clock_day(gone[0][1]['at'])} alarm, sir."
    if len(left) == 1:
        msg += f" The {_clock_day(left[0])} one is still set."
    elif left:
        msg += f" {len(left)} others are still set."
    return msg


def stop_alarm(_: str = "") -> str:
    if _stop_ringing("voice"):
        return "Alarm off, sir."
    return "No alarm is ringing, sir."


def snooze(args: str = "") -> str:
    m = re.search(r"\d+", args or "")
    mins = int(m.group(0)) if m else ALARM_SNOOZE_MIN
    mins = max(1, min(60, mins))
    with _lock:
        me = _ringing[0]
        label = me["label"] if me else ""
    if not _stop_ringing("snooze"):
        return "No alarm is ringing, sir."
    _arm(mins * 60.0, label)
    _persist()
    return f"Snoozed, sir. It rings again in {mins} minutes."


def register(actions):
    _actions_ref[0] = actions
    actions["set_alarm"] = set_alarm
    actions["list_alarms"] = list_alarms
    actions["cancel_alarm"] = cancel_alarm
    actions["stop_alarm"] = stop_alarm
    actions["snooze_alarm"] = snooze
    try:
        n = restore_persisted_alarms()
        if n:
            print(f"  [alarm] re-armed {n} saved alarm(s)")
    except Exception as e:
        print(f"  [alarm] restore failed ({e})")
