"""What the proactive speech queue may still say, and how old it may be
(v3.0.11, PENDING_SOURCE_POLICY / PENDING_GREETING_FIRST).

THE INCIDENT THIS EXISTS FOR (live 2026-11-05, boot 10:52). pending_speech.json
survives a restart, and nothing limited how old a queued line could be. The
last run stopped on 2026-10-29 at 22:46 with the owner away, so the evening
briefing and the recap queued at 22:30, a clap acknowledgement from 22:42 and a
microphone change from 22:45 all waited on disk for seven days. At 10:54 he came
back and heard, in the morning: "today played 1 tracks", "Good evening, sir. A
quiet day ..." with a week-old forecast and headlines, then "While you were
away, sir: You rang; and Switched to the Blue Snowball." His real greeting,
queued last, came after all of it. No scheduled job misfired.

THE RULE. One table, QUEUE_RULES: a queued line belongs to a class; the class
says how old the line may be and whether it may be folded into the "While you
were away" recap. A line past its age is dropped with a log line.

His own reminders (core/owner_presence.EXEMPT_SOURCES) and urgent lines are in
no class: they keep their value however late they are, as before. An entry with
no usable 'ts' counts as fresh, as in plan_return_drain.

Pure logic with an injectable clock and hour; stdlib only; nothing here raises.
"""
from __future__ import annotations

import re
import time

from core.owner_presence import EXEMPT_SOURCES, is_urgent, source_root

_HOUR = 3600.0

# class: (how old the line may be in seconds, may it be folded into a recap)
QUEUE_RULES = {
    # "You rang, sir?" answers a clap. Live 11-05 11:06:55 and 11:26:48 it came
    # 96 s and 67 s late, after room talk had held the queue.
    "clap_ack":   (10.0, False),
    # "Switched to the Blue Snowball" is news for a moment, and read as a
    # fragment of a recap it says nothing.
    "mic_change": (300.0, False),
    # A RAM figure a few minutes old is not true any more; a line that is still
    # true comes round again on the alert's own cool-down. Live 11-05 12:55:14
    # and 14:44:19 a recap read out figures that had waited.
    "ram":        (180.0, False),
    # A briefing or digest is about the day it was built on: its "today", its
    # forecast and its headlines. Past this it is not read out, and it is never
    # cut to a fragment.
    "briefing":   (6 * _HOUR, False),
    # Anything else that waited this long is not worth a recap fragment.
    "other":      (24 * _HOUR, True),
}

# Queued sources that are a briefing or a digest (before any ':'). The presence
# recap ("recap:presence") is built at drain time, so its 'ts' is always fresh.
BRIEFING_SOURCES = frozenset({
    "morning", "arrival", "arrival_v2", "evening", "daily", "news", "recap",
    "handoff", "weekly_digest_briefing", "anticipation_briefing",
})

# "Good evening" before this hour is wrong however it got there; a line younger
# than EVENING_LINE_FRESH_S is one he has just asked for by name, which still
# speaks. Small hours are left to the age limit: 00:30 is still his evening.
EVENING_FROM_HOUR = 17
EVENING_UNTIL_HOUR = 4
EVENING_LINE_FRESH_S = 120.0

_TAGS_RE = re.compile(r"^(?:\s*\[[a-z_]+(?::[^\]]*)?\]\s*)+", re.IGNORECASE)
_CLAP_ACK_RE = re.compile(r"^you rang\b", re.IGNORECASE)
_MIC_CHANGE_RE = re.compile(r"^(?:switched to\b|headset (?:on|off)\b)",
                            re.IGNORECASE)
_RAM_RE = re.compile(r"\bmemory usage is at\b", re.IGNORECASE)
_GREETING_RE = re.compile(r"^good (?:morning|afternoon|evening)\b",
                          re.IGNORECASE)
_EVENING_RE = re.compile(r"^good evening\b", re.IGNORECASE)


def _spoken_text(entry) -> str:
    try:
        return _TAGS_RE.sub("", str(entry.get("message") or "")).strip()
    except Exception:
        return ""


def _exempt(entry) -> bool:
    return source_root(entry) in EXEMPT_SOURCES or is_urgent(entry)


def line_class(entry) -> str:
    """The QUEUE_RULES class of a queued entry; '' when it is his own or
    urgent (no rule applies) or not an entry at all."""
    try:
        if not isinstance(entry, dict) or _exempt(entry):
            return ""
        root = source_root(entry)
        text = _spoken_text(entry)
        if root == "clap" and _CLAP_ACK_RE.match(text):
            return "clap_ack"
        if root == "audio" and _MIC_CHANGE_RE.match(text):
            return "mic_change"
        if root == "system_monitor" and _RAM_RE.search(text):
            return "ram"
        if root in BRIEFING_SOURCES:
            return "briefing"
        return "other"
    except Exception:
        return ""


def never_fold(entry) -> bool:
    """plan_return_drain's never_fold: True when the line must not become a
    recap fragment (it is spoken whole while young enough, then dropped)."""
    cls = line_class(entry)
    return bool(cls) and not QUEUE_RULES[cls][1]


def _age_s(entry, now: float) -> float:
    try:
        return float(now) - float(entry.get("ts"))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def too_old_reason(entry, *, now: float, local_hour: int) -> str:
    """Why this queued line must not be spoken now ('' = it may be)."""
    cls = line_class(entry)
    if not cls:
        return ""
    age = _age_s(entry, now)
    limit = QUEUE_RULES[cls][0]
    if age > limit:
        return f"{cls}: waited {age:.0f} s, limit {limit:.0f} s"
    if (EVENING_UNTIL_HOUR <= int(local_hour) < EVENING_FROM_HOUR
            and age > EVENING_LINE_FRESH_S
            and _EVENING_RE.match(_spoken_text(entry))):
        return f"good evening at {int(local_hour):02d}h"
    return ""


def _local_hour(now: float) -> int:
    """The local hour at ``now`` (a seam: tests pin it)."""
    return time.localtime(float(now)).tm_hour


def plan_ages(items: list, *, now: float, local_hour: "int | None" = None
              ) -> tuple:
    """(entries still fit to say, in queue order, [(dropped entry, reason)]).
    Never raises: on an error the queue is returned unchanged."""
    try:
        if local_hour is None:
            local_hour = _local_hour(now)
        kept: list = []
        dropped: list = []
        for item in items:
            why = too_old_reason(item, now=now, local_hour=local_hour)
            if why:
                dropped.append((item, why))
            else:
                kept.append(item)
        return kept, dropped
    except Exception:
        return list(items), []


def is_greeting(entry) -> bool:
    """A briefing or arrival line that opens by greeting him."""
    try:
        return (isinstance(entry, dict)
                and source_root(entry) in BRIEFING_SOURCES
                and bool(_GREETING_RE.match(_spoken_text(entry))))
    except Exception:
        return False


def greeting_first(items: list) -> list:
    """The queue with its greeting lines moved ahead of everything else except
    his own reminders and urgent lines at the head of the queue. Live 11-05
    "Good morning, sir." was queued at 10:54:24 and spoken at 10:55:11, last,
    behind the recap. The rest keeps its order. Never raises."""
    try:
        greetings = [e for e in items if is_greeting(e)]
        if not greetings:
            return items
        rest = [e for e in items if not is_greeting(e)]
        at = 0
        while (at < len(rest) and isinstance(rest[at], dict)
               and _exempt(rest[at])):
            at += 1
        return rest[:at] + greetings + rest[at:]
    except Exception:
        return items
