"""core/queue_receipts.py - what became of a queued proactive line (2026-11-17).

The speech-queue drain (bobert_companion._speak_pending) reports two things
here: a line was SPOKEN (note_spoken), or it was DROPPED without a word
(note_dropped: too old - core/pending_policy.plan_ages; a moment-line that
expired while he was away - core/owner_presence.plan_return_drain).

* A skill may register a receipt for its queued source (register). The
  presence-triggered morning briefing (skills/daily_briefing/arrival_v2)
  learns whether he HEARD it. Live 11-17 06:35:30 it was queued as his face
  appeared, held behind his video and room talk while he got ready, then
  "owner away" for hours, and dropped at 16:02:41 ("waited 34031 s") - while
  the day had counted as briefed from the moment it was queued.
* The lines dropped as too old are remembered for a while (missed /
  take_missed), so his "did anything happen while I was away" can name them
  (bobert_companion._away_recap_shortcut, missed_phrase): at 16:02:41 the
  briefing, four memory alerts and a microphone change went without a word.

Stdlib only; thread-safe; the note_* / missed* calls never raise. In memory: a restart forgets what was missed (the queue
itself survives restarts; while he is away the presence hold leaves it
unclaimed, so its too-old drops are made by the drain that runs when he is
back - the run he asks in).
"""
from __future__ import annotations

import threading
import time

MISSED_CAP = 30
MISSED_KEEP_S = 12 * 3600.0
# Classes (core/pending_policy.QUEUE_RULES) not worth naming as missed: a clap
# answer or a wake acknowledgement is over the moment it is late, and a stale
# mic-health alert ("has been completely silent", "appears to be down") is
# false once he is talking to JARVIS again (a mic still dead is warned about
# again on the alert's own cool-down).
NOT_NEWS = frozenset({"clap_ack", "wake_ack", "mic_health"})

_lock = threading.Lock()
_hooks: dict = {}     # source root -> (on_spoken, on_dropped)
_missed: list = []    # [(dropped_at, entry, line class)]


def _root(entry) -> str:
    try:
        from core.owner_presence import source_root
        return source_root(entry)
    except Exception:
        return ""


def register(source_root: str, *, on_spoken=None, on_dropped=None) -> None:
    """Call ``on_spoken(entry)`` when a line queued under ``source_root`` (its
    source before any ':') is spoken, ``on_dropped(entry, why)`` when one is
    dropped unsaid. A later register for the same root replaces it."""
    with _lock:
        _hooks[str(source_root)] = (on_spoken, on_dropped)


def unregister(source_root: str) -> None:
    with _lock:
        _hooks.pop(str(source_root), None)


def _hook(entry, which: int):
    with _lock:
        pair = _hooks.get(_root(entry))
    return pair[which] if pair else None


def note_spoken(entry) -> None:
    """The drain spoke ``entry``. Never raises."""
    try:
        fn = _hook(entry, 0)
        if callable(fn):
            fn(entry)
    except Exception as e:
        print(f"  [queue] spoken receipt failed: {type(e).__name__}: {e}")


def note_dropped(entry, why: str = "", *, remember: bool = True,
                 now: "float | None" = None) -> None:
    """The drain dropped ``entry`` unsaid. ``remember`` False: a moment-line
    (a break nudge) - its skill still hears of it, but it is not news he
    missed. Never raises."""
    try:
        fn = _hook(entry, 1)
        if callable(fn):
            fn(entry, str(why or ""))
    except Exception as e:
        print(f"  [queue] dropped receipt failed: {type(e).__name__}: {e}")
    if not remember:
        return
    try:
        from core.pending_policy import line_class
        cls = line_class(entry) or "other"
        if cls in NOT_NEWS or not isinstance(entry, dict):
            return
        at = time.time() if now is None else float(now)
        with _lock:
            _missed.append((at, dict(entry), cls))
            del _missed[:-MISSED_CAP]
    except Exception:
        pass


def _fresh(now: float) -> list:
    return [m for m in _missed if 0 <= now - m[0] <= MISSED_KEEP_S]


def missed(now: "float | None" = None) -> list:
    """[(dropped_at, entry, line class)] dropped within MISSED_KEEP_S."""
    try:
        now = time.time() if now is None else float(now)
        with _lock:
            _missed[:] = _fresh(now)
            return list(_missed)
    except Exception:
        return []


def take_missed(now: "float | None" = None) -> list:
    """missed(), then forget them (they have been told). Never raises."""
    try:
        now = time.time() if now is None else float(now)
        with _lock:
            out = _fresh(now)
            _missed.clear()
            return out
    except Exception:
        return []


def clear() -> None:
    with _lock:
        _missed.clear()


def reset() -> None:
    """Forget every receipt and every missed line (the test harness, so one
    test's hook or drop never reaches another)."""
    with _lock:
        _missed.clear()
        _hooks.clear()


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten")
# Briefing sources by name; any other briefing source is a morning one.
_BRIEFING_NAMES = {
    "evening": "the evening briefing",
    "recap": "the day's recap",
    "weekly_digest_briefing": "the weekly digest",
    "news": "the news headlines",
}
_COUNTED = (
    ("ram", "memory warning"),
    ("mic_change", "microphone change"),
    ("weather", "weather note"),
    ("other", "other notice"),
)


def _counted(n: int, noun: str) -> str:
    if n == 1:
        return ("one " if noun.startswith("other ") else "a ") + noun
    word = _WORDS[n] if 0 <= n < len(_WORDS) else str(n)
    return f"{word} {noun}s"


def missed_phrase(entries) -> str:
    """'the morning briefing, four memory warnings and a microphone change'
    for missed() / take_missed() rows ('' for none). Briefings by name, once
    each; the rest counted by kind. Never raises."""
    try:
        names: list = []
        counts: dict = {}
        for row in entries or ():
            entry, cls = row[1], row[2]
            if cls == "briefing":
                name = _BRIEFING_NAMES.get(_root(entry), "the morning briefing")
                if name not in names:
                    names.append(name)
            else:
                counts[cls] = counts.get(cls, 0) + 1
        for cls, noun in _COUNTED:
            if counts.get(cls):
                names.append(_counted(counts[cls], noun))
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + " and " + names[-1]
    except Exception:
        return ""
