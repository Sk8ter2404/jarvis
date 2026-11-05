"""core/repeat_line.py - "what did you say?" gets the last line, word for word
(v3.0.11, REPEAT_LAST_LINE).

Live 2026-11-05 14:57:42 "Jarvis, what did you say?" went to the brain, which
retold an earlier action result ("I've just confirmed your identity via the
camera, sir. That's you, sir.") instead of the sentence JARVIS had just said.
A request to hear a line again needs no brain: LastLine keeps what was last
voiced (every sentence of one owner turn together, tags already off) and the
monolith speaks it back.

What counts is core.tone_detector.is_repeat_request, with two narrowings a mood
check does not need:
  * "what is that" is a question about something, never a repeat request here;
  * "what was that" may be about a noise, so it only counts straight after a
    line (SHORT_ASK_WINDOW_S). The others count for ASK_WINDOW_S; after that
    there is nothing fresh to repeat and the old path answers.

Stdlib only, no I/O, nothing here raises.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

from core.tone_detector import is_repeat_request

ASK_WINDOW_S = 600.0
SHORT_ASK_WINDOW_S = 45.0
# Sentences of one owner turn further apart than this are separate lines (a
# proactive remark long after the reply, on the same turn number).
JOIN_WINDOW_S = 30.0

_WHAT_IS_THAT_RE = re.compile(r"\bwhat(?:'s| is|s) that\b")
_WHAT_WAS_THAT_RE = re.compile(r"\bwhat was that\b")


def repeat_line_on() -> bool:
    """REPEAT_LAST_LINE (core/config.py), read at call time. Never raises."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "REPEAT_LAST_LINE", False))
    except Exception:
        return False


def is_repeat_ask(text) -> bool:
    """True when the whole line asks to hear JARVIS's last line again. Never
    raises."""
    try:
        if not is_repeat_request(text):
            return False
        return _WHAT_IS_THAT_RE.search(str(text).lower()) is None
    except Exception:
        return False


class LastLine:
    """What was last voiced. Thread-safe: _speak runs on many threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parts: list = []
        self._turn = None
        self._at = 0.0

    def note(self, text, turn=None, now: Optional[float] = None) -> None:
        """Record a voiced line; a sentence of the same turn joins it. Never
        raises."""
        try:
            s = " ".join(str(text or "").split())
            if not s:
                return
            when = time.time() if now is None else float(now)
            with self._lock:
                if (self._parts and turn == self._turn
                        and when - self._at <= JOIN_WINDOW_S):
                    self._parts.append(s)
                else:
                    self._parts = [s]
                self._turn = turn
                self._at = when
        except Exception:
            pass

    def line_for(self, text, now: Optional[float] = None) -> Optional[str]:
        """The line to say back for this utterance, else None (no repeat ask,
        nothing voiced yet, or it is too old). Never raises."""
        try:
            if not is_repeat_ask(text):
                return None
            when = time.time() if now is None else float(now)
            with self._lock:
                parts, at = list(self._parts), self._at
            if not parts:
                return None
            short = _WHAT_WAS_THAT_RE.search(str(text).lower()) is not None
            if when - at > (SHORT_ASK_WINDOW_S if short else ASK_WINDOW_S):
                return None
            return " ".join(parts)
        except Exception:
            return None
