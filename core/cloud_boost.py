"""Time-boxed Claude boost (CLOUD_BOOST_UNTIL, 2026-10-12).

JARVIS is local-first. While the owner still has Claude API credits that will
expire, he may set CLOUD_BOOST_UNTIL to an ISO date ("2026-11-05"): until the
end of that day, HARD questions and screen-reading questions are answered by
Claude instead of the local model, and a local answer that fails falls to
Claude even on an AI_BACKEND "ollama" install. The day after, everything
returns to normal by itself, with one log line.

Empty / unparseable = OFF = no behaviour change anywhere. This module only
decides; the callers in bobert_companion.py own the Claude calls and still
need an ANTHROPIC_API_KEY (no key = no boost = the local path, as before).
"""
from __future__ import annotations

import datetime as _dt
import re

_state = {"announced": None}     # last logged state: "on" | "expired" | None


def _parse(until) -> "_dt.date | None":
    try:
        s = str(until or "").strip()
        return _dt.date.fromisoformat(s[:10]) if s else None
    except Exception:
        return None


def boost_active(until, today: "_dt.date | None" = None, log=print) -> bool:
    """True while today <= the CLOUD_BOOST_UNTIL date. Logs ONE line when the
    boost is first seen on, and ONE when it is first seen expired. Never
    raises; empty or invalid = False, silently."""
    try:
        end = _parse(until)
        if end is None:
            return False
        today = today or _dt.date.today()
        if today <= end:
            if _state["announced"] != "on":
                _state["announced"] = "on"
                log(f"  [cloud-boost] ON until {end.isoformat()} - hard "
                    f"questions and screen reading go to Claude")
            return True
        if _state["announced"] != "expired":
            _state["announced"] = "expired"
            log(f"  [cloud-boost] expired {end.isoformat()} - back to the "
                f"local-first defaults")
        return False
    except Exception:
        return False


_HARD_PHRASES = re.compile(
    r"\b(think (?:hard|carefully|deeply|it through)|look (?:it|this|that) up|"
    r"look up|do some research|research (?:this|that|it)|step by step|"
    r"in (?:great )?detail|deep dive|walk me through|pros and cons|"
    r"compare (?:and contrast )?\w+ (?:and|with|to|vs)|"
    r"explain (?:why|how)|analy[sz]e|figure out why)\b", re.I)
_PART_SPLIT = re.compile(r"\b(?:and then|and also|also|then|after that|"
                         r"second(?:ly)?|finally|first(?:ly)?)\b", re.I)


def is_hard_turn(text: str) -> bool:
    """No existing hardness signal in the repo, so a conservative one: an
    explicit "think hard / look it up / step by step / analyze" ask, or a long
    multi-part question (>= 40 words), or a medium one (>= 20 words) with
    several parts (2+ question marks or 2+ connectors). Short commands and
    chit-chat are never hard."""
    try:
        t = str(text or "").strip()
        if not t:
            return False
        words = len(t.split())
        if words < 6:
            return False
        if _HARD_PHRASES.search(t):
            return True
        if words >= 40:
            return True
        parts = t.count("?") + len(_PART_SPLIT.findall(t))
        return words >= 20 and parts >= 2
    except Exception:
        return False
