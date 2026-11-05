"""core/chain_lines.py - a follow-up round does not say the same line again
(v3.0.11, CHAIN_LEAD_DEDUPE).

Live 2026-11-05 14:58:15, :20 and :23: "Apologies, sir." was spoken three times
in one turn, once per follow-up round, because each round's reply opened with
it. A lead sentence identical to one already spoken this turn is not spoken
again; the rest of the reply is.

Stdlib only, nothing here raises.
"""
from __future__ import annotations

import re

_LEAD_TAGS_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def line_key(sentence) -> str:
    """The sentence compared: lower case, letters and digits only."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(sentence or "").lower()).split())


def spoken_keys(text) -> set:
    """The keys of every sentence in ``text`` (leading tags off). Never
    raises."""
    try:
        src = str(text or "")
        body = src[_LEAD_TAGS_RE.match(src).end():]
        return {k for k in (line_key(p) for p in _SENTENCE_RE.split(body)) if k}
    except Exception:
        return set()


def drop_repeated_lead(text, seen) -> tuple:
    """(``text`` without leading sentences already in ``seen``, how many went).
    Leading tags stay unless nothing is left. ``text`` unchanged and 0 when its
    lead is new. Never raises."""
    try:
        src = str(text or "")
        if not src.strip() or not seen:
            return text, 0
        lead = _LEAD_TAGS_RE.match(src).group(0)
        parts = [p for p in _SENTENCE_RE.split(src[len(lead):]) if p and p.strip()]
        n = 0
        while n < len(parts) and line_key(parts[n]) in seen:
            n += 1
        if not n:
            return text, 0
        rest = " ".join(p.strip() for p in parts[n:]).strip()
        return ((lead.strip() + " " + rest).strip() if rest else ""), n
    except Exception:
        return text, 0
