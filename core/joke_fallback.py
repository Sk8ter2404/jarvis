"""core/joke_fallback.py — a joke request must get a joke.

WHY THIS MODULE EXISTS
======================
Live v2.0.131 (2026-09-29, local model): "tell me a short joke" was answered
with "[intent:dry_wit] I'm afraid I've run out of material, sir; my humor
processors seem to have hit a bit of a wall." A refusal dressed as wit. The
same model told a joke fine earlier the same day, so this is the persona
("I'm afraid..." openers, pick-a-stance) occasionally winning over the
request. The prompt now says to deliver (core/prompts.py, "DELIVER WHAT WAS
ASKED"); this module is the small deterministic safety net for the most common
case.

THE RULE
========
``apply(reply, user_text)`` returns a bundled one-liner ONLY when all hold:
  1. the owner's turn is an explicit joke request ("tell me a joke", "got any
     jokes?", "tell me another one" is NOT enough — it names no joke);
  2. the reply matches a refusal pattern ("run out of material", "not in the
     mood", "hit a wall", "I'll pass", ...);
  3. the reply carries no joke-like content (no setup question answered in
     the same reply, no "walks into a bar", no "because ..." punchline).
Otherwise the reply is returned unchanged. The list is short, clean and
generic on purpose: the repo is public and this is a fallback, not a
repertoire.

Pure: stdlib only, no monolith import — tested on the light-deps CI runner
(tests/test_joke_fallback.py).
"""
from __future__ import annotations

import random
import re
from typing import Optional

__all__ = ["FALLBACK_JOKES", "is_joke_request", "looks_like_refusal",
           "has_joke_content", "apply"]

FALLBACK_JOKES: tuple[str, ...] = (
    "Why don't scientists trust atoms, sir? They make up everything.",
    "I'm reading a book on anti-gravity, sir. It's impossible to put down.",
    "Parallel lines have so much in common, sir. A pity they'll never meet.",
    "Why do programmers prefer dark mode, sir? Because light attracts bugs.",
    "I'd tell you a joke about UDP, sir, but you might not get it.",
    "Why did the scarecrow win an award, sir? He was outstanding in his field.",
    "There are ten kinds of people, sir: those who understand binary and "
    "those who don't.",
    "I told the computer I needed a break, sir. It went to sleep.",
)

_FILLER_RE = re.compile(
    r"^(?:(?:hey|hi|ok(?:ay)?|so|and|well|um+|uh+|jarvis|sir|please|alright|"
    r"come\s+on)[,.!\s]+)+")
_JOKE_REQUEST_RE = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
    r"(?:tell|give|hit|share\s+with)\s+(?:me|us)(?:\s+with)?\s+"
    r"(?:a|an|another|one|some|your)?\s*(?:[\w'-]+\s+){0,3}?jokes?\b"
    r"|^(?:do\s+you\s+)?(?:know|got|have)\s+(?:any|a|another)\s+"
    r"(?:[\w'-]+\s+){0,2}?jokes?\b"
    r"|^(?:make\s+me\s+laugh|say\s+something\s+funny)\b"
    r"|^(?:a\s+|another\s+)?jokes?(?:\s+please)?[.!?]*$")


def _norm(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


def is_joke_request(user_text: str) -> bool:
    """True for an explicit request for a joke ("tell me a short joke",
    "got any good jokes?", "make me laugh"). "Are you still making jokes at
    me" and "that was a bad joke" are not requests."""
    t = _FILLER_RE.sub("", _norm(user_text)).strip()
    return bool(t) and bool(_JOKE_REQUEST_RE.match(t))


_REFUSAL_RE = re.compile(
    r"\b(?:(?:run|ran|running)\s+out\s+of\s+(?:material|jokes|ideas|humou?r)|"
    r"(?:fresh|clean|all)\s+out\s+of\s+(?:material|jokes|humou?r)|"
    r"out\s+of\s+(?:material|jokes)|"
    r"hit\s+(?:a\s+)?(?:bit\s+of\s+a\s+)?(?:wall|snag|block)|"
    r"(?:can'?t|cannot|couldn'?t|unable\s+to)\s+(?:think\s+of|come\s+up\s+with|"
    r"muster|summon|find)\s+(?:a|an|any|one)\b|"
    r"(?:not|hardly)\s+(?:in\s+the\s+mood|feeling\s+(?:particularly\s+)?"
    r"(?:funny|humorous|witty))|"
    r"(?:i'?ll|i\s+will|i'?d\s+rather|i\s+would\s+rather|i\s+must|"
    r"i'?m\s+going\s+to)\s+"
    r"(?:pass|decline|refrain|abstain|not\b|sit\s+this\s+one\s+out)|"
    # "My humour circuits" alone is persona colour, often the preamble to a
    # REAL joke ("my humour circuits are rusty, sir, but here goes: ..."), so
    # only a stated failure makes it a refusal.
    r"(?:humou?r|comedy|comedic|joke)\s+(?:processors?|circuits?|modules?|"
    r"subroutines?|reserves?|banks?|database)\s+(?:[\w']+\s+){0,4}?"
    r"(?:offline|down|empty|exhausted|depleted|drained|jammed|fried|"
    r"on\s+strike|out\s+of\s+order|failed|failing|broken|dry)\b|"
    r"no\s+jokes?\s+(?:today|tonight|at\s+the\s+moment|right\s+now)|"
    # "...any jokes about paper, sir. They're tearable." is a pun, not a
    # refusal: "about" keeps it out.
    r"(?:don'?t|do\s+not)\s+(?:have|know)\s+(?:a|any)\s+(?:good\s+)?"
    r"jokes?\b(?!\s+about\b))")

# Joke-like content. A setup is a wh-/aux-question ("Why did the ...?", "What
# do you call ...?") followed by a punchline; "A joke, sir? I'm afraid I've run
# out of material." is a question back to the owner, not a setup. A bare
# "because" is not a punchline either ("..., because you've heard them all").
_JOKE_SHAPE_RE = re.compile(
    r"\b(?:why|what|what'?s|how|where|who|when|which|did|do\s+you|"
    r"have\s+you)\b[^.!?]*\?\s*\S+.{3,}"
    r"|\bwalks?\s+into\s+a\b"
    r"|\bknock,?\s+knock\b"
    r"|\bhere\s+goes\b\s*[:,.!-]*\s*\S"
    r"|\b(?:here'?s|try\s+this)\s+(?:one|a\s+\w+|an\s+old\s+one)\s*[:,.!-]\s*\S")


def looks_like_refusal(reply: str) -> bool:
    return bool(_REFUSAL_RE.search(_norm(reply)))


def has_joke_content(reply: str) -> bool:
    return bool(_JOKE_SHAPE_RE.search(_norm(reply)))


_last: list[Optional[str]] = [None]


def _pick(rng) -> str:
    pool = [j for j in FALLBACK_JOKES if j != _last[0]] or list(FALLBACK_JOKES)
    joke = rng.choice(pool)
    _last[0] = joke
    return joke


def apply(reply: str, user_text: str, *, rng=None) -> Optional[str]:
    """The bundled one-liner to speak INSTEAD of ``reply`` when the owner asked
    for a joke and the reply refused without telling one; None otherwise
    (keep the reply). ``rng`` — a ``random.Random``-like object (tests pass a
    seeded one). Never raises."""
    try:
        if not reply or not is_joke_request(user_text):
            return None
        if not looks_like_refusal(reply) or has_joke_content(reply):
            return None
        return _pick(rng or random)
    except Exception:
        return None
