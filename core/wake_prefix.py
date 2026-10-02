"""core/wake_prefix.py — the ONE "is this utterance addressed to JARVIS by
name at its start?" rule.

Wake-word mode (config REQUIRE_WAKE_MODE), the music gates, the follow-up
window, the standby wake that carries a command and the learn gate's
``wake=`` signal all ask the same question. It used to be answered by two
hand-copied checks (bobert_companion._text_has_wake_prefix and
skills/standby_audio_detect.should_refuse_wake) that only accepted "jarvis"
as the FIRST word (or "hey / ok / okay jarvis"), plus a regex stripper
(core/fast_paths._WAKE_LEAD_RE) with its own idea of the same lead.

Live 2026-10-01 20:57: "What Jarvis what model are you?" was dropped as
overheard audio in wake-word mode ("[bg-audio] wake-word mode — ignoring
non-wake utterance") — a spoken interjection before the name is normal
speech, and Whisper writes it down.

The rule
--------
The wake word may sit at word 1, 2 or 3 (MAX_WAKE_POSITION), and every word
before it must be a lead interjection (WAKE_LEAD_FILLERS: "what", "okay",
"hey", "yo", "so", "um", "uh", "oh", "alright", "all right", ...). So "What
Jarvis, what model are you?", "Um, okay, Jarvis, pause", "All right, Jarvis"
and "Yo Jarvis" are addressed;
"I asked Jarvis yesterday", "tell Jarvis that ..." (a non-filler lead) and
"so um uh Jarvis" (word 4) are not.

A filler-led name followed by a third-person reporting verb is somebody
talking ABOUT him ("So Jarvis said it would rain", "What Jarvis told me"),
not to him (_MENTION_VERBS). The guard applies to the NEW filler-led forms
only: the forms that always passed ("Jarvis ...", "hey / ok / okay Jarvis
...") pass exactly as before.

Punctuation around a word never matters ("What? Jarvis, ..." is the same
as "what jarvis ..."); a possessive ("Jarvis's") is not the wake word.

Public API (pure stdlib, never raises, safe on the light-deps CI runner):
    has_wake_prefix(text)  -> bool  the gate question
    strip_wake_lead(text)  -> str   the command with filler + wake removed
    canonical_wake_text(text) -> str  a filler-led wake rewritten to the
                                      plain "Jarvis ..." form every
                                      downstream handler already strips
"""
from __future__ import annotations

import re

WAKE_WORD = "jarvis"

# The wake word may be word 1, 2 or 3 of the utterance — never deeper.
MAX_WAKE_POSITION = 3

# Interjections that may come before the wake word. Keep this to sounds and
# discourse markers that carry no request of their own: a content word here
# ("tell", "ask", "I") would turn a mention into a wake. tools/web_interface.py
# mirrors this list in its standby guard; tests/test_wake_prefix.py pins the
# two together.
WAKE_LEAD_FILLERS = frozenset({
    "what", "okay", "ok", "hey", "yo", "so", "um", "umm", "uh", "uhh",
    "oh", "alright",
})
# Two-word interjections, each word counting toward MAX_WAKE_POSITION:
# Whisper writes "alright" as "All right," about as often as not.
WAKE_LEAD_PHRASES = frozenset({("all", "right")})

# The single-word leads that always passed ("hey Jarvis", "ok Jarvis",
# "okay Jarvis"). Utterances in these legacy forms are never rewritten and
# never mention-guarded, so their behaviour is exactly what it was.
_LEGACY_LEADS = frozenset({"hey", "ok", "okay"})

# After a FILLER-led name these make it a third-person mention, not an
# address. Only unambiguous reporting forms: an auxiliary ("is", "did",
# "can't") also opens a question put TO him ("So Jarvis, is it raining?"),
# so none is listed.
_MENTION_VERBS = frozenset({
    "said", "says", "told", "tells", "thinks", "thought", "knows", "knew",
    "wants", "wanted", "meant", "means", "heard", "keeps", "kept", "seems",
    "seemed", "sounds", "sounded", "likes", "liked", "needs", "needed",
})

# Characters trimmed from both ends of a word before it is compared.
_WORD_TRIM = ",.!?;:-—–…\"'()[]“”‘’"
# Separators dropped between the wake word and the command it leads.
_LEAD_SEP = " \t\r\n,.!?;:-—–…"

_TOKEN_RE = re.compile(r"\S+")


def _norm(token: str) -> str:
    return token.strip(_WORD_TRIM).lower()


def _words(text):
    """``(match, normalised word)`` for the first MAX_WAKE_POSITION + 1
    words of ``text`` — enough to find a wake word at word 1-3 and the word
    after it. A bare "-" / "..." is not a word."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        w = _norm(m.group(0))
        if w:
            out.append((m, w))
            if len(out) > MAX_WAKE_POSITION:
                break
    return out


def _scan(text):
    """``(wake_match, leads, next_word)`` for an utterance led by the wake
    word, else None. ``wake_match`` is the regex match of the wake token in
    ``text``; ``leads`` the normalised fillers before it; ``next_word`` the
    normalised word after it ("" at the end)."""
    if not isinstance(text, str):
        return None
    words = _words(text)
    leads: list = []
    i = 0
    while i < min(len(words), MAX_WAKE_POSITION):
        m, w = words[i]
        if w == WAKE_WORD:
            nxt = words[i + 1][1] if i + 1 < len(words) else ""
            return m, leads, nxt
        pair = tuple(x[1] for x in words[i:i + 2])
        if pair in WAKE_LEAD_PHRASES:
            leads.append(" ".join(pair))
            i += 2
            continue
        if w in WAKE_LEAD_FILLERS:
            leads.append(w)
            i += 1
            continue
        return None
    return None


def _is_legacy(leads) -> bool:
    return not leads or (len(leads) == 1 and leads[0] in _LEGACY_LEADS)


def _addressed(text):
    """The _scan result when ``text`` is addressed to JARVIS, else None."""
    hit = _scan(text)
    if hit is None:
        return None
    _m, leads, nxt = hit
    if not _is_legacy(leads) and nxt in _MENTION_VERBS:
        return None
    return hit


def has_wake_prefix(text) -> bool:
    """True when ``text`` addresses JARVIS by name at its start (see the
    module docstring for the rule). Any length after the name. Never
    raises."""
    try:
        return _addressed(text) is not None
    except Exception:
        return False


def strip_wake_lead(text) -> str:
    """``text`` with its lead fillers, the wake word and the separators after
    it removed ("Um, Jarvis, turn off the lights" -> "turn off the lights").
    Text not led by the wake word comes back unchanged; a bare wake comes
    back "". Never raises."""
    try:
        hit = _addressed(text)
        if hit is None:
            return text if isinstance(text, str) else ""
        return text[hit[0].end():].lstrip(_LEAD_SEP)
    except Exception:
        return text if isinstance(text, str) else ""


def canonical_wake_text(text) -> str:
    """A filler-led wake rewritten to the plain prefix form — "What Jarvis
    what model are you?" -> "Jarvis what model are you?" — so every
    downstream handler that already strips a leading "Jarvis" (lead_fillers,
    yes_no, the fast paths, the skill routes) sees the command the way the
    prefix path hands it over.

    Unchanged: text not addressed to JARVIS, the legacy forms ("Jarvis ...",
    "hey / ok / okay Jarvis ..."), and a name that ENDS the utterance
    ("Alright, Jarvis." — the lead IS the message there, a yes to a
    question). Never raises."""
    try:
        hit = _addressed(text)
        if hit is None:
            return text
        m, leads, nxt = hit
        if _is_legacy(leads) or not nxt:
            return text
        return text[m.start():]
    except Exception:
        return text
