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
not to him (_MENTION_VERBS). So is one followed - with NO vocative
punctuation after the name - by an auxiliary, copula or modal that does not
open a question ("So Jarvis is broken again", "Oh Jarvis can't hear us";
but "So Jarvis is it raining" and "Um Jarvis can you hear me" are questions
put to him), or by a past-tense verb ("So Jarvis shut down.", "So Jarvis
turned off the lights"). Review 2026-10-02: those passed, and
canonical_wake_text then turned the mention into a command-shaped "Jarvis
shut down." that armed the shutdown prompt. "So Jarvis, is it raining?"
(a comma, a vocative) still passes. The guard applies to the NEW filler-led
forms only: the forms that always passed ("Jarvis ...", "hey / ok / okay
Jarvis ...") pass exactly as before.

Punctuation around a word does not change the word ("What? Jarvis, ..." is
the same as "what jarvis ..."), but it is read for one thing: whether the
name is set off as a vocative. A possessive ("Jarvis's", "Jarvis'") is not
the wake word in any form.

"Hay" (2026-10-05). Parakeet writes "Hey Jarvis" as "Hay Jarvis": 18 of
its 22 refusals on clean synthetic commands. "hay" is read exactly as
"hey" (a legacy lead: no mention guard), and canonical_wake_text hands
the line on as "Hey Jarvis ..." so every downstream "hey jarvis" stripper
sees the form it knows. Fuzzy spellings of the NAME ("Jervis", "Jarva")
stay out: they would widen the rule for every caller.

Sentence re-anchor (2026-10-05, WAKE_REANCHOR_MODE). Over a video the
owner speaks into a capture that is already running, so his "Jarvis, pause
the music" lands behind the video's words and the whole line fails the
rule above. reanchor() tries each later SENTENCE (split at . ! ?) and
returns the first one that is addressed, with the rest of the line as the
command. It never changes has_wake_prefix: the caller decides what to do
with a re-anchored line (shadow counts, or a voice check and a fresh decode
of the audio from the name before it is accepted - the monolith's
_wake_reanchor).

Public API (pure stdlib, never raises, safe on the light-deps CI runner):
    has_wake_prefix(text)  -> bool  the gate question
    strip_wake_lead(text)  -> str   the command with filler + wake removed
    canonical_wake_text(text) -> str  a filler-led wake rewritten to the
                                      plain "Jarvis ..." form every
                                      downstream handler already strips
    reanchor(text) -> Reanchor | None  the first later sentence that is
                                      addressed, for a line that is not
"""
from __future__ import annotations

import re
from typing import NamedTuple

WAKE_WORD = "jarvis"

# The wake word may be word 1, 2 or 3 of the utterance — never deeper.
MAX_WAKE_POSITION = 3

# Interjections that may come before the wake word. Keep this to sounds and
# discourse markers that carry no request of their own: a content word here
# ("tell", "ask", "I") would turn a mention into a wake. tools/web_interface.py
# mirrors this list in its standby guard; tests/test_wake_prefix.py pins the
# two together.
WAKE_LEAD_FILLERS = frozenset({
    "what", "okay", "ok", "hey", "hay", "yo", "so", "um", "umm", "uh",
    "uhh", "oh", "alright",
})
# Two-word interjections, each word counting toward MAX_WAKE_POSITION:
# Whisper writes "alright" as "All right," about as often as not.
WAKE_LEAD_PHRASES = frozenset({("all", "right")})

# The single-word leads that always passed ("hey Jarvis", "ok Jarvis",
# "okay Jarvis"). Utterances in these legacy forms are never rewritten and
# never mention-guarded, so their behaviour is exactly what it was. "hay" is
# Parakeet's spelling of "hey" (2026-10-05) and is read as "hey".
_LEGACY_LEADS = frozenset({"hey", "hay", "ok", "okay"})

# A legacy lead that is only a MISHEARING of another, and the word it is:
# canonical_wake_text rewrites it so the handlers downstream (which strip
# "hey jarvis", never "hay jarvis") see the form they know.
_MISHEARD_LEADS = {"hay": "Hey"}

# After a FILLER-led name these make it a third-person mention, not an
# address. Only unambiguous reporting forms: an auxiliary ("is", "did",
# "can't") also opens a question put TO him ("So Jarvis, is it raining?"),
# so none is listed.
_MENTION_VERBS = frozenset({
    "said", "says", "told", "tells", "thinks", "thought", "knows", "knew",
    "wants", "wanted", "meant", "means", "heard", "keeps", "kept", "seems",
    "seemed", "sounds", "sounded", "likes", "liked", "needs", "needed",
})

# After a filler-led name with NO vocative punctuation (review 2026-10-02):
# an auxiliary / copula / modal is a statement about him ("So Jarvis is
# broken again", "What Jarvis did was wrong", "Oh Jarvis can't hear us")
# unless the word after it is a subject - then it is a question put to him
# ("um Jarvis is it raining", "Um Jarvis can you hear me"). "do" and "have"
# are left out: they also open an imperative ("um Jarvis do me a favour").
_AUX_VERBS = frozenset({
    "is", "was", "are", "were", "isn't", "wasn't", "aren't", "weren't",
    "does", "did", "doesn't", "didn't", "don't", "has", "had", "hasn't",
    "haven't", "hadn't", "can", "can't", "cannot", "could", "couldn't",
    "will", "won't", "would", "wouldn't", "should", "shouldn't", "must",
    "might", "may", "shall",
})
# A subject right after the auxiliary makes it an inverted question.
_INVERSION_SUBJECTS = frozenset({
    "you", "i", "we", "it", "there", "they", "he", "she", "that", "this",
    "these", "those",
})
# Past tenses with no "-ed". Only forms that are never also the imperative
# ("set", "put", "cut", "quit", "read" are left out) - except "shut", which
# the review names: an unpunctuated "um Jarvis shut down" is refused, the
# safe side for a shutdown.
_IRREGULAR_PAST = frozenset({
    "shut", "broke", "went", "got", "gave", "took", "made", "left", "came",
    "ran", "saw", "felt", "found", "froze", "fell", "forgot", "lost", "woke",
    "hung", "spoke", "wrote", "began", "became", "bought", "brought",
    "caught", "sold", "sent", "slept", "stood", "won", "drove", "flew",
    "threw",
})
# "-ed" words that are not past tenses (the "-eed" ones are excluded by
# shape: "need", "speed", "feed", "proceed").
_ED_NOT_PAST = frozenset({"embed", "imbed", "shred"})
# Punctuation after the name that sets it off as a vocative ("So Jarvis, is
# it raining?" / "Um, Jarvis. Lights off.").
_VOCATIVE_PUNCT = frozenset(",.!?;:—–-…")
# Apostrophes: "Jarvis'" / "Jarvis’" is a possessive, never the wake word.
_APOSTROPHES = ("'", "’")

# Characters trimmed from both ends of a word before it is compared.
_WORD_TRIM = ",.!?;:-—–…\"'()[]“”‘’"
# Separators dropped between the wake word and the command it leads.
_LEAD_SEP = " \t\r\n,.!?;:-—–…"

_TOKEN_RE = re.compile(r"\S+")


def _norm(token: str) -> str:
    return token.strip(_WORD_TRIM).lower()


def _words(text):
    """``(match, normalised word)`` for the first MAX_WAKE_POSITION + 2
    words of ``text`` — enough to find a wake word at word 1-3 and the two
    words after it. A bare "-" / "..." is not a word."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        w = _norm(m.group(0))
        if w:
            out.append((m, w))
            if len(out) > MAX_WAKE_POSITION + 1:
                break
    return out


class _Hit(NamedTuple):
    """A wake word found at word 1-3 behind lead fillers."""
    start: int        # offset of the name's first letter in the text
    end: int          # offset just after the token holding the name
    leads: list       # the normalised fillers before it
    nxt: str          # the normalised word after it ("" at the end)
    nxt2: str         # ...and the one after that
    vocative: bool    # punctuation sets the name off ("Jarvis," / "Jarvis.")


def _scan(text):
    """A _Hit for an utterance led by the wake word, else None (also for a
    possessive "Jarvis'")."""
    if not isinstance(text, str):
        return None
    words = _words(text)
    leads: list = []
    i = 0
    while i < min(len(words), MAX_WAKE_POSITION):
        m, w = words[i]
        if w == WAKE_WORD:
            at = m.group(0).lower().find(WAKE_WORD)
            start = m.start() + max(at, 0)
            name_end = start + len(WAKE_WORD)
            if text[name_end:name_end + 1] in _APOSTROPHES:
                return None      # "Jarvis' voice": a possessive
            nxt_m = words[i + 1][0] if i + 1 < len(words) else None
            gap = text[name_end:nxt_m.start() if nxt_m else len(text)]
            return _Hit(
                start, m.end(), leads,
                words[i + 1][1] if i + 1 < len(words) else "",
                words[i + 2][1] if i + 2 < len(words) else "",
                any(c in _VOCATIVE_PUNCT for c in gap))
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


def _is_past_tense(word: str) -> bool:
    if word in _IRREGULAR_PAST:
        return True
    return (len(word) >= 4 and word.endswith("ed")
            and not word.endswith("eed") and word not in _ED_NOT_PAST)


def _is_mention(hit: _Hit) -> bool:
    """A filler-led name that is talked ABOUT, not to (see the module
    docstring). Never called for the legacy forms."""
    if hit.nxt in _MENTION_VERBS:
        return True
    if hit.vocative:
        return False
    if hit.nxt in _AUX_VERBS:
        return hit.nxt2 not in _INVERSION_SUBJECTS
    return _is_past_tense(hit.nxt)


def _addressed(text):
    """The _scan result when ``text`` is addressed to JARVIS, else None."""
    hit = _scan(text)
    if hit is None:
        return None
    if not _is_legacy(hit.leads) and _is_mention(hit):
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
        return text[hit.end:].lstrip(_LEAD_SEP)
    except Exception:
        return text if isinstance(text, str) else ""


def canonical_wake_text(text) -> str:
    """A filler-led wake rewritten to the plain prefix form — "What Jarvis
    what model are you?" -> "Jarvis what model are you?" — so every
    downstream handler that already strips a leading "Jarvis" (lead_fillers,
    yes_no, the fast paths, the skill routes) sees the command the way the
    prefix path hands it over.

    Punctuation-only tokens before the name ("- Jarvis, turn it off.",
    '"Jarvis, pause"') are dropped too: the gate admits that text, and a
    stripper that expects a leading "Jarvis" would otherwise see "- jarvis
    turn it off" and miss (review 2026-10-02).

    Unchanged: text not addressed to JARVIS, the legacy forms ("Jarvis ...",
    "hey / ok / okay Jarvis ..."), and a name that ENDS the utterance
    ("Alright, Jarvis." — the lead IS the message there, a yes to a
    question). A misheard legacy lead is the one rewrite of a legacy form:
    "Hay Jarvis, pause" -> "Hey Jarvis, pause" (_MISHEARD_LEADS). Never
    raises."""
    try:
        hit = _addressed(text)
        if hit is None:
            return text
        if not hit.leads:
            # No filler: only punctuation can sit before the name.
            return text[hit.start:] if text[:hit.start].strip() else text
        if len(hit.leads) == 1 and hit.leads[0] in _MISHEARD_LEADS:
            # "Hay Jarvis, ..." -> "Hey Jarvis, ..." (the lead was misheard).
            return f"{_MISHEARD_LEADS[hit.leads[0]]} {text[hit.start:]}"
        if _is_legacy(hit.leads) or not hit.nxt:
            return text
        return text[hit.start:]
    except Exception:
        return text


# ── sentence re-anchor (2026-10-05) ───────────────────────────────────────
# A sentence ends at . ! ? or an ellipsis followed by whitespace. A decimal
# ("3.5") or an abbreviation without a following space never splits.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])[\"”’')\]]*\s+")


class Reanchor(NamedTuple):
    """A line re-anchored at a later sentence (see reanchor)."""
    command: str      # the line from that sentence to its end
    sentence: int     # which sentence (1 = the second; 0 never: the whole
                      # line would have passed has_wake_prefix)
    start_word: int   # index of the sentence's first word in text.split()
    name_word: int    # index of the wake word in text.split()
    words: int        # len(text.split()) - for the caller's timestamp check


def _word_index(text: str, offset: int) -> int:
    """How many whitespace-separated words of ``text`` start before
    ``offset`` - the index, in text.split(), of the word at ``offset``."""
    return sum(1 for _ in _TOKEN_RE.finditer(text[:max(0, offset)]))


def reanchor(text) -> "Reanchor | None":
    """For a line that is NOT addressed to JARVIS at its start
    (has_wake_prefix False): the first later sentence - split at . ! ? - that
    IS addressed, by the same rule, with the rest of the line as its command.
    None when the line already passes (nothing to re-anchor), when no
    sentence passes, or for anything that is not a string. A video's
    "...that was close. Jarvis, pause the music." -> command "Jarvis, pause
    the music.", sentence 1. The mention guards apply to each sentence as
    they do to a whole line ("...and then. So Jarvis said no." stays
    refused). Never raises."""
    try:
        if not isinstance(text, str) or has_wake_prefix(text):
            return None
        sentence = 0
        for m in _SENTENCE_END_RE.finditer(text):
            sentence += 1
            start = m.end()
            rest = text[start:]
            if not rest.strip():
                break
            hit = _addressed(rest)
            if hit is None:
                continue
            return Reanchor(rest, sentence, _word_index(text, start),
                            _word_index(text, start + hit.start),
                            len(text.split()))
        return None
    except Exception:
        return None


def name_word_index(text) -> "int | None":
    """The index, in text.split(), of the wake word of a line addressed to
    JARVIS at its start (has_wake_prefix), else None. The loopback veto
    (D2) asks when the name was heard from the transcript's word times.
    Never raises."""
    try:
        hit = _addressed(text)
        if hit is None:
            return None
        return _word_index(text, hit.start)
    except Exception:
        return None
