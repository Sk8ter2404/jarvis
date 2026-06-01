"""core/tone_detector.py — cheap pre-LLM tone classifier.

MCU JARVIS reads Tony's mood and adjusts his register on the fly. This is the
tiny pure-Python classifier that runs on every transcribed utterance BEFORE the
LLM call: it scans for stress markers (swearing, urgency words, clipped
imperatives, exclamation, cross-turn repetition) and returns a single tone
label, plus the per-turn system-prompt addendum for that label.

Extracted verbatim from bobert_companion.py so the ~230 lines of tone data +
heuristics live in one small, testable module. The ONLY change vs the inline
version: detect_tone() takes the previous user utterance as a parameter
(prev_user_text) instead of reaching into bobert_companion.conversation_history,
so it stays pure and side-effect-free. bobert_companion keeps a thin wrapper
that supplies prev_user_text from the live history, so every call site is
unchanged. The per-utterance caches (_last_user_tone etc.) stay in the monolith
— they're runtime state, not classification logic.
"""
from __future__ import annotations

import datetime
import re

TONE_DETECTION_ENABLED = True

_STRESS_SWEAR_WORDS = (
    "fuck", "fucking", "fuckin", "shit", "shitty", "damn", "damnit",
    "dammit", "goddamn", "bullshit", "bloody", "bollocks", "wtf",
)

_URGENCY_WORDS = (
    "now", "just", "finally", "hurry", "quick", "quickly", "asap",
    "immediately", "already", "still",
)

_REPETITION_PHRASES = (
    "i said", "i told you", "again", "like i said", "as i said",
    "i just said", "for the third time", "for the second time",
    "still not", "why isnt", "why isn't", "why won't", "why wont",
    "you didn't", "you didnt", "that's not", "thats not",
    "no no", "no that's", "no thats",
)

_TIREDNESS_PHRASES = (
    "im tired", "i'm tired", "im exhausted", "i'm exhausted",
    "im knackered", "i'm knackered", "going to bed", "off to bed",
    "call it a night",
)

_PLAYFUL_MARKERS = (
    "lol", "haha", "hehe", "lmao", "ha ha", "rofl",
)

# Short, imperative-style commands that often signal a stressed user barking an
# order. Only counts when the whole utterance is short (≤3 words) — "wait" in
# the middle of a long sentence isn't stress.
_CLIPPED_IMPERATIVES = (
    "stop", "no", "wait", "now", "go", "do it", "shut up",
    "be quiet", "quiet", "enough", "cancel", "kill it", "abort",
)

# High-energy positive markers. Distinguishes excited utterances from stressed
# ones — both can carry exclamation marks, but excitement pairs them with
# positive content rather than swearing or clipped imperatives.
_EXCITEMENT_PHRASES = (
    "amazing", "awesome", "incredible", "fantastic", "brilliant",
    "excellent", "perfect", "love it", "love this", "i love",
    "let's go", "lets go", "let's do", "lets do", "can't wait",
    "cant wait", "so excited", "so good", "so cool", "yes yes",
    "yesss", "yessss", "woohoo", "woo hoo", "yay", "epic",
    "killer", "nailed it", "beautiful", "magnificent", "splendid",
    "hell yes", "hell yeah", "yeah baby",
)

# Local-clock hours when "late-night" applies as a fallback tone. The range
# wraps midnight: 22:00–04:59 inclusive.
_LATE_NIGHT_START_HOUR = 22
_LATE_NIGHT_END_HOUR   = 5


def _is_late_night_hour(now: "datetime.datetime | None" = None) -> bool:
    """True if the local hour falls in the late-night band (22:00–04:59).
    Callers can pass an explicit datetime (tests, the voice-emotion router)."""
    h = (now or datetime.datetime.now()).hour
    return h >= _LATE_NIGHT_START_HOUR or h < _LATE_NIGHT_END_HOUR


def detect_tone(user_text: str, prev_user_text: str | None = None) -> str | None:
    """Classify the emotional tone of a transcribed user utterance.

    Returns one of: 'frustrated' | 'stressed' | 'rushed' | 'tired' |
    'playful' | 'excited' | 'late_night', or None when nothing notable is
    detected (= default calm register, no system-prompt modification).

    `prev_user_text` is the user's PREVIOUS utterance (or None); when it shares
    a majority of content words with this one the user is restating themselves,
    a strong frustration signal. 'late_night' is a time-of-day fallback applied
    only when no other tone fires, so explicit signals still win after midnight.

    Pure-Python heuristics only — no LLM call, no model load, side-effect free.
    """
    if not TONE_DETECTION_ENABLED or not user_text:
        return None

    raw = user_text.strip()
    if not raw:
        return None

    excl_count = raw.count("!")

    clean = re.sub(r"[^a-z' ]+", " ", raw.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None

    n_words = len(clean.split())

    def _has_any(phrases) -> bool:
        for p in phrases:
            if re.search(r'\b' + re.escape(p) + r'\b', clean):
                return True
        return False

    has_swear   = _has_any(_STRESS_SWEAR_WORDS)
    has_urgency = _has_any(_URGENCY_WORDS)
    has_repeat  = _has_any(_REPETITION_PHRASES)
    has_tired   = _has_any(_TIREDNESS_PHRASES)
    has_playful = _has_any(_PLAYFUL_MARKERS)
    has_excited = _has_any(_EXCITEMENT_PHRASES)
    is_clipped  = (n_words <= 3 and _has_any(_CLIPPED_IMPERATIVES))

    # Cross-turn repetition: if the previous utterance shares a majority of its
    # content words with this one, the user is restating — a strong frustration
    # signal even without explicit "I said" markers.
    similar_to_last = False
    try:
        if prev_user_text:
            p = re.sub(r"[^a-z' ]+", " ", str(prev_user_text).strip().lower())
            p = re.sub(r"\s+", " ", p).strip()
            if p and p != clean:
                pw = set(p.split())
                cw = set(clean.split())
                overlap = len(pw & cw)
                if overlap >= max(2, min(len(pw), len(cw)) // 2):
                    similar_to_last = True
    except Exception:
        similar_to_last = False

    # Priority: frustrated > excited > stressed > rushed > tired > playful >
    # late_night (time-based fallback). Frustration trumps stress because the
    # response strategy differs; excited fires before stressed because both can
    # carry exclamation marks but excitement pairs them with positive markers
    # and no swearing / clipped-imperative pattern.
    if has_repeat or similar_to_last or (has_swear and (has_urgency or is_clipped)):
        return "frustrated"

    if has_excited and not has_swear and not is_clipped:
        return "excited"

    if has_swear or excl_count >= 2 or (is_clipped and excl_count >= 1):
        return "stressed"

    if has_urgency or is_clipped:
        return "rushed"

    if has_tired:
        return "tired"

    if has_playful:
        return "playful"

    if _is_late_night_hour():
        return "late_night"

    return None


_TONE_HINTS: dict[str, str] = {
    "frustrated": (
        "USER_TONE: frustrated — the user appears to be repeating "
        "themselves or pushing back. Skip pleasantries, skip 'sir' "
        "filler, briefly acknowledge the misfire ('Apologies, sir — "
        "trying again.') and immediately try a different approach. "
        "Do NOT defend the previous attempt. Do NOT explain. Act."
    ),
    "stressed": (
        "USER_TONE: stressed — be extra calm, efficient, and skip "
        "pleasantries. One short sentence. No 'sir' embellishments "
        "unless natural. Lead with the action, not the acknowledgement."
    ),
    "rushed": (
        "USER_TONE: rushed — the user wants this done now. Acknowledge "
        "in ≤5 words ('On it.') then act. Skip preamble and framing."
    ),
    "tired": (
        "USER_TONE: tired — speak softer and shorter than usual. Avoid "
        "stat dumps and dry humour. One gentle sentence is plenty."
    ),
    "playful": (
        "USER_TONE: playful — a touch of dry wit lands well here. Still "
        "≤2 sentences, but a quip in character is welcome."
    ),
    "excited": (
        "USER_TONE: excited — match sir's energy without losing the dry "
        "British register. Affirm enthusiastically in one line ('Splendid, "
        "sir.' / 'Quite agree, sir.') and feel free to add a wry quip. "
        "Do NOT damp the mood with caveats or hedging."
    ),
    "late_night": (
        "USER_TONE: late-night — it's past sir's normal hours. Speak "
        "softer and shorter than usual; lean into gentle understatement. "
        "Avoid stat dumps and full status reports unless asked. One quiet "
        "sentence is plenty."
    ),
}


def _tone_system_addendum(tone: str | None) -> str:
    """Per-turn system-prompt addition for a detected tone, or '' when no
    special handling applies (= default register)."""
    if not tone:
        return ""
    hint = _TONE_HINTS.get(tone)
    if not hint:
        return ""
    return "\n\n[Per-turn tone hint]\n" + hint
