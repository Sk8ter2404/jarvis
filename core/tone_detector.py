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


def night_quiet_enabled() -> bool:
    """NIGHT_QUIET_ENABLED (core/night_quiet.py), read at call time.

    Imported lazily so this module still imports with no package on sys.path
    (`python core/tone_detector.py`); there it keeps the old behaviour (on).
    """
    try:
        from core.night_quiet import night_quiet_enabled as _enabled
    except ImportError:
        return True
    return _enabled()


_STRESS_SWEAR_WORDS = (
    "fuck", "fucking", "fuckin", "shit", "shitty", "damn", "damnit",
    "dammit", "goddamn", "bullshit", "bloody", "bollocks", "wtf",
)

_URGENCY_WORDS = (
    "now", "just", "finally", "hurry", "quick", "quickly", "asap",
    "immediately", "already",
)

# Words that signal push-back ONLY in context. 'still' used to be a plain
# urgency word, so "I'm still having USB issues" -- the owner REPORTING a
# lingering fault, often the first time he mentions it -- came out 'rushed'
# ("acknowledge in <=5 words, then act"), and with any swear word attached,
# 'frustrated' ("Do NOT explain. Act."). Both registers push the model to
# guess an action when the turn needs a question or a diagnostic. It now
# counts, as frustration, only when the previous JARVIS turn failed or the
# owner is restating himself; on its own it counts for nothing. Multi-word
# phrases that contain it ("still not" in _REPETITION_PHRASES) are unchanged.
_CONTEXT_GATED_FRUSTRATION_WORDS = ("still",)

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


def _late_night_tone_applies(now: "datetime.datetime | None" = None) -> bool:
    """True when the CLOCK alone should put JARVIS in the late-night register:
    inside the late-night band AND NIGHT_QUIET_ENABLED is on
    (core/night_quiet.py). The one gate for both clock-driven late-night
    paths: detect_tone()'s fallback tone and core/voice_emotion's mood. The
    raw hour test stays in _is_late_night_hour()."""
    return night_quiet_enabled() and _is_late_night_hour(now)


def _clean(text) -> str:
    """Lowercase, letters/apostrophes/spaces only, whitespace collapsed."""
    s = re.sub(r"[^a-z' ]+", " ", str(text).strip().lower())
    return re.sub(r"\s+", " ", s).strip()


# Words that carry no request of their own: articles, pronouns, auxiliaries,
# question words, politeness and "again"-type fillers. Comparing requests on
# these made every pair of short questions a "restatement" (live 2026-09-29:
# "what time is it" after "what day is it" shared what/is/it -> frustrated ->
# a stressed voice and 15 min of proactive silence).
_RESTATE_FILLER = frozenset((
    "a an the this that these those it its it's i i'm me my you your you're "
    "we us our he him his she her they them their is are was were be been am "
    "do does did done can could would will should shall may might must have "
    "has had what what's whats which who whom whose when where why how "
    "there here to of for in at by with from as and or but if so then than "
    "please now just again still keep keeps kept really very jarvis sir hey "
    "ok okay um uh well also too any some all").split())

# A request that differs only by one of these pairs asks for something else.
_RESTATE_OPPOSITES = (
    ("on", "off"), ("up", "down"), ("open", "close"), ("start", "stop"),
    ("enable", "disable"), ("lock", "unlock"), ("show", "hide"),
    ("more", "less"), ("louder", "quieter"), ("higher", "lower"),
    ("increase", "decrease"), ("next", "previous"), ("yes", "no"),
    ("left", "right"), ("forward", "back"),
    ("brighter", "dimmer"), ("mute", "unmute"), ("add", "remove"),
)
_RESTATE_NUMBER_WORDS = frozenset((
    "zero one two three four five six seven eight nine ten eleven twelve "
    "fifteen twenty thirty forty fifty sixty hundred thousand half quarter "
    "first second third").split())


def _restate_words(text) -> set:
    return {w for w in _clean(text).split() if w not in _RESTATE_FILLER}


def is_restatement(user_text: str, prev_user_text) -> bool:
    """True when ``user_text`` restates ``prev_user_text``: the same request
    again, reworded or padded ("turn off the lights" -> "turn off the lights
    now", "the usb hub keeps dropping out" -> "the usb hub is still dropping
    out"). Compared on the words that carry the request (fillers dropped):
    at least 2 shared, and one contains the other or they mostly overlap
    (Jaccard >= 0.6). A pair that differs by opposites (on/off, up/down) or
    by a number asks for something else, so it never counts. An IDENTICAL
    previous line does not count either -- callers take "previous" from a
    history the current turn may already sit in.

    Shared with core.emotion_tracker so both classifiers agree on what
    "repeating himself" means. Never raises: anything that cannot be read as
    text is simply not a restatement."""
    try:
        if not prev_user_text or not user_text:
            return False
        cur = _clean(user_text)
        prev = _clean(prev_user_text)
        if not prev or not cur or prev == cur:
            return False
        pw, cw = _restate_words(prev), _restate_words(cur)
        shared = pw & cw
        if len(shared) < 2:
            return False
        only_prev, only_cur = pw - cw, cw - pw
        for a, b in _RESTATE_OPPOSITES:
            if ((a in only_prev and b in only_cur)
                    or (b in only_prev and a in only_cur)):
                return False
        if (only_prev & _RESTATE_NUMBER_WORDS
                and only_cur & _RESTATE_NUMBER_WORDS):
            return False
        if pw <= cw or cw <= pw:
            return True
        return len(shared) / len(pw | cw) >= 0.6
    except Exception:
        return False


def detect_tone(user_text: str, prev_user_text: str | None = None,
                prev_turn_failed: bool = False) -> str | None:
    """Classify the emotional tone of a transcribed user utterance.

    Returns one of: 'frustrated' | 'stressed' | 'rushed' | 'tired' |
    'playful' | 'excited' | 'late_night', or None when nothing notable is
    detected (= default calm register, no system-prompt modification).

    `prev_user_text` is the user's PREVIOUS utterance (or None); when it shares
    a majority of content words with this one the user is restating themselves,
    a strong frustration signal. `prev_turn_failed` says JARVIS's previous turn
    did not do what was asked; it defaults to False because the live caller
    has no such signal today. Either one turns a bare 'still' into
    frustration (see _CONTEXT_GATED_FRUSTRATION_WORDS); without them 'still'
    is neutral. 'late_night' is a time-of-day fallback applied only when no
    other tone fires, so explicit signals still win after midnight, and only
    while NIGHT_QUIET_ENABLED is on (see _late_night_tone_applies).

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
    similar_to_last = is_restatement(clean, prev_user_text)

    # 'still' is push-back only in context. A restatement is frustrated on
    # its own (similar_to_last below), so the one extra case is a 'still'
    # right after a failed turn. Alone it is a status report, not frustration.
    gated_pushback = (prev_turn_failed
                      and _has_any(_CONTEXT_GATED_FRUSTRATION_WORDS))

    # Priority: frustrated > excited > stressed > rushed > tired > playful >
    # late_night (time-based fallback). Frustration trumps stress because the
    # response strategy differs; excited fires before stressed because both can
    # carry exclamation marks but excitement pairs them with positive markers
    # and no swearing / clipped-imperative pattern.
    if (has_repeat or similar_to_last or gated_pushback
            or (has_swear and (has_urgency or is_clipped))):
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

    if _late_night_tone_applies():
        return "late_night"

    return None


_TONE_HINTS: dict[str, str] = {
    # Rewritten 2026-09-29. The old hint ended "Do NOT explain. Act." -- an
    # order to GUESS: a frustrated owner describing a fault got a random
    # neighbouring action instead of the question or check that would find
    # the cause. Still terse, still no defending the last attempt, but the
    # next move is now a clarifying question or the matching diagnostic.
    "frustrated": (
        "USER_TONE: frustrated — the user appears to be repeating "
        "themselves or pushing back. Skip pleasantries and 'sir' filler, "
        "acknowledge the misfire in a few words ('Apologies, sir.') and do "
        "NOT defend the previous attempt. Do NOT guess at a different "
        "action: if what he wants is unclear, ask ONE short clarifying "
        "question; if he is describing a fault, run the matching "
        "diagnostic or status check and report what it shows. Act "
        "directly only when the request is unambiguous."
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
