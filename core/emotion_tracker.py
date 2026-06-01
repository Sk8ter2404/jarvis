"""
core/emotion_tracker.py

Classifies the user's emotional state from a transcribed utterance into
five labels — stressed, frustrated, excited, focused, tired — using word
choice, sentence length, and optional prosody hints supplied by the
caller (e.g. mic RMS level, speech rate). Returns a per-turn system-
prompt addendum and the matching TTS preset name.

This is a self-contained classifier — no LLM call, no model load. It runs
on every utterance, so heuristics must stay cheap. Designed to slot
alongside bobert_companion.detect_tone() rather than replace it: the
existing detector covers stressed/frustrated/rushed/tired/playful/excited
plus a late-night fallback; this tracker adds an explicit `focused`
bucket (engineering-vocabulary + measured cadence) and a clean mapping
to the five canonical Paul-Bettany-range TTS presets the spec calls out:
'calm' for stressed, 'amused' for excited, 'concerned' for tired, etc.

Public API:

    classify_emotion(text, prosody=None) -> EmotionResult
    system_prompt_addendum(label)        -> str
    tts_preset_for(label)                -> str | None

EmotionResult fields:
    label    : 'stressed' | 'frustrated' | 'excited' | 'focused' | 'tired' | None
    addendum : per-turn LLM system-prompt extension (may be empty)
    tts_preset : TTS preset name to feed _resolve_tts_preset (may be None)
    reason   : short debug string explaining which rule fired
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
#  WORD-CHOICE LEXICONS
# ──────────────────────────────────────────────────────────────────────────

# Actual profanity. Used both as a stress signal and (combined with a
# clipped imperative) as a frustration signal — only real swearing escalates
# a short command into "frustrated", not generic emphatic words like "stop".
_SWEAR_WORDS: tuple[str, ...] = (
    "damn", "damnit", "dammit", "shit", "crap", "fuck", "fucking",
    "bloody", "hell", "screw", "freaking", "frigging",
)

# Emphatic / intensifying vocabulary. Counts toward "stressed" but never
# toward "frustrated" on its own — "Stop it now!" is stress, not frustration.
_STRESS_INTENSIFIERS: tuple[str, ...] = (
    "stop", "wait", "no no no", "ugh", "argh",
)

# Combined view used everywhere we previously matched _STRESS_WORDS.
_STRESS_WORDS: tuple[str, ...] = _SWEAR_WORDS + _STRESS_INTENSIFIERS

# Frustration markers: phrases that indicate the user is repeating themselves
# or pushing back on a prior failure.
_FRUSTRATION_PHRASES: tuple[str, ...] = (
    "i said", "i told you", "again", "still", "you keep", "you're not",
    "you are not listening", "not listening", "just do", "for the last time",
    "we already", "i already", "you missed", "you didn't", "you did not",
    "that's not what", "that is not what", "wrong", "incorrect",
)

# Excitement markers: positive, high-energy vocabulary.
_EXCITEMENT_PHRASES: tuple[str, ...] = (
    "yes", "yess", "yesss", "finally", "let's go", "lets go", "woo", "woohoo",
    "amazing", "awesome", "incredible", "perfect", "brilliant", "fantastic",
    "love it", "love this", "epic", "killer", "nice", "yay", "wow",
    "great job", "well done", "excellent",
)

# Focused markers: engineering/work vocabulary delivered in measured tone.
# These shift the assistant into a precise, minimal-chatter register.
_FOCUSED_PHRASES: tuple[str, ...] = (
    "let's", "lets", "let me", "let us",
    "implement", "implementing", "refactor", "refactoring", "rewrite",
    "debug", "debugging", "fix the", "trace", "stack trace", "the bug",
    "the issue", "the problem", "investigate", "diagnose",
    "merge", "deploy", "build", "compile", "tests", "test the",
    "review the", "audit the", "the function", "the method", "the class",
    "the variable", "the parameter", "the argument", "the import",
    "in the file", "in the module", "the codebase", "the repo",
    "step through", "walk through", "go through",
)

# Tired markers: fatigue and end-of-session vocabulary.
_TIRED_PHRASES: tuple[str, ...] = (
    "i'm tired", "i am tired", "exhausted", "knackered", "wiped",
    "going to bed", "going to sleep", "off to bed", "bedtime",
    "good night", "goodnight", "g'night", "calling it a night",
    "wrapping up", "done for the night", "done for today",
    "long day", "yawn", "drained", "fading",
)

# Clipped imperatives — short commands that, combined with exclamation or
# repetition, signal stress.
_CLIPPED_IMPERATIVES: tuple[str, ...] = (
    "stop", "cancel", "abort", "now", "quiet", "shut up", "enough",
    "kill it", "kill that", "end it",
)

# Late-night fallback hours (24h local).
_LATE_NIGHT_START_HOUR = 22
_LATE_NIGHT_END_HOUR = 5


# ──────────────────────────────────────────────────────────────────────────
#  TTS PRESET MAP
#
#  Maps each emotion label to the TTS preset name that should drive
#  prosody for the assistant's reply this turn. Names match
#  bobert_companion._TTS_EMOTION_PRESETS — except 'calm', which this
#  module declares as a new preset and which bobert_companion registers
#  alongside the existing entries. The intent (per spec) is:
#
#      stressed  → calm     (slow, low, soft — defuse the user's tension)
#      frustrated → calm    (same — never add energy to a frustrated user)
#      excited   → amused   (bright, slightly quicker, gentle smile)
#      focused   → briefing (measured, news-anchor cadence — work mode)
#      tired     → concerned (softer + slower, a touch lower)
# ──────────────────────────────────────────────────────────────────────────

TTS_PRESETS: dict[str, str] = {
    "stressed":   "calm",
    "frustrated": "calm",
    "excited":    "amused",
    "focused":    "briefing",
    "tired":      "concerned",
}


# ──────────────────────────────────────────────────────────────────────────
#  SYSTEM-PROMPT ADDENDA
#
#  One short, behavioural directive per label. These are concatenated onto
#  the base system prompt for the single turn in which the emotion fires —
#  they shape reply LENGTH and REGISTER, not the underlying personality.
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_HINTS: dict[str, str] = {
    "stressed": (
        "USER_EMOTION: stressed — sir is under pressure. Reply in ONE "
        "short sentence. Skip pleasantries, skip 'sir' filler if it "
        "doesn't land naturally. Lead with the action, not the "
        "acknowledgement. Stay calm; do not match the panic."
    ),
    "frustrated": (
        "USER_EMOTION: frustrated — sir appears to be repeating "
        "themselves or pushing back on a prior failure. Briefly "
        "acknowledge the misfire ('Apologies, sir — trying again.') "
        "and immediately try a different approach. Do NOT defend the "
        "previous attempt. Do NOT explain. Act."
    ),
    "excited": (
        "USER_EMOTION: excited — match sir's energy without losing the "
        "dry British register. Affirm in one line ('Splendid, sir.' / "
        "'Quite, sir.' / 'A bold choice, if I may say so.') and feel "
        "free to add a wry quip. Do NOT damp the mood with caveats."
    ),
    "focused": (
        "USER_EMOTION: focused — sir is in work mode. Be precise, "
        "concise, and technical. Skip filler. Mirror the engineering "
        "vocabulary already in use. One or two factual sentences; no "
        "dry humour, no preamble, no 'shall I' framing — just the "
        "answer or the action."
    ),
    "tired": (
        "USER_EMOTION: tired — speak softer and shorter than usual. "
        "Avoid stat dumps and dry humour. One gentle sentence is "
        "plenty. If sir is signing off for the night, acknowledge "
        "warmly and let them go."
    ),
}


# ──────────────────────────────────────────────────────────────────────────
#  PROSODY HINT TYPE
#
#  Callers may optionally pass measured prosody features. None of them
#  are required — the classifier falls back to text-only heuristics when
#  they're absent. Fields:
#
#    rms       : normalized RMS amplitude of the utterance, 0.0–1.0+
#    rms_baseline : the caller's running baseline so we can detect a
#                   sudden volume spike (signals stress/excitement) or
#                   a drop (signals tired/focused).
#    speech_rate_wps : words-per-second from STT timing (lower = tired
#                      or focused; higher = stressed or excited).
#    hour      : 0–23 local hour, used for the tired-time fallback.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ProsodyHints:
    rms: Optional[float] = None
    rms_baseline: Optional[float] = None
    speech_rate_wps: Optional[float] = None
    hour: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────
#  CLASSIFICATION RESULT
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    label: Optional[str]
    addendum: str
    tts_preset: Optional[str]
    reason: str

    def __bool__(self) -> bool:  # truthy iff something was detected
        return self.label is not None


_EMPTY = EmotionResult(label=None, addendum="", tts_preset=None, reason="")


# ──────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation except apostrophes, collapse whitespace."""
    s = re.sub(r"[^a-z' ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _has_any_phrase(text: str, phrases) -> Optional[str]:
    """Word-boundary match against any phrase. Returns the matched phrase
    (for the reason string) or None."""
    for p in phrases:
        if re.search(r"\b" + re.escape(p) + r"\b", text):
            return p
    return None


def _in_late_night_band(hour: int) -> bool:
    """22:00–04:59 local, wrap-around."""
    return hour >= _LATE_NIGHT_START_HOUR or hour < _LATE_NIGHT_END_HOUR


def _current_hour() -> int:
    import datetime as _dt
    return _dt.datetime.now().hour


# ──────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def classify_emotion(
    text: str,
    prosody: Optional[ProsodyHints] = None,
) -> EmotionResult:
    """Classify the emotional state of one user utterance.

    Returns an EmotionResult. When no emotion is detected, the result's
    `label` is None and both addendum and tts_preset are empty — the
    caller should fall through to the default register.

    Priority order (first match wins):
        frustrated > stressed > excited > tired > focused > (late_night→tired)

    'Frustrated' beats 'stressed' because the response strategy is
    different — frustrated wants an acknowledgement + retry; stressed
    just wants quiet competence.

    'Focused' is positioned below the high-arousal emotions because a
    focused user with engineering vocabulary AND profanity is still
    frustrated, not focused — the negative signal dominates.
    """
    if not text:
        return _EMPTY

    clean = _normalize(text)
    if not clean:
        return _EMPTY

    raw = text.strip()
    excl_count = raw.count("!")
    n_words = len(clean.split())
    sentence_len = len(raw)

    p = prosody or ProsodyHints()
    loud_spike = _is_loud_spike(p)
    quiet_drop = _is_quiet_drop(p)
    fast_rate  = (p.speech_rate_wps is not None and p.speech_rate_wps >= 3.8)
    slow_rate  = (p.speech_rate_wps is not None and p.speech_rate_wps <= 1.6)

    # ── frustrated ────────────────────────────────────────────────
    # Word-choice signal: repetition/blame phrases ("I said", "still", etc.)
    # OR actual profanity combined with a clipped imperative. Generic
    # emphatic words ("stop", "wait") are NOT enough on their own — those
    # fall through to the stressed branch below.
    frust_phrase  = _has_any_phrase(clean, _FRUSTRATION_PHRASES)
    swear_phrase  = _has_any_phrase(clean, _SWEAR_WORDS)
    stress_phrase = _has_any_phrase(clean, _STRESS_WORDS)
    clipped = (n_words <= 3) and (_has_any_phrase(clean, _CLIPPED_IMPERATIVES) is not None)
    if frust_phrase or (swear_phrase and clipped):
        return _build(
            "frustrated",
            f"phrase={frust_phrase or swear_phrase!r} clipped={clipped}",
        )

    # ── stressed ──────────────────────────────────────────────────
    # Word-choice: profanity or stress vocabulary.
    # Sentence-length: clipped (≤3 words) with at least one exclamation.
    # Prosody: sudden volume spike or notably fast cadence.
    if stress_phrase or excl_count >= 2 or (clipped and excl_count >= 1):
        return _build(
            "stressed",
            f"phrase={stress_phrase!r} excl={excl_count} clipped={clipped}",
        )
    if loud_spike and (excl_count >= 1 or clipped):
        return _build("stressed", "prosody=loud_spike")
    if fast_rate and (stress_phrase or excl_count >= 1):
        return _build("stressed", "prosody=fast_rate")

    # ── excited ───────────────────────────────────────────────────
    # Word-choice: positive high-energy markers WITHOUT swearing.
    # Sentence-length irrelevant — short or long can both be excited.
    # Prosody: loud spike paired with positive vocabulary.
    exc_phrase = _has_any_phrase(clean, _EXCITEMENT_PHRASES)
    if exc_phrase and not stress_phrase and not clipped:
        return _build("excited", f"phrase={exc_phrase!r}")
    if excl_count >= 2 and not stress_phrase:
        return _build("excited", f"excl={excl_count}")
    if loud_spike and exc_phrase:
        return _build("excited", "prosody=loud_spike+phrase")

    # ── tired ─────────────────────────────────────────────────────
    # Word-choice: explicit fatigue phrases.
    # Prosody: slow cadence + quiet drop.
    tired_phrase = _has_any_phrase(clean, _TIRED_PHRASES)
    if tired_phrase:
        return _build("tired", f"phrase={tired_phrase!r}")
    if slow_rate and quiet_drop:
        return _build("tired", "prosody=slow+quiet")

    # ── focused ───────────────────────────────────────────────────
    # Word-choice: engineering vocabulary.
    # Sentence-length: usually moderate (4+ words) and declarative
    # (no exclamation, no all-caps).
    # Prosody: stable, neutral, not loud.
    focus_phrase = _has_any_phrase(clean, _FOCUSED_PHRASES)
    if (
        focus_phrase
        and excl_count == 0
        and n_words >= 4
        and not stress_phrase
    ):
        return _build("focused", f"phrase={focus_phrase!r} words={n_words}")

    # ── tired fallback (time-of-day) ──────────────────────────────
    # No explicit textual or prosodic signal, but it's past sir's
    # normal hours. Treated as tired for delivery (softer / shorter),
    # but only when the utterance is short and declarative — at 2 AM,
    # a long focused engineering ask should stay 'focused', not be
    # downgraded to a tired register.
    hour = p.hour if p.hour is not None else _current_hour()
    if _in_late_night_band(hour) and n_words <= 8 and excl_count == 0:
        return _build("tired", f"time=late_night hour={hour}")

    return _EMPTY


# ──────────────────────────────────────────────────────────────────────────
#  CONVENIENCE ACCESSORS
# ──────────────────────────────────────────────────────────────────────────

def system_prompt_addendum(label: Optional[str]) -> str:
    """Per-turn LLM system-prompt extension for `label`, or '' when
    nothing applies. Public so callers that already have a label (from a
    cache, a follow-up call, etc.) can replay the addendum without
    re-classifying."""
    if not label:
        return ""
    hint = _SYSTEM_PROMPT_HINTS.get(label)
    if not hint:
        return ""
    return "\n\n[Per-turn emotion hint]\n" + hint


def tts_preset_for(label: Optional[str]) -> Optional[str]:
    """TTS preset name (calm/amused/briefing/concerned/...) for `label`,
    or None when no override applies."""
    if not label:
        return None
    return TTS_PRESETS.get(label)


# ──────────────────────────────────────────────────────────────────────────
#  INTERNALS
# ──────────────────────────────────────────────────────────────────────────

def _build(label: str, reason: str) -> EmotionResult:
    return EmotionResult(
        label=label,
        addendum=system_prompt_addendum(label),
        tts_preset=tts_preset_for(label),
        reason=reason,
    )


def _is_loud_spike(p: ProsodyHints) -> bool:
    """RMS notably above baseline. Threshold is intentionally permissive
    since baselines drift with room noise; we only need to catch obvious
    shouts and emphatic punches, not subtle inflection."""
    if p.rms is None:
        return False
    base = p.rms_baseline if p.rms_baseline is not None else 0.02
    if base <= 0:
        base = 0.02
    return p.rms >= base * 1.8


def _is_quiet_drop(p: ProsodyHints) -> bool:
    """RMS notably below baseline — soft, fading delivery."""
    if p.rms is None:
        return False
    base = p.rms_baseline if p.rms_baseline is not None else 0.02
    if base <= 0:
        base = 0.02
    return p.rms <= base * 0.55


# ──────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("Stop it now!",                                  "stressed"),
        ("I said open the file",                          "frustrated"),
        ("This is amazing, look at that!",                "excited"),
        ("Let's refactor the dispatcher in bobert_companion.py", "focused"),
        ("I'm exhausted, going to bed",                   "tired"),
        ("What's the weather today",                      None),
        ("Hello sir how are you doing today",             None),
        ("Wait wait wait STOP!",                          "stressed"),
        ("Fix the bug in the click handler",              "focused"),
    ]
    width = max(len(t) for t, _ in cases) + 2
    for text, expected in cases:
        r = classify_emotion(text)
        ok = "OK " if r.label == expected else "BAD"
        print(f"  {ok}  {text!r:<{width}} -> {r.label!r:<13} preset={r.tts_preset!r:<12} ({r.reason})")
