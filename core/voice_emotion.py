"""core/voice_emotion.py — higher-level voice mood router.

Fuses the text-tone classifier (core.tone_detector.detect_tone) with
time-of-day and an 'excited' detector into ONE stable mood label that drives
both the per-turn LLM register (reply length / signature phrasing) AND the TTS
prosody preset downstream. detect_tone() classifies the text alone; this router
adds the 'late_night' bucket (doesn't fall out of pure text), an 'excited'
bucket (which detect_tone calls 'playful' or misses), and the stable label.

Extracted from bobert_companion.py. Pure: route_voice_emotion() takes the
previous user utterance as a parameter (prev_user_text) so detect_tone can spot
cross-turn repetition without this module reaching into conversation_history.
bobert_companion keeps a thin route_voice_emotion(user_text, now=None) wrapper
that supplies prev from the live history. The _last_voice_route cache stays in
the monolith (runtime state).
"""
from __future__ import annotations

import datetime
import re

from core.tone_detector import (
    detect_tone,
    _late_night_tone_applies,
    is_repeat_request,
    _EXCITEMENT_PHRASES,
    _STRESS_SWEAR_WORDS,
)

VOICE_EMOTION_ROUTER_ENABLED = True

# Per-mood LLM system-prompt addenda. These shape REPLY LENGTH and register;
# TTS prosody is handled separately by _USER_TONE_TTS using the same mood label
# as the lookup key.
_VOICE_MOOD_HINTS: dict[str, str] = {
    "stressed": (
        "USER_TONE: stressed — speak as though defusing tension. ≤1 short "
        "sentence. Calm, low-key, no humour, no embellishments. Lead with "
        "the action or the headline. 'On it, sir.' / 'Right away.' If "
        "there is bad news, lead with 'I'm afraid' and keep it factual."
    ),
    "late_night": (
        "USER_TONE: late_night — it is past 22:00 (or before 05:00). "
        "Speak softer, slower, shorter. One gentle sentence. No stat "
        "dumps, no dry quips, no 'I've taken the liberty of'. "
        "Acknowledge the hour naturally if it fits ('A bit late for "
        "that, sir, but very well.')."
    ),
    "excited": (
        "USER_TONE: excited — match sir's energy with dry wit. Up to 2 "
        "sentences. A signature quip is welcome ('Quite, sir.' / 'A bold "
        "choice, if I may say so.' / 'Well, that escalated quickly, "
        "sir.'). Stay in character — JARVIS is amused, not bouncy."
    ),
    "casual": "",
}


def _detect_excited(user_text: str) -> bool:
    """True when the utterance carries high-energy positive markers — at least
    one excitement phrase OR ≥2 exclamation marks without swearing. Kept
    separate from detect_tone() so the router can promote excitement above
    'playful' (which is gentler than what an excited user wants).

    A request to repeat ("Say that again!!", "What?! Say that again!") is
    never excitement: he did not hear JARVIS, and its '!!' is emphasis, not
    energy. Same rule as detect_tone / emotion_tracker.classify_emotion
    (core.tone_detector.is_repeat_request), so all three agree and the
    repeated line is not answered with a quip in the faster 'excited'
    voice."""
    if not user_text:
        return False
    clean = re.sub(r"[^a-z' ]+", " ", user_text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean or is_repeat_request(clean):
        return False
    for phrase in _EXCITEMENT_PHRASES:
        if re.search(r'\b' + re.escape(phrase) + r'\b', clean):
            return True
    if user_text.count("!") >= 2:
        for sw in _STRESS_SWEAR_WORDS:
            if re.search(r'\b' + re.escape(sw) + r'\b', clean):
                return False
        return True
    return False


def route_voice_emotion(user_text: str, now: float | None = None,
                        prev_user_text: str | None = None) -> dict:
    """Classify (text tone + time-of-day) → a voice mood route.

    Returns {'mood': 'stressed'|'late_night'|'excited'|'casual',
             'addendum': per-turn LLM system-prompt extension (may be empty)}.

    Priority: stressed > excited > late_night > casual. A stressed late-night
    utterance gets stress protocol (terse, calm) not late-night softness;
    excited beats late-night so a 2 AM 'yes!! finally!!' still gets an energy
    match. `prev_user_text` feeds detect_tone's cross-turn repetition check.
    """
    if not VOICE_EMOTION_ROUTER_ENABLED:
        return {"mood": "casual", "addendum": ""}

    text = (user_text or "").strip()
    tone = detect_tone(text, prev_user_text=prev_user_text) if text else None
    excited = _detect_excited(text) if text else False

    # Time-of-day branch. The helper expects a datetime, so build one from
    # `now` when a caller passes a Unix timestamp (tests do this; the live
    # path leaves now=None and the helper uses the current clock). Always
    # False while NIGHT_QUIET_ENABLED is off (core/night_quiet.py); a 'tired'
    # tone from the owner's own words still routes to late_night below.
    when = datetime.datetime.fromtimestamp(now) if now is not None else None
    late_night = _late_night_tone_applies(when)

    # Bucket detect_tone()'s 7-label output into the 4 router moods.
    # 'frustrated' folds into 'stressed' (same calm + terse strategy).
    # 'rushed' and 'playful' stay 'casual' so synthesise() falls back to the
    # fine-grained tone rather than being downgraded to the bad_news cadence.
    if tone in ("frustrated", "stressed"):
        mood = "stressed"
    elif tone == "excited" or excited:
        mood = "excited"
    elif tone == "late_night" or tone == "tired" or late_night:
        mood = "late_night"
    else:
        mood = "casual"

    hint = _VOICE_MOOD_HINTS.get(mood, "")
    addendum = ("\n\n[Per-turn voice tone]\n" + hint) if hint else ""

    return {"mood": mood, "addendum": addendum}
