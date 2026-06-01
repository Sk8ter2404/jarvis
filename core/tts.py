"""
core/tts.py

Tone-aware TTS layer. Sits between the LLM's reply and edge-tts and picks
a (rate, pitch, gain) preset based on:

    1. an explicit [wry] marker on the reply (dry-humour deliveries)
    2. an explicit [intent:xxx] tag (LLM-chosen prosody)
    3. the five-label emotional state from core/emotion_tracker
    4. the voice_emotion_router user-tone
    5. signature phrases in the outgoing text ("I'm afraid…", etc.)

The actual edge-tts call and audio decoding live in bobert_companion.py —
this module owns the *selection* and the two cosmetic helpers
(parse_wry_tag + split_for_wry_pause) so the per-spec wry preset can land
a beat before the punchline.

Per spec (jarvis_todo.md line 120):
    calm + slower  when the user is stressed/frustrated
    faster/clipped when the user is excited
    default dry register otherwise
    'concerned' preset → rate=-15%, pitch=-4Hz
    'wry' preset      → rate=+3%, pitch=+1Hz, slight pause before the punchline
"""

from __future__ import annotations

import datetime
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
#  MUTE_TTS — speakers off, pipeline still alive
#
#  When the MUTE_TTS env var is set (any truthy value), the playback layer
#  in bobert_companion.play_with_lipsync skips the audio device write but
#  still runs synthesise() end-to-end. Used by:
#    • blue/green staging — green must not fight prod for the speakers,
#      but we still want a real edge-tts render so a broken voice fails
#      the smoke gate.
#    • headless test runs / CI — exercise the prosody pipeline without
#      requiring an output device.
#  The setting is read on every call (not cached) so an operator can flip
#  it mid-session via `os.environ["MUTE_TTS"] = "1"`.
# ──────────────────────────────────────────────────────────────────────────

_MUTE_TTS_ENV = "MUTE_TTS"


def is_muted() -> bool:
    """True if MUTE_TTS is set (any non-empty/non-zero value)."""
    raw = (os.environ.get(_MUTE_TTS_ENV) or "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


# ──────────────────────────────────────────────────────────────────────────
#  TTS RENDER CACHE
#
#  edge-tts is a ~0.5-1.5 s network round-trip, and JARVIS re-synthesises the
#  SAME short confirmations constantly — "Yes, sir?", "At your service.",
#  "Done." This LRU caches the decoded audio per (voice, rate, pitch, text) so
#  repeated phrases play instantly. Keyed on the PRE-gain render (the caller in
#  bobert_companion.synthesise re-applies per-mood gain afterwards), so one
#  cached render is reusable across moods. Only short phrases are cached — long
#  unique replies would just churn it. Thread-safe, and it copies on store AND
#  on return so a caller that mutates the buffer in place (e.g. applies gain
#  with gain==1.0 and edits the array) can never corrupt a cached entry.
# ──────────────────────────────────────────────────────────────────────────

TTS_CACHE_MAX_ENTRIES = 96
TTS_CACHE_MAX_CHARS   = 80     # only cache short, repeatable confirmations

_tts_cache: "OrderedDict[str, tuple]" = OrderedDict()
_tts_cache_lock = threading.Lock()
_tts_cache_stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}


def tts_cache_key(text: str, voice: str, rate: str, pitch: str) -> str:
    return f"{voice}|{rate}|{pitch}|{(text or '').strip()}"


def _copy_audio(audio):
    """Defensive copy of an audio buffer. Works for numpy arrays and plain
    lists alike (both expose .copy()), so core/tts stays numpy-free."""
    cp = getattr(audio, "copy", None)
    return cp() if callable(cp) else audio


def tts_cache_get(text: str, voice: str, rate: str, pitch: str):
    """Return a cached (audio_copy, sr) for this key, or None on miss."""
    key = tts_cache_key(text, voice, rate, pitch)
    with _tts_cache_lock:
        hit = _tts_cache.get(key)
        if hit is None:
            _tts_cache_stats["misses"] += 1
            return None
        _tts_cache.move_to_end(key)
        _tts_cache_stats["hits"] += 1
        audio, sr = hit
        return _copy_audio(audio), sr


def tts_cache_put(text: str, voice: str, rate: str, pitch: str, audio, sr) -> bool:
    """Store a render. No-op (returns False) for empty/over-long text so the
    cache only ever holds short, repeatable phrases."""
    t = (text or "").strip()
    if not t or len(t) > TTS_CACHE_MAX_CHARS:
        return False
    key = tts_cache_key(t, voice, rate, pitch)
    with _tts_cache_lock:
        _tts_cache[key] = (_copy_audio(audio), sr)
        _tts_cache.move_to_end(key)
        _tts_cache_stats["stores"] += 1
        while len(_tts_cache) > TTS_CACHE_MAX_ENTRIES:
            _tts_cache.popitem(last=False)
            _tts_cache_stats["evictions"] += 1
    return True


def tts_cache_stats() -> dict:
    """Snapshot of cache counters + entries + hit_rate (for diagnostics)."""
    with _tts_cache_lock:
        s = dict(_tts_cache_stats)
        s["entries"] = len(_tts_cache)
    total = s["hits"] + s["misses"]
    s["hit_rate"] = round(s["hits"] / total, 3) if total else 0.0
    return s


def tts_cache_clear() -> None:
    """Drop all cached renders AND reset the counters — a clean slate."""
    with _tts_cache_lock:
        _tts_cache.clear()
        for k in _tts_cache_stats:
            _tts_cache_stats[k] = 0


# ──────────────────────────────────────────────────────────────────────────
#  PRESETS THIS LAYER OWNS
#
#  These two named constants are the per-spec 'concerned' / 'wry' values,
#  kept as standalone exports for any direct importer. The full prosody
#  palette now lives in the _TTS_EMOTION_PRESETS table further down (it was
#  consolidated into this module out of bobert_companion).
# ──────────────────────────────────────────────────────────────────────────

CONCERNED_PRESET: dict[str, object] = {"rate": "-15%", "pitch": "-4Hz", "gain": 0.92}
WRY_PRESET:       dict[str, object] = {"rate": "+3%",  "pitch": "+1Hz", "gain": 1.0}

# Silence to splice between setup and punchline when wry. ~180 ms reads as a
# deliberate beat without sounding like a stall.
WRY_PAUSE_MS: int = 180


# ──────────────────────────────────────────────────────────────────────────
#  EMOTION / TONE → PRESET MAPPING
#
#  Mirrors emotion_tracker.TTS_PRESETS but kept local so this module is
#  importable on its own. 'frustrated' shares 'calm' with 'stressed' — per
#  spec we never escalate energy when the user is already upset.
# ──────────────────────────────────────────────────────────────────────────

_EMOTION_TO_PRESET: dict[str, str] = {
    "stressed":   "calm",
    "frustrated": "calm",
    "excited":    "amused",
    "focused":    "briefing",
    "tired":      "concerned",
}

_USER_TONE_TO_PRESET: dict[str, str] = {
    "stressed":   "calm",
    "frustrated": "calm",
    "late_night": "concerned",
    "tired":      "concerned",
    "excited":    "amused",
    "rushed":     "confirmation",
    "playful":    "amused",
}


# ──────────────────────────────────────────────────────────────────────────
#  [wry] TAG PARSER
#
#  Mirrors the [intent:xxx] parser in bobert_companion. Anchored to the
#  start of the string so the LLM has to LEAD with the marker — that way
#  a reply that merely contains the word 'wry' doesn't trigger the preset.
# ──────────────────────────────────────────────────────────────────────────

_WRY_TAG_RE = re.compile(r"^\s*\[\s*wry\s*\]\s*", re.IGNORECASE)


def parse_wry_tag(text: str) -> Tuple[bool, str]:
    """If `text` starts with [wry], return (True, text_with_tag_stripped).
    Otherwise return (False, text)."""
    if not text:
        return False, text
    m = _WRY_TAG_RE.match(text)
    if not m:
        return False, text
    return True, text[m.end():]


# ──────────────────────────────────────────────────────────────────────────
#  SPLIT FOR PUNCHLINE PAUSE
#
#  The pause-before-the-punchline behaviour is implemented by splitting the
#  reply on its final natural break, synthesising each half separately, and
#  splicing WRY_PAUSE_MS of silence between them. We try sentence ends
#  first (period / exclamation / question mark); fall back to the last
#  comma; bail out if neither leaves at least 2 words on each side, so we
#  never marrow a useful clause down to a single tag word.
# ──────────────────────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"[.!?…]\s+")


def split_for_wry_pause(text: str) -> Tuple[str, Optional[str]]:
    """Split `text` into (setup, punchline) at the latest natural break.

    Returns (text, None) when no good split is available — caller should
    render the whole reply in a single pass and skip the pause."""
    if not text:
        return text, None

    matches = list(_SENTENCE_END_RE.finditer(text))
    for m in reversed(matches):
        cut = m.end()
        head = text[:cut].strip()
        tail = text[cut:].strip()
        if head and tail and len(head.split()) >= 2 and len(tail.split()) >= 2:
            return head, tail

    last_comma = text.rfind(",")
    if last_comma > 0:
        head = text[:last_comma].strip()
        tail = text[last_comma + 1:].strip()
        if head and tail and len(head.split()) >= 2 and len(tail.split()) >= 2:
            return head, tail

    return text, None


# ──────────────────────────────────────────────────────────────────────────
#  CONTEXT-AWARE PRESET DETECTORS
#
#  Three small detectors the caller can feed into select_preset() (and the
#  legacy bobert_companion._resolve_tts_preset wrapper) to pick a preset
#  from live context — late-night quiet, vocal stress, panic keywords —
#  rather than only the model's [intent:xxx] tag or last-sampled mood.
#
#  These all degrade silently to False on any failure: stale or missing
#  files, missing audio processor, empty input strings. The whole point is
#  to layer prosody on top of the existing pipeline, not gate it.
# ──────────────────────────────────────────────────────────────────────────

LATE_HOUR_THRESHOLD_HOUR: int = 23     # 11 PM
LATE_HOUR_STATE_MAX_AGE_S: float = 3600.0  # last_trigger older than 1 h is stale
STRESS_RMS_THRESHOLD: float = 0.08     # peak RMS over the recent window

# Keywords the user might shout when something has actually gone wrong. Kept
# small and unambiguous — false positives are worse than misses because the
# brisk_alert preset is conspicuously different from a normal reply.
_EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "fuck", "shit", "help",
)
_EMERGENCY_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _EMERGENCY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Project root is two levels up from core/tts.py (core/ → JARVIS/).
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANTICIPATION_STATE_PATH = os.path.join(_PROJECT_DIR, "anticipation_state.json")


def detect_late_hour(
    *,
    state_path: str = _ANTICIPATION_STATE_PATH,
    now: Optional[datetime.datetime] = None,
) -> bool:
    """True when JARVIS should adopt the late-night hushed register.

    Returns True if EITHER:
      * the local hour is at/after LATE_HOUR_THRESHOLD_HOUR (23, i.e. 11 PM),
      * the anticipation engine recently fired a 'late_hour' trigger
        (state file's last_trigger == 'late_hour' AND younger than 1 h).

    Either signal alone is enough — the wall-clock check covers the case
    where anticipation hasn't fired yet, and the state-file check catches
    daylight-saving / clock-skew edge cases where the trigger fired but the
    local hour rolled back.
    """
    cur = now if now is not None else datetime.datetime.now()
    try:
        if cur.hour >= int(LATE_HOUR_THRESHOLD_HOUR):
            return True
    except Exception:
        pass

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return False
    if not isinstance(state, dict):
        return False
    last_trigger = state.get("last_trigger")
    last_at = state.get("last_proactive_at")
    if last_trigger != "late_hour":
        return False
    try:
        age = time.time() - float(last_at)
    except Exception:
        return False
    return 0.0 <= age <= float(LATE_HOUR_STATE_MAX_AGE_S)


def detect_stress_from_rms(
    peak_rms: Optional[float] = None,
    *,
    threshold: float = STRESS_RMS_THRESHOLD,
) -> bool:
    """True when the user's recent vocal energy is above `threshold`.

    `peak_rms` is the highest RMS observed over the recent VAD window. If
    None, we read it from core.audio_processor.recent_peak_rms() so callers
    that don't track RMS themselves still get a meaningful signal.
    """
    pr: float
    if peak_rms is None:
        try:
            from core.audio_processor import recent_peak_rms
            pr = float(recent_peak_rms())
        except Exception:
            return False
    else:
        try:
            pr = float(peak_rms)
        except (TypeError, ValueError):
            return False
    if pr <= 0.0:
        return False
    return pr > float(threshold)


def detect_emergency_keywords(user_text: Optional[str]) -> bool:
    """True when `user_text` contains an emergency keyword as a whole word.

    Anchored on word boundaries so 'helpful' does not trip 'help' and a
    chat about /shittake mushrooms/ doesn't trip 'shit'.
    """
    if not user_text:
        return False
    return _EMERGENCY_KEYWORD_RE.search(str(user_text)) is not None


def detect_context_preset(
    user_text: Optional[str] = None,
    peak_rms: Optional[float] = None,
    *,
    now: Optional[datetime.datetime] = None,
    state_path: str = _ANTICIPATION_STATE_PATH,
) -> Optional[str]:
    """Combined helper: return the highest-priority context preset name, or
    None when no context applies.

    Priority (per spec — brisk_alert is the most intrusive so a real
    emergency overrides quiet late-night cadence):
        1. emergency keywords  → 'brisk_alert'
        2. late hour           → 'hushed_late'
        3. vocal stress (RMS)  → 'calm_low'
    """
    if detect_emergency_keywords(user_text):
        return "brisk_alert"
    if detect_late_hour(state_path=state_path, now=now):
        return "hushed_late"
    if detect_stress_from_rms(peak_rms):
        return "calm_low"
    return None


# ──────────────────────────────────────────────────────────────────────────
#  PRESET SELECTION
#
#  Returns just the preset NAME. The caller (bobert_companion._resolve_tts_preset)
#  resolves it against its own _TTS_EMOTION_PRESETS dict — this keeps the
#  edge-tts-facing (rate, pitch, gain) triples in one place.
# ──────────────────────────────────────────────────────────────────────────

def select_preset(
    text: str = "",
    *,
    emotion_label: Optional[str] = None,
    user_tone: Optional[str] = None,
    intent: Optional[str] = None,
    wry: bool = False,
    text_emotion: Optional[str] = None,
) -> str:
    """Pick the preset name for the next utterance.

    Priority (first match wins):
        1. wry=True       → 'wry'
        2. intent         → caller maps the intent string itself
        3. emotion_label  → calm / amused / briefing / concerned
        4. user_tone      → same family
        5. text_emotion   → bad_news / warning / confirmation
        6. 'neutral'

    Exception kept consistent with the legacy _resolve_tts_preset: an
    'excited' / 'amused' preset never overrides a bad_news text — the
    outgoing line's gravitas wins.
    """
    if wry:
        return "wry"
    if intent:
        return intent

    if emotion_label:
        mapped = _EMOTION_TO_PRESET.get(emotion_label)
        if mapped and not (text_emotion == "bad_news" and mapped == "amused"):
            return mapped

    if user_tone:
        mapped = _USER_TONE_TO_PRESET.get(user_tone)
        if mapped and not (text_emotion == "bad_news" and mapped == "amused"):
            return mapped

    if text_emotion:
        return text_emotion

    return "neutral"


# ──────────────────────────────────────────────────────────────────────────
#  QUIP LAYER
#
#  When the LLM emits a short confirmation ("Right away, sir.", "Done.",
#  "Very good, sir."), randomly append a dry MCU-style aside so JARVIS
#  doesn't sound like a button-press confirmation prompt. Pools are keyed
#  by action category — destructive actions get pointed asides, music/UI
#  get gentler ones.
#
#  Defaults: 15% chance, only when the reply is < 6 words. Both knobs are
#  arguments so tests can pin them. random.Random is also injectable so
#  the unit test in __main__ is deterministic.
# ──────────────────────────────────────────────────────────────────────────

QUIP_PROBABILITY: float = 0.15
QUIP_MAX_WORDS:   int   = 6

# Action-name → category. Anything not listed routes to 'default'.
_DESTRUCTIVE_ACTIONS: frozenset[str] = frozenset({
    # Self-modification / process control
    "upgrade_jarvis", "start_overnight_upgrade", "restart_jarvis",
    "shutdown_jarvis", "reload_config",
    # OS-level destructive
    "shutdown", "restart", "reboot", "sleep_pc", "lock_pc", "log_off",
    "kill_process", "force_quit",
    # Window/app slaughter
    "close_window", "close_all_windows", "close_app",
    # File system
    "delete_file", "empty_recycle_bin", "wipe_screenshots",
})

_MEDIA_ACTIONS: frozenset[str] = frozenset({
    "play_music", "pause_music", "resume_music",
    "next_song", "previous_song",
    "apple_music", "spotify", "youtube_play", "youtube",
    "netflix", "prime_video", "disney_plus", "hulu", "max",
    "play_streaming", "play_unheard", "play_vibe", "skip_track",
    "media_next", "media_prev", "media_playpause",
    "volume_up", "volume_down", "mute", "set_volume",
})

_UI_ACTIONS: frozenset[str] = frozenset({
    "focus_window", "switch_window", "snap_window", "snap_left", "snap_right",
    "minimize", "maximize", "restore_window",
    "switch_monitor", "move_to_monitor", "move_window",
    "open_app", "launch_app", "open_url",
    "set_brightness", "dim_screen",
    "type_text", "press_key", "hotkey",
})

_QUIP_POOLS: dict[str, tuple[str, ...]] = {
    # Destructive: drier, more pointed. Stays this side of vulgar — JARVIS
    # is sardonic, not crass.
    "destructive": (
        "Try not to break anything, sir.",
        "I'll be here, sir, contemplating the inevitable.",
        "A bold choice, if I may.",
        "Do mind the load-bearing wall on the way out, sir.",
        "I'll keep the recovery image warm.",
        "Hope is, as ever, not a strategy.",
        "If anything's left standing, I'll let you know.",
        "Your funeral, sir. I'll catalogue the remains.",
    ),
    # Music/streaming: gentler, lightly amused.
    "media": (
        "Enjoy, sir.",
        "Tasteful as always, sir.",
        "An inspired choice, if I'm allowed an opinion.",
        "Try not to sing along too audibly.",
        "I'll keep the volume reasonable on your behalf.",
        "Mood-setting, sir. Quite right.",
    ),
    # Window juggling, focus shifts: neutral, observational.
    "ui": (
        "Tidying up, sir.",
        "Order, however briefly, restored.",
        "Cleaner already.",
        "Done — you can pretend you intended that.",
        "Pretending you meant to do that, sir.",
    ),
    # Everything else.
    "default": (
        "Noted, sir.",
        "I'll add it to the pile.",
        "Quietly judging, sir.",
        "If you say so, sir.",
        "Filed under 'inevitable', sir.",
        "I'll pretend I didn't see that.",
    ),
}


def classify_action_for_quip(action_name: Optional[str]) -> str:
    """Map an action name to one of: 'destructive', 'media', 'ui', 'default'."""
    if not action_name:
        return "default"
    n = action_name.strip().lower()
    if n in _DESTRUCTIVE_ACTIONS:
        return "destructive"
    if n in _MEDIA_ACTIONS:
        return "media"
    if n in _UI_ACTIONS:
        return "ui"
    return "default"


def jarvis_quip_layer(
    text: str,
    action_name: Optional[str] = None,
    *,
    rng: Optional[random.Random] = None,
    probability: float = QUIP_PROBABILITY,
    max_words: int = QUIP_MAX_WORDS,
) -> str:
    """Maybe append a dry MCU-style aside to a short confirmation.

    Returns `text` unchanged when:
        * `text` is empty
        * `text` has >= `max_words` words (not a short confirmation)
        * the probability roll fails

    The aside is drawn from `_QUIP_POOLS[classify_action_for_quip(action_name)]`.
    `rng` (a `random.Random`) is injectable so the unit test is deterministic.
    """
    if not text:
        return text
    if max_words is not None and len(text.split()) >= max_words:
        return text
    r = rng if rng is not None else random
    if r.random() >= probability:
        return text
    category = classify_action_for_quip(action_name)
    pool = _QUIP_POOLS.get(category, _QUIP_POOLS["default"])
    if not pool:
        return text
    aside = r.choice(pool)
    head = text.rstrip()
    if head and head[-1] not in ".?!…":
        head += "."
    return f"{head} {aside}"


# ──────────────────────────────────────────────────────────────────────────
#  EMOTION-AWARE PROSODY PRESETS + PRIORITY RESOLVER
#
#  Consolidated here from bobert_companion: the edge-tts (rate, pitch, gain)
#  triples AND the resolver that picks one now live in a single module.
#  bobert_companion re-exports _TTS_EMOTION_PRESETS / _INTENT_PRESETS /
#  _USER_TONE_TTS / _TTS_EMOTION_KEYWORDS / detect_tts_emotion off its own
#  namespace so _parse_intent_tag, the thin _resolve_tts_preset shim, the
#  night_owl_mode monkeypatch, and tools/audit_codebase.py keep working.
#
#  resolve_tts_preset() is pure: the monolith shim reads the live _last_*
#  singletons + recent vocal RMS and forwards them as arguments, so this
#  selection logic is unit-testable without booting the assistant.
# ──────────────────────────────────────────────────────────────────────────

# Emotion-aware TTS prosody. Edge-tts accepts `rate` (percent) and `pitch` (Hz)
# parameters on Communicate(); we scan the outgoing text for category-tagging
# phrases and pick a (rate, pitch) preset so JARVIS doesn't sound monotone.
_TTS_EMOTION_PRESETS: dict[str, dict[str, object]] = {
    "bad_news":     {"rate": "-10%", "pitch": "-6Hz", "gain": 1.0},   # slower + lower — gravitas for "I'm afraid"
    "warning":      {"rate": "-7%",  "pitch": "-3Hz", "gain": 1.0},   # slightly slower + slightly lower — "Slight problem"
    "confirmation": {"rate": "+6%",  "pitch": "+4Hz", "gain": 1.0},   # slightly faster + brighter — "Very good", "Complete"
    "neutral":      {"rate": "+0%",  "pitch": "+0Hz", "gain": 1.0},
    # User-tone-driven presets (selected by voice_emotion_router from
    # _last_user_tone[0] before TTS, taking precedence over the
    # text-keyword preset when both apply). See _USER_TONE_TTS.
    "late_night":   {"rate": "-12%", "pitch": "-4Hz", "gain": 0.65},  # quieter + slower for 2 AM
    "excited":      {"rate": "+9%",  "pitch": "+5Hz", "gain": 1.05},  # match sir's energy, dry-wit cadence
    # Intent-driven presets (selected by _parse_intent_tag from a leading
    # [intent:xxx] marker in the assistant reply). Highest priority — wins
    # over both _USER_TONE_TTS and detect_tts_emotion. See _INTENT_PRESETS.
    "urgent_alert": {"rate": "+14%", "pitch": "+8Hz", "gain": 1.05},  # faster + higher — true alerts, time-critical
    "dry_wit":      {"rate": "-3%",  "pitch": "-2Hz", "gain": 1.0},   # flat deadpan — wit lands on understatement
    "briefing":     {"rate": "-3%",  "pitch": "+0Hz", "gain": 1.0},   # measured, news-anchor cadence for briefings
    # Expanded delivery palette — gives the LLM a wider tonal vocabulary so the
    # same words read differently depending on the situation. Each entry has a
    # distinct (rate, pitch, gain) so they're audibly separable from the
    # confirmation/dry_wit/bad_news triad they sit between.
    "amused":         {"rate": "+3%",  "pitch": "+3Hz", "gain": 1.0},   # gentle smile in the voice — slightly brighter + quicker
    "calm":           {"rate": "-8%",  "pitch": "-3Hz", "gain": 0.88},  # slow + low + softer — defusing tension for a stressed sir (core/emotion_tracker preset)
    "conspiratorial": {"rate": "-7%",  "pitch": "-4Hz", "gain": 0.78},  # quieter + slower + low — secrets, asides ("Between you and me, sir…")
    "stern":          {"rate": "-6%",  "pitch": "-5Hz", "gain": 1.05},  # slower + lower + firm — strong pushback, "I'd strongly advise"
    "proud":          {"rate": "+1%",  "pitch": "+2Hz", "gain": 1.05},  # warmer + slightly bright — quiet pride, mild praise
    "concerned":      {"rate": "-15%", "pitch": "-4Hz", "gain": 0.92},  # softer + slower — per spec line 120, lingering worry, sits between calm and bad_news
    "wry":            {"rate": "+3%",  "pitch": "+1Hz", "gain": 1.0},   # dry-humour deliveries [wry] — slight beat inserted before the punchline by synthesise()
    # voice_mood layer (core/voice_mood_selector) — context-aware overrides
    # passed in explicitly via _speak(mood=...) or the [mood:xxx] tag. Picks
    # land between the [intent:xxx] override and the emotion-tracker /
    # user-tone fallbacks so a skill saying "this is urgent" beats the
    # passive read of sir's current mood.
    "calm_efficient": {"rate": "+0%",  "pitch": "+0Hz", "gain": 1.0},   # default baseline mood — same prosody as 'neutral', distinct name so the log makes the routing intent obvious
    "urgent_clipped": {"rate": "+12%", "pitch": "+6Hz", "gain": 1.05},  # faster + brighter + a touch hotter — alerts, VIP calls during work hours
    "dry_amused":     {"rate": "+2%",  "pitch": "+2Hz", "gain": 1.0},   # slight smile in the voice — Chappie recall, banter
    "concerned_soft": {"rate": "-12%", "pitch": "-3Hz", "gain": 0.85},  # slower + lower + softer — real self-diagnostic problem, care rather than alarm
    # Context-aware presets driven by detect_late_hour / detect_stress_from_rms /
    # detect_emergency_keywords (above). Sit between [intent:xxx] and the
    # emotion-tracker fallback in resolve_tts_preset so a live emergency keyword
    # on user input beats whatever passive tone was last sampled.
    "calm_low":     {"rate": "-10%", "pitch": "-3Hz", "gain": 0.85},  # high-pitch/fast-cadence sir → slow, low, soft to defuse
    "hushed_late":  {"rate": "-12%", "pitch": "-5Hz", "gain": 0.55},  # post-23:00 anticipation context → quiet, gentle
    "brisk_alert":  {"rate": "+15%", "pitch": "+7Hz", "gain": 1.10},  # emergency keyword detected → crisp, attentive, awake
}

# Map a leading [intent:xxx] tag emitted by the LLM → preset name. The LLM
# is instructed to start each reply with one of these tags (see
# BASE_SYSTEM_PROMPT's "Voice intent tagging" section). The tag is stripped
# from the text before TTS but routes the preset selection. Both 'urgent'
# and 'alert' map to the same urgent_alert preset since the LLM can pick
# whichever reads more naturally for the situation.
_INTENT_PRESETS: dict[str, str] = {
    "urgent":         "urgent_alert",
    "alert":          "urgent_alert",
    "bad_news":       "bad_news",
    "dry_wit":        "dry_wit",
    "wit":            "dry_wit",
    "dry":            "dry_wit",        # shorter alias — LLM tends to pick this naturally
    "confirmation":   "confirmation",
    "briefing":       "briefing",
    "amused":         "amused",
    "conspiratorial": "conspiratorial",
    "secret":         "conspiratorial", # alias — same hushed cadence
    "stern":          "stern",
    "proud":          "proud",
    "concerned":      "concerned",
    "worried":        "concerned",      # alias — same softer-bad-news prosody
}

# Map a detected user tone → preset name to apply at TTS time. Drives the
# "voice_emotion_router" delivery contract: stressed gets the calm
# bad_news preset; late-night gets quiet + slow; excited gets brighter +
# faster. Tones absent from this mapping fall through to the text-based
# emotion (detect_tts_emotion) so existing per-phrase cadence still works.
_USER_TONE_TTS: dict[str, str] = {
    "stressed":   "bad_news",      # calm + slower — counterbalance the user's panic
    "frustrated": "bad_news",      # don't add energy to a frustrated user
    "late_night": "late_night",    # quieter + slower for past-midnight
    "tired":      "late_night",    # treat fatigue the same as late-night
    "excited":    "excited",       # match the energy
    "rushed":     "confirmation",  # crisp + brisk acknowledgement
    "playful":    "confirmation",  # brighter to land the wit
}

# Lowercase keyword → category. Longer/more-specific phrases first so that
# e.g. "I'm afraid that's inadvisable" picks bad_news instead of warning.
_TTS_EMOTION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("i'm afraid",         "bad_news"),
    ("i am afraid",        "bad_news"),
    ("inadvisable",        "bad_news"),
    ("rather concerning",  "bad_news"),
    ("rather unfortunate", "bad_news"),
    ("we have a situation","bad_news"),
    ("failed",             "bad_news"),
    ("error",              "bad_news"),
    ("slight problem",     "warning"),
    ("a trifle awkward",   "warning"),
    ("less than ideal",    "warning"),
    ("suboptimal",         "warning"),
    ("not entirely ideal", "warning"),
    ("with respect",       "warning"),
    ("very good",          "confirmation"),
    ("complete",           "confirmation"),
    ("completed",          "confirmation"),
    ("as you wish",        "confirmation"),
    ("right away",         "confirmation"),
    ("on it, sir",         "confirmation"),
    ("done, sir",          "confirmation"),
    ("nominal",            "confirmation"),
    ("at your service",    "confirmation"),
    ("understood, sir",    "confirmation"),
)


def detect_tts_emotion(text: str) -> str:
    """Return one of 'bad_news' | 'warning' | 'confirmation' | 'neutral'
    based on signature phrases in the outgoing text. First match wins; the
    keyword list is ordered most-specific first."""
    if not text:
        return "neutral"
    low = text.lower()
    for kw, cat in _TTS_EMOTION_KEYWORDS:
        if kw in low:
            return cat
    return "neutral"


def resolve_tts_preset(
    text: str,
    user_tone: Optional[str],
    *,
    wry: bool = False,
    intent_override: Optional[str] = None,
    mood: Optional[str] = None,
    user_text: Optional[str] = None,
    peak_rms: float = 0.0,
    emotion_preset: Optional[str] = None,
    now: Optional[datetime.datetime] = None,
    state_path: str = _ANTICIPATION_STATE_PATH,
) -> Tuple[str, dict]:
    """Pick the TTS preset for one synthesis. Pure: all runtime state is
    passed in (the bobert_companion._resolve_tts_preset shim supplies the
    live _last_* singletons + recent vocal RMS). Priority order:

    0. ``wry`` (highest — dry-humour deliveries always render in the wry
       preset so synthesise() can insert a beat before the punchline).
    1. ``intent_override`` — an [intent:xxx] tag parsed off the reply.
    2. ``mood`` — explicit voice_mood layer (mood= kwarg / [mood:xxx]).
    3. context presets (emergency / late-hour / vocal-stress) from
       detect_context_preset(user_text, peak_rms).
    4. ``emotion_preset`` — the five-label emotion-tracker preset.
    5. ``user_tone`` via _USER_TONE_TTS.
    6. the text-keyword emotion from detect_tts_emotion (fallback).

    Exceptions preserved from the legacy resolver: a bad_news line never gets
    an 'excited'/'amused' preset layered on top — the outgoing line's
    gravitas wins.
    """
    if wry:
        preset = _TTS_EMOTION_PRESETS.get("wry", _TTS_EMOTION_PRESETS["neutral"])
        return "wry", preset

    if intent_override and intent_override in _INTENT_PRESETS:
        chosen = _INTENT_PRESETS[intent_override]
        preset = _TTS_EMOTION_PRESETS.get(chosen, _TTS_EMOTION_PRESETS["neutral"])
        return chosen, preset

    # voice_mood layer — explicit per-call mood= kwarg or [mood:xxx] tag.
    # Sits between [intent:xxx] and the emotion-tracker / user-tone fallbacks:
    # a skill saying "this is urgent" outranks the passive read of sir's
    # current mood, but an [intent:xxx] the LLM emits still wins because
    # intents are the model's explicit prosody pick.
    if mood and mood in _TTS_EMOTION_PRESETS:
        return mood, _TTS_EMOTION_PRESETS[mood]

    # Context-aware presets (detect_context_preset):
    #   emergency keywords  → brisk_alert  (most intrusive, always wins)
    #   late hour           → hushed_late  (quiet after 23:00)
    #   high RMS energy     → calm_low     (vocal stress proxy)
    # Sits above the emotion-tracker / user-tone / text-emotion fallbacks.
    # Falls through silently when none apply.
    try:
        ctx_preset = detect_context_preset(
            user_text=user_text,
            peak_rms=peak_rms,
            now=now,
            state_path=state_path,
        )
    except Exception:
        ctx_preset = None
    if ctx_preset and ctx_preset in _TTS_EMOTION_PRESETS:
        return ctx_preset, _TTS_EMOTION_PRESETS[ctx_preset]

    text_emotion = detect_tts_emotion(text)
    chosen = text_emotion

    # Five-label emotion tracker preset (core/emotion_tracker). Wins over
    # _USER_TONE_TTS because it makes the explicit per-spec mappings
    # (stressed→calm, excited→amused, focused→briefing, tired→concerned)
    # the source of truth. Exception preserved: a bad_news line never gets
    # an excited preset layered on top — the text wins there.
    if emotion_preset and emotion_preset in _TTS_EMOTION_PRESETS:
        if not (text_emotion == "bad_news" and emotion_preset == "amused"):
            chosen = emotion_preset
            preset = _TTS_EMOTION_PRESETS[chosen]
            return chosen, preset

    if user_tone and user_tone in _USER_TONE_TTS:
        # Never override an explicit bad_news line with an excited
        # user-tone preset — the text is more authoritative there.
        if not (text_emotion == "bad_news" and user_tone == "excited"):
            chosen = _USER_TONE_TTS[user_tone]

    preset = _TTS_EMOTION_PRESETS.get(chosen, _TTS_EMOTION_PRESETS["neutral"])
    return chosen, preset


# ──────────────────────────────────────────────────────────────────────────
#  SELF-TEST
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        # (text, emotion, tone, intent, wry, expected)
        ("Quite, sir.",                None,        None,         None,    True,  "wry"),
        ("Right away, sir.",           None,        None,         "urgent",False, "urgent"),
        ("Working on it.",             "stressed",  None,         None,    False, "calm"),
        ("Splendid choice, sir.",      "excited",   None,         None,    False, "amused"),
        ("Let's review the diff.",     "focused",   None,         None,    False, "briefing"),
        ("Going to bed, sir.",         "tired",     None,         None,    False, "concerned"),
        ("Hello there.",               None,        None,         None,    False, "neutral"),
        ("I'm afraid that's failed.",  "excited",   None,         None,    False, "bad_news"),
    ]
    for text, emo, tone, intent, wry, expected in cases:
        text_emo = "bad_news" if "i'm afraid" in text.lower() else None
        got = select_preset(
            text,
            emotion_label=emo,
            user_tone=tone,
            intent=intent,
            wry=wry,
            text_emotion=text_emo,
        )
        ok = "OK " if got == expected else "BAD"
        print(f"  {ok}  emo={emo!r:<13} wry={wry!s:<5} intent={intent!r:<9} text_emo={text_emo!r:<10} -> {got!r:<10} expected={expected!r}")

    print()
    print("  parse_wry_tag tests:")
    for s, expected in [
        ("[wry] Quite, sir.",              (True,  "Quite, sir.")),
        ("[ WRY ]   Indeed.",              (True,  "Indeed.")),
        ("Just regular text",              (False, "Just regular text")),
        ("",                               (False, "")),
        ("Some [wry] mid-sentence tag",    (False, "Some [wry] mid-sentence tag")),
    ]:
        got = parse_wry_tag(s)
        ok = "OK " if got == expected else "BAD"
        print(f"    {ok}  {s!r:<40} -> {got}")

    print()
    print("  split_for_wry_pause tests:")
    for s in [
        "Quite, sir. I had wondered when you'd ask.",
        "I'd advise against it, sir — though if you insist, your funeral.",
        "Short.",
        "Two sentences. Yes.",
        "Very good. Right away, sir.",
    ]:
        head, tail = split_for_wry_pause(s)
        print(f"    {s!r}\n       head={head!r}\n       tail={tail!r}")

    print()
    print("  jarvis_quip_layer tests:")
    # Deterministic RNG: probability=1.0 forces a quip, seeded Random pins the
    # pool selection so we can assert exact output.
    cases_q = [
        ("Right away, sir.",  "upgrade_jarvis",   "destructive"),
        ("Done.",              "play_music",       "media"),
        ("Very good, sir.",   "focus_window",     "ui"),
        ("Noted.",             "see_screen",       "default"),
        ("Noted.",             None,               "default"),
    ]
    for text, action, expected_cat in cases_q:
        got_cat = classify_action_for_quip(action)
        ok = "OK " if got_cat == expected_cat else "BAD"
        out = jarvis_quip_layer(text, action,
                                rng=random.Random(0), probability=1.0)
        print(f"    {ok}  action={action!r:<22} cat={got_cat:<11} -> {out!r}")

    # Long text (>= 6 words): no quip even at 100%.
    long_text = "I will close all the windows for you now."
    out_long = jarvis_quip_layer(long_text, "close_all_windows",
                                 rng=random.Random(0), probability=1.0)
    ok = "OK " if out_long == long_text else "BAD"
    print(f"    {ok}  long-text guard: {out_long!r}")

    # Probability=0.0 → never quips.
    out_zero = jarvis_quip_layer("Done.", "upgrade_jarvis",
                                 rng=random.Random(0), probability=0.0)
    ok = "OK " if out_zero == "Done." else "BAD"
    print(f"    {ok}  probability=0 guard: {out_zero!r}")
