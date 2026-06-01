"""core/speech_filter.py — Whisper transcription gating.

Two pure functions that run on every transcription BEFORE it reaches the LLM:

  is_ambient_music(text)  — True when Whisper emitted a [Music]/♪ marker instead
                            of words (the mic picked up music, not speech).
  is_valid_speech(text, conf, peak_rms) — (is_valid, reason) filter that drops
                            hallucinations, sub-threshold confidence, too-short
                            mumbles, and (optionally) anything missing the wake
                            word, while always accepting a small set of common
                            single-word commands.

Extracted verbatim from bobert_companion.py along with their tuning constants
so the gate logic is testable in isolation (it had no coverage before) and the
~70 lines leave the monolith. Pure stdlib (`re`) — no model load, no I/O.
bobert_companion re-exports is_valid_speech / is_ambient_music / WHISPER_TRUST_RMS
(the one threshold the main loop also reads); the remaining constants are used
only inside is_valid_speech and live here.
"""
from __future__ import annotations

import re

# Whisper confidence filtering — prevents Bobert from responding to music,
# background noise, or garbled audio. Raise thresholds if too many things are
# getting filtered; lower them if junk is still getting through.
WHISPER_MIN_WORDS          = 2     # discard transcriptions shorter than this
WHISPER_MAX_NO_SPEECH_PROB = 0.85  # Whisper's "this isn't speech" score (0-1)
WHISPER_MIN_AVG_LOGPROB    = -1.5  # confidence; less negative = more confident

# Short single words to always accept (overrides MIN_WORDS) — useful for
# confirmations, stop commands, quick answers.
WHISPER_ALWAYS_ACCEPT = {
    "yes", "yeah", "yep", "yup", "no", "nope", "nah",
    "stop", "cancel", "wait", "okay", "ok", "sure",
    "done", "go", "back", "next", "quit", "exit", "help",
    "confirm", "proceed", "continue", "pause",
    # Greetings
    "hello", "hi", "hey", "morning", "goodbye", "bye",
    # Common single-word commands
    "louder", "quieter", "mute", "unmute", "skip", "repeat",
    "play", "resume", "again", "more", "less",
    # Control / restart / upgrade
    "restart", "reboot", "reload", "refresh", "start", "run", "upgrade",
    # Navigation / misc single-word commands
    "map", "search", "open", "close", "show", "hide", "check", "status",
    "volume", "timer", "weather", "news", "time", "date", "lock", "screenshot",
    "hud", "toggle",
    # JARVIS wake / sleep words
    "jarvis", "wake", "sleep", "standby",
}

# If the recording's peak RMS exceeded this level during capture, trust the
# transcription regardless of Whisper's confidence scores (real loud speech
# sometimes gets bad confidence scores anyway).
WHISPER_TRUST_RMS = 0.025

# Known Whisper hallucinations — phrases it commonly outputs on silence/music
# or noisy audio with no real speech. Discarded automatically.
WHISPER_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thanks", "thanks.",
    "thanks for watching", "thanks for watching.", "thank you for watching",
    "thank you for watching.", "please subscribe", "subscribe", "like and subscribe",
    "bye", "bye.", "bye bye", "okay", "okay.", "ok", "ok.",
    "music", "[music]", "(music)", "music playing", "[music playing]",
    ".", "...", "!", "?", "yeah", "yeah.", "mm", "hmm", "uh", "um",
}

# Wake word — if set, Bobert only responds when his name is detected.
# Recommended when you have music playing or work in a noisy room.
# Examples: "bobert", "hey bobert", "hey jarvis"
WAKE_WORD = None

# Whisper emits explicit non-speech markers when it hears music / applause /
# instrumental audio it can't transcribe as words. These are the strongest
# signal we have that the mic is picking up ambient music vs. real speech.
_MUSIC_MARKERS = (
    "[music]", "(music)", "♪", "♬", "♫",
    "[singing]", "(singing)", "[instrumental]", "(instrumental)",
    "[chorus]", "(chorus)", "[applause]", "(applause)",
    "music playing", "[music playing]",
)


def is_ambient_music(text: str) -> bool:
    """Return True if the transcription looks like Whisper picked up music
    rather than speech — explicit marker like [Music], (Music), ♪ etc."""
    if not text:
        return False
    t = text.lower()
    return any(m in t for m in _MUSIC_MARKERS)


def is_valid_speech(text: str, conf: dict, peak_rms: float = 0.0) -> tuple[bool, str]:
    """Decide if this transcription is real speech worth responding to.
    Returns (is_valid, reason_if_filtered)."""
    if not text:
        return False, "empty"

    # Normalised form for the various checks below
    normalized = text.lower().strip().rstrip(".,!?")
    words      = re.sub(r"[^\w\s]", " ", text).split()

    # Always accept short common words even if MIN_WORDS would reject them
    if len(words) == 1 and normalized in WHISPER_ALWAYS_ACCEPT:
        return True, ""

    # Always reject known hallucinations regardless of anything else
    if normalized in WHISPER_HALLUCINATIONS:
        return False, f"hallucination match: '{normalized}'"

    # Wake word always required if configured
    if WAKE_WORD and WAKE_WORD.lower() not in text.lower():
        return False, f"missing wake word '{WAKE_WORD}'"

    # If audio level was clearly speech, trust the transcription regardless
    # of Whisper's own confidence scores.
    high_rms = peak_rms >= WHISPER_TRUST_RMS

    # Minimum-character gate that ALSO applies under high_rms — prevents
    # loud mumbles like "ops" / "uh" / "mm" from being dispatched to the LLM
    # just because they were said at speaking volume.
    if len(normalized.replace(" ", "")) < 4 and normalized not in WHISPER_ALWAYS_ACCEPT:
        return False, f"too short ({len(normalized)} chars)"

    # Word count gate (unless we're trusting on RMS)
    if not high_rms and len(words) < WHISPER_MIN_WORDS:
        return False, f"too short ({len(words)} words)"

    # Confidence gates (unless we're trusting on RMS)
    if not high_rms:
        if conf["no_speech_prob"] > WHISPER_MAX_NO_SPEECH_PROB:
            return False, f"no_speech_prob={conf['no_speech_prob']:.2f}"
        if conf["avg_logprob"] < WHISPER_MIN_AVG_LOGPROB:
            return False, f"low confidence (logprob={conf['avg_logprob']:.2f})"

    return True, ""
