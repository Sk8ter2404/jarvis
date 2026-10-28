"""core/speech_filter.py — Whisper transcription gating.

Pure functions that run on every transcription BEFORE it reaches the LLM:

  is_ambient_music(text)  — True when Whisper emitted a [Music]/♪ marker instead
                            of words (the mic picked up music, not speech).
  is_valid_speech(text, conf, peak_rms) — (is_valid, reason) filter that drops
                            hallucinations, sub-threshold confidence, too-short
                            mumbles, and (optionally) anything missing the wake
                            word, while always accepting a small set of common
                            single-word commands.
  hallucination_verdict(...) — (R10) for a transcript that is ONLY a known
                            hallucination phrase: noise (drop), a genuine short
                            reply (keep), or neither (is_valid_speech decides).
                            Also drops a degenerate transcript — one word over
                            and over, a run of fillers, near-zero lexical
                            diversity (repetition_reason) — whatever the context.

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


def is_valid_speech(text: str, conf: dict, peak_rms: float = 0.0,
                    reply: bool = False) -> tuple[bool, str]:
    """Decide if this transcription is real speech worth responding to.
    Returns (is_valid, reason_if_filtered).

    ``reply``: the caller's hallucination_verdict() judged this transcript a
    genuine short reply ("thank you" right after JARVIS answered). Only then
    is a reply-able hallucination phrase (REPLY_PHRASES) accepted instead of
    rejected as a hallucination match; every other transcript is judged
    exactly as without it."""
    if not text:
        return False, "empty"

    # Normalised form for the various checks below
    normalized = text.lower().strip().rstrip(".,!?")
    words      = re.sub(r"[^\w\s]", " ", text).split()

    # Always accept short common words even if MIN_WORDS would reject them
    if len(words) == 1 and normalized in WHISPER_ALWAYS_ACCEPT:
        return True, ""

    # A judged reply: the phrase is the owner's, not Whisper's. The verdict
    # already required Whisper's own no-speech / confidence scores to pass.
    if reply and _norm_phrase(text) in _reply_norms():
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


# ── Hallucination-only transcripts: noise, or a real short reply? ──────────
# (2026-09-29, R10)
#
# THE LIVE INCIDENT (session_2026-09-29_19-17-23.log, 19:18:55). Nobody home,
# wake-word mode off, no media playing. Whisper turned room noise (peak RMS
# 0.0119 against the 0.008 VAD threshold: 1.49x) into "Bye." and the main loop
# answered it as an owner turn with a full LLM call. It got through because
# "bye" is in BOTH lists above and is_valid_speech's single-word
# WHISPER_ALWAYS_ACCEPT shortcut runs BEFORE the hallucination check — so
# "bye" / "okay" / "ok" / "yeah" were accepted however they were heard, while
# "thank you" was rejected however it was said, even right after JARVIS
# answered the owner.
#
# hallucination_verdict() decides the phrase-ONLY transcripts on evidence
# instead of on the word:
#   "noise" — drop it, when ANY of:
#     * Whisper itself says so: no_speech_prob above WHISPER_MAX_NO_SPEECH_PROB
#       or avg_logprob below WHISPER_MIN_AVG_LOGPROB (is_valid_speech's own
#       thresholds — and NOT bypassed by a loud peak here);
#     * the capture was only marginally louder than the VAD trip: peak RMS
#       under NOISE_RMS_MARGIN x the VAD threshold. Measured over every
#       September session log on the owner's desk mic: 58% of hallucination-
#       only transcripts peaked under 1.5x the threshold, against 11% of all
#       other transcripts (his real speech sits at ~1.1-2x on that quiet mic,
#       which is why the margin is not higher);
#     * the owner has not spoken for NOISE_OWNER_IDLE_S and the phrase is not
#       an answer to a question JARVIS asked in the last NOISE_REPLY_WINDOW_S.
#   "reply" — keep it, when the phrase is one people answer with
#     (REPLY_PHRASES) and the context makes it clearly a reply: the owner is
#     in a conversation (spoke within NOISE_OWNER_IDLE_S) and JARVIS just
#     spoke (within NOISE_REPLY_WINDOW_S) or is waiting on a confirmation /
#     yes-no prompt. That context outranks the two CIRCUMSTANTIAL signals
#     (level, idle owner) but never Whisper's own verdict.
#   ""      — neither: is_valid_speech decides exactly as before (so a
#     "thank you" that is not a reply is still a hallucination match).
#
# Pure: the caller supplies the context. The reason string is numbers only —
# callers log it, never the transcript.
NOISE_RMS_MARGIN     = 1.5    # x the VAD threshold = "only marginally above"
NOISE_OWNER_IDLE_S   = 300.0  # owner silent this long = no conversation
NOISE_REPLY_WINDOW_S = 20.0   # "right after JARVIS spoke"

# Hallucination phrases a person really answers with. Everything else in
# WHISPER_HALLUCINATIONS ("you", "thanks for watching", "subscribe", fillers,
# music tags, punctuation) is never taken as a reply.
REPLY_PHRASES = frozenset({
    "thank you", "thanks", "bye", "bye bye", "okay", "ok", "yeah",
})


def _norm_phrase(text) -> str:
    """Lower-case words only: 'Bye-bye!' -> 'bye bye', 'Thank you.' ->
    'thank you'. Never raises."""
    try:
        return " ".join(re.sub(r"[^\w\s]", " ", str(text or "").lower()).split())
    except Exception:
        return ""


def _hallucination_norms() -> frozenset:
    # Derived from the live set at call time, so the two can never drift.
    return frozenset(n for n in map(_norm_phrase, WHISPER_HALLUCINATIONS) if n)


def _reply_norms() -> frozenset:
    return (frozenset(map(_norm_phrase, REPLY_PHRASES))
            & _hallucination_norms())


def is_hallucination_only(text) -> bool:
    """True when the WHOLE transcript is one known Whisper-hallucination
    phrase (case and punctuation ignored): 'Bye.', 'Thank you!', 'You'."""
    n = _norm_phrase(text)
    return bool(n) and n in _hallucination_norms()


def _conf_value(conf, key):
    try:
        v = conf.get(key) if isinstance(conf, dict) else None
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Degenerate transcripts: one word over and over (2026-09-30) ─────────────
#
# THE LIVE INCIDENT (session_2026-09-29_22-06-02.log, 22:53:08). Mid-evening,
# the owner in a conversation, sustained room music. Whisper turned 9.8 s of
# audio into "I I I I I I I I I I I I I" and the main loop answered it as an
# owner turn ("Very good, sir.") with a full LLM call. It is not a known
# hallucination PHRASE, so hallucination_verdict let it through, and
# is_valid_speech saw thirteen words.
#
# That shape is Whisper looping, not a person: a person does not say one word
# thirteen times. repetition_reason() names it — from the words' COUNTS only,
# never their content — and hallucination_verdict() drops it as noise in ANY
# context (a conversation does not make "I I I I I" an answer):
#   * one word repeated: REPEAT_MIN_WORDS+ words, a single distinct word;
#   * one or two words are nearly all of it: the two commonest words make up
#     REPEAT_TOP2_SHARE+ of a transcript of REPEAT_TOP2_MIN_WORDS+ words;
#   * a run of filler syllables: FILLER_RUN_MIN+ words, every one a filler
#     ("uh um uh", "la la la la");
#   * near-zero lexical diversity: distinct / total at or under
#     REPEAT_MAX_DIVERSITY over REPEAT_DIVERSITY_MIN_WORDS+ words.
# EXEMPT: a transcript made only of REPEAT_EXEMPT_WORDS — stop and
# confirmation words ("stop stop stop", "no no no", "yes yes"), the wake /
# attention words and a mic check — is a real, emphatic command; it is left
# to the other gates exactly as before. Shorter than REPEAT_MIN_WORDS is left
# to them too ("I I" is already "too short").
REPEAT_MIN_WORDS           = 3
REPEAT_TOP2_SHARE          = 0.9
REPEAT_TOP2_MIN_WORDS      = 6
FILLER_RUN_MIN             = 3
REPEAT_MAX_DIVERSITY       = 0.25
REPEAT_DIVERSITY_MIN_WORDS = 8

REPEAT_EXEMPT_WORDS = frozenset({
    # stop / cancel
    "stop", "cancel", "wait", "pause", "quit", "exit", "enough", "halt",
    # confirmation / refusal
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "okay", "ok", "sure",
    "confirm", "proceed", "continue", "go", "right", "correct", "please",
    # wake / attention, and a mic check
    "jarvis", "hey", "hi", "hello", "test", "testing",
})

FILLER_WORDS = frozenset({
    "uh", "um", "umm", "uhm", "er", "erm", "ah", "ahh", "aah", "eh", "oh",
    "ooh", "hmm", "hm", "mm", "mmm", "ha", "haha", "la", "na", "da",
})


def repetition_reason(text) -> str:
    """Why this transcript is degenerate repetition (numbers only — never a
    word from it), or "" when it is not (or is exempt, or too short to
    judge). See the block comment above. Never raises."""
    try:
        words = _norm_phrase(text).split()
        n = len(words)
        if n < REPEAT_MIN_WORDS:
            return ""
        if all(w in REPEAT_EXEMPT_WORDS for w in words):
            return ""
        counts: dict = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        distinct = len(counts)
        if distinct == 1:
            return f"1 distinct word in {n}"
        if n >= FILLER_RUN_MIN and all(w in FILLER_WORDS for w in words):
            return f"filler run of {n} words"
        if n >= REPEAT_TOP2_MIN_WORDS:
            top2 = sum(sorted(counts.values(), reverse=True)[:2])
            if top2 / n >= REPEAT_TOP2_SHARE:
                return f"2 words are {100 * top2 / n:.0f}% of {n}"
        if n >= REPEAT_DIVERSITY_MIN_WORDS:
            diversity = distinct / n
            if diversity <= REPEAT_MAX_DIVERSITY:
                return f"lexical diversity {diversity:.2f} over {n} words"
        return ""
    except Exception:
        return ""


def hallucination_verdict(text, conf, peak_rms, *, vad_threshold,
                          owner_idle_s=None, since_jarvis_s=None,
                          jarvis_asked=False,
                          prompt_pending=False,
                          primary_heard_nothing=False) -> tuple[str, str]:
    """("noise" | "reply" | "", reason) for one mic transcript. See the block
    comments above.

    ``owner_idle_s``: seconds since the owner's last accepted turn (None: not
    this session). ``since_jarvis_s``: seconds since JARVIS's last audible
    line finished (None: none). ``jarvis_asked``: that line asked something.
    ``prompt_pending``: a confirmation / yes-no prompt awaits an answer.
    ``primary_heard_nothing``: the primary decoder (Parakeet) found no words
    in this capture and Whisper's 'empty' rescue produced the text (see
    PRIMARY_EMPTY below). Never raises — an internal error returns ("", ""),
    the old behaviour."""
    try:
        # Degenerate repetition is noise whatever the context (22:53:08).
        rep = repetition_reason(text)
        if rep:
            return ("noise", rep)
        if not is_hallucination_only(text):
            return ("", "")
        norm = _norm_phrase(text)
        nsp = _conf_value(conf, "no_speech_prob")
        alp = _conf_value(conf, "avg_logprob")
        poor = []
        if nsp is not None and nsp > WHISPER_MAX_NO_SPEECH_PROB:
            poor.append(f"no_speech_prob {nsp:.2f}")
        if alp is not None and alp < WHISPER_MIN_AVG_LOGPROB:
            poor.append(f"avg_logprob {alp:.2f}")
        if poor:
            return ("noise", "whisper " + ", ".join(poor))
        idle = None if owner_idle_s is None else max(0.0, float(owner_idle_s))
        since = None if since_jarvis_s is None else float(since_jarvis_s)
        owner_recent = idle is not None and idle < NOISE_OWNER_IDLE_S
        jarvis_recent = (since is not None
                         and 0.0 <= since <= NOISE_REPLY_WINDOW_S)
        # PRIMARY_EMPTY: Parakeet heard no words, Whisper then "heard" a
        # stock phrase. Outranks the reply context; only a waiting prompt or
        # a question JARVIS just asked keeps it (a lost "yeah" there costs
        # more than a phantom).
        if (primary_heard_nothing and not prompt_pending
                and not (jarvis_recent and jarvis_asked)):
            return ("noise", "the primary decoder heard no words")
        if norm in _reply_norms() and owner_recent:
            if jarvis_recent:
                return ("reply", f"JARVIS spoke {since:.0f} s ago, the owner "
                                 f"{idle:.0f} s ago")
            if prompt_pending:
                return ("reply", "a prompt is waiting for an answer")
        vad = float(vad_threshold)
        peak = float(peak_rms or 0.0)
        if vad > 0 and peak < vad * NOISE_RMS_MARGIN:
            return ("noise", f"peak {peak / vad:.2f}x the VAD threshold")
        if not owner_recent and not (jarvis_recent and jarvis_asked):
            who = ("owner silent this session" if idle is None
                   else f"owner silent {idle:.0f} s")
            return ("noise", f"{who}, not a reply")
        return ("", "")
    except Exception:
        return ("", "")


# ── PRIMARY_EMPTY: a stock phrase only Whisper heard (2026-10-27) ───────────
#
# THE LIVE INCIDENT (v3.0.0, 20:39:40). The TV had just come on; nobody spoke
# to JARVIS. Parakeet, the primary decoder, found no words in the capture, so
# the 'empty' rescue handed the same audio to Whisper, which made it "Thank
# you." JARVIS had spoken 14 s earlier and the owner 18 s earlier, so the
# reply rule above kept it, and the owner had to say "I didn't say that."
# Not a self-echo: no JARVIS audio was playing during the capture.
#
# Over every session log on disk, 8 of the 21 Parakeet-empty rescues were
# "Thank you." and Parakeet itself never produced a hallucination-only phrase
# once. Two decoders disagreeing, the one that does not hallucinate hearing
# nothing, is the strongest noise evidence the gate gets, so it outranks the
# reply context. A waiting confirmation / yes-no prompt still keeps it.


# ── Unclear short utterances (2026-10-27) ───────────────────────────────────
#
# THE LIVE INCIDENT (v3.0.0, 20:41:26). Right after a search answer, a
# capture peaking at 1.25x the VAD threshold (barely above the room's own
# ambient level) came back from Parakeet as "Enough and awesome." The model
# answered "On it, sir. [ACTION: web_search, enough and awesome]" and read
# three rounds of results about it.
#
# unclear_reason() names a 2-4 word transcript that is BOTH
#   * heard at a marginal level: peak under NOISE_RMS_MARGIN x the VAD
#     threshold (the same "only marginally above" measure the noise gate
#     uses), and
#   * carries nothing a request or a reply is made of: no word from
#     SENSE_WORDS (wake word, pronouns, question and helper verbs, negations
#     and answers, command verbs and particles, media / device nouns,
#     greetings, every WHISPER_ALWAYS_ACCEPT word), no contraction, no "-ing"
#     / "-ed" verb form, no digit and no Capitalised name after the first
#     word.
# The caller asks the owner to repeat himself instead of acting on it. A
# clearly heard capture is never judged (the level test comes first), and a
# one-word line is left to is_valid_speech. Pure; the reason holds numbers
# only, never a word from the transcript.
UNCLEAR_MIN_WORDS = 2
UNCLEAR_MAX_WORDS = 4

SENSE_WORDS = frozenset({
    # address
    "jarvis",
    # pronouns, determiners that point at something
    "i", "me", "my", "mine", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "we", "us", "our", "they", "them", "their", "this", "that",
    "these", "those", "there", "here",
    # questions and helper verbs
    "what", "who", "whom", "whose", "how", "why", "when", "where", "which",
    "is", "are", "was", "were", "am", "be", "been", "do", "does", "did",
    "can", "could", "will", "would", "should", "shall", "may", "might",
    "must", "have", "has", "had", "let", "lets",
    # negations and answers
    "not", "never", "nothing", "please", "thanks", "thank", "sorry",
    # command verbs
    "turn", "set", "put", "make", "give", "tell", "show", "find", "read",
    "get", "send", "switch", "move", "call", "look", "launch", "take",
    "bring", "keep", "add", "remind", "delete", "clear", "dim", "raise",
    "lower", "increase", "decrease", "need", "want", "try", "say", "go",
    "come", "leave", "rewind", "forward", "shuffle", "silence",
    # particles a command ends on
    "on", "off", "up", "down", "out", "in", "over", "away",
    # media and devices
    "lights", "light", "lamp", "music", "song", "songs", "track", "video",
    "tv", "screen", "monitor", "headset", "speaker", "speakers", "mode",
    "page", "app", "window", "tab", "alarm", "reminder", "camera", "mic",
    "microphone", "print", "printer",
    # greetings
    "good", "night", "afternoon", "evening",
    # answers that pick or size something ("the second one", "too loud",
    # "episode five"): numbers in words, ordinals, sides, levels
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "half", "max",
    "first", "second", "third", "fourth", "fifth", "last", "previous",
    "left", "right", "both", "all", "too", "loud", "quiet", "bright",
    "higher", "brighter", "dimmer", "episode", "season", "brightness",
    "color", "colour", "minute", "minutes", "hour", "hours", "sec",
}) | frozenset(WHISPER_ALWAYS_ACCEPT)

_SENSE_SUFFIXES = ("ing", "ed")


def _sense_word(word: str, position: int) -> bool:
    w = word.lower()
    if w in SENSE_WORDS or any(ch.isdigit() for ch in w):
        return True
    if "'" in w or "’" in w:
        return True        # a contraction: "it's", "don't", "I'm"
    if len(w) >= 5 and w.endswith(_SENSE_SUFFIXES):
        return True        # a verb form: "tracking", "switched"
    if position > 0 and word[:1].isupper():
        return True        # a name Parakeet capitalised mid-line
    return False


def unclear_reason(text, peak_rms, *, vad_threshold) -> str:
    """Why this transcript is too unclear to act on (numbers only), or "".
    See the block comment above. Never raises."""
    try:
        vad = float(vad_threshold)
        peak = float(peak_rms or 0.0)
        if vad <= 0 or peak <= 0 or peak >= vad * NOISE_RMS_MARGIN:
            return ""
        words = [w for w in re.findall(r"[\w'’]+", str(text or ""))
                 if w.strip("'’")]
        n = len(words)
        if not (UNCLEAR_MIN_WORDS <= n <= UNCLEAR_MAX_WORDS):
            return ""
        if any(_sense_word(w, i) for i, w in enumerate(words)):
            return ""
        return (f"{n} words, none a request or reply word, peak "
                f"{peak / vad:.2f}x the VAD threshold")
    except Exception:
        return ""


# ── Per-install tuning ──────────────────────────────────────────────────────
# These thresholds depend on the MICROPHONE: its gain decides what RMS "loud"
# is, and its noise floor shapes Whisper's confidence. An install whose mic
# differs from the desktop's (e.g. a laptop edge node) overrides them through
# core.config.SPEECH_FILTER_OVERRIDES (data/user_settings.json); the monolith
# calls apply_overrides() once at import. This module stays pure: it never
# imports config and does no I/O.
#
# Only these are overridable, each with a type and a sane range. Anything
# else -- unknown names, wrong types, bools, out-of-range values, a
# non-integral word count -- is skipped rather than half-applied. The three
# NOISE_* knobs (R10) belong here for the same reason: how far above the VAD
# threshold real speech lands is a property of the microphone.
_OVERRIDABLE = {
    "WHISPER_MIN_WORDS":          (int,   1,     10),
    "WHISPER_MAX_NO_SPEECH_PROB": (float, 0.0,   1.0),
    "WHISPER_MIN_AVG_LOGPROB":    (float, -10.0, 0.0),
    "WHISPER_TRUST_RMS":          (float, 0.0,   1.0),
    "NOISE_RMS_MARGIN":           (float, 1.0,   10.0),
    "NOISE_OWNER_IDLE_S":         (float, 0.0,   86400.0),
    "NOISE_REPLY_WINDOW_S":       (float, 0.0,   600.0),
}
_DEFAULTS = {name: globals()[name] for name in _OVERRIDABLE}


def apply_overrides(overrides) -> dict:
    """Apply validated overrides to this module's thresholds.

    Returns ``{name: value}`` for what was ACTUALLY applied (never raises).
    Callers that re-export a threshold by value must re-read it afterwards."""
    applied = {}
    if not isinstance(overrides, dict):
        return applied
    g = globals()
    for name, val in overrides.items():
        spec = _OVERRIDABLE.get(name)
        if spec is None or isinstance(val, bool):
            continue
        typ, lo, hi = spec
        try:
            if typ is int:
                if isinstance(val, float) and not val.is_integer():
                    continue
                v = int(val)
            else:
                v = float(val)
        except (TypeError, ValueError):
            continue
        if not (lo <= v <= hi):
            continue
        g[name] = v
        applied[name] = v
    return applied


def reset_overrides() -> None:
    """Restore every overridable threshold to its built-in default."""
    globals().update(_DEFAULTS)
