"""core/device_speech_filter.py — known-device speech must never command JARVIS.

WHY THIS MODULE EXISTS
======================
A talking device in the room (a smart speaker, a toy, a gadget) speaks a
known, finite set of lines. The mic hears them like any other voice, so a
device line that happens to start with the wake word was taken as the owner
talking and triggered unwanted actions, and with wake-word mode
off a misheard device line got a spoken reply and was LEARNED as a topic.

The fix is a phrase list per device: every ``*.json`` file in the gitignored
``<data dir>/device_phrases/`` directory, shaped

    {"source": "<device name>", "phrases": ["...", "..."]}

and a matcher the main loop consults right after transcription, BEFORE the
background/wake gate, the LLM and any learning. The phrase files are private
(they live only on the box, never in the repo); the matcher is generic.

MATCHING RULES (conservative — a real owner command must always get through)
  * PROTECTED owner vocabulary never matches: an utterance that, normalised,
    IS one of OWNER_PHRASES (confirmations like "yes" / "go ahead", replies
    like "maybe" / "got it", greetings, media transport like "next track"),
    one of core/speech_filter's always-accepted single words, or one of the
    caller's ``never_match`` phrases (the monolith passes its wake / sleep /
    shutdown-prompt phrases). A device list full of generic one- and two-word
    lines must never swallow the owner's "yes" to a pending confirmation
    (R3 review, 2026-09-29).
  * An utterance containing a STOP word (stop, halt, freeze, abort, emergency,
    estop / e-stop, cancel — exact words, not inflections) NEVER matches, so a
    stop command is never swallowed even when it sounds like a device line. A
    device line like "... stopping" is not a stop command and is still
    filtered.
  * Exact normalised equality with any phrase matches (score 1.0).
  * Phrases of 1-2 words: exact normalised equality only (short strings are
    too easy to hit fuzzily). Known mishearings of a short line go into the
    phrase file as literal extra phrases.
  * Phrases of 3+ words: difflib.SequenceMatcher ratio of the WHOLE normalised
    utterance against the WHOLE normalised phrase, with these guards:
      - the utterance itself must have 3+ words (a 1-2 word owner utterance
        is exact-only too: "the timer" scored 0.80 against "the time is");
      - a wake phrase leading BOTH the utterance and the phrase is dropped
        from both before the ratio (callers pass ``wake_phrases``), so the
        shared "<wake word> " prefix cannot inflate an owner's
        "<wake word> back online" against a device's "<wake word> link online";
      - threshold FUZZY_MIN_RATIO (0.80) for a long compared phrase and the
        stricter SHORT_FUZZY_MIN_RATIO for a short one (fewer than
        SHORT_PHRASE_MIN_WORDS words or SHORT_PHRASE_MIN_CHARS characters):
        on a short string one different word already scores ~0.80-0.89.
    Measured live: misheard device lines scored 0.81-0.82, real owner commands
    0.55-0.60. The ratio also covers a Whisper-truncated device line: a prefix
    of a phrase only reaches 0.80 when it keeps ~2/3 of the phrase.
  * Whisper noise is dropped before the fuzzy rule when 3+ words remain:
    leading fillers (oh, uh, um, ah, er, hmm) and one trailing hallucinated
    "you" / "thank you" ("<line> you" is a classic Whisper tail).
  * An utterance that CONTAINS a device phrase verbatim inside a longer
    sentence never matches that phrase — that is the owner quoting or
    extending it, not the device speaking. (Deliberate addition to the spec.)
  * Word-count guard (fuzzy rule only): the utterance may run at most
    max(1, words // 3) words longer than the phrase. Whisper splitting a word
    ("pairing" -> "pear ring") adds one; an owner command built around a
    device line adds several ("desk speaker are you ready to play" scores 0.87
    against the 5-word "desk speaker ready to play" and must still pass).
    (Deliberate addition to the spec.)

KNOWN RESIDUALS: a misheard SHORT (1-2 word) line is not caught unless the
mishearing is listed literally in the phrase file; a device line split across
two ambient-listen batches is only caught on the second fragment; the neural
standby wake detector (WAKE_WORD_AUTOSTART) has no transcript to check.

FAILURE POSTURE: a missing directory, an unreadable or malformed file, or any
internal error means "no filtering" for that file / call. Nothing here raises.
Loaded lists are cached by each file's (mtime, size), so edits apply without a
restart of the matcher but the directory is only re-parsed when it changes.

Log hygiene: callers print the SOURCE name only, never the utterance or the
matched phrase (the phrase list is private).

Pure stdlib (difflib, json, os, re, threading).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
from typing import Optional

from core import paths as _paths

# Sub-directory of the (staging-aware) data dir that holds the phrase files.
PHRASES_SUBDIR = "device_phrases"

# Fuzzy threshold for a long compared phrase (see module docstring).
FUZZY_MIN_RATIO = 0.80
# Stricter threshold when the compared phrase is short: fewer than
# SHORT_PHRASE_MIN_WORDS words or SHORT_PHRASE_MIN_CHARS normalised characters.
SHORT_FUZZY_MIN_RATIO = 0.90
SHORT_PHRASE_MIN_WORDS = 4
SHORT_PHRASE_MIN_CHARS = 20
# Phrases AND utterances with at most this many words need an exact
# normalised match.
EXACT_ONLY_MAX_WORDS = 2

# Stop words: exact tokens (the spec's list). "e-stop" normalises to "e stop",
# whose "stop" token already counts.
_STOP_WORDS = frozenset(
    ("stop", "halt", "freeze", "abort", "emergency", "estop", "cancel"))

# Owner vocabulary a device line can never shadow: confirmations, replies,
# greetings and media transport. Matched as WHOLE utterances (normalised), so
# only a bare "yes" is protected, not every sentence containing it.
OWNER_PHRASES = frozenset((
    # confirmations / refusals (the confirmation prompts ask for "yes")
    "yes", "yeah", "yep", "yup", "yes please", "yes go ahead", "yes do it",
    "no", "nope", "nah", "no thanks", "no thank you", "not now",
    "ok", "okay", "sure", "confirm", "confirmed", "proceed", "do it",
    "go ahead", "go for it", "correct", "right", "wrong", "never mind",
    "absolutely", "definitely", "of course", "please",
    # replies to a question
    "maybe", "perhaps", "probably", "likely", "very likely", "unlikely",
    "not really", "i dont know", "got it", "understood", "noted",
    "sounds good", "thats right", "it is", "it isnt", "i am", "later",
    "thanks", "thank you",
    # greetings
    "hi", "hey", "hello", "hello there", "good morning", "good afternoon",
    "good evening", "good night", "goodbye", "bye", "bye bye",
    # media transport / volume
    "play", "pause", "resume", "skip", "next", "previous", "next track",
    "previous track", "next song", "previous song", "skip track", "skip song",
    "louder", "quieter", "mute", "unmute", "volume up", "volume down",
))

# Whisper noise dropped before the fuzzy rule (see module docstring).
_LEADING_FILLERS = frozenset(("oh", "uh", "um", "umm", "ah", "er", "hmm"))
_TRAILING_TAILS = (("thank", "you"), ("you",))

_APOSTROPHES_RE = re.compile(r"['‘’`]")
_NON_WORD_RE = re.compile(r"[^\w\s]|_")
_SPACE_RE = re.compile(r"\s+")

_lock = threading.Lock()
# (directory, signature) -> list of (source, normalised phrase, word count)
_cache: dict = {"key": None, "phrases": []}
# Files already reported as unusable, keyed by (path, mtime, size) so a fixed
# file is re-read and a still-broken one is not re-reported every turn.
_reported_bad: set = set()


def normalise(text) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Apostrophes are deleted ("it's" -> "its"); every other punctuation mark
    becomes a space ("e-stop" -> "e stop", "online." -> "online"). Never raises;
    a non-string yields "".
    """
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _APOSTROPHES_RE.sub("", s)
    s = _NON_WORD_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def has_stop_word(text) -> bool:
    """True when ``text`` contains a stop word (see module docstring)."""
    norm = normalise(text)
    if not norm:
        return False
    return any(tok in _STOP_WORDS for tok in norm.split())


def _protected_owner_phrases() -> frozenset:
    """OWNER_PHRASES plus core/speech_filter's always-accepted single words
    (the confirmations / quick answers the speech gate already whitelists),
    normalised. Never raises."""
    extra = ()
    try:
        from core.speech_filter import WHISPER_ALWAYS_ACCEPT as extra
    except Exception:
        pass
    return frozenset(normalise(p) for p in (*OWNER_PHRASES, *extra)) - {""}


_OWNER_PROTECTED = _protected_owner_phrases()


def _strip_whisper_noise(words: list) -> list:
    """``words`` minus leading fillers and one trailing hallucinated tail —
    only when 3+ words remain; otherwise ``words`` unchanged."""
    out = list(words)
    while out and out[0] in _LEADING_FILLERS:
        out.pop(0)
    for tail in _TRAILING_TAILS:
        if len(out) > len(tail) and tuple(out[-len(tail):]) == tail:
            out = out[:-len(tail)]
            break
    return out if len(out) >= 3 else list(words)


def _wake_prefix_len(words: list, wake_seqs) -> int:
    """Word count of the longest wake phrase ``words`` starts with (and is
    longer than), else 0."""
    best = 0
    for seq in wake_seqs:
        n = len(seq)
        if n > best and len(words) > n and tuple(words[:n]) == seq:
            best = n
    return best


def phrases_dir() -> str:
    """The phrase directory for THIS process, resolved at call time through
    core.paths so staging / test redirects (JARVIS_DATA_DIR, JARVIS_STAGING)
    apply. Never creates anything."""
    return os.path.join(_paths.data_dir(create=False), PHRASES_SUBDIR)


def _dir_signature(directory: str):
    """((name, mtime_ns, size), ...) for every *.json file, sorted; None when
    the directory is missing/unreadable."""
    try:
        names = sorted(n for n in os.listdir(directory)
                       if n.lower().endswith(".json"))
    except Exception:
        return None
    sig = []
    for name in names:
        try:
            st = os.stat(os.path.join(directory, name))
        except Exception:
            continue
        sig.append((name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _parse_file(path: str, fallback_source: str):
    """Return [(source, normalised phrase, word count), ...] or None if the
    file is unusable. Never raises."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    phrases = data.get("phrases")
    if not isinstance(phrases, list):
        return None
    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        source = fallback_source
    source = source.strip()
    out = []
    seen = set()
    for raw in phrases:
        norm = normalise(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append((source, norm, len(norm.split())))
    return out


def load_phrases(directory: Optional[str] = None) -> list:
    """All loaded (source, normalised phrase, word count) tuples, cached by the
    directory's file signature. [] when the directory is missing or holds no
    usable file. Never raises."""
    try:
        d = directory or phrases_dir()
        sig = _dir_signature(d)
        if not sig:
            return []
        key = (os.path.normcase(os.path.abspath(d)), sig)
        with _lock:
            if _cache["key"] == key:
                return _cache["phrases"]
        loaded = []
        for name, mtime_ns, size in sig:
            path = os.path.join(d, name)
            parsed = _parse_file(path, os.path.splitext(name)[0])
            if parsed is None:
                bad_key = (path, mtime_ns, size)
                if bad_key not in _reported_bad:
                    _reported_bad.add(bad_key)
                    # File NAME only — never its contents.
                    print(f"  [device-speech] skipped unusable phrase file "
                          f"{name}")
                continue
            loaded.extend(parsed)
        with _lock:
            _cache["key"] = key
            _cache["phrases"] = loaded
        return loaded
    except Exception:
        return []


def _contains_verbatim(utt_words: list, phrase_words: list) -> bool:
    """True when ``phrase_words`` occurs as a contiguous run inside a STRICTLY
    longer ``utt_words`` (the owner quoting/extending a device line)."""
    n, m = len(utt_words), len(phrase_words)
    if m == 0 or n <= m:
        return False
    for i in range(n - m + 1):
        if utt_words[i:i + m] == phrase_words:
            return True
    return False


def match(utterance, directory: Optional[str] = None, never_match=(),
          wake_phrases=()):
    """``(source, phrase, score)`` when ``utterance`` is a known device line,
    else None. ``phrase`` is the NORMALISED phrase (never log it).

    ``never_match``: protected phrases on top of OWNER_PHRASES — an utterance
    that, normalised, equals one of them is never filtered. Callers pass the
    wake / sleep / confirmation phrases, so a device list holding a bare
    "<wake word>" or "yes" line can never lock the owner out.
    ``wake_phrases``: a wake phrase leading BOTH the utterance and a phrase is
    dropped from both before the fuzzy ratio (see module docstring).
    Never raises; any internal error is treated as "no match"."""
    try:
        utt = normalise(utterance)
        if not utt:
            return None
        protected = _OWNER_PROTECTED
        if never_match:
            protected = protected | {normalise(p) for p in never_match}
        if utt in protected:
            return None
        utt_words = utt.split()
        if any(tok in _STOP_WORDS for tok in utt_words):
            return None
        phrases = load_phrases(directory)
        if not phrases:
            return None
        fz_words = _strip_whisper_noise(utt_words)
        fz_utt = " ".join(fz_words)
        wake_seqs = {tuple(w) for w in (normalise(p).split()
                                        for p in (wake_phrases or ())) if w}
        u_wake = _wake_prefix_len(fz_words, wake_seqs)
        best = None
        sm = difflib.SequenceMatcher(autojunk=False)
        for source, phrase, n_words in phrases:
            if utt == phrase or fz_utt == phrase:
                return (source, phrase, 1.0)
            if n_words <= EXACT_ONLY_MAX_WORDS:
                continue
            if len(fz_words) <= EXACT_ONLY_MAX_WORDS:
                continue
            if len(fz_words) > n_words + max(1, n_words // 3):
                continue
            p_words = phrase.split()
            if _contains_verbatim(fz_words, p_words):
                continue
            a, b = fz_words, p_words
            if u_wake and p_words[:u_wake] == fz_words[:u_wake]:
                a, b = fz_words[u_wake:], p_words[u_wake:]
            a_s, b_s = " ".join(a), " ".join(b)
            short = (len(b) < SHORT_PHRASE_MIN_WORDS
                     or len(b_s) < SHORT_PHRASE_MIN_CHARS)
            need = SHORT_FUZZY_MIN_RATIO if short else FUZZY_MIN_RATIO
            sm.set_seqs(b_s, a_s)
            if sm.real_quick_ratio() < need or sm.quick_ratio() < need:
                continue
            score = sm.ratio()
            if score >= need and (best is None or score > best[2]):
                best = (source, phrase, score)
        return best
    except Exception:
        return None


def _reset_cache_for_tests() -> None:
    with _lock:
        _cache["key"] = None
        _cache["phrases"] = []
    _reported_bad.clear()
