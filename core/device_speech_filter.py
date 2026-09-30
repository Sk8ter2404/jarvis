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

EXPECTED LINES / DIALOGUE (transient, in memory)
  A caller that is about to make a device speak a line it composed at run
  time (a scripted back-and-forth between JARVIS and a device) registers that
  line first with ``expect(source, text, window_s=...)`` and marks it finished
  with ``expect_done(handle)``. ``match`` then also filters those lines, but
  only inside a SHORT time window, so the owner's own reply a few seconds
  later is never swallowed:
    * an entry is live until ``now + window_s + EXPECT_GRACE_S`` (the predicted
      end of the line plus 4 s), never longer than EXPECT_HARD_TTL_S (30 s)
      after it was registered; ``expect_done`` moves the expiry to
      ``done + EXPECT_GRACE_S`` (still capped by the hard TTL);
    * at most EXPECT_MAX entries (oldest dropped);
    * the SAME bars as the phrase files (0.80 whole ratio, 0.90 for a short
      line, <= 2 words exact-only, the verbatim-containment and word-count
      guards), plus a FRAGMENT rule for a line Whisper only caught part of: an
      utterance of at least EXPECTED_FRAGMENT_MIN_WORDS words scored against
      every same-length word window of the line, at
      EXPECTED_FRAGMENT_MIN_RATIO (0.85). Nothing is loosened during a
      dialogue;
    * protected owner phrases and stop words never match (the same early
      returns as the file phrases).
  ``begin_dialogue(source, max_s)`` / ``end_dialogue(token, tail_s)`` bracket
  a dialogue; ``dialogue_active()`` is True from begin until ``tail_s`` after
  end (or ``max_s`` after begin when nobody ends it), so the other listeners
  (the standby loop, the ambient daemon, the learners) can hold still while
  it runs. When the tail ends, every expected entry of that source is
  dropped. A stale token (an older dialogue) can never end a newer one.
  All of this uses the monotonic clock (``now=`` injects one in tests) and is
  lock-protected; it never touches disk.

FAILURE POSTURE: a missing directory, an unreadable or malformed file, or any
internal error means "no filtering" for that file / call. Nothing here raises.
Loaded lists are cached by each file's (mtime, size), so edits apply without a
restart of the matcher but the directory is only re-parsed when it changes.

Log hygiene: callers print the SOURCE name only, never the utterance or the
matched phrase (the phrase list is private).

Pure stdlib (difflib, json, os, re, threading, time).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
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
# Public alias: callers (core/dialogue.py, a skill's line checker) test their
# own text against the SAME list the matcher protects.
STOP_WORDS = _STOP_WORDS

# Expected (transient) device lines: see "EXPECTED LINES / DIALOGUE" above.
EXPECT_MAX = 32
EXPECT_GRACE_S = 4.0
EXPECT_HARD_TTL_S = 30.0
EXPECTED_FRAGMENT_MIN_WORDS = 4
EXPECTED_FRAGMENT_MIN_RATIO = 0.85

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


def _whole_score(fz_words: list, u_wake: int, p_words: list, sm,
                 n_words: int) -> Optional[float]:
    """The fuzzy whole-utterance score of ``fz_words`` against one phrase, or
    None when a guard rules it out or it misses its bar. The rules shared by
    the phrase files and the expected lines (see the module docstring)."""
    if n_words <= EXACT_ONLY_MAX_WORDS:
        return None
    if len(fz_words) <= EXACT_ONLY_MAX_WORDS:
        return None
    if len(fz_words) > n_words + max(1, n_words // 3):
        return None
    if _contains_verbatim(fz_words, p_words):
        return None
    a, b = fz_words, p_words
    if u_wake and p_words[:u_wake] == fz_words[:u_wake]:
        a, b = fz_words[u_wake:], p_words[u_wake:]
    a_s, b_s = " ".join(a), " ".join(b)
    short = (len(b) < SHORT_PHRASE_MIN_WORDS
             or len(b_s) < SHORT_PHRASE_MIN_CHARS)
    need = SHORT_FUZZY_MIN_RATIO if short else FUZZY_MIN_RATIO
    sm.set_seqs(b_s, a_s)
    if sm.real_quick_ratio() < need or sm.quick_ratio() < need:
        return None
    score = sm.ratio()
    return score if score >= need else None


def _fragment_score(fz_words: list, p_words: list, sm) -> Optional[float]:
    """Best score of a SHORTER utterance (at least EXPECTED_FRAGMENT_MIN_WORDS
    words) against every same-length word window of an expected line, or
    None below EXPECTED_FRAGMENT_MIN_RATIO. Covers a device line Whisper only
    caught part of (the capture started late, or the line was split)."""
    k = len(fz_words)
    if k < EXPECTED_FRAGMENT_MIN_WORDS or k >= len(p_words):
        return None
    a_s = " ".join(fz_words)
    best = None
    for i in range(len(p_words) - k + 1):
        b_s = " ".join(p_words[i:i + k])
        sm.set_seqs(b_s, a_s)
        if sm.real_quick_ratio() < EXPECTED_FRAGMENT_MIN_RATIO:
            continue
        if sm.quick_ratio() < EXPECTED_FRAGMENT_MIN_RATIO:
            continue
        score = sm.ratio()
        if score >= EXPECTED_FRAGMENT_MIN_RATIO and (best is None
                                                     or score > best):
            best = score
    return best


def match(utterance, directory: Optional[str] = None, never_match=(),
          wake_phrases=(), *, now: Optional[float] = None):
    """``(source, phrase, score)`` when ``utterance`` is a known device line,
    else None. ``phrase`` is the NORMALISED phrase (never log it).

    Checks the live EXPECTED lines (see expect()) first, then the phrase
    files. ``now``: monotonic time for the expected-line windows (tests).

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
        expected = _expected_snapshot(now)
        phrases = load_phrases(directory)
        if not phrases and not expected:
            return None
        fz_words = _strip_whisper_noise(utt_words)
        fz_utt = " ".join(fz_words)
        wake_seqs = {tuple(w) for w in (normalise(p).split()
                                        for p in (wake_phrases or ())) if w}
        u_wake = _wake_prefix_len(fz_words, wake_seqs)
        sm = difflib.SequenceMatcher(autojunk=False)
        best = None
        for source, phrase, n_words in expected:
            if utt == phrase or fz_utt == phrase:
                return (source, phrase, 1.0)
            p_words = phrase.split()
            whole = _whole_score(fz_words, u_wake, p_words, sm, n_words)
            frag = (_fragment_score(fz_words, p_words, sm)
                    if n_words > EXACT_ONLY_MAX_WORDS else None)
            scores = [x for x in (whole, frag) if x is not None]
            score = max(scores) if scores else None
            if score is not None and (best is None or score > best[2]):
                best = (source, phrase, score)
        if best is not None:
            return best
        for source, phrase, n_words in phrases:
            if utt == phrase or fz_utt == phrase:
                return (source, phrase, 1.0)
            score = _whole_score(fz_words, u_wake, phrase.split(), sm,
                                 n_words)
            if score is not None and (best is None or score > best[2]):
                best = (source, phrase, score)
        return best
    except Exception:
        return None


# ── expected lines + dialogue bracket (transient, in memory) ───────────────
_exp_lock = threading.Lock()
# handle -> {"source", "norm", "n", "expires", "hard"}; insertion-ordered, so
# the first key is the oldest (EXPECT_MAX eviction).
_expected: dict = {}
_exp_next = [0]
# The one dialogue bracket: {"token", "source", "until", "tail_until"} or None.
_dialogue: list = [None]
_dialogue_next = [0]


def _clock(now) -> float:
    return time.monotonic() if now is None else float(now)


def _gc_locked(now: float) -> None:
    """Drop expired entries, and an ended dialogue together with its source's
    entries. Caller holds _exp_lock."""
    d = _dialogue[0]
    if d is not None:
        limit = d["until"] if d["tail_until"] is None else d["tail_until"]
        if now >= limit:
            _dialogue[0] = None
            for h in [h for h, e in _expected.items()
                      if e["source"] == d["source"]]:
                del _expected[h]
    for h in [h for h, e in _expected.items() if now >= e["expires"]]:
        del _expected[h]


def _expected_snapshot(now=None) -> list:
    """[(source, normalised line, word count), ...] of the live expected
    lines, newest first. Never raises."""
    try:
        t = _clock(now)
        with _exp_lock:
            _gc_locked(t)
            return [(e["source"], e["norm"], e["n"])
                    for e in reversed(list(_expected.values()))]
    except Exception:
        return []


def expect(source: str, text: str, *, window_s: float,
           now: Optional[float] = None) -> Optional[int]:
    """Register a line a device is ABOUT to speak. ``window_s`` is how long the
    caller predicts the line takes; the entry stays live until then plus
    EXPECT_GRACE_S, never past EXPECT_HARD_TTL_S. Returns a handle for
    expect_done(), or None for an empty / non-printable text or a blank
    source. Never raises."""
    try:
        if not isinstance(source, str) or not source.strip():
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        if not text.strip().isprintable():
            return None
        norm = normalise(text)
        if not norm:
            return None
        t = _clock(now)
        try:
            w = max(0.0, float(window_s))
        except Exception:
            w = 0.0
        hard = t + EXPECT_HARD_TTL_S
        with _exp_lock:
            _gc_locked(t)
            while len(_expected) >= EXPECT_MAX:
                del _expected[next(iter(_expected))]
            _exp_next[0] += 1
            handle = _exp_next[0]
            _expected[handle] = {
                "source": source.strip(), "norm": norm,
                "n": len(norm.split()),
                "expires": min(hard, t + w + EXPECT_GRACE_S), "hard": hard}
        return handle
    except Exception:
        return None


def expect_done(handle, *, now: Optional[float] = None) -> None:
    """The device finished the line: it stays filtered EXPECT_GRACE_S more
    (the echo and a late transcript), capped by its hard TTL. An unknown or
    expired handle is ignored. Never raises."""
    try:
        t = _clock(now)
        with _exp_lock:
            _gc_locked(t)          # an already-expired entry stays expired
            e = _expected.get(handle)
            if e is not None:
                e["expires"] = min(e["hard"], t + EXPECT_GRACE_S)
    except Exception:
        pass


def forget_expected(source: Optional[str] = None) -> int:
    """Drop every expected line (of ``source`` only, when given). Returns how
    many were dropped. Never raises."""
    try:
        want = None if source is None else str(source).strip()
        with _exp_lock:
            gone = [h for h, e in _expected.items()
                    if want is None or e["source"] == want]
            for h in gone:
                del _expected[h]
            return len(gone)
    except Exception:
        return 0


def begin_dialogue(source: str, max_s: float = 60.0, *,
                   now: Optional[float] = None) -> int:
    """Open the dialogue bracket for ``source`` (it replaces any earlier
    one). It lapses by itself ``max_s`` after begin if nobody ends it.
    Returns the token end_dialogue() needs. Never raises (0 on error)."""
    try:
        t = _clock(now)
        with _exp_lock:
            _gc_locked(t)
            _dialogue_next[0] += 1
            tok = _dialogue_next[0]
            _dialogue[0] = {"token": tok, "source": str(source or "").strip(),
                            "until": t + max(0.0, float(max_s)),
                            "tail_until": None}
            return tok
    except Exception:
        return 0


def end_dialogue(token: int, tail_s: float = 4.0, *,
                 now: Optional[float] = None) -> None:
    """Close the dialogue ``token`` opened: it stays active ``tail_s`` more
    (the last echo), then its source's expected lines are dropped. A stale
    token, or a second end, does nothing. Never raises."""
    try:
        t = _clock(now)
        with _exp_lock:
            d = _dialogue[0]
            if d is None or d["token"] != token or d["tail_until"] is not None:
                return
            d["tail_until"] = min(d["until"], t + max(0.0, float(tail_s)))
            _gc_locked(t)
    except Exception:
        pass


def dialogue_active(*, now: Optional[float] = None) -> bool:
    """True while a dialogue is open or in its end tail. Never raises."""
    try:
        t = _clock(now)
        with _exp_lock:
            _gc_locked(t)
            return _dialogue[0] is not None
    except Exception:
        return False


def dialogue_source(*, now: Optional[float] = None) -> Optional[str]:
    """The active dialogue's source name, else None. Never raises."""
    try:
        t = _clock(now)
        with _exp_lock:
            _gc_locked(t)
            d = _dialogue[0]
            return d["source"] if d is not None else None
    except Exception:
        return None


def _reset_cache_for_tests() -> None:
    with _lock:
        _cache["key"] = None
        _cache["phrases"] = []
    _reported_bad.clear()
    with _exp_lock:
        _expected.clear()
        _dialogue[0] = None
