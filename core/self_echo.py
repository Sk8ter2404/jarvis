"""core/self_echo.py — JARVIS must never answer his own voice.

WHY THIS MODULE EXISTS
======================
The main loop's mic capture (record_speech) and JARVIS's own speech were only
ever kept apart by ORDER: the main thread speaks, then captures, so the two
never overlap. Nothing gated a line spoken from ANOTHER thread (the tray
drainer, a timer, a proactive/background announcement) while the main loop was
mid-capture. Live 2026-09-29: the tray's force_wake said "At your service,
sir." while the loop was listening; the desk mic picked it up (VAD peak 0.016
over a 0.008 threshold), Whisper transcribed it, and the turn ran as the owner
— JARVIS answered himself, four times in two minutes.

Two independent layers, both fed by the monolith:

1. TIMING (``capture_overlap``). play_with_lipsync registers every audible
   playback interval (``playback_begin`` / ``playback_end``); record_speech
   reports when its stream opened, when the VAD tripped and when the capture
   ended. A capture whose utterance overlaps a playback, or whose VAD tripped
   within ``tail_s`` after one ended, is JARVIS's own voice — provided the
   capture stream was already open while that playback was live. A capture
   that opened only AFTER the line finished (the ordinary speak-then-listen
   sequence on the main thread) is deliberately NOT tail-gated: it cannot
   contain the line, and a tail there would swallow a quick "yes" to JARVIS's
   own question.

2. CONTENT (``remember`` / ``match``), independent of timing. Every line
   _speak voices is remembered for ``window_s`` (counted from when it finished
   playing). A transcript that fuzzy-matches a remembered line is an echo:
   normalised (core/device_speech_filter.normalise), whole-utterance
   difflib ratio >= FUZZY_MIN_RATIO when both sides have 3+ words, exact
   normalised equality when either side has 1-2 words. Each sentence of a
   multi-sentence line is also remembered (3+ word sentences only), since the
   mic may catch just one of them.

   Physics bound (``clip_start``, reviewer fix 2026-09-29): when the caller
   knows where the captured clip begins, a line that had FINISHED playing
   before that instant cannot be in the clip, so it is not compared at all.
   Without this the content layer ate the owner in the ordinary
   speak-then-listen turn: JARVIS says "Turning off the desk lamp." (or asks
   "Should I lock the front door?"), the lamp stays on, the owner says "turn
   off the desk lamp" (or answers "lock the front door") a second later in a
   capture that opened only after the line ended — a char-level ratio of
   0.86 (0.81), silently dropped as an echo. The bound uses the content
   layer's OWN line timestamps (_speak's remember / refresh), not the
   playback registry, so it still backstops a timing-layer miss. With no
   clip bound (the realtime path) the match is purely on content.

The CALLER (bobert_companion._self_echo_ignored) owns the policy exemptions
that must stay identical to the existing rules: stop words always pass, a
wake-word transcript overlapping a playback passes unless that playback itself
said "jarvis" (the barge-in echo gate), and typed / injected turns are never
checked. ``match`` additionally never fires on a stop word, the caller's
``protected`` phrases (wake / sleep phrases) or the device filter's owner
vocabulary ("yes", "okay", ...), so a remembered "Okay, sir." can never eat the
owner's "okay".

Guards copied from core/device_speech_filter.py so the two filters can be
unified later (a parallel feature may add an expect(source, text, window_s)
API there; this module deliberately does not depend on it):
  * leading Whisper fillers and one trailing hallucinated "you" /
    "thank you" are dropped before the fuzzy rule;
  * an utterance that CONTAINS a remembered line verbatim inside a longer
    sentence is the owner quoting / extending it, never an echo;
  * the utterance may run at most max(1, words // 3) words longer than the
    line.

Log hygiene: nothing here logs. Callers print numbers only (score, age, gap),
never the transcript or the remembered line.

Clock: every timestamp comes from ``now()`` (``time.monotonic`` by default),
so record_speech, play_with_lipsync and the gate share one clock and tests
can freeze it by patching ``_clock``.

FAILURE POSTURE: nothing here raises; any internal error means "not an echo".
Pure stdlib (collections, difflib, re, threading, time).
"""
from __future__ import annotations

import collections
import difflib
import itertools
import re
import threading
import time
from typing import Optional

from core import device_speech_filter as _dsf

DEFAULT_WINDOW_S = 20.0
DEFAULT_TAIL_S = 0.8
FUZZY_MIN_RATIO = 0.80
EXACT_ONLY_MAX_WORDS = 2

# Bounded memory: far more than a 20 s window can ever hold.
_MAX_LINES = 64
_MAX_PLAYBACKS = 64
# Playbacks are kept this long after they end (the tail is sub-second; the
# slack covers a long capture that started before a playback ended).
_PLAYBACK_KEEP_S = 120.0

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")

_clock = time.monotonic
_lock = threading.Lock()
_ids = itertools.count(1)
# token -> [start, end or None (still playing), says_jarvis]
_playbacks: "collections.OrderedDict[int, list]" = collections.OrderedDict()
# token -> [ts (refreshed when the line finishes), [(norm, n_words), ...],
#          finished (True once refresh() ran: ts is then the line's END)]
_lines: "collections.OrderedDict[int, list]" = collections.OrderedDict()


def now() -> float:
    """The shared self-echo clock (monotonic seconds)."""
    try:
        return float(_clock())
    except Exception:
        return time.monotonic()


def _owner_protected() -> frozenset:
    base = getattr(_dsf, "_OWNER_PROTECTED", None)
    if isinstance(base, frozenset):
        return base
    return frozenset(_dsf.normalise(p) for p in _dsf.OWNER_PHRASES) - {""}


def _strip_noise(words: list) -> list:
    fn = getattr(_dsf, "_strip_whisper_noise", None)
    if callable(fn):
        try:
            return fn(words)
        except Exception:
            pass
    return list(words)


def _contains_verbatim(utt_words: list, line_words: list) -> bool:
    n, m = len(utt_words), len(line_words)
    if m == 0 or n <= m:
        return False
    return any(utt_words[i:i + m] == line_words for i in range(n - m + 1))


# ── timing layer ───────────────────────────────────────────────────────────

def playback_begin(text: str = "", at: Optional[float] = None) -> int:
    """Register an AUDIBLE playback starting now (or at ``at``). ``text`` is
    only inspected for the word "jarvis" (the barge-in echo rule); it is not
    stored. Returns a token for ``playback_end``. Never raises."""
    try:
        t = now() if at is None else float(at)
        says = "jarvis" in (text or "").lower()
        with _lock:
            tok = next(_ids)
            _playbacks[tok] = [t, None, says]
            _prune_playbacks_locked(t)
        return tok
    except Exception:
        return 0


def playback_end(token: int, at: Optional[float] = None) -> None:
    """Mark the playback ``token`` finished. Never raises."""
    try:
        t = now() if at is None else float(at)
        with _lock:
            p = _playbacks.get(token)
            if p is not None and p[1] is None:
                p[1] = max(t, p[0])
    except Exception:
        pass


def playback_live() -> bool:
    """True while any registered playback has not ended."""
    try:
        with _lock:
            return any(p[1] is None for p in _playbacks.values())
    except Exception:
        return False


def last_playback_end() -> Optional[float]:
    """When JARVIS's latest audible playback ended (the shared clock):
    now() while one is still playing, None when none is remembered. The
    always-open mic (bobert_companion, MIC_BUS_MODE) never starts a capture
    from ring audio older than this (2026-10-09 review: his own last words
    sat in the next capture's pre-roll). Never raises (None)."""
    try:
        with _lock:
            if not _playbacks:
                return None
            if any(p[1] is None for p in _playbacks.values()):
                live = True
                end = None
            else:
                live = False
                end = max(float(p[1]) for p in _playbacks.values())
        return now() if live else end
    except Exception:
        return None


def _prune_playbacks_locked(t: float) -> None:
    for tok in [k for k, p in _playbacks.items()
                if p[1] is not None and t - p[1] > _PLAYBACK_KEEP_S]:
        del _playbacks[tok]
    while len(_playbacks) > _MAX_PLAYBACKS:
        _playbacks.popitem(last=False)


def capture_overlap(open_ts: float, vad_ts: float, end_ts: float,
                    tail_s: float = DEFAULT_TAIL_S,
                    at: Optional[float] = None) -> Optional[dict]:
    """Did this capture hear JARVIS's own playback?

    ``open_ts``: the capture stream started; ``vad_ts``: the VAD tripped (the
    utterance began); ``end_ts``: the capture ended. A playback [start, end]
    (end = now while still playing) is heard when

        start <= end_ts            (it began before the capture ended),
        open_ts <= end             (the stream was open while it played), and
        vad_ts <= end + tail_s     (the utterance began during it or within
                                    the tail after it).

    Returns None when no playback qualifies, else
    ``{"gap": vad_ts - end of the latest-ending hit (negative = the utterance
    began while it was still playing), "says_jarvis": any hit said "jarvis",
    "count": number of hits}``. Never raises (None on error)."""
    try:
        t = now() if at is None else float(at)
        tail = max(0.0, float(tail_s))
        o, v, e = float(open_ts), float(vad_ts), float(end_ts)
        hits = []
        with _lock:
            for start, end, says in _playbacks.values():
                p_end = t if end is None else end
                if start <= e and o <= p_end and v <= p_end + tail:
                    hits.append((p_end, says))
        if not hits:
            return None
        latest = max(h[0] for h in hits)
        return {"gap": v - latest,
                "says_jarvis": any(h[1] for h in hits),
                "count": len(hits)}
    except Exception:
        return None


# ── content layer ──────────────────────────────────────────────────────────

def _variants(text: str) -> list:
    """[(normalised, word count), ...]: the whole line, plus each 3+ word
    sentence of a multi-sentence line. Deduplicated, order kept."""
    out, seen = [], set()
    whole = _dsf.normalise(text)
    if whole:
        out.append((whole, len(whole.split())))
        seen.add(whole)
    parts = _SENTENCE_SPLIT_RE.split(text or "")
    if len(parts) > 1:
        for part in parts:
            norm = _dsf.normalise(part)
            n = len(norm.split())
            if norm and n > EXACT_ONLY_MAX_WORDS and norm not in seen:
                seen.add(norm)
                out.append((norm, n))
    return out


def remember(text: str, at: Optional[float] = None) -> int:
    """Remember a line JARVIS is about to speak. Returns a token for
    ``refresh`` (call it when the line finishes, so the window counts from
    the end of a long reply). 0 when there was nothing to remember. Never
    raises."""
    try:
        variants = _variants(text)
        if not variants:
            return 0
        t = now() if at is None else float(at)
        with _lock:
            tok = next(_ids)
            _lines[tok] = [t, variants, False]
            while len(_lines) > _MAX_LINES:
                _lines.popitem(last=False)
        return tok
    except Exception:
        return 0


def refresh(token: int, at: Optional[float] = None) -> None:
    """Restart the window of the remembered line ``token``. Never raises."""
    try:
        if not token:
            return
        t = now() if at is None else float(at)
        with _lock:
            entry = _lines.get(token)
            if entry is not None:
                entry[0] = max(entry[0], t)
                entry[2] = True
    except Exception:
        pass


def match(utterance: str, window_s: float = DEFAULT_WINDOW_S, protected=(),
          at: Optional[float] = None,
          clip_start: Optional[float] = None) -> Optional[tuple]:
    """``(score, age_s)`` when ``utterance`` is an echo of a line remembered
    within the last ``window_s`` seconds, else None. ``clip_start`` (optional,
    same clock): the earliest instant the captured clip can contain — a line
    that finished playing before it is skipped (see the module docstring's
    physics bound); a line still playing is always compared. See the module
    docstring for the rules. Never raises (None on error)."""
    try:
        utt = _dsf.normalise(utterance)
        if not utt:
            return None
        if _dsf.has_stop_word(utt):
            return None
        prot = _owner_protected()
        if protected:
            prot = prot | {_dsf.normalise(p) for p in protected}
        if utt in prot:
            return None
        t = now() if at is None else float(at)
        window = max(0.0, float(window_s))
        bound = None if clip_start is None else float(clip_start)
        with _lock:
            for tok in [k for k, v in _lines.items() if t - v[0] > window]:
                del _lines[tok]
            live = [(v[0], list(v[1])) for v in _lines.values()
                    if bound is None or not v[2] or v[0] >= bound]
        if not live:
            return None
        utt_words = utt.split()
        fz_words = _strip_noise(utt_words)
        fz_utt = " ".join(fz_words)
        best = None
        sm = difflib.SequenceMatcher(autojunk=False)
        for ts, variants in live:
            age = max(0.0, t - ts)
            for norm, n_words in variants:
                if utt == norm or fz_utt == norm:
                    return (1.0, age)
                if n_words <= EXACT_ONLY_MAX_WORDS:
                    continue
                if len(fz_words) <= EXACT_ONLY_MAX_WORDS:
                    continue
                if len(fz_words) > n_words + max(1, n_words // 3):
                    continue
                line_words = norm.split()
                if _contains_verbatim(fz_words, line_words):
                    continue
                sm.set_seqs(norm, fz_utt)
                if (sm.real_quick_ratio() < FUZZY_MIN_RATIO
                        or sm.quick_ratio() < FUZZY_MIN_RATIO):
                    continue
                score = sm.ratio()
                if score >= FUZZY_MIN_RATIO and (best is None
                                                 or score > best[0]):
                    best = (score, age)
        return best
    except Exception:
        return None


def _reset_for_tests() -> None:
    with _lock:
        _playbacks.clear()
        _lines.clear()
