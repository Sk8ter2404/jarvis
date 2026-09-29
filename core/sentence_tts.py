"""core/sentence_tts.py -- per-sentence speech for a reply that is already known.

`_speak` used to render the WHOLE reply and only then start playing it, so the
listener waited for the full synthesis of a multi-sentence answer before
hearing its first word. This module lets `_speak` start playing sentence 1 as
soon as it is rendered while a worker thread renders the rest.

Unlike streaming the LLM (core/stream_speech.py), the complete reply and all of
its actions are already settled when `_speak` runs, so voicing it sentence by
sentence has none of the early-speech risks: the words and their order are
exactly the words `_speak` was handed.

Two pieces, both pure (no audio, no monolith import) so they unit-test without
a speaker:

  * `split_sentences` / `plan_chunks` -- a CONSERVATIVE splitter. A missed
    split only costs latency; a wrong split would break the prosody of a
    phrase, so every doubtful boundary stays joined (decimals, abbreviations,
    initials, "2:30 p.m.", ellipses).
  * `play_pipelined` -- renders chunk 1, plays it, and meanwhile renders the
    rest on ONE worker thread. The caller's thread makes every `play` call, in
    order, one per chunk, one at a time (never two at once), so the single
    playback stream, lip-sync and ducking behave exactly as for one utterance.

The monolith injects `synth`, `play` and `should_stop` (the interrupt / mute
check), so this module never touches sounddevice.

One deliberate exception to "renders happen only while the speaker lock is
held": when a reply is STOPPED, the render already in flight on the worker
cannot be cancelled (a native ONNX run), so it finishes -- at most that one
sentence -- after `play_pipelined` has returned, and its audio is discarded.
The worker starts no further render once stopped. Joining it instead would add
up to a sentence's render time to every barge-in before the mic reopens.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, List, Optional, Tuple

__all__ = ["MIN_CHARS", "SENTENCE_GAP_S", "split_sentences", "plan_chunks",
           "play_pipelined", "PipelineResult", "SentenceFallback"]

# Below this many characters a reply is voiced whole, exactly as before: the
# render of a short reply is already quick, and one play call is one stream.
MIN_CHARS = 120

# Silence appended after every sentence but the reply's last. Kokoro trims the
# leading/trailing silence off each render, so back-to-back sentence clips
# would lose the natural pause a whole-text render keeps at each full stop.
# Separate plays also pay ~0.1 s of stream setup, so the heard gap is a little
# longer than this.
SENTENCE_GAP_S = 0.15

# Words that end in "." without ending a sentence. Compared lower-case, with
# the dot removed. Deliberately generous: a missed split only costs latency.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "gen", "col",
    "lt", "sgt", "capt", "cmdr", "gov", "sen", "rep", "hon", "mt", "ft",
    "ave", "blvd", "rd", "ln", "apt", "dept", "est", "inc", "ltd", "co",
    "corp", "vs", "etc", "approx", "fig", "vol", "ch", "sec", "min", "max",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun", "am", "pm", "eg", "ie", "cf", "al", "ca",
})

# A candidate boundary: terminator run, optional closing quotes/brackets, then
# whitespace, then (lookahead) an uppercase letter or a digit, optionally
# preceded by ONE opening quote/bracket (a quote alone is not enough -- the
# quoted word must itself start with a capital or a digit). An ellipsis is
# matched as a candidate only so _is_real_boundary can refuse it.
_BOUNDARY_RE = re.compile(
    r"([.!?\u2026]+)([\"')\]\u201d\u2019]*)(\s+)"
    r"(?=[\"'(\[\u201c\u2018]?[A-Z0-9])")

_LEADING_OPENERS = "\"'([\u201c\u2018"
_DOTTED_ABBREV_RE = re.compile(r"(?:[A-Za-z]{1,3}\.)+[A-Za-z]{1,3}")


def _is_real_boundary(text: str, m: "re.Match[str]",
                      piece_start: int = 0) -> bool:
    """False for every doubtful '.' boundary (and every ellipsis).
    `piece_start` is where the current (unsplit) piece begins."""
    term = m.group(1)
    # Ellipses ("...", "\u2026", "?..") are a pause inside a thought.
    if "\u2026" in term or ".." in term:
        return False
    if not term.endswith("."):
        return True                       # "!" / "?" -- always a sentence end
    # The token immediately before the dot.
    start = m.start(1)
    tok_start = start
    while tok_start > 0 and not text[tok_start - 1].isspace():
        tok_start -= 1
    token = text[tok_start:start].lstrip(_LEADING_OPENERS)
    if not token:
        return False
    low = token.lower()
    # "e.g." / "i.e." / "p.m." / "U.S." / "Ph.D." -- a dotted run of short
    # letter groups is an abbreviation ("2.718." and "v2.0.115." are not).
    if _DOTTED_ABBREV_RE.fullmatch(token):
        return False
    if low in _ABBREVIATIONS:
        return False
    # Initials ("J. Smith") -- a lone letter before the dot.
    if len(token) == 1 and token.isalpha():
        return False
    # "1. Open it. 2. Close it." / "Steps: 1. Open it." -- a bare list number
    # opening a piece (or right after a colon) is not a sentence of its own
    # ("1." would be voiced alone). "I counted 12. Then..." still splits.
    if token.isdigit() and len(token) <= 2:
        before = text[piece_start:tok_start].rstrip()
        if not before or before.endswith(":"):
            return False
    # "No. 5" -- "No." before a number is the abbreviation for number.
    after = m.end()
    if low == "no" and after < len(text) and text[after].isdigit():
        return False
    return True


def split_sentences(text: str) -> List[str]:
    """Split `text` into sentences, conservatively. Never returns an empty
    list for non-blank text; the pieces are stripped and, joined with single
    spaces, read exactly as the whitespace-normalised input."""
    t = (text or "").strip()
    if not t:
        return []
    out: List[str] = []
    last = 0
    for m in _BOUNDARY_RE.finditer(t):
        if not _is_real_boundary(t, m, last):
            continue
        cut = m.end(2)                    # keep the terminator + closing quote
        piece = t[last:cut].strip()
        if piece:
            out.append(piece)
        last = m.end(3)
    tail = t[last:].strip()
    if tail:
        out.append(tail)
    return out or [t]


def plan_chunks(text: str, min_chars: int = MIN_CHARS) -> List[str]:
    """The chunks `_speak` voices: the whole text as ONE chunk when it is short
    or a single sentence (today's path), else one chunk per sentence."""
    t = (text or "").strip()
    if not t:
        return []
    if len(t) < min_chars:
        return [t]
    parts = split_sentences(t)
    return parts if len(parts) > 1 else [t]




class SentenceFallback(Exception):
    """Raised by a `synth` that could not voice one sentence on its own (the
    per-sentence engine timed out or returned nothing). `play_pipelined` then
    voices that sentence and every one after it as ONE block through
    `synth_rest` -- one fallback render in one voice, instead of paying the
    engine's timeout again for each remaining sentence."""


class PipelineResult:
    """What `play_pipelined` did.

    plays            -- play calls made (one per chunk; a fallback block is one)
    sentences_played -- chunks covered by those plays
    stopped          -- a stop cut the reply short
    error            -- the exception that ended a reply AFTER at least one
                        sentence was heard (None otherwise); an error before
                        anything played is raised instead
    fell_back        -- the rest of the reply went through `synth_rest`"""

    __slots__ = ("plays", "sentences_played", "stopped", "error", "fell_back")

    def __init__(self) -> None:
        self.plays = 0
        self.sentences_played = 0
        self.stopped = False
        self.error: Optional[BaseException] = None
        self.fell_back = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PipelineResult(plays={self.plays}, "
                f"sentences_played={self.sentences_played}, "
                f"stopped={self.stopped}, error={self.error!r}, "
                f"fell_back={self.fell_back})")


_DONE = object()


def play_pipelined(
    chunks: List[str],
    synth: Callable[[str], Tuple[object, int]],
    play: Callable[[object, int], None],
    should_stop: Callable[[], bool],
    *,
    synth_rest: Optional[Callable[[str], Tuple[object, int]]] = None,
    pad: Optional[Callable[[object, int], object]] = None,
    on_first_play: Optional[Callable[[], None]] = None,
    wait_timeout: float = 90.0,
    poll_s: float = 0.05,
) -> PipelineResult:
    """Render chunk 1, play it; while it plays, ONE worker renders the rest.

    * The CALLER's thread makes every `play` call, sequentially, exactly one
      per chunk (a `synth_rest` block counts as one) -- never two at once, so
      one playback stream at a time.
    * `should_stop()` is checked while waiting for the next render (every
      `poll_s`) and again right before each play; a True ends the reply there
      and tells the worker to START no further render. A render already in
      flight finishes on the worker and is discarded (it cannot be cancelled;
      the caller does not wait for it).
    * `synth` may raise `SentenceFallback`: that chunk and all after it are
      rendered as one block by `synth_rest` (re-raised when it is None).
    * `pad(audio, sr)` is applied to every piece except the reply's last, so
      each sentence boundary gets the same pause (the engine trims each
      render's trailing silence).
    * Errors: one before anything was heard (chunk 1's render or play) is
      raised, so the caller's whole-text error path runs. One after at least
      one sentence was heard -- a later render or play error, or no render
      within `wait_timeout` of waiting -- ends the reply there and is
      returned as `result.error`: those sentences WERE spoken.
    """
    res = PipelineResult()
    if not chunks:
        return res
    n = len(chunks)

    def _render(i: int):
        """(audio, sr, chunks covered) for chunk i, padded unless last."""
        try:
            audio, sr = synth(chunks[i])
            covered = 1
        except SentenceFallback:
            if synth_rest is None:
                raise
            res.fell_back = True
            audio, sr = synth_rest(" ".join(chunks[i:]))
            covered = n - i
        if pad is not None and i + covered < n:
            audio = pad(audio, sr)
        return audio, sr, covered

    first = _render(0)
    q: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    next_i = first[2]

    def _worker() -> None:
        i = next_i
        try:
            while i < n:
                if stop.is_set():
                    return
                try:
                    item = _render(i)
                except BaseException as e:  # noqa: BLE001 - handed to caller
                    q.put(("err", e))
                    return
                q.put(("ok", item))
                i += item[2]
        finally:
            q.put(_DONE)

    def _wait_next():
        """The worker's next item, or None when a stop landed first."""
        deadline = time.monotonic() + wait_timeout
        while True:
            if should_stop():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"sentence render exceeded {wait_timeout:.0f}s")
            try:
                return q.get(timeout=min(poll_s, remaining))
            except queue.Empty:
                continue

    if next_i < n:
        threading.Thread(target=_worker, name="sentence-tts-synth",
                         daemon=True).start()
    try:
        if on_first_play is not None:
            on_first_play()
        play(first[0], first[1])
        res.plays += 1
        res.sentences_played += first[2]
        while res.sentences_played < n:
            item = _wait_next()
            if item is None:
                res.stopped = True
                return res
            if item is _DONE:           # worker ended early (never expected)
                return res
            kind, val = item
            if kind == "err":
                raise val
            if should_stop():
                res.stopped = True
                return res
            audio, sr, covered = val
            play(audio, sr)
            res.plays += 1
            res.sentences_played += covered
        return res
    except Exception as e:  # noqa: BLE001
        if res.sentences_played == 0:
            raise
        res.error = e
        return res
    finally:
        stop.set()
