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
held": when a reply is STOPPED, a Kokoro render already in flight on the
worker cannot be cancelled (a native ONNX run), so it finishes -- at most that
one sentence -- after `play_pipelined` has returned, and its audio is
discarded. The worker starts no further render once stopped. Joining it
instead would add up to a sentence's render time to every barge-in before the
mic reopens. (A clone voice server render is an HTTP wait, which CAN be
abandoned: `reply_stopped()` below.)

The clone voice server (2026-10-04) adds three things, all inert for Kokoro:

  * `needed_by()` -- while `play_pipelined` renders a chunk AHEAD of playback,
    the time that chunk will actually be needed (when the audio queued ahead
    of it runs out). A slow engine can wait that long instead of giving the
    line to a fallback voice mid-reply. None for a reply's first chunk.
  * `reply_stopped()` -- for the same renders, an Event set once the reply
    has ended (stopped, failed or finished): after that nothing will play
    the chunk, so an engine that is still waiting can give up at once.
  * `plan_clone_chunks` -- a long first line is split at a clause boundary
    into a short head (first audio sooner) and the rest (rendered while the
    head plays), joined by a short pause (`Chunk.gap_s`).
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Callable, List, Optional, Tuple

__all__ = ["MIN_CHARS", "SENTENCE_GAP_S", "split_sentences", "plan_chunks",
           "play_pipelined", "PipelineResult", "SentenceFallback",
           "CLAUSE_SPLIT_MIN_CHARS", "CLAUSE_GAP_S", "Chunk",
           "split_first_clause", "plan_clone_chunks", "needed_by",
           "reply_stopped"]

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


# ═══════════════════════════════════════════════════════════════════════════
#  First-line clause split (the clone voice server only, 2026-10-04)
# ═══════════════════════════════════════════════════════════════════════════
# The clone renders one line in roughly 0.3 s + ~25 ms per character on a
# busy GPU (measured live 2026-10-04: 18 chars 1.0 s, 116 chars 3.2 s), so a
# long first line kept the listener waiting ~3 s for the first word. Its
# audio runs ~55 ms per character, so a head of ~2/5 of the line plays long
# enough to cover the render of the rest.
#
# A first line longer than this is split at a clause boundary.
CLAUSE_SPLIT_MIN_CHARS = 70
# The head is at least this long (no "Sir," on its own) ...
HEAD_MIN_CHARS = 12
# ... and leaves at least this much for the rest. (With the thresholds below
# HEAD_MAX_FRAC already leaves a line over CLAUSE_SPLIT_MIN_CHARS a rest of
# 24+ chars, so this only bites if those thresholds are ever lowered.)
REST_MIN_CHARS = 20
# Among the usable boundaries, the one whose head is nearest this share of
# the line wins; a head longer than HEAD_MAX_FRAC of it saves too little.
HEAD_TARGET_FRAC = 0.4
HEAD_MAX_FRAC = 0.65
# Silence after the head. The clone keeps ~0.10 s after a render and ~0.04 s
# before one, and each play pays ~0.1 s of stream setup, so the heard pause is
# ~0.3 s: a comma, not a full stop (SENTENCE_GAP_S).
CLAUSE_GAP_S = 0.05

_CLAUSE_PUNCT = ",;:"
_DASHES = "-–—"
# A spaced dash between two of these is a range, not a clause break:
# 'Monday - Friday', 'noon - 2 pm' (a number on both sides is one too).
_RANGE_WORDS = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "noon", "midnight",
})
_WORD_RE = re.compile(r"[^\s\-–—]+")
_WORD_EDGE = ".,;:!?\"'()[]“”‘’"


class Chunk(str):
    """A chunk with hints for the player and the clone engine; a plain str to
    everything else (joins and slices of it are plain str again).

    gap_s        -- silence after this chunk (None: the player's own pad)
    budget_chars -- the clone's render budget is computed for this many
                    characters (None: its own length). The rest of a split
                    first line keeps the WHOLE line's budget, so splitting
                    never gives it less time than the unsplit line had.
    clause       -- 'head' / 'tail' for the two halves of a split line."""

    gap_s: Optional[float] = None
    budget_chars: Optional[int] = None
    clause: str = ""

    def __new__(cls, text: str, *, gap_s: Optional[float] = None,
                budget_chars: Optional[int] = None, clause: str = ""):
        obj = super().__new__(cls, text)
        obj.gap_s = gap_s
        obj.budget_chars = budget_chars
        obj.clause = clause
        return obj


def _token_before(text: str, end: int) -> str:
    """The whitespace-delimited token ending at `end` (exclusive)."""
    start = end
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:end]


def _is_abbreviation(token: str) -> bool:
    """'e.g.' / 'etc.' / 'a.m.' / 'Dr.' -- a token that ends in an
    abbreviation's dot."""
    tok = token.lstrip(_LEADING_OPENERS)
    if not tok.endswith("."):
        return False
    low = tok.rstrip(".").lower()
    return (low in _ABBREVIATIONS or bool(_DOTTED_ABBREV_RE.fullmatch(tok))
            or bool(_DOTTED_ABBREV_RE.fullmatch(tok.rstrip(".")))
            or (len(low) == 1 and low.isalpha()))


def _inside_quotes(head: str) -> bool:
    """An opened quote or bracket is still open at the end of `head`.

    Single quotes double as apostrophes, so one between two letters
    ("don't", "it’s") is not a quote mark. Any other straight ' counts both
    ways (an odd number is open -- a possessive "students'" then reads as
    open, which only costs a missed split); a curly ‘ opens and a ’ that is
    not an apostrophe closes."""
    straight = curly_close = 0
    for k, c in enumerate(head):
        if c not in "'’":
            continue
        prev = head[k - 1] if k > 0 else ""
        nxt = head[k + 1] if k + 1 < len(head) else ""
        if prev.isalpha() and nxt.isalpha():
            continue                              # an apostrophe
        if c == "'":
            straight += 1
        else:
            curly_close += 1
    return (head.count('"') % 2 == 1
            or head.count("“") > head.count("”")
            or straight % 2 == 1
            or head.count("‘") > curly_close
            or head.count("(") > head.count(")")
            or head.count("[") > head.count("]"))


def _range_end(word: str) -> bool:
    """A word that can end a range: has a digit, or is a weekday, a month,
    noon or midnight."""
    w = word.strip(_WORD_EDGE).lower()
    return any(c.isdigit() for c in w) or w in _RANGE_WORDS


def _is_range_dash(text: str, i: int) -> bool:
    """The dash at text[i] joins the two ends of a range or a score: a range
    end within two words on BOTH sides -- '31 - 17', '2 pm - 4 pm',
    '5 km - 8 km', 'Monday - Friday', 'noon - 2 pm'."""
    before = _WORD_RE.findall(text[:i])[-2:]
    after = _WORD_RE.findall(text[i + 1:])[:2]
    return any(map(_range_end, before)) and any(map(_range_end, after))


def _clause_candidates(text: str) -> List[Tuple[str, str]]:
    """Every (head, rest) split at a clause boundary that is safe to pause
    at: ', ' / '; ' / ': ', a spaced dash (' - ', ' – ', ' — ') or
    an em dash between words. Never inside a number ('1,500', '2:30',
    '3, 4'), a range or a score ('3 - 5', '102 – 98', '2 pm - 4 pm',
    'Monday - Friday'), after an abbreviation ('e.g.,', 'a.m.,'), or inside
    quotes or brackets. The punctuation stays with the head."""
    out: List[Tuple[str, str]] = []
    n = len(text)
    for i, ch in enumerate(text):
        nxt = text[i + 1] if i + 1 < n else ""
        prev = text[i - 1] if i > 0 else ""
        if ch in _CLAUSE_PUNCT:
            if not nxt.isspace():
                continue                      # '1,500', '2:30', 'a,b'
            after = text[i + 1:].lstrip()
            if prev.isdigit() and after[:1].isdigit():
                continue                      # '3, 4', '10: 30'
            if _is_abbreviation(_token_before(text, i)):
                continue                      # 'e.g.,' / 'a.m.,'
        elif ch in _DASHES:
            spaced = prev.isspace() and nxt.isspace()
            joined = (ch == "—" and prev.isalpha() and nxt.isalpha())
            if not (spaced or joined):
                continue                      # 'well-known', '5-3'
            if _is_range_dash(text, i):
                continue                      # '3 - 5', 'Monday - Friday'
        else:
            continue
        head = text[:i + 1].strip()
        rest = text[i + 1:].strip()
        if not head or not rest or _inside_quotes(head):
            continue
        out.append((head, rest))
    return out


def split_first_clause(text: str) -> Optional[Tuple[str, str]]:
    """(head, rest) for a line longer than CLAUSE_SPLIT_MIN_CHARS, split at the
    safe clause boundary whose head is nearest HEAD_TARGET_FRAC of the line
    (head HEAD_MIN_CHARS..HEAD_MAX_FRAC of it, rest >= REST_MIN_CHARS); None
    when the line is short or no boundary qualifies -- a missed split only
    costs latency, a wrong one breaks the phrase."""
    t = (text or "").strip()
    n = len(t)
    if n <= CLAUSE_SPLIT_MIN_CHARS:
        return None
    best = None
    for head, rest in _clause_candidates(t):
        h = len(head)
        if h < HEAD_MIN_CHARS or h > HEAD_MAX_FRAC * n:
            continue
        if len(rest) < REST_MIN_CHARS:
            continue
        score = abs(h - HEAD_TARGET_FRAC * n)
        if best is None or score < best[0]:
            best = (score, head, rest)
    return None if best is None else (best[1], best[2])


def plan_clone_chunks(text: str, min_chars: int = MIN_CHARS) -> List[str]:
    """`plan_chunks` for the clone voice server: the same chunks, except that
    a first line longer than CLAUSE_SPLIT_MIN_CHARS is voiced in pieces so
    the first audio comes sooner --

      * a short reply of several sentences (one chunk for Kokoro) is voiced
        sentence by sentence, every sentence after the first a Chunk that
        keeps the whole reply's render budget;
      * a first sentence still that long is split at a clause
        (split_first_clause) into a head Chunk (gap_s CLAUSE_GAP_S) and a
        tail Chunk that keeps the whole sentence's render budget (the whole
        reply's, when the sentence came from such a short reply).

    No piece ever gets less render budget than the unsplit text it came from
    had as one render. Never used for Kokoro (its replies are planned by
    plan_chunks)."""
    chunks = plan_chunks(text, min_chars)
    if not chunks or len(chunks[0]) <= CLAUSE_SPLIT_MIN_CHARS:
        return chunks
    budget = len(chunks[0])
    if len(chunks) == 1:
        parts = split_sentences(chunks[0])
        if len(parts) > 1:
            chunks = [parts[0]] + [Chunk(p, budget_chars=budget)
                                   for p in parts[1:]]
            if len(chunks[0]) <= CLAUSE_SPLIT_MIN_CHARS:
                return chunks
    first = chunks[0]
    cut = split_first_clause(first)
    if cut is None:
        return chunks
    head, rest = cut
    return ([Chunk(head, gap_s=CLAUSE_GAP_S, clause="head"),
             Chunk(rest, budget_chars=max(budget, len(first)), clause="tail")]
            + list(chunks[1:]))


# ═══════════════════════════════════════════════════════════════════════════
#  When a chunk rendered ahead is needed (2026-10-04)
# ═══════════════════════════════════════════════════════════════════════════
_render_ctx = threading.local()


def needed_by() -> Optional[float]:
    """For a render running inside `play_pipelined` on THIS thread: the
    time.monotonic() at which its chunk will be needed -- when the sentence
    playing now and every rendered sentence waiting behind it have played
    (an estimate from their durations; the real moment is a little later,
    each play pays its stream setup). None for a reply's first chunk (nothing
    plays ahead of it: the listener is waiting now) and for any render
    outside play_pipelined (a whole-text reply, the R3 pre-render, the
    filler)."""
    return getattr(_render_ctx, "needed_by", None)


def reply_stopped() -> Optional[threading.Event]:
    """For a render running inside `play_pipelined`'s worker on THIS thread:
    an Event set once the reply has ended -- stopped by the listener, failed,
    or finished -- after which nothing will play the chunk being rendered.
    An engine that waits on something it can abandon (the clone voice
    server's HTTP reply) checks it and gives up at once; a native render
    simply finishes and is discarded. None for a reply's first chunk (it is
    rendered before playback starts, on the caller's thread) and for any
    render outside play_pipelined."""
    return getattr(_render_ctx, "stop", None)


def _duration_s(audio, sr) -> float:
    """Seconds of `audio` at `sr`; 0.0 when it cannot be told. Never raises."""
    try:
        return max(0.0, float(len(audio)) / float(sr))
    except Exception:
        return 0.0


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
    first_rendered: Optional[Tuple[object, int]] = None,
) -> PipelineResult:
    """Render chunk 1, play it; while it plays, ONE worker renders the rest.

    * The CALLER's thread makes every `play` call, sequentially, exactly one
      per chunk (a `synth_rest` block counts as one) -- never two at once, so
      one playback stream at a time.
    * `should_stop()` is checked while waiting for the next render (every
      `poll_s`) and again right before each play; a True ends the reply there
      and tells the worker to START no further render. A render already in
      flight finishes on the worker and is discarded (the caller does not
      wait for it); `reply_stopped()` lets an engine that can abandon its
      wait do so at once.
    * `synth` may raise `SentenceFallback`: that chunk and all after it are
      rendered as one block by `synth_rest` (re-raised when it is None).
    * `pad(audio, sr)` is applied to every piece except the reply's last, so
      each sentence boundary gets the same pause (the engine trims each
      render's trailing silence). A `Chunk` with a `gap_s` (the head of a
      clause split) is padded with `pad(audio, sr, gap_s)` instead.
    * Errors: one before anything was heard (chunk 1's render or play) is
      raised, so the caller's whole-text error path runs. One after at least
      one sentence was heard -- a later render or play error, or no render
      within `wait_timeout` of waiting -- ends the reply there and is
      returned as `result.error`: those sentences WERE spoken.
    * `first_rendered` (audio, sr): chunk 1, already rendered by the caller
      (the speed-plan R3 filler pre-render). It is padded like any chunk and
      never rendered again; the worker starts on chunk 2 at once. None (the
      default) renders chunk 1 here, exactly as before.
    * While the worker renders chunk i, `needed_by()` on its thread is when
      chunk i will be played: the end of the chunk playing now plus the
      length of every rendered chunk waiting to play (padding included), and
      `reply_stopped()` is the Event set when this call returns. Chunk 1 is
      rendered with both None.
    """
    res = PipelineResult()
    if not chunks:
        return res
    n = len(chunks)
    # The playback schedule the worker reads for needed_by(): when the chunk
    # handed to play() last will end, and the length of the rendered chunks
    # still waiting in the queue. The caller writes busy_until right before
    # each play; the worker adds each render it queues.
    sched_mu = threading.Lock()
    sched = {"busy_until": 0.0, "queued_s": 0.0}

    def _padded(audio, sr, last_i: int):
        """`audio` padded for the boundary after chunk `last_i` (not the
        reply's last)."""
        if pad is None or last_i + 1 >= n:
            return audio
        gap = getattr(chunks[last_i], "gap_s", None)
        return pad(audio, sr) if gap is None else pad(audio, sr, gap)

    def _render(i: int, need: Optional[float] = None,
                stopped: Optional[threading.Event] = None):
        """(audio, sr, chunks covered) for chunk i, padded unless last."""
        _render_ctx.needed_by = need
        _render_ctx.stop = stopped
        try:
            try:
                audio, sr = synth(chunks[i])
                covered = 1
            except SentenceFallback:
                if synth_rest is None:
                    raise
                res.fell_back = True
                audio, sr = synth_rest(" ".join(chunks[i:]))
                covered = n - i
        finally:
            _render_ctx.needed_by = None
            _render_ctx.stop = None
        return _padded(audio, sr, i + covered - 1), sr, covered

    if first_rendered is None:
        first = _render(0)
    else:
        f_audio, f_sr = first_rendered
        first = (_padded(f_audio, f_sr, 0), f_sr, 1)
    q: "queue.Queue" = queue.Queue()
    # Set when this call returns (the finally below): the worker starts no
    # further render, and a render in flight may abandon its wait.
    stop = threading.Event()
    next_i = first[2]
    # Chunk 1 starts playing right after the worker starts: set its end now,
    # so the worker's first needed_by() already counts it.
    with sched_mu:
        sched["busy_until"] = time.monotonic() + _duration_s(first[0], first[1])

    def _worker() -> None:
        i = next_i
        try:
            while i < n:
                if stop.is_set():
                    return
                with sched_mu:
                    need = (max(time.monotonic(), sched["busy_until"])
                            + sched["queued_s"])
                try:
                    item = _render(i, need, stop)
                except BaseException as e:  # noqa: BLE001 - handed to caller
                    q.put(("err", e))
                    return
                with sched_mu:
                    sched["queued_s"] += _duration_s(item[0], item[1])
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
            dur = _duration_s(audio, sr)
            with sched_mu:
                sched["queued_s"] = max(0.0, sched["queued_s"] - dur)
                sched["busy_until"] = time.monotonic() + dur
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
