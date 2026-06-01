"""core/stream_speech.py — safe sentence-gating for streamed TTS early-speech.

The perceived-latency win of streaming is: start *speaking* the first complete
sentence of a reply while the rest is still being generated, instead of waiting
for the whole completion (~3-4s of dead silence today).

But JARVIS's speak path is **action-parse-BEFORE-speak** by design:

    reply = get_response_with_animation(text)          # full LLM reply
    spoken_text, results = parse_and_run_actions(reply) # strips [ACTION:] + runs
    _post_dispatch_followups(...)                        # voices spoken_text

Early-speaking naively would risk (a) speaking an ``[ACTION:...]`` token aloud,
or (b) speaking a *preemptive hallucination* ("I've set the timer") before the
action actually ran. This module exists to make early-speech provably safe:

  * It NEVER releases a sentence once a ``[`` has appeared in the stream (the
    instant an action token *starts*, early-speech aborts — the blocking path
    then handles everything not already spoken).
  * It NEVER releases a sentence that matches the preemptive-hallucination
    guard (the caller injects the monolith's exact ``_detect_preemptive_
    hallucination`` so the gate can't drift from the guard).
  * It only releases a sentence once a sentence *terminator followed by more
    text* confirms the boundary — so the final/last sentence is always left to
    the normal quip+speak path, and decimals like ``3.5`` never split.

Crucially, for an action-free reply the set of early-spoken sentences is exactly
a prefix of the final ``spoken_text`` (the strip is a no-op when there's no
``[``, and any sentence the guard would cut is one we already refused to speak).
So ``remainder(spoken_text)`` — normalized ``spoken_text`` with the early-spoken
prefix removed — lets the blocking path voice only what's left, with no
double-speak and no missing words. That parity property is what the tests pin.

This module imports nothing from bobert_companion (loads clean mid-boot) and
does no I/O — the actual ``_speak`` call and the hallucination detector are
injected, so the whole gate is unit-testable without audio or an LLM.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

__all__ = ["norm_ws", "split_complete_sentences", "EarlySpeaker",
           "looks_like_execution_claim"]


# ── whitespace normalisation ────────────────────────────────────────────────
_WS_RE = re.compile(r"\s+")


def norm_ws(s: str) -> str:
    """Collapse runs of whitespace to a single space and strip the ends.

    TTS doesn't care about original spacing, so comparing/subtracting on the
    normalised form is both robust (immune to the whitespace-collapse that
    parse_and_run_actions does) and lossless for speech purposes."""
    return _WS_RE.sub(" ", s or "").strip()


# ── incremental sentence splitting ──────────────────────────────────────────
_TERMINATORS = ".!?"
_CLOSERS = "\"')]"


def split_complete_sentences(buf: str, start: int) -> tuple[List[str], int]:
    """Find *confirmed-complete* sentences in ``buf[start:]``.

    Returns ``(sentences, new_start)``. A sentence boundary is confirmed only
    when a terminator (``. ! ?``), optional repeats and a closing quote/paren,
    is followed by **whitespace** — i.e. there is more text after it. A
    terminator at the very end of the buffer is NOT a confirmed boundary (more
    may still stream, e.g. ``3.`` then ``5``), so that trailing sentence stays
    buffered. A ``.`` between two digits is treated as a decimal, not a
    boundary. ``new_start`` is where the next call should resume.
    """
    sentences: List[str] = []
    n = len(buf)
    last_cut = start
    pos = start
    while pos < n:
        ch = buf[pos]
        if ch in _TERMINATORS:
            # decimal point inside a number (3.5) — not a sentence end
            if (ch == "." and pos > start and pos + 1 < n
                    and buf[pos - 1].isdigit() and buf[pos + 1].isdigit()):
                pos += 1
                continue
            end = pos + 1
            while end < n and buf[end] in _TERMINATORS:
                end += 1
            if end < n and buf[end] in _CLOSERS:
                end += 1
            if end < n and buf[end].isspace():
                seg = buf[last_cut:end].strip()
                if seg:
                    sentences.append(seg)
                while end < n and buf[end].isspace():
                    end += 1
                last_cut = end
                pos = end
                continue
            # terminator with no following char yet → unconfirmed; stop here.
            if end >= n:
                break
        pos += 1
    return sentences, last_cut


# ── default hallucination backstop (caller normally injects the real one) ───
# Compact superset of the monolith's preemptive-execution-claim shapes, used
# only when the caller does NOT inject the authoritative detector. Keeps the
# module safe-by-default standalone; the monolith injects
# `_detect_preemptive_hallucination` so the gate matches the guard exactly.
_EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:i(?:'ve| have)|i'?ll|i'?m going to|let me|going to|now)\b.{0,40}"
    r"\b(?:set|sett|schedul|creat|add|turn|play|sen[dt]|email|messag|remind|"
    r"open|launch|start|stop|disabl|enabl|dim|brighten|lower|rais|mov|pan|"
    r"tilt|rotat|aim|point)",
    re.IGNORECASE,
)


def looks_like_execution_claim(text: str) -> bool:
    """True if ``text`` reads like a claim that an action was/will be performed.
    Backstop heuristic only — the monolith injects the real detector."""
    return bool(_EXECUTION_CLAIM_RE.search(text or ""))


# ── the early speaker ───────────────────────────────────────────────────────
class EarlySpeaker:
    """Accumulates streamed text and voices safe complete sentences early.

    Parameters
    ----------
    speak_fn:
        Called with each sentence that's safe to voice now (no quip layer —
        the final remainder gets the quip via the normal path). Must not raise;
        a raising speak_fn aborts early-speech (fail-safe to the blocking path).
    is_unsafe_fn:
        ``is_unsafe_fn(sentence) -> bool``; True means "do NOT early-speak this;
        abort". The monolith injects ``lambda t: _detect_preemptive_
        hallucination(t) is not None``. Defaults to ``looks_like_execution_claim``.
    enabled:
        When False the speaker is inert (``feed`` only buffers); the caller then
        runs the pure blocking path. Lets the LLM_STREAMING flag gate cheaply.
    """

    def __init__(
        self,
        speak_fn: Callable[[str], None],
        is_unsafe_fn: Optional[Callable[[str], bool]] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._speak = speak_fn
        self._is_unsafe = is_unsafe_fn or looks_like_execution_claim
        self.enabled = enabled
        self._buf: List[str] = []
        self._buf_str = ""
        self._cursor = 0
        self._spoken: List[str] = []
        self.aborted = False

    # -- public state --------------------------------------------------------
    @property
    def spoke_anything(self) -> bool:
        return bool(self._spoken)

    def spoken_concat(self) -> str:
        """Normalised concatenation of everything early-spoken (for subtraction)."""
        return norm_ws(" ".join(self._spoken))

    # -- streaming feed ------------------------------------------------------
    def feed(self, chunk: str) -> None:
        """Consume one streamed text chunk; voice any newly-safe sentences."""
        if not chunk:
            return
        self._buf.append(chunk)
        if not self.enabled or self.aborted:
            return
        self._buf_str += chunk
        # The instant an action token *starts*, stop early-speech entirely.
        # Anything not yet released is left to the blocking action-parse path.
        if "[" in self._buf_str:
            self.aborted = True
            return
        try:
            sentences, new_cursor = split_complete_sentences(
                self._buf_str, self._cursor)
        except Exception:
            self.aborted = True
            return
        for seg in sentences:
            # Redundant safety: a sentence must be free of '[' and must not
            # read as a preemptive claim the guard would cut/refuse.
            if "[" in seg or self._is_unsafe(seg):
                self.aborted = True
                return
            try:
                self._speak(seg)
            except Exception:
                # A speak hiccup must not corrupt the turn — abort early-speech
                # and let the blocking path voice the full reply.
                self.aborted = True
                return
            self._spoken.append(seg)
            self._cursor = new_cursor

    # -- post-stream reconciliation -----------------------------------------
    def remainder(self, final_spoken_text: str) -> str:
        """What the blocking path should still voice, given the final
        post-action ``spoken_text``.

        For an action-free reply the early-spoken sentences are a normalised
        prefix of ``final_spoken_text``, so we return the suffix. If nothing was
        early-spoken we return the text unchanged. If (defensively) the early
        speech is NOT a prefix — which shouldn't happen — we return the full
        text so the user always hears the complete final answer (a rare repeat
        beats a dropped sentence)."""
        if not self._spoken:
            return final_spoken_text
        full = norm_ws(final_spoken_text)
        already = self.spoken_concat()
        if not already:
            return final_spoken_text
        if full == already:
            return ""
        if full.startswith(already):
            return full[len(already):].strip()
        # Defensive: divergence (e.g. guard cut something we didn't model).
        return final_spoken_text
