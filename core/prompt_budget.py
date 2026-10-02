"""Local prompt budget: keep a local chat prompt inside the model's window.

WHY THIS EXISTS (2026-10-01)
============================
Ollama does not reject a prompt longer than ``num_ctx``. It truncates it and
says so only in its own server log::

    msg="truncating input prompt" limit=8195 prompt=17958 keep=5 new=8195

It keeps the first ``keep`` tokens and the TAIL, so a 17.9k-token prompt
reaches the model as ~8.2k tokens: almost the whole system prompt (identity,
action grammar, safety rules) is gone, and nothing on JARVIS's side ever learns
of it. The live session logs show it twice on 2026-10-01 (the turn-timing line
reads ``prompt_eval_count=8195`` instead of the usual 13-16k), and a 'web'
brain-eval prompt hit the same wall at ~18.5k tokens. Both live cases were a
long turn: twelve to fourteen thousand characters of routed PC sections on top
of a ~51k-character cache-stable system prompt and a full history.

WHAT IT DOES
============
``fit_chat`` estimates the prompt's size and, ONLY when the estimate is over
the budget, trims in a fixed order until it fits:

  1. the oldest history messages, down to the last ``keep_recent`` (the most
     recent exchange: a short follow-up's subject lives there);
  2. the per-turn parts, lowest rank first (``RANK_*`` below; the largest
     part goes first within a rank, so the fewest capabilities are lost);
  3. the rest of the history.

It never touches the system prompt (the cache-stable prefix, which carries the
action grammar and the safety core) or the final message (the user's words, or
a follow-up round's action results). So when those two alone are over the
budget, trimming cannot make the prompt fit and nothing is trimmed: Ollama
keeps the TAIL of a truncated prompt, which is where the history and the
per-turn context sit, so dropping them would only lose more. The prompt is
sent unchanged and ``describe`` says it cannot fit.

A prompt that fits comes back exactly as the caller would have sent it without
the budget: same messages, same bytes. Trimming only runs on an overflowing
turn. Trimming history moves the front of the message list, which costs that
one turn a full prompt evaluation (see _STABLE_LOCAL_PREFIX in the monolith).
That is far cheaper than a reply from a model that never saw its own
instructions.

CALIBRATION (read-only, from the live session logs and the Ollama server log)
=============================================================================
``prompt_eval_count`` is the total prompt size even on a warm prefix (13.6k
evaluated in 641 ms on a cached turn), so it can be compared with character
counts directly:

  * system prompt alone: each session's first idle re-prime (sent before the
    first user turn, so no history) against that session's ``sys_chars`` plus
    the local-mode directive. 10 sessions, 3.83-3.85 characters per token.
  * per-turn context (routed PC section bodies + addenda): a turn's
    ``prompt_eval_count`` minus the re-prime that warmed its prefix, against
    ``turn_ctx_chars`` + the user's words + the context wrapper. 23 turns over
    2,000 characters: 3.69 minimum, 4.06 median, 4.52 maximum.

``CHARS_PER_TOKEN = 3.7`` sits at the bottom of both ranges, so the estimate is
at or above the real count for every material measured: about 3.6 % high on
the system prompt, about right on the densest per-turn context. The budget
reserves the reply's own ``max_tokens``, so the 16384 window allows 15,884
estimated prompt tokens on a voice turn. It is not a flat ~14,000: the system
prompt alone is ~13.4k tokens, and of 397 logged turns (2026-09-29 to 10-01;
median 13,847 tokens, max 16,338 untruncated) 157 were over 14,000. A flat
14k cap would have trimmed history on about 40 % of ordinary turns.

Pure and stdlib-only, so the CI-light tier covers it (tests/test_prompt_budget.py).
"""
from __future__ import annotations

import math
from typing import Callable, List, NamedTuple, Optional, Sequence

# Characters per token for the local brain's tokenizer on JARVIS's prompt
# material. See CALIBRATION above: the bottom of the measured range, so the
# estimate errs high.
CHARS_PER_TOKEN = 3.7

# Chat-template cost of one message (role marker + turn delimiters) and of the
# prompt as a whole (BOS + the generation prompt). Small, and generous.
MESSAGE_OVERHEAD_TOKENS = 6
PROMPT_OVERHEAD_TOKENS = 8

# Never compute a budget below this. A misconfigured num_ctx or max_tokens
# must not make every call trim its whole history.
MIN_BUDGET_TOKENS = 1024

# History messages that survive the first trim pass: the most recent exchange.
KEEP_RECENT_MESSAGES = 2

# Per-turn part ranks. LOWER is dropped FIRST.
RANK_STYLE_HINT = 10   # phrasebook "last used" rotation hint
RANK_REGISTER = 20     # tone / emotion / voice-mood register hints
RANK_MEMORY = 30       # per-turn long-term-memory recall
RANK_MODE = 40         # agent-mode PLAN/EXECUTE directive
RANK_SECTION = 50      # PC_CONTROL section bodies (the turn's action grammar)


class TurnPart(NamedTuple):
    """One piece of per-turn context. The parts' texts, joined in order,
    are exactly the context string the caller would otherwise attach."""
    label: str
    text: str
    rank: int


class Fit(NamedTuple):
    """What fit_chat did. ``messages`` is what to send."""
    messages: list
    before: int            # estimated prompt tokens before trimming
    after: int             # estimated prompt tokens of ``messages``
    budget: int
    dropped_history: int   # history messages dropped (oldest first)
    dropped_parts: tuple   # labels of the dropped per-turn parts, in order
    floor: int = 0         # the system prompt + the final message alone
                           # (0 when the prompt fit and nothing was measured)

    @property
    def trimmed(self) -> bool:
        return bool(self.dropped_history or self.dropped_parts)

    @property
    def fits(self) -> bool:
        return self.after <= self.budget


def estimate_tokens(text) -> int:
    """Estimated tokens in ``text`` (0 for empty or non-text). Never raises."""
    try:
        n = len(text) if isinstance(text, str) else 0
    except Exception:
        return 0
    return int(math.ceil(n / CHARS_PER_TOKEN)) if n else 0


def _content_text(content) -> str:
    """Plain text of a message's content: a string, or the text blocks of a
    list of content blocks. Anything else counts as empty."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def estimate_chat_tokens(system, messages) -> int:
    """Estimated tokens of a whole chat prompt: system + every message +
    template overhead. Never raises."""
    total = PROMPT_OVERHEAD_TOKENS + MESSAGE_OVERHEAD_TOKENS
    total += estimate_tokens(system)
    for m in messages or ():
        total += MESSAGE_OVERHEAD_TOKENS
        if isinstance(m, dict):
            total += estimate_tokens(_content_text(m.get("content")))
    return total


def budget_for(num_ctx, max_tokens) -> int:
    """Estimated prompt tokens allowed in a ``num_ctx`` window when the reply
    may run to ``max_tokens``. The reply's room is reserved because a prompt
    that fills the window leaves the model nothing to answer in."""
    try:
        ctx = int(num_ctx)
    except Exception:
        ctx = 0
    try:
        reply = max(0, int(max_tokens))
    except Exception:
        reply = 0
    return max(MIN_BUDGET_TOKENS, ctx - reply)


def _default_attach(messages: list, turn_ctx: str) -> list:
    """Prepend ``turn_ctx`` to the final message's content (a copy)."""
    if not turn_ctx or not messages:
        return messages
    out = list(messages)
    last = out[-1]
    if isinstance(last, dict) and isinstance(last.get("content"), str):
        out[-1] = dict(last, content=turn_ctx + last["content"])
    return out


def _drop_oldest(head: list) -> int:
    """Drop the oldest history message, then any leading non-user ones (the
    cloud fallback rejects a list that opens with 'assistant'). Returns how
    many went."""
    n = 0
    if head:
        head.pop(0)
        n = 1
    while head and not (isinstance(head[0], dict)
                        and head[0].get("role") == "user"):
        head.pop(0)
        n += 1
    return n


def _pick_victim(kept: List[TurnPart]) -> int:
    """Index of the part to drop next: the lowest rank, the largest text
    within it, the latest on a tie."""
    best = 0
    for i, p in enumerate(kept):
        b = kept[best]
        if (p.rank, -len(p.text)) <= (b.rank, -len(b.text)):
            best = i
    return best


def fit_chat(messages: Sequence, parts: Sequence[TurnPart] = (), *,
             budget: int,
             measure: Callable[[list], int],
             attach: Optional[Callable[[list, str], list]] = None,
             keep_recent: int = KEEP_RECENT_MESSAGES) -> Fit:
    """Fit ``messages`` (+ the per-turn ``parts``) inside ``budget``.

    ``messages`` is the chat history ending with the CURRENT message, without
    the per-turn context. ``attach(messages, turn_ctx)`` attaches the rendered
    parts the way the caller always has (the monolith passes
    _with_turn_context). ``measure(messages)`` returns the estimated tokens of
    the whole prompt those final messages make, so the caller decides how the
    system prompt is shaped and counted.

    The input is never mutated. Within budget, or when even the system
    prompt + the final message alone are over it, the result is exactly
    ``attach(messages, "".join(texts))``. Otherwise the result fits. See the
    module docstring for the trim order."""
    attach = attach or _default_attach
    parts = [p for p in (parts or ()) if p.text]

    def build(head, kept):
        ctx = "".join(p.text for p in kept)
        return attach(list(head) + [current], ctx)

    if not messages:
        out = attach(list(messages or ()), "".join(p.text for p in parts))
        n = measure(out)
        return Fit(out, n, n, budget, 0, (), n)
    current = messages[-1]
    full = attach(list(messages), "".join(p.text for p in parts))
    before = measure(full)
    if before <= budget:
        return Fit(full, before, before, budget, 0, ())

    # Can trimming fit it at all? The system prompt + the final message are
    # never trimmed, so if they alone are over, trimming only throws context
    # away: Ollama keeps the TAIL of a truncated prompt, which is exactly where
    # the history and the per-turn context sit. Send it unchanged and say so.
    floor = measure(attach([current], ""))
    if floor > budget:
        return Fit(full, before, before, budget, 0, (), floor)

    head = list(messages[:-1])
    kept = list(parts)
    dropped_history = 0
    dropped_parts: List[str] = []
    out, now = full, before

    # 1. Oldest history, down to the most recent exchange.
    while now > budget and len(head) > max(0, keep_recent):
        dropped_history += _drop_oldest(head)
        out = build(head, kept)
        now = measure(out)
    # 2. Per-turn parts, lowest priority first.
    while now > budget and kept:
        dropped_parts.append(kept.pop(_pick_victim(kept)).label)
        out = build(head, kept)
        now = measure(out)
    # 3. Whatever history is left. Ends at the floor at worst, which fits.
    while now > budget and head:
        dropped_history += _drop_oldest(head)
        out = build(head, kept)
        now = measure(out)
    return Fit(out, before, now, budget, dropped_history,
               tuple(dropped_parts), floor)


def describe(fit: Fit, where: str = "local", num_ctx=None,
             max_labels: int = 6) -> str:
    """The one-line ``[prompt-budget]`` console note for a trimmed or still
    over-budget fit."""
    head = (f"[prompt-budget] {where}: ~{fit.before:,} tok > budget "
            f"{fit.budget:,}")
    if num_ctx:
        head += f" (num_ctx {num_ctx})"
    did = []
    if fit.dropped_history:
        did.append(f"{fit.dropped_history} oldest history msg(s)")
    if fit.dropped_parts:
        labels = list(fit.dropped_parts[:max_labels])
        more = len(fit.dropped_parts) - len(labels)
        if more > 0:
            labels.append(f"+{more} more")
        did.append(f"{len(fit.dropped_parts)} turn part(s) "
                   f"[{', '.join(labels)}]")
    if not did and not fit.fits:
        return head + (f" - CANNOT FIT: the system prompt + the current "
                       f"message alone are ~{fit.floor:,} tok (never "
                       f"trimmed); sent unchanged")
    body = (" - dropped " + " + ".join(did)) if did else " - nothing dropped"
    return head + body + f" -> ~{fit.after:,} tok"
