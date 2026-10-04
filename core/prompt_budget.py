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

  1. the per-turn parts ranked below the section grammar (phrase rotation,
     tone, a section only the history routed, long-term memory, agent mode),
     lowest rank first - they sit at the END of the prompt, so dropping them
     breaks no cached prefix;
  2. the oldest history messages, down to the last ``keep_recent`` (the most
     recent exchange: a short follow-up's subject lives there);
  3. the remaining parts - the routed PC sections, largest first within a
     rank, so the fewest capabilities are lost;
  4. the rest of the history.

Review 2026-10-02: the order used to start with the history, which moves the
front of the message list and costs the turn a full prompt re-evaluation
(~2.5 s, see _STABLE_LOCAL_PREFIX in the monolith) while the cheap tail
parts stayed.

It never touches the system prompt (the cache-stable prefix, which carries the
action grammar and the safety core) or the final message (the user's words, or
a follow-up round's action results). With ``pin_last_user`` (the follow-up
round) the owner's last turn and the chain after it are never dropped either:
there the final message is the machine-made results, and the owner's request
is what the round exists to finish.

When those never-trimmed messages alone are over the budget the prompt cannot
fit, and EVERYTHING else is dropped anyway (review 2026-10-02 - this used to
send the prompt unchanged). Ollama keeps the first ``numKeep`` tokens and cuts
the next (length - num_ctx), so every history or part token left in costs one
token from the START of the system prompt: the identity and the safety rules.
``describe`` says it cannot fit; a follow-up caller can clip its results
(``clip_middle``) and try again.

A prompt that fits comes back exactly as the caller would have sent it without
the budget: same messages, same bytes. Trimming only runs on an overflowing
turn. That is far cheaper than a reply from a model that never saw its own
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
reserves room for the reply: REPLY_RESERVE_TOKENS (200), or less when the
call's ``max_tokens`` is smaller - voice replies run ~15-60 tokens, and
reserving the full 500 trimmed turns of 15.2k-16.4k real tokens that Ollama
would never have cut (review 2026-10-02). The 16384 window allows 16,184
estimated prompt tokens. It is not a flat ~14,000: the system
prompt alone is ~13.4k tokens, and of 397 logged turns (2026-09-29 to 10-01;
median 13,847 tokens, max 16,338 untruncated) 157 were over 14,000. A flat
14k cap would have trimmed history on about 40 % of ordinary turns.

TOKEN-AWARE: EXACT COUNTS FOR WHAT WAS ALREADY SENT (2026-10-04)
================================================================
The character estimate is ~3.6 % high on the system prompt (~490 tokens of a
13.6k prompt) and LOW on dense text (the topics list runs 2.5 characters per
token), so it neither guarantees the fit nor uses the window well. Ollama
reports the exact size of every prompt it evaluated (``prompt_eval_count`` is
the whole prompt even when most of it came from the cache), and JARVIS
re-sends the same prefix over and over: the idle re-prime posts exactly the
next turn minus its user message, and its stage-A posts the system prompt's
stable head alone. ``ExactCounts`` (``EXACT``) remembers those sizes, keyed
by model, system prompt and message list, and ``measure_chat_tokens`` counts
the longest known prefix exactly and estimates only what follows it. A count
far from the estimate of the same prompt (``EXACT_SANITY``: a truncated
prompt, or a server that reports only the uncached part) is never stored.
``budget_for`` keeps ``SAFETY_MARGIN_TOKENS`` free on top of the reply's
room, so system prompt + turn context + reply stay inside the real window
with a margin. The system prompt is still never trimmed.

WHEN THE WINDOW IS SMALLER THAN num_ctx (review 2026-10-02)
==========================================================
The 10-01 incident read ``limit=8195`` - half the 16k num_ctx the budget
assumes (a runner loaded with a different context, or parallel slots). No
estimate can see that, so the monolith compares each reply's
``prompt_eval_count`` with the estimate of what it sent (``looks_truncated``):
a count far under the estimate is a truncation. It logs a loud
``[prompt-budget] TRUNCATED`` line and ``ObservedWindow`` budgets the next
prompts to the observed limit for OBSERVED_LIMIT_TTL_S.

Pure and stdlib-only, so the CI-light tier covers it (tests/test_prompt_budget.py).
"""
from __future__ import annotations

import math
import threading
import time
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

# The reply's room in the window (see CALIBRATION): at most this, or the
# call's max_tokens when that is smaller.
REPLY_RESERVE_TOKENS = 200

# Kept free on top of the reply's room (2026-10-04): the estimated part of a
# prompt (what follows its exactly-known prefix: the new user message, the
# per-turn context) may run a little denser than CHARS_PER_TOKEN.
SAFETY_MARGIN_TOKENS = 128

# An exact count is stored only when it is within this band of the
# character estimate of the same prompt (see TOKEN-AWARE above). Measured
# real/estimate on whole local prompts: 0.93-0.98.
EXACT_SANITY = (0.80, 1.20)

# Per-turn part ranks. LOWER is dropped FIRST. Every rank below RANK_SECTION
# goes before any history does.
RANK_STYLE_HINT = 10   # phrasebook "last used" rotation hint
RANK_REGISTER = 20     # tone / emotion / voice-mood register hints
RANK_INHERITED = 25    # a section only the HISTORY routed (a short follow-up's
                       # inheritance, prompt_router.inherited_turn_sections):
                       # below the turn's own memory recall (review 2026-10-02)
RANK_MEMORY = 30       # per-turn long-term-memory recall
RANK_MODE = 40         # agent-mode PLAN/EXECUTE directive
RANK_SECTION = 50      # PC_CONTROL section bodies (the turn's action grammar)

# A follow-up round whose results alone overflow clips each result to these
# sizes in turn (head and tail kept, clip_middle) before giving up.
RESULT_CLIP_STEPS = (4000, 1500, 600, 250)

# Truncation detection (looks_truncated / ObservedWindow).
TRUNCATION_RATIO = 0.75         # evaluated < 75 % of the estimate = cut
TRUNCATION_MIN_ESTIMATE = 2048  # smaller prompts are never judged
OBSERVED_LIMIT_TTL_S = 900.0    # how long an observed limit budgets prompts


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
    floor: int = 0         # the system prompt + the never-trimmed messages
                           # alone (the final one, plus the pinned owner turn
                           # and chain with pin_last_user); 0 when the prompt
                           # fit and nothing was measured

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
    """Prompt tokens allowed in a ``num_ctx`` window when the reply may run
    to ``max_tokens``. Room for the reply is reserved - a prompt that fills
    the window leaves the model nothing to answer in - but only
    REPLY_RESERVE_TOKENS of it: a voice reply is far shorter than its cap.
    SAFETY_MARGIN_TOKENS more stay free for the estimated part of the
    prompt (2026-10-04)."""
    try:
        ctx = int(num_ctx)
    except Exception:
        ctx = 0
    try:
        reply = min(max(0, int(max_tokens)), REPLY_RESERVE_TOKENS)
    except Exception:
        reply = 0
    return max(MIN_BUDGET_TOKENS, ctx - reply - SAFETY_MARGIN_TOKENS)


def _sha(text) -> str:
    import hashlib
    return hashlib.sha1(str(text).encode("utf-8", "surrogatepass")).hexdigest()


def _message_key(m) -> str:
    """One message as a hash of its role and text (images are not counted:
    a prompt with images is never stored, see ExactCounts.note)."""
    if not isinstance(m, dict):
        return _sha(repr(m))
    return _sha(f"{m.get('role', '')}\x00{_content_text(m.get('content'))}")


def _has_images(messages) -> bool:
    for m in messages or ():
        if isinstance(m, dict) and m.get("images"):
            return True
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, (list, tuple)) and any(
                isinstance(b, dict) and b.get("type") not in (None, "text")
                for b in c):
            return True
    return False


class ExactCounts:
    """Exact prompt sizes Ollama reported for prompts JARVIS sent.

    ``note(model, system, messages, count)`` after a reply: the whole prompt
    of (system, messages) was ``count`` tokens. ``note_head(model, head,
    count)`` after a system-only prompt whose system text is ``head`` (the
    re-prime's stage-A): any later system prompt that STARTS with ``head``
    has that many tokens up to the end of it. ``lookup`` / ``head_for`` find
    the longest known prefix. Bounded (MAX_ENTRIES / MAX_HEADS, newest
    kept); thread-safe; never raises."""

    MAX_ENTRIES = 8
    MAX_HEADS = 4

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = []   # [(model, sys_hash, (msg_hash, ...), count)]
        self._heads = []     # [(model, head_len, head_hash, count)]

    @staticmethod
    def _sane(estimated, count) -> bool:
        try:
            est = float(estimated)
            got = float(count)
        except Exception:
            return False
        lo, hi = EXACT_SANITY
        return est > 0 and got > 0 and lo * est <= got <= hi * est

    def note(self, model, system, messages, count) -> bool:
        try:
            messages = list(messages or ())
            if _has_images(messages):
                return False
            if not self._sane(estimate_chat_tokens(system, messages), count):
                return False
            key = (str(model or ""), _sha(system),
                   tuple(_message_key(m) for m in messages))
            with self._lock:
                self._entries = [e for e in self._entries if e[:3] != key]
                self._entries.append(key + (int(count),))
                del self._entries[:-self.MAX_ENTRIES]
            return True
        except Exception:
            return False

    def note_head(self, model, head, count) -> bool:
        try:
            if not isinstance(head, str) or not head:
                return False
            if not self._sane(estimate_chat_tokens(head, []), count):
                return False
            entry = (str(model or ""), len(head), _sha(head), int(count))
            with self._lock:
                self._heads = [h for h in self._heads if h[:3] != entry[:3]]
                self._heads.append(entry)
                del self._heads[:-self.MAX_HEADS]
            return True
        except Exception:
            return False

    def lookup(self, model, system, messages):
        """(count, k): the longest stored prompt with this model and system
        whose messages are the first k of ``messages``; None when none is."""
        try:
            sh = _sha(system)
            keys = tuple(_message_key(m) for m in (messages or ()))
            best = None
            with self._lock:
                entries = list(self._entries)
            for m, s, msgs, count in entries:
                if (m != str(model or "") or s != sh or len(msgs) > len(keys)
                        or keys[:len(msgs)] != msgs):
                    continue
                if best is None or len(msgs) > best[1]:
                    best = (count, len(msgs))
            return best
        except Exception:
            return None

    def head_for(self, model, system):
        """(head_len, count) of the longest stored head ``system`` starts
        with (this model); None when none."""
        try:
            if not isinstance(system, str):
                return None
            with self._lock:
                heads = list(self._heads)
            best = None
            for m, n, h, count in heads:
                if m != str(model or "") or n > len(system):
                    continue
                if (best is None or n > best[0]) and _sha(system[:n]) == h:
                    best = (n, count)
            return best
        except Exception:
            return None

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._heads = []


# The process-wide record the monolith's local calls feed and read.
EXACT = ExactCounts()


def measure_chat_tokens(system, messages, model="", exact=None) -> int:
    """Tokens of a whole chat prompt, exact where Ollama already reported a
    prefix of it (``EXACT``), estimated (estimate_chat_tokens' rules) for the
    rest. A known prefix's count already includes the chat template's
    opening and closing tokens, so each further message adds its own
    MESSAGE_OVERHEAD_TOKENS and text; a known head (a prefix of the system
    text) adds the estimate of the system text after it. Never raises."""
    ex = EXACT if exact is None else exact
    msgs = list(messages or ())
    try:
        hit = ex.lookup(model, system, msgs)
        if hit is not None:
            count, k = hit
            total = int(count)
            for m in msgs[k:]:
                total += MESSAGE_OVERHEAD_TOKENS
                if isinstance(m, dict):
                    total += estimate_tokens(_content_text(m.get("content")))
            return total
        head = ex.head_for(model, system)
        if head is not None:
            n, count = head
            total = int(count) + estimate_tokens(system[n:])
            for m in msgs:
                total += MESSAGE_OVERHEAD_TOKENS
                if isinstance(m, dict):
                    total += estimate_tokens(_content_text(m.get("content")))
            return total
    except Exception:
        pass
    return estimate_chat_tokens(system, msgs)


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


def _pick_victim(kept: List[TurnPart], below: Optional[int] = None) -> int:
    """Index of the part to drop next: the lowest rank, the largest text
    within it, the latest on a tie. With ``below``, only a part ranked under
    it qualifies (-1 when none does)."""
    best = -1
    for i, p in enumerate(kept):
        if below is not None and p.rank >= below:
            continue
        b = kept[best] if best >= 0 else None
        if b is None or (p.rank, -len(p.text)) <= (b.rank, -len(b.text)):
            best = i
    return best


def _pin_index(head: list) -> int:
    """Index of the last user message in ``head`` (len(head) when none)."""
    for i in range(len(head) - 1, -1, -1):
        m = head[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return len(head)


def fit_chat(messages: Sequence, parts: Sequence[TurnPart] = (), *,
             budget: int,
             measure: Callable[[list], int],
             attach: Optional[Callable[[list, str], list]] = None,
             keep_recent: int = KEEP_RECENT_MESSAGES,
             pin_last_user: bool = False) -> Fit:
    """Fit ``messages`` (+ the per-turn ``parts``) inside ``budget``.

    ``messages`` is the chat history ending with the CURRENT message, without
    the per-turn context. ``attach(messages, turn_ctx)`` attaches the rendered
    parts the way the caller always has (the monolith passes
    _with_turn_context). ``measure(messages)`` returns the estimated tokens of
    the whole prompt those final messages make, so the caller decides how the
    system prompt is shaped and counted. ``pin_last_user``: the last user
    message before the current one, and everything after it, is never
    dropped (a follow-up round: the owner's request and the chain so far).

    The input is never mutated. Within budget the result is exactly
    ``attach(messages, "".join(texts))``. Otherwise trimming runs in the
    module docstring's order; when the never-trimmed messages alone are over
    the budget everything else goes and ``fits`` is False."""
    attach = attach or _default_attach
    parts = [p for p in (parts or ()) if p.text]

    if not messages:
        out = attach(list(messages or ()), "".join(p.text for p in parts))
        n = measure(out)
        return Fit(out, n, n, budget, 0, (), n)
    current = messages[-1]
    full = attach(list(messages), "".join(p.text for p in parts))
    before = measure(full)
    if before <= budget:
        return Fit(full, before, before, budget, 0, ())

    history = list(messages[:-1])
    at = _pin_index(history) if pin_last_user else len(history)
    head, pinned = history[:at], history[at:]
    floor = measure(attach(pinned + [current], ""))

    kept = list(parts)
    dropped_history = 0
    dropped_parts: List[str] = []
    out, now = full, before

    def build():
        ctx = "".join(p.text for p in kept)
        return attach(head + pinned + [current], ctx)

    # 1. Parts ranked below the section grammar: the cheap tail.
    while now > budget:
        i = _pick_victim(kept, below=RANK_SECTION)
        if i < 0:
            break
        dropped_parts.append(kept.pop(i).label)
        out = build()
        now = measure(out)
    # 2. Oldest history, down to the most recent exchange.
    while now > budget and head and len(head) + len(pinned) > max(0, keep_recent):
        dropped_history += _drop_oldest(head)
        out = build()
        now = measure(out)
    # 3. The section parts, largest first.
    while now > budget and kept:
        dropped_parts.append(kept.pop(_pick_victim(kept)).label)
        out = build()
        now = measure(out)
    # 4. Whatever history is left. Ends at the floor at worst: within budget,
    #    or - when even the floor is over - as little as can be sent.
    while now > budget and head:
        dropped_history += _drop_oldest(head)
        out = build()
        now = measure(out)
    return Fit(out, before, now, budget, dropped_history,
               tuple(dropped_parts), floor)


def clip_middle(text, max_chars: int) -> str:
    """``text`` cut to about ``max_chars``: its head and tail kept, the middle
    replaced by a one-line note (an action result's verdict tends to sit at
    either end). Text that fits comes back unchanged. Never raises."""
    if not isinstance(text, str):
        return ""
    try:
        cap = max(40, int(max_chars))
    except Exception:
        cap = 40
    if len(text) <= cap:
        return text
    head = (cap * 2) // 3
    tail = cap - head
    cut = len(text) - head - tail
    return (text[:head].rstrip() + f"\n[... {cut:,} characters cut ...]\n"
            + text[-tail:].lstrip())


def looks_truncated(estimated, prompt_eval_count) -> bool:
    """True when Ollama evaluated far fewer prompt tokens than the estimate of
    what was sent: under TRUNCATION_RATIO of it, on a prompt of at least
    TRUNCATION_MIN_ESTIMATE. The estimate errs HIGH by a few percent (see
    CALIBRATION; 4.52 chars/token, the loosest material measured, is ~18 %
    high), so an honest count never gets near the ratio. Never raises."""
    try:
        est = int(estimated)
        got = int(prompt_eval_count)
    except Exception:
        return False
    return (est >= TRUNCATION_MIN_ESTIMATE and 0 < got
            and got < est * TRUNCATION_RATIO)


class ObservedWindow:
    """The window Ollama ACTUALLY gave the last truncated prompt.

    ``note(estimated, prompt_eval_count)`` after each local reply: a
    truncation (looks_truncated) records the evaluated count as the limit and
    returns True; a later prompt evaluated whole at more than that limit
    clears it (the runner was reloaded). ``effective(num_ctx)`` is the window
    to budget for: the observed limit while it is fresh
    (OBSERVED_LIMIT_TTL_S) and smaller, else ``num_ctx``. Thread-safe; never
    raises."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.limit = 0
        self._at = 0.0

    def note(self, estimated, prompt_eval_count, num_ctx=None) -> bool:
        """``num_ctx``: the window the prompt was sent for. A prompt whose
        estimate was already over it was cut because it was too big, not
        because the runner's window is small (2026-10-04: the proactive
        remark's ~18.4k prompt taught "the window is 8,195" and every local
        prompt for 15 minutes was budgeted to that), so it teaches nothing."""
        try:
            if num_ctx is not None:
                try:
                    if int(estimated) > int(num_ctx):
                        return False
                except Exception:
                    pass
            if looks_truncated(estimated, prompt_eval_count):
                with self._lock:
                    self.limit = int(prompt_eval_count)
                    self._at = self._clock()
                return True
            got = int(prompt_eval_count or 0)
            with self._lock:
                if self.limit and got > self.limit:
                    self.limit = 0
            return False
        except Exception:
            return False

    def effective(self, num_ctx) -> int:
        try:
            ctx = int(num_ctx)
        except Exception:
            return num_ctx
        try:
            with self._lock:
                limit, at = self.limit, self._at
            if limit and self._clock() - at <= OBSERVED_LIMIT_TTL_S:
                return min(ctx, limit)
        except Exception:
            pass
        return ctx

    def clear(self) -> None:
        with self._lock:
            self.limit = 0
            self._at = 0.0


# The process-wide observation the monolith's local calls feed and read.
OBSERVED_WINDOW = ObservedWindow()


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
    cannot = (f" - CANNOT FIT: the system prompt + the never-trimmed "
              f"messages alone are ~{fit.floor:,} tok")
    if not fit.fits:
        if not did:
            return head + cannot + "; sent as is"
        return (head + " - dropped " + " + ".join(did)
                + f" -> ~{fit.after:,} tok" + cannot
                + "; Ollama will cut the start")
    body = (" - dropped " + " + ".join(did)) if did else " - nothing dropped"
    return head + body + f" -> ~{fit.after:,} tok"
