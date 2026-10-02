"""core/llm_client.py — one place for the Anthropic (Claude) API mechanics.

Before this module the exact same call —

    anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
        model=..., max_tokens=..., system=..., messages=...
    ).content[0].text

— was inlined at six sites in the 15k-line bobert_companion.py monolith, each
with its own subtly-different error handling. Centralising it means:

  * the self-upgrade pipeline (and a human) edits the LLM call in ONE small,
    readable file instead of hunting through the monolith,
  * model / timeout / token defaults live in a single place,
  * the streaming entrypoint (stream_text) has a home, so the perceived-latency
    win of speaking partial replies can be wired in later without touching the
    call sites again.

This module imports `anthropic` lazily (inside the functions) and never imports
bobert_companion, so it loads cleanly mid-boot and the dependency stays optional
until a Claude call actually happens. Exceptions are NOT swallowed here — each
call site keeps its own bespoke `except anthropic.BadRequestError ...` handling,
so behaviour is identical to the inlined version; this module only removes the
duplicated construction boilerplate.

Per-model request shaping (2026-10-01, the Sonnet 5.5 / Opus 5.5 upgrade)
------------------------------------------------------------------------
EVERY Claude call in the tree — this module's complete()/stream_text(), the
monolith's direct calls, the orchestrator, the deep-audit daemon and the skills —
goes through ``build_request()`` (usually via ``create_message()``) and reads the
reply with ``response_text()``. tests/test_llm_client_models.py holds an AST
check that no other production file calls ``.messages.create(`` /
``.messages.stream(`` directly. The rules live in ONE place because the current
models changed the request surface:

  * Sonnet 5.5 / Opus 5.5 / Fable 5.1 THINK BY DEFAULT and the thinking tokens
    count toward ``max_tokens`` even though their text is not returned. A
    voice-sized cap (8 … 500) can be eaten entirely by thinking and come back
    with no text at all — so those models get a ``max_tokens`` FLOOR per
    purpose (reply length is governed by the prompt, not the cap; you pay only
    for tokens actually generated).
  * Their effort is set with ``output_config={"effort": …}``. The installed SDK
    (anthropic 0.76.0) has no typed ``output_config`` parameter on
    ``messages.create``/``stream``, so it travels in ``extra_body``.
  * They reject non-default ``temperature`` / ``top_p`` / ``top_k`` (Opus 4.7+
    rejects ANY value). ``build_request`` strips them for every model not
    known to accept them, so a sampling knob can never 400 a call.
  * A reply can START with a ``thinking`` block, and a safety decline arrives as
    HTTP 200 with ``stop_reason == "refusal"`` and possibly no text.
    ``response_text`` reads blocks by TYPE and raises ``CloudReplyError`` for a
    refusal or a reply with no text block, so the caller's existing cloud-
    failure fallback runs instead of JARVIS speaking an empty string.

Unknown / older models (Haiku 4.5, Sonnet 4.x, Opus 4.x, a test id) get NO extra
fields, so an owner who picks an older model in Settings keeps exactly the
request he had. The rules per model are in ``request_options``.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence

# Mirror of bobert_companion._ANTHROPIC_TIMEOUT_S. Kept here so the default
# lives with the client; callers may override per-call.
DEFAULT_TIMEOUT_S: float = 30.0

# `timeout` in the Anthropic SDK is PER ATTEMPT, and the SDK's own default is
# max_retries=2 — i.e. up to THREE attempts plus exponential backoff between
# them. So a "30 second timeout" was really a ~92 s worst case, and these calls
# run on the MAIN VOICE THREAD: JARVIS would go silent and deaf for a minute and
# a half while the SDK quietly retried behind his back. Nobody chose 92 s; it was
# inherited. Pin it. One retry still absorbs the transient 429/529/socket blips
# retries exist for, while the worst case a caller can actually feel becomes
# 2 x timeout + backoff (~62 s at the default) instead of 3 x. 2026-07-14 audit.
DEFAULT_MAX_RETRIES: int = 1


def _client(timeout: float, max_retries: int = DEFAULT_MAX_RETRIES):
    """Construct an Anthropic client. Lazy import so `anthropic` stays an
    optional dependency until a cloud call is actually made."""
    import anthropic  # noqa: WPS433 — intentional lazy import
    return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)


# ── Per-model request shaping ─────────────────────────────────────────────
#
# What a call is FOR decides how hard the model should think and how much
# room the reply needs. Every call site names one of these.
#   voice    — a reply JARVIS speaks (main chat turn, follow-up round, the
#              local-route cloud fallback, the phone channel)
#   quick    — background one-shots (_llm_quick: learning / extraction /
#              proactive lines; news-headline summaries; orchestrator workers)
#   classify — one-label triage (email / notification classifiers)
#   vision   — a question about a screenshot
#   compose  — write a finished text from material already gathered
#              (orchestrator merger, email draft)
#   plan     — decompose a request into sub-tasks (orchestrator planner)
#   deep     — unattended analysis where quality beats latency (deep code
#              audit, overnight idea generation)
#   ping     — a reachability probe; the reply is ignored (caller's cap kept)
PURPOSES = ("voice", "quick", "classify", "vision", "compose", "plan",
            "deep", "ping")

# Effort for the current-generation models. Anthropic's guidance for Sonnet 5.5:
# `low` for chat, content generation, classification, extraction — from
# `medium` up it "thinks briefly before almost every reply, even a greeting",
# and at `low` it skips thinking on most simple requests. Measured (Artificial
# Analysis, 2026-10-01): Sonnet 5.5 is ~1.2 s to the first answer token at
# `low` but ~12.9 s at its default `high` — so EVERY latency-bound purpose
# (anything the owner waits on) is `low`.
#
# The orchestrator PLANNER is latency-bound too: it runs inside a spoken turn
# under a 20 s per-attempt timeout (ORCHESTRATOR_PLANNER_TIMEOUT_S) and only
# emits a short JSON task list over a handful of sub-agents, so it gets `low`
# like the merger. Only `deep` (unattended: the deep code audit, overnight idea
# generation) gets `medium` — Opus 5.5's own default, which in Anthropic's
# testing matches or beats Opus 5 at `high`. Opus 5.5 always thinks (~13 s to
# the first token even at `low`, ~22 s at `medium`), which is why it is only
# ever the model for `deep` jobs.
_EFFORT_BY_PURPOSE: dict[str, str] = {
    "voice": "low",
    "quick": "low",
    "classify": "low",
    "vision": "low",
    "compose": "low",
    "plan": "low",
    "deep": "medium",
    "ping": "low",
}

# max_tokens FLOOR for models that think by default (thinking counts toward
# max_tokens). Voice-class purposes get 2048 (Anthropic's guidance: size the cap
# for thinking + reply); the planner 4096 (a JSON task list); deep work 16000
# (kept under the SDK's ~21k non-streaming ceiling). `ping` keeps the caller's
# cap (1 token — the reply is never read; only the HTTP status matters).
_MAX_TOKENS_FLOOR: dict[str, int] = {
    "voice": 2048,
    "quick": 2048,
    "classify": 2048,
    "vision": 2048,
    "compose": 2048,
    "plan": 4096,
    "deep": 16000,
    "ping": 0,
}

# Current generation: adaptive thinking always/by default on, `output_config.
# effort` supported, sampling params rejected, forced tool_choice rejected.
_EFFORT_MODELS = frozenset({
    "claude-sonnet-5-5", "claude-opus-5-5",
    "claude-fable-5-1", "claude-mythos-5-1",
})
# The previous 5.x generation also thinks by default (omitting `thinking` runs
# adaptive), so a tiny cap returns no text there too — they get the floor but
# keep their default effort ("older model → no extra fields" otherwise).
_THINKS_BY_DEFAULT = _EFFORT_MODELS | frozenset({
    "claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-mythos-5",
})
# Models that reject forced tool_choice ({"type": "any"|"tool"} → 400).
_REJECTS_FORCED_TOOL_CHOICE = _EFFORT_MODELS

# Sampling params pass through ONLY for models known to accept them (the 4.5 /
# 4.6 line, Haiku 4.5, the deprecated Claude 4 models, Claude 3). Everything
# else — Opus 4.7+, Sonnet 5+, Opus 5+, Fable/Mythos, and any id we don't
# recognise — has them stripped: dropping a temperature never fails a call,
# sending one to a model that rejects it always does.
_ACCEPTS_SAMPLING = frozenset({
    "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-1",
    "claude-sonnet-4-0", "claude-opus-4-0", "claude-sonnet-4", "claude-opus-4",
})
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def base_model_id(model: Any) -> str:
    """Normalise a model id for rule lookup: lower-case, no Bedrock
    ``anthropic.`` prefix, no ``-YYYYMMDD`` snapshot suffix. Exact-set lookups
    on the result (never ``startswith``) — ``claude-sonnet-5-5`` starts with
    ``claude-sonnet-5``, and the two need different rules."""
    m = str(model or "").strip().lower()
    if m.startswith("anthropic."):
        m = m[len("anthropic."):]
    return _DATE_SUFFIX_RE.sub("", m)


def accepts_sampling_params(model: Any) -> bool:
    """True only for models known to accept temperature / top_p / top_k."""
    base = base_model_id(model)
    return base in _ACCEPTS_SAMPLING or base.startswith("claude-3")


def rejects_forced_tool_choice(model: Any) -> bool:
    """True for models where tool_choice {"type": "any"|"tool"} is a 400."""
    return base_model_id(model) in _REJECTS_FORCED_TOOL_CHOICE


def request_options(model: Any, purpose: str = "voice",
                    max_tokens: Optional[int] = None) -> dict:
    """The EXTRA ``messages.create`` kwargs this model needs for this purpose.

    Rules (exact model-id match, date suffix ignored):
      * claude-sonnet-5-5 / claude-opus-5-5 / claude-fable-5-1 (+ mythos-5-1):
        ``extra_body={"output_config": {"effort": E}}`` with E = low for
        voice / quick / classify / vision / compose / plan / ping and medium
        for deep; plus the max_tokens floor below.
      * claude-sonnet-5 / claude-opus-5 / claude-fable-5 (think by default):
        the max_tokens floor only — their default effort is left alone.
      * everything else (Haiku 4.5, Sonnet 4.x, Opus 4.x, unknown ids): {}.

    The floor (voice-class 2048, plan 4096, deep 16000, ping none) is returned
    as ``max_tokens`` only when it RAISES the caller's value.
    Raises ValueError for an unknown purpose (a typo must fail loudly in tests,
    not silently pick a tuning)."""
    if purpose not in _EFFORT_BY_PURPOSE:
        raise ValueError(f"unknown llm purpose {purpose!r}; "
                         f"expected one of {PURPOSES}")
    base = base_model_id(model)
    opts: dict[str, Any] = {}
    if base in _THINKS_BY_DEFAULT:
        floor = _MAX_TOKENS_FLOOR[purpose]
        if floor and (max_tokens is None or int(max_tokens) < floor):
            opts["max_tokens"] = floor
    if base in _EFFORT_MODELS:
        opts["extra_body"] = {
            "output_config": {"effort": _EFFORT_BY_PURPOSE[purpose]}}
    return opts


def build_request(*, purpose: str = "voice", **kwargs: Any) -> dict:
    """Full ``messages.create`` / ``messages.stream`` kwargs: the caller's own
    kwargs + ``request_options`` for (model, purpose), with sampling params
    stripped for models that reject them. A caller-supplied ``extra_body`` is
    merged, and a caller-supplied ``output_config.effort`` wins."""
    req: dict[str, Any] = dict(kwargs)
    model = req.get("model")
    opts = request_options(model, purpose, req.get("max_tokens"))
    if "max_tokens" in opts:
        req["max_tokens"] = opts["max_tokens"]
    if "extra_body" in opts:
        extra = dict(req.get("extra_body") or {})
        oc = dict(extra.get("output_config") or {})
        for k, v in opts["extra_body"]["output_config"].items():
            oc.setdefault(k, v)
        extra["output_config"] = oc
        req["extra_body"] = extra
    if not accepts_sampling_params(model):
        for p in _SAMPLING_PARAMS:
            req.pop(p, None)
        if isinstance(req.get("extra_body"), dict):
            eb = {k: v for k, v in req["extra_body"].items()
                  if k not in _SAMPLING_PARAMS}
            req["extra_body"] = eb
    return req


def create_message(client: Any, *, purpose: str = "voice", **kwargs: Any) -> Any:
    """``client.messages.create`` with the per-model request shaping applied.
    Returns the raw Message; read it with ``response_text``."""
    return client.messages.create(**build_request(purpose=purpose, **kwargs))


def stream_message(client: Any, *, purpose: str = "voice", **kwargs: Any) -> Any:
    """``client.messages.stream`` (a context manager) with the per-model
    request shaping applied."""
    return client.messages.stream(**build_request(purpose=purpose, **kwargs))


class CloudReplyError(RuntimeError):
    """The API answered (HTTP 200) but with nothing JARVIS can say. Raised so
    the caller's existing cloud-failure fallback runs (local model, honest
    line) instead of an empty string being spoken."""

    def __init__(self, message: str, *, stop_reason: Any = None,
                 category: Any = None):
        super().__init__(message)
        self.stop_reason = stop_reason
        self.category = category


class CloudRefusalError(CloudReplyError):
    """``stop_reason == "refusal"`` — a safety-classifier decline."""


class CloudEmptyReplyError(CloudReplyError):
    """No text block at all (e.g. thinking used the whole max_tokens)."""


def _refusal_category(msg: Any) -> Any:
    details = getattr(msg, "stop_details", None)
    if isinstance(details, dict):
        return details.get("category")
    return getattr(details, "category", None)


def _check_refusal(msg: Any) -> None:
    stop = getattr(msg, "stop_reason", None)
    if isinstance(stop, str) and stop == "refusal":
        cat = _refusal_category(msg)
        raise CloudRefusalError(
            f"Claude declined the request (stop_reason=refusal"
            f"{', category=' + str(cat) if cat else ''})",
            stop_reason=stop, category=cat)


def _text_blocks(msg: Any) -> list[str]:
    """Text of every TEXT block, in order. A block whose ``type`` is a string
    counts only when it is "text" (so a leading ``thinking`` /
    ``redacted_thinking`` / ``tool_use`` block is skipped); a block without a
    string ``type`` (hand-rolled stand-ins) counts when its ``.text`` is a str."""
    out: list[str] = []
    try:
        blocks = list(getattr(msg, "content", None) or [])
    except Exception:
        return out
    for block in blocks:
        btype = getattr(block, "type", None)
        if isinstance(btype, str) and btype != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            out.append(text)
    return out


def response_text(msg: Any) -> str:
    """The reply text of a Messages response — all text blocks joined, read by
    block TYPE (never ``content[0]``). Raises ``CloudRefusalError`` on
    ``stop_reason == "refusal"`` and ``CloudEmptyReplyError`` when the reply
    carries no text block at all; both are ``CloudReplyError`` (a
    RuntimeError) so ``except Exception`` fallbacks catch them."""
    _check_refusal(msg)
    parts = _text_blocks(msg)
    if not parts:
        raise CloudEmptyReplyError(
            "Claude returned no text "
            f"(stop_reason={getattr(msg, 'stop_reason', None)!r})",
            stop_reason=getattr(msg, "stop_reason", None))
    return "".join(parts)


def complete(
    *,
    model: str,
    messages: Sequence[dict],
    system: Optional[str | list] = None,
    max_tokens: int = 500,
    timeout: float = DEFAULT_TIMEOUT_S,
    purpose: str = "voice",
) -> str:
    """Blocking completion. Returns the reply text (``response_text``).

    ``system`` accepts either a plain string OR a list of Anthropic content
    blocks (the prompt-caching form: a stable block carrying
    ``cache_control={"type": "ephemeral"}`` plus an optional volatile tail —
    see bobert_companion._cached_system_param). Both are forwarded verbatim.
    ``purpose`` picks the per-model shaping (see ``request_options``).

    Raises the underlying anthropic.* exception on failure (the caller decides
    how to degrade), and ``CloudReplyError`` for a refusal / text-less reply."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }
    if system is not None:
        kwargs["system"] = system
    msg = create_message(_client(timeout), purpose=purpose, **kwargs)
    _log_cache_usage(getattr(msg, "usage", None))
    return response_text(msg)


def stream_text(
    *,
    model: str,
    messages: Sequence[dict],
    system: Optional[str | list] = None,
    max_tokens: int = 500,
    timeout: float = DEFAULT_TIMEOUT_S,
    on_delta: Optional[Callable[[str], None]] = None,
    purpose: str = "voice",
) -> str:
    """Streaming completion. Accumulates and returns the full text, invoking
    `on_delta(chunk)` for each text chunk as it arrives.

    This is the seam for the perceived-latency win (speak the first complete,
    action-free sentence as it streams). `on_delta` callbacks must be cheap and
    must never raise — a raising callback is swallowed so a downstream hiccup
    can't abort the stream. Returns identical text to complete(): only TEXT
    deltas are accumulated (thinking never reaches on_delta), and a refusal or
    a stream that produced no text raises ``CloudReplyError``."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }
    if system is not None:
        kwargs["system"] = system
    parts: list[str] = []
    final = None
    with stream_message(_client(timeout), purpose=purpose, **kwargs) as stream:
        for chunk in stream.text_stream:
            parts.append(chunk)
            if on_delta is not None:
                try:
                    on_delta(chunk)
                except Exception:
                    pass
        try:
            final = stream.get_final_message()
            _log_cache_usage(getattr(final, "usage", None))
        except Exception:
            final = None   # telemetry only — never let it taint a good stream
    if final is not None:
        _check_refusal(final)
    if not parts:
        raise CloudEmptyReplyError(
            "Claude stream produced no text "
            f"(stop_reason={getattr(final, 'stop_reason', None)!r})",
            stop_reason=getattr(final, "stop_reason", None))
    return "".join(parts)


# Rolling cache-hit telemetry. One short line per cloud call so the session
# log shows whether the prompt-cache split (_cached_system_param) is actually
# landing: `cache_read` tokens are billed at 10% — a healthy steady state is
# a large constant cache_read with a small volatile `input` remainder.
# Kept module-global (not per-call) so a tail of the log tells the story.
last_usage: dict = {}


def _log_cache_usage(usage: Any) -> None:
    if usage is None:
        return
    try:
        read = getattr(usage, "cache_read_input_tokens", None) or 0
        made = getattr(usage, "cache_creation_input_tokens", None) or 0
        raw = getattr(usage, "input_tokens", None) or 0
        out = getattr(usage, "output_tokens", None) or 0
        last_usage.update(cache_read=read, cache_creation=made,
                          input=raw, output=out)
        print(f"  [llm] tokens in={raw} out={out} "
              f"cache_read={read} cache_write={made}")
    except Exception:
        pass
