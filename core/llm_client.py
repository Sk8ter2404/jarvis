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
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

# Mirror of bobert_companion._ANTHROPIC_TIMEOUT_S. Kept here so the default
# lives with the client; callers may override per-call.
DEFAULT_TIMEOUT_S: float = 30.0


def _client(timeout: float):
    """Construct an Anthropic client. Lazy import so `anthropic` stays an
    optional dependency until a cloud call is actually made."""
    import anthropic  # noqa: WPS433 — intentional lazy import
    return anthropic.Anthropic(timeout=timeout)


def complete(
    *,
    model: str,
    messages: Sequence[dict],
    system: Optional[str] = None,
    max_tokens: int = 500,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Blocking completion. Returns the first text block of the reply.

    Raises the underlying anthropic.* exception on failure (the caller decides
    how to degrade) — identical semantics to the previous inlined call."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }
    if system is not None:
        kwargs["system"] = system
    msg = _client(timeout).messages.create(**kwargs)
    return _first_text(msg)


def stream_text(
    *,
    model: str,
    messages: Sequence[dict],
    system: Optional[str] = None,
    max_tokens: int = 500,
    timeout: float = DEFAULT_TIMEOUT_S,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    """Streaming completion. Accumulates and returns the full text, invoking
    `on_delta(chunk)` for each text chunk as it arrives.

    This is the seam for the perceived-latency win (speak the first complete,
    action-free sentence as it streams). `on_delta` callbacks must be cheap and
    must never raise — a raising callback is swallowed so a downstream hiccup
    can't abort the stream. Returns identical text to complete()."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }
    if system is not None:
        kwargs["system"] = system
    parts: list[str] = []
    with _client(timeout).messages.stream(**kwargs) as stream:
        for chunk in stream.text_stream:
            parts.append(chunk)
            if on_delta is not None:
                try:
                    on_delta(chunk)
                except Exception:
                    pass
    return "".join(parts)


def _first_text(msg: Any) -> str:
    """Extract the first text block from a Messages response, tolerant of
    responses whose leading block is a non-text (e.g. tool_use) block."""
    content = getattr(msg, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    # Fall back to the historical access pattern so a shape we didn't expect
    # surfaces the same way it used to rather than silently returning "".
    return msg.content[0].text
