"""Model catalog — the selectable LLM models with pricing + an estimated cost
PER CONVERSATION, so you can see how fast each one burns through credit and pick
accordingly (cheap+fast vs smart+pricey vs free+local).

Cloud prices are USD per 1M tokens (input / output) — the public Anthropic
list-price *tiers*, approximate and for GUIDANCE ONLY (not billing); see
``PRICING_AS_OF``. Local Ollama models are $0 (you pay in GPU, not credit). The
per-conversation estimate multiplies a typical-conversation token profile
(env-tunable) by the model's prices.

Import-light + CI-safe: stdlib only, no network, total functions.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

#: The month these list-price tiers were recorded. They drift; treat as a guide.
PRICING_AS_OF = "2026-07"


def _conv_tokens() -> Tuple[int, int]:
    """Typical cumulative (input, output) tokens for one JARVIS conversation —
    a few turns plus tool results + memory context. Conservative defaults,
    tunable via ``JARVIS_CONV_INPUT_TOKENS`` / ``JARVIS_CONV_OUTPUT_TOKENS`` so
    the estimate can match real usage. Never raises."""
    def _int(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default
    return _int("JARVIS_CONV_INPUT_TOKENS", 12000), _int("JARVIS_CONV_OUTPUT_TOKENS", 1500)


class Model:
    """One selectable model + its pricing."""

    __slots__ = ("id", "label", "backend", "in_price", "out_price", "tier", "note")

    def __init__(self, id: str, label: str, backend: str, in_price: float,
                 out_price: float, tier: str, note: str = ""):
        self.id = id              # value passed to switch_llm / written to config
        self.label = label        # human label
        self.backend = backend    # "claude" | "ollama"
        self.in_price = in_price  # USD / 1M input tokens (0.0 for local)
        self.out_price = out_price
        self.tier = tier          # short speed/cost descriptor
        self.note = note

    def cost_per_conversation(self, in_tokens: Optional[int] = None,
                              out_tokens: Optional[int] = None) -> float:
        """Estimated USD for one conversation. Uses the env-tuned token profile
        unless explicit token counts are given. Local models are always 0.0."""
        di, do = _conv_tokens()
        it = di if in_tokens is None else in_tokens
        ot = do if out_tokens is None else out_tokens
        return (it / 1_000_000.0) * self.in_price + (ot / 1_000_000.0) * self.out_price


# ── Local (Ollama, $0) entries — DERIVED, not hard-coded ────────────────────
# 2026-07-21 audit: the catalog's local rows were a stale copy of the local-
# model identity (still qwen/llama while the shipped brain moved to gemma4),
# so 'show LLM stats' couldn't price the default brain. The primary entry now
# comes from core.config.LOCAL_LLM_MODEL (guarded import; core.config is
# stdlib-constants so this module stays import-light/CI-safe and never
# raises). The static failover tags mirror the resolver chain
# (bobert_companion._LOCAL_LLM_PREFERENCE) — mirrored as literals rather than
# imported, because importing the monolith here would break the CI-safe
# contract. by_id()'s prefix match makes the un-quantised ids below cover the
# quantised installed tags (e.g. qwen2.5:14b-instruct → ...-q5_K_M).
_LOCAL_FAILOVER_TAGS = (
    ("gemma4:12b", "Gemma4 12B (local)",
     "lower-VRAM multimodal fallback, also $0"),
    ("qwen2.5:14b-instruct", "Qwen 2.5 14B (local)",
     "runs on your own GPU, $0 per conversation"),
    ("qwen3:14b", "Qwen 3 14B (local)",
     "dense 14B failover, also $0"),
    ("llama3.1:8b", "Llama 3.1 8B (local)",
     "smaller local model, also $0"),
)


def _local_models() -> List[Model]:
    """The local catalog entries: the configured default brain (and the
    shipped default, when the owner overrode it) first — read from core.config
    at import — then the resolver chain's static failover tags, de-duplicated
    in order. All cost $0 — you pay in GPU, not credit."""
    heads: List[Tuple[str, str]] = []
    try:
        from core import config as _cfg
        primary = str(getattr(_cfg, "LOCAL_LLM_MODEL", "") or "").strip()
        shipped = str(getattr(_cfg, "_SHIPPED_LOCAL_LLM_MODEL", "") or "").strip()
        if primary:
            heads.append((primary, f"{primary} (local, default)"))
        if shipped and shipped != primary:
            heads.append((shipped, f"{shipped} (local)"))
    except Exception:  # pragma: no cover - config always importable in practice
        heads = []
    models: List[Model] = []
    for tag, label in heads:
        covered = any(tag == t or tag.startswith(t)
                      for t, _, _ in _LOCAL_FAILOVER_TAGS)
        if not covered and all(m.id != tag for m in models):
            models.append(Model(tag, label, "ollama", 0.0, 0.0, "local / free",
                                "the local brain — your own GPU, $0 per "
                                "conversation"))
    for tag, label, note in _LOCAL_FAILOVER_TAGS:
        models.append(Model(tag, label, "ollama", 0.0, 0.0, "local / free", note))
    return models


# Ordered cheapest -> priciest so the list reads as a "how fast it burns" dial.
CATALOG: List[Model] = _local_models() + [
    Model("claude-haiku-4-5", "Claude Haiku", "claude", 1.0, 5.0,
          "fastest / cheapest cloud", "snappy + inexpensive; great for everyday"),
    Model("claude-sonnet-4-6", "Claude Sonnet 4.6", "claude", 3.0, 15.0,
          "balanced (previous gen)", "strong reasoning at a moderate cost"),
    Model("claude-sonnet-5", "Claude Sonnet 5", "claude", 3.0, 15.0,
          "balanced (default)", "near-Opus smarts at the same Sonnet price"),
    Model("claude-opus-4-6", "Claude Opus", "claude", 5.0, 25.0,
          "smart / pricier", "previous-gen Opus"),
    Model("claude-opus-4-8", "Claude Opus 4.8", "claude", 5.0, 25.0,
          "smartest / priciest", "most capable; ~1.7x Sonnet's burn rate"),
]


def catalog() -> List[Model]:
    """A copy of the model list (cheapest first)."""
    return list(CATALOG)


def by_id(model_id: str) -> Optional[Model]:
    """The catalog entry whose id matches ``model_id`` (exact, or by prefix for
    Ollama tags like ``qwen2.5:14b-instruct-q5_K_M``), or None."""
    if not model_id:
        return None
    for m in CATALOG:
        if m.id == model_id or model_id.startswith(m.id):
            return m
    return None


def _fmt_usd(x: float) -> str:
    if x <= 0:
        return "$0 (local)"
    if x < 0.01:
        return f"~${x:.3f}/conv"
    return f"~${x:.2f}/conv"


def format_catalog() -> str:
    """A voice/menu-friendly listing: each model + tier + est. cost/conversation,
    cheapest first."""
    lines = ["Model options, cheapest first (estimated cost per conversation):"]
    for m in CATALOG:
        lines.append(f"  - {m.label} [{m.tier}] {_fmt_usd(m.cost_per_conversation())}"
                     f" - {m.note}")
    lines.append(f"(Cloud rates are approximate Anthropic list prices as of "
                 f"{PRICING_AS_OF}; tune the estimate with JARVIS_CONV_INPUT_TOKENS"
                 f" / JARVIS_CONV_OUTPUT_TOKENS.)")
    return "\n".join(lines)
