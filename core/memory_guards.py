"""core/memory_guards.py — write-time guards for bobert_memory.json.

bobert_memory.json is dumped verbatim into the system prompt on EVERY
conversation turn, so anything stored there is sent to the cloud LLM each time.
Two pure guards run before a candidate fact/project is written:

  _is_secret_fact(fact)        — drops credentials/secrets (passwords, API keys,
                                 tokens, SSNs, card numbers …). A "User's Deco
                                 password is …" fact leaked a router admin
                                 password to the cloud on every turn until this
                                 was added (2026-05-30 security audit).
  _is_internal_noise_fact(t)   — drops JARVIS-internal artifacts (self-diag
                                 anomaly IDs, exception traces, [regression]/
                                 [overnight] task tags, placeholder "None") that
                                 the local fallback model otherwise learned as
                                 bogus "projects" (2026-05-30 live-watch fix).
  _clamp_fact_len(text)        — truncates an over-long candidate to MAX_FACT_LEN
                                 chars (on a word boundary when one is close) so
                                 a single garbage-long "fact" can't bloat the
                                 cloud system prompt on every turn (2026-06-07).

Extracted from bobert_companion.py so this security-critical logic is unit
tested in isolation. Pure stdlib (`re`); bobert_companion re-exports both
functions for merge_memory.
"""
from __future__ import annotations

import re

# Credentials / secrets that must never reach cloud-bound memory. Genuine
# secrets JARVIS needs (Deco password, Bambu access code) live in their own
# local-only config files and are read directly by the skills that use them.
_SECRET_FACT_PATTERNS = re.compile(
    r"\b("
    r"password|passwd|passphrase|"
    r"api[\s_-]?key|secret|token|"
    r"access[\s_-]?code|"
    r"credential|"
    r"ssn|social security|"
    r"credit[\s_-]?card|cvv|"
    r"routing[\s_-]?number|account[\s_-]?number"
    r")\b",
    re.IGNORECASE,
)


def _is_secret_fact(fact: str) -> bool:
    """True if a candidate memory fact appears to contain a credential or
    other sensitive secret that must never be written to cloud-bound memory."""
    return bool(_SECRET_FACT_PATTERNS.search(fact or ""))


# JARVIS-internal artifacts the local fallback model began learning as bogus
# "projects" (e.g. "Running diagnostics on anomaly-1780175157", a literal
# "None") — never durable USER memory.
_NOISE_FACT_PATTERNS = re.compile(
    r"(anomaly[-\s]?\d|anomaly investigation|unhandled exception|"
    r"exception[\s_-]?trace|exception_burst|vad[\s_-]?stall|"
    r"running diagnostics on|deep-?audit|"
    r"\[(anomaly|regression|self-?heal|self-?diag|deep-?audit|overnight)\]|"
    r"traceback)",
    re.IGNORECASE,
)


def _is_internal_noise_fact(text: str) -> bool:
    """True if a candidate fact/project is a JARVIS-internal artifact (a
    self-diagnostic anomaly, exception trace, internal task tag) rather than
    real user memory. Also rejects empty/placeholder values like 'None'."""
    t = (text or "").strip()
    if t.lower() in {"none", "n/a", "na", "null", "unknown", ""}:
        return True
    return bool(_NOISE_FACT_PATTERNS.search(t))


# Per-fact length cap. Each stored fact/project is concatenated into the system
# prompt sent to the cloud LLM on EVERY turn, so one runaway "fact" (a pasted
# log line, a rambling local-model hallucination) is permanent token bloat that
# degrades the prompt until it's manually pruned. A real durable fact ("User's
# name is Alex", "Building the REPO animatronic") fits comfortably in 300 chars.
MAX_FACT_LEN = 300


def _clamp_fact_len(text: str, cap: int = MAX_FACT_LEN) -> str:
    """Truncate an over-long candidate fact/project to `cap` characters.

    Cuts on a word boundary when a space falls in the back portion of the
    window (so we don't slice a word in half), otherwise hard-cuts at `cap`
    (an unbroken garbage token). A trailing '…' marks that truncation
    happened. Short inputs are returned unchanged (no marker), so this is a
    no-op for normal facts and is safe to apply on both insert and load.
    """
    if not isinstance(text, str):
        return text
    if len(text) <= cap:
        return text
    head = text[:cap]
    cut = head.rfind(" ")
    # Only honour the word boundary if it isn't chopping off most of the text
    # (e.g. a 300-char unbroken token whose only space is at index 4).
    if cut >= int(cap * 0.6):
        head = head[:cut]
    return head.rstrip() + "…"
