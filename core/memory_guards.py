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
