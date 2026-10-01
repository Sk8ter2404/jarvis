"""The cloud gate for skills that call Claude DIRECTLY (2026-10-01).

The chat path asks the monolith before any cloud call: _claude_reachable()
(AI_BACKEND == "claude" AND a key) and the local branch whenever
MODEL_ROUTING['chat'] is 'local'. The skills that build their OWN anthropic
client — email triage and drafts, news summaries, notification triage, the
phone-bridge fallback — checked only for a key in the environment. So on a
local-only install (AI_BACKEND=ollama, chat routed local, a key still in the
User env) a "morning briefing" sent each unread mail's sender, subject and
body preview to Claude Haiku on that key. They ask this one predicate first.

A separate module so a skill can ask without importing the monolith. The
answer still comes from the RUNNING monolith (bobert_companion.
_chat_cloud_allowed), which owns the live AI_BACKEND a switch_llm / set_brain
may have moved since boot.
"""
from __future__ import annotations

import sys


def chat_cloud_allowed() -> bool:
    """May a skill send this data to Claude? The running monolith's
    _chat_cloud_allowed(): AI_BACKEND "claude", a key present, and
    MODEL_ROUTING['chat'] not 'local'. A gate that raises counts as "no"
    (fail closed — the caller's local fallback answers).

    With no monolith loaded (a skill run on its own, a unit test with a
    stand-in module) there is no live setting to consult: this does not veto,
    and the caller's own precondition (its key / AI_BACKEND check) decides,
    exactly as before. In JARVIS the monolith is always loaded — it aliases
    itself into sys.modules before any skill runs."""
    bc = sys.modules.get("bobert_companion")
    gate = getattr(bc, "_chat_cloud_allowed", None) if bc is not None else None
    if not callable(gate):
        return True
    try:
        return bool(gate())
    except Exception:
        return False
