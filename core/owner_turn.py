"""What did the owner actually say, for the turn that is running right now?

A tool's answer must match the question that was ACTUALLY asked. The LLM picks
the action, but it does not always pass the detail that decides the answer: on
2026-09-29 "what's the weather going to be like tomorrow" produced a bare
``[ACTION: weather_briefing]`` and JARVIS read out the CURRENT conditions. The
owner's own words were sitting in ``bobert_companion._last_user_text`` the
whole time. Handlers that need the detail (which day, which kind of recall) read
it through this one helper, so there is a single rule for when it is safe to.

THE RULE: only while an owner turn is in progress. ``_last_user_text`` is never
cleared, so outside a turn it holds whatever he said LAST — possibly an hour
ago. A scheduled or proactive call to the same action must not inherit that old
question (a 7 AM weather job would otherwise answer "tomorrow" because of
something said the night before). ``_turn_in_progress`` is set by the main loop
when a transcript is accepted, voice or typed, and cleared at the top of the
next iteration, so it brackets exactly the turn whose actions are running.

Strict shape checks throughout: a test double (``mock.Mock``) is truthy and
subscriptable, and must read as "no utterance", never as text. Never raises.
"""
from __future__ import annotations

import sys


def _companion():
    """The ALREADY-LOADED monolith, or None — never an import (importing it
    runs its boot code). Same rule as core/draft_preview_gate._loaded_companion."""
    return sys.modules.get("bobert_companion")


def current_owner_utterance(bc=None) -> str:
    """The owner's utterance for the turn in progress, or "".

    ``bc`` is the monolith (or a stand-in); defaults to the loaded module.
    Returns "" when there is no monolith, no turn in progress, or the cells do
    not have their real single-element-list shape."""
    try:
        mod = bc if bc is not None else _companion()
        if mod is None:
            return ""
        turn = getattr(mod, "_turn_in_progress", None)
        if not (isinstance(turn, list) and turn and turn[0] is True):
            return ""
        cell = getattr(mod, "_last_user_text", None)
        if isinstance(cell, list) and cell and isinstance(cell[0], str):
            return cell[0].strip()
    except Exception:
        pass
    return ""
