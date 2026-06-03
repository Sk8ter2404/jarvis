"""mode_router — Conversation mode router (Controlled / Smart / Agent).

JARVIS supports three conversation modes that change how an utterance gets
dispatched:

  CONTROLLED — only run the exact named skill via deterministic intent
               matching (the same rules core.dispatcher uses for the chain
               resolver). No LLM improvisation. Unrecognised utterances
               get a polite refusal. Use for trusted automations ("play
               music") where the LLM's tendency to embroider or wander
               off-task would be a regression.
  SMART      — default. Current Claude tool-calling behavior; the LLM
               decides which [ACTION: ...] tokens to emit. No prompt
               changes.
  AGENT      — autonomous multi-step planning with self-critique. The
               system prompt is extended with an "Agent mode" directive
               that asks the LLM to PLAN → EXECUTE → CRITIQUE → REPORT
               and chain actions until the goal is genuinely met. The
               follow-up loop depth is also boosted (handled by the
               caller via is_in_agent_mode()).

Mode is persisted to data/conversation_mode.json so the chosen mode
survives JARVIS restarts.

Public API:
  current_mode()                                  → str
  set_mode(mode)                                  → str (the new mode)
  is_in_controlled_mode() / is_in_agent_mode()    → bool
  maybe_handle_mode_toggle(text)                  → str | None
      Pre-router. Detect mode-toggle / status utterances and switch.
      Returns the confirmation line for TTS, or None if `text` isn't a
      mode utterance.
  controlled_dispatch(text, actions)              → str | None
      Try to match `text` to one or more registered actions via the
      core.dispatcher intent rules and execute them. Returns the
      action result or a refusal line. Returns None when not in
      controlled mode (caller should fall through to the LLM path).
  system_prompt_addendum()                        → str
      Addendum to inject into the system prompt for the current mode.
      Empty for smart/controlled, planning directive for agent.
  followup_loop_depth(default=5)                  → int
      Returns the recommended follow-up loop depth for the current
      mode (smart/controlled: default; agent: 3x default capped at 15).

This module is self-contained — it does not import bobert_companion or
any skill module, so it loads cleanly mid-boot.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path


# ── modes ────────────────────────────────────────────────────────────────

MODE_CONTROLLED = "controlled"
MODE_SMART      = "smart"
MODE_AGENT      = "agent"

_VALID_MODES = (MODE_CONTROLLED, MODE_SMART, MODE_AGENT)
_DEFAULT_MODE = MODE_SMART


# ── persisted state ─────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_STATE_FILE   = _DATA_DIR / "conversation_mode.json"

_LOCK = threading.Lock()
_state: dict = {"mode": _DEFAULT_MODE}


def _ensure_data_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_state() -> None:
    if not _STATE_FILE.exists():
        return
    try:
        raw = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        m = (raw.get("mode") or "").lower()
        if m in _VALID_MODES:
            _state["mode"] = m
    except Exception:
        # Corrupt state file — fall back to the default rather than
        # crashing the boot path.
        pass


def _save_state() -> None:
    _ensure_data_dir()
    # Prefer the shared atomic writer when available; fall back to a
    # plain write so this module loads even if core/atomic_io.py is
    # missing or partially initialised at import time.
    try:
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(str(_STATE_FILE), _state)
        return
    except Exception:
        pass
    try:
        _STATE_FILE.write_text(json.dumps(_state, indent=2), encoding="utf-8")
    except Exception:
        pass


_load_state()


# ── mode accessors ──────────────────────────────────────────────────────

def current_mode() -> str:
    with _LOCK:
        return _state["mode"]


def set_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in _VALID_MODES:
        raise ValueError(f"unknown mode '{mode}' (expected {_VALID_MODES})")
    with _LOCK:
        _state["mode"] = m
        _save_state()
    return m


def is_in_controlled_mode() -> bool:
    return current_mode() == MODE_CONTROLLED


def is_in_agent_mode() -> bool:
    return current_mode() == MODE_AGENT


# ── toggle pre-router ───────────────────────────────────────────────────
#
# The utterance is normalised (lead-in fillers stripped, trailing punctuation
# trimmed) before pattern matching so "JARVIS, please switch to agent mode."
# and "agent mode" both land on the same handler.

_PUNCT_TAIL = re.compile(r"[.!?,;:]+\s*$")

_LEAD_FILLERS = (
    "could you ", "can you ", "would you ", "please ",
    "jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, ",
    "ok jarvis ", "ok jarvis, ", "okay jarvis ", "okay jarvis, ",
    "go ahead and ", "i need you to ", "i'd like you to ",
)


def _strip_lead_filler(s: str) -> str:
    # Strip STACKED lead-ins ("JARVIS, please switch to agent mode") by looping
    # until stable — a single pass left the second filler in place and broke the
    # toggle match (caught by tests/test_mode_router). Each pass removes at most
    # one filler, then re-checks the shortened head.
    cur = s.strip()
    changed = True
    while changed:
        changed = False
        low = cur.lower()
        for f in _LEAD_FILLERS:
            if low.startswith(f):
                cur = cur[len(f):].lstrip()
                changed = True
                break
    return cur


_SWITCH_VERBS = (
    r"switch\s+(?:to\s+)?|change\s+(?:to\s+)?|"
    r"enter\s+|use\s+|activate\s+|enable\s+|engage\s+|"
    r"go\s+(?:to\s+|into\s+)|put\s+(?:yourself\s+)?(?:in|into)\s+|"
    r"set\s+(?:yourself\s+)?to\s+"
)

_TOGGLE_WITH_MODE = re.compile(
    rf"^(?:{_SWITCH_VERBS})?(controlled|smart|agent)\s+mode$",
    re.IGNORECASE,
)
_TOGGLE_NO_MODE_WORD = re.compile(
    rf"^(?:{_SWITCH_VERBS})(controlled|smart|agent)$",
    re.IGNORECASE,
)
_STATUS_QUERY = re.compile(
    r"^(?:what(?:'s|\s+is)?\s+(?:your\s+|the\s+current\s+)?(?:current\s+)?"
    r"(?:mode|conversation\s+mode))(?:\s+are\s+you\s+(?:in|on))?\??$",
    re.IGNORECASE,
)
_STATUS_QUERY_SHORT = re.compile(
    r"^(?:current\s+mode|mode\s+status|which\s+mode)\??$",
    re.IGNORECASE,
)


def maybe_handle_mode_toggle(text: str) -> str | None:
    """Detect mode toggle / status utterances.

    Returns the confirmation line to speak, or None if the text isn't a
    mode utterance. Safe to call on every turn — non-mode utterances are
    a cheap no-op.
    """
    if not text or not text.strip():
        return None
    s = _strip_lead_filler(text)
    s = _PUNCT_TAIL.sub("", s).strip()
    if not s:
        return None

    # Status queries: "what mode are you in?" / "current mode"
    if _STATUS_QUERY.match(s) or _STATUS_QUERY_SHORT.match(s):
        return f"I am in {current_mode()} mode, sir."

    # Switch: "controlled mode" / "switch to agent mode" / "use smart"
    m = _TOGGLE_WITH_MODE.match(s) or _TOGGLE_NO_MODE_WORD.match(s)
    if m:
        new_mode = m.group(1).lower()
        prior = current_mode()
        if new_mode == prior:
            return f"Already in {new_mode} mode, sir."
        try:
            set_mode(new_mode)
        except Exception as e:
            return f"Could not switch modes, sir ({type(e).__name__})."
        return _confirmation_for(new_mode)

    return None


def _confirmation_for(mode: str) -> str:
    if mode == MODE_CONTROLLED:
        return (
            "Controlled mode engaged, sir. I'll only run exact commands "
            "now — no improvisation."
        )
    if mode == MODE_AGENT:
        return (
            "Agent mode engaged, sir. I'll plan and iterate "
            "autonomously until the task is done."
        )
    return "Smart mode engaged, sir."


# ── controlled mode dispatch ────────────────────────────────────────────

_REFUSAL_LINE = (
    "I'm in controlled mode, sir — that wasn't a recognised command. "
    "Say 'smart mode' to re-enable conversational replies."
)


def controlled_dispatch(text: str, actions: dict) -> str | None:
    """In controlled mode, try to match `text` to a registered action and
    execute it. Returns the action's reply string (or a refusal line for
    unrecognised input). Returns None when NOT in controlled mode, so the
    caller can fall through to the normal LLM path.

    Resolution order:
      1. Multi-step chain (existing core.dispatcher.resolve_and_dispatch)
      2. Single-intent match via core.dispatcher.match_single_intent
      3. Refusal
    """
    if not is_in_controlled_mode():
        return None
    if not text or not text.strip():
        return _REFUSAL_LINE

    # Defer the dispatcher import so a partially-initialised
    # core.dispatcher (broken intent rule, etc.) doesn't take this
    # module down with it.
    try:
        from core.dispatcher import (
            resolve_and_dispatch as _chain_dispatch,
            match_single_intent as _single_match,
        )
    except Exception:
        return _REFUSAL_LINE

    # Try multi-step first (cheap if no chain separators are present —
    # returns None immediately for single-intent utterances).
    try:
        chain_reply = _chain_dispatch(text, actions)
    except Exception:
        chain_reply = None
    if chain_reply is not None:
        return chain_reply

    # Single-intent fallback.
    try:
        step = _single_match(text, actions.keys())
    except Exception:
        step = None
    if step is None:
        return _REFUSAL_LINE

    fn = actions.get(step.action)
    if fn is None:
        # Race: action was registered when we resolved but isn't now.
        return _REFUSAL_LINE
    try:
        rv = fn(step.arg)
    except Exception as e:
        return f"That action failed, sir ({type(e).__name__})."
    if isinstance(rv, str) and rv.strip():
        return rv
    # Action returned nothing meaningful — fall back to the resolver's
    # canned confirmation so the user still gets audible feedback.
    return f"{step.confirmation.capitalize()}, sir."


# ── agent mode system-prompt addendum ──────────────────────────────────

_AGENT_ADDENDUM = (
    "\n\n# AGENT MODE\n"
    "You are operating in autonomous Agent mode. The user has delegated "
    "this task to you and expects you to work through it across multiple "
    "steps without checking in between every action.\n"
    "\n"
    "Approach every turn as PLAN → EXECUTE → CRITIQUE → REPORT:\n"
    "  1. PLAN — in one short line (≤ 1 sentence), outline the steps "
    "you will take. Do not enumerate a long checklist; the user does "
    "not need to read your plan.\n"
    "  2. EXECUTE — emit [ACTION: ...] tokens to make progress. Chain "
    "multiple actions in a single reply when the goal requires it. Do "
    "NOT stop after the first action if the task is not yet complete.\n"
    "  3. CRITIQUE — after each [ACTION:] result comes back via the "
    "follow-up loop, ask yourself 'did that actually achieve the user's "
    "goal?' If not, emit the next [ACTION: ...] toward the goal. Only "
    "declare the task done when the result is genuinely complete.\n"
    "  4. REPORT — when finished, give a one-line in-character summary "
    "of what you did and the outcome.\n"
    "\n"
    "Self-critique rules:\n"
    "  - If an action returns an error or partial success, explicitly "
    "try an alternative — do not repeat the same failing call.\n"
    "  - Do not promise actions in prose without backing them with an "
    "[ACTION: ...] token in the same reply.\n"
    "  - Do not stop mid-chain. If you said you would do three things, "
    "emit tokens for all three.\n"
    "  - Stay in this mode until the user toggles back to smart mode.\n"
)


def system_prompt_addendum() -> str:
    """Return the prompt addendum for the current mode. Empty for
    smart/controlled, planning directive for agent."""
    if is_in_agent_mode():
        return _AGENT_ADDENDUM
    return ""


# ── follow-up loop depth ───────────────────────────────────────────────

def followup_loop_depth(default: int = 5) -> int:
    """Recommended follow-up loop depth for the current mode.

    Smart/controlled return `default` (current behavior). Agent mode
    boosts to 3x default, capped at 15, so an autonomous task has more
    room to plan-execute-critique-repeat before the safety cap kicks in.
    """
    if is_in_agent_mode():
        return min(15, max(default, default * 3))
    return default
