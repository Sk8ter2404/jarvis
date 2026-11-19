"""Carry the recent conversation across a quick restart (2026-11-17).

WHY
===
Every restart started the conversation from zero, even one seconds after the
owner was talking: on 2026-11-17 three restarts between 16:59 and 17:05
(two through the tray's hardened teardown, i.e. _act_restart) each logged
"[session_resume] no greeting: restart 0 min after the last session" - the
same session as far as the greeting goes - while conversation_history was
empty. The blue/green upgrade swap already carried a conversation tail across
(data/handoff.json, only with --resume-handoff); an ordinary restart did not.

WHAT
====
_act_restart writes the tail through the SAME handoff file
(blue_green_manager.write_handoff_state) with ``kind: "restart"``; the next
prod process takes it at boot (consume_handoff_state(kind="restart")) when it
was not started by the blue/green swap, and only when:

  * the owner's last turn is recent: at most CONVERSATION_RESUME_MAX_AGE_S
    (core/config.py, default 900 s = SESSION_RESUME_QUIET_S, the window the
    welcome-back greeting already treats as "the same session"), and the
    handoff file itself is under its HANDOFF_STATE_TTL_SECONDS (600 s);
  * it is bounded: the newest CONVERSATION_RESUME_MAX_MESSAGES messages
    (default 10), each at most MAX_MESSAGE_CHARS, all together at most
    MAX_TOTAL_CHARS;
  * no message that reads like a credential (core.memory_guards.
    _is_secret_fact) travels - nor the replies to such a turn, which may
    echo it - checked when written AND when read back.

The tail always starts with an owner turn and ends with a reply (a trailing
owner turn is the restart ask itself, still unanswered). Pure and stdlib-only;
the monolith owns the I/O (_save_conversation_for_restart /
_resume_conversation_after_restart).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

KIND = "restart"

DEFAULT_MAX_AGE_S = 900.0
DEFAULT_MAX_MESSAGES = 10
# One message's text at most (head kept); a read-out email or web page must
# not ride the next session whole.
MAX_MESSAGE_CHARS = 2000
# The whole tail at most; the oldest exchanges go first.
MAX_TOTAL_CHARS = 8000
# A last-turn stamp this far in the FUTURE is a clock jump, not a turn.
CLOCK_SKEW_S = 120.0

_ROLES = ("user", "assistant")


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + " [...]"


def _secret(is_secret: Optional[Callable[[str], bool]], text: str) -> bool:
    if is_secret is None:
        return False
    try:
        return bool(is_secret(text))
    except Exception:
        return True          # cannot tell: it does not travel


def tail_for_restart(history, max_messages: int = DEFAULT_MAX_MESSAGES,
                     is_secret: Optional[Callable[[str], bool]] = None,
                     max_chars: int = MAX_MESSAGE_CHARS,
                     max_total: int = MAX_TOTAL_CHARS) -> List[dict]:
    """The conversation tail that may cross a restart: plain
    ``{"role", "content"}`` copies of the newest user / assistant messages
    of ``history``, secrets out (a secret owner turn takes its replies with
    it), each clipped to ``max_chars``, the newest ``max_messages`` within
    ``max_total`` characters, starting with an owner turn and ending with a
    reply. [] when nothing qualifies. Never raises."""
    try:
        cap = max(0, int(max_messages))
    except Exception:
        cap = DEFAULT_MAX_MESSAGES
    if not cap:
        return []
    out: List[dict] = []
    try:
        skip_replies = False
        for m in list(history or ()):
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role not in _ROLES or not isinstance(content, str):
                continue
            if not content.strip():
                continue
            if role == "user":
                skip_replies = _secret(is_secret, content)
                if skip_replies:
                    continue
            elif skip_replies or _secret(is_secret, content):
                continue
            out.append({"role": role, "content": _clip(content, max_chars)})
        # The unanswered owner turn at the end (the restart ask) goes first,
        # so it never costs the tail one of its messages.
        while out and out[-1]["role"] != "assistant":
            out.pop()
        out = out[-cap:]
        while out and sum(len(m["content"]) for m in out) > max_total:
            out.pop(0)
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out
    except Exception:
        return []


def build_payload(tail: List[dict], last_turn_at: float, now: float,
                  opening_turns=None, opening_ts=None,
                  opening_lost: bool = False, reason: str = "restart") -> dict:
    """The handoff payload a restart writes (blue_green_manager's
    write_handoff_state). ``signaled_at`` is what its TTL counts from;
    ``last_turn_at`` (epoch seconds) what the age rule counts from."""
    return {
        "kind": KIND,
        "reason": str(reason or KIND)[:40],
        "conversation_tail": list(tail or ()),
        "last_turn_at": float(last_turn_at),
        "signaled_at": float(now),
        "session_opening_turns": list(opening_turns or ()),
        "session_opening_ts": list(opening_ts or ()),
        "session_opening_lost": bool(opening_lost),
    }


def accept(payload, now: float, max_age_s: float = DEFAULT_MAX_AGE_S,
           max_messages: int = DEFAULT_MAX_MESSAGES,
           is_secret: Optional[Callable[[str], bool]] = None
           ) -> Tuple[List[dict], str]:
    """``(messages, why_not)`` for a payload read back at boot: the tail to
    put back (re-checked with tail_for_restart's rules), or ``[]`` and a
    short reason - not a restart payload, no last-turn stamp, the last turn
    too long ago (or in the future), nothing left. Never raises."""
    try:
        if not isinstance(payload, dict) or payload.get("kind") != KIND:
            return [], "not a restart handoff"
        at = payload.get("last_turn_at")
        if not isinstance(at, (int, float)) or isinstance(at, bool) or at <= 0:
            return [], "no last-turn time"
        age = float(now) - float(at)
        if age < -CLOCK_SKEW_S:
            return [], "the last turn is stamped in the future"
        try:
            limit = max(0.0, float(max_age_s))
        except Exception:
            limit = DEFAULT_MAX_AGE_S
        if age > limit:
            return [], (f"the last turn was {age / 60.0:.0f} min ago "
                        f"(over {limit / 60.0:.0f} min)")
        tail = tail_for_restart(payload.get("conversation_tail"),
                                max_messages, is_secret)
        if not tail:
            return [], "nothing to restore"
        return tail, ""
    except Exception:
        return [], "unreadable"


def age_s(payload, now: float) -> float:
    """Seconds since the payload's last turn (0.0 when unknown)."""
    try:
        return max(0.0, float(now) - float(payload.get("last_turn_at")))
    except Exception:
        return 0.0
