"""Owner-spoke-during-call guard for background LLM results (v3.0.27,
overhaul finding reliability-07, the part v3.0.26 left open).

WHY
===
A background brain call (a proactive remark, a screen remark, the boot
changelog summary, a learning batch) cannot be cancelled once its HTTP request
is out. v3.0.26 only made sure none STARTS while the owner is talking. One
that is already running when he speaks still finishes, and its line was then
spoken or queued over his turn or straight after it (the 11-16/17 logs show
calls of ~50 s running through his turns).

WHAT THIS MODULE DOES (stdlib only)
===================================
A process-wide counter of ACCEPTED owner turns (voice or typed). A caller
takes ``mark()`` just before the call and asks ``should_skip(...)`` when the
answer is back: the owner spoke during the call when the counter moved, or
when the caller says a turn / capture of his is in progress right now. A
skipped result is never spoken or queued; pure data writes (learning records)
are never skipped - only the caller decides what is a spoken line.

BG_SKIP_IF_OWNER_SPOKE (core/config.py, default True) False restores the
v3.0.26 behaviour: ``should_skip`` always answers False.

Log line (caller tag only, never the line's text):
``[bg-skip] <caller>: the owner spoke during the call - its line is not
spoken``
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_turns = [0]


def note_owner_turn() -> None:
    """An owner turn was accepted ("You:"), voice or typed. Never raises."""
    try:
        with _lock:
            _turns[0] += 1
    except Exception:
        pass


def mark() -> int:
    """The current owner-turn count: take it right before a background call.
    Never raises."""
    try:
        with _lock:
            return int(_turns[0])
    except Exception:
        return 0


def owner_spoke_since(mark_value, *, active: bool = False) -> bool:
    """True when an owner turn was accepted since ``mark_value`` or one of
    his turns / captures is in progress now (``active``). A mark that cannot
    be read counts as "he spoke" (doubt = do not talk over him). Never
    raises."""
    if active:
        return True
    try:
        return mark() != int(mark_value)
    except Exception:
        return True


def should_skip(caller: str, mark_value, *, active: bool = False,
                enabled: bool = True, log=None) -> bool:
    """Should a background result whose call began at ``mark_value`` NOT be
    spoken / queued? ``enabled`` is BG_SKIP_IF_OWNER_SPOKE (False = never
    skip, the v3.0.26 behaviour). Logs one ``[bg-skip]`` line on a skip.
    Never raises (False on an error: the old behaviour)."""
    try:
        if not enabled:
            return False
        if not owner_spoke_since(mark_value, active=active):
            return False
        if log is not None:
            try:
                tag = "_".join(str(caller or "").split())[:40] or "background"
                log(f"  [bg-skip] {tag}: the owner spoke during the call - "
                    "its line is not spoken")
            except Exception:
                pass
        return True
    except Exception:
        return False
