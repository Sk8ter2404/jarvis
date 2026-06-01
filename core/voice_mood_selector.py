"""
core/voice_mood_selector.py

Voice mood layer for the JARVIS TTS pipeline.

The base prosody stack (bobert_companion._TTS_EMOTION_PRESETS) is a
homogeneous "neutral baseline + tone variants" set. This module sits on top
of it as a context-aware mood router so the same words can be delivered
with the cadence the moment calls for:

    calm_efficient   — default. Neutral, brisk-but-relaxed. The baseline
                       MCU JARVIS register.
    urgent_clipped   — alerts, a VIP's calls during work hours, anything
                       time-critical. Faster, slightly higher, crisp.
    dry_amused       — banter, Chappie recall, lightly mocking asides.
                       Slight smile in the voice.
    concerned_soft   — self-diagnostic fired a HIGH-severity problem,
                       or anything where lower energy + softer gain reads
                       as care rather than alarm.

`select_mood(context)` returns one of those four strings. The caller
(bobert_companion._resolve_tts_preset) looks the name up in
_TTS_EMOTION_PRESETS to get the (rate, pitch, gain) triple.

Context dict keys (all optional):
    self_diagnostic_problem : bool   — recent HIGH-severity probe failure
    vip_intercept           : bool   — a VIP is the call/DM source right now
    work_hours              : bool   — override the time-of-day check
    now                     : float  — override time.time() for testing
    chappie_mode            : bool   — Chappie recall / observational reply
    banter                  : bool   — banter skill is driving this reply
    conversation_mode       : str    — e.g. "chappie", "banter", "default"

Priority (first match wins) mirrors the spec ordering — diagnostic concern
overrides everything else because a real system problem outranks tone.
"""

from __future__ import annotations

import time
from typing import Optional


CALM_EFFICIENT = "calm_efficient"
URGENT_CLIPPED = "urgent_clipped"
DRY_AMUSED     = "dry_amused"
CONCERNED_SOFT = "concerned_soft"

VALID_MOODS: frozenset[str] = frozenset({
    CALM_EFFICIENT, URGENT_CLIPPED, DRY_AMUSED, CONCERNED_SOFT,
})

# Local hours considered "work hours" — a VIP's calls during this window get
# the urgent_clipped treatment. Outside this window a VIP call is unusual
# enough that the calm_efficient default reads better.
WORK_HOURS_START = 8     # 08:00 local
WORK_HOURS_END   = 18    # 18:00 local


def _is_work_hours(now: Optional[float] = None) -> bool:
    """True iff the current local hour is within [WORK_HOURS_START, WORK_HOURS_END)."""
    ts = time.time() if now is None else now
    try:
        h = time.localtime(ts).tm_hour
    except Exception:
        return False
    return WORK_HOURS_START <= h < WORK_HOURS_END


def select_mood(context: Optional[dict] = None) -> str:
    """Return the mood name to use for the next utterance.

    See module docstring for the recognised context keys and priority.
    Unknown keys are ignored. Falls back to `calm_efficient` so a missing
    context never silences variation."""
    ctx = context or {}

    if ctx.get("self_diagnostic_problem"):
        return CONCERNED_SOFT

    mode = (ctx.get("conversation_mode") or "").strip().lower()
    if ctx.get("chappie_mode") or ctx.get("banter") or mode in ("chappie", "banter"):
        return DRY_AMUSED

    if ctx.get("vip_intercept"):
        work_override = ctx.get("work_hours")
        if work_override is None:
            work_override = _is_work_hours(ctx.get("now"))
        if work_override:
            return URGENT_CLIPPED

    return CALM_EFFICIENT


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    cases: list[tuple[dict, str]] = [
        ({},                                                       CALM_EFFICIENT),
        ({"self_diagnostic_problem": True},                        CONCERNED_SOFT),
        ({"self_diagnostic_problem": True, "vip_intercept": True}, CONCERNED_SOFT),
        ({"chappie_mode": True},                                   DRY_AMUSED),
        ({"banter": True},                                         DRY_AMUSED),
        ({"conversation_mode": "chappie"},                         DRY_AMUSED),
        ({"vip_intercept": True, "work_hours": True},              URGENT_CLIPPED),
        ({"vip_intercept": True, "work_hours": False},             CALM_EFFICIENT),
        ({"vip_intercept": True, "now": time.mktime((2026, 5, 29, 10, 0, 0, 0, 0, -1))},
                                                                   URGENT_CLIPPED),
        ({"vip_intercept": True, "now": time.mktime((2026, 5, 29, 22, 0, 0, 0, 0, -1))},
                                                                   CALM_EFFICIENT),
    ]
    for ctx, expected in cases:
        got = select_mood(ctx)
        ok = "OK " if got == expected else "BAD"
        print(f"  {ok}  {ctx!r:<70} -> {got!r:<18} expected={expected!r}")
