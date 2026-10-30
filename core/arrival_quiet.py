"""core/arrival_quiet.py - a short quiet period after he arrives (v3.0.10).

Live 2026-10-29 07:21-07:23 he walked in to the "while you were away" recap,
the morning briefing, an anticipation pattern offer and a screen-watch
stretch nudge inside 90 seconds. The arrival briefing (skills/daily_briefing/
arrival_v2._fire_arrival) now stamps the moment here, and the skills that
start a remark of their own (skills/screen_watch.py, skills/
anticipation_engine.py) wait ARRIVAL_QUIET_MINUTES (core/config.py, read live)
before they speak. 0 minutes = no quiet period (the 3.0.9 behaviour).

One process-wide cell (this module is imported once as core.arrival_quiet by
every skill). Stdlib only; nothing here raises.
"""
from __future__ import annotations

import time

# time.monotonic() of the last arrival stamp; 0.0 = none this session.
_arrived_at = [0.0]


def _quiet_seconds() -> float:
    """core.config.ARRIVAL_QUIET_MINUTES in seconds, read live (0 = off).
    Never raises (0)."""
    try:
        import core.config as _cfg
        return max(0.0, float(getattr(_cfg, "ARRIVAL_QUIET_MINUTES", 0) or 0)) * 60.0
    except Exception:
        return 0.0


def note_arrival(now: float | None = None) -> None:
    """Stamp an arrival (the briefing was just queued). Never raises."""
    try:
        _arrived_at[0] = float(time.monotonic() if now is None else now)
    except Exception:
        pass


def arrival_quiet_active(now: float | None = None) -> bool:
    """True within ARRIVAL_QUIET_MINUTES of the last note_arrival(). False
    with the setting at 0, before any arrival, or on any error."""
    try:
        quiet = _quiet_seconds()
        at = float(_arrived_at[0] or 0.0)
        if quiet <= 0.0 or at <= 0.0:
            return False
        t = float(time.monotonic() if now is None else now)
        return 0.0 <= t - at < quiet
    except Exception:
        return False
