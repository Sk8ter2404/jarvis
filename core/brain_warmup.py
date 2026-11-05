"""Is the local brain still doing its boot warm-up? (v3.0.11,
ARRIVAL_NEWS_WAIT_FOR_BRAIN). The monolith's warm-up thread notes its start and
end; a skill that needs the model at boot can ask without importing the monolith.
A warm-up that never reported its end stops counting after WARMUP_MAX_S. Never
raises."""
from __future__ import annotations

import threading
import time

WARMUP_MAX_S = 300.0

_lock = threading.Lock()
_started_at = [None]


def note_start(now: "float | None" = None) -> None:
    with _lock:
        _started_at[0] = time.monotonic() if now is None else float(now)


def note_done() -> None:
    with _lock:
        _started_at[0] = None


def warming(now: "float | None" = None) -> bool:
    """True while a warm-up is running (and younger than WARMUP_MAX_S)."""
    with _lock:
        at = _started_at[0]
    if at is None:
        return False
    t = time.monotonic() if now is None else float(now)
    return t - at <= WARMUP_MAX_S
