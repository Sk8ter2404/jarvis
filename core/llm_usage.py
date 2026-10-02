"""Month-to-date Claude token tally, persisted across restarts.

core.llm_client keeps this session's per-model token counts in memory
(``session_usage``). This module folds them into a small JSON file in the
JARVIS data dir so running_costs can report the cloud spend for the whole
calendar month, not just since the last restart.

File (``core.paths.data_file("llm_usage_month.json")``, so staging and the
test runner's JARVIS_DATA_DIR redirect apply) — COUNTS ONLY, never prompt or
reply text:

    {"month": "2026-10",
     "models": {"claude-sonnet-5-5": {"calls": 12, "input": 3400,
                                      "output": 800, "cache_read": 90000,
                                      "cache_write": 4000}},
     "previous": {"month": "2026-09", "models": {...}},
     "updated": 1790000000.0}

Everything read back is re-validated (a ``YYYY-MM`` month, model-id-shaped
keys, non-negative integer counts), so even a hand-edited file cannot carry
text through a rewrite. A missing or corrupt file starts fresh; the first
write in a new month moves the old month's totals under ``previous``.

Writes are debounced: ``note_usage()`` (called by llm_client for every reply
it tallies) schedules ONE timer-driven ``flush()``, immediately when the last
write attempt is FLUSH_INTERVAL_S old and otherwise when that interval is up,
so the voice thread never does file I/O and the file is written at most once
a minute. ``flush()`` also runs at atexit and from the monolith's
``_hard_exit`` (which terminates the process and skips atexit). Each write is
atomic (temp file + fsync + os.replace via core.atomic_io).

Never raises: telemetry must not break a cloud call or an exit.
"""
from __future__ import annotations

import atexit
import json
import re
import threading
import time
from typing import Optional

FILE_NAME = "llm_usage_month.json"
FLUSH_INTERVAL_S = 60.0
_LOCK_TIMEOUT_S = 2.0

_FIELDS = ("calls", "input", "output", "cache_read", "cache_write")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._:\-]{0,79}$")

_clock = time.monotonic
_lock = threading.Lock()         # guards the debounce state below
_flush_lock = threading.Lock()   # one reader-modify-writer of the file at a time
_flushed: dict = {}              # session totals already merged into the file
_last_attempt = [None]           # _clock() of the last write attempt
_timer = [None]                  # the pending flush timer, if any
_atexit_armed = [False]


def usage_path() -> str:
    from core import paths
    return paths.data_file(FILE_NAME)


def _month(now: Optional[float] = None) -> str:
    return time.strftime("%Y-%m",
                         time.localtime(time.time() if now is None else now))


def _count(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def _clean_models(raw) -> dict:
    """Only model-id-shaped keys with integer counts survive."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for model, row in raw.items():
        if (isinstance(model, str) and _MODEL_RE.match(model)
                and isinstance(row, dict)):
            out[model] = {f: _count(row.get(f)) for f in _FIELDS}
    return out


def _load(path: str, month: str) -> tuple[dict, bool]:
    """(record for ``month``, whether the file already held that month).
    Missing or corrupt starts fresh; another month rolls over into
    ``previous``."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = None
    file_month = raw.get("month") if isinstance(raw, dict) else None
    if not isinstance(file_month, str) or not _MONTH_RE.match(file_month):
        return {"month": month, "models": {}}, False
    models = _clean_models(raw.get("models"))
    if file_month != month:
        return {"month": month, "models": {},
                "previous": {"month": file_month, "models": models}}, False
    rec = {"month": month, "models": models}
    prev = raw.get("previous")
    if (isinstance(prev, dict) and isinstance(prev.get("month"), str)
            and _MONTH_RE.match(prev["month"])):
        rec["previous"] = {"month": prev["month"],
                           "models": _clean_models(prev.get("models"))}
    return rec, True


def _session_snapshot() -> dict:
    from core import llm_client
    return llm_client.session_usage_snapshot()


def _delta(session: dict, flushed: dict) -> dict:
    """Per-model counts in ``session`` not yet in ``flushed``."""
    out: dict = {}
    for model, row in (session or {}).items():
        if not isinstance(row, dict):
            continue
        prev = flushed.get(model) or {}
        d = {f: max(0, _count(row.get(f)) - _count(prev.get(f)))
             for f in _FIELDS}
        if any(d.values()):
            out[model] = d
    return out


def _add(models: dict, delta: dict) -> None:
    for model, d in delta.items():
        key = model if isinstance(model, str) and _MODEL_RE.match(model) \
            else "other"
        row = models.setdefault(key, dict.fromkeys(_FIELDS, 0))
        for f in _FIELDS:
            row[f] += d[f]


def flush(now: Optional[float] = None) -> bool:
    """Merge the session tally's unflushed counts into the month file.
    True when the file is up to date (or there was nothing to add)."""
    if not _flush_lock.acquire(timeout=_LOCK_TIMEOUT_S):
        return False
    try:
        session = _session_snapshot()
        with _lock:
            delta = _delta(session, _flushed)
            if not delta:
                return True
            _last_attempt[0] = _clock()
        now = time.time() if now is None else now
        path = usage_path()
        rec, _ = _load(path, _month(now))
        _add(rec["models"], delta)
        rec["updated"] = round(now, 1)
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(path, rec)
        with _lock:
            _flushed.clear()
            _flushed.update({m: dict(r) for m, r in session.items()
                             if isinstance(r, dict)})
        return True
    except Exception as e:
        print(f"  [llm-usage] month tally write failed: {type(e).__name__}: {e}")
        return False
    finally:
        _flush_lock.release()


def _start_timer(delay: float, fn) -> threading.Timer:
    t = threading.Timer(delay, fn)
    t.daemon = True
    t.start()
    return t


def _timer_flush() -> None:
    try:
        flush()
    finally:
        with _lock:
            _timer[0] = None
        # A reply tallied while that write was in flight is caught by the
        # next debounced write instead of waiting for the next reply.
        try:
            session = _session_snapshot()
            with _lock:
                pending = _delta(session, _flushed)
            if pending:
                note_usage()
        except Exception:
            pass


def note_usage() -> None:
    """A reply was tallied: make sure a debounced flush is scheduled."""
    try:
        if not _atexit_armed[0]:
            _atexit_armed[0] = True
            atexit.register(flush)
        with _lock:
            if _timer[0] is not None:
                return
            last = _last_attempt[0]
            wait = 0.0 if last is None else max(
                0.0, last + FLUSH_INTERVAL_S - _clock())
            _timer[0] = _start_timer(wait, _timer_flush)
    except Exception:
        pass


def month_usage(now: Optional[float] = None) -> Optional[dict]:
    """This calendar month's per-model counts — the file's plus the session's
    not-yet-flushed ones — or None when nothing is persisted for this month
    (missing, corrupt or last month's file)."""
    if not _flush_lock.acquire(timeout=_LOCK_TIMEOUT_S):
        return None
    try:
        rec, current = _load(usage_path(), _month(now))
        if not current:
            return None
        session = _session_snapshot()
        with _lock:
            pending = _delta(session, _flushed)
        _add(rec["models"], pending)
        return rec["models"]
    except Exception:
        return None
    finally:
        _flush_lock.release()
