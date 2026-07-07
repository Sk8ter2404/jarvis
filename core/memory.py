"""
core/memory.py — Contextual callback / pending-promise mechanism for JARVIS.

When JARVIS announces a long-running task ('I'll let you know when the print
finishes', 'I'll tell you once the bed cools'), it should not be a fire-and-
forget line — the announcement is a *promise* and the user should hear the
follow-up automatically when the underlying condition is actually met.

This module stores those promises, persists them to disk (so they survive
restarts), runs a watcher thread that evaluates their conditions, and pushes
the deferred message into JARVIS's normal proactive-speech queue when a
condition fires.

Promises are also surfaced as a plain-text-ish JSON file at
    memory/pending_promises.json
so the user can inspect outstanding promises any time.

Public API
──────────
    make_promise(message, condition, *, params=None, deadline_s=None,
                 source='unknown') -> int
        Store a new promise and return its integer id.

    list_promises(include_delivered=False) -> list[dict]
        Snapshot of all known promises (deep-copied).

    cancel_promise(promise_id) -> bool
        Mark a pending promise cancelled (no announcement fires).

    fulfil_promise(promise_id) -> bool
        Force-fire a promise immediately (used by the 'manual' condition or
        by callers that already know the condition was met out-of-band).

    register_condition(name, predicate) -> None
        Add or override a condition. The predicate is called with
        (promise: dict) and must return True iff the promise should fire now.

    start_watcher(announce_callable=None, *, interval_s=30.0) -> None
        Kick off the background watcher thread. Idempotent — calling twice
        is a no-op. `announce_callable` accepts (message: str, source: str)
        and is used to surface fired promises; if omitted, the watcher does
        a lazy import of bobert_companion.proactive_announce.

    stop_watcher() -> None
        Signal the watcher to exit (used in tests).

Built-in conditions
───────────────────
    bambu_print_finish    Bambu monitor reports gcode_state=='FINISH'.
                          Params: {} (uses fresh-after_ts so a finish from
                          before the promise was made doesn't trigger).
    bambu_bed_cool        Bambu bed_temper drops below threshold AFTER a
                          FINISH has been observed since the promise was
                          made. Params: {threshold_c: float, default 40.0}.
    time_at               Wall-clock time reaches a given epoch second.
                          Params: {epoch: float}.
    time_after            N seconds elapse from promise creation.
                          Params: {delay_s: float}.
    manual                Never auto-fires; only fulfil_promise() releases it.

Skills can register their own conditions with register_condition().
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────────────────────────────────
#  PATHS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEM_DIR      = os.path.join(_PROJECT_DIR, "memory")
_PROMISES_FILE = os.path.join(_MEM_DIR, "pending_promises.json")

# How long delivered/cancelled promises stick around in the file so the
# user can review what got fired. Once older than this they're pruned on
# the next save.
_RETENTION_S = 7 * 24 * 3600

# Hard cap on the number of *pending* promises retained on disk. A 'manual'
# or never-true-condition promise stays pending forever and the retention
# prune above never touches pending entries — so without a cap they grow
# unbounded. Beyond this we drop the OLDEST pending promises (keeping the
# newest _MAX_PENDING_PROMISES). Delivered/cancelled/expired entries are
# unaffected (the retention prune governs those).
_MAX_PENDING_PROMISES = 200

# Hard cap on params dict size — defence against a runaway skill stuffing
# huge payloads into the persisted file. (Sane skills use tiny dicts.)
_MAX_PARAMS_BYTES = 4096


# ──────────────────────────────────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────────────────────────────────

# All shared mutable state goes under this lock. The watcher thread also
# uses it for read-modify-write on the promises list.
_lock = threading.RLock()

# In-memory promise registry. Loaded lazily from disk on first access.
# Each promise:
#   {
#     "id": int,
#     "created_at": float (epoch),
#     "deadline":   float | None,
#     "message":    str,
#     "condition":  str,
#     "params":     dict,
#     "source":     str,
#     "status":     "pending" | "delivered" | "cancelled" | "expired",
#     "fired_at":   float | None,
#   }
_promises: list[dict] = []
_next_id: list[int] = [1]
# Set by a condition predicate that MUTATED a promise's params (e.g.
# _cond_bambu_bed_cool latching "_finish_seen") without firing — so _tick knows
# to persist the change even though nothing fired/expired this pass. Without it,
# the mutation lived only in memory and the next tick's _load_locked() reloaded
# the un-persisted promise from disk, wiping the latch (2026-07-07 bug-hunt).
_promises_dirty: list[bool] = [False]


def _mark_promises_dirty() -> None:
    """Called by a predicate that changed a promise's stored params but did not
    fire it, so _tick persists the change this pass."""
    _promises_dirty[0] = True
_loaded: list[bool] = [False]

# Condition registry: name -> predicate(promise: dict) -> bool
_conditions: dict[str, Callable[[dict], bool]] = {}

# Watcher thread plumbing.
_watcher_thread: list[Optional[threading.Thread]] = [None]
_watcher_stop = threading.Event()
_announce_fn: list[Optional[Callable[[str, str], Any]]] = [None]


# ──────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    try:
        os.makedirs(_MEM_DIR, exist_ok=True)
    except Exception:
        pass


def _load_locked() -> None:
    """Load promises from disk into the in-memory registry. Caller must
    already hold _lock."""
    if _loaded[0]:
        return
    _loaded[0] = True
    if not os.path.exists(_PROMISES_FILE):
        return
    try:
        with open(_PROMISES_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return
        decoded, _ = json.JSONDecoder().raw_decode(raw)
        if not isinstance(decoded, list):
            return
    except Exception as e:
        print(f"  [promises] failed to load {_PROMISES_FILE}: {e}")
        return

    max_seen = 0
    for entry in decoded:
        if not isinstance(entry, dict):
            continue
        # Defensive defaults so a hand-edited file can't crash the watcher.
        entry.setdefault("id", 0)
        entry.setdefault("created_at", 0.0)
        entry.setdefault("deadline", None)
        entry.setdefault("message", "")
        entry.setdefault("condition", "manual")
        entry.setdefault("params", {})
        entry.setdefault("source", "unknown")
        entry.setdefault("status", "pending")
        entry.setdefault("fired_at", None)
        try:
            entry["id"] = int(entry["id"])
        except (TypeError, ValueError):
            entry["id"] = 0
        if not isinstance(entry["params"], dict):
            entry["params"] = {}
        if entry["id"] > max_seen:
            max_seen = entry["id"]
        _promises.append(entry)
    _next_id[0] = max_seen + 1


def _save_locked() -> None:
    """Atomically write promises to disk. Caller must already hold _lock."""
    _ensure_dir()
    now = time.time()
    # Prune promises that are old AND no longer pending.
    keep = [
        p for p in _promises
        if p.get("status") == "pending"
        or (now - (p.get("fired_at") or p.get("created_at") or now)) < _RETENTION_S
    ]
    # Cap pending promises: a 'manual'/never-true condition never fires and is
    # never pruned above, so pending entries can accumulate without bound. If
    # over the cap, drop the OLDEST pending ones (by created_at) and keep the
    # newest _MAX_PENDING_PROMISES. Non-pending entries are left untouched so
    # the retention window above still governs delivered/cancelled/expired.
    pending = [p for p in keep if p.get("status") == "pending"]
    if len(pending) > _MAX_PENDING_PROMISES:
        survivors = sorted(
            pending, key=lambda p: p.get("created_at", 0.0)
        )[-_MAX_PENDING_PROMISES:]
        survivor_ids = {id(p) for p in survivors}
        dropped = len(pending) - len(survivors)
        keep = [
            p for p in keep
            if p.get("status") != "pending" or id(p) in survivor_ids
        ]
        print(f"  [promises] pending cap reached — dropped {dropped} oldest "
              f"pending (keeping newest {_MAX_PENDING_PROMISES}).")
    if len(keep) != len(_promises):
        _promises[:] = keep
    try:
        fd, tmp = tempfile.mkstemp(dir=_MEM_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_promises, f, indent=2)
            os.replace(tmp, _PROMISES_FILE)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            raise
    except Exception as e:
        print(f"  [promises] failed to save {_PROMISES_FILE}: {e}")


# ──────────────────────────────────────────────────────────────────────────
#  BUILT-IN CONDITIONS
# ──────────────────────────────────────────────────────────────────────────

def _bambu_state() -> dict:
    """Pull the latest Bambu monitor snapshot, or {} if the skill isn't
    loaded / hasn't received any push messages yet."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return {}
    try:
        with mod._state_lock:                    # type: ignore[attr-defined]
            return dict(mod._state)              # type: ignore[attr-defined]
    except Exception:
        return {}


def _cond_bambu_print_finish(promise: dict) -> bool:
    st = _bambu_state()
    if not st:
        return False
    if (st.get("last_update") or 0) <= promise.get("created_at", 0.0):
        # The printer hasn't reported anything since the promise was made,
        # so we can't yet trust the current state.
        return False
    return (st.get("gcode_state") or "").upper() == "FINISH"


def _cond_bambu_bed_cool(promise: dict) -> bool:
    st = _bambu_state()
    if not st:
        return False
    threshold = float(promise.get("params", {}).get("threshold_c", 40.0))
    # Wait until we've actually seen a FINISH at-or-after the promise was
    # created — otherwise "bed below 40°C" trivially fires on an idle printer.
    if not promise["params"].get("_finish_seen"):
        if (st.get("gcode_state") or "").upper() == "FINISH" \
           and (st.get("last_update") or 0) >= promise.get("created_at", 0.0):
            promise["params"]["_finish_seen"] = True
            # Persist the latch this tick — otherwise a restart in the
            # FINISH→bed-cool window loses it and the promise waits forever for
            # a second FINISH that never comes (the print already finished).
            _mark_promises_dirty()
        else:
            return False
    bed = st.get("bed_temper")
    if bed is None:
        return False
    try:
        return float(bed) < threshold
    except (TypeError, ValueError):
        return False


def _cond_time_at(promise: dict) -> bool:
    try:
        target = float(promise["params"].get("epoch", 0.0))
    except (TypeError, ValueError):
        return False
    return target > 0 and time.time() >= target


def _cond_time_after(promise: dict) -> bool:
    try:
        delay = float(promise["params"].get("delay_s", 0.0))
    except (TypeError, ValueError):
        return False
    return delay > 0 and (time.time() - promise.get("created_at", 0.0)) >= delay


def _cond_manual(_: dict) -> bool:
    return False  # only fulfil_promise() can release these


_conditions.update({
    "bambu_print_finish": _cond_bambu_print_finish,
    "bambu_bed_cool":     _cond_bambu_bed_cool,
    "time_at":            _cond_time_at,
    "time_after":         _cond_time_after,
    "manual":             _cond_manual,
})


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────

def register_condition(name: str, predicate: Callable[[dict], bool]) -> None:
    """Register a custom condition predicate. Overwrites any existing
    condition with the same name."""
    with _lock:
        _conditions[str(name)] = predicate


def make_promise(message: str,
                 condition: str,
                 *,
                 params: Optional[dict] = None,
                 deadline_s: Optional[float] = None,
                 source: str = "unknown") -> int:
    """Store a new pending promise. Returns the new promise's id.

    `condition` must be a registered condition name. Unknown conditions
    are stored anyway — the watcher just won't fire them — so the user
    can still see the promise in pending_promises.json.

    `deadline_s` is a wall-clock-relative timeout in seconds after which
    the promise auto-expires without firing (and is logged). None = forever.
    """
    if not message:
        raise ValueError("promise needs a non-empty message")
    if not condition:
        raise ValueError("promise needs a condition name")
    if params is None:
        params = {}
    if len(json.dumps(params, default=str)) > _MAX_PARAMS_BYTES:
        raise ValueError(f"params dict exceeds {_MAX_PARAMS_BYTES} bytes")

    now = time.time()
    with _lock:
        _load_locked()
        pid = _next_id[0]
        _next_id[0] += 1
        promise = {
            "id":         pid,
            "created_at": now,
            "deadline":   (now + deadline_s) if deadline_s else None,
            "message":    str(message),
            "condition":  str(condition),
            "params":     dict(params),
            "source":     str(source),
            "status":     "pending",
            "fired_at":   None,
        }
        _promises.append(promise)
        _save_locked()
    print(f"  [promises] #{pid} stored ({condition}) from {source}: {message!r}")
    return pid


def list_promises(include_delivered: bool = False) -> list[dict]:
    """Snapshot of promises, deep-copied so callers can mutate freely."""
    with _lock:
        _load_locked()
        out = []
        for p in _promises:
            if include_delivered or p.get("status") == "pending":
                out.append(copy.deepcopy(p))
        return out


def cancel_promise(promise_id: int) -> bool:
    """Mark a pending promise cancelled. Returns True iff a pending promise
    with that id was found."""
    with _lock:
        _load_locked()
        for p in _promises:
            if p.get("id") == promise_id and p.get("status") == "pending":
                p["status"] = "cancelled"
                p["fired_at"] = time.time()
                _save_locked()
                print(f"  [promises] #{promise_id} cancelled")
                return True
    return False


def fulfil_promise(promise_id: int) -> bool:
    """Force-fire a pending promise now. Returns True on success."""
    with _lock:
        _load_locked()
        for p in _promises:
            if p.get("id") == promise_id and p.get("status") == "pending":
                _fire_promise_locked(p)
                _save_locked()
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────
#  WATCHER
# ──────────────────────────────────────────────────────────────────────────

def _resolve_announce_fn() -> Callable[[str, str], Any]:
    """Return whatever callable surfaces a promise message. Tries the
    explicitly-registered announce_fn first, then a lazy import of
    bobert_companion.proactive_announce, then a console-print fallback."""
    if _announce_fn[0] is not None:
        return _announce_fn[0]
    try:
        import bobert_companion  # lazy — module may not be importable in tests
        fn = getattr(bobert_companion, "proactive_announce", None)
        if callable(fn):
            return fn
    except Exception:
        pass
    return lambda message, source="promise": print(
        f"  [promises:{source}] (no announcer) {message}"
    )


def _fire_promise_locked(promise: dict) -> None:
    """Send the promise's message to the speech queue and mark it delivered.
    Caller must already hold _lock."""
    announcer = _resolve_announce_fn()
    msg = promise.get("message", "")
    src = f"promise:{promise.get('source', 'unknown')}"
    try:
        announcer(msg, src)
    except Exception as e:
        print(f"  [promises] announcer raised on #{promise.get('id')}: {e}")
    promise["status"] = "delivered"
    promise["fired_at"] = time.time()
    print(f"  [promises] #{promise.get('id')} delivered: {msg!r}")


def _tick() -> None:
    """One pass over all pending promises. Fires any whose condition is
    true, expires any past their deadline."""
    now = time.time()
    any_change = False
    with _lock:
        _load_locked()
        _promises_dirty[0] = False   # predicates below may set it via _mark_promises_dirty
        for p in _promises:
            if p.get("status") != "pending":
                continue
            # Deadline expiry takes precedence over the condition check.
            dl = p.get("deadline")
            if dl is not None and now >= dl:
                p["status"] = "expired"
                p["fired_at"] = now
                any_change = True
                print(f"  [promises] #{p.get('id')} expired without firing")
                continue
            cond_name = p.get("condition") or ""
            pred = _conditions.get(cond_name)
            if pred is None:
                continue
            try:
                ready = bool(pred(p))
            except Exception as e:
                print(f"  [promises] condition {cond_name!r} raised on #{p.get('id')}: {e}")
                continue
            if ready:
                _fire_promise_locked(p)
                any_change = True
        if any_change or _promises_dirty[0]:
            _save_locked()


def _watcher_loop(interval_s: float) -> None:
    print(f"  [promises] watcher started (interval={interval_s:.0f}s)")
    while not _watcher_stop.is_set():
        try:
            try:
                _tick()
            except Exception as e:
                print(f"  [promises] watcher tick raised: {e}")
            # Wait returns True if the stop event was set during the sleep, so
            # we exit promptly when stop_watcher() is called.
            if _watcher_stop.wait(interval_s):
                break
        except Exception:
            logging.exception("[promises] watcher loop iteration crashed")
            # Avoid a tight crash loop if something keeps raising.
            if _watcher_stop.wait(interval_s):
                break
    print("  [promises] watcher stopped")


def start_watcher(announce_callable: Optional[Callable[[str, str], Any]] = None,
                  *,
                  interval_s: float = 30.0) -> None:
    """Start the background watcher thread. Idempotent."""
    if announce_callable is not None:
        _announce_fn[0] = announce_callable
    with _lock:
        if _watcher_thread[0] is not None and _watcher_thread[0].is_alive():
            return
        _watcher_stop.clear()
        t = threading.Thread(
            target=_watcher_loop,
            args=(max(1.0, float(interval_s)),),
            daemon=True,
            name="promises-watcher",
        )
        _watcher_thread[0] = t
        t.start()


def stop_watcher() -> None:
    """Ask the watcher thread to exit. Returns after it joins (with a short
    timeout) or immediately if no thread is running."""
    _watcher_stop.set()
    t = _watcher_thread[0]
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _watcher_thread[0] = None


# ──────────────────────────────────────────────────────────────────────────
#  ACTION HELPERS
# ──────────────────────────────────────────────────────────────────────────
#
#  These are wired into bobert_companion's ACTIONS dict so the user can ask
#  JARVIS verbally about outstanding promises.

def action_list_promises(_: str = "") -> str:
    """JARVIS-style summary of pending promises."""
    pending = list_promises(include_delivered=False)
    if not pending:
        return "No outstanding promises, sir."
    lines = []
    now = time.time()
    for p in pending:
        age = int(now - p.get("created_at", now))
        if age < 60:
            age_str = f"{age}s ago"
        elif age < 3600:
            age_str = f"{age // 60}m ago"
        else:
            age_str = f"{age // 3600}h{(age % 3600) // 60:02d}m ago"
        lines.append(f"  #{p['id']} [{p['condition']}, {age_str}] {p['message']}")
    head = f"{len(pending)} outstanding promise(s), sir:"
    return head + "\n" + "\n".join(lines)


def action_cancel_promise(args: str) -> str:
    """Cancel a pending promise by id."""
    args = (args or "").strip()
    if not args:
        return "format: cancel_promise, <id>"
    try:
        pid = int(args)
    except ValueError:
        return f"could not parse promise id from {args!r}"
    if cancel_promise(pid):
        return f"Very good, sir — promise #{pid} cancelled."
    return f"I don't have a pending promise with id #{pid}, sir."


def register_actions(actions: dict) -> None:
    """Add the inspection actions to a bobert_companion-style ACTIONS dict."""
    actions["list_promises"]  = action_list_promises
    actions["cancel_promise"] = action_cancel_promise
