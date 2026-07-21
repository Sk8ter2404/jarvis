"""
core.scheduler — APScheduler-backed cron/interval/one-shot/conditional job engine.

Persistent SQLite jobstore at ``data/scheduler.db`` so cron entries survive
JARVIS restarts.  Three trigger families are supported natively:

    * cron      — ``minute / hour / day / month / day_of_week`` fields
    * interval  — every N seconds/minutes/hours
    * date      — fire once at an absolute timestamp

A fourth family — *conditional* — is implemented in-process because
APScheduler doesn't model "fire when predicate flips False→True".  Each
conditional spec lives in ``data/scheduler_conditions.json`` and a single
poller thread re-evaluates them every ``_CONDITION_POLL_SECONDS``.

The module is deliberately tolerant of missing deps and missing config: if
APScheduler isn't installed, every public entry point degrades to a stub
and ``is_available()`` returns False so the skill loader can surface a
clean install hint instead of crashing the boot sequence.

Why module-level dispatch?
--------------------------
APScheduler serialises jobs by importable path.  Persisted jobs therefore
have to point at a callable that can be re-resolved on the next boot.  All
job dispatch goes through ``run_action`` (module-level, see below) which
looks up ``action`` in the in-process ACTIONS dict installed by
``bootstrap()``.  Jobs persisted to the SQLite store reference
``core.scheduler:run_action`` so they re-bind cleanly after a restart.

Public API
----------
    is_available()                   -> True iff APScheduler installed
    bootstrap(actions)               -> starts the scheduler + condition poller
    schedule_cron(...)               -> add a cron job
    schedule_interval(...)           -> add an interval job
    schedule_once(...)               -> add a one-shot date job
    schedule_when(name, condition, action, arg=..., chain=...)
                                     -> add a conditional trigger
    register_condition(name, fn)     -> register a custom boolean predicate
    list_jobs()                      -> list of {id, kind, next_run, action, arg, ...}
    list_conditions()                -> list of conditional triggers
    available_conditions()           -> registered condition names
    cancel_job(job_id)               -> remove a job (cron/interval/date OR conditional)
    fire_now(job_id)                 -> dispatch a job's action immediately
    shutdown(wait=False)             -> stop the scheduler
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable


_log = logging.getLogger("jarvis.scheduler")

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_DB_PATH      = os.path.join(_DATA_DIR, "scheduler.db")
_CONDITIONS_PATH = os.path.join(_DATA_DIR, "scheduler_conditions.json")

# How often the condition poller re-evaluates every registered predicate.
# 30s keeps the overhead negligible while still feeling responsive for the
# Bambu-print-finished / disk-low style triggers the spec calls out.
_CONDITION_POLL_SECONDS = 30.0
# Wait this long after boot before the first condition sweep — gives skills
# that publish state files (bambu, system_monitor) time to come online so
# we don't spuriously fire a "printer idle" trigger during early boot.
_CONDITION_BOOT_DELAY   = 20.0


# ── atomic write helper ─────────────────────────────────────────────
# Same fallback pattern as voice_id / notification_triage — prefer
# core.atomic_io but inline an equivalent so this module loads cleanly
# during early boot before core.atomic_io's import has completed.
try:
    from core.atomic_io import _atomic_write_json  # type: ignore
except Exception:  # pragma: no cover - boot-order safety
    import tempfile

    def _atomic_write_json(path: str, data: Any, *, indent: int = 2) -> None:
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise


# ── lazy APScheduler import ─────────────────────────────────────────
_imports: dict[str, Any] = {}
_import_error: str | None = None


def _aps_imports() -> dict[str, Any]:
    """Resolve APScheduler symbols lazily. Cached on first success."""
    global _import_error
    if _imports:
        return _imports
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
        from apscheduler.triggers.cron     import CronTrigger              # type: ignore
        from apscheduler.triggers.interval import IntervalTrigger          # type: ignore
        from apscheduler.triggers.date     import DateTrigger              # type: ignore
        try:
            from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore  # type: ignore
        except Exception:
            SQLAlchemyJobStore = None  # falls back to in-memory jobstore
    except Exception as e:
        _import_error = f"{type(e).__name__}: {e}"
        return {}
    _imports.update({
        "BackgroundScheduler": BackgroundScheduler,
        "CronTrigger":         CronTrigger,
        "IntervalTrigger":     IntervalTrigger,
        "DateTrigger":         DateTrigger,
        "SQLAlchemyJobStore":  SQLAlchemyJobStore,
    })
    return _imports


def is_available() -> bool:
    return bool(_aps_imports())


# ── shared state ────────────────────────────────────────────────────
_lock = threading.RLock()
_state: dict[str, Any] = {
    "scheduler":    None,    # BackgroundScheduler instance
    "actions":      None,    # ACTIONS dict from bobert_companion (injected at bootstrap)
    "started_at":   None,
    "conditions":   {},      # name → callable returning bool
    "cond_thread":  None,
    "cond_stop":    None,    # threading.Event
    "cond_state":   {},      # condition_name → last_seen_bool (debounce)
    "last_error":   None,
}


# ── module-level dispatch (persistable) ─────────────────────────────
def run_action(action: str, arg: str = "", chain: list | None = None) -> str:
    """The single entry point every persisted job points at.

    APScheduler resolves persisted jobs by importable path, so jobs must
    point at a top-level callable (not a closure over the actions dict).
    Resolution happens at fire-time against ``_state["actions"]`` —
    installed by ``bootstrap()`` and refreshed automatically on every
    JARVIS boot.

    ``chain`` is an optional list of ``[{"action": "...", "arg": "..."}]``
    entries dispatched in order after ``action`` so a single scheduled
    job can run "brief emails, then weather, then play lo-fi".
    """
    with _lock:
        actions = _state.get("actions")
    if not actions:
        _log.warning("scheduler.run_action fired before bootstrap (action=%s)", action)
        return f"scheduler not bootstrapped (skipped {action})"

    results: list[str] = []
    steps: list[dict] = [{"action": action, "arg": arg}]
    if chain:
        for entry in chain:
            if isinstance(entry, dict) and entry.get("action"):
                steps.append({
                    "action": entry["action"],
                    "arg":    entry.get("arg") or "",
                })

    for step in steps:
        name = step["action"]
        a    = step.get("arg") or ""
        fn   = actions.get(name)
        if fn is None:
            results.append(f"{name}: not registered")
            continue
        try:
            rv = fn(a)
            results.append(f"{name}: {rv}" if isinstance(rv, str) else f"{name}: ok")
        except Exception as e:
            _log.exception("scheduled action %s raised", name)
            results.append(f"{name}: {type(e).__name__}: {e}")
    return " | ".join(results)


# ── condition registry ──────────────────────────────────────────────
# Built-in predicates wire into existing JARVIS state files.  Custom
# predicates can be added by other skills via register_condition().
def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cond_bambu_print_finished() -> bool:
    """True when the Bambu printer has just transitioned to FINISH state.

    Reads ``bambu_overlay_state.json`` which is the canonical state file
    written by ``skills/bambu_monitor.py``.  The poller debounces by
    tracking previous values, so this returns the *current* status and
    the poller fires only on False→True transitions.
    """
    state = _read_json_safe(os.path.join(_PROJECT_DIR, "bambu_overlay_state.json"))
    gcode = (state.get("gcode_state") or "").upper()
    return gcode in {"FINISH", "FINISHED", "SUCCESS"}


def _cond_bambu_print_failed() -> bool:
    state = _read_json_safe(os.path.join(_PROJECT_DIR, "bambu_overlay_state.json"))
    gcode = (state.get("gcode_state") or "").upper()
    if gcode in {"FAILED", "ABORTED", "CANCELLED"}:
        return True
    err = state.get("print_error") or state.get("error") or ""
    return bool(err and str(err).strip() not in {"", "0", "None"})


def _cond_bambu_print_started() -> bool:
    state = _read_json_safe(os.path.join(_PROJECT_DIR, "bambu_overlay_state.json"))
    gcode = (state.get("gcode_state") or "").upper()
    return gcode in {"RUNNING", "PRINTING", "PREPARE"}


def _cond_disk_low() -> bool:
    """True when the project drive has < 1 GB free."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(_PROJECT_DIR)
        return free < 1024 * 1024 * 1024
    except Exception:
        return False


def _cond_ram_high() -> bool:
    """True when host RAM usage is over 90%."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().percent >= 90.0
    except Exception:
        return False


_BUILTIN_CONDITIONS: dict[str, Callable[[], bool]] = {
    "bambu_print_finished": _cond_bambu_print_finished,
    "bambu_print_failed":   _cond_bambu_print_failed,
    "bambu_print_started":  _cond_bambu_print_started,
    "disk_low":             _cond_disk_low,
    "ram_high":             _cond_ram_high,
}


def register_condition(name: str, fn: Callable[[], bool]) -> None:
    """Register a custom boolean predicate other skills can trigger off."""
    if not name or not callable(fn):
        raise ValueError("register_condition requires (name, callable)")
    with _lock:
        _state["conditions"][name] = fn


def available_conditions() -> list[str]:
    with _lock:
        return sorted(set(_BUILTIN_CONDITIONS.keys()) | set(_state["conditions"].keys()))


def _resolve_condition(name: str) -> Callable[[], bool] | None:
    with _lock:
        custom = _state["conditions"].get(name)
    return custom or _BUILTIN_CONDITIONS.get(name)


# ── conditional-trigger persistence ─────────────────────────────────
def _read_conditions() -> list[dict]:
    if not os.path.exists(_CONDITIONS_PATH):
        return []
    try:
        with open(_CONDITIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("triggers"), list):
            return data["triggers"]
    except Exception as e:
        _log.warning("failed to read scheduler_conditions.json: %s", e)
    return []


def _write_conditions(triggers: list[dict]) -> None:
    _atomic_write_json(_CONDITIONS_PATH, triggers)


# ── condition poller thread ─────────────────────────────────────────
def _condition_poller() -> None:
    stop: threading.Event = _state["cond_stop"]  # type: ignore[assignment]
    if stop.wait(_CONDITION_BOOT_DELAY):
        return
    while not stop.is_set():
        try:
            _evaluate_conditions_once()
        except Exception as e:
            _log.exception("condition poller iteration failed: %s", e)
        if stop.wait(_CONDITION_POLL_SECONDS):
            return


def _evaluate_conditions_once() -> None:
    """Re-evaluate every persisted conditional trigger.

    Fires the trigger's action on a False→True transition (rising edge).
    The previous boolean is tracked in ``_state["cond_state"]`` so a long
    "FINISH" state on the Bambu printer doesn't re-fire every 30 s.
    """
    triggers = _read_conditions()
    if not triggers:
        return
    with _lock:
        prev_state: dict[str, bool] = dict(_state["cond_state"])

    fired_any = False
    # Build the next state fresh from THIS sweep's trigger ids so ids for
    # deleted triggers are pruned rather than leaking across every sweep.
    updated_state: dict[str, bool] = {}
    dropped_ids: set[str] = set()

    for trig in triggers:
        try:
            tid     = trig.get("id")
            cond    = trig.get("condition")
            action  = trig.get("action")
            arg     = trig.get("arg") or ""
            chain   = trig.get("chain") or []
            if not tid or not cond or not action:
                continue
            fn = _resolve_condition(cond)
            if fn is None:
                # Keep a known previous value (if any) so re-registering the
                # condition later doesn't reset its edge state mid-uptime.
                if tid in prev_state:
                    updated_state[tid] = prev_state[tid]
                continue
            try:
                value = bool(fn())
            except Exception as e:
                _log.warning("condition %s raised: %s", cond, e)
                if tid in prev_state:
                    updated_state[tid] = prev_state[tid]
                continue
            # Seed an unseen id from its CURRENT value (not False) so a
            # condition that's already true on first observation doesn't
            # spuriously fire a False→True edge on the very first sweep.
            prev = prev_state.get(tid, value)
            updated_state[tid] = value
            if value and not prev:
                # rising edge — fire the action chain
                try:
                    run_action(action, arg, chain=chain if isinstance(chain, list) else None)
                    fired_any = True
                except Exception as e:
                    _log.exception("conditional trigger %s dispatch failed: %s", tid, e)
                # If the trigger was one-shot, mark it for removal; rewrite once at end.
                if trig.get("one_shot"):
                    dropped_ids.add(tid)
        except Exception:
            _log.exception("malformed conditional trigger entry: %r", trig)

    if dropped_ids:
        # Re-read under the lock and drop only the fired one-shots, so a trigger
        # added concurrently during this (action-dispatching) sweep isn't
        # clobbered by writing back this sweep's stale snapshot.
        try:
            with _lock:
                current = _read_conditions()
                _write_conditions([t for t in current
                                   if t.get("id") not in dropped_ids])
        except Exception:
            pass
        # Drop fired one-shots from the edge-state too so it tracks the
        # surviving on-disk trigger set exactly.
        for tid in dropped_ids:
            updated_state.pop(tid, None)

    with _lock:
        _state["cond_state"] = updated_state
    _ = fired_any  # surfaced via fire_now / list_conditions if needed


# ── bootstrap / shutdown ────────────────────────────────────────────
def bootstrap(actions: dict) -> bool:
    """Start the scheduler and condition poller. Idempotent."""
    if not is_available():
        with _lock:
            _state["last_error"] = f"APScheduler unavailable: {_import_error}"
        return False

    with _lock:
        # Re-bind ACTIONS on every call so a reload swaps the live dict in.
        _state["actions"] = actions
        if _state["scheduler"] is not None:
            return True

    aps = _aps_imports()
    os.makedirs(_DATA_DIR, exist_ok=True)

    jobstores: dict[str, Any] = {}
    SQLAlchemyJobStore = aps.get("SQLAlchemyJobStore")
    if SQLAlchemyJobStore is not None:
        try:
            jobstores["default"] = SQLAlchemyJobStore(url=f"sqlite:///{_DB_PATH}")
        except Exception as e:
            _log.warning("SQLAlchemyJobStore failed (%s) — using in-memory store", e)
            jobstores = {}

    BackgroundScheduler = aps["BackgroundScheduler"]
    try:
        # Only pass jobstores when we actually have one configured. APScheduler's
        # _configure() reads jobstores via `config.get("jobstores", {}).items()`,
        # so any kwarg that isn't a real dict (None, etc.) trips an AttributeError
        # on .start(). Omitting the kwarg lets APScheduler install its default
        # in-memory MemoryJobStore.
        scheduler_kwargs: dict[str, Any] = {
            "job_defaults": {
                # Coalesce missed runs — if JARVIS was off when a 7am cron
                # should have fired, we want at most one make-up fire when
                # it boots back up, not one per missed slot.
                "coalesce":          True,
                "max_instances":     1,
                "misfire_grace_time": 60 * 60,  # 1 h grace for missed cron fires
            },
        }
        # Type-check first (defends against None/list/str), then truthiness
        # (skips empty dict), then verify every entry is a real jobstore
        # instance — a dict like {"default": None} would still crash
        # APScheduler downstream when it tries to call methods on the store.
        if (
            isinstance(jobstores, dict)
            and jobstores
            and all(v is not None for v in jobstores.values())
        ):
            scheduler_kwargs["jobstores"] = jobstores
        scheduler = BackgroundScheduler(**scheduler_kwargs)
        scheduler.start()
    except Exception as e:
        with _lock:
            _state["last_error"] = f"BackgroundScheduler.start failed: {type(e).__name__}: {e}"
        _log.exception("scheduler.bootstrap failed")
        return False

    with _lock:
        _state["scheduler"]  = scheduler
        _state["started_at"] = time.time()
        # Spin up the condition poller daemon.
        stop = threading.Event()
        _state["cond_stop"] = stop
        t = threading.Thread(target=_condition_poller,
                             name="scheduler-conditions",
                             daemon=True)
        _state["cond_thread"] = t
        t.start()

    return True


def shutdown(wait: bool = False) -> None:
    with _lock:
        sched = _state.get("scheduler")
        stop  = _state.get("cond_stop")
    if stop is not None:
        try:
            stop.set()
        except Exception:
            pass
    if sched is not None:
        try:
            sched.shutdown(wait=wait)
        except Exception:
            pass
    with _lock:
        _state["scheduler"] = None


# ── job construction ────────────────────────────────────────────────
def _scheduler() -> Any:
    with _lock:
        s = _state.get("scheduler")
    if s is None:
        raise RuntimeError("scheduler not bootstrapped — call core.scheduler.bootstrap(actions) first")
    return s


def _make_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime for ``dt``.

    The scheduler is constructed without an explicit ``timezone`` kwarg (see
    ``bootstrap``), so APScheduler defaults to the host's local zone — and it
    stamps every DateTrigger built from a *naive* datetime with that zone.
    If the scheduler/jobstore ends up resolving to UTC, a naive local
    wall-clock time would therefore fire at the wrong hour. We pin the instant
    by interpreting a naive ``dt`` as local wall-clock and attaching the
    host's local tzinfo via ``astimezone()``. An already-aware datetime is
    returned unchanged."""
    if dt.tzinfo is not None:
        return dt
    # astimezone() on a naive datetime treats it as local wall-clock and
    # attaches the host's local tzinfo — exactly the intent of a reminder
    # phrased in local time.
    return dt.astimezone()


def _job_kwargs(action: str, arg: str, chain: list | None) -> dict:
    """Build the kwargs APScheduler hands to ``run_action`` at fire time."""
    kw: dict[str, Any] = {"action": action, "arg": arg or ""}
    if chain:
        kw["chain"] = list(chain)
    return kw


def _short_id(prefix: str) -> str:
    """Generate a short stable-ish job id when the caller didn't supply one."""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def schedule_cron(
    *,
    action: str,
    arg: str = "",
    job_id: str | None = None,
    chain: list | None = None,
    minute: str | int | None = None,
    hour:   str | int | None = None,
    day:    str | int | None = None,
    month:  str | int | None = None,
    day_of_week: str | int | None = None,
    timezone: str | None = None,
    replace_existing: bool = True,
) -> str:
    """Add a cron-style recurring job.

    Any APScheduler CronTrigger field can be left None to mean "any".  A
    pure "every morning at 8am" job is therefore ``hour=8, minute=0``.
    """
    aps = _aps_imports()
    CronTrigger = aps["CronTrigger"]
    sched = _scheduler()
    trigger = CronTrigger(
        minute=minute, hour=hour, day=day, month=month,
        day_of_week=day_of_week, timezone=timezone,
    )
    jid = job_id or _short_id("cron")
    sched.add_job(
        run_action,
        trigger=trigger,
        kwargs=_job_kwargs(action, arg, chain),
        id=jid,
        replace_existing=replace_existing,
        name=f"cron:{action}",
    )
    return jid


def schedule_interval(
    *,
    action: str,
    arg: str = "",
    job_id: str | None = None,
    chain: list | None = None,
    seconds: float = 0,
    minutes: float = 0,
    hours:   float = 0,
    start_date: datetime | None = None,
    replace_existing: bool = True,
) -> str:
    """Add an interval-style recurring job."""
    aps = _aps_imports()
    IntervalTrigger = aps["IntervalTrigger"]
    if seconds == 0 and minutes == 0 and hours == 0:
        raise ValueError("schedule_interval requires at least one of seconds/minutes/hours > 0")
    sched = _scheduler()
    trigger = IntervalTrigger(
        seconds=seconds, minutes=minutes, hours=hours,
        start_date=start_date,
    )
    jid = job_id or _short_id("intv")
    sched.add_job(
        run_action,
        trigger=trigger,
        kwargs=_job_kwargs(action, arg, chain),
        id=jid,
        replace_existing=replace_existing,
        name=f"interval:{action}",
    )
    return jid


def schedule_once(
    *,
    action: str,
    run_at: datetime | float,
    arg: str = "",
    job_id: str | None = None,
    chain: list | None = None,
    replace_existing: bool = True,
) -> str:
    """Add a one-shot job that fires at ``run_at`` and then expires."""
    aps = _aps_imports()
    DateTrigger = aps["DateTrigger"]
    sched = _scheduler()
    if isinstance(run_at, (int, float)):
        when = datetime.fromtimestamp(float(run_at))
    else:
        when = run_at
    # Pin the instant to a concrete tz. A naive datetime here would be
    # localised by APScheduler against the scheduler's configured zone, so a
    # local wall-clock reminder fires at the wrong hour if that zone is UTC.
    when = _make_aware(when)
    trigger = DateTrigger(run_date=when)
    jid = job_id or _short_id("once")
    sched.add_job(
        run_action,
        trigger=trigger,
        kwargs=_job_kwargs(action, arg, chain),
        id=jid,
        replace_existing=replace_existing,
        name=f"once:{action}",
    )
    return jid


def schedule_when(
    *,
    name: str,
    condition: str,
    action: str,
    arg: str = "",
    chain: list | None = None,
    one_shot: bool = False,
) -> str:
    """Add a conditional trigger that fires when ``condition`` flips True.

    Conditional triggers persist to ``data/scheduler_conditions.json``
    rather than the SQLite jobstore, because APScheduler doesn't model
    predicate-based triggers natively.  The condition poller thread (see
    ``_condition_poller``) does the rising-edge detection.
    """
    if not condition or condition not in available_conditions():
        raise ValueError(
            f"unknown condition '{condition}'. "
            f"Available: {', '.join(available_conditions())}"
        )
    # Hold _lock across the read-modify-write so a concurrent schedule_when /
    # cancel_job pass can't interleave and clobber a just-added trigger (RLock,
    # so the cond_state seed below re-enters safely).
    with _lock:
        triggers = _read_conditions()
        # Replace by id if it already exists, otherwise append.
        triggers = [t for t in triggers if t.get("id") != name]
        triggers.append({
            "id":        name,
            "condition": condition,
            "action":    action,
            "arg":       arg or "",
            "chain":     list(chain) if chain else [],
            "one_shot":  bool(one_shot),
            "created":   time.time(),
        })
        _write_conditions(triggers)
    # Seed cond_state so the trigger doesn't fire immediately on its
    # initial value (e.g. a printer already in FINISH state at bootstrap).
    with _lock:
        try:
            fn = _resolve_condition(condition)
            _state["cond_state"][name] = bool(fn()) if fn else False
        except Exception:
            _state["cond_state"][name] = False
    return name


# ── listing / inspection ────────────────────────────────────────────
def _describe_trigger(trigger: Any) -> tuple[str, str]:
    """Return (kind, human_summary) for a job's trigger."""
    name = type(trigger).__name__
    if name == "CronTrigger":
        try:
            fields = ", ".join(f"{f.name}={f}" for f in trigger.fields if not f.is_default)
            return "cron", fields or "cron(*)"
        except Exception:
            return "cron", str(trigger)
    if name == "IntervalTrigger":
        try:
            return "interval", f"every {trigger.interval}"
        except Exception:
            return "interval", str(trigger)
    if name == "DateTrigger":
        try:
            return "date", f"at {trigger.run_date}"
        except Exception:
            return "date", str(trigger)
    return name.lower(), str(trigger)


def _job_summary(job: Any) -> dict:
    kind, summary = _describe_trigger(job.trigger)
    kw = getattr(job, "kwargs", {}) or {}
    next_run = getattr(job, "next_run_time", None)
    return {
        "id":       job.id,
        "kind":     kind,
        "trigger":  summary,
        "action":   kw.get("action"),
        "arg":      kw.get("arg") or "",
        "chain":    kw.get("chain") or [],
        "next_run": next_run.isoformat() if next_run else None,
        "name":     job.name,
    }


def list_jobs() -> list[dict]:
    """List every APScheduler-managed job (cron/interval/date)."""
    with _lock:
        sched = _state.get("scheduler")
    if sched is None:
        return []
    try:
        return [_job_summary(j) for j in sched.get_jobs()]
    except Exception as e:
        _log.warning("list_jobs failed: %s", e)
        return []


def list_conditions() -> list[dict]:
    """List every conditional trigger persisted to disk."""
    triggers = _read_conditions()
    with _lock:
        state = dict(_state["cond_state"])
    return [
        {
            **t,
            "current_value": state.get(t.get("id"), None),
        }
        for t in triggers
    ]


def cancel_job(job_id: str) -> bool:
    """Cancel a cron/interval/date OR a conditional trigger by id."""
    removed = False
    with _lock:
        sched = _state.get("scheduler")
    if sched is not None:
        try:
            sched.remove_job(job_id)
            removed = True
        except Exception:
            # not in the APScheduler store — fall through to conditions
            pass

    with _lock:
        triggers = _read_conditions()
        new_triggers = [t for t in triggers if t.get("id") != job_id]
        if len(new_triggers) != len(triggers):
            try:
                _write_conditions(new_triggers)
                removed = True
            except Exception as e:
                _log.warning("cancel_job: failed to rewrite conditions: %s", e)

    with _lock:
        _state["cond_state"].pop(job_id, None)
    return removed


def fire_now(job_id: str) -> str:
    """Dispatch the action of an existing job immediately, regardless of
    its schedule.  Useful for testing a freshly-created cron entry
    without waiting for the next trigger time."""
    with _lock:
        sched = _state.get("scheduler")
    if sched is not None:
        try:
            job = sched.get_job(job_id)
        except Exception:
            job = None
        if job is not None:
            kw = getattr(job, "kwargs", {}) or {}
            return run_action(
                kw.get("action") or "",
                kw.get("arg")    or "",
                chain=kw.get("chain"),
            )

    # Conditional trigger fallback
    for trig in _read_conditions():
        if trig.get("id") == job_id:
            return run_action(
                trig.get("action") or "",
                trig.get("arg")    or "",
                chain=trig.get("chain"),
            )
    return f"job '{job_id}' not found"


def status() -> dict:
    """One-call health snapshot used by the schedule_status action."""
    with _lock:
        s = _state.get("scheduler")
        started_at = _state.get("started_at")
        last_error = _state.get("last_error")
    jobs       = list_jobs()
    conditions = list_conditions()
    return {
        "available":         is_available(),
        "running":           s is not None and getattr(s, "running", False),
        "started_at":        started_at,
        "uptime_seconds":    (time.time() - started_at) if started_at else 0,
        "job_count":         len(jobs),
        "condition_count":   len(conditions),
        "registered_conditions": available_conditions(),
        "last_error":        last_error,
    }


# ── time-string parsing helpers ─────────────────────────────────────
def parse_clock(s: str) -> tuple[int, int] | None:
    """Parse '8am', '8:30 am', '20:15', '8 pm' → (hour, minute) on 24h clock.

    Returns None on parse failure so callers can surface a usage hint.
    """
    import re
    if not s:
        return None
    txt = s.strip().lower().replace(".", "")
    m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?$", txt)
    if not m:
        return None
    h = int(m.group(1))
    mm = int(m.group(2)) if m.group(2) else 0
    ampm = m.group(3)
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= mm <= 59):
        return None
    return h, mm


_WEEKDAY_TO_CRON = {
    "mon": "mon", "monday": "mon",
    "tue": "tue", "tues": "tue", "tuesday": "tue",
    "wed": "wed", "weds": "wed", "wednesday": "wed",
    "thu": "thu", "thur": "thu", "thurs": "thu", "thursday": "thu",
    "fri": "fri", "friday": "fri",
    "sat": "sat", "saturday": "sat",
    "sun": "sun", "sunday": "sun",
}


def parse_dow(token: str) -> str | None:
    """'monday' / 'mon' / 'weekdays' / 'weekends' → APScheduler day_of_week."""
    if not token:
        return None
    t = token.strip().lower()
    if t in ("daily", "everyday", "every day", "any"):
        return None
    if t in ("weekday", "weekdays"):
        return "mon-fri"
    if t in ("weekend", "weekends"):
        return "sat,sun"
    parts = [p.strip() for p in t.replace("/", ",").replace(" and ", ",").split(",") if p.strip()]
    mapped: list[str] = []
    for p in parts:
        if p in _WEEKDAY_TO_CRON:
            mapped.append(_WEEKDAY_TO_CRON[p])
    if not mapped:
        return None
    return ",".join(mapped)


def parse_every(token: str) -> dict | None:
    """'30 minutes' / '2 hours' / '45 seconds' → interval kwargs."""
    import re
    if not token:
        return None
    m = re.match(
        r"^\s*(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)\s*$",
        token, re.IGNORECASE,
    )
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("sec"):
        return {"seconds": n}
    if unit.startswith("min"):
        return {"minutes": n}
    if unit.startswith(("hr", "hour")):
        return {"hours": n}
    return None  # pragma: no cover - unreachable: the regex above only admits sec/min/hr/hour units, all handled above


def parse_when(token: str) -> datetime | None:
    """Best-effort parse of a one-shot run timestamp.

    Accepts:
      * ISO 8601: '2026-06-01T08:00:00'
      * Date + time: '2026-06-01 08:00'
      * 'tomorrow 8am'
      * 'in 30 minutes' / 'in 2 hours'
    """
    import re
    if not token:
        return None
    txt = token.strip().lower()

    # All returned datetimes are made timezone-aware (local zone) via
    # _make_aware so APScheduler doesn't re-interpret a naive local time
    # against a UTC scheduler/jobstore and fire at the wrong hour.
    now = datetime.now().astimezone()

    # in <n> <unit>
    m = re.match(r"^in\s+(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)$", txt)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        seconds = n
        if unit.startswith("min"):
            seconds = n * 60
        elif unit.startswith(("hr", "hour")):
            seconds = n * 3600
        return now + timedelta(seconds=seconds)

    # tomorrow [at] <clock>
    m = re.match(r"^tomorrow(?:\s+at)?\s+(.+)$", txt)
    if m:
        clock = parse_clock(m.group(1))
        if clock is not None:
            h, mm = clock
            base = (now + timedelta(days=1)).replace(hour=h, minute=mm, second=0, microsecond=0)
            return base

    # today [at] <clock>
    m = re.match(r"^today(?:\s+at)?\s+(.+)$", txt)
    if m:
        clock = parse_clock(m.group(1))
        if clock is not None:
            h, mm = clock
            base = now.replace(hour=h, minute=mm, second=0, microsecond=0)
            # A "today at <past-time>" would otherwise schedule in the past and
            # an immediate-misfire would fire it instantly. Roll to tomorrow.
            if base <= now:
                base = base + timedelta(days=1)
            return base

    # ISO / datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return _make_aware(datetime.strptime(token.strip(), fmt))
        except ValueError:
            continue
    return None
