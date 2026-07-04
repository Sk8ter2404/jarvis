"""
Always-on diagnostic daemons.

Four background threads run continuously while JARVIS is up:

    1. SelfDiagDaemon    — invokes skills/self_diagnostic.run_diagnostic
                           every ~5 min (4-min dedup throttle).
    2. CrashWatcherDaemon — polls Windows Event Log every 30 s for
                            APPCRASH events whose faulting application is
                            python(w).exe AND whose path resolves to a
                            JARVIS-related cwd. Queues a [crash-watch]
                            task to jarvis_todo.md.
    3. DeepAuditDaemon   — cost-throttled. Fires after every
                            DEEP_AUDIT_BATCH_SIZE pipeline task completions
                            (counted from data/pipeline_runs.jsonl). Also
                            fires when any source file's mtime is newer than
                            the last-audit timestamp AND >1 h has elapsed.
                            Each run spawns an Anthropic Messages call asking
                            it to inspect the N most-recently-modified files
                            and return ranked JSON findings, which become
                            [deep-audit] tasks. Hard caps: max 5 calls / h,
                            $JARVIS_DEEP_AUDIT_BUDGET_USD per day (default 5).
    4. AnomalyWatchDaemon — local-only (no LLM, no shell-out). Every ~90 s
                            tails the freshest session log + checks for
                            stale hud_state.json mtime + reads
                            data/boot_failures.jsonl. Queues [anomaly]
                            tasks for: repeated single-skill failures,
                            unhandled exception bursts, stuck main loop,
                            and recent boot failures. 30-min dedup window
                            per signature to avoid todo flooding.

The four daemons are coordinated by module-level singletons and share a
single on-disk state file (data/diagnostic_daemons.json) so pauses, last-run
timestamps, and daily budget survive restarts.

Lifecycle
---------
    start_diagnostic_daemons()  — call once from bobert_companion.main()
                                  AFTER load_skills(). Idempotent.
    stop_diagnostic_daemons()   — call from _act_shutdown_jarvis. Signals
                                  each thread to wind down; returns after
                                  a short bounded join.
    pause_diagnostics()         — set the global paused flag (persists).
    resume_diagnostics()        — clear it.
    diagnostic_daemon_status()  — dict snapshot of last-run times, budget
                                  remaining, pending findings count.

Failure policy: every external operation (event log probe, Anthropic API
call, file write) is wrapped so a single failure cannot crash the daemon.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from typing import Any

from core.atomic_io import _atomic_write_json

# ──────────────────────────── module paths ───────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
STATE_FILE = os.path.join(DATA_DIR, "diagnostic_daemons.json")
TODO_FILE = os.path.join(PROJECT_DIR, "jarvis_todo.md")
PIPELINE_RUNS_FILE = os.path.join(DATA_DIR, "pipeline_runs.jsonl")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
BOOT_FAILURES_FILE = os.path.join(DATA_DIR, "boot_failures.jsonl")
HUD_STATE_FILE = os.path.join(PROJECT_DIR, "hud_state.json")

# ──────────────────────────── tuning ─────────────────────────────────

SELF_DIAG_INTERVAL_S = 300            # 5 minutes
SELF_DIAG_DEDUP_MIN_GAP_S = 240       # skip if <4 min since last run

CRASH_POLL_INTERVAL_S = 30
CRASH_FAULTING_APP_RE = re.compile(r"python(w)?\.exe", re.IGNORECASE)

DEEP_AUDIT_BATCH_SIZE = 10
DEEP_AUDIT_AGGRESSIVE_GAP_S = 3600    # 1h: aggressive mtime-triggered fallback
DEEP_AUDIT_FILES_PER_RUN = 8          # files passed to the auditor model
DEEP_AUDIT_MAX_RUNS_PER_HOUR = 5
# Default daily spend ceiling for the deep-audit daemon. Sourced from
# core/config.py (Settings GUI knob) so a saved override applies; the
# JARVIS_DEEP_AUDIT_BUDGET_USD env var still takes priority in
# _deep_audit_budget_usd(). Falls back to 5.0 if core.config is unimportable.
try:
    from core.config import DEEP_AUDIT_BUDGET_USD as DEEP_AUDIT_DEFAULT_BUDGET_USD
except Exception:
    DEEP_AUDIT_DEFAULT_BUDGET_USD = 5.0
DEEP_AUDIT_MODEL = os.environ.get(
    "JARVIS_DEEP_AUDIT_MODEL", "claude-sonnet-4-6"
)
# Sonnet 4.6: $3 in / $15 out per million. A typical audit run sends
# ~30 KB of code (~10k tokens) and gets back ~2 KB (~500 tokens) of JSON.
# Estimated cost per run ≈ $0.04. We use this estimate for the budget
# counter — actual usage may differ; the daily cap is a safety net not
# an accounting tool.
DEEP_AUDIT_ESTIMATED_COST_PER_RUN_USD = 0.05

THREAD_JOIN_TIMEOUT_S = 5.0

# ── anomaly-watch tuning ──
# Watch session logs + boot_failures.jsonl + hud_state mtime for signs that
# JARVIS is sick: skills failing repeatedly, tracebacks piling up, main loop
# stuck, or recent boot crashes. Thresholds are deliberately conservative —
# a slow disk, a network blip, or a one-off transient must NOT flood
# jarvis_todo with [anomaly] entries. Five prior tester rejections all stemmed
# from too-eager flagging, so we err on the side of silence.
ANOMALY_INITIAL_DELAY_S = 120          # let boot settle before first sweep
ANOMALY_POLL_INTERVAL_S = 90           # check every 90s
ANOMALY_WINDOW_S = 600                 # 10-min window for repeated failures
ANOMALY_FAILURE_THRESHOLD = 5          # same skill failing N+ times in tail
ANOMALY_EXCEPTION_THRESHOLD = 6        # N+ tracebacks in tail
ANOMALY_STUCK_LOOP_THRESHOLD_S = 300   # hud_state stale N+ s = loop stuck
ANOMALY_STUCK_LOOP_MIN_CONSECUTIVE = 2 # require N consecutive stale sweeps
ANOMALY_DEDUP_GAP_S = 1800             # 30-min cooldown per signature
ANOMALY_LOG_TAIL_BYTES = 64 * 1024     # tail size per scan (cheap)
ANOMALY_MAX_QUEUED_PER_SWEEP = 1       # hard cap: avoid bursting the todo

# Skills that frequently fail for legitimate transient reasons (network blip,
# device temporarily busy, external service down). One-off failures from
# these are *expected* and should not be reported as anomalies — they're
# only flagged if they spike well above the normal floor.
_TRANSIENT_SKILL_NAMES: frozenset[str] = frozenset({
    "weather", "weather_check", "network_check", "internet_check",
    "ping", "dns_lookup", "calendar_fetch", "bambu_status",
    "media_playback", "apple_music",
})

# Substrings indicating an action failure was a known transient — DNS hiccup,
# COM brief unavailability, request timeout. These get filtered out of the
# count BEFORE the threshold is applied so a flaky external dependency
# doesn't generate alert fatigue.
_TRANSIENT_ERROR_TOKENS: tuple[str, ...] = (
    "getaddrinfo failed",
    "name or service not known",
    "temporary failure in name resolution",
    "no address associated",
    "winerror 10060",      # connection timed out
    "winerror 10061",      # connection refused
    "winerror 10054",      # connection reset
    "winerror 10065",      # host unreachable
    "rpc server is unavailable",
    "rpc_e_server_died",
    "co_e_server_exec_failure",
    "comerror",
    "the rpc server is too busy",
    "timed out",
    "timeout",
    "read timed out",
)

# ──────────────────────────── shared state ───────────────────────────

_state_lock = threading.RLock()
_stop_event = threading.Event()
_threads: list[threading.Thread] = []
_started = False
_lock_threads = threading.Lock()


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts or _now()))


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime(_now()))


# ──────────────────────────── persistence ────────────────────────────

_DEFAULT_STATE: dict[str, Any] = {
    "paused": False,
    "self_diag": {
        "last_run_ts": 0.0, "last_run_iso": None, "runs": 0,
        "alive_ts": 0.0,
    },
    "crash_watch": {
        "last_poll_ts": 0.0,
        "last_seen_record_id": 0,
        "detections": 0,
        "alive_ts": 0.0,
    },
    "deep_audit": {
        "last_run_ts": 0.0,
        "last_run_iso": None,
        "runs": 0,
        "last_pipeline_event_count": 0,
        "hourly_window_start_ts": 0.0,
        "hourly_window_count": 0,
        "daily_budget_date": None,
        "daily_budget_spent_usd": 0.0,
        "pending_findings": 0,
        "alive_ts": 0.0,
    },
    "anomaly_watch": {
        "last_poll_ts": 0.0,
        "last_poll_iso": None,
        "last_boot_failure_offset": 0,    # byte offset into boot_failures.jsonl
        "queued_signatures": {},          # signature → ts (for dedup)
        "detections": 0,
        "stuck_loop_consecutive_misses": 0,  # consecutive sweeps with stale hud
        "alive_ts": 0.0,
    },
}


def _read_state() -> dict[str, Any]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return json.loads(json.dumps(_DEFAULT_STATE))
    except Exception:
        return json.loads(json.dumps(_DEFAULT_STATE))
    # Backfill any missing keys so older state files still load.
    merged = json.loads(json.dumps(_DEFAULT_STATE))
    for k, v in raw.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def _write_state(state: dict[str, Any]) -> None:
    # Route through the shared atomic writer (unique mkstemp tempfile + fsync +
    # a WinError-5/PermissionError retry on the final replace). The old fixed
    # "<state>.json.tmp" name raced across the four daemon threads and fired
    # live: "[diag-daemons] state write failed: [WinError 5] Access is denied:
    # ...diagnostic_daemons.json.tmp -> ...json".
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        _atomic_write_json(STATE_FILE, state, indent=2)
    except Exception as e:
        print(f"  [diag-daemons] state write failed: {e}")


def _update_state(mutator) -> dict[str, Any]:
    with _state_lock:
        state = _read_state()
        mutator(state)
        _write_state(state)
        return state


# ──────────────────────────── todo writer ────────────────────────────

_todo_lock = threading.Lock()


def _existing_todo_text() -> str:
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _append_todo_task(body: str, tag: str) -> bool:
    """Append a single task line to jarvis_todo.md.

    Dedup: if a line containing the same body already exists in the file, do
    nothing and return False. Returns True when a new task is appended.
    """
    body = body.strip()
    if not body:
        return False
    with _todo_lock:
        existing = _existing_todo_text()
        if body in existing:
            return False
        slug = f"{_today()} {tag}-{int(_now())}"
        line = f"- [ ] **{slug}** {body}\n"
        try:
            with open(TODO_FILE, "a", encoding="utf-8") as f:
                # Ensure separation from prior content.
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(line)
        except Exception as e:
            print(f"  [diag-daemons] todo append failed: {e}")
            return False
    return True


# ──────────────────────────── self-diag daemon ───────────────────────

def _run_self_diag_once() -> None:
    try:
        # Lazy import so daemon module can load even when self_diagnostic
        # hasn't been registered yet.
        from skills import self_diagnostic  # type: ignore
    except Exception as e:
        print(f"  [diag-daemons] self_diagnostic import failed: {e}")
        return
    try:
        self_diagnostic.run_diagnostic("")
    except Exception:
        print("  [diag-daemons] self_diagnostic.run_diagnostic raised:")
        traceback.print_exc()


def _self_diag_loop() -> None:
    print("  [diag-daemons] self-diag daemon online "
          f"(interval={SELF_DIAG_INTERVAL_S}s, "
          f"dedup_gap={SELF_DIAG_DEDUP_MIN_GAP_S}s)")
    # Don't fire immediately on boot — give self_diagnostic.register() its
    # own ON_BOOT_DELAY first sweep, then we take over.
    if _stop_event.wait(SELF_DIAG_INTERVAL_S):
        return
    while not _stop_event.is_set():
        try:
            _update_state(lambda s: s["self_diag"].update({"alive_ts": _now()}))
            state = _read_state()
            if state.get("paused"):
                if _stop_event.wait(30):
                    return
                continue
            last = float(state.get("self_diag", {}).get("last_run_ts", 0.0))
            gap = _now() - last
            if gap < SELF_DIAG_DEDUP_MIN_GAP_S:
                # Some other path (manual /run_diagnostic, scheduler) ran it
                # recently. Sleep until the dedup window clears.
                sleep_for = max(5.0, SELF_DIAG_DEDUP_MIN_GAP_S - gap)
                if _stop_event.wait(sleep_for):
                    return
                continue
            _run_self_diag_once()
            _update_state(lambda s: s["self_diag"].update({
                "last_run_ts": _now(),
                "last_run_iso": _iso(),
                "runs": int(s["self_diag"].get("runs", 0)) + 1,
            }))
            if _stop_event.wait(SELF_DIAG_INTERVAL_S):
                return
        except Exception:
            print("  [diag-daemons] self-diag loop iteration raised:")
            traceback.print_exc()
            if _stop_event.wait(SELF_DIAG_INTERVAL_S):
                return


# ─────────────────────────── crash watcher ───────────────────────────

def _import_win_event_api():
    try:
        import win32evtlog          # type: ignore
        return win32evtlog
    except Exception as e:
        print(f"  [diag-daemons] win32evtlog unavailable ({e}); "
              "crash watcher disabled")
        return None


def _path_is_jarvis(path_or_text: str) -> bool:
    if not path_or_text:
        return False
    norm = path_or_text.replace("/", "\\").lower()
    return (
        "\\jarvis" in norm
        or norm.endswith("jarvis")
        or "bobert_companion" in norm
        or PROJECT_DIR.lower() in norm
    )


def _latest_session_log_tail(n_lines: int = 30) -> str:
    try:
        files = [
            os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR)
            if f.startswith("session_") and f.endswith(".log")
        ]
    except FileNotFoundError:
        return ""
    if not files:
        return ""
    latest = max(files, key=lambda p: os.path.getmtime(p))
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = "".join(lines[-n_lines:]).strip().replace("\n", " | ")
        # Cap so a single huge tail can't bloat the todo file.
        return tail[:1500]
    except Exception:
        return ""


def _scan_event_log_for_crashes(win32evtlog, last_record_id: int) -> tuple[list[dict], int]:
    """Read Application log for new APPCRASH events since last_record_id.

    Returns (list_of_crash_dicts, max_record_id_seen).
    """
    hits: list[dict] = []
    # Track the TRUE max record id independently of the filter bound. Seeding
    # max_seen from last_record_id was a permanent-blindness bug: the seed call
    # passes last_record_id=10**12 to skip historical crashes, but because
    # events are read newest-first the early-return below fires on the first
    # event before max_seen can drop to the real head — so it returned 10**12,
    # which _crash_watch_loop then persisted as last_seen_record_id. Every later
    # poll (last_seen=10**12) filtered out every real event, so no APPCRASH was
    # ever surfaced. Starting at 0 lets the newest real rec_id win.
    max_seen = 0
    try:
        handle = win32evtlog.OpenEventLog(None, "Application")
    except Exception as e:
        print(f"  [diag-daemons] OpenEventLog failed: {e}")
        return hits, max_seen
    flags = (
        win32evtlog.EVENTLOG_BACKWARDS_READ |
        win32evtlog.EVENTLOG_SEQUENTIAL_READ
    )
    try:
        # Bound the scan: don't read indefinitely far back when we have no
        # last_record_id baseline (first run). Cap at ~200 records.
        scanned = 0
        SCAN_CAP = 200
        while scanned < SCAN_CAP:
            try:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
            except Exception:
                break
            if not events:
                break
            for ev in events:
                scanned += 1
                try:
                    rec_id = int(getattr(ev, "RecordNumber", 0) or 0)
                except Exception:
                    rec_id = 0
                if rec_id and rec_id > max_seen:
                    max_seen = rec_id
                # Stop once we go past the baseline (events come newest-first).
                if last_record_id and rec_id and rec_id <= last_record_id:
                    return hits, max_seen
                source = (getattr(ev, "SourceName", "") or "").strip()
                if source.lower() != "application error":
                    continue
                strings = list(getattr(ev, "StringInserts", None) or [])
                # APPCRASH event 1000 strings: [0]=app name, [1]=app version,
                # [2]=app timestamp, [3]=module, [4]=module version, [5]=mod
                # timestamp, [6]=exception code, [7]=fault offset, [8]=process
                # id, [9]=app start time, [10]=app path, [11]=mod path, ...
                app_name = strings[0] if strings else ""
                if not CRASH_FAULTING_APP_RE.search(app_name):
                    continue
                # Look at any string for a JARVIS path hint.
                blob = " ".join(strings)
                if not _path_is_jarvis(blob):
                    continue
                offset = strings[7] if len(strings) > 7 else "?"
                ts = getattr(ev, "TimeGenerated", None)
                ts_str = ts.Format() if ts and hasattr(ts, "Format") else _iso()
                hits.append({
                    "ts": ts_str,
                    "app": app_name,
                    "offset": offset,
                    "record_id": rec_id,
                })
                if scanned >= SCAN_CAP:
                    break
            if scanned >= SCAN_CAP:
                break
    finally:
        try:
            win32evtlog.CloseEventLog(handle)
        except Exception:
            pass
    return hits, max_seen


def _crash_watch_loop() -> None:
    api = _import_win_event_api()
    if api is None:
        return
    print(f"  [diag-daemons] crash watcher online (interval={CRASH_POLL_INTERVAL_S}s)")
    # On first run, seed last_seen_record_id to the current head so we don't
    # spam the queue with historical crashes.
    state = _read_state()
    seed_id = int(state.get("crash_watch", {}).get("last_seen_record_id", 0))
    if seed_id == 0:
        _, head = _scan_event_log_for_crashes(api, last_record_id=10**12)
        _update_state(lambda s: s["crash_watch"].update({
            "last_seen_record_id": head,
            "last_poll_ts": _now(),
        }))

    while not _stop_event.is_set():
        try:
            _update_state(lambda s: s["crash_watch"].update({"alive_ts": _now()}))
            state = _read_state()
            if state.get("paused"):
                if _stop_event.wait(CRASH_POLL_INTERVAL_S):
                    return
                continue
            last_seen = int(state.get("crash_watch", {}).get("last_seen_record_id", 0))
            try:
                crashes, new_head = _scan_event_log_for_crashes(api, last_seen)
            except Exception:
                print("  [diag-daemons] crash watcher scan raised:")
                traceback.print_exc()
                if _stop_event.wait(CRASH_POLL_INTERVAL_S):
                    return
                continue
            if new_head > last_seen:
                _update_state(lambda s: s["crash_watch"].update({
                    "last_seen_record_id": new_head,
                    "last_poll_ts": _now(),
                }))
            for crash in crashes:
                tail = _latest_session_log_tail(30)
                body = (
                    f"[crash-watch] APPCRASH at {crash['ts']}, "
                    f"offset {crash['offset']}, last log: {tail}"
                )
                queued = _append_todo_task(body, tag="crash-watch")
                if queued:
                    _update_state(lambda s: s["crash_watch"].update({
                        "detections": int(s["crash_watch"].get("detections", 0)) + 1,
                    }))
                    print(f"  [diag-daemons] queued crash-watch task: {crash['app']} "
                          f"offset={crash['offset']}")
            if _stop_event.wait(CRASH_POLL_INTERVAL_S):
                return
        except Exception:
            print("  [diag-daemons] crash-watch loop iteration raised:")
            traceback.print_exc()
            if _stop_event.wait(CRASH_POLL_INTERVAL_S):
                return


# ──────────────────────────── deep audit ─────────────────────────────

def _deep_audit_budget_usd() -> float:
    try:
        v = float(os.environ.get("JARVIS_DEEP_AUDIT_BUDGET_USD",
                                 DEEP_AUDIT_DEFAULT_BUDGET_USD))
        return max(0.0, v)
    except Exception:
        return DEEP_AUDIT_DEFAULT_BUDGET_USD


_count_cache_lock = threading.Lock()
_count_cache: dict[str, Any] = {"mtime": 0.0, "size": -1, "count": 0}


def _count_pipeline_task_completions() -> int:
    """Count successful task completions in data/pipeline_runs.jsonl.

    Treats any line whose payload includes 'verdict':'approve' OR
    'event':'task_completed' as a completion. Best-effort — corrupt lines
    are skipped.

    Caches (mtime, size) → count so subsequent calls with an unchanged
    file return instantly. The 120s deep-audit tick would otherwise re-parse
    the whole jsonl every iteration (~2s at 10k tasks).
    """
    if not os.path.exists(PIPELINE_RUNS_FILE):
        return 0
    try:
        st = os.stat(PIPELINE_RUNS_FILE)
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        return 0
    with _count_cache_lock:
        if (_count_cache["mtime"] == mtime
                and _count_cache["size"] == size):
            return int(_count_cache["count"])
    n = 0
    try:
        with open(PIPELINE_RUNS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                verdict = (obj.get("verdict") or "").lower()
                event = (obj.get("event") or "").lower()
                if verdict in ("approve", "approve_with_warnings"):
                    n += 1
                elif event in ("task_completed", "task_done", "approved"):
                    n += 1
    except Exception:
        return n
    with _count_cache_lock:
        _count_cache["mtime"] = mtime
        _count_cache["size"] = size
        _count_cache["count"] = n
    return n


def _recently_modified_source_files(limit: int) -> list[str]:
    """Return up to `limit` *.py files in the project root + key dirs,
    sorted newest-mtime first. Skips backups/, logs/, data/, venv-ish dirs."""
    excluded_parts = {
        "backups", "logs", "data", "__pycache__", ".git", "venv", ".venv",
        "memory",
    }
    candidates: list[tuple[float, str]] = []
    for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
        rel = os.path.relpath(dirpath, PROJECT_DIR)
        parts = set(rel.split(os.sep))
        if parts & excluded_parts:
            # Prune subtree.
            dirnames[:] = []  # pragma: no cover - unreachable: the dirnames filter on the next iteration's parent already strips excluded child dirs before os.walk descends, so an excluded dir is never yielded as dirpath
            continue  # pragma: no cover - see above
        dirnames[:] = [d for d in dirnames if d not in excluded_parts]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            candidates.append((mtime, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:limit]]


def _newest_source_mtime() -> float:
    files = _recently_modified_source_files(1)
    if not files:
        return 0.0
    try:
        return os.path.getmtime(files[0])
    except Exception:
        return 0.0


def _deep_audit_due(state: dict) -> tuple[bool, str]:
    """Return (should_fire, reason)."""
    audit = state.get("deep_audit", {})
    completed = _count_pipeline_task_completions()
    last_count = int(audit.get("last_pipeline_event_count", 0))
    if completed - last_count >= DEEP_AUDIT_BATCH_SIZE:
        return True, f"batch_of_{DEEP_AUDIT_BATCH_SIZE}"
    last_audit_ts = float(audit.get("last_run_ts", 0.0))
    newest = _newest_source_mtime()
    if (
        newest > last_audit_ts
        and (_now() - last_audit_ts) > DEEP_AUDIT_AGGRESSIVE_GAP_S
    ):
        return True, "aggressive_mtime_fallback"
    return False, ""


def _deep_audit_budget_ok(state: dict) -> tuple[bool, str]:
    audit = state.get("deep_audit", {})
    # Hourly cap.
    window_start = float(audit.get("hourly_window_start_ts", 0.0))
    window_count = int(audit.get("hourly_window_count", 0))
    if _now() - window_start > 3600:
        window_start = _now()
        window_count = 0
        _update_state(lambda s: s["deep_audit"].update({
            "hourly_window_start_ts": window_start,
            "hourly_window_count": 0,
        }))
    if window_count >= DEEP_AUDIT_MAX_RUNS_PER_HOUR:
        return False, "hourly_cap"
    # Daily $$ cap.
    today = _today()
    if audit.get("daily_budget_date") != today:
        _update_state(lambda s: s["deep_audit"].update({
            "daily_budget_date": today,
            "daily_budget_spent_usd": 0.0,
        }))
        spent = 0.0
    else:
        spent = float(audit.get("daily_budget_spent_usd", 0.0))
    if spent + DEEP_AUDIT_ESTIMATED_COST_PER_RUN_USD > _deep_audit_budget_usd():
        return False, "daily_budget_exhausted"
    return True, "ok"


_DEEP_AUDIT_PROMPT_TEMPLATE = """\
You are a senior code auditor inspecting the most recently modified files in
the JARVIS PC companion (Python, Windows). Find issues a static linter would
MISS — focus on:

  * Native-binding races (PortAudio threads, COM apartment-init, OpenCV
    capture handles, CUDA stream releases).
  * Thread-safety holes (locks held across blocking I/O, shared mutable
    state, leaked atexit handlers, daemon threads that never wind down).
  * Dependency leaks (file handles, subprocesses, sockets not closed on the
    error path).
  * Regression risk: behaviour changes that other JARVIS subsystems quietly
    depend on (e.g. a renamed action key, a removed broadcaster).

Return STRICT JSON ONLY (no prose, no markdown fences):

{{
  "findings": [
    {{
      "rank": 1,                  // 1 = most severe
      "file": "core/foo.py",
      "line": 123,                // best estimate, 0 if unknown
      "category": "thread-safety",
      "summary": "one sentence describing the bug",
      "fix_hint": "one sentence on how to fix it"
    }}
  ]
}}

If no real issues, return {{"findings": []}}.

Files to inspect:

{file_blob}
"""


def _build_audit_file_blob(paths: list[str], per_file_byte_cap: int = 12000) -> str:
    pieces: list[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            content = f"<<unable to read: {e}>>"
        if len(content) > per_file_byte_cap:
            content = content[:per_file_byte_cap] + "\n# ... [truncated]\n"
        rel = os.path.relpath(p, PROJECT_DIR)
        pieces.append(f"=== FILE: {rel} ===\n{content}\n")
    return "\n".join(pieces)


def _call_anthropic_auditor(prompt: str) -> str | None:
    """Invoke the Anthropic Messages API as a one-shot research-agent.

    Uses the in-process anthropic SDK (NOT a subprocess) so we stay aligned
    with the 'do not shell out' constraint. Returns the raw text or None on
    any failure.
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:
        print(f"  [diag-daemons] anthropic SDK unavailable: {e}")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [diag-daemons] ANTHROPIC_API_KEY missing — skipping audit")
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=DEEP_AUDIT_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        # Collect text from all content blocks; some SDK versions return a
        # list with multiple TextBlocks.
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip() or None
    except Exception as e:
        # Do NOT dump a full traceback here. This daemon is non-essential and
        # runs on a timer; when the Claude API is capped/throttled it raised a
        # full traceback EVERY cycle (17+/session in the logs), which tripped
        # the anomaly watcher's exception_burst detector → false self-heal
        # noise. Log a single calm line; recognise the recoverable/cap cases
        # explicitly. 2026-05-30 log-audit.
        _s = str(e).lower()
        if any(p in _s for p in ("usage limit", "regain access", "credit balance",
                                 "quota", "rate limit", "reached your",
                                 "overloaded", "exceeded", "529", "too low")):
            print(f"  [diag-daemons] deep-audit skipped — Claude API "
                  f"unavailable ({type(e).__name__})")
        else:
            print(f"  [diag-daemons] anthropic audit call failed: "
                  f"{type(e).__name__}: {e}")
        return None


def _parse_audit_findings(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip optional ```json fences.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except Exception:
        # Try to grab the first {...} block.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return []
    findings = obj.get("findings") if isinstance(obj, dict) else None
    if not isinstance(findings, list):
        return []
    cleaned: list[dict] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        cleaned.append({
            "rank": f.get("rank", 99),
            "file": str(f.get("file", "")).strip(),
            "line": f.get("line", 0),
            "category": str(f.get("category", "")).strip(),
            "summary": str(f.get("summary", "")).strip(),
            "fix_hint": str(f.get("fix_hint", "")).strip(),
        })
    cleaned.sort(key=lambda d: d.get("rank", 99))
    return cleaned


def _run_deep_audit_once(reason: str) -> int:
    files = _recently_modified_source_files(DEEP_AUDIT_FILES_PER_RUN)
    if not files:
        return 0
    prompt = _DEEP_AUDIT_PROMPT_TEMPLATE.format(
        file_blob=_build_audit_file_blob(files)
    )
    raw = _call_anthropic_auditor(prompt)
    if not raw:
        return 0
    findings = _parse_audit_findings(raw)
    queued = 0
    for f in findings:
        body = (
            f"[deep-audit] {f.get('category') or 'issue'} in "
            f"{f.get('file') or '?'}:{f.get('line') or 0} — "
            f"{f.get('summary') or ''}. Fix hint: "
            f"{f.get('fix_hint') or 'investigate'}."
        )
        if _append_todo_task(body, tag="deep-audit"):
            queued += 1
    completed = _count_pipeline_task_completions()
    _update_state(lambda s: s["deep_audit"].update({
        "last_run_ts": _now(),
        "last_run_iso": _iso(),
        "runs": int(s["deep_audit"].get("runs", 0)) + 1,
        "last_pipeline_event_count": completed,
        "hourly_window_count":
            int(s["deep_audit"].get("hourly_window_count", 0)) + 1,
        "daily_budget_spent_usd":
            float(s["deep_audit"].get("daily_budget_spent_usd", 0.0))
            + DEEP_AUDIT_ESTIMATED_COST_PER_RUN_USD,
        "pending_findings":
            int(s["deep_audit"].get("pending_findings", 0)) + queued,
    }))
    print(f"  [diag-daemons] deep-audit fired (reason={reason}) — "
          f"{queued} new finding(s) queued")
    return queued


def _deep_audit_loop() -> None:
    print(f"  [diag-daemons] deep-audit daemon online (budget="
          f"${_deep_audit_budget_usd():.2f}/day, max="
          f"{DEEP_AUDIT_MAX_RUNS_PER_HOUR}/h)")
    # Light cadence — we check every ~120 s and only act when due AND budget
    # allows; the daemon is fundamentally event-driven (batch / mtime).
    check_interval_s = 120
    while not _stop_event.is_set():
        if _stop_event.wait(check_interval_s):
            return
        try:
            _update_state(lambda s: s["deep_audit"].update({"alive_ts": _now()}))
            state = _read_state()
            if state.get("paused"):
                continue
            # BILLING-RULE GATE (2026-05-30 log-audit): this daemon calls the
            # Claude API directly on metered CREDITS (claude-sonnet-4-6) for
            # autonomous background code-auditing. The user's standing rule is
            # "API credits for conversational turns ONLY — everything else runs
            # on the Max subscription or not at all." So gate the credit-spend
            # on the same flag as the overnight self-improvement engine: when
            # autonomous upgrades are paused, do NOT burn credits auditing in
            # the background (this is also what was hammering the API cap every
            # cycle as files were edited — 17+ tracebacks/session in the logs).
            try:
                from core.config import OVERNIGHT_UPGRADE_ENABLED as _ovr_on
            except Exception:
                _ovr_on = False
            if not _ovr_on:
                continue
            due, reason = _deep_audit_due(state)
            if not due:
                continue
            ok, why = _deep_audit_budget_ok(state)
            if not ok:
                print(f"  [diag-daemons] deep-audit skipped: {why}")
                continue
            try:
                _run_deep_audit_once(reason)
            except Exception:
                print("  [diag-daemons] deep-audit run raised:")
                traceback.print_exc()
        except Exception:
            print("  [diag-daemons] deep-audit loop iteration raised:")
            traceback.print_exc()


# ──────────────────────────── anomaly watch ──────────────────────────
#
# Lightweight reactive monitor — runs every ~90s after a 60s boot delay.
# NEVER calls an LLM or shells out; just reads files. Findings are queued
# as `[anomaly]` tasks to jarvis_todo.md with a 30-min dedup window per
# signature so a single sick skill doesn't flood the todo list.

# Match lines like:
#   "  [action] system_pulse failed: ..."
#   "  [action] system_pulse error: ..."
#   "  [action] system_pulse raised: ..."
_ACTION_FAILURE_RE = re.compile(
    r"\[action\]\s+([A-Za-z0-9_.\-]+)\s*[:\-]?\s*(?:failed|error|raised|exception)",
    re.IGNORECASE,
)
# Lines indicating an in-process traceback (Python or our own '... raised:'
# style print).
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*\w+Error:|raised:\s*$",
    re.MULTILINE,
)


def _prune_dedup_signatures(state: dict) -> None:
    """Drop any signature whose dedup window has expired so the dict can't
    grow unbounded across days of uptime."""
    aw = state.get("anomaly_watch", {})
    sigs = aw.get("queued_signatures") or {}
    if not isinstance(sigs, dict):
        aw["queued_signatures"] = {}
        return
    cutoff = _now() - ANOMALY_DEDUP_GAP_S
    pruned = {k: v for k, v in sigs.items()
              if isinstance(v, (int, float)) and float(v) >= cutoff}
    aw["queued_signatures"] = pruned


def _claim_signature(signature: str) -> bool:
    """Atomic check-and-set under _state_lock. Returns True if the signature
    was NOT in the dedup window (and is now reserved); False if it was already
    claimed within ANOMALY_DEDUP_GAP_S.

    Folding the read-modify-write into one critical section closes the
    race where two checks both passed before either mark landed — a real
    risk now that the dedup-window is the only line of defence (we removed
    the broken bucket-rotation trick that was masking duplicates by
    rotating the key)."""
    claimed = {"ok": False}

    def _mut(s: dict) -> None:
        aw = s.setdefault("anomaly_watch", {})
        sigs = aw.setdefault("queued_signatures", {})
        if not isinstance(sigs, dict):
            sigs = {}
            aw["queued_signatures"] = sigs
        last = sigs.get(signature)
        now = _now()
        if last is not None:
            try:
                if (now - float(last)) < ANOMALY_DEDUP_GAP_S:
                    return  # still inside window — do NOT claim
            except (TypeError, ValueError):
                pass
        sigs[signature] = now
        _prune_dedup_signatures(s)
        claimed["ok"] = True

    _update_state(_mut)
    return claimed["ok"]


def _release_signature(signature: str) -> None:
    """Roll back a claim when the todo append fails so the next sweep can
    legitimately retry rather than silently swallowing the failure."""
    def _mut(s: dict) -> None:
        aw = s.setdefault("anomaly_watch", {})
        sigs = aw.get("queued_signatures") or {}
        if isinstance(sigs, dict) and signature in sigs:
            sigs.pop(signature, None)
    _update_state(_mut)


def _bump_detection_count() -> None:
    def _mut(s: dict) -> None:
        aw = s.setdefault("anomaly_watch", {})
        aw["detections"] = int(aw.get("detections", 0)) + 1
    _update_state(_mut)


def _queue_anomaly(body: str, signature: str,
                   already_queued_this_sweep: int = 0) -> bool:
    """Append an [anomaly] task if signature isn't in the dedup window.

    Returns True if a new task was queued. Enforces a per-sweep cap so a
    pathological log can't burst N anomalies in a single tick."""
    if already_queued_this_sweep >= ANOMALY_MAX_QUEUED_PER_SWEEP:
        return False
    # Atomic claim — if another check already marked this signature within
    # the cooldown, we bail before even touching the todo file.
    if not _claim_signature(signature):
        return False
    queued = _append_todo_task(body, tag="anomaly")
    if queued:
        _bump_detection_count()
        print(f"  [diag-daemons] queued anomaly: {signature}")
    else:
        # The append was a no-op (file write failed OR body already on disk
        # from a previous boot). Release the claim so the next sweep can
        # retry rather than wedging on a stale lock.
        _release_signature(signature)
    return queued


def _tail_bytes(path: str, n_bytes: int) -> str:
    """Read up to the last n_bytes of a file. Returns "" on any failure."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _check_boot_failures(queued_so_far: list[int]) -> None:
    """Scan data/boot_failures.jsonl for new entries since last poll. Queue
    one [anomaly] task per distinct boot failure kind, capped by the
    per-sweep budget so a freshly populated file can't drain it in one
    pass."""
    if not os.path.exists(BOOT_FAILURES_FILE):
        return
    state = _read_state()
    last_offset = int(state.get("anomaly_watch", {})
                      .get("last_boot_failure_offset", 0))
    try:
        size = os.path.getsize(BOOT_FAILURES_FILE)
    except OSError:
        return
    if size < last_offset:
        # File was truncated/rotated — start over from zero.
        last_offset = 0
    if size == last_offset:
        return
    try:
        with open(BOOT_FAILURES_FILE, "r", encoding="utf-8",
                  errors="replace") as f:
            f.seek(last_offset)
            new_lines = f.readlines()
            new_offset = f.tell()
    except OSError:
        return
    seen_kinds: set[str] = set()
    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        kind = (obj.get("kind") or "boot_failure").strip()
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        iso = obj.get("iso") or _iso(obj.get("ts"))
        err = (obj.get("error_repr") or "").strip()
        body = (
            f"[anomaly] boot failure detected ({kind}) at {iso}: "
            f"jarvis failed to acquire its singleton lock — "
            f"path={obj.get('lock_path')!r}, "
            f"winerror={obj.get('winerror')}, errno={obj.get('errno')}"
        )
        if err:
            body += f". last error: {err[:200]}"
        signature = f"boot_failure:{kind}:{obj.get('winerror')}:{obj.get('errno')}"
        if _queue_anomaly(body, signature, queued_so_far[0]):
            queued_so_far[0] += 1
    _update_state(lambda s: s["anomaly_watch"].update({
        "last_boot_failure_offset": new_offset,
    }))


def _check_stuck_loop(queued_so_far: list[int]) -> None:
    """Use hud_state.json mtime as a main-loop heartbeat. The HUD publisher
    runs in the main thread on every audio tick, so a stale hud_state.json
    is a *candidate* signal the event loop has wedged — but a single slow
    I/O sweep can also briefly stall HUD writes, so we require
    ANOMALY_STUCK_LOOP_MIN_CONSECUTIVE consecutive stale sweeps before we
    queue anything. That removes the worst false-positive class (heavy
    disk/network I/O blocking the main loop for ~5 min while JARVIS is
    perfectly healthy)."""
    if not os.path.exists(HUD_STATE_FILE):
        # HUD might be disabled — don't false-positive in that case.
        _update_state(lambda s: s["anomaly_watch"].update(
            {"stuck_loop_consecutive_misses": 0}))
        return
    try:
        age = _now() - os.path.getmtime(HUD_STATE_FILE)
    except OSError:
        return

    if age < ANOMALY_STUCK_LOOP_THRESHOLD_S:
        # Healthy: reset the consecutive-miss streak so a future stall has
        # to clear the bar from scratch.
        state = _read_state()
        prev = int(state.get("anomaly_watch", {})
                   .get("stuck_loop_consecutive_misses", 0))
        if prev:
            _update_state(lambda s: s["anomaly_watch"].update(
                {"stuck_loop_consecutive_misses": 0}))
        return

    # Increment the miss streak and only fire once we've cleared the bar.
    state = _update_state(lambda s: s["anomaly_watch"].update(
        {"stuck_loop_consecutive_misses":
         int(s["anomaly_watch"].get("stuck_loop_consecutive_misses", 0)) + 1}))
    misses = int(state.get("anomaly_watch", {})
                 .get("stuck_loop_consecutive_misses", 0))
    if misses < ANOMALY_STUCK_LOOP_MIN_CONSECUTIVE:
        return

    body = (
        f"[anomaly] main loop appears stuck — hud_state.json hasn't been "
        f"updated in {int(age)}s across {misses} consecutive sweeps "
        f"(threshold {ANOMALY_STUCK_LOOP_THRESHOLD_S}s × "
        f"{ANOMALY_STUCK_LOOP_MIN_CONSECUTIVE}). Possible deadlock or "
        f"blocking call in the audio/dispatch path. Inspect threads and "
        f"recent session_*.log for the last call before the silence."
    )
    # Stable signature — the 30-min cooldown is the only dedup; once the
    # window expires and we're STILL stuck, we'll re-queue, which is the
    # correct behaviour.
    if _queue_anomaly(body, "stuck_loop", queued_so_far[0]):
        queued_so_far[0] += 1
        # Reset the streak so a re-fire after the dedup window doesn't
        # report a misleading ever-growing "across N consecutive sweeps"
        # count (3, 4, 5, ..., 240). Guard on the queue-success return so
        # we don't mask the streak when the anomaly was suppressed.
        _update_state(lambda s: s["anomaly_watch"].update(
            {"stuck_loop_consecutive_misses": 0}))


def _looks_like_text_log(sample: str) -> bool:
    """Cheap heuristic — a healthy session log is mostly printable ASCII
    plus newlines. If the tail comes back full of NULs or binary noise the
    file was probably truncated mid-write or rotated under us; skip rather
    than parsing garbage."""
    if not sample:
        return False
    snippet = sample[:4096]
    if not snippet:  # pragma: no cover - unreachable: sample was proven truthy one line above, so sample[:4096] is always non-empty for a str
        return False
    if "\x00" in snippet:
        return False
    printable = sum(1 for c in snippet if c == "\n" or c == "\t" or
                    (32 <= ord(c) < 127) or ord(c) >= 160)
    return (printable / max(1, len(snippet))) >= 0.85


def _line_at(text: str, m_start: int, m_end: int) -> str:
    start = text.rfind("\n", 0, m_start) + 1
    end = text.find("\n", m_end)
    if end == -1:
        end = len(text)
    return text[start:end]


def _is_transient_failure_line(line: str) -> bool:
    """Filter out action failures that are known-transient (DNS, brief COM
    glitch, timeout). One-off transients of this flavour are normal and
    shouldn't trip the per-skill threshold."""
    low = line.lower()
    for tok in _TRANSIENT_ERROR_TOKENS:
        if tok in low:
            return True
    return False


def _check_log_failures(queued_so_far: list[int]) -> None:
    """Tail the freshest session log and count repeated action failures /
    tracebacks within the recent window. Each detector is independently
    guarded so a single bad regex match can't take the loop down."""
    log_path = _latest_session_log_path()
    if not log_path:
        return
    try:
        text = _tail_bytes(log_path, ANOMALY_LOG_TAIL_BYTES)
    except Exception:
        return
    if not text or not _looks_like_text_log(text):
        return

    # Count per-skill action failures, filtering known transients.
    action_counts: dict[str, int] = {}
    try:
        matches = list(_ACTION_FAILURE_RE.finditer(text))
    except Exception:
        matches = []
    for m in matches:
        try:
            name = m.group(1).strip().lower()
        except Exception:
            continue
        if not name:
            continue
        # Whitelist skills whose normal failure rate is non-zero (network,
        # weather, external services).
        if name in _TRANSIENT_SKILL_NAMES:
            continue
        # Whitelist this specific failure if the line carries a transient
        # error token (timeout, DNS hiccup, transient COM error).
        try:
            line = _line_at(text, m.start(), m.end())
        except Exception:
            line = ""
        if _is_transient_failure_line(line):
            continue
        action_counts[name] = action_counts.get(name, 0) + 1

    for skill, count in sorted(action_counts.items(),
                               key=lambda kv: kv[1], reverse=True):
        if count < ANOMALY_FAILURE_THRESHOLD:
            continue
        body = (
            f"[anomaly] skill '{skill}' failed {count} times in the recent "
            f"session log tail (threshold {ANOMALY_FAILURE_THRESHOLD}, "
            f"transients excluded). Inspect the skill handler and any "
            f"external dependency it calls (network, file I/O, COM)."
        )
        # Stable signature: same skill within the cooldown collapses to one
        # task. After 30 min the cooldown lifts; if the spike is still
        # happening we'll re-queue legitimately.
        if _queue_anomaly(body, f"skill_failure:{skill}", queued_so_far[0]):
            queued_so_far[0] += 1

    # Count tracebacks/exceptions.
    try:
        tb_hits = list(_TRACEBACK_RE.finditer(text))
    except Exception:
        tb_hits = []
    if len(tb_hits) >= ANOMALY_EXCEPTION_THRESHOLD:
        sample = ""
        try:
            m = tb_hits[-1]
            sample = _line_at(text, m.start(), m.end()).strip()[:240]
        except Exception:
            sample = "(sample unavailable)"
        body = (
            f"[anomaly] {len(tb_hits)} unhandled exception traces seen in "
            f"the recent session log tail (threshold "
            f"{ANOMALY_EXCEPTION_THRESHOLD}). Sample: {sample}"
        )
        if _queue_anomaly(body, "exception_burst", queued_so_far[0]):
            queued_so_far[0] += 1


def _latest_session_log_path() -> str | None:
    try:
        files = [
            os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR)
            if f.startswith("session_") and f.endswith(".log")
        ]
    except FileNotFoundError:
        return None
    if not files:
        return None
    return max(files, key=lambda p: os.path.getmtime(p))


def _anomaly_watch_loop() -> None:
    print(f"  [diag-daemons] anomaly watcher online "
          f"(initial_delay={ANOMALY_INITIAL_DELAY_S}s, "
          f"interval={ANOMALY_POLL_INTERVAL_S}s)")
    # Defer the first sweep so we don't compete with boot-time I/O or
    # mis-classify a slow startup as a stuck loop.
    if _stop_event.wait(ANOMALY_INITIAL_DELAY_S):
        return
    while not _stop_event.is_set():
        try:
            _update_state(lambda s: s["anomaly_watch"].update(
                {"alive_ts": _now()}))
            state = _read_state()
            if state.get("paused"):
                if _stop_event.wait(ANOMALY_POLL_INTERVAL_S):
                    return
                continue
            # Each check is wrapped — one failing detector must not knock
            # the others out. The shared `queued_so_far` counter enforces
            # ANOMALY_MAX_QUEUED_PER_SWEEP across all detectors so a sick
            # log + a fresh boot_failure can't double-queue in one tick.
            queued_so_far: list[int] = [0]
            for fn in (_check_boot_failures, _check_stuck_loop,
                       _check_log_failures):
                try:
                    fn(queued_so_far)
                except Exception:
                    print(f"  [diag-daemons] anomaly check {fn.__name__} raised:")
                    traceback.print_exc()
            _update_state(lambda s: s["anomaly_watch"].update({
                "last_poll_ts": _now(),
                "last_poll_iso": _iso(),
            }))
        except Exception:
            print("  [diag-daemons] anomaly-watch loop iteration raised:")
            traceback.print_exc()
        if _stop_event.wait(ANOMALY_POLL_INTERVAL_S):
            return


# ──────────────────────────── lifecycle ──────────────────────────────

def start_diagnostic_daemons() -> bool:
    """Spin up all three daemon threads. Idempotent — second call is a
    no-op. Returns True when threads were actually started.

    Caller (bobert_companion.main) wraps this in a try/except, but we also
    guard each step here so a transient disk/JSON hiccup on the state file
    cannot prevent the daemon threads from coming up. Threads themselves
    own all subsequent I/O and are individually fault-tolerant."""
    global _started
    with _lock_threads:
        if _started:
            return False
        _stop_event.clear()
        # Best-effort: refresh the state file shape. If the disk is wedged
        # or the JSON file is corrupt we still want the threads to start —
        # they each re-read state on every tick and will recover on the
        # next successful write.
        try:
            _update_state(lambda s: None)
        except Exception as _e:
            print(f"  [diag-daemons] state init skipped: {_e}")
        threads = [
            threading.Thread(target=_self_diag_loop,
                             name="jarvis-self-diag-daemon", daemon=True),
            threading.Thread(target=_crash_watch_loop,
                             name="jarvis-crash-watch-daemon", daemon=True),
            threading.Thread(target=_deep_audit_loop,
                             name="jarvis-deep-audit-daemon", daemon=True),
            threading.Thread(target=_anomaly_watch_loop,
                             name="jarvis-anomaly-watch-daemon", daemon=True),
        ]
        for t in threads:
            try:
                t.start()
            except Exception as _e:
                print(f"  [diag-daemons] thread '{t.name}' failed to start: {_e}")
                continue
            _threads.append(t)
        _started = bool(_threads)
    print(f"  [diag-daemons] {len(_threads)}/{len(threads)} daemons running")
    return True


def stop_diagnostic_daemons(join_timeout: float = THREAD_JOIN_TIMEOUT_S) -> None:
    """Signal the threads to stop and wait briefly for them. Safe to call
    multiple times."""
    global _started
    with _lock_threads:
        if not _started:
            return
        _stop_event.set()
        for t in list(_threads):
            try:
                t.join(timeout=join_timeout)
            except Exception:
                pass
        _threads.clear()
        _started = False
    print("  [diag-daemons] shutdown complete")


def pause_diagnostics() -> str:
    _update_state(lambda s: s.update({"paused": True}))
    return "Diagnostics paused, sir."


def resume_diagnostics() -> str:
    _update_state(lambda s: s.update({"paused": False}))
    return "Diagnostics resumed, sir."


def diagnostic_daemon_status() -> dict[str, Any]:
    state = _read_state()
    self_diag = state.get("self_diag", {})
    crash = state.get("crash_watch", {})
    audit = state.get("deep_audit", {})
    budget = _deep_audit_budget_usd()
    spent = float(audit.get("daily_budget_spent_usd", 0.0))
    now = _now()
    # A daemon is "alive" if its loop has updated alive_ts within 2x its
    # tick interval. Each loop heartbeats at the top of every iteration, so
    # a silent thread death is detectable here even though _started stays
    # True until shutdown.
    self_diag_alive = (
        _started
        and (now - float(self_diag.get("alive_ts", 0.0)))
        < 2 * SELF_DIAG_INTERVAL_S
    )
    crash_alive = (
        _started
        and (now - float(crash.get("alive_ts", 0.0)))
        < 2 * CRASH_POLL_INTERVAL_S
    )
    # deep-audit uses an internal 120s tick interval; mirror it here.
    deep_audit_alive = (
        _started
        and (now - float(audit.get("alive_ts", 0.0))) < 2 * 120
    )
    anomaly = state.get("anomaly_watch", {})
    anomaly_alive = (
        _started
        and (now - float(anomaly.get("alive_ts", 0.0)))
        < 2 * ANOMALY_POLL_INTERVAL_S
    )
    return {
        "paused": bool(state.get("paused")),
        "self_diag_last_iso": self_diag.get("last_run_iso"),
        "self_diag_runs": int(self_diag.get("runs", 0)),
        "self_diag_alive": self_diag_alive,
        "crash_watch_last_poll_iso":
            _iso(crash.get("last_poll_ts") or 0),
        "crash_watch_detections":
            int(crash.get("detections", 0)),
        "crash_watch_alive": crash_alive,
        "deep_audit_last_iso": audit.get("last_run_iso"),
        "deep_audit_runs": int(audit.get("runs", 0)),
        "deep_audit_budget_remaining_usd": round(max(0.0, budget - spent), 4),
        "deep_audit_pending_findings": int(audit.get("pending_findings", 0)),
        "deep_audit_alive": deep_audit_alive,
        "anomaly_watch_last_poll_iso": anomaly.get("last_poll_iso"),
        "anomaly_watch_detections": int(anomaly.get("detections", 0)),
        "anomaly_watch_alive": anomaly_alive,
        "started": _started,
    }


def diagnostic_daemon_status_spoken(_: str = "") -> str:
    """Voice-friendly status summary for 'JARVIS, diagnostic status'."""
    s = diagnostic_daemon_status()
    if s["paused"]:
        prefix = "Diagnostics are paused, sir. "
    else:
        prefix = ""
    self_diag = s["self_diag_last_iso"] or "never"
    audit_last = s["deep_audit_last_iso"] or "never"
    anomaly_last = s.get("anomaly_watch_last_poll_iso") or "never"
    return (
        f"{prefix}Last self-diagnostic at {self_diag}, "
        f"{s['self_diag_runs']} runs total. "
        f"Crash watcher last polled at {s['crash_watch_last_poll_iso']}, "
        f"{s['crash_watch_detections']} detections. "
        f"Deep audit last at {audit_last}, "
        f"{s['deep_audit_runs']} runs, "
        f"${s['deep_audit_budget_remaining_usd']:.2f} budget remaining, "
        f"{s['deep_audit_pending_findings']} pending findings. "
        f"Anomaly watcher last polled at {anomaly_last}, "
        f"{s.get('anomaly_watch_detections', 0)} detections."
    )


# Voice-action shims (so the existing action-dispatch infrastructure can
# route "pause diagnostics" etc. through these names).

def act_pause_diagnostics(_: str = "") -> str:
    return pause_diagnostics()


def act_resume_diagnostics(_: str = "") -> str:
    return resume_diagnostics()


def act_diagnostic_status(_: str = "") -> str:
    return diagnostic_daemon_status_spoken(_)
