"""Blue/green deployment manager for zero-downtime JARVIS upgrades.

Owns the isolation primitives that let two JARVIS processes coexist on the
same machine — one PROD (blue) serving the user, one STAGING (green)
running smoke tests against the candidate code. After staging passes, the
upgrade pipeline calls into here to swap the active version atomically.

Key data files (all live next to bobert_companion.py):

  data/deployment_state.json   — the source of truth for active role/version
  data/instances.json          — per-PID heartbeats (role, version, started_at)
  data/handoff.json            — conversation/runtime state ferried prod→green
  data/handoff.signal          — one-shot signal: "graceful idle then exit"
  data_staging/                — green's parallel working state directory
  jarvis.lock                  — PROD singleton (PID-locked)
  jarvis_staging.lock          — STAGING singleton (PID-locked)
  injected_commands_staging.json — green's text-input channel

The functions here are intentionally side-effect-light: they create
directories on demand, read/write JSON atomically, and never reach into
the running JARVIS process directly. The runtime hooks (lock files, mic
disable, HUD skip, state isolation) live in bobert_companion.py guarded
by `is_staging()`.

Public surface:

  is_staging()                   — True when this process was launched with --staging
  resolve_role()                 — "prod" | "staging" based on argv/env
  resource_paths(role)           — dict of paths for that role's lock, data dir, ...
  paths_for_current_process()    — shortcut: resource_paths(resolve_role())
  read_state() / write_state()   — deployment_state.json (atomic)
  register_instance(role, ...)   — heartbeat into instances.json
  unregister_instance(pid)       — remove from instances.json (atexit)
  signal_handoff(...)            — write handoff.signal so prod idles + exits
  consume_handoff_signal()       — prod-side: pop the signal, return its payload
  promote_staging(...)           — atomically swap lock pointers + bump version
  rollback(...)                  — remove staging artifacts; deployment_state stays on prod

This file deliberately has no imports from bobert_companion so it stays
unit-testable and safe to import from the upgrade pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
DATA_STAGING_DIR = os.path.join(PROJECT_DIR, "data_staging")

DEPLOYMENT_STATE_FILE = os.path.join(DATA_DIR, "deployment_state.json")
INSTANCES_FILE        = os.path.join(DATA_DIR, "instances.json")
HANDOFF_FILE          = os.path.join(DATA_DIR, "handoff.json")
HANDOFF_SIGNAL_FILE   = os.path.join(DATA_DIR, "handoff.signal")
# blue-green-2: secondary one-shot signals so prod can voice an
# abort / failure during the cinematic handoff.
UPGRADE_ABORT_FILE    = os.path.join(DATA_DIR, "upgrade_aborted.signal")
HANDOFF_FAILURE_FILE  = os.path.join(DATA_DIR, "handoff_failure.signal")
VERSION_FILE          = os.path.join(DATA_DIR, "version.json")

PROD_LOCK_FILE        = os.path.join(PROJECT_DIR, "jarvis.lock")
STAGING_LOCK_FILE     = os.path.join(PROJECT_DIR, "jarvis_staging.lock")

PROD_INJECT_FILE      = os.path.join(PROJECT_DIR, "injected_commands.json")
STAGING_INJECT_FILE   = os.path.join(PROJECT_DIR, "injected_commands_staging.json")

STAGING_FLAG = "--staging"
RESUME_HANDOFF_FLAG = "--resume-handoff"


def _ensure_data_dir() -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError:
        pass


def _atomic_write_json(path: str, payload: dict) -> bool:
    """Write `payload` to `path` via tmp+replace. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _read_json(path: str, default: dict | list | None = None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ── role detection ───────────────────────────────────────────────────────


def is_staging(argv: list[str] | None = None) -> bool:
    """True if this process was launched with --staging on argv or via the
    JARVIS_STAGING=1 env var (the env path lets test harnesses opt in
    without rewriting argv)."""
    if argv is None:
        argv = sys.argv
    if STAGING_FLAG in argv:
        return True
    return os.environ.get("JARVIS_STAGING", "").strip() == "1"


def resolve_role(argv: list[str] | None = None) -> str:
    return "staging" if is_staging(argv) else "prod"


# ── per-role resource isolation ──────────────────────────────────────────


def resource_paths(role: str) -> dict:
    """Return the resource paths a process of the given role should use.
    Centralising this here means bobert_companion.py never has to branch
    on role beyond a single lookup at startup."""
    role = "staging" if role == "staging" else "prod"
    if role == "staging":
        return {
            "role": "staging",
            "lock_file":     STAGING_LOCK_FILE,
            "data_dir":      DATA_STAGING_DIR,
            "logs_dir":      os.path.join(PROJECT_DIR, "logs_staging"),
            "memory_dir":    os.path.join(DATA_STAGING_DIR, "memory"),
            "hud_state_file": os.path.join(DATA_STAGING_DIR, "hud_state.json"),
            "inject_file":   STAGING_INJECT_FILE,
            "tray_enabled":  False,
            # blue-green-2: staging HUD is VISIBLE so the user can watch the
            # ceremony. It lives on a separate state file (above), pinned to
            # `monitor_name` (left), so it never blinks the prod HUD.
            "hud_enabled":   True,
            "hud_enabled_in_staging": True,
            "monitor_name":  "left",
            "mic_enabled":   False,
            "tts_audio_out": False,
            "bambu_enabled": False,
            "camera_enabled": False,
            "replies_file":  os.path.join(DATA_STAGING_DIR, "replies.jsonl"),
            "test_state_template": {
                "current_test_case": "",
                "tests_passed":      0,
                "tests_remaining":   0,
            },
        }
    return {
        "role": "prod",
        "lock_file":     PROD_LOCK_FILE,
        "data_dir":      DATA_DIR,
        "logs_dir":      os.path.join(PROJECT_DIR, "logs"),
        "memory_dir":    os.path.join(PROJECT_DIR, "memory"),
        "hud_state_file": os.path.join(PROJECT_DIR, "hud_state.json"),
        "inject_file":   PROD_INJECT_FILE,
        "tray_enabled":  True,
        "hud_enabled":   True,
        "hud_enabled_in_staging": False,
        "monitor_name":  "top",
        "mic_enabled":   True,
        "tts_audio_out": True,
        "bambu_enabled": True,
        "camera_enabled": True,
        "replies_file":  os.path.join(DATA_DIR, "replies.jsonl"),
        "test_state_template": {},
    }


def paths_for_current_process(argv: list[str] | None = None) -> dict:
    return resource_paths(resolve_role(argv))


def ensure_role_dirs(role: str) -> None:
    """Pre-create the directories the role needs so first-time staging
    boots don't trip on missing-dir errors."""
    p = resource_paths(role)
    for key in ("data_dir", "logs_dir", "memory_dir"):
        try:
            os.makedirs(p[key], exist_ok=True)
        except OSError:
            pass


# ── deployment_state.json ────────────────────────────────────────────────
# Schema:
#   {
#     "active_role": "prod" | "staging",   # who currently owns the user
#     "prod_version":    "1.0.5",
#     "staging_version": "1.0.6",          # null when no staging is running
#     "staging_pid":     12345 | null,
#     "prod_pid":        12340 | null,
#     "last_promotion_at": "2026-05-28T01:23:45",
#     "last_rollback_at":  null
#   }


_DEFAULT_STATE = {
    "active_role":     "prod",
    "prod_version":    None,
    "staging_version": None,
    "prod_pid":        None,
    "staging_pid":     None,
    "last_promotion_at": None,
    "last_rollback_at":  None,
}


def read_state() -> dict:
    state = _read_json(DEPLOYMENT_STATE_FILE, dict(_DEFAULT_STATE))
    if not isinstance(state, dict):
        return dict(_DEFAULT_STATE)
    out = dict(_DEFAULT_STATE)
    out.update(state)
    return out


def write_state(updates: dict) -> dict:
    """Merge `updates` into deployment_state.json and persist atomically.
    Returns the merged state."""
    _ensure_data_dir()
    state = read_state()
    state.update(updates)
    _atomic_write_json(DEPLOYMENT_STATE_FILE, state)
    return state


# ── instances.json — per-PID heartbeats ──────────────────────────────────


def register_instance(role: str, version: str | None = None,
                       extra: dict | None = None) -> dict:
    """Record this PID's role/version into instances.json. Called from
    bobert_companion startup AFTER the singleton check passes so we know
    we hold the lock for our role."""
    _ensure_data_dir()
    pid = os.getpid()
    data = _read_json(INSTANCES_FILE, {})
    if not isinstance(data, dict):
        data = {}
    # Drop dead PIDs while we're here — keeps the file from accumulating
    # crash residue across upgrade cycles.
    live: dict = {}
    for k, v in data.items():
        try:
            if _pid_alive(int(k)):
                live[k] = v
        except (TypeError, ValueError):
            continue
    entry = {
        "pid":        pid,
        "role":       role,
        "version":    version,
        "started_at": time.time(),
        "heartbeat_at": time.time(),
    }
    if extra:
        entry.update(extra)
    live[str(pid)] = entry
    _atomic_write_json(INSTANCES_FILE, live)
    return entry


def heartbeat(role: str, version: str | None = None,
              extra: dict | None = None) -> None:
    """Refresh this PID's instances.json entry. Called periodically from
    the main loop's background tick."""
    if not os.path.exists(INSTANCES_FILE):
        register_instance(role, version, extra)
        return
    data = _read_json(INSTANCES_FILE, {})
    if not isinstance(data, dict):
        data = {}
    key = str(os.getpid())
    entry = data.get(key) or {
        "pid": os.getpid(), "role": role, "version": version,
        "started_at": time.time(),
    }
    entry["role"] = role
    entry["version"] = version
    entry["heartbeat_at"] = time.time()
    if extra:
        entry.update(extra)
    data[key] = entry
    _atomic_write_json(INSTANCES_FILE, data)


def unregister_instance(pid: int | None = None) -> None:
    if pid is None:
        pid = os.getpid()
    if not os.path.exists(INSTANCES_FILE):
        return
    data = _read_json(INSTANCES_FILE, {})
    if not isinstance(data, dict):
        return
    data.pop(str(pid), None)
    _atomic_write_json(INSTANCES_FILE, data)


def list_instances() -> dict:
    data = _read_json(INSTANCES_FILE, {})
    return data if isinstance(data, dict) else {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ── handoff signal — prod idles + exits on next user-quiet window ────────


def signal_handoff(reason: str = "upgrade",
                    target_version: str | None = None,
                    grace_seconds: int = 10) -> bool:
    """Write a handoff signal that the prod JARVIS will see on its next
    tick. Reads as: 'finish whatever you're saying, then exit cleanly.'
    The pipeline waits grace_seconds + a small margin before assuming the
    handoff has happened."""
    payload = {
        "signaled_at":    time.time(),
        "reason":         reason,
        "target_version": target_version,
        "grace_seconds":  int(grace_seconds),
    }
    return _atomic_write_json(HANDOFF_SIGNAL_FILE, payload)


def consume_handoff_signal() -> dict | None:
    """Called by the prod JARVIS's main-loop watchdog. If the signal file
    is present, return the payload and DELETE the file so it only fires
    once. Returns None when no signal is pending."""
    if not os.path.exists(HANDOFF_SIGNAL_FILE):
        return None
    payload = _read_json(HANDOFF_SIGNAL_FILE, {})
    try:
        os.remove(HANDOFF_SIGNAL_FILE)
    except OSError:
        pass
    return payload if isinstance(payload, dict) else None


def write_handoff_state(state: dict) -> bool:
    """Prod writes its in-flight conversation/timers state here so the
    next prod (formerly green) can pick up seamlessly via
    `--resume-handoff` on boot."""
    return _atomic_write_json(HANDOFF_FILE, dict(state))


HANDOFF_STATE_TTL_SECONDS = 600


def consume_handoff_state() -> dict | None:
    """New-prod-side equivalent of consume_handoff_signal — read + delete.

    Rejects payloads older than HANDOFF_STATE_TTL_SECONDS (10 min) as a
    secondary defense against orphaned handoff.json files left behind by
    an aborted blue-green run. Without this, a regular full-upgrade days
    later would replay a multi-day-old conversation_tail into the fresh
    session and JARVIS would greet mid-sentence about a stale topic."""
    if not os.path.exists(HANDOFF_FILE):
        return None
    payload = _read_json(HANDOFF_FILE, {})
    try:
        os.remove(HANDOFF_FILE)
    except OSError:
        pass
    if not isinstance(payload, dict):
        return None
    signaled_at = payload.get("signaled_at")
    if isinstance(signaled_at, (int, float)):
        age = time.time() - float(signaled_at)
        if age > HANDOFF_STATE_TTL_SECONDS:
            return None
    return payload


# ── blue-green-2: secondary signals for the cinematic handoff ────────────


def signal_upgrade_aborted(reason: str = "smoke-failed") -> bool:
    """Write a one-shot 'upgrade aborted' signal that the prod main loop
    picks up and announces ('I'm afraid the upgrade was aborted, sir.').
    Used when staging smoke tests fail so the user knows the candidate
    build didn't ship without watching the upgrade console."""
    payload = {"signaled_at": time.time(), "reason": reason}
    return _atomic_write_json(UPGRADE_ABORT_FILE, payload)


def consume_upgrade_aborted_signal() -> dict | None:
    if not os.path.exists(UPGRADE_ABORT_FILE):
        return None
    payload = _read_json(UPGRADE_ABORT_FILE, {})
    try:
        os.remove(UPGRADE_ABORT_FILE)
    except OSError:
        pass
    return payload if isinstance(payload, dict) else None


def signal_handoff_failure(reason: str = "timeout") -> bool:
    """Write a one-shot 'handoff failure' signal so prod can announce
    'Handoff failure, sir — staying on the current version.' and stop
    waiting for the takeover."""
    payload = {"signaled_at": time.time(), "reason": reason}
    return _atomic_write_json(HANDOFF_FAILURE_FILE, payload)


def consume_handoff_failure_signal() -> dict | None:
    if not os.path.exists(HANDOFF_FAILURE_FILE):
        return None
    payload = _read_json(HANDOFF_FAILURE_FILE, {})
    try:
        os.remove(HANDOFF_FAILURE_FILE)
    except OSError:
        pass
    return payload if isinstance(payload, dict) else None


# ── blue-green-2: staging test progress published into instances.json ────


def publish_test_state(test_state: dict, role: str | None = None) -> None:
    """Called by the staging instance (or the smoke-test driver) to attach
    `test_state` to this PID's instances.json entry. Prod and the staging
    HUD both poll this so a quick glance at the LEFT monitor shows which
    smoke case is currently running."""
    if role is None:
        role = resolve_role()
    try:
        heartbeat(role=role, version=read_version(),
                  extra={"test_state": dict(test_state or {})})
    except Exception:
        pass


def read_staging_test_state(max_age_s: float = 30.0) -> dict | None:
    """Prod-side read of the staging PID's published `test_state`. Returns
    None when there is no live staging instance or it has not yet
    published any progress."""
    insts = list_instances()
    now = time.time()
    for entry in insts.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "staging":
            continue
        hb = float(entry.get("heartbeat_at") or 0)
        if hb and (now - hb) > max_age_s:
            continue
        ts = entry.get("test_state")
        if isinstance(ts, dict):
            return ts
    return None


# ── promote / rollback ───────────────────────────────────────────────────


def promote_staging(new_version: str | None = None,
                    staging_pid: int | None = None) -> dict:
    """Mark the staging build as the new prod build in deployment_state.
    The actual lock-file swap happens when prod exits (it releases its
    lock) and green's restart picks up jarvis.lock at boot. This call
    just records the intent so observers (tray, voice, HUD) see the
    state flip immediately.

    Called by the upgrade pipeline AFTER staging's smoke tests pass and
    AFTER prod has been signaled to idle.
    """
    state = read_state()
    if new_version is None:
        new_version = state.get("staging_version") or state.get("prod_version")
    return write_state({
        "active_role":       "prod",   # green will rebrand as prod on next boot
        "prod_version":      new_version,
        "staging_version":   None,
        "staging_pid":       None,
        "prod_pid":          staging_pid,
        "last_promotion_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def rollback(reason: str = "smoke-test-failed") -> dict:
    """Tear down staging without touching prod. Removes the staging lock
    file (in case it was orphaned) and clears staging fields in
    deployment_state. The on-disk data_staging/ tree is left alone — the
    pipeline reads it for post-mortem if needed."""
    try:
        if os.path.exists(STAGING_LOCK_FILE):
            os.remove(STAGING_LOCK_FILE)
    except OSError:
        pass
    state = write_state({
        "staging_version":  None,
        "staging_pid":      None,
        "last_rollback_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_rollback_reason": reason,
    })
    return state


# ── helpers used by the upgrade pipeline ─────────────────────────────────


def seed_staging_data() -> None:
    """Copy a minimal slice of data/ into data_staging/ so green boots
    with believable state rather than a blank profile. We deliberately
    avoid mirroring everything — large caches, audio recordings, and
    the prod memory directory are skipped to keep boot fast. State that
    legitimately needs to influence the smoke-test (version.json, recent
    intent caches) is copied; everything else green creates fresh.

    Safe to call multiple times — uses dirs_exist_ok semantics."""
    import shutil
    ensure_role_dirs("staging")
    seed_files = [
        "version.json",
        "user_settings.json",
    ]
    for fname in seed_files:
        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(DATA_STAGING_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass


def read_version() -> str:
    payload = _read_json(VERSION_FILE, {})
    if isinstance(payload, dict):
        v = payload.get("version") or payload.get("v")
        if isinstance(v, str):
            return v
    return "0.0.0"


def staging_is_running() -> bool:
    """Quick check for the upgrade pipeline: is there a live PID locked
    against jarvis_staging.lock right now?"""
    if not os.path.exists(STAGING_LOCK_FILE):
        return False
    try:
        with open(STAGING_LOCK_FILE, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


def prod_is_running() -> bool:
    if not os.path.exists(PROD_LOCK_FILE):
        return False
    try:
        with open(PROD_LOCK_FILE, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


if __name__ == "__main__":
    # Tiny CLI for ops poking: `python blue_green_manager.py status`.
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps({
            "deployment_state": read_state(),
            "instances":        list_instances(),
            "prod_running":     prod_is_running(),
            "staging_running":  staging_is_running(),
            "version":          read_version(),
        }, indent=2))
    elif cmd == "rollback":
        print(json.dumps(rollback("manual-cli"), indent=2))
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)
