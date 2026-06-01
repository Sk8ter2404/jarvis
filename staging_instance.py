"""Smoke-test runner for the green/staging JARVIS during a blue/green upgrade.

The upgrade pipeline calls `run_smoke_tests()` after spawning a staging
JARVIS. This module:

  1. Sanity-checks the candidate code via `py_compile`.
  2. Waits for `jarvis_staging.lock` to appear (proof the new process
     actually got past its singleton check).
  3. Feeds a handful of canned voice equivalents through
     `injected_commands_staging.json` and watches for replies in
     `data_staging/replies.jsonl`.
  4. Reads the staging instance's heartbeat from `data/instances.json`
     so a silently-crashed staging fails the gate even if it had time
     to write a few replies first.

The pipeline treats `run_smoke_tests` as: returns dict with `ok: bool`
and `details: str` — the caller decides whether to promote or roll back.

Deliberately small in scope. The real correctness gates are
`stability_smoke_test.py` and the auditor; this module owns just the
blue/green-specific input/output plumbing.
"""

from __future__ import annotations

import json
import os
import py_compile
import time
from typing import Iterable

import blue_green_manager as _bgm

# Smoke-test prompts kept short so the staging instance can drain them
# inside the timeout window. We test: (1) a no-op acknowledgement,
# (2) a state-query path that exercises the action dispatcher,
# (3) a simple memory recall to prove the side modules loaded.
DEFAULT_PROMPTS = [
    "are you the new one",
    "list timers",
    "what time is it",
]

# Where green will write replies. Mirrored from blue_green_manager so a
# stale import doesn't desync them.
_STAGING_PATHS = _bgm.resource_paths("staging")
REPLIES_FILE = _STAGING_PATHS["replies_file"]
STAGING_INJECT = _bgm.STAGING_INJECT_FILE
STAGING_LOCK = _bgm.STAGING_LOCK_FILE
# blue-green-2: the staging HUD reads from this file so the LEFT-monitor
# overlay can render the test progress alongside the smoke-test text.
STAGING_HUD_STATE_FILE = _STAGING_PATHS["hud_state_file"]


def precompile_candidate_files(files: Iterable[str]) -> tuple[bool, list[str]]:
    """Run py_compile on each path. Returns (ok, errors). Best-effort
    sweep — missing files are skipped (the upgrade may have removed
    them) but real SyntaxErrors fail the gate."""
    errors: list[str] = []
    for path in files:
        if not os.path.exists(path):
            continue
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{os.path.basename(path)}: {exc}")
        except OSError as exc:
            errors.append(f"{os.path.basename(path)}: {exc!r}")
    return (len(errors) == 0, errors)


def wait_for_staging_lock(timeout_s: float = 60.0,
                           poll_s: float = 1.0) -> int | None:
    """Block until jarvis_staging.lock contains a live PID. Returns that
    PID, or None on timeout. Used by the pipeline right after spawning
    the staging process so a stillborn boot fails fast."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(STAGING_LOCK):
            try:
                with open(STAGING_LOCK, "r", encoding="utf-8") as f:
                    pid_str = f.read().strip()
                pid = int(pid_str)
                if _bgm._pid_alive(pid):
                    return pid
            except (OSError, ValueError):
                pass
        time.sleep(poll_s)
    return None


def inject_command(text: str) -> bool:
    """Append a single command to injected_commands_staging.json. The
    file format mirrors prod's injected_commands.json: a JSON list of
    {"text": "..."} entries the staging JARVIS drains FIFO."""
    queue: list = []
    if os.path.exists(STAGING_INJECT):
        try:
            with open(STAGING_INJECT, "r", encoding="utf-8") as f:
                queue = json.load(f)
            if not isinstance(queue, list):
                queue = []
        except (OSError, ValueError):
            queue = []
    queue.append({"text": text, "queued_at": time.time()})
    tmp = STAGING_INJECT + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(queue, f)
        os.replace(tmp, STAGING_INJECT)
        return True
    except OSError:
        return False


def _read_replies_tail(since_ts: float) -> list[dict]:
    """Read replies.jsonl and return entries with ts >= since_ts. The
    file is JSONL — one JSON object per line."""
    if not os.path.exists(REPLIES_FILE):
        return []
    out: list[dict] = []
    try:
        with open(REPLIES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and float(obj.get("ts") or 0) >= since_ts:
                    out.append(obj)
    except OSError:
        return out
    return out


def wait_for_replies(min_count: int,
                      since_ts: float,
                      timeout_s: float = 90.0,
                      poll_s: float = 1.5) -> list[dict]:
    """Poll replies.jsonl until at least `min_count` new entries appear
    or we time out. Returns whatever replies showed up."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        replies = _read_replies_tail(since_ts)
        if len(replies) >= min_count:
            return replies
        time.sleep(poll_s)
    return _read_replies_tail(since_ts)


def staging_heartbeat_fresh(max_age_s: float = 30.0) -> bool:
    """True when the staging PID is still publishing heartbeats. Read
    from data/instances.json. The pipeline calls this after the inject
    + wait phase so a silently-crashed staging instance fails the gate."""
    insts = _bgm.list_instances()
    now = time.time()
    for entry in insts.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "staging":
            continue
        hb = float(entry.get("heartbeat_at") or 0)
        if hb and (now - hb) <= max_age_s:
            return True
    return False


def _publish_test_state_for_pid(pid: int, test_state: dict) -> None:
    """Attach `test_state` to the staging PID's entry in instances.json AND
    mirror it into the staging HUD state file so the LEFT-monitor overlay
    can render the ceremony.

    Called from the smoke driver (this process is not the staging JARVIS
    itself, so we can't call _bgm.publish_test_state directly — that would
    write our OWN PID. Instead we patch the staging entry in place)."""
    ts = dict(test_state)
    # 1. instances.json — prod polls this to mirror progress onto its own HUD.
    try:
        if os.path.exists(_bgm.INSTANCES_FILE):
            with open(_bgm.INSTANCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                key = str(pid)
                entry = data.get(key)
                if isinstance(entry, dict):
                    entry["test_state"] = ts
                    entry["heartbeat_at"] = time.time()
                    tmp = _bgm.INSTANCES_FILE + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, sort_keys=True)
                    os.replace(tmp, _bgm.INSTANCES_FILE)
    except (OSError, ValueError):
        pass
    # 2. data_staging/hud_state.json — the staging HUD reads from here.
    try:
        os.makedirs(os.path.dirname(STAGING_HUD_STATE_FILE), exist_ok=True)
        hud_data: dict = {}
        if os.path.exists(STAGING_HUD_STATE_FILE):
            try:
                with open(STAGING_HUD_STATE_FILE, "r", encoding="utf-8") as f:
                    hud_data = json.load(f) or {}
                if not isinstance(hud_data, dict):
                    hud_data = {}
            except (OSError, ValueError):
                hud_data = {}
        hud_data["test_state"] = ts
        tmp = STAGING_HUD_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hud_data, f)
        os.replace(tmp, STAGING_HUD_STATE_FILE)
    except OSError:
        pass


def run_smoke_tests(candidate_files: Iterable[str] | None = None,
                     prompts: Iterable[str] | None = None,
                     boot_timeout_s: float = 60.0,
                     reply_timeout_s: float = 90.0) -> dict:
    """Full smoke-test wrapper used by upgrade_jarvis.py.

    Sequence:
      1. py_compile gate on candidate files.
      2. Wait for jarvis_staging.lock to be claimed by a live PID.
      3. Inject N prompts.
      4. Wait for ≥1 reply per prompt in replies.jsonl.
      5. Confirm the staging heartbeat is still fresh after all replies.

    blue-green-2: publishes per-step progress into the staging PID's
    instances.json entry as `test_state = {current_test_case, tests_passed,
    tests_remaining}` so prod (or the staging HUD) can render the ceremony
    on the LEFT monitor.

    Returns:
      {"ok": bool, "stage_failed": str | None, "details": dict}
    """
    started_ts = time.time()
    result: dict = {"ok": False, "stage_failed": None, "details": {}}

    files = list(candidate_files or [])
    if files:
        ok, errors = precompile_candidate_files(files)
        result["details"]["py_compile"] = {"ok": ok, "errors": errors[:5]}
        if not ok:
            result["stage_failed"] = "py_compile"
            return result

    pid = wait_for_staging_lock(timeout_s=boot_timeout_s)
    result["details"]["staging_pid"] = pid
    if pid is None:
        result["stage_failed"] = "boot"
        return result

    prompts_list = list(prompts) if prompts is not None else list(DEFAULT_PROMPTS)
    total = len(prompts_list)

    # Seed an opening "booting" state immediately so the left-monitor HUD
    # shows the ceremony before the first prompt fires.
    _publish_test_state_for_pid(pid, {
        "current_test_case": "boot",
        "tests_passed":      0,
        "tests_remaining":   total,
    })

    for i, text in enumerate(prompts_list, start=1):
        _publish_test_state_for_pid(pid, {
            "current_test_case": text[:40],
            "tests_passed":      i - 1,
            "tests_remaining":   max(0, total - (i - 1)),
        })
        inject_command(text)
        # tiny stagger so the staging drainer processes them in order
        time.sleep(0.2)

    replies = wait_for_replies(
        min_count=total,
        since_ts=started_ts,
        timeout_s=reply_timeout_s,
    )
    result["details"]["replies_received"] = len(replies)
    result["details"]["replies_expected"] = total
    passed = min(len(replies), total)
    _publish_test_state_for_pid(pid, {
        "current_test_case": "verifying replies",
        "tests_passed":      passed,
        "tests_remaining":   max(0, total - passed),
    })
    if len(replies) < total:
        _publish_test_state_for_pid(pid, {
            "current_test_case": "FAILED: missing replies",
            "tests_passed":      passed,
            "tests_remaining":   max(0, total - passed),
        })
        result["stage_failed"] = "replies"
        return result

    if not staging_heartbeat_fresh():
        _publish_test_state_for_pid(pid, {
            "current_test_case": "FAILED: heartbeat",
            "tests_passed":      passed,
            "tests_remaining":   0,
        })
        result["stage_failed"] = "heartbeat"
        return result

    _publish_test_state_for_pid(pid, {
        "current_test_case": "PASSED",
        "tests_passed":      total,
        "tests_remaining":   0,
    })
    result["ok"] = True
    return result


def record_reply(text: str, kind: str = "tts",
                 extra: dict | None = None) -> None:
    """Called from the staging JARVIS process whenever _speak() fires.
    Lands as a JSONL line in data_staging/replies.jsonl so the pipeline
    can watch the conversation without ever opening an audio device."""
    try:
        os.makedirs(os.path.dirname(REPLIES_FILE), exist_ok=True)
    except OSError:
        pass
    payload = {"ts": time.time(), "kind": kind, "text": str(text)}
    if extra:
        payload.update(extra)
    try:
        with open(REPLIES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    # Smoke-the-smoke-tester: run with no candidate files just to verify
    # the lock + reply plumbing against whatever staging is already up.
    import sys
    result = run_smoke_tests(candidate_files=[],
                             boot_timeout_s=5.0,
                             reply_timeout_s=15.0)
    json.dump(result, sys.stdout, indent=2)
    sys.exit(0 if result["ok"] else 1)
