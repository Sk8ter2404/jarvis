#!/usr/bin/env python3
"""
JARVIS post-pipeline stability smoke test.

Launches JARVIS via _boot_jarvis.ps1, lets it run for STABILITY_WAIT_SECONDS
(default 300 s = 5 min), then checks three signals:

    1. JARVIS process still alive (PID from jarvis.lock found in tasklist).
    2. No APPCRASH events in the Windows Application Log since launch.
    3. No `[FATAL]` lines in the freshest session log under C:\\JARVIS\\logs.

If every signal passes, writes stability_smoke_PASS.json and exits 0.
Otherwise writes stability_smoke_FAIL.json with per-check details and exits 1.

Designed as the final post-pipeline gate: run it once the crash-fix-* tasks
have landed, before declaring a build trustworthy.

    python tools/stability_smoke_test.py
    python tools/stability_smoke_test.py --wait 120     # shorter dry-run
    python tools/stability_smoke_test.py --no-launch    # use whatever's running
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT_SCRIPT = os.path.join(ROOT, "_boot_jarvis.ps1")
LOCK_FILE = os.path.join(ROOT, "jarvis.lock")
BOOT_ERR_FILE = os.path.join(ROOT, "jarvis_boot_error.txt")
LOGS_DIR = os.path.join(ROOT, "logs")
PASS_REPORT = os.path.join(ROOT, "stability_smoke_PASS.json")
FAIL_REPORT = os.path.join(ROOT, "stability_smoke_FAIL.json")

DEFAULT_WAIT_SECONDS = 300
LOCK_POLL_TIMEOUT = 30
LOCK_POLL_INTERVAL = 1.0


def _launch_jarvis() -> tuple[str, str]:
    """Spawn _boot_jarvis.ps1 in a detached PowerShell. Returns the booter's
    captured (stdout, stderr) so a failing lock-wait can surface what the
    booter actually said (e.g. 'WARNING: ANTHROPIC_API_KEY not found')."""
    if not os.path.isfile(BOOT_SCRIPT):
        raise FileNotFoundError(f"boot script missing: {BOOT_SCRIPT}")
    r = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            # -Headless: the pipeline's smoke boots must be SILENT + DEAF (muted
            # TTS, mic forced off) so a test instance never speaks over the user
            # or acts on ambient audio mid-run. PROD lock/logs are unchanged.
            "-File", BOOT_SCRIPT, "-Headless",
        ],
        cwd=ROOT,
        timeout=60,
        check=False,
        capture_output=True,
        text=True,
    )
    return (r.stdout or "", r.stderr or "")


def _list_jarvis_processes() -> list[str]:
    """Return short descriptions of any python(w).exe processes whose command
    line references bobert_companion — so a lock-wait failure can tell the
    difference between 'JARVIS never started' and 'JARVIS started but never
    wrote its lock'."""
    try:
        r = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe' } | "
                "Where-Object { $_.CommandLine -like '*bobert_companion*' } | "
                "ForEach-Object { \"$($_.ProcessId) $($_.Name) :: $($_.CommandLine)\" }",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"<process probe failed: {exc!r}>"]
    if r.returncode != 0:
        return [f"<process probe rc={r.returncode}: {r.stderr.strip()[:200]}>"]
    out = (r.stdout or "").strip()
    return [line for line in out.splitlines() if line.strip()] or ["<no bobert_companion processes>"]


def _latest_log_tail(n: int = 40) -> dict:
    """Return {'path': ..., 'tail': [...]} for the freshest session log — used
    to diagnose what JARVIS got stuck on if the lock never appeared."""
    path = _latest_session_log()
    if path is None:
        return {"path": None, "tail": []}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {"path": path, "tail": [ln.rstrip("\n") for ln in lines[-n:]]}
    except OSError as exc:
        return {"path": path, "tail": [f"<could not read: {exc}>"]}


def _read_lock_pid() -> int | None:
    """Return the PID recorded in jarvis.lock, or None if absent/unparseable."""
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_boot_error() -> str:
    """Return the contents of jarvis_boot_error.txt, written by
    bobert_companion._early_boot_singleton_lock() when the lock-file write
    fails after retries. Empty string if no marker was left behind."""
    try:
        with open(BOOT_ERR_FILE, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _wait_for_lock_pid(timeout: float) -> int | None:
    """Poll the lock file up to `timeout` seconds for a PID to appear.

    Short-circuits if jarvis_boot_error.txt appears — that means the
    early-boot singleton lock decided to fast-fail and there's no point
    waiting the full timeout for a lock that will never materialise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = _read_lock_pid()
        if pid is not None:
            return pid
        # Fast path: a non-empty boot_error.txt means JARVIS already
        # decided it couldn't acquire the lock. Bail out so the test
        # surfaces the cause in <1s instead of waiting 30s.
        if _read_boot_error():
            return None
        time.sleep(LOCK_POLL_INTERVAL)
    return None


def _pid_alive(pid: int) -> bool:
    """tasklist-based liveness check, matching the singleton logic in
    bobert_companion._enforce_singleton."""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return str(pid) in r.stdout and "python" in r.stdout.lower()


def _latest_session_log() -> str | None:
    """Newest C:\\JARVIS\\logs\\session_*.log by mtime, or None if logs/
    is empty/missing."""
    if not os.path.isdir(LOGS_DIR):
        return None
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(LOGS_DIR):
        if not (name.startswith("session_") and name.endswith(".log")):
            continue
        path = os.path.join(LOGS_DIR, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


CRASH_TRACES_LOG = os.path.join(LOGS_DIR, "crash_traces.log")


def _scan_crash_traces_since(since_epoch: float) -> dict:
    """Scan crash_traces.log for faulthandler dumps written after `since_epoch`.

    The session-log scan catches Python-level [FATAL] / Traceback hits, but
    native crashes (SIGSEGV inside a C extension — sounddevice, numpy, opencv,
    comtypes) bypass the session log entirely and only land here. JARVIS's
    setup_logging() routes faulthandler.enable(file=...) at this exact path;
    any line starting with 'Fatal Python error:', 'Windows fatal exception:',
    or 'Thread 0x...' inside a faulthandler dump marks the start of a crash.

    Returns:
        {"new_dumps": [...], "head_signatures": [...], "since_epoch": ...}

    Discovered 2026-05-30 by tracing a real silent crash where JARVIS died
    mid-daily-briefing inside core/audio_processor.py:_ns while formatting a
    numpy exception; session log ended at 08:30:21 with no fatal marker, but
    crash_traces.log got the full thread dump 4s later. The scan-session-only
    gate said `ok=True` for that boot even though the process was dead.
    """
    result: dict = {
        "path": CRASH_TRACES_LOG,
        "new_dumps": [],
        "head_signatures": [],
        "since_epoch": since_epoch,
    }
    if not os.path.exists(CRASH_TRACES_LOG):
        return result
    try:
        mtime = os.path.getmtime(CRASH_TRACES_LOG)
    except OSError:
        return result
    # If the file hasn't been touched since launch, no new dump.
    if mtime < since_epoch:
        return result
    try:
        with open(CRASH_TRACES_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        result["new_dumps"].append(f"<could not read crash_traces.log: {exc}>")
        return result

    # Faulthandler doesn't timestamp its dumps, so we can't perfectly bound
    # 'since launch' by line. Instead, return the LAST dump as a proxy —
    # faulthandler appends, so the bottom is the most recent crash. A dump
    # is bracketed by 'Fatal Python error:' / 'Windows fatal exception:' at
    # its start and 'Current thread' / blank line at its end.
    dump_starts: list[int] = []
    for i, ln in enumerate(lines):
        if (ln.startswith("Fatal Python error:")
                or ln.startswith("Windows fatal exception:")
                or ln.startswith("Thread 0x")):
            # Only treat 'Thread 0x' as a dump start if the previous line
            # is empty or the file just began — otherwise we'd flag every
            # thread inside a single dump.
            if ln.startswith("Thread 0x"):
                if i > 0 and lines[i - 1].strip():
                    continue
            dump_starts.append(i)
    if not dump_starts:
        return result

    last_start = dump_starts[-1]
    # Take up to 40 lines from the last dump for the report.
    dump = "".join(lines[last_start:last_start + 40])
    result["new_dumps"].append(dump.rstrip())
    # Extract a one-line signature for the regression task summary.
    # We want the TOP-of-stack JARVIS frame in the "Current thread" block
    # (which is where the crash actually happened), NOT the deepest one
    # (which is just <module> at boot). faulthandler prints
    # 'Current thread 0x... (most recent call first):' as the section
    # header, then frames in order from top (crash site) to bottom
    # (<module>). So the FIRST JARVIS frame inside that block is the
    # site we want.
    in_current_thread = False
    for ln in lines[last_start:last_start + 40]:
        stripped = ln.strip()
        if stripped.startswith("Current thread "):
            in_current_thread = True
            continue
        if not in_current_thread:
            continue
        if not stripped.startswith("File "):
            continue
        if "JARVIS" in stripped and "in <module>" not in stripped:
            # First JARVIS frame inside the current thread = crash site.
            result["head_signatures"].append(stripped[:200])
            break
    # Fallback: if 'Current thread' wasn't present (rare on older dumps),
    # take the topmost JARVIS frame in the whole dump that isn't <module>.
    if not result["head_signatures"]:
        for ln in lines[last_start:last_start + 40]:
            stripped = ln.strip()
            if not stripped.startswith("File "):
                continue
            if "JARVIS" in stripped and "in <module>" not in stripped:
                result["head_signatures"].append(stripped[:200])
                break
    return result


def _scan_log_for_fatal(path: str) -> list[str]:
    """Return up to 20 lines containing the literal token `[FATAL]`.

    Kept as a thin wrapper for back-compat — `_scan_log_for_issues` does
    the categorized scan that the reviewer actually wants. The pipeline
    still calls this for the strict pass/fail gate; the richer scan goes
    into the failure report so a regression task description has the
    runtime context, not just 'no fatal lines found'.
    """
    return _scan_log_for_issues(path).get("fatal", [])


def _scan_log_for_issues(path: str) -> dict:
    """Categorized log scan — returns a dict with per-category hits so the
    reviewer/regression-task description has more than `[FATAL]` to work
    with. The previous `_scan_log_for_fatal` only flagged hard fatals;
    everything else (Python tracebacks, skill load failures, STT/cuda
    errors, missing deps) flew under the radar and forced the reviewer to
    guess from a thin log tail.

    Categories returned:
      fatal              — `[FATAL]` lines (the hard-fail signal)
      tracebacks         — multi-line Python tracebacks (head + tail lines)
      errors             — `[ERROR]` / `ERROR:` / `error:` (lowercased) lines
      skill_failures     — `[skill] X: failed` or `load_skills exception`
      transcribe_failures — `[transcribe] failed` (cublas dll miss, etc.)
      missing_deps       — `not installed` / `pip install` / `not available`
      native_crashes     — faulthandler-style `Thread 0x...` headers,
                           `Segmentation fault`, `SIGSEGV`, `SIGABRT`
      boot_milestones    — booleans for known healthy-boot markers we
                           expect to see in the first ~120s of a normal
                           boot (Listening, skills loaded, HUD spawned)
      ok                 — True ONLY when fatal + native_crashes + at
                           least one tracebacks entry are all absent AND
                           every boot_milestones flag is True (so a boot
                           that hangs before "Listening..." still fails
                           the gate even with no [FATAL] line)
    """
    findings: dict = {
        "fatal": [],
        "tracebacks": [],
        "errors": [],
        "skill_failures": [],
        "transcribe_failures": [],
        "missing_deps": [],
        "native_crashes": [],
        "boot_milestones": {
            "listening": False,        # main loop reached
            "skills_loaded": False,    # at least one [skill] X: added actions
            "diag_daemons": False,     # 4/4 daemons running
            "hud_spawned": False,      # [hud] launched
            "faulthandler": False,     # crash handler armed
        },
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except OSError as exc:
        findings["fatal"].append(f"<could not read log: {exc}>")
        findings["ok"] = False
        return findings

    in_traceback = False
    traceback_buf: list[str] = []
    for ln in lines:
        # Strict fatal — primary pass/fail signal, preserved verbatim.
        if "[FATAL]" in ln:
            if len(findings["fatal"]) < 20:
                findings["fatal"].append(ln)

        # Boot milestones (positive signals — track whichever we saw).
        ms = findings["boot_milestones"]
        # "Listening..." is the marker that the main loop reached the
        # record→transcribe→speak cycle. JARVIS writes it with a Unicode
        # ellipsis (U+2026), so accept either glyph variant.
        if "Listening" in ln and ("..." in ln or "…" in ln):
            ms["listening"] = True
        if "[skill]" in ln and "added actions" in ln:
            ms["skills_loaded"] = True
        if "[diag-daemons]" in ln and "4/4 daemons running" in ln:
            ms["diag_daemons"] = True
        if "[hud]" in ln and "launched" in ln:
            ms["hud_spawned"] = True
        if "[faulthandler]" in ln and "enabled" in ln:
            ms["faulthandler"] = True

        # Multi-line Python traceback capture.
        if "Traceback (most recent call last):" in ln:
            in_traceback = True
            traceback_buf = [ln]
            continue
        if in_traceback:
            traceback_buf.append(ln)
            # End-of-traceback heuristic: the last line is `<ExcName>:
            # ...` (no leading indent). Cap each entry at 12 lines so
            # one runaway exception doesn't fill the report.
            if ln and not ln.startswith(" ") and not ln.startswith("\t"):
                if len(findings["tracebacks"]) < 10:
                    findings["tracebacks"].append(
                        "\n".join(traceback_buf[:12]))
                in_traceback = False
                traceback_buf = []
                continue
            if len(traceback_buf) > 30:
                # Defensive cap if the end-marker never matches.
                if len(findings["tracebacks"]) < 10:
                    findings["tracebacks"].append(
                        "\n".join(traceback_buf[:12]) + "\n…(truncated)")
                in_traceback = False
                traceback_buf = []
                continue

        # Generic error / warning patterns.
        lowered = ln.lower()
        if ("[error]" in lowered or "error:" in lowered) and "[FATAL]" not in ln:
            if len(findings["errors"]) < 30:
                findings["errors"].append(ln)
        # Skill registration failures.
        if "[skill]" in ln and ("failed" in lowered or "exception" in lowered):
            if len(findings["skill_failures"]) < 15:
                findings["skill_failures"].append(ln)
        # STT/transcribe failures (e.g. cublas64_12.dll missing).
        if "[transcribe]" in ln and "failed" in lowered:
            if len(findings["transcribe_failures"]) < 15:
                findings["transcribe_failures"].append(ln)
        # Missing dep / install hint surfaced by skills.
        if any(kw in lowered for kw in (
            "not installed", "pip install", "not available", "modulenotfounderror",
        )):
            if len(findings["missing_deps"]) < 15:
                findings["missing_deps"].append(ln)
        # Native crashes — faulthandler dumps thread headers + 'Segfault'.
        if (ln.startswith("Thread 0x") or "Segmentation fault" in ln
                or "SIGSEGV" in ln or "SIGABRT" in ln):
            if len(findings["native_crashes"]) < 15:
                findings["native_crashes"].append(ln)

    # Overall gate logic — strict, but more discerning than before:
    #   • Any [FATAL] line → FAIL.
    #   • Any native-crash signature (SIGSEGV / faulthandler dump) → FAIL.
    #   • Any Python traceback in the log → FAIL.
    #   • Missing the "Listening..." milestone after a full wait window →
    #     FAIL (boot hung before main loop). The other milestones are
    #     reported for context but don't block the gate on their own,
    #     since some installs legitimately skip the HUD subprocess.
    ms = findings["boot_milestones"]
    findings["ok"] = (
        not findings["fatal"]
        and not findings["native_crashes"]
        and not findings["tracebacks"]
        and bool(ms["listening"])
    )
    return findings


def _check_appcrash_events(since: datetime) -> dict:
    """Query the Windows Application Event Log for APPCRASH-class events
    raised at or after `since`. Returns a dict with `ok`, `count`,
    `events` (up to 10), and optional `error` (e.g. PowerShell missing,
    access denied).

    Event source 'Application Error' / event ID 1000 covers user-mode
    crashes (APPCRASH bucket). We also include the .NET Runtime fatal
    bucket (event ID 1026)."""
    ps_since = since.strftime("%Y-%m-%dT%H:%M:%S")
    script = (
        "$ErrorActionPreference = 'Stop';"
        f"$since = [datetime]'{ps_since}';"
        "$filter = @{LogName='Application';"
        " ProviderName=@('Application Error','.NET Runtime');"
        " StartTime=$since};"
        "try {"
        "  $evts = Get-WinEvent -FilterHashtable $filter -ErrorAction Stop;"
        "} catch [System.Exception] {"
        "  if ($_.Exception.Message -match 'No events were found') {"
        "    $evts = @();"
        "  } else { throw }"
        "};"
        "$evts |"
        " Select-Object -First 10 TimeCreated,Id,ProviderName,"
        "    @{Name='Message';Expression={($_.Message -split \"`n\")[0]}} |"
        " ConvertTo-Json -Depth 3 -Compress"
    )
    try:
        r = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command", script,
            ],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "count": 0, "events": [],
                "error": f"powershell invocation failed: {exc!r}"}
    if r.returncode != 0:
        return {"ok": False, "count": 0, "events": [],
                "error": f"Get-WinEvent failed: {r.stderr.strip()[:500]}"}
    raw = r.stdout.strip()
    if not raw:
        return {"ok": True, "count": 0, "events": []}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "count": 0, "events": [],
                "error": f"could not parse PS output ({exc}): {raw[:300]}"}
    if isinstance(parsed, dict):
        parsed = [parsed]
    return {"ok": len(parsed) == 0, "count": len(parsed), "events": parsed}


def _write_report(path: str, payload: dict) -> None:
    other = FAIL_REPORT if path == PASS_REPORT else PASS_REPORT
    try:
        if os.path.exists(other):
            os.remove(other)
    except OSError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wait", type=int, default=DEFAULT_WAIT_SECONDS,
        help="seconds to let JARVIS run before sampling (default 300)",
    )
    ap.add_argument(
        "--no-launch", action="store_true",
        help="skip the boot step; assume JARVIS is already running",
    )
    args = ap.parse_args(argv)

    launched_at = datetime.now()
    pre_launch_log = _latest_session_log()
    boot_stdout = ""
    boot_stderr = ""

    if not args.no_launch:
        print(f"[stability] launching JARVIS via {BOOT_SCRIPT}")
        try:
            boot_stdout, boot_stderr = _launch_jarvis()
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            payload = {
                "result": "FAIL",
                "phase": "launch",
                "launched_at": launched_at.isoformat(timespec="seconds"),
                "error": f"could not launch booter: {exc!r}",
            }
            _write_report(FAIL_REPORT, payload)
            print(f"[stability] FAIL — {payload['error']}")
            return 1

    print(f"[stability] waiting up to {LOCK_POLL_TIMEOUT}s for jarvis.lock to appear")
    pid = _wait_for_lock_pid(LOCK_POLL_TIMEOUT)
    if pid is None:
        # Surface enough state to tell whether JARVIS never started, started but
        # crashed before writing the lock, or started and is still loading.
        boot_error = _read_boot_error()
        payload = {
            "result": "FAIL",
            "phase": "lock_wait",
            "launched_at": launched_at.isoformat(timespec="seconds"),
            "error": f"jarvis.lock never appeared within {LOCK_POLL_TIMEOUT}s",
            "boot_stdout_tail": boot_stdout[-2000:] if boot_stdout else "",
            "boot_stderr_tail": boot_stderr[-2000:] if boot_stderr else "",
            "early_boot_error": boot_error,
            "jarvis_processes": _list_jarvis_processes(),
            "latest_session_log": _latest_log_tail(40),
        }
        _write_report(FAIL_REPORT, payload)
        print(f"[stability] FAIL — {payload['error']}")
        if boot_error:
            print(f"[stability]   early-boot error marker: {boot_error[:400]}")
        if payload["jarvis_processes"]:
            print("[stability]   processes seen:")
            for line in payload["jarvis_processes"][:5]:
                print(f"    {line[:200]}")
        log_info = payload["latest_session_log"]
        if log_info.get("path"):
            print(f"[stability]   newest session log: {log_info['path']}")
            for line in log_info.get("tail", [])[-10:]:
                print(f"    {line[:200]}")
        return 1
    print(f"[stability] JARVIS PID {pid} recorded in lock; sleeping {args.wait}s")

    time.sleep(args.wait)

    checks: dict = {}

    # 1. Process liveness.
    alive = _pid_alive(pid)
    checks["process_alive"] = {"ok": alive, "pid": pid}

    # 2. APPCRASH / fatal CLR events since launch (small slack for clock skew).
    window_start = launched_at - timedelta(seconds=10)
    checks["appcrash_events"] = _check_appcrash_events(window_start)

    # 4. crash_traces.log scan — catches native crashes (SIGSEGV in C
    # extensions like sounddevice / numpy / opencv) that bypass the
    # session-log [FATAL] gate. The session log's main loop dies
    # before excepthook can write anything, but faulthandler is wired
    # to a dedicated fd that survives the C-level abort and dumps the
    # full thread trace 1-4s later. Skipping this check is how the
    # 08:30 2026-05-30 crash slipped past the gate.
    crash_dump = _scan_crash_traces_since(launched_at.timestamp() - 10)
    checks["crash_traces"] = {
        "path": crash_dump["path"],
        "ok": not crash_dump["new_dumps"],
        "new_dumps": crash_dump["new_dumps"],
        "head_signatures": crash_dump["head_signatures"],
    }

    # 3. Categorized session-log scan (fatal + tracebacks + skill failures +
    # transcribe failures + missing deps + native crashes + positive boot
    # milestones). The reviewer/regression-task description gets every
    # category so it has the runtime context to plan the next iteration
    # without an additional log-read round-trip.
    session_log = _latest_session_log()
    log_info: dict = {"path": session_log}
    if session_log is None:
        log_info["ok"] = False
        log_info["error"] = "no session log files found under logs/"
    else:
        if session_log == pre_launch_log:
            log_info["warning"] = (
                "freshest session log predates launch — JARVIS may not have"
                " written one yet; scanning it anyway"
            )
        scan = _scan_log_for_issues(session_log)
        # Preserve the legacy fields so any external consumers reading the
        # report by name (dashboards, the pipeline reviewer's prompt) keep
        # working — but enrich with the new categories underneath them.
        log_info["fatal_count"] = len(scan.get("fatal", []))
        log_info["fatal_lines"] = scan.get("fatal", [])
        log_info["tracebacks"]           = scan.get("tracebacks", [])
        log_info["errors"]               = scan.get("errors", [])
        log_info["skill_failures"]       = scan.get("skill_failures", [])
        log_info["transcribe_failures"]  = scan.get("transcribe_failures", [])
        log_info["missing_deps"]         = scan.get("missing_deps", [])
        log_info["native_crashes"]       = scan.get("native_crashes", [])
        log_info["boot_milestones"]      = scan.get("boot_milestones", {})
        log_info["ok"]                   = bool(scan.get("ok", False))
    checks["session_log_fatal"] = log_info

    all_ok = all(c.get("ok", False) for c in checks.values())

    payload = {
        "result": "PASS" if all_ok else "FAIL",
        "launched_at": launched_at.isoformat(timespec="seconds"),
        "sampled_at": datetime.now().isoformat(timespec="seconds"),
        "wait_seconds": args.wait,
        "pid": pid,
        "checks": checks,
    }

    if all_ok:
        _write_report(PASS_REPORT, payload)
        print(f"[stability] PASS — wrote {PASS_REPORT}")
        return 0

    _write_report(FAIL_REPORT, payload)
    print(f"[stability] FAIL — wrote {FAIL_REPORT}")
    for name, info in checks.items():
        if not info.get("ok", False):
            print(f"  - {name}: {info}")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint, exercised via main()
    sys.exit(main())
