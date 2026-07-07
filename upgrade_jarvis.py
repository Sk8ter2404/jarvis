#!/usr/bin/env python3
"""
JARVIS Upgrade Pipeline
─────────────────────────
Invoked by JARVIS's `upgrade` action or run manually.

Flow:
1. Snapshot current code into backups/<timestamp>/ (last 5 kept)
2. Read all unticked items from jarvis_todo.md
3. Kill any running JARVIS instance
4. Spawn Claude Code CLI (--print --dangerously-skip-permissions) in a
   new visible PowerShell window with the pending queue as the prompt;
   Claude Code exits automatically when the work is done.
5. After it exits, this script runs a final py_compile sweep of every
   touched file, writes .last_upgrade_summary.json so the next JARVIS
   start can announce what changed, and (if --relaunch) launches JARVIS.

Manual usage:
    python upgrade_jarvis.py                # multi-agent pipeline (default).
                                            # 4 stages per task: planner →
                                            # implementer → reviewer → tester.
    python upgrade_jarvis.py --relaunch     # auto-relaunch JARVIS after upgrade
    python upgrade_jarvis.py --dry-audit    # read-only preview of what Claude would do
                                            # (no changes, no backup, no JARVIS kill)
    python upgrade_jarvis.py --single-stage # legacy fallback: one Claude invocation
                                            # per task, no review/test gates. Faster
                                            # and cheaper but no regression guards —
                                            # for emergencies only.

Pipeline model knobs (env vars, optional):
    JARVIS_PIPELINE_PLANNER_MODEL       default "haiku"
    JARVIS_PIPELINE_IMPLEMENTER_MODEL   default "opus"
    JARVIS_PIPELINE_REVIEWER_MODEL      default "sonnet"
    JARVIS_PIPELINE_TESTER_WAIT_S       default 90
    JARVIS_PIPELINE_SKIP_TESTER         "1" to skip the tester stage (debug)

Requires:
    - `claude` CLI installed and on PATH (Claude Code)
    - A Claude Max subscription — the pipeline strips ANTHROPIC_API_KEY and
      runs Claude Code on the Max plan, NOT on metered API credits.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

# Make all console output UTF-8-safe. The pipeline's status lines use arrows
# ('→'), box-drawing chars, etc., and a forced one-shot gate run on 2026-05-31
# crashed with UnicodeEncodeError on a '→' in a snapshot print because stdout
# was a redirected file under Windows cp1252. errors='replace' degrades an
# unmappable glyph to '?' instead of taking the whole pipeline down mid-cycle.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    import blue_green_manager as _bgm  # type: ignore
except Exception as _bge:
    _bgm = None  # type: ignore
    print(f"[upgrade] blue_green_manager unavailable: {_bge!r}")

try:
    import staging_instance as _stg  # type: ignore
except Exception as _sge:
    _stg = None  # type: ignore
    print(f"[upgrade] staging_instance unavailable: {_sge!r}")

PROJECT_DIR          = os.path.dirname(os.path.abspath(__file__))
TODO_FILE            = os.path.join(PROJECT_DIR, "jarvis_todo.md")
SCRIPT               = os.path.join(PROJECT_DIR, "bobert_companion.py")
UPGRADE_LOG          = os.path.join(PROJECT_DIR, "upgrade_log.txt")
UPGRADE_STREAM_LOG   = os.path.join(PROJECT_DIR, "upgrade_stream.log")
BACKUP_DIR           = os.path.join(PROJECT_DIR, "backups")
UPGRADE_SUMMARY_FILE = os.path.join(PROJECT_DIR, ".last_upgrade_summary.json")
CHANGELOG_FILE       = os.path.join(PROJECT_DIR, "CHANGELOG.md")
VERSION_FILE         = os.path.join(PROJECT_DIR, "data", "version.json")
STABILITY_GATES_LOG  = os.path.join(PROJECT_DIR, "data", "stability_gates.jsonl")
GATE_SNAPSHOT_ROOT   = os.path.join(PROJECT_DIR, "backups")
BOOT_SCRIPT          = os.path.join(PROJECT_DIR, "_boot_jarvis.ps1")
LOCK_FILE            = os.path.join(PROJECT_DIR, "jarvis.lock")
STABILITY_SMOKE_TOOL = os.path.join(PROJECT_DIR, "tools", "stability_smoke_test.py")
MAX_BACKUPS          = 5   # keep the last 5 snapshots


def _gate_config() -> tuple[int, int, bool]:
    """Return (interval, duration_s, disabled) for the stability gate.
    Env vars: STABILITY_GATE_INTERVAL (default 10), STABILITY_GATE_DURATION_S
    (default 300), STABILITY_GATE_DISABLE ('1' to skip — emergency only)."""
    try:
        interval = max(1, int(os.environ.get("STABILITY_GATE_INTERVAL", "10")))
    except ValueError:
        interval = 10
    try:
        duration = max(30, int(os.environ.get("STABILITY_GATE_DURATION_S", "300")))
    except ValueError:
        duration = 300
    disabled = os.environ.get("STABILITY_GATE_DISABLE", "").strip() == "1"
    return (interval, duration, disabled)


# Note: no module-level cache of (interval, duration_s). Every call site reads
# _gate_config() fresh so STABILITY_GATE_INTERVAL / STABILITY_GATE_DURATION_S /
# STABILITY_GATE_DISABLE can be toggled mid-session (e.g. by an operator
# tweaking the env var between pipeline runs) without re-importing.

TASK_RE = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)
DONE_RE = re.compile(r"^- \[x\] (.+)$", re.MULTILINE)


def _bump_version(current: str, has_features: bool) -> str:
    """Bump semver. Patch for bug-fix-only runs, minor when any wish/research feature lands.
    Major is never auto-bumped (reserved for the user)."""
    try:
        major, minor, patch = [int(x) for x in current.split(".")[:3]]
    except Exception:
        return "1.0.1"
    if has_features:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _classify_completed(titles: list[str]) -> dict[str, list[str]]:
    """Group completed-task titles by category for changelog readability."""
    out = {"audit": [], "wish": [], "research": [], "self_diag": [], "other": []}
    for t in titles:
        low = t.lower()
        if "[audit" in low or "audit-v2" in low:
            out["audit"].append(t)
        elif "[self-diag" in low or "self-diag" in low:
            out["self_diag"].append(t)
        elif "research-" in low or "[top priority]" in low or "[high priority]" in low:
            out["research"].append(t)
        elif " wish" in low or "wish-list" in low:
            out["wish"].append(t)
        else:
            out["other"].append(t)
    return out


def _short_title(line: str) -> str:
    """Strip checkbox / bold-dates / category tags / DONE-suffix; keep first ~160 chars."""
    s = re.sub(r"^- \[x\] ", "", line)
    s = re.sub(r"\*\*[^*]+\*\*", "", s)
    s = re.sub(r"\[(audit-v2|research-?\w*|self-diag|wish|TOP PRIORITY|HIGH PRIORITY)\]",
               "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+✓ DONE.*$", "", s)
    s = re.sub(r"  +", " ", s).strip(" -—.")
    return s[:160].strip()


def _append_changelog_entry(tasks_before: list[str], syntax_ok: bool,
                            audit_p0: int, audit_p1: int, audit_p2: int) -> None:
    """Write a new dated, versioned CHANGELOG.md entry for this pipeline run.

    Diffs the unticked-tasks-before-run against the post-run jarvis_todo.md
    state to identify what was completed THIS run, categorises them, bumps the
    version, and prepends the entry to CHANGELOG.md (so newest is on top, after
    the file header)."""
    try:
        # Read the TODO file ONCE so both regex passes see a single,
        # consistent snapshot (a concurrent edit between two reads would
        # otherwise yield an inconsistent changelog).
        with open(TODO_FILE, "r", encoding="utf-8") as _tf:
            todo_text = _tf.read()
        # What's done now?
        done_titles = [_short_title(line) for line in
                       DONE_RE.findall(todo_text)]
        # Tasks the pipeline saw at start vs unfinished now:
        still_open = set(TASK_RE.findall(todo_text))
        completed_this_run = [t for t in tasks_before if t not in still_open]
        # Filter the full done-titles to ones that match the completed-this-run set
        completed_titles = []
        for ct in completed_this_run:
            short = _short_title("- [x] " + ct)
            completed_titles.append(short)

        if not completed_titles:
            return    # nothing to log

        # Load + bump version
        version_data: dict = {}
        if os.path.exists(VERSION_FILE):
            try:
                with open(VERSION_FILE, "r", encoding="utf-8") as _vf:
                    version_data = json.load(_vf)
            except Exception:
                pass
        cur_ver = version_data.get("version", "1.0.0")
        bucketed = _classify_completed(completed_titles)
        has_features = bool(bucketed["wish"] or bucketed["research"])
        new_ver = _bump_version(cur_ver, has_features)

        # Build entry
        now_str = time.strftime("%Y-%m-%d %H:%M")
        lines = [
            f"## v{new_ver} — {now_str}",
            "",
            f"**{len(completed_titles)} task(s) completed this run.**",
            "",
        ]
        if not syntax_ok:
            lines.append(f"⚠  Syntax check FAILED — review backups before relaunch.")
            lines.append("")
        if audit_p0:
            lines.append(f"⚠  Auditor found {audit_p0} P0 finding(s) post-run.")
            lines.append("")
        if audit_p1:
            lines.append(f"ℹ  Auditor found {audit_p1} P1 finding(s) post-run.")
            lines.append("")

        labels = [("research", "Research / big features"),
                  ("wish",     "Wish-list features"),
                  ("audit",    "Bug fixes (audit-v2)"),
                  ("self_diag","Self-healing fixes"),
                  ("other",    "Other")]
        for key, label in labels:
            items = bucketed[key]
            if not items:
                continue
            lines.append(f"### {label} ({len(items)})")
            lines.append("")
            for it in items[:30]:
                lines.append(f"- {it}")
            if len(items) > 30:
                lines.append(f"- _(...and {len(items) - 30} more)_")
            lines.append("")
        lines.append("---")
        lines.append("")
        entry = "\n".join(lines)

        # Prepend new entry to CHANGELOG.md, keeping the original header intact
        header_marker = "---"
        old = ""
        if os.path.exists(CHANGELOG_FILE):
            with open(CHANGELOG_FILE, "r", encoding="utf-8") as _cf:
                old = _cf.read()
        if header_marker in old:
            header, rest = old.split(header_marker, 1)
            new_body = header + header_marker + "\n\n" + entry + rest.lstrip("\n")
        else:
            new_body = ("# JARVIS Changelog\n\nAuto-appended on every upgrade run.\n\n---\n\n"
                        + entry + old)

        # Atomic write via tempfile in same dir
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix=".changelog_", dir=PROJECT_DIR, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as _wf:
            _wf.write(new_body)
        os.replace(tmp, CHANGELOG_FILE)

        # Update version.json
        version_data["version"] = new_ver
        version_data["last_upgrade_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        version_data["last_upgrade_unix"] = int(time.time())
        version_data["last_pipeline_tasks_count"] = len(completed_titles)
        version_data["total_tasks_completed_lifetime"] = len(done_titles)
        os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".version_", dir=os.path.dirname(VERSION_FILE), text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as _wf:
            json.dump(version_data, _wf, indent=2)
        os.replace(tmp, VERSION_FILE)

        print(f"  CHANGELOG: bumped {cur_ver} → {new_ver} ({len(completed_titles)} tasks)")
    except Exception as _ce:
        print(f"  Couldn't update CHANGELOG: {_ce}")


def backup_codebase() -> str:
    """Snapshot all key source files into backups/<timestamp>/ before any
    upgrade runs. Keeps MAX_BACKUPS most-recent snapshots; prunes older ones.
    Returns the backup folder path."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    dest = os.path.join(BACKUP_DIR, ts)
    os.makedirs(dest, exist_ok=True)

    # Files to always back up
    always = [
        "bobert_companion.py",
        "overnight_upgrade.py",
        "upgrade_jarvis.py",
        "jarvis_todo.md",
    ]
    for fname in always:
        src = os.path.join(PROJECT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, fname))

    # Entire skills/ directory
    skills_src = os.path.join(PROJECT_DIR, "skills")
    if os.path.isdir(skills_src):
        shutil.copytree(skills_src, os.path.join(dest, "skills"), dirs_exist_ok=True)

    # Entire hud/ directory if it exists
    hud_src = os.path.join(PROJECT_DIR, "hud")
    if os.path.isdir(hud_src):
        shutil.copytree(hud_src, os.path.join(dest, "hud"), dirs_exist_ok=True)

    # core/ and tools/ too — the pipeline edits them and _revert_to_snapshot()
    # restores them, so they MUST be in the snapshot. Otherwise a syntax-fail
    # revert silently leaves a broken core/ file in place (revert skips dirs
    # missing from the snapshot) and relaunch still boots un-importable code.
    for _sub in ("core", "tools"):
        _src = os.path.join(PROJECT_DIR, _sub)
        if os.path.isdir(_src):
            shutil.copytree(_src, os.path.join(dest, _sub), dirs_exist_ok=True)

    # Prune oldest backups beyond MAX_BACKUPS. Only this function's own
    # timestamped snapshots (YYYY-MM-DD_HH-MM-SS) are eligible. backups/ also
    # holds the pipeline's per-task rollback tree (backups/pipeline/) and the
    # stability gate's gate_* snapshots; a blind listdir sort would let those
    # count against MAX_BACKUPS (evicting real code snapshots first) and could
    # rmtree live rollback state. Match the timestamp pattern explicitly.
    try:
        _ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        snapshots = sorted(
            d for d in os.listdir(BACKUP_DIR)
            if _ts_re.match(d) and os.path.isdir(os.path.join(BACKUP_DIR, d))
        )
        for old in snapshots[:-MAX_BACKUPS]:
            shutil.rmtree(os.path.join(BACKUP_DIR, old), ignore_errors=True)
    except Exception:
        pass

    return dest


def get_pending_tasks() -> list[str]:
    if not os.path.exists(TODO_FILE):
        return []
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return TASK_RE.findall(f.read())
    except Exception:
        return []


def kill_running_jarvis() -> int:
    """Stop any python.exe processes running bobert_companion.py.
    Returns the number killed."""
    # PROD killer: exclude any --staging (green) instance so we never take
    # down staging. Mirrors the PROD contract in _boot_jarvis.ps1
    # ($_.CommandLine -notlike '*--staging*').
    ps_cmd = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.CommandLine -like '*bobert_companion*' "
        "-and $_.CommandLine -notlike '*--staging*' } "
        "| ForEach-Object { $_.ProcessId }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        print("  [kill] PID enumeration timed out — nothing killed")
        return 0
    pids = [int(p.strip()) for p in result.stdout.splitlines() if p.strip().isdigit()]
    for pid in pids:
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
                capture_output=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            print(f"  [kill] Stop-Process for PID {pid} timed out")
    return len(pids)


# ─────────────────────── Stability gate ───────────────────────────────────
# Runs every STABILITY_GATE_INTERVAL successful task completions in the
# multi-agent pipeline (multi_agent_pipeline.run_pipeline_loop_driver calls
# `_stability_gate` via lazy import below). The gate snapshots state, boots
# JARVIS for STABILITY_GATE_DURATION_S, then verdicts on three signals:
# process alive, no APPCRASH, no [FATAL] in session log. On FAIL it
# robocopy-reverts to the snapshot AND appends a [regression] task — the
# pipeline pauses unless invoked with --force-continue-after-regression.


def _log_gate_event(record: dict) -> None:
    """Append one JSON line to data/stability_gates.jsonl. Best-effort —
    never raises so a logging hiccup can't take the gate down."""
    try:
        os.makedirs(os.path.dirname(STABILITY_GATES_LOG), exist_ok=True)
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(STABILITY_GATES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _gate_snapshot(batch_n: int) -> str:
    """Copy the source tree's mutable surface into
    backups/gate_<timestamp>_batch<N>/ so a FAIL verdict can robocopy /MIR
    revert. Returns the snapshot directory path."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    dest = os.path.join(GATE_SNAPSHOT_ROOT, f"gate_{ts}_batch{batch_n}")
    os.makedirs(dest, exist_ok=True)

    # Same surface as backup_codebase() — entry points + skills/ + hud/.
    always = [
        "bobert_companion.py",
        "overnight_upgrade.py",
        "upgrade_jarvis.py",
        "jarvis_todo.md",
    ]
    for fname in always:
        src = os.path.join(PROJECT_DIR, fname)
        if os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(dest, fname))
            except OSError:
                pass

    for sub in ("skills", "hud", "tools", "core"):
        src = os.path.join(PROJECT_DIR, sub)
        if os.path.isdir(src):
            try:
                shutil.copytree(src, os.path.join(dest, sub), dirs_exist_ok=True)
            except OSError:
                pass

    return dest


def _eventlog_appcrash_dump(since_iso: str) -> str:
    """Get-WinEvent APPCRASH dump since `since_iso` (datetime '%Y-%m-%dT%H:%M:%S')
    as a short JSON string. Best-effort; returns '' on any failure."""
    ps = (
        "$ErrorActionPreference = 'Stop';"
        f"$since = [datetime]'{since_iso}';"
        "$filter = @{LogName='Application';"
        " ProviderName=@('Application Error','.NET Runtime');"
        " StartTime=$since};"
        "try { $e = Get-WinEvent -FilterHashtable $filter -ErrorAction Stop } "
        "catch { if ($_.Exception.Message -match 'No events were found') "
        "{ $e = @() } else { throw } };"
        "$e | Select-Object -First 5 TimeCreated,Id,ProviderName,"
        "@{Name='Message';Expression={($_.Message -split \"`n\")[0]}} |"
        " ConvertTo-Json -Depth 3 -Compress"
    )
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", ps],
            capture_output=True, text=True, timeout=30,
        )
        return (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _latest_session_log_tail(n: int = 50) -> list[str]:
    """Return the last `n` lines of the freshest logs/session_*.log, or []."""
    logs_dir = os.path.join(PROJECT_DIR, "logs")
    if not os.path.isdir(logs_dir):
        return []
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(logs_dir):
        if name.startswith("session_") and name.endswith(".log"):
            p = os.path.join(logs_dir, name)
            try:
                candidates.append((os.path.getmtime(p), p))
            except OSError:
                continue
    if not candidates:
        return []
    candidates.sort()
    latest = candidates[-1][1]
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [ln.rstrip("\n") for ln in lines[-n:]]


def _run_stability_smoke_test(duration_s: int | None = None) -> dict:
    """Boot JARVIS via _boot_jarvis.ps1, wait `duration_s`, verdict on the
    three smoke signals (process alive, no APPCRASH, no FATAL in session
    log). Returns:

        {"ok": bool,
         "rc": int|None,
         "report": <smoke-test JSON>,
         "stdout_tail": str,
         "error": str | None}

    Reuses tools/stability_smoke_test.py which already encapsulates the
    booter spawn + lock-wait + three-check logic. We just pass --wait to
    parameterise the sample window.

    duration_s=None re-reads STABILITY_GATE_DURATION_S from the env via
    _gate_config() at call time, so an operator can re-tune the env var
    between runs without re-importing.
    """
    if duration_s is None:
        duration_s = _gate_config()[1]
    if not os.path.exists(STABILITY_SMOKE_TOOL):
        return {"ok": False, "rc": None, "report": None, "stdout_tail": "",
                "error": f"missing {STABILITY_SMOKE_TOOL}"}

    # Belt-and-braces: kill any stale JARVIS + clear the lock before handing
    # off. The booter does this too, but doing it here means the smoke
    # test's "lock just appeared" signal is unambiguous.
    try:
        kill_running_jarvis()
    except Exception:
        pass
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except OSError:
        pass

    # Headroom: smoke test does its own launch + 30s lock-wait + duration sleep.
    timeout_s = duration_s + 120
    try:
        result = subprocess.run(
            [sys.executable, STABILITY_SMOKE_TOOL, "--wait", str(duration_s)],
            cwd=PROJECT_DIR, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "rc": None, "report": None,
                "stdout_tail": (exc.stdout or "")[-1500:],
                "error": f"smoke test timed out after {timeout_s}s"}
    except OSError as exc:
        return {"ok": False, "rc": None, "report": None, "stdout_tail": "",
                "error": f"could not spawn smoke test: {exc!r}"}

    report: dict | None = None
    for report_path in (
        os.path.join(PROJECT_DIR, "stability_smoke_PASS.json"),
        os.path.join(PROJECT_DIR, "stability_smoke_FAIL.json"),
    ):
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                break
            except (OSError, ValueError):
                continue

    return {
        "ok": result.returncode == 0,
        "rc": result.returncode,
        "report": report,
        "stdout_tail": (result.stdout or "")[-1500:],
        "error": None,
    }


def _revert_to_snapshot(snapshot_dir: str) -> tuple[bool, str]:
    """Restore PROJECT_DIR from snapshot_dir. Returns (ok, log_tail).

    Two-phase to avoid the /MIR foot-gun where any file in dest not in src is
    deleted unconditionally — that wiped runtime artifacts and made revert
    failure modes opaque. Instead:
      1. robocopy /E src dst  (recursive copy, no delete) restores content.
      2. Explicit delete pass walks dst and removes regular files whose
         relative path is not in the snapshot manifest. __pycache__ subtrees
         are skipped (runtime artifacts, regenerated on next import).
    robocopy exit codes 0-7 mean success (8+ is failure)."""
    if not os.path.isdir(snapshot_dir):
        return (False, f"snapshot dir missing: {snapshot_dir}")

    tails: list[str] = []
    overall_ok = True

    for fname in ("bobert_companion.py", "overnight_upgrade.py",
                  "upgrade_jarvis.py", "jarvis_todo.md"):
        src = os.path.join(snapshot_dir, fname)
        if os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(PROJECT_DIR, fname))
            except OSError as exc:
                overall_ok = False
                tails.append(f"copy {fname} failed: {exc}")

    for sub in ("skills", "hud", "tools", "core"):
        src = os.path.join(snapshot_dir, sub)
        dst = os.path.join(PROJECT_DIR, sub)
        if not os.path.isdir(src):
            continue

        manifest: set[str] = set()
        try:
            for root, _dirs, files in os.walk(src):
                rel_root = os.path.relpath(root, src)
                for f in files:
                    rel = f if rel_root == "." else os.path.join(rel_root, f)
                    manifest.add(os.path.normpath(rel).lower())
        except OSError as exc:
            overall_ok = False
            tails.append(f"manifest {sub} failed: {exc!r}")
            continue

        try:
            r = subprocess.run(
                ["robocopy", src, dst, "/E", "/NFL", "/NDL", "/NJH", "/NJS",
                 "/NC", "/NS", "/NP"],
                capture_output=True, text=True, timeout=120,
            )
            tails.append(f"[{sub}] rc={r.returncode}")
            if r.returncode >= 8:
                overall_ok = False
                tails.append((r.stdout or "")[-200:])
                continue
        except (OSError, subprocess.SubprocessError) as exc:
            overall_ok = False
            tails.append(f"robocopy {sub} failed: {exc!r}")
            continue

        deleted = 0
        purge_errors = 0
        for root, dirs, files in os.walk(dst, topdown=True):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            rel_root = os.path.relpath(root, dst)
            for f in files:
                rel = f if rel_root == "." else os.path.join(rel_root, f)
                key = os.path.normpath(rel).lower()
                if key in manifest:
                    continue
                try:
                    os.remove(os.path.join(root, f))
                    deleted += 1
                except OSError:
                    purge_errors += 1
        if deleted or purge_errors:
            tails.append(
                f"[{sub}] purged {deleted} extra file(s)"
                + (f" ({purge_errors} errors)" if purge_errors else "")
            )

    return (overall_ok, " | ".join(tails))


def _stability_gate(batch_n: int, recent_task_titles: list[str],
                    batch_size: int = 10) -> dict:
    """Run one stability gate after `batch_size` task completions.

    Steps:
      1. Snapshot PROJECT_DIR to backups/gate_<ts>_batch<N>/
      2. Call _run_stability_smoke_test(duration_s=<env STABILITY_GATE_DURATION_S>)
      3. PASS  → append PASS jsonl record + marker file + kill JARVIS.
      4. FAIL  → append FAIL record (with EventLog dump + session-log tail) +
                 revert via robocopy + queue a [regression] task pointing at
                 the suspect batch.

    Returns a dict with at least {"ok": bool, "verdict": "PASS"|"FAIL"|"SKIP",
    "snapshot_dir": str|None, "report": dict|None, "details": str}.
    Caller (the pipeline driver loop) decides whether to pause.
    """
    interval, duration_s, disabled = _gate_config()
    if disabled:
        _log_gate_event({"batch": batch_n, "verdict": "SKIP",
                         "reason": "STABILITY_GATE_DISABLE=1"})
        return {"ok": True, "verdict": "SKIP", "snapshot_dir": None,
                "report": None,
                "details": "stability gate disabled via env var"}

    print(f"\n=== STABILITY GATE batch_{batch_n} "
          f"(after {batch_size} tasks, {duration_s}s smoke) ===")

    snapshot_dir = _gate_snapshot(batch_n)
    print(f"  [gate] snapshot → {os.path.basename(snapshot_dir)}")

    launched_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    smoke = _run_stability_smoke_test(duration_s=duration_s)

    # Always kill JARVIS after the smoke test so subsequent pipeline tasks
    # start from a known-clean state. Match the existing pipeline behaviour.
    try:
        kill_running_jarvis()
    except Exception:
        pass

    if smoke["ok"]:
        marker = os.path.join(
            snapshot_dir, f"PASS_batch_{batch_n}_at_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(f"PASS batch_{batch_n} at "
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                        f"duration_s={duration_s}\n")
        except OSError:
            pass
        _log_gate_event({"batch": batch_n, "verdict": "PASS",
                         "duration_s": duration_s,
                         "snapshot": os.path.basename(snapshot_dir),
                         "report": smoke.get("report")})
        print(f"  [gate] PASS — batch_{batch_n} stable for {duration_s}s")
        return {"ok": True, "verdict": "PASS", "snapshot_dir": snapshot_dir,
                "report": smoke.get("report"),
                "details": f"PASS batch_{batch_n}"}

    # FAIL path — collect diagnostics, revert, queue regression.
    eventlog_dump = _eventlog_appcrash_dump(launched_at_iso)
    session_tail = _latest_session_log_tail(n=50)
    fail_record = {
        "batch": batch_n, "verdict": "FAIL",
        "duration_s": duration_s,
        "snapshot": os.path.basename(snapshot_dir),
        "smoke_error": smoke.get("error"),
        "smoke_rc": smoke.get("rc"),
        "smoke_stdout_tail": smoke.get("stdout_tail", "")[-800:],
        "report": smoke.get("report"),
        "eventlog_appcrash": eventlog_dump[:2000],
        "session_log_tail": session_tail,
        "recent_task_titles": recent_task_titles[-batch_size:],
    }
    _log_gate_event(fail_record)

    print(f"  [gate] FAIL — reverting via robocopy /E + manifest purge from "
          f"{os.path.basename(snapshot_dir)}")
    revert_ok, revert_log = _revert_to_snapshot(snapshot_dir)
    print(f"  [gate] revert {'OK' if revert_ok else 'FAILED'}: {revert_log[:200]}")

    # Append a high-priority regression task — give the user enough context
    # to bisect which of the N tasks broke things. The smoke-test report
    # now carries a categorized scan of the session log; pull the most
    # actionable lines into the regression description so the next
    # iteration's planner has runtime context, not just 'tester failed'.
    titles_blob = "; ".join(t[:80] for t in recent_task_titles[-batch_size:])

    # Compose a focused symptom string. Priority order:
    #   1. hard error from the smoke tool itself (subprocess failure)
    #   2. [FATAL] line from the JARVIS session log
    #   3. first traceback head
    #   4. native-crash signature
    #   5. failed boot milestones (which step didn't happen)
    #   6. fall back to stdout tail / jsonl pointer
    rich_report = (smoke.get("report") or {}) if isinstance(smoke, dict) else {}
    rich_checks = (rich_report.get("checks") or {}) if isinstance(rich_report, dict) else {}
    log_findings = rich_checks.get("session_log_fatal") or {}
    crash_findings = rich_checks.get("crash_traces") or {}
    fatal_lines       = log_findings.get("fatal_lines") or []
    tracebacks        = log_findings.get("tracebacks") or []
    native_crashes    = log_findings.get("native_crashes") or []
    skill_failures    = log_findings.get("skill_failures") or []
    transcribe_fails  = log_findings.get("transcribe_failures") or []
    milestones        = log_findings.get("boot_milestones") or {}
    missed_milestones = [k for k, v in milestones.items() if v is False] if milestones else []
    # Faulthandler-caught native crashes (sounddevice / numpy / opencv
    # SIGSEGV that never reach the session log's [FATAL] gate). These are
    # the silent crashes the user has been chasing all night.
    crash_dumps      = crash_findings.get("new_dumps") or []
    crash_signatures = crash_findings.get("head_signatures") or []

    symptom_blob = (
        smoke.get("error")
        or (fatal_lines[0] if fatal_lines else "")
        or (tracebacks[0].splitlines()[-1] if tracebacks else "")
        # Prefer the faulthandler signature over a raw [FATAL] line —
        # it points at the actual user-code frame that crashed.
        or (f"native crash: {crash_signatures[0]}"
            if crash_signatures else "")
        or (native_crashes[0] if native_crashes else "")
        or (f"boot stalled — never reached: {', '.join(missed_milestones)}"
            if missed_milestones else "")
        or (smoke.get("stdout_tail", "") or "")[-200:]
        or "see data/stability_gates.jsonl"
    )

    # Build a context block that the next iteration's planner / implementer
    # can actually act on. Only include categories that fired so we don't
    # pad the description with empty headers.
    context_chunks: list[str] = []
    if crash_signatures and not symptom_blob.startswith("native crash:"):
        # Don't double-print the signature; include only if we picked a
        # different symptom (e.g. fatal_line) but a crash also fired.
        context_chunks.append(
            f"native crash signature: {crash_signatures[0]}")
    if transcribe_fails:
        context_chunks.append(
            f"transcribe failures: {transcribe_fails[0][:150]}")
    if skill_failures:
        context_chunks.append(
            f"skill failures: {'; '.join(s[:80] for s in skill_failures[:3])}")
    if missed_milestones and not (fatal_lines or tracebacks or native_crashes or crash_dumps):
        context_chunks.append(
            f"missing boot milestones: {', '.join(missed_milestones)}")
    context_blob = (" | " + " | ".join(context_chunks)) if context_chunks else ""

    regression_line = (
        f"- [ ] **[regression]** [HIGH PRIORITY] Stability gate batch {batch_n} "
        f"broke JARVIS — symptom: {symptom_blob[:300]}{context_blob[:400]}. "
        f"Investigate which of the following {batch_size} tasks caused it: "
        f"{titles_blob[:600]}. Revert was automatic (snapshot "
        f"{os.path.basename(snapshot_dir)}); you must fix and re-test.\n"
    )
    try:
        with open(TODO_FILE, "a", encoding="utf-8") as f:
            f.write(regression_line)
    except OSError:
        pass

    return {"ok": False, "verdict": "FAIL", "snapshot_dir": snapshot_dir,
            "report": smoke.get("report"),
            "details": f"FAIL batch_{batch_n}: {symptom_blob[:200]}"}


def find_claude_cli() -> str | None:
    """Locate the claude executable. Returns the full path, or None if not found.
    Checks PATH first, then known install locations."""
    # Try PATH (works in shells that loaded the new PATH variable)
    import shutil
    found = shutil.which("claude")
    if found:
        return found

    # Fallback: known Windows install locations for Claude Code
    candidates = [
        os.path.expanduser(r"~\.local\bin\claude.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude-code\claude.exe"),
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
        os.path.expanduser(r"~\AppData\Roaming\npm\claude.cmd"),
        os.path.expandvars(r"%USERPROFILE%\.npm-global\claude.cmd"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def spawn_claude_code(tasks: list[str], claude_path: str,
                      single_stage: bool = False) -> subprocess.Popen | None:
    """Launch a Python loop driver in a new visible console.

    Two driver flavors:

    • Multi-agent pipeline (DEFAULT) — each iteration runs the 4-stage
      pipeline (planner → implementer → reviewer → tester) from
      tools/multi_agent_pipeline.py. The driver tempfile is thin: it just
      sets sys.path and calls run_pipeline_loop_driver(). All the
      orchestration lives in the module so it stays grep-able and testable.

    • Single-stage (legacy, --single-stage) — each iteration invokes
      `claude --print` ONCE to implement one task and exit, then the loop
      re-reads jarvis_todo.md and fires another invocation. No review or
      test gates. Faster and cheaper, but a bad change can ship straight to
      bobert_companion.py with no checks. Kept for emergencies.

    Why the loop in both modes? `--print` is one-shot. Without the loop, a
    15-task queue would only get the first 1-3 tasks done before Claude
    Code exited. The loop guarantees the queue drains fully (or until the
    safety cap is hit).

    Why a Python driver instead of a PowerShell wrapper? The old
    PowerShell-hosted pipe (`claude ... | python -u pp.py`) printed nothing
    inside CREATE_NEW_CONSOLE windows — PS pipe buffering across the new-
    console boundary swallowed every stream-json event until claude exited.
    The driver does the pipe in pure Python via subprocess.Popen.stdout
    (line-buffered, byte-passthrough) so events render in real time, and
    additionally tees every line to upgrade_stream.log so the user can
    `Get-Content -Wait` from any other window if the visible one breaks
    again."""
    # Hard safety cap. The driver loop also exits early when the queue is
    # empty AND when no progress is made for several iterations, so this just
    # needs to be high enough that a legitimately-growing queue (e.g. the
    # auditor appending findings mid-run) can still drain. Was 25 — that
    # silently capped audit-fix runs at 25 iterations even when the queue
    # held 400+ tasks.
    MAX_ITER = max(len(tasks) * 2 + 50, 1000)

    # Per-iteration prompt: do ONE task, mark it done, exit. No multi-task
    # plan, no codebase review at the end — the loop handles batching.
    per_task_prompt = (
        f"You are upgrading the JARVIS PC companion at {PROJECT_DIR}. "
        f"Read jarvis_todo.md and find the FIRST line that starts with '- [ ]' "
        f"(an unchecked task). Implement that ONE task end-to-end:\n"
        f"  1. Make the code changes the task describes.\n"
        f"  2. Run python -m py_compile on every file you touched. Fix any errors.\n"
        f"  3. Edit jarvis_todo.md to change '- [ ]' to '- [x]' on that task's line, "
        f"and append a ' ✓ DONE — <one-line note>' summary of what you did.\n"
        f"  4. Exit. Do NOT try to handle more than one task in this invocation — "
        f"a wrapper loop will call you again for the next task.\n"
        f"If there are no unchecked tasks left, just print 'QUEUE EMPTY' and exit. "
        f"If a task is hardware-dependent or genuinely impossible (e.g. 'check that "
        f"the cables are plugged in'), tick it off with a note explaining why it "
        f"can't be auto-fixed."
    )

    mode_label = "SINGLE-STAGE (legacy)" if single_stage else "MULTI-AGENT pipeline"
    log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Spawning Claude Code LOOP "
        f"[{mode_label}] ({len(tasks)} task(s), max {MAX_ITER} iterations) "
        f"via {claude_path}\n"
    )
    with open(UPGRADE_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)

    try:
        # Strip ANTHROPIC_API_KEY from the spawned env so Claude Code bills
        # to the Max subscription rather than API credits.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        # Render the driver template to a tempfile. Sentinel substitution
        # (not str.format) so we don't fight Python brace-escaping with the
        # f-strings and dict literals in the template body.
        import tempfile
        if single_stage:
            template = _LOOP_DRIVER_TEMPLATE
            suffix = "_jarvis_upgrade_loop.py"
        else:
            template = _PIPELINE_LOOP_DRIVER_TEMPLATE
            suffix = "_jarvis_pipeline_loop.py"
        driver_src = (template
            .replace("__PROJECT_DIR__",  repr(PROJECT_DIR))
            .replace("__CLAUDE_PATH__",  repr(claude_path))
            .replace("__PROMPT__",       repr(per_task_prompt))
            .replace("__STREAM_LOG__",   repr(UPGRADE_STREAM_LOG))
            .replace("__MAX_ITER__",     str(MAX_ITER))
            .replace("__TASK_COUNT__",   str(len(tasks)))
        )
        drv_fd, drv_path = tempfile.mkstemp(suffix=suffix, text=True)
        with os.fdopen(drv_fd, "w", encoding="utf-8") as _df:
            _df.write(driver_src)

        # Spawn the driver in its own console window. -u keeps stdout line-
        # buffered. CREATE_NEW_CONSOLE gives it a visible terminal on Windows.
        proc = subprocess.Popen(
            [sys.executable, "-u", drv_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            env=env,
            cwd=PROJECT_DIR,
            close_fds=True,
        )
        return proc
    except Exception as e:
        print(f"Failed to launch Claude Code: {e}")
        return None


# ───────────────── Upgrade-loop driver (rendered to tempfile) ─────────────────
# This template is rendered with sentinel substitution and written to a temp
# .py file that gets spawned in its own console window. Doing the
# claude→pretty-printer pipe inside one Python process (via Popen.stdout)
# bypasses the PowerShell pipe-buffering bug that made the spawned window
# go silent. The driver also tees every rendered line to upgrade_stream.log
# so the user can `Get-Content -Wait` it from any other window — guarantees
# visibility even if the new console itself is broken.
_LOOP_DRIVER_TEMPLATE = r'''#!/usr/bin/env python3
"""JARVIS upgrade loop driver (spawned by upgrade_jarvis.py).

Loops up to MAX_ITER times: counts unchecked tasks, runs `claude --print`
as a subprocess, reads its stream-json stdout line-by-line, pretty-prints
each event to both console AND a log file. Doing the pipe in pure Python
sidesteps the PowerShell pipe-buffering issue that made the spawned loop
window silent.
"""
import json, os, re, subprocess, sys, threading, time

PROJECT_DIR = __PROJECT_DIR__
CLAUDE_PATH = __CLAUDE_PATH__
PROMPT      = __PROMPT__
STREAM_LOG  = __STREAM_LOG__
MAX_ITER    = __MAX_ITER__
TASK_COUNT  = __TASK_COUNT__

TODO_FILE = os.path.join(PROJECT_DIR, "jarvis_todo.md")

# Enable ANSI escape sequences on Windows 10+ consoles so colors render.
if os.name == "nt":
    try:
        import ctypes
        _k32 = ctypes.windll.kernel32
        _k32.SetConsoleMode(_k32.GetStdHandle(-11), 7)
    except Exception:
        pass

CYAN   = "\x1b[36m"
YELLOW = "\x1b[33m"
GREEN  = "\x1b[32m"
RED    = "\x1b[31m"
GRAY   = "\x1b[90m"
RESET  = "\x1b[0m"

_log_fp = open(STREAM_LOG, "a", encoding="utf-8", buffering=1)
_log_fp.write(
    "\n=== upgrade loop started "
    + time.strftime("%Y-%m-%d %H:%M:%S")
    + " ===\n"
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
def emit(s):
    """Write to console (flushed) and append plain-text copy to STREAM_LOG."""
    print(s, flush=True)
    _log_fp.write(_ANSI_RE.sub("", s) + "\n")
    _log_fp.flush()

def _fmt_tool_use(c):
    name = c.get("name", "?")
    inp = c.get("input", {}) or {}
    if name == "Read":      return f"  [read]  {inp.get('file_path','')}"
    if name == "Edit":      return f"  [edit]  {inp.get('file_path','')}"
    if name == "Write":     return f"  [write] {inp.get('file_path','')}"
    if name == "Bash":      return f"  [bash]  {(inp.get('command','') or '')[:120]}"
    if name == "Grep":      return f"  [grep]  {(inp.get('pattern','') or '')[:80]}"
    if name == "Glob":      return f"  [glob]  {inp.get('pattern','')}"
    if name == "TodoWrite": return f"  [todo]  {len(inp.get('todos', []))} item(s)"
    return f"  [tool]  {name}"

def _fmt(e):
    t = e.get("type", "?")

    if t == "assistant":
        msg = e.get("message", {})
        lines = []
        for c in msg.get("content", []):
            ct = c.get("type")
            if ct == "tool_use":
                lines.append(_fmt_tool_use(c))
            elif ct == "text":
                tx = (c.get("text", "") or "").strip()
                if tx:
                    lines.append(f"  [say]   {tx[:240]}")
        return "\n".join(lines)

    if t == "user":
        msg = e.get("message", {})
        for c in msg.get("content", []):
            if c.get("type") == "tool_result":
                out = c.get("content", "")
                if isinstance(out, list):
                    out = " ".join(b.get("text","") for b in out if isinstance(b, dict))
                # BUG FIX: `splitlines()[0]` raised IndexError on tool_results
                # whose joined text was empty or whitespace-only (e.g. some
                # ToolSearch responses). Take the first non-empty line, or
                # skip the event entirely if there is none.
                _lines = (str(out) or "").strip().splitlines()
                _first = next((ln for ln in _lines if ln.strip()), "")
                if _first:
                    return f"          {_first[:120]}"
        return ""

    if t == "stream_event":
        ev = e.get("event", {}) or {}
        if ev.get("type") == "content_block_start":
            cb = ev.get("content_block", {}) or {}
            if cb.get("type") == "tool_use":
                return _fmt_tool_use(cb)
        return ""

    if t == "result":
        ms   = e.get("duration_ms", 0)
        cost = e.get("total_cost_usd", 0)
        return f"  [done]  {e.get('subtype','?')} in {ms/1000:.1f}s, ${cost:.4f}"

    if t == "system":
        if e.get("subtype") == "init":
            sid = (e.get("session_id", "?") or "?")[:8]
            return f"  [init]  session {sid} model={e.get('model','?')}"
        return ""

    if t == "rate_limit_event":
        info = e.get("rate_limit_info", {}) or {}
        if info.get("status") != "allowed":
            return (
                f"  [!]     rate limit: {info.get('status','?')} "
                f"reason={info.get('overageDisabledReason','?')}"
            )
        return ""

    return ""

def count_pending():
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return len(re.findall(r"^- \[ \]", f.read(), re.MULTILINE))
    except Exception:
        return 0

def run_one_claude_call():
    cmd = [
        CLAUDE_PATH, "--print", "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--dangerously-skip-permissions",
        PROMPT,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=PROJECT_DIR,
        )
    except FileNotFoundError as e:
        emit(f"{RED}[loop] claude binary not found at {CLAUDE_PATH}: {e}{RESET}")
        return 127
    except Exception as e:
        emit(f"{RED}[loop] failed to spawn claude: {e}{RESET}")
        return 1
    assert proc.stdout is not None
    # Watchdog: a hung `claude --print` child (network stall, stuck stream, no
    # EOF) would otherwise block the read loop AND proc.wait() forever, freezing
    # the whole upgrade loop with JARVIS already killed. Kill the child past a
    # hard wall-clock deadline so the loop drains EOF and returns a failure code.
    _CALL_TIMEOUT_S = 3600  # 1h backstop — a single task call should never approach this
    _killed = threading.Event()
    def _watchdog():
        try:
            proc.kill()
        finally:
            _killed.set()
    _timer = threading.Timer(_CALL_TIMEOUT_S, _watchdog)
    _timer.daemon = True
    _timer.start()
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped:
                continue
            try:
                out = _fmt(json.loads(stripped))
                if out:
                    emit(out)
            except json.JSONDecodeError:
                # Non-JSON line (likely a stderr message merged in). Surface it
                # so a silently-broken invocation isn't invisible.
                emit(f"  [?]     {line[:160]}")
        proc.wait()
    finally:
        _timer.cancel()
    if _killed.is_set():
        emit(f"{RED}[loop] claude call exceeded {_CALL_TIMEOUT_S}s — killed to "
             f"unblock the upgrade loop.{RESET}")
        return 124
    return proc.returncode

def main():
    # Self-delete the rendered driver tempfile so %TEMP% doesn't accumulate
    # one *_jarvis_upgrade_loop.py per upgrade cycle. CPython reads .py source
    # files at compile time and closes the handle before main() runs, so the
    # unlink succeeds even on Windows. Wrapped in try/except: if it ever does
    # fail (e.g. AV scanner holding the handle), continue silently rather than
    # crash the loop driver.
    try:
        os.unlink(sys.argv[0])
    except OSError:
        pass

    emit(
        f"{CYAN}=== JARVIS UPGRADE: {TASK_COUNT} task(s), looping "
        f"(max {MAX_ITER} iterations) ==={RESET}"
    )
    emit(f"{GRAY}Live log: {STREAM_LOG}{RESET}")
    emit(
        f'{GRAY}Tail from any other window with: '
        f'Get-Content -Wait "{STREAM_LOG}"{RESET}'
    )

    # Loop with two guards:
    #   (1) Stop early when queue empty (normal success path).
    #   (2) No-progress watchdog: if the unchecked count fails to drop for
    #       NO_PROGRESS_LIMIT consecutive iterations, something is stuck —
    #       give up rather than burn cycles in a no-op.
    NO_PROGRESS_LIMIT = 5
    no_progress = 0
    last_pending = None
    for i in range(1, MAX_ITER + 1):
        pending = count_pending()
        if pending == 0:
            emit("")
            emit(f"{GREEN}[loop] queue empty — all tasks done{RESET}")
            break
        if last_pending is not None and pending >= last_pending:
            no_progress += 1
            if no_progress >= NO_PROGRESS_LIMIT:
                emit("")
                emit(
                    f"{RED}[loop] no queue progress for "
                    f"{NO_PROGRESS_LIMIT} iterations — stopping{RESET}"
                )
                break
        else:
            no_progress = 0
        last_pending = pending
        emit("")
        emit(
            f"{YELLOW}[loop] iteration {i} of {MAX_ITER} — "
            f"{pending} task(s) remain{RESET}"
        )
        rc = run_one_claude_call()
        if rc != 0:
            emit(f"{RED}[loop] Claude exited code {rc} — stopping{RESET}")
            break

    emit("")
    emit(f"{YELLOW}Verifying syntax...{RESET}")
    rc = subprocess.run(
        [sys.executable, "-m", "py_compile", "bobert_companion.py"],
        cwd=PROJECT_DIR,
    ).returncode
    if rc == 0:
        emit(f"{GREEN}bobert_companion.py: OK{RESET}")
    else:
        emit(f"{RED}SYNTAX ERROR in bobert_companion.py — check above!{RESET}")
    emit(f"{GREEN}Upgrade complete.{RESET}")

    try:
        _log_fp.close()
    except Exception:
        pass

    time.sleep(5)

if __name__ == "__main__":
    main()
'''


# ─────────── Multi-agent pipeline driver (rendered to tempfile) ───────────
# Thin wrapper: all orchestration lives in tools/multi_agent_pipeline.py.
# Keeping the driver itself small makes the per-iteration logic easy to
# read/modify in the module file rather than buried in an f-string template.
_PIPELINE_LOOP_DRIVER_TEMPLATE = r'''#!/usr/bin/env python3
"""JARVIS multi-agent upgrade loop driver (spawned by upgrade_jarvis.py).

Each iteration runs the 4-stage pipeline on the first unchecked task:
    planner (cheap, read-only) → implementer (Opus, writes) →
    reviewer (sonnet, read-only) → tester (boot smoke test).
"""
import os, sys, time

PROJECT_DIR = __PROJECT_DIR__
CLAUDE_PATH = __CLAUDE_PATH__
STREAM_LOG  = __STREAM_LOG__
MAX_ITER    = __MAX_ITER__
TASK_COUNT  = __TASK_COUNT__

# Self-delete the rendered tempfile so %TEMP% doesn't accumulate one per
# upgrade. CPython reads/closes the .py source before main() runs, so the
# unlink is safe on Windows.
try:
    os.unlink(sys.argv[0])
except OSError:
    pass

sys.path.insert(0, PROJECT_DIR)
try:
    from tools.multi_agent_pipeline import run_pipeline_loop_driver
except Exception as exc:
    print(f"[pipeline-driver] failed to import multi_agent_pipeline: {exc!r}")
    print(f"[pipeline-driver] PROJECT_DIR={PROJECT_DIR}")
    time.sleep(10)
    sys.exit(2)

rc = run_pipeline_loop_driver(
    project_dir=PROJECT_DIR,
    claude_path=CLAUDE_PATH,
    stream_log=STREAM_LOG,
    max_iter=MAX_ITER,
    task_count=TASK_COUNT,
)

# Pause so the user can read the final summary in the spawned console
# before it closes.
time.sleep(5)
sys.exit(rc)
'''


def spawn_claude_dry_audit(tasks: list[str], claude_path: str) -> subprocess.Popen | None:
    """Launch Claude Code in --print mode WITHOUT --dangerously-skip-permissions
    to produce a read-only audit/code-review report on the pending queue.

    No backup, no JARVIS kill, no loop — single invocation that asks Claude to
    review each unchecked task, describe what it would do, flag risks, and
    rate confidence — without touching any files. Useful before big upgrades
    so the human can preview Claude's plan."""
    audit_prompt = (
        f"You are running a DRY AUDIT for the JARVIS PC companion at {PROJECT_DIR}. "
        f"This is READ-ONLY: do NOT modify any files, do NOT use Edit/Write/"
        f"NotebookEdit or any mutating tool. You may only Read, Glob, and Grep.\n\n"
        f"Read jarvis_todo.md and review every unchecked '- [ ]' task. For each one:\n"
        f"  - Briefly explain what you would change and which files you would touch.\n"
        f"  - Flag concerns: ambiguity, missing context, regression risk, hardware "
        f"dependency, or genuine impossibility.\n"
        f"  - Rate confidence (HIGH / MEDIUM / LOW) that you could complete it cleanly.\n\n"
        f"End the report with a one-paragraph recommendation: which tasks are safe to "
        f"run autonomously next, and which need human review before kicking off the "
        f"real upgrade."
    )
    safe_prompt = audit_prompt.replace("'", "''")

    report_path = os.path.join(
        PROJECT_DIR, f"dry_audit_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md"
    )

    log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Spawning Claude Code DRY AUDIT "
        f"({len(tasks)} task(s)) via {claude_path}\n"
    )
    with open(UPGRADE_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)

    try:
        # Strip ANTHROPIC_API_KEY so Claude Code bills to the Max subscription.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        # Single Claude invocation — no loop, no skip-permissions flag.
        # Tee-Object captures the report to a markdown file alongside stdout.
        ps_cmd = (
            f"$env:ANTHROPIC_API_KEY=''; "
            f"cd '{PROJECT_DIR}'; "
            f"Write-Host '=== JARVIS DRY AUDIT: {len(tasks)} task(s) (read-only) ===' "
            f"-ForegroundColor Cyan; "
            f"Write-Host 'Report will be saved to:' -ForegroundColor DarkGray; "
            f"Write-Host '  {report_path}' -ForegroundColor DarkGray; "
            f"Write-Host ''; "
            f"& '{claude_path}' --print --verbose '{safe_prompt}' "
            f"| Tee-Object -FilePath '{report_path}'; "
            f"Write-Host ''; "
            f"if ($LASTEXITCODE -eq 0) {{ "
            f"  Write-Host \"Audit complete. Report saved to {report_path}\" "
            f"  -ForegroundColor Green "
            f"}} else {{ "
            f"  Write-Host \"Audit exited with code $LASTEXITCODE — check report.\" "
            f"  -ForegroundColor Yellow "
            f"}}; "
            f"Write-Host 'Press Enter to close this window.' -ForegroundColor DarkGray; "
            f"Read-Host"
        )

        proc = subprocess.Popen(
            ["powershell", "-Command", ps_cmd],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            env=env,
            close_fds=True,
        )
        return proc
    except Exception as e:
        print(f"Failed to launch Claude Code (dry-audit): {e}")
        return None


def relaunch_jarvis(with_handoff: bool = False):
    """Start JARVIS in a new visible PowerShell window.
    No -NoExit: when Python exits (restart/upgrade), PowerShell also exits
    and the console window closes automatically — no orphaned windows.

    Only the blue-green caller wrote a fresh handoff.json on the way out,
    so only it should ask the new prod to resume from it. Default False
    keeps regular full-upgrades from replaying stale orphaned handoffs
    left behind by an aborted blue-green run."""
    extra_args = " --resume-handoff" if with_handoff else ""
    # Pipeline relaunch boots SILENT into ambient-learning standby: JARVIS
    # listens + learns but won't speak until the user says the wake word.
    # JARVIS_AMBIENT_LEARNING=0 in the environment opts back out (e.g. a manual
    # restart that should come up fully interactive). Wake-resume sub-mode
    # (answer_then_quiet | stay_talkative) is propagated through unchanged.
    _ambient = os.environ.get("JARVIS_AMBIENT_LEARNING", "1").strip() or "1"
    _wake_resume = os.environ.get("JARVIS_WAKE_RESUME", "answer_then_quiet").strip()
    ps_inline = (
        f"$env:ANTHROPIC_API_KEY = "
        f"[System.Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User'); "
        # Bambu credentials follow the same durable-secret pattern: the setup
        # wizard (skills/bambu_setup.py) writes them to HKCU\Environment, and
        # the relaunch pulls them forward for core/config.py's os.getenv().
        f"$env:BAMBU_PRINTER_IP = "
        f"[System.Environment]::GetEnvironmentVariable('BAMBU_PRINTER_IP', 'User'); "
        f"$env:BAMBU_ACCESS_CODE = "
        f"[System.Environment]::GetEnvironmentVariable('BAMBU_ACCESS_CODE', 'User'); "
        f"$env:BAMBU_SERIAL = "
        f"[System.Environment]::GetEnvironmentVariable('BAMBU_SERIAL', 'User'); "
        f"$env:JARVIS_AMBIENT_LEARNING = '{_ambient}'; "
        f"$env:JARVIS_WAKE_RESUME = '{_wake_resume}'; "
        f"cd '{PROJECT_DIR}'; python bobert_companion.py{extra_args}"
    )
    subprocess.Popen(
        ["powershell", "-Command", ps_inline],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        close_fds=True,
    )


# ─────────────────────── Blue / green deployment ─────────────────────────
# Zero-downtime upgrades: prod keeps serving the user while a green/staging
# JARVIS boots from the candidate code and runs smoke tests in isolation.
# After staging passes, the pipeline writes a handoff signal that prod
# picks up on its next tick (10s grace), then exits. The green process
# is then relaunched as the new prod from the candidate build.


def spawn_staging_jarvis() -> subprocess.Popen | None:
    """Boot the green/staging JARVIS in a detached, hidden process. Uses
    pythonw.exe so no console window pops onto the user's desktop while
    they're working. Returns the Popen handle, or None on failure."""
    pythonw_candidates = [
        os.path.join(os.path.dirname(sys.executable), "pythonw.exe"),
        sys.executable.replace("python.exe", "pythonw.exe"),
        sys.executable,
    ]
    exe = next((p for p in pythonw_candidates if p and os.path.exists(p)),
               sys.executable)
    env = dict(os.environ)
    # Belt-and-braces: the bobert_companion module detects --staging from
    # argv, but JARVIS_STAGING=1 is the secondary trigger so any helper
    # script that re-execs picks up the role too.
    env["JARVIS_STAGING"] = "1"
    env.setdefault(
        "ANTHROPIC_API_KEY",
        os.environ.get("ANTHROPIC_API_KEY", "") or "",
    )
    try:
        proc = subprocess.Popen(
            [exe, "bobert_companion.py", "--staging"],
            cwd=PROJECT_DIR,
            env=env,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
            close_fds=True,
        )
        return proc
    except OSError as exc:
        print(f"  [blue-green] could not spawn staging JARVIS: {exc}")
        return None


def _prod_state_label() -> str:
    """Return prod's published HUD state ('Idle' / 'Speaking' / etc.) or
    empty string when hud_state.json is missing/unreadable. Used during
    blue-green-2 to wait for prod to finish whatever it was saying before
    the handoff announcement fires."""
    hud_state_path = os.path.join(PROJECT_DIR, "hud_state.json")
    if not os.path.exists(hud_state_path):
        return ""
    try:
        with open(hud_state_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            return str(data.get("state") or "")
    except (OSError, ValueError):
        pass
    return ""


def _wait_for_prod_idle(timeout_s: float = 8.0, poll_s: float = 0.25) -> bool:
    """Block until prod's hud_state.state reads 'idle' (case-insensitive)
    or `timeout_s` elapses. Returns True if we observed idle. Used by the
    cinematic handoff to avoid stepping on prod mid-sentence."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = _prod_state_label().lower()
        if s in ("", "idle", "standby"):
            return True
        time.sleep(poll_s)
    return False


def _wait_for_new_prod_heartbeat(timeout_s: float = 30.0,
                                   poll_s: float = 0.5) -> bool:
    """After promote_staging() flips the deployment state, the relaunched
    prod must claim jarvis.lock AND publish a fresh prod heartbeat to
    instances.json. Returns True when both happen inside the window."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _bgm is not None and _bgm.prod_is_running():
            try:
                insts = _bgm.list_instances()
            except Exception:
                insts = {}
            now = time.time()
            for entry in insts.values():
                if not isinstance(entry, dict):
                    continue
                if entry.get("role") != "prod":
                    continue
                hb = float(entry.get("heartbeat_at") or 0)
                if hb and (now - hb) < 30.0:
                    return True
        time.sleep(poll_s)
    return False


def run_blue_green_handoff(timeout_boot_s: float = 60.0,
                            timeout_replies_s: float = 90.0,
                            grace_seconds: int = 10) -> dict:
    """Full blue/green upgrade ceremony — called AFTER Claude Code has
    finished editing the candidate code, in lieu of the old kill-and-
    relaunch pattern.

    Sequence:
      1. Pre-flight: py_compile the entry points; abort if any fail.
      2. Spawn the staging JARVIS (--staging). Wait for its lock to
         appear inside the boot timeout.
      3. Run the smoke tests defined in staging_instance.py.
      4. On PASS: signal prod to idle, wait for prod's HUD state to
         report 'idle' (so we don't talk over the user mid-sentence),
         give prod ~3 seconds to land the 'Switching to new version sir'
         line, then promote_staging() and wait for the new prod to
         heartbeat. If prod doesn't yield inside the grace window,
         signal_handoff_failure() so prod announces it and stays.
      5. On FAIL: roll back — kill staging, leave prod untouched, log
         the failure into deployment_state.json, signal_upgrade_aborted()
         so prod announces 'Upgrade aborted sir.'

    Returns a dict shaped like:
        {"ok": bool, "stage_failed": str | None, "details": dict}
    """
    if _bgm is None or _stg is None:
        return {"ok": False,
                "stage_failed": "import",
                "details": {"error": "blue_green_manager or staging_instance not importable"}}

    print("\n=== BLUE/GREEN HANDOFF ===")

    candidate_files = [
        SCRIPT,
        os.path.join(PROJECT_DIR, "upgrade_jarvis.py"),
        os.path.join(PROJECT_DIR, "blue_green_manager.py"),
        os.path.join(PROJECT_DIR, "staging_instance.py"),
    ]

    print("[blue-green] running py_compile gate on candidate files...")
    ok, errors = _stg.precompile_candidate_files(candidate_files)
    if not ok:
        print(f"[blue-green] py_compile FAILED:")
        for e in errors[:5]:
            print(f"    {e}")
        _bgm.rollback(reason="py_compile_failed")
        # Let prod voice the abort so the user doesn't have to read logs.
        try:
            _bgm.signal_upgrade_aborted(reason="py_compile_failed")
        except Exception:
            pass
        return {"ok": False, "stage_failed": "py_compile",
                "details": {"errors": errors[:5]}}

    # Drop a stale staging lock if a prior cancelled run left one behind.
    if os.path.exists(_bgm.STAGING_LOCK_FILE):
        try:
            os.remove(_bgm.STAGING_LOCK_FILE)
        except OSError:
            pass

    _bgm.seed_staging_data()
    proc = spawn_staging_jarvis()
    if proc is None:
        _bgm.rollback(reason="staging_spawn_failed")
        try:
            _bgm.signal_upgrade_aborted(reason="staging_spawn_failed")
        except Exception:
            pass
        return {"ok": False, "stage_failed": "spawn",
                "details": {"error": "could not spawn --staging"}}
    print(f"[blue-green] staging JARVIS spawned (PID {proc.pid}). "
          f"Waiting for it to claim {os.path.basename(_bgm.STAGING_LOCK_FILE)}...")

    smoke = _stg.run_smoke_tests(
        candidate_files=[],   # already pre-compiled above
        boot_timeout_s=timeout_boot_s,
        reply_timeout_s=timeout_replies_s,
    )
    print(f"[blue-green] smoke verdict: ok={smoke['ok']} "
          f"failed_stage={smoke['stage_failed']} details={smoke['details']}")

    if not smoke["ok"]:
        # Tear down staging — leave prod intact.
        try:
            proc.terminate()
        except Exception:
            pass
        _bgm.rollback(reason=f"smoke_failed:{smoke.get('stage_failed')}")
        try:
            _bgm.signal_upgrade_aborted(
                reason=f"smoke_failed:{smoke.get('stage_failed')}",
            )
        except Exception:
            pass
        return {"ok": False,
                "stage_failed": smoke.get("stage_failed") or "smoke",
                "details": smoke.get("details") or {}}

    # ── Staging passed: wait for prod idle, signal handoff, let prod
    #     land the cinematic 'Switching to new version sir' line. ──
    target_version = _bgm.read_version()
    print(f"[blue-green] staging passed — waiting for prod idle...")
    _wait_for_prod_idle(timeout_s=8.0)
    print(f"[blue-green] signaling prod handoff "
          f"(grace {grace_seconds}s, target version {target_version})")
    _bgm.signal_handoff(reason="upgrade",
                        target_version=target_version,
                        grace_seconds=grace_seconds)

    # Give prod ~3 seconds to read the signal and land the announcement
    # before we start polling for its exit. Matches the choreography spec
    # in jarvis_todo.md (blue-green-2): announce → 3s → promote.
    time.sleep(3)

    # Give prod up to grace + 8s to exit. The grace is 10s by default —
    # plenty of margin for an in-flight TTS clip to drain. We poll the
    # lock file rather than trust a fixed sleep because prod may need a
    # second extra to finish a long sentence.
    handoff_deadline = time.time() + grace_seconds + 8
    while time.time() < handoff_deadline:
        if not _bgm.prod_is_running():
            break
        time.sleep(0.5)
    if _bgm.prod_is_running():
        print("[blue-green] WARN — prod did not exit within grace window. "
              "Signaling handoff failure and aborting the takeover.")
        try:
            _bgm.signal_handoff_failure(reason="prod_did_not_yield")
        except Exception:
            pass
        # Tear down staging so we don't leak two instances. Prod stays.
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        _bgm.rollback(reason="handoff_timeout")
        return {"ok": False, "stage_failed": "handoff_timeout",
                "details": {"target_version": target_version}}

    # ── Stop staging and bring up the new prod. ──
    print("[blue-green] stopping staging instance before new prod boot...")
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass
    # Ensure both lock files are cleared so the new prod can claim
    # jarvis.lock without tripping the singleton check.
    for stale in (_bgm.PROD_LOCK_FILE, _bgm.STAGING_LOCK_FILE):
        try:
            if os.path.exists(stale):
                os.remove(stale)
        except OSError:
            pass

    # Record the promotion in deployment_state.json.
    _bgm.promote_staging(new_version=target_version,
                         staging_pid=None)

    print("[blue-green] relaunching as new prod...")
    relaunch_jarvis(with_handoff=True)

    # Verify the new prod actually came up — heartbeat in instances.json
    # within 30s. Failure here is non-fatal (the launcher already kicked
    # PowerShell off), but it logs the outcome so the operator sees it.
    if _wait_for_new_prod_heartbeat(timeout_s=30.0):
        print("[blue-green] new prod heartbeat seen — handoff complete.")
        return {"ok": True, "stage_failed": None,
                "details": {"target_version": target_version,
                            "new_prod_heartbeat": True}}

    print("[blue-green] WARN — new prod did not heartbeat within 30s. "
          "Relaunch may still be in progress; check jarvis.lock.")
    return {"ok": True, "stage_failed": None,
            "details": {"target_version": target_version,
                        "new_prod_heartbeat": False}}


def main():
    auto_relaunch = "--relaunch" in sys.argv
    dry_audit     = "--dry-audit" in sys.argv
    single_stage  = "--single-stage" in sys.argv
    blue_green    = "--blue-green" in sys.argv

    # --force-trigger-gate-now: run a one-shot stability gate immediately and
    # exit. Used for the verify step in the task spec: deliberately break a
    # file, run this flag, and watch the gate catch + auto-revert + queue a
    # regression task. No backup/kill ceremony — the gate's smoke test
    # handles boot lifecycle.
    if "--force-trigger-gate-now" in sys.argv:
        print("=== STABILITY GATE: forced one-shot ===")
        result = _stability_gate(
            batch_n=0, recent_task_titles=["(forced-trigger)"],
            batch_size=_gate_config()[0],
        )
        print(f"\nverdict={result['verdict']}  ok={result['ok']}")
        print(f"details: {result['details']}")
        return

    # --blue-green: run the zero-downtime handoff ceremony directly.
    # Assumes the candidate code is already on disk (caller has run the
    # implementer + reviewer stages). Prod stays serving the user the
    # whole time; on PASS, we hand off in ~10s; on FAIL, we roll back.
    if "--blue-green" in sys.argv:
        print("=== JARVIS Upgrade Pipeline (BLUE/GREEN) ===")
        result = run_blue_green_handoff()
        print(f"\nverdict={'OK' if result['ok'] else 'FAIL'}  "
              f"stage_failed={result.get('stage_failed')}")
        sys.exit(0 if result["ok"] else 1)

    print(f"=== JARVIS Upgrade Pipeline ===")
    if dry_audit:
        print("(DRY AUDIT mode — read-only preview, no changes will be made)")
    elif single_stage:
        print("(SINGLE-STAGE mode — legacy one-call-per-task, no review/test gates)")
    else:
        print("(MULTI-AGENT mode — planner → implementer → reviewer → tester per task)")

    tasks = get_pending_tasks()
    print(f"Pending tasks in queue: {len(tasks)}")
    for t in tasks:
        print(f"  - {t[:90]}")

    if not tasks:
        print("\nNothing to upgrade. Exiting.")
        return

    if not dry_audit:
        # Snapshot current code before anything changes
        backup_path = backup_codebase()
        print(f"\nBackup saved → {backup_path}")

        killed = kill_running_jarvis()
        print(f"Stopped {killed} running JARVIS process(es).")
        time.sleep(2)
    else:
        # Dry-audit is read-only — no backup needed and no reason to disturb
        # any running JARVIS instance.
        print("\nSkipping backup + JARVIS-kill (dry-audit is read-only).")

    claude_path = find_claude_cli()
    if claude_path and dry_audit:
        # ---- Mode C: Dry audit — read-only review, no changes ----
        print(f"Found Claude Code at: {claude_path}")
        print("Launching DRY AUDIT in a new window...")
        proc = spawn_claude_dry_audit(tasks, claude_path)
        if proc is None:
            print("Failed. Aborting.")
            return
        print(f"\nClaude Code (PID {proc.pid}) running audit in its own window.")
        print("The report will be saved to dry_audit_<timestamp>.md in the project root.")
        return

    if claude_path:
        # ---- Mode A: Autonomous via `claude` CLI ----
        print(f"Found Claude Code at: {claude_path}")
        print("Launching Claude Code in a new window...")
        proc = spawn_claude_code(tasks, claude_path, single_stage=single_stage)
        if proc is None:
            print("Failed. Aborting.")
            return

        if auto_relaunch:
            print(f"\nWaiting for Claude Code (PID {proc.pid}) to finish...")
            try:
                proc.wait()
                print("\nClaude Code finished. Running syntax check...")
                time.sleep(1)

                # ── Syntax check all key files ───────────────────────────────
                import py_compile, glob as _glob
                syntax_errors: list[str] = []
                _files = (
                    [SCRIPT]
                    + _glob.glob(os.path.join(PROJECT_DIR, "skills", "*.py"))
                    + _glob.glob(os.path.join(PROJECT_DIR, "hud", "*.py"))
                    # core/ and tools/ are edited by the pipeline AND imported
                    # by bobert_companion at startup (`from core.* import *`), so
                    # a syntax error there bricks the boot just as surely as one
                    # in the entry point. Without compiling them, syntax_ok is a
                    # false green and the relaunch gate ships un-importable code.
                    # Recurse so package subdirs are covered.
                    + _glob.glob(os.path.join(PROJECT_DIR, "core", "**", "*.py"),
                                 recursive=True)
                    + _glob.glob(os.path.join(PROJECT_DIR, "tools", "**", "*.py"),
                                 recursive=True)
                )
                for _f in _files:
                    try:
                        py_compile.compile(_f, doraise=True)
                    except py_compile.PyCompileError as _ce:
                        syntax_errors.append(os.path.basename(_f) + ": " + str(_ce))
                syntax_ok = len(syntax_errors) == 0
                if syntax_ok:
                    print(f"  Syntax: OK ({len(_files)} files clean)")
                else:
                    print(f"  Syntax ERRORS ({len(syntax_errors)}):")
                    for _e in syntax_errors:
                        print(f"    {_e}")

                # ── Codebase auditor (post-upgrade gate) ─────────────────────
                # Runs tools/audit_codebase.py and surfaces any new P0/P1
                # findings before relaunch. Exit codes:
                #   0 = clean, 1 = P0, 2 = P1, 3 = only P2.
                audit_p0 = 0
                audit_p1 = 0
                audit_p2 = 0
                audit_script = os.path.join(PROJECT_DIR, "tools", "audit_codebase.py")
                if os.path.exists(audit_script):
                    print("\nRunning codebase auditor...")
                    try:
                        rc = subprocess.run(
                            [sys.executable, audit_script, "--quiet"],
                            cwd=PROJECT_DIR, timeout=120,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        rc = -1
                        print("  [audit] timed out after 120s — skipping")
                    report_path = os.path.join(PROJECT_DIR, "audit_report.json")
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, "r", encoding="utf-8") as _rf:
                                _audit = json.load(_rf)
                            audit_p0 = _audit.get("summary", {}).get("p0", 0)
                            audit_p1 = _audit.get("summary", {}).get("p1", 0)
                            audit_p2 = _audit.get("summary", {}).get("p2", 0)
                        except Exception:
                            pass
                    if rc == 0:
                        print(f"  Audit: CLEAN (0 findings)")
                    elif rc == 1:
                        print(f"  Audit: FAIL — {audit_p0} P0 finding(s); see audit_report.md")
                    elif rc == 2:
                        print(f"  Audit: WARN — {audit_p1} P1 finding(s); see audit_report.md")
                    elif rc == 3:
                        print(f"  Audit: INFO — {audit_p2} P2 finding(s); see audit_report.md")
                    else:
                        print(f"  Audit: skipped (rc={rc})")

                # ── Write summary so JARVIS announces changes on next start ──
                try:
                    with open(UPGRADE_SUMMARY_FILE, "w", encoding="utf-8") as _sf:
                        json.dump({
                            "upgraded_at": time.strftime("%H:%M"),
                            "tasks":       tasks,
                            "syntax_ok":   syntax_ok,
                            "syntax_errors": syntax_errors[:3],
                            "audit_p0":    audit_p0,
                            "audit_p1":    audit_p1,
                            "audit_p2":    audit_p2,
                        }, _sf, indent=2)
                    print(f"  Upgrade summary written → {UPGRADE_SUMMARY_FILE}")
                except Exception as _we:
                    print(f"  Couldn't write upgrade summary: {_we}")

                # ── Append a versioned changelog entry ───────────────────────
                _append_changelog_entry(
                    tasks_before=tasks,
                    syntax_ok=syntax_ok,
                    audit_p0=audit_p0,
                    audit_p1=audit_p1,
                    audit_p2=audit_p2,
                )

                # ── Relaunch gate ────────────────────────────────────────────
                # NEVER bring JARVIS back up on syntactically broken source — it
                # would crash on import and brick the assistant until a manual
                # fix. A failed syntax check reverts to the pre-upgrade snapshot
                # (backup_path, taken before the run) and relaunches THAT
                # known-good code, so JARVIS always comes back alive. This makes
                # the changelog's own "review backups before relaunch" note
                # (previously advisory only) actually enforcing, and mirrors the
                # stability gate's revert-on-fail contract. A P0 audit finding on
                # otherwise-valid syntax is surfaced loudly but does not auto-
                # revert a working upgrade (the auditor can false-positive, and
                # the pre-upgrade tree may carry the same pre-existing finding).
                if not syntax_ok:
                    print(f"\n⚠  Syntax check FAILED ({len(syntax_errors)} file(s)) — "
                          f"NOT relaunching broken code. Reverting to pre-upgrade backup…")
                    try:
                        revert_ok, revert_log = _revert_to_snapshot(backup_path)
                        print(f"  Revert {'OK' if revert_ok else 'FAILED'}: {revert_log[:200]}")
                    except Exception as _re:
                        print(f"  Revert FAILED: {_re} — JARVIS source may be broken; "
                              f"restore manually from {backup_path}")
                elif audit_p0:
                    print(f"\n⚠  Auditor flagged {audit_p0} P0 finding(s) post-upgrade — "
                          f"relaunching (syntax is valid) but review audit_report.md.")

                print("\nRelaunching JARVIS...")
                time.sleep(2)
                relaunch_jarvis()
                print("JARVIS relaunched.")
            except KeyboardInterrupt:
                print("\nInterrupted - JARVIS not relaunched.")
        else:
            print("\nWhen Claude Code finishes, manually run:")
            print(f"  python {os.path.basename(SCRIPT)}")
        return

    # ---- Mode B: Fallback — copy tasks to clipboard, open todo file ----
    print("\n⚠  `claude` CLI not found anywhere.")
    print("Falling back to manual handoff mode:")
    print("  1. Copying the pending tasks to your clipboard")
    print("  2. Opening jarvis_todo.md in your editor")
    print("  3. Paste them into your Claude Code chat (Cursor/VS Code/web)")
    print("  4. Tell Claude Code to implement them and tick them off")
    print("  5. Manually relaunch JARVIS when done\n")

    # Build a clipboard payload
    task_block = "\n".join(f"- [ ] {t}" for t in tasks)
    clipboard_text = (
        f"Hey Claude Code — please implement these pending tasks for the JARVIS PC companion at "
        f"{PROJECT_DIR}. Main script is bobert_companion.py. Read jarvis_todo.md for full context. "
        f"Tick items off as you complete them.\n\n"
        f"Pending tasks:\n{task_block}"
    )

    # Copy to Windows clipboard via PowerShell (no extra deps)
    try:
        ps = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
            stdin=subprocess.PIPE, text=True, close_fds=True,
        )
        ps.communicate(clipboard_text)
        print("  ✓ Copied to clipboard")
    except Exception as e:
        print(f"  ✗ Could not copy to clipboard: {e}")
        print("\n--- Paste this manually into Claude Code: ---")
        print(clipboard_text)
        print("---")

    # Open todo file in default editor
    try:
        os.startfile(TODO_FILE)
        print(f"  ✓ Opened {TODO_FILE}")
    except Exception as e:
        print(f"  ✗ Could not open editor: {e}")

    print("\nWhen Claude Code is done, relaunch JARVIS manually:")
    print(f"  python bobert_companion.py")
    print("\nOr install the `claude` CLI for full autonomy:")
    print("  https://docs.claude.com/en/docs/claude-code")
    print("\n(Press Enter to close this window)")
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint; never run under unittest
    main()
