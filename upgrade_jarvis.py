#!/usr/bin/env python3
"""
JARVIS Upgrade entry point (SAFE design, 2026-10-15)
────────────────────────────────────────────────────
Started only when the owner asks (the `upgrade` action / tray "Run Upgrade
Now") - never by an idle timer. The work itself lives in tools/selfup_runner.py:

1. Create a fresh git worktree off origin/main (branch auto/selfup-<stamp>).
2. Take the owner-approved tasks from jarvis_todo.md ([self-heal] / [anomaly]
   lines are skipped; jarvis_todo_proposed.md is never read).
3. For each task: planner (read-only) -> implementer (locked-down Claude Code,
   edits only the worktree) -> reviewer (read-only) -> headless compile smoke
   -> ONE git commit, or a git revert.
4. Run the real gate once (tools/gate_run.py + tools/run_tests.py with a
   1800 s budget, then tools/check_no_pii.py) and write a report under
   data/selfup_reports/.
The live checkout is never edited and the live JARVIS is never killed,
relaunched or booted. No VERSION bump, no push, no tag: the branch waits for
a normal release session.

Manual usage:
    python upgrade_jarvis.py                # the safe run above
    python upgrade_jarvis.py --dry-audit    # read-only preview of what Claude would do
                                            # (no changes, no worktree)
    python upgrade_jarvis.py --blue-green   # owner-typed zero-downtime handoff of
                                            # code ALREADY on disk (stops/starts JARVIS)
    python upgrade_jarvis.py --force-trigger-gate-now   # one-shot stability gate

Pipeline model knobs (env vars, optional):
    JARVIS_PIPELINE_PLANNER_MODEL       default "haiku"
    JARVIS_PIPELINE_IMPLEMENTER_MODEL   default "opus"
    JARVIS_PIPELINE_REVIEWER_MODEL      default "sonnet"
    JARVIS_PIPELINE_MAX_USD             default 25 (per-run cost cap)
    JARVIS_SELFUP_MAX_TASKS             default 5

Requires:
    - `claude` CLI installed and on PATH (Claude Code)
    - A Claude subscription - the pipeline strips ANTHROPIC_API_KEY and
      runs Claude Code on a subscription, NOT on metered API credits.
      core/claude_accounts.py picks which one per call (CLAUDE_PRO_FIRST_UNTIL:
      a second Pro profile first while it lasts, Max on its limits).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
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

# ═════════════ Upgrade-in-progress marker (watchdog handshake) ════════════
# UPDATE 2026-10-15: the default run (tools/selfup_runner.py) works in a
# separate git worktree and never stops or boots JARVIS, so it holds no marker;
# only the owner-typed --blue-green / --force-trigger-gate-now flags do. The
# history below describes the retired live-tree run and is kept for the lease
# design, which those flags still use.
#
# WHY THIS EXISTS (H-7, 2026-08-20).
# tools/jarvis_watchdog.py resurrects JARVIS whenever it sees "no
# bobert_companion process AND no data/clean_shutdown.flag". That is exactly
# the state an upgrade run leaves behind for HOURS:
#
#   • The production entry point is JARVIS itself: _overnight_upgrade_thread
#     -> _act_upgrade spawns `python upgrade_jarvis.py --relaunch` and then
#     self-exits via _hard_exit(clean=True), which DOES write
#     data/clean_shutdown.flag. So at pipeline START the watchdog correctly
#     stands down. That part was never broken.
#   • The flag does not SURVIVE the run. The pipeline's tester stage
#     (tools/multi_agent_pipeline.py stage 4/4, and _run_stability_smoke_test
#     / _stability_gate below) boots a real PROD JARVIS for the smoke test,
#     and bobert_companion's boot path DELETES data/clean_shutdown.flag
#     unconditionally ("delete the flag NOW so a crash after a prior clean
#     shutdown is still resurrected"). The tester then kills that instance
#     with `Stop-Process -Force` = TerminateProcess, which never reaches
#     atexit / _write_clean_shutdown_flag. Neither killer writes the flag
#     back: `clean_shutdown` has ZERO hits in upgrade_jarvis.py,
#     tools/multi_agent_pipeline.py, tools/stability_smoke_test.py and
#     _boot_jarvis.ps1.
#
# From the FIRST task's tester stage onward, for the rest of a multi-hour
# run, the flag is gone and JARVIS is dead — so the watchdog fires every 5
# minutes and boots a full interactive JARVIS (mic live, TTS unmuted) out of
# a tree the implementer stage is mid-rewrite on. The same window opens for a
# manual `python upgrade_jarvis.py` run, where main()'s kill_running_jarvis()
# TerminateProcess-es a live JARVIS without leaving a flag.
#
# THE MIRROR-IMAGE BUG WE MUST NOT RECREATE. From 2026-07-15 to 07-21 a crash
# left clean_shutdown.flag behind and the watchdog politely declined to
# resurrect anything for six days. A marker that is never cleared silently
# disables the resurrection net, which is WORSE than the bug it fixes. So
# this marker is bounded three independent ways:
#
#   pid           owner process id. A dead owner = abandoned marker.
#   expires_at    short lease, refreshed by a heartbeat thread every
#                 _MARKER_HEARTBEAT_S. Owner dies or wedges -> the lease
#                 lapses within ttl (default 10 min) and the net re-arms.
#   hard_deadline absolute ceiling stamped ONCE at acquire and never
#                 extended, so even a heartbeat thread still running after
#                 the upgrade has wedged cannot park the watchdog past
#                 max_age (default 6 h).
#
# The two deadlines are written into the file as ABSOLUTE EPOCH TIMESTAMPS on
# purpose. The reader (tools/jarvis_watchdog._upgrade_in_progress) then needs
# no copy of this policy — it only compares two numbers against now(). That
# makes this project's #1 bug class (the STALE DUPLICATE: a rule fixed in one
# copy while the other rots) structurally impossible here: retune a timeout
# below and the watchdog follows automatically, with nothing to keep in sync.

_MARKER_HEARTBEAT_S = 60          # lease-refresh cadence


def _marker_config() -> tuple[int, int]:
    """Return (ttl_s, max_age_s) for the upgrade-in-progress marker.

    Env: JARVIS_UPGRADE_MARKER_TTL_S (default 600) and
    JARVIS_UPGRADE_MARKER_MAX_AGE_S (default 21600 = 6h). The ttl floor is
    two heartbeats — a lease shorter than that would expire between
    refreshes and re-open the very window it exists to close."""
    try:
        ttl = max(_MARKER_HEARTBEAT_S * 2,
                  int(os.environ.get("JARVIS_UPGRADE_MARKER_TTL_S", "600")))
    except ValueError:
        ttl = 600
    try:
        max_age = int(os.environ.get("JARVIS_UPGRADE_MARKER_MAX_AGE_S", "21600"))
    except ValueError:
        max_age = 21600
    return ttl, max(ttl, max_age)


def _upgrade_marker_path() -> str:
    """Marker path, derived at CALL time rather than import time.

    Deliberate: the unit suite redirects PROJECT_DIR into a per-test tempdir,
    and a module-level constant baked at import would make every main() test
    write a REAL C:\\JARVIS\\data\\upgrade_in_progress.flag — parking the live
    watchdog on the developer's machine for hours."""
    return os.path.join(PROJECT_DIR, "data", "upgrade_in_progress.flag")


# depth: re-entrancy refcount within THIS process.
# owned: True only when this process actually wrote the file, so a nested or
#        child acquire can never delete an outer owner's marker.
_marker_state: dict = {"depth": 0, "owned": False, "stop": None,
                       "thread": None, "started_at": 0.0,
                       "hard_deadline": 0.0}


def _marker_pid_alive(pid: int) -> bool:
    """Is `pid` genuinely executing? Uses core.parent_watch
    (GetExitCodeProcess + WaitForSingleObject), which is corpse-aware — a
    kernel-stuck 'terminating forever' row must not count as a live owner.

    Unknown -> True (assume alive). Safe, because expires_at / hard_deadline
    already bound the marker regardless of what the pid check says."""
    try:
        from core.parent_watch import parent_is_alive  # type: ignore
    except Exception:
        return True
    try:
        return bool(parent_is_alive(pid))
    except Exception:
        return True


def _write_upgrade_marker() -> bool:
    """(Re)write the marker atomically with a refreshed lease. Never raises."""
    ttl, _ = _marker_config()
    now = time.time()
    path = _upgrade_marker_path()
    payload = {
        "pid":           os.getpid(),
        "started_at":    _marker_state["started_at"],
        "heartbeat_at":  now,
        "expires_at":    now + ttl,
        "hard_deadline": _marker_state["hard_deadline"],
        "argv":          " ".join(sys.argv)[:300],
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # Atomic swap: the watchdog polls this file from another process every
        # 5 minutes and must never read a half-written one.
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _read_upgrade_marker() -> "dict | None":
    """Parse the marker, or None when absent / unreadable / malformed."""
    try:
        with open(_upgrade_marker_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _marker_is_live(data: "dict | None") -> bool:
    """True while `data` still holds the watchdog off.

    The same three bounds the watchdog applies, read out of the file rather
    than duplicated: lease not lapsed, hard ceiling not reached, owner alive."""
    if not data:
        return False
    now = time.time()
    try:
        if now >= float(data.get("expires_at", 0)):
            return False
        hard = float(data.get("hard_deadline", 0))
        if hard and now >= hard:
            return False
    except (TypeError, ValueError):
        return False
    pid = data.get("pid")
    if isinstance(pid, int) and pid > 0 and not _marker_pid_alive(pid):
        return False
    return True


def acquire_upgrade_marker() -> bool:
    """Declare "the JARVIS source tree is being rewritten" BEFORE the first
    mutating step, so the watchdog stops reading "dead + no clean-shutdown
    flag" as a crash to resurrect.

    Re-entrant within a process (refcount). Across processes: a LIVE marker
    owned by somebody else is left alone and False is returned — that owner
    releases it, never us. An ABANDONED marker (dead owner, lapsed lease, or
    past the hard ceiling) is taken over. Returns True when WE own it.
    Never raises."""
    _marker_state["depth"] += 1
    if _marker_state["owned"]:
        return True
    existing = _read_upgrade_marker()
    if _marker_is_live(existing) and existing.get("pid") != os.getpid():
        print(f"  [upgrade-marker] pid {existing.get('pid')} already holds it "
              f"— running under their marker, not taking it over")
        return False
    _, max_age = _marker_config()
    now = time.time()
    _marker_state["started_at"] = now
    _marker_state["hard_deadline"] = now + max_age
    if not _write_upgrade_marker():
        print("  [upgrade-marker] COULD NOT WRITE the marker — the watchdog "
              "may resurrect JARVIS from half-written source mid-upgrade")
        return False
    _marker_state["owned"] = True
    stop = threading.Event()
    _marker_state["stop"] = stop

    def _heartbeat() -> None:
        # Event.wait, NOT time.sleep: the unit suite stubs time.sleep to a
        # no-op, and a heartbeat built on it would busy-spin a daemon thread
        # for the whole test run.
        while not stop.wait(_MARKER_HEARTBEAT_S):
            if time.time() >= _marker_state["hard_deadline"]:
                break          # never extend past the ceiling
            if not _marker_state["owned"]:
                break
            _write_upgrade_marker()

    thread = threading.Thread(target=_heartbeat,
                              name="upgrade-marker-heartbeat", daemon=True)
    _marker_state["thread"] = thread
    thread.start()
    print(f"  [upgrade-marker] held (pid {os.getpid()}, ceiling {max_age}s) "
          f"— watchdog resurrection paused for this run")
    return True


def release_upgrade_marker() -> None:
    """Drop the marker. MUST run from a finally: an unreleased marker disables
    the resurrection net until its lease lapses. Only the process that wrote
    the file deletes it, and only while it still owns it, so a child or nested
    acquire can never clear its parent's. Never raises."""
    if _marker_state["depth"] > 0:
        _marker_state["depth"] -= 1
    if _marker_state["depth"] > 0 or not _marker_state["owned"]:
        return
    stop = _marker_state.get("stop")
    thread = _marker_state.get("thread")
    _marker_state["owned"] = False
    if stop is not None:
        try:
            stop.set()
        except Exception:
            pass
    if thread is not None:
        # Join before unlinking. The heartbeat is parked in Event.wait(), so
        # this returns immediately; without it the thread can wake between
        # `owned = False` and the unlink and rewrite the marker we just tried
        # to delete. Bounded (it would lapse on its own inside the lease), but
        # the deterministic fix costs nothing.
        try:
            thread.join(timeout=5)
        except Exception:
            pass
    _marker_state["stop"] = None
    _marker_state["thread"] = None
    data = _read_upgrade_marker()
    if data is not None and data.get("pid") not in (None, os.getpid()):
        return          # somebody took it over after our lease lapsed
    try:
        os.remove(_upgrade_marker_path())
    except OSError:
        pass


class upgrade_in_progress:
    """`with upgrade_in_progress():` — acquire/release around a whole run.

    .owned tells the caller whether the marker is actually ours; the body
    runs either way (a missing marker degrades to the pre-H-7 behaviour — it
    must never abort an upgrade)."""

    def __init__(self) -> None:
        self.owned = False

    def __enter__(self) -> "upgrade_in_progress":
        self.owned = acquire_upgrade_marker()
        return self

    def __exit__(self, *exc) -> bool:
        release_upgrade_marker()
        return False


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
        # Strip ANTHROPIC_API_KEY so Claude Code bills to a subscription.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        # The call goes through tools/claude_account_run.py (2026-10-14): the
        # owner's Pro account first while it lasts, Max when Pro is at a limit.
        runner = os.path.join(PROJECT_DIR, "tools", "claude_account_run.py")

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
            f"& '{sys.executable}' '{runner}' '{claude_path}' --print --verbose "
            f"'{safe_prompt}' "
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
    # H-7: reap any PROD JARVIS before spawning the replacement. This path
    # does NOT go through _boot_jarvis.ps1 (which kills stale prod itself), so
    # without this an instance the WATCHDOG resurrected mid-upgrade still holds
    # jarvis.lock; the upgraded process then trips the singleton check, prints
    # "[singleton] Another JARVIS already holds the lock", sys.exit(0)s into a
    # PowerShell window opened without -NoExit, and the caller cheerfully
    # reports success for a relaunch that never happened. kill_running_jarvis()
    # excludes --staging, so a green/blue-green instance is never touched.
    try:
        reaped = kill_running_jarvis()
        if reaped:
            print(f"  [relaunch] reaped {reaped} stale PROD JARVIS process(es) "
                  f"before relaunch")
            time.sleep(2)
    except Exception as _ke:
        print(f"  [relaunch] pre-kill failed ({_ke!r}) — spawning anyway")
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

    # Prod is confirmed gone. If the signal is STILL on disk, nobody ever acted
    # on it: prod was already down when the ceremony started, or it died inside
    # the ~3-21 s window between signal_handoff() and its next tick. Either way
    # the grace loop broke on its first poll, so signal_handoff_failure() was
    # never written and nothing else in the tree deletes this file. Leaving it
    # makes the freshly promoted prod consume a takeover addressed to its
    # PREDECESSOR: it announces "Switching to the new version, sir" and exits
    # 10 s into its first main loop, while this pipeline prints "handoff
    # complete" and returns ok:True. consume_handoff_signal() now refuses a
    # pre-boot signal as a second line of defence; this is the first.
    try:
        if _bgm.clear_unconsumed_handoff_signal():
            print("[blue-green] the handoff signal was never consumed (prod "
                  "was not running) — cleared it before promote so the new "
                  "prod cannot inherit its predecessor's takeover.")
    except Exception as _cue:
        print(f"[blue-green] WARN — could not clear the handoff signal: "
              f"{type(_cue).__name__}: {_cue}")

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


# Flags that still reach LIVE-tree machinery (a staging handoff / a one-shot
# gate). They are owner-typed maintenance commands - JARVIS's voice, tray and
# idle paths never start them - and they are the only runs that stop or
# relaunch JARVIS, so they alone hold the upgrade-in-progress marker.
_LIVE_MAINTENANCE_FLAGS = ("--blue-green", "--force-trigger-gate-now")


def main():
    """The upgrade-in-progress marker (see above) parks the watchdog so it does
    not resurrect JARVIS from half-written source while a run has him stopped.
    The default run no longer stops JARVIS (it works in a separate git
    worktree), so only the live maintenance flags hold the marker; --dry-audit
    is read-only and deliberately NOT covered either."""
    if not any(f in sys.argv for f in _LIVE_MAINTENANCE_FLAGS):
        return _main_body()
    with upgrade_in_progress():
        return _main_body()


def _main_body():
    dry_audit = "--dry-audit" in sys.argv

    # --force-trigger-gate-now: run a one-shot stability gate immediately and
    # exit. Used for the verify step in the task spec: deliberately break a
    # file, run this flag, and watch the gate catch + auto-revert + queue a
    # regression task. No backup/kill ceremony - the gate's smoke test
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

    for gone in ("--relaunch", "--single-stage"):
        if gone in sys.argv:
            print(f"({gone} no longer exists: an upgrade run works in a separate "
                  f"git worktree, never edits or restarts the live JARVIS, and "
                  f"always runs the review + test gates.)")

    print("=== JARVIS Upgrade Pipeline ===")
    claude_path = find_claude_cli()

    if dry_audit:
        # Read-only preview of the approved queue: no worktree, no changes.
        print("(DRY AUDIT mode - read-only preview, no changes will be made)")
        tasks = get_pending_tasks()
        print(f"Pending tasks in queue: {len(tasks)}")
        for t in tasks:
            print(f"  - {t[:90]}")
        if not tasks:
            print("\nNothing to upgrade. Exiting.")
            return
        if not claude_path:
            print("`claude` CLI not found - cannot run the audit.")
            return
        print(f"Found Claude Code at: {claude_path}")
        print("Launching DRY AUDIT in a new window...")
        proc = spawn_claude_dry_audit(tasks, claude_path)
        if proc is None:
            print("Failed. Aborting.")
            return
        print(f"\nClaude Code (PID {proc.pid}) running audit in its own window.")
        print("The report will be saved to dry_audit_<timestamp>.md in the project root.")
        return

    # ---- The run: request-only, isolated worktree, locked-down Claude ----
    # tools/selfup_runner.py owns everything (worktree, one commit per task,
    # the real gate, the report). Nothing here touches the live tree.
    if not claude_path:
        print("`claude` CLI not found - nothing can be upgraded. Install Claude "
              "Code (https://docs.claude.com/en/docs/claude-code) and re-run.")
        sys.exit(127)
    from tools import selfup_runner
    result = selfup_runner.run(project_dir=PROJECT_DIR, claude_path=claude_path)
    sys.exit(1 if result.get("error") else 0)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint; never run under unittest
    main()
