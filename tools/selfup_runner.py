#!/usr/bin/env python3
"""
JARVIS SAFE self-upgrade runner (request-only, isolated, locked down).

WHY THIS EXISTS (read-only audit, 2026-10-14)
=============================================
The old design edited the LIVE C:\\JARVIS checkout with
--dangerously-skip-permissions, fired on 30 idle minutes and chained, killed
and relaunched JARVIS without _boot_jarvis.ps1, booted the real JARVIS for 90 s
per task, gated on a suite with a 900 s default (the suite needs ~16 min, so
every task "failed"), rolled back .py files only, skipped VERSION / PII / tags /
the crash watch, and could commit and push to the PUBLIC repo. This runner
replaces all of it. It is started ONLY by an explicit owner request (the
"upgrade" voice action or the tray's Run Upgrade Now) and works like this:

  1. ISOLATED   - `git worktree add -b auto/selfup-<stamp>` off origin/main in
                  a sibling directory (C:\\jarvis_wt_selfup_<stamp>). The live
                  checkout is never edited, and JARVIS is never killed,
                  relaunched or booted for a test.
  2. APPROVED   - tasks come from jarvis_todo.md (the owner's approved queue),
                  never from jarvis_todo_proposed.md. Lines tagged [self-heal]
                  or [anomaly] are machine-written and skipped by default.
  3. LOCKED DOWN- planner / reviewer get Read, Glob, Grep only; the implementer
                  gets an allow-list plus a deny-list (no git push / tag /
                  merge / reset / checkout / commit, nothing under data/) - see
                  tools/multi_agent_pipeline.implementer_permission_args.
  4. ONE COMMIT PER TASK - the runner commits after a task passes; a task that
                  fails is reverted with git and logged. Rollback of the whole
                  run = drop the branch.
  5. REAL GATE  - once, at the end:
                    python tools/gate_run.py <log> --cwd <wt> -- python tools/run_tests.py
                  with JARVIS_TEST_TOTAL_TIMEOUT_S=1800, then check_no_pii.py.
  6. NO RELEASE - no VERSION bump, no push, no tag. The branch is left for the
                  owner / a normal release session (VERSION, gates, push, tag,
                  restart, 30-minute crash watch).
  7. REPORT     - data/selfup_reports/selfup_<stamp>.md (+ latest.json with one
                  honest spoken sentence JARVIS says when the run ends).
  8. COST GUARD - the per-run USD cap (JARVIS_PIPELINE_MAX_USD), a task cap and
                  a consecutive-failure stop; a Claude usage limit that survives
                  core.claude_accounts' Pro -> Max fallback stops the run.

usage:
    python tools/selfup_runner.py [--list] [--max-tasks N] [--include-machine]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Callable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import multi_agent_pipeline as mp  # noqa: E402 - needs _ROOT on sys.path

PROJECT_DIR = _ROOT
TODO_FILE = os.path.join(PROJECT_DIR, "jarvis_todo.md")        # approved queue
REMOTE = "origin"
BASE_REF = "origin/main"
BRANCH_PREFIX = "auto/selfup-"
WORKTREE_PREFIX = "jarvis_wt_selfup_"

# The suite needs ~16 minutes; the old 900 s default failed every task.
GATE_TOTAL_TIMEOUT_S = 1800
_GATE_CEILING_S = GATE_TOTAL_TIMEOUT_S * 4 + 600 + 300   # gate_run's own ceiling + slack
DEFAULT_MAX_TASKS = 5
NO_PROGRESS_LIMIT = 5            # consecutive failed tasks before the run stops
LOCK_MAX_AGE_S = 8 * 3600

UNCHECKED_RE = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)
# Machine-written queue entries (core/diagnostic_daemons, boot self-heal).
MACHINE_TAG_RE = re.compile(r"\[(?:self-heal|anomaly)\]", re.IGNORECASE)

# Paths a task may never change (checked on the staged diff, on top of the
# implementer's own deny-list).
_FORBIDDEN_PREFIXES = ("data/", ".git/", ".claude/")
_FORBIDDEN_FILES = ("VERSION", "jarvis_todo.md")


# ───────────────────────────── approved queue ─────────────────────────────

def approved_tasks(todo_path: str | None = None, *,
                   include_machine: bool = False) -> list[str]:
    """Unchecked '- [ ]' lines of the approved queue, in file order. Lines
    tagged [self-heal] / [anomaly] are machine-written and skipped unless
    ``include_machine``. jarvis_todo_proposed.md is NEVER read here."""
    try:
        with open(todo_path or TODO_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    out = []
    for m in UNCHECKED_RE.finditer(text):
        line = m.group(1).strip()
        if not include_machine and MACHINE_TAG_RE.search(line):
            continue
        out.append(line)
    return out


def _task_key(task: str) -> str:
    return hashlib.sha1(task.strip().encode("utf-8")).hexdigest()[:16]


def _load_ledger(report_dir: str) -> dict:
    try:
        with open(os.path.join(report_dir, "ledger.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


# ──────────────────────────────── git helpers ────────────────────────────────

def _git(cwd: str, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          env=env)


def _git_out(cwd: str, *args: str) -> str:
    return (_git(cwd, *args).stdout or "").strip()


def _assert_isolated(wt: str, project_dir: str) -> None:
    """Destructive git (reset --hard / clean) runs ONLY in the run's own
    worktree: never the live checkout, never a plain clone."""
    a, b = os.path.realpath(wt), os.path.realpath(project_dir)
    if a == b or a.startswith(b + os.sep) or not os.path.isfile(os.path.join(wt, ".git")):
        raise RuntimeError(f"refusing git changes outside an isolated worktree: {wt}")


def create_worktree(project_dir: str, stamp: str, emit: Callable[[str], None]
                    ) -> tuple[str, str]:
    """`git worktree add -b auto/selfup-<stamp> <sibling dir> origin/main`.
    Returns (worktree_path, branch). Raises RuntimeError on failure."""
    fetch = _git(project_dir, "fetch", "-q", REMOTE)
    if fetch.returncode != 0:
        emit(f"  [git] fetch failed ({(fetch.stderr or '').strip()[:120]}); "
             f"using the local {BASE_REF}")
    if _git(project_dir, "rev-parse", "--verify", "-q", BASE_REF + "^{commit}").returncode != 0:
        raise RuntimeError(f"{BASE_REF} does not exist in {project_dir}")
    branch = BRANCH_PREFIX + stamp
    wt = os.path.join(os.path.dirname(os.path.abspath(project_dir)),
                      WORKTREE_PREFIX + stamp)
    add = _git(project_dir, "worktree", "add", "-b", branch, wt, BASE_REF)
    if add.returncode != 0:
        raise RuntimeError("git worktree add failed: "
                           + ((add.stderr or add.stdout or "").strip()[:300]))
    return wt, branch


def _revert(wt: str, project_dir: str, to: str = "HEAD") -> None:
    _assert_isolated(wt, project_dir)
    _git(wt, "reset", "--hard", "-q", to)
    _git(wt, "clean", "-fdq")


def _staged_paths(wt: str) -> list[str]:
    _git(wt, "add", "-A")
    out = _git(wt, "diff", "--cached", "--name-only", "-z").stdout or ""
    return [p for p in out.split("\0") if p]


def _forbidden(paths: list[str]) -> list[str]:
    return [p for p in paths
            if p in _FORBIDDEN_FILES or p.startswith(_FORBIDDEN_PREFIXES)]


def _ignored_in_data(wt: str) -> list[str]:
    """Git-ignored files under data/ never show up in the staged diff, so ask
    for them explicitly: a task that wrote there is a scope violation too."""
    out = _git(wt, "ls-files", "--others", "--ignored", "--exclude-standard",
               "-z", "--", "data").stdout or ""
    return [p for p in out.split("\0") if p]


def _smoke(wt: str, paths: list[str]) -> tuple[bool, str]:
    """Headless smoke: every changed .py must compile. compile() only parses -
    nothing is imported or executed, so it cannot open a mic, camera or
    window, and writes no __pycache__. (The real JARVIS is never booted.)"""
    for rel in paths:
        if not rel.endswith(".py"):
            continue
        full = os.path.join(wt, rel)
        if not os.path.isfile(full):
            continue            # deleted by the task
        try:
            with open(full, "r", encoding="utf-8") as f:
                compile(f.read(), rel, "exec")
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            return False, f"{rel}: {exc}"
    return True, "ok"


# ──────────────────────────────── one task ────────────────────────────────

def _is_limit(rc: Any) -> bool:
    return rc == 429


def run_one_task(task: str, *, wt: str, branch: str, project_dir: str,
                 claude_path: str, emit: Callable[[str], None]) -> dict[str, Any]:
    """plan -> implement -> diff -> review -> smoke -> COMMIT, or revert.
    Returns {task, status: done|failed|skipped, stage, sha, note, limit,
    stages}. Never raises for a task-level problem."""
    res: dict[str, Any] = {"task": task, "status": "failed", "stage": "",
                           "sha": "", "note": "", "limit": False, "stages": {}}
    stages = res["stages"]
    before = _git_out(wt, "rev-parse", "HEAD")

    def fail(stage: str, note: str, *, limit: bool = False) -> dict[str, Any]:
        _revert(wt, project_dir, before)
        res.update(status="failed", stage=stage, note=note[:300], limit=limit)
        emit(f"  [revert] {stage}: {note[:160]}")
        mp._log_pipeline_event({"stage": stage, "ok": False, "task": task[:200],
                                "note": note[:300], "branch": branch})
        return res

    emit("  [1/4] planner (read-only)")
    plan = mp._run_planner(task, claude_path=claude_path, project_dir=wt)
    stages["planner"] = plan
    if "error" in plan:
        return fail("planner", plan["error"], limit=_is_limit(plan.get("rc")))
    approach = (plan.get("approach") or "").strip()
    if approach.upper().startswith("IMPOSSIBLE:"):
        res.update(status="skipped", stage="planner", note=approach[:300])
        emit(f"  [skip] planner: {approach[:140]}")
        return res

    emit(f"  [2/4] implementer ({len(plan.get('files_to_touch', []))} file(s) planned)")
    impl = mp._run_implementer(task, plan, claude_path=claude_path, project_dir=wt)
    stages["implementer"] = impl
    if impl["rc"] != 0:
        return fail("implementer", f"implementer exited rc={impl['rc']}",
                    limit=_is_limit(impl["rc"]))
    head = _git_out(wt, "rev-parse", "HEAD")
    cur = _git_out(wt, "rev-parse", "--abbrev-ref", "HEAD")
    if head != before or cur != branch:
        if cur != branch:
            _git(wt, "checkout", "-f", "-q", branch)
        return fail("implementer", "implementer moved HEAD or the branch "
                    "(it must not commit or switch branches)")
    if impl["impossible"]:
        _revert(wt, project_dir, before)
        res.update(status="skipped", stage="implementer", note=impl["note"][:300])
        emit(f"  [skip] implementer: {impl['note'][:140]}")
        return res

    paths = _staged_paths(wt)
    bad = _forbidden(paths) + _ignored_in_data(wt)
    if bad:
        return fail("scope", "task touched forbidden path(s): " + ", ".join(bad[:5]))
    diff = (_git(wt, "diff", "--cached", "--no-color").stdout or "").strip() \
        or "(no file changes)"
    emit(f"  [3/4] reviewer (read-only) - {len(paths)} file(s) changed")
    review = mp._run_reviewer(task, plan, diff, claude_path=claude_path,
                              project_dir=wt,
                              impl_already_done=bool(impl.get("already_done")),
                              impl_note=impl.get("note") or "")
    stages["reviewer"] = review
    verdict = review.get("verdict", "approve_with_warnings")
    score = review.get("risk_score", 5)
    concerns = list(review.get("concerns") or [])
    if _is_limit(review.get("rc")):
        return fail("reviewer", "Claude usage limit during review", limit=True)
    # A high risk_score BLOCKS even on an approve verdict; degraded reviewer
    # paths (infra error / bad JSON) are scored 8 and fail closed here.
    if verdict in ("approve", "approve_with_warnings") \
            and mp._coerce_risk(score) >= mp._max_risk_score():
        concerns.append(f"risk_score {score} >= threshold {mp._max_risk_score()}")
        verdict = "reject_and_redo"
    emit(f"        verdict={verdict} risk={score} concerns={len(concerns)}")
    if verdict == "reject_and_redo":
        return fail("reviewer", "; ".join(str(c) for c in concerns[:3]) or "rejected")
    if verdict == "already_done" or not paths:
        _revert(wt, project_dir, before)
        res.update(status="skipped", stage="reviewer",
                   note=f"already done: {(impl.get('note') or '')[:200]}")
        return res

    ok, detail = _smoke(wt, paths)
    if not ok:
        return fail("smoke", detail)

    title = re.sub(r"\s+", " ", re.sub(r"[*`]", "", task)).strip()
    msg = [f"selfup: {title[:72]}", "",
           f"Task: {task[:400]}", f"Implementer: {(impl.get('note') or '-')[:200]}",
           f"Reviewer: {verdict}, risk {score}", "",
           "Committed by tools/selfup_runner.py; not released (no VERSION bump, "
           "no push, no tag)."]
    commit = _git(wt, "commit", "-q", "-m", "\n".join(msg))
    if commit.returncode != 0:
        return fail("commit", (commit.stderr or commit.stdout or "commit failed").strip())
    res.update(status="done", stage="complete", sha=_git_out(wt, "rev-parse", "HEAD"),
               note=(impl.get("note") or "")[:300])
    emit(f"  [4/4] committed {res['sha'][:10]}")
    mp._log_pipeline_event({"stage": "complete", "ok": True, "task": task[:200],
                            "verdict": verdict, "risk_score": score,
                            "sha": res["sha"], "branch": branch})
    return res


# ───────────────────────────────── the gate ─────────────────────────────────

def run_gate(wt: str, report_dir: str, stamp: str, *,
             runner: Callable = subprocess.run) -> dict[str, Any]:
    """The REAL gate on the finished branch: the full suite through
    tools/gate_run.py with the 1800 s whole-run budget, then the PII scan."""
    os.makedirs(report_dir, exist_ok=True)
    log = os.path.join(report_dir, f"gate_{stamp}.log")
    env = dict(os.environ, JARVIS_TEST_TOTAL_TIMEOUT_S=str(GATE_TOTAL_TIMEOUT_S))
    out: dict[str, Any] = {"log": log, "budget_s": GATE_TOTAL_TIMEOUT_S,
                           "tests_ok": False, "tests_rc": None,
                           "pii_ok": False, "pii_rc": None}
    try:
        t = runner([sys.executable, "tools/gate_run.py", log, "--cwd", wt, "--",
                    sys.executable, "tools/run_tests.py"],
                   cwd=wt, env=env, capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=_GATE_CEILING_S)
        out["tests_rc"] = t.returncode
        out["tests_ok"] = t.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        out["tests_error"] = repr(exc)[:200]
    try:
        p = runner([sys.executable, "tools/check_no_pii.py"], cwd=wt, env=env,
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", timeout=600)
        out["pii_rc"] = p.returncode
        out["pii_ok"] = p.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        out["pii_error"] = repr(exc)[:200]
    out["ok"] = bool(out["tests_ok"] and out["pii_ok"])
    return out


# ───────────────────────────────── reporting ─────────────────────────────────

def spoken_summary(r: dict[str, Any]) -> str:
    """One honest sentence for JARVIS to say when the run ends."""
    if r.get("error"):
        return f"The upgrade could not start, sir: {str(r['error'])[:90]}."
    if not r.get("attempted"):
        return "The approved upgrade queue has nothing to run, sir."
    done, failed = len(r.get("done", [])), len(r.get("failed", []))
    stopped = " I stopped early on a Claude usage limit." if r.get("stopped") == "limit" else ""
    if not done:
        return (f"The upgrade run finished with no changes, sir: "
                f"{failed} of {r['attempted']} tasks failed.{stopped}")
    noun = "task" if done == 1 else "tasks"
    gate = r.get("gate") or {}
    if gate.get("ok"):
        return (f"The upgrade branch is ready for review, sir: "
                f"{done} {noun} done, tests passed.{stopped}")
    why = "the tests failed" if not gate.get("tests_ok") else "the privacy scan failed"
    return (f"The upgrade branch needs a look before anything ships, sir: "
            f"{done} {noun} done, but {why}.{stopped}")


def write_report(report_dir: str, stamp: str, r: dict[str, Any]) -> str:
    """Markdown report + latest.json. Returns the markdown path."""
    os.makedirs(report_dir, exist_ok=True)
    gate = r.get("gate")
    lines = [f"# Self-upgrade run {stamp}", "",
             f"- Branch: `{r.get('branch') or '(none)'}`",
             f"- Worktree: `{r.get('worktree') or '(none)'}`",
             f"- Base: `{BASE_REF}`",
             f"- Stopped: {r.get('stopped') or 'queue / task cap reached'}",
             f"- Est. spend: ${r.get('spend_usd', 0.0):.2f}", ""]
    if r.get("error"):
        lines += [f"**Error:** {r['error']}", ""]
    lines.append(f"## Done ({len(r.get('done', []))})")
    lines += [f"- `{d['sha'][:10]}` {d['task'][:140]}" for d in r.get("done", [])] or ["- none"]
    lines += ["", f"## Failed, reverted ({len(r.get('failed', []))})"]
    lines += [f"- [{d['stage']}] {d['task'][:110]} - {d['note'][:120]}"
              for d in r.get("failed", [])] or ["- none"]
    lines += ["", f"## Skipped ({len(r.get('skipped', []))})"]
    lines += [f"- {d['task'][:110]} - {d['note'][:100]}"
              for d in r.get("skipped", [])] or ["- none"]
    lines += ["", "## Gate"]
    if gate:
        lines += [f"- Tests (budget {gate.get('budget_s')} s): "
                  f"{'PASS' if gate.get('tests_ok') else 'FAIL'} (rc {gate.get('tests_rc')})",
                  f"- PII scan: {'PASS' if gate.get('pii_ok') else 'FAIL'} (rc {gate.get('pii_rc')})",
                  f"- Log: `{gate.get('log')}`"]
    else:
        lines.append("- not run (no task was committed)")
    lines += ["", "## Next steps (owner)",
              "- Review: `git log origin/main.." + str(r.get("branch") or "<branch>") + "`",
              "- Ship via a normal release session: VERSION, gates, push, tag, restart, "
              "30-minute crash watch. Then tick the finished lines in jarvis_todo.md.",
              "- Discard: `git worktree remove --force <worktree>` and `git branch -D <branch>`.",
              "", f"_{r.get('spoken', '')}_", ""]
    path = os.path.join(report_dir, f"selfup_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    r["report"] = path
    _write_json_atomic(os.path.join(report_dir, "latest.json"), r)
    return path


# ─────────────────────────────────── lock ───────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return bool(psutil.pid_exists(pid))
    except Exception:
        return True            # unknown: assume alive; LOCK_MAX_AGE_S bounds it


def is_running(report_dir: str) -> bool:
    try:
        with open(os.path.join(report_dir, ".running"), encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - float(data.get("at", 0)) > LOCK_MAX_AGE_S:
            return False
        return _pid_alive(int(data.get("pid", 0)))
    except (OSError, ValueError, TypeError):
        return False


def _acquire(report_dir: str) -> bool:
    if is_running(report_dir):
        return False
    _write_json_atomic(os.path.join(report_dir, ".running"),
                       {"pid": os.getpid(), "at": time.time()})
    return True


def _release(report_dir: str) -> None:
    try:
        os.remove(os.path.join(report_dir, ".running"))
    except OSError:
        pass


# ──────────────────────────────────── run ────────────────────────────────────

def run(*, project_dir: str = PROJECT_DIR, claude_path: str,
        todo_path: str | None = None, report_dir: str | None = None,
        include_machine: bool = False, max_tasks: int | None = None,
        emit: Callable[[str], None] = print,
        gate_fn: Callable[..., dict] | None = None,
        stamp: str | None = None) -> dict[str, Any]:
    """One whole run. Returns the result dict (also written as the report)."""
    report_dir = report_dir or os.path.join(project_dir, "data", "selfup_reports")
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    if max_tasks is None:
        try:
            max_tasks = int(os.environ.get("JARVIS_SELFUP_MAX_TASKS", DEFAULT_MAX_TASKS))
        except ValueError:
            max_tasks = DEFAULT_MAX_TASKS
    r: dict[str, Any] = {"stamp": stamp, "branch": "", "worktree": "",
                         "attempted": 0, "done": [], "failed": [], "skipped": [],
                         "spend_usd": 0.0, "stopped": "", "gate": None}
    os.makedirs(report_dir, exist_ok=True)
    if not _acquire(report_dir):
        r["error"] = "another upgrade run is already in progress"
        r["spoken"] = spoken_summary(r)
        return r
    try:
        ledger = _load_ledger(report_dir)
        queue = [t for t in approved_tasks(
                    todo_path or os.path.join(project_dir, "jarvis_todo.md"),
                    include_machine=include_machine)
                 if ledger.get(_task_key(t), {}).get("status") != "done"]
        queue = queue[:max(1, max_tasks)]
        if not queue:
            r["spoken"] = spoken_summary(r)
            write_report(report_dir, stamp, r)
            return r
        try:
            wt, branch = create_worktree(project_dir, stamp, emit)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            r["error"] = str(exc)[:300]
            r["spoken"] = spoken_summary(r)
            write_report(report_dir, stamp, r)
            return r
        r.update(branch=branch, worktree=wt)
        emit(f"=== JARVIS safe self-upgrade: {len(queue)} approved task(s) -> "
             f"{branch} ({wt}) ===")

        cap = mp._pipeline_max_usd()
        streak = 0
        for n, task in enumerate(queue, 1):
            if cap > 0 and r["spend_usd"] >= cap:
                r["stopped"] = "budget"
                emit(f"[budget] est. ${r['spend_usd']:.2f} reached the ${cap:.2f} cap - stopping")
                break
            emit(f"[task {n}/{len(queue)}] {task[:120]}")
            try:
                out = run_one_task(task, wt=wt, branch=branch, project_dir=project_dir,
                                   claude_path=claude_path, emit=emit)
            except Exception as exc:        # a crash in one task is that task's failure
                try:
                    _revert(wt, project_dir)
                except Exception:
                    pass
                out = {"task": task, "status": "failed", "stage": "exception",
                       "note": repr(exc)[:300], "sha": "", "limit": False, "stages": {}}
            r["attempted"] += 1
            r["spend_usd"] += mp._estimate_task_cost_usd(out)
            kind = {"done": "done", "skipped": "skipped"}.get(out["status"], "failed")
            r[kind].append({k: out.get(k, "") for k in ("task", "stage", "sha", "note")})
            ledger[_task_key(task)] = {"status": out["status"], "branch": branch,
                                       "sha": out.get("sha", ""), "at": stamp}
            _write_json_atomic(os.path.join(report_dir, "ledger.json"), ledger)
            if out.get("limit"):
                r["stopped"] = "limit"
                emit("[limit] Claude refused on Pro and Max - stopping the run")
                break
            streak = streak + 1 if out["status"] == "failed" else 0
            if streak >= NO_PROGRESS_LIMIT:
                r["stopped"] = "no-progress"
                emit(f"[loop] {streak} failures in a row - stopping")
                break

        if r["done"]:
            emit("=== final gate (full suite, 1800 s budget, then PII scan) ===")
            try:
                r["gate"] = (gate_fn or run_gate)(wt, report_dir, stamp)
            except Exception as exc:
                r["gate"] = {"ok": False, "tests_ok": False, "pii_ok": False,
                             "tests_rc": None, "pii_rc": None, "log": "",
                             "budget_s": GATE_TOTAL_TIMEOUT_S, "error": repr(exc)[:200]}
        else:
            _assert_isolated(wt, project_dir)       # nothing kept: tidy up our own branch
            _git(project_dir, "worktree", "remove", "--force", wt)
            _git(project_dir, "branch", "-D", branch)
            r["worktree"] = r["branch"] = ""
        r["spoken"] = spoken_summary(r)
        write_report(report_dir, stamp, r)
        emit(r["spoken"])
        return r
    finally:
        _release(report_dir)


# ───────────────────────────────────── CLI ─────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="JARVIS safe self-upgrade runner "
                                 "(request-only, isolated worktree, locked down).")
    ap.add_argument("--list", action="store_true",
                    help="print the approved tasks that would run, then exit")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--include-machine", action="store_true",
                    help="also run [self-heal] / [anomaly] lines (skipped by default)")
    args = ap.parse_args(argv)
    tasks = approved_tasks(include_machine=args.include_machine)
    if args.list:
        for t in tasks:
            print("-", t[:140])
        print(f"{len(tasks)} approved task(s)")
        return 0
    claude_path = mp._find_claude_cli()
    if not claude_path:
        print("ERROR: claude CLI not found.", file=sys.stderr)
        return 127
    r = run(claude_path=claude_path, include_machine=args.include_machine,
            max_tasks=args.max_tasks)
    return 0 if not r.get("error") else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
