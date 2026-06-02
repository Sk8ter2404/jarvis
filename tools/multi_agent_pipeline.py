#!/usr/bin/env python3
"""
JARVIS multi-agent upgrade pipeline.

Replaces the single `claude --print` per task in upgrade_jarvis.py with a
4-stage pipeline. Each stage is a SCOPED Claude invocation (different model,
different tool grant, narrower context):

    1. PLANNER     — cheap model, READ-only. Outputs JSON:
                       files_to_touch, regression_risks, tests_to_run, approach.
    2. IMPLEMENTER — Opus, write-enabled. Gets ONLY the files the planner
                       identified + the planner's risk notes. The only stage
                       that uses --dangerously-skip-permissions.
    3. REVIEWER    — Sonnet, READ-only. Sees the diff (computed via difflib
                       against the per-task backup) + planner risks. Outputs
                       JSON: risk_score 0-10, concerns[], verdict
                       (approve | reject_and_redo | approve_with_warnings).
    4. TESTER      — local, no Claude. Spawns JARVIS via _boot_jarvis.ps1,
                       waits 90s, checks process liveness + APPCRASH +
                       FATAL lines in session log.

Safety gates (between stages 3 and 4, and after 4):
  * RISK GATE — a reviewer risk_score AT/ABOVE JARVIS_PIPELINE_MAX_RISK
    (default 7) is escalated to reject_and_redo even on an approve verdict.
    Degraded reviewer paths (infra error / unparseable JSON) are scored 8 so
    they fail CLOSED through this gate instead of shipping unreviewed.
  * CORRECTNESS GATE — after the tester proves JARVIS still BOOTS, the unit
    suite (tools/run_tests.py) runs; a change that breaks a tested contract
    rolls back even though it booted clean.

Approve (risk < threshold) → tick the task, continue.
Reject / risk >= threshold → restore files from per-task backup, append
           [regression] task, do NOT tick the original.
Tester or unit-suite fail → same as reject_and_redo (restore + queue regression).

Configuration (env vars, all optional):
    JARVIS_PIPELINE_PLANNER_MODEL       default "haiku"
    JARVIS_PIPELINE_IMPLEMENTER_MODEL   default "opus"
    JARVIS_PIPELINE_REVIEWER_MODEL      default "sonnet"
    JARVIS_PIPELINE_TESTER_WAIT_S       default 90  (smoke-test deadline)
    JARVIS_PIPELINE_MAX_RISK            default 7   (risk_score >= this blocks)
    JARVIS_PIPELINE_SKIP_TESTER         "1" to skip stage 4 (debug only)
    JARVIS_PIPELINE_SKIP_SUITE          "1" to skip the correctness gate (debug)
    JARVIS_PIPELINE_MAX_USD             default 25.0 (per-RUN dollar ceiling;
                                          the loop stops at a task boundary once
                                          estimated spend reaches it)

Safety budget cap (mirrors core/diagnostic_daemons.py's DeepAudit daily cap):
  The loop driver tallies a COARSE per-stage cost ESTIMATE (see
  _STAGE_COST_ESTIMATE_USD) as each task runs and stops cleanly — exactly like
  the no-progress watchdog — once the running total reaches JARVIS_PIPELINE_MAX_USD.
  A recent uncapped run cost $413.90 over 579 iterations; this ceiling bounds
  the blast radius of an autonomous loop. It stops at a TASK boundary so an
  in-flight apply is never severed mid-edit. (Real per-call cost is available
  from `claude --print --output-format json` via `total_cost_usd`; this proxy is
  intentionally conservative because the cap is a safety net, not an accountant.)

The single-stage flow is still reachable via `upgrade_jarvis.py --single-stage`
for emergencies.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

# Make console output UTF-8-safe (this pipeline prints arrows '→' / box-chars);
# a redirected file or legacy cp1252 console otherwise crashes with
# UnicodeEncodeError mid-cycle (see upgrade_jarvis.py — 2026-05-31 gate crash).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - import-time guard for consoles lacking reconfigure
        pass


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODO_FILE = os.path.join(PROJECT_DIR, "jarvis_todo.md")
PIPELINE_LOG = os.path.join(PROJECT_DIR, "data", "pipeline_runs.jsonl")
PIPELINE_BACKUP_ROOT = os.path.join(PROJECT_DIR, "backups", "pipeline")
STABILITY_SCRIPT = os.path.join(PROJECT_DIR, "tools", "stability_smoke_test.py")
BOOT_SCRIPT = os.path.join(PROJECT_DIR, "_boot_jarvis.ps1")
LOCK_FILE = os.path.join(PROJECT_DIR, "jarvis.lock")

UNCHECKED_RE = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Substrings that the Claude CLI prints to stdout/stderr when it hits a usage
# cap. The CLI exits 0 in this case, so we have to sniff the message to avoid
# falsely treating the rate-limited response as a successful invocation.
_RATE_LIMIT_NEEDLES: tuple[str, ...] = (
    # Tightened 2026-05-28 (round3-M3-3): removed "please try again later"
    # which matched organic LLM prose. Each remaining needle is specific
    # enough that it only appears in actual rate-limit error responses,
    # not in conversational output the LLM might emit while implementing
    # a fix.
    "claude ai usage limit reached",
    "claude usage limit reached",
    "usage limit reached",
    "rate_limit_error",
    "rate limit exceeded",
    "you have been rate limited",
    "anthropic rate limit",
)


def _looks_rate_limited(stdout: str, stderr: str) -> bool:
    """True if either stream contains a known Claude-CLI rate-limit marker.
    Match is case-insensitive and substring-based; we only inspect the first
    ~4 KB of each stream because the marker, when present, sits at the top."""
    haystack = ((stdout or "")[:4096] + "\n" + (stderr or "")[:4096]).lower()
    return any(needle in haystack for needle in _RATE_LIMIT_NEEDLES)


def _env_model(name: str, default: str) -> str:
    val = os.environ.get(name, "").strip()
    return val if val else default


def _planner_model() -> str:
    return _env_model("JARVIS_PIPELINE_PLANNER_MODEL", "haiku")


def _implementer_model() -> str:
    return _env_model("JARVIS_PIPELINE_IMPLEMENTER_MODEL", "opus")


def _reviewer_model() -> str:
    return _env_model("JARVIS_PIPELINE_REVIEWER_MODEL", "sonnet")


def _tester_wait_s() -> int:
    try:
        return max(20, int(os.environ.get("JARVIS_PIPELINE_TESTER_WAIT_S", "90")))
    except ValueError:
        return 90


def _tester_disabled() -> bool:
    return os.environ.get("JARVIS_PIPELINE_SKIP_TESTER", "").strip() == "1"


# ──────────────────────────── log helpers ────────────────────────────

def _log_pipeline_event(event: dict[str, Any]) -> None:
    """Append a single JSON line to data/pipeline_runs.jsonl. Best-effort —
    never raises, so the pipeline itself can't be derailed by a log write."""
    try:
        os.makedirs(os.path.dirname(PIPELINE_LOG), exist_ok=True)
        event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


# ──────────────────────────── task helpers ────────────────────────────

def _read_todo() -> str:
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        return f.read()


def _first_unchecked_task() -> str | None:
    """Return the FIRST unchecked task text (without the '- [ ] ' prefix),
    or None if none remain."""
    try:
        text = _read_todo()
    except OSError:
        return None
    m = UNCHECKED_RE.search(text)
    return m.group(1).strip() if m else None


def _count_unchecked() -> int:
    try:
        text = _read_todo()
    except OSError:
        return 0
    return len(UNCHECKED_RE.findall(text))


def _tick_task(task_text: str, done_note: str) -> bool:
    """Flip `- [ ] <task_text>` to `- [x] <task_text> ✓ DONE — <note>` in
    jarvis_todo.md. Returns True on success."""
    try:
        text = _read_todo()
    except OSError:
        return False
    needle = f"- [ ] {task_text}"
    if needle not in text:
        return False
    replacement = f"- [x] {task_text} ✓ DONE — {done_note}"
    new_text = text.replace(needle, replacement, 1)
    try:
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            f.write(new_text)
    except OSError:
        return False
    return True


def _append_regression_task(parent_task: str, stage: str, details: str) -> None:
    """Append a [regression] task to the END of jarvis_todo.md so the queue
    naturally picks it up on a future iteration."""
    short = parent_task[:160]
    snippet = details[:400].replace("\n", " ")
    line = (
        f"- [ ] **[regression]** Pipeline {stage} failed on "
        f"'{short}'. Details: {snippet}\n"
    )
    try:
        with open(TODO_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


# ─────────────────────────── backup helpers ───────────────────────────

def _backup_files(files: list[str], task_slug: str) -> str:
    """Copy the given files (paths relative to PROJECT_DIR OR absolute) into
    a per-task subdirectory of backups/pipeline/<ts>_<slug>/. Returns that
    directory path. Files that don't exist yet are recorded as MISSING so
    restore knows to delete them on rollback (i.e. the implementer created
    them)."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", task_slug)[:40] or "task"
    dest = os.path.join(PIPELINE_BACKUP_ROOT, f"{ts}_{safe_slug}")
    os.makedirs(dest, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for f in files:
        abs_src = f if os.path.isabs(f) else os.path.join(PROJECT_DIR, f)
        rel = os.path.relpath(abs_src, PROJECT_DIR)
        # Refuse to back up files outside the project root.
        if rel.startswith(".."):
            continue
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(abs_src):
            try:
                shutil.copy2(abs_src, target)
                manifest.append({"path": rel, "existed_before": True})
            except OSError as exc:
                manifest.append({"path": rel, "existed_before": True,
                                 "backup_error": str(exc)})
        else:
            manifest.append({"path": rel, "existed_before": False})
    try:
        with open(os.path.join(dest, "_manifest.json"), "w", encoding="utf-8") as mf:
            json.dump({"ts": ts, "files": manifest}, mf, indent=2)
    except OSError:
        pass
    return dest


def _restore_files(backup_dir: str) -> tuple[int, int]:
    """Roll back to the state captured in _backup_files. Returns
    (restored_count, deleted_count). Files that didn't exist before the
    implementer ran are deleted; files that did exist are copied back."""
    manifest_path = os.path.join(backup_dir, "_manifest.json")
    restored = 0
    deleted = 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
    except (OSError, ValueError):
        return (0, 0)
    for entry in manifest.get("files", []):
        rel = entry.get("path", "")
        if not rel:
            continue
        abs_target = os.path.join(PROJECT_DIR, rel)
        if entry.get("existed_before"):
            src = os.path.join(backup_dir, rel)
            if os.path.exists(src):
                try:
                    os.makedirs(os.path.dirname(abs_target), exist_ok=True)
                    shutil.copy2(src, abs_target)
                    restored += 1
                except OSError:
                    pass
        else:
            if os.path.exists(abs_target):
                try:
                    os.remove(abs_target)
                    deleted += 1
                except OSError:
                    pass
    return (restored, deleted)


def _compute_diff(backup_dir: str, files: list[str]) -> str:
    """Unified diff of every file in `files` (current vs backed-up). Returns
    a single concatenated diff string. Truncates each file's diff to 400
    lines so the reviewer's prompt doesn't blow past its context limit."""
    chunks: list[str] = []
    for rel in files:
        abs_now = os.path.join(PROJECT_DIR, rel) if not os.path.isabs(rel) \
            else rel
        rel_norm = os.path.relpath(abs_now, PROJECT_DIR)
        before_path = os.path.join(backup_dir, rel_norm)

        try:
            if os.path.exists(before_path):
                with open(before_path, "r", encoding="utf-8",
                          errors="replace") as f:
                    before_lines = f.read().splitlines()
            else:
                before_lines = []
        except OSError:
            before_lines = []
        try:
            if os.path.exists(abs_now):
                with open(abs_now, "r", encoding="utf-8",
                          errors="replace") as f:
                    after_lines = f.read().splitlines()
            else:
                after_lines = []
        except OSError:
            after_lines = []

        if before_lines == after_lines:
            continue

        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f"a/{rel_norm}", tofile=f"b/{rel_norm}",
            lineterm="",
        ))
        if not diff:
            continue
        if len(diff) > 400:
            diff = diff[:400] + [f"... (diff truncated at 400 lines, "
                                 f"{len(diff) - 400} more)"]
        chunks.append("\n".join(diff))
    return "\n\n".join(chunks) if chunks else "(no file changes)"


# ──────────────────────────── Claude wrapper ────────────────────────────

def _strip_json(raw: str) -> str:
    """Pull a JSON object out of model output. Handles ```json fences,
    leading prose, and trailing prose. Returns the matched JSON substring,
    or the original input if nothing matched."""
    raw = (raw or "").strip()
    m = JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    # Fallback: scan for the first '{' and try to find a balanced closer.
    # Walk the string respecting JSON "..." quotes (with \" escapes) so that
    # braces inside string literals — e.g. {"approach": "use {placeholder}"} —
    # don't fool the depth counter into returning a truncated object.
    start = raw.find("{")
    if start < 0:
        return raw
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return raw[start:]


def _invoke_claude(
    *,
    prompt: str,
    model: str,
    claude_path: str,
    allow_writes: bool,
    project_dir: str = PROJECT_DIR,
    timeout_s: int = 600,
    extra_args: list[str] | None = None,
) -> tuple[int, str, str]:
    """Invoke `claude --print` in headless mode and return
    (returncode, stdout, stderr). The implementer stage passes
    allow_writes=True (which adds --dangerously-skip-permissions); the
    planner/reviewer stages get read-only access.

    Stdout is the FINAL textual response. We use plain --print (no
    --output-format stream-json) here because the orchestrator wants the
    last reply, not a tool-use trace. The driver still emits its own
    progress lines so the user can see stage transitions."""
    cmd = [claude_path, "--print"]
    if model:
        cmd += ["--model", model]
    if allow_writes:
        cmd.append("--dangerously-skip-permissions")
    if extra_args:
        cmd += extra_args
    cmd.append("--")
    cmd.append(prompt)

    env = os.environ.copy()
    # Bill to Max subscription, not API credits.
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return (124, exc.stdout or "", f"timeout after {timeout_s}s")
    except FileNotFoundError as exc:
        return (127, "", f"claude binary not found: {exc}")
    except OSError as exc:
        return (1, "", f"failed to spawn claude: {exc!r}")

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    rc = result.returncode
    # The claude CLI exits 0 even when it returned a "you hit your usage
    # limit" message instead of doing the work. Downstream stages treat rc=0
    # as success, so we synthesise rc=429 to surface the rate-limit clearly.
    if rc == 0 and _looks_rate_limited(stdout, stderr):
        return (429, stdout,
                (stderr + "\nrate-limited: claude CLI returned a usage-limit "
                          "message; synthesised rc=429").strip())
    return (rc, stdout, stderr)


# ────────────────────────────── stage 1: PLANNER ──────────────────────────────

_PLANNER_PROMPT = """You are the PLANNER stage of the JARVIS multi-agent upgrade pipeline.

Your job: scope a single task. You are READ-ONLY. Do NOT use Edit, Write,
NotebookEdit, or any other mutating tool. You MAY use Read, Glob, Grep.

The task (verbatim from jarvis_todo.md):
─────────────────────────────────────
{task_line}
─────────────────────────────────────

Project root: {project_dir}
Main script:  bobert_companion.py
Skills:       skills/*.py
Tools:        tools/*.py

Read the task carefully, then explore ONLY the files you need to scope the
work. Don't read the whole codebase — read the entry points + the files the
task is clearly going to touch.

Your final reply MUST be a single JSON object (no prose around it) with EXACTLY
these keys:

  {{
    "files_to_touch":    ["relative/path/to/file.py", ...],
    "regression_risks":  ["one sentence per risk", ...],
    "tests_to_run":      ["py_compile bobert_companion.py", "boot smoke test", ...],
    "approach":          "one paragraph describing the implementation strategy"
  }}

Rules:
  - files_to_touch MUST be paths relative to {project_dir}. Use forward slashes.
  - If the task is genuinely impossible or hardware-dependent, set
    "approach" to start with "IMPOSSIBLE:" and explain why. The pipeline will
    skip implementation and tick the task with that note.
  - Be concrete. "Various files" is not a valid files_to_touch entry.
  - Cap files_to_touch at 12. If the change is broader, pick the highest-impact
    12 and note the rest in regression_risks.
"""


def _run_planner(task_line: str, *, claude_path: str,
                 project_dir: str = PROJECT_DIR) -> dict[str, Any]:
    """Run the planner stage. Returns the parsed JSON plan, or a dict with
    `{"error": ...}` if parsing failed."""
    prompt = _PLANNER_PROMPT.format(task_line=task_line, project_dir=project_dir)
    rc, out, err = _invoke_claude(
        prompt=prompt,
        model=_planner_model(),
        claude_path=claude_path,
        allow_writes=False,
        project_dir=project_dir,
        timeout_s=600,
    )
    if rc != 0:
        return {"error": f"planner exited rc={rc}", "stderr": err[:500]}
    raw = _strip_json(out)
    try:
        plan = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return {"error": f"could not parse planner JSON: {exc}",
                "raw": out[:1000]}
    # Normalize / sanity-check
    plan.setdefault("files_to_touch", [])
    plan.setdefault("regression_risks", [])
    plan.setdefault("tests_to_run", [])
    plan.setdefault("approach", "")
    if not isinstance(plan["files_to_touch"], list):
        plan["files_to_touch"] = []
    plan["files_to_touch"] = [str(p).replace("\\", "/")
                              for p in plan["files_to_touch"]][:12]
    return plan


# ──────────────────────────── stage 2: IMPLEMENTER ────────────────────────────

_IMPLEMENTER_PROMPT = """You are the IMPLEMENTER stage of the JARVIS multi-agent
upgrade pipeline. You have FULL tool access (--dangerously-skip-permissions).

The task (verbatim from jarvis_todo.md):
─────────────────────────────────────
{task_line}
─────────────────────────────────────

The PLANNER scoped the work for you. STICK TO IT unless you discover the plan
is wrong — then narrow your changes, do not balloon scope.

Plan approach:
{approach}

Files the planner identified to touch (focus your edits here):
{files_block}

Regression risks the planner flagged (avoid these specifically):
{risks_block}

Project root: {project_dir}. Do all work end-to-end:
  1. Read whatever you need.
  2. Make the code changes.
  3. Run `python -m py_compile` on every file you touched. Fix any errors.
  4. Do NOT edit jarvis_todo.md — the orchestrator handles ticking the task.
  5. Do NOT spawn JARVIS, run boot scripts, or shell out to long-running
     subprocesses — the TESTER stage handles that.
  6. When done, print a one-line summary starting with "DONE:" so the
     orchestrator can capture your note.

If the task turns out to be hardware-dependent or impossible, print a single
line starting with "IMPOSSIBLE:" explaining why. The pipeline will still tick
the task with that note (no rollback) so it stops blocking the queue.

If the task is ALREADY COMPLETE — i.e. the target file/code already exists in
the working tree, py_compile is clean, sibling dependencies resolve, and there
is genuinely nothing to do — print a single line starting with "ALREADY_DONE:"
followed by concrete evidence (e.g. "ALREADY_DONE: skills/foo.py exists at
rev abc123, py_compile clean, imported by bar.py:42"). The reviewer will
VERIFY your claim (check the file exists, compiles, and deps resolve) and
either approve or reject. Use this ONLY when the working tree already
satisfies the task — do NOT use it to skip work you should have done.
"""


def _run_implementer(task_line: str, plan: dict[str, Any], *,
                     claude_path: str,
                     project_dir: str = PROJECT_DIR) -> dict[str, Any]:
    files = plan.get("files_to_touch", [])
    risks = plan.get("regression_risks", [])
    files_block = ("\n".join(f"  - {p}" for p in files)
                   if files else "  (planner did not name specific files)")
    risks_block = ("\n".join(f"  - {r}" for r in risks)
                   if risks else "  (none flagged)")
    prompt = _IMPLEMENTER_PROMPT.format(
        task_line=task_line,
        approach=plan.get("approach", "(no approach provided)"),
        files_block=files_block,
        risks_block=risks_block,
        project_dir=project_dir,
    )
    rc, out, err = _invoke_claude(
        prompt=prompt,
        model=_implementer_model(),
        claude_path=claude_path,
        allow_writes=True,
        project_dir=project_dir,
        timeout_s=1800,  # implementer can take a while on big tasks
    )
    note = ""
    impossible = False
    already_done = False
    # Parse the LAST matching marker in stdout so a model that prints
    # exploration prose containing the word "DONE:" before its real verdict
    # doesn't fool us. Markers are mutually exclusive; the last one wins.
    for line in (out or "").splitlines():
        s = line.strip()
        if s.startswith("DONE:"):
            note = s[len("DONE:"):].strip()
            impossible = False
            already_done = False
        elif s.startswith("IMPOSSIBLE:"):
            note = s[len("IMPOSSIBLE:"):].strip()
            impossible = True
            already_done = False
        elif s.startswith("ALREADY_DONE:"):
            note = s[len("ALREADY_DONE:"):].strip()
            already_done = True
            impossible = False
    return {
        "rc": rc,
        "stdout_tail": (out or "")[-2000:],
        "stderr_tail": (err or "")[-500:],
        "note": note,
        "impossible": impossible,
        "already_done": already_done,
    }


# ────────────────────────────── stage 3: REVIEWER ──────────────────────────────

_REVIEWER_PROMPT = """You are the REVIEWER stage of the JARVIS multi-agent
upgrade pipeline. You are READ-ONLY. Do NOT use Edit, Write, NotebookEdit, or
any mutating tool. You MAY use Read, Glob, Grep to investigate further.

The task being implemented:
─────────────────────────────────────
{task_line}
─────────────────────────────────────

The PLANNER flagged these regression risks:
{risks_block}

Implementer's claim: {impl_claim}

Diff produced by the IMPLEMENTER (against the per-task backup):
─────────────────────────────────────
{diff}
─────────────────────────────────────

KNOWN PATTERNS TO FLAG (be aggressive — this is the last gate before tests):
  • Native binding races: PortAudio thread, COM apartment, OpenCV release
    paths, cv2.VideoCapture not released, pyttsx3/sapi5 reused across threads.
  • Unchecked exceptions in hot paths (audio loop, wake-word loop, main
    dispatch loop) that would kill the daemon thread silently.
  • Undocumented actions: new ACTIONS entries that aren't added to
    PC_CONTROL_PROMPT in bobert_companion.py.
  • Prompt drift: changes to PC_CONTROL_PROMPT that drop coverage of
    existing actions, or actions that no longer match documented aliases.
  • subprocess.Popen / subprocess.run without timeout=, threading.Thread
    without daemon=True, open() without encoding=, bare `except:` clauses.
  • Globals mutated from non-main threads without locks.
  • Files written non-atomically (json.dump straight to a state file
    instead of tempfile+os.replace via core/atomic_io.py).

──── ALREADY-DONE VERIFICATION PATH ────
If the implementer claimed "ALREADY_DONE" (see "Implementer's claim" above),
your job is NOT to review a diff (there is none). Your job is to VERIFY the
claim by checking the working tree:
  1. Does the file/code the task asks for actually exist? (Use Read/Glob.)
  2. Does `python -m py_compile` pass on it? (Trust the implementer's note if
     they cited it; spot-check with Read if anything looks off.)
  3. Do the sibling dependencies the task implies actually resolve?
     (e.g. if the task says "wire X into Y", grep that Y imports/calls X.)
  4. Is there an OBVIOUS bug visible on a Read of the file? (Surface-level
     only — you are not doing a deep audit, just confirming the claim.)
If 1-3 hold and 4 finds nothing damning: verdict "already_done", risk_score
0-3, concerns describing what you verified. If the claim is FALSE (file
missing, doesn't compile, deps unresolved, or there's a clear bug):
verdict "reject_and_redo", risk_score 7+, concerns naming what's missing.
Do NOT use "already_done" if you see ANY diff lines above — that path is
exclusively for the no-diff + ALREADY_DONE claim combination.

Do NOT reject an already-done claim because of hypothetical concerns (TOCTOU
windows, future drain order, edge cases that don't exist in the current tree).
Verify the claim against the CURRENT state, not against imagined futures.

Your final reply MUST be a single JSON object (no prose around it):

  {{
    "risk_score":  0,
    "concerns":    ["one sentence per concern", ...],
    "verdict":     "approve" | "approve_with_warnings" | "reject_and_redo" | "already_done"
  }}

Rules:
  - risk_score is 0-10 (0 = bulletproof, 10 = definitely broken).
  - Use "reject_and_redo" ONLY if the change introduces a regression or fails
    to address the task. Style nits are "approve_with_warnings".
  - Use "already_done" ONLY when the implementer claimed ALREADY_DONE and you
    verified the claim per the section above.
  - If the diff is empty AND the implementer didn't claim IMPOSSIBLE or
    ALREADY_DONE, treat that as a failure: verdict "reject_and_redo",
    risk_score 8+, concern "implementer made no file changes".
"""


def _run_reviewer(task_line: str, plan: dict[str, Any], diff: str, *,
                  claude_path: str,
                  project_dir: str = PROJECT_DIR,
                  impl_impossible: bool = False,
                  impl_already_done: bool = False,
                  impl_note: str = "") -> dict[str, Any]:
    risks = plan.get("regression_risks", [])
    risks_block = ("\n".join(f"  - {r}" for r in risks)
                   if risks else "  (none flagged)")
    # Cap diff in the prompt so we don't OOM the context.
    if len(diff) > 24_000:
        diff = diff[:24_000] + "\n... (diff truncated at 24000 chars)"
    if impl_already_done:
        claim_line = (f"ALREADY_DONE — task target already exists in tree. "
                      f"Evidence: {impl_note[:300] or '(no evidence cited)'}")
    elif impl_impossible:
        claim_line = (f"IMPOSSIBLE — implementer aborted. "
                      f"Reason: {impl_note[:300] or '(no reason cited)'}")
    else:
        claim_line = f"DONE — {impl_note[:300] or '(no note)'}"
    prompt = _REVIEWER_PROMPT.format(
        task_line=task_line,
        risks_block=risks_block,
        impl_claim=claim_line,
        diff=diff,
    )
    rc, out, err = _invoke_claude(
        prompt=prompt,
        model=_reviewer_model(),
        claude_path=claude_path,
        allow_writes=False,
        project_dir=project_dir,
        timeout_s=600,
    )

    # An empty diff with no IMPOSSIBLE marker means the implementer silently
    # did nothing (most often: rate-limited claude call returning rc=0 with a
    # usage-limit message). Override locally — we don't trust the reviewer's
    # fallback paths to catch this, since both reviewer infra errors and
    # unparseable JSON resolve to approve_with_warnings.
    no_changes = diff.strip() == "(no file changes)"

    def _force_reject(extra_concern: str) -> dict[str, Any]:
        return {
            "verdict": "reject_and_redo",
            "risk_score": 9,
            "concerns": [
                "implementer produced no file changes "
                "(likely rate-limited or aborted)",
                extra_concern,
            ],
            "_local_override": True,
        }

    if rc != 0:
        if no_changes and not impl_impossible and not impl_already_done:
            return _force_reject(f"reviewer infra error rc={rc}: {err[:200]}")
        # FAIL CLOSED: a reviewer infra failure means the change went UNreviewed.
        # We score it 8 (>= the default risk threshold) so the safety gate in
        # run_pipeline_on_task escalates it to a rollback rather than shipping an
        # unreviewed self-edit. (A persistently-flaky reviewer stalls the queue,
        # which the no-progress watchdog + circuit breaker then halt — the safe
        # failure mode for a system that edits its own brain.)
        return {
            "verdict": "approve_with_warnings",
            "risk_score": 8,
            "concerns": [f"reviewer infra error rc={rc}: {err[:200]} "
                         "(unreviewed — failing closed)"],
            "_infra_error": True,
        }
    raw = _strip_json(out)
    try:
        review = json.loads(raw)
    except (ValueError, TypeError):
        if no_changes and not impl_impossible and not impl_already_done:
            return _force_reject("reviewer returned unparseable JSON")
        # FAIL CLOSED (see infra-error path above): an unparseable review is no
        # review — score it 8 so the safety gate rolls it back.
        return {
            "verdict": "approve_with_warnings",
            "risk_score": 8,
            "concerns": ["reviewer returned unparseable JSON "
                         "(unreviewed — failing closed)"],
            "raw": out[:800],
        }
    review.setdefault("verdict", "approve_with_warnings")
    review.setdefault("risk_score", 5)
    review.setdefault("concerns", [])
    if review["verdict"] not in ("approve", "approve_with_warnings",
                                 "reject_and_redo", "already_done"):
        review["verdict"] = "approve_with_warnings"
    # Guard: the already_done verdict is only valid when the implementer
    # actually claimed it. If the reviewer freelances "already_done" on a
    # normal DONE flow, downgrade to approve_with_warnings so we still run
    # the tester. Mirror-image: if there IS a diff but reviewer said
    # already_done, that's also invalid.
    if review["verdict"] == "already_done" and (
            not impl_already_done or not no_changes):
        review["verdict"] = "approve_with_warnings"
        concerns = list(review.get("concerns") or [])
        concerns.insert(0, "reviewer returned already_done but implementer "
                           "did not claim it (or diff is non-empty); "
                           "downgraded to approve_with_warnings")
        review["concerns"] = concerns
        review["_local_override"] = True
    # Empty diff with no IMPOSSIBLE/ALREADY_DONE claim → force reject.
    if no_changes and not impl_impossible and not impl_already_done \
            and review["verdict"] != "reject_and_redo":
        concerns = list(review.get("concerns") or [])
        concerns.insert(0, "implementer produced no file changes "
                           "(likely rate-limited or aborted)")
        review["verdict"] = "reject_and_redo"
        review["risk_score"] = max(int(review.get("risk_score") or 0), 9)
        review["concerns"] = concerns
        review["_local_override"] = True
    return review


# ─────────────────────────────── stage 4: TESTER ───────────────────────────────

def _run_tester(*, project_dir: str = PROJECT_DIR) -> dict[str, Any]:
    """Spawn JARVIS via _boot_jarvis.ps1, wait JARVIS_PIPELINE_TESTER_WAIT_S
    seconds (default 90), check process liveness + APPCRASH + FATAL lines.

    Reuses the existing tools/stability_smoke_test.py which already
    implements all three checks. We pass --wait to shorten its default 300s
    window down to the per-task deadline."""
    if _tester_disabled():
        return {"ok": True, "skipped": True,
                "reason": "JARVIS_PIPELINE_SKIP_TESTER=1"}

    if not os.path.exists(STABILITY_SCRIPT):
        return {"ok": True, "skipped": True,
                "reason": f"missing {STABILITY_SCRIPT}"}

    wait_s = _tester_wait_s()
    # +60s headroom: smoke test does its own launch + lock polling.
    timeout_s = wait_s + 90

    try:
        result = subprocess.run(
            [sys.executable, STABILITY_SCRIPT, "--wait", str(wait_s)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "skipped": False,
                "error": f"smoke test timed out after {timeout_s}s",
                "stdout_tail": (exc.stdout or "")[-1500:]}
    except OSError as exc:
        return {"ok": False, "skipped": False,
                "error": f"could not spawn smoke test: {exc!r}"}

    # The smoke test writes its own PASS/FAIL json next to the project root;
    # rc==0 is the canonical success signal.
    ok = result.returncode == 0
    return {
        "ok": ok,
        "rc": result.returncode,
        "stdout_tail": (result.stdout or "")[-1500:],
        "stderr_tail": (result.stderr or "")[-500:],
    }


def _suite_disabled() -> bool:
    return os.environ.get("JARVIS_PIPELINE_SKIP_SUITE", "").strip() == "1"


def _max_risk_score() -> int:
    """A reviewer ``risk_score`` AT OR ABOVE this value BLOCKS the change
    (escalates to reject_and_redo + rollback) even when the verdict was
    ``approve``/``approve_with_warnings``. Before this gate the score was purely
    advisory and a risk=9 change shipped if it merely booted. Env override:
    ``JARVIS_PIPELINE_MAX_RISK`` (default 7; clamped to 0-10)."""
    try:
        return max(0, min(10, int(os.environ.get("JARVIS_PIPELINE_MAX_RISK", "7"))))
    except (TypeError, ValueError):
        return 7


# ───────────────────────── per-run USD budget cap ─────────────────────────
# A per-RUN dollar ceiling on the loop, mirroring the daily USD cap the
# DeepAudit daemon already enforces (core/diagnostic_daemons.py:
# DEEP_AUDIT_DEFAULT_BUDGET_USD / _deep_audit_budget_usd /
# _deep_audit_budget_ok). The main pipeline had NO cap — a single uncapped run
# cost $413.90 over 579 iterations. We stop the loop CLEANLY at a task boundary
# (like the no-progress watchdog) once the running spend reaches the ceiling,
# so an in-flight apply is never severed mid-edit.

PIPELINE_DEFAULT_MAX_USD = 25.0

# Coarse per-STAGE cost ESTIMATE (USD), summed per task as a defensible proxy
# for real spend. Basis: the implementer is Opus with write access and by far
# the most expensive turn (large context + long agentic edit loop); planner is
# a cheap read-only model; reviewer is a mid model over a capped diff; the
# tester/suite are LOCAL (no Claude call → $0). These are order-of-magnitude
# figures, deliberately conservative — the cap is a safety net, not an
# accounting tool (same caveat as DEEP_AUDIT_ESTIMATED_COST_PER_RUN_USD).
# `claude --print --output-format json` exposes a real per-call `total_cost_usd`;
# wiring that through every stage's stdout parser is a larger change, so this
# proxy gates the loop today without destabilising the tested _invoke_claude
# contract.
_STAGE_COST_ESTIMATE_USD: dict[str, float] = {
    "planner":     0.05,
    "implementer": 1.50,
    "reviewer":    0.20,
}
# Fallback when a task records an unknown/partial stage set — bill it as one
# full planner+implementer+reviewer cycle so the estimate never UNDER-counts a
# real Claude task to zero.
_FULL_TASK_COST_ESTIMATE_USD = sum(_STAGE_COST_ESTIMATE_USD.values())


def _pipeline_max_usd() -> float:
    """Per-run USD ceiling. Env override ``JARVIS_PIPELINE_MAX_USD`` (default
    25.0; clamped to >= 0). A value of 0 means "no cap" (the loop never stops on
    budget) — mirrors how a 0 daily budget would disable the DeepAudit cap."""
    try:
        v = float(os.environ.get("JARVIS_PIPELINE_MAX_USD",
                                 PIPELINE_DEFAULT_MAX_USD))
        return max(0.0, v)
    except (TypeError, ValueError):
        return PIPELINE_DEFAULT_MAX_USD


def _estimate_task_cost_usd(result: dict[str, Any] | None) -> float:
    """Coarse USD estimate for ONE completed task, from which stages actually
    ran (``result["stages"]``). A planner-only IMPOSSIBLE bail costs ~planner;
    a full approve+test cycle costs planner+implementer+reviewer. Unknown shape
    → bill a full cycle so we never under-count a real Claude task to $0."""
    if not result:
        return _FULL_TASK_COST_ESTIMATE_USD
    stages = result.get("stages")
    if not isinstance(stages, dict) or not stages:
        return _FULL_TASK_COST_ESTIMATE_USD
    total = 0.0
    for stage_name, est in _STAGE_COST_ESTIMATE_USD.items():
        if stage_name in stages:
            total += est
    # If none of the billable Claude stages were recorded (e.g. a stages dict
    # that only carries local "tester"/"suite" keys), fall back rather than
    # report $0 for what was still a real run.
    return total if total > 0 else _FULL_TASK_COST_ESTIMATE_USD


# ───────────────── refuse autonomous run with safety nets off ──────────────
# Each of these env flags SILENTLY removes a safety net. That's defensible for
# a human-driven single-task debug run, but DANGEROUS for an autonomous /
# overnight loop that edits JARVIS's own brain unattended. The loop-driver
# entry REFUSES to start when any are set on an autonomous run, unless the
# operator passes an explicit --force-unsafe ack. (env value → human label.)
_UNSAFE_SKIP_FLAGS: tuple[tuple[str, str], ...] = (
    ("STABILITY_GATE_DISABLE",     "stability gate (boot smoke test) DISABLED"),
    ("JARVIS_PIPELINE_SKIP_TESTER", "tester stage (per-task boot check) SKIPPED"),
    ("JARVIS_PIPELINE_SKIP_SUITE",  "correctness gate (unit suite) SKIPPED"),
)


def _active_unsafe_skip_flags() -> list[str]:
    """Human-readable labels for every safety-net skip flag currently set to
    "1". Empty list = all safety nets armed."""
    return [label for env_name, label in _UNSAFE_SKIP_FLAGS
            if os.environ.get(env_name, "").strip() == "1"]


def _force_unsafe_acked(argv: list[str] | None = None) -> bool:
    """True if the operator explicitly acknowledged running with safety nets
    off, via the ``--force-unsafe`` CLI flag or ``JARVIS_PIPELINE_FORCE_UNSAFE=1``.
    Mirrors how ``--force-continue-after-regression`` is recognised."""
    args = sys.argv if argv is None else argv
    return ("--force-unsafe" in args
            or os.environ.get("JARVIS_PIPELINE_FORCE_UNSAFE", "").strip() == "1")


def _run_test_suite(*, project_dir: str = PROJECT_DIR) -> dict[str, Any]:
    """CORRECTNESS GATE — run the unit suite (``tools/run_tests.py``) after a
    self-edit. The tester only proves JARVIS still *boots*; this proves the
    tested behaviour contracts (intent routing, memory, action handlers, the
    skill registry, …) are still intact, so a change that boots clean but breaks
    a contract rolls back instead of shipping.

    Returns ``{"ok": True, "skipped": True}`` (never wedges the queue) only when
    ``JARVIS_PIPELINE_SKIP_SUITE=1`` or the runner is absent."""
    if _suite_disabled():
        return {"ok": True, "skipped": True,
                "reason": "JARVIS_PIPELINE_SKIP_SUITE=1"}
    runner = os.path.join(project_dir, "tools", "run_tests.py")
    if not os.path.exists(runner):
        return {"ok": True, "skipped": True, "reason": f"missing {runner}"}
    try:
        result = subprocess.run(
            [sys.executable, runner],
            cwd=project_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "skipped": False,
                "summary": "unit suite timed out after 1200s"}
    except OSError as exc:
        return {"ok": False, "skipped": False,
                "summary": f"could not spawn unit suite: {exc!r}"}
    out = (result.stdout or "") + (result.stderr or "")
    summary = next((ln.strip() for ln in reversed(out.splitlines())
                    if "TESTS:" in ln), "") or f"rc={result.returncode}"
    return {"ok": result.returncode == 0, "skipped": False,
            "summary": summary, "stdout_tail": out[-1500:]}


def _kill_jarvis() -> int:
    """Stop any python.exe/pythonw.exe running bobert_companion.py. Returns
    the count killed. Mirrors upgrade_jarvis.kill_running_jarvis()."""
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
    except (OSError, subprocess.SubprocessError):
        return 0
    pids = [int(p.strip()) for p in result.stdout.splitlines()
            if p.strip().isdigit()]
    for pid in pids:
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Stop-Process -Id {pid} -Force"],
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    # Clear stale lock so the next test launch doesn't refuse.
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except OSError:
        pass
    return len(pids)


# ───────────────────────────── orchestration ─────────────────────────────

_GUARD_SURFACE_DIRS = ("skills", "hud", "core", "tools", "audio")


def _guard_surface(project_dir: str, planned: list[str]) -> list[str]:
    """The wider code surface to snapshot per task. The implementer runs with
    --dangerously-skip-permissions and CAN edit any file; if it touches a file
    outside plan['files_to_touch'], that edit is otherwise invisible to the
    reviewer diff AND never reverted on rollback (it silently ships). Snapshot
    every project .py (root entry points + skills/hud/core/tools/audio) so
    _compute_diff surfaces every real change and _restore_files reverts every
    one. Returns project-root-relative forward-slash paths, unioned with the
    planner's files (which may name not-yet-existing new files — those are
    recorded MISSING and deleted on rollback). (Residual: a brand-new file the
    implementer creates *outside* the plan isn't pre-enumerable, so it won't be
    auto-deleted on rollback — the common out-of-plan EDIT case is covered.)"""
    surface: set[str] = set()
    try:
        for f in os.listdir(project_dir):
            if f.endswith(".py") and os.path.isfile(os.path.join(project_dir, f)):
                surface.add(f)
    except OSError:
        pass
    for sub in _GUARD_SURFACE_DIRS:
        base = os.path.join(project_dir, sub)
        if not os.path.isdir(base):
            continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in fnames:
                if fn.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, fn), project_dir)
                    surface.add(rel.replace("\\", "/"))
    for p in planned:
        surface.add(str(p).replace("\\", "/"))
    return sorted(surface)


def _prune_pipeline_backups(keep: int = 15) -> None:
    """Bound backups/pipeline/ growth — now that each task snapshots the wider
    code surface, keep only the most recent `keep` task snapshots. Names are
    `<YYYY-MM-DD_HH-MM-SS>_<slug>`, so a plain name sort is chronological."""
    try:
        dirs = sorted(
            os.path.join(PIPELINE_BACKUP_ROOT, d)
            for d in os.listdir(PIPELINE_BACKUP_ROOT)
            if os.path.isdir(os.path.join(PIPELINE_BACKUP_ROOT, d))
        )
        for old in dirs[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def run_pipeline_on_task(task_line: str, *, claude_path: str,
                         project_dir: str = PROJECT_DIR,
                         emit=print) -> dict[str, Any]:
    """Run all 4 stages on a single task. Returns a result dict:

        {"ok": bool, "ticked": bool, "stage_failed": str|None, "stages": {...}}

    Side effects:
      - Implementer may modify files on disk.
      - On reviewer reject OR tester fail: files are restored from backup
        AND a [regression] task is appended to jarvis_todo.md AND the
        original task is NOT ticked.
      - On approve + tester pass: the original task IS ticked.
    """
    emit("  [stage 1/4] planner — scoping the work...")
    plan = _run_planner(task_line, claude_path=claude_path,
                        project_dir=project_dir)
    if "error" in plan:
        _log_pipeline_event({"stage": "planner", "ok": False,
                             "task": task_line[:200], "plan": plan})
        emit(f"  [stage 1/4] planner FAILED: {plan['error']}")
        # Planner failure: don't tick, don't append regression, just bail.
        # Next iteration will try again with a fresh planner call.
        return {"ok": False, "ticked": False, "stage_failed": "planner",
                "stages": {"planner": plan}}

    approach = (plan.get("approach", "") or "").strip()
    if approach.upper().startswith("IMPOSSIBLE:"):
        note = approach[len("IMPOSSIBLE:"):].strip() or "marked impossible by planner"
        emit(f"  [stage 1/4] planner marked task IMPOSSIBLE: {note[:120]}")
        ticked = _tick_task(task_line, f"IMPOSSIBLE (planner): {note[:140]}")
        _log_pipeline_event({"stage": "planner", "ok": True,
                             "verdict": "impossible", "task": task_line[:200],
                             "ticked": ticked})
        return {"ok": True, "ticked": ticked, "stage_failed": None,
                "stages": {"planner": plan}}

    files = plan.get("files_to_touch", [])
    emit(f"  [stage 1/4] planner OK — {len(files)} file(s) to touch, "
         f"{len(plan.get('regression_risks', []))} risk(s)")

    # Snapshot the WIDER surface, not just plan['files_to_touch'], so an
    # out-of-plan implementer edit is still diffed + reverted (see _guard_surface).
    guard_files = _guard_surface(project_dir, files)
    backup_dir = _backup_files(guard_files, task_slug=task_line[:60])
    _prune_pipeline_backups()
    emit(f"  [backup] snapshotted {len(guard_files)} file(s) "
         f"(plan named {len(files)}) → {os.path.basename(backup_dir)}")

    emit("  [stage 2/4] implementer — making code changes...")
    impl = _run_implementer(task_line, plan, claude_path=claude_path,
                            project_dir=project_dir)
    if impl["rc"] != 0:
        emit(f"  [stage 2/4] implementer FAILED rc={impl['rc']} — "
             f"rolling back")
        r, d = _restore_files(backup_dir)
        _log_pipeline_event({"stage": "implementer", "ok": False,
                             "task": task_line[:200],
                             "rc": impl["rc"], "rollback": [r, d]})
        return {"ok": False, "ticked": False, "stage_failed": "implementer",
                "stages": {"planner": plan, "implementer": impl}}

    if impl["impossible"]:
        note = impl["note"] or "marked impossible by implementer"
        emit(f"  [stage 2/4] implementer marked task IMPOSSIBLE: {note[:120]}")
        # Don't roll back — implementer may have left useful diagnostic state.
        ticked = _tick_task(task_line, f"IMPOSSIBLE (implementer): {note[:140]}")
        _log_pipeline_event({"stage": "implementer", "ok": True,
                             "verdict": "impossible", "task": task_line[:200],
                             "ticked": ticked})
        return {"ok": True, "ticked": ticked, "stage_failed": None,
                "stages": {"planner": plan, "implementer": impl}}

    emit(f"  [stage 2/4] implementer DONE — {impl['note'][:120]}")

    diff = _compute_diff(backup_dir, guard_files)
    if diff == "(no file changes)":
        if impl.get("already_done"):
            emit("  [diff] no file changes — implementer claimed "
                 "ALREADY_DONE; reviewer will verify the claim")
        else:
            emit("  [diff] implementer made no file changes — "
                 "flagging to reviewer")

    impl_already_done = bool(impl.get("already_done"))
    if impl_already_done:
        emit(f"  [stage 2/4] implementer claims ALREADY_DONE: "
             f"{impl['note'][:120]}")

    emit("  [stage 3/4] reviewer — checking diff for regressions...")
    review = _run_reviewer(task_line, plan, diff, claude_path=claude_path,
                           project_dir=project_dir,
                           impl_impossible=bool(impl.get("impossible")),
                           impl_already_done=impl_already_done,
                           impl_note=impl.get("note") or "")
    verdict = review.get("verdict", "approve_with_warnings")
    score = review.get("risk_score", 5)
    concerns = review.get("concerns", [])
    # SAFETY GATE: a high risk_score must BLOCK, not merely annotate. Before
    # this, the reviewer recorded risk_score advisorily while
    # approve_with_warnings shipped regardless — so a risk=9 change reached the
    # tester and shipped if it merely booted. Now a score AT/ABOVE the threshold
    # is escalated to a rejection + rollback. This is also what makes the
    # degraded reviewer paths (infra error / unparseable JSON, scored 8) fail
    # CLOSED instead of shipping unreviewed.
    try:
        _score_int = int(score)
    except (TypeError, ValueError):
        _score_int = 9  # unparseable score → treat as high risk (fail closed)
    if verdict in ("approve", "approve_with_warnings") \
            and _score_int >= _max_risk_score():
        concerns = list(concerns) + [
            f"risk_score {_score_int} at/above safety threshold "
            f"{_max_risk_score()} — auto-escalated to reject_and_redo"]
        emit(f"  [safety] risk_score {_score_int} >= threshold "
             f"{_max_risk_score()} — escalating {verdict} -> reject_and_redo")
        verdict = "reject_and_redo"
    emit(f"  [stage 3/4] reviewer verdict={verdict} risk={score} "
         f"concerns={len(concerns)}")
    for c in concerns[:5]:
        emit(f"            • {str(c)[:140]}")

    if verdict == "reject_and_redo":
        r, d = _restore_files(backup_dir)
        details = "; ".join(str(c) for c in concerns[:3])[:400]
        _append_regression_task(task_line, "reviewer", details)
        _log_pipeline_event({"stage": "reviewer", "ok": False,
                             "task": task_line[:200], "review": review,
                             "rollback": [r, d]})
        emit(f"  [rollback] reviewer rejected — restored {r} file(s), "
             f"deleted {d} new file(s); regression task queued")
        return {"ok": False, "ticked": False, "stage_failed": "reviewer",
                "stages": {"planner": plan, "implementer": impl,
                           "reviewer": review}}

    if verdict == "already_done":
        # No diff to test — reviewer verified the working tree already
        # satisfies the task. Tick immediately and skip the tester stage.
        note = impl.get("note") or "verified already complete"
        done_note = f"ALREADY_DONE — {note[:130]} [reviewer verified, risk={score}]"
        ticked = _tick_task(task_line, done_note)
        _log_pipeline_event({"stage": "complete", "ok": True,
                             "task": task_line[:200],
                             "verdict": "already_done", "risk_score": score,
                             "ticked": ticked,
                             "backup_dir": os.path.basename(backup_dir)})
        emit("  [stage 3/4] reviewer verified ALREADY_DONE — "
             "skipping tester, ticking task")
        return {"ok": True, "ticked": ticked, "stage_failed": None,
                "stages": {"planner": plan, "implementer": impl,
                           "reviewer": review}}

    emit("  [stage 4/4] tester — booting JARVIS for smoke test...")
    # Tester always starts from a clean slate.
    killed = _kill_jarvis()
    if killed:
        emit(f"            killed {killed} stale JARVIS process(es)")
    test_result = _run_tester(project_dir=project_dir)
    # Make sure JARVIS is shut down before returning to the loop, so the
    # next task's tester run starts clean.
    _kill_jarvis()

    if not test_result.get("ok"):
        details = (test_result.get("error")
                   or (test_result.get("stdout_tail", "") or "")[-400:]
                   or "see pipeline log")
        _append_regression_task(task_line, "tester", details)
        r, d = _restore_files(backup_dir)
        _log_pipeline_event({"stage": "tester", "ok": False,
                             "task": task_line[:200], "test": test_result,
                             "rollback": [r, d]})
        emit(f"  [rollback] tester failed — restored {r} file(s), "
             f"deleted {d} new file(s); regression task queued")
        return {"ok": False, "ticked": False, "stage_failed": "tester",
                "stages": {"planner": plan, "implementer": impl,
                           "reviewer": review, "tester": test_result}}

    if test_result.get("skipped"):
        emit(f"  [stage 4/4] tester SKIPPED ({test_result.get('reason')})")
    else:
        emit("  [stage 4/4] tester PASSED — JARVIS booted clean")

    # CORRECTNESS GATE: the tester only proves JARVIS still BOOTS. Run the unit
    # suite so a self-edit that breaks a tested contract (intent routing,
    # memory, action handlers, the skill registry) rolls back instead of
    # shipping green-on-boot. Skips cleanly (missing runner / opt-out env) so it
    # never wedges the queue.
    suite = _run_test_suite(project_dir=project_dir)
    if not suite.get("ok") and not suite.get("skipped"):
        details = (suite.get("summary") or "unit suite failed")[:400]
        _append_regression_task(task_line, "test-suite", details)
        r, d = _restore_files(backup_dir)
        _log_pipeline_event({"stage": "test-suite", "ok": False,
                             "task": task_line[:200], "suite": suite,
                             "rollback": [r, d]})
        emit(f"  [rollback] unit suite FAILED ({details}) — restored {r} "
             f"file(s), deleted {d} new file(s); regression task queued")
        return {"ok": False, "ticked": False, "stage_failed": "test-suite",
                "stages": {"planner": plan, "implementer": impl,
                           "reviewer": review, "tester": test_result,
                           "suite": suite}}
    if suite.get("skipped"):
        emit(f"  [gate] unit suite SKIPPED ({suite.get('reason')})")
    else:
        emit(f"  [gate] unit suite PASSED — {suite.get('summary')}")

    impl_note = impl.get("note") or "implemented"
    review_tag = "approve" if verdict == "approve" else "approve+warnings"
    done_note = f"{impl_note[:120]} [{review_tag}, risk={score}]"
    ticked = _tick_task(task_line, done_note)
    _log_pipeline_event({"stage": "complete", "ok": True,
                         "task": task_line[:200],
                         "verdict": verdict, "risk_score": score,
                         "ticked": ticked,
                         "backup_dir": os.path.basename(backup_dir)})
    return {"ok": True, "ticked": ticked, "stage_failed": None,
            "stages": {"planner": plan, "implementer": impl,
                       "reviewer": review, "tester": test_result,
                       "suite": suite}}


# ─────────────────────────── driver entry point ───────────────────────────
# The driver tempfile rendered by upgrade_jarvis.py imports this function
# and runs it. Keeping the loop in a real module (not inside the template
# string) makes it grep-able, testable, and trivial to extend.

def run_pipeline_loop_driver(*, project_dir: str, claude_path: str,
                             stream_log: str, max_iter: int,
                             task_count: int,
                             interactive: bool = False) -> int:
    """Loop until the queue is empty (or MAX_ITER, or NO_PROGRESS_LIMIT).
    Each iteration picks the first unchecked task and runs the 4-stage
    pipeline on it. Returns 0 on clean exit, non-zero on hard failure.

    ``interactive`` distinguishes a human-driven, attended invocation from an
    AUTONOMOUS one. The driver is the autonomous entry point: it is rendered to
    a tempfile and spawned DETACHED by upgrade_jarvis.py / the overnight engine,
    with no human watching. So it defaults to ``interactive=False`` (autonomous)
    — only a caller that KNOWS a human is present (e.g. an operator at a
    terminal) passes ``interactive=True``. On an autonomous run, having any
    safety-net skip flag set (STABILITY_GATE_DISABLE / JARVIS_PIPELINE_SKIP_TESTER
    / JARVIS_PIPELINE_SKIP_SUITE) is REFUSED unless ``--force-unsafe`` is acked,
    because an unattended loop that edits JARVIS's own brain with the gates off
    can ship a regression with nothing to catch it. A human single-task debug
    run (the ``--task`` / ``--first-unchecked`` CLI paths, which call
    run_pipeline_on_task directly and never reach here) may still use the flags
    freely."""
    # Enable ANSI colors on Windows so the stream pops.
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
        except Exception:
            pass

    CYAN, YELLOW, GREEN, RED, GRAY, RESET = (
        "\x1b[36m", "\x1b[33m", "\x1b[32m", "\x1b[31m", "\x1b[90m", "\x1b[0m"
    )
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    # ── REFUSE an autonomous run with safety nets disabled ───────────────
    # Bail BEFORE opening the stream log / touching any task so an unattended
    # loop can't churn the codebase with the gates off. A human single-task
    # debug run never reaches here (it calls run_pipeline_on_task directly), so
    # this only fires on the autonomous loop. --force-unsafe is the explicit ack.
    _unsafe = _active_unsafe_skip_flags()
    if _unsafe and not interactive and not _force_unsafe_acked():
        banner = "\n".join([
            f"{RED}{'=' * 64}{RESET}",
            f"{RED}[REFUSED] autonomous pipeline run with safety nets "
            f"DISABLED{RESET}",
            *[f"{RED}    - {label}{RESET}" for label in _unsafe],
            f"{YELLOW}An unattended loop edits JARVIS's own code; with these "
            f"gates off a{RESET}",
            f"{YELLOW}regression could ship unreviewed/untested. Refusing to "
            f"start.{RESET}",
            f"{GRAY}    Re-run a human-attended single task instead "
            f"(--task / --first-unchecked),{RESET}",
            f"{GRAY}    or pass --force-unsafe (or set "
            f"JARVIS_PIPELINE_FORCE_UNSAFE=1) to override.{RESET}",
            f"{RED}{'=' * 64}{RESET}",
        ])
        print(banner, flush=True)
        _log_pipeline_event({"stage": "refused-unsafe", "ok": False,
                             "skip_flags": _unsafe, "interactive": interactive})
        return 5

    try:
        log_fp = open(stream_log, "a", encoding="utf-8", buffering=1)
    except OSError:
        log_fp = None

    def emit(s: str) -> None:
        print(s, flush=True)
        if log_fp is not None:
            try:
                log_fp.write(ANSI_RE.sub("", s) + "\n")
                log_fp.flush()
            except OSError:
                pass

    if log_fp is not None:
        log_fp.write("\n=== multi-agent pipeline started "
                     + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")

    emit(f"{CYAN}=== JARVIS MULTI-AGENT PIPELINE ==="
         f"{RESET}")
    emit(f"{GRAY}  planner={_planner_model()}  "
         f"implementer={_implementer_model()}  "
         f"reviewer={_reviewer_model()}  "
         f"tester_wait={_tester_wait_s()}s"
         f"{'  (TESTER OFF)' if _tester_disabled() else ''}"
         f"{RESET}")
    emit(f"{GRAY}  {task_count} task(s) at start, max {max_iter} iterations"
         f"{RESET}")
    emit(f"{GRAY}  log: {stream_log}{RESET}")
    emit("")

    # Watchdog: count consecutive iterations where the pipeline failed to
    # tick *anything*. Pending-delta would falsely trip when diagnostic
    # daemons or our own regression-task generator append "- [ ]" lines
    # during the run — producers can hide real drain progress. Counting
    # actual ticks is immune to that.
    NO_PROGRESS_LIMIT = 5
    no_progress = 0
    rc_out = 0

    # ── Per-run USD budget cap ───────────────────────────────────────────
    # Tally a coarse per-task cost ESTIMATE and STOP the loop cleanly (at a
    # task boundary, like the no-progress watchdog) once it reaches the
    # ceiling. Mirrors the DeepAudit daily-budget cap. A 0 ceiling = no cap.
    _budget_cap_usd = _pipeline_max_usd()
    _spend_usd = 0.0
    if _budget_cap_usd > 0:
        emit(f"{GRAY}  budget cap: ${_budget_cap_usd:.2f}/run "
             f"(coarse estimate; stops at a task boundary){RESET}")
    else:
        emit(f"{GRAY}  budget cap: disabled (JARVIS_PIPELINE_MAX_USD=0){RESET}")

    # ── Stability gate setup ─────────────────────────────────────────────
    # Lazy import to avoid a circular dep (upgrade_jarvis -> driver template
    # -> this module). We only need the helper when the gate actually
    # fires, so deferring import is also faster on the no-gate path.
    try:
        sys.path.insert(0, project_dir)
        from upgrade_jarvis import (  # type: ignore
            _stability_gate as _run_stability_gate,
            _gate_config as _read_gate_config,
        )
    except Exception as _gate_imp_exc:
        emit(f"{YELLOW}[gate] stability-gate helper unavailable "
             f"({_gate_imp_exc!r}); gate disabled this run{RESET}")
        _run_stability_gate = None
        _read_gate_config = lambda: (10, 300, False)  # noqa: E731

    # Force-continue lets the user resume the pipeline after a gate FAIL +
    # revert. Without it, a FAIL pauses the loop so we don't pile more
    # changes on top of broken state.
    _force_continue_after_regression = (
        "--force-continue-after-regression" in sys.argv
        or os.environ.get("JARVIS_PIPELINE_FORCE_CONTINUE_AFTER_REGRESSION",
                          "").strip() == "1"
    )
    _completed_since_gate = 0
    _batch_n = 0
    _recent_titles: list[str] = []

    for i in range(1, max_iter + 1):
        pending = _count_unchecked()
        if pending == 0:
            emit(f"{GREEN}[loop] queue empty — all tasks done{RESET}")
            break

        task = _first_unchecked_task()
        if task is None:
            emit(f"{GREEN}[loop] no unchecked tasks — done{RESET}")
            break

        emit("")
        emit(f"{YELLOW}[loop] iter {i}/{max_iter}  "
             f"{pending} task(s) remain{RESET}")
        emit(f"{GRAY}        task: {task[:140]}{RESET}")

        ticked_this_iter = False
        try:
            result = run_pipeline_on_task(
                task, claude_path=claude_path,
                project_dir=project_dir, emit=emit,
            )
        except Exception as exc:
            emit(f"{RED}[loop] pipeline raised on this task: {exc!r}{RESET}")
            _log_pipeline_event({"stage": "exception", "ok": False,
                                 "task": task[:200], "exc": repr(exc)})
            result = None

        if result is None:
            pass
        elif result.get("ok") and result.get("ticked"):
            emit(f"{GREEN}[loop] task ticked clean{RESET}")
            ticked_this_iter = True
            _completed_since_gate += 1
            _recent_titles.append(task[:160])
        elif result.get("ok"):
            # Marked impossible but couldn't tick (rare — likely a todo
            # file write race).
            emit(f"{YELLOW}[loop] task resolved but not ticked — "
                 f"investigate{RESET}")
        else:
            emit(f"{RED}[loop] task FAILED at stage="
                 f"{result.get('stage_failed')}; "
                 f"rolled back, regression queued{RESET}")

        # Tally estimated spend for THIS task (a rolled-back task still burned
        # its planner/implementer/reviewer Claude calls, so it counts too).
        # result is None only when run_pipeline_on_task raised — bill it as a
        # full task so a crashing loop still draws down the budget.
        _spend_usd += _estimate_task_cost_usd(result)
        if _budget_cap_usd > 0:
            emit(f"{GRAY}[budget] est. spend ${_spend_usd:.2f} / "
                 f"${_budget_cap_usd:.2f} cap{RESET}")

        if ticked_this_iter:
            no_progress = 0
        else:
            no_progress += 1
            # round3-followup-1: exponential backoff with jitter on
            # consecutive failures, so a rate-limited or transient API
            # error doesn't burn the no-progress watchdog in seconds.
            # cap at 120s so we don't sleep forever. import inside the
            # branch so a missing `random` doesn't break the hot path.
            import random
            _backoff_s = min(2 ** min(no_progress, 6) + random.uniform(0, 1),
                             120.0)
            emit(f"{GRAY}[loop] consecutive failure #{no_progress} — "
                 f"backing off {_backoff_s:.1f}s before next iter{RESET}")
            time.sleep(_backoff_s)
            if no_progress >= NO_PROGRESS_LIMIT:
                emit(f"{RED}[loop] no task ticked for "
                     f"{NO_PROGRESS_LIMIT} iterations — stopping{RESET}")
                rc_out = 2
                break

        # ── Stability gate: every STABILITY_GATE_INTERVAL successful task
        # completions the pipeline pauses, snapshots, boots JARVIS for a
        # 5-minute smoke test, and verdicts on process-alive / no-APPCRASH /
        # no-FATAL. A FAIL auto-reverts via robocopy and pauses the loop
        # (unless the operator passed --force-continue-after-regression).
        # _read_gate_config() is re-called every iteration so an operator
        # can re-tune STABILITY_GATE_INTERVAL via env var mid-run.
        _gate_interval_now = _read_gate_config()[0]
        if (_run_stability_gate is not None
                and _completed_since_gate >= max(1, _gate_interval_now)):
            _batch_n += 1
            emit("")
            emit(f"{CYAN}[gate] hitting stability gate after "
                 f"{_completed_since_gate} completions (batch {_batch_n}){RESET}")
            try:
                _gate_result = _run_stability_gate(
                    batch_n=_batch_n,
                    recent_task_titles=list(_recent_titles),
                    batch_size=_completed_since_gate,
                )
            except Exception as _gate_exc:
                emit(f"{RED}[gate] gate raised: {_gate_exc!r} — "
                     f"continuing without verdict{RESET}")
                _gate_result = {"ok": True, "verdict": "ERROR",
                                "details": repr(_gate_exc)}

            _completed_since_gate = 0
            _recent_titles = []
            if _gate_result.get("verdict") == "PASS":
                emit(f"{GREEN}[gate] PASS — resuming pipeline{RESET}")
            elif _gate_result.get("verdict") == "SKIP":
                emit(f"{GRAY}[gate] {_gate_result.get('details','')}{RESET}")
            elif _gate_result.get("verdict") == "FAIL":
                emit(f"{RED}[gate] FAIL — auto-reverted; "
                     f"regression task queued{RESET}")
                if _force_continue_after_regression:
                    emit(f"{YELLOW}[gate] --force-continue-after-regression "
                         f"set; resuming despite FAIL{RESET}")
                else:
                    emit(f"{RED}[gate] pausing pipeline; rerun with "
                         f"--force-continue-after-regression to override"
                         f"{RESET}")
                    rc_out = 3
                    break

        # ── Budget cap: stop CLEANLY at this task boundary once estimated
        # spend reaches the ceiling. Checked LAST in the iteration body so the
        # current task fully finished (applied + reviewed + tested or rolled
        # back) before we stop — we never sever an in-flight apply. Mirrors the
        # no-progress watchdog's stop-and-break shape.
        if _budget_cap_usd > 0 and _spend_usd >= _budget_cap_usd:
            emit(f"{RED}[budget] estimated spend ${_spend_usd:.2f} reached "
                 f"the ${_budget_cap_usd:.2f}/run cap — stopping at task "
                 f"boundary{RESET}")
            _log_pipeline_event({"stage": "budget-cap", "ok": True,
                                 "spend_usd": round(_spend_usd, 4),
                                 "cap_usd": _budget_cap_usd,
                                 "iterations": i})
            rc_out = 4
            break

    # Final py_compile sweep, matching the single-stage driver.
    emit("")
    emit(f"{YELLOW}Verifying syntax of bobert_companion.py...{RESET}")
    try:
        rc = subprocess.run(
            [sys.executable, "-m", "py_compile", "bobert_companion.py"],
            cwd=project_dir, timeout=60,
        ).returncode
        if rc == 0:
            emit(f"{GREEN}bobert_companion.py: OK{RESET}")
        else:
            emit(f"{RED}SYNTAX ERROR in bobert_companion.py — check above{RESET}")
    except (OSError, subprocess.SubprocessError) as exc:
        emit(f"{RED}py_compile failed to run: {exc!r}{RESET}")

    if _budget_cap_usd > 0:
        emit(f"{GRAY}Estimated spend this run: ${_spend_usd:.2f} "
             f"(cap ${_budget_cap_usd:.2f}){RESET}")
    emit(f"{GREEN}Pipeline complete.{RESET}")

    if log_fp is not None:
        try:
            log_fp.close()
        except OSError:
            pass
    return rc_out


# ─────────────────────────────── CLI entry ───────────────────────────────
# Allows manual single-task invocation for testing:
#     python tools/multi_agent_pipeline.py --task "the task line"
#     python tools/multi_agent_pipeline.py --first-unchecked
# Useful for the "Verify by running it on one fake task" workflow.

def _find_claude_cli() -> str | None:
    """Mirror of upgrade_jarvis.find_claude_cli() so this module is
    standalone-runnable without importing the orchestrator."""
    found = shutil.which("claude")
    if found:
        return found
    for path in [
        os.path.expanduser(r"~\.local\bin\claude.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude-code\claude.exe"),
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
        os.path.expanduser(r"~\AppData\Roaming\npm\claude.cmd"),
        os.path.expandvars(r"%USERPROFILE%\.npm-global\claude.cmd"),
    ]:
        if os.path.exists(path):
            return path
    return None


def _cli(argv: list[str]) -> int:
    import argparse
    # Plain-ASCII description so argparse's help printer doesn't crash on
    # Windows consoles using cp1252 (the module docstring has Unicode arrows).
    ap = argparse.ArgumentParser(
        description=(
            "JARVIS multi-agent upgrade pipeline. "
            "Stages: planner -> implementer -> reviewer -> tester."
        ),
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=str,
                   help="Run the pipeline on this exact task string.")
    g.add_argument("--first-unchecked", action="store_true",
                   help="Pick the first '- [ ]' task from jarvis_todo.md.")
    g.add_argument("--loop", action="store_true",
                   help="Run the full driver loop (drain the queue).")
    ap.add_argument("--max-iter", type=int, default=50,
                    help="Iteration cap for --loop (default 50).")
    args = ap.parse_args(argv)

    claude_path = _find_claude_cli()
    if not claude_path:
        print("ERROR: claude CLI not found.", file=sys.stderr)
        return 127

    if args.loop:
        rc = run_pipeline_loop_driver(
            project_dir=PROJECT_DIR, claude_path=claude_path,
            stream_log=os.path.join(PROJECT_DIR, "upgrade_stream.log"),
            max_iter=args.max_iter,
            task_count=_count_unchecked(),
        )
        # Queue drained — bring JARVIS back up SILENT in ambient-learning
        # standby (listens + learns, speaks only after the 'JARVIS' wake
        # word). Only the standalone --loop entry relaunches; when this loop
        # is driven by upgrade_jarvis.py the orchestrator owns the relaunch,
        # so we must NOT relaunch here too (would race the singleton lock).
        try:
            sys.path.insert(0, PROJECT_DIR)
            from upgrade_jarvis import relaunch_jarvis  # type: ignore
            relaunch_jarvis()
            print("[pipeline] JARVIS relaunched in ambient-learning standby "
                  "(say 'JARVIS' to wake).")
        except Exception as _relaunch_exc:
            print(f"[pipeline] ambient relaunch skipped ({_relaunch_exc!r}); "
                  f"start manually: python bobert_companion.py")
        return rc

    if args.first_unchecked:
        task = _first_unchecked_task()
        if not task:
            print("QUEUE EMPTY")
            return 0
    else:
        task = args.task

    result = run_pipeline_on_task(task, claude_path=claude_path)
    print(json.dumps({
        "ok":            result.get("ok"),
        "ticked":        result.get("ticked"),
        "stage_failed":  result.get("stage_failed"),
    }, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via _cli() in tests
    sys.exit(_cli(sys.argv[1:]))
