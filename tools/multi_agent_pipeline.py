#!/usr/bin/env python3
"""
JARVIS self-upgrade STAGE LIBRARY (planner / implementer / reviewer).

The orchestration lives in tools/selfup_runner.py: it creates an isolated git
worktree off origin/main, runs these stages inside it task by task, commits
each passing task, reverts each failing one with git, runs the real test gate
at the end and leaves the branch for the owner. This module only holds the
three SCOPED Claude invocations and their helpers:

    1. PLANNER     - cheap model, READ-ONLY (Read, Glob, Grep). Outputs JSON:
                       files_to_touch, regression_risks, tests_to_run, approach.
    2. IMPLEMENTER - Opus, may edit files INSIDE THE WORKTREE only. It runs
                       with an explicit allow-list and a deny-list (see
                       implementer_permission_args): no git push / tag / merge /
                       reset / checkout / commit, no writes under data/, no
                       network tools, and NEVER --dangerously-skip-permissions.
    3. REVIEWER    - Sonnet, READ-ONLY. Sees the git diff + the planner's risks.
                       Outputs JSON: risk_score 0-10, concerns[], verdict
                       (approve | reject_and_redo | approve_with_warnings |
                       already_done).

The CLI flag names used for the lock-down (--tools, --allowedTools,
--disallowedTools, --permission-mode dontAsk) are the ones `claude --help`
prints (Claude Code 2.1.204, checked 2026-10-15).

Configuration (env vars, all optional):
    JARVIS_PIPELINE_PLANNER_MODEL       default "haiku"
    JARVIS_PIPELINE_IMPLEMENTER_MODEL   default "opus"
    JARVIS_PIPELINE_REVIEWER_MODEL      default "sonnet"
    JARVIS_PIPELINE_MAX_RISK            default 7   (risk_score >= this blocks)
    JARVIS_PIPELINE_MAX_USD             default 25.0 (per-RUN dollar ceiling;
                                          the runner stops at a task boundary
                                          once estimated spend reaches it)

Per-run budget cap (mirrors core/diagnostic_daemons.py's DeepAudit daily cap):
  The runner tallies a COARSE per-stage cost ESTIMATE (_STAGE_COST_ESTIMATE_USD)
  as each task runs and stops cleanly once the running total reaches
  JARVIS_PIPELINE_MAX_USD. A recent uncapped run cost $413.90 over 579
  iterations; this ceiling bounds the blast radius.
"""
from __future__ import annotations

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
# UnicodeEncodeError mid-cycle (see upgrade_jarvis.py - 2026-05-31 gate crash).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - import-time guard for consoles lacking reconfigure
        pass


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_LOG = os.path.join(PROJECT_DIR, "data", "pipeline_runs.jsonl")

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
    # Claude Code 2.1.x words a subscription limit "You've hit your session
    # limit · resets 9pm" (also weekly / Opus / Sonnet / usage) - none of the
    # needles above matched it (2026-10-14).
    "hit your session limit",
    "hit your weekly limit",
    "hit your opus limit",
    "hit your sonnet limit",
    "hit your usage limit",
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


def _claude_accounts():
    """core.claude_accounts (Pro-first profile choice), or None when it can't
    be imported - then the call runs as before, on the default profile."""
    try:
        from core import claude_accounts
        return claude_accounts
    except Exception:
        return None


# ───────────────────────── permission lock-down ─────────────────────────
# Flag names verified against `claude --help` (2.1.204, 2026-10-15):
#   --tools <list>             built-in tools that exist at all
#   --allowedTools <list>      pre-approved tools / patterns, e.g. "Bash(git *)"
#   --disallowedTools <list>   denied tools / patterns (deny beats allow)
#   --permission-mode dontAsk  anything not pre-approved is refused, never asked
# There is deliberately NO --dangerously-skip-permissions anywhere.

READ_ONLY_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep")
# Tools no stage ever gets: nothing mutating beyond the implementer's own
# Edit/Write, nothing that reaches the network.
_NEVER_TOOLS: tuple[str, ...] = ("WebFetch", "WebSearch", "NotebookEdit")
IMPLEMENTER_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Edit", "Write", "Bash")
IMPLEMENTER_ALLOW: tuple[str, ...] = (
    "Read", "Glob", "Grep", "Edit", "Write",
    "Bash(python -m py_compile *)", "Bash(python -m pyflakes *)",
    "Bash(git diff *)", "Bash(git status *)", "Bash(git log *)",
)
IMPLEMENTER_DENY: tuple[str, ...] = _NEVER_TOOLS + (
    # git: the runner owns every commit; nothing may publish, move or rewrite
    "Bash(git push *)", "Bash(git tag *)", "Bash(git merge *)",
    "Bash(git reset *)", "Bash(git checkout *)", "Bash(git switch *)",
    "Bash(git commit *)", "Bash(git rebase *)", "Bash(git branch *)",
    "Bash(git remote *)", "Bash(git stash *)", "Bash(git clean *)",
    "Bash(git restore *)", "Bash(git pull *)", "Bash(git fetch *)",
    "Bash(git cherry-pick *)", "Bash(git worktree *)", "Bash(git config *)",
    "Bash(gh *)",
    # data/ is the owner's live state; VERSION belongs to a release session
    "Edit(data/**)", "Write(data/**)", "Edit(VERSION)", "Write(VERSION)",
    "Edit(.git/**)", "Write(.git/**)", "Edit(.claude/**)", "Write(.claude/**)",
    "Read(.env)", "Read(.env.*)",
)
# Credentials a child could otherwise use to publish.
_CHILD_ENV_DROP: tuple[str, ...] = ("ANTHROPIC_API_KEY", "GH_TOKEN",
                                    "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN")


def read_only_permission_args() -> list[str]:
    """Planner / reviewer: Read, Glob and Grep exist, nothing else does."""
    return ["--tools", ",".join(READ_ONLY_TOOLS),
            "--allowedTools", ",".join(READ_ONLY_TOOLS),
            "--disallowedTools", ",".join(
                ("Edit", "Write", "Bash", "PowerShell") + _NEVER_TOOLS),
            "--permission-mode", "dontAsk"]


def implementer_permission_args() -> list[str]:
    """Implementer: edits inside its cwd (the worktree) and a short Bash
    allow-list; the deny-list closes git publishing, data/ and VERSION."""
    return ["--tools", ",".join(IMPLEMENTER_TOOLS),
            "--allowedTools", ",".join(IMPLEMENTER_ALLOW),
            "--disallowedTools", ",".join(IMPLEMENTER_DENY),
            "--permission-mode", "dontAsk"]


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
    allow_writes=True (implementer_permission_args); the planner/reviewer
    stages get read_only_permission_args. ``project_dir`` is the cwd and must
    be the run's worktree, never the live checkout.

    Stdout is the FINAL textual response. We use plain --print (no
    --output-format stream-json) here because the orchestrator wants the
    last reply, not a tool-use trace."""
    cmd = [claude_path, "--print"]
    if model:
        cmd += ["--model", model]
    cmd += (implementer_permission_args() if allow_writes
            else read_only_permission_args())
    if extra_args:
        cmd += extra_args
    cmd.append("--")
    cmd.append(prompt)

    env = os.environ.copy()
    # Bill to a Claude subscription, not API credits. core/claude_accounts
    # also picks the profile: the owner's Pro account first while it lasts,
    # Max whenever Pro is at its 5-hour / weekly limit (2026-10-14).
    for _name in _CHILD_ENV_DROP:
        env.pop(_name, None)
    acct = _claude_accounts()

    try:
        run_kw = dict(cwd=project_dir, capture_output=True, text=True,
                      encoding="utf-8", errors="replace", timeout=timeout_s)
        if acct is not None:
            result = acct.run(cmd, env=env, runner=subprocess.run,
                              log=lambda m: print(f"  {m}", flush=True), **run_kw)
        else:
            result = subprocess.run(cmd, env=env, **run_kw)
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
Never run git push / tag / merge, never touch data/, never read or print
secrets (.env files, tokens, keys).

The task (verbatim from the approved queue):
─────────────────────────────────────
{task_line}
─────────────────────────────────────

Project root (an isolated git worktree): {project_dir}
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
    skip implementation and record the task as skipped.
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
        return {"error": f"planner exited rc={rc}", "rc": rc,
                "stderr": err[:500]}
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
upgrade pipeline. You work inside an ISOLATED git worktree of the JARVIS repo
({project_dir}) on a throwaway branch. You can read and edit files in it, and
run py_compile / pyflakes / git diff / git status.

HARD RULES (the permission system enforces them; breaking one fails the task):
  - NO git push, tag, merge, reset, checkout, commit, branch or stash. The
    orchestrator makes the one commit per task itself.
  - NO edits to anything under data/, to VERSION, to .git/ or to .claude/.
  - NO secrets: never read, print or write tokens, keys or .env files.
  - Stay inside {project_dir}; never touch any other directory.

The task (verbatim from the approved queue):
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

Do all work end-to-end:
  1. Read whatever you need.
  2. Make the code changes.
  3. Run `python -m py_compile` on every file you touched. Fix any errors.
  4. Do NOT edit jarvis_todo.md - the owner ticks the queue after review.
  5. Do NOT spawn JARVIS, run boot scripts, or shell out to long-running
     subprocesses; the full test suite runs once at the end of the run.
  6. When done, print a one-line summary starting with "DONE:" so the
     orchestrator can capture your note.

If the task turns out to be hardware-dependent or impossible, print a single
line starting with "IMPOSSIBLE:" explaining why. The pipeline records the task
as skipped (nothing is committed).

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
Never run git push / tag / merge, never touch data/, never read or print
secrets (.env files, tokens, keys).

The task being implemented (from the approved queue):
─────────────────────────────────────
{task_line}
─────────────────────────────────────

The PLANNER flagged these regression risks:
{risks_block}

Implementer's claim: {impl_claim}

Diff produced by the IMPLEMENTER (git diff against the branch HEAD):
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


def _coerce_risk(value, default: int = 9) -> int:
    """``int(value)`` for a reviewer risk_score, or ``default`` when it is not
    a plain number: a word, a list, or the NaN / Infinity that json.loads
    accepts. The default 9 fails CLOSED, so an unparseable score is treated
    as high risk instead of crashing the pipeline (audit A77)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


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
            "rc": rc,
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
            "rc": rc,
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
        review["risk_score"] = max(_coerce_risk(review.get("risk_score") or 0), 9)
        review["concerns"] = concerns
        review["_local_override"] = True
    return review

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
