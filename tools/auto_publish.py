#!/usr/bin/env python3
"""Land a self-upgrade pipeline run on GitHub as a REVIEWABLE pull request.

The overnight self-upgrade pipeline (`upgrade_jarvis.py`) edits the LOCAL working
tree but never touches git: no commit, no push, no version bump beyond the
internal `data/version.json` counter. That makes every overnight run an
untracked, un-pushed, un-PII-screened island that silently diverges from GitHub.

This tool closes that gap by landing a run the SAME way a human change does:

    git checkout -b auto/overnight-<stamp>      # isolate the run's work
    git add (scoped) && git commit              # the pre-commit PII guard runs HERE
    git push -u origin auto/overnight-<stamp>
    -> open a pull request the owner reviews    # never auto-merged

Staging is SCOPED, never `git add -A`: only tracked-file edits (`git add -u`) plus
new files that git itself does not ignore are staged, and a post-stage guard
refuses to commit if anything gitignored, a log/transcript, or an embedded git
repo (gitlink) slipped into the index. So owner transcripts (logs/), runtime
data (data/), and nested repo clones can never reach a PUBLIC PR.

So autonomous work becomes tracked, PII-guarded, and visible on GitHub, while a
human still approves what actually merges. The release VERSION bump stays a
review/merge-time decision (a human owns the version line).

    python tools/auto_publish.py --summary "fixed X, added Y"   # branch+commit+push+PR
    python tools/auto_publish.py --no-pr                        # stop before the PR

Total + injectable: every git/GitHub call goes through a runner/poster, so the
orchestration is unit-tested without real git or network.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OWNER = os.environ.get("JARVIS_GITHUB_OWNER", "Sk8ter2404")
_REPO = os.environ.get("JARVIS_GITHUB_REPO", "jarvis")
_BASE = os.environ.get("JARVIS_GITHUB_BASE", "main")

Runner = Callable[..., "subprocess.CompletedProcess"]
Poster = Callable[[str, str, str], Optional[str]]


def _git(args: List[str], runner: Runner) -> "subprocess.CompletedProcess":
    return runner(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                  timeout=300)


def working_changes(runner: Runner) -> List[str]:
    """Paths changed in the working tree (the pipeline's output). Empty list if
    the tree is clean or git errored — never raises."""
    try:
        r = _git(["status", "--porcelain"], runner)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]


def make_branch_name(stamp: str) -> str:
    """A run's isolation branch. `stamp` is injected (no wall-clock here) so the
    name is deterministic + testable."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in stamp)
    return f"auto/overnight-{safe}"


# Belt-and-braces guard list. `.gitignore` (consulted via `git check-ignore`) is
# the PRIMARY source of truth — these patterns are a hard backstop for the
# never-publish classes even if a path is not (yet) gitignored: owner
# transcripts/logs, runtime data, and the embedded `.git` of a nested clone.
_NEVER_STAGE_DIRS = ("logs/", "logs_staging/", "data/", "data_staging/",
                     "memory/", "backups/", "_backups/")
_NEVER_STAGE_SUFFIXES = (".log",)
# substrings that flag a transcript or a nested-repo / embedded-repo path
_NEVER_STAGE_SUBSTRINGS = ("transcript", "/.git/", ".git/")
# Owner-authored prose/scratch that none of the classes above catch (it lives at
# the repo root and ends in .md/.txt/.ps1) yet must never reach a public PR.
# Matched on the BASENAME so a "mine" substring elsewhere can't false-trip a
# legitimate source file. .gitignore is the primary guard; this is the hard
# backstop for a tree that has not (yet) ignored them. See REVIEW_FINDINGS_2 P0-4.
_NEVER_STAGE_BASENAMES = ("voice_command_tests.md",)
_NEVER_STAGE_BASENAME_PREFIXES = ("_mine_",)


def _norm(path: str) -> str:
    """Forward-slash, unquoted form of a git path (git may quote/backslash)."""
    p = path.strip().strip('"')
    return p.replace("\\", "/")


def _is_never_stage(path: str) -> bool:
    """True if `path` is in a class that must NEVER reach a public PR
    (transcript/log, runtime data, or an embedded/nested git repo). This is the
    backstop applied IN ADDITION to git's own .gitignore."""
    p = _norm(path)
    low = p.lower()
    if low.endswith(_NEVER_STAGE_SUFFIXES):
        return True
    if any(seg in low for seg in _NEVER_STAGE_SUBSTRINGS):
        return True
    base = low.rsplit("/", 1)[-1]
    if base in _NEVER_STAGE_BASENAMES or base.startswith(_NEVER_STAGE_BASENAME_PREFIXES):
        return True
    return any(p == d.rstrip("/") or p.startswith(d) for d in _NEVER_STAGE_DIRS)


def _git_ignores(path: str, runner: Runner) -> bool:
    """True if the repo's .gitignore (the real source of truth) ignores `path`.
    `git check-ignore` exits 0 and echoes the path when it is ignored. Any git
    error is treated as "ignored" (fail safe — refuse rather than leak)."""
    try:
        r = _git(["check-ignore", "-q", "--", path], runner)
    except Exception:
        return True
    return r.returncode == 0


def _untracked_to_add(runner: Runner) -> List[str]:
    """New (untracked) paths that are SAFE to stage: `git status --porcelain`
    already omits gitignored files, so this is the not-ignored new work. We then
    drop the never-stage classes and any untracked *directory* (a trailing-slash
    porcelain entry — an embedded repo shows up this way and `git add`-ing it
    would create a gitlink). Returns [] on any git error."""
    try:
        r = _git(["status", "--porcelain", "--untracked-files=all"], runner)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out: List[str] = []
    for ln in r.stdout.splitlines():
        if not ln.strip() or ln[:2] != "??":
            continue
        p = _norm(ln[3:])             # path after the "?? " status prefix
        if p.endswith("/"):           # untracked dir (incl. embedded repos) — skip
            continue
        if _is_never_stage(p):
            continue
        out.append(p)
    return out


def _scoped_stage(runner: Runner) -> Tuple[bool, str]:
    """Stage ONLY intended source changes, never `git add -A`.

    1. `git add -u` stages edits/deletions to already-TRACKED files (never adds
       a new untracked path, so logs/transcripts/embedded clones can't enter).
    2. New, not-gitignored files are added explicitly, minus the never-stage
       classes and untracked dirs (embedded-repo gitlinks).
    3. A post-stage guard re-reads the index and ABORTS if anything gitignored,
       a log/transcript, or an embedded `.git` path slipped in.

    Returns (True, "") on a clean staged set, else (False, reason)."""
    u = _git(["add", "-u"], runner)
    if u.returncode != 0:
        return False, "stage (tracked) failed: " + (u.stderr.strip() or "?")
    for path in _untracked_to_add(runner):
        # plain `git add --` honours .gitignore; -f is deliberately NOT used.
        a = _git(["add", "--", path], runner)
        if a.returncode != 0:
            return False, f"stage failed for {path}: " + (a.stderr.strip() or "?")
    # Guard: inspect what is actually staged and refuse never-publish paths.
    g = _git(["diff", "--cached", "--name-only"], runner)
    if g.returncode != 0:
        return False, "could not inspect staged set: " + (g.stderr.strip() or "?")
    staged = [ln for ln in g.stdout.splitlines() if ln.strip()]
    bad = [s for s in staged if _is_never_stage(s) or _git_ignores(s, runner)]
    if bad:
        return False, ("refusing to commit — disallowed paths staged "
                       "(logs/transcripts/data/ignored/embedded-repo): "
                       + ", ".join(_norm(b) for b in bad[:8]))
    if not staged:
        return False, "nothing to stage after scoping"
    return True, ""


def commit_to_branch(branch: str, message: str, runner: Runner) -> Tuple[bool, str]:
    """Create `branch`, stage only the intended source changes (SCOPED — never
    `git add -A`), and commit. Staging refuses owner transcripts/logs, runtime
    data, gitignored paths, and embedded-repo gitlinks; the pre-commit PII guard
    then runs at the commit step. A staging block, a guard block, or any git
    error returns (False, detail) without raising, so a tainted autonomous run
    can never reach `push`."""
    try:
        c = _git(["checkout", "-b", branch], runner)
        if c.returncode != 0:
            return False, "branch create failed: " + (c.stderr.strip() or "?")
        staged_ok, why = _scoped_stage(runner)
        if not staged_ok:
            return False, why
        m = _git(["commit", "-m", message], runner)
        if m.returncode != 0:
            detail = (m.stdout.strip() + " " + m.stderr.strip()).strip() or "?"
            return False, "commit blocked/failed: " + detail
        return True, m.stdout.strip() or "committed"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"commit error: {e}"


def push_branch(branch: str, runner: Runner) -> bool:
    try:
        r = _git(["push", "-u", "origin", branch], runner)
        return r.returncode == 0
    except Exception:  # pragma: no cover - defensive
        return False


def _token() -> Optional[str]:
    for key in ("JARVIS_GITHUB_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v
    return None


def _pr_head(branch: str) -> str:
    """The PR 'head' ref. A contributor running their OWN JARVIS instance pushes
    to their FORK and sets JARVIS_GITHUB_HEAD_OWNER to their GitHub username, so
    the PR opens cross-fork (`youruser:branch`) against the upstream repo
    (JARVIS_GITHUB_OWNER/REPO). Unset → same-repo (`branch`), the owner's case."""
    owner = os.environ.get("JARVIS_GITHUB_HEAD_OWNER", "").strip()
    return f"{owner}:{branch}" if owner else branch


def _default_poster(branch: str, title: str, body: str) -> Optional[str]:  # pragma: no cover - real network
    tok = _token()
    if not tok:
        return None
    payload = json.dumps({"title": title, "head": _pr_head(branch), "base": _BASE,
                          "body": body}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{_OWNER}/{_REPO}/pulls",
        data=payload, method="POST",
        headers={"Authorization": f"Bearer {tok}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json; charset=utf-8",
                 "User-Agent": "jarvis-auto-publish"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("html_url")
    except (urllib.error.URLError, ValueError, OSError):
        return None


def run(argv: Optional[List[str]] = None, *, runner: Optional[Runner] = None,
        poster: Optional[Poster] = None, stamp: str = "manual",
        out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="auto_publish",
                                 description="Land a self-upgrade run as a reviewable PR.")
    ap.add_argument("--summary", default="autonomous self-upgrade run",
                    help="one-line description of what the run changed")
    ap.add_argument("--stamp", default=stamp,
                    help="branch suffix (default: injected stamp)")
    ap.add_argument("--no-pr", action="store_true",
                    help="commit + push only; don't open a pull request")
    args = ap.parse_args(argv)
    runner = runner or subprocess.run

    changes = working_changes(runner)
    if not changes:
        out("No working-tree changes to publish.")
        return 0
    out(f"{len(changes)} changed file(s) from the run.")

    branch = make_branch_name(args.stamp)
    title = f"Overnight self-upgrade: {args.summary}"
    body = (f"Autonomous self-upgrade run.\n\nSummary: {args.summary}\n\n"
            "Opened by tools/auto_publish.py. Review before merge — the release "
            "VERSION bump is a merge-time decision (a human owns the version line).")
    ok, detail = commit_to_branch(branch, title + "\n\n" + body, runner)
    if not ok:
        out(f"Aborted (nothing pushed): {detail}")
        return 1
    out(f"Committed run to {branch}.")

    if not push_branch(branch, runner):
        out(f"Push failed — work is committed locally on {branch}; push manually.")
        return 1
    out(f"Pushed {branch}.")

    if args.no_pr:
        out(f"Skipped PR (--no-pr). Open one for {branch} when ready.")
        return 0
    pr_poster = poster if poster is not None else _default_poster
    url = pr_poster(branch, title, body)
    if url:
        out(f"Pull request opened: {url}")
    else:
        out("Couldn't open the PR automatically (no token / API error). "
            f"Open one for {branch} manually.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(run(sys.argv[1:]))
