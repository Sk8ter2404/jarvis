#!/usr/bin/env python3
"""Land a self-upgrade pipeline run on GitHub as a REVIEWABLE pull request.

The overnight self-upgrade pipeline (`upgrade_jarvis.py`) edits the LOCAL working
tree but never touches git: no commit, no push, no version bump beyond the
internal `data/version.json` counter. That makes every overnight run an
untracked, un-pushed, un-PII-screened island that silently diverges from GitHub.

This tool closes that gap by landing a run the SAME way a human change does:

    git checkout -b auto/overnight-<stamp>      # isolate the run's work
    git add -A && git commit                    # the pre-commit PII guard runs HERE
    git push -u origin auto/overnight-<stamp>
    -> open a pull request the owner reviews    # never auto-merged

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


def commit_to_branch(branch: str, message: str, runner: Runner) -> Tuple[bool, str]:
    """Create `branch`, stage everything, and commit. The pre-commit PII guard
    runs at the commit step — a block (or any git error) returns (False, detail)
    without raising, so a tainted autonomous run can never reach `push`."""
    try:
        c = _git(["checkout", "-b", branch], runner)
        if c.returncode != 0:
            return False, "branch create failed: " + (c.stderr.strip() or "?")
        a = _git(["add", "-A"], runner)
        if a.returncode != 0:
            return False, "stage failed: " + (a.stderr.strip() or "?")
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


def _default_poster(branch: str, title: str, body: str) -> Optional[str]:  # pragma: no cover - real network
    tok = _token()
    if not tok:
        return None
    payload = json.dumps({"title": title, "head": branch, "base": _BASE,
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
