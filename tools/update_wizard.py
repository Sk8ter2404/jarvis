#!/usr/bin/env python3
"""Update wizard — check for a newer JARVIS release and apply it safely.

    python tools/update_wizard.py              # interactive: check -> confirm -> pull -> verify
    python tools/update_wizard.py --check      # just report whether an update exists
    python tools/update_wizard.py --yes        # non-interactive (assume yes)
    python tools/update_wizard.py --skip-verify   # don't re-run the test gate after pulling

Flow:
  1. Ask core.update_checker whether a newer GitHub release exists.
  2. If so, show current -> latest + the release URL.
  3. Refuse to touch a DIRTY working tree — your gitignored personal layer
     (skills/vip_*, data/, .env, creds) is fine, but uncommitted *tracked*
     changes abort the update so nothing of yours is clobbered.
  4. git fetch + FAST-FORWARD main only (never a merge/rebase that could
     conflict or rewrite a diverged local).
  5. Re-run the test gate to confirm the new build is healthy; on failure it
     prints the exact one-line rollback command.
  6. Tell you to restart JARVIS to load it.

Safety: every external command goes through an injectable runner; nothing
destructive happens without confirmation; ff-only guarantees a diverged local
can never be silently rewritten. The functions are total (never raise) so the
wizard always exits with a clean status + message.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Callable, Optional, Tuple

from core import update_checker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, runner):
    """Run a command from the repo root via the injected runner (subprocess.run
    by default). Returns the CompletedProcess-like result."""
    return runner(cmd, cwd=_ROOT, capture_output=True, text=True, timeout=600)


def current_commit(runner) -> str:
    """Short HEAD sha, or '?' if git can't be queried. Never raises."""
    try:
        r = _run(["git", "rev-parse", "--short", "HEAD"], runner)
        return r.stdout.strip() if r.returncode == 0 else "?"
    except Exception:
        return "?"


def tracked_tree_dirty(runner) -> bool:
    """True if there are uncommitted TRACKED changes (untracked + gitignored
    files are deliberately ignored, so the personal layer never blocks an
    update). Conservative: returns False if git can't be queried."""
    try:
        r = _run(["git", "status", "--porcelain", "--untracked-files=no"], runner)
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False


def apply_update(runner) -> Tuple[bool, str]:
    """git fetch origin main, then fast-forward. Returns (ok, detail). Never
    raises. A diverged local fails the ff cleanly rather than being rewritten."""
    try:
        f = _run(["git", "fetch", "origin", "main"], runner)
        if f.returncode != 0:
            return False, "git fetch failed: " + (f.stderr.strip() or "unknown")
        m = _run(["git", "merge", "--ff-only", "origin/main"], runner)
        if m.returncode != 0:
            return False, ("fast-forward failed — your local main has diverged: "
                           + (m.stderr.strip() or m.stdout.strip() or "unknown"))
        return True, (m.stdout.strip() or "fast-forwarded")
    except Exception as e:
        return False, f"update error: {e}"


def verify(runner) -> Tuple[bool, str]:
    """Run the light test gate to confirm the freshly-pulled build is healthy.
    Returns (ok, detail). Never raises."""
    try:
        r = _run([sys.executable, "tools/run_tests.py"], runner)
        if r.returncode == 0:
            return True, "tests passed"
        body = (r.stdout or r.stderr or "").strip().splitlines()
        tail = " | ".join(body[-3:]) if body else "see output"
        return False, "tests FAILED: " + tail
    except Exception as e:
        return False, f"verify error: {e}"


def main(argv: Optional[list] = None, *, runner: Optional[Callable] = None,
         input_fn: Callable = input, out: Callable = print) -> int:
    parser = argparse.ArgumentParser(prog="update_wizard",
                                     description="Check for and apply a JARVIS update.")
    parser.add_argument("--check", action="store_true",
                        help="report whether an update exists, then stop")
    parser.add_argument("--yes", action="store_true",
                        help="apply without an interactive confirmation")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the post-update test run")
    args = parser.parse_args(argv)
    runner = runner or subprocess.run

    res = update_checker.check_for_update()
    if not res.get("checked"):
        out(f"Couldn't check for updates: {res.get('detail')}")
        out("(Private repo? Set JARVIS_GITHUB_TOKEN with read-only Contents access.)")
        return 2
    if not res.get("update_available"):
        out(f"You're up to date - {res.get('current')}.")
        return 0

    out(f"Update available: {res.get('current')} -> {res.get('latest')}")
    if res.get("release_url"):
        out(f"  Release notes: {res.get('release_url')}")
    if args.check:
        return 0

    if not args.yes:
        ans = (input_fn("Apply this update now? [y/N] ") or "").strip().lower()
        if ans not in ("y", "yes"):
            out("Skipped.")
            return 0

    if tracked_tree_dirty(runner):
        out("Aborting: you have uncommitted tracked changes. Commit or stash "
            "them first, then re-run.")
        return 1

    prev = current_commit(runner)
    ok, detail = apply_update(runner)
    if not ok:
        out(f"Update failed: {detail}")
        return 1
    out(f"Updated ({detail}).")

    if not args.skip_verify:
        out("Verifying the new build (running the test gate)...")
        vok, vdetail = verify(runner)
        if not vok:
            out(f"WARNING: {vdetail}")
            out(f"Roll back with:  git reset --hard {prev}")
            return 1
        out(f"Verified - {vdetail}.")

    out(f"Done. Restart JARVIS to load {res.get('latest')}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
