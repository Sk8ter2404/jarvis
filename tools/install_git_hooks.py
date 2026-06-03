#!/usr/bin/env python3
"""Install JARVIS's git hooks — currently a pre-commit PII/secret guard.

Run once per clone (the hook itself lives under .git/, which git never tracks):

    python tools/install_git_hooks.py

The installed pre-commit hook runs ``tools/check_no_pii.py`` before every commit
and BLOCKS the commit on a HARD finding. On the owner's machine it additionally
loads the gitignored owner-specific patterns (``tools/pii_local.py``), so real
personal data (name, email, LAN subnet, device codes) is caught before it can
ever enter history — the strongest line of defence against an accidental leak.

Idempotent: safe to re-run (it overwrites the hook with the current version).
Bypass a genuine false positive on a single commit with ``git commit --no-verify``.

Zero external deps (stdlib only) so it runs in the App-Control-locked env.
"""
from __future__ import annotations

import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The hook body. MUST be written with LF endings (it runs under git's POSIX sh,
# which chokes on CRLF). Blocks only on exit-code 1 (a HARD finding); a scanner
# that can't run at all degrades to a warning rather than bricking commits.
PRE_COMMIT_HOOK = """#!/bin/sh
# JARVIS pre-commit guard (installed by tools/install_git_hooks.py).
# Blocks commits that introduce HARD PII/secret findings. On the owner's machine
# this also loads tools/pii_local.py so real personal data is caught here.
# Bypass a genuine false positive with: git commit --no-verify
python tools/check_no_pii.py
code=$?
if [ "$code" -eq 1 ]; then
  echo "" 1>&2
  echo "[pre-commit] BLOCKED: check_no_pii found HARD PII/secret finding(s) above." 1>&2
  echo "[pre-commit] Move the value to .env or a gitignored file, then re-commit." 1>&2
  echo "[pre-commit] (Genuine false positive? re-run with: git commit --no-verify)" 1>&2
  exit 1
fi
exit 0
"""


def hooks_dir() -> str:
    """Resolve the real hooks directory, honouring core.hooksPath and worktrees
    by asking git directly; falls back to ``<root>/.git/hooks`` if git can't be
    queried (e.g. not a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            p = out.stdout.strip()
            return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)
    except Exception:
        pass
    return os.path.join(_PROJECT_ROOT, ".git", "hooks")


def install(target_dir: str | None = None) -> str | None:
    """Write the pre-commit hook into ``target_dir`` (or the resolved hooks dir).
    Returns the path written, or None on failure (never raises)."""
    hooks = target_dir or hooks_dir()
    try:
        os.makedirs(hooks, exist_ok=True)
    except OSError as e:
        print(f"[install-hooks] could not create {hooks}: {e}")
        return None
    dest = os.path.join(hooks, "pre-commit")
    try:
        # newline="" + explicit \n in the string => LF endings on every OS.
        with open(dest, "w", encoding="utf-8", newline="") as f:
            f.write(PRE_COMMIT_HOOK)
    except OSError as e:
        print(f"[install-hooks] could not write {dest}: {e}")
        return None
    try:
        os.chmod(dest, 0o755)  # real on POSIX; harmless on Windows
    except OSError:
        pass
    return dest


def main() -> int:
    dest = install()
    if dest is None:
        return 1
    print(f"[install-hooks] installed pre-commit guard -> {dest}")
    print("[install-hooks] runs tools/check_no_pii.py and blocks commits with "
          "HARD PII/secret findings.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
