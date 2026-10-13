#!/usr/bin/env python3
"""Install JARVIS's git hooks — pre-commit, commit-msg and pre-push PII/secret guards.

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
#
# FAIL CLOSED (2026-07-14 audit). This used to block ONLY on exit 1 — but
# check_no_pii deliberately exits 2 when it CANNOT SCAN SAFELY (a present-but-
# broken pii_local.py: the fail-closed path added on 2026-07-04 precisely so a
# degraded scanner can't wave secrets through). exit 2 fell into the `exit 0`
# tail, so the hook that exists to protect a PUBLIC repo silently PASSED the
# commit in exactly the situation it was designed to catch. Any nonzero status
# now blocks, except 127 (python missing entirely — degrade to a warning rather
# than bricking every commit on a machine without the interpreter).
python tools/check_no_pii.py
code=$?
if [ "$code" -eq 127 ]; then
  echo "[pre-commit] WARNING: python not found — PII scan SKIPPED." 1>&2
  exit 0
fi
if [ "$code" -ne 0 ]; then
  echo "" 1>&2
  if [ "$code" -eq 1 ]; then
    echo "[pre-commit] BLOCKED: check_no_pii found HARD PII/secret finding(s) above." 1>&2
    echo "[pre-commit] Move the value to .env or a gitignored file, then re-commit." 1>&2
  else
    echo "[pre-commit] BLOCKED: check_no_pii could not scan safely (exit $code)." 1>&2
    echo "[pre-commit] Refusing to commit to a PUBLIC repo with a degraded scanner." 1>&2
  fi
  echo "[pre-commit] (Genuine false positive? re-run with: git commit --no-verify)" 1>&2
  exit 1
fi
exit 0
"""


# commit-msg: scans the message AND the author/committer identity git is about
# to record (the 2026-10-05 leak was a personal e-mail in the author field of two
# commits made from a shell with no repo-local user.email).
COMMIT_MSG_HOOK = """#!/bin/sh
# JARVIS commit-msg guard (installed by tools/install_git_hooks.py).
python tools/check_no_pii.py --commit-msg "$1"
code=$?
if [ "$code" -eq 127 ]; then
  echo "[commit-msg] WARNING: python not found - identity scan SKIPPED." 1>&2
  exit 0
fi
if [ "$code" -ne 0 ]; then
  echo "[commit-msg] BLOCKED: commit identity/message failed the PII gate." 1>&2
  exit 1
fi
exit 0
"""

# pre-push: the last line of defence for commits that skipped the other hooks
# (made with --no-verify, in another clone, or by an agent). Scans every pushed
# commit's author, committer and message.
PRE_PUSH_HOOK = """#!/bin/sh
# JARVIS pre-push guard (installed by tools/install_git_hooks.py).
python tools/check_no_pii.py --pre-push
code=$?
if [ "$code" -eq 127 ]; then
  echo "[pre-push] WARNING: python not found - metadata scan SKIPPED." 1>&2
  exit 0
fi
if [ "$code" -ne 0 ]; then
  echo "[pre-push] BLOCKED: a commit being pushed carries personal data in its" 1>&2
  echo "[pre-push] author/committer identity or message (see above)." 1>&2
  exit 1
fi
exit 0
"""

_HOOKS = (("pre-commit", PRE_COMMIT_HOOK), ("commit-msg", COMMIT_MSG_HOOK),
          ("pre-push", PRE_PUSH_HOOK))


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
    for name, body in _HOOKS:
        path = os.path.join(hooks, name)
        try:
            # newline="" + LF line feeds in the string => LF on every OS.
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(body)
        except OSError as e:
            print(f"[install-hooks] could not write {path}: {e}")
            return None
        try:
            os.chmod(path, 0o755)  # real on POSIX; harmless on Windows
        except OSError:
            pass
    return dest


def main() -> int:
    dest = install()
    if dest is None:
        return 1
    print(f"[install-hooks] installed pre-commit, commit-msg and pre-push "
          f"guards -> {os.path.dirname(dest)}")
    print("[install-hooks] they run tools/check_no_pii.py and block commits and "
          "pushes with HARD PII/secret findings (file contents, commit "
          "identity, commit messages).")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
