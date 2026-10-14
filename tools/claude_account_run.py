#!/usr/bin/env python3
"""claude_account_run - run a Claude Code job on the right subscription, or
report which one that is (core/claude_accounts.py, 2026-10-14).

    python tools/claude_account_run.py --status
        Which profile the NEXT job runs on and why (Pro first until
        CLAUDE_PRO_FIRST_UNTIL, Max while Pro is at a limit or afterwards).
    python tools/claude_account_run.py --check
        Sign-in state of both profiles via `claude auth status` - logged in or
        not, the account's email and plan. Prints no token, ever.
    python tools/claude_account_run.py CLAUDE_PATH [claude args ...]
        Run one CAPTURED job (e.g. `--print ...`) Pro first; when Pro refuses
        (limit / signed out) the same job runs again on Max. Prints the job's
        output and exits with its code. upgrade_jarvis.py's dry audit uses it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import claude_accounts as acct  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# The only `claude auth status` fields ever printed.
_SAFE_FIELDS = ("loggedIn", "authMethod", "apiProvider", "email", "emailAddress",
                "subscriptionType", "plan", "orgName", "organizationName")


def _claude() -> str:
    return shutil.which("claude") or "claude"


def status() -> int:
    until = acct.pro_until()
    st = acct.load_state()
    print(acct.status_line())
    print(f"  Pro profile folder : {acct.pro_dir()}")
    print(f"  Pro signed in      : {'yes' if acct.pro_signed_in() else 'no'}")
    print(f"  Pro first until    : {until.isoformat() if until else 'off'}")
    blocked = float(st.get('pro_blocked_until') or 0)
    if blocked:
        kind = acct.LABELS.get(st.get("pro_blocked_kind", ""), "limit")
        print(f"  Last Pro refusal   : {kind}, parked until {acct._hhmm(blocked)}")
    return 0


def _auth_status(profile: str) -> dict:
    try:
        r = subprocess.run([_claude(), "auth", "status", "--json"],
                           env=acct.env_for(profile), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        data = json.loads(r.stdout or "{}")
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {"error": type(e).__name__}


def check() -> int:
    for profile, label in ((acct.PRO, f"Pro  ({acct.pro_dir()})"),
                           (acct.MAX, "Max  (default profile)")):
        data = _auth_status(profile)
        shown = {k: data[k] for k in _SAFE_FIELDS if k in data}
        if "error" in data:
            shown["error"] = data["error"]
        hidden = len([k for k in data if k not in _SAFE_FIELDS and k != "error"])
        extra = f" (+{hidden} other field(s) not shown)" if hidden else ""
        print(f"{label}: {json.dumps(shown)}{extra}")
    print(acct.status_line())
    return 0


def run_job(argv: list) -> int:
    res = acct.run(argv, env=os.environ.copy(), capture_output=True, text=True,
                   encoding="utf-8", errors="replace",
                   log=lambda m: print(m, file=sys.stderr, flush=True))
    sys.stdout.write(res.stdout or "")
    sys.stderr.write(res.stderr or "")
    sys.stdout.flush()
    return int(res.returncode or 0)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] == "--status":
        return status()
    if argv[0] == "--check":
        return check()
    return run_job(argv)


if __name__ == "__main__":
    sys.exit(main())
