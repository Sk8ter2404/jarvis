#!/usr/bin/env python3
"""Run a test gate as a CHILD and guarantee it ends — the outside half of
tools/test_watchdog.py.

THE INCIDENT (2026-10-13)
=========================
The v2.0.191 local gate (``python tools/run_tests.py`` under a hand-rolled
scratchpad wrapper, started 09:15:57) tripped the in-process watchdog, which
printed both stack dumps and "[test-watchdog] exiting 3" at 09:43:39. The
python.exe then lived ~26 more hours — flat CPU, 5,520 MB private — and the
wrapper sat in ``subprocess.call`` the whole time, until the process was killed
by PID on 2026-10-14. The watchdog's own exit now uses TerminateProcess (see
its module docstring), but an in-process bound has limits no in-process fix can
remove: its thread has to be scheduled and has to get the GIL. That same dump
fired at 374.9 s on a 120 s budget — 255 s late, cause undetermined.

So the PARENT enforces two things this module owns:

  * EXIT GRACE — once the log carries "[test-watchdog] exiting", the child has
    ``--exit-grace`` seconds (default 60) to be gone, or it is killed. The
    watchdog's verdict stands: the run reports the watchdog's exit code.
  * WALL-CLOCK CEILING — ``--max-seconds``; by default the watchdog's own
    whole-run budget (``JARVIS_TEST_TOTAL_TIMEOUT_S``, else 900 s) times the
    largest multiplier any runner applies to it (4x: run_coverage.py and
    run_tests_ci_sim.py --coverage) plus 10 minutes. A run the watchdog never
    managed to end reports 124, like a shell ``timeout``. A disabled total
    budget leaves this off unless --max-seconds says otherwise.

It keeps the scratchpad wrapper's habits: IDLE priority on Windows, CUDA hidden
(``CUDA_VISIBLE_DEVICES=-1``), UTF-8 output, everything into one log file and a
final ``EXIT=<code>`` line. Output goes to a FILE, never a pipe, so every wait
here is a real bound (a pipe a grandchild inherited is what turns
``subprocess.run(timeout=...)`` into an unbounded drain on Windows).

usage:
    python tools/gate_run.py LOG [--cwd DIR] [--max-seconds N]
                             [--exit-grace N] -- CMD [ARGS...]
e.g.
    python tools/gate_run.py C:/tmp/gate.log --cwd C:/jarvis_wt_x -- \\
        python tools/run_tests.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools import test_watchdog  # noqa: E402 - stdlib-only, safe anywhere

# What tools/test_watchdog.py writes right before it kills its own process.
MARKER = b"[test-watchdog] exiting"
DEFAULT_EXIT_GRACE_S = 60.0
# run_coverage.py and run_tests_ci_sim.py --coverage arm the watchdog with
# DEFAULT_TOTAL_S * 4; the ceiling has to sit above the slowest legal run.
CEILING_FACTOR = 4
CEILING_SLACK_S = 600.0
TIMEOUT_EXIT_CODE = 124
# How long to wait for a killed child to actually go. A thread parked inside a
# driver can outlive even TerminateProcess; then we say so and stop waiting —
# the one thing this module must never do is become the next thing that hangs.
_REAP_S = 30.0
_POLL_S = 1.0


def default_ceiling() -> float:
    """Wall-clock ceiling in seconds, from the watchdog's own budget; 0 = off."""
    total = test_watchdog._budget(test_watchdog.TOTAL_ENV,
                                  test_watchdog.DEFAULT_TOTAL_S)
    return total * CEILING_FACTOR + CEILING_SLACK_S if total else 0.0


def _child_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _note(log: str, text: str) -> None:
    """Append one line to the log. Never raises — it runs on the kill path."""
    try:
        with open(log, "ab") as fh:
            fh.write(text.encode("utf-8", "replace"))
    except OSError:
        pass


class _MarkerScan:
    """Reads only the bytes the child added since the last look, keeping a
    short tail so a marker split across two reads is still found."""

    def __init__(self, log: str):
        self.log = log
        self.offset = 0
        self.tail = b""

    def seen(self) -> bool:
        try:
            with open(self.log, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return False
        self.offset += len(chunk)
        window = self.tail + chunk
        self.tail = window[-(len(MARKER) - 1):]
        return MARKER in window


def _kill(proc: subprocess.Popen, log: str, why: str) -> None:
    _note(log, "\n[gate-run] %s — killing PID %d\n" % (why, proc.pid))
    try:
        proc.kill()                     # TerminateProcess on Windows
    except OSError:
        pass
    try:
        proc.wait(timeout=_REAP_S)
    except subprocess.TimeoutExpired:
        _note(log, "[gate-run] PID %d is STILL alive %ss after the kill (a "
                   "thread parked in a driver?) — not waiting any longer\n"
              % (proc.pid, test_watchdog._fmt(_REAP_S)))


def run(log: str, cmd: list, *, cwd: str | None = None,
        max_seconds: float | None = None,
        exit_grace: float = DEFAULT_EXIT_GRACE_S) -> int:
    """Run ``cmd`` to completion or to a kill; return the exit code to report."""
    ceiling = default_ceiling() if max_seconds is None else max_seconds
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=_child_env(), stdin=subprocess.DEVNULL,
            stdout=fh, stderr=subprocess.STDOUT,
            # 0 off Windows, where Popen rejects any creationflags.
            creationflags=getattr(subprocess, "IDLE_PRIORITY_CLASS", 0))
    scan = _MarkerScan(log)
    start = time.monotonic()
    marked_at = None
    rc = None
    while rc is None:
        try:
            rc = proc.wait(timeout=_POLL_S)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if marked_at is None and scan.seen():
            marked_at = now
        if marked_at is not None and now - marked_at > exit_grace:
            _kill(proc, log, "the test watchdog said it was exiting %ss ago "
                             "and the process is still alive"
                  % test_watchdog._fmt(round(now - marked_at, 1)))
            rc = test_watchdog.EXIT_CODE
        elif ceiling and now - start > ceiling:
            _kill(proc, log, "wall-clock ceiling of %ss blown and the test "
                             "watchdog never ended the run"
                  % test_watchdog._fmt(ceiling))
            rc = TIMEOUT_EXIT_CODE
    _note(log, "\nEXIT=%d\n" % rc)
    return rc


def main(argv: list) -> int:
    if "--" not in argv:
        print(__doc__.split("usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="gate_run.py")
    parser.add_argument("log")
    parser.add_argument("--cwd")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--exit-grace", type=float,
                        default=DEFAULT_EXIT_GRACE_S)
    args = parser.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        parser.error("no command after --")
    rc = run(args.log, cmd, cwd=args.cwd, max_seconds=args.max_seconds,
             exit_grace=args.exit_grace)
    print("EXIT", rc, flush=True)
    return rc


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))
