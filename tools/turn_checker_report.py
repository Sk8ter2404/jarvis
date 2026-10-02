#!/usr/bin/env python3
"""Turn-checker verdict counts from JARVIS session logs.

Runs core/turn_checker.check_turn over every owner turn in the logs and
prints how many land in each verdict kind, plus how many should_escalate
would retry on Claude, so the escalation tier's threshold can be judged on
real traffic before it is wired in.

    python tools/turn_checker_report.py C:\\JARVIS\\logs
    python tools/turn_checker_report.py session_2026-10-02_09-00-00.log

A path is a session log or a folder of them (its session_*.log files);
several may be given.

READ-ONLY, COUNTS ONLY. The logs are opened for reading and nothing is
written: the report goes to stdout. It prints NO transcript, reply, action
argument, path or other log text, only counts (the same rule as
tools/turn_latency_report.py). Stdlib + repo modules only.

WHAT A TURN IS. A "You:" line opens one; it closes at its "[turn-timing]
kind=..." line, a "[proactive]" line or the next "You:". Inside it:
  * "JARVIS:" lines are the reply (first round and follow-up rounds); their
    [ACTION: name] tokens are the emitted names;
  * "[action] name: ..." / "[action] name failed ..." lines are actions that
    ran; "[action] ⚠ ..." (a confirmation prompt) and "[pushback]" mark the
    turn as needing confirmation;
  * "[autocorrect] ambiguous ..." means the runtime asked "did you mean X or
    Y?", so the turn asked the owner a question;
  * "[fast-path] <kind>" means no LLM answered (a deterministic reply or a
    read-out action that logs no "[action]" line): the turn is counted but
    not checked, since there is no local reply to escalate.

THE REGISTRY. A log does not list the live registry, so each log's is rebuilt
from it: every name that ran or was emitted, minus every name an
"[autocorrect] ..." line handled (it was not registered). command_no_action
therefore counts only actions that log used somewhere: an undercount, never
an overcount.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import turn_checker  # noqa: E402

# bobert_companion._ACTION_RE, verbatim (tests pin the two together).
ACTION_TOKEN_RE = re.compile(r"\[ACTION:\s*([a-z0-9_]+)\s*(?:,\s*(.+?))?\s*\]", re.IGNORECASE)

_TS = re.compile(r"^\[\d\d:\d\d:\d\d\]\s?")
_RAN = re.compile(r"^\[action\]\s+([A-Za-z0-9_]+)(?::|\s+failed\b)")
_FAST_PATH = re.compile(r"^\[fast-path\]\s+[a-z][a-z0-9_-]*$")
# Every name autocorrect handled was not registered: "'x' -> 'y'",
# "ambiguous 'x' -> ..." and "no match for 'x'".
_UNREGISTERED = re.compile(
    r"^\[autocorrect\]\s+(?:no match for |ambiguous )?'([^']+)'")


def log_paths(paths) -> list:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "session_*.log"))))
        elif os.path.isfile(p):
            out.append(p)
    return out


def _new_turn(user: str) -> dict:
    return {"user": user, "reply": [], "emitted": [], "ran": set(),
            "asked": False, "confirm": False, "fast_path": False}


def parse_log(path: str):
    """One session log -> (turns, unknown action names)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    turns, unknown = [], set()
    turn = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = _TS.sub("", line, count=1).strip()
        if s.startswith("You:"):
            turn = _new_turn(s[len("You:"):].strip())
            turns.append(turn)
            continue
        m = _UNREGISTERED.match(s)
        if m:
            unknown.add(m.group(1).strip().lower())
        if turn is None:
            continue
        if s.startswith("[turn-timing] kind=") or s.startswith("[proactive]"):
            turn = None
        elif s.startswith("JARVIS:"):
            text = s[len("JARVIS:"):].strip()
            turn["reply"].append(text)
            turn["emitted"].extend(
                m.group(1).lower() for m in ACTION_TOKEN_RE.finditer(text))
        elif s.startswith("[action]"):
            m = _RAN.match(s)
            if m:
                turn["ran"].add(m.group(1).lower())
            elif "⚠" in s:
                turn["confirm"] = True
        elif s.startswith("[pushback]"):
            turn["confirm"] = True
        elif s.startswith("[autocorrect] ambiguous"):
            turn["asked"] = True
        elif _FAST_PATH.match(s):
            turn["fast_path"] = True
    return turns, unknown


def verdicts(turns, unknown):
    """(verdict, needs_confirmation) for each turn of one log; None in place
    of the verdict for a fast-path turn (not checked)."""
    registry = set()
    for t in turns:
        registry |= t["ran"] | set(t["emitted"])
    registry -= unknown
    out = []
    for t in turns:
        if t["fast_path"]:
            out.append((None, t["confirm"]))
            continue
        v = turn_checker.check_turn(
            t["user"], " ".join(t["reply"]), t["emitted"], t["ran"],
            registry, asked_question=True if t["asked"] else None)
        out.append((v, t["confirm"]))
    return out


def report(paths) -> str:
    files = log_paths(paths)
    counts = dict.fromkeys(turn_checker.KINDS, 0)
    escalate = fast = 0
    for p in files:
        try:
            turns, unknown = parse_log(p)
        except OSError:
            print("  (skipped a log that could not be read)", file=sys.stderr)
            continue
        for v, confirm in verdicts(turns, unknown):
            if v is None:
                fast += 1
                continue
            counts[v.kind] += 1
            if turn_checker.should_escalate(v, cloud_allowed=True,
                                            already_escalated=False,
                                            needs_confirmation=confirm):
                escalate += 1
    total = sum(counts.values()) + fast
    lines = ["JARVIS turn-checker report (counts only; read-only)",
             f"logs: {len(files)} session file(s), {total} turn(s), {fast} "
             f"on a fast path (no LLM, not checked)"]
    for kind in turn_checker.KINDS:
        lines.append(f"  {kind:<18} {counts[kind]:>6}")
    lines.append(f"  {'would escalate':<18} {escalate:>6}   (cloud allowed, "
                 f"confidence >= {turn_checker.ESCALATE_MIN_CONFIDENCE})")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Count core/turn_checker verdicts over JARVIS session "
                    "logs (counts only, never transcript text).")
    ap.add_argument("paths", nargs="+",
                    help="session log file(s) or folder(s) of session_*.log")
    args = ap.parse_args(argv)
    if not log_paths(args.paths):
        print("no session logs found", file=sys.stderr)
        return 2
    sys.stdout.write(report(args.paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
