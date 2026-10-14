#!/usr/bin/env python3
"""The weekly success score: how JARVIS did over the last 7 days.

    python tools/weekly_score.py              # print the summary, write the JSON
    python tools/weekly_score.py --days 14
    python tools/weekly_score.py --logs-dir logs --no-write

Reads logs/session_*.log (the session files touched inside the window) and
counts, nothing else:
  * owner turns        "[turn-timing] kind=voice" lines (one per spoken turn);
  * turns answered     those whose outcome= is ok or shortcut;
  * median latency     first_play minus you (ms from the transcript to the
                       first audible sound), voice turns that played;
  * actions run/failed "[action] <name>: <result>" lines; a result that opens
                       with a failure phrase counts as failed;
  * top 3 failing actions, by name;
  * turn-checker flags "[turn-check]" lines.

COUNTS AND ACTION NAMES ONLY. No conversation text is kept or printed, a
malformed line is skipped, and nothing leaves the machine. Writes
data/weekly_score.json (core.paths.data_file) unless --no-write.
Stdlib + core.paths only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_DAYS = 7
OUT_NAME = "weekly_score.json"
_NAME_TS_RE = re.compile(r"session_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
_TIMING_RE = re.compile(r"\[turn-timing\]\s+kind=(\w+)\s+outcome=(\w+)")
_FIELD_RE = re.compile(r"\b(you|first_play)=(\d+)")
_ACTION_RE = re.compile(r"\[action\]\s+([a-z0-9_]{1,64}):\s*(.*)")
_FAIL_RE = re.compile(
    r"^(?:failed|could not|couldn'?t|can'?t|cannot|unable|error|no window|"
    r"not found|timed out|i haven'?t)", re.I)
_ANSWERED = frozenset({"ok", "shortcut"})


def default_logs_dir() -> str:
    from core import paths
    return os.path.join(paths.PROJECT_DIR, "logs")


def default_out_path() -> str:
    from core import paths
    return paths.data_file(OUT_NAME)


def _session_start(path: str) -> float:
    m = _NAME_TS_RE.search(os.path.basename(path))
    if m:
        try:
            return time.mktime(time.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S"))
        except (ValueError, OverflowError):
            pass
    return 0.0


def session_files(logs_dir: str, now: float, days: float) -> list:
    """Session logs started or last written inside the window."""
    since = now - days * 86400.0
    out = []
    for p in glob.glob(os.path.join(logs_dir, "session_*.log")):
        try:
            last = os.path.getmtime(p)
        except OSError:
            continue
        if max(_session_start(p), last) >= since:
            out.append(p)
    return sorted(out)


def _failed(result: str) -> bool:
    return bool(_FAIL_RE.match((result or "").strip()))


def compute(lines, days: float = DEFAULT_DAYS) -> dict:
    """Counts for an iterable of log lines. Malformed lines are skipped."""
    turns = answered = inject = checks = run = failed = 0
    lat: list = []
    fails: dict = {}
    for line in lines:
        try:
            if "[turn-timing]" in line:
                m = _TIMING_RE.search(line)
                if not m:
                    continue
                if m.group(1) == "voice":
                    turns += 1
                    if m.group(2) in _ANSWERED:
                        answered += 1
                    f = {k: int(v) for k, v in _FIELD_RE.findall(line)}
                    if ("first_play" in f and "you" in f
                            and f["first_play"] >= f["you"]):
                        lat.append(f["first_play"] - f["you"])
                elif m.group(1) == "inject":
                    inject += 1
            elif "[action]" in line:
                m = _ACTION_RE.search(line)
                if not m:
                    continue
                run += 1
                if _failed(m.group(2)):
                    failed += 1
                    fails[m.group(1)] = fails.get(m.group(1), 0) + 1
            elif "[turn-check]" in line:
                checks += 1
        except Exception:
            continue
    top = sorted(fails.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {
        "days": days,
        "owner_turns": turns,
        "turns_answered": answered,
        "answer_rate": round(answered / turns, 3) if turns else None,
        "injected_turns": inject,
        "median_latency_ms": int(statistics.median(lat)) if lat else None,
        "actions_run": run,
        "actions_failed": failed,
        "action_success_rate": round((run - failed) / run, 3) if run else None,
        "top_failing_actions": [{"action": a, "failures": n} for a, n in top],
        "turn_check_flags": checks,
    }


def build(logs_dir=None, now=None, days: float = DEFAULT_DAYS) -> dict:
    now = time.time() if now is None else float(now)
    logs_dir = logs_dir or default_logs_dir()
    files = session_files(logs_dir, now, days)

    def _lines():
        for p in files:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    for ln in fh:
                        yield ln[:600]
            except OSError:
                continue

    res = compute(_lines(), days)
    res["sessions"] = len(files)
    res["generated"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    return res


def summary_text(r: dict) -> str:
    """Two or three spoken sentences from build()'s dict."""
    if not r.get("owner_turns") and not r.get("actions_run"):
        return ("I have no turns logged for the past week, sir, "
                "so there is nothing to score.")
    parts = []
    s = (f"This week you spoke to me {r['owner_turns']} times and I answered "
         f"{r['turns_answered']} of them")
    if r.get("median_latency_ms"):
        s += (f", with a median of {r['median_latency_ms'] / 1000:.1f} "
              "seconds to first sound")
    parts.append(s + ".")
    if r["actions_run"]:
        s = f"I ran {r['actions_run']} actions and {r['actions_failed']} failed"
        top = r.get("top_failing_actions") or []
        if top:
            s += ", mostly " + ", ".join(
                x["action"].replace("_", " ") for x in top)
        parts.append(s + ".")
    return " ".join(parts)


def write_json(result: dict, path=None) -> str:
    path = path or default_out_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=DEFAULT_DAYS)
    ap.add_argument("--logs-dir")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    res = build(a.logs_dir, days=a.days)
    print(summary_text(res))
    print(json.dumps(res, indent=2))
    if not a.no_write:
        print("wrote", write_json(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
