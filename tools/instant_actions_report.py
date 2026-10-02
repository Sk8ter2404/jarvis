#!/usr/bin/env python3
"""The weekly score of INSTANT_ACTIONS (core/instant_actions.py).

    python tools/instant_actions_report.py
    python tools/instant_actions_report.py --log backup/instant_actions.jsonl
    python tools/instant_actions_report.py --days 14

Reads data/instant_actions.jsonl (and its rotated .1 file), the log the
monolith writes for every turn an instant rule matched, and prints:
  * per action: shadow turns and how many of them the brain answered with
    the SAME action ("agree"), "on" turns and how many ran cleanly ("ok");
  * the agreement rate over every shadow row: the precision of the rules —
    given the same words, how often the brain chose the same action;
  * the pass rate over the last --days days (7 by default): shadow rows that
    agreed plus "on" rows that ran cleanly, out of every row in that window.
    That is the weekly score; judge it before switching the mode to "on".

READ-ONLY, COUNTS ONLY. The log is opened for reading and nothing is written:
the report goes to stdout. The log holds only times and action names, and
the report prints only action names and counts.

Exit status: 0 = report printed (it may say there are no rows yet); 2 = no
log file to read, so nothing was checked. Stdlib + repo modules only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import instant_actions  # noqa: E402

DEFAULT_DAYS = 7


def default_log_path() -> str:
    """data/instant_actions.jsonl as the running JARVIS resolves it
    (core.paths: JARVIS_DATA_DIR and staging redirect it)."""
    from core import paths
    return paths.data_file(instant_actions.LOG_NAME, create_dir=False)


def _passed(row: dict) -> bool:
    if row.get("mode") == "shadow":
        return bool(row.get("agree"))
    return bool(row.get("ok"))


def summarize(rows, now: float, days: float = DEFAULT_DAYS) -> dict:
    """Counts for ``rows`` (core.instant_actions.read_rows output)."""
    per: dict = {}
    agree = shadow = 0
    window = passed = 0
    since = float(now) - float(days) * 86400.0
    for row in rows:
        a = per.setdefault(row["action"],
                           {"shadow": 0, "agree": 0, "on": 0, "ok": 0})
        if row["mode"] == "shadow":
            a["shadow"] += 1
            shadow += 1
            if row["agree"]:
                a["agree"] += 1
                agree += 1
        else:
            a["on"] += 1
            if row["ok"]:
                a["ok"] += 1
        if since <= float(row["ts"]) <= float(now):
            window += 1
            if _passed(row):
                passed += 1
    return {"rows": len(rows), "per_action": per, "shadow": shadow,
            "agree": agree, "window": window, "passed": passed,
            "days": days}


def _rate(k: int, n: int) -> str:
    if not n:
        return "n/a"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def report(rows, now: float, days: float = DEFAULT_DAYS) -> str:
    s = summarize(rows, now, days)
    days_txt = f"{days:g}"
    lines = ["JARVIS instant-actions report (counts only; read-only)",
             f"log: {s['rows']} row(s)"]
    if s["per_action"]:
        lines.append(f"  {'action':<20} {'shadow':>6} {'agree':>6} "
                     f"{'on':>6} {'ok':>6}")
        for name in sorted(s["per_action"]):
            a = s["per_action"][name]
            lines.append(f"  {name:<20} {a['shadow']:>6} {a['agree']:>6} "
                         f"{a['on']:>6} {a['ok']:>6}")
    else:
        lines.append("  no instant-action turns logged yet")
    lines.append(f"agreement with the brain (all shadow rows): "
                 f"{_rate(s['agree'], s['shadow'])}")
    lines.append(f"pass rate, last {days_txt} days: "
                 f"{_rate(s['passed'], s['window'])}")
    return "\n".join(lines) + "\n"


def main(argv=None, now=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score INSTANT_ACTIONS from data/instant_actions.jsonl "
                    "(action names and counts only).")
    ap.add_argument("--log", default=None,
                    help="the log file (default: data/instant_actions.jsonl)")
    ap.add_argument("--days", type=float, default=DEFAULT_DAYS,
                    help="pass-rate window in days (default 7)")
    args = ap.parse_args(argv)
    path = args.log or default_log_path()
    if not (os.path.isfile(path) or os.path.isfile(path + ".1")):
        print("no instant-actions log found - nothing to score",
              file=sys.stderr)
        return 2
    rows = instant_actions.read_rows(path)
    sys.stdout.write(report(rows, time.time() if now is None else now,
                            args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
