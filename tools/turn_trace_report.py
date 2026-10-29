#!/usr/bin/env python3
"""Summarise one day of the turn trace (v3 W20, core/turn_trace.py).

    python tools/turn_trace_report.py                    # today
    python tools/turn_trace_report.py --day 2026-10-27
    python tools/turn_trace_report.py --day 2026-10-27 --kind voice
    python tools/turn_trace_report.py --dir <copy of data/turns> --json
    python tools/turn_trace_report.py --logs C:\\JARVIS\\logs   # + log cross-check

Reads ``data/turns/turns-<day>*.jsonl`` (the day file and its size-rotated
parts) and prints:

  * how many turns, by channel / kind / outcome;
  * latency p50 / p95 per stage (stt, llm, actions, first_play, audible,
    end), in ms, from the [turn-timing] numbers each record carries;
  * the prefill state counts (warm / partial / full) and "partial+full / N",
    the figure the W09 learn pre-filter is judged by (baseline 27/149);
  * actions run / failed (and the ones that failed most), follow-up rounds
    and why chains were cut;
  * the validators that fired;
  * the shadow-mode counters of W02 [head-hold], W09 [learn-prefilter], W11
    [action-fit] and W21 [loop-guard] ("would ..." lines), from the records.
    With --logs the same lines are also counted in that day's session logs
    (``session_<day>_*.log``) as a cross-check; a tag with no line anywhere
    prints 0 (its item may not be in this build yet).
  * W09's "(ambient fragment)" lines on a row of their own (3.0.6): the
    ambient-learn path prints them for room / TV audio, not for an owner
    turn (103 of 116 W09 log lines on 2026-10-28), so they are kept out of
    the per-turn W09 figure in both the trace and the log counts. Trace
    records written before 3.0.6 still have them inside learn-prefilter.

READ-ONLY and NUMBERS ONLY: nothing is written and neither the owner's words
nor any reply text is ever printed (the 2026-09-06 PII near-miss rule that
tools/turn_latency_report.py follows too). The percentile method matches that
tool's (linear interpolation, rounded to an int), so the two agree on a day.
Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_HERE)
DEFAULT_DIR = os.path.join(PROJECT_DIR, "data", "turns")

STAGES = ("stt", "llm", "actions", "first_play", "audible", "end")
SHADOW_TAGS = ("head-hold", "learn-prefilter", "action-fit", "loop-guard")
SHADOW_LABEL = {"head-hold": "W02", "learn-prefilter": "W09",
                "action-fit": "W11", "loop-guard": "W21"}
# core.turn_trace.AMBIENT_SHADOW_KEY (kept literal: this tool is stdlib only).
AMBIENT_KEY = "learn-prefilter-ambient"
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LOG_LINE = re.compile(r"^\s*(?:\[\d\d:\d\d:\d\d\]\s*)?\[([a-z0-9_-]+)\]\s?(.*)")


def pct(xs, p):
    """Linear-interpolated percentile of the non-None values, rounded to an
    int (the same method as tools/turn_latency_report.pct); None if empty."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return int(round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo)))


def day_files(directory: str, day: str) -> list:
    """The day file and its size-rotated parts, oldest part first."""
    base = os.path.join(directory, f"turns-{day}.jsonl")
    parts = glob.glob(os.path.join(directory, f"turns-{day}.*.jsonl"))

    def n(path):
        m = re.search(r"\.(\d+)\.jsonl$", path)
        return int(m.group(1)) if m else 0
    out = sorted(parts, key=n)
    if os.path.isfile(base):
        out.append(base)
    return out


def load_day(directory: str, day: str) -> list:
    """Every parseable record of ``day``. A torn or foreign line is skipped,
    never fatal (a crash can leave half a line at the end of a file)."""
    recs = []
    for path in day_files(directory, day):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        recs.append(rec)
        except OSError:
            continue
    return recs


def log_shadow_counts(log_dir: str, day: str) -> dict:
    """{tag: lines} of "would" lines for the four shadow tags in that day's
    session logs, with W09's ambient-fragment lines under AMBIENT_KEY (the
    same split core.turn_trace.observe makes). Counted only; no text is
    kept."""
    counts = {t: 0 for t in SHADOW_TAGS}
    counts[AMBIENT_KEY] = 0
    for path in glob.glob(os.path.join(log_dir, f"session_{day}_*.log")):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "would" not in line:
                        continue
                    m = _LOG_LINE.match(line)
                    if m and m.group(1) in SHADOW_TAGS and "would" in m.group(2):
                        tag = m.group(1)
                        if tag == "learn-prefilter" and "ambient fragment" in m.group(2):
                            tag = AMBIENT_KEY
                        counts[tag] += 1
        except OSError:
            continue
    return counts


def summarize(recs: list, kind=None) -> dict:
    """Every number the report prints, as plain data."""
    if kind:
        recs = [r for r in recs if r.get("kind") == kind]
    out = {"turns": len(recs), "by_channel": {}, "by_kind": {},
           "by_outcome": {}, "latency": {}, "pe_state": {}, "actions": {},
           "cuts": {}, "validators": {},
           "shadow": dict({t: 0 for t in SHADOW_TAGS}, **{AMBIENT_KEY: 0}),
           "rounds": 0}

    def bump(d, k, n=1):
        d[k] = d.get(k, 0) + n

    stage_vals = {s: [] for s in STAGES}
    ran = failed = 0
    failed_by = {}
    ran_by = {}
    for r in recs:
        bump(out["by_channel"], str(r.get("channel") or "?"))
        bump(out["by_kind"], str(r.get("kind") or "?"))
        bump(out["by_outcome"], str(r.get("outcome") or "?"))
        st = r.get("stages") if isinstance(r.get("stages"), dict) else {}
        for s in STAGES:
            v = st.get(s)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stage_vals[s].append(v)
        pe = r.get("pe_state")
        if isinstance(pe, str) and pe:
            bump(out["pe_state"], pe)
        for a in r.get("actions") or ():
            if not isinstance(a, dict):
                continue
            ran += 1
            name = str(a.get("n") or "?")
            bump(ran_by, name)
            if a.get("ok") is False:
                failed += 1
                bump(failed_by, name)
        cut = r.get("cut")
        if cut:
            bump(out["cuts"], str(cut))
        if isinstance(r.get("rounds"), int):
            out["rounds"] += r["rounds"]
        for tag, n in (r.get("validators") or {}).items():
            if isinstance(n, int):
                bump(out["validators"], str(tag), n)
        for tag, n in (r.get("shadow") or {}).items():
            if isinstance(n, int):
                bump(out["shadow"], str(tag), n)
    for s in STAGES:
        xs = stage_vals[s]
        out["latency"][s] = {"n": len(xs), "p50": pct(xs, 50),
                             "p95": pct(xs, 95)}
    out["actions"] = {
        "ran": ran, "failed": failed,
        "top_failed": sorted(failed_by.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
        "top_ran": sorted(ran_by.items(), key=lambda kv: (-kv[1], kv[0]))[:8]}
    pe = out["pe_state"]
    known = sum(pe.values())
    out["prefill"] = {"partial_plus_full": pe.get("partial", 0) + pe.get("full", 0),
                      "known": known}
    return out


def _fmt(v):
    return "-" if v is None else str(v)


def render(day: str, s: dict, log_counts=None) -> str:
    L = [f"Turn trace {day}: {s['turns']} turn(s)"]
    if not s["turns"]:
        L.append("  (no records for this day)")
    for label, key in (("channel", "by_channel"), ("kind", "by_kind"),
                       ("outcome", "by_outcome")):
        if s[key]:
            L.append(f"  by {label}: " + ", ".join(
                f"{k}={v}" for k, v in sorted(s[key].items())))
    L.append("")
    L.append("  latency (ms)        n     p50     p95")
    for st in STAGES:
        row = s["latency"][st]
        L.append(f"  {st:<16} {row['n']:>5} {_fmt(row['p50']):>7} "
                 f"{_fmt(row['p95']):>7}")
    L.append("")
    pf = s["prefill"]
    pe = s["pe_state"]
    L.append("  prefill: " + (", ".join(f"{k}={pe[k]}" for k in sorted(pe))
                              or "none recorded")
             + f"  -> partial+full {pf['partial_plus_full']}/{pf['known']}")
    a = s["actions"]
    L.append(f"  actions: {a['ran']} ran, {a['failed']} failed; "
             f"follow-up rounds {s['rounds']}")
    if a["top_failed"]:
        L.append("    failed most: " + ", ".join(
            f"{n} x{c}" for n, c in a["top_failed"]))
    if s["cuts"]:
        L.append("  chain cuts: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(s["cuts"].items())))
    if s["validators"]:
        L.append("  validators fired: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(s["validators"].items())))
    else:
        L.append("  validators fired: none")
    L.append("  shadow 'would' lines (trace"
             + (" / session logs" if log_counts is not None else "") + "):")
    for tag in SHADOW_TAGS:
        n = s["shadow"].get(tag, 0)
        extra = ""
        if log_counts is not None:
            extra = f" / {log_counts.get(tag, 0)}"
        L.append(f"    {SHADOW_LABEL[tag]} [{tag}]: {n}{extra}")
    n = s["shadow"].get(AMBIENT_KEY, 0)
    extra = f" / {log_counts.get(AMBIENT_KEY, 0)}" if log_counts is not None else ""
    L.append(f"    W09 ambient fragments (no owner turn): {n}{extra}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarise one day of the JARVIS turn trace "
                    "(read-only; numbers only; prints to stdout).")
    ap.add_argument("--day", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help=f"trace folder (default {DEFAULT_DIR})")
    ap.add_argument("--kind", choices=("voice", "inject", "realtime"),
                    help="only turns of this [turn-timing] kind")
    ap.add_argument("--logs", help="session log folder: also count the "
                                   "shadow 'would' lines there")
    ap.add_argument("--json", action="store_true",
                    help="print the numbers as JSON")
    a = ap.parse_args(argv)
    day = a.day or time.strftime("%Y-%m-%d")
    if not _DAY.match(day):
        ap.error("--day must be YYYY-MM-DD")
    if not os.path.isdir(a.dir):
        print(f"no such trace folder: {a.dir}", file=sys.stderr)
        return 2
    summary = summarize(load_day(a.dir, day), kind=a.kind)
    logs = None
    if a.logs:
        if not os.path.isdir(a.logs):
            print(f"no such log folder: {a.logs}", file=sys.stderr)
            return 2
        logs = log_shadow_counts(a.logs, day)
    if a.json:
        payload = dict(summary, day=day)
        if logs is not None:
            payload["log_shadow"] = logs
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render(day, summary, logs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
