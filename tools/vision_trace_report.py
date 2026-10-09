"""tools/vision_trace_report.py - a TEXT summary of JARVIS's vision trace
(core/vision_trace.py, owner-approved 2026-10-05).

For a debugging session: hit rate per tier / source and per window, the
misses with the owner's words and the model's raw answer, wrong-monitor
clicks, private skips. It reads data/vision_trace/index.jsonl ONLY - it
never opens an image (image files are listed by name; a debugging session
opens one only with the owner's go-ahead).

    python tools/vision_trace_report.py            # last 7 days
    python tools/vision_trace_report.py --days 1 --misses 30
    python tools/vision_trace_report.py --json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_HIT = ("verified", "found", "already_playing", "read")
_MISS = ("not_found", "no_change", "changed_wrong", "ambiguous", "failed",
         "unfinished")


def summarise(entries, monitors=None) -> dict:
    """The report as a dict (pure: no I/O)."""
    from core import monitor_geometry as _mg
    out = {"entries": len(entries), "steps": collections.Counter(),
           "by_source": {}, "by_window": {}, "misses": [], "private": 0,
           "wrong_monitor": 0, "images": 0, "image_bytes": 0,
           "first": None, "last": None, "cloud_calls": 0}
    by_src = collections.defaultdict(lambda: [0, 0])
    by_win = collections.defaultdict(lambda: [0, 0])
    for e in entries:
        ts = float(e.get("ts") or 0)
        out["first"] = ts if out["first"] is None else min(out["first"], ts)
        out["last"] = ts if out["last"] is None else max(out["last"], ts)
        step = str(e.get("step") or "?")
        out["steps"][step] += 1
        if e.get("privacy"):
            out["private"] += 1
            continue
        imgs = e.get("images") or []
        out["images"] += len(imgs)
        out["image_bytes"] += sum(int(i.get("bytes") or 0) for i in imgs)
        if e.get("cloud"):
            out["cloud_calls"] += 1
        outcome = str(e.get("outcome") or "")
        if step not in ("click", "find", "pick", "scene", "vision",
                        "see_screen"):
            continue
        src = str(e.get("source") or "-")
        scope = e.get("scope") or {}
        win = (scope.get("process") or "?") + " " + (scope.get("url_host")
                                                     or "")
        hit = outcome in _HIT
        by_src[src][0] += hit
        by_src[src][1] += 1
        by_win[win.strip()][0] += hit
        by_win[win.strip()][1] += 1
        named = _mg.monitor_named_in(e.get("utterance") or "", monitors or {})
        if named and scope.get("monitor") and named != scope.get("monitor") \
                and step == "click" and outcome == "verified":
            out["wrong_monitor"] += 1
        if outcome in _MISS:
            out["misses"].append({
                "when": time.strftime("%m-%d %H:%M:%S", time.localtime(ts)),
                "id": e.get("id"), "step": step, "source": src,
                "outcome": outcome, "utterance": e.get("utterance", ""),
                "chosen": e.get("chosen"), "evidence": e.get("evidence", ""),
                "raw_answer": str(e.get("raw_answer") or "")[:300],
                "monitor": scope.get("monitor"),
                "images": [i.get("file") for i in imgs]})
    out["by_source"] = {k: {"hits": v[0], "n": v[1],
                            "rate": round(v[0] / v[1], 3) if v[1] else None}
                        for k, v in sorted(by_src.items())}
    out["by_window"] = {k: {"hits": v[0], "n": v[1],
                            "rate": round(v[0] / v[1], 3) if v[1] else None}
                        for k, v in sorted(by_win.items(),
                                           key=lambda kv: -kv[1][1])[:15]}
    out["steps"] = dict(out["steps"])
    return out


def render(rep: dict, misses: int = 20) -> str:
    lines = []
    if not rep["entries"]:
        return "vision trace: no entries"
    f = time.strftime("%Y-%m-%d %H:%M", time.localtime(rep["first"]))
    l = time.strftime("%Y-%m-%d %H:%M", time.localtime(rep["last"]))
    lines.append(f"vision trace: {rep['entries']} entries, {f} .. {l}")
    lines.append(f"  steps: {rep['steps']}")
    lines.append(f"  private skips: {rep['private']}   wrong-monitor clicks: "
                 f"{rep['wrong_monitor']}   cloud calls: {rep['cloud_calls']}")
    lines.append(f"  images: {rep['images']} "
                 f"({rep['image_bytes'] / 1048576:.1f} MB) - not opened")
    lines.append("  hit rate by source:")
    for k, v in rep["by_source"].items():
        lines.append(f"    {k:<10} {v['hits']}/{v['n']}  ({v['rate']})")
    lines.append("  hit rate by window:")
    for k, v in rep["by_window"].items():
        lines.append(f"    {k[:40]:<40} {v['hits']}/{v['n']}  ({v['rate']})")
    lines.append(f"  misses (newest {misses}):")
    for m in rep["misses"][-misses:]:
        lines.append(f"    {m['when']} {m['step']}/{m['source']} "
                     f"{m['outcome']} on {m['monitor']}: "
                     f"said {m['utterance'][:80]!r} chose {m['chosen']!r}")
        if m["raw_answer"]:
            lines.append(f"        model said: {m['raw_answer'][:160]!r}")
        if m["evidence"]:
            lines.append(f"        evidence: {m['evidence'][:160]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--misses", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    from core import vision_trace as _vt
    try:
        from core.config import MONITORS
    except Exception:
        MONITORS = {}
    cut = time.time() - a.days * 86400
    entries = [e for e in _vt.read_index()
               if float(e.get("ts") or 0) >= cut]
    rep = summarise(entries, MONITORS)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(render(rep, a.misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
