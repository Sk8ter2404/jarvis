"""tools/vision_bench/resolver_bench.py - offline accuracy / latency of the
on-screen click resolver (core.screen_resolve) over SYNTHETIC pages.

Inputs are the UIA trees and OCR lines of the synthetic pages in
tests/_screen_pages.json (read from a throw-away Chrome profile; no owner
screen), turned into candidates the way core.grounded_click does
(_cands_from_snapshot / core.screen_ocr.lines_to_cands), plus - with
``--capture`` - the live UIA captures that tools/vision_bench/
live_uia_bench.py writes (the FRESH2 page). No model, no network.

    python tools/vision_bench/resolver_bench.py
    python tools/vision_bench/resolver_bench.py --capture <live.json> --json
    python tools/vision_bench/resolver_bench.py --baseline <resolver.py>

``--baseline`` loads another resolver module exposing
resolve(query, cands, playing_title=...) (e.g. the research prototype) and
reports both side by side.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import screen_resolve as R                      # noqa: E402
from core.screen_ocr import lines_to_cands               # noqa: E402
from tools.vision_bench import pages as P                # noqa: E402
from tools.vision_bench import queries as Q              # noqa: E402

FIXTURES = os.path.join(_PROJECT, "tests", "_screen_pages.json")


def uia_cands(page: dict) -> tuple:
    """(cands, origin, playing title) from a fixture / capture UIA page,
    filtered exactly like core.grounded_click._cands_from_snapshot."""
    els = page["elements"]
    doc = None
    for e in els:
        if e.get("type") == "Document" and e.get("rect") and \
                e.get("fw", "Chrome") == "Chrome":
            doc = e["rect"]
            break
    out = []
    for e in els:
        if e.get("type") in ("Document", "Edit") or not str(
                e.get("name") or "").strip():
            continue
        if e.get("off") or e.get("pw") or not e.get("rect"):
            continue
        r = e["rect"]
        in_doc = bool(doc and r[1] >= doc[1] - 1 and r[0] >= doc[0] - 1)
        if doc is not None and not in_doc and e.get("type") != "TabItem":
            continue
        out.append({"text": " ".join(str(e["name"]).split()),
                    "rect": list(r), "type": e["type"],
                    "href": e.get("href", "")})
    origin = (doc[0], doc[1]) if doc else (0.0, 0.0)
    return out, origin, str(page.get("title", "")).rsplit(" - ", 1)[0]


def ocr_cands(page: dict) -> tuple:
    cands = [c for c in lines_to_cands(page["lines"]) if c["rect"][1] >= 88]
    return cands, (0.0, 88.0), str(page.get("title", "")).rsplit(" - ", 1)[0]


def run(label, src, rows, fn, meta) -> dict:
    n = ok = wrong = dec = 0
    fails = []
    t0 = time.perf_counter()
    for page, query, exp in rows:
        if page not in src:
            continue
        cands, origin, playing = src[page]
        res = fn(query, cands, playing_title=playing)
        st = res.get("status")
        tgt = res.get("target") or {}
        opts = [R.label_of(o) for o in res.get("options") or ()]
        v = Q.judge("ok" if st == "ok" else st, R.label_of(tgt),
                    tgt.get("rect") or (0, 0, 0, 0), opts, exp, page, meta,
                    origin)
        n += 1
        ok += v == "ok"
        wrong += v == "wrong"
        dec += v == "declined"
        if v != "ok":
            fails.append({"verdict": v, "query": query, "expected": str(exp),
                          "got": R.label_of(tgt)[:60] if st == "ok"
                          else (opts[:3] or st)})
    ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
    return {"label": label, "n": n, "ok": ok, "wrong": wrong,
            "declined": dec, "ms_per_query": round(ms, 2), "fails": fails}


def _load_baseline(path):
    spec = importlib.util.spec_from_file_location("vision_bench_baseline",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", help="live_uia_bench capture JSON")
    ap.add_argument("--baseline", help="another resolver module to compare")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    with open(FIXTURES, encoding="utf-8") as f:
        fx = json.load(f)
    meta = P.meta_dict()
    uia = {k: uia_cands(v) for k, v in fx["uia"].items()}
    ocr = {k: ocr_cands(v) for k, v in fx["ocr"].items()}
    if a.capture:
        with open(a.capture, encoding="utf-8") as f:
            cap = json.load(f)
        for name, pg in (cap.get("captures") or {}).items():
            uia[name] = uia_cands(pg)
    resolvers = [("shipped", R.resolve)]
    if a.baseline:
        resolvers.insert(0, ("baseline", _load_baseline(a.baseline)))
    out = []
    for rname, fn in resolvers:
        for sname, rows in Q.SETS.items():
            for src_name, src in (("UIA", uia), ("OCR", ocr)):
                r = run(f"{rname} {src_name} {sname}", src, rows, fn, meta)
                if r["n"]:
                    out.append(r)
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for r in out:
            print(f"{r['label']:<28} {r['ok']:>3}/{r['n']:<3} ok  "
                  f"{r['wrong']} wrong  {r['declined']} declined  "
                  f"{r['ms_per_query']:.2f} ms/query")
            for fl in r["fails"]:
                print("     ", fl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
