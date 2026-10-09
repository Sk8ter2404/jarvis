"""tools/vision_bench/live_probe.py - runs INSIDE the hidden bench desktop
that tools/vision_bench/live_uia_bench.py creates (never on the owner's).

It drives the SHIPPED code - core.screen_text (UI Automation), core.
screen_scope, core.grounded_click, core.screen_digest - against our own
throw-away Chrome showing synthetic pages, and records timings and
accuracy. Rules it keeps:

  * windows are only those of the Chrome PIDs passed in (the enumerator is
    pinned to them), so no other window's content is ever read;
  * "clicks" are UI Automation Invoke on the element under the point -
    never mouse or keyboard input (the hidden desktop gets no input, and
    the owner's desktop must never get any);
  * no screenshots, no OCR, no model calls (capture / vision are off);
  * page navigation and the ground-truth URL come from Chrome's DevTools
    protocol on 127.0.0.1.

    python live_probe.py <out.json> <pages_dir> <cdp_port> <pid,pid,...>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.request

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

GT_JS = r"""
() => Array.from(document.querySelectorAll('[data-gt]')).map(el => {
  const r = el.getBoundingClientRect();
  return {kind: el.dataset.gt, id: el.dataset.id,
          text: (el.innerText || el.value || el.placeholder || '').trim(),
          rect: [r.left, r.top, r.width, r.height]};
})
"""

CLICK_CASES = [
    # (page, utterance, expected watch page or None, expected outcome)
    ("fresh2_light", "click that MrBeast video", "watch_f04.html", "verified"),
    ("fresh2_light", "click the drone video", "watch_f01.html", "verified"),
    ("fresh2_light", "click the fifth video", "watch_f05.html", "verified"),
    ("fresh2_light", "play the chess one", "watch_f10.html", "verified"),
    ("fresh2_light", "click the kai senat video", "watch_f13.html", "verified"),
    ("home_dark", "click that Mr. Beast video", "watch_v01.html", "verified"),
    ("holdout_light", "click the IShowSpeed video", "watch_h03.html", "verified"),
    ("fresh2_light", "click the dragon video", None, "not_found"),
    ("watch_dark", "click the one that's playing", None, "not_found"),
    ("chooser_light", "click the test user account", None, "refused_auth"),
    ("signin_light", "click next", None, "refused_auth"),
]


PRIVATE_PAGES = ("chooser_light", "signin_light")


class _CDP:
    def __init__(self, port):
        import websocket
        self.port = port
        tg = []
        for _ in range(80):
            try:
                tg = [t for t in json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json", timeout=1))
                    if t.get("type") == "page"]
                if tg:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not tg:
            raise RuntimeError("no DevTools page target")
        # No Origin header: Chrome then needs no --remote-allow-origins, so
        # no web page anywhere can drive this browser.
        self.ws = websocket.create_connection(tg[0]["webSocketDebuggerUrl"],
                                              timeout=20,
                                              suppress_origin=True)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method,
                                 "params": params or {}}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self._id:
                return m.get("result", m)

    def eval(self, expr):
        r = self.call("Runtime.evaluate", {"expression": expr,
                                           "returnByValue": True})
        return (r.get("result") or {}).get("value")

    def goto(self, url, settle=0.8):
        self.call("Page.navigate", {"url": url})
        t0 = time.time()
        while time.time() - t0 < 10:
            if self.eval("document.readyState") == "complete" and \
                    self.eval("location.href") == url:
                break
            time.sleep(0.1)
        time.sleep(settle)

    def href(self):
        return self.eval("location.href") or ""


def _file_url(pages_dir, name):
    return "file:///" + os.path.join(pages_dir, name).replace("\\", "/")


def main():
    out_path, pages_dir, port, pid_arg = sys.argv[1:5]
    pids = {int(x) for x in pid_arg.split(",") if x}
    res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pages": {},
           "queries": [], "clicks": [], "captures": {}, "errors": []}
    cpu0 = time.process_time()
    t_start = time.perf_counter()
    try:
        _run(res, pages_dir, int(port), pids)
    except Exception:
        res["errors"].append(traceback.format_exc()[-2000:])
    res["probe_cpu_s"] = round(time.process_time() - cpu0, 3)
    res["probe_wall_s"] = round(time.perf_counter() - t_start, 2)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)


def _run(res, pages_dir, port, pids):
    from core import screen_privacy as priv
    if priv.reads_blocked():
        raise RuntimeError("JARVIS_NO_SCREEN_READ / JARVIS_TEST_MODE is set")
    from core import grounded_click as G
    from core import screen_digest as D
    from core import screen_resolve as R
    from core import screen_scope as S
    from core import onscreen_refs as REFS
    from core import screen_text as T
    from core import uia_host
    from tools.vision_bench import pages as P
    from tools.vision_bench import queries as Q

    def ours():
        return [w for w in S._win32_windows() if w.pid in pids]
    S.set_enumerator(ours)

    class BenchBackend(G.Backend):
        """The production backend, minus everything that could reach past
        our own hidden Chrome."""
        invokes = []

        def windows(self, include_jarvis=False):
            return S.visible_windows(include_jarvis=include_jarvis)

        def foreground(self):
            w = self.windows()
            return w[0].hwnd if w else None

        def ledger(self):
            return None

        def top_hwnds(self):
            return {w.hwnd for w in self.windows(True)}

        def click(self, x, y):
            """What is under the point gets PRESSED through UI Automation
            (core.screen_text._press) - the stand-in for a mouse click,
            which must never be sent."""
            def job(uia):
                from comtypes.gen import UIAutomationClient as UIA
                e = uia.ElementFromPoint(UIA.tagPOINT(int(x), int(y)))
                walker = uia.ControlViewWalker
                for _ in range(6):
                    if T._nil(e):
                        return None
                    how = T._press(e)
                    if how:
                        return f"{e.CurrentName} ({how})"
                    e = walker.GetParentElement(e)
                return None
            ok, name = uia_host.call(job, timeout_s=1.0)
            self.invokes.append((int(x), int(y), name if ok else None))
            if not ok or name is None:
                raise RuntimeError("bench: nothing pressable at the point")

        def hotkey(self, *keys):
            raise RuntimeError("bench: no keyboard input")

        def focus(self, hwnd):
            return True

        def close_window(self, hwnd):
            return False

        def close_last_opened(self):
            return "nothing to close"

        def now_playing(self):
            return None

        def capture(self, rect, target_hwnd=None, windows=None):
            return None

        def ocr(self, img):
            return None

        def vision(self, prompt, png):
            return None

        def vision_usable(self):
            return False

        def legacy_find(self, desc, monitor):
            return None

        def is_self_close(self, desc):
            return False

        def turn_vision(self, add=0):
            return 0

        def screen_texts(self):
            return []

    b = BenchBackend()
    cdp = _CDP(port)
    meta = P.meta_dict()

    # UIA host start-up (first COM init + type-library generation).
    t0 = time.perf_counter()
    res["uia_host_ready"] = uia_host.available(wait_s=20.0)
    res["uia_host_start_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    wins = []
    for _ in range(40):
        wins = b.windows()
        if wins:
            break
        time.sleep(0.25)
    res["windows"] = [{"title": w.title, "process": w.process,
                       "monitor": w.monitor, "rect": list(w.rect)}
                      for w in wins]
    if not wins:
        raise RuntimeError("our Chrome window was not found")

    rows_by_page = {}
    for sname, rows in Q.SETS.items():
        for page, q, exp in rows:
            rows_by_page.setdefault(page, []).append((sname, q, exp))

    for page in ("fresh2_light", "home_dark", "watch_dark", "holdout_light",
                 "console_light", "chooser_light", "signin_light"):
        cdp.goto(_file_url(pages_dir, page + ".html"))
        w = b.windows()[0]
        pr = {"title": w.title}
        snaps = []
        for _ in range(3):
            s = b.snapshot(w)
            snaps.append(s)
        s = snaps[-1]
        pr["snapshot_ms"] = [x.ms if x else None for x in snaps]
        if s is None:
            pr["error"] = "snapshot failed"
            res["pages"][page] = pr
            continue
        pr["elements"] = len(s.elements)
        pr["has_document"] = s.doc_rect is not None
        pr["heavy"] = s.heavy
        pr["hrefs"] = sum(1 for e in s.elements if e.href)
        pr["url_from_snapshot_ok"] = bool(s.url) and \
            s.url.replace("\\", "/").lower().endswith(page + ".html")
        t0 = time.perf_counter()
        u = None
        for _ in range(20):
            u = b.read_url(w.hwnd)
        pr["read_url_ms"] = round((time.perf_counter() - t0) * 1000 / 20, 2)
        pr["read_url_ok"] = bool(u) and u.lower().endswith(page + ".html")
        titles = [e for e in s.elements if e.ctype == "Hyperlink"
                  and e.in_document and len(e.name) > 20]
        if titles:
            e0 = titles[0]
            t0 = time.perf_counter()
            ea = b.element_at(e0.rect[0] + e0.rect[2] / 2,
                              e0.rect[1] + e0.rect[3] / 2)
            pr["element_at_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            pr["element_at_ok"] = bool(ea) and R.squash(ea.get("name")) == \
                R.squash(e0.name)
        gt = cdp.eval(f"({GT_JS})()") or []
        names = {R.squash(e.name) for e in s.elements}
        want = [g for g in gt if g["kind"] == "title" and g["text"]]
        pr["gt_titles_in_uia"] = (sum(1 for g in want
                                      if R.squash(g["text"]) in names),
                                  len(want))
        t0 = time.perf_counter()
        dg = D.digest("window", hwnd=w.hwnd, backend=b, ocr_when_thin=False)
        pr["digest_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        pr["digest_chars"] = dg.get("chars")
        pr["digest_private"] = dg.get("private")
        pr["digest_titles"] = sum(1 for g in want
                                  if g["text"][:30] in (dg.get("text") or ""))
        res["pages"][page] = pr
        if page == "fresh2_light":
            doc = s.doc_rect
            res["captures"][page] = {
                "title": "Home - VideoSite",
                "elements": ([{"name": "doc", "type": "Document",
                               "rect": list(doc), "fw": "Chrome"}]
                             if doc else []) + [
                    {"name": e.name, "type": e.ctype, "rect": list(e.rect),
                     "fw": "Chrome", "pw": e.is_password,
                     "off": e.offscreen} for e in s.elements]}
        # find-mode accuracy (end to end through grounded_click.run)
        origin = (s.doc_rect[0], s.doc_rect[1]) if s.doc_rect else (0, 0)
        for sname, q, exp in rows_by_page.get(page, ()):
            G.reset_state()
            t0 = time.perf_counter()
            r = G.run(q, said=q, mode="find", backend=b)
            ms = (time.perf_counter() - t0) * 1000
            # the same decision straight from the resolver, for the rect
            direct = R.resolve(q, G._cands_from_snapshot(s),
                               playing_title=None)
            rect = (direct.get("target") or {}).get("rect") or (0, 0, 0, 0)
            if r.outcome == G.FOUND:
                outcome = "ok"
            elif r.outcome == G.AMBIGUOUS:
                outcome = "ambiguous"
            else:
                outcome = r.outcome
            import re as _re
            opts = _re.findall(r"'(.*?)' on the ", r.text or "")
            v = Q.judge(outcome, r.label, rect, opts, exp, page, meta, origin)
            if page in PRIVATE_PAGES:
                # sign-in pages are never read (by design): right = nothing
                v = "wrong" if r.outcome == G.FOUND else "ok"
                sname = sname + "-private"
            res["queries"].append({"set": sname, "page": page, "query": q,
                                   "expected": str(exp), "outcome": r.outcome,
                                   "label": r.label[:70], "verdict": v,
                                   "ms": round(ms, 1)})

    # clicks: grounded_click.run_bounded(mode="click") -> verify -> undo
    for page, q, want_page, want in CLICK_CASES:
        home = _file_url(pages_dir, page + ".html")
        cdp.goto(home)
        G.reset_state()
        b.invokes.clear()
        t0 = time.perf_counter()
        # the argument the built-in click route passes (core.dispatcher)
        arg = REFS.onscreen_click_target(q) or q
        r = G.run_bounded(arg, said=q, mode="click", backend=b)
        ms = (time.perf_counter() - t0) * 1000
        time.sleep(0.4)
        after = cdp.href()
        row = {"page": page, "utterance": q, "arg": arg,
               "outcome": r.outcome, "pressed": [i[2] for i in b.invokes],
               "text": r.text[:160], "ms": round(ms, 1),
               "invokes": len(b.invokes),
               "landed": after.rsplit("/", 1)[-1],
               "expected_landing": want_page or page + ".html",
               "expected_outcome": want}
        row["landed_ok"] = row["landed"] == row["expected_landing"]
        row["outcome_ok"] = r.outcome == want
        if want == "verified" and r.outcome == G.VERIFIED:
            t0 = time.perf_counter()
            u = G.undo(False, said="go back", backend=b)
            row["undo_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            time.sleep(0.4)
            row["undo_outcome"] = u.outcome
            row["undo_text"] = u.text[:120]
            row["back_ok"] = cdp.href() == home
        res["clicks"].append(row)

    # "not that one": a click, then a correction that asks from the scene
    page = "fresh2_light"
    home = _file_url(pages_dir, page + ".html")
    cdp.goto(home)
    G.reset_state()
    G.freeze_scene("click the drone video", backend=b)
    r = G.run_bounded("click the drone video", said="click the drone video",
                      mode="click", backend=b)
    time.sleep(0.3)
    t0 = time.perf_counter()
    u = G.undo(True, said="not that one", backend=b)
    res["not_that_one"] = {
        "click": r.outcome, "undo_outcome": u.outcome, "text": u.text[:200],
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "back_ok": (time.sleep(0.4) or cdp.href()) == home,
        "pending": bool(G.pending_choice())}
    if G.pending_choice():
        t0 = time.perf_counter()
        p = G.pick(1, said="the first one", backend=b)
        time.sleep(0.4)
        res["not_that_one"]["pick"] = p.outcome
        res["not_that_one"]["pick_ms"] = round(
            (time.perf_counter() - t0) * 1000, 1)
        res["not_that_one"]["pick_landed"] = cdp.href().rsplit("/", 1)[-1]
    res["uia_status"] = uia_host.status()
    cdp.ws.close()


if __name__ == "__main__":
    main()
