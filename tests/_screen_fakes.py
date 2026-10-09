"""Fakes for the screen-vision tests (2026-10-05): a fake desktop of
browser windows showing the SYNTHETIC research pages (tests/_screen_pages.
json - UIA trees and OCR lines read from pages rendered in a throw-away
Chrome profile; no owner screen, no real UIA / OCR / pointer / model).

  FakeBackend   - every method core.grounded_click.Backend has, over fake
                  windows; clicks change the fake URL / title / windows the
                  way a browser would, so verify / undo are exercised for
                  real.
  page_snapshot - a core.screen_text.Snapshot of a fixture page placed in a
                  window rect.
  MONITORS      - a synthetic 4-monitor layout (left / middle / right / top).
"""
from __future__ import annotations

import json
import os
import threading
import time

from core.screen_scope import Win
from core.screen_text import El, Snapshot

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "_screen_pages.json"), encoding="utf-8") as _f:
    PAGES = json.load(_f)

MONITORS = {"left": (-2560, 0, 2560, 1440), "middle": (0, 0, 2560, 1440),
            "right": (2560, 0, 2560, 1440), "top": (0, -1440, 2560, 1440)}


def rect_on(monitor):
    return tuple(float(v) for v in MONITORS[monitor])


def page_elements(page, dx=0.0, dy=0.0):
    """(elements, doc_rect, url) of fixture ``page`` shifted by (dx, dy)."""
    pg = PAGES["uia"][page]
    doc = None
    for e in pg["elements"]:
        if e["type"] == "Document" and e.get("fw") == "Chrome":
            r = e["rect"]
            doc = (r[0] + dx, r[1] + dy, r[2], r[3])
            break
    els = []
    for i, e in enumerate(pg["elements"]):
        if e["type"] in ("Document", "Edit") or not e["name"].strip():
            continue
        if e.get("off"):
            continue
        r = e["rect"]
        rr = (r[0] + dx, r[1] + dy, r[2], r[3])
        in_doc = bool(doc and rr[1] >= doc[1] - 1 and rr[0] >= doc[0] - 1)
        els.append(El(name=" ".join(e["name"].split()), ctype=e["type"],
                      rect=rr, href=e.get("href", ""),
                      is_password=bool(e.get("pw")),
                      invokable=e["type"] in ("Hyperlink", "Button",
                                              "TabItem", "MenuItem"),
                      in_document=in_doc, ref=(page, i)))
    return els, doc, pg.get("url", "")


def page_snapshot(page, hwnd, rect, monitor, title=None, url=None,
                  process="chrome.exe"):
    els, doc, purl = page_elements(page, rect[0], rect[1])
    pg = PAGES["uia"][page]
    has_pw = any(e.get("pw") for e in pg["elements"])
    return Snapshot(hwnd=hwnd, title=title or pg["title"] + " - Google Chrome",
                    process=process, pid=1000 + hwnd, url=url or purl,
                    monitor=monitor, rect=tuple(rect), elements=tuple(els),
                    partial=doc is None, ms=12.0, doc_rect=doc,
                    has_password=has_pw, heavy=False, at=time.time())


class FakeWindow:
    def __init__(self, hwnd, page, monitor, title=None, url=None,
                 process="chrome.exe", jarvis=False, rect=None):
        self.hwnd = hwnd
        self.page = page
        self.monitor = monitor
        self.rect = tuple(rect) if rect else rect_on(monitor)
        pg = PAGES["uia"].get(page) if page else None
        base = pg["title"] if pg else (title or "Window")
        self.title = title or f"{base} - Google Chrome"
        self.url = url if url is not None else (pg or {}).get("url", "")
        self.process = process
        self.jarvis = jarvis
        self.history = []
        self.tabs = [base]
        self.alive = True
        self.elements_override = None

    def win(self, z=0):
        return Win(hwnd=self.hwnd, title=self.title, process=self.process,
                   pid=1000 + self.hwnd, rect=self.rect, monitor=self.monitor,
                   cls="Chrome_WidgetWin_1", z=z, jarvis=self.jarvis)


class FakeBackend:
    """Fake desktop. ``on_click``: "navigate" (a link opens its href in the
    same window), "new_window", "new_tab", "nothing", or "wrong" (opens a
    different page)."""

    def __init__(self, windows, *, fg=None, ledger=None, now_playing=None,
                 vision_answers=None, vision_ok=False, ocr_lines=None,
                 on_click="navigate"):
        self.wins = list(windows)
        self.fg = fg
        self.ledger_entry = ledger
        self.np = now_playing
        self.vision_answers = list(vision_answers or [])
        self.vision_ok = vision_ok
        self.ocr_lines = ocr_lines
        self.on_click = on_click
        self.clicks = []
        self.invokes = []
        self.snapshots = 0
        self.vision_calls = []
        self.captures = 0
        self.closed = []
        self.hotkeys = []
        self.focused = []
        self.backs = []
        self._tl = threading.local()
        self.auth_refusals = 0
        self.href_reads = 0
        self._next_hwnd = 9000
        self.closed_last_opened = 0
        self.lock = threading.Lock()

    # -- lookup ----------------------------------------------------------
    def _w(self, hwnd):
        return next((w for w in self.wins if w.hwnd == hwnd and w.alive), None)

    def windows(self, include_jarvis=False):
        out = []
        for z, w in enumerate(x for x in self.wins if x.alive):
            if w.jarvis and not include_jarvis:
                continue
            out.append(w.win(z))
        return out

    def foreground(self):
        return self.fg

    def ledger(self):
        return self.ledger_entry

    def snapshot(self, win, budget_ms=600):
        w = self._w(win.hwnd)
        if w is None or not w.page:
            return None
        self.snapshots += 1
        snap = page_snapshot(w.page, w.hwnd, w.rect, w.monitor, title=w.title,
                             url=w.url, process=w.process)
        if w.elements_override is not None:
            snap = snap._replace(elements=tuple(w.elements_override))
        return snap

    def read_url(self, hwnd):
        w = self._w(hwnd)
        return w.url if w else None

    def _elements(self, w):
        if not w.page:
            return []
        if w.elements_override is not None:
            return list(w.elements_override)
        return page_elements(w.page, w.rect[0], w.rect[1])[0]

    def root_at(self, x, y):
        for w in self.wins:
            if not w.alive:
                continue
            rx, ry, rw, rh = w.rect
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return w.hwnd
        return 0

    def element_at(self, x, y):
        root = self.root_at(x, y)
        w = self._w(root)
        if w is None:
            return None
        best = None
        for el in self._elements(w):
            ex, ey, ew, eh = el.rect
            if ex <= x <= ex + ew and ey <= y <= ey + eh:
                if best is None or ew * eh < best.rect[2] * best.rect[3]:
                    best = el
        if best is None:
            return {"name": "", "ctype": "Pane", "rect": w.rect,
                    "root_hwnd": root}
        return {"name": best.name, "ctype": best.ctype, "rect": best.rect,
                "root_hwnd": root}

    # -- effects ---------------------------------------------------------
    def _effect(self, w, el):
        mode = self.on_click
        if mode == "nothing" or el is None:
            return
        href = el.href or ""
        if mode == "new_window":
            self._next_hwnd += 1
            nw = FakeWindow(self._next_hwnd, None, w.monitor,
                            title=f"{el.name} - Google Chrome", url=href)
            self.wins.insert(0, nw)
            return
        if mode == "new_tab":
            w.tabs.append(el.name)
            return
        name = el.name
        if mode == "wrong":
            # a different page: its own address AND its own title (the
            # clicked label's title would read as the right page)
            href = "https://videosite.example/@ch-elsewhere"
            name = "Channel Elsewhere"
        if href:
            w.history.append((w.url, w.title, w.page))
            w.url = href
            w.title = f"{name} - VideoSite - Google Chrome"
            w.page = None if "watch" in href or "@" in href else w.page

    def click(self, x, y):
        with self.lock:
            self.clicks.append((int(x), int(y)))
            root = self.root_at(x, y)
            w = self._w(root)
            if w is None:
                return
            best = None
            for el in self._elements(w):
                ex, ey, ew, eh = el.rect
                if ex <= x <= ex + ew and ey <= y <= ey + eh:
                    if best is None or ew * eh < best.rect[2] * best.rect[3]:
                        best = el
            self._effect(w, best)

    def invoke(self, el):
        self.invokes.append(el.name if el is not None else None)
        if el is None:
            return False
        w = next((x for x in self.wins if x.alive and x.page
                  and any(e.ref == el.ref for e in self._elements(x))), None)
        if w is not None:
            self._effect(w, el)
        return True

    def scroll_into_view(self, el):
        return None

    def toggle_state(self, el):
        return (None, None)

    def tabs(self, hwnd):
        w = self._w(hwnd)
        return list(w.tabs) if w else None

    def back(self, hwnd):
        w = self._w(hwnd)
        self.backs.append(hwnd)
        if w is None or not w.history:
            return False
        w.url, w.title, w.page = w.history.pop()
        return True

    def close_tab(self, hwnd, name):
        w = self._w(hwnd)
        if w is None or name not in w.tabs:
            return False
        w.tabs.remove(name)
        return True

    def window_title(self, hwnd):
        w = self._w(hwnd)
        return w.title if w else ""

    def window_alive(self, hwnd):
        return self._w(hwnd) is not None

    def top_hwnds(self):
        return {w.hwnd for w in self.wins if w.alive}

    def close_window(self, hwnd):
        w = self._w(hwnd)
        if w is not None:
            w.alive = False
            self.closed.append(hwnd)
        return True

    def close_last_opened(self):
        self.closed_last_opened += 1
        if self.ledger_entry and self.ledger_entry[0]:
            w = self._w(self.ledger_entry[0])
            if w is not None:
                w.alive = False
                self.closed.append(w.hwnd)
                return f"closed the {w.title} window I opened"
        return "the window I opened is already closed"

    def focus(self, hwnd):
        self.focused.append(hwnd)
        return True

    def hotkey(self, *keys):
        self.hotkeys.append(keys)

    def now_playing(self):
        return self.np

    def capture(self, rect, target_hwnd=None, windows=None):
        self.captures += 1
        from PIL import Image
        return Image.new("RGB", (max(1, int(rect[2])), max(1, int(rect[3]))),
                         (40, 40, 40))

    def ocr(self, img):
        return self.ocr_lines

    def vision(self, prompt, png):
        self.vision_calls.append(prompt)
        return self.vision_answers.pop(0) if self.vision_answers else None

    def vision_usable(self):
        return self.vision_ok

    def legacy_find(self, desc, monitor):
        return None

    def is_self_close(self, desc):
        d = str(desc).lower()
        return "close" in d and ("powershell" in d or "terminal" in d)

    # The owner turn's frame is PER THREAD, like the monolith's
    # _turn_grounding (review 2026-10-05: this fake used to keep one shared
    # counter, so the 4-calls-per-turn cap "worked" here while production's
    # click worker thread saw no turn at all). A test that wants a turn sets
    # one: b.adopt_frame({"user_text": ...}).
    def turn_frame(self):
        return getattr(self._tl, "frame", None)

    def adopt_frame(self, frame):
        prev = getattr(self._tl, "frame", None)
        self._tl.frame = frame
        return prev

    def turn_vision(self, add=0):
        frame = self.turn_frame()
        if frame is None:
            return 0
        frame["vision_calls"] = int(frame.get("vision_calls", 0)) + add
        return frame["vision_calls"]

    def screen_texts(self):
        frame = self.turn_frame() or {}
        return list(frame.get("screen") or [])

    def auth_context(self):
        frame = self.turn_frame() or {}
        return {"owner_text": str(frame.get("user_text") or ""),
                "screen_texts": list(frame.get("screen") or []),
                "looked_for": list(frame.get("looked_for") or []),
                "refused_before": bool(frame.get("auth_refused"))}

    def note_auth_refused(self):
        self.auth_refusals += 1
        frame = self.turn_frame()
        if frame is not None:
            frame["auth_refused"] = True

    def href_of(self, el):
        self.href_reads += 1
        return getattr(el, "href", "") or ""

    def sleep(self, s):
        time.sleep(min(float(s), 0.01))
