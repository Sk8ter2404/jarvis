"""core/screen_digest.py - "what's on my screen" answered from the page's
own TEXT (2026-10-05).

WHY THIS EXISTS
===============
Live 00:28:01 JARVIS opened YouTube and was asked to look: see_screen sent
four 1024-px monitor shots to the local model, which answered "the YouTube
page displays several video thumbnails and categories" - not one title.
Every title, channel and view count was sitting in Chrome's accessibility
tree. This module reads them (UI Automation, OCR when UIA is thin) and
writes a compact digest the brain quotes from:

  [middle] Chrome - 'YouTube' (youtube.com) | headings: ... | videos:
  "Title" - Channel - 1.2M views; ...

Scope (core.actions.see_screen decides): a named monitor, the focused
window, the page JARVIS opened, or an overview (every monitor's window
inventory plus a digest of the top window). Private / excluded / sign-in /
password windows are listed as "(private)" and never read. Limits: 1,200
characters per window, 3,000 in all. Never raises.
"""
from __future__ import annotations

import re
import urllib.parse

from core import screen_privacy as _priv
from core import screen_resolve as _R

__all__ = ["digest", "window_digest", "inventory", "PER_WINDOW", "TOTAL",
           "VISUAL_Q_RE"]

PER_WINDOW = 1200
TOTAL = 3000
# A browser's tab strip + toolbar when UIA did not say where the page starts
# (core.grounded_click._BROWSER_TOOLBAR_PX).
_TOOLBAR_PX = 88
# A question about how things LOOK needs the vision model; anything else is
# a reading question.
VISUAL_Q_RE = re.compile(
    r"\b(?:colou?rs?|looks?\s+like|what\s+does\s+(?:it|this|that|the\s+\w+)"
    r"\s+look|pictures?|images?|photos?|charts?|graphs?|faces?|wearing|"
    r"layout|design|logos?|drawing|diagram|screenshot\s+of|how\s+does\s+it\s+"
    r"look|appearance|style|font|icons?)\b", re.IGNORECASE)
_VIEWS_RE = re.compile(r"[\d.,]+\s*[KMB]?\s+views?", re.IGNORECASE)
_BROWSER_NAMES = {"chrome.exe": "Chrome", "msedge.exe": "Edge",
                  "firefox.exe": "Firefox", "brave.exe": "Brave",
                  "opera.exe": "Opera", "vivaldi.exe": "Vivaldi"}


def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _host(url) -> str:
    try:
        s = str(url or "")
        if not s:
            return ""
        h = urllib.parse.urlsplit(s if "://" in s else "https://" + s).hostname
        return (h or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _app(win) -> str:
    p = str(_g(win, "process", "") or "").lower()
    if p in _BROWSER_NAMES:
        return _BROWSER_NAMES[p]
    return p[:-4] if p.endswith(".exe") else (p or "app")


def _page_title(title) -> str:
    t = re.sub(r"\s+-\s+(?:Google\s+Chrome|Microsoft\s*Edge|Brave|Opera|"
               r"Vivaldi|Mozilla\s+Firefox)\s*$", "", str(title or ""),
               flags=re.IGNORECASE)
    return t.strip()


def _clip(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def window_digest(win, snap=None, ocr_lines=None, url="",
                  limit: int = PER_WINDOW) -> str:
    """One window's digest line from its UIA snapshot (and/or OCR lines)."""
    try:
        mon = _g(win, "monitor") or "?"
        title = _page_title(_g(win, "title", ""))
        host = _host(url or (snap.url if snap is not None else ""))
        head = f"[{mon}] {_app(win)} — '{title}'" + (f" ({host})"
                                                          if host else "")
        parts = []
        if snap is not None:
            els = [e for e in snap.elements
                   if (e.in_document or snap.doc_rect is None)
                   and not e.is_password]
            cands = [{"text": e.name, "rect": list(e.rect), "type": e.ctype}
                     for e in els]
            prepared = _R.prepare(cands)
            grid = _R.video_cards(prepared)
            used = set()
            if grid:
                vids = []
                for unit in _R.reading_order(grid):
                    t = _R.card_target(unit)
                    lab = _R.label_of(t)
                    if not lab or lab in used:
                        continue
                    used.add(lab)
                    # The channel: the short line just UNDER the title (not
                    # the thumbnail's own link, not the views line).
                    below = sorted((c for c in unit if c is not t
                                    and c["rect"][1] > t["rect"][1]
                                    and c["rect"][3] <= 60
                                    and len(c["text"]) < 40
                                    and not _VIEWS_RE.search(c["text"])
                                    and not _R.DUR.match(c["text"])),
                                   key=lambda c: c["rect"][1])
                    ch = t.get("channel") or (below[0]["text"] if below
                                              else "")
                    views = next((m.group(0) for c in unit
                                  for m in [_VIEWS_RE.search(
                                      c.get("raw") or c["text"])] if m), "")
                    bits = [f'"{lab}"'] + [b for b in (ch, views) if b]
                    vids.append(" — ".join(bits))
                    for c in unit:
                        used.add(c["text"])
                if vids:
                    parts.append("videos: " + "; ".join(vids))
            heads = []
            for e in sorted(els, key=lambda e: (e.rect[1], e.rect[0])):
                if e.name in used:
                    continue
                if e.ctype == "Text" and 4 <= len(e.name) <= 120 \
                        and e.name not in heads:
                    heads.append(e.name)
                if len(heads) >= 8:
                    break
            if heads:
                parts.insert(0, "headings: " + " / ".join(heads))
            links = []
            for e in sorted(els, key=lambda e: (e.rect[1], e.rect[0])):
                if (e.ctype in ("Hyperlink", "Button", "TabItem")
                        and e.name not in used and e.name not in links
                        and len(e.name) >= 3):
                    links.append(e.name)
                if len(links) >= 25:
                    break
            if links:
                parts.append("links: " + ", ".join(links))
        if ocr_lines:
            texts = [" ".join(str(ln.get("t") or "").split())
                     for ln in ocr_lines]
            texts = [t for t in texts if len(t) >= 3]
            if texts:
                parts.append("text (OCR): " + " / ".join(texts[:40]))
        return _clip(head + (" | " + " | ".join(parts) if parts else ""),
                     limit)
    except Exception:
        return ""


def inventory(windows, url_of=None) -> str:
    """Every visible window per monitor: title, app, URL host. A browser
    window is judged with its address too (``url_of(hwnd)``: a bank tab
    titled "Accounts Overview" is "(a private window)" - review
    2026-10-05)."""
    by_mon: dict = {}
    for w in windows or ():
        by_mon.setdefault(_g(w, "monitor") or "?", []).append(w)
    lines = []
    for mon in sorted(by_mon):
        items = []
        for w in by_mon[mon][:6]:
            if _priv.live_private(w, url_of=url_of):
                items.append("(a private window)")
            else:
                items.append(f"'{_clip(_page_title(_g(w, 'title', '')), 70)}'"
                             f" [{_app(w)}]")
        lines.append(f"[{mon}] " + "; ".join(items))
    return "\n".join(lines)


def digest(scope="overview", *, said="", monitor=None, hwnd=None,
           backend=None, ocr_when_thin: bool = True) -> dict:
    """{"text": digest, "windows": [hwnd...], "private": n, "chars": n,
    "uia": bool} for a scope: "monitor" (``monitor``), "window" (``hwnd``)
    or "overview". "text" is "" when nothing could be read (the caller then
    falls back to vision). Never raises."""
    out = {"text": "", "windows": [], "private": 0, "chars": 0, "uia": False}
    try:
        if backend is None:
            from core.grounded_click import Backend
            backend = Backend()
        wins = list(backend.windows())
        if scope == "window":
            targets = [w for w in wins if _g(w, "hwnd") == hwnd]
        elif scope == "monitor":
            targets = [w for w in wins if _g(w, "monitor") == monitor][:2]
        else:
            targets = []
            seen = set()
            for w in sorted(wins, key=lambda w: _g(w, "z", 0)):
                if _g(w, "monitor") in seen:
                    continue
                seen.add(_g(w, "monitor"))
                targets.append(w)
            fg = backend.foreground()
            targets.sort(key=lambda w: (_g(w, "hwnd") != fg, _g(w, "z", 0)))
            targets = targets[:1]
        parts = []
        url_of = getattr(backend, "read_url", None)
        if scope == "overview":
            inv = inventory(wins, url_of=url_of)
            if inv:
                parts.append("Windows on screen:\n" + inv)
        total = sum(len(p) for p in parts)
        hidden = "(a private window - not read)"
        for w in targets:
            if total >= TOTAL:
                break
            # The address FIRST (review 2026-10-05): a page private by its
            # address alone - a bank tab titled "Accounts Overview" - is
            # never read, OCR'd or quoted.
            browser = _priv.is_browser_process(_g(w, "process"))
            url0 = ""
            if browser and callable(url_of):
                try:
                    url0 = url_of(_g(w, "hwnd")) or ""
                except Exception:
                    url0 = ""
            if _priv.live_private(w, url=url0 or None):
                out["private"] += 1
                parts.append(f"[{_g(w, 'monitor')}] {hidden}")
                continue
            snap = backend.snapshot(w)
            url = (snap.url if snap is not None else "") or url0
            if snap is not None and (snap.has_password or _priv.live_private(
                    {"hwnd": _g(w, "hwnd"), "title": _g(w, "title"),
                     "process": _g(w, "process"), "url": url},
                    url=url or None)):
                out["private"] += 1
                parts.append(f"[{_g(w, 'monitor')}] {hidden}")
                continue
            if snap is not None:
                out["uia"] = True
            text_chars = (sum(len(e.name) for e in snap.elements)
                          if snap is not None else 0)
            ocr_lines = None
            if ocr_when_thin and text_chars < 150:
                img = backend.capture(_g(w, "rect"), target_hwnd=_g(w, "hwnd"))
                if img is not None:
                    ocr_lines = backend.ocr(img)
                # A browser whose address is still unknown: its address bar,
                # as OCR read it, decides (the click path's rule).
                if ocr_lines and browser and not url:
                    top = (snap.doc_rect[1] - _g(w, "rect")[1]
                           if snap is not None and snap.doc_rect
                           else _TOOLBAR_PX)
                    if _priv.address_bar_private(ocr_lines, top):
                        out["private"] += 1
                        parts.append(f"[{_g(w, 'monitor')}] {hidden}")
                        continue
            line = window_digest(w, snap, ocr_lines, url,
                                 limit=min(PER_WINDOW, TOTAL - total))
            if line:
                parts.append(line)
                total += len(line)
                out["windows"].append(_g(w, "hwnd"))
        text = "\n".join(parts)
        out["text"] = _clip(text, TOTAL)
        out["chars"] = len(out["text"])
        return out
    except Exception:
        return out
