"""core/screen_text.py - read a window's links, buttons and text by NAME
through Windows UI Automation (2026-10-05).

WHY THIS EXISTS
===============
Every link on the YouTube page JARVIS opened at 00:27:57 had its exact
title and rectangle sitting in Chrome's accessibility tree; JARVIS instead
photographed four monitors and asked a vision model, which answered "the
YouTube page displays several video thumbnails and categories". One
cached UIA query returns them all in 20-55 ms on a normal page (research
2026-10-05, synthetic pages in a throw-away Chrome profile).

Rules:
  * every call runs on core.uia_host (one thread, bounded waits);
  * EDIT VALUES ARE NEVER READ - the one exception is Chrome's address bar
    ("Address and search bar"), for the page's URL. A password box is only
    noted (IsPassword), never read;
  * element pointers never leave the host thread: an ``El`` carries a
    handle (snapshot id, index) the host resolves for invoke / scroll;
  * a page that answers with browser chrome only (Chrome turns web
    accessibility on lazily: 36 -> 132 elements) is read again once after
    300 ms; a window whose read took > 300 ms or > 1,000 elements is marked
    HEAVY for 60 s (the OCR path reads it instead);
  * elements are clipped to the window, IsOffscreen ones dropped, and one
    whose centre is covered by another window is marked ``occluded``;
  * PRESSING (invoke / back_button / close_tab) uses the element's
    LegacyIAccessible default action first: Chrome answers a UIA
    InvokePattern.Invoke on a link or its Back button only after ~2 s
    (live bench, tools/vision_bench/live_uia_bench.py, 2026-10-05) while
    DoDefaultAction returns in ~1 ms. A press that outlives its wait
    returns None ("sent, may still land") - never False - so no caller
    fires a fallback keystroke on top of it (a double Back / a second
    closed tab);
  * a missing pattern / element comes back from comtypes as a NULL
    pointer, not None - every check uses _nil().

Never raises at the public API.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import NamedTuple, Optional

from core import uia_host

__all__ = ["El", "Snapshot", "snapshot", "read_url", "element_at",
           "invoke", "scroll_into_view", "back_button", "tab_items",
           "close_tab", "href_of", "is_heavy", "toggle_state", "CT_NAMES"]

# UIA property ids
P_RECT, P_CTYPE, P_NAME = 30001, 30003, 30005
P_HASFOCUS, P_PASSWORD, P_OFFSCREEN = 30008, 30019, 30022
P_FRAMEWORK, P_INVOKE_AVAIL, P_VALUE = 30024, 30031, 30045
P_TOGGLE_STATE, P_SELECTED = 30086, 30079
P_HEADING = 30173
# pattern ids
PAT_INVOKE, PAT_VALUE, PAT_SELITEM = 10000, 10002, 10010
PAT_TOGGLE, PAT_SCROLLITEM, PAT_LEGACY = 10015, 10017, 10018
TS_DESCENDANTS, TS_CHILDREN = 4, 2

CT_NAMES = {
    50000: "Button", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit",
    50005: "Hyperlink", 50006: "Image", 50007: "ListItem", 50008: "List",
    50011: "MenuItem", 50013: "RadioButton", 50018: "Tab", 50019: "TabItem",
    50020: "Text", 50021: "ToolBar", 50024: "TreeItem", 50025: "Custom",
    50026: "Group", 50029: "DataItem", 50030: "Document", 50031: "SplitButton",
    50032: "Window", 50033: "Pane",
}
_WANTED = (50005, 50000, 50007, 50019, 50011, 50020, 50006, 50002, 50013,
           50003, 50024, 50029, 50031, 50030, 50004)
_ADDRESS_BAR_NAMES = ("Address and search bar", "Search or enter address",
                      "Search or enter web address")
_HEAVY_MS = 300
_HEAVY_ELEMENTS = 1000
_HEAVY_TTL_S = 60.0
_KEEP_SNAPSHOTS = 12


class El(NamedTuple):
    name: str
    ctype: str
    rect: tuple                # (x, y, w, h) screen px, clipped to the window
    href: str = ""
    is_password: bool = False
    offscreen: bool = False
    occluded: bool = False
    invokable: bool = False
    heading_level: int = 0
    in_document: bool = True   # inside the page (not the browser's toolbar)
    ref: tuple = ()            # (snapshot id, index) - host-side handle


class Snapshot(NamedTuple):
    hwnd: int
    title: str
    process: str
    pid: int
    url: str
    monitor: Optional[str]
    rect: tuple
    elements: tuple
    partial: bool
    ms: float
    doc_rect: Optional[tuple] = None
    has_password: bool = False
    heavy: bool = False
    at: float = 0.0


# Host-side state (touched only inside uia_host jobs).
_snaps: "collections.OrderedDict" = collections.OrderedDict()
_omni: dict = {}
_seq = [0]
_heavy: dict = {}
_heavy_lock = threading.Lock()


def is_heavy(hwnd) -> bool:
    with _heavy_lock:
        until = _heavy.get(hwnd)
        if until is None:
            return False
        if until < time.time():
            _heavy.pop(hwnd, None)
            return False
        return True


def _mark_heavy(hwnd) -> None:
    with _heavy_lock:
        _heavy[hwnd] = time.time() + _HEAVY_TTL_S


def _rect_of(r) -> Optional[tuple]:
    try:
        if r is None:
            return None
        if hasattr(r, "left"):
            x, y = float(r.left), float(r.top)
            return (x, y, float(r.right) - x, float(r.bottom) - y)
        vals = [float(v) for v in list(r)[:4]]
        if len(vals) == 4:
            return tuple(vals)
    except Exception:
        return None
    return None


def _clip(r, w):
    x0, y0 = max(r[0], w[0]), max(r[1], w[1])
    x1 = min(r[0] + r[2], w[0] + w[2])
    y1 = min(r[1] + r[3], w[1] + w[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _cache_request(uia):
    cr = uia.CreateCacheRequest()
    for pid in (P_NAME, P_CTYPE, P_RECT, P_PASSWORD, P_OFFSCREEN, P_FRAMEWORK,
                P_INVOKE_AVAIL):
        cr.AddProperty(pid)
    return cr


def _condition(uia):
    conds = [uia.CreatePropertyCondition(P_CTYPE, ct) for ct in _WANTED]
    c = conds[0]
    for nxt in conds[1:]:
        c = uia.CreateOrCondition(c, nxt)
    return c


def _win32():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.GetAncestor.restype = wintypes.HWND
    return ctypes, wintypes, user32


def _root_at(x, y) -> int:
    try:
        ctypes, wintypes, user32 = _win32()
        h = user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
        r = user32.GetAncestor(h, 2) if h else None          # GA_ROOT
        return int(r or 0)
    except Exception:
        return 0


def _cached(e, pid):
    try:
        return e.GetCachedPropertyValue(pid)
    except Exception:
        return None


def snapshot(hwnd, *, win_rect=None, title="", process="", pid=0,
             monitor=None, budget_ms: int = 600, want_hrefs: bool = True,
             retry_chrome: bool = True) -> Optional[Snapshot]:
    """The window's readable elements (see module docstring), or None when
    UIA is unavailable / the read failed / timed out. Never raises."""
    try:
        if not isinstance(hwnd, int) or hwnd <= 0:
            return None
        t0 = time.perf_counter()
        res = _snapshot_once(hwnd, win_rect, title, process, pid, monitor,
                             budget_ms, want_hrefs)
        if (res is not None and retry_chrome and res.doc_rect is None
                and str(process).lower() in ("chrome.exe", "msedge.exe")):
            time.sleep(0.3)
            again = _snapshot_once(hwnd, win_rect, title, process, pid,
                                   monitor, budget_ms, want_hrefs)
            if again is not None:
                res = again
        if res is not None:
            total = (time.perf_counter() - t0) * 1000
            res = res._replace(ms=round(total, 1))
            if res.heavy:
                _mark_heavy(hwnd)
        return res
    except Exception:
        return None


def _snapshot_once(hwnd, win_rect, title, process, pid, monitor, budget_ms,
                   want_hrefs) -> Optional[Snapshot]:
    def job(uia):
        t0 = time.perf_counter()
        root = uia.ElementFromHandle(hwnd)
        arr = root.FindAllBuildCache(TS_DESCENDANTS, _condition(uia),
                                     _cache_request(uia))
        n = int(getattr(arr, "Length", 0) or 0)
        wr = win_rect
        if wr is None:
            try:
                wr = _rect_of(root.CurrentBoundingRectangle)
            except Exception:
                wr = None
        _seq[0] += 1
        sid = _seq[0]
        keep = []
        doc_rect = None
        url = ""
        has_pw = False
        items = []
        for i in range(n):
            try:
                e = arr.GetElement(i)
            except Exception:
                continue
            try:
                ct = int(_cached(e, P_CTYPE) or 0)
                name = str(_cached(e, P_NAME) or "")
                r = _rect_of(_cached(e, P_RECT))
                pw = bool(_cached(e, P_PASSWORD))
                off = bool(_cached(e, P_OFFSCREEN))
                inv = bool(_cached(e, P_INVOKE_AVAIL))
            except Exception:
                continue
            if ct == 50030 and doc_rect is None and r and r[2] > 50:
                doc_rect = r
                continue
            if ct == 50004:                     # Edit: never its value...
                if pw:
                    has_pw = True
                if name in _ADDRESS_BAR_NAMES and not url:
                    _omni[hwnd] = e             # ...except the address bar
                    url = _value_of(uia, e)
                continue
            if off or r is None or not name.strip():
                continue
            if wr is not None:
                r = _clip(r, wr)
                if r is None:
                    continue
            items.append((e, ct, name, r, pw, inv))
        els = []
        for idx, (e, ct, name, r, pw, inv) in enumerate(items):
            in_doc = bool(doc_rect and r[1] >= doc_rect[1] - 1
                          and r[0] >= doc_rect[0] - 1)
            href = ""
            if (want_hrefs and ct == 50005 and in_doc
                    and len(name) >= 12):
                href = _value_of(uia, e)
            keep.append(e)
            els.append(El(name=" ".join(name.split()),
                          ctype=CT_NAMES.get(ct, str(ct)), rect=r,
                          href=href, is_password=pw, invokable=inv,
                          in_document=in_doc, ref=(sid, len(keep) - 1)))
        _snaps[sid] = keep
        while len(_snaps) > _KEEP_SNAPSHOTS:
            _snaps.popitem(last=False)
        ms = (time.perf_counter() - t0) * 1000
        return els, doc_rect, url, has_pw, n, ms, wr
    ok, val = uia_host.call(job, timeout_s=max(0.2, budget_ms / 1000.0))
    if not ok:
        return None
    els, doc_rect, url, has_pw, n, ms, wr = val
    # Occlusion (outside the host: plain Win32, no COM).
    marked = []
    for el in els:
        cx, cy = el.rect[0] + el.rect[2] / 2, el.rect[1] + el.rect[3] / 2
        root = _root_at(cx, cy)
        marked.append(el._replace(occluded=bool(root and root != hwnd)))
    return Snapshot(hwnd=hwnd, title=str(title or ""),
                    process=str(process or ""), pid=int(pid or 0),
                    url=str(url or ""), monitor=monitor,
                    rect=tuple(wr) if wr else (),
                    elements=tuple(marked), partial=doc_rect is None,
                    ms=round(ms, 1), doc_rect=doc_rect, has_password=has_pw,
                    heavy=(ms > _HEAVY_MS or n > _HEAVY_ELEMENTS),
                    at=time.time())


def _nil(p) -> bool:
    """None, or a NULL COM pointer (comtypes returns those for a missing
    pattern / element - they are not None, only falsy)."""
    try:
        return p is None or not bool(p)
    except Exception:
        return True


def _press(e) -> str:
    """Press element ``e`` (host thread only): LegacyIAccessible
    DoDefaultAction when it has a default action (fast everywhere), else
    InvokePattern.Invoke. 'legacy' / 'invoke', or '' when it has neither."""
    from comtypes.gen import UIAutomationClient as UIA
    try:
        p = e.GetCurrentPattern(PAT_LEGACY)
        if not _nil(p):
            lp = p.QueryInterface(UIA.IUIAutomationLegacyIAccessiblePattern)
            if str(lp.CurrentDefaultAction or "").strip():
                lp.DoDefaultAction()
                return "legacy"
    except Exception:
        pass
    p = e.GetCurrentPattern(PAT_INVOKE)
    if not _nil(p):
        p.QueryInterface(UIA.IUIAutomationInvokePattern).Invoke()
        return "invoke"
    return ""


def _pressed(ok, val):
    """True pressed / False not pressed / None sent but unconfirmed (the
    wait ran out while the press was still running - it may land)."""
    if ok:
        return bool(val)
    return None if val == "timeout" else False


def _value_of(uia, e) -> str:
    try:
        from comtypes.gen import UIAutomationClient as UIA
        pat = e.GetCurrentPattern(PAT_VALUE)
        if _nil(pat):
            return ""
        vp = pat.QueryInterface(UIA.IUIAutomationValuePattern)
        return str(vp.CurrentValue or "")
    except Exception:
        try:
            return str(e.GetCurrentPropertyValue(P_VALUE) or "")
        except Exception:
            return ""


def read_url(hwnd, timeout_s: float = 0.5) -> Optional[str]:
    """The browser window's address-bar text (7 ms the first time, ~0.1 ms
    after), or None. Never raises."""
    def job(uia):
        e = _omni.get(hwnd)
        if not _nil(e):
            v = _value_of(uia, e)
            if v:
                return v
        root = uia.ElementFromHandle(hwnd)
        for nm in _ADDRESS_BAR_NAMES:
            cond = uia.CreateAndCondition(
                uia.CreatePropertyCondition(P_CTYPE, 50004),
                uia.CreatePropertyCondition(P_NAME, nm))
            e = root.FindFirst(TS_DESCENDANTS, cond)
            if not _nil(e):
                _omni[hwnd] = e
                return _value_of(uia, e)
        return ""
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return val if ok and isinstance(val, str) else None
    except Exception:
        return None


def _resolve(ref):
    try:
        sid, idx = ref
        return _snaps[sid][idx]
    except Exception:
        return None


def element_at(x, y, timeout_s: float = 0.3):
    """{name, ctype, rect, root_hwnd, invokable} of the element at a screen
    point (2-10 ms), or None. Never raises."""
    def job(uia):
        import ctypes
        from ctypes import wintypes

        class _PT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]
        try:
            from comtypes.gen import UIAutomationClient as UIA
            pt = UIA.tagPOINT(int(x), int(y))
        except Exception:
            pt = _PT(int(x), int(y))
        e = uia.ElementFromPoint(pt)
        if _nil(e):
            return None
        try:
            ct = int(e.CurrentControlType)
        except Exception:
            ct = 0
        return {"name": " ".join(str(e.CurrentName or "").split()),
                "ctype": CT_NAMES.get(ct, str(ct)),
                "rect": _rect_of(e.CurrentBoundingRectangle)}
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        if not ok or not val:
            return None
        val["root_hwnd"] = _root_at(x, y)
        return val
    except Exception:
        return None


def invoke(el, timeout_s: float = 0.8) -> Optional[bool]:
    """Press a snapshot element (_press). True pressed, False could not,
    None sent but unconfirmed (see the module docstring)."""
    def job(uia):
        e = _resolve(el.ref)
        if _nil(e):
            return False
        return bool(_press(e))
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return _pressed(ok, val)
    except Exception:
        return False


def scroll_into_view(el, timeout_s: float = 0.8) -> Optional[tuple]:
    """ScrollItemPattern.ScrollIntoView, then the element's NEW rect."""
    def job(uia):
        from comtypes.gen import UIAutomationClient as UIA
        e = _resolve(el.ref)
        if _nil(e):
            return None
        p = e.GetCurrentPattern(PAT_SCROLLITEM)
        if _nil(p):
            return None
        p.QueryInterface(UIA.IUIAutomationScrollItemPattern).ScrollIntoView()
        time.sleep(0.15)
        return _rect_of(e.CurrentBoundingRectangle)
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return val if ok else None
    except Exception:
        return None


def toggle_state(el, timeout_s: float = 0.3):
    """(toggle state, selected) of a snapshot element, None parts unknown."""
    def job(uia):
        e = _resolve(el.ref)
        if _nil(e):
            return (None, None)
        t = s = None
        try:
            t = e.GetCurrentPropertyValue(P_TOGGLE_STATE)
        except Exception:
            pass
        try:
            s = e.GetCurrentPropertyValue(P_SELECTED)
        except Exception:
            pass
        return (t, s)
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return val if ok else (None, None)
    except Exception:
        return (None, None)


def href_of(el, timeout_s: float = 0.3) -> str:
    def job(uia):
        e = _resolve(el.ref)
        return "" if _nil(e) else _value_of(uia, e)
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return val if ok and isinstance(val, str) else ""
    except Exception:
        return ""


def _toolbar_button(uia, hwnd, names):
    root = uia.ElementFromHandle(hwnd)
    for nm in names:
        cond = uia.CreateAndCondition(
            uia.CreatePropertyCondition(P_CTYPE, 50000),
            uia.CreatePropertyCondition(P_NAME, nm))
        e = root.FindFirst(TS_DESCENDANTS, cond)
        if not _nil(e):
            return e
    return None


def back_button(hwnd, timeout_s: float = 0.8) -> Optional[bool]:
    """Press the browser toolbar's "Back" button in that window (no
    keystroke, no focus change). True / False / None as invoke()."""
    def job(uia):
        e = _toolbar_button(uia, hwnd, ("Back", "Click to go back, hold to "
                                                "see history"))
        if _nil(e):
            return False
        return bool(_press(e))
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return _pressed(ok, val)
    except Exception:
        return False


def tab_items(hwnd, timeout_s: float = 0.5) -> Optional[list]:
    """The browser window's tab titles (TabItems), or None."""
    def job(uia):
        root = uia.ElementFromHandle(hwnd)
        arr = root.FindAll(TS_DESCENDANTS,
                           uia.CreatePropertyCondition(P_CTYPE, 50019))
        out = []
        for i in range(int(getattr(arr, "Length", 0) or 0)):
            try:
                out.append(str(arr.GetElement(i).CurrentName or ""))
            except Exception:
                continue
        return out
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return list(val) if ok and isinstance(val, list) else None
    except Exception:
        return None


def close_tab(hwnd, tab_name, timeout_s: float = 0.8) -> Optional[bool]:
    """Press the "Close" button inside the TabItem named ``tab_name``.
    True / False / None as invoke()."""
    def job(uia):
        root = uia.ElementFromHandle(hwnd)
        cond = uia.CreateAndCondition(
            uia.CreatePropertyCondition(P_CTYPE, 50019),
            uia.CreatePropertyCondition(P_NAME, str(tab_name)))
        tab = root.FindFirst(TS_DESCENDANTS, cond)
        if _nil(tab):
            return False
        btn = tab.FindFirst(TS_DESCENDANTS, uia.CreateAndCondition(
            uia.CreatePropertyCondition(P_CTYPE, 50000),
            uia.CreatePropertyCondition(P_NAME, "Close")))
        if _nil(btn):
            return False
        return bool(_press(btn))
    try:
        ok, val = uia_host.call(job, timeout_s=timeout_s)
        return _pressed(ok, val)
    except Exception:
        return False
