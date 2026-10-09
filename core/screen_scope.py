"""core/screen_scope.py - which windows are on screen, and which of them a
spoken "click that X" is about (2026-10-05).

WHY THIS EXISTS
===============
Live 00:28-00:30 the clicks were aimed by guesswork: the whole 7680x2880
desktop shrunk to 1568 px, or "the PRIMARY (middle) monitor" by default
(core/prompts.py), or the monitor the model typed. The page the owner
meant was on the MIDDLE monitor while the model kept saying TOP.

Scope rules (SPEC 4.1):
  * a monitor named in the OWNER's words is a HARD filter;
  * the window JARVIS opened (core.opened_ledger) and the foreground
    window are soft priors (+0.10 each), the model's ``monitor:`` prefix a
    weaker one (+0.05) - never a filter;
  * otherwise the topmost window per monitor; at most 6 windows.
JARVIS's own windows are never in scope: its process tree (HUD, reticle,
settings), its console, terminals / python hosts, and a Chrome window
showing the loopback dashboard.

Win32 through ctypes (EnumWindows in z-order, DWM cloaking, tool windows
skipped); an enumerator can be injected for tests. Never raises.
"""
from __future__ import annotations

import os
import threading
import time
from typing import NamedTuple, Optional

__all__ = ["Win", "Scope", "visible_windows", "scope_for", "is_browser",
           "jarvis_pids", "set_enumerator", "TERMINAL_PROCESSES"]

TERMINAL_PROCESSES = frozenset({
    "python.exe", "pythonw.exe", "py.exe", "powershell.exe", "pwsh.exe",
    "cmd.exe", "windowsterminal.exe", "conhost.exe", "openconsole.exe",
    "wt.exe", "mintty.exe", "bash.exe",
})
BROWSER_PROCESSES = frozenset({"chrome.exe", "msedge.exe", "firefox.exe",
                               "brave.exe", "opera.exe", "vivaldi.exe"})
_MIN_W, _MIN_H = 200, 150
MAX_SCOPE = 6


class Win(NamedTuple):
    hwnd: int
    title: str
    process: str
    pid: int
    rect: tuple            # (x, y, w, h)
    monitor: Optional[str]
    cls: str = ""
    z: int = 0             # 0 = topmost
    jarvis: bool = False   # one of JARVIS's own windows
    url: str = ""


class Scope(NamedTuple):
    windows: tuple         # Win, in search order
    hard_monitor: Optional[str]
    priors: dict           # hwnd -> weight


_enum_override = [None]
_pid_cache = {"at": 0.0, "pids": frozenset()}
_proc_cache: dict = {}
_lock = threading.Lock()


def set_enumerator(fn) -> None:
    """Tests: ``fn() -> [Win | dict]`` replaces the Win32 enumeration."""
    _enum_override[0] = fn


def is_browser(win) -> bool:
    try:
        return str(_g(win, "process") or "").lower() in BROWSER_PROCESSES
    except Exception:
        return False


def _g(win, key, default=None):
    if isinstance(win, dict):
        return win.get(key, default)
    return getattr(win, key, default)


def jarvis_pids() -> frozenset:
    """This process and every descendant (HUD Qt subprocesses, the reticle,
    the settings window), cached 30 s."""
    now = time.time()
    with _lock:
        if now - _pid_cache["at"] < 30 and _pid_cache["pids"]:
            return _pid_cache["pids"]
    pids = {os.getpid()}
    try:
        import psutil
        for c in psutil.Process(os.getpid()).children(recursive=True):
            pids.add(c.pid)
    except Exception:
        pass
    out = frozenset(pids)
    with _lock:
        _pid_cache.update(at=now, pids=out)
    return out


def _proc_name(pid: int) -> str:
    with _lock:
        hit = _proc_cache.get(pid)
        if hit and time.time() - hit[1] < 60:
            return hit[0]
    name = ""
    try:
        import psutil
        name = psutil.Process(pid).name()
    except Exception:
        name = ""
    with _lock:
        _proc_cache[pid] = (name, time.time())
        if len(_proc_cache) > 512:
            for k in list(_proc_cache)[:256]:
                _proc_cache.pop(k, None)
    return name


def _monitors():
    try:
        from core.config import MONITORS
        return MONITORS
    except Exception:
        return {}


def _forbidden_title(title: str) -> bool:
    t = (title or "").lower()
    return ("jarvis" in t and any(w in t for w in (
        "hud", "reticle", "console", "settings", "dashboard", "overlay")))


def _win32_windows() -> list:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    dwm = None
    try:
        dwm = ctypes.WinDLL("dwmapi")
    except Exception:
        dwm = None
    kernel32 = ctypes.WinDLL("kernel32")
    console = 0
    try:
        console = int(kernel32.GetConsoleWindow() or 0)
    except Exception:
        console = 0
    out = []
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowLongW.restype = ctypes.c_long
    rect = wintypes.RECT()
    buf = ctypes.create_unicode_buffer(512)
    cbuf = ctypes.create_unicode_buffer(128)
    mons = _monitors()
    from core import monitor_geometry as _mg

    def cb(h, _lp):
        try:
            hwnd = int(h)
            if not user32.IsWindowVisible(h) or user32.IsIconic(h):
                return True
            if user32.GetWindowLongW(h, -20) & 0x00000080:   # WS_EX_TOOLWINDOW
                return True
            if dwm is not None:
                cloaked = ctypes.c_int(0)
                if dwm.DwmGetWindowAttribute(h, 14, ctypes.byref(cloaked),
                                             ctypes.sizeof(cloaked)) == 0 \
                        and cloaked.value:
                    return True
            if not user32.GetWindowRect(h, ctypes.byref(rect)):
                return True
            w = rect.right - rect.left
            hh = rect.bottom - rect.top
            if w < _MIN_W or hh < _MIN_H:
                return True
            user32.GetWindowTextW(h, buf, 512)
            title = buf.value or ""
            if not title:
                return True
            user32.GetClassNameW(h, cbuf, 128)
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
            r = (rect.left, rect.top, w, hh)
            out.append(Win(hwnd=hwnd, title=title,
                           process=_proc_name(int(pid.value)),
                           pid=int(pid.value), rect=r,
                           monitor=_mg.monitor_for_rect(*r, mons),
                           cls=cbuf.value or "", z=len(out),
                           jarvis=(hwnd == console)))
        except Exception:
            pass
        return True
    user32.EnumWindows(EnumProc(cb), 0)
    return out


def visible_windows(include_jarvis: bool = False) -> list:
    """Visible, un-minimised, un-cloaked top-level windows (>= 200x150, a
    title), topmost first, each marked ``jarvis`` when it is one of
    JARVIS's own. JARVIS's are dropped unless ``include_jarvis``."""
    try:
        fn = _enum_override[0]
        if fn is None:
            from core.screen_privacy import reads_blocked
            if reads_blocked():
                return []
        raw = fn() if fn is not None else _win32_windows()
        own = jarvis_pids()
        out = []
        for i, w in enumerate(raw or ()):
            if isinstance(w, dict):
                w = Win(**{k: w[k] for k in Win._fields if k in w})
            proc = (w.process or "").lower()
            mine = (w.jarvis or w.pid in own or proc in TERMINAL_PROCESSES
                    or _forbidden_title(w.title))
            if mine != w.jarvis:
                w = w._replace(jarvis=mine)
            if w.z != i:
                w = w._replace(z=i)
            if mine and not include_jarvis:
                continue
            out.append(w)
        return out
    except Exception:
        return []


def _loopback(url: str) -> bool:
    try:
        import urllib.parse
        h = (urllib.parse.urlsplit(url if "://" in url else "http://" + url)
             .hostname or "").lower()
        return h in ("127.0.0.1", "localhost", "::1", "[::1]") or h.endswith(
            ".localhost")
    except Exception:
        return False


def scope_for(said="", model_monitor=None, ledger_hwnd=None,
              foreground_hwnd=None, windows=None, url_of=None) -> Scope:
    """The windows to search for a spoken target, in search order (SPEC
    4.1). ``url_of(hwnd)`` (optional) reads a browser's URL so a Chrome
    window showing JARVIS's own dashboard is dropped. Never raises."""
    try:
        from core import monitor_geometry as _mg
        wins = list(windows) if windows is not None else visible_windows()
        wins = [w for w in wins if not _g(w, "jarvis")]
        hard = _mg.monitor_named_in(said or "", _monitors())
        if hard:
            wins = [w for w in wins if _g(w, "monitor") == hard]
        if url_of is not None:
            kept = []
            for w in wins:
                if is_browser(w):
                    try:
                        u = url_of(_g(w, "hwnd")) or ""
                    except Exception:
                        u = ""
                    if u and _loopback(u):
                        continue
                kept.append(w)
            wins = kept
        priors: dict = {}
        order: list = []

        def _add(w, weight):
            h = _g(w, "hwnd")
            priors[h] = round(priors.get(h, 0.0) + weight, 3)
            if all(_g(o, "hwnd") != h for o in order):
                order.append(w)

        by_hwnd = {_g(w, "hwnd"): w for w in wins}
        if ledger_hwnd in by_hwnd:
            _add(by_hwnd[ledger_hwnd], 0.10)
        if foreground_hwnd in by_hwnd:
            _add(by_hwnd[foreground_hwnd], 0.10)
        if model_monitor:
            for w in wins:
                if _g(w, "monitor") == model_monitor:
                    _add(w, 0.05)
                    break
        # Then the top window of every monitor, then the second (a small
        # window - Notepad, a chat - on top of a maximised browser must not
        # hide the page under it).
        per_mon: dict = {}
        for w in order:
            per_mon[_g(w, "monitor")] = per_mon.get(_g(w, "monitor"), 0) + 1
        for rank in (1, 2):
            for w in sorted(wins, key=lambda w: _g(w, "z", 0)):
                mon = _g(w, "monitor")
                if any(_g(o, "hwnd") == _g(w, "hwnd") for o in order):
                    continue
                if per_mon.get(mon, 0) >= rank:
                    continue
                per_mon[mon] = per_mon.get(mon, 0) + 1
                _add(w, 0.0)
        # Owner-named monitor: every window there is fair game (side by
        # side), topmost first, after the priors.
        if hard:
            for w in sorted(wins, key=lambda w: _g(w, "z", 0)):
                _add(w, 0.0)
        return Scope(windows=tuple(order[:MAX_SCOPE]), hard_monitor=hard,
                     priors={k: v for k, v in priors.items()
                             if any(_g(w, "hwnd") == k
                                    for w in order[:MAX_SCOPE])})
    except Exception:
        return Scope(windows=(), hard_monitor=None, priors={})
