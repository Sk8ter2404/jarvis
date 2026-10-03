"""core/window_scope.py - which open windows a window command may touch.

WHY THIS EXISTS (2026-10-03)
============================
Live 17:22 the owner asked JARVIS to close every window but the Claude app.
There was no action for that, so the brain ran list_windows and improvised six
minimize_window actions: one of the owner's folders, the JARVIS HUD, the JARVIS
Reticle overlay, a media player, "Program Manager" (the shell's desktop) and
"Windows Input Experience" (a cloaked system frame). list_windows had handed
it JARVIS's own windows and the shell's windows as if they were the owner's
apps, and nothing on the minimize / close / move paths stopped them.

THE ONE RULE
============
Every enumeration of "the owner's windows" goes through this module:
  * list_windows                       (core.actions._act_list_windows)
  * the bulk "all windows except X"    (core.actions, close / minimize)
  * name lookups for close_window, minimize_window, focus_window,
    move_window_to_monitor and close_window's pushback count
    (bobert_companion._find_windows_by_title -> matching_windows)
  * close_window's process-name lookup (core.actions._windows_of_process).

Never a target:
  SYSTEM  the shell's own windows: "Program Manager" (WM_CLOSE on it opens
          Windows' shut-down dialog), "Windows Input Experience", "Microsoft
          Text Input Application", the taskbar and tray hosts (by window
          class), and frames the owner cannot see - cloaked (suspended UWP
          frames, windows on another virtual desktop) or zero-size.
  JARVIS  JARVIS's own windows - the HUD, the Reticle and the other overlays,
          the console, the settings window and the dashboard page: windows
          owned by this process or one of its python / console child
          processes (the HUD and overlays are child processes), this
          process's console window, and the known JARVIS window titles
          (the tray's windows and the dashboard live in other processes).

A JARVIS window is still a target for a SINGLE-window command when it is
named explicitly ("close the HUD"): matching_windows lets it through when the
query names it AND, during an owner turn, the owner's own words name it too -
so a brain that copies "JARVIS HUD" out of an old listing cannot minimize it
for a request that never mentioned it. Bulk commands (user_windows) never
include one.

Win32 reads (owning process, window class, DWM cloak) go through ``probe``,
the single seam tests patch; each fails soft to "unknown", which keeps the
window. Stdlib only at import (ctypes / psutil are imported lazily); never
raises.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Iterable, NamedTuple, Optional

__all__ = [
    "JARVIS_WINDOW_TITLES",
    "SYSTEM_WINDOW_CLASSES",
    "SYSTEM_WINDOW_TITLES",
    "WindowFacts",
    "is_jarvis_title",
    "is_jarvis_window",
    "is_system_window",
    "matching_windows",
    "names_jarvis_window",
    "own_pids",
    "own_window_handles",
    "probe",
    "user_windows",
]

# The shell's own top-level windows, by title (case-insensitive).
SYSTEM_WINDOW_TITLES = frozenset({
    "program manager",                    # the desktop (explorer.exe)
    "windows input experience",           # touch keyboard / emoji host
    "microsoft text input application",   # the text input host
    "windows shell experience host",
    "start",
    "search",
    "task view",
    "task switching",
    "notification center",
    "action center",
    "taskbar",
})

# The shell's own top-level windows, by WINDOW CLASS (case-insensitive): the
# desktop, the taskbars on every monitor and the tray / overflow hosts.
SYSTEM_WINDOW_CLASSES = frozenset({
    "progman",
    "workerw",
    "shell_traywnd",
    "shell_secondarytraywnd",
    "notifyiconoverflowwindow",
    "topleveloverflowiconwindow",
    "windows.ui.core.corewindow",
    "xamlexplorerhostislandwindow",
    "foregroundstaging",
    "multitaskingviewframe",
})

# JARVIS's own window titles (case-insensitive), where the window lives in a
# process that is not this one or its child: the tray's dialogs, the settings
# window, the overlays when another launcher started them. Kept honest by
# tests/test_window_scope.py, which reads every window title literal in the
# tree and fails on one this rule does not recognise.
JARVIS_WINDOW_TITLES = frozenset({
    "jarvis",
    "jarvis hud",
    "jarvis reticle",
    "jarvis globe",
    "jarvis air cursor",
    "jarvis air cursor 2",
    "jarvis holographic",
    "jarvis settings",
    "jarvis arc reactor status hud",
    "jarvis bambu camera hud",
    "jarvis bambu h2d overlay",
    "jarvis workshop canvas",
    "jarvis workshop hud",
    "jarvis workshop print monitor",
    "jarvis stark status ring",
    "about jarvis",
    "what jarvis knows about me",
})
# A title is JARVIS's when its first " — " part is one of the above: "JARVIS
# — Live" (the dashboard page; inside a browser window the title also carries
# the browser's suffix), "JARVIS Settings — <file>". Or its last part is
# "JARVIS": "Today's Summary — JARVIS".
_TITLE_DASH = " — "
_JARVIS_TITLE_SUFFIX = " — jarvis"
_DASHBOARD_PREFIX = "jarvis — live"

# Child processes of JARVIS that are JARVIS: the interpreter running a HUD /
# overlay / settings script, and the console host of a python console.
_OWN_CHILD_STEMS = ("python", "pythonw", "conhost", "openconsole")
_OWN_PIDS_TTL_S = 2.0

# DWMWA_CLOAKED
_DWMWA_CLOAKED = 14

_INVISIBLE_RE = re.compile("[​-‏‪-‮⁦-⁩﻿]")


class WindowFacts(NamedTuple):
    hwnd: Optional[int]
    pid: Optional[int]
    class_name: str     # lower-case; "" when unknown
    cloaked: bool


_UNKNOWN = WindowFacts(None, None, "", False)


def _norm_title(title) -> str:
    """Lower-case title with invisible marks dropped and spacing folded."""
    t = _INVISIBLE_RE.sub("", str(title or "")).replace(" ", " ")
    return " ".join(t.split()).casefold()


def _hwnd_of(w) -> Optional[int]:
    h = getattr(w, "_hWnd", None)
    if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
        return None
    return h


def probe(w) -> WindowFacts:
    """Owning process id, window class and DWM cloak state of pygetwindow
    window ``w``, each "unknown" (None / "" / False) when it can't be read:
    no native handle, not Windows, any fault. Read-only Win32 queries."""
    hwnd = _hwnd_of(w)
    if hwnd is None:
        return _UNKNOWN
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
    except Exception:
        return WindowFacts(hwnd, None, "", False)
    pid = None
    try:
        d = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(d))
        pid = int(d.value) or None
    except Exception:
        pid = None
    cls = ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        if user32.GetClassNameW(wintypes.HWND(hwnd), buf, 256):
            cls = str(buf.value or "").lower()
    except Exception:
        cls = ""
    cloaked = False
    try:
        val = wintypes.DWORD(0)
        hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), _DWMWA_CLOAKED, ctypes.byref(val),
            ctypes.sizeof(val))
        cloaked = hr == 0 and bool(val.value)
    except Exception:
        cloaked = False
    return WindowFacts(hwnd, pid, cls, cloaked)


_own_lock = threading.Lock()
_own_cache: list = [None, -1, frozenset()]   # [stamp, pid, pids]


def own_pids() -> frozenset:
    """This process's id plus its python / console-host descendants (the HUD,
    the overlays, the settings window, the console). An app JARVIS launched
    is NOT included: it is not a python or console-host process. Cached for
    two seconds; on any fault just this process."""
    me = os.getpid()
    now = time.monotonic()
    with _own_lock:
        stamp, pid, pids = _own_cache
        if pid == me and stamp is not None and 0 <= now - stamp < _OWN_PIDS_TTL_S:
            return pids
    found = {me}
    try:
        import psutil
        for child in psutil.Process(me).children(recursive=True):
            try:
                name = str(child.name() or "").lower()
            except Exception:
                continue
            stem = name[:-4] if name.endswith(".exe") else name
            if stem.startswith(_OWN_CHILD_STEMS):
                found.add(int(child.pid))
    except Exception:
        pass
    pids = frozenset(found)
    with _own_lock:
        _own_cache[:] = [now, me, pids]
    return pids


def own_window_handles() -> frozenset:
    """This process's console window, when it has one."""
    try:
        import ctypes
        h = int(ctypes.windll.kernel32.GetConsoleWindow() or 0)
        return frozenset({h}) if h > 0 else frozenset()
    except Exception:
        return frozenset()


def _zero_size(w) -> bool:
    try:
        return int(getattr(w, "width")) <= 0 or int(getattr(w, "height")) <= 0
    except Exception:
        return False


def is_system_window(w, facts: Optional[WindowFacts] = None) -> bool:
    """True for a shell window no command may touch (see the module doc)."""
    try:
        if _norm_title(getattr(w, "title", "")) in SYSTEM_WINDOW_TITLES:
            return True
        f = facts if facts is not None else probe(w)
        if f.class_name and f.class_name in SYSTEM_WINDOW_CLASSES:
            return True
        if f.cloaked:
            return True
        return _zero_size(w)
    except Exception:
        return False


def is_jarvis_title(title) -> bool:
    """True when ``title`` is one of JARVIS's own window titles."""
    t = _norm_title(title)
    if not t:
        return False
    head = t.split(_TITLE_DASH, 1)[0].strip()
    return head in JARVIS_WINDOW_TITLES or t.endswith(_JARVIS_TITLE_SUFFIX)


def is_jarvis_window(w, facts: Optional[WindowFacts] = None,
                     pids: Optional[Iterable[int]] = None,
                     handles: Optional[Iterable[int]] = None) -> bool:
    """True for one of JARVIS's own windows: owned by this process or one of
    its python / console-host children, this process's console, or a known
    JARVIS title."""
    try:
        if is_jarvis_title(getattr(w, "title", "")):
            return True
        f = facts if facts is not None else probe(w)
        own = own_pids() if pids is None else pids
        if f.pid is not None and f.pid in own:
            return True
        hs = own_window_handles() if handles is None else handles
        return f.hwnd is not None and f.hwnd in hs
    except Exception:
        return False


# Words in a JARVIS title that do not name WHICH JARVIS window it is.
_NAME_STOP = frozenset({
    "jarvis", "the", "a", "an", "my", "your", "window", "windows", "about",
    "what", "knows", "me", "s", "live", "starting", "google", "chrome",
    "microsoft", "edge", "firefox", "brave", "opera", "vivaldi", "mozilla",
})


def _jarvis_name_words(title) -> frozenset:
    """The words that name a JARVIS window: "hud" for "JARVIS HUD",
    "dashboard" for the "JARVIS — Live" page, "overlay" too for the
    reticle and the air cursor."""
    t = _norm_title(title)
    if t.startswith(_DASHBOARD_PREFIX):
        return frozenset({"dashboard"})
    words = set(re.findall(r"[a-z]+", t)) - _NAME_STOP
    if words & {"reticle", "cursor"}:
        words.add("overlay")
    return frozenset(words)


def names_jarvis_window(text, title) -> bool:
    """True when ``text`` names the JARVIS window titled ``title`` by one of
    its own words ("close the HUD", "hide the reticle", "the dashboard").
    "Jarvis" alone names none of them."""
    try:
        said = set(re.findall(r"[a-z]+", _norm_title(text)))
        return bool(said & _jarvis_name_words(title))
    except Exception:
        return False


def _titled(w) -> str:
    return str(getattr(w, "title", "") or "")


def user_windows(windows) -> list:
    """The owner's own windows among ``windows`` (pygetwindow windows, in
    order): titled, not a system window, not one of JARVIS's. What
    list_windows shows and what every bulk window command may act on."""
    out = []
    try:
        pids = handles = None
        for w in list(windows or ()):
            if not _titled(w).strip():
                continue
            facts = probe(w)
            if is_system_window(w, facts):
                continue
            if pids is None:
                pids, handles = own_pids(), own_window_handles()
            if is_jarvis_window(w, facts, pids, handles):
                continue
            out.append(w)
    except Exception:
        return out
    return out


def matching_windows(windows, query, owner_text: str = "") -> list:
    """The windows among ``windows`` whose title contains ``query``
    (case-insensitive), in order, for a SINGLE-window command by name: never
    a system window; a JARVIS window only when ``query`` names it and, when
    ``owner_text`` (the owner's words this turn) is given, those name it
    too."""
    q = str(query or "").lower().strip()
    out = []
    try:
        pids = handles = None
        for w in list(windows or ()):
            title = _titled(w)
            if not title or q not in title.lower():
                continue
            facts = probe(w)
            if is_system_window(w, facts):
                continue
            if pids is None:
                pids, handles = own_pids(), own_window_handles()
            if is_jarvis_window(w, facts, pids, handles):
                if not names_jarvis_window(query, title):
                    continue
                if owner_text and not names_jarvis_window(owner_text, title):
                    continue
            out.append(w)
    except Exception:
        return out
    return out
