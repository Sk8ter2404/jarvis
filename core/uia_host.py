"""core/uia_host.py - the ONE thread that talks to Windows UI Automation
(2026-10-05).

WHY THIS EXISTS
===============
Reading a page's links by name (core.screen_text) means COM: an apartment,
a CUIAutomation object, element pointers that belong to the thread that
made them. Three lessons from this codebase decide the shape:

  * NEVER A THREAD PER CALL - v2.0.100's bounded camera opens spawned one
    worker each, and the wedged ones never exited: 14,507 threads, 1.4 M
    handles, a dead process at 3.5 h. Here every UIA call runs on ONE
    long-lived daemon thread, fed by a queue.
  * NEVER LET IT EXIT - core/ct2_host's lesson: native state torn down by a
    thread exit is a crash class of its own. The host loop cannot end.
  * NOTHING UNBOUNDED ON THE VOICE THREAD - ``call`` waits at most its
    timeout; IUIAutomation2 ConnectionTimeout (1.5 s) / TransactionTimeout
    (1 s) bound a hung provider inside COM too.

WEDGE POLICY. A call past its timeout returns (False, "timeout"). If the
host is still busy on that job 3 s later it is RETIRED (left parked - it is
a daemon) and ONE replacement starts; a second wedge disables UIA for the
session, logs "[uia] disabled: host wedged twice", and every caller gets
(False, "disabled") - core.grounded_click then says so once.

The comtypes wrapper for UIAutomationCore.dll is generated into
``data/comtypes_gen`` (never site-packages: App Control blocks writes
there). Under JARVIS_TEST_MODE the real COM object is refused unless a
test injected a backend (``set_backend``): tests never reach real UIA.
Never raises at the public API.
"""
from __future__ import annotations

import os
import queue
import threading
import time

__all__ = ["call", "submit", "available", "set_backend", "status",
           "reset_for_tests", "DISABLED", "SWITCHED_OFF", "WEDGE_GRACE_S",
           "switched_off"]

DISABLED = "disabled"
SWITCHED_OFF = "switched off"
WEDGE_GRACE_S = 3.0
_CONNECTION_TIMEOUT_MS = 1500
_TRANSACTION_TIMEOUT_MS = 1000

_lock = threading.Lock()
_backend = [None]          # tests: a factory () -> fake uia object
_hosts: list = []          # every host ever started (retired ones parked)
_state = {"host": None, "wedges": 0, "disabled": False, "told": False,
          "dpi_logged": False}


class _Job:
    __slots__ = ("fn", "done", "ok", "value", "abandoned", "started")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.ok = False
        self.value = None
        self.abandoned = False
        self.started = 0.0


class _Host:
    def __init__(self, n: int):
        self.n = n
        self.q: "queue.Queue" = queue.Queue()
        self.uia = None
        self.init_error = ""
        self.ready = threading.Event()
        self.current: "_Job | None" = None
        self.retired = False
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"uia-host{'' if n == 1 else n}")
        self.thread.start()

    def _run(self) -> None:            # never returns
        try:
            self.uia = _make_uia()
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"[:200]
            print(f"  [uia] not available: {self.init_error}", flush=True)
        finally:
            self.ready.set()
        while True:
            try:
                job = self.q.get()
            except Exception:
                time.sleep(0.5)
                continue
            if job.abandoned:
                job.done.set()
                continue
            self.current = job
            job.started = time.monotonic()
            try:
                if self.uia is None:
                    raise RuntimeError(self.init_error or "UIA unavailable")
                job.value = job.fn(self.uia)
                job.ok = True
            except Exception as e:
                job.ok = False
                job.value = f"{type(e).__name__}: {e}"[:200]
            finally:
                self.current = None
                job.done.set()


def _test_mode() -> bool:
    from core.screen_privacy import reads_blocked
    return reads_blocked()


def _gen_dir() -> str:
    from core.paths import data_dir
    d = os.path.join(data_dir(), "comtypes_gen")
    os.makedirs(d, exist_ok=True)
    return d


def _make_uia():
    """The UIA client object for this (host) thread."""
    factory = _backend[0]
    if factory is not None:
        return factory()
    if _test_mode():
        raise RuntimeError("real UI Automation is refused in a test process "
                           "(inject a backend)")
    import comtypes
    import comtypes.client
    import comtypes.gen
    gen = _gen_dir()
    comtypes.client.gen_dir = gen
    try:
        if gen not in list(comtypes.gen.__path__):
            comtypes.gen.__path__.insert(0, gen)
    except Exception:
        pass
    comtypes.CoInitializeEx(0x2)          # COINIT_APARTMENTTHREADED
    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as UIA
    obj = comtypes.client.CreateObject(UIA.CUIAutomation8,
                                       interface=UIA.IUIAutomation)
    try:
        obj2 = obj.QueryInterface(UIA.IUIAutomation2)
        obj2.ConnectionTimeout = _CONNECTION_TIMEOUT_MS
        obj2.TransactionTimeout = _TRANSACTION_TIMEOUT_MS
        obj = obj2
    except Exception:
        pass
    _log_dpi_once()
    return obj


def _log_dpi_once() -> None:
    """Boot log: the DPI of every monitor (expected 96 on all four)."""
    if _state["dpi_logged"]:
        return
    _state["dpi_logged"] = True
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32")
        shcore = ctypes.WinDLL("shcore")
        dpis = []
        Proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR,
                                  wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                  wintypes.LPARAM)

        def _cb(hmon, _hdc, rect, _lp):
            x, y = ctypes.c_uint(), ctypes.c_uint()
            if shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(x),
                                       ctypes.byref(y)) == 0:
                r = rect.contents
                dpis.append(f"({r.left},{r.top}) {x.value}")
            return True
        user32.EnumDisplayMonitors(None, None, Proc(_cb), 0)
        print(f"  [uia] monitor DPI: {', '.join(dpis) or 'unknown'}",
              flush=True)
    except Exception:
        pass


def set_backend(factory) -> None:
    """Tests: run every job against ``factory()`` (a fake UIA object)
    instead of COM. ``None`` restores the real backend. Resets the host."""
    with _lock:
        _backend[0] = factory
        _state.update(host=None, wedges=0, disabled=False, told=False)


def reset_for_tests() -> None:
    with _lock:
        _state.update(host=None, wedges=0, disabled=False, told=False)


def switched_off() -> bool:
    """SCREEN_UIA_ENABLED is False: no UI Automation at all - the ONE
    chokepoint every reader passes (core.screen_text: page reads, address
    reads, presses, the privacy gate's address check, the page wait after
    open_url, the search-results line). Review 2026-10-05: the switch only
    stopped the click path's page reads. Never raises."""
    try:
        from core import config as _c
        return getattr(_c, "SCREEN_UIA_ENABLED", True) is False
    except Exception:
        return False


def _host() -> "_Host | None":
    if switched_off():
        return None
    with _lock:
        if _state["disabled"]:
            return None
        h = _state["host"]
        if h is None:
            if _backend[0] is None and _test_mode():
                return None
            h = _Host(len(_hosts) + 1)
            _hosts.append(h)
            _state["host"] = h
        return h


def available(wait_s: float = 0.0) -> bool:
    """True when the host is up and has a UIA object."""
    h = _host()
    if h is None:
        return False
    if wait_s and not h.ready.is_set():
        h.ready.wait(wait_s)
    return h.ready.is_set() and h.uia is not None


def _check_wedge(h: "_Host") -> None:
    """Retire ``h`` when its current job has run past timeout + grace."""
    job = h.current
    if job is None or not job.abandoned:
        return
    if time.monotonic() - job.started < WEDGE_GRACE_S:
        return
    with _lock:
        if h.retired or _state["host"] is not h:
            return
        h.retired = True
        _state["wedges"] += 1
        if _state["wedges"] >= 2:
            _state["disabled"] = True
            _state["host"] = None
            print("  [uia] disabled: host wedged twice", flush=True)
        else:
            _state["host"] = None
            print("  [uia] host wedged - starting one replacement",
                  flush=True)


def call(fn, timeout_s: float = 1.0):
    """Run ``fn(uia)`` on the host thread. (True, value) / (False, why):
    why is "timeout", "disabled", "unavailable" or the exception text.
    Waits at most ``timeout_s`` (plus the host's start-up the first time,
    bounded too). Never raises."""
    try:
        if switched_off():
            return False, SWITCHED_OFF
        h = _host()
        if h is None:
            return False, DISABLED if _state["disabled"] else "unavailable"
        _check_wedge(h)
        h = _host()
        if h is None:
            return False, DISABLED if _state["disabled"] else "unavailable"
        if not h.ready.is_set() and not h.ready.wait(max(2.0, timeout_s)):
            return False, "timeout"
        if h.uia is None:
            return False, "unavailable"
        job = _Job(fn)
        h.q.put(job)
        if job.done.wait(max(0.01, float(timeout_s))):
            return job.ok, job.value
        job.abandoned = True
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}"


def submit(fn) -> bool:
    """Queue ``fn(uia)`` without waiting (a scene freeze). False when the
    host is unavailable."""
    try:
        h = _host()
        if h is None or (h.ready.is_set() and h.uia is None):
            return False
        h.q.put(_Job(fn))
        return True
    except Exception:
        return False


def disabled_notice() -> str:
    """The one-time line for the next screen action after UIA was disabled
    ('' when not disabled, or already said)."""
    with _lock:
        if _state["disabled"] and not _state["told"]:
            _state["told"] = True
            return ("My page reader stopped responding twice, sir, so I've "
                    "switched it off until I restart.")
    return ""


def status() -> dict:
    with _lock:
        h = _state["host"]
        return {"hosts": len(_hosts), "wedges": _state["wedges"],
                "disabled": _state["disabled"],
                "ready": bool(h and h.ready.is_set() and h.uia is not None),
                "queue": h.q.qsize() if h else 0}
