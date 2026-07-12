"""Definitive parent-process liveness for the overlay children (unified HUD,
tray, reticle, air-cursor).

Every watcher used ``psutil.pid_exists(parent_pid)`` — which on Windows reads
TRUE (``Process(pid).status()`` even says 'running'!) for a DEAD-but-unreaped
process: ANY open handle (a monitoring shell, WMI, a child) keeps the process
row enumerable after termination. Live 2026-07-12: a terminated JARVIS's HUD +
tray survived it by 25 minutes — the owner saw two of everything ("theres
still multiple jarvis") while four independent copies of the same broken
liveness check all agreed the corpse was alive.

``WaitForSingleObject`` on a ``SYNCHRONIZE`` handle is authoritative: a
terminated process object is SIGNALED the instant it dies, no matter who
still holds handles to it. Non-Windows / failure falls back to psutil.
"""
from __future__ import annotations

import sys

_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
_WAIT_TIMEOUT = 0x00000102
_STILL_ACTIVE = 259


def parent_is_alive(pid: int) -> bool:
    """True while the process behind ``pid`` is genuinely still executing.

    Two dead states must BOTH read dead (live 2026-07-12, same afternoon):
      * terminated-but-unreaped — object SIGNALED, row still enumerable
        while anything holds a handle (WaitForSingleObject catches it);
      * TERMINATING-FOREVER — GetExitCodeProcess already returns a real
        exit code but the object is never signaled because a thread is
        pinned inside a kernel driver (the ExitProcess loader-lock zombies:
        exitcode=0, wait=TIMEOUT, unkillable, visible in Task Manager until
        reboot). The exit-code check catches this one.

    pid <= 0 → True (the "no parent to watch" convention every caller uses).
    A pid we cannot OPEN at all → False: the overlays' parent is always the
    same-user JARVIS process, so access-denied means it's gone (a recycled
    pid grabbed by a privileged stranger also — correctly — reads dead).
    """
    if pid is None or pid <= 0:
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                        wintypes.DWORD]
            k32.WaitForSingleObject.restype = wintypes.DWORD
            k32.WaitForSingleObject.argtypes = [wintypes.HANDLE,
                                                wintypes.DWORD]
            k32.GetExitCodeProcess.restype = wintypes.BOOL
            k32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            h = k32.OpenProcess(
                _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION,
                False, int(pid))
            if not h:
                return False
            try:
                code = wintypes.DWORD()
                if (k32.GetExitCodeProcess(h, ctypes.byref(code))
                        and code.value != _STILL_ACTIVE):
                    return False        # exited (even if never signaled)
                return k32.WaitForSingleObject(h, 0) == _WAIT_TIMEOUT
            finally:
                k32.CloseHandle(h)
        except Exception:
            pass
    try:
        import psutil
        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return True
