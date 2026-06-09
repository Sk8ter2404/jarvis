"""Auto-YIELD-to-real-input watcher for the Kinect air-mouse.

THE PROBLEM this solves
=======================
The air-mouse drives the OS cursor (SetCursorPos) and clicks (mouse_event /
SendInput). If the owner reaches for their REAL mouse or keyboard while the
air-mouse is active, the two fight over the cursor. This module makes the
air-mouse YIELD: the instant any REAL (hardware) input arrives, the air-mouse
force-disengages and stays SUPPRESSED until ~1.5 s after the most recent real
input — so touching the real mouse/keyboard always wins, immediately.

HOW it detects REAL input (and ignores its OWN)
===============================================
A low-level Windows hook on a DEDICATED THREAD with its own message pump:
  • SetWindowsHookEx(WH_MOUSE_LL)    — every mouse event system-wide.
  • SetWindowsHookEx(WH_KEYBOARD_LL) — every keypress system-wide.
Each callback records a MONOTONIC timestamp of the last real input.

CRITICAL — do not self-trigger:
  • The air-mouse MOVES the cursor with SetCursorPos, which does NOT generate
    WH_MOUSE_LL events at all — so cursor motion never looks like real input.
  • The air-mouse CLICKS with mouse_event / SendInput, which DO generate
    WH_MOUSE_LL events, but with the LLMHF_INJECTED flag (0x01) SET. The mouse
    callback IGNORES any event whose MSLLHOOKSTRUCT.flags has LLMHF_INJECTED set,
    so the air-mouse's own clicks are not counted as real input.
  • The air-mouse never types, so EVERY keyboard event is real input.

GRACEFUL DEGRADATION
====================
install() is lazy + best-effort. If SetWindowsHookEx fails (or ctypes/user32 is
unavailable — e.g. the light-tier CI runner), it logs a warning and the watcher
FALLS BACK to polling GetLastInputInfo: it compares the OS "last input" tick to
the air-mouse's OWN last-injected-action time, and treats the OS input as real
only when it is NEWER than our own action. Never raises out to the poller.

PUBLIC API (all NEVER raise)
============================
  install()                       — idempotent; start the hook thread (or arm the
                                     polling fallback). Safe to call every tick.
  mark_self_action()              — the air-mouse calls this whenever IT moves /
                                     clicks the cursor, so the polling fallback can
                                     discount its own activity.
  seconds_since_real_input(now)   — seconds since the last REAL input (huge if
                                     none / unavailable).
  real_input_recent(window, now)  — True if real input occurred within `window` s
                                     (→ the air-mouse must yield + stay suppressed).
  note_real_input_for_test(ts)    — TEST seam: inject the last-real-input
                                     timestamp without a real hook.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


# Win32 constants (avoid importing win32con so this works ctypes-only).
_WH_MOUSE_LL = 14
_WH_KEYBOARD_LL = 13
_LLMHF_INJECTED = 0x00000001     # MSLLHOOKSTRUCT.flags bit: event was injected
_LLKHF_INJECTED = 0x00000010     # KBDLLHOOKSTRUCT.flags bit: injected keystroke
_WM_QUIT = 0x0012


# ─── shared state (one process-wide watcher) ─────────────────────────────────
_lock = threading.Lock()
# Monotonic timestamp of the last REAL (non-injected) hardware input. Starts at
# -inf so "seconds since" is huge until something real actually happens (the
# air-mouse is NOT suppressed at boot).
_last_real_input = float("-inf")
# Monotonic timestamp of the air-mouse's OWN last injected action (cursor move or
# click), used only by the GetLastInputInfo polling fallback to discount itself.
_last_self_action = float("-inf")
# Wall-clock (time.time) of the last self action, to compare against
# GetLastInputInfo's wall-clock-derived "last input" in the fallback path.
_last_self_action_wall = float("-inf")

_install_lock = threading.Lock()
_installed = False               # a hook thread or the fallback is armed
_hook_ok = False                 # the LL hook installed successfully
_thread: "Optional[threading.Thread]" = None
# Keep references to the ctypes callback trampolines for the life of the process —
# if they're GC'd while the hook is live, Windows calls freed memory (crash).
_mouse_cb = None
_kbd_cb = None
_warned = [False]                # one-shot warning when the hook can't install


def _now_mono() -> float:
    return time.monotonic()


def _record_real_input() -> None:
    """Stamp 'a real input just happened' (monotonic). Called from the hook
    callbacks for non-injected events."""
    global _last_real_input
    with _lock:
        _last_real_input = _now_mono()


def mark_self_action(now: "Optional[float]" = None) -> None:
    """The air-mouse calls this whenever IT moves or clicks the cursor. Only the
    polling fallback uses it (to avoid mistaking our own activity for the owner's);
    the LL-hook path ignores injected events directly, so this is harmless there.
    NEVER raises."""
    global _last_self_action, _last_self_action_wall
    try:
        with _lock:
            _last_self_action = _now_mono() if now is None else float(now)
            _last_self_action_wall = time.time()
    except Exception:
        pass


def note_real_input_for_test(ts: "Optional[float]" = None) -> None:
    """TEST seam: set the last-real-input timestamp directly (monotonic seconds),
    so the suppression logic can be exercised WITHOUT installing a real hook."""
    global _last_real_input
    with _lock:
        _last_real_input = _now_mono() if ts is None else float(ts)


def _last_real_input_mono() -> float:
    """The monotonic timestamp of the last real input. When the LL hook ISN'T
    active, fall back to polling GetLastInputInfo and fold the result in: if the OS
    reports input NEWER than the air-mouse's own last injected action, treat it as
    real input now. NEVER raises."""
    with _lock:
        latest = _last_real_input
        self_wall = _last_self_action_wall
    # Only consult the GetLastInputInfo polling fallback once install() has run AND
    # the LL hook did NOT come up — i.e. the live degraded path. Before install()
    # (and in unit tests, which inject the timestamp directly), we trust the
    # injected/hook value alone and never poll the OS, so a test machine's real
    # recent input can't leak into the pure tests.
    if _installed and not _hook_ok:
        polled = _poll_last_input_is_real(self_wall)
        if polled is not None and polled > latest:
            latest = polled
    return latest


def seconds_since_real_input(now: "Optional[float]" = None) -> float:
    """Seconds since the last REAL input (monotonic). Huge when nothing real has
    happened yet / the watcher is unavailable. NEVER raises."""
    try:
        t = _now_mono() if now is None else float(now)
        return t - _last_real_input_mono()
    except Exception:
        return float("inf")


def real_input_recent(window: float, now: "Optional[float]" = None) -> bool:
    """True when REAL input occurred within the last `window` seconds — i.e. the
    air-mouse must YIELD (force-disengage) and stay SUPPRESSED. NEVER raises."""
    try:
        return seconds_since_real_input(now) < float(window)
    except Exception:
        return False


# ─── GetLastInputInfo polling fallback ───────────────────────────────────────
def _poll_last_input_is_real(self_action_wall: float) -> "Optional[float]":
    """Polling fallback when the LL hook isn't installed. Reads the OS 'last input'
    time (GetLastInputInfo, a GetTickCount-based ms counter) and, if that input is
    MORE RECENT than the air-mouse's own last injected action, returns a MONOTONIC
    timestamp marking 'real input just now'. Returns None when it can't tell (no
    real input newer than ours, or the API is unavailable). NEVER raises.

    The OS counter can't distinguish injected from hardware input, so we
    approximate: any OS input within a small slop AFTER our own last action is
    assumed to be ours; anything newer than that is treated as the owner's."""
    try:
        import ctypes
        from ctypes import wintypes

        class _LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return None
        tick_now = ctypes.windll.kernel32.GetTickCount()
        # Age of the OS's last input, in seconds (how long ago it happened).
        age_s = max(0.0, (int(tick_now) - int(lii.dwTime)) / 1000.0)
        last_input_wall = time.time() - age_s
        # If the OS input is within 150 ms AFTER our own last injected action, it's
        # almost certainly our injected click → not real. Otherwise count it.
        if (self_action_wall != float("-inf")
                and last_input_wall <= self_action_wall + 0.15):
            return None
        # Map the wall-clock "last input" onto the monotonic clock: it happened
        # `age_s` ago, so its monotonic stamp is now-minus-age.
        return _now_mono() - age_s
    except Exception:
        return None


# ─── low-level hook install (dedicated thread + message pump) ─────────────────
def install() -> bool:
    """Install the WH_MOUSE_LL + WH_KEYBOARD_LL hooks on a dedicated daemon thread
    with its own message pump. Idempotent + lazy + graceful: safe to call every
    tick; on failure logs ONE warning and arms the GetLastInputInfo fallback.
    Returns True when the LL hook is active, False when running on the fallback.
    NEVER raises."""
    global _installed, _thread
    with _install_lock:
        if _installed:
            return _hook_ok
        _installed = True
        try:
            import ctypes  # noqa: F401  (probe availability before spawning)
            _ = ctypes.windll.user32
        except Exception:
            _warn_once("ctypes/user32 unavailable")
            return False
        try:
            _thread = threading.Thread(target=_hook_thread, daemon=True,
                                       name="kinect-air-mouse-yield-hook")
            _thread.start()
        except Exception as e:   # pragma: no cover - thread spawn is platform I/O
            _warn_once(f"hook thread failed to start: {e}")
            return False
    # Give the thread a moment to set up the hooks (it flips _hook_ok). Brief +
    # bounded so a wedged install can't hang the caller.
    for _ in range(50):
        if _hook_ok:
            break
        time.sleep(0.005)
    return _hook_ok


def _warn_once(msg: str) -> None:
    if not _warned[0]:
        _warned[0] = True
        try:
            print(f"  [air-mouse] auto-yield: LL hook unavailable ({msg}); "
                  "falling back to GetLastInputInfo polling")
        except Exception:
            pass


def _hook_thread() -> None:  # pragma: no cover - needs a real Windows message loop
    """Dedicated thread: install both LL hooks, then run a GetMessage pump so the
    callbacks fire. The hooks are thread-affine + require a message loop, which is
    exactly why this lives on its own thread. NEVER raises out."""
    global _hook_ok, _mouse_cb, _kbd_cb
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # LRESULT CALLBACK LowLevelProc(int nCode, WPARAM wParam, LPARAM lParam)
        HOOKPROC = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        class _MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [("pt", wintypes.POINT),
                        ("mouseData", wintypes.DWORD),
                        ("flags", wintypes.DWORD),
                        ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

        class _KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [("vkCode", wintypes.DWORD),
                        ("scanCode", wintypes.DWORD),
                        ("flags", wintypes.DWORD),
                        ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]

        def _mouse_proc(nCode, wParam, lParam):
            try:
                if nCode >= 0:
                    ms = ctypes.cast(
                        lParam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                    # IGNORE the air-mouse's OWN injected clicks (LLMHF_INJECTED);
                    # count only genuine hardware mouse events as real input.
                    if not (ms.flags & _LLMHF_INJECTED):
                        _record_real_input()
            except Exception:
                pass
            return user32.CallNextHookEx(None, nCode, wParam, lParam)

        def _kbd_proc(nCode, wParam, lParam):
            try:
                if nCode >= 0:
                    kb = ctypes.cast(
                        lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                    # The air-mouse never types, so all keypresses are real — but
                    # still skip any injected keystroke for correctness.
                    if not (kb.flags & _LLKHF_INJECTED):
                        _record_real_input()
            except Exception:
                pass
            return user32.CallNextHookEx(None, nCode, wParam, lParam)

        _mouse_cb = HOOKPROC(_mouse_proc)
        _kbd_cb = HOOKPROC(_kbd_proc)
        h_mod = kernel32.GetModuleHandleW(None)
        user32.SetWindowsHookExW.restype = wintypes.HHOOK
        h_mouse = user32.SetWindowsHookExW(_WH_MOUSE_LL, _mouse_cb, h_mod, 0)
        h_kbd = user32.SetWindowsHookExW(_WH_KEYBOARD_LL, _kbd_cb, h_mod, 0)
        if not h_mouse and not h_kbd:
            _warn_once("SetWindowsHookEx returned NULL")
            return
        _hook_ok = True

        # Message pump — REQUIRED for LL hooks to deliver. GetMessage blocks the
        # thread until a message arrives (the hooks themselves wake it).
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except Exception as e:
        _warn_once(f"hook thread error: {e}")
        return
