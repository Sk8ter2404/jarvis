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

64-BIT CTYPES (B057, 2026-10-01): every Win32 call in the hook path goes through
PRIVATE, fully-typed WinDLL handles (_win32_hook_api). Untyped, ctypes returns a
32-bit int, which truncated GetModuleHandleW's HMODULE so SetWindowsHookExW got a
bogus hMod and returned NULL on EVERY boot — the hook had never once installed.

HOOK HEALTH CHECK: a Python hook callback needs the GIL. If it misses Windows'
LowLevelHooksTimeout even once, Windows SILENTLY removes the hook (no
notification), and an LL hook also never sees input aimed at an elevated window.
Either way the OS registers input the hook never reported. So while the hook is
"up", each read also checks GetLastInputInfo: OS input more than
_HOOK_STALE_SLOP_S newer than the hook's last callback means the hook is blind,
and that read uses the polling fallback instead. Self-healing: the next hook
callback makes the hook path authoritative again.

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
# OS input this much newer than the hook's last callback ⇒ the hook is blind
# (removed by LowLevelHooksTimeout, or input to an elevated window). Generous next
# to GetTickCount's ~16 ms granularity; a live hook stamps the same input within ms.
_HOOK_STALE_SLOP_S = 1.0


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
# Monotonic timestamp of the hook's last callback for ANY event (injected or not):
# proof the hook is still alive. Feeds the health check in _last_real_input_mono.
_last_hook_event = float("-inf")

_install_lock = threading.Lock()
_installed = False               # a hook thread or the fallback is armed
_hook_ok = False                 # the LL hook installed successfully
_thread: "Optional[threading.Thread]" = None
# Keep references to the ctypes callback trampolines for the life of the process —
# if they're GC'd while the hook is live, Windows calls freed memory (crash).
_mouse_cb = None
_kbd_cb = None
_warned = [False]                # one-shot warning when the hook can't install
_stale_warned = [False]          # one-shot warning when the live hook goes blind
_LASTINPUTINFO = None            # lazily-built ctypes struct (polled every tick)


def _now_mono() -> float:
    return time.monotonic()


def _record_real_input() -> None:
    """Stamp 'a real input just happened' (monotonic). Called from the hook
    callbacks for non-injected events."""
    global _last_real_input
    with _lock:
        _last_real_input = _now_mono()


def _note_hook_event(real: bool) -> None:
    """One hook callback's whole job (kept tiny: it runs for EVERY system-wide
    mouse/key event while Windows waits on it). Stamps 'the hook is alive' and,
    for a non-injected event, 'real input just happened'."""
    global _last_hook_event, _last_real_input
    t = _now_mono()
    with _lock:
        _last_hook_event = t
        if real:
            _last_real_input = t


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
    active (or the health check finds it blind), fall back to polling
    GetLastInputInfo and fold the result in: if the OS reports input NEWER than the
    air-mouse's own last injected action, treat it as real input now. NEVER
    raises."""
    with _lock:
        latest = _last_real_input
        self_wall = _last_self_action_wall
        last_hook_event = _last_hook_event
    # Only consult the GetLastInputInfo polling fallback once install() has run AND
    # the LL hook is NOT trustworthy — it never came up, or it is up but blind (the
    # health check). Before install() (and in unit tests, which inject the
    # timestamp directly), we trust the injected/hook value alone and never poll
    # the OS, so a test machine's real recent input can't leak into the pure tests.
    if _installed:
        polled = None
        if not _hook_ok:
            polled = _poll_last_input_is_real(self_wall)
        else:
            age_s = _os_last_input_age_s()
            if age_s is not None and _hook_looks_stale(age_s, last_hook_event):
                polled = _os_input_as_real(age_s, self_wall)
        if polled is not None and polled > latest:
            latest = polled
    return latest


def _hook_looks_stale(age_s: float, last_hook_event: float) -> bool:
    """Health check for a hook that installed: True when the OS saw input (it
    happened `age_s` ago) more than _HOOK_STALE_SLOP_S AFTER the hook's last
    callback — Windows removed the hook (LowLevelHooksTimeout) or the input went
    to a window the hook can't see. Warns once. NEVER raises."""
    try:
        gap = (_now_mono() - age_s) - last_hook_event
        if gap <= _HOOK_STALE_SLOP_S:
            return False
        # A read can also land while a live hook's callback is still waiting for
        # the GIL. Polling for that one read is no worse than the old always-poll
        # path, but only input left unreported for over the slop proves the hook
        # is blind, so only that is logged.
        if age_s > _HOOK_STALE_SLOP_S and not _stale_warned[0]:
            _stale_warned[0] = True
            print("  [air-mouse] auto-yield: LL hook missed OS input "
                  f"{min(gap, 9999.0):.1f}s newer than its last callback (removed "
                  "by LowLevelHooksTimeout, or input it cannot see: an elevated "
                  "window / the lock screen); polling GetLastInputInfo until the "
                  "hook reports again")
        return True
    except Exception:
        return False


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
    age_s = _os_last_input_age_s()
    if age_s is None:
        return None
    return _os_input_as_real(age_s, self_action_wall)


def _tick_age_s(tick_now: int, last_tick: int) -> float:
    """Seconds between two GetTickCount-style DWORD millisecond ticks, done in
    32-bit modular arithmetic. Untyped ctypes returns GetTickCount as a SIGNED
    c_int (negative after 24.8 days of uptime) and the counter wraps at 49.7 days;
    plain subtraction then goes negative, which clamped to 'input 0 s ago' and
    flagged REAL input on every poll."""
    delta_ms = (int(tick_now) - int(last_tick)) & 0xFFFFFFFF
    if delta_ms > 0xFFFFFFFF - 1000:
        delta_ms = 0     # last_tick a hair AHEAD of tick_now (read race): now, not 49.7 days
    return delta_ms / 1000.0


def _os_last_input_age_s() -> "Optional[float]":
    """How long ago (seconds) the OS saw ANY input, injected or not, from
    GetLastInputInfo. None when unavailable. Read-only use of the shared
    ctypes.windll handles (no argtypes/restype set on them). NEVER raises."""
    global _LASTINPUTINFO
    try:
        import ctypes
        if _LASTINPUTINFO is None:
            from ctypes import wintypes

            class _LII(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]
            _LASTINPUTINFO = _LII
        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return None
        return _tick_age_s(ctypes.windll.kernel32.GetTickCount(), lii.dwTime)
    except Exception:
        return None


def _os_input_as_real(age_s: float, self_action_wall: float) -> "Optional[float]":
    """The fallback's verdict on OS input that happened `age_s` ago: a MONOTONIC
    'real input' stamp, or None when it is within 150 ms after the air-mouse's
    own last action (then it is almost certainly our injected click)."""
    last_input_wall = time.time() - age_s
    if (self_action_wall != float("-inf")
            and last_input_wall <= self_action_wall + 0.15):
        return None
    # Map the wall-clock "last input" onto the monotonic clock: it happened
    # `age_s` ago, so its monotonic stamp is now-minus-age.
    return _now_mono() - age_s


# ─── low-level hook install (dedicated thread + message pump) ─────────────────
def install() -> bool:
    """Install the WH_MOUSE_LL + WH_KEYBOARD_LL hooks on a dedicated daemon thread
    with its own message pump. Idempotent + lazy + graceful: safe to call every
    tick; on failure logs ONE warning and arms the GetLastInputInfo fallback.
    Returns True when the LL hook is active, False when running on the fallback.
    NEVER raises."""
    global _installed
    with _install_lock:
        if _installed:
            return _hook_ok
        _installed = True
        if not _start_hook_thread():
            return False
    # Give the thread a moment to set up the hooks (it flips _hook_ok). Brief +
    # bounded so a wedged install can't hang the caller.
    for _ in range(50):
        if _hook_ok:
            break
        time.sleep(0.005)
    return _hook_ok


def _start_hook_thread() -> bool:
    """The setting gate, the user32 probe and the hook thread start (install()
    holds _install_lock). True when the thread was started. Never raises."""
    global _thread
    if not _ll_hook_enabled():
        # 2026-10-01 (B057 review): with the ctypes types fixed the hook
        # really installs — and then every mouse/keyboard event on the PC
        # waits for this Python callback, which needs the GIL of a process
        # running ~3 cores of work: ~15.6 ms per event measured, 311 ms for
        # a 20-event burst (games included). Off by default until a
        # GIL-free design (separate process / raw input) lands; the
        # GetLastInputInfo fallback keeps the real-input yield working.
        _warn_once("off by setting: AIR_MOUSE_LL_HOOK_ENABLED=False")
        return False
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
    return True


def _ll_hook_enabled() -> bool:
    """AIR_MOUSE_LL_HOOK_ENABLED from the running monolith (owner settings
    applied) or core.config; False when unreadable."""
    import sys
    for mod in (sys.modules.get("bobert_companion"), sys.modules.get("core.config")):
        v = getattr(mod, "AIR_MOUSE_LL_HOOK_ENABLED", None) if mod is not None else None
        if isinstance(v, bool):
            return v
    return False


def _warn_once(msg: str) -> None:
    if not _warned[0]:
        _warned[0] = True
        try:
            print(f"  [air-mouse] auto-yield: LL hook unavailable ({msg}); "
                  "falling back to GetLastInputInfo polling")
        except Exception:
            pass


def _win32_hook_api():  # pragma: no cover - Windows-only ctypes (WinDLL); unit-tested on Windows
    """PRIVATE, fully-typed user32/kernel32 handles for the LL hooks (B057).

    Private WinDLL instances, NOT ctypes.windll.*: argtypes/restype live on a
    function object that ctypes.windll shares process-wide, so declaring them
    there would silently retype the call for every other module in the monolith.

    Untyped, ctypes returns a 32-bit c_int and converts int args as c_int. On
    64-bit Python that (a) truncated GetModuleHandleW's HMODULE, so
    SetWindowsHookExW got a bogus hMod and returned NULL on every boot, and
    (b) makes CallNextHookEx raise "int too long to convert" for an lParam
    pointer above 4 GB. LRESULT is pointer-sized (LONG_PTR) = c_ssize_t."""
    import ctypes
    import types
    from ctypes import wintypes

    LRESULT = ctypes.c_ssize_t
    # LRESULT CALLBACK LowLevelProc(int nCode, WPARAM wParam, LPARAM lParam)
    HOOKPROC = ctypes.WINFUNCTYPE(
        LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    LPMSG = ctypes.POINTER(wintypes.MSG)

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    user32.SetWindowsHookExW.argtypes = [
        ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = [
        wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = LRESULT
    user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [LPMSG, wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [LPMSG]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [LPMSG]
    user32.DispatchMessageW.restype = LRESULT
    return types.SimpleNamespace(ctypes=ctypes, wintypes=wintypes,
                                 user32=user32, kernel32=kernel32,
                                 LRESULT=LRESULT, HOOKPROC=HOOKPROC)


def _build_hook_procs(api):  # pragma: no cover - Windows-only ctypes (wintypes); unit-tested on Windows
    """The two LowLevel*Proc bodies (plain Python functions; _hook_thread wraps
    them in api.HOOKPROC). Windows calls them for EVERY system-wide mouse/key
    event and waits for the answer, and each needs the GIL, so they are minimal
    and exception-proof: one flags read, one stamp, then CallNextHookEx. A raise
    inside a ctypes callback prints a traceback per event, so nothing may raise,
    CallNextHookEx included (0 = let the input through; never block it)."""
    ctypes, wintypes = api.ctypes, api.wintypes
    call_next = api.user32.CallNextHookEx

    class _MSLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("pt", wintypes.POINT),
                    ("mouseData", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]      # ULONG_PTR

    class _KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("vkCode", wintypes.DWORD),
                    ("scanCode", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_size_t)]      # ULONG_PTR

    dword_at = wintypes.DWORD.from_address
    mouse_flags_off = _MSLLHOOKSTRUCT.flags.offset
    kbd_flags_off = _KBDLLHOOKSTRUCT.flags.offset

    def _mouse_proc(nCode, wParam, lParam):
        try:
            if nCode >= 0:
                # IGNORE injected events (LLMHF_INJECTED: the air-mouse's own
                # clicks, AFK-Helper, any SendInput); count only hardware input.
                _note_hook_event(
                    not (dword_at(lParam + mouse_flags_off).value & _LLMHF_INJECTED))
        except Exception:
            pass
        try:
            return call_next(None, nCode, wParam, lParam)
        except Exception:
            return 0

    def _kbd_proc(nCode, wParam, lParam):
        try:
            if nCode >= 0:
                # The air-mouse never types, so all keypresses are real — but
                # still skip any injected keystroke for correctness.
                _note_hook_event(
                    not (dword_at(lParam + kbd_flags_off).value & _LLKHF_INJECTED))
        except Exception:
            pass
        try:
            return call_next(None, nCode, wParam, lParam)
        except Exception:
            return 0

    return _mouse_proc, _kbd_proc


def _hook_thread() -> None:  # pragma: no cover - needs a real Windows message loop
    """Dedicated thread: install both LL hooks, then run a GetMessage pump so the
    callbacks fire. The hooks are thread-affine + require a message loop, which is
    exactly why this lives on its own thread. Both hooks or neither: a half-
    installed pair would silently stop yielding to one device, so a partial
    install is undone and the polling fallback takes over. Whenever the pump
    ends, the hooks are removed first (a hook with no pump stalls every input
    event system-wide until Windows times it out). NEVER raises out."""
    global _hook_ok, _mouse_cb, _kbd_cb, _last_hook_event
    user32 = None
    hooks = []
    try:
        api = _win32_hook_api()
        user32 = api.user32
        mouse_proc, kbd_proc = _build_hook_procs(api)
        _mouse_cb = api.HOOKPROC(mouse_proc)
        _kbd_cb = api.HOOKPROC(kbd_proc)
        # The full, correctly-typed module handle (hMod=None also works for LL
        # hooks; the real handle is the documented form).
        h_mod = api.kernel32.GetModuleHandleW(None)
        errors = []
        for hook_id, cb, name in ((_WH_MOUSE_LL, _mouse_cb, "mouse"),
                                  (_WH_KEYBOARD_LL, _kbd_cb, "keyboard")):
            h = user32.SetWindowsHookExW(hook_id, cb, h_mod, 0)
            if h:
                hooks.append(h)
            else:
                errors.append(f"{name} err {api.ctypes.get_last_error()}")
        if errors:
            _warn_once("SetWindowsHookEx returned NULL (" + ", ".join(errors) + ")")
            return
        with _lock:
            _last_hook_event = _now_mono()   # health-check baseline: alive now
        _hook_ok = True

        # Message pump — REQUIRED for LL hooks to deliver. GetMessage blocks the
        # thread until a message arrives (the hooks themselves wake it); it
        # returns 0 on WM_QUIT and -1 on error, both of which end the pump.
        msg = api.wintypes.MSG()
        p_msg = api.ctypes.byref(msg)
        while user32.GetMessageW(p_msg, None, 0, 0) > 0:
            user32.TranslateMessage(p_msg)
            user32.DispatchMessageW(p_msg)
        _warn_once("hook message pump ended")
    except Exception as e:
        _warn_once(f"hook thread error: {e}")
    finally:
        _hook_ok = False                     # → the polling fallback takes over
        for h in hooks:
            try:
                user32.UnhookWindowsHookEx(h)
            except Exception:
                pass
