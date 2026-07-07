"""apple_music_app — lazy bridge to the new UWP **Apple Music** app.

WHY THIS MODULE EXISTS
======================
The classic desktop **iTunes** is gone from this machine: the
``iTunes.Application`` COM server is no longer registered and
``C:\\Program Files\\iTunes\\iTunes.exe`` is absent, so
``audio.itunes_bridge.get_client()`` now always returns ``(None, error)``
and every COM-driven music action is dead.

The replacement is the Microsoft-Store **Apple Music** app
(package ``AppleInc.AppleMusicWin``, process ``AppleMusic.exe``,
AppUserModelID ``AppleInc.AppleMusicWin_nzyj5cx40ttqa!App``). It exposes
**no COM automation surface**, so JARVIS controls it the only legitimate
way: launch it, drive transport with OS-level **media keys**, and read its
window title for a best-effort "now playing". There is deliberately NO
UI automation here (no typing into its search, no clicking its buttons) —
that is policy-restricted. "Play a specific song" stays on the existing
browser ``apple_music`` action (music.apple.com).

DESIGN (mirrors itunes_bridge.py)
=================================
  * NEVER import psutil / subprocess / pygetwindow at module-import time
    in a way that can raise. Every optional dependency is imported lazily
    INSIDE the function that needs it and guarded — importing this bridge
    from anywhere costs nothing beyond loading a handful of functions.
  * Every public function is best-effort and NEVER raises: a missing
    dependency, a dead shell-out, or an absent app degrades to a sensible
    default (``False`` / ``None`` / ``(False, reason)``), never a stack
    trace into the action dispatcher.

PUBLIC API
==========
  aumid()              -> str         the Apple Music AppUserModelID
  is_installed()       -> bool        package present
  is_running()         -> bool        AppleMusic.exe in the process list
  launch()             -> (bool, str|None)   start the app via explorer shell:AppsFolder
  ensure_running(...)  -> (bool, str|None)   launch + poll until running
  now_playing()        -> str|None    best-effort from the window title
  is_active_media_app()-> bool        running and/or a music window present
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

# JARVIS runs as a DETACHED pythonw (no console), so any child that would spawn
# a console flashes a window over the user's work. CREATE_NO_WINDOW keeps these
# helper subprocesses invisible — the rest of the codebase (itunes_bridge et al.)
# already passes it; these three sites were the un-flagged gap that flashed a
# PowerShell/Explorer window on every Apple-Music launch (2026-07-07 bug-hunt).
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# The known AppUserModelID for the Microsoft-Store Apple Music app on this
# machine. `aumid()` returns this unless a dynamic Get-StartApps lookup
# resolves a different one (cached). Kept module-level so tests can assert it.
_KNOWN_AUMID = r"AppleInc.AppleMusicWin_nzyj5cx40ttqa!App"

# The package family prefix and the running process name.
_PACKAGE_PREFIX = "AppleInc.AppleMusicWin"
_PROCESS_NAME = "applemusic.exe"   # compared case-insensitively

# A resolved AUMID is cached here after the first successful lookup so we
# don't shell out to PowerShell on every transport action.
_aumid_cache: list[Optional[str]] = [None]


# ─── AppUserModelID resolution ────────────────────────────────────────────

def _resolve_aumid_via_startapps() -> Optional[str]:
    """Best-effort dynamic lookup of the Apple Music AppID via Get-StartApps.
    Returns the AppID string or None. Never raises — a missing PowerShell, a
    non-zero exit, or an empty result all degrade to None so the caller falls
    back to the known constant."""
    try:
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-StartApps | Where-Object {$_.Name -match 'Apple Music'})"
                ".AppID",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    # Get-StartApps can return multiple lines if more than one app matches;
    # take the first non-empty line.
    for line in out.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def aumid() -> str:
    """Return the Apple Music AppUserModelID.

    Prefers a cached value, then a dynamic Get-StartApps resolution (cached
    on success), then the known constant. Never raises — always returns a
    usable string so `launch()` has something to hand to explorer.exe."""
    if _aumid_cache[0]:
        return _aumid_cache[0]
    resolved = _resolve_aumid_via_startapps()
    if resolved:
        _aumid_cache[0] = resolved
        return resolved
    # Fall back to the known constant (do NOT cache it, so a later call can
    # still pick up a dynamic resolution once the shell is responsive).
    return _KNOWN_AUMID


# ─── process / package inspection ─────────────────────────────────────────

def is_running() -> bool:
    """True iff AppleMusic.exe is in the process list. Cheap psutil scan;
    returns False (never raises) if psutil is absent."""
    try:
        import psutil
    except Exception:
        return False
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                if (proc.info.get("name") or "").lower() == _PROCESS_NAME:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
    except Exception:
        return False
    return False


def is_installed() -> bool:
    """Best-effort check that the Apple Music package is installed. Tries
    Get-AppxPackage; returns False (never raises) if PowerShell is absent or
    the lookup fails. A False here should be treated as 'unknown', not a hard
    'not installed' — launch() still tries the known AUMID regardless."""
    try:
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"(Get-AppxPackage -Name '{_PACKAGE_PREFIX}*').Name",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_NO_WINDOW,
        )
    except Exception:
        return False
    return _PACKAGE_PREFIX.lower() in (proc.stdout or "").lower()


# ─── launch ───────────────────────────────────────────────────────────────

def launch() -> tuple[bool, Optional[str]]:
    """Start the Apple Music app via ``explorer.exe shell:AppsFolder\\<aumid>``.
    Returns (launched, error_msg). Never raises — a failed spawn degrades to
    (False, reason)."""
    target = f"shell:AppsFolder\\{aumid()}"
    try:
        subprocess.Popen(
            ["explorer.exe", target],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=_NO_WINDOW,
        )
        return True, None
    except Exception as e:
        return False, f"failed to launch Apple Music: {e}"


def ensure_running(timeout: float = 8.0, poll: float = 0.5) -> tuple[bool, Optional[str]]:
    """Ensure the Apple Music app is running: if it already is, return
    (True, None) immediately; otherwise launch() and poll is_running() until
    `timeout` seconds elapse. Returns (running, error_msg). Never raises."""
    if is_running():
        return True, None
    ok, err = launch()
    if not ok:
        return False, err
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return True, None
        time.sleep(poll)
    # Launch fired but the process didn't show within the window — explorer
    # may still be starting it. Report optimistically-but-honestly.
    return is_running(), None


# ─── now playing (best-effort, window title only) ─────────────────────────

def _music_window_titles() -> list[str]:
    """Return the titles of visible windows that look like the Apple Music
    app. Best-effort via pygetwindow; [] if pygetwindow is absent or errors.
    Never raises."""
    titles: list[str] = []
    try:
        import pygetwindow as gw
    except Exception:
        return titles
    try:
        for w in gw.getAllWindows():
            t = (getattr(w, "title", "") or "").strip()
            if t and "apple music" in t.lower():
                titles.append(t)
    except Exception:
        return titles
    return titles


def now_playing() -> Optional[str]:
    """Best-effort "what's playing in the Apple Music app".

    Reads the app's window title via pygetwindow and returns the song/artist
    portion when the title carries one. The Apple Music app titles its window
    "<Song> — <Artist>" (or similar) while playing, and just "Apple Music"
    when idle. Returns None when nothing useful is known (idle title, no
    window, or pygetwindow absent). Never raises.
    """
    for title in _music_window_titles():
        low = title.lower().strip()
        # The bare app name (any casing, with/without trailing junk) carries
        # no track info → not useful.
        if low in ("apple music", "apple music app"):
            continue
        # Strip a trailing/leading " - Apple Music" / " — Apple Music" decorator
        # so we surface just the track, mirroring browser-tab title handling.
        cleaned = title
        for sep in (" - Apple Music", " — Apple Music",
                    " | Apple Music", " - apple music"):
            if cleaned.lower().endswith(sep.lower()):
                cleaned = cleaned[: -len(sep)].strip()
                break
        cleaned = cleaned.strip()
        if cleaned and cleaned.lower() not in ("apple music", "apple music app"):
            return cleaned
        # Title was only the decorator — fall through to the next window.
    return None


# ─── active-media-app predicate ───────────────────────────────────────────

def is_active_media_app() -> bool:
    """True when the Apple Music app is the live media app: the process is
    running and/or one of its windows is visible. Used by the transport
    actions to decide whether to drive playback with media keys. Never raises.
    """
    if is_running():
        return True
    # Process scan can miss it (psutil absent); a visible window is a second
    # signal.
    return bool(_music_window_titles())
