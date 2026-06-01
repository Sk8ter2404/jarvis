"""
itunes_bridge — lazy iTunes COM client.

WHY THIS MODULE EXISTS
======================
iTunes was opening on every JARVIS boot. The cause: the apple_music_intel
listen loop and a couple of other call sites were calling _get_itunes()
unconditionally, and _get_itunes() in turn called win32com.client.Dispatch
("iTunes.Application"). Dispatch SPAWNS iTunes.exe via its COM server when
no running instance exists — that single line is "music command pops iTunes
open by surprise" in concentrated form.

The fix has two parts:

  1. NEVER import or touch win32com / pythoncom at module-import time. The
     `import` statements live inside `get_client()`, after the gating check.
     Importing this bridge from anywhere — bobert_companion, apple_music_intel,
     a hud widget — has zero cost beyond loading three Python functions.

  2. `get_client()` short-circuits and returns (None, error) unless one of
     two preconditions holds:
       a) `is_running()` returns True — iTunes.exe is already in the process
          list, so binding to it via GetActiveObject is safe and won't spawn
          a new instance.
       b) The caller passed `force=True` — used ONLY for explicit
          iTunes-keyed user intents (e.g. `play_music, library:<query>` or
          the `play_unheard` action). Even with force=True, we still
          honour the auto-launch gate before invoking subprocess.Popen.

CONFIGURATION
=============
The bridge has two configuration hooks set by bobert_companion at startup:

  set_auto_launch(True/False)  — mirror of ITUNES_AUTO_LAUNCH. When False
      (the default), `get_client()` will NOT spawn iTunes.exe even when
      `force=True` — it returns a friendly error instead. When True, an
      explicit iTunes-keyed action may launch iTunes if it isn't running.

  set_apple_music_active_check(callable) — predicate returning True when
      the user's actual player is Apple Music in a browser tab. When set
      and force=False, `get_client()` returns early without touching COM,
      because routing music actions to iTunes while Apple Music owns the
      audio session would steal it. force=True bypasses this guard (the
      user typed `library:` — they really mean iTunes COM).

Both hooks are optional: with neither installed, get_client() behaves as
"only attach when iTunes is already running, otherwise return an error."

PER-THREAD COM BOOKKEEPING
==========================
Preserved from the original bobert_companion implementation: CoInitialize
is reference-counted per thread, and `get_client` is called repeatedly by
apple_music_intel's polling loop (every POLL_SECONDS = 12s). Calling
CoInitialize on every call without a matching CoUninitialize ratchets the
apartment refcount up indefinitely; eventually OLE RPC channels exhaust
and the next allocation tramples heap. Pairing every CoInitialize with a
try/finally CoUninitialize inside `get_client` is not viable — the
function returns a live COM object the caller dereferences AFTER
get_client exits, and CoUninitialize would tear down the apartment.

So: CoInitialize exactly once per thread (TLS flag set, subsequent calls
no-op) and rely on Python thread cleanup to balance the refcount when the
thread terminates. The polling thread is long-lived, so this caps the
leak at one refcount per thread rather than one per call.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable, Optional


# ─── configuration hooks ──────────────────────────────────────────────────

_AUTO_LAUNCH: bool = False
_apple_music_active_check: Optional[Callable[[], bool]] = None


def set_auto_launch(enabled: bool) -> None:
    """Mirror bobert_companion's ITUNES_AUTO_LAUNCH into the bridge so a
    cold-start launch is only attempted when the user opted in."""
    global _AUTO_LAUNCH
    _AUTO_LAUNCH = bool(enabled)


def get_auto_launch() -> bool:
    return _AUTO_LAUNCH


def set_apple_music_active_check(fn: Optional[Callable[[], bool]]) -> None:
    """Install the Apple-Music-in-browser predicate used to short-circuit
    iTunes COM calls when the user's actual player is the web app. Pass
    None to clear."""
    global _apple_music_active_check
    _apple_music_active_check = fn


# ─── process inspection ───────────────────────────────────────────────────

_ITUNES_EXE_CANDIDATES = (
    r"C:\Program Files\iTunes\iTunes.exe",
    r"C:\Program Files (x86)\iTunes\iTunes.exe",
)


def find_itunes_exe() -> str | None:
    for p in _ITUNES_EXE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def is_running() -> bool:
    """Cheap psutil scan — True iff iTunes.exe is in the process list.
    Used as the primary gate: if iTunes isn't running, we never touch
    win32com / pythoncom (calling Dispatch would spawn it)."""
    try:
        import psutil
    except ImportError:
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "itunes.exe":
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def launch() -> tuple[bool, str | None]:
    """Spawn iTunes.exe detached. Returns (launched, error_msg).
    Only called from inside get_client() when the auto-launch gate is on."""
    exe = find_itunes_exe()
    if exe is None:
        return False, "iTunes.exe not found in Program Files"
    try:
        creationflags = 0
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — iTunes survives
            # our exit and doesn't inherit our console.
            creationflags = 0x00000008 | 0x00000200
        subprocess.Popen(
            [exe],
            creationflags=creationflags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return True, None
    except Exception as e:
        return False, f"failed to launch iTunes: {e}"


# ─── per-thread COM init bookkeeping ──────────────────────────────────────

_itunes_com_tls = threading.local()


def _ensure_com_init(pythoncom_mod) -> None:
    if not getattr(_itunes_com_tls, "initialized", False):
        pythoncom_mod.CoInitialize()
        _itunes_com_tls.initialized = True


# ─── the main entry point ─────────────────────────────────────────────────

def get_client(wait_for_ready: bool = True, timeout: float = 12.0,
               force: bool = False):
    """Return the iTunes COM Application object, or (None, error_msg).

    Gating order (each guard returns early without touching COM):

      1. Apple-Music-in-browser check (if installed and force=False) —
         routing music actions to iTunes while the user is using the
         Apple Music web app would steal the audio session.

      2. is_running() — if iTunes.exe isn't in the process list AND
         auto-launch is disabled, return early with a friendly error
         BEFORE importing win32com / pythoncom at all. This is the
         specific guard that fixes "iTunes opens on every boot": the
         apple_music_intel polling thread (and other lazy callers) hit
         this branch on a clean boot and never touch COM.

      3. Auto-launch gate — only when force=True AND we still couldn't
         bind to a running instance do we spawn iTunes.exe via
         subprocess.Popen, then poll Dispatch until it succeeds.

    `force=True` is the "user said iTunes" escape hatch. It bypasses the
    Apple-Music-active short-circuit (the user is being explicit) but
    still honours the auto-launch gate (we don't spawn cold without
    explicit opt-in).
    """
    if not force and _apple_music_active_check is not None:
        try:
            if _apple_music_active_check():
                return None, ("Apple Music is the active player — iTunes COM "
                              "is intentionally not initialised. Music actions "
                              "route through browser media keys.")
        except Exception:
            # If the predicate itself crashes, treat as "not active" rather
            # than failing the whole call — the in_running check below is
            # the load-bearing guard.
            pass

    running = is_running()
    if not running and not _AUTO_LAUNCH:
        return None, ("iTunes isn't running — auto-launch is disabled "
                      "(set ITUNES_AUTO_LAUNCH=True to enable). Open "
                      "iTunes manually, or use apple_music for browser "
                      "streaming.")

    try:
        import win32com.client
        import pythoncom
    except ImportError:
        return None, "pywin32 not installed — pip install pywin32"

    _ensure_com_init(pythoncom)

    app = None
    dispatch_err: Exception | None = None

    if running:
        try:
            app = win32com.client.GetActiveObject("iTunes.Application")
        except Exception as ex:
            dispatch_err = ex
            # iTunes is running but GetActiveObject couldn't bind (rare —
            # usually a ROT registration race during cold-start). Fall back
            # to Dispatch. Safe because iTunes is already running, so
            # Dispatch attaches to the existing instance rather than
            # spawning a new one.
            try:
                app = win32com.client.Dispatch("iTunes.Application")
                dispatch_err = None
            except Exception as ex2:
                dispatch_err = ex2

    # iTunes is not running — only proceed if the user has opted into
    # auto-launch. The pre-import bail above already short-circuits this
    # branch unless _AUTO_LAUNCH is True; this is defence in depth.
    launched = False
    if app is None and not running:
        if not _AUTO_LAUNCH:
            return None, ("iTunes isn't running — auto-launch is disabled "
                          "(set ITUNES_AUTO_LAUNCH=True to enable). Open "
                          "iTunes manually, or use apple_music for browser "
                          "streaming.")
        ok, launch_err = launch()
        if not ok:
            return None, f"could not launch iTunes; {launch_err}"
        launched = True
        # Give iTunes a moment to start its COM server before dispatching.
        launch_deadline = time.time() + 30.0
        while time.time() < launch_deadline:
            time.sleep(1.0)
            try:
                app = win32com.client.Dispatch("iTunes.Application")
                dispatch_err = None
                break
            except Exception as ex:
                dispatch_err = ex
                app = None
        if app is None:
            return None, f"launched iTunes but COM never became available: {dispatch_err}"

    if app is None:
        return None, f"could not connect to iTunes: {dispatch_err}"

    if not wait_for_ready:
        return app, None

    # Poll until the library is actually accessible. COM Dispatch can
    # return an iTunes object that's still launching, and any library
    # call will throw E_UNEXPECTED until the library finishes loading.
    # When we had to launch iTunes ourselves, extend the timeout — cold
    # start can take 30+ seconds depending on library size.
    effective_timeout = max(timeout, 45.0) if launched else timeout
    deadline = time.time() + effective_timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            lib = app.LibraryPlaylist
            _ = lib.Tracks.Count   # touch count so any deferred init triggers
            return app, None
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return None, f"iTunes did not become ready within {effective_timeout}s ({last_err})"
