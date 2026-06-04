"""apple_music_keeper — optional autostart + keep-alive for the UWP Apple Music app.

WHY THIS EXISTS
===============
The Microsoft-Store **Apple Music** app (process ``AppleMusic.exe``) has no
system tray of its own, so the user asked JARVIS to "keep Apple Music always
open" and host the controls in JARVIS's tray. This module is the keep-it-open
half: behind two opt-in flags it (1) launches the app once at boot and (2)
re-launches it if it ever gets closed.

It is deliberately the MINIMAL, LEGITIMATE mechanism — it only ever calls
``apple_music_app.launch()`` (explorer ``shell:AppsFolder`` AUMID). There is NO
UI automation here: it never clicks the app's buttons, types into it, sends
app-specific shortcuts, or moves/minimises its window. That automation is
policy-restricted; transport stays on OS media keys (the existing
``media_playpause`` / ``media_next`` / ``media_prev`` actions).

SAFETY / DESIGN (mirrors the other opt-in daemons, e.g. skills/kinect_gestures)
==============================================================================
  * Both flags (``APPLE_MUSIC_AUTOSTART`` / ``APPLE_MUSIC_KEEP_OPEN``) default
    OFF in core.config; nothing launches unless the user opts in.
  * HARD-GATED OFF in staging/test: ``_is_staging()`` (JARVIS_STAGING env or the
    monolith's own ``_is_staging``) short-circuits every launch, so the test
    suite and the blue/green staging box NEVER pop the real app open.
  * The keep-alive loop only (re)launches when the app is NOT already running —
    so it never steals focus on a tick where the app is already up.
  * Idempotent: ``start_keeper()`` guards by OS thread name, so a skill reload
    can't spawn a second keeper loop.
  * Never raises into the caller: a missing dependency, a dead shell-out, or an
    absent app degrades to a quiet no-op. ``start_keeper()`` never delays boot
    (the launch + the loop both run on a background daemon thread).
  * TERMINABLE: the keep-alive loop waits on a module-level ``_STOP`` event
    instead of sleeping, so ``stop_keeper()`` wakes it and it exits cleanly.
    ``stop_keeping_music_open`` (the voice toggle) and test teardown both call
    it, so disabling the feature really stops the watchdog and no test can leak
    the daemon into a later test.

The bobert startup wires this in via ``start_keeper()`` (gated on the flags),
next to where the other bridges/daemons are started.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

# How often the keep-alive loop checks that the app is still running. 120s is
# gentle — the app dying is rare, and we only relaunch when it's actually gone.
KEEP_ALIVE_INTERVAL_SECONDS = 120.0
# Let the monolith + tray come up before the first keep-alive check fires, so a
# boot-time autostart launch isn't immediately double-checked.
INITIAL_DELAY_SECONDS = 30.0
_THREAD_NAME = "apple-music-keeper"

# Stop signal for the keep-alive loop. The loop waits on this event instead of
# sleeping, so ``stop_keeper()`` can wake it instantly and let it exit cleanly
# (no more unkillable daemon that outlives a shutdown — or a test). It starts
# SET to "not stopping" semantics via clear() in start_keeper(); a fresh process
# has it clear, which is fine because the loop isn't running yet.
_STOP = threading.Event()


def _app():
    """The lazy ``audio.apple_music_app`` bridge, or None. Prefer the instance
    the monolith already imported; fall back to a direct import. Never raises."""
    mod = sys.modules.get("audio.apple_music_app")
    if mod is not None:
        return mod
    try:
        from audio import apple_music_app as _am
        return _am
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None — for the staging gate."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — the keeper must NEVER launch the real
    app there. Matches the monolith's own gate plus the raw env var so the check
    holds even before the monolith is importable (mirrors skills/kinect_gestures)."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


# ─── single launch (best-effort, never raises, never in staging) ───────────

def _launch_once() -> bool:
    """Launch the Apple Music app exactly once IF it isn't already running.
    Returns True iff a launch was fired. No-op (returns False) in staging, when
    the bridge is unavailable, or when the app is already up. Never raises."""
    if _is_staging():
        return False
    amapp = _app()
    if amapp is None:
        return False
    try:
        if amapp.is_running():
            return False
    except Exception:
        # If we can't even tell, don't blindly launch — stay conservative.
        return False
    try:
        ok, err = amapp.launch()
    except Exception as e:
        print(f"  [apple-music-keeper] launch raised: {e}")
        return False
    if ok:
        print("  [apple-music-keeper] launched Apple Music")
    else:
        print(f"  [apple-music-keeper] launch failed: {err}")
    return bool(ok)


# ─── keep-alive loop ───────────────────────────────────────────────────────

def _keep_alive_tick() -> Optional[bool]:
    """One keep-alive check: if KEEP_OPEN is on (and not staging) and the app is
    NOT running, relaunch it. Returns True if it relaunched, False if it did
    nothing because the app was already up, None if disabled/unavailable/staging.
    Factored out (and side-effect-light) so tests can drive it directly without
    the loop. NEVER raises."""
    if not _cfg_flag("APPLE_MUSIC_KEEP_OPEN"):
        return None
    if _is_staging():
        return None
    amapp = _app()
    if amapp is None:
        return None
    try:
        if amapp.is_running():
            return False
    except Exception:
        return None
    # App is gone — relaunch it (only here, so we never steal focus while it's up).
    return _launch_once()


def _keep_alive_loop() -> None:  # pragma: no cover - daemon loop; each tick delegates to _keep_alive_tick (unit-tested directly) and the wait/stop wiring is covered via _STOP
    # Wait out the initial settle window on the stop EVENT, not a bare sleep, so
    # a stop_keeper() during startup makes the loop exit immediately instead of
    # blocking for INITIAL_DELAY_SECONDS. wait() returns True iff the event was
    # set (i.e. we've been asked to stop) — in which case we never enter the loop.
    if _STOP.wait(INITIAL_DELAY_SECONDS):
        return
    # Keep the original cadence: tick first (right after the settle window),
    # THEN wait the interval on the same event. wait() returns True the moment
    # _STOP is set, so the loop unblocks and terminates promptly; it returns
    # False on a normal timeout, which is when we loop round to the next tick.
    while True:
        try:
            _keep_alive_tick()
        except Exception as e:
            print(f"  [apple-music-keeper] keep-alive error: {e}")
        if _STOP.wait(KEEP_ALIVE_INTERVAL_SECONDS):
            return


# ─── public entry point (wired into bobert startup) ────────────────────────

def start_keeper() -> bool:
    """Start the Apple Music autostart + keep-alive behaviour, both opt-in.

    * When ``APPLE_MUSIC_AUTOSTART`` is on, launch the app once (in the
      background — never blocks/delays boot).
    * When ``APPLE_MUSIC_KEEP_OPEN`` is on, start the daemon keep-alive loop.

    Returns True iff a keeper thread was started (the loop) — for tests/logging.
    Hard no-op in staging/test and when both flags are off. Idempotent: a
    second call won't spawn a duplicate loop (guarded by OS thread name, like
    the other JARVIS daemons). Never raises."""
    if _is_staging():
        return False

    autostart = _cfg_flag("APPLE_MUSIC_AUTOSTART")
    keep_open = _cfg_flag("APPLE_MUSIC_KEEP_OPEN")
    if not autostart and not keep_open:
        return False

    # Autostart launch runs OFF the boot thread so a slow explorer shell-out
    # can never delay startup. is_running() is checked inside _launch_once.
    if autostart:
        try:
            threading.Thread(
                target=_launch_once, daemon=True,
                name="apple-music-autostart").start()
        except Exception as e:
            print(f"  [apple-music-keeper] autostart spawn failed: {e}")

    if not keep_open:
        return False

    # Idempotent: don't spawn a second keep-alive loop on a skill reload.
    if any(th.name == _THREAD_NAME and th.is_alive()
           for th in threading.enumerate()):
        print("  [apple-music-keeper] keep-alive already running — skipping duplicate")
        return False
    # Clear any prior stop signal so the freshly-spawned loop actually runs
    # (a previous stop_keeper() would otherwise make it exit on its first wait).
    _STOP.clear()
    try:
        threading.Thread(target=_keep_alive_loop, daemon=True,
                         name=_THREAD_NAME).start()
        print("  [apple-music-keeper] keep-alive active "
              f"(every ~{KEEP_ALIVE_INTERVAL_SECONDS:.0f}s; opt-in via "
              "APPLE_MUSIC_KEEP_OPEN)")
        return True
    except Exception as e:
        print(f"  [apple-music-keeper] keep-alive spawn failed: {e}")
        return False


def stop_keeper(timeout: float = 2.0) -> bool:
    """Stop the keep-alive watchdog if it's running: signal the loop to exit and
    join its thread (briefly). Idempotent and safe to call even when nothing is
    running — it just sets the stop event. Returns True iff no live keep-alive
    thread remains afterwards. Never raises.

    Used by the ``stop_keeping_music_open`` voice action (so disabling actually
    stops the watchdog, not just flips the flag) and by test teardown (so a test
    that spawned a real loop can't leak the daemon into a later test)."""
    _STOP.set()
    try:
        # Join EVERY live keep-alive thread (normally at most one, but be
        # thorough). The single _STOP event releases them all; each exits after
        # one wait() wake, so the join returns promptly.
        for th in list(threading.enumerate()):
            if (th.name == _THREAD_NAME and th.is_alive()
                    and th is not threading.current_thread()):
                try:
                    th.join(timeout)
                except Exception:
                    pass
    except Exception:
        # Enumerating/joining threads should never fail, but stay graceful.
        return False
    # Report whether the watchdog is truly gone now.
    try:
        return not any(th.name == _THREAD_NAME and th.is_alive()
                       and th is not threading.current_thread()
                       for th in threading.enumerate())
    except Exception:
        return False
