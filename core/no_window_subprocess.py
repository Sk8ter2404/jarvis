"""Process-wide CREATE_NO_WINDOW safety net for console-subsystem spawns.

JARVIS runs as pythonw (GUI subsystem, no console). On Windows, ANY console
app it spawns without a no-window flag makes the OS allocate a console and
pop a visible window — and on this box the default-terminal delegation
routes that to Windows Terminal, which piles up "ghost" windows the owner
can't close. v2.0.32 fixed the 9 spawn sites an audit confirmed, but the
ghosts came back within hours through sites the audit missed (the unified
HUD's ~3s nvidia-smi utilization poll, the monolith's GPU-temp poll, a
health ping): ~38 windows in 30 minutes. Fixing sites one by one loses to
entropy — every future skill is one bare subprocess.run() away from
re-introducing the leak.

install() patches subprocess.Popen.__init__ so a spawn that specifies
NEITHER creationflags NOR startupinfo gets CREATE_NO_WINDOW by default:

  • subprocess.run/call/check_output/check_call all route through Popen,
    so one patch covers every stdlib entry point.
  • A caller that passes ANY creationflags (DETACHED_PROCESS,
    CREATE_NEW_CONSOLE, its own CREATE_NO_WINDOW…) is left untouched —
    deliberate console windows stay possible, explicitly.
  • A caller that passes startupinfo is left untouched (it is already
    managing window visibility via STARTF_USESHOWWINDOW).
  • GUI-subsystem children (pythonw, .pyw overlays) ignore the flag, so
    blanket application is harmless to them.
  • No-op on non-Windows and on double-install.

Call install() ONCE, as early as possible, in EVERY JARVIS process that can
spawn helpers: the monolith, the HUD/reticle/air-cursor overlays, the tray.
2026-07-10."""
from __future__ import annotations

import os
import subprocess

# The original Popen.__init__, kept for uninstall() and for tests. A
# single-element list per house style (mutated, never rebound).
_ORIG_INIT = [None]


def install() -> bool:
    """Activate the safety net. Returns True if (already) active."""
    if os.name != "nt":
        return False
    if _ORIG_INIT[0] is not None:
        return True          # already installed — idempotent
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    orig = subprocess.Popen.__init__
    _ORIG_INIT[0] = orig

    def _no_window_init(self, *args, **kwargs):
        if not kwargs.get("creationflags") and kwargs.get("startupinfo") is None:
            kwargs["creationflags"] = create_no_window
        return orig(self, *args, **kwargs)

    subprocess.Popen.__init__ = _no_window_init
    return True


def uninstall() -> None:
    """Restore the stock Popen (tests only)."""
    if _ORIG_INIT[0] is not None:
        subprocess.Popen.__init__ = _ORIG_INIT[0]
        _ORIG_INIT[0] = None
