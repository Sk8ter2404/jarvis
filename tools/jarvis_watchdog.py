"""jarvis_watchdog.py — resurrect JARVIS after an UNINTENDED death.

Runs from a Windows scheduled task every 5 minutes UNDER PYTHONW — the GUI
subsystem allocates no console at all, so the watchdog itself can never
flash or leak a terminal window (the first .ps1 version ran via powershell
and left a Windows Terminal window per tick when its exit code was
nonzero). All child spawns inherit CREATE_NO_WINDOW via the same net the
app uses.

Semantics (see also the atexit handshake in bobert_companion.main):
  • data/clean_shutdown.flag  — present = the owner MEANT to stop JARVIS
    (clean exits write it via atexit; boot deletes it) → stay dead.
  • data/watchdog_disabled.flag — manual master off-switch.
  • No flag + no process = crash / driver swap / external kill (live case
    2026-07-10 11:32: an iCUE reinstall swapped the audio stack under
    JARVIS's WASAPI streams and the process vanished traceless) → boot.

Register (once, as the logged-in user):
  schtasks /Create /SC MINUTE /MO 5 /TN "JARVIS Watchdog" /F /TR
    "<pythonw.exe> C:\\JARVIS\\tools\\jarvis_watchdog.py"
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJ)
try:
    from core.no_window_subprocess import install as _install_no_window
    _install_no_window()
except Exception:
    pass


def _note(msg: str) -> None:
    try:
        with open(os.path.join(PROJ, "logs", "watchdog.log"), "a",
                  encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")
    except Exception:
        pass


def _jarvis_running() -> bool:
    """True when a bobert_companion process is GENUINELY EXECUTING.

    CORPSE BLINDNESS (fixed 2026-07-14): this used to COUNT the CIM rows and
    call any count > 0 alive. But a kernel-stuck 'terminating forever'
    process — a thread parked in a CUDA/audio driver at exit — keeps its row
    enumerable FOREVER (until Windows reboots), with its command line intact.
    So a single corpse permanently convinced the watchdog that JARVIS was
    running, and the resurrection net silently stopped resurrecting: JARVIS
    died at 10:49 today and the 5-minute ticks all no-opped against two
    corpses from yesterday. Ask each PID whether it is really alive
    (core.parent_watch: GetExitCodeProcess + WaitForSingleObject) instead of
    trusting the row's existence."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR "
             "Name='python.exe'\" | Where-Object { $_.CommandLine -match "
             "'bobert_companion' } | ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=30,
        )
        pids = [int(p) for p in (out.stdout or "").split() if p.strip().isdigit()]
    except Exception:
        return True     # fail SAFE: never double-boot on an uncertain read
    if not pids:
        return False
    try:
        from core.parent_watch import parent_is_alive
    except Exception:
        # Helper unavailable — fall back to the historical (corpse-blind)
        # behaviour rather than risking a double boot.
        return True
    live = [p for p in pids if parent_is_alive(p)]
    corpses = [p for p in pids if p not in live]
    if corpses and not live:
        _note(f"only CORPSE pids present {corpses} — treating JARVIS as DEAD "
              f"(kernel-stuck rows never disappear until reboot)")
    return bool(live)


def main() -> int:
    if os.path.exists(os.path.join(PROJ, "data", "watchdog_disabled.flag")):
        return 0
    if os.path.exists(os.path.join(PROJ, "data", "clean_shutdown.flag")):
        return 0
    if _jarvis_running():
        return 0
    # Grace: a boot may be mid-flight — newest session log written <90s ago.
    try:
        logs = [os.path.join(PROJ, "logs", f)
                for f in os.listdir(os.path.join(PROJ, "logs"))
                if f.startswith("session_") and f.endswith(".log")]
        newest = max(logs, key=os.path.getmtime) if logs else None
        if newest and time.time() - os.path.getmtime(newest) < 90:
            return 0
    except Exception:
        pass
    _note("JARVIS not running and no clean-shutdown flag — resurrecting.")
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", os.path.join(PROJ, "_boot_jarvis.ps1")],
            capture_output=True, text=True, timeout=120, cwd=PROJ,
        )
        _note("boot script invoked.")
    except Exception as e:
        _note(f"boot failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
