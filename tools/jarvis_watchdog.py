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
    """True when a pythonw/python process is running bobert_companion.
    tasklist would need per-process command lines, so use PowerShell's CIM
    query — spawned WITHOUT a window via the safety net."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR "
             "Name='python.exe'\" | Where-Object { $_.CommandLine -match "
             "'bobert_companion' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=30,
        )
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:
        return True     # fail SAFE: never double-boot on an uncertain read


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
