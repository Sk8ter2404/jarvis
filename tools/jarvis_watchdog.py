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
  • data/watchdog_disabled.flag — manual master off-switch, now EXPIRING
    (see _disable_flag_holds): honoured for 4h off its mtime unless the file
    body says 'permanent'. An off-switch with no expiry is how a resurrection
    net dies quietly.
  • data/upgrade_in_progress.flag — an upgrade run is rewriting the source
    tree (see _upgrade_in_progress). Written by upgrade_jarvis.py before its
    first mutating step and carrying its own absolute deadlines.
  • No flag + no process = crash / driver swap / external kill (live case
    2026-07-10 11:32: an iCUE reinstall swapped the audio stack under
    JARVIS's WASAPI streams and the process vanished traceless) → boot.

H-7 (2026-08-20) — why the two upgrade gates exist. "Dead + no clean-shutdown
flag" is ALSO the normal state for hours in the middle of an upgrade run, and
it is not a crash. The overnight path does write clean_shutdown.flag on its
way out, so the watchdog correctly stands down when the pipeline STARTS — but
the flag does not survive the run: the pipeline's tester stage boots a real
prod JARVIS for its smoke test (bobert_companion's boot path DELETES the flag
unconditionally) and then kills it with Stop-Process -Force, which never
reaches atexit, and no killer writes the flag back. From the first task's
tester stage onward the watchdog therefore booted a full interactive JARVIS —
mic live, TTS unmuted — every 5 minutes out of a tree the implementer stage
was mid-rewrite on. watchdog.log 2026-08-08 shows the shape: 03:39 resurrect
→ 03:41 boot failed (TimeoutExpired) → 03:44 resurrect again.

EVERY GATE HERE IS BOUNDED, ON PURPOSE. The mirror-image bug already cost this
project six days of downtime (2026-07-15 → 07-21): a crash left
clean_shutdown.flag behind and the watchdog politely declined to resurrect
anything. A gate that can latch on forever is worse than the bug it closes.
So: the marker carries expiry timestamps and is DELETED once abandoned, the
manual off-switch expires off its mtime, and _pipeline_running() fails OPEN on
every uncertain read because it has no expiry of its own.

Register (once, as the logged-in user):
  schtasks /Create /SC MINUTE /MO 5 /TN "JARVIS Watchdog" /F /TR
    "<pythonw.exe> C:\\JARVIS\\tools\\jarvis_watchdog.py"
"""
from __future__ import annotations

import json
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


# How long an UNPARSEABLE marker is honoured off its mtime. os.replace() makes
# a torn read impossible in practice, so this only covers "somebody dropped a
# hand-written file here" — short, and bounded either way.
_MARKER_MALFORMED_GRACE_S = 900

# Default life of the manual off-switch. Nothing in the repo writes
# watchdog_disabled.flag today (it was pure dead config until H-7), so this
# expiry can only ever act on a file a human put there by hand.
_DISABLE_FLAG_MAX_AGE_S = 4 * 3600

# Command lines that mean "an upgrade pipeline process". Four sentinels because
# the run can be carried by any of them: upgrade_jarvis.py itself, either
# rendered loop-driver tempfile (*_jarvis_pipeline_loop.py /
# *_jarvis_upgrade_loop.py, spawned into their own console), or a direct
# tools/multi_agent_pipeline.py invocation.
#
# The upgrade_jarvis.py alternative is anchored to a path separator or
# start-of-string ON PURPOSE: an unanchored 'upgrade_jarvis\.py' also matches
# tests/test_upgrade_jarvis.py, so running the unit suite would have parked the
# resurrection net for the length of the test run. Regex syntax is the subset
# .NET and Python re agree on, so the constant is directly testable.
_PIPELINE_CMDLINE_RE = (r"(^|[\\/])upgrade_jarvis\.py"
                        r"|_jarvis_pipeline_loop"
                        r"|_jarvis_upgrade_loop"
                        r"|multi_agent_pipeline")


def _pid_alive(pid: int, unknown: bool) -> bool:
    """Is `pid` genuinely executing? core.parent_watch is corpse-aware
    (GetExitCodeProcess + WaitForSingleObject) — a kernel-stuck 'terminating
    forever' row keeps its CIM entry until Windows reboots and must not count.

    `unknown` is what to answer when the helper is unavailable or throws, and
    each caller picks it deliberately:
      • the marker gate passes True (assume alive) because expires_at /
        hard_deadline already bound how long the marker can hold;
      • _pipeline_running() passes False (assume dead) because it has NO
        expiry, so an unresolvable pid there could park the net forever."""
    try:
        from core.parent_watch import parent_is_alive
    except Exception:
        return unknown
    try:
        return bool(parent_is_alive(pid))
    except Exception:
        return unknown


def _discard(path: str) -> None:
    """Best-effort unlink of a marker WE own. Never raises."""
    try:
        os.remove(path)
    except OSError:
        pass


def _disable_flag_holds() -> bool:
    """The manual master off-switch — honoured, but with an expiry.

    Absent -> False. Present and younger than JARVIS_WATCHDOG_DISABLE_MAX_AGE_S
    (default 4h, measured from mtime) -> True. Present, older, and NOT marked
    permanent -> ignored, with a log line saying so.

    Rationale: an off-switch with no expiry is exactly how this project lost
    six days in July. `touch` renews it; putting the word 'permanent' (or
    'never') in the file body opts out of the expiry entirely for a deliberate
    long-term disable. The file is never deleted here — a human wrote it, so
    it stays visible; only its authority lapses."""
    path = os.path.join(PROJ, "data", "watchdog_disabled.flag")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            body = f.read(200).lower()
    except OSError:
        body = ""
    if "permanent" in body or "never" in body:
        return True
    try:
        max_age = int(os.environ.get("JARVIS_WATCHDOG_DISABLE_MAX_AGE_S",
                                     str(_DISABLE_FLAG_MAX_AGE_S)))
    except ValueError:
        max_age = _DISABLE_FLAG_MAX_AGE_S
    age = time.time() - mtime
    if age < max_age:
        return True
    _note(f"watchdog_disabled.flag is {int(age)}s old (limit {max_age}s) — "
          f"EXPIRED, resuming resurrection. Touch it to renew, or put the word "
          f"'permanent' in the file to disable the watchdog indefinitely.")
    return False


def _upgrade_in_progress() -> bool:
    """True while an upgrade run says the source tree is being rewritten.

    upgrade_jarvis.acquire_upgrade_marker() writes data/upgrade_in_progress.flag
    BEFORE its first mutating step and refreshes it from a heartbeat thread.
    The file carries its own ABSOLUTE deadlines, so this reader holds no copy
    of the timeout policy — it just compares two numbers to now(). That is
    deliberate: it makes the project's #1 bug class (the stale duplicate — a
    rule fixed in one copy while the other rots) impossible across the two
    files. Retuning a timeout in upgrade_jarvis.py needs no edit here.

    THREE INDEPENDENT WAYS THIS RELEASES, so a crashed upgrade cannot park the
    resurrection net forever:
      expires_at    short lease (default 10 min), refreshed every 60s;
      hard_deadline absolute ceiling (default 6h), never extended;
      pid           owner gone -> abandoned immediately.
    On any of them the marker is DELETED and the watchdog re-arms on this same
    tick. An absent or unreadable marker never blocks."""
    path = os.path.join(PROJ, "data", "upgrade_in_progress.flag")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return False        # absent, or unreadable — never park the net on it
    now = time.time()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("marker is not a JSON object")
        expires = float(data.get("expires_at", 0) or 0)
        hard = float(data.get("hard_deadline", 0) or 0)
        pid = data.get("pid")
    except (TypeError, ValueError):
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            return False
        if age < _MARKER_MALFORMED_GRACE_S:
            _note(f"upgrade marker unreadable but only {int(age)}s old — "
                  f"standing down for up to {_MARKER_MALFORMED_GRACE_S}s")
            return True
        _note("upgrade marker unreadable AND stale — discarding it and "
              "re-arming the resurrection net")
        _discard(path)
        return False

    owner_alive = (not isinstance(pid, int) or pid <= 0
                   or _pid_alive(pid, unknown=True))
    if expires and now < expires and (not hard or now < hard) and owner_alive:
        _note(f"upgrade in progress (pid {pid}, lease {int(expires - now)}s "
              f"left) — NOT resurrecting into a half-written tree")
        return True

    if not expires or now >= expires:
        why = "lease lapsed"
    elif hard and now >= hard:
        why = f"past the {int(hard - float(data.get('started_at', hard)))}s ceiling"
    else:
        why = f"owner pid {pid} is gone"
    _note(f"ABANDONED upgrade marker ({why}) — discarding it and re-arming "
          f"the resurrection net")
    _discard(path)
    return False


def _pipeline_running() -> bool:
    """True when an upgrade pipeline process is GENUINELY EXECUTING.

    Belt-and-braces beside _upgrade_in_progress(), and it covers a window the
    marker cannot: invoked WITHOUT --relaunch, upgrade_jarvis.main() returns
    as soon as it has spawned the loop driver, so its marker is released while
    the driver process keeps rewriting the tree for hours. The driver is a
    python.exe running the rendered *_jarvis_pipeline_loop.py tempfile, which
    imports tools/multi_agent_pipeline.py — hence the four cmdline sentinels.

    Needs no writer and self-clears the instant the process dies, so it can
    never leave the net parked. Same CIM shape bobert_companion already uses
    as its pipeline singleton guard ("scan for any running upgrade_jarvis.py;
    if found, skip").

    FAILS OPEN (returns False) on every uncertain read — timeout, exception,
    corpse-only rows, or an unavailable liveness helper. This predicate has no
    expiry of its own, so it must never be able to block on a doubt; that is
    the park-forever hazard. _jarvis_running() already fails safe on an
    unreadable process table, so the tick is still protected.

    The cmdline pattern is the module constant _PIPELINE_CMDLINE_RE so a unit
    test can prove `test_upgrade_jarvis.py` does not match it."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR "
             "Name='python.exe'\" | Where-Object { $_.CommandLine -match "
             "'" + _PIPELINE_CMDLINE_RE + "' } | ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=30,
        )
        pids = [int(p) for p in (out.stdout or "").split() if p.strip().isdigit()]
    except Exception:
        return False
    pids = [p for p in pids if p != os.getpid()]
    if not pids:
        return False
    live = [p for p in pids if _pid_alive(p, unknown=False)]
    if not live:
        _note(f"only CORPSE pipeline pids present {pids} — NOT treating an "
              f"upgrade as in flight")
        return False
    _note(f"upgrade pipeline still running (pids {live}) — NOT resurrecting "
          f"into a half-written tree")
    return True


def main() -> int:
    if _disable_flag_holds():
        return 0
    if os.path.exists(os.path.join(PROJ, "data", "clean_shutdown.flag")):
        return 0
    # H-7 gate: an upgrade run is rewriting the tree, so "dead + no flag" is
    # the EXPECTED state, not a crash. Checked before _jarvis_running() on
    # purpose — it is a single file read, it cannot be defeated by an
    # unreadable process table, and it is the one gate that carries its own
    # expiry.
    if _upgrade_in_progress():
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
    # Last chance before we spend a boot: catch an upgrade whose marker is
    # gone but whose driver process is demonstrably still rewriting the tree
    # (a plain no-relaunch invocation). Costs one PowerShell spawn, and only on
    # a tick that would otherwise have resurrected.
    if _pipeline_running():
        return 0
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
