"""Bounce JARVIS: kill the running bobert_companion process and relaunch it
with the user's ANTHROPIC_API_KEY (read from HKCU\\Environment) so the Claude
bonus stays armed. Used when the PowerShell launch path is unavailable.

Only the LIVE instance is touched: a staging one (a blue/green "green"
candidate, a sweep, a test run) is left alone (audit A103)."""
import os
import sys
import time
import subprocess

PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
if not os.path.exists(PYW):
    PYW = sys.executable
JARVIS_DIR = r"C:\JARVIS"

# A staging JARVIS carries the same marks everywhere in the repo: --staging on
# its command line (the prod-only killers in tools/multi_agent_pipeline.py and
# upgrade_jarvis.py) or JARVIS_STAGING=1 in its
# environment (core.paths.is_staging, blue_green_manager.is_staging, the
# monolith's singleton lock).
STAGING_FLAG = "--staging"


def _is_live_jarvis(p) -> bool:
    """True for a live (prod) bobert_companion python process only."""
    cl = " ".join(p.info.get("cmdline") or [])
    nm = (p.info.get("name") or "").lower()
    if "bobert_companion" not in cl or "python" not in nm:
        return False
    if STAGING_FLAG in cl:
        return False
    try:
        env = p.environ()
    except Exception:
        env = None  # unreadable: the command line is all we have
    if env and (env.get("JARVIS_STAGING") or "").strip() == "1":
        return False
    return True


def kill_live_jarvis() -> list:
    """Kill the running live JARVIS; returns the killed PIDs."""
    killed = []
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if _is_live_jarvis(p):
                    p.kill()
                    killed.append(p.pid)
            except Exception:
                pass
    except Exception as e:
        print("psutil kill failed:", e)
    return killed


def main() -> int:
    # 1. Kill the running JARVIS (only the live bobert_companion process).
    killed = kill_live_jarvis()
    print("killed:", killed)
    time.sleep(2.5)

    # 2. Read the user's API key from the registry (User-scope env var).
    env = os.environ.copy()
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        val, _ = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")
        if val:
            env["ANTHROPIC_API_KEY"] = val
            print("api key injected (len %d)" % len(val))
    except Exception as e:
        print("key read failed (will boot credits-optional):", e)

    # 3. Relaunch detached.
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        p = subprocess.Popen([PYW, "bobert_companion.py"], cwd=JARVIS_DIR,
                             env=env, creationflags=flags, close_fds=True)
        print("relaunched pid", p.pid)
    except Exception as e:
        print("relaunch failed:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
