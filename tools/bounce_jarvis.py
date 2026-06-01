"""Bounce JARVIS: kill the running bobert_companion process and relaunch it
with the user's ANTHROPIC_API_KEY (read from HKCU\\Environment) so the Claude
bonus stays armed. Used when the PowerShell launch path is unavailable."""
import os
import sys
import time
import subprocess

PYW = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
if not os.path.exists(PYW):
    PYW = sys.executable
JARVIS_DIR = r"C:\JARVIS"

# 1. Kill the running JARVIS (only the bobert_companion process).
killed = []
try:
    import psutil
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            nm = (p.info.get("name") or "").lower()
            if "bobert_companion" in cl and ("python" in nm):
                p.kill()
                killed.append(p.pid)
        except Exception:
            pass
except Exception as e:
    print("psutil kill failed:", e)
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
    sys.exit(1)
