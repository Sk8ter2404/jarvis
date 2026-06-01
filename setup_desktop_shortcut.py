#!/usr/bin/env python3
r"""Create / refresh the JARVIS Windows shortcuts.

What this writes:
  • Desktop\J.A.R.V.I.S..lnk
        Launches _boot_jarvis.ps1 (PROD mode). Uses the arc-reactor ICO
        from assets/jarvis_icon.ico so the shortcut reads as JARVIS at a
        glance.
  • Optional: Startup\J.A.R.V.I.S..lnk  (--startup flag)
        Same target, dropped into the user's Startup folder so JARVIS
        auto-launches at login.

Usage:
    python setup_desktop_shortcut.py             # desktop only
    python setup_desktop_shortcut.py --startup   # desktop + Startup
    python setup_desktop_shortcut.py --remove    # delete both shortcuts
    python setup_desktop_shortcut.py --staging   # desktop shortcut for staging

Implementation uses the Windows Shell COM API via pywin32 if available,
falling back to a PowerShell one-liner (which uses the same WScript.Shell
COM object) if pywin32 isn't installed. Either path produces a real .lnk
file readable by Explorer and pinnable to the taskbar.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_DIR     = os.path.dirname(os.path.abspath(__file__))
BOOT_SCRIPT     = os.path.join(PROJECT_DIR, "_boot_jarvis.ps1")
ICON_PATH       = os.path.join(PROJECT_DIR, "assets", "jarvis_icon.ico")
POWERSHELL_EXE  = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _desktop_dir() -> str:
    """User's Desktop. Prefer the registered shell folder so OneDrive-
    redirected desktops resolve correctly; fall back to %USERPROFILE%."""
    try:
        # Lazy import — winreg is Windows-only; the script is too, but
        # importing at module load would break test imports on other OSes.
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as k:
            value, _ = winreg.QueryValueEx(k, "Desktop")
            # Value may contain %USERPROFILE% — expand env vars.
            return os.path.expandvars(value)
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Desktop")


def _startup_dir() -> str:
    """User's Startup folder (no admin required, per-user auto-start)."""
    appdata = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Roaming",
    )
    return os.path.join(
        appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup",
    )


def _build_lnk_pywin32(lnk_path: str, target: str, args: str,
                      workdir: str, icon: str, description: str) -> bool:
    """Create a .lnk via pywin32's win32com.shell."""
    try:
        from win32com.client import Dispatch  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        shell = Dispatch("WScript.Shell")
        sc = shell.CreateShortCut(lnk_path)
        sc.TargetPath       = target
        sc.Arguments        = args
        sc.WorkingDirectory = workdir
        sc.Description      = description
        sc.IconLocation     = icon
        # WindowStyle: 1 normal, 7 minimised, 3 maximised. PowerShell
        # launches Start-Process -WindowStyle Hidden internally, so the
        # initial PS host can be minimised without flashing.
        sc.WindowStyle      = 7
        sc.Save()
        return True
    except Exception as e:
        print(f"[shortcut] pywin32 path failed: {e}")
        return False


def _build_lnk_powershell(lnk_path: str, target: str, args: str,
                         workdir: str, icon: str, description: str) -> bool:
    """Fallback: shell out to PowerShell which talks to the same COM API.

    Used when pywin32 isn't installed. Slower (process spawn) but no
    extra dependency required.
    """
    if not os.path.exists(POWERSHELL_EXE):
        print(f"[shortcut] powershell.exe not found at {POWERSHELL_EXE}")
        return False

    def _ps_quote(s: str) -> str:
        # PowerShell single-quoted strings escape ' as ''. No other
        # escaping needed since we control the inputs (paths from os.path).
        return "'" + s.replace("'", "''") + "'"

    ps_cmd = (
        f"$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut({_ps_quote(lnk_path)}); "
        f"$sc.TargetPath = {_ps_quote(target)}; "
        f"$sc.Arguments = {_ps_quote(args)}; "
        f"$sc.WorkingDirectory = {_ps_quote(workdir)}; "
        f"$sc.Description = {_ps_quote(description)}; "
        f"$sc.IconLocation = {_ps_quote(icon)}; "
        f"$sc.WindowStyle = 7; "
        f"$sc.Save()"
    )
    try:
        result = subprocess.run(
            [POWERSHELL_EXE, "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"[shortcut] powershell exit {result.returncode}: "
                  f"{result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"[shortcut] powershell path failed: {e}")
        return False


def _create_shortcut(lnk_path: str, target: str, args: str, workdir: str,
                    icon: str, description: str) -> bool:
    """Try pywin32 first, fall back to PowerShell. Either way, the result
    is a real .lnk file Explorer treats as native."""
    os.makedirs(os.path.dirname(lnk_path), exist_ok=True)
    if _build_lnk_pywin32(lnk_path, target, args, workdir, icon, description):
        print(f"[shortcut] wrote {lnk_path} (pywin32)")
        return True
    if _build_lnk_powershell(lnk_path, target, args, workdir, icon, description):
        print(f"[shortcut] wrote {lnk_path} (powershell)")
        return True
    print(f"[shortcut] failed to write {lnk_path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", action="store_true",
                        help="Also drop a shortcut in the user Startup folder")
    parser.add_argument("--staging", action="store_true",
                        help="Create a STAGING launcher (passes -Staging to "
                             "_boot_jarvis.ps1)")
    parser.add_argument("--remove", action="store_true",
                        help="Delete both shortcuts instead of creating")
    parser.add_argument("--name", default=None,
                        help="Override shortcut filename (without .lnk)")
    args = parser.parse_args()

    if not os.path.exists(BOOT_SCRIPT):
        print(f"[shortcut] boot script missing: {BOOT_SCRIPT}")
        return 2

    base_name = args.name or ("J.A.R.V.I.S. (Staging)" if args.staging
                              else "J.A.R.V.I.S.")
    lnk_name  = base_name + ".lnk"
    desktop_lnk = os.path.join(_desktop_dir(), lnk_name)
    startup_lnk = os.path.join(_startup_dir(), lnk_name)

    if args.remove:
        ok = True
        for p in (desktop_lnk, startup_lnk):
            if os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"[shortcut] removed {p}")
                except Exception as e:
                    print(f"[shortcut] could not remove {p}: {e}")
                    ok = False
            else:
                print(f"[shortcut] (no file at {p})")
        return 0 if ok else 1

    # Target is powershell.exe; arguments pin the script path and (for
    # staging) the -Staging flag. -ExecutionPolicy Bypass keeps the
    # shortcut working under default Restricted policy.
    target  = POWERSHELL_EXE
    ps_args = (f"-NoProfile -ExecutionPolicy Bypass "
               f"-File \"{BOOT_SCRIPT}\"")
    if args.staging:
        ps_args += " -Staging"
    icon = ICON_PATH if os.path.exists(ICON_PATH) else target
    desc = ("Launch J.A.R.V.I.S. (staging instance)" if args.staging
            else "Launch J.A.R.V.I.S.")

    ok = _create_shortcut(
        desktop_lnk,
        target=target,
        args=ps_args,
        workdir=PROJECT_DIR,
        icon=icon,
        description=desc,
    )
    if args.startup:
        ok = _create_shortcut(
            startup_lnk,
            target=target,
            args=ps_args,
            workdir=PROJECT_DIR,
            icon=icon,
            description=desc + " (auto-start)",
        ) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
