#!/usr/bin/env python3
"""First-run setup wizard — configure a fresh JARVIS clone and get it running.

    python tools/setup_wizard.py
    python tools/setup_wizard.py --defaults   # accept every default, just write the files

Walks through the essentials:
  1. ANTHROPIC_API_KEY — the one thing JARVIS can't run without — written to .env
     (gitignored). Skipped if it's already set in the environment or .env.
  2. A few high-impact choices (AI backend, TTS voice, voice mode, and which
     subsystems to enable) -> data/user_settings.json.
  3. Points you at the Settings GUI + SETUP.md for everything else.

Reuses the schema + load/save from tools/settings_window.py, so what the wizard
writes is exactly what the GUI + core.config read. Total/safe: never overwrites a
value you didn't change; all I/O is injectable so it's fully testable + needs no
tkinter, network, or API key to run the tests.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from tools import settings_window as sw

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_ROOT, ".env")

# Curated essentials to prompt for (each must be a key in sw.SCHEMA).
_ESSENTIALS = ["AI_BACKEND", "TTS_VOICE", "VOICE_MODE", "HUD_ENABLED",
               "TRAY_ENABLED", "AMBIENT_LISTENING_ENABLED"]


def env_has_api_key(env_path: str) -> bool:
    """True if ANTHROPIC_API_KEY is set in the process env, or present + non-empty
    in the .env file. Never raises."""
    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return True
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("ANTHROPIC_API_KEY="):
                    return bool(s.split("=", 1)[1].strip())
    except OSError:
        pass
    return False


def upsert_env(env_path: str, key: str, value: str) -> bool:
    """Add or update ``KEY=value`` in .env, preserving every other line. Returns
    True on success, False on write failure (never raises)."""
    lines: list = []
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
    except OSError:
        lines = []
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key + "="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    try:
        with open(env_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        return True
    except OSError:
        return False


def _ask_value(spec: dict, current, input_fn):
    """Prompt for one schema key; return the (coerced) value, or the current
    value when the user just hits enter."""
    typ = spec.get("type")
    label = spec.get("label", "")
    if typ == "bool":
        hint = "Y/n" if current else "y/N"
        ans = (input_fn(f"  {label}? [{hint}] ") or "").strip().lower()
        if not ans:
            return bool(current)
        return ans in ("y", "yes")
    if typ == "enum":
        choices = spec.get("choices", [])
        ans = (input_fn(f"  {label} {choices} [{current}]: ") or "").strip()
        return sw.coerce_value(spec, ans) if ans else current
    ans = (input_fn(f"  {label} [{current}]: ") or "").strip()
    return sw.coerce_value(spec, ans) if ans else current


def find_claude_cli() -> Optional[str]:
    """Path to the Claude Code CLI if installed, else None. When present, the
    optional self-upgrade pipeline runs on a Claude Code SUBSCRIPTION (via
    ``claude --print``) instead of burning metered Anthropic API credit."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    for p in (
        os.path.expanduser(r"~\.local\bin\claude.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude-code\claude.exe"),
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
    ):
        if p and os.path.exists(p):
            return p
    return None


def run(*, input_fn=input, out=print, env_path: Optional[str] = None,
        settings_path: Optional[str] = None, defaults: bool = False) -> int:
    env_path = env_path or _ENV_PATH
    settings_path = settings_path or sw.SETTINGS_PATH
    out("JARVIS setup wizard")
    out("-" * 40)

    # 1. ANTHROPIC_API_KEY -> .env
    if env_has_api_key(env_path):
        out("ANTHROPIC_API_KEY already set - skipping.")
    elif defaults:
        out("No ANTHROPIC_API_KEY found; set it in .env before running JARVIS.")
    else:
        key = (input_fn("Anthropic API key (sk-ant-...), or blank to skip: ") or "").strip()
        if key:
            ok = upsert_env(env_path, "ANTHROPIC_API_KEY", key)
            out("  saved to .env." if ok else "  could not write .env.")
        else:
            out("  skipped - set ANTHROPIC_API_KEY in .env before running JARVIS.")

    # 2. essentials -> data/user_settings.json
    settings = sw.load_settings(settings_path)
    if not defaults:
        out("\nEssentials (press enter to keep the shown default):")
        for k in _ESSENTIALS:
            spec = sw.SCHEMA.get(k)
            if spec:
                settings[k] = _ask_value(spec, settings.get(k), input_fn)
    try:
        sw.save_settings(settings, settings_path)
        out(f"\nSaved settings -> {settings_path}")
    except Exception as e:
        out(f"\nCould not save settings: {e}")
        return 1

    # 3. Claude Code (optional) — lets the self-upgrade pipeline run on a
    #    subscription instead of metered API credit.
    if find_claude_cli():
        out("\nClaude Code detected - the optional self-upgrade pipeline will use\n"
            "your subscription instead of API credit.")
    else:
        out("\nTip: install Claude Code (https://claude.com/claude-code) + sign in\n"
            "so the optional self-upgrade pipeline runs on your subscription, not\n"
            "API credit. Without it, that pipeline simply stays off.")

    out("\nDone. Fine-tune anything else in the Settings GUI:")
    out("  python tools/settings_window.py")
    out("Then start JARVIS (see SETUP.md).")
    return 0


def main(argv: Optional[list] = None, **kw) -> int:
    defaults = bool(argv and "--defaults" in argv)
    return run(defaults=defaults, **kw)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))
