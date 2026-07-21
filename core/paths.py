"""Canonical runtime-state paths — ONE staging-aware source of truth.

WHY THIS MODULE EXISTS
======================
`JARVIS_STAGING=1` is supposed to mean "this process must not touch the live
box's runtime state". For a long time it only actually meant that for two
things: the monolith's own module-level paths, and — after the incident
recorded in `tools/settings_window.settings_path()` — `user_settings.json`.

Every other runtime-state writer resolved its own directory with a private
`_DATA_DIR = os.path.join(_PROJECT_DIR, "data")` bound at import. Roughly
twenty modules did this: the smart-home catalog, the per-brand smart-home
credential/state files (ecobee, hue, govee, nest, ring, tuya, kasa), the
scheduler, the diagnostic daemons, ambient capture, pattern learning, the
browser agent, network_deco, RAG. None of them honoured staging, so a
staging-isolated sweep wrote straight into the LIVE `data/`.

OBSERVED 2026-07-21: a full `tools/action_smoke.py` sweep (which sets
JARVIS_STAGING=1 and a redirected JARVIS_SETTINGS_PATH, and whose md5 tripwire
on user_settings.json stayed green) nonetheless rewrote the live
`data/smart_home_devices.json` — its discovery actions ran, `[sh-discover]
get_devices failed`, and the live catalog was replaced with `device_count: 0`.
The tripwire did not catch it because the tripwire only watched settings. This
is the same rule ("staging must redirect writes") fixed in one copy while the
rest rotted — this codebase's signature bug class.

USE THIS, not a private join. `data_dir()` resolves at CALL time so a redirect
set after import still takes effect, exactly like settings_path().
"""
from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Env var that force-redirects the whole data directory (mirrors
# JARVIS_SETTINGS_PATH's role for the settings file). Highest precedence.
DATA_DIR_ENV = "JARVIS_DATA_DIR"


def is_staging() -> bool:
    """True on a staging / blue-green GREEN candidate / sweep process.

    Same signal settings_window.settings_path() uses, so the two can never
    disagree about which box's state a process owns.
    """
    return (os.environ.get("JARVIS_STAGING", "").strip() == "1"
            or "--staging" in sys.argv)


def data_dir(create: bool = True) -> str:
    """The directory runtime state belongs in for THIS process.

    `JARVIS_DATA_DIR` wins; then `data_staging/` for a staging role; then the
    live `data/`. Creates the directory by default so a staging process writing
    its first file doesn't fail on a missing dir (the live `data/` always
    exists, so this is effectively staging-only).
    """
    override = (os.environ.get(DATA_DIR_ENV) or "").strip()
    if override:
        path = override
    elif is_staging():
        path = os.path.join(PROJECT_DIR, "data_staging")
    else:
        path = os.path.join(PROJECT_DIR, "data")
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            # Never let path resolution raise — callers use this at import
            # time and a read-only FS must degrade, not crash the loader.
            pass
    return path


def data_file(name: str, create_dir: bool = True) -> str:
    """`data_dir()`-relative path for a single runtime-state file."""
    return os.path.join(data_dir(create=create_dir), name)
