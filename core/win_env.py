"""Shared environment probes for skills (window titles, sleep/standby).

Canonical copies of helpers that were duplicated verbatim across the
briefing / banter / anticipation skills. Modules import them under the same
local names so call sites and `mock.patch.object(mod, "_name")` keep working.
"""
import sys


def _all_window_titles() -> list[str]:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return []
    out: list[str] = []
    try:
        for w in gw.getAllWindows():
            t = getattr(w, "title", "") or ""
            if t.strip():
                out.append(t)
    except Exception:
        pass
    return out


def _is_sleep_or_standby() -> bool:
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return False
    try:
        if getattr(bc, "_sleep_mode")[0]:
            return True
        if getattr(bc, "_standby_mode")[0]:
            return True
    except Exception:
        return False
    return False
