"""
Skill shim — exposes `core.smart_home_router`'s actions to the JARVIS
skill loader. The router itself lives in `core/` (it's not a leaf skill
in spirit — it dispatches to other skills) but JARVIS' `load_skills()`
only scans `skills/*.py`, so this file forwards the `register()` call.

If the core module fails to import for any reason (e.g. partial install,
corrupt catalog), this shim degrades silently and the rest of JARVIS
keeps loading.
"""
from __future__ import annotations


def register(actions: dict) -> None:
    try:
        from core import smart_home_router  # type: ignore
    except Exception as e:
        print(f"  [sh-router-shim] could not import core.smart_home_router: {e}")
        return
    try:
        smart_home_router.register(actions)
    except Exception as e:
        print(f"  [sh-router-shim] register failed: {e}")
