"""Guest mode: visitors are in the room, so JARVIS learns nothing.

While guest mode is on JARVIS still answers every turn normally, but it
writes NOTHING to its long-term stores: no learned facts (bobert_memory.json
or the tiered semantic store), no projects, no topics or topic sightings, no
episodes (the verbatim turn log and the session summaries) and no
voice-command log entries. A visitor's sentence can therefore never become a
"fact about the owner", and what was said in company is not kept.

The owner turns it on and off by voice ("guest mode on", "we have guests",
"the guests have left"), or from the web dashboard. The flip persists like
wake-word mode does: core.config.GUEST_MODE, written to user_settings.json by
the monolith's _act_guest_mode_set, and re-applied at boot by
bobert_companion._guest_mode_boot. It stays on across restarts until he turns
it off.

ONE live flag, read by every writer (the monolith's learners and turn
recorder, core/long_term_memory, the ambient skills). Stdlib-only and import
light on purpose: core.long_term_memory and the skills import it, and it never
reads core.config itself. The boot seed is the monolith's job, so a process
that merely imports a module (a test, a tool) always starts with guest mode
OFF whatever the owner's settings file says.

The slot is a single-element list, the core/state.py contract: the identity
never changes, only ``_on[0]``. Never raises.
"""
from __future__ import annotations

_on: list[bool] = [False]


def is_on() -> bool:
    """True while guest mode is on. Never raises."""
    try:
        return bool(_on[0])
    except Exception:
        return False


def set_on(on: bool) -> None:
    """Set the live flag (no persistence; the monolith's action persists).
    Never raises."""
    try:
        _on[0] = bool(on)
    except Exception:
        pass
