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

Visitors' VOICES (review 2026-10-02): a live "guest mode on" also lets them
through the voice-ID gates (the wake listener's strict gate, the media voice
gate) -- the older wake-listener "guest mode" did only that. That half is a
security bypass and stays what it always was, per run: only the live flip
opens it (set_voices_open), never the boot seed, and turning guest mode off
closes it. After a restart a saved guest mode still keeps memory shut, but
the gates are strict again until "guest mode on" is said in that run.

The slots are single-element lists, the core/state.py contract: the identity
never changes, only ``_on[0]`` / ``_voices[0]``. Never raises.
"""
from __future__ import annotations

_on: list[bool] = [False]
# Visitors' voices pass the voice-ID gates: set only by a live flip.
_voices: list[bool] = [False]


def is_on() -> bool:
    """True while guest mode is on. Never raises."""
    try:
        return bool(_on[0])
    except Exception:
        return False


def set_on(on: bool) -> None:
    """Set the live flag (no persistence; the monolith's action persists).
    Off also closes the voice-ID bypass (see voices_open). Never raises."""
    try:
        _on[0] = bool(on)
    except Exception:
        pass
    if not on:
        set_voices_open(False)


def voices_open() -> bool:
    """True while visitors' voices pass the voice-ID gates: guest mode is on
    AND it was turned on live in this run (never by the boot seed alone).
    Never raises."""
    try:
        return bool(_on[0]) and bool(_voices[0])
    except Exception:
        return False


def set_voices_open(on: bool) -> None:
    """Open / close the voice-ID bypass (the monolith's live flip calls this;
    the boot seed never does). Never persisted. Never raises."""
    try:
        _voices[0] = bool(on)
    except Exception:
        pass
