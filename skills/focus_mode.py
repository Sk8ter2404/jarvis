"""
Focus mode / do-not-disturb — HOLD unsolicited announcements, RECAP on resume.
─────────────────────────────────────────────────────────────────────────────

The owner works long heads-down stretches and wants JARVIS to stop volunteering
things (print milestones, weather, Teams nudges, timer reminders, self-diag…)
while he's concentrating — then, when he resumes, hear a quick RECAP of what he
missed. He never wants his OWN commands silenced: wake-word + direct command
replies must keep working the whole time.

HOW IT WORKS (the design — see bobert_companion for the other half)
───────────────────────────────────────────────────────────────────
`bobert_companion.proactive_announce(...)` is the SINGLE chokepoint for ALL
unsolicited/proactive speech (it appends to pending_speech.json which the main
loop drains). Direct command responses do NOT go through it — they call _speak()
directly. So gating proactive_announce cleanly suppresses ONLY unsolicited output
while leaving wake-word + command responses fully working.

This skill is the CONTROL surface. It flips the monolith's focus flag and reads
its bounded "missed" buffer, all through small NEVER-RAISES helpers on the
monolith:

    focus_mode_active()            -> bool   (also self-resumes a lapsed timer)
    set_focus_mode(on, until=…)    -> None
    focus_missed_count()           -> int
    focus_missed_snapshot()        -> list   (does NOT clear)
    clear_focus_missed()           -> None
    _build_focus_recap(clear=…)    -> str    (summarise + optionally clear)

Actions registered
───────────────────
  focus_mode_on / do_not_disturb / quiet_mode
        Engage. Arg may carry a duration ("30 minutes", "an hour", "45m"); no
        arg = indefinite. Sets focus on, arms an auto-resume Timer if timed.
  focus_mode_off / resume / end_focus_mode
        Disengage + return a RECAP one-liner of what was missed, then clear the
        buffer. (end_focus_mode ALSO chains the pre-existing dnd_focus_mode
        handler of the same name so its Windows-Focus-Assist / Teams-presence
        teardown still runs — see _chain_prior below.)
  whats_missed / focus_mode_status
        Report whether focus is on (+ remaining time if timed) and how many
        announcements are held — WITHOUT clearing. A pure status query.

Auto-resume
───────────
When a duration is given we arm a daemon threading.Timer that, at expiry, flips
focus OFF and enqueues the recap via proactive_announce (now un-gated, so it
speaks normally): "Focus time's up, sir — here's what you missed…". Daemon so it
never blocks shutdown. Belt-and-braces: focus_mode_active() ALSO self-heals a
lapsed timer on the next read, so even if the Timer is cancelled at shutdown the
owner still gets the recap the next time the flag is consulted.

Coexistence with skills/dnd_focus_mode.py
─────────────────────────────────────────
dnd_focus_mode.py is a SEPARATE, older DND skill that toggles Windows Focus
Assist + Teams presence and suppresses a few named skills' nudges. It registers
`focus_mode`, `end_focus_mode`, `focus_mode_status`. We load alphabetically AFTER
it (d < f), so to avoid clobbering its OS-level teardown we CHAIN the two names
we share (`end_focus_mode`, `focus_mode_status`): call its handler first, then
append our recap/status. Our unique names (focus_mode_on/off, resume,
do_not_disturb, quiet_mode, whats_missed) never collide.
"""
from __future__ import annotations

import re
import sys
import threading
import time


# ─── reaching the monolith (defensive, mirrors the other skills) ─────────────

def _bc():
    """Live monolith module (main entrypoint or by-name import), or None.
    Every accessor below tolerates None so this skill degrades to graceful
    spoken messages if the monolith isn't importable (e.g. isolated tests)."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _is_active() -> bool:
    bc = _bc()
    if bc is None:
        return False
    try:
        fn = getattr(bc, "focus_mode_active", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _set_focus(on: bool, *, until: float = 0.0) -> None:
    bc = _bc()
    if bc is None:
        return
    try:
        fn = getattr(bc, "set_focus_mode", None)
        if callable(fn):
            fn(on, until=until)
    except Exception:
        pass


def _missed_count() -> int:
    bc = _bc()
    if bc is None:
        return 0
    try:
        fn = getattr(bc, "focus_missed_count", None)
        return int(fn()) if callable(fn) else 0
    except Exception:
        return 0


def _build_recap(*, clear: bool, prefix: str) -> str:
    """Summarise the monolith's missed buffer. Falls back to a plain sentence
    if the monolith helper is unavailable (isolated test / early boot)."""
    bc = _bc()
    if bc is not None:
        try:
            fn = getattr(bc, "_build_focus_recap", None)
            if callable(fn):
                return fn(clear=clear, prefix=prefix)
        except Exception:
            pass
    return f"{prefix} — nothing came up while you were focused."


# ─── auto-resume timer ───────────────────────────────────────────────────────
# One live daemon Timer at a time. Held in a single-element list (GIL-atomic)
# under a lock so re-engaging focus cancels any prior timer cleanly.
_resume_timer: list = [None]
_timer_lock = threading.Lock()


def _cancel_resume_timer() -> None:
    with _timer_lock:
        t = _resume_timer[0]
        _resume_timer[0] = None
    if t is not None:
        try:
            t.cancel()
        except Exception:
            pass


def _arm_resume_timer(seconds: float) -> None:
    """Arm a daemon Timer that fires the auto-resume recap at expiry. Replaces
    any existing timer. Daemon so it never blocks process shutdown."""
    _cancel_resume_timer()
    if seconds <= 0:
        return

    def _fire():
        # Only act if we're still actually focused (a manual 'resume' may have
        # beaten us here — in which case focus is already off and the buffer
        # already recapped + cleared).
        try:
            if not _is_active():
                return
            _set_focus(False)
            recap = _build_recap(clear=True, prefix="Focus time's up, sir")
            bc = _bc()
            if bc is not None and recap:
                announcer = getattr(bc, "proactive_announce", None)
                if callable(announcer):
                    # Focus is OFF now, so this enqueues + speaks normally.
                    announcer(recap, source="focus_mode")
        except Exception:
            pass
        finally:
            with _timer_lock:
                _resume_timer[0] = None

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    with _timer_lock:
        _resume_timer[0] = t
    t.start()


# ─── duration parsing (best-effort) ──────────────────────────────────────────

# Word-number map for spoken durations ("an hour", "half an hour", "ninety").
_WORD_NUMS = {
    "an": 1, "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fortyfive": 45, "forty five": 45, "fifty": 50,
    "sixty": 60, "ninety": 90,
}


def _parse_duration_seconds(text: str):
    """Return seconds for a spoken duration, or None if none is expressed.

    Handles: '30 minutes', '45m', '1 hour 30 min', '2h', 'an hour',
    'half an hour', 'ninety minutes', and a bare number ('30' → 30 minutes).
    Best-effort by design: an unparseable/blank arg → None → indefinite focus."""
    if not text:
        return None
    t = text.strip().lower()

    # Common spoken phrases first.
    if "half an hour" in t or "half hour" in t:
        return 30 * 60
    if re.search(r"\ban?\s+hour\b", t) and not re.search(r"\d", t):
        # "an hour" / "a hour" with no explicit number.
        return 3600

    total = 0
    found = False

    # Numeric quantities with units.
    for n, unit in re.findall(
        r"(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b", t
    ):
        n = int(n)
        if unit in ("s", "sec", "secs", "second", "seconds"):
            total += n
        elif unit in ("m", "min", "mins", "minute", "minutes"):
            total += n * 60
        else:  # h / hr / hrs / hour / hours
            total += n * 3600
        found = True
    if found:
        return total if total > 0 else None

    # Word-number quantities with units ("ninety minutes", "two hours").
    m = re.search(
        r"\b([a-z\- ]+?)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?)\b", t
    )
    if m:
        word = m.group(1).strip().split()[-1]  # last word before the unit
        qty = _WORD_NUMS.get(word)
        if qty is not None:
            unit = m.group(2)
            if unit.startswith("h"):
                return qty * 3600
            if unit.startswith("s"):
                return qty
            return qty * 60

    # Bare number → assume minutes ("focus mode 30" = 30 min).
    bare = re.fullmatch(r"\s*(\d+)\s*", t)
    if bare:
        return int(bare.group(1)) * 60
    return None


def _spoken_duration(seconds: int) -> str:
    """Human phrasing for a duration: '30 minutes', '1 hour', '1 hour 30 minutes'."""
    if seconds < 60:
        return f"{seconds} seconds"
    m = round(seconds / 60)
    if m < 60:
        return "1 minute" if m == 1 else f"{m} minutes"
    h, rem = divmod(m, 60)
    hpart = "1 hour" if h == 1 else f"{h} hours"
    if rem == 0:
        return hpart
    return f"{hpart} {rem} minutes"


# ─── chaining the pre-existing dnd_focus_mode handlers (coexistence) ─────────

def _chain_prior(actions: dict, name: str):
    """Return the handler already registered under `name` (dnd_focus_mode's, if
    it loaded first), or None. We call it BEFORE our own logic for the two names
    we share so its Windows-Focus-Assist / Teams-presence teardown still runs.
    Captured at register() time so a later re-register can't recurse into us."""
    prior = actions.get(name)
    return prior if callable(prior) else None


# ─── action registration ─────────────────────────────────────────────────────

def register(actions):
    # Capture any pre-existing handlers for the names we share with
    # dnd_focus_mode BEFORE we overwrite them (see module docstring).
    _prior_end    = _chain_prior(actions, "end_focus_mode")
    _prior_status = _chain_prior(actions, "focus_mode_status")
    _prior_engage = _chain_prior(actions, "focus_mode")

    def focus_mode_on(args: str = "") -> str:
        secs = _parse_duration_seconds(args) if args else None
        if secs is not None and secs > 0:
            until = time.time() + secs
            _set_focus(True, until=until)
            _arm_resume_timer(secs)
            return (f"Focus mode on, sir — I'll hold notifications for "
                    f"{_spoken_duration(int(secs))}.")
        # No/blank/unparseable duration → indefinite.
        _cancel_resume_timer()
        _set_focus(True, until=0.0)
        return "Focus mode on, sir — I'll hold notifications until you resume."

    def focus_mode_off(_: str = "") -> str:
        # Cancel any pending auto-resume first so it can't double-fire.
        _cancel_resume_timer()
        was_active = _is_active()
        # Build the recap (and clear the buffer) BEFORE flipping the flag — same
        # ordering the monolith uses on auto-expiry; harmless either way since
        # we clear here.
        recap = _build_recap(clear=True, prefix="While you were focused, sir")
        _set_focus(False)
        if not was_active:
            return "Focus mode wasn't on, sir — nothing was held."
        return recap

    def resume(args: str = "") -> str:
        # 'resume' / 'I'm back' / 'what did I miss' → same as focus_mode_off.
        return focus_mode_off(args)

    def engage(args: str = "") -> str:
        # 2026-07-07 fix: the phrase "focus mode" routes to the SHARED name
        # `focus_mode` (dnd_focus_mode's engage), NOT our `focus_mode_on`. Before
        # this, saying "focus mode" turned on dnd's OS-level Focus Assist but left
        # OUR announcement-holding gate OFF — so notifications were NOT actually
        # held and `focus_mode_status` reported a contradiction (dnd engaged / us
        # off). Chain dnd's engage (Focus Assist / Teams presence) for side
        # effects, then engage our state too, so one command does both and status
        # stays consistent. We return OUR message (dnd's return is discarded).
        if _prior_engage is not None:
            try:
                _prior_engage(args)
            except Exception:
                pass
        return focus_mode_on(args)

    def end_focus_mode(args: str = "") -> str:
        # Chain the pre-existing dnd_focus_mode teardown (Focus Assist / Teams)
        # first, then return OUR recap so the owner hears what he missed.
        if _prior_end is not None:
            try:
                _prior_end(args)
            except Exception:
                pass
        return focus_mode_off(args)

    def whats_missed(_: str = "") -> str:
        # Pure status query — never clears the buffer.
        held = _missed_count()
        if not _is_active():
            if held:
                return (f"Focus mode is off, sir — {held} announcement"
                        f"{'s' if held != 1 else ''} still waiting from your "
                        f"last session. Say 'what did I miss' to hear them.")
            return "Focus mode is off, sir — nothing is being held."
        # Active — report remaining time if it's a timed block.
        remaining_txt = ""
        bc = _bc()
        try:
            until = float(getattr(bc, "_focus_until", [0.0])[0]) if bc else 0.0
        except Exception:
            until = 0.0
        if until:
            rem = int(max(0, until - time.time()))
            remaining_txt = f", about {_spoken_duration(rem)} left"
        noun = "announcement" if held == 1 else "announcements"
        if held:
            return (f"Focus mode is on{remaining_txt}, sir — holding {held} "
                    f"{noun} so far.")
        return (f"Focus mode is on{remaining_txt}, sir — nothing held yet.")

    def focus_mode_status(args: str = "") -> str:
        # Chain the pre-existing dnd_focus_mode status (its OS-level state)
        # first, then append our held-count so a single query reports both.
        prior_txt = ""
        if _prior_status is not None:
            try:
                prior_txt = (_prior_status(args) or "").strip()
            except Exception:
                prior_txt = ""
        ours = whats_missed(args)
        if prior_txt and prior_txt not in ours:
            return f"{prior_txt} {ours}"
        return ours

    # Unique names — never collide with dnd_focus_mode.
    actions["focus_mode_on"]  = focus_mode_on
    actions["do_not_disturb"] = focus_mode_on
    actions["quiet_mode"]     = focus_mode_on
    actions["focus_mode_off"] = focus_mode_off
    actions["resume"]         = resume
    actions["whats_missed"]   = whats_missed
    # Shared names — chained to preserve dnd_focus_mode's OS-level behaviour.
    actions["focus_mode"]        = engage
    actions["end_focus_mode"]    = end_focus_mode
    actions["focus_mode_status"] = focus_mode_status

    print("  [focus_mode] ready — actions: focus_mode_on / do_not_disturb / "
          "quiet_mode / focus_mode_off / resume / whats_missed "
          "(+ chained end_focus_mode / focus_mode_status)")
