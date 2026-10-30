"""Voice-mood response adapter.

Closes the loop on the existing voice-mood router: when route_voice_emotion()
flags the user as 'stressed', this module reacts on behalf of JARVIS instead
of merely logging the label. Side effects:

  1. Marks 'voice_mood_stressed_until' = now + 900 (seconds) in long-term
     memory so should_be_proactive() will skip the next 15 minutes of
     idle-time nudges. Parallel to the late-night suppression mechanism;
     uses the same _memory_lock for write safety.

  2. Fires a non-blocking smart-home call ('dim all lights to 15% warm
     2700K') via core.smart_home_router.smart_home_control in a daemon
     thread. The LLM response path is never blocked by Hue/Govee latency
     or device unavailability — failures are logged and dropped.
     Since v3.0.10 only as VOICE_MOOD_CALMING_LIGHTS says: 'off' (default)
     sends nothing, 'dimmable' needs a reachable catalog light that can dim
     and waits 15 min between dims, 'legacy' is the old path.

  3. Returns an extra system-prompt addendum the caller stacks onto
     sys_prompt_now, instructing the LLM to open with a brief deferential
     acknowledgement ("Understood, sir. I'll keep it brief.") and to keep
     the response short. Since v3.0.10 (VOICE_MOOD_HONEST_ADDENDUM) the
     note forces no opener and mentions the lights only when a dim was sent.

The TTS preset shift to a calmer/slower variant is already handled by the
existing 'stressed' entry in bobert_companion._USER_TONE_TTS (selected by
synthesise() via _last_voice_route[0]['mood']), so this adapter does not
touch TTS directly — it relies on that path remaining the single source of
truth for mood→prosody.
"""

from __future__ import annotations

import threading
import time
from typing import Any


STRESS_SUPPRESSION_SECS = 15 * 60  # 15 minutes
MEMORY_KEY = "voice_mood_stressed_until"

# Stack-on-top addendum for the LLM. Kept short so it doesn't blow out the
# existing voice-mood hint already stacked by route_voice_emotion().
STRESS_DEFERENTIAL_ADDENDUM = (
    "\n\n[Voice-mood response: stressed]\n"
    "User audio cues suggest acute stress. Open with one extra-deferential, "
    "concise acknowledgement in JARVIS's register — e.g. \"Understood, sir. "
    "I'll keep it brief.\" — then deliver the response in ≤1 short sentence. "
    "No humour, no embellishments, no follow-up questions unless strictly "
    "necessary. Smart-home dimming is already in motion; do not narrate it."
)

# v3.0.10 (VOICE_MOOD_HONEST_ADDENDUM): the same register with no forced
# opener and no claim about the lights. Live 2026-10-28 22:34:00 and
# 2026-10-29 18:54:37 the model opened with the canned "Understood, sir.
# I'll keep it brief." and the prompt said "dimming is already in motion"
# while the smart-home call only read device status. The lights line is added
# (STRESS_LIGHTS_NOTE) only when a dim was really sent this turn.
STRESS_HONEST_ADDENDUM = (
    "\n\n[Voice-mood response: stressed]\n"
    "User audio cues suggest acute stress. Keep it short: deliver the "
    "response in ≤1 short sentence, calm and plain. No humour, no "
    "embellishments, no follow-up questions unless strictly necessary."
)
STRESS_LIGHTS_NOTE = " The lights are being dimmed; do not narrate it."

# VOICE_MOOD_CALMING_LIGHTS modes (core/config.py): 'off' never dims,
# 'dimmable' dims only with a reachable catalog device that can dim and at
# most once per CALMING_LIGHTS_COOLDOWN_SECS, 'legacy' is the old path.
CALMING_LIGHTS_MODES = ("off", "dimmable", "legacy")
CALMING_LIGHTS_COOLDOWN_SECS = 15 * 60
# time.monotonic() of the last dim sent in 'dimmable' mode (0.0 = none yet).
_last_dim_sent_at = [0.0]

# Single-shot daemon-thread guard. Prevents a runaway burst of identical
# smart-home calls when the user fires several stressed utterances back to
# back — the first dim is in flight, subsequent calls within the same window
# are no-ops at the dispatch layer.
_smart_home_in_flight = threading.Event()


def _dispatch_calming_lights() -> None:
    """Background-thread target: ask the smart-home router to dim lights warm.

    Swallows every exception — this runs on a daemon thread and there is
    nothing the LLM path could do with a failure here. We do log to stdout
    so the operator can see whether the call actually landed.
    """
    try:
        from core import smart_home_router  # local import: avoids cycles
        result = smart_home_router.smart_home_control(
            "dim all lights to 15% warm 2700K"
        )
        print(f"  [voice-mood] smart-home: {result}")
    except Exception as err:
        print(f"  [voice-mood] smart-home dispatch failed: {err}")
    finally:
        _smart_home_in_flight.clear()


def _smart_home_catalog_ready() -> bool:
    """True only when a smart-home catalog with at least one device exists.

    Mirrors the guard smart_home_control() itself uses
    (``_ensure_catalog()`` → ``catalog.get("devices")``) so we don't fire
    the calming-lights reflex — and print the resulting "No smart-home
    catalog yet, sir." line — on every stressed utterance when no devices
    are configured. Best-effort: any import/lookup failure is treated as
    "not ready" so a broken router never blocks the LLM path.
    """
    try:
        from core import smart_home_router  # local import: avoids cycles
        catalog = smart_home_router._ensure_catalog()
    except Exception:
        return False
    return bool(catalog and catalog.get("devices"))


def _calming_lights_mode() -> str:
    """VOICE_MOOD_CALMING_LIGHTS (core/config.py) normalised to one of
    CALMING_LIGHTS_MODES, read at call time. Anything else, or any trouble
    reading it, is 'off' (no call). Never raises."""
    try:
        from core import config as _cfg
        mode = str(getattr(_cfg, "VOICE_MOOD_CALMING_LIGHTS", "off"))
        mode = mode.strip().lower()
    except Exception:
        return "off"
    return mode if mode in CALMING_LIGHTS_MODES else "off"


def _honest_addendum_on() -> bool:
    """VOICE_MOOD_HONEST_ADDENDUM (core/config.py), default True; True on any
    trouble reading it. Never raises."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, "VOICE_MOOD_HONEST_ADDENDUM", True))
    except Exception:
        return True


def _has_reachable_dimmable(catalog) -> bool:
    """True when the catalog lists a device that can dim ('dim' in its
    capabilities) and that something can reach: a controller skill (named,
    or from its brand) and a LAN address or an Alexa entity. On/off plugs,
    cameras and scenes never count. Never raises (False)."""
    try:
        from core import smart_home_router  # local import: avoids cycles
        brand_skill = getattr(smart_home_router, "_controller_for", None)
        for d in (catalog or {}).get("devices") or []:
            if not isinstance(d, dict):
                continue
            caps = [str(c).strip().lower()
                    for c in (d.get("capabilities") or [])]
            if "dim" not in caps:
                continue
            skill = d.get("controller_skill") or (
                brand_skill(str(d.get("brand") or ""))
                if callable(brand_skill) else None)
            if skill and (d.get("lan_ip") or d.get("alexa_entity_id")):
                return True
    except Exception:
        return False
    return False


def _kick_calming_lights(now: float | None = None) -> bool:
    """The calming-lights dispatch as VOICE_MOOD_CALMING_LIGHTS says. True
    only when a dim was SENT on this call. 'off' (default) sends nothing;
    'dimmable' sends one only when a reachable dimmable device is in the
    catalog, none is in flight and the cooldown has run out; 'legacy' is
    _kick_smart_home exactly. Never raises (False)."""
    try:
        mode = _calming_lights_mode()
        if mode == "legacy":
            return _kick_smart_home()
        if mode != "dimmable":
            return False
        t = time.monotonic() if now is None else float(now)
        last = float(_last_dim_sent_at[0] or 0.0)
        if last and t - last < CALMING_LIGHTS_COOLDOWN_SECS:
            return False
        if _smart_home_in_flight.is_set():
            return False
        from core import smart_home_router  # local import: avoids cycles
        if not _has_reachable_dimmable(smart_home_router._ensure_catalog()):
            return False
        _last_dim_sent_at[0] = t
        return _kick_smart_home()
    except Exception as err:
        print(f"  [voice-mood] calming-lights check failed: "
              f"{type(err).__name__}")
        return False


def _kick_smart_home() -> bool:
    """Fire-and-forget calming-lights dispatch on a daemon thread.

    Gated on a configured smart-home catalog: with no devices set up there
    is nothing to dim, so we skip silently rather than dispatch a call that
    can only return "No smart-home catalog yet, sir." (logged ×N otherwise).

    Returns True when a dim was dispatched (or one was already in flight),
    False when skipped for lack of a catalog — the caller uses this to keep
    its "dimming lights" log line honest.
    """
    if not _smart_home_catalog_ready():
        return False
    if _smart_home_in_flight.is_set():
        return True
    _smart_home_in_flight.set()
    t = threading.Thread(
        target=_dispatch_calming_lights,
        name="voice-mood-lights",
        daemon=True,
    )
    t.start()
    return True


def is_stress_suppression_active(memory: dict, now: float | None = None) -> bool:
    """True iff a stress-detected suppression window is still open.

    Read by should_be_proactive() in bobert_companion to skip idle nudges
    while the user is recovering. The 'until' field is a Unix timestamp;
    a stale value (in the past) is treated as inactive and ignored.
    """
    if not memory:
        return False
    until = memory.get(MEMORY_KEY)
    if not isinstance(until, (int, float)):
        return False
    return (now if now is not None else time.time()) < float(until)


def apply_voice_mood_response(
    route: dict[str, Any] | None,
    user_text: str,
    memory_lock: Any = None,
    load_memory: Any = None,
    save_memory: Any = None,
) -> str:
    """Top-level entry point.

    Caller (bobert_companion._call_llm) invokes this immediately after
    route_voice_emotion() returns. When mood != 'stressed' this is a no-op
    that returns ''. When mood == 'stressed' it:
      - persists the 15-minute suppression timestamp into memory,
      - kicks off the non-blocking smart-home dim,
      - returns the deferential system-prompt addendum to be appended to
        sys_prompt_now.

    The load_memory / save_memory / memory_lock callables are passed in so
    the adapter stays loosely coupled to bobert_companion's persistence
    helpers (which would otherwise create an import cycle).
    """
    if not route or route.get("mood") != "stressed":
        return ""

    deadline = time.time() + STRESS_SUPPRESSION_SECS
    if load_memory is not None and save_memory is not None:
        try:
            if memory_lock is not None:
                with memory_lock:
                    mem = load_memory()
                    mem[MEMORY_KEY] = deadline
                    save_memory(mem)
            else:
                mem = load_memory()
                mem[MEMORY_KEY] = deadline
                save_memory(mem)
        except Exception as err:
            print(f"  [voice-mood] memory persist failed: {err}")

    # v3.0.10: VOICE_MOOD_CALMING_LIGHTS decides whether a dim is sent, and
    # VOICE_MOOD_HONEST_ADDENDUM drops the canned opener and mentions the
    # lights only when one was. Both at their old values = the old path.
    dimming = _kick_calming_lights()
    lights_note = "; dimming lights" if dimming else ""
    print(
        f"  [voice-mood] stressed → proactive suppression "
        f"for {STRESS_SUPPRESSION_SECS // 60} min{lights_note}"
    )
    if not _honest_addendum_on():
        return STRESS_DEFERENTIAL_ADDENDUM
    # A 'dimmable' dim is not yet proven on this hardware and its result
    # lands 20-40 s later (v3.0.10 review: the generic "dim all lights" can
    # end in a status read), so the prompt claims nothing about the lights;
    # only 'legacy' keeps the lights line.
    claim_lights = dimming and _calming_lights_mode() == "legacy"
    return STRESS_HONEST_ADDENDUM + (STRESS_LIGHTS_NOTE if claim_lights
                                     else "")
