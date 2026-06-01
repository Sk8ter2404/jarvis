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

  3. Returns an extra system-prompt addendum the caller stacks onto
     sys_prompt_now, instructing the LLM to open with a brief deferential
     acknowledgement ("Understood, sir. I'll keep it brief.") and to keep
     the response short.

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

    dimming = _kick_smart_home()
    lights_note = "; dimming lights" if dimming else ""
    print(
        f"  [voice-mood] stressed → proactive suppression "
        f"for {STRESS_SUPPRESSION_SECS // 60} min{lights_note}"
    )
    return STRESS_DEFERENTIAL_ADDENDUM
