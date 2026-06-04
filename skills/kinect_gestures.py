"""
kinect_gestures skill — gesture CONTROL for JARVIS via the Kinect v2.

The pure recognizer lives in audio/kinect_gestures.py; this skill is the live
wiring around it: a background poll loop that reads the Kinect skeleton stream
at ~18 Hz, runs the recognizer, and maps the discrete gestures it emits onto
JARVIS actions — all behind opt-in flags and a hard staging gate so it never
fires in tests / the staging instance.

GESTURE → ACTION MAP
====================
  WAVE       → wake JARVIS if it's dormant (the tray force_wake path).
               "You waved me over, sir." No-op if already awake.
  RAISE_HAND → confirm the pending confirmation if one is queued (the same as
               saying "yes"). No-op when nothing is pending.
  SWIPE      → "never mind": interrupt any current TTS and clear a pending
               confirmation. The stop/cancel path.

EVERYTHING is opt-in + safe:
  • The whole loop is gated by core.config.KINECT_GESTURES_ENABLED (default
    False), re-read each tick so a Settings toggle takes effect with no restart.
  • A staging / test instance (JARVIS_STAGING / bobert_companion._is_staging())
    NEVER runs the loop — gesture control must not fire during the exhaustive
    test suite or on the blue/green staging box.
  • All sensor contact is via audio/kinect_bridge (accessors never raise); a
    missing / disabled sensor degrades to a quiet no-op.
  • Every action side-effect is wrapped — a mapping failure logs and is dropped,
    it never crashes the poller.

Voice actions:
  gesture_status            — "what gestures can you see" / "is gesture control
                              on" — reports enabled state + whether a body is in
                              view.
  gestures_on / gestures_off — toggle KINECT_GESTURES_ENABLED live, persisting
                              via the same Settings writer model_picker uses.
"""
from __future__ import annotations

import os
import sys
import threading
import time


# ─── tunables ────────────────────────────────────────────────────────────
GESTURE_POLL_HZ = 18.0                       # recognizer poll rate
GESTURE_POLL_INTERVAL = 1.0 / GESTURE_POLL_HZ
INITIAL_DELAY_SECONDS = 6.0                  # let the monolith + bridge come up
_THREAD_NAME = "kinect-gestures-skill"


def _recognizer_module():
    """The pure recognizer module (audio.kinect_gestures). Imported lazily so a
    failure can't stop the skill from registering its voice actions."""
    mod = sys.modules.get("audio.kinect_gestures")
    if mod is not None:
        return mod
    try:
        from audio import kinect_gestures as _kg
        return _kg
    except Exception:
        return None


def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_vision)."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — gesture control must NEVER fire
    there. Matches the monolith's own gate plus the raw env var so the check
    holds even before the monolith is importable."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def _gestures_enabled() -> bool:
    """The master gate for the live loop: opt-in flag ON and not staging."""
    return _cfg_flag("KINECT_GESTURES_ENABLED") and not _is_staging()


# ─── action mapping (each branch fully guarded + error-swallowing) ─────────

def _in_standby(bc) -> bool:
    """True when JARVIS is dormant (sleep or standby)."""
    try:
        return bool(bc._standby_mode[0]) or bool(bc._sleep_mode[0])
    except Exception:
        return False


def _do_wave(bc) -> None:
    """WAVE → wake if dormant. No-op when already awake. Clears the standby
    flags under the same lock the tray force_wake path uses so a concurrent
    auto-engage can't immediately re-assert standby."""
    if not _in_standby(bc):
        return
    lock = getattr(bc, "_standby_auto_engage_lock", None)
    try:
        sleep_flag = getattr(bc, "_sleep_mode", None)
        standby_flag = getattr(bc, "_standby_mode", None)
        if sleep_flag is None or standby_flag is None:
            return
        if lock is not None:
            with lock:
                sleep_flag[0] = False
                standby_flag[0] = False
        else:
            sleep_flag[0] = False
            standby_flag[0] = False
        try:
            bc._write_hud_state(sleep_mode=False, standby_mode=False, state="Idle")
        except Exception:
            pass
        _speak(bc, "You waved me over, sir.")
        print("  [gestures] WAVE -> woke from standby")
    except Exception as e:
        print(f"  [gestures] wave-wake failed: {e}")


def _do_raise_hand(bc) -> None:
    """RAISE_HAND → confirm a pending confirmation (equivalent to 'yes').
    No-op when nothing is queued. Reuses handle_confirmation_response so the
    confirm path (execute + spoken feedback) is identical to a voice 'yes'."""
    try:
        pending = getattr(bc, "_pending_confirmation", None)
        if not pending:
            return
        handler = getattr(bc, "handle_confirmation_response", None)
        if callable(handler):
            handler("yes")
            print("  [gestures] RAISE_HAND -> confirmed pending action(s)")
    except Exception as e:
        print(f"  [gestures] raise-hand confirm failed: {e}")


def _do_swipe(bc) -> None:
    """SWIPE → 'never mind': interrupt current TTS AND clear a pending
    confirmation. Setting _barge_in_interrupted is the same flag the mic
    barge-in uses; the play_with_lipsync watch thread sees it and stops
    playback. Clearing _pending_confirmation cancels a queued action."""
    stopped_something = False
    # Interrupt any in-flight speech.
    try:
        if bool(getattr(bc, "_tts_playback_active", [False])[0]):
            bc._barge_in_interrupted = True
            stopped_something = True
            print("  [gestures] SWIPE -> interrupted speech")
    except Exception as e:
        print(f"  [gestures] swipe-stop-tts failed: {e}")
    # Cancel a pending confirmation (a 'never mind').
    try:
        pending = getattr(bc, "_pending_confirmation", None)
        if pending:
            try:
                pending.clear()
            except Exception:
                del pending[:]
            stopped_something = True
            _speak(bc, "Never mind, sir.")
            print("  [gestures] SWIPE -> cleared pending confirmation")
    except Exception as e:
        print(f"  [gestures] swipe-cancel-confirm failed: {e}")
    return stopped_something


def _speak(bc, text: str) -> None:
    """Speak via the skill_utils seam if present, else the monolith _speak.
    Best-effort and silent on failure."""
    su = globals().get("skill_utils")
    if isinstance(su, dict):
        speaker = su.get("speak")
        if callable(speaker):
            try:
                speaker(text)
                return
            except Exception:
                pass
    try:
        fn = getattr(bc, "_speak", None) or getattr(bc, "speak", None)
        if callable(fn):
            fn(text)
    except Exception:
        pass


# Map each recognizer gesture name onto its handler. Swipe-left and swipe-right
# both dismiss (direction is irrelevant for "never mind").
def _dispatch(bc, gesture: str) -> None:
    kg = _recognizer_module()
    if kg is None:
        return
    if gesture == kg.WAVE:
        _do_wave(bc)
    elif gesture == kg.RAISE_HAND:
        _do_raise_hand(bc)
    elif gesture in (kg.SWIPE_LEFT, kg.SWIPE_RIGHT):
        _do_swipe(bc)


# ─── poll loop ─────────────────────────────────────────────────────────────

def _poll_once(rec, bc) -> str | None:
    """One recognizer tick: read the skeleton stream, update the recognizer,
    dispatch any gesture. Returns the gesture name (for tests) or None. NEVER
    raises. Respects the live gate so toggling KINECT_GESTURES_ENABLED off
    stops dispatch mid-session (the recognizer is still fed so it doesn't see a
    huge time gap when re-enabled — but nothing is dispatched)."""
    kb = _bridge()
    if kb is None:
        return None
    try:
        if not kb.get_enabled():
            return None
        ok, _reason = kb.available()
        if not ok:
            return None
        bodies = kb.get_bodies()
    except Exception:
        return None
    try:
        gesture = rec.update(bodies)
    except Exception:
        return None
    if not gesture:
        return None
    # Gate the SIDE EFFECT (not the recognition) so flipping the flag off stops
    # actions instantly without leaving the recognizer in a stale state.
    if not _gestures_enabled():
        return None
    if bc is None:
        return gesture
    try:
        _dispatch(bc, gesture)
    except Exception as e:   # pragma: no cover - defensive: dispatch swallows internally
        print(f"  [gestures] dispatch error: {e}")
    return gesture


def _poll_loop() -> None:  # pragma: no cover - non-terminating daemon; each tick delegates to _poll_once, which is unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    kg = _recognizer_module()
    if kg is None:
        print("  [gestures] recognizer module unavailable — poller exiting")
        return
    rec = kg.GestureRecognizer()
    while True:
        try:
            bc = _bc()
            _poll_once(rec, bc)
        except Exception as e:
            print(f"  [gestures] poll error: {e}")
        time.sleep(GESTURE_POLL_INTERVAL)


# ─── persistence (reuse the Settings-GUI atomic, merge-not-clobber writer) ──

def _persist_setting(key: str, value) -> bool:
    """Write {key: value} into data/user_settings.json WITHOUT clobbering the
    owner's other saved settings — the EXACT path model_picker._persist_setting
    uses (settings_window.load_settings + save_settings). Best-effort: returns
    False on any error (the live toggle already took effect)."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current)
        return True
    except Exception:
        return False


def _set_gestures_enabled(on: bool) -> bool:
    """Flip KINECT_GESTURES_ENABLED live (core.config + the monolith's mirror if
    it imported config) and persist it. Returns the persisted flag."""
    try:
        import core.config as _cfg
        _cfg.KINECT_GESTURES_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("KINECT_GESTURES_ENABLED", bool(on))


# ─── actions ─────────────────────────────────────────────────────────────

def _body_in_view() -> bool | None:
    """True/False if the Kinect can tell whether a body is in view; None when
    the sensor is off/absent so the caller can phrase it honestly."""
    kb = _bridge()
    if kb is None:
        return None
    try:
        if not kb.get_enabled():
            return None
        ok, _reason = kb.available()
        if not ok:
            return None
        presence = kb.get_presence()
        return bool(presence.get("present"))
    except Exception:
        return None


def gesture_status(_: str = "") -> str:
    """Report whether gesture control is on + whether a body is currently in
    view. 'what gestures can you see' / 'is gesture control on'."""
    enabled = _cfg_flag("KINECT_GESTURES_ENABLED")
    in_view = _body_in_view()
    gestures = "wave to wake me, raise a hand to confirm, or swipe to cancel"
    if not enabled:
        return ("Gesture control is off, sir — say 'turn on gesture control' to "
                f"enable it. Once on, you can {gestures}.")
    # Enabled — describe whether I can actually see a body.
    if in_view is None:
        return ("Gesture control is on, sir, but the Kinect is off or "
                "unavailable, so I can't see any gestures right now.")
    if in_view:
        return (f"Gesture control is on and I can see you, sir — {gestures}.")
    return ("Gesture control is on, sir, but no one is in the Kinect's view at "
            f"the moment. When you step in, {gestures}.")


def gestures_on(_: str = "") -> str:
    """Turn gesture control on (live + persisted)."""
    if _cfg_flag("KINECT_GESTURES_ENABLED"):
        already = "Gesture control is already on, sir."
    else:
        already = None
    persisted = _set_gestures_enabled(True)
    kb = _bridge()
    sensor_note = ""
    if kb is not None:
        try:
            if not kb.get_enabled():
                sensor_note = (" Note the Kinect itself is still off — enable it "
                               "so I can actually see your gestures.")
        except Exception:
            pass
    if already:
        return already + sensor_note
    msg = "Gesture control on, sir — wave to wake me, raise a hand to confirm, swipe to cancel."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + sensor_note


def gestures_off(_: str = "") -> str:
    """Turn gesture control off (live + persisted)."""
    if not _cfg_flag("KINECT_GESTURES_ENABLED"):
        return "Gesture control is already off, sir."
    persisted = _set_gestures_enabled(False)
    msg = "Gesture control off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["gesture_status"] = gesture_status
    actions["gestures_on"] = gestures_on
    actions["gestures_off"] = gestures_off

    # Guard against duplicate pollers on skill reload (same OS-thread-name check
    # face_tracker uses). The loop self-gates on KINECT_GESTURES_ENABLED +
    # staging each tick, so it's cheap to leave running even when disabled.
    if any(th.name == _THREAD_NAME and th.is_alive()
           for th in threading.enumerate()):
        print("  [gestures] poller already running — skipping duplicate (reload)")
    else:
        t = threading.Thread(target=_poll_loop, daemon=True, name=_THREAD_NAME)
        t.start()
        print(f"  [gestures] gesture poller active (~{GESTURE_POLL_HZ:.0f} Hz; "
              "opt-in via KINECT_GESTURES_ENABLED, off by default)")
