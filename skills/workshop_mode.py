"""
Workshop mode skill for JARVIS.

Polls open windows every WORKSHOP_POLL_SECONDS for the title of any known
3D-modeling / CAD / slicer app (Bambu Studio, Fusion 360, FreeCAD, SolidWorks,
OnShape, Blender, OpenSCAD, OrcaSlicer, PrusaSlicer, Cura, etc.). When a CAD
app appears, JARVIS automatically enters 'workshop mode':

  • TTS playback volume is scaled to WORKSHOP_TTS_SCALE (0.7 = 30% reduction).
    Implemented by wrapping bobert_companion.play_with_lipsync so the audio
    array is multiplied in place — restored on exit.
  • The system prompt gets a one-line addendum telling JARVIS to keep replies
    to a single sentence unless asked for more. Applied by mutating
    bobert_companion._system_prompt (the per-call sites read this string
    freshly, so the change takes effect on the very next LLM turn).
  • Announces entry with 'Workshop mode engaged, sir.'
  • If the Bambu H2D is mid-print at entry time, also queues a one-line
    print status offer: 'The H2D is at layer 47 of 312, sir — about 18
    minutes remaining.'

When every known CAD window is closed (or replaced by something else for at
least WORKSHOP_EXIT_GRACE_SECONDS), JARVIS exits workshop mode, restores
playback / prompt to defaults, and says 'Workshop mode disengaged, sir.'

Actions added:
  workshop_status   — verbally report whether workshop mode is engaged
                      ('Workshop mode is engaged, sir — Bambu Studio is open.'
                      / 'Workshop mode is not currently engaged, sir.')

Configurable at the top of this file. No external deps beyond pygetwindow,
which is already used by bobert_companion.
"""
import importlib
import logging
import os
import sys
import threading
import time

# ── Config ────────────────────────────────────────────────────────────────
WORKSHOP_POLL_SECONDS      = 30.0
WORKSHOP_TTS_SCALE         = 0.7     # 30% volume reduction (1.0 - 0.3)
WORKSHOP_EXIT_GRACE_SECONDS = 60.0   # CAD must be gone this long before exit

# Window-title fragments (lowercased) that indicate a 3D / CAD app. Matched
# substring against window titles, longest first so 'autodesk fusion 360'
# beats a stray 'fusion' in some other app.
CAD_WINDOW_HINTS = (
    "autodesk fusion 360",
    "bambu studio",
    "orcaslicer",
    "prusaslicer",
    "ultimaker cura",
    "solidworks",
    "freecad",
    "openscad",
    "onshape",
    "tinkercad",
    "blender",
    "fusion 360",
    "fusion360",
    "cura",
)

WORKSHOP_PROMPT_ADDENDUM = (
    "\n\n[Workshop mode]\n"
    "Sir is in the workshop — keep responses to one sentence unless asked "
    "for more. Skip pleasantries. He's concentrating on a 3D / CAD task."
)

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Mode state ────────────────────────────────────────────────────────────
_mode_lock          = threading.Lock()
# _state_lock guards the lightweight cross-thread state cells (_last_cad_seen_at,
# and — see the line-273 finding — _current_app_title when written from the
# poll loop). Separate from _mode_lock so the poll loop can update the seen-at
# timestamp without blocking on enter/exit, which themselves hold _mode_lock.
_state_lock         = threading.Lock()
_workshop_active    = [False]
_current_app_title  = [None]      # str or None — last matched CAD window title
_last_cad_seen_at   = [0.0]       # epoch — used for the exit-grace timer

# Saved originals so we can restore on exit / shutdown
_saved_play_with_lipsync = [None]   # callable
_saved_system_prompt     = [None]   # str


def _enqueue_speech(message: str) -> None:
    """Route a workshop alert through bobert_companion.proactive_announce() —
    the canonical writer for pending_speech.json. Funnelling every skill
    through that one helper eliminates the cross-skill read-modify-write race
    that an independent local fallback would reintroduce. If the parent module
    isn't loaded yet (import-time / unit tests) or the announce call fails,
    the alert is logged to the console so it isn't silently lost."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer) and announcer(message, source="workshop"):
            return
    except Exception as e:
        print(f"  [workshop] speech-queue write failed ({e}); message: {message}")
        return
    print(f"  [workshop] speech-queue unavailable; message: {message}")


def _find_cad_window():
    """Return (matched_hint, full_window_title) of the first open CAD window,
    or (None, None) when no known CAD app is visible."""
    try:
        import pygetwindow as gw
    except ImportError:
        return None, None
    try:
        titles = [(w.title or "") for w in gw.getAllWindows() if w.title]
    except Exception:
        return None, None
    for hint in CAD_WINDOW_HINTS:
        for t in titles:
            if hint in t.lower():
                return hint, t
    return None, None


def _friendly_app_name(hint: str) -> str:
    """Map an internal hint to a speakable display name."""
    return {
        "autodesk fusion 360": "Fusion 360",
        "fusion 360":          "Fusion 360",
        "fusion360":           "Fusion 360",
        "bambu studio":        "Bambu Studio",
        "orcaslicer":          "OrcaSlicer",
        "prusaslicer":         "PrusaSlicer",
        "ultimaker cura":      "Cura",
        "cura":                "Cura",
        "solidworks":          "SolidWorks",
        "freecad":             "FreeCAD",
        "openscad":            "OpenSCAD",
        "onshape":             "OnShape",
        "tinkercad":           "Tinkercad",
        "blender":             "Blender",
    }.get(hint, hint)


def _maybe_get_print_status_line() -> str | None:
    """If the Bambu monitor is loaded and a print is running, return a one-line
    JARVIS-style status sentence. None when nothing useful to say."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return None
    try:
        state_lock = getattr(mod, "_state_lock")
        state      = getattr(mod, "_state")
        with state_lock:
            gcode_state = (state.get("gcode_state") or "").upper()
            layer       = state.get("layer_num")
            total       = state.get("total_layer")
            remaining   = state.get("mc_remaining")
            last_update = state.get("last_update", 0.0)
        if last_update == 0.0:
            return None
        if gcode_state != "RUNNING":
            return None
        parts = []
        if layer and total:
            parts.append(f"the H2D is at layer {layer} of {total}")
        else:
            parts.append("the H2D is mid-print")
        try:
            fmt = getattr(mod, "_format_minutes")
            rem_str = fmt(remaining) if remaining else ""
        except Exception:
            rem_str = ""
        if rem_str:
            parts.append(f"about {rem_str} remaining")
        return "By the way, " + ", ".join(parts) + ", sir."
    except Exception:
        return None


def _enter_workshop_mode(matched_hint: str, full_title: str) -> None:
    """Apply TTS scaling + prompt addendum + announce entry. Idempotent."""
    with _mode_lock:
        if _workshop_active[0]:
            return
        _workshop_active[0]   = True
        _current_app_title[0] = full_title

        # Wrap play_with_lipsync to scale audio
        try:
            import bobert_companion
            if _saved_play_with_lipsync[0] is None:
                original = bobert_companion.play_with_lipsync
                _saved_play_with_lipsync[0] = original

                def _scaled(audio, sr, _orig=original):
                    try:
                        if _workshop_active[0] and audio is not None:
                            audio = audio * WORKSHOP_TTS_SCALE
                    except Exception:
                        # Worst case: pass the original audio through
                        pass
                    return _orig(audio, sr)

                bobert_companion.play_with_lipsync = _scaled

            # Extend the system prompt for as long as we're in workshop mode.
            # _call_llm and get_followup_response both read _system_prompt
            # freshly each turn, so mutating it here takes effect immediately.
            if _saved_system_prompt[0] is None:
                _saved_system_prompt[0] = bobert_companion._system_prompt
            bobert_companion._system_prompt = (
                _saved_system_prompt[0] + WORKSHOP_PROMPT_ADDENDUM
            )
        except Exception as e:
            print(f"  [workshop] failed to install hooks: {e}")

    app = _friendly_app_name(matched_hint)
    print(f"  [workshop] entering workshop mode — detected '{full_title}' (→ {app})")
    _enqueue_speech("Workshop mode engaged, sir.")
    print_line = _maybe_get_print_status_line()
    if print_line:
        _enqueue_speech(print_line)


def _exit_workshop_mode() -> None:
    """Restore TTS + prompt to their pre-workshop state and announce."""
    with _mode_lock:
        if not _workshop_active[0]:
            return
        _workshop_active[0]   = False
        _current_app_title[0] = None

        try:
            import bobert_companion
            if _saved_play_with_lipsync[0] is not None:
                bobert_companion.play_with_lipsync = _saved_play_with_lipsync[0]
                _saved_play_with_lipsync[0] = None
            if _saved_system_prompt[0] is not None:
                bobert_companion._system_prompt = _saved_system_prompt[0]
                _saved_system_prompt[0] = None
        except Exception as e:
            print(f"  [workshop] failed to restore hooks: {e}")

    print("  [workshop] exiting workshop mode — no CAD windows visible")
    _enqueue_speech("Workshop mode disengaged, sir.")


def _poll_loop() -> None:
    """Watch open windows every WORKSHOP_POLL_SECONDS and toggle mode."""
    try:
        # Initial settle delay so we don't race with whatever's coming up at boot
        time.sleep(WORKSHOP_POLL_SECONDS)
        while True:
            try:
                hint, title = _find_cad_window()
                now = time.time()
                if hint is not None:
                    with _state_lock:
                        _last_cad_seen_at[0] = now
                    if not _workshop_active[0]:
                        _enter_workshop_mode(hint, title)
                    else:
                        # Track the latest matched title so workshop_status can
                        # report which app is currently driving the mode.
                        with _state_lock:
                            _current_app_title[0] = title
                else:
                    # No CAD window visible. Only exit after a grace period so a
                    # brief Bambu Studio reload doesn't toggle us off-and-on.
                    if _workshop_active[0]:
                        with _state_lock:
                            last = _last_cad_seen_at[0]
                        if last == 0.0:
                            # Shouldn't happen, but exit safely
                            _exit_workshop_mode()
                        elif (now - last) >= WORKSHOP_EXIT_GRACE_SECONDS:
                            _exit_workshop_mode()
            except Exception:
                logging.exception("  [workshop] poll loop iteration error")
            time.sleep(WORKSHOP_POLL_SECONDS)
    except Exception:
        logging.exception("  [workshop] poll loop crashed — thread exiting")


def register(actions):
    def workshop_status(_: str = "") -> str:
        with _mode_lock:
            active = _workshop_active[0]
            title  = _current_app_title[0]
        if active and title:
            return f"Workshop mode is engaged, sir — '{title}' is open."
        if active:
            return "Workshop mode is engaged, sir."
        return "Workshop mode is not currently engaged, sir."

    actions["workshop_status"] = workshop_status

    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print(f"  [workshop] watcher active — polling for CAD apps every "
          f"{WORKSHOP_POLL_SECONDS:.0f}s")
