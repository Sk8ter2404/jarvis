"""
Suit-up sequence skill — cinematic morning warm-restart boot.

Spec (jarvis_todo.md 2026-05-27 11:26, suit_up_sequence): play a 6–8 second
cinematic boot sequence on the holographic_overlay HUD when JARVIS first
wakes each morning. Three coordinated pieces:

  1. Arc-reactor spin-up animation on the HUD (driven by writing
     boot_phase / boot_started_at / boot_duration into hud_state.json — the
     same hook iron_man_boot.py uses).
  2. Sequential system check readouts, spoken one line at a time so the
     dry MCU "diagnostics → online → standing by" cadence reads on TTS:
        "Diagnostics: nominal."
        "Network: online."
        "Audio: <speaker_name>, connected."
        "Workshop: standing by."
  3. Final TTS line: "Welcome back, sir. Systems are yours."

Triggered from the session_resume startup hook on first warm-restart of
the day (replacing the plain warm-restart greeting). The 'first warm-
restart of the day' gate uses suit_up_state.json so a same-day JARVIS
restart doesn't re-fire the cinematic.

Actions registered:
  suit_up / suit_up_sequence — manually fire the full sequence. Bypasses
                               the once-per-day gate so the user can ask
                               for it verbally whenever they like.

Public API for bobert_companion.py:
  maybe_play_morning_suit_up(speak_fn, write_hud_state=None,
                             speaker_name=None) -> str
      Fires the sequence iff (a) we have a warm restart on record AND
      (b) suit_up hasn't fired today yet. Returns the final spoken line
      on a fire, '' otherwise. The caller can skip the plain warm-restart
      resume greeting when this returns non-empty.

  play_suit_up_sequence(speak_fn, write_hud_state=None,
                        speaker_name=None) -> str
      Fires unconditionally. Returns the final spoken line.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE   = os.path.join(_PROJECT_DIR, "suit_up_state.json")

# Animation budget. The HUD's iron-man overlay reads boot_started_at +
# boot_duration to know how long to keep the arc-reactor expanding. Tuned
# to match a typical 4-line readout + welcome (~6.5 s on a moderate TTS
# backend) so the rings finish filling roughly when the welcome line ends.
SUIT_UP_ANIMATION_SECONDS = 7.0

# Brief beat between diagnostic lines so each one lands distinctly rather
# than running into the next. 200 ms reads as snappy without sounding
# rushed; 0 ms makes the readouts blur together on a fast TTS backend.
INTER_LINE_PAUSE_SECONDS = 0.20

# Warm-restart window — must match bobert_companion.WARM_RESTART_WINDOW_SECONDS
# (18 hours). Re-declared here so the skill is self-contained: if
# bobert_companion's constant moves, the worst case is the cinematic fires
# slightly outside the official warm window — harmless.
WARM_RESTART_WINDOW_SECONDS = 18 * 3600

_state_lock = threading.Lock()


# ─── state file helpers ──────────────────────────────────────────────────

def _load_state() -> dict:
    with _state_lock:
        if not os.path.exists(_STATE_FILE):
            return {}
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}


def _save_state(state: dict) -> None:
    with _state_lock:
        try:
            dir_ = os.path.dirname(_STATE_FILE)
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp, _STATE_FILE)
            except Exception:
                try: os.unlink(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  [suit_up] state write failed: {e}")


# ─── warm-restart probe (mirrors bobert_companion._last_session_end_ts) ──

def _last_session_end_ts() -> float:
    """Best-effort epoch seconds of the most-recent previous session's end.
    Returns 0.0 when no prior session is on record or memory module isn't
    importable. Mirrors bobert_companion._last_session_end_ts so the skill
    stays self-contained even if the host module is mid-reload."""
    try:
        memory = importlib.import_module("memory")
        recent = memory.get_session_summaries("", limit=1)
        if not recent:
            return 0.0
        row = recent[0]
        ts = row.get("ts")
        if ts:
            try:
                return float(ts)
            except Exception:
                pass
        iso = row.get("iso_end") or row.get("iso_start") or ""
        if iso:
            try:
                return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))
            except Exception:
                pass
        date = row.get("date") or ""
        if date:
            try:
                return time.mktime(time.strptime(date + "T20:00:00",
                                                 "%Y-%m-%dT%H:%M:%S"))
            except Exception:
                pass
    except Exception:
        pass
    return 0.0


def _is_warm_restart() -> bool:
    last_ts = _last_session_end_ts()
    if last_ts <= 0:
        return False
    age = time.time() - last_ts
    return 0 < age <= WARM_RESTART_WINDOW_SECONDS


# ─── audio device name lookup (best-effort, never raises) ────────────────

def _resolve_speaker_name(explicit: Optional[str]) -> str:
    """Return a short speakable device name for the readout line.
    Caller can pass an explicit name (e.g. bobert_companion already knows
    'Gaming Headset'); otherwise we probe sounddevice ourselves. Falls back
    to 'system default' so the line still parses cleanly."""
    if explicit:
        s = str(explicit).strip()
        if s:
            return s
    try:
        bc = importlib.import_module("bobert_companion")
        raw = bc.get_current_speaker_name()
        # 'get_current_speaker_name' returns "[N] Description, API" — strip.
        if raw and raw.startswith("[") and "] " in raw:
            raw = raw.split("] ", 1)[1]
        friendly = bc._friendly_device_name(raw)
        if friendly:
            return friendly
    except Exception:
        pass
    return "system default"


# ─── holographic_overlay coordination ───────────────────────────────────

def _holo_overlay_module():
    """Return the loaded holographic_overlay skill module, or None when the
    skill failed to load. load_skills() exposes it as skill_holographic_overlay
    in sys.modules; we look it up there to avoid a hard import dependency
    (this skill must still work if holographic_overlay is disabled)."""
    return sys.modules.get("skill_holographic_overlay")


def _ensure_holo_overlay_up() -> bool:
    """Bring the fullscreen holographic_overlay HUD up for the cinematic.

    Returns True iff WE launched it (so the caller can tear it back down
    afterwards — if it was already engaged by the user, we leave it alone).
    Silent and returns False on any error so a missing overlay never blocks
    the spoken sequence."""
    mod = _holo_overlay_module()
    if mod is None:
        return False
    try:
        was_alive = mod._overlay_is_alive()
    except Exception:
        was_alive = False
    if was_alive:
        return False
    try:
        ok, _msg = mod._launch_overlay()
        return bool(ok)
    except Exception as e:
        print(f"  [suit_up] overlay launch failed: {e}")
        return False


def _dismiss_holo_overlay() -> None:
    """Tear down the fullscreen holographic_overlay HUD after the cinematic
    so it doesn't cover the user's desktop for the rest of the morning.
    Caller only invokes this when _ensure_holo_overlay_up() returned True."""
    mod = _holo_overlay_module()
    if mod is None:
        return
    try:
        mod._shutdown_overlay()
    except Exception as e:
        print(f"  [suit_up] overlay dismiss failed: {e}")


# ─── cinematic sequence ──────────────────────────────────────────────────

def _start_animation(write_hud_state: Optional[Callable]) -> float:
    """Kick off the HUD arc-reactor spin-up. Returns the start timestamp so
    the caller can measure elapsed time when it's time to clear the phase.

    Writes boot_phase='powering' (NOT 'suit_up') so the existing
    jarvis_hud.py boot-animation hook fires — that hook only matches
    'powering' literally, so any other value would render no animation at
    all. The fullscreen holographic_overlay is the spec's primary stage;
    the slim HUD animation is a complementary signal."""
    started_at = time.time()
    if write_hud_state is None:
        return started_at
    try:
        write_hud_state(
            boot_phase="powering",
            boot_started_at=started_at,
            boot_duration=SUIT_UP_ANIMATION_SECONDS,
            state="Initialising",
        )
    except Exception as e:
        print(f"  [suit_up] HUD start publish failed: {e}")
    return started_at


def _clear_animation(write_hud_state: Optional[Callable],
                     started_at: float) -> None:
    """Clear the HUD boot_phase so the overlay returns to its normal
    rendering. Held open at least until SUIT_UP_ANIMATION_SECONDS so the
    rings get a chance to fill all the way out on a fast TTS backend."""
    if write_hud_state is None:
        return
    try:
        elapsed = time.time() - started_at
        if elapsed < SUIT_UP_ANIMATION_SECONDS:
            time.sleep(SUIT_UP_ANIMATION_SECONDS - elapsed)
        write_hud_state(
            boot_phase="",
            boot_started_at=0.0,
            state="Idle",
        )
    except Exception as e:
        print(f"  [suit_up] HUD clear failed: {e}")


def _build_diagnostic_lines(speaker_name: str) -> list[str]:
    """Compose the four sequential system check readouts. Speaker name is
    interpolated so the audio line names the actual headset."""
    audio_line = (f"Audio: {speaker_name}, connected." if speaker_name
                  else "Audio: connected.")
    return [
        "Diagnostics: nominal.",
        "Network: online.",
        audio_line,
        "Workshop: standing by.",
    ]


def play_suit_up_sequence(speak_fn: Callable[[str], None],
                          write_hud_state: Optional[Callable] = None,
                          speaker_name: Optional[str] = None) -> str:
    """Fire the cinematic boot sequence unconditionally.

    Args:
        speak_fn: synchronous TTS callable. Each line blocks until spoken.
        write_hud_state: optional HUD state writer. None skips the visual;
            the spoken lines still fire.
        speaker_name: optional pre-resolved short speaker name. When None
            we probe sounddevice ourselves.

    Returns:
        The final spoken line, so callers can append it to
        conversation_history.
    """
    speaker = _resolve_speaker_name(speaker_name)
    diagnostic_lines = _build_diagnostic_lines(speaker)
    welcome = "Welcome back, sir. Systems are yours."

    # Bring the fullscreen holographic_overlay HUD up so the arc-reactor
    # has a stage to spin up on (the spec's primary visual surface).
    # `we_launched_overlay` tells us whether to dismiss it ourselves at
    # the end — if the user already had the overlay engaged, leave it
    # alone.
    we_launched_overlay = _ensure_holo_overlay_up()

    started_at = _start_animation(write_hud_state)

    for line in diagnostic_lines:
        try:
            speak_fn(line)
        except Exception as e:
            print(f"  [suit_up] readout speak failed ({line!r}): {e}")
        # Brief beat between readouts — keeps each one distinct.
        try:
            time.sleep(INTER_LINE_PAUSE_SECONDS)
        except Exception:
            pass

    try:
        speak_fn(welcome)
    except Exception as e:
        print(f"  [suit_up] welcome speak failed: {e}")

    _clear_animation(write_hud_state, started_at)

    if we_launched_overlay:
        _dismiss_holo_overlay()

    return welcome


def maybe_play_morning_suit_up(speak_fn: Callable[[str], None],
                               write_hud_state: Optional[Callable] = None,
                               speaker_name: Optional[str] = None) -> str:
    """Fire the suit-up sequence iff (a) we have a warm restart on record
    AND (b) the cinematic hasn't fired today yet. Returns the final spoken
    line on a fire, '' otherwise. The caller should skip the plain
    warm-restart resume greeting when this returns non-empty."""
    today = time.strftime("%Y-%m-%d")
    state = _load_state()
    if state.get("last_fired_date") == today:
        return ""
    if not _is_warm_restart():
        return ""

    print(f"  [suit_up] firing morning cinematic (first warm restart {today})")
    welcome = play_suit_up_sequence(speak_fn=speak_fn,
                                    write_hud_state=write_hud_state,
                                    speaker_name=speaker_name)
    state["last_fired_date"] = today
    state["last_fired_ts"]   = time.time()
    state["last_reason"]     = "first warm restart of day"
    _save_state(state)
    return welcome


# ─── action registration ────────────────────────────────────────────────

def _act_suit_up(_: str = "") -> str:
    """Verbal trigger: 'suit up' / 'run the suit-up sequence'. Bypasses the
    once-per-day gate so the user can request it whenever they like."""
    # Bind speak_fn + write_hud_state lazily so the skill stays self-
    # contained — importing bobert_companion at module load would create
    # a circular import during load_skills().
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception as e:
        return f"I'm afraid I can't reach the speech subsystem, sir — {e}."
    speak_fn        = getattr(bc, "_speak", None)
    write_hud_state = getattr(bc, "_write_hud_state", None)
    if not callable(speak_fn):
        return "I'm afraid the TTS layer is offline, sir."
    play_suit_up_sequence(speak_fn=speak_fn,
                          write_hud_state=write_hud_state)
    # Record a manual fire so the morning auto-trigger doesn't double up
    # if the user runs this in the morning before the warm-restart hook.
    today = time.strftime("%Y-%m-%d")
    state = _load_state()
    state["last_fired_date"] = today
    state["last_fired_ts"]   = time.time()
    state["last_reason"]     = "manual trigger"
    _save_state(state)
    # The spoken welcome line is the user-visible answer — replicate it
    # so the dispatcher prints something sensible in its echo log.
    return "Welcome back, sir. Systems are yours."


def register(actions: dict) -> None:
    actions["suit_up"]          = _act_suit_up
    actions["suit_up_sequence"] = _act_suit_up
