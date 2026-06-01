"""
presence_wellness skill — soft nudge after 90 minutes of uninterrupted desk
presence.

Concept: a "focus block" is a continuous stretch in which JARVIS has reason to
believe the user is actively at the desk. Reasons (any one is enough):
  • face_tracker has recently seen the user at a monitor (not "away"),
  • workshop_mode is engaged (a CAD / slicer app is open),
  • the system has registered keyboard/mouse input within RECENT_INPUT_WINDOW.

The block survives short interruptions — a 30-second glance away, a quick
walk to grab coffee — but resets when no presence signal has fired for
BREAK_RESET_SECONDS (default 5 min).

Once the block reaches FOCUS_BLOCK_SECONDS (default 90 min) and no gate is
blocking, JARVIS volunteers one of:
  "You've been at it for an hour and a half, sir. Hydration, perhaps?"
  "If I may, sir — your eyes will thank you for a thirty-second break."

After firing the nudge enters a SNOOZE_SECONDS (default 60 min) cooldown,
during which the block tracker keeps running but no further nudge fires.

Gates (all must allow before nudging):
  • bobert_companion._sleep_mode[0] / _standby_mode[0] must be False
  • No window matches CALL_WINDOW_HINTS (Teams / Zoom / Meet / Webex / Discord)
  • skill_bambu_monitor must not report an active print (gcode_state RUNNING)

Actions registered:
  wellness_status — verbally report current block length, snooze remaining,
                    and which gates (if any) are currently blocking.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import random
import sys
import threading
import time

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_speech_lock = threading.Lock()

# ── Config ────────────────────────────────────────────────────────────────
WELLNESS_POLL_SECONDS    = 60.0
FOCUS_BLOCK_SECONDS      = 90 * 60   # 90 minutes uninterrupted to trigger
BREAK_RESET_SECONDS      = 5 * 60    # no presence for 5 min resets the block
RECENT_INPUT_WINDOW      = 5 * 60    # keyboard/mouse within 5 min counts as present
SNOOZE_SECONDS           = 60 * 60   # 1 hour cooldown after a fired nudge
INITIAL_DELAY_SECONDS    = 120.0     # let JARVIS boot before we start counting

NUDGE_LINES = (
    "You've been at it for an hour and a half, sir. Hydration, perhaps?",
    "If I may, sir — your eyes will thank you for a thirty-second break.",
)

# Window-title fragments indicating the user is on a call — mirrors the list
# used by anticipation_engine so wellness nudges follow the same etiquette.
CALL_WINDOW_HINTS = (
    "meeting now",
    "meeting in ",
    " | microsoft teams meeting",
    "microsoft teams meeting |",
    "zoom meeting",
    "zoom - meeting",
    "webex meetings",
    "google meet -",
    "meet -",
    "discord call",
)

# ── State ─────────────────────────────────────────────────────────────────
_state_lock        = threading.Lock()
_block_started_at  = [0.0]   # epoch — when the current focus block began
_last_presence_at  = [0.0]   # epoch — most recent presence signal observed
_last_nudge_at     = [0.0]   # epoch — most recent fired nudge
_last_phrase_idx   = [-1]    # avoid back-to-back repeats of the same line


def _enqueue_speech(message: str) -> None:
    """Route a wellness nudge through bobert_companion.proactive_announce()
    — the canonical writer for pending_speech.json. Funnelling every skill
    through that one helper eliminates the cross-skill read-modify-write race.
    Falls back to core.atomic_io._atomic_write_json — the same shared helper
    skills/timer.py uses — when the parent module isn't loaded yet
    (import-time / unit tests) or the announce call fails, so the nudge isn't
    silently lost and the fallback shares one race-safe write path with
    timer.py rather than rolling its own."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer) and announcer(message, source="wellness"):
            return
    except Exception:
        pass

    with _speech_lock:
        data: list = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [wellness] speech-queue write failed ({e}); nudge: {message}")


# ── Presence signals ──────────────────────────────────────────────────────

def _get_system_idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input system-wide (Windows).
    Returns a very large value on non-Windows / on failure so missing input
    info isn't interpreted as "user is here"."""
    try:
        from ctypes import Structure, c_uint, sizeof, windll, byref
        class _LASTINPUTINFO(Structure):
            _fields_ = [("cbSize", c_uint), ("dwTime", c_uint)]
        info = _LASTINPUTINFO()
        info.cbSize = sizeof(info)
        if not windll.user32.GetLastInputInfo(byref(info)):
            return float("inf")
        millis = windll.kernel32.GetTickCount() - info.dwTime
        if millis < 0:
            return float("inf")
        return millis / 1000.0
    except Exception:
        return float("inf")


def _face_tracker_at_desk() -> bool | None:
    """True if face_tracker has recently seen the user at a monitor, False if
    'away', None if the tracker isn't loaded or hasn't established gaze."""
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return None
    snap_func = getattr(mod, "_snapshot_state", None)
    if snap_func is None:
        return None
    try:
        snap = snap_func()
    except Exception:
        return None
    if not snap.get("last_sample_at"):
        return None
    monitor = snap.get("current_monitor")
    if monitor in ("left", "right", "middle_or_top"):
        return True
    if monitor == "away":
        return False
    return None


def _workshop_mode_active() -> bool:
    mod = sys.modules.get("skill_workshop_mode")
    if mod is None:
        return False
    try:
        flag = getattr(mod, "_workshop_active", None)
        return bool(flag and flag[0])
    except Exception:
        return False


def _recent_input() -> bool:
    """True if there's been keyboard/mouse input within RECENT_INPUT_WINDOW."""
    return _get_system_idle_seconds() <= RECENT_INPUT_WINDOW


def _user_present() -> bool:
    """Composite presence signal — any one source is sufficient."""
    if _face_tracker_at_desk() is True:
        return True
    if _workshop_mode_active():
        return True
    if _recent_input():
        return True
    return False


# ── Gate checks ───────────────────────────────────────────────────────────

def _is_sleep_or_standby() -> bool:
    mod = sys.modules.get("bobert_companion")
    if mod is None:
        return False
    try:
        return bool(mod._sleep_mode[0]) or bool(mod._standby_mode[0])
    except Exception:
        return False


def _is_in_call() -> bool:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return False
    try:
        titles = [(getattr(w, "title", "") or "").lower()
                  for w in gw.getAllWindows()]
    except Exception:
        return False
    return any(hint in t for hint in CALL_WINDOW_HINTS for t in titles if t)


def _bambu_print_active() -> bool:
    """True if a Bambu print is currently running. We give the user space —
    a print in progress is exactly the kind of moment a wellness nudge would
    feel mistimed."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return False
    try:
        lock = getattr(mod, "_state_lock", None)
        state = getattr(mod, "_state", None)
        if state is None:
            return False
        if lock is not None:
            with lock:
                gcode_state = (state.get("gcode_state") or "").upper()
        else:
            gcode_state = (state.get("gcode_state") or "").upper()
        return gcode_state == "RUNNING"
    except Exception:
        return False


def _gate_reasons() -> list[str]:
    reasons: list[str] = []
    if _is_sleep_or_standby():
        reasons.append("sleep mode")
    if _is_in_call():
        reasons.append("on a call")
    if _bambu_print_active():
        reasons.append("Bambu print active")
    return reasons


# ── Poll loop ─────────────────────────────────────────────────────────────

def _pick_nudge_line() -> str:
    """Pick a nudge phrase, avoiding back-to-back repeats."""
    if len(NUDGE_LINES) == 1:
        return NUDGE_LINES[0]
    while True:
        idx = random.randrange(len(NUDGE_LINES))
        if idx != _last_phrase_idx[0]:
            _last_phrase_idx[0] = idx
            return NUDGE_LINES[idx]


def _poll_once() -> None:
    now = time.time()
    present = _user_present()

    with _state_lock:
        if present:
            if _block_started_at[0] == 0.0:
                _block_started_at[0] = now
            _last_presence_at[0] = now
        else:
            # No presence signal this tick. If we've gone without one for
            # BREAK_RESET_SECONDS, reset the block — the user has actually
            # stepped away.
            if (_last_presence_at[0]
                    and (now - _last_presence_at[0]) >= BREAK_RESET_SECONDS):
                _block_started_at[0] = 0.0

        if _block_started_at[0] == 0.0:
            return
        block_duration = now - _block_started_at[0]
        last_nudge = _last_nudge_at[0]

    if block_duration < FOCUS_BLOCK_SECONDS:
        return

    if last_nudge and (now - last_nudge) < SNOOZE_SECONDS:
        return

    gates = _gate_reasons()
    if gates:
        return

    line = _pick_nudge_line()
    with _state_lock:
        _last_nudge_at[0] = now

    print(f"  [wellness] firing presence nudge — {block_duration / 60:.0f} min "
          f"focus block: {line}")
    _enqueue_speech(line)


def _poll_loop() -> None:
    try:
        time.sleep(INITIAL_DELAY_SECONDS)
        while True:
            try:
                _poll_once()
            except Exception:
                logging.exception("[wellness] poll error")
            time.sleep(WELLNESS_POLL_SECONDS)
    except Exception:
        logging.exception("[wellness] poll loop terminated unexpectedly")


# ── Action handler ────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def register(actions):
    def wellness_status(_: str = "") -> str:
        now = time.time()
        with _state_lock:
            started   = _block_started_at[0]
            last_seen = _last_presence_at[0]
            last_fire = _last_nudge_at[0]
        block = (now - started) if started else 0.0
        idle  = _get_system_idle_seconds()
        gates = _gate_reasons()
        gate_str = ", ".join(gates) if gates else "none"

        if not started:
            return (f"No active focus block, sir — last input "
                    f"{_fmt_duration(idle)} ago.")

        snooze_remaining = max(0.0, SNOOZE_SECONDS - (now - last_fire)) if last_fire else 0.0
        snooze_str = (f"snoozed for {_fmt_duration(snooze_remaining)}"
                      if snooze_remaining > 0 else "ready to fire")
        last_seen_str = (f"{_fmt_duration(now - last_seen)} ago"
                         if last_seen else "never")
        return (f"Focus block running {_fmt_duration(block)}, sir — "
                f"last presence {last_seen_str}, gates: {gate_str}, "
                f"nudge {snooze_str}.")

    actions["wellness_status"] = wellness_status

    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print(f"  [wellness] presence watcher active — "
          f"focus threshold {FOCUS_BLOCK_SECONDS // 60} min, "
          f"snooze {SNOOZE_SECONDS // 60} min, poll every "
          f"{WELLNESS_POLL_SECONDS:.0f}s")
