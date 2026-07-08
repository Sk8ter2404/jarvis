"""
Anticipation engine — proactive in-character commentary.

Runs a 60-second background poll and, when the stars align, volunteers ONE
JARVIS-style line through the pending_speech queue. Inputs:
  • pattern_memory.maybe_pattern_offer()  — strong day-of-week / time-of-day habits
  • focused window dwell                  — long sessions in productivity apps
  • current local time                    — late-hour active sessions

Gating (all must pass before any trigger is considered):
  • Cooldown:        no more than once every ANTICIPATION_COOLDOWN_MINUTES
                     (default 20). Persisted across restarts.
  • Not in a call:   suppressed while any window matches a known
                     Teams / Zoom / Meet / Webex meeting title.
  • Not asleep:      suppressed while bobert_companion._sleep_mode[0] OR
                     _standby_mode[0] is True.
  • Not away:        if face_tracker reports gaze == "away", skip. Unknown /
                     no-tracker is treated permissively (don't suppress).
  • Late-night:      between 23:00 and 07:00 we only fire if the user has
                     spoken in the last 30 minutes (no whispering at an
                     empty desk in the middle of the night).
  • Probability:     even when a trigger matches, fire with probability
                     FIRE_PROBABILITY (0.35) so the engine feels rare,
                     not punctual.

Actions registered:
  anticipation_status — short status report on the engine: last fire,
                        cooldown remaining, current dwell window, in-call.

Config knobs in bobert_companion.py:
  ANTICIPATION_ENABLED            (bool, default True)
  ANTICIPATION_COOLDOWN_MINUTES   (int,  default 20)
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "anticipation_state.json")

POLL_INTERVAL_SECONDS = 60.0
INITIAL_DELAY_SECONDS = 90       # let JARVIS finish booting
FIRE_PROBABILITY      = 0.35     # per-poll roll after a trigger matches

# Long-dwell trigger thresholds
LONG_DWELL_MIN_SECONDS = 2 * 3600    # 2 hours on same productivity window
LONG_DWELL_REPEAT_GAP  = 90 * 60     # don't remark on same window again within 90 min

# Late-hour active-session trigger
LATE_HOUR_THRESHOLD_HOUR = 23        # 23:00 onward counts as late
LATE_HOUR_END_HOUR       = 7         # ...up until 07:00 the next morning
LATE_HOUR_ACTIVE_WINDOW  = 30 * 60   # user must have spoken within 30 min

# Window-title fragments that indicate productivity / deep-work apps where a
# multi-hour dwell is worth remarking on. Matched case-insensitively as
# substrings, longest-first so 'autodesk fusion 360' beats a generic 'fusion'.
PRODUCTIVITY_WINDOW_HINTS = (
    "bambu studio",
    "autodesk fusion 360",
    "fusion 360",
    "orcaslicer",
    "prusaslicer",
    "cura",
    "solidworks",
    "freecad",
    "openscad",
    "onshape",
    "blender",
    "visual studio code",
    "vscode",
    "intellij",
    "pycharm",
    "webstorm",
    "android studio",
    "xcode",
    "photoshop",
    "illustrator",
    "premiere pro",
    "after effects",
    "davinci resolve",
    "figma",
    "notion",
    "obsidian",
    "logic pro",
    "ableton",
)

# Window-title fragments indicating the user is currently on a call. If ANY
# window title contains one of these, suppress the engine.
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

_state_lock  = threading.Lock()
_speech_lock = threading.Lock()

# Per-process dwell tracking (resets on restart — acceptable, since multi-hour
# windows survive a quick restart and the trigger re-arms on its own).
_dwell_lock  = threading.Lock()
_dwell_state: dict = {
    "window":      "",      # current focused window title
    "started_at":  0.0,     # when we first observed it
    "last_seen":   0.0,
}


# ─── speech queue ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Route a proactive line through bobert_companion.proactive_announce()
    — the canonical, serialized writer for pending_speech.json — and only fall
    back to a direct atomic write when the parent module isn't importable yet
    (import-time / unit tests). Going through proactive_announce restores the
    shared-lock serialization, the focus / DND gate, and the 50-entry cap that a
    bare local write silently bypasses. Mirrors the exact pattern in
    skills/wellness.py and skills/daily_recap.py so every co-writer of
    pending_speech.json funnels through the same path. 2026-07-08.
    """
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer) and announcer(message, source="anticipation"):
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
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
            print(f"  [anticipate] speech-queue write failed ({e}); line: {message}")


# ─── config + persistent state ───────────────────────────────────────────

def _read_config() -> dict:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    return {
        "enabled":  bool(getattr(bc, "ANTICIPATION_ENABLED",         True)) if bc else True,
        "cooldown": int (getattr(bc, "ANTICIPATION_COOLDOWN_MINUTES", 20))  if bc else 20,
    }


def _load_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    with _state_lock:
        try:
            fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp, _STATE_FILE)
            except Exception:
                try: os.unlink(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  [anticipate] could not persist state: {e}")


# ─── environment inspection ──────────────────────────────────────────────

def _all_window_titles() -> list[str]:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return []
    out: list[str] = []
    try:
        for w in gw.getAllWindows():
            t = getattr(w, "title", "") or ""
            if t.strip():
                out.append(t)
    except Exception:
        pass
    return out


def _focused_window_title() -> str:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return ""
    try:
        w = gw.getActiveWindow()
        return (getattr(w, "title", "") or "").strip()
    except Exception:
        return ""


def _is_in_call() -> bool:
    titles = _all_window_titles()
    if not titles:
        return False
    lowered = [t.lower() for t in titles]
    for hint in CALL_WINDOW_HINTS:
        for t in lowered:
            if hint in t:
                return True
    return False


def _is_sleep_or_standby() -> bool:
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return False
    try:
        if getattr(bc, "_sleep_mode")[0]:
            return True
        if getattr(bc, "_standby_mode")[0]:
            return True
    except Exception:
        return False
    return False


def _user_at_desk() -> bool | None:
    """True if face_tracker has recently seen the user, False if 'away', None
    if tracker isn't loaded / has no data."""
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


def _last_speech_age_seconds() -> float | None:
    """Seconds since the user last spoke, per bobert_companion.last_speech_time.
    Returns None if the value isn't available."""
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return None
    ts = getattr(bc, "last_speech_time", None)
    if not isinstance(ts, (int, float)):
        return None
    return max(0.0, time.time() - float(ts))


# ─── dwell tracking ──────────────────────────────────────────────────────

def _shorten_app_name(title: str) -> str:
    """Map a verbose window title to a clean app name for the spoken line."""
    if not title:
        return ""
    lower = title.lower()
    for hint in sorted(PRODUCTIVITY_WINDOW_HINTS, key=len, reverse=True):
        if hint in lower:
            return _title_case_app(hint)
    # Fallback: take the right-hand component after the last " - " or " | "
    for sep in (" — ", " - ", " | "):
        if sep in title:
            tail = title.split(sep)[-1].strip()
            if tail and len(tail) <= 40:
                return tail
    return title[:40]


def _title_case_app(name: str) -> str:
    """Title-case an app hint while keeping known acronyms uppercase."""
    upper_words = {"vscode", "intellij", "freecad", "openscad", "ableton"}
    if name in upper_words:
        return {
            "vscode":    "VS Code",
            "intellij":  "IntelliJ",
            "freecad":   "FreeCAD",
            "openscad":  "OpenSCAD",
            "ableton":   "Ableton",
        }[name]
    return " ".join(w.capitalize() for w in name.split())


def _is_productivity_window(title: str) -> bool:
    if not title:
        return False
    lower = title.lower()
    return any(hint in lower for hint in PRODUCTIVITY_WINDOW_HINTS)


def _update_dwell(focused: str) -> None:
    """Refresh the in-memory dwell-tracking record. Called once per poll."""
    now = time.time()
    with _dwell_lock:
        if focused != _dwell_state["window"]:
            _dwell_state["window"]     = focused
            _dwell_state["started_at"] = now
            _dwell_state["last_seen"]  = now
        else:
            _dwell_state["last_seen"]  = now


def _current_dwell_seconds() -> tuple[str, float]:
    with _dwell_lock:
        if not _dwell_state["window"] or not _dwell_state["started_at"]:
            return "", 0.0
        return _dwell_state["window"], time.time() - _dwell_state["started_at"]


# ─── trigger selection ───────────────────────────────────────────────────

def _format_hours_minutes(seconds: float) -> str:
    total_min = int(seconds // 60)
    h, m = divmod(total_min, 60)
    if h == 0:
        return f"{m} minutes"
    if m == 0:
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{h} hour{'s' if h != 1 else ''} and {m} minutes"


def _format_clock(now: time.struct_time) -> str:
    hour = now.tm_hour
    minute = now.tm_min
    suffix = "AM" if hour < 12 else "PM"
    disp_hour = hour % 12 or 12
    return f"{disp_hour}:{minute:02d} {suffix}"


def _try_pattern_offer() -> str:
    """Surface a strong day-of-week / time-of-day habit from the pattern
    memory (top-level ``memory`` module).
    maybe_pattern_offer() self-throttles to once-per-day per pattern key."""
    try:
        pm = importlib.import_module("memory")
    except Exception:
        return ""
    try:
        return pm.maybe_pattern_offer() or ""
    except Exception as e:
        print(f"  [anticipate] pattern_offer failed: {e}")
        return ""


def _try_long_dwell(state: dict) -> tuple[str, str]:
    """Productivity-window dwell trigger. Returns (line, dwell_key) or ('','')."""
    title, dwell = _current_dwell_seconds()
    if not title or dwell < LONG_DWELL_MIN_SECONDS:
        return "", ""
    if not _is_productivity_window(title):
        return "", ""
    app = _shorten_app_name(title)
    dwell_key = f"dwell:{app}"
    last_for_app = float(state.get("last_dwell_remark_at", {}).get(app, 0.0))
    if last_for_app and (time.time() - last_for_app) < LONG_DWELL_REPEAT_GAP:
        return "", ""

    now = time.localtime()
    clock = _format_clock(now)
    duration = _format_hours_minutes(dwell)
    # One-sentence line that matches the spec example phrasing.
    line = (
        f"Sir, it's {clock} and you've been in {app} for {duration}. "
        f"Shall I queue a coffee timer, or are we pushing through?"
    )
    return line, dwell_key


def _try_late_hour_active() -> str:
    """If past LATE_HOUR_THRESHOLD_HOUR and user has spoken within the last
    LATE_HOUR_ACTIVE_WINDOW, suggest pacing."""
    now = time.localtime()
    if not (now.tm_hour >= LATE_HOUR_THRESHOLD_HOUR or now.tm_hour < LATE_HOUR_END_HOUR):
        return ""
    age = _last_speech_age_seconds()
    if age is None or age > LATE_HOUR_ACTIVE_WINDOW:
        return ""
    clock = _format_clock(now)
    return (
        f"It's {clock}, sir. We've been at this a while — "
        f"a brief stretch would not go amiss."
    )


# ─── main loop ───────────────────────────────────────────────────────────

def _should_skip_late_night() -> bool:
    """Between LATE_HOUR_THRESHOLD_HOUR and LATE_HOUR_END_HOUR, only fire when
    the user has spoken in the last 30 minutes — no whispering at an empty desk
    at 4am."""
    lt = time.localtime()
    if not (lt.tm_hour >= LATE_HOUR_THRESHOLD_HOUR or lt.tm_hour < LATE_HOUR_END_HOUR):
        return False
    age = _last_speech_age_seconds()
    if age is None:
        return True   # no activity record, assume idle → skip
    return age > LATE_HOUR_ACTIVE_WINDOW


def _scheduler_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            cfg = _read_config()
            if not cfg["enabled"]:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Always refresh dwell tracking — even when we won't fire, we want
            # a valid record for the eventual moment when gates open.
            focused = _focused_window_title()
            _update_dwell(focused)

            # Hard gates
            if _is_sleep_or_standby():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            if _is_in_call():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            if _user_at_desk() is False:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            if _should_skip_late_night():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Cooldown
            state = _load_state()
            cooldown_s = max(60, int(cfg["cooldown"]) * 60)
            last_fire = float(state.get("last_proactive_at", 0.0) or 0.0)
            if last_fire and (time.time() - last_fire) < cooldown_s:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Pick a trigger — pattern offer wins over dwell wins over late-hour
            line = ""
            trigger = ""
            dwell_key_for_app: str = ""

            offer = _try_pattern_offer()
            if offer:
                line = offer
                trigger = "pattern"
            if not line:
                dwell_line, dwell_app_key = _try_long_dwell(state)
                if dwell_line:
                    line = dwell_line
                    trigger = "long_dwell"
                    dwell_key_for_app = dwell_app_key
            if not line:
                late_line = _try_late_hour_active()
                if late_line:
                    line = late_line
                    trigger = "late_hour"

            if not line:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Probabilistic gate — even with a match we keep silent most ticks
            # so the engine feels like a butler, not a clock. Pattern offers
            # skip this gate: they self-throttle once-per-day already.
            if trigger != "pattern" and random.random() > FIRE_PROBABILITY:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Fire
            print(f"  [anticipate] firing ({trigger}): {line}")
            _enqueue_speech(line)

            # Persist
            state["last_proactive_at"] = time.time()
            state["last_trigger"]      = trigger
            state["last_line"]         = line
            if trigger == "long_dwell" and dwell_key_for_app:
                per_app = dict(state.get("last_dwell_remark_at") or {})
                per_app[dwell_key_for_app.split(":", 1)[1]] = time.time()
                state["last_dwell_remark_at"] = per_app
            _save_state(state)

        except Exception:
            logging.exception("  [anticipate] scheduler error")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        time.sleep(POLL_INTERVAL_SECONDS)


# ─── action: anticipation_status ─────────────────────────────────────────

def _format_status() -> str:
    cfg = _read_config()
    state = _load_state()
    last_fire = float(state.get("last_proactive_at", 0.0) or 0.0)
    cooldown_s = max(60, int(cfg["cooldown"]) * 60)
    parts: list[str] = []
    if not cfg["enabled"]:
        parts.append("engine disabled in config")
    if last_fire:
        age = int(time.time() - last_fire)
        ago = _format_hours_minutes(age) if age >= 60 else f"{age} seconds"
        parts.append(f"last fire {ago} ago")
        remaining = cooldown_s - age
        if remaining > 0:
            parts.append(f"{_format_hours_minutes(remaining)} until next eligible")
    else:
        parts.append("no fires yet this session")
    title, dwell = _current_dwell_seconds()
    if title and dwell > 60:
        parts.append(f"focused window: {_shorten_app_name(title)} for {_format_hours_minutes(dwell)}")
    if _is_in_call():
        parts.append("currently in a call (suppressed)")
    if _is_sleep_or_standby():
        parts.append("sleep/standby active (suppressed)")
    return "Anticipation engine — " + "; ".join(parts) + "."


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    def anticipation_status(_: str = "") -> str:
        try:
            return _format_status()
        except Exception as e:
            return f"anticipation status failed: {e}"

    actions["anticipation_status"] = anticipation_status

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [anticipate] ANTICIPATION_ENABLED is False — engine disabled")
        return

    # Guard against duplicate loops on skill reload: load_skills() re-execs
    # this module (fresh globals), so a module-level flag can't catch a prior
    # load's still-running thread — only an OS-thread name check survives.
    if any(th.name == "anticipation-scheduler" and th.is_alive()
           for th in threading.enumerate()):
        print("  [anticipate] scheduler already running — skipping duplicate "
              "(skill reload)")
    else:
        t = threading.Thread(target=_scheduler_loop, daemon=True,
                             name="anticipation-scheduler")
        t.start()
        print(
            f"  [anticipate] background loop running "
            f"(poll {int(POLL_INTERVAL_SECONDS)}s, cooldown {cfg['cooldown']}m, "
            f"p(fire)={FIRE_PROBABILITY:.2f})"
        )
