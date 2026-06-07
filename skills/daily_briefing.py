"""
Daily briefing skill for JARVIS.

Fires once per day at a configurable local clock time (default 08:00) and
delivers a proactive JARVIS-style briefing covering:
  • Current local time
  • Outdoor weather (wttr.in, geolocated by IP)
  • First Teams / Outlook calendar meeting of the day (best effort via the
    Outlook COM interface — silently skipped if Outlook isn't installed)
  • Active Bambu print status (read from skills/bambu_monitor.py state)

Actions added:
  daily_briefing   — manually trigger the briefing. Returns the briefing text
                     (which is also enqueued for spoken delivery).

Scheduler behaviour:
  A background thread polls every 60 seconds. When local time crosses the
  configured hour/minute AND the briefing hasn't fired today, the skill
  waits up to DAILY_BRIEFING_WAIT_MINUTES for the user to appear in view
  (via skills/face_tracker.py's gaze state). If they appear, JARVIS speaks
  immediately; if not, the briefing is delivered anyway at the end of the
  wait window so it isn't silently skipped.

  Persistence: the last-fired ISO date is stored in
  `daily_briefing_state.json` next to bobert_companion.py so the skill
  survives restarts without re-firing twice in one day.

Config knobs live in bobert_companion.py:
  DAILY_BRIEFING_ENABLED      (bool, default True)
  DAILY_BRIEFING_HOUR         (int 0–23, default 8)
  DAILY_BRIEFING_MINUTE       (int 0–59, default 0)
  DAILY_BRIEFING_WAIT_MINUTES (int, default 30)
"""
import datetime
import importlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "daily_briefing_state.json")

WTTR_URL     = "https://wttr.in/?format=j1"
WTTR_TIMEOUT = 6.0

POLL_INTERVAL_SECONDS  = 60.0
PRESENCE_POLL_SECONDS  = 30.0
INITIAL_DELAY_SECONDS  = 45    # let the rest of JARVIS finish booting

# How far past the scheduled time we'll still fire (in case JARVIS was off when
# the schedule window opened). Keeps us from firing at, say, 6pm on a machine
# that just booted up.
CATCHUP_WINDOW_MINUTES = 120

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()


# ─── speech queue ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Route a proactive briefing through bobert_companion's public
    proactive_announce() API — the canonical helper for pending_speech.json —
    falling back to a direct atomic write if the parent module hasn't loaded
    yet (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising)."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="daily")
            return
    except Exception:
        # Fall through to local write — never let a broken parent import
        # silence the briefing.
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
            # Atomic write failed (read-only share, full disk, permission
            # denied). Fall back to console so the briefing isn't silently
            # lost.
            print(f"  [daily] speech-queue write failed ({e}); briefing: {message}")


# ─── config + persistent state ───────────────────────────────────────────

def _read_config() -> dict:
    """Pull live config from bobert_companion at call time."""
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    return {
        "enabled":  bool(getattr(bc, "DAILY_BRIEFING_ENABLED",      True)) if bc else True,
        "hour":     int (getattr(bc, "DAILY_BRIEFING_HOUR",         8))    if bc else 8,
        "minute":   int (getattr(bc, "DAILY_BRIEFING_MINUTE",       0))    if bc else 0,
        "wait_min": int (getattr(bc, "DAILY_BRIEFING_WAIT_MINUTES", 30))   if bc else 30,
    }


def _load_last_fired_date() -> str:
    if not os.path.exists(_STATE_FILE):
        return ""
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("last_fired_date", "") or ""
    except Exception:
        return ""


def _save_last_fired_date(iso_date: str) -> None:
    with _state_lock:
        try:
            fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"last_fired_date": iso_date}, f, indent=2)
                os.replace(tmp, _STATE_FILE)
            except Exception:
                try: os.unlink(tmp)
                except Exception: pass  # pragma: no cover - defensive cleanup-of-cleanup: temp unlink rarely fails
                raise
        except Exception as e:
            print(f"  [daily] could not persist last-fired date: {e}")


# ─── weather ─────────────────────────────────────────────────────────────

def _briefing_sources():
    """Lazy-load skills/briefing_sources.py so import errors don't blow up the
    whole scheduler thread on startup."""
    try:
        from . import briefing_sources  # type: ignore
        return briefing_sources  # pragma: no cover - reached only when loaded as a package (skills.daily_briefing); the live/test flat loader uses the import-by-name fallback below
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import briefing_sources  # type: ignore
            return briefing_sources
        except Exception as e:
            print(f"  [daily] briefing_sources unavailable: {e}")
            return None


def _fetch_weather() -> str:
    """Return a short weather phrase like '57 degrees and overcast' or ''.

    Temperature is spoken in Fahrenheit (sir's preference): briefing_sources
    stores Celsius, converted here with the canonical c*9/5+32 formula so the
    daily briefing agrees with morning_briefing / weather_briefing.

    Routed through skills/briefing_sources.py so the daily briefing has the
    wttr → Open-Meteo → cached-last-known fallback chain instead of dropping
    silently when wttr.in is unreachable."""
    bs = _briefing_sources()
    if bs is None:
        return ""
    data = bs.get_weather_data()
    if not data:
        return ""
    try:
        temp_c = int(data["temp_c"])
    except (KeyError, TypeError, ValueError):
        return ""
    temp_f = int(round(temp_c * 9 / 5 + 32))
    desc = (data.get("desc") or "").strip().lower()
    suffix = ""
    if data.get("source") == "cache" and data.get("stale"):
        suffix = " (cached)"
    if desc:
        return f"outside temperature is {temp_f} degrees and {desc}{suffix}"
    return f"outside temperature is {temp_f} degrees{suffix}"


# ─── Outlook calendar (best-effort) ──────────────────────────────────────

def _first_meeting_today() -> str:
    """Return a phrase like 'your first meeting today is at 9:30 AM with X'
    or '' if no source returned a meeting.

    Routed through skills/briefing_sources.py so the chain is now
    Outlook COM → Microsoft Graph → Google Calendar ICS instead of just
    Outlook (which fails outright when the user isn't signed in)."""
    bs = _briefing_sources()
    if bs is None:
        return ""
    data = bs.get_first_meeting_data("today")
    if not data:
        return ""

    start_dt = data.get("start")
    if not isinstance(start_dt, datetime.datetime):
        return ""
    hour = start_dt.hour
    minute = start_dt.minute
    suffix = "AM" if hour < 12 else "PM"
    disp_hour = hour % 12 or 12
    tstr = f"{disp_hour}:{minute:02d} {suffix}"

    organizer = (data.get("organizer") or "").strip()
    who = ""
    if organizer and organizer.lower() not in (os.getenv("JARVIS_USER_NAME", "").lower(), "me", ""):
        name = organizer.split("<")[0].strip()
        if name and "@" not in name:
            who = f" with {name}"

    subject = (data.get("subject") or "").strip()
    if subject:
        return f"your first meeting today is at {tstr}{who} — {subject}"
    return f"your first meeting today is at {tstr}{who}"


# ─── Bambu print status (cross-skill read) ───────────────────────────────

def _bambu_status() -> str:
    """Pull current print status from skills/bambu_monitor's module-level
    state. Returns a short phrase or ''."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return ""
    try:
        lock = getattr(mod, "_state_lock", None)
        raw_state = getattr(mod, "_state", None)
        if raw_state is None:
            return ""
        if lock is not None:
            with lock:
                state = dict(raw_state)
        else:
            state = dict(raw_state)
    except Exception:
        return ""

    if state.get("last_update", 0.0) == 0.0:
        return ""

    gcode_state = (state.get("gcode_state") or "").upper()
    fname = ""
    try:
        fname = getattr(mod, "_strip_filename")(state.get("filename") or "")
    except Exception:
        pass

    if gcode_state == "FINISH":
        # If the print just finished overnight, give the user a heads up.
        # Heuristic: if last_update was within the last 12h, mention it.
        age_s = time.time() - state.get("last_update", 0.0)
        when = ""
        if 0 < age_s < 12 * 3600:
            finished_at = time.localtime(state["last_update"])
            hr = finished_at.tm_hour
            mn = finished_at.tm_min
            suffix = "AM" if hr < 12 else "PM"
            disp_hour = hr % 12 or 12
            when = f" at {disp_hour}:{mn:02d} {suffix}"
        if fname:
            return f"your H2D finished printing '{fname}'{when}"
        return f"your H2D finished its overnight print{when}"

    if gcode_state == "FAILED":
        return "I'm afraid the H2D flagged a print failure overnight"

    if gcode_state in ("RUNNING", "PREPARE", "PAUSE"):
        layer = state.get("layer_num")
        total = state.get("total_layer")
        remaining = state.get("mc_remaining")
        parts = []
        if fname:
            parts.append(f"the H2D is mid-print on '{fname}'")
        else:
            parts.append("the H2D has a print in progress")
        if layer and total:
            parts.append(f"layer {layer} of {total}")
        if remaining:
            try:
                m = int(remaining)
                if m > 0:
                    if m < 60:
                        parts.append(f"about {m} minutes remaining")
                    else:
                        h, rm = divmod(m, 60)
                        if rm == 0:
                            parts.append(f"about {h} hour{'s' if h != 1 else ''} remaining")
                        else:
                            parts.append(
                                f"about {h} hour{'s' if h != 1 else ''} "
                                f"and {rm} minutes remaining"
                            )
            except (TypeError, ValueError):
                pass
        return ", ".join(parts)

    return ""


# ─── face tracker presence (cross-skill read) ────────────────────────────

def _user_at_desk() -> bool | None:
    """Return True if face_tracker has recently seen the user, False if it
    has seen 'away', or None if face_tracker isn't loaded / has no data."""
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


# ─── briefing assembly ───────────────────────────────────────────────────

def _format_time_phrase(now: time.struct_time) -> str:
    hour = now.tm_hour
    minute = now.tm_min
    suffix = "AM" if hour < 12 else "PM"
    disp_hour = hour % 12 or 12
    return f"{disp_hour}:{minute:02d} {suffix}"


def _build_briefing() -> str:
    now = time.localtime()
    time_phrase = _format_time_phrase(now)

    weather = _fetch_weather()
    meeting = _first_meeting_today()
    bambu   = _bambu_status()

    pieces = [f"Good morning, sir. It is currently {time_phrase}"]
    extras = [p for p in (weather, meeting, bambu) if p]

    if not extras:
        pieces.append("and there is nothing remarkable to report")
    else:
        pieces.append(", ".join(extras))

    sentence = ", ".join(pieces) + "."
    return sentence


# ─── scheduler thread ────────────────────────────────────────────────────

def _wait_for_presence(max_wait_s: float) -> bool:
    """Poll face_tracker every PRESENCE_POLL_SECONDS until the user is in
    view or max_wait_s elapses. Returns True if presence was detected."""
    deadline = time.time() + max_wait_s
    # If face_tracker isn't loaded at all, don't wait — just fire.
    if _user_at_desk() is None:
        return False
    while time.time() < deadline:
        present = _user_at_desk()
        if present:
            return True
        time.sleep(PRESENCE_POLL_SECONDS)
    return False


def _fire_briefing(reason: str = "scheduled") -> str:
    text = _build_briefing()
    print(f"  [daily] firing briefing ({reason}): {text}")
    _enqueue_speech(text)
    _save_last_fired_date(datetime.date.today().isoformat())
    return text


def _scheduler_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            cfg = _read_config()
            if not cfg["enabled"]:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            now = datetime.datetime.now()
            today_iso = now.date().isoformat()
            last_fired = _load_last_fired_date()

            if last_fired == today_iso:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            scheduled = now.replace(
                hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0,
            )
            if now < scheduled:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Don't fire if we're way past the scheduled time (e.g. machine
            # was off all day and just booted at 6pm).
            if (now - scheduled).total_seconds() > CATCHUP_WINDOW_MINUTES * 60:
                # Mark today as "done" anyway so we don't keep checking.
                _save_last_fired_date(today_iso)
                print(
                    f"  [daily] past catch-up window for "
                    f"{cfg['hour']:02d}:{cfg['minute']:02d}, skipping today"
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Within fire window. Wait for user presence (best effort).
            wait_seconds = max(0, cfg["wait_min"]) * 60
            present = _wait_for_presence(wait_seconds) if wait_seconds > 0 else False
            _fire_briefing("user-present" if present else "timed-out")

        except Exception:
            logging.exception("  [daily] scheduler error")
        time.sleep(POLL_INTERVAL_SECONDS)


# ─── action registration ─────────────────────────────────────────────────

def register(actions):
    def daily_briefing(_: str = "") -> str:
        try:
            text = _build_briefing()
            _enqueue_speech(text)
            _save_last_fired_date(datetime.date.today().isoformat())
            return text
        except Exception as e:
            return f"daily briefing failed: {e}"

    actions["daily_briefing"] = daily_briefing

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [daily] DAILY_BRIEFING_ENABLED is False — scheduler disabled")
        return

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [daily] briefing scheduled for "
        f"{cfg['hour']:02d}:{cfg['minute']:02d} "
        f"(wait up to {cfg['wait_min']} min for user presence)"
    )
