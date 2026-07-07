"""
Evening briefing skill for JARVIS.

Fires once per day at a configurable local clock time (default 22:00) and
delivers a proactive JARVIS-style end-of-day summary covering:
  * Number of voice interactions logged today (across all session_*.log files)
  * Tasks completed from jarvis_todo.md today (counted from the markdown)
  * Active Bambu print status (read from skills/bambu_monitor.py state)
  * Tomorrow's weather forecast (wttr.in's day-1 forecast block)
  * Tomorrow's first calendar appointment (Outlook MAPI, best effort)
  * One dry observation drawn from today's session logs (e.g. repeated
    actions or repeated "play X" requests — "You said 'play Michael
    Jackson' four times today, sir. A pattern emerges.")

Actions added:
  evening_briefing  -- manually trigger the briefing. Returns the briefing
                       text (which is also enqueued for spoken delivery).

Scheduler behaviour mirrors skills/daily_briefing.py:
  * Background thread polls every 60 seconds.
  * When local time crosses HH:MM AND the briefing hasn't fired today, it
    waits up to EVENING_BRIEFING_WAIT_MINUTES for the user to appear in
    view (via skills/face_tracker.py's gaze state). If they appear, JARVIS
    speaks immediately; if not, the briefing fires anyway so it isn't
    silently skipped.
  * Persistence: the last-fired ISO date is stored in
    `evening_briefing_state.json` next to bobert_companion.py so the skill
    survives restarts without re-firing twice in one evening.

Config knobs live in bobert_companion.py:
  EVENING_BRIEFING_ENABLED       (bool, default True)
  EVENING_BRIEFING_HOUR          (int 0-23, default 22)
  EVENING_BRIEFING_MINUTE        (int 0-59, default 0)
  EVENING_BRIEFING_WAIT_MINUTES  (int, default 30)
"""
import datetime
import glob
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
from collections import Counter

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "evening_briefing_state.json")
_LOGS_DIR     = os.path.join(_PROJECT_DIR, "logs")
_TODO_FILE    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")

WTTR_URL     = "https://wttr.in/?format=j1"
WTTR_TIMEOUT = 6.0
# One quick retry on wttr before falling through to Open-Meteo — wttr.in is
# prone to transient 5xx / timeouts that clear on an immediate re-try.
WTTR_RETRIES = 1

# Open-Meteo daily-forecast fallback so a wttr outage doesn't silently drop
# tomorrow's weather from the evening briefing. Location is resolved via
# briefing_sources._resolve_location() (shared ipapi cache + config-pinned
# lat/lon), the same plumbing weather_briefing reuses.
OPEN_METEO_URL     = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT = 6.0
# WMO weather codes → short descriptions. Inlined (as elsewhere in the
# codebase) rather than reaching into briefing_sources' private table.
_WMO_DESCRIPTIONS = {
    0:  "clear",         1:  "mainly clear",   2:  "partly cloudy",
    3:  "overcast",      45: "foggy",          48: "freezing fog",
    51: "light drizzle", 53: "drizzle",        55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain",    63: "rain",           65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow",    73: "snow",           75: "heavy snow",
    77: "snow grains",
    80: "rain showers",  81: "rain showers",   82: "heavy rain showers",
    85: "snow showers",  86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}

POLL_INTERVAL_SECONDS = 60.0
PRESENCE_POLL_SECONDS = 30.0
INITIAL_DELAY_SECONDS = 45    # let the rest of JARVIS finish booting

# How far past the scheduled time we'll still fire (e.g. machine just booted
# at 23:30 after the 22:00 slot opened). Beyond this we mark today done and
# move on rather than dropping a midnight briefing into a sleep cycle.
CATCHUP_WINDOW_MINUTES = 120

# Only count an action as "common enough to remark on" if it shows up at
# least this many times in today's logs.
DRY_OBS_MIN_COUNT = 3

# Actions so routine that mentioning them would just be tedious noise.
_BORING_ACTIONS = {
    "see_screen", "see_user", "which_monitor", "focus_window",
    "list_windows", "scroll", "type", "press", "hotkey", "click",
    "gaze_status", "gaze_stats", "face_track_status", "check_system",
    "audio_music_status",
}

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()


def _show_card_safe() -> None:
    """Pop the transient briefing card. Imported lazily so the skill keeps
    working if hud_card.py is missing or fails to import."""
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
    try:
        hud_card = importlib.import_module("hud_card")
        hud_card.show_card("evening")
    except Exception as e:
        print(f"  [evening] hud_card.show_card failed: {e}")


# --- speech queue --------------------------------------------------------

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
            announcer(message, source="evening")
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
            print(f"  [evening] speech-queue write failed ({e}); briefing: {message}")


# --- config + persistent state -------------------------------------------

def _read_config() -> dict:
    """Pull live config from bobert_companion at call time."""
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    return {
        "enabled":  bool(getattr(bc, "EVENING_BRIEFING_ENABLED",      True)) if bc else True,
        "hour":     int (getattr(bc, "EVENING_BRIEFING_HOUR",         22))   if bc else 22,
        "minute":   int (getattr(bc, "EVENING_BRIEFING_MINUTE",       0))    if bc else 0,
        "wait_min": int (getattr(bc, "EVENING_BRIEFING_WAIT_MINUTES", 30))   if bc else 30,
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
            _atomic_write_json(_STATE_FILE, {"last_fired_date": iso_date})
        except Exception as e:
            print(f"  [evening] could not persist last-fired date: {e}")


# --- weather (tomorrow) --------------------------------------------------

def _phrase_tomorrow(max_f: int, min_f: int, desc: str) -> str:
    """Compose the spec's 'tomorrow looks like a high of X, low of Y, and
    DESC' phrasing shared by the wttr and Open-Meteo paths. Temperatures are
    FAHRENHEIT — the project-wide spoken-weather convention (the morning
    briefing says "96 degrees"); the evening path used to read Celsius, so
    "a high of 18" was spoken on an 18°C / 64°F day (2026-07-06 audit)."""
    desc = (desc or "").strip().lower()
    if desc:
        return f"tomorrow looks like a high of {max_f}, low of {min_f}, and {desc}"
    return f"tomorrow looks like a high of {max_f} and a low of {min_f}"


def _tomorrow_weather_from_wttr() -> str:
    """wttr.in day-1 forecast → phrase, with a short retry. '' on failure."""
    data = None
    last_err = None
    for attempt in range(WTTR_RETRIES + 1):
        try:
            req = urllib.request.Request(WTTR_URL, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=WTTR_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            last_err = e
    if data is None:
        print(f"  [evening] wttr weather fetch failed: {last_err}")
        return ""
    try:
        # weather[0] = today, weather[1] = tomorrow
        forecast = data["weather"][1]
        # Fahrenheit — the spoken-weather convention (was reading maxtempC).
        max_f = int(float(forecast.get("maxtempF", "0")))
        min_f = int(float(forecast.get("mintempF", "0")))
        # Hourly buckets: pick the noon entry for a representative condition
        hourly = forecast.get("hourly") or []
        desc = ""
        noon = next((h for h in hourly if str(h.get("time", "")) in ("1200", "1100", "1300")), None)
        if noon is None and hourly:
            noon = hourly[len(hourly) // 2]
        if noon:
            try:
                desc = (noon.get("weatherDesc", [{}])[0].get("value", "") or "").strip().lower()
            except (KeyError, IndexError, TypeError, AttributeError):
                # AttributeError guards a string-shaped weatherDesc (e.g.
                # "Sunny"): [0] then yields 'S', whose .get() would raise.
                desc = ""
        return _phrase_tomorrow(max_f, min_f, desc)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        return ""


def _tomorrow_weather_from_open_meteo() -> str:
    """Open-Meteo daily forecast for tomorrow → phrase. '' on failure.
    Location is resolved through briefing_sources._resolve_location() so we
    share the morning briefing's ipapi cache + config-pinned lat/lon."""
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
    try:
        from skills import briefing_sources  # type: ignore
    except Exception:
        try:
            import briefing_sources  # type: ignore
        except Exception as e:
            print(f"  [evening] open-meteo fallback unavailable: {e}")
            return ""
    try:
        loc = briefing_sources._resolve_location()
    except Exception as e:
        print(f"  [evening] open-meteo location resolution failed: {e}")
        return ""
    if not loc:
        return ""
    lat, lon = loc
    import urllib.parse
    qs = urllib.parse.urlencode({
        "latitude":  f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily":     "temperature_2m_max,temperature_2m_min,weather_code",
        "temperature_unit": "fahrenheit",   # spoken-weather convention
        "timezone":  "auto",
        "forecast_days": "2",
    })
    url = f"{OPEN_METEO_URL}?{qs}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jarvis/1.0"})
        with urllib.request.urlopen(req, timeout=OPEN_METEO_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [evening] open-meteo weather fetch failed: {e}")
        return ""
    try:
        daily = data.get("daily") or {}
        # index 0 = today, index 1 = tomorrow (values now Fahrenheit via the
        # temperature_unit=fahrenheit query param above).
        max_f = int(round(float(daily["temperature_2m_max"][1])))
        min_f = int(round(float(daily["temperature_2m_min"][1])))
        code = int(daily.get("weather_code", [None, -1])[1])
        desc = _WMO_DESCRIPTIONS.get(code, "")
        return _phrase_tomorrow(max_f, min_f, desc)
    except (KeyError, IndexError, ValueError, TypeError):
        return ""


def _fetch_tomorrow_weather() -> str:
    """Return a short phrase like 'tomorrow looks like a high of 18, low of
    9, and partly cloudy' or '' if every weather source is unreachable.

    Resilience: wttr.in first (with a short retry), then an Open-Meteo daily
    forecast fallback so a wttr outage doesn't silently drop tomorrow's
    weather from the briefing."""
    phrase = _tomorrow_weather_from_wttr()
    if phrase:
        return phrase
    return _tomorrow_weather_from_open_meteo()


# --- Outlook calendar (tomorrow's first meeting, best-effort) ------------

def _first_meeting_tomorrow() -> str:
    """Return 'your first meeting tomorrow is at 9:30 AM with Sam -- subj'
    or '' if Outlook unavailable / no meetings."""
    try:
        import pythoncom                  # type: ignore
        import win32com.client            # type: ignore
    except Exception:
        return ""

    try:
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        outlook = win32com.client.Dispatch("Outlook.Application")
        namespace = outlook.GetNamespace("MAPI")
        calendar = namespace.GetDefaultFolder(9)   # 9 == olFolderCalendar
        items = calendar.Items
        items.IncludeRecurrences = True
        items.Sort("[Start]")

        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        start_of_day = datetime.datetime.combine(tomorrow, datetime.time(0, 0))
        end_of_day = datetime.datetime.combine(tomorrow, datetime.time(23, 59, 59))
        fmt = "%m/%d/%Y %I:%M %p"
        restriction = (
            f"[Start] >= '{start_of_day.strftime(fmt)}' AND "
            f"[Start] <= '{end_of_day.strftime(fmt)}'"
        )
        try:
            tomorrows = items.Restrict(restriction)
        except Exception:
            tomorrows = items

        for appt in tomorrows:
            try:
                start = appt.Start
                if hasattr(start, "Format"):
                    start_dt = datetime.datetime(
                        start.year, start.month, start.day,
                        start.hour, start.minute,
                    )
                else:
                    start_dt = start
                if start_dt < start_of_day or start_dt > end_of_day:
                    continue
                subject = (getattr(appt, "Subject", "") or "").strip()
                organizer = (getattr(appt, "Organizer", "") or "").strip()

                hour = start_dt.hour
                minute = start_dt.minute
                suffix = "AM" if hour < 12 else "PM"
                disp_hour = hour % 12 or 12
                tstr = f"{disp_hour}:{minute:02d} {suffix}"

                who = ""
                if organizer and organizer.lower() not in (os.getenv("JARVIS_USER_NAME", "").lower(), "me", ""):
                    name = organizer.split("<")[0].strip()
                    if name and "@" not in name:
                        who = f" with {name}"

                if subject:
                    return f"your first meeting tomorrow is at {tstr}{who} -- {subject}"
                return f"your first meeting tomorrow is at {tstr}{who}"
            except Exception:
                continue
        return ""
    except Exception as e:
        print(f"  [evening] outlook query failed: {e}")
        return ""
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


# --- Bambu print status (cross-skill read) -------------------------------

def _bambu_status() -> str:
    """Pull current print status from skills/bambu_monitor's state. Returns
    a short phrase suitable for an evening report or ''."""
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
        # Did it finish in the last 12h? If so, worth mentioning at 22:00.
        age_s = time.time() - state.get("last_update", 0.0)
        if 0 < age_s < 12 * 3600:
            if fname:
                return f"the H2D finished '{fname}' earlier today"
            return "the H2D finished its print earlier today"
        return ""

    if gcode_state == "FAILED":
        return "I'm afraid the H2D flagged a print failure earlier"

    if gcode_state in ("RUNNING", "PREPARE", "PAUSE"):
        layer = state.get("layer_num")
        total = state.get("total_layer")
        remaining = state.get("mc_remaining")
        parts = []
        if fname:
            parts.append(f"the H2D is still printing '{fname}'")
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


# --- face tracker presence (cross-skill read) ----------------------------

def _user_at_desk():
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


# --- session log scraping ------------------------------------------------

def _todays_log_paths() -> list:
    """All session_<TODAY>_*.log files. Date prefix matches the filename
    convention bobert_companion uses for new session logs."""
    today_iso = datetime.date.today().isoformat()
    pattern = os.path.join(_LOGS_DIR, f"session_{today_iso}_*.log")
    return sorted(glob.glob(pattern))


def _count_voice_interactions_today() -> int:
    """Count user utterance lines across today's session logs.

    Each user utterance is logged via `print(f"  You:    {text}")` so a
    simple substring scan suffices.
    """
    needle = "  You:    "
    count = 0
    for path in _todays_log_paths():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if needle in line:
                        count += 1
        except Exception:
            continue
    return count


def _count_tasks_completed_today() -> int:
    """Count jarvis_todo.md lines that begin with '- [x]' AND contain today's
    ISO date somewhere on the line."""
    if not os.path.exists(_TODO_FILE):
        return 0
    today_iso = datetime.date.today().isoformat()
    count = 0
    try:
        with open(_TODO_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.lstrip()
                if s.startswith("- [x]") and today_iso in line:
                    count += 1
    except Exception:
        return 0
    return count


_ACTION_RE  = re.compile(r"\[action\]\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:")
_YOU_RE     = re.compile(r"  You:    (.+?)\s*$")
_PLAY_RE    = re.compile(r"\b(?:play|put on|queue|throw on)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)
# Whitelisted "small words" we don't want as the head of a "you kept asking
# for X" remark.
_PLAY_STOPWORDS = {
    "music", "something", "a song", "the song", "a track", "the track",
    "something good", "anything", "more", "again", "it", "that",
}


def _scan_today_for_patterns():
    """Return (action_counter, play_phrase_counter, you_count) drawn from
    today's session logs."""
    actions = Counter()
    plays = Counter()
    you_count = 0
    for path in _todays_log_paths():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = _ACTION_RE.search(line)
                    if m:
                        actions[m.group(1)] += 1
                    ym = _YOU_RE.search(line)
                    if ym:
                        you_count += 1
                        utterance = ym.group(1)
                        pm = _PLAY_RE.search(utterance)
                        if pm:
                            target = pm.group(1).strip().lower()
                            # Trim trailing fillers
                            target = re.sub(r"\s+(please|now|again|sir)\.?$", "", target).strip()
                            target = target.strip(" .,!?\"'")
                            if target and target not in _PLAY_STOPWORDS and len(target) <= 60:
                                plays[target] += 1
        except Exception:
            continue
    return actions, plays, you_count


def _humanize_count(n: int) -> str:
    """3 -> 'three times', 4 -> 'four times', else 'N times'."""
    words = {2: "twice", 3: "three times", 4: "four times", 5: "five times",
             6: "six times", 7: "seven times", 8: "eight times", 9: "nine times",
             10: "ten times"}
    if n in words:
        return words[n]
    return f"{n} times"


def _dry_observation():
    """Return a one-line JARVIS-style dry remark drawn from today's logs,
    or '' if nothing stood out."""
    actions, plays, _ = _scan_today_for_patterns()

    # Prefer a "you said play X N times" remark -- most personal-feeling.
    if plays:
        target, n = plays.most_common(1)[0]
        if n >= DRY_OBS_MIN_COUNT:
            return (
                f"you said 'play {target}' {_humanize_count(n)} today, sir. "
                "A pattern emerges."
            )

    # Otherwise, surface a repeated non-routine action.
    if actions:
        # Filter out routine/boring actions before picking the top.
        filtered = Counter({k: v for k, v in actions.items()
                            if k not in _BORING_ACTIONS})
        if filtered:
            name, n = filtered.most_common(1)[0]
            if n >= DRY_OBS_MIN_COUNT:
                spoken = name.replace("_", " ")
                return (
                    f"I ran '{spoken}' {_humanize_count(n)} today -- "
                    "well above the usual rate, sir."
                )

    return ""


# --- briefing assembly ---------------------------------------------------

def _fetch_news() -> str:
    """Pull the news_briefing module's headline paragraph, or '' if it's
    unavailable / disabled / every feed failed."""
    try:
        from . import news_briefing  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import news_briefing  # type: ignore
        except Exception as e:
            print(f"  [evening] news_briefing unavailable: {e}")
            return ""
    try:
        return news_briefing.get_news_text()
    except Exception as e:
        print(f"  [evening] news_briefing failed: {e}")
        return ""


def _fetch_tomorrow_umbrella() -> str:
    """Forward-looking precipitation alert for tomorrow, drawn from the
    weather_briefing skill's Open-Meteo hourly forecast. Empty when the
    skill is unavailable or no notable precipitation is expected."""
    try:
        from . import weather_briefing  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import weather_briefing  # type: ignore
        except Exception as e:
            print(f"  [evening] weather_briefing unavailable: {e}")
            return ""
    try:
        return weather_briefing.get_umbrella_alert("tomorrow")
    except Exception as e:
        print(f"  [evening] weather_briefing umbrella alert failed: {e}")
        return ""


def _build_briefing() -> str:
    interactions = _count_voice_interactions_today()
    completed    = _count_tasks_completed_today()
    bambu        = _bambu_status()
    weather      = _fetch_tomorrow_weather()
    meeting      = _first_meeting_tomorrow()
    observation  = _dry_observation()
    news         = _fetch_news()
    umbrella     = _fetch_tomorrow_umbrella()

    # Opening line varies slightly with how busy the day was
    if interactions == 0:
        opener = "Good evening, sir. A quiet day on the voice channel"
    elif interactions == 1:
        opener = "Good evening, sir. One voice interaction logged today"
    else:
        opener = f"Good evening, sir. {interactions} voice interactions logged today"

    if completed > 0:
        plural = "task" if completed == 1 else "tasks"
        opener += f" and {completed} {plural} cleared from the queue"
    opener += "."

    # "Right now" segment
    now_bits = []
    if bambu:
        now_bits.append(bambu)
    now_line = ""
    if now_bits:
        now_line = "Currently, " + ", ".join(now_bits) + "."

    # Tomorrow segment
    tomorrow_bits = [b for b in (weather, meeting) if b]
    tomorrow_line = ""
    if tomorrow_bits:
        tomorrow_line = "For tomorrow, " + ", and ".join(tomorrow_bits) + "."

    pieces = [opener]
    if now_line:
        pieces.append(now_line)
    if tomorrow_line:
        pieces.append(tomorrow_line)
    if umbrella:
        pieces.append(umbrella)
    if observation:
        # Capitalise first letter for sentence form
        obs = observation[0].upper() + observation[1:] if observation else ""
        pieces.append(obs)
    if news:
        pieces.append(news)

    body = " ".join(pieces)
    # Prepend the briefing intent tag when news is included so the whole
    # report reads with the measured "briefing" TTS preset. The tag is
    # stripped by _parse_intent_tag before TTS — never spoken aloud.
    if news:
        return f"[intent:briefing] {body}"
    return body


# --- scheduler thread ----------------------------------------------------

def _wait_for_presence(max_wait_s: float) -> bool:
    """Poll face_tracker every PRESENCE_POLL_SECONDS until the user is in
    view or max_wait_s elapses. Returns True if presence was detected."""
    deadline = time.time() + max_wait_s
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
    print(f"  [evening] firing briefing ({reason}): {text}")
    _enqueue_speech(text)
    _show_card_safe()
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

            if (now - scheduled).total_seconds() > CATCHUP_WINDOW_MINUTES * 60:
                _save_last_fired_date(today_iso)
                print(
                    f"  [evening] past catch-up window for "
                    f"{cfg['hour']:02d}:{cfg['minute']:02d}, skipping today"
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            wait_seconds = max(0, cfg["wait_min"]) * 60
            present = _wait_for_presence(wait_seconds) if wait_seconds > 0 else False
            # Re-check the same-day flag after the (potentially long) presence
            # wait — a manual evening_briefing invocation or a second instance
            # may have fired during the window (TOCTOU, matches
            # morning_briefing's pre-check -> delay -> re-check pattern).
            if _load_last_fired_date() == today_iso:
                print("  [evening] suppressing — briefing already fired during presence wait")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            _fire_briefing("user-present" if present else "timed-out")

        except Exception:
            logging.exception("  [evening] scheduler error")
        time.sleep(POLL_INTERVAL_SECONDS)


# --- action registration -------------------------------------------------

def register(actions):
    def evening_briefing(_: str = "") -> str:
        try:
            text = _build_briefing()
            _enqueue_speech(text)
            _show_card_safe()
            _save_last_fired_date(datetime.date.today().isoformat())
            return text
        except Exception as e:
            return f"evening briefing failed: {e}"

    actions["evening_briefing"] = evening_briefing

    # The manual evening_briefing action runs an Outlook COM calendar query
    # plus three serial network fetches (weather/news/umbrella) with no
    # aggregate timeout — routinely 15-20 s. Register it as long-running so
    # the dispatcher's mid-task status timer speaks one dry "working on it,
    # sir" line at the 8 s mark instead of leaving dead air. Same pattern
    # skills/dossier.py uses; the "_generic" bucket is the right fit here.
    try:
        import bobert_companion as _bc  # type: ignore
        _long_running = getattr(_bc, "LONG_RUNNING_ACTIONS", None)
        if isinstance(_long_running, set):
            _long_running.add("evening_briefing")
        _bucket_map = getattr(_bc, "_MID_TASK_STATUS_BUCKET", None)
        if isinstance(_bucket_map, dict):
            _bucket_map["evening_briefing"] = "_generic"
    except Exception as e:
        print(f"  [evening] couldn't register mid-task status bridge: {e}")

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [evening] EVENING_BRIEFING_ENABLED is False -- scheduler disabled")
        return

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [evening] briefing scheduled for "
        f"{cfg['hour']:02d}:{cfg['minute']:02d} "
        f"(wait up to {cfg['wait_min']} min for user presence)"
    )
