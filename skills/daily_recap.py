"""
Daily recap skill for JARVIS.

Fires once per day at a configurable local clock time (default 22:30) and
delivers a spoken JARVIS-style end-of-day SUMMARY of what actually happened,
synthesising every signal we already log:

  * pattern_learning JSONL  (data/usage_patterns.jsonl)
        — what actions JARVIS ran, and their args (artists, apps, URLs).
  * focused-window history  (today's session logs)
        — mined from `[action] focus_window: ...` and `[action] launch_app: ...`
          plus dwell remarks left by skills/anticipation_engine, giving a
          rough proxy for "time spent in X".
  * Teams events            (today's session logs)
        — counted via teams_screener / teams_nudge alert lines, with the
          name of any VIP caller surfaced.
  * Bambu print logs        (today's session logs + bambu_monitor state)
        — counts started + completed + failed prints today, and reports
          the in-flight print if one is still running.

Sample line the skill aims to produce:
  "Sir, today you spent 2 hours 40 minutes in Bambu Studio, completed one
   print, took 4 Teams calls including one from a colleague, and played 11
   tracks. Shall I queue the same morning briefing for
   tomorrow?"

Actions added:
  daily_recap   -- manually trigger the recap (e.g. "JARVIS, recap my day").
                   Returns the recap text (which is also enqueued for spoken
                   delivery).

Scheduler behaviour mirrors skills/evening_briefing.py:
  * Background thread polls every 60 seconds.
  * Fires at HH:MM with persistence in `daily_recap_state.json` so it
    doesn't re-fire on restart.
  * Catch-up window: if we boot >120 min past the slot, we mark today
    done rather than barge in at 4am.

Config knobs live in bobert_companion.py:
  DAILY_RECAP_ENABLED   (bool, default True)
  DAILY_RECAP_HOUR      (int 0-23, default 22)
  DAILY_RECAP_MINUTE    (int 0-59, default 30)
"""
from __future__ import annotations

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
from collections import Counter

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "daily_recap_state.json")
_LOGS_DIR     = os.path.join(_PROJECT_DIR, "logs")
_TODO_FILE    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")
_PATTERNS_JSONL = os.path.join(_PROJECT_DIR, "data", "usage_patterns.jsonl")

POLL_INTERVAL_SECONDS = 60.0
INITIAL_DELAY_SECONDS = 45     # let the rest of JARVIS finish booting
CATCHUP_WINDOW_MINUTES = 120   # past this, mark today done rather than fire mid-night

# Window-title fragments we want to credit when summarising focus time. Kept
# in sync (loosely) with anticipation_engine.PRODUCTIVITY_WINDOW_HINTS so the
# recap reports the same apps the engine already cares about.
_APP_HINTS_DISPLAY = [
    ("bambu studio",       "Bambu Studio"),
    ("orcaslicer",         "OrcaSlicer"),
    ("prusaslicer",        "PrusaSlicer"),
    ("autodesk fusion 360","Fusion 360"),
    ("fusion 360",         "Fusion 360"),
    ("solidworks",         "SolidWorks"),
    ("freecad",            "FreeCAD"),
    ("onshape",            "Onshape"),
    ("blender",            "Blender"),
    ("visual studio code", "VS Code"),
    ("vscode",             "VS Code"),
    ("intellij",           "IntelliJ"),
    ("pycharm",            "PyCharm"),
    ("photoshop",          "Photoshop"),
    ("illustrator",        "Illustrator"),
    ("premiere pro",       "Premiere Pro"),
    ("after effects",      "After Effects"),
    ("davinci resolve",    "DaVinci Resolve"),
    ("figma",              "Figma"),
    ("notion",             "Notion"),
    ("obsidian",           "Obsidian"),
    ("logic pro",          "Logic Pro"),
    ("ableton",            "Ableton"),
    ("chrome",             "Chrome"),
    ("edge",               "Edge"),
    ("firefox",            "Firefox"),
    ("teams",              "Microsoft Teams"),
    ("outlook",            "Outlook"),
    ("excel",              "Excel"),
    ("word",               "Word"),
    ("powerpoint",         "PowerPoint"),
    ("spotify",            "Spotify"),
    ("apple music",        "Apple Music"),
    ("youtube",            "YouTube"),
]

# Actions that count as "playing music" -- used both as a count and as a
# source for the dominant artist/title arg.
_MUSIC_ACTIONS = {
    "play_music", "apple_music", "spotify", "youtube_play",
    "resume_music", "play_streaming", "media_play",
}

# Actions so routine they shouldn't drive the "headline action" remark.
_BORING_ACTIONS = {
    "see_screen", "see_user", "which_monitor", "focus_window", "list_windows",
    "scroll", "type", "press", "hotkey", "click", "find_on_screen",
    "gaze_status", "gaze_stats", "face_track_status", "check_system",
    "audio_music_status",
}

_speech_lock = threading.Lock()
_state_lock  = threading.Lock()


# --- speech queue --------------------------------------------------------

def _enqueue_speech(message: str) -> None:
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. offline smoke test, import-time skill registration before
    bobert_companion finishes initialising). Matches the canonical pattern
    used by skills/credits_monitor.py, skills/bambu_monitor.py,
    skills/weather_briefing.py, etc., so every co-writer of
    pending_speech.json funnels through the same atomic helper and there's no
    per-skill race drift."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="recap")
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
            print(f"  [recap] speech-queue write failed ({e}); recap: {message}")


# --- config + persistent state -------------------------------------------

def _read_config() -> dict:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    return {
        "enabled": bool(getattr(bc, "DAILY_RECAP_ENABLED", True)) if bc else True,
        "hour":    int (getattr(bc, "DAILY_RECAP_HOUR",    22))   if bc else 22,
        "minute":  int (getattr(bc, "DAILY_RECAP_MINUTE",  30))   if bc else 30,
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
            print(f"  [recap] could not persist last-fired date: {e}")


# --- todays log paths ----------------------------------------------------

def _todays_log_paths() -> list:
    today_iso = datetime.date.today().isoformat()
    pattern = os.path.join(_LOGS_DIR, f"session_{today_iso}_*.log")
    return sorted(glob.glob(pattern))


# --- pattern_learning JSONL today ---------------------------------------

def _todays_pattern_events() -> list[dict]:
    """Return only today's events from data/usage_patterns.jsonl. The file
    may not exist yet (skill hasn't logged anything) -- that's fine."""
    if not os.path.exists(_PATTERNS_JSONL):
        return []
    today_iso = datetime.date.today().isoformat()
    out: list[dict] = []
    try:
        with open(_PATTERNS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("date") == today_iso:
                    out.append(e)
    except Exception:
        return []
    return out


# --- session log mining --------------------------------------------------

_ACTION_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\]\s+\[action\]\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:?\s*(.*?)\s*$"
)
_FOCUS_WINDOW_RE = re.compile(
    r"\[action\]\s+focus_window\s*:?\s*(.+?)\s*$", re.IGNORECASE
)
_LAUNCH_APP_RE = re.compile(
    r"\[action\]\s+launch_app\s*:?\s*(.+?)\s*$", re.IGNORECASE
)
_OPEN_URL_RE = re.compile(
    r"\[action\]\s+open_url\s*:?\s*(.+?)\s*$", re.IGNORECASE
)
_DWELL_REMARK_RE = re.compile(
    r"you've been in\s+([A-Za-z0-9 +./-]+?)\s+for\s+(\d+)\s+hour", re.IGNORECASE
)
_MUSIC_ARG_RE = re.compile(r"['\"]([^'\"]{2,80})['\"]")

# Teams alerts -- lines emitted by teams_screener.py / teams_nudge.py via
# pending_speech.json end up in the session log when the main loop reads
# the queue. We look for distinctive phrases that don't appear elsewhere.
_TEAMS_CALL_RE = re.compile(
    r"(?:Incoming call from|is calling you|on Teams|teams call|Teams meeting|"
    r"unread messages on Teams)\b.*?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?)?",
    re.IGNORECASE,
)
# Conservative VIP-name extraction: capitalised first name optionally
# followed by a capitalised last name, after a "from " keyword.
_FROM_NAME_RE = re.compile(
    r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)

# Bambu lifecycle phrases the printer-monitor emits.
_PRINT_STARTED_RE  = re.compile(r"Print started, sir", re.IGNORECASE)
_PRINT_FINISHED_RE = re.compile(r"Print complete, sir", re.IGNORECASE)
_PRINT_FAILED_RE   = re.compile(r"print has failed", re.IGNORECASE)


def _normalize_app_name(text: str) -> str:
    """Map an arbitrary window title / app arg to a clean app display name,
    or '' if nothing matches."""
    if not text:
        return ""
    lower = text.lower()
    for hint, display in sorted(_APP_HINTS_DISPLAY, key=lambda x: -len(x[0])):
        if hint in lower:
            return display
    return ""


def _scan_session_logs() -> dict:
    """Mine today's session logs for everything the recap wants. Returns:
        {
          "voice_count":      int,
          "action_counts":    Counter,
          "music_titles":     Counter,  # quoted titles passed to music actions
          "app_minutes":      Counter,  # estimated minutes per app (rough)
          "teams_alerts":     int,
          "teams_vips":       Counter,
          "print_started":    int,
          "print_finished":   int,
          "print_failed":     int,
        }
    """
    voice_count       = 0
    action_counts     = Counter()
    music_titles      = Counter()
    teams_alerts      = 0
    teams_vips: Counter = Counter()
    print_started     = 0
    print_finished    = 0
    print_failed      = 0

    # For app dwell estimation we track distinct minute-of-day buckets per
    # app -- a far cheaper proxy than full event-stream reconstruction.
    app_minute_buckets: dict[str, set[tuple[int, int, int]]] = {}

    for path in _todays_log_paths():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "  You:    " in line:
                        voice_count += 1

                    m = _ACTION_RE.search(line)
                    if m:
                        ts_str, action, payload = m.group(1), m.group(2), m.group(3) or ""
                        action_counts[action] += 1

                        if action in _MUSIC_ACTIONS:
                            qm = _MUSIC_ARG_RE.search(payload)
                            if qm:
                                title = qm.group(1).strip().lower()
                                if 2 <= len(title) <= 80:
                                    music_titles[title] += 1

                        # Bambu lifecycle picked up via action emissions too
                        # (some are emitted by bambu_monitor through speech;
                        # _PRINT_*_RE catches those via the raw line check
                        # below, but the explicit check_print action also
                        # counts as a print-related interaction we don't
                        # need to special-case here).

                        # Focus/app mining
                        app_name = ""
                        if action == "focus_window":
                            app_name = _normalize_app_name(payload)
                        elif action == "launch_app":
                            app_name = _normalize_app_name(payload)
                        elif action == "open_url":
                            app_name = _normalize_app_name(payload) or "Browser"
                        else:
                            # Some actions still mention the app in their payload
                            # (e.g. apple_music "playing X on Apple Music").
                            app_name = _normalize_app_name(payload)

                        if app_name:
                            try:
                                hh, mm, ss = ts_str.split(":")
                                bucket = (int(hh), int(mm), 0)
                                app_minute_buckets.setdefault(app_name, set()).add(bucket)
                            except (ValueError, TypeError):  # pragma: no cover - defensive: _ACTION_RE already guarantees ts_str is \d{2}:\d{2}:\d{2}
                                pass

                    # Anticipation-engine dwell remark: "you've been in <app>
                    # for N hour(s) and M minutes" -- gives us an explicit
                    # dwell signal we should trust over the minute-bucket
                    # estimate.
                    dm = _DWELL_REMARK_RE.search(line)
                    if dm:
                        app_name = _normalize_app_name(dm.group(1))
                        if app_name:
                            try:
                                hours = int(dm.group(2))
                                # Each minute of remarked dwell stands in for
                                # a "we know they were in this app this
                                # minute" data point. We synthesize that many
                                # bucket entries so the dwell line can
                                # outweigh sparse action sampling.
                                synth = app_minute_buckets.setdefault(app_name, set())
                                for k in range(hours * 60):
                                    synth.add(("synth", app_name, k))
                            except (ValueError, TypeError):  # pragma: no cover - defensive: _DWELL_REMARK_RE group(2) is always \d+
                                pass

                    # Teams alerts
                    low = line.lower()
                    is_teams_alert = False
                    if "incoming call from" in low or "is calling you" in low \
                            or "unread messages on teams" in low \
                            or "calling — i'd recommend taking it" in low \
                            or "teams call" in low:
                        is_teams_alert = True
                    if is_teams_alert:
                        teams_alerts += 1
                        fm = _FROM_NAME_RE.search(line)
                        if fm:
                            teams_vips[fm.group(1).strip()] += 1

                    # Bambu lifecycle (raw spoken phrases reach the log via
                    # the "speaking: ..." TTS trace lines).
                    if _PRINT_STARTED_RE.search(line):
                        print_started += 1
                    if _PRINT_FINISHED_RE.search(line):
                        print_finished += 1
                    if _PRINT_FAILED_RE.search(line):
                        print_failed += 1
        except Exception:
            continue

    # Collapse the dwell buckets into minutes per app. Buckets keyed by
    # (hour, minute, 0) collapse cleanly; "synth" buckets each represent
    # exactly one minute.
    app_minutes: Counter = Counter()
    for app, buckets in app_minute_buckets.items():
        app_minutes[app] = len(buckets)

    return {
        "voice_count":     voice_count,
        "action_counts":   action_counts,
        "music_titles":    music_titles,
        "app_minutes":     app_minutes,
        "teams_alerts":    teams_alerts,
        "teams_vips":      teams_vips,
        "print_started":   print_started,
        "print_finished":  print_finished,
        "print_failed":    print_failed,
    }


# --- pattern_learning supplement ----------------------------------------

def _supplement_with_pattern_jsonl(report: dict) -> None:
    """Fold today's data/usage_patterns.jsonl entries into the report.
    Cheaper, less noisy, and date-tagged -- so when it's available it
    refines the session-log estimates."""
    events = _todays_pattern_events()
    if not events:
        return
    for e in events:
        action = e.get("action") or ""
        arg = (e.get("arg") or "").strip()
        if not action:
            continue
        report["action_counts"][action] += 1
        if action in _MUSIC_ACTIONS and arg:
            # Normalise: lowercase, strip junk so "Michael Jackson Essentials"
            # and "michael jackson essentials" merge.
            key = arg.strip(" .,!?\"'").lower()
            if 2 <= len(key) <= 80:
                report["music_titles"][key] += 1


# --- bambu print cross-skill read ----------------------------------------

def _bambu_now() -> dict:
    """Snapshot of bambu_monitor._state if loaded, else {}."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return {}
    try:
        lock = getattr(mod, "_state_lock", None)
        raw_state = getattr(mod, "_state", None)
        if raw_state is None:
            return {}
        if lock is not None:
            with lock:
                return dict(raw_state)
        return dict(raw_state)
    except Exception:
        return {}


def _bambu_strip(filename: str) -> str:
    """Use bambu_monitor's own filename cleaner when available."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return filename or ""
    try:
        return getattr(mod, "_strip_filename")(filename or "")
    except Exception:
        return filename or ""


# --- helpers -------------------------------------------------------------

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}


def _number_word(n: int) -> str:
    if n in _NUMBER_WORDS:
        return _NUMBER_WORDS[n]
    return str(n)


def _format_duration_minutes(minutes: int) -> str:
    if minutes <= 0:
        return ""
    h, m = divmod(int(minutes), 60)
    if h == 0:
        return f"{m} minute{'s' if m != 1 else ''}"
    if m == 0:
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"


def _titlecase(s: str) -> str:
    small = {"and", "or", "of", "the", "in", "on", "for", "to", "a", "an"}
    parts = (s or "").split()
    out = []
    for i, w in enumerate(parts):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _count_tasks_completed_today() -> int:
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


# --- recap assembly ------------------------------------------------------

def _build_recap() -> str:
    report = _scan_session_logs()
    _supplement_with_pattern_jsonl(report)

    parts: list[str] = []

    # 1) Top app + minutes-in-app
    app_minutes: Counter = report["app_minutes"]
    if app_minutes:
        top_app, top_min = app_minutes.most_common(1)[0]
        # Only mention if it's a meaningful chunk -- below 10 minutes of
        # samples it's noise.
        if top_min >= 10:
            dur = _format_duration_minutes(top_min)
            parts.append(f"you spent {dur} in {top_app}")

    # 2) Bambu prints today
    finished = report["print_finished"]
    failed   = report["print_failed"]
    bambu    = _bambu_now()
    gcode_state = (bambu.get("gcode_state") or "").upper()
    in_flight_name = _bambu_strip(bambu.get("filename") or "")

    # Treat the live state as authoritative for the "active print" hint, but
    # the session-log counts for what's already happened today.
    if finished >= 1:
        if finished == 1:
            parts.append("completed one print")
        else:
            parts.append(f"completed {_number_word(finished)} prints")
    if failed >= 1:
        if failed == 1:
            parts.append("had one print fail on you")
        else:
            parts.append(f"had {_number_word(failed)} prints fail on you")
    if gcode_state in ("RUNNING", "PREPARE", "PAUSE"):
        if in_flight_name:
            parts.append(f"and the H2D is still printing '{in_flight_name}'")
        else:
            parts.append("and the H2D is still in flight")

    # 3) Teams events
    teams_alerts = report["teams_alerts"]
    teams_vips: Counter = report["teams_vips"]
    if teams_alerts >= 1:
        if teams_alerts == 1:
            teams_chunk = "took one Teams call"
        else:
            teams_chunk = f"took {_number_word(teams_alerts)} Teams calls"
        if teams_vips:
            top_vip, _ = teams_vips.most_common(1)[0]
            first_name = top_vip.split()[0] if top_vip else ""
            if first_name:
                teams_chunk += f" including one from {first_name}"
        parts.append(teams_chunk)

    # 4) Music plays
    music_total = sum(report["action_counts"].get(a, 0) for a in _MUSIC_ACTIONS)
    music_titles: Counter = report["music_titles"]
    if music_total >= 1:
        if music_titles:
            top_title, top_n = music_titles.most_common(1)[0]
            display = _titlecase(top_title)
            # Use the dominant title's count when it's clearly the headline
            # (else just say "N tracks"). "Michael Jackson Essentials" → 11
            # plays reads better than "11 tracks" generically.
            if top_n >= max(2, music_total // 2):
                if top_n == 1:  # pragma: no cover - unreachable: max(2,..)>=2 forces top_n>=2 in this branch
                    parts.append(f"played one {display} track")
                else:
                    parts.append(f"played {top_n} {display} tracks")
            else:
                parts.append(f"played {music_total} tracks")
        else:
            if music_total == 1:
                parts.append("played one track")
            else:
                parts.append(f"played {music_total} tracks")

    # 5) Tasks shipped from the queue
    tasks_done = _count_tasks_completed_today()
    if tasks_done >= 1:
        if tasks_done == 1:
            parts.append("cleared one task from the queue")
        else:
            parts.append(f"cleared {_number_word(tasks_done)} tasks from the queue")

    # 6) Voice-channel volume -- always honest about the lower bound, even
    # if everything else is empty.
    if not parts:
        vc = report["voice_count"]
        if vc == 0:
            parts.append("nothing of note made it onto the record today, sir")
        elif vc == 1:
            parts.append("just one voice interaction reached me today")
        else:
            parts.append(f"{vc} voice interactions reached me today")

    # Stitch into a single JARVIS-style sentence. The opening varies very
    # slightly depending on how much we have.
    if len(parts) == 1:
        body = f"Sir, today {parts[0]}."
    else:
        # The bambu "and the H2D is still printing X" segment is meant to
        # tail at the end of the prints chunk -- if it landed in the middle
        # of the list, join with comma+'and' rather than just commas.
        # Simple rule: last item gets ", and " unless it already starts
        # with "and ".
        head = parts[:-1]
        tail = parts[-1]
        if tail.startswith("and "):
            sentence = ", ".join(parts)
        else:
            sentence = ", ".join(head) + (", and " if head else "") + tail
        body = f"Sir, today you {sentence}."
        # The first chunk already starts with "you spent..." or similar --
        # patch double-"you" if so.
        body = body.replace("Sir, today you you ", "Sir, today you ")
        # If the first chunk started with "nothing"/"just one"/N voice etc.
        # (no leading "you ..."), the "today you" prefix needs to fall away.
        if parts[0].startswith(("nothing", "just one", "no ", "1 ", "2 ",
                                "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ")) \
                or re.match(r"^\d", parts[0]):  # pragma: no cover - defensive: no multi-chunk first part is digit/keyword-led (voice fallback is single-chunk)
            body = f"Sir, today {sentence}."

    # Closing nudge -- the spec's headline ask.
    closing = " Shall I queue the same morning briefing for tomorrow?"
    text = body + closing

    # Briefing intent tag so TTS uses the measured "briefing" preset rather
    # than the default neutral one. The dispatcher strips the tag before
    # speaking.
    return f"[intent:briefing] {text}"


# --- card popup (best effort) --------------------------------------------

def _show_card_safe() -> None:
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
    try:
        hud_card = importlib.import_module("hud_card")
        hud_card.show_card("recap")
    except Exception as e:
        # Card is purely decorative -- never block the spoken recap on it.
        print(f"  [recap] hud_card.show_card failed: {e}")


# --- scheduler thread ----------------------------------------------------

def _fire_recap(reason: str = "scheduled") -> str:
    text = _build_recap()
    print(f"  [recap] firing recap ({reason}): {text}")
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
                    f"  [recap] past catch-up window for "
                    f"{cfg['hour']:02d}:{cfg['minute']:02d}, skipping today"
                )
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            _fire_recap("scheduled")
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception:
            logging.exception("[recap] scheduler error")
            time.sleep(POLL_INTERVAL_SECONDS)


# --- action registration -------------------------------------------------

def register(actions):
    def daily_recap(_: str = "") -> str:
        try:
            text = _build_recap()
            _enqueue_speech(text)
            _show_card_safe()
            _save_last_fired_date(datetime.date.today().isoformat())
            return text
        except Exception as e:
            return f"daily recap failed: {e}"

    actions["daily_recap"] = daily_recap

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [recap] DAILY_RECAP_ENABLED is False -- scheduler disabled")
        return

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [recap] daily recap scheduled for "
        f"{cfg['hour']:02d}:{cfg['minute']:02d}"
    )


# --- offline smoke test --------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - manual offline smoke test entry point
    print("Running offline smoke test...")
    text = _build_recap()
    print(text)
