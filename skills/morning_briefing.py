"""
Morning briefing skill for JARVIS.

Actions:
  morning_briefing   — manually trigger the briefing. Returns the briefing
                       text (also spoken).

Auto-trigger:
  Driven by skills/morning_chain.py — a single controller polls
  bobert_companion._last_wake_date for the day's first wake event and picks
  ONE of {morning_arrival, morning_handoff, morning_briefing} to dispatch
  based on day-of-week / DEFAULT_MORNING_SKILL / time-of-day. When the chain
  picks briefing, it calls _fire_from_chain() here. Same-day suppression is
  persisted via .morning_briefing_last so a JARVIS restart doesn't re-fire.
  This skill no longer spawns its own wake-watcher thread.

  The briefing covers:
    • Day-of-week + date
    • Current weather (wttr.in, no API key)
    • Pending task count (jarvis_todo.md)
    • Optional dry remark if the last session ended very late

Style is authentic JARVIS — dry, two or three sentences, no preamble.
"""
import importlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_MEMORY_FILE  = os.path.join(_PROJECT_DIR, "bobert_memory.json")
_TODO_FILE    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")

# Hours of day (local) when the briefing should auto-fire
BRIEFING_START_HOUR = 6
BRIEFING_END_HOUR   = 11

# Seconds between the wake event and the briefing being queued. Long enough
# for the wake greeting to finish speaking before the briefing lines up
# behind it on pending_speech.json.
BRIEFING_DELAY_SECONDS = 8

# Background poll interval for the first-wake watcher. Matches the value used
# in morning_handoff's watcher so the two skills feel symmetric.
WAKE_WATCH_POLL_SECONDS = 5.0

# wttr.in URL — empty location = geolocate by IP. Format ?format=j1 gives JSON.
WTTR_URL    = "https://wttr.in/?format=j1"
WTTR_TIMEOUT = 6.0

# Hard timeout for the MS Graph outlook lookup so the briefing still fires
# even if Graph hangs (network drop, expired token mid-refresh, etc.).
OUTLOOK_TIMEOUT_SECONDS = 10.0

_speech_lock = threading.Lock()


def _show_card_safe() -> None:
    """Pop the transient briefing card. Imported lazily so the skill stays
    usable even if hud_card.py is missing or fails to import."""
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
    try:
        hud_card = importlib.import_module("hud_card")
        hud_card.show_card("morning")
    except Exception as e:
        print(f"  [morning] hud_card.show_card failed: {e}")


def _enqueue_speech(message: str) -> None:
    """Route a spoken briefing through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising). Matches the canonical pattern in
    skills/bambu_monitor.py so co-writers of pending_speech.json funnel
    through the same helper and can't race each other."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="morning")
            return
    except Exception:
        # Fall through to local atomic write — never let a broken parent
        # import silence the morning briefing.
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
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the briefing isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [morning] speech-queue write failed ({e}); briefing: {message}")


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _count_pending_tasks() -> int:
    if not os.path.exists(_TODO_FILE):
        return 0
    try:
        with open(_TODO_FILE, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip().startswith("- [ ]"))
    except Exception:
        return 0


def _fetch_weather() -> str:
    """Return a short weather phrase like '64 degrees and overcast' or ''.

    Temperature is spoken in Fahrenheit (sir's preference): briefing_sources
    stores Celsius, so it's converted here with the canonical c*9/5+32 formula.

    Routed through skills/briefing_sources.py so the morning briefing has the
    wttr → Open-Meteo → cached-last-known fallback chain instead of dropping
    silently when wttr.in is unreachable."""
    try:
        from . import briefing_sources  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import briefing_sources  # type: ignore
        except Exception as e:
            print(f"  [morning] briefing_sources unavailable: {e}")
            return ""

    data = briefing_sources.get_weather_data()
    if not data:
        return ""
    try:
        temp_c = int(data["temp_c"])
    except (KeyError, TypeError, ValueError):
        return ""
    # briefing_sources stores Celsius; sir wants Fahrenheit spoken. Convert with
    # the project's canonical store-Celsius/speak-Fahrenheit formula so this line
    # can never drift from weather_briefing's _current_conditions_line() or
    # morning_arrival's _section_weather_phrase() (both: int(round(c*9/5+32))).
    temp_f = int(round(temp_c * 9 / 5 + 32))
    desc = (data.get("desc") or "").strip().lower()
    suffix = ""
    if data.get("source") == "cache" and data.get("stale"):
        # Be honest if we're quoting an aging cached reading
        suffix = " (cached)"
    if not desc:
        return f"{temp_f} degrees outside{suffix}"
    return f"{temp_f} degrees and {desc} in your area{suffix}"


def _outlook_summary_blocking() -> str:
    """Body of the outlook summary — may block on MS Graph calls. Always
    invoke through ``_outlook_summary`` so the hard timeout is enforced."""
    try:
        from . import ms_graph                          # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import ms_graph                             # type: ignore
        except Exception:
            return ""

    parts: list[str] = []

    try:
        unread = ms_graph.get_unread_mail_count()
    except Exception:
        unread = None
    if isinstance(unread, int) and unread > 0:
        if unread == 1:
            parts.append("one unread email")
        else:
            parts.append(f"{unread} unread emails")

    try:
        meeting = ms_graph.get_first_meeting("today")
    except Exception:
        meeting = None
    if meeting:
        sdt = meeting.get("start")
        if hasattr(sdt, "hour"):
            hour = sdt.hour
            minute = sdt.minute
            disp_hour = hour % 12 or 12
            suffix = "AM" if hour < 12 else "PM"
            tstr = f"{disp_hour}:{minute:02d} {suffix}"
            subject = (meeting.get("subject") or "").strip()
            organizer = (meeting.get("organizer") or "").strip()
            who = ""
            if organizer and organizer.lower() not in (os.getenv("JARVIS_USER_NAME", "").lower(), "me"):
                first_name = organizer.split("<")[0].strip().split()[0] if organizer.split("<")[0].strip() else ""
                if first_name and "@" not in first_name:
                    who = f" with {first_name}"
            phrase = f"your first meeting is at {tstr}{who}"
            if subject:
                phrase = f"{phrase} — {subject}"
            parts.append(phrase)

    return "; ".join(parts)


def _outlook_summary() -> str:
    """Run :func:`_outlook_summary_blocking` with a hard wall-clock timeout
    so a hung MS Graph call can't stall the entire morning briefing. Returns
    ''. on timeout / executor failure."""
    # Why a fresh executor per call: this runs at most once per briefing
    # (typically once per morning), so the spin-up cost is negligible and we
    # avoid leaking a module-level pool across reloads.
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_outlook_summary_blocking)
            try:
                return future.result(timeout=OUTLOOK_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                print(
                    f"  [morning] outlook summary timed out after "
                    f"{OUTLOOK_TIMEOUT_SECONDS:.0f}s — skipping"
                )
                return ""
            except Exception as e:
                print(f"  [morning] outlook summary failed: {e}")
                return ""
    except Exception as e:
        print(f"  [morning] outlook executor unavailable: {e}")
        return ""


def _bed_remark() -> str:
    """If the previous session ended very late (after midnight, before 5 AM),
    add a dry remark. Falls back to '' if we can't tell.

    Note: bobert_memory.json doesn't store hour granularity, so the signal is
    the most recent log file's mtime as a proxy for "last activity" — the
    memory file itself is deliberately NOT required."""
    logs_dir = os.path.join(_PROJECT_DIR, "logs")
    if not os.path.isdir(logs_dir):
        return ""
    try:
        latest_mtime = max(
            (os.path.getmtime(os.path.join(logs_dir, f))
             for f in os.listdir(logs_dir) if f.endswith(".log")),
            default=0.0,
        )
    except Exception:
        latest_mtime = 0.0
    if latest_mtime == 0.0:
        return ""

    end = time.localtime(latest_mtime)
    # Only remark if last activity was between 00:00 and 04:59
    if 0 <= end.tm_hour < 5:
        hr = end.tm_hour or 12
        suffix = "AM" if end.tm_hour < 12 else "PM"
        return (
            f" — I should mention you were still at it until "
            f"{hr}:{end.tm_min:02d} {suffix}, so do try to pace yourself today"
        )
    return ""


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
            print(f"  [morning] news_briefing unavailable: {e}")
            return ""
    try:
        return news_briefing.get_news_text()
    except Exception as e:
        print(f"  [morning] news_briefing failed: {e}")
        return ""


def _fetch_umbrella_alert() -> str:
    """Forward-looking precipitation alert for today, drawn from the
    weather_briefing skill's Open-Meteo hourly forecast. Empty when the
    skill is unavailable or no notable precipitation is expected."""
    try:
        from . import weather_briefing  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import weather_briefing  # type: ignore
        except Exception as e:
            print(f"  [morning] weather_briefing unavailable: {e}")
            return ""
    try:
        return weather_briefing.get_umbrella_alert("today")
    except Exception as e:
        print(f"  [morning] weather_briefing umbrella alert failed: {e}")
        return ""


def _fetch_robot_volunteer() -> str:
    """One-line REPO Robot remark from skills/repo_robot.py when the project
    has actionable progress to volunteer (part arrived, blocker cleared).
    '' when the skill isn't loaded or there's nothing notable to flag."""
    try:
        from . import repo_robot  # type: ignore
    except Exception:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import repo_robot  # type: ignore
        except Exception:
            return ""
    try:
        return repo_robot.get_morning_volunteer_text() or ""
    except Exception as e:
        print(f"  [morning] repo_robot volunteer failed: {e}")
        return ""


def _build_briefing() -> str:
    now = time.localtime()
    day = time.strftime("%A", now)
    date_str = _ordinal(now.tm_mday)
    weather = _fetch_weather()
    pending = _count_pending_tasks()
    bed = _bed_remark()
    news = _fetch_news()
    umbrella = _fetch_umbrella_alert()

    parts = [f"Good morning, sir. It's {day} the {date_str}"]
    if weather:
        parts.append(f"{weather}")
    sentence_one = ", ".join(parts) + "."

    # The bed remark rides along regardless of task count — the docstring
    # promises it whenever the last session ran past midnight.
    if pending == 0:
        sentence_two = f"Your task queue is, for once, mercifully empty{bed}."
    elif pending == 1:
        sentence_two = f"You have one task queued{bed}."
    else:
        sentence_two = f"You have {pending} tasks queued{bed}."

    outlook = _outlook_summary()
    if outlook:
        head = f"{sentence_one} {sentence_two} From Outlook: {outlook}."
    else:
        head = f"{sentence_one} {sentence_two}"

    if umbrella:
        head = f"{head} {umbrella}"

    robot = _fetch_robot_volunteer()
    if robot:
        head = f"{head} {robot}"

    # If we've got news, prepend the briefing intent tag so the whole
    # message reads with the measured "briefing" TTS preset. The tag is
    # parsed off and stripped before TTS — it never gets spoken aloud.
    if news:
        return f"[intent:briefing] {head} {news}"
    return head


_BRIEFING_FLAG_FILE = os.path.join(_PROJECT_DIR, ".morning_briefing_last")


def _briefing_already_fired_today() -> bool:
    """True if the auto-briefing has already spoken today. Prevents the
    briefing from re-reading on every JARVIS restart when the upgrade
    pipeline kills and relaunches multiple times during the 6-11 window."""
    if not os.path.exists(_BRIEFING_FLAG_FILE):
        return False
    try:
        with open(_BRIEFING_FLAG_FILE, encoding="utf-8") as f:
            last = f.read().strip()
        return last == time.strftime("%Y-%m-%d")
    except Exception:
        return False


def _mark_briefing_fired_today() -> None:
    try:
        with open(_BRIEFING_FLAG_FILE, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d"))
    except Exception:
        pass


def _fire_briefing(reason: str, *, force: bool = False) -> None:
    """Queue the briefing for the main loop to speak. Marks the flag BEFORE
    building so a crash mid-build still prevents re-trigger on relaunch.

    When ``force`` is False (the auto-trigger path), the flag is re-checked
    immediately so a TOCTOU race between the watcher's pre-check and a
    parallel manual invocation can't double-speak the briefing. Manual
    invocations pass force=True to always run."""
    if not force and _briefing_already_fired_today():
        print(f"  [morning] suppressing ({reason}) — already fired today")
        return
    _mark_briefing_fired_today()
    try:
        text = _build_briefing()
    except Exception as e:
        print(f"  [morning] briefing build failed: {e}")
        return
    print(f"  [morning] queuing briefing ({reason}): {text}")
    _enqueue_speech(text)
    _show_card_safe()


def _fire_from_chain(reason: str = "morning_chain") -> None:
    """Auto-trigger entry called by skills/morning_chain.py once it has
    decided briefing is today's pick. Preserves the original watcher's
    TOCTOU-safe pattern verbatim: pre-check → delay → re-check → fire.
    Manual triggers ("morning briefing") still bypass via force=True."""
    if _briefing_already_fired_today():
        return
    time.sleep(BRIEFING_DELAY_SECONDS)
    if _briefing_already_fired_today():
        return
    _fire_briefing(reason)


def register(actions):
    def morning_briefing(_: str = "") -> str:
        # Manual invocation ALWAYS works — user explicitly asked for it.
        # Updates the flag so the auto-trigger won't fire afterward.
        try:
            text = _build_briefing()
            _show_card_safe()
            _mark_briefing_fired_today()
            return text
        except Exception as e:
            return f"morning briefing failed: {e}"

    actions["morning_briefing"] = morning_briefing

    # Auto-trigger is owned by skills/morning_chain.py — no per-skill watcher.
