"""
weather_briefing skill — forward-looking weather intelligence for JARVIS.

The existing morning/evening briefings only carry a "current conditions"
weather phrase (via skills/briefing_sources.py → wttr.in / Open-Meteo
current endpoint). This skill adds an hourly forecast layer on top, so
JARVIS can:

  * Tell the user to grab an umbrella before they leave the house
    ("I'd suggest the umbrella today, sir — 80% chance of rain at 3 PM").
  * Proactively warn about significant weather transitions ~2 hours out
    ("Sir, the forecast turns notably wetter at 5 PM — 75% chance of
    rain. Worth knowing before you head outside.").

Public API (called by morning_briefing / evening_briefing):
  get_umbrella_alert(when="today") -> str
      Short JARVIS-style umbrella line for the rest of today (or tomorrow
      when when="tomorrow"). "" if no notable precipitation is expected.

  get_two_hour_alert() -> str
      Same shape as the proactive scheduler alert but synchronous, for
      callers that want to know "is something brewing in the next 2h?".
      "" if nothing significant is incoming.

Actions registered:
  weather_briefing   — manual: returns a single short forecast sentence
                       (umbrella alert if any, otherwise the "looks dry"
                       confirmation).
  weather_forecast   — alias of weather_briefing.

Scheduler:
  Background thread polls every WEATHER_POLL_MINUTES (default 30). If a
  notable transition (precipitation probability jumping, temperature
  dropping ≥5 °C, or the weather code transitioning into a meaningfully
  different category) is forecast within the next 2 hours, it enqueues a
  spoken alert via pending_speech.json. A 4-hour cooldown per-alert-class
  prevents repeated re-fires of the same warning.

Config knobs (read live from bobert_companion at call time):
  WEATHER_BRIEFING_ENABLED          bool, default True
  WEATHER_BRIEFING_PROACTIVE        bool, default True   (background watcher)
  WEATHER_POLL_MINUTES              int,  default 30
  WEATHER_UMBRELLA_PROB_THRESHOLD   int,  default 50    (% precip prob)
  WEATHER_LOOKAHEAD_HOURS           int,  default 2     (alert window)
  WEATHER_SIGNIFICANT_TEMP_DROP_C   int,  default 5
  WEATHER_ALERT_COOLDOWN_HOURS      int,  default 4
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE   = os.path.join(_PROJECT_DIR, "weather_briefing_state.json")

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.weather_briefing` or run directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_OPEN_METEO_URL     = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_TIMEOUT = 6.0

# Same WMO code descriptions used by briefing_sources.py — kept inline so this
# skill doesn't reach into a private table.
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

# Weather "category" for change-detection. The proactive watcher only fires
# when category transitions between dissimilar groups (clear → rain, rain →
# snow, anything → thunderstorm), not on adjacent shading (clear → cloudy).
def _weather_category(code: int) -> str:
    if code in (95, 96, 99):       return "thunderstorm"
    if code in (71, 73, 75, 77, 85, 86, 66, 67): return "snow"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82): return "rain"
    if code in (45, 48):           return "fog"
    if code in (3,):               return "overcast"
    if code in (1, 2):             return "cloudy"
    if code in (0,):               return "clear"
    return "unknown"


_DEFAULT_POLL_MINUTES         = 30
_DEFAULT_LOOKAHEAD_HOURS      = 2
_DEFAULT_UMBRELLA_PROB        = 50
_DEFAULT_SIG_TEMP_DROP_C      = 5
_DEFAULT_ALERT_COOLDOWN_HOURS = 4

_state_lock  = threading.Lock()


# ─── config helper ───────────────────────────────────────────────────────

def _config(name: str, default):
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return default
    return getattr(bc, name, default)


def _read_config() -> dict:
    return {
        "enabled":         bool(_config("WEATHER_BRIEFING_ENABLED",        True)),
        "proactive":       bool(_config("WEATHER_BRIEFING_PROACTIVE",      True)),
        "poll_minutes":    int (_config("WEATHER_POLL_MINUTES",            _DEFAULT_POLL_MINUTES)),
        "umbrella_prob":   int (_config("WEATHER_UMBRELLA_PROB_THRESHOLD", _DEFAULT_UMBRELLA_PROB)),
        "lookahead_h":     int (_config("WEATHER_LOOKAHEAD_HOURS",         _DEFAULT_LOOKAHEAD_HOURS)),
        "sig_temp_drop_c": int (_config("WEATHER_SIGNIFICANT_TEMP_DROP_C", _DEFAULT_SIG_TEMP_DROP_C)),
        "cooldown_h":      int (_config("WEATHER_ALERT_COOLDOWN_HOURS",    _DEFAULT_ALERT_COOLDOWN_HOURS)),
    }


# ─── JSON I/O ────────────────────────────────────────────────────────────

def _safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── speech queue ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Route a proactive announcement through bobert_companion.proactive_announce()
    — the canonical writer for pending_speech.json. Funnelling every skill
    through that one helper eliminates the cross-skill read-modify-write race
    that an independent local fallback would reintroduce. If the parent module
    isn't loaded yet (import-time / unit tests) or the announce call fails, the
    message is logged to the console so it isn't silently lost."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer) and announcer(message, source="weather"):
            return
    except Exception as e:
        print(f"  [weather] speech-queue write failed ({e}); alert: {message}")
        return
    print(f"  [weather] speech-queue unavailable; alert: {message}")


# ─── location ────────────────────────────────────────────────────────────

def _resolve_location() -> tuple | None:
    """Defer to briefing_sources._resolve_location so we share the same
    ipapi cache + config-pinned lat/lon plumbing the morning briefing uses."""
    try:
        from . import briefing_sources  # type: ignore
    except Exception:
        try:
            if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import briefing_sources  # type: ignore
        except Exception as e:
            print(f"  [weather] briefing_sources unavailable: {e}")
            return None
    try:
        return briefing_sources._resolve_location()
    except Exception as e:
        print(f"  [weather] location resolution failed: {e}")
        return None


# ─── Open-Meteo hourly fetch ─────────────────────────────────────────────

def _fetch_hourly_forecast() -> list:
    """Return a list of hourly dicts {ts, hour_local, temp_c, precip_prob,
    precip_mm, weather_code, desc, category} for roughly the next 48 hours,
    or [] on any error."""
    loc = _resolve_location()
    if loc is None:
        return []
    lat, lon = loc
    qs = urllib.parse.urlencode({
        "latitude":  f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly":    "temperature_2m,precipitation_probability,precipitation,weather_code",
        "forecast_days": "2",
        "timezone":  "auto",
    })
    url = f"{_OPEN_METEO_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_OPEN_METEO_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [weather] open-meteo hourly fetch failed: {e}")
        return []
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    probs = hourly.get("precipitation_probability") or []
    mms   = hourly.get("precipitation") or []
    codes = hourly.get("weather_code") or []
    out = []
    for i, tstr in enumerate(times):
        try:
            dt = datetime.fromisoformat(tstr)
        except Exception:
            continue
        try:
            temp_c = float(temps[i]) if i < len(temps) and temps[i] is not None else None
        except (TypeError, ValueError):
            temp_c = None
        try:
            prob = int(probs[i]) if i < len(probs) and probs[i] is not None else 0
        except (TypeError, ValueError):
            prob = 0
        try:
            mm = float(mms[i]) if i < len(mms) and mms[i] is not None else 0.0
        except (TypeError, ValueError):
            mm = 0.0
        try:
            code = int(codes[i]) if i < len(codes) and codes[i] is not None else -1
        except (TypeError, ValueError):
            code = -1
        out.append({
            "dt":           dt,
            "hour_local":   dt.hour,
            "temp_c":       temp_c,
            "precip_prob":  prob,
            "precip_mm":    mm,
            "weather_code": code,
            "desc":         _WMO_DESCRIPTIONS.get(code, ""),
            "category":     _weather_category(code),
        })
    return out


# ─── formatting helpers ──────────────────────────────────────────────────

def _format_hour(hour_24: int) -> str:
    suffix = "AM" if hour_24 < 12 else "PM"
    disp = hour_24 % 12 or 12
    return f"{disp} {suffix}"


def _slice_for_day(hourly: list, when: str) -> list:
    """Return only the hours falling on the requested local day. Hours that
    have already passed today are dropped so a 7 PM 'rain at 3 PM' alert
    doesn't fire after the fact."""
    now = datetime.now()
    if when == "tomorrow":
        target = (now + timedelta(days=1)).date()
        return [h for h in hourly if h["dt"].date() == target]
    target = now.date()
    return [h for h in hourly if h["dt"].date() == target and h["dt"] >= now.replace(minute=0, second=0, microsecond=0)]


# ─── public: umbrella alert ──────────────────────────────────────────────

def get_umbrella_alert(when: str = "today") -> str:
    """JARVIS-style umbrella warning for today (default) or tomorrow.

    Returns "" if no hour in the window meets the configured probability
    threshold, so callers can simply concatenate the result to the briefing.
    """
    cfg = _read_config()
    if not cfg["enabled"]:
        return ""

    hourly = _fetch_hourly_forecast()
    if not hourly:
        return ""
    window = _slice_for_day(hourly, when)
    if not window:
        return ""

    threshold = cfg["umbrella_prob"]
    # Find the most-likely-to-rain hour above threshold.
    rainy = [h for h in window
             if h["precip_prob"] >= threshold
             and h["category"] in ("rain", "snow", "thunderstorm")]
    if not rainy:
        # Also flag hours with measurable precip (≥1 mm) even if prob is missing.
        rainy = [h for h in window if h["precip_mm"] >= 1.0
                 and h["category"] in ("rain", "snow", "thunderstorm")]
    if not rainy:
        return ""

    peak = max(rainy, key=lambda h: (h["precip_prob"], h["precip_mm"]))
    label = "snow" if peak["category"] == "snow" else (
        "thunderstorms" if peak["category"] == "thunderstorm" else "rain"
    )
    time_phrase = f"at {_format_hour(peak['hour_local'])}"
    prob = peak["precip_prob"]
    day_phrase = "today" if when == "today" else "tomorrow"
    if label == "snow":
        return (
            f"I'd suggest layering up {day_phrase}, sir — "
            f"{prob}% chance of snow {time_phrase}."
        )
    if label == "thunderstorms":
        return (
            f"There are thunderstorms forecast {day_phrase} {time_phrase}, sir — "
            f"{prob}% probability. Best stay indoors if you can."
        )
    return (
        f"I'd suggest the umbrella {day_phrase}, sir — "
        f"{prob}% chance of rain {time_phrase}."
    )


# ─── public: two-hour change detector ────────────────────────────────────

def _detect_two_hour_change(hourly: list, cfg: dict):
    """Return (alert_class, message) for the next significant change within
    cfg['lookahead_h'] hours, or (None, '') if nothing notable is imminent.

    alert_class is one of: 'precip_jump', 'temp_drop', 'category_change',
    'thunderstorm_incoming'. Used as the cooldown key so the watcher won't
    re-fire the same class within cfg['cooldown_h'] hours.
    """
    now = datetime.now()
    cutoff = now + timedelta(hours=cfg["lookahead_h"])
    window = [h for h in hourly if now <= h["dt"] <= cutoff]
    if len(window) < 2:
        return None, ""

    # Compare against "now-ish" — first hourly bucket whose dt straddles now.
    current = None
    for h in hourly:
        if h["dt"] <= now < h["dt"] + timedelta(hours=1):
            current = h
            break
    if current is None and window:
        current = window[0]

    # 1) Thunderstorms incoming take priority — they're the high-stakes case.
    for h in window:
        if h["category"] == "thunderstorm":
            return (
                "thunderstorm_incoming",
                f"Sir, thunderstorms are forecast around {_format_hour(h['hour_local'])} — "
                f"{h['precip_prob']}% probability. Worth being aware.",
            )

    # 2) Precipitation jump — current low, upcoming high.
    if current is not None:
        cur_prob = current["precip_prob"]
        for h in window:
            if (h["precip_prob"] - cur_prob) >= 40 and h["precip_prob"] >= cfg["umbrella_prob"]:
                label = "snow" if h["category"] == "snow" else "rain"
                return (
                    "precip_jump",
                    f"Sir, the forecast turns notably wetter at "
                    f"{_format_hour(h['hour_local'])} — {h['precip_prob']}% chance of {label}. "
                    f"Worth knowing before you head outside.",
                )

    # 3) Significant temperature drop within window.
    if current is not None and current["temp_c"] is not None:
        cur_t = current["temp_c"]
        for h in window:
            if h["temp_c"] is None:
                continue
            drop = cur_t - h["temp_c"]
            if drop >= cfg["sig_temp_drop_c"]:
                # Speak the delta in Fahrenheit (sir's preference). A temperature
                # drop is a span, so convert with *9/5 only -- no +32 offset. The
                # internal threshold stays Celsius; a 5C drop voices as ~9F.
                drop_f = int(round(drop * 9 / 5))
                return (
                    "temp_drop",
                    f"Sir, the temperature drops about {drop_f} degrees by "
                    f"{_format_hour(h['hour_local'])} — you may want a jacket.",
                )

    # 4) Category transition into a precipitation/fog state.
    if current is not None:
        cur_cat = current["category"]
        for h in window:
            new_cat = h["category"]
            if new_cat == cur_cat:
                continue
            if new_cat in ("rain", "snow", "fog"):
                return (
                    "category_change",
                    f"Sir, conditions shift to {h['desc']} around "
                    f"{_format_hour(h['hour_local'])}.",
                )

    return None, ""


def get_two_hour_alert() -> str:
    """Synchronous version of the proactive watcher — returns the message
    that would have been spoken, or '' if nothing notable is incoming."""
    cfg = _read_config()
    if not cfg["enabled"]:
        return ""
    hourly = _fetch_hourly_forecast()
    if not hourly:
        return ""
    _klass, msg = _detect_two_hour_change(hourly, cfg)
    return msg


# ─── proactive watcher ───────────────────────────────────────────────────

def _alert_cooldown_active(alert_class: str, cooldown_seconds: float) -> bool:
    with _state_lock:
        state = _safe_load_json(_STATE_FILE) or {}
    last = (state.get("alerts") or {}).get(alert_class)
    if last is None:
        return False
    try:
        return (time.time() - float(last)) < cooldown_seconds
    except (TypeError, ValueError):
        return False


def _record_alert(alert_class: str) -> None:
    with _state_lock:
        state = _safe_load_json(_STATE_FILE) or {}
        alerts = state.get("alerts") or {}
        alerts[alert_class] = time.time()
        state["alerts"] = alerts
        try:
            _atomic_write_json(_STATE_FILE, state)
        except Exception as e:
            print(f"  [weather] state write failed: {e}")


def _watch_loop() -> None:
    # Small initial delay so JARVIS finishes booting before we hit the network.
    time.sleep(60)
    while True:
        try:
            cfg = _read_config()
            if not (cfg["enabled"] and cfg["proactive"]):
                time.sleep(cfg["poll_minutes"] * 60)
                continue
            try:
                hourly = _fetch_hourly_forecast()
                if hourly:
                    klass, msg = _detect_two_hour_change(hourly, cfg)
                    if klass and msg:
                        cooldown_s = cfg["cooldown_h"] * 3600
                        if not _alert_cooldown_active(klass, cooldown_s):
                            print(f"  [weather] proactive alert ({klass}): {msg}")
                            _enqueue_speech(msg)
                            _record_alert(klass)
            except Exception as e:
                print(f"  [weather] watcher iteration failed: {e}")
            time.sleep(max(60, cfg["poll_minutes"] * 60))
        except Exception:
            logging.exception("[weather] _watch_loop iteration crashed")
            time.sleep(60)


# ─── action registration ─────────────────────────────────────────────────

def register(actions):
    def weather_briefing(_: str = "") -> str:
        try:
            umbrella = get_umbrella_alert("today")
            if umbrella:
                return umbrella
            two_hour = get_two_hour_alert()
            if two_hour:
                return two_hour
            return "Forecast looks unremarkable for the rest of the day, sir."
        except Exception as e:
            return f"weather briefing failed: {e}"

    actions["weather_briefing"] = weather_briefing
    actions["weather_forecast"] = weather_briefing

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [weather] WEATHER_BRIEFING_ENABLED is False — watcher disabled")
        return
    if not cfg["proactive"]:
        print("  [weather] proactive watcher disabled (WEATHER_BRIEFING_PROACTIVE=False)")
        return
    t = threading.Thread(target=_watch_loop, daemon=True)
    t.start()
    print(
        f"  [weather] proactive watcher armed "
        f"(poll {cfg['poll_minutes']} min, lookahead {cfg['lookahead_h']} h)"
    )


# ─── manual smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("umbrella (today):   ", get_umbrella_alert("today"))
    print("umbrella (tomorrow):", get_umbrella_alert("tomorrow"))
    print("two-hour alert:     ", get_two_hour_alert())
