"""
briefing_sources.py — centralised data sources for the morning_briefing and
daily_briefing skills, with a fallback chain so a single dead service can't
silently degrade the briefing into the "wttr fetch failed / outlook query
failed" no-op that was showing up in nearly every session log.

Weather chain (get_weather_data):
  1. wttr.in (?format=j1, geolocated by IP)
  2. Open-Meteo (free, no API key; lat/lon resolved via ipapi.co or
     bobert_companion.OPEN_METEO_LAT / OPEN_METEO_LON if configured)
  3. weather_cache.json (last-known good value, written on every success)

Calendar chain (get_first_meeting_data):
  1. Outlook COM (pythoncom + win32com.client)
  2. Microsoft Graph REST (best-effort; requires a token file produced by an
     external OAuth flow — when not configured, this layer silently skips.
     NOTE: the spec referenced the mcp_claude_ai_Microsoft_365 MCP tool. MCP
     tools are exposed to Claude Code / claude.ai, not to runtime Python, so
     they cannot be invoked from this file. This layer is therefore a stub
     that consults microsoft_graph_token.json if the user has configured one.)
  3. Google Calendar (secret-address ICS URL via
     bobert_companion.GOOGLE_CALENDAR_ICS_URL; standard library only)

Both getters return a dict with a 'source' field naming the layer that
succeeded, or None if every layer failed. Formatting is left to callers so
morning_briefing's "18 degrees and overcast in your area" and
daily_briefing's "outside temperature is 14 degrees and overcast" wording
can each stay intact.
"""
from __future__ import annotations

import datetime
import importlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_PROJECT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEATHER_CACHE_FILE = os.path.join(_PROJECT_DIR, "weather_cache.json")
_GEO_CACHE_FILE     = os.path.join(_PROJECT_DIR, "geo_cache.json")
_GRAPH_TOKEN_FILE   = os.path.join(_PROJECT_DIR, "microsoft_graph_token.json")

_WTTR_URL          = "https://wttr.in/?format=j1"
_WTTR_TIMEOUT      = 6.0
_OPEN_METEO_URL    = "https://api.open-meteo.com/v1/forecast"
_OPEN_METEO_TIMEOUT = 6.0
_IPAPI_URL         = "https://ipapi.co/json/"
_IPAPI_TIMEOUT     = 4.0
_ICS_TIMEOUT       = 6.0
_GRAPH_TIMEOUT     = 6.0

_GEO_CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600   # 30 days
_WEATHER_CACHE_USABLE_MAX  = 12 * 3600        # don't quote a >12h-old reading

_cache_lock = threading.Lock()


# ─── config helpers ──────────────────────────────────────────────────────

def _config(name: str, default):
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        return default
    return getattr(bc, name, default)


def _atomic_write_json(path: str, payload: dict) -> None:
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise


def _safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── weather: wttr.in ────────────────────────────────────────────────────

def _weather_from_wttr() -> dict | None:
    req = urllib.request.Request(_WTTR_URL, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=_WTTR_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    current = data["current_condition"][0]
    temp_c = int(float(current.get("temp_C", "0")))
    desc = (current.get("weatherDesc", [{}])[0].get("value", "") or "").strip().lower()
    return {"temp_c": temp_c, "desc": desc}


# ─── weather: Open-Meteo ─────────────────────────────────────────────────

# WMO weather codes → short human descriptions
_OPEN_METEO_DESCRIPTIONS = {
    0:  "clear",
    1:  "mainly clear",
    2:  "partly cloudy",
    3:  "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def _resolve_location() -> tuple[float, float] | None:
    """Return (lat, lon) from config, fresh ipapi lookup, or cached geo —
    in that order. Caches successful IP-based lookups for 30 days."""
    lat = _config("OPEN_METEO_LAT", None)
    lon = _config("OPEN_METEO_LON", None)
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass

    # Try a cached geolocation first to avoid hammering ipapi
    cached = _safe_load_json(_GEO_CACHE_FILE)
    if cached and (time.time() - cached.get("ts", 0.0)) < _GEO_CACHE_MAX_AGE_SECONDS:
        try:
            return float(cached["lat"]), float(cached["lon"])
        except (KeyError, TypeError, ValueError):
            pass

    # Fall back to a fresh ipapi lookup
    try:
        req = urllib.request.Request(_IPAPI_URL, headers={"User-Agent": "jarvis/1.0"})
        with urllib.request.urlopen(req, timeout=_IPAPI_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        flat = float(payload["latitude"])
        flon = float(payload["longitude"])
    except Exception as e:
        print(f"  [briefing-src] ipapi geolocation failed: {e}")
        # Fall back to a stale cached value if we have one
        if cached:
            try:
                return float(cached["lat"]), float(cached["lon"])
            except (KeyError, TypeError, ValueError):
                pass
        return None

    try:
        with _cache_lock:
            _atomic_write_json(_GEO_CACHE_FILE, {"lat": flat, "lon": flon, "ts": time.time()})
    except Exception:
        pass
    return flat, flon


def _weather_from_open_meteo() -> dict | None:
    loc = _resolve_location()
    if loc is None:
        return None
    lat, lon = loc
    qs = urllib.parse.urlencode({
        "latitude":  f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "current":   "temperature_2m,weather_code",
        "timezone":  "auto",
    })
    url = f"{_OPEN_METEO_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis/1.0"})
    with urllib.request.urlopen(req, timeout=_OPEN_METEO_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    current = data.get("current") or {}
    if "temperature_2m" not in current:
        return None
    temp_c = int(round(float(current["temperature_2m"])))
    code = int(current.get("weather_code", -1))
    desc = _OPEN_METEO_DESCRIPTIONS.get(code, "")
    return {"temp_c": temp_c, "desc": desc}


# ─── weather: cache fallback ─────────────────────────────────────────────

def _save_weather_cache(data: dict) -> None:
    payload = {
        "temp_c": data.get("temp_c"),
        "desc":   data.get("desc", ""),
        "source": data.get("source", ""),
        "ts":     time.time(),
    }
    try:
        with _cache_lock:
            _atomic_write_json(_WEATHER_CACHE_FILE, payload)
    except Exception as e:
        print(f"  [briefing-src] weather cache write failed: {e}")


def _weather_from_cache() -> dict | None:
    payload = _safe_load_json(_WEATHER_CACHE_FILE)
    if not payload or "temp_c" not in payload:
        return None
    # If the cache is fresher than _WEATHER_CACHE_USABLE_MAX, just hand it back
    # silently. If it's older, still return it but mark it stale so callers can
    # caveat the report if they want to.
    age = time.time() - float(payload.get("ts", 0.0))
    return {
        "temp_c":      payload["temp_c"],
        "desc":        payload.get("desc", "") or "",
        "stale":       age > _WEATHER_CACHE_USABLE_MAX,
        "cached_age_s": age,
    }


# ─── weather: public chain entry point ───────────────────────────────────

def get_weather_data() -> dict | None:
    """Try each weather source in order. Returns {temp_c, desc, source[, stale]}
    or None if every layer failed."""
    chain = [
        ("wttr",       _weather_from_wttr),
        ("open-meteo", _weather_from_open_meteo),
    ]
    for label, fn in chain:
        try:
            data = fn()
        except Exception as e:
            print(f"  [briefing-src] {label} fetch failed: {e}")
            continue
        if data:
            data["source"] = label
            _save_weather_cache(data)
            return data

    cached = _weather_from_cache()
    if cached:
        cached["source"] = "cache"
        return cached
    return None


# ─── calendar: Outlook COM ───────────────────────────────────────────────

def _meeting_window(when: str) -> tuple[datetime.datetime, datetime.datetime]:
    """Return (start, end) datetime window for 'today' or 'tomorrow'."""
    now = datetime.datetime.now()
    if when == "tomorrow":
        d = (now + datetime.timedelta(days=1)).date()
        return (
            datetime.datetime.combine(d, datetime.time(0, 0)),
            datetime.datetime.combine(d, datetime.time(23, 59, 59)),
        )
    # default: today, from now → end of day
    return now, now.replace(hour=23, minute=59, second=59, microsecond=0)


def _meeting_from_outlook(when: str) -> dict | None:
    try:
        import pythoncom                  # type: ignore
        import win32com.client            # type: ignore
    except Exception:
        return None

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

        start_dt, end_dt = _meeting_window(when)
        fmt = "%m/%d/%Y %I:%M %p"
        restriction = (
            f"[Start] >= '{start_dt.strftime(fmt)}' AND "
            f"[Start] <= '{end_dt.strftime(fmt)}'"
        )
        try:
            window_items = items.Restrict(restriction)
        except Exception:
            window_items = items

        for appt in window_items:
            try:
                start = appt.Start
                if hasattr(start, "Format"):
                    s_dt = datetime.datetime(
                        start.year, start.month, start.day,
                        start.hour, start.minute,
                    )
                else:
                    s_dt = start
                if s_dt < start_dt or s_dt > end_dt:
                    continue
                subject = (getattr(appt, "Subject", "") or "").strip()
                organizer = (getattr(appt, "Organizer", "") or "").strip()
                return {
                    "start":     s_dt,
                    "subject":   subject,
                    "organizer": organizer,
                }
            except Exception:
                continue
        return None
    except Exception as e:
        print(f"  [briefing-src] outlook query failed: {e}")
        return None
    finally:
        try:
            pythoncom.CoUninitialize()  # type: ignore
        except Exception:
            pass


# ─── calendar: Microsoft Graph (best-effort, token-file based) ───────────

def _meeting_from_graph(when: str) -> dict | None:
    """Use Microsoft Graph's /me/calendarView endpoint if the user has a
    valid bearer token cached in microsoft_graph_token.json. The MCP tool
    referenced in the original spec (mcp_claude_ai_Microsoft_365) is a
    Claude Code tool and cannot be invoked from runtime Python, so this is
    the closest direct-API equivalent.

    Expected token-file shape:
      {"access_token": "...", "expires_at": <epoch>}
    """
    payload = _safe_load_json(_GRAPH_TOKEN_FILE)
    if not payload or not payload.get("access_token"):
        return None
    if payload.get("expires_at") and float(payload["expires_at"]) < time.time():
        # Token expired — refresh flow is out of scope here; skip silently.
        return None

    start_dt, end_dt = _meeting_window(when)
    qs = urllib.parse.urlencode({
        "startDateTime": start_dt.isoformat(),
        "endDateTime":   end_dt.isoformat(),
        "$orderby":      "start/dateTime",
        "$top":          "1",
        "$select":       "subject,start,organizer",
    })
    url = f"https://graph.microsoft.com/v1.0/me/calendarView?{qs}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {payload['access_token']}",
        "Accept":        "application/json",
        "Prefer":        'outlook.timezone="UTC"',
    })
    try:
        with urllib.request.urlopen(req, timeout=_GRAPH_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  [briefing-src] graph http {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"  [briefing-src] graph query failed: {e}")
        return None

    events = body.get("value") or []
    if not events:
        return None
    evt = events[0]
    try:
        start_iso = evt["start"]["dateTime"]
        # Graph returns naive ISO; treat as UTC and convert to local
        s_dt_utc = datetime.datetime.fromisoformat(start_iso.replace("Z", ""))
        s_dt_local = s_dt_utc.replace(tzinfo=datetime.timezone.utc).astimezone(None).replace(tzinfo=None)
    except Exception:
        return None
    organizer = ""
    try:
        organizer = (evt["organizer"]["emailAddress"].get("name") or "").strip()
    except Exception:
        pass
    return {
        "start":     s_dt_local,
        "subject":   (evt.get("subject") or "").strip(),
        "organizer": organizer,
    }


# ─── calendar: Google Calendar ICS (public secret-address URL) ───────────

_ICS_VEVENT_RE  = re.compile(r"BEGIN:VEVENT(.+?)END:VEVENT", re.DOTALL)
_ICS_DTSTART_RE = re.compile(r"^DTSTART(?:;[^:]*)?:(.+)$", re.MULTILINE)
_ICS_SUMMARY_RE = re.compile(r"^SUMMARY:(.+)$", re.MULTILINE)
_ICS_ORG_RE     = re.compile(r"^ORGANIZER(?:;[^:]*CN=([^;:]+))?", re.MULTILINE)


def _parse_ics_dtstart(raw: str) -> datetime.datetime | None:
    """Parse an ICS DTSTART value into a naive local datetime."""
    raw = raw.strip()
    # All-day events use YYYYMMDD; we treat them as midnight local.
    if re.fullmatch(r"\d{8}", raw):
        return datetime.datetime.strptime(raw, "%Y%m%d")
    # UTC: YYYYMMDDTHHMMSSZ
    if raw.endswith("Z"):
        try:
            dt = datetime.datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
            return dt.replace(tzinfo=datetime.timezone.utc).astimezone(None).replace(tzinfo=None)
        except ValueError:
            return None
    # Local: YYYYMMDDTHHMMSS
    try:
        return datetime.datetime.strptime(raw, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _meeting_from_google_ics(when: str) -> dict | None:
    url = (_config("GOOGLE_CALENDAR_ICS_URL", "") or "").strip()
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jarvis/1.0"})
        with urllib.request.urlopen(req, timeout=_ICS_TIMEOUT) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [briefing-src] google-ics fetch failed: {e}")
        return None

    # ICS line-unfolding: continuation lines start with a space or tab.
    text = re.sub(r"\r?\n[ \t]", "", text)

    start_dt, end_dt = _meeting_window(when)
    best: dict | None = None
    for m in _ICS_VEVENT_RE.finditer(text):
        block = m.group(1)
        ds_m = _ICS_DTSTART_RE.search(block)
        if not ds_m:
            continue
        s_dt = _parse_ics_dtstart(ds_m.group(1))
        if s_dt is None or s_dt < start_dt or s_dt > end_dt:
            continue
        subj_m = _ICS_SUMMARY_RE.search(block)
        org_m  = _ICS_ORG_RE.search(block)
        candidate = {
            "start":     s_dt,
            "subject":   (subj_m.group(1).strip() if subj_m else ""),
            "organizer": (org_m.group(1).strip() if (org_m and org_m.group(1)) else ""),
        }
        if best is None or candidate["start"] < best["start"]:
            best = candidate
    return best


# ─── calendar: public chain entry point ──────────────────────────────────

def get_first_meeting_data(when: str = "today") -> dict | None:
    """Try each calendar source in order. Returns {start, subject, organizer,
    source} or None if every layer failed or no events were found.

    ``when`` is 'today' (now → end-of-day) or 'tomorrow'.
    """
    chain = [
        ("outlook",   _meeting_from_outlook),
        ("graph",     _meeting_from_graph),
        ("google-ics", _meeting_from_google_ics),
    ]
    for label, fn in chain:
        try:
            data = fn(when)
        except Exception as e:
            print(f"  [briefing-src] {label} lookup raised: {e}")
            continue
        if data:
            data["source"] = label
            return data
    return None


# ─── manual smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("weather:", json.dumps(get_weather_data(), default=str, indent=2))
    print("meeting (today):", json.dumps(get_first_meeting_data("today"), default=str, indent=2))
    print("meeting (tomorrow):", json.dumps(get_first_meeting_data("tomorrow"), default=str, indent=2))
