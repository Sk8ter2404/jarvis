"""
JARVIS transient briefing card overlay.

Companion to hud/jarvis_hud.py — that one is a permanent corner ring. This
one pops up a larger, transient card on the top monitor when JARVIS
delivers morning_briefing / evening_briefing / status_report, showing:

  * Today's date
  * Current weather (icon + temp + conditions)
  * 3-day forecast (icon, label, high/low, conditions)
  * Up to 3 upcoming Outlook calendar items
  * Active Bambu H2D print status (filename, layer, percent, ETA)

Auto-dismisses after 20 seconds, or earlier when JARVIS detects "thank
you, JARVIS" / "dismiss" / "close that" in a transcribed utterance.

Public API (called from skills + main loop):
    show_card(card_type, duration_seconds=20.0)
    dismiss_card()
    is_card_active() -> bool
    matches_dismiss_phrase(text) -> bool

Subprocess entry point (renderer):
    python hud_card.py --render --parent-pid <pid>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Optional

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_STATE_FILE  = os.path.join(_PROJECT_DIR, "hud_card_state.json")
_PID_FILE    = os.path.join(_PROJECT_DIR, "hud_card.pid")

DEFAULT_DURATION_SECONDS = 20.0

DISMISS_PHRASES = (
    "thank you jarvis", "thank you, jarvis", "thanks jarvis", "thanks, jarvis",
    "dismiss the card", "dismiss that", "dismiss the briefing", "close that",
    "close the card", "close the briefing", "got it jarvis", "got it, jarvis",
)

WTTR_URL     = "https://wttr.in/?format=j1"
WTTR_TIMEOUT = 6.0

_lock = threading.Lock()


# ─── public API ───────────────────────────────────────────────────────────

def show_card(card_type: str,
              duration_seconds: float = DEFAULT_DURATION_SECONDS) -> None:
    """Build a briefing card, write the state file, and ensure the renderer
    subprocess is running. Safe to call repeatedly — the renderer picks up
    the new state on its next 250 ms poll.

    card_type: 'morning' | 'evening' | 'status' (controls only the title).
    """
    try:
        # STAGING/harness processes must never pop a renderer window on the
        # live desktop — JARVIS_STAGING reroutes state files, but the SCREEN
        # is shared. Six skills (briefings, dossier, calendar, status panel)
        # all funnel through this one entry point, so the guard lives here
        # rather than in each caller (the 2026-07-11 action sweep rendered a
        # card on the owner's monitor through exactly this path).
        try:
            from core import is_staging
            if is_staging():
                print(f"  [hud_card] staging role — suppressing '{card_type}' "
                      f"card render")
                return
        except Exception:
            pass
        state = _build_card_state(card_type, duration_seconds)
        _write_state(state)
        _ensure_renderer_running()
    except Exception as e:
        print(f"  [hud_card] show_card failed: {e}")


def dismiss_card() -> None:
    """Mark the card dismissed so the renderer closes on next poll."""
    try:
        with _lock:
            if not os.path.exists(_STATE_FILE):
                return
            try:
                with open(_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return
            data["dismissed"] = True
            tmp = _STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, _STATE_FILE)
    except Exception as e:
        print(f"  [hud_card] dismiss_card failed: {e}")


def is_card_active() -> bool:
    """True if a card is currently displayed (state file exists, not expired,
    not dismissed). False on any read failure — fail closed."""
    if not os.path.exists(_STATE_FILE):
        return False
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if data.get("dismissed"):
        return False
    try:
        return float(data.get("expiry_ts", 0.0)) > time.time()
    except (TypeError, ValueError):
        return False


def matches_dismiss_phrase(text: str) -> bool:
    """Quick check: does this transcribed utterance contain a dismiss phrase?
    Used by the main loop right after whisper validates a turn."""
    if not text:
        return False
    t = text.strip().lower()
    return any(p in t for p in DISMISS_PHRASES)


# ─── card-state builder ───────────────────────────────────────────────────

_TITLE_MAP = {
    "morning": "Morning Briefing",
    "evening": "Evening Briefing",
    "status":  "Status Report",
}


def _build_card_state(card_type: str, duration_seconds: float) -> dict:
    now = time.time()
    title = _TITLE_MAP.get(card_type, (card_type or "Briefing").title())
    geom  = _get_top_monitor_geometry_inproc()
    return {
        "id":         str(int(now * 1000)),
        "card_type":  card_type,
        "title":      title,
        "date_line":  _date_line(),
        "weather":    _gather_weather_now(),
        "forecast":   _gather_forecast(),
        "calendar":   _gather_calendar(),
        "unread_mail": _gather_unread_mail(),
        "bambu":      _gather_bambu(),
        "geometry":   list(geom),
        "shown_at":   now,
        "expiry_ts":  now + max(1.0, float(duration_seconds)),
        "dismissed":  False,
    }


def _date_line() -> str:
    now = time.localtime()
    return (
        f"{time.strftime('%A', now)}, "
        f"{time.strftime('%B', now)} {_ordinal(now.tm_mday)}, "
        f"{now.tm_year}"
    )


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Map weather description keywords to a unicode glyph. Order matters — longest
# / most-specific keys first. Segoe UI Emoji on Win10/11 renders these.
_WEATHER_EMOJI = [
    ("thunder",       "⛈"),   # ⛈
    ("partly cloudy", "⛅"),   # ⛅
    ("rain",          "\U0001F327"),
    ("drizzle",       "\U0001F326"),
    ("snow",          "❄"),
    ("sleet",         "\U0001F328"),
    ("fog",           "\U0001F32B"),
    ("mist",          "\U0001F32B"),
    ("cloud",         "☁"),
    ("overcast",      "☁"),
    ("clear",         "☀"),
    ("sunny",         "☀"),
]


def _emoji_for_desc(desc: str) -> str:
    d = (desc or "").lower()
    for needle, glyph in _WEATHER_EMOJI:
        if needle in d:
            return glyph
    return "\U0001F321"   # thermometer


# Cached wttr.in fetch — the same JSON drives both current + forecast,
# and multiple briefings firing the same morning shouldn't slam the API.
_wttr_cache = {"ts": 0.0, "data": None}
_WTTR_CACHE_TTL_SECONDS = 600.0


def _fetch_wttr() -> Optional[dict]:
    now = time.time()
    if _wttr_cache["data"] and (now - _wttr_cache["ts"]) < _WTTR_CACHE_TTL_SECONDS:
        return _wttr_cache["data"]
    try:
        req = urllib.request.Request(WTTR_URL, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=WTTR_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _wttr_cache["data"] = data
        _wttr_cache["ts"] = now
        return data
    except Exception as e:
        print(f"  [hud_card] wttr fetch failed: {e}")
        return None


def _gather_weather_now() -> Optional[dict]:
    data = _fetch_wttr()
    if not data:
        return None
    try:
        current = data["current_condition"][0]
        temp_c = int(float(current.get("temp_C", "0")))
        desc = (current.get("weatherDesc", [{}])[0].get("value", "") or "").strip()
        return {
            "temp_c": temp_c,
            "desc": desc,
            "emoji": _emoji_for_desc(desc),
        }
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _gather_forecast() -> list:
    data = _fetch_wttr()
    if not data:
        return []
    out = []
    weather = data.get("weather") or []
    for idx, day in enumerate(weather[:3]):
        try:
            max_c = int(float(day.get("maxtempC", "0")))
            min_c = int(float(day.get("mintempC", "0")))
            hourly = day.get("hourly") or []
            noon = next(
                (h for h in hourly if str(h.get("time", "")) in ("1200", "1100", "1300")),
                None,
            )
            if noon is None and hourly:
                noon = hourly[len(hourly) // 2]
            desc = ""
            if noon:
                try:
                    desc = (noon.get("weatherDesc", [{}])[0].get("value", "") or "").strip()
                except (KeyError, IndexError, TypeError):
                    desc = ""
            if idx == 0:
                label = "Today"
            elif idx == 1:
                label = "Tomorrow"
            else:
                dstr = day.get("date") or ""
                try:
                    label = time.strftime("%A", time.strptime(dstr, "%Y-%m-%d"))
                except (ValueError, TypeError):
                    label = f"+{idx}"
            out.append({
                "label":  label,
                "high_c": max_c,
                "low_c":  min_c,
                "desc":   desc,
                "emoji":  _emoji_for_desc(desc),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _import_ms_graph():
    """Lazy-load skills/ms_graph.py so this module stays importable even if
    the skills directory hasn't been added to sys.path yet."""
    skills_dir = os.path.join(_PROJECT_DIR, "skills")
    if skills_dir not in sys.path:
        sys.path.insert(0, skills_dir)
    try:
        import ms_graph                # type: ignore
        return ms_graph
    except Exception as e:
        print(f"  [hud_card] ms_graph unavailable: {e}")
        return None


def _gather_calendar() -> list:
    """Top 3 upcoming events in the next 14 days via Microsoft Graph.
    Returns [] when Graph isn't authenticated — no Outlook desktop dependency."""
    import datetime as _dt

    mg = _import_ms_graph()
    if mg is None:
        return []
    try:
        events = mg.get_upcoming_events(top_n=3, when="next_14_days")
    except Exception as e:
        print(f"  [hud_card] graph calendar failed: {e}")
        return []

    now_dt = _dt.datetime.now()
    out: list = []
    for evt in events:
        try:
            sdt = evt.get("start")
            if not isinstance(sdt, _dt.datetime):
                continue
            subject = (evt.get("subject") or "").strip()
            hour = sdt.hour
            minute = sdt.minute
            disp_hour = hour % 12 or 12
            suffix = "AM" if hour < 12 else "PM"
            if sdt.date() == now_dt.date():
                day_label = ""
            elif sdt.date() == now_dt.date() + _dt.timedelta(days=1):
                day_label = "Tomorrow "
            else:
                day_label = sdt.strftime("%a ")
            out.append({
                "time":    f"{day_label}{disp_hour}:{minute:02d} {suffix}",
                "subject": subject or "(no subject)",
            })
            if len(out) >= 3:
                break
        except Exception:
            continue
    return out


def _gather_unread_mail() -> Optional[int]:
    """Inbox unread count via Microsoft Graph, or None when unavailable."""
    mg = _import_ms_graph()
    if mg is None:
        return None
    try:
        return mg.get_unread_mail_count()
    except Exception as e:
        print(f"  [hud_card] graph unread mail failed: {e}")
        return None


def _gather_bambu() -> Optional[dict]:
    """Pull current Bambu H2D state via skills/bambu_monitor.py if loaded."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return None
    try:
        lock = getattr(mod, "_state_lock", None)
        if lock is not None:
            with lock:
                state = dict(getattr(mod, "_state", {}))
        else:
            state = dict(getattr(mod, "_state", {}))
    except Exception:
        return None
    if not state or state.get("last_update", 0.0) == 0.0:
        return None
    gcode_state = (state.get("gcode_state") or "").upper()
    if gcode_state not in ("RUNNING", "PREPARE", "PAUSE", "FINISH", "FAILED"):
        return None
    fname = state.get("filename") or ""
    try:
        stripper = getattr(mod, "_strip_filename", None)
        if stripper is not None:
            fname = stripper(fname) or fname
    except Exception:
        pass
    return {
        "status":            gcode_state,
        "filename":          fname,
        "percent":           int(state.get("mc_percent") or 0),
        "layer":             int(state.get("layer_num") or 0),
        "total_layers":      int(state.get("total_layer") or 0),
        "remaining_minutes": int(state.get("mc_remaining") or 0),
    }


def _get_top_monitor_geometry_inproc() -> tuple:
    """Read MONITORS['top'] from the already-loaded bobert_companion module,
    falling back to a sensible default if it isn't loaded (e.g. running this
    file standalone for demo)."""
    bc = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    try:
        m = getattr(bc, "MONITORS", None) if bc else None
        if isinstance(m, dict):
            if "top" in m:
                return tuple(m["top"])
            if m:
                return tuple(next(iter(m.values())))
    except Exception:
        pass
    return (0, 0, 1920, 1080)


# ─── state I/O + subprocess management ───────────────────────────────────

def _write_state(state: dict) -> None:
    with _lock:
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _STATE_FILE)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil   # type: ignore
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def _renderer_alive() -> bool:
    if not os.path.exists(_PID_FILE):
        return False
    try:
        with open(_PID_FILE, "r", encoding="utf-8") as f:
            pid = int((f.read() or "0").strip() or "0")
    except Exception:
        return False
    return _pid_alive(pid)


def _ensure_renderer_running() -> None:
    # Hold _lock across the alive-check and the spawn so two near-simultaneous
    # show_card() calls can't both pass the check and spawn duplicate renderer
    # subprocesses. The caller (show_card) does not hold _lock here.
    with _lock:
        if _renderer_alive():
            return
        try:
            parent_pid = os.getpid()
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 "--render", "--parent-pid", str(parent_pid)],
                creationflags=creationflags,
                close_fds=True,
            )
        except Exception as e:
            print(f"  [hud_card] failed to spawn renderer: {e}")


# ─── renderer (subprocess entry point) ────────────────────────────────────

def _renderer_main(parent_pid: int) -> int:
    import tkinter as tk

    try:
        with open(_PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    state = _load_state_safe()
    if not state:
        return 0

    geom = state.get("geometry") or [0, 0, 1920, 1080]
    try:
        mon_x, mon_y, mon_w, mon_h = (int(v) for v in geom[:4])
    except (TypeError, ValueError):
        mon_x, mon_y, mon_w, mon_h = 0, 0, 1920, 1080

    # 80% × 80%, centered on the top monitor — large enough to feel
    # "transient full-screen card" without literally covering everything.
    CARD_W = max(800,  int(mon_w * 0.80))
    CARD_H = max(500,  int(mon_h * 0.80))
    CARD_X = mon_x + (mon_w - CARD_W) // 2
    CARD_Y = mon_y + (mon_h - CARD_H) // 2

    BG       = "#04080d"
    BORDER   = "#4cc9ff"
    TITLE_FG = "#9ee7ff"
    TEXT_FG  = "#cfeefb"
    DIM_FG   = "#5d8aa3"
    GOLD     = "#ffd166"
    GREEN    = "#9ff9c4"
    ALERT    = "#ff5b5b"

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{CARD_W}x{CARD_H}+{CARD_X}+{CARD_Y}")
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.94)
    except Exception:
        pass
    root.configure(bg=BG)

    border_frame = tk.Frame(root, bg=BORDER, padx=2, pady=2)
    border_frame.pack(fill="both", expand=True)
    inner = tk.Frame(border_frame, bg=BG)
    inner.pack(fill="both", expand=True, padx=2, pady=2)

    title_lbl = tk.Label(
        inner, text=state.get("title", "Briefing"),
        font=("Segoe UI", 30, "bold"), fg=TITLE_FG, bg=BG,
    )
    title_lbl.pack(pady=(22, 4))
    tk.Label(
        inner, text=state.get("date_line", ""),
        font=("Segoe UI", 16), fg=DIM_FG, bg=BG,
    ).pack(pady=(0, 18))

    body = tk.Frame(inner, bg=BG)
    body.pack(fill="both", expand=True, padx=40, pady=(0, 8))
    body.grid_columnconfigure(0, weight=1, uniform="col")
    body.grid_columnconfigure(1, weight=1, uniform="col")
    body.grid_rowconfigure(0, weight=1)

    # ── LEFT: Weather now + 3-day forecast ──
    weather_col = tk.Frame(body, bg=BG)
    weather_col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

    weather = state.get("weather")
    if weather:
        big = tk.Frame(weather_col, bg=BG)
        big.pack(anchor="w", pady=(0, 6))
        tk.Label(
            big, text=weather.get("emoji", "\U0001F321"),
            font=("Segoe UI Emoji", 64), fg=TEXT_FG, bg=BG,
        ).pack(side="left")
        tcol = tk.Frame(big, bg=BG)
        tcol.pack(side="left", padx=(18, 0), anchor="s")
        tk.Label(
            tcol, text=f"{weather.get('temp_c', '?')}°C",
            font=("Segoe UI", 44, "bold"), fg=TEXT_FG, bg=BG,
        ).pack(anchor="w")
        tk.Label(
            tcol, text=(weather.get("desc", "") or "").title(),
            font=("Segoe UI", 15), fg=DIM_FG, bg=BG,
        ).pack(anchor="w")
    else:
        tk.Label(
            weather_col, text="Weather unavailable",
            font=("Segoe UI", 14), fg=DIM_FG, bg=BG,
        ).pack(anchor="w", pady=(18, 0))

    tk.Label(
        weather_col, text="3-DAY FORECAST",
        font=("Segoe UI", 11, "bold"), fg=DIM_FG, bg=BG,
    ).pack(anchor="w", pady=(22, 6))

    forecast = state.get("forecast") or []
    if forecast:
        fcast_frame = tk.Frame(weather_col, bg=BG)
        fcast_frame.pack(anchor="w", fill="x")
        for f in forecast:
            row = tk.Frame(fcast_frame, bg=BG)
            row.pack(anchor="w", pady=4, fill="x")
            tk.Label(
                row, text=f.get("emoji", "\U0001F321"),
                font=("Segoe UI Emoji", 22), fg=TEXT_FG, bg=BG, width=2,
            ).pack(side="left")
            tk.Label(
                row, text=f.get("label", "")[:12],
                font=("Segoe UI", 14, "bold"), fg=TEXT_FG, bg=BG,
                width=10, anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=f"{f.get('high_c', '?')}° / {f.get('low_c', '?')}°",
                font=("Segoe UI", 14), fg=TEXT_FG, bg=BG, width=10, anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=(f.get("desc", "") or "").title()[:24],
                font=("Segoe UI", 13), fg=DIM_FG, bg=BG, anchor="w",
            ).pack(side="left", padx=(8, 0))
    else:
        tk.Label(
            weather_col, text="Forecast unavailable",
            font=("Segoe UI", 13), fg=DIM_FG, bg=BG,
        ).pack(anchor="w")

    # ── RIGHT: Calendar (top) + Bambu (bottom) ──
    right_col = tk.Frame(body, bg=BG)
    right_col.grid(row=0, column=1, sticky="nsew", padx=(16, 0))

    upcoming_header = tk.Frame(right_col, bg=BG)
    upcoming_header.pack(anchor="w", fill="x", pady=(0, 6))
    tk.Label(
        upcoming_header, text="UPCOMING",
        font=("Segoe UI", 11, "bold"), fg=DIM_FG, bg=BG,
    ).pack(side="left")
    unread = state.get("unread_mail")
    if isinstance(unread, int) and unread > 0:
        tk.Label(
            upcoming_header,
            text=f"  •  {unread} unread email{'s' if unread != 1 else ''}",
            font=("Segoe UI", 11, "bold"), fg=GOLD, bg=BG,
        ).pack(side="left")
    calendar = state.get("calendar") or []
    if calendar:
        for c in calendar[:3]:
            row = tk.Frame(right_col, bg=BG)
            row.pack(anchor="w", pady=3, fill="x")
            tk.Label(
                row, text=c.get("time", ""),
                font=("Segoe UI", 14, "bold"), fg=GOLD, bg=BG,
                width=15, anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=(c.get("subject", "") or "")[:48],
                font=("Segoe UI", 14), fg=TEXT_FG, bg=BG, anchor="w",
            ).pack(side="left")
    else:
        tk.Label(
            right_col, text="No upcoming appointments",
            font=("Segoe UI", 14), fg=DIM_FG, bg=BG,
        ).pack(anchor="w", pady=4)

    tk.Label(
        right_col, text="BAMBU H2D",
        font=("Segoe UI", 11, "bold"), fg=DIM_FG, bg=BG,
    ).pack(anchor="w", pady=(24, 6))

    bambu = state.get("bambu")
    if bambu:
        status = (bambu.get("status") or "").upper()
        fname = bambu.get("filename") or "(unnamed)"
        pct = max(0, min(100, int(bambu.get("percent") or 0)))
        layer = int(bambu.get("layer") or 0)
        total = int(bambu.get("total_layers") or 0)
        rem = int(bambu.get("remaining_minutes") or 0)

        if status == "FAILED":
            scol = ALERT
        elif status == "FINISH":
            scol = GREEN
        else:
            scol = GOLD

        tk.Label(
            right_col, text=f"{status}  •  {fname[:42]}",
            font=("Segoe UI", 14, "bold"), fg=scol, bg=BG, anchor="w",
        ).pack(anchor="w")

        bar_w = 30
        filled = int(round(bar_w * (pct / 100.0)))
        filled = max(0, min(bar_w, filled))
        bar = "█" * filled + "░" * (bar_w - filled)
        tk.Label(
            right_col, text=f"{bar}  {pct}%",
            font=("Consolas", 13), fg=TEXT_FG, bg=BG, anchor="w",
        ).pack(anchor="w", pady=(4, 4))

        details = []
        if layer and total:
            details.append(f"Layer {layer} / {total}")
        if rem > 0:
            hrs, mins = divmod(rem, 60)
            if hrs > 0:
                details.append(f"{hrs}h {mins}m remaining")
            else:
                details.append(f"{mins}m remaining")
        if details:
            tk.Label(
                right_col, text="  •  ".join(details),
                font=("Segoe UI", 13), fg=DIM_FG, bg=BG, anchor="w",
            ).pack(anchor="w")
    else:
        tk.Label(
            right_col, text="No active print",
            font=("Segoe UI", 14), fg=DIM_FG, bg=BG,
        ).pack(anchor="w", pady=4)

    # Bottom hint with live countdown
    hint = tk.Label(
        inner,
        text="",
        font=("Segoe UI", 11), fg=DIM_FG, bg=BG,
    )
    hint.pack(side="bottom", pady=12)

    def _tick():
        try:
            cur = _load_state_safe()
            now = time.time()
            if cur is None or cur.get("dismissed"):
                root.destroy()
                return
            expiry = float(cur.get("expiry_ts", 0.0))
            if now >= expiry:
                root.destroy()
                return
            if parent_pid > 0 and not _pid_alive(parent_pid):
                root.destroy()
                return
            rem = max(0, int(expiry - now))
            hint.config(
                text=f"Auto-dismiss in {rem}s  •  "
                     f"say 'thank you, JARVIS' to close"
            )
            root.after(250, _tick)
        except Exception:
            try:
                root.destroy()
            except Exception:
                pass

    root.after(50, _tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.remove(_PID_FILE)
        except Exception:
            pass
    return 0


def _load_state_safe() -> Optional[dict]:
    if not os.path.exists(_STATE_FILE):
        return None
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS briefing card overlay")
    parser.add_argument("--render", action="store_true",
                        help="Renderer subprocess mode (read state, draw card)")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="Exit when this PID dies")
    parser.add_argument("--demo", choices=["morning", "evening", "status"],
                        help="Show a demo card for 20 s (manual test)")
    args = parser.parse_args()

    if args.demo:
        show_card(args.demo)
        # Keep the parent alive so the spawned renderer can read the state
        # file and finish rendering before this process exits.
        while is_card_active():
            time.sleep(0.5)
        sys.exit(0)
    elif args.render:
        sys.exit(_renderer_main(args.parent_pid))
    else:
        parser.print_help()
