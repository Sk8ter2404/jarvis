"""
JARVIS holographic globe skill — put a rotating wireframe Earth on any
monitor and drop pins on it.

The renderer is hud/globe_hud.py (tkinter), run as its own subprocess with
the same launch/stop contract as the holographic overlays in
skills/holographic_overlay: --x/--y/--width/--height/--parent-pid,
CREATE_NO_WINDOW, a dedicated control file the HUD polls, and
terminate → wait → kill on dismissal. Nothing launches at boot; the globe
only appears when asked for.

Actions:
  show_globe [| monitor]      → open the globe (default: the HUD monitor);
                                 naming another monitor while it is up moves
                                 it there, pins and all.
  hide_globe                  → close it.
  globe_pin <place> [| label] → pin a city from the bundled table (~260
                                 major cities and capitals, Natural Earth) or
                                 'lat, lon'; opens the globe if needed and
                                 turns it to face the pin. Consecutive pins
                                 are joined by great-circle arcs.
  globe_clear                 → remove every pin.

Control file: globe_hud_state.json at the project root —
  {"mode": "on"|"off", "pins": [{"lat", "lon", "label"}], "seq": n}.
"seq" bumps on every pin so the HUD re-focuses even on a repeated city.

tkinter is never imported here: the skill only shells out to the HUD.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import unicodedata

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GLOBE_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "globe_hud.py")
_CONTROL_FILE = os.path.join(_PROJECT_DIR, "globe_hud_state.json")
# Made with Natural Earth - public domain (see the file's "_source").
_CITIES_FILE = os.path.join(_PROJECT_DIR, "hud", "data", "world_cities.json")

_GLOBE_PROCESS = None
_GLOBE_MONITOR = None        # MONITORS key the live globe is on
_GLOBE_LOCK = threading.Lock()

# The globe window is a square this fraction of the monitor's short side,
# centred on the monitor.
_GLOBE_SIZE_FRAC = 0.62
# Matches hud/globe_hud.py MAX_PINS: the oldest pin drops off past this.
_MAX_PINS = 12

_CITIES = None               # [(name, country, lat, lon)], largest first
_CITY_INDEX = None           # folded name / alias / "name country" → row


# ─── monitors ─────────────────────────────────────────────────────────────

def _monitors() -> dict:
    try:
        from core.config import MONITORS
        return dict(MONITORS)
    except Exception:
        return {}


def _default_monitor(monitors: dict):
    """The HUD's monitor (core.config.HUD_MONITOR), else the primary (the
    one at the origin), else the first configured."""
    try:
        from core.config import HUD_MONITOR
    except Exception:
        HUD_MONITOR = "top"
    if HUD_MONITOR in monitors:
        return HUD_MONITOR
    for key, rect in monitors.items():
        if tuple(rect[:2]) == (0, 0):
            return key
    return next(iter(monitors), None)


def _resolve_monitor(name: str):
    """MONITORS key for a spoken monitor name — core.actions' resolver, so
    'the left monitor' / 'main' mean what they mean everywhere else."""
    try:
        from core.actions import _resolve_monitor as resolve
        return resolve(name)
    except Exception:
        s = (name or "").strip().lower()
        return s if s in _monitors() else None


def _globe_rect(rect) -> tuple:
    """(x, y, w, h) of the square globe window centred on a monitor rect."""
    mx, my, mw, mh = (int(v) for v in list(rect)[:4])
    side = max(200, int(min(mw, mh) * _GLOBE_SIZE_FRAC))
    return mx + (mw - side) // 2, my + (mh - side) // 2, side, side


# ─── control file ─────────────────────────────────────────────────────────

def _read_control() -> dict:
    try:
        with open(_CONTROL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_control(**updates) -> None:
    """Atomic-write the control file the HUD polls (merge, then replace)."""
    try:
        data = _read_control()
        data.update(updates)
        tmp = _CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _CONTROL_FILE)
    except Exception:
        # Best-effort — terminate() is the fallback for "off".
        pass


# ─── process lifecycle ────────────────────────────────────────────────────

def _spawn(argv: list):
    """Start the HUD subprocess. The one seam tests replace."""
    return subprocess.Popen(
        argv,
        creationflags=(subprocess.CREATE_NO_WINDOW
                       if sys.platform == "win32" else 0),
        close_fds=True,
    )


def _globe_is_alive() -> bool:
    proc = _GLOBE_PROCESS
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def _stop_locked() -> None:
    """Terminate the live globe (caller holds _GLOBE_LOCK)."""
    global _GLOBE_PROCESS, _GLOBE_MONITOR
    proc = _GLOBE_PROCESS
    _GLOBE_PROCESS = None
    _GLOBE_MONITOR = None
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            # terminate() didn't take — kill, then ALWAYS wait so the OS
            # handle is released (see holographic_overlay._shutdown_overlay).
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.1)
            except Exception:
                pass
    except Exception:
        pass


def _launch(monitor: str) -> tuple:
    """Show the globe on MONITORS[monitor]. Returns (ok, message). Idempotent
    on the same monitor; on another monitor it moves, keeping the pins."""
    global _GLOBE_PROCESS, _GLOBE_MONITOR
    monitors = _monitors()
    rect = monitors.get(monitor)
    if rect is None:
        return False, "REFUSED: I've no monitor layout to put the globe on, sir."
    with _GLOBE_LOCK:
        moving = _globe_is_alive()
        if moving and _GLOBE_MONITOR == monitor:
            return True, "The globe is already up, sir."
        if moving:
            _stop_locked()
        if not os.path.exists(_GLOBE_SCRIPT):
            return False, "REFUSED: the globe renderer is missing, sir."
        # A fresh globe starts clean (a stale file may hold a crashed
        # session's pins); a move keeps them. Either way clear any "off".
        if moving:
            _write_control(mode="on")
        else:
            _write_control(mode="on", pins=[])
        x, y, w, h = _globe_rect(rect)
        try:
            _GLOBE_PROCESS = _spawn(
                [sys.executable, _GLOBE_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())])
        except Exception as e:
            _GLOBE_PROCESS = None
            return False, f"I couldn't start the globe, sir — {e}."
        _GLOBE_MONITOR = monitor
        if moving:
            return True, f"Globe moved to the {monitor} monitor, sir."
        return True, f"Globe up on the {monitor} monitor, sir."


# ─── places ───────────────────────────────────────────────────────────────

# Letters NFKD does not decompose into base + accent.
_FOLD_EXTRA = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ł": "l",
                             "đ": "d", "ð": "d", "þ": "th", "ı": "i"})


def _fold(text: str) -> str:
    """Case-, accent- and punctuation-insensitive form of a place name:
    'São Paulo' / 'sao paulo' / 'SAO-PAULO' → 'sao paulo', and
    'Washington, D.C.' / 'washington dc' → 'washington dc'."""
    s = unicodedata.normalize("NFKD", str(text or "")).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(_FOLD_EXTRA)
    s = re.sub(r"[.'’]", "", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _load_cities() -> list:
    global _CITIES, _CITY_INDEX
    if _CITIES is not None:
        return _CITIES
    rows, index = [], {}
    try:
        with open(_CITIES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f).get("cities") or []
    except Exception:
        raw = []
    for r in raw:
        try:
            row = (str(r[0]), str(r[1]), float(r[2]), float(r[3]))
        except (IndexError, TypeError, ValueError):
            continue
        rows.append(row)
        aliases = r[4] if len(r) > 4 and isinstance(r[4], list) else []
        for key in [row[0], *aliases, f"{row[0]} {row[1]}"]:
            # Rows are largest-first, so a shared name keeps the bigger city.
            index.setdefault(_fold(key), row)
    _CITIES, _CITY_INDEX = rows, index
    return rows


def _find_city(query: str):
    """The (name, country, lat, lon) row for a spoken place, or None.
    Tolerates case, accents, punctuation, a trailing country ('Paris,
    France'), a leading 'the' and a trailing 'city' ('New York City')."""
    _load_cities()
    q = _fold(query)
    if not q:
        return None
    tries = [q]
    if q.startswith("the "):
        tries.append(q[4:])
    tries += [t[:-5] for t in list(tries) if t.endswith(" city")]
    for t in tries:
        row = _CITY_INDEX.get(t)
        if row is not None:
            return row
    return None


_LATLON_RE = re.compile(
    r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*°?\s*([NS])?\s*[,\s]\s*"
    r"(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([EW])?\s*$", re.IGNORECASE)


def _parse_latlon(text: str):
    """'35.7, 139.7' / '33.9S 18.4E' → (lat, lon), or None."""
    m = _LATLON_RE.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(3))
    if (m.group(2) or "").upper() == "S":
        lat = -abs(lat)
    if (m.group(4) or "").upper() == "W":
        lon = -abs(lon)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


# ─── actions ──────────────────────────────────────────────────────────────

def show_globe(args: str = "") -> str:
    """show_globe [| monitor]"""
    asked = (args or "").strip().strip("|").strip()
    monitors = _monitors()
    if asked:
        monitor = _resolve_monitor(asked)
        if monitor is None:
            return (f"Unknown monitor '{asked}', sir — I have "
                    f"{', '.join(monitors) or 'none configured'}.")
    else:
        if _globe_is_alive():
            return "The globe is already up, sir."
        monitor = _default_monitor(monitors)
    _ok, msg = _launch(monitor)
    return msg


def hide_globe(_: str = "") -> str:
    with _GLOBE_LOCK:
        # Belt-and-braces: the control file asks a globe this process lost
        # track of to close too; terminate() covers a missed poll.
        _write_control(mode="off")
        if not _globe_is_alive():
            _stop_locked()
            return "The globe isn't up, sir."
        _stop_locked()
    return "Globe dismissed, sir."


def globe_pin(args: str = "") -> str:
    """globe_pin <city or 'lat, lon'> [| label]"""
    place, _, label = (args or "").partition("|")
    place, label = place.strip(), label.strip()
    if not place:
        return "format: globe_pin, <city or 'lat, lon'> [| label]"
    coords = _parse_latlon(place)
    if coords is not None:
        lat, lon = coords
        name = label or f"{lat:.1f}, {lon:.1f}"
    else:
        row = _find_city(place)
        if row is None:
            return (f"I couldn't place '{place}', sir — I only know "
                    f"{len(_load_cities())} major cities and capitals. Try a "
                    f"bigger one nearby, or give me 'latitude, longitude'.")
        name, _country, lat, lon = row
        name = label or name
    if not _globe_is_alive():
        ok, msg = _launch(_default_monitor(_monitors()))
        if not ok:
            return msg
    data = _read_control()
    pins = [p for p in data.get("pins") or []
            if isinstance(p, dict)
            and (p.get("lat"), p.get("lon")) != (lat, lon)]
    pins.append({"lat": lat, "lon": lon, "label": name})
    try:
        seq = int(data.get("seq") or 0) + 1
    except (TypeError, ValueError):
        seq = 1
    _write_control(mode="on", pins=pins[-_MAX_PINS:], seq=seq)
    return f"Pinned {name}, sir."


def globe_clear(_: str = "") -> str:
    _write_control(pins=[])
    if not _globe_is_alive():
        return "The globe isn't up, sir."
    return "Pins cleared, sir."


def register(actions: dict):
    actions["show_globe"] = show_globe
    actions["hide_globe"] = hide_globe
    actions["globe_pin"] = globe_pin
    actions["globe_clear"] = globe_clear
