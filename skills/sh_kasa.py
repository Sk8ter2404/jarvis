"""
sh_kasa — TP-Link Kasa / Tapo controller skill.

Wraps the `python-kasa` library so `core.smart_home_router` can dispatch
to TP-Link smart plugs, switches and bulbs without going through Alexa.

Tapo devices are also handled by python-kasa via its experimental
SMART protocol — requires the user's TP-Link cloud email + password in
data/sh_kasa_config.json for those devices:
    {"username": "...", "password": "..."}
Pure Kasa devices (older SMARTPLUG protocol) need no credentials.

Discovery: `python-kasa` Discover.discover() broadcasts UDP 9999 on the
LAN and returns a {ip → SmartDevice} dict. Cached for 30s. If the user's
device catalog already carries a `lan_ip`, we go direct rather than
broadcast.

All public functions degrade gracefully — if `python-kasa` isn't
installed, they return informative error dicts.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_CONFIG_PATH = os.path.join(_DATA_DIR, "sh_kasa_config.json")

_DISCOVERY_TTL = 30.0
_lock = threading.Lock()
_state: dict[str, Any] = {"by_ip": {}, "by_name": {}, "fetched_at": 0.0}


# ── dep import ─────────────────────────────────────────────────────
def _kasa():
    try:
        import kasa  # type: ignore
        return kasa
    except Exception:
        return None


def is_available() -> bool:
    return _kasa() is not None


# ── async runner ──────────────────────────────────────────────────
def _run_async(coro):
    """Run `coro` to completion, even if the calling thread is itself in
    an event loop (delegated to a worker thread in that case)."""
    try:
        asyncio.get_running_loop()
        nested = True
    except RuntimeError:
        nested = False
    if not nested:
        return asyncio.run(coro)
    box: dict = {}
    def _go() -> None:
        try:
            box["v"] = asyncio.run(coro)
        except Exception as e:
            box["err"] = e
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join()
    if "err" in box:
        raise box["err"]
    return box.get("v")


# ── config ────────────────────────────────────────────────────────
def _read_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# ── discovery ─────────────────────────────────────────────────────
async def _discover_async() -> dict[str, Any]:
    kasa = _kasa()
    if kasa is None:
        return {}
    Discover = getattr(kasa, "Discover", None)
    if Discover is None or not hasattr(Discover, "discover"):
        return {}
    cfg = _read_config()
    kwargs: dict[str, Any] = {"timeout": 5}
    # python-kasa ≥0.6 uses Credentials() for Tapo cloud auth.
    if cfg.get("username") and cfg.get("password"):
        try:
            Credentials = getattr(kasa, "Credentials", None)
            if Credentials is not None:
                kwargs["credentials"] = Credentials(cfg["username"], cfg["password"])
        except Exception:
            pass
    try:
        return await Discover.discover(**kwargs)
    except TypeError:
        try:
            return await Discover.discover()
        except Exception as e:
            print(f"  [sh-kasa] discover failed: {e}")
            return {}
    except Exception as e:
        print(f"  [sh-kasa] discover failed: {e}")
        return {}


def _refresh_discovery(force: bool = False) -> dict[str, Any]:
    with _lock:
        if not force and (time.monotonic() - _state["fetched_at"]) < _DISCOVERY_TTL:
            return dict(_state["by_ip"])
    try:
        found = _run_async(_discover_async()) or {}
    except Exception as e:
        print(f"  [sh-kasa] discovery error: {e}")
        found = {}
    by_name: dict[str, Any] = {}
    for ip, dev in found.items():
        try:
            alias = (getattr(dev, "alias", None) or "").strip()
            if alias:
                by_name[alias.lower()] = dev
        except Exception:
            pass
    with _lock:
        _state["by_ip"]     = dict(found)
        _state["by_name"]   = by_name
        _state["fetched_at"] = time.monotonic()
    return found


async def _device_from_ip_async(ip: str) -> Any:
    """Connect directly to a known IP — avoids the broadcast latency."""
    kasa = _kasa()
    if kasa is None or not ip:
        return None
    # python-kasa ≥0.6 prefers `Discover.discover_single(host=ip)` because
    # it autodetects protocol; fall back to SmartPlug for old kasa releases.
    Discover = getattr(kasa, "Discover", None)
    if Discover is not None and hasattr(Discover, "discover_single"):
        try:
            return await Discover.discover_single(ip)
        except Exception:
            pass
    SmartPlug = getattr(kasa, "SmartPlug", None)
    if SmartPlug is not None:
        try:
            dev = SmartPlug(ip)
            await dev.update()
            return dev
        except Exception:
            return None
    return None


def _device_for(device_record: dict) -> Any:
    """Resolve a catalog record → a live python-kasa device handle."""
    ip = (device_record.get("lan_ip") or "").strip()
    name = (device_record.get("name") or "").strip().lower()
    # Direct by IP when possible.
    if ip:
        try:
            dev = _run_async(_device_from_ip_async(ip))
            if dev is not None:
                return dev
        except Exception:
            pass
    # Otherwise scan and look up by alias.
    _refresh_discovery()
    with _lock:
        dev = _state["by_name"].get(name)
    return dev


# ── public API ────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    found = _refresh_discovery()
    out: list[dict] = []
    for ip, dev in found.items():
        try:
            alias = getattr(dev, "alias", None) or ""
            model = getattr(dev, "model", None) or ""
            caps = ["on_off"]
            if getattr(dev, "is_dimmable", False):
                caps.append("dim")
            if getattr(dev, "is_color", False):
                caps.append("color")
            if getattr(dev, "is_variable_color_temp", False):
                caps.append("color_temperature")
            dtype = "plug"
            if getattr(dev, "is_bulb", False):
                dtype = "light"
            elif getattr(dev, "is_strip", False):
                dtype = "strip"
            elif getattr(dev, "is_dimmer", False):
                dtype = "dimmer"
            out.append({
                "name": alias,
                "brand": "TP-Link",
                "model": model,
                "type":  dtype,
                "capabilities": sorted(set(caps)),
                "lan_ip": ip,
            })
        except Exception:
            continue
    return out


def get_state(device: dict) -> dict:
    dev = _device_for(device)
    if dev is None:
        return {"error": f"kasa device '{device.get('name')}' not found"}
    try:
        _run_async(dev.update())
        return {
            "on":          bool(getattr(dev, "is_on", False)),
            "brightness":  int(getattr(dev, "brightness", 0) or 0) if getattr(dev, "is_dimmable", False) else None,
            "alias":       getattr(dev, "alias", ""),
            "model":       getattr(dev, "model", ""),
        }
    except Exception as e:
        return {"error": f"kasa state read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    dev = _device_for(device)
    if dev is None:
        return {"error": f"kasa device '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}

    async def _apply() -> None:
        await dev.update()
        if "on" in kwargs and kwargs["on"] is not None:
            if kwargs["on"]:
                await dev.turn_on()
                applied["on"] = True
            else:
                await dev.turn_off()
                applied["on"] = False
        if "brightness" in kwargs and kwargs["brightness"] is not None:
            pct = max(0, min(100, int(kwargs["brightness"])))
            if getattr(dev, "is_dimmable", False):
                try:
                    await dev.set_brightness(pct)
                    applied["brightness"] = pct
                except Exception:
                    pass
            if pct > 0 and not applied.get("on", False):
                try:
                    await dev.turn_on()
                    applied["on"] = True
                except Exception:
                    pass
        if "color_temperature" in kwargs and kwargs["color_temperature"]:
            if getattr(dev, "is_variable_color_temp", False):
                try:
                    await dev.set_color_temp(int(kwargs["color_temperature"]))
                    applied["color_temperature_k"] = int(kwargs["color_temperature"])
                except Exception:
                    pass
        if "color" in kwargs and kwargs["color"]:
            if getattr(dev, "is_color", False):
                try:
                    h, s, v = _rgb_to_hsv(kwargs["color"])
                    await dev.set_hsv(h, s, v)
                    applied["color"] = list(kwargs["color"])
                except Exception:
                    pass

    try:
        _run_async(_apply())
    except Exception as e:
        return {"error": f"kasa set_state failed: {e}", "partial": applied}

    return {"ok": True, "applied": applied}


def _rgb_to_hsv(rgb) -> tuple[int, int, int]:
    """Kasa bulbs expect HSV with H 0..360, S 0..100, V 0..100."""
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if df == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    else:
        h = (60 * ((r - g) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx * 100
    v = mx * 100
    return (int(h), int(s), int(v))


def kasa_list(_: str = "") -> str:
    devs = list_devices()
    if not devs:
        return ("No Kasa/Tapo devices discovered on the LAN, sir. "
                "Check that UDP 9999 broadcasts aren't blocked.")
    names = [d["name"] or d.get("lan_ip", "?") for d in devs]
    return f"{len(names)} Kasa device(s): " + ", ".join(names[:10]) + (
        " (+more)" if len(names) > 10 else ""
    )


def _tuya_mod():
    """Locate the loaded sh_tuya skill module (name varies by loader) so the
    unified control below can drive Tuya devices too. None if not loaded."""
    import sys
    import importlib
    for nm in ("skill_sh_tuya", "sh_tuya", "skills.sh_tuya"):
        m = sys.modules.get(nm)
        if m is not None:
            return m
    for nm in ("sh_tuya", "skills.sh_tuya"):
        try:
            return importlib.import_module(nm)
        except Exception:
            pass
    return None


def smart_home_control(request: str = "") -> str:
    """Voice control for the LAN smart plugs: 'turn on the entry light',
    'turn off dining room', 'toggle kitchen 2', 'are the lights on?'.

    Parses on/off/toggle + the device name out of the request, matches it
    against the live Kasa discovery, and drives it directly over the LAN — no
    Amazon/Alexa needed. 2026-05-30 (added after Amazon locked down the Alexa
    cookie API; these TP-Link Kasa plugs are controlled locally instead)."""
    import re as _re
    req = (request or "").strip()
    if not req:
        return "What would you like me to control, sir?"
    low = req.lower()
    # Intent — word-boundary so 'office' isn't read as 'off', 'nook' not 'on'.
    # INTERROGATIVE GUARD (2026-07-07 bug-hunt): a STATUS QUESTION like "are the
    # lights on" contains the word "on" and would otherwise be read as an ON
    # command that actually SWITCHES the plug. A leading question word or a
    # trailing '?' → status query (intent=None), which reads live state below.
    if (_re.match(r"^\s*(?:are|is|was|were|do|does|did|has|have|can|could|"
                  r"what'?s|how'?s)\b", low)
            or low.rstrip().endswith("?")):
        intent = None
    elif _re.search(r"\btoggle\b", low):
        intent = "toggle"
    elif _re.search(r"\b(off|shut)\b", low):
        intent = "off"
    elif _re.search(r"\b(on|enable)\b", low):
        intent = "on"
    else:
        intent = None  # status query

    # Combined device list — Kasa (live LAN discovery) + Tuya (catalog).
    devs = []
    try:
        for d in list_devices():
            d["_ctl"] = "kasa"
            devs.append(d)
    except Exception:
        pass
    tmod = _tuya_mod()
    if tmod is not None:
        try:
            for d in tmod.list_devices():
                d["_ctl"] = "tuya"
                devs.append(d)
        except Exception:
            pass
    if not devs:
        return ("I don't see any controllable smart devices on the network "
                "yet, sir.")

    # Match the device whose name appears in the request; else best word-overlap.
    def _clean(s):
        return (s or "").strip()
    matches = [d for d in devs if _clean(d.get("name")).lower()
               and _clean(d.get("name")).lower() in low]
    if not matches:
        req_words = set(_re.findall(r"[a-z0-9]+", low))
        best, best_score = None, 0
        for d in devs:
            nm_words = set(_re.findall(r"[a-z0-9]+", _clean(d.get("name")).lower()))
            score = len(req_words & nm_words)
            if score > best_score:
                best, best_score = d, score
        if best and best_score > 0:
            matches = [best]
    # 'all'/'everything' → every device.
    if _re.search(r"\b(all|everything|every)\b", low):
        matches = devs
    if not matches:
        names = ", ".join(_clean(d.get("name")) for d in devs)
        return (f"I'm not sure which one you meant, sir. I can control: {names}.")

    out = []
    for d in matches:
        rec = {"name": _clean(d.get("name")), "lan_ip": d.get("lan_ip"),
               "_tuya": d.get("_tuya")}
        nm = rec["name"]
        # Route to the right controller (Kasa local API vs Tuya/tinytuya).
        if d.get("_ctl") == "tuya" and tmod is not None:
            _set, _get = tmod.set_state, tmod.get_state
        else:
            _set, _get = set_state, get_state
        if intent in ("on", "off"):
            r = _set(rec, on=(intent == "on"))
            out.append(f"{nm} {intent}" if r.get("ok") else f"{nm} (failed)")
        elif intent == "toggle":
            st = _get(rec)
            new = not bool(st.get("on"))
            r = _set(rec, on=new)
            out.append(f"{nm} {'on' if new else 'off'}" if r.get("ok")
                       else f"{nm} (failed)")
        else:
            st = _get(rec)
            out.append(f"{nm} is {'on' if st.get('on') else 'off'}")
    if intent is None:
        return "Status, sir — " + "; ".join(out) + "."
    return "Done, sir — " + "; ".join(out) + "."


def register(actions: dict) -> None:
    actions["kasa_list_devices"] = kasa_list
    actions["kasa_list"]         = kasa_list
    actions["tplink_list"]       = kasa_list
    # Voice control for the discovered plugs (no Alexa needed).
    actions["smart_home_control"] = smart_home_control
    actions["kasa_control"]       = smart_home_control
    actions["control_device"]     = smart_home_control
    actions["control_plug"]       = smart_home_control
    actions["control_light"]      = smart_home_control
