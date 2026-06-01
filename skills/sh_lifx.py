"""
sh_lifx — LIFX controller skill.

Wraps `lifxlan` so `core.smart_home_router` can talk to LIFX bulbs over
the LAN protocol (UDP 56700) without going through Alexa or the LIFX
cloud.

LIFX devices announce themselves on the LAN, so there's no configuration
required beyond `pip install lifxlan`. Discovery is broadcast-based and
cached for `_DISCOVERY_TTL` seconds to avoid hammering the network on
every voice command.

Uniform API: list_devices / get_state / set_state. set_state kwargs
match the rest of the smart_home stack:
    on:                bool
    brightness:        0..100
    color:             (r, g, b)
    color_temperature: kelvin (LIFX native; 2500..9000 typical range)

All functions degrade gracefully if `lifxlan` isn't installed.
"""
from __future__ import annotations

import threading
import time
from typing import Any


_DISCOVERY_TTL = 30.0
_lock = threading.Lock()
_state: dict[str, Any] = {"by_name": {}, "by_mac": {}, "fetched_at": 0.0}


_DISCOVERY_TIMEOUT = 3.0


def _lifxlan():
    try:
        import lifxlan  # type: ignore
        return lifxlan
    except Exception:
        return None


def is_available() -> bool:
    return _lifxlan() is not None


def _discover_bulbs(lifxlan: Any, timeout: float = _DISCOVERY_TIMEOUT):
    """Run LifxLAN().get_lights() (full UDP broadcast discovery, 2-5s) in a
    daemon thread with a hard timeout. Returns the bulb list, or None if the
    discovery is still running when the timeout fires — the caller then keeps
    its cached bulbs. Without a device-count hint, get_lights() blocks on a
    fixed broadcast wait, so bounding it here keeps it off the voice thread."""
    result: dict[str, Any] = {"bulbs": None}

    def _worker() -> None:
        try:
            lan = lifxlan.LifxLAN()
            result["bulbs"] = lan.get_lights() or []
        except Exception as e:
            print(f"  [sh-lifx] discover failed: {e}")
            result["bulbs"] = []

    t = threading.Thread(target=_worker, name="sh_lifx-discover", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return result["bulbs"]


def _refresh(force: bool = False) -> dict[str, Any]:
    with _lock:
        if not force and (time.monotonic() - _state["fetched_at"]) < _DISCOVERY_TTL:
            return dict(_state["by_name"])
    lifxlan = _lifxlan()
    if lifxlan is None:
        return {}
    bulbs = _discover_bulbs(lifxlan)
    if bulbs is None:
        # Discovery exceeded the timeout — keep whatever we already had so a
        # slow LAN broadcast can't stall the voice turn.
        print(f"  [sh-lifx] discovery timed out after {_DISCOVERY_TIMEOUT:.0f}s "
              "— using cached bulbs.")
        with _lock:
            return dict(_state["by_name"])
    by_name: dict[str, Any] = {}
    by_mac: dict[str, Any] = {}
    for b in bulbs:
        try:
            label = (b.get_label() or "").strip()
            mac = (b.get_mac_addr() or "").strip().lower()
            if label:
                by_name[label.lower()] = b
            if mac:
                by_mac[mac] = b
        except Exception:
            continue
    with _lock:
        _state["by_name"]   = by_name
        _state["by_mac"]    = by_mac
        _state["fetched_at"] = time.monotonic()
    return by_name


def _bulb_for(device: dict) -> Any:
    name = (device.get("name") or "").strip().lower()
    mac  = (device.get("lan_mac") or "").strip().lower()
    _refresh()
    with _lock:
        if mac and mac in _state["by_mac"]:
            return _state["by_mac"][mac]
        if name and name in _state["by_name"]:
            return _state["by_name"][name]
    return None


# ── color helpers ─────────────────────────────────────────────────
def _rgb_to_hsbk(rgb, kelvin: int = 3500) -> tuple[int, int, int, int]:
    """Convert sRGB → LIFX HSBK (each channel 0..65535 except kelvin).

    H: 0..65535 = 0..360°
    S: 0..65535 = 0..1
    B: 0..65535 = 0..1
    K: kelvin
    """
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
    s = 0 if mx == 0 else df / mx
    v = mx
    return (
        int(h / 360 * 65535),
        int(s * 65535),
        int(v * 65535),
        int(kelvin),
    )


# ── public API ────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    bulbs = _refresh()
    out: list[dict] = []
    for name, b in bulbs.items():
        try:
            caps = ["on_off", "dim", "color_temperature"]
            if getattr(b, "supports_color", lambda: False)():
                caps.append("color")
        except Exception:
            caps = ["on_off", "dim"]
        out.append({
            "name": b.get_label(),
            "brand": "LIFX",
            "type": "light",
            "capabilities": sorted(set(caps)),
            "lan_mac": b.get_mac_addr(),
        })
    return out


def get_state(device: dict) -> dict:
    b = _bulb_for(device)
    if b is None:
        return {"error": f"lifx bulb '{device.get('name')}' not found"}
    try:
        power = b.get_power()
        h, s, br, k = b.get_color()
        return {
            "on":         bool(power),
            "brightness": int(round(br / 65535 * 100)),
            "color_temperature_k": int(k),
        }
    except Exception as e:
        return {"error": f"lifx state read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    b = _bulb_for(device)
    if b is None:
        return {"error": f"lifx bulb '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}
    try:
        if "on" in kwargs and kwargs["on"] is not None:
            b.set_power("on" if kwargs["on"] else "off")
            applied["on"] = bool(kwargs["on"])
        if "color" in kwargs and kwargs["color"]:
            hsbk = _rgb_to_hsbk(kwargs["color"],
                                kelvin=int(kwargs.get("color_temperature") or 3500))
            b.set_color(hsbk)
            applied["color"] = list(kwargs["color"])
        elif "brightness" in kwargs and kwargs["brightness"] is not None:
            pct = max(0, min(100, int(kwargs["brightness"])))
            try:
                h, s, _, k = b.get_color()
            except Exception:
                h, s, k = 0, 0, 3500
            b.set_color((h, s, int(pct / 100 * 65535), k))
            applied["brightness"] = pct
            if pct > 0 and not applied.get("on", False):
                b.set_power("on")
                applied["on"] = True
        elif "color_temperature" in kwargs and kwargs["color_temperature"]:
            try:
                h, s, br, _ = b.get_color()
            except Exception:
                h, s, br = 0, 0, 32768
            b.set_color((h, 0, br, int(kwargs["color_temperature"])))
            applied["color_temperature_k"] = int(kwargs["color_temperature"])
    except Exception as e:
        return {"error": f"lifx set_state failed: {e}", "partial": applied}

    return {"ok": True, "applied": applied}


def lifx_list(_: str = "") -> str:
    devs = list_devices()
    if not devs:
        return ("No LIFX bulbs found on the LAN, sir — UDP 56700 broadcasts "
                "may be blocked.")
    names = [d["name"] for d in devs]
    return f"{len(names)} LIFX bulb(s): " + ", ".join(names[:10]) + (
        " (+more)" if len(names) > 10 else ""
    )


def register(actions: dict) -> None:
    actions["lifx_list_devices"] = lifx_list
    actions["lifx_list"]         = lifx_list
