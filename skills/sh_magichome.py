"""Magic Home / Flux LED controller skill (LAN only, no cloud).

Drives Magic Home (aka Flux LED / "LEDnet") Wi-Fi LED strip controllers
directly over the local network using the ``flux_led`` library (UDP 48899
discovery, TCP 5577 control). The smart-home router calls ``set_state`` first
and falls back to Alexa when it returns ``{"error": ...}``, so this skill only
reports success when a follow-up status read matches what was asked for.

Public contract (same shape as sh_kasa / sh_govee):
  * ``list_devices()``  -> [{name, brand, type, capabilities, lan_ip, model}]
  * ``set_state(device, on=, brightness=, color=(r,g,b), ...)``
  * ``get_state(device)``
  * ``is_available()``
Nothing here ever raises out of a public function.
"""
from __future__ import annotations

import os
import time
from typing import Any

_SCAN_TIMEOUT_SEC = 3        # bounded UDP broadcast listen window
_CONNECT_TIMEOUT_SEC = 3.0   # per-device TCP connect/read timeout
_SETTLE_SEC = 0.25           # let the controller apply before the verify read
_COLOR_TOLERANCE = 24        # per-channel slack on the read-back (0-255)
_BRIGHT_TOLERANCE_PCT = 8    # slack on the brightness read-back (percent)


def _enabled() -> bool:
    """MAGICHOME_ENABLED (env or core.config), default ON."""
    env = os.environ.get("MAGICHOME_ENABLED")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        from core import config
        return bool(getattr(config, "MAGICHOME_ENABLED", True))
    except Exception:
        return True


def _flux() -> Any:
    """Import flux_led lazily; None when it is not installed."""
    try:
        import flux_led
        return flux_led
    except Exception:
        return None


def is_available() -> bool:
    return _enabled() and _flux() is not None


# -- discovery ---------------------------------------------------------
def _scan() -> list[dict]:
    """Broadcast-scan the LAN. Returns raw info dicts; [] on any failure."""
    fl = _flux()
    if fl is None:
        return []
    try:
        scanner = fl.BulbScanner()
        found = scanner.scan(timeout=_SCAN_TIMEOUT_SEC) or []
    except Exception:
        return []
    out: list[dict] = []
    for item in found:
        try:
            if isinstance(item, dict):
                ip, ident, model = item.get("ipaddr"), item.get("id"), item.get("model")
            else:
                ip = getattr(item, "ipaddr", None)
                ident = getattr(item, "id", None)
                model = getattr(item, "model", None)
            if ip:
                out.append({"ip": str(ip), "id": str(ident or ""),
                            "model": str(model or "")})
        except Exception:
            continue
    return out


def list_devices() -> list[dict]:
    if not _enabled():
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for info in _scan():
        ip = info["ip"]
        if ip in seen:
            continue
        seen.add(ip)
        tail = (info["id"] or ip.replace(".", ""))[-6:]
        out.append({
            "name": f"Magic Home {tail}",
            "brand": "Magic Home",
            "type": "light",
            "capabilities": ["on_off", "dim", "color"],
            "lan_ip": ip,
            "model": info["model"],
        })
    return out


# -- device access -----------------------------------------------------
def _resolve_ip(device: dict) -> str:
    ip = str((device or {}).get("lan_ip") or "").strip()
    if ip:
        return ip
    # No IP in the catalog: match by name against a fresh scan.
    want = str((device or {}).get("name") or "").strip().lower()
    for d in list_devices():
        if want and d["name"].lower() == want:
            return d["lan_ip"]
    return ""


def _open(ip: str) -> Any:
    fl = _flux()
    if fl is None:
        raise RuntimeError("flux_led is not installed")
    return fl.WifiLedBulb(ip, timeout=_CONNECT_TIMEOUT_SEC)


def _close(bulb: Any) -> None:
    try:
        bulb.close()
    except Exception:
        pass


def _refresh(bulb: Any) -> None:
    bulb.update_state()


def _read(bulb: Any) -> dict:
    raw = getattr(bulb, "rgb_unscaled", None) or getattr(bulb, "rgb", (0, 0, 0))
    rgb = [int(c) for c in raw]
    try:
        pct = round(int(bulb.brightness) * 100 / 255)
    except Exception:
        pct = None
    return {"on": bool(bulb.is_on), "rgb": rgb, "brightness": pct}


def get_state(device: dict) -> dict:
    if not _enabled():
        return {"error": "Magic Home disabled"}
    ip = _resolve_ip(device)
    if not ip:
        return {"error": "Magic Home device has no known IP"}
    bulb = None
    try:
        bulb = _open(ip)
        _refresh(bulb)
        return _read(bulb)
    except Exception as e:
        return {"error": f"Magic Home status read failed: {e}"}
    finally:
        if bulb is not None:
            _close(bulb)


def set_state(device: dict, **kwargs) -> dict:
    if not _enabled():
        return {"error": "Magic Home disabled"}
    if _flux() is None:
        return {"error": "flux_led not installed"}
    if kwargs.get("color_temperature"):
        return {"error": "Magic Home color temperature not supported"}
    ip = _resolve_ip(device)
    if not ip:
        return {"error": "Magic Home device has no known IP"}

    want_on = kwargs.get("on")
    pct = kwargs.get("brightness")
    color = kwargs.get("color")
    if pct is not None:
        pct = max(0, min(100, int(pct)))
    if color:
        color = tuple(max(0, min(255, int(c))) for c in color)
    if want_on is None and pct is None and not color:
        return {"error": "nothing to do"}
    # Brightness 0 means off.
    turning_off = (want_on is False and not color) or pct == 0

    bulb = None
    try:
        bulb = _open(ip)
        _refresh(bulb)
        if turning_off:
            bulb.turnOff()
        else:
            bulb.turnOn()
            if color:
                b255 = None if pct is None else round(pct * 255 / 100)
                bulb.setRgb(*color, brightness=b255)
            elif pct is not None:
                cur = tuple(getattr(bulb, "rgb_unscaled", None)
                            or getattr(bulb, "rgb", (255, 255, 255)))
                if not any(cur):
                    cur = (255, 255, 255)
                bulb.setRgb(*cur, brightness=round(pct * 255 / 100))
        time.sleep(_SETTLE_SEC)
        _refresh(bulb)
        got = _read(bulb)
    except Exception as e:
        return {"error": f"Magic Home command failed: {e}"}
    finally:
        if bulb is not None:
            _close(bulb)

    # Honest verification against the read-back.
    applied: dict[str, Any] = {}
    if turning_off:
        if got["on"]:
            return {"error": "Magic Home did not turn off (status read still on)"}
        applied["on"] = False
        return {"ok": True, "applied": applied, "state": got, "path": "lan"}
    if not got["on"]:
        return {"error": "Magic Home did not turn on (status read still off)"}
    applied["on"] = True
    if color:
        # rgb_unscaled is the colour at full brightness, so pct is ignored.
        if any(abs(a - b) > _COLOR_TOLERANCE for a, b in zip(got["rgb"], color)):
            return {"error": f"Magic Home colour mismatch (read {got['rgb']})"}
        applied["color"] = list(color)
    if pct is not None:
        if got["brightness"] is None or abs(got["brightness"] - pct) > _BRIGHT_TOLERANCE_PCT:
            return {"error": f"Magic Home brightness mismatch (read {got['brightness']})"}
        applied["brightness"] = pct
    return {"ok": True, "applied": applied, "state": got, "path": "lan"}


# -- voice action ------------------------------------------------------
def magichome_list(_: str = "") -> str:
    devs = list_devices()
    if not devs:
        return "No Magic Home controllers found on the network, sir."
    names = [d["name"] for d in devs]
    return f"{len(names)} Magic Home controller(s): " + ", ".join(names[:10])


def register(actions: dict) -> None:
    actions["magichome_list_devices"] = magichome_list
