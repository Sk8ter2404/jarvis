"""
sh_govee — Govee controller skill.

Talks to Govee devices via two layers, in priority order:

  1. **LAN API** (UDP 4001/4003) — many recent Govee bulbs/strips expose
     a JSON-over-UDP control surface that requires no internet round-trip.
     Per Govee's WLAN guide, devices receive multicast scan requests on
     239.255.255.250:4001 and control commands on <ip>:4003. The canonical
     reply port for scans is 4002, but we avoid binding it: the Govee Home
     desktop app also binds 4002 and Windows can't cleanly multiplex the
     two. Instead _lan_scan sends from an ephemeral port and devices reply
     unicast to that source port (reply-to semantics). Toggled per-device
     in the Govee app under Settings → LAN Control. See
     https://app-h5.govee.com/user-manual/wlan-guide.

  2. **Cloud REST API** at https://developer-api.govee.com — requires the
     user's Govee API key in env var GOVEE_API_KEY or data/sh_govee_config.json
     ({"api_key": "..."}). Adds ~300 ms latency but works for every Govee
     model.

If neither path is configured, set_state returns an informative error.

Catalog records that arrive from the discovery wizard with a `lan_ip`
take the LAN path; without one the skill falls back to the cloud and
matches devices by name.
"""
from __future__ import annotations

import json
import os
import socket
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
_CONFIG_PATH = os.path.join(_DATA_DIR, "sh_govee_config.json")

# Govee LAN ports (per WLAN guide at https://app-h5.govee.com/user-manual/wlan-guide).
#   4001 — device-side scan-request port (we send to 239.255.255.250:4001)
#   4003 — device-side command port (we send control payloads to <ip>:4003)
# We deliberately do NOT bind the canonical app-side reply port 4002:
# the Govee Home desktop app also binds 4002 with SO_REUSEADDR, and on
# Windows two processes sharing a UDP port receive packets non-
# deterministically (discovery would return 0 or duplicate devices).
# Instead _lan_scan binds an ephemeral port and uses that same socket
# to send the scan request; devices reply unicast to the scan's source
# port (the WLAN guide's reply-to semantics).
_LAN_MULTICAST = "239.255.255.250"
_LAN_SCAN_PORT = 4001  # device-side: receives multicast scan request
_LAN_CMD_PORT  = 4003  # device-side: receives control commands

_CLOUD_BASE = "https://developer-api.govee.com/v1"
_CLOUD_TTL_SECS = 60.0
_LAN_TTL_SECS   = 60.0

_lock = threading.Lock()
_state: dict[str, Any] = {
    "lan_devices":   {},     # ip → {model, sku}
    "lan_fetched":   0.0,
    "cloud_devices": [],     # raw cloud device dicts
    "cloud_fetched": 0.0,
}


# ── config / API key ───────────────────────────────────────────────
def _api_key() -> str | None:
    key = os.environ.get("GOVEE_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            return (cfg.get("api_key") or "").strip() or None
        except Exception:
            return None
    return None


def is_available() -> bool:
    """LAN broadcast always works at OS level (sockets are stdlib); cloud
    needs `requests` plus an API key. We report True as long as either
    path could plausibly succeed — set_state still returns a useful
    error if neither responds."""
    return True


# ── LAN scan ──────────────────────────────────────────────────────
def _lan_scan(timeout: float = 1.5) -> dict[str, dict]:
    """Broadcast the Govee LAN scan request, collect replies for `timeout`
    seconds, return {ip → info dict}.

    Uses a single UDP socket bound to an ephemeral port for both send and
    receive. The scan datagram's source port (= our ephemeral port) is what
    devices reply to per the WLAN guide's reply-to semantics, so we never
    need to occupy the canonical 4002 port that the Govee Home desktop app
    also wants."""
    sock = None
    found: dict[str, dict] = {}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        try:
            sock.bind(("0.0.0.0", 0))
        except OSError as e:
            print(f"  [sh-govee] LAN socket bind failed: {e}")
            return {}
        sock.settimeout(0.2)

        payload = json.dumps({
            "msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}
        }).encode("utf-8")

        try:
            sock.sendto(payload, (_LAN_MULTICAST, _LAN_SCAN_PORT))
        except Exception as e:
            print(f"  [sh-govee] LAN scan broadcast failed: {e}")
            return {}

        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8", errors="replace"))
            except Exception:
                continue
            d = (msg.get("msg") or {}).get("data") or {}
            ip = d.get("ip") or addr[0]
            found[ip] = {
                "ip":     ip,
                "sku":    d.get("sku") or "",
                "device": d.get("device") or "",
                "ble":    d.get("bleVersionHard"),
            }
    finally:
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass
    return found


def _refresh_lan() -> dict[str, dict]:
    with _lock:
        if (time.monotonic() - _state["lan_fetched"]) < _LAN_TTL_SECS:
            return dict(_state["lan_devices"])
    found = _lan_scan()
    with _lock:
        _state["lan_devices"] = found
        _state["lan_fetched"] = time.monotonic()
    return found


def _send_lan_cmd(ip: str, cmd: dict) -> dict:
    """Send one LAN command and (optionally) read one reply."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.6)
        try:
            payload = json.dumps({"msg": cmd}).encode("utf-8")
            sock.sendto(payload, (ip, _LAN_CMD_PORT))
            return {"ok": True}
        finally:
            sock.close()
    except Exception as e:
        return {"error": f"lan send failed: {e}"}


# ── cloud REST ─────────────────────────────────────────────────────
def _requests():
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        return None


def _cloud_devices() -> list[dict]:
    key = _api_key()
    if not key:
        return []
    with _lock:
        if (time.monotonic() - _state["cloud_fetched"]) < _CLOUD_TTL_SECS \
                and _state["cloud_devices"]:
            return list(_state["cloud_devices"])
    req = _requests()
    if req is None:
        return []
    try:
        r = req.get(f"{_CLOUD_BASE}/devices",
                    headers={"Govee-API-Key": key}, timeout=6)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as e:
        print(f"  [sh-govee] cloud list failed: {e}")
        return []
    devices = (data.get("data") or {}).get("devices") or []
    with _lock:
        _state["cloud_devices"] = devices
        _state["cloud_fetched"] = time.monotonic()
    return list(devices)


def _cloud_match(device: dict) -> dict | None:
    name = (device.get("name") or "").lower().strip()
    if not name:
        return None
    for d in _cloud_devices():
        if (d.get("deviceName") or "").lower().strip() == name:
            return d
    return None


def _cloud_control(d: dict, cmd: str, value: Any) -> dict:
    key = _api_key()
    if not key:
        return {"error": "no Govee API key"}
    req = _requests()
    if req is None:
        return {"error": "requests not installed"}
    body = {
        "device": d.get("device"),
        "model":  d.get("model"),
        "cmd":    {"name": cmd, "value": value},
    }
    try:
        r = req.put(f"{_CLOUD_BASE}/devices/control",
                    headers={"Govee-API-Key": key, "Content-Type": "application/json"},
                    json=body, timeout=6)
        r.raise_for_status()
        return {"ok": True, "cloud": r.json()}
    except Exception as e:
        return {"error": f"cloud control failed: {e}"}


# ── public API ─────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    out: list[dict] = []
    for ip, info in _refresh_lan().items():
        out.append({
            "name": info.get("device") or info.get("sku") or ip,
            "brand": "Govee",
            "type": "light",
            "capabilities": ["on_off", "dim", "color"],
            "lan_ip": ip,
            "model": info.get("sku") or "",
        })
    for d in _cloud_devices():
        if not any(o.get("model") == d.get("model")
                   and o.get("name") == d.get("deviceName") for o in out):
            caps = ["on_off"]
            if d.get("supportCmds"):
                if "brightness" in d["supportCmds"]:
                    caps.append("dim")
                if "color" in d["supportCmds"]:
                    caps.append("color")
                if "colorTem" in d["supportCmds"]:
                    caps.append("color_temperature")
            out.append({
                "name": d.get("deviceName"),
                "brand": "Govee",
                "type": "light",
                "capabilities": sorted(set(caps)),
                "model": d.get("model"),
            })
    return out


def get_state(device: dict) -> dict:
    """Govee doesn't expose a read endpoint on the LAN protocol; we issue
    a cloud `state` GET when we have an API key, else return 'unknown'."""
    d = _cloud_match(device)
    if d is None:
        return {"on": "unknown", "note": "Govee LAN protocol is write-only"}
    key = _api_key()
    req = _requests()
    if not key or req is None:
        return {"on": "unknown"}
    try:
        r = req.get(f"{_CLOUD_BASE}/devices/state",
                    headers={"Govee-API-Key": key},
                    params={"device": d.get("device"), "model": d.get("model")},
                    timeout=6)
        r.raise_for_status()
        payload = r.json() or {}
    except Exception as e:
        return {"error": f"cloud state read failed: {e}"}
    props = ((payload.get("data") or {}).get("properties") or [])
    state: dict[str, Any] = {}
    for p in props:
        for k, v in (p or {}).items():
            state[k] = v
    return state


def set_state(device: dict, **kwargs) -> dict:
    ip = (device.get("lan_ip") or "").strip()
    applied: dict[str, Any] = {}

    if ip:
        # LAN path — preferred when an IP is known.
        if "on" in kwargs and kwargs["on"] is not None:
            r = _send_lan_cmd(ip, {"cmd": "turn",
                                    "data": {"value": 1 if kwargs["on"] else 0}})
            if "error" in r:
                return r
            applied["on"] = bool(kwargs["on"])
        # Color/temperature MUST be sent before brightness: on many Govee
        # models the `colorwc` command resets brightness to full, so sending
        # brightness first would be clobbered. Apply color first, then set
        # brightness last so the requested level sticks.
        if "color" in kwargs and kwargs["color"]:
            r, g, b = kwargs["color"]
            r2 = _send_lan_cmd(ip, {"cmd": "colorwc",
                                     "data": {"color": {"r": int(r),
                                                         "g": int(g),
                                                         "b": int(b)},
                                              "colorTemInKelvin": 0}})
            if "error" in r2:
                return r2
            applied["color"] = [int(r), int(g), int(b)]
        if "color_temperature" in kwargs and kwargs["color_temperature"]:
            k = int(kwargs["color_temperature"])
            r = _send_lan_cmd(ip, {"cmd": "colorwc",
                                    "data": {"color": {"r": 0, "g": 0, "b": 0},
                                             "colorTemInKelvin": k}})
            if "error" in r:
                return r
            applied["color_temperature_k"] = k
        if "brightness" in kwargs and kwargs["brightness"] is not None:
            pct = max(0, min(100, int(kwargs["brightness"])))
            r = _send_lan_cmd(ip, {"cmd": "brightness", "data": {"value": pct}})
            if "error" in r:
                return r
            applied["brightness"] = pct
        return {"ok": True, "applied": applied, "path": "lan"}

    # Cloud path.
    cloud = _cloud_match(device)
    if cloud is None:
        return {"error": "Govee device not found via LAN or cloud"}
    if "on" in kwargs and kwargs["on"] is not None:
        r = _cloud_control(cloud, "turn", "on" if kwargs["on"] else "off")
        if "error" in r:
            return r
        applied["on"] = bool(kwargs["on"])
    if "brightness" in kwargs and kwargs["brightness"] is not None:
        pct = max(0, min(100, int(kwargs["brightness"])))
        r = _cloud_control(cloud, "brightness", pct)
        if "error" in r:
            return r
        applied["brightness"] = pct
    if "color" in kwargs and kwargs["color"]:
        r, g, b = kwargs["color"]
        r2 = _cloud_control(cloud, "color",
                            {"r": int(r), "g": int(g), "b": int(b)})
        if "error" in r2:
            return r2
        applied["color"] = [int(r), int(g), int(b)]
    if "color_temperature" in kwargs and kwargs["color_temperature"]:
        k = int(kwargs["color_temperature"])
        r = _cloud_control(cloud, "colorTem", k)
        if "error" in r:
            return r
        applied["color_temperature_k"] = k
    return {"ok": True, "applied": applied, "path": "cloud"}


def govee_list(_: str = "") -> str:
    devs = list_devices()
    if not devs:
        return ("No Govee devices found, sir — LAN scan empty and no cloud "
                "API key (set GOVEE_API_KEY or data/sh_govee_config.json).")
    names = [d.get("name") or d.get("model") or "?" for d in devs]
    return f"{len(names)} Govee device(s): " + ", ".join(names[:10]) + (
        " (+more)" if len(names) > 10 else ""
    )


def register(actions: dict) -> None:
    actions["govee_list_devices"] = govee_list
    actions["govee_list"]         = govee_list
