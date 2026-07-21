"""Tuya / Smart-Life local control (no Amazon, no live cloud calls).

Controls Tuya v3.3 plugs/switches/bulbs over the LAN using tinytuya. Each
device needs a per-device *local key* (encrypted by Tuya) which is fetched once
via a free Tuya IoT developer account + `python -m tinytuya wizard`; the keys
land in data/tuya_devices.json. Devices without a key yet are surfaced as
"discovered but not set up" so the user knows what's pending.

Public API (mirrors sh_kasa so the smart-home router treats both uniformly):
    list_devices() -> list[dict]   # only key-equipped (controllable) devices
    get_state(rec) -> dict
    set_state(rec, on=bool, brightness=int) -> dict

2026-05-30: added after Amazon locked down the Alexa cookie API; these LAN
Tuya devices are controlled directly instead.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CATALOG = os.path.join(_DATA_DIR, "tuya_devices.json")
_lock = threading.Lock()
_tinytuya = None


def _tt():
    global _tinytuya
    if _tinytuya is None:
        try:
            import tinytuya  # type: ignore
            _tinytuya = tinytuya
        except Exception:
            _tinytuya = False
    return _tinytuya or None


def _load_catalog() -> list[dict]:
    try:
        with open(_CATALOG, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in (data.get("devices") or []) if isinstance(d, dict)]
    except Exception:
        return []


def _all_devices() -> list[dict]:
    """Every catalog device (including not-yet-keyed ones)."""
    return _load_catalog()


def _ready_devices() -> list[dict]:
    """Only devices that have a local key — i.e. actually controllable."""
    return [d for d in _load_catalog() if (d.get("key") or "").strip()]


def _device(rec: dict):
    """Build a live tinytuya handle from a catalog record (needs a key)."""
    tt = _tt()
    if tt is None or not (rec.get("key") or "").strip():
        return None
    try:
        ver = float(rec.get("version") or 3.3)
    except (TypeError, ValueError):
        ver = 3.3
    try:
        d = tt.OutletDevice(dev_id=rec.get("id"), address=rec.get("ip"),
                            local_key=rec.get("key"), version=ver)
        d.set_socketTimeout(3)
        return d
    except Exception:
        return None


# ── public API (router-compatible) ────────────────────────────────────
def list_devices() -> list[dict]:
    out = []
    for d in _ready_devices():
        out.append({
            "name":   (d.get("name") or d.get("id") or "").strip(),
            "brand":  "Tuya",
            "model":  d.get("product") or "",
            "type":   "plug",
            "capabilities": ["on_off"],
            "lan_ip": d.get("ip"),
            "_tuya":  d,            # carry the full record for control
        })
    return out


def _find_record(device: dict) -> dict | None:
    if device.get("_tuya"):
        return device["_tuya"]
    name = (device.get("name") or "").strip().lower()
    ip = (device.get("lan_ip") or device.get("ip") or "").strip()
    for d in _ready_devices():
        if ip and d.get("ip") == ip:
            return d
        if name and (d.get("name") or "").strip().lower() == name:
            return d
    return None


def get_state(device: dict) -> dict:
    rec = _find_record(device)
    dev = _device(rec) if rec else None
    if dev is None:
        return {"error": f"tuya device '{device.get('name')}' not ready"}
    try:
        st = dev.status() or {}
        dps = st.get("dps") or {}
        # dps '1' is the switch on most Tuya plugs/switches.
        on = bool(dps.get("1", dps.get("20", False)))
        return {"on": on, "raw": dps}
    except Exception as e:
        return {"error": f"tuya state read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    rec = _find_record(device)
    dev = _device(rec) if rec else None
    if dev is None:
        return {"error": f"tuya device '{device.get('name')}' not ready"}
    applied = {}
    try:
        if "on" in kwargs and kwargs["on"] is not None:
            if kwargs["on"]:
                dev.turn_on()
            else:
                dev.turn_off()
            applied["on"] = bool(kwargs["on"])
        if kwargs.get("brightness") is not None:
            # Tuya brightness dp is usually '2'/'22' (0..1000). Best-effort.
            pct = max(0, min(100, int(kwargs["brightness"])))
            try:
                dev.set_value(22, int(pct * 10))
                applied["brightness"] = pct
            except Exception:
                pass
        return {"ok": True, "applied": applied}
    except Exception as e:
        return {"error": f"tuya set_state failed: {e}", "partial": applied}


# ── voice helpers ──────────────────────────────────────────────────────
def tuya_list(_: str = "") -> str:
    ready = _ready_devices()
    pending = [d for d in _all_devices() if not (d.get("key") or "").strip()]
    if not _all_devices():
        return "No Tuya devices catalogued yet, sir."
    bits = []
    if ready:
        names = ", ".join((d.get("name") or d.get("id"))[:24] for d in ready)
        bits.append(f"{len(ready)} Tuya device(s) ready: {names}.")
    if pending:
        bits.append(f"{len(pending)} discovered but awaiting their local key "
                    "(run the Tuya setup, sir).")
    return " ".join(bits)


def register(actions: dict) -> None:
    actions["tuya_list"]         = tuya_list
    actions["tuya_list_devices"] = tuya_list
    actions["smart_life_list"]   = tuya_list
