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


def _switch_dp(dps: dict) -> str | None:
    """The on/off data point: '20' on Tuya lights/bulbs (category dj; their
    dp '1' does not exist), '1' on plugs and switches. None when neither is
    reported. (2026-10-13: the old code always drove dp 1 through
    OutletDevice.turn_on, so a bulb 'turned on' did nothing yet said OK.)"""
    for dp in ("20", "1"):
        if dp in dps:
            return dp
    return None


def _brightness_dp(dps: dict):
    """(dp, lowest, highest) for brightness: v2 lights use dp '22' (10..1000),
    older ones dp '3' (25..255). None when the device has no brightness."""
    if "22" in dps:
        return "22", 10, 1000
    if "3" in dps and "20" not in dps:
        return "3", 25, 255
    return None


def _read_dps(dev) -> tuple[dict, str]:
    """(dps, error). tinytuya reports a failed read as {'Error': ...}."""
    try:
        st = dev.status() or {}
    except Exception as e:
        return {}, f"tuya state read failed: {e}"
    if st.get("Error"):
        return {}, f"tuya device didn't answer: {st.get('Error')}"
    return (st.get("dps") or {}), ""


def get_state(device: dict) -> dict:
    rec = _find_record(device)
    dev = _device(rec) if rec else None
    if dev is None:
        return {"error": f"tuya device '{device.get('name')}' not ready"}
    dps, err = _read_dps(dev)
    if err:
        return {"error": err}
    sw = _switch_dp(dps)
    out = {"on": bool(dps.get(sw)) if sw else False, "raw": dps}
    b = _brightness_dp(dps)
    if b and isinstance(dps.get(b[0]), (int, float)):
        out["brightness"] = max(0, min(100, round(dps[b[0]] * 100 / b[2])))
    return out


def set_state(device: dict, **kwargs) -> dict:
    """On/off and brightness over the LAN, on the device's REAL data points
    (read first), confirmed by a read-back. Anything this skill can't do
    locally (colour, colour temperature), an unreachable device, or a device
    that doesn't confirm returns {"error"} so the router falls back to Alexa
    - never an "ok" for a change that didn't happen."""
    rec = _find_record(device)
    dev = _device(rec) if rec else None
    if dev is None:
        return {"error": f"tuya device '{device.get('name')}' not ready"}
    want_on = kwargs.get("on")
    pct = kwargs.get("brightness")
    extras = [k for k in ("color", "color_name", "color_temperature")
              if kwargs.get(k) is not None]
    if extras and want_on is None and pct is None:
        return {"error": f"tuya: {', '.join(extras)} isn't supported locally"}
    dps, err = _read_dps(dev)
    if err:
        return {"error": err}
    sw = _switch_dp(dps)
    bdp = _brightness_dp(dps)
    if pct is not None:
        pct = max(0, min(100, int(pct)))
        if pct == 0:
            want_on, pct = False, None
        elif want_on is None:
            want_on = True                  # "set it to 40%" means on
    applied: dict = {}
    target: dict = {}
    try:
        if want_on is not None:
            if sw is None:
                return {"error": "tuya device reports no on/off switch"}
            dev.set_value(int(sw), bool(want_on))
            applied["on"] = target[sw] = bool(want_on)
        if pct is not None and want_on is not False:
            if bdp is None:
                return {"error": "tuya device has no brightness control",
                        "partial": applied}
            dp, lo, hi = bdp
            value = max(lo, min(hi, round(hi * pct / 100)))
            dev.set_value(int(dp), value)
            applied["brightness"] = pct
            target[dp] = value
    except Exception as e:
        return {"error": f"tuya set_state failed: {e}", "partial": applied}
    after, err = _read_dps(dev)
    if err:
        return {"error": f"tuya change not confirmed ({err})", "partial": applied}
    for dp, want in target.items():
        got = after.get(dp)
        if isinstance(want, bool):
            ok = got is want
        else:
            ok = isinstance(got, (int, float)) and abs(got - want) <= max(5, hi * 0.02)
        if not ok:
            return {"error": f"tuya device did not confirm dp {dp} (wanted "
                             f"{want}, reads {got})", "partial": applied}
    return {"ok": True, "applied": applied}


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
    actions["smart_home_account"] = smart_home_account
    try:
        su = globals().get("skill_utils") or {}
        reg = su.get("register_utterance_route") if isinstance(su, dict) else None
        if callable(reg):
            reg(_account_route, "smart home account")
    except Exception:
        pass


# ── "which account are my smart devices on?" (2026-10-19) ──────────────
# Live 10-18 21:57 the owner asked which login his Tuya devices use; the model
# said it had "no way to check". The JARVIS Tuya project's app link answers it:
# the linked user's uid (data/tuya_wizard/tuya-raw.json) and Tuya's own
# /v1.0/users/{uid}/infos (the project's keys, data/tuya_wizard/tinytuya.json,
# never printed). The link was made with the Smart Life app on 2026-10-13.
# Spoken: only the last 4 digits of a phone login.
import re as _re
import time as _time

SPEAK_VERBATIM_ACTIONS = ("smart_home_account",)
LINKED_APP = "Smart Life"
_WIZARD_DIR = os.path.join(_DATA_DIR, "tuya_wizard")
_ACCOUNT_TTL_S = 6 * 3600.0
_account_cache: dict = {"at": 0.0, "info": None}


def _linked_uid(path: str | None = None) -> str:
    try:
        with open(path or os.path.join(_WIZARD_DIR, "tuya-raw.json"), "r", encoding="utf-8") as f:
            raw = json.load(f)
        for d in raw.get("result") or []:
            if isinstance(d, dict) and d.get("uid"):
                return str(d["uid"])
    except Exception:
        pass
    return ""


def _cloud_account(uid: str, cloud=None):
    """Tuya's user record for `uid`, or None. Never raises."""
    try:
        if cloud is None:
            tt = _tt()
            cfg_path = os.path.join(_WIZARD_DIR, "tinytuya.json")
            if tt is None or not os.path.exists(cfg_path):
                return None
            cloud = tt.Cloud(configFile=cfg_path)
        r = cloud.cloudrequest(f"/v1.0/users/{uid}/infos")
        return r.get("result") if isinstance(r, dict) and r.get("success") else None
    except Exception:
        return None


def describe_account(info, n_devices: int) -> str:
    if not info:
        return ("I couldn't reach Tuya to check the account just now, sir. Your devices "
                f"were linked from the {LINKED_APP} app.")
    login = str(info.get("username") or info.get("mobile") or info.get("email") or "")
    if "@" in login:
        how = f"the email {login}"
    else:
        digits = _re.sub(r"\D", "", login)
        how = f"your phone number ending {digits[-4:]}" if len(digits) >= 4 else "a login I can't read"
    since = ""
    try:
        since = " (an account from " + _time.strftime("%B %Y", _time.localtime(int(info["create_time"]))) + ")"
    except Exception:
        pass
    count = f"Your {n_devices} Tuya device{'' if n_devices == 1 else 's'} are" if n_devices else "Your Tuya devices are"
    if n_devices == 1:
        count = "Your Tuya device is"
    return (f"{count} in the {LINKED_APP} app, sir, signed in with {how}{since}. "
            f"The Tuya Smart app is a separate account, so they won't show up there.")


def smart_home_account(_: str = "", cloud=None, uid: str | None = None) -> str:
    uid = _linked_uid() if uid is None else uid
    if not uid:
        return ("I don't have a linked Tuya account on file, sir - the Tuya setup "
                "hasn't been run on this PC.")
    now = _time.time()
    if cloud is None and _account_cache["info"] and now - _account_cache["at"] < _ACCOUNT_TTL_S:
        info = _account_cache["info"]
    else:
        info = _cloud_account(uid, cloud)
        if info:
            _account_cache.update(at=now, info=info)
    return describe_account(info, len(_all_devices()))


_ACCOUNT_RE = _re.compile(r"\b(account|login|log ?in|sign(ed)? ?in|username)\b", _re.I)
_SMART_RE = _re.compile(r"\b(tuya|2 ?ua|smart ?life|smart (home|devices?|plugs?|lights?|fans?|lamps?))\b", _re.I)
_NOT_ACCOUNT_RE = _re.compile(r"\b(add|create|delete|remove|sign out|log ?out|link|unlink|reset)\b", _re.I)


def _account_route(text):
    t = str(text or "")
    if (len(t) <= 160 and _ACCOUNT_RE.search(t) and _SMART_RE.search(t)
            and not _NOT_ACCOUNT_RE.search(t)):
        return "[ACTION: smart_home_account]"
    return None
