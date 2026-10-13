"""BroadLink RM Pro local IR/RF control (no cloud, no Amazon).

The RM Pro hub blasts IR/RF codes. The BroadLink app keeps its own codes in its
cloud, which cannot be read locally, so JARVIS learns each code ONCE from the
owner's real remote and stores it in data/broadlink_codes.json:

    {"<name>": {"code": "<base64>", "type": "ir"|"rf", "learned_at": "<iso>"}}

Public API (same contract as sh_kasa / sh_tuya, so the smart-home router
treats it uniformly):
    set_state(device, on=bool) -> {"ok": ...} | {"error": ...}
    get_state(device)          -> always {"error": ...}: IR/RF is one-way, the
                                  hub cannot know what state the device is in.
    learn(name, kind="ir")     -> str   (owner-initiated only, bounded wait)
    list_codes() / forget(name) / discover()

A device with no learned code returns {"error": ...}, so the router falls back
to the Alexa scene until a code is learned. When on and off are the same
(toggle) code the result says so; a resulting state is never claimed.

Voice actions (owner-initiated, never auto-learned): broadlink_learn,
broadlink_list, broadlink_forget.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

from core.atomic_io import _atomic_write_json

try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CODES_PATH = os.path.join(_DATA_DIR, "broadlink_codes.json")

DISCOVER_TIMEOUT_S = 4
LEARN_TIMEOUT_S = 25
_POLL_S = 0.5
_HUB_TTL_S = 300

SPEAK_VERBATIM_ACTIONS = ("broadlink_learn", "learn_remote_code", "broadlink_list",
                          "list_remote_codes", "broadlink_forget", "forget_remote_code")

_lock = threading.Lock()
_hub = {"dev": None, "at": 0.0}
_bl = None


def _broadlink():
    """Lazy import; None when python-broadlink is not installed."""
    global _bl
    if _bl is None:
        try:
            import broadlink  # type: ignore
            _bl = broadlink
        except Exception:
            _bl = False
    return _bl or None


# ── name + code store ──────────────────────────────────────────────────
def _norm(name) -> str:
    s = re.sub(r"\s+", " ", str(name or "").strip().lower())
    s = re.sub(r"^(the|my)\s+", "", s)
    s = re.sub(r"\s+(remote\s+)?code$", "", s)
    s = re.sub(r"\s+remote$", "", s)
    return s.strip()


def _load_codes() -> dict:
    """Never raises; a missing, corrupt or wrongly-shaped file reads as empty."""
    try:
        with open(_CODES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("code"), str)}


def _save_codes(codes: dict) -> None:
    os.makedirs(os.path.dirname(_CODES_PATH) or ".", exist_ok=True)
    _atomic_write_json(_CODES_PATH, codes)


def _store_code(name: str, raw: bytes, kind: str) -> None:
    with _lock:
        codes = _load_codes()
        codes[name] = {
            "code": base64.b64encode(raw).decode("ascii"),
            "type": kind,
            "learned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save_codes(codes)


# ── hub discovery ──────────────────────────────────────────────────────
def discover(timeout: float = DISCOVER_TIMEOUT_S):
    """Find the RM hub on the LAN. Returns an authenticated device or None.
    Never raises; bounded by `timeout`. The result is cached for a few minutes."""
    bl = _broadlink()
    if bl is None:
        return None
    now = time.time()
    with _lock:
        if _hub["dev"] is not None and now - _hub["at"] < _HUB_TTL_S:
            return _hub["dev"]
    try:
        found = bl.discover(timeout=timeout) or []
    except Exception:
        return None
    for dev in found:
        if not (hasattr(dev, "send_data") and hasattr(dev, "enter_learning")):
            continue            # a BroadLink plug/bulb, not a remote hub
        try:
            dev.auth()
        except Exception:
            continue
        with _lock:
            _hub["dev"], _hub["at"] = dev, now
        return dev
    return None


def _forget_hub() -> None:
    with _lock:
        _hub["dev"], _hub["at"] = None, 0.0


# ── send ───────────────────────────────────────────────────────────────
def _send_raw(entry: dict):
    """Blast one stored code. Returns None on success, else an error string."""
    try:
        raw = base64.b64decode(entry["code"], validate=True)
    except (binascii.Error, ValueError, KeyError):
        return "stored code is corrupt"
    dev = discover()
    if dev is None:
        return "BroadLink hub not found on the network"
    try:
        dev.send_data(raw)
    except Exception as e:
        _forget_hub()
        return f"send failed: {e}"
    return None


def set_state(device: dict, **kwargs) -> dict:
    """Router entry point. Looks for '<name> on'/'<name> off', then a single
    '<name>' code (a toggle). No learned code -> {"error"} so the router falls
    back to the Alexa scene."""
    on = kwargs.get("on")
    if on is None:
        return {"error": "broadlink: only on/off is supported"}
    name = _norm((device or {}).get("name"))
    if not name:
        return {"error": "broadlink: device has no name"}
    codes = _load_codes()
    word = "on" if on else "off"
    explicit = codes.get(f"{name} {word}")
    # A catalog entry already named like 'room on' is itself the on code.
    entry = explicit or codes.get(name)
    if entry is None:
        return {"error": f"no learned BroadLink code for '{name}' "
                         f"(say 'learn the {name} remote code')"}
    err = _send_raw(entry)
    if err:
        return {"error": f"broadlink {err}"}
    if explicit is not None:
        return {"ok": True, "applied": {"on": bool(on)}, "sent": f"{name} {word}"}
    other = codes.get(f"{name} {'off' if on else 'on'}")
    if other is None or other.get("code") == entry.get("code"):
        return {"ok": True, "applied": {}, "sent": name, "toggle": True,
                "state_unknown": True,
                "note": f"sent the toggle code for {name}; I can't tell "
                        "what state it ended up in"}
    return {"ok": True, "applied": {}, "sent": name}


def get_state(device: dict) -> dict:
    return {"error": "IR/RF is one-way; BroadLink cannot report device state"}


def list_devices() -> list:
    return []


# ── learn ──────────────────────────────────────────────────────────────
def _poll(fn, deadline: float, sleep):
    while time.time() < deadline:
        try:
            data = fn()
            if data:
                return bytes(data)
        except Exception:
            pass                # StorageError / "nothing yet": keep waiting
        sleep(_POLL_S)
    return None


def learn(name: str, kind: str = "ir", timeout: float = LEARN_TIMEOUT_S,
          say=None, _sleep=time.sleep) -> str:
    """Learn one code from the owner's remote. Owner-initiated only. `say`
    (optional) voices the instruction before the bounded wait starts."""
    key = _norm(name)
    if not key:
        return "Which remote code should I learn, sir?"
    if _broadlink() is None:
        return "The BroadLink library isn't installed, sir."
    dev = discover()
    if dev is None:
        return "I couldn't find the BroadLink hub on the network, sir."
    kind = "rf" if kind == "rf" else "ir"
    try:
        if kind == "rf":
            if not hasattr(dev, "sweep_frequency"):
                return "That hub doesn't do RF learning, sir."
            if say:
                say("Hold the button on the remote until I say it's locked, "
                    f"for the {key} code.")
            dev.sweep_frequency()
            # python-broadlink 0.19: check_frequency() -> (found, MHz). A
            # tuple is always truthy, so read the flag, not the return value
            # (2026-10-13 review); older builds returned a bare bool.
            freq_box: list = []

            def _locked():
                got = dev.check_frequency()
                found, mhz = (got if isinstance(got, tuple) else (got, None))
                if found:
                    freq_box.append(mhz)
                    return b"1"
                return None

            locked = _poll(_locked, time.time() + timeout, _sleep)
            if locked is None:
                try:
                    dev.cancel_sweep_frequency()
                except Exception:
                    pass
                return f"No RF signal heard in {int(timeout)} seconds; nothing learned, sir."
            if say:
                say("Frequency locked. Now tap the button once.")
            mhz = freq_box[0] if freq_box else None
            try:
                dev.find_rf_packet(frequency=mhz) if mhz else dev.find_rf_packet()
            except TypeError:           # an older build without the kwarg
                dev.find_rf_packet()
            raw = _poll(dev.check_data, time.time() + timeout, _sleep)
        else:
            if say:
                say("Point the remote at the hub and press the button now "
                    f"for the {key} code.")
            dev.enter_learning()
            raw = _poll(dev.check_data, time.time() + timeout, _sleep)
    except Exception as e:
        _forget_hub()
        return f"Learning failed: {e}"
    if raw is None:
        return f"Nothing heard in {int(timeout)} seconds; the {key} code was not learned, sir."
    try:
        _store_code(key, raw, kind)
    except Exception as e:
        return f"I heard the code but couldn't save it: {e}"
    return f"Learned the {key} {kind.upper()} code, sir."


def list_codes() -> str:
    codes = _load_codes()
    if not codes:
        return "No remote codes learned yet, sir."
    names = ", ".join(f"{n} ({v.get('type', 'ir')})" for n, v in sorted(codes.items()))
    return f"{len(codes)} learned remote code(s): {names}."


def forget(name: str) -> str:
    key = _norm(name)
    with _lock:
        codes = _load_codes()
        if key not in codes:
            return f"I have no learned code called {key or 'that'}, sir."
        del codes[key]
        _save_codes(codes)
    return f"Forgot the {key} code, sir."


# ── voice actions ──────────────────────────────────────────────────────
def _say():
    try:
        fn = (globals().get("skill_utils") or {}).get("speak_line")
    except Exception:
        return None
    if not callable(fn):
        return None

    def _s(text):
        try:
            fn(text)
        except Exception:
            pass
    return _s


def broadlink_learn(arg: str = "") -> str:
    text = str(arg or "")
    kind = "rf" if re.search(r"\brf\b", text, re.I) else "ir"
    text = re.sub(r"\b(rf|ir)\b", " ", text, flags=re.I)
    text = re.sub(r"^\s*(learn|record|teach)\s+(me\s+)?", "", text, flags=re.I)
    return learn(text, kind=kind, say=_say())


def broadlink_list(_: str = "") -> str:
    return list_codes()


def broadlink_forget(arg: str = "") -> str:
    text = re.sub(r"^\s*(forget|delete|remove)\s+", "", str(arg or ""), flags=re.I)
    return forget(text)


def register(actions: dict) -> None:
    actions["broadlink_learn"] = broadlink_learn
    actions["learn_remote_code"] = broadlink_learn
    actions["broadlink_list"] = broadlink_list
    actions["list_remote_codes"] = broadlink_list
    actions["broadlink_forget"] = broadlink_forget
    actions["forget_remote_code"] = broadlink_forget
