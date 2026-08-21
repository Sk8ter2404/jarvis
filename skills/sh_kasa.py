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

Two contracts this module must keep (2026-08-20):
  * BOUNDED — every await goes through `_run_async`, which is bounded on both
    its paths. These calls sit on the voice dispatch thread and nothing above
    them applies a timeout.
  * HONEST — `set_state` returns ok:True only when the whole request landed;
    a failed or capability-skipped sub-command yields ok:False plus `failed`/
    `skipped` naming the device and what did not happen, and logs it.
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
# BOUNDED (2026-08-20). Everything below runs ON THE VOICE DISPATCH THREAD.
# The old runner had no bound at all — `asyncio.run(coro)` on the direct path
# and a bare `t.join()` on the nested one — so a half-open plug could stall
# JARVIS indefinitely (python-kasa retries 4x5 s with 1 s backoff ~= 23 s per
# query; update()+turn_on() ~= 46 s, and a toggle ~= 79 s, past the 60 s
# _MAIN_LOOP_HEARTBEAT_TIMEOUT) with no spoken "still working" line, because
# no smart-home action is in LONG_RUNNING_ACTIONS.
#
# The in-repo twin skills/smart_home_discover._run_async was given exactly this
# shape on 2026-07-08 and this copy never received the edit — bug class #1, the
# stale duplicate. Keep the two in step.
#
# asyncio.wait_for is the load-bearing part: it CANCELS the pending python-kasa
# query rather than orphaning a daemon thread holding an open socket. The join
# grace margin only covers a coroutine that ignores cancellation.
_CALL_TIMEOUT_SECS = 12.0        # one device call (update / turn_on / set_*)
_DISCOVERY_TIMEOUT_SECS = 10.0   # UDP broadcast: the library's own 5 s + margin,
                                 # kept under smart_home_discover's
                                 # _LAN_SOURCE_TIMEOUT_SEC = 12.0 ceiling.
# Whole-turn ceiling for smart_home_control, measured from function entry so
# discovery counts against it: "turn off everything" fans out sequentially, so
# a per-call bound alone still multiplies by the device count.
_CONTROL_BUDGET_SECS = 20.0
# Extra wall-clock allowed on the worker-thread join beyond the coroutine's own
# budget, for a coro blocked in a native call that never sees the cancel.
_JOIN_GRACE_SECS = 5.0

_USE_DEFAULT_TIMEOUT = object()   # sentinel: the budgets stay patchable/tunable
                                  # at call time instead of being frozen into a
                                  # default argument at import.


def _run_async(coro, timeout=_USE_DEFAULT_TIMEOUT, what: str = "call"):
    """Run `coro` to completion, even if the calling thread is itself in
    an event loop (delegated to a worker thread in that case) — bounded by
    `timeout` seconds on BOTH paths.

    On expiry raises TimeoutError carrying what timed out and after how long;
    every caller turns that into an honest {"error": ...} or a logged
    degradation, never a silent one. `timeout=None` restores the old unbounded
    behaviour and is used by nothing."""
    if timeout is _USE_DEFAULT_TIMEOUT:
        timeout = _CALL_TIMEOUT_SECS
    msg = (f"kasa {what} timed out after {timeout:g}s"
           if timeout is not None else f"kasa {what} timed out")

    async def _bounded():
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # asyncio.wait_for raises a bare TimeoutError with no message;
            # restate it so the log/spoken line says WHAT timed out.
            raise TimeoutError(msg) from None

    try:
        asyncio.get_running_loop()
        nested = True
    except RuntimeError:
        nested = False
    if not nested:
        return asyncio.run(_bounded())
    box: dict = {}
    def _go() -> None:
        try:
            box["v"] = asyncio.run(_bounded())
        except Exception as e:
            box["err"] = e
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(None if timeout is None else timeout + _JOIN_GRACE_SECS)
    if t.is_alive():
        raise TimeoutError(msg + " (worker thread abandoned)")
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
        found = _run_async(_discover_async(),
                           timeout=_DISCOVERY_TIMEOUT_SECS,
                           what="LAN discovery") or {}
    except Exception as e:
        print(f"  [sh-kasa] discovery error: {type(e).__name__}: {e}")
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
            dev = _run_async(_device_from_ip_async(ip),
                             what=f"connect {ip}")
            if dev is not None:
                return dev
        except Exception as e:
            # A DEGRADED PATH MUST LOG. This used to be `except: pass`, so a
            # timeout here was invisible and we silently fell through to a full
            # LAN broadcast — doubling the wait with no trace of why.
            print(f"  [sh-kasa] direct connect to {ip} failed "
                  f"({type(e).__name__}: {e}); falling back to broadcast")
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
        _run_async(dev.update(),
                   what=f"state read '{device.get('name')}'")
        return {
            "on":          bool(getattr(dev, "is_on", False)),
            "brightness":  int(getattr(dev, "brightness", 0) or 0) if getattr(dev, "is_dimmable", False) else None,
            "alias":       getattr(dev, "alias", ""),
            "model":       getattr(dev, "model", ""),
        }
    except Exception as e:
        return {"error": f"kasa state read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    """Drive one Kasa/Tapo device. THE RESULT REPORTS WHAT ACTUALLY HAPPENED.

    HONEST-FAILURE CONTRACT (2026-08-20). Every sub-command used to be wrapped
    in a bare `except Exception: pass` and the function returned a flat
    {"ok": True} regardless, so a request the hardware never honoured was
    reported as a success — e.g. "set the desk lamp to 30 percent" against a
    non-dimmable plug switched it fully ON, recorded nothing, and the router
    spoke "Set to 30%, sir". The owner walks away believing a light dimmed.

    Return shape:
      * everything requested landed  -> {"ok": True, "device", "applied"}
      * anything failed or was skipped ->
            {"ok": False, "device", "applied", "partial", "failed", "skipped",
             "error": "<device>: <per-sub-command detail> (applied: ...)"}
        `ok` is True ONLY when the whole request landed. `failed` holds
        sub-commands that were attempted and raised; `skipped` holds ones the
        hardware cannot do (capability gate) so they were never attempted.
        Both are also printed, so a degraded run is visible in the log even if
        a caller ignores the dict.
      * power (`on`) is deliberately NOT swallowed: if turn_on/turn_off raises,
        the whole apply aborts into the error branch — there is no point
        dimming a device we could not switch.
    """
    name = (device.get("name") or "").strip() or "device"
    dev = _device_for(device)
    if dev is None:
        return {"error": f"kasa device '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}
    failed:  dict[str, str] = {}   # attempted and raised
    skipped: dict[str, str] = {}   # never attempted (device can't do it)

    def _fail(key: str, exc: Exception) -> None:
        failed[key] = f"{type(exc).__name__}: {exc}"
        print(f"  [sh-kasa] {name}: {key} FAILED — {type(exc).__name__}: {exc}")

    def _skip(key: str, why: str) -> None:
        skipped[key] = why
        print(f"  [sh-kasa] {name}: {key} NOT applied — {why}")

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
                except Exception as e:
                    _fail("brightness", e)
            else:
                # The owner's TP-Link fleet is plugs, so this is the live case:
                # a brightness request lands here, the plug is switched fully
                # on below, and without this marker the result claimed success.
                _skip("brightness", "device is not dimmable")
            if pct > 0 and not applied.get("on", False):
                try:
                    await dev.turn_on()
                    applied["on"] = True
                except Exception as e:
                    _fail("on", e)
        if "color_temperature" in kwargs and kwargs["color_temperature"]:
            if getattr(dev, "is_variable_color_temp", False):
                try:
                    await dev.set_color_temp(int(kwargs["color_temperature"]))
                    applied["color_temperature_k"] = int(kwargs["color_temperature"])
                except Exception as e:
                    _fail("color_temperature", e)
            else:
                _skip("color_temperature",
                      "device has no variable colour temperature")
        if "color" in kwargs and kwargs["color"]:
            if getattr(dev, "is_color", False):
                try:
                    h, s, v = _rgb_to_hsv(kwargs["color"])
                    await dev.set_hsv(h, s, v)
                    applied["color"] = list(kwargs["color"])
                except Exception as e:
                    _fail("color", e)
            else:
                _skip("color", "device is not colour-capable")

    try:
        _run_async(_apply(), what=f"set_state '{name}'")
    except Exception as e:
        return {"error": f"kasa set_state failed: {e}",
                "device": name, "applied": applied, "partial": applied,
                "failed": failed, "skipped": skipped}

    if not failed and not skipped:
        return {"ok": True, "device": name, "applied": applied}

    problems = [f"{k} failed ({v})" for k, v in failed.items()]
    problems += [f"{k} not applied ({v})" for k, v in skipped.items()]
    did = ", ".join(f"{k}={v}" for k, v in applied.items()) or "nothing"
    return {
        "ok": False,
        "device":  name,
        "applied": applied,
        "partial": applied,
        "failed":  failed,
        "skipped": skipped,
        "error":   f"kasa '{name}': " + "; ".join(problems)
                   + f" (applied: {did})",
    }


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
    unified control below can drive Tuya devices too. None if not loaded.

    Deliberate sibling copy of core.smart_home_router._skill_module — the
    lookup rule's one shared home — kept local so this skill works when the
    router isn't loaded at all. Change BOTH if the loader naming changes."""
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


def _failure_reason(result: dict) -> str:
    """Short, speakable reason out of a skill result dict. Never invents one —
    an unlabelled failure says so rather than being dressed up."""
    txt = str((result or {}).get("error") or "").strip()
    if not txt:
        txt = "no reason reported"
    return txt if len(txt) <= 140 else txt[:137] + "..."


def smart_home_control(request: str = "") -> str:
    """Voice control for the LAN smart plugs: 'turn on the entry light',
    'turn off dining room', 'toggle kitchen 2', 'are the lights on?'.

    Parses on/off/toggle + the device name out of the request, matches it
    against the live Kasa discovery, and drives it directly over the LAN — no
    Amazon/Alexa needed. 2026-05-30 (added after Amazon locked down the Alexa
    cookie API; these TP-Link Kasa plugs are controlled locally instead)."""
    import re as _re
    # Whole-turn deadline (2026-08-20). Bounding each device call is not
    # enough: "turn off everything" fans out sequentially, so N devices still
    # multiply N x the per-call bound. Started before discovery so the
    # broadcast counts against the same budget.
    _turn_deadline = time.monotonic() + _CONTROL_BUDGET_SECS
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

    out: list[str] = []
    failures = 0
    not_attempted: list[str] = []
    for d in matches:
        rec = {"name": _clean(d.get("name")), "lan_ip": d.get("lan_ip"),
               "_tuya": d.get("_tuya")}
        nm = rec["name"]
        if time.monotonic() >= _turn_deadline:
            # Out of budget. Say so — do NOT quietly drop the rest and then
            # report "Done" for a set we never touched.
            not_attempted.append(nm)
            continue
        # Route to the right controller (Kasa local API vs Tuya/tinytuya).
        if d.get("_ctl") == "tuya" and tmod is not None:
            _set, _get = tmod.set_state, tmod.get_state
        else:
            _set, _get = set_state, get_state
        if intent in ("on", "off"):
            r = _set(rec, on=(intent == "on"))
            if r.get("ok"):
                out.append(f"{nm} {intent}")
            else:
                failures += 1
                out.append(f"{nm} (failed: {_failure_reason(r)})")
        elif intent == "toggle":
            st = _get(rec)
            if "error" in st or st.get("on") is None:
                # NEVER GUESS. `not bool(st.get("on"))` on an unreadable device
                # resolved to True, so a failed state read silently became a
                # "turn it on" command whose result was then reported as fact.
                failures += 1
                out.append(f"{nm} (failed: couldn't read state — "
                           f"{_failure_reason(st)})")
                continue
            new_on = not bool(st.get("on"))
            r = _set(rec, on=new_on)
            if r.get("ok"):
                out.append(f"{nm} {'on' if new_on else 'off'}")
            else:
                failures += 1
                out.append(f"{nm} (failed: {_failure_reason(r)})")
        else:
            st = _get(rec)
            if "error" in st or st.get("on") is None:
                # An unreadable device used to be reported as "is off".
                failures += 1
                out.append(f"{nm} status unknown ({_failure_reason(st)})")
            else:
                out.append(f"{nm} is {'on' if st.get('on') else 'off'}")
    if not_attempted:
        # Phrased with "couldn't" on purpose: core.failure_markers.FAILURE_MARKERS
        # is what bobert_companion._is_failure / dispatcher._is_failure_result
        # classify on, and "not attempted" alone matches no marker — the line
        # would have been filed as a SUCCESS.
        out.append(f"{len(not_attempted)} not attempted "
                   f"({', '.join(not_attempted)}) — I couldn't get to them "
                   f"in time")
    if intent is None:
        return "Status, sir — " + "; ".join(out) + "."
    attempted = len(matches) - len(not_attempted)
    if failures and failures >= attempted:
        return "That didn't work, sir — " + "; ".join(out) + "."
    if failures or not_attempted:
        return "Partly done, sir — " + "; ".join(out) + "."
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
