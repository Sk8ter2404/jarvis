"""
sh_nest — Nest thermostat / camera controller skill.

Google migrated Nest devices to the Smart Device Management API (SDM)
years ago, so this skill targets `google-nest-sdm` (alias
`python-google-nest`) rather than the deprecated Works-with-Nest API.

OAuth setup is non-trivial — the user has to:
  1. Pay the one-time $5 Device Access fee at console.nest.google.com.
  2. Create a project; note the Project ID.
  3. Create OAuth 2.0 client credentials in console.cloud.google.com.
  4. Drop both into `data/sh_nest_config.json`:
        {
          "project_id":   "...",
          "client_id":    "...",
          "client_secret": "...",
          "refresh_token": "..."        // produced by sh_nest_authorize
        }

If any of those are missing, the skill keeps every other piece of
JARVIS happy — set_state returns a clean error and the router falls
back to the Alexa cookie path.

Uniform set_state kwargs honored:
    temperature : int   (°F)
    mode        : 'HEAT' | 'COOL' | 'HEATCOOL' | 'OFF'
    on          : bool  (True → 'HEATCOOL', False → 'OFF')
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
_CONFIG_PATH = os.path.join(_DATA_DIR, "sh_nest_config.json")

_lock = threading.Lock()
# Serializes client construction so two concurrent cache-misses can't each
# build (and leak) an aiohttp.ClientSession. Held across the whole build,
# unlike _lock which only guards the brief cache reads/writes.
_build_lock = threading.Lock()
_state: dict[str, Any] = {
    "client": None,
    "devices": [],
    "fetched_at": 0.0,
    "loop": None,
    "thread": None,
}
_DEV_TTL = 60.0


def _sdm():
    try:
        from google_nest_sdm import (  # type: ignore
            auth as _auth,
            device_manager as _dm,
            google_nest_subscriber as _gns,
        )
        return _auth, _dm, _gns
    except Exception:
        return None


def _read_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def is_available() -> bool:
    if _sdm() is None:
        return False
    cfg = _read_config()
    return all(cfg.get(k) for k in ("project_id", "client_id",
                                     "client_secret", "refresh_token"))


# ── async runner ───────────────────────────────────────────────────
def _start_loop_thread() -> asyncio.AbstractEventLoop:
    """Start (idempotently) the daemon thread that owns the asyncio loop.

    aiohttp.ClientSession objects are bound to the loop they were created
    on. By keeping a single long-lived loop on its own thread we can cache
    the session across calls — otherwise each `asyncio.run(...)` opens and
    closes a loop, leaving the cached session unusable on the next call.
    """
    with _lock:
        loop = _state["loop"]
        if loop is not None and not loop.is_closed():
            return loop

    ready = threading.Event()
    new_loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(new_loop)
        ready.set()
        try:
            new_loop.run_forever()
        finally:
            try:
                new_loop.close()
            except Exception:  # pragma: no cover - defensive: loop.close() in the daemon thread's finally only runs at interpreter teardown
                pass

    t = threading.Thread(target=_run, name="sh-nest-asyncio", daemon=True)
    t.start()
    if not ready.wait(timeout=5.0):
        raise RuntimeError("sh-nest asyncio loop failed to start within 5s")

    with _lock:
        _state["loop"]   = new_loop
        _state["thread"] = t
    return new_loop


def _run_async(coro, timeout: float = 10.0):
    """Run `coro` on the shared loop thread and wait — BOUNDED.

    fut.result() with NO timeout froze the VOICE LOOP indefinitely whenever the
    Nest cloud hung (a 5xx that never closes, a dead TCP connection): these
    actions run ON the voice thread, so JARVIS simply stopped responding until
    the process was killed. Every sibling smart-home skill (sh_hue, sh_ecobee, …)
    already bounds this — sh_nest was the only one that did not (2026-07-14
    audit). On timeout: cancel the future and raise TimeoutError, which the
    callers already degrade into an honest "couldn't reach Nest" reply."""
    loop = _start_loop_thread()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeoutError:
        fut.cancel()
        raise TimeoutError(
            f"Nest did not respond within {timeout:.0f}s") from None


# ── client ─────────────────────────────────────────────────────────
async def _build_client_async() -> Any:
    sdm = _sdm()
    if sdm is None:
        return None
    _auth, _dm, _gns = sdm
    cfg = _read_config()
    if not all(cfg.get(k) for k in ("project_id", "client_id",
                                     "client_secret", "refresh_token")):
        return None
    try:
        try:
            import aiohttp  # type: ignore
        except Exception:
            return None
        # google-nest-sdm API: AbstractAuth subclass needed in newer versions.
        # We use the low-level Token approach via aiohttp_oauth2 if available;
        # otherwise fall back to the legacy AccessTokenAuth.
        AuthClass = getattr(_auth, "AbstractAuth", None)
        AccessTokenAuth = getattr(_auth, "AccessTokenAuth", None)
        session = aiohttp.ClientSession()
        if AccessTokenAuth is not None:
            # Exchange refresh token for access token via Google OAuth.
            try:
                import requests  # type: ignore
                r = requests.post(
                    "https://www.googleapis.com/oauth2/v4/token",
                    data={
                        "client_id":     cfg["client_id"],
                        "client_secret": cfg["client_secret"],
                        "refresh_token": cfg["refresh_token"],
                        "grant_type":    "refresh_token",
                    }, timeout=10,
                )
                r.raise_for_status()
                access_token = r.json().get("access_token")
            except Exception as e:
                await session.close()
                print(f"  [sh-nest] token exchange failed: {e}")
                return None
            client = AccessTokenAuth(session, access_token,
                                      "https://smartdevicemanagement.googleapis.com/v1")
            return (client, cfg["project_id"], session)
        await session.close()
        return None
    except Exception as e:
        print(f"  [sh-nest] client init failed: {e}")
        return None


def _get_client():
    """Returns (auth_client, project_id, session) or None."""
    with _lock:
        c = _state["client"]
        if c is not None and (time.monotonic() - _state["fetched_at"]) < _DEV_TTL:
            return c
    # Build under a dedicated lock so two concurrent cache-misses don't each
    # open an aiohttp.ClientSession (the loser would be overwritten and leaked).
    with _build_lock:
        # Re-check inside the lock: another caller may have just built one.
        with _lock:
            c = _state["client"]
            fetched = _state["fetched_at"]
        if c is not None and (time.monotonic() - fetched) < _DEV_TTL:
            return c
        old = c  # an expired (client, project_id, session) we must close
        try:
            new = _run_async(_build_client_async())
        except Exception as e:
            print(f"  [sh-nest] client build failed: {e}")
            new = None
        if new is not None:
            # Close the stale session before replacing it so we don't leak it.
            if old is not None:
                try:
                    old_session = old[2]
                    _run_async(old_session.close())
                except Exception as e:
                    print(f"  [sh-nest] stale session close failed: {e}")
            with _lock:
                _state["client"]     = new
                _state["fetched_at"] = time.monotonic()
        return new


# ── public API ────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    sdm = _sdm()
    c = _get_client()
    if sdm is None or c is None:
        return []
    _auth, _dm, _gns = sdm
    client, project_id, _session = c
    try:
        async def _go():
            mgr = _dm.DeviceManager()
            devices = await client.request("get", f"enterprises/{project_id}/devices")
            return devices
        devices = _run_async(_go())
    except Exception as e:
        print(f"  [sh-nest] device list failed: {e}")
        return []
    items = (devices or {}).get("devices", []) if isinstance(devices, dict) else []
    out: list[dict] = []
    for d in items:
        traits = d.get("traits") or {}
        dtype = d.get("type", "")
        is_thermo = "sdm.devices.types.THERMOSTAT" in dtype
        is_cam = "sdm.devices.types.CAMERA" in dtype or "DOORBELL" in dtype
        name = ""
        for label in (d.get("parentRelations") or []):
            if "displayName" in label:
                name = label["displayName"]
                break
        name = name or d.get("name", "").rsplit("/", 1)[-1] or "Nest device"
        out.append({
            "name": name,
            "brand": "Nest",
            "type": "thermostat" if is_thermo else ("camera" if is_cam else "unknown"),
            "capabilities": (["thermostat"] if is_thermo else []) +
                             (["camera"] if is_cam else []),
            "native_id": d.get("name"),
        })
    return out


def _device_id(device: dict) -> str | None:
    nid = device.get("native_id")
    if nid:
        return nid
    name_low = (device.get("name") or "").lower().strip()
    if not name_low:
        return None
    for d in list_devices():
        if (d.get("name") or "").lower().strip() == name_low:
            return d.get("native_id")
    return None


def get_state(device: dict) -> dict:
    c = _get_client()
    if c is None:
        return {"error": "nest client not initialized"}
    client, _project_id, _session = c
    nid = _device_id(device)
    if not nid:
        return {"error": f"nest device '{device.get('name')}' not found"}
    try:
        async def _go():
            return await client.request("get", nid)
        data = _run_async(_go()) or {}
    except Exception as e:
        return {"error": f"nest get_state failed: {e}"}
    traits = data.get("traits") or {}
    temp = ((traits.get("sdm.devices.traits.Temperature") or {})
            .get("ambientTemperatureCelsius"))
    mode = ((traits.get("sdm.devices.traits.ThermostatMode") or {})
            .get("mode"))
    setpoint = (traits.get("sdm.devices.traits.ThermostatTemperatureSetpoint")
                or {})
    return {
        "actual_c": temp,
        "actual_f": (temp * 9 / 5 + 32) if temp is not None else None,
        "mode": mode,
        "heat_set_c": setpoint.get("heatCelsius"),
        "cool_set_c": setpoint.get("coolCelsius"),
    }


def set_state(device: dict, **kwargs) -> dict:
    c = _get_client()
    if c is None:
        return {"error": "nest client not initialized "
                          "(missing OAuth config — run sh_nest_authorize)"}
    client, _project_id, _session = c
    nid = _device_id(device)
    if not nid:
        return {"error": f"nest device '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}
    mode = kwargs.get("mode")
    if kwargs.get("on") is False:
        mode = "OFF"
    if kwargs.get("on") is True and not mode:
        mode = "HEATCOOL"

    async def _set_mode(m: str) -> dict:
        try:
            await client.request(
                "post", f"{nid}:executeCommand",
                json={"command": "sdm.devices.commands.ThermostatMode.SetMode",
                      "params": {"mode": m}},
            )
            applied["mode"] = m
            return {"ok": True}
        except Exception as e:
            return {"error": f"set_mode failed: {e}"}

    async def _set_setpoint(f: int) -> dict:
        c_ = (f - 32) * 5 / 9
        # Use the heat command in heat-only contexts; for HEATCOOL we'd
        # need a range. Keep it simple — apply both.
        cmd = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetHeat"
        try:
            await client.request(
                "post", f"{nid}:executeCommand",
                json={"command": cmd, "params": {"heatCelsius": c_}},
            )
            applied["temperature"] = f
            return {"ok": True}
        except Exception as e:
            return {"error": f"set_setpoint failed: {e}"}

    async def _go() -> dict:
        if mode:
            r = await _set_mode(mode)
            if "error" in r:
                return r
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            r = await _set_setpoint(int(kwargs["temperature"]))
            if "error" in r:
                return r
        return {"ok": True, "applied": applied}

    try:
        return _run_async(_go())
    except Exception as e:
        return {"error": f"nest set_state failed: {e}", "partial": applied}


def sh_nest_authorize(_: str = "") -> str:
    """One-shot OAuth instructions; the full PKCE dance is interactive
    in a browser. We just print the URL the user has to visit."""
    cfg = _read_config()
    cid = cfg.get("client_id")
    pid = cfg.get("project_id")
    if not cid or not pid:
        return ("Need project_id + client_id in data/sh_nest_config.json "
                "first, sir — see https://developers.google.com/nest/device-access.")
    url = (
        "https://nestservices.google.com/partnerconnections/"
        f"{pid}/auth?redirect_uri=https://www.google.com&"
        f"access_type=offline&prompt=consent&client_id={cid}&"
        "response_type=code&scope=https://www.googleapis.com/auth/sdm.service"
    )
    print(f"  [sh-nest] Open this URL in a browser to authorize:\n  {url}")
    print("  After accepting, copy the `code=` value from the redirect and"
          " exchange it for tokens via the standard Google OAuth POST. Store"
          " the refresh_token in data/sh_nest_config.json.")
    return "Nest authorization URL printed to console, sir."


def register(actions: dict) -> None:
    actions["nest_list_devices"] = lambda _="": f"{len(list_devices())} Nest device(s)."
    actions["nest_authorize"]    = sh_nest_authorize
