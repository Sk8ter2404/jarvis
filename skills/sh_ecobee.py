"""
sh_ecobee — Ecobee thermostat controller skill.

Wraps `pyecobee` so `core.smart_home_router` can read and set Ecobee
thermostat setpoints, hold modes, and HVAC mode (heat/cool/auto/off).

Initial OAuth dance is interactive: the user has to register a free
developer app at https://www.ecobee.com/developers/, paste the API key
into `data/sh_ecobee_config.json` (`{"api_key": "..."}`), and complete a
one-time PIN authorization. `pyecobee` persists refresh tokens on disk
so subsequent calls run unattended.

Uniform set_state kwargs honored:
    temperature : int (°F, 50..90 typical range)
    mode        : 'heat' | 'cool' | 'auto' | 'off'
    on          : bool   (True → 'auto', False → 'off')

If `pyecobee` isn't installed or no API key is configured, every call
returns a clean error dict — the router will fall back to the Alexa
cookie path.
"""
from __future__ import annotations

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
_CONFIG_PATH = os.path.join(_DATA_DIR, "sh_ecobee_config.json")
_TOKEN_PATH  = os.path.join(_DATA_DIR, "sh_ecobee_tokens.json")

_lock = threading.Lock()
_state: dict[str, Any] = {"service": None, "fetched_at": 0.0,
                           "thermostats": []}
_SERVICE_TTL = 300.0


def _pyecobee():
    try:
        import pyecobee  # type: ignore
        return pyecobee
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
    return _pyecobee() is not None and bool(_read_config().get("api_key"))


# ── service init ───────────────────────────────────────────────────
def _save_tokens(service: Any) -> None:
    try:
        # Atomic write so a crash/power-loss mid-refresh can't truncate the
        # token file and force a full PIN re-authorization (matches sh_ring).
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(_TOKEN_PATH, {
            "access_token":   getattr(service, "access_token", ""),
            "refresh_token":  getattr(service, "refresh_token", ""),
            "authorization_token": getattr(service, "authorization_token", ""),
        })
    except Exception as e:
        print(f"  [sh-ecobee] token save failed: {e}")


def _load_tokens() -> dict:
    if not os.path.exists(_TOKEN_PATH):
        return {}
    try:
        with open(_TOKEN_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _run_with_timeout(fn, timeout: float = 8.0) -> dict:
    """Run a blocking callable in a daemon thread with a hard timeout.

    Returns {"ok", "error", "timed_out"}. `refresh_tokens()` is a blocking
    HTTPS POST with no timeout of its own; on the voice dispatch thread a
    stalled exchange would hang the turn indefinitely, so we cap it here.
    A timed-out worker is left as a daemon thread (it holds no resource we
    must reclaim) and the caller retries on the next call."""
    result: dict[str, Any] = {"ok": False, "error": None, "timed_out": False}

    def _worker() -> None:
        try:
            fn()
            result["ok"] = True
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_worker, name="sh_ecobee-refresh", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["timed_out"] = True
    return result


def _get_service() -> Any:
    pyecobee = _pyecobee()
    if pyecobee is None:
        return None
    with _lock:
        svc = _state["service"]
        if svc is not None and (time.monotonic() - _state["fetched_at"]) < _SERVICE_TTL:
            return svc
    cfg = _read_config()
    api_key = cfg.get("api_key")
    if not api_key:
        return None
    tokens = _load_tokens()
    try:
        EcobeeService = getattr(pyecobee, "EcobeeService", None)
        if EcobeeService is None:
            return None
        svc = EcobeeService(
            thermostat_name=cfg.get("thermostat_name", ""),
            application_key=api_key,
            access_token=tokens.get("access_token") or None,
            refresh_token=tokens.get("refresh_token") or None,
            authorization_token=tokens.get("authorization_token") or None,
        )
        # If we have a refresh token, exchange it. If we don't, the user
        # needs to run the interactive PIN flow (separate action).
        if svc.refresh_token:
            res = _run_with_timeout(svc.refresh_tokens, timeout=8.0)
            if res["timed_out"]:
                # Do NOT cache: a stalled refresh shouldn't poison the
                # service for the full TTL — let the next call retry cleanly.
                print("  [sh-ecobee] token refresh timed out after 8s — "
                      "not caching; will retry next call.")
                return None
            if res["error"] is not None:
                # Transient refresh failure: don't cache the service as valid
                # (the old code did, so it re-failed every 5 min forever).
                print(f"  [sh-ecobee] token refresh failed: {res['error']} — "
                      "not caching; will retry next call.")
                return None
            _save_tokens(svc)
    except Exception as e:
        print(f"  [sh-ecobee] service init failed: {e}")
        return None
    with _lock:
        _state["service"]    = svc
        _state["fetched_at"] = time.monotonic()
    return svc


def _fetch_thermostats(svc: Any) -> list[Any]:
    pyecobee = _pyecobee()
    if pyecobee is None or svc is None:
        return []
    try:
        Selection = getattr(pyecobee, "Selection")
        SelectionType = getattr(pyecobee, "SelectionType")
        selection = Selection(
            selection_type=SelectionType.REGISTERED.value,
            selection_match="",
            include_runtime=True,
            include_settings=True,
            include_program=True,
            include_events=True,
        )
        resp = svc.request_thermostats(selection)
        thermostats = getattr(resp, "thermostat_list", None) or []
        return list(thermostats)
    except Exception as e:
        print(f"  [sh-ecobee] thermostat fetch failed: {e}")
        return []


# ── public API ────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    svc = _get_service()
    if svc is None:
        return []
    out: list[dict] = []
    for t in _fetch_thermostats(svc):
        out.append({
            "name": getattr(t, "name", "Ecobee"),
            "brand": "ecobee",
            "type": "thermostat",
            "capabilities": ["thermostat", "temperature"],
            "native_id": getattr(t, "identifier", None),
            "model": getattr(t, "model_number", ""),
        })
    return out


def _match_thermostat(svc: Any, device: dict) -> Any:
    name = (device.get("name") or "").strip().lower()
    nid  = device.get("native_id")
    for t in _fetch_thermostats(svc):
        if nid and getattr(t, "identifier", None) == nid:
            return t
        if name and (getattr(t, "name", "") or "").lower() == name:
            return t
    return None


def get_state(device: dict) -> dict:
    svc = _get_service()
    if svc is None:
        return {"error": "ecobee service not initialized"}
    t = _match_thermostat(svc, device)
    if t is None:
        return {"error": f"thermostat '{device.get('name')}' not found"}
    try:
        runtime = getattr(t, "runtime", None)
        settings = getattr(t, "settings", None)
        # Ecobee temps are tenths of a degree F.
        actual = getattr(runtime, "actual_temperature", None)
        cool_hold = getattr(runtime, "desired_cool", None)
        heat_hold = getattr(runtime, "desired_heat", None)
        hvac_mode = getattr(settings, "hvac_mode", "")
        return {
            "actual_f":  (actual / 10.0) if actual is not None else None,
            "cool_set":  (cool_hold / 10.0) if cool_hold is not None else None,
            "heat_set":  (heat_hold / 10.0) if heat_hold is not None else None,
            "mode":      hvac_mode,
        }
    except Exception as e:
        return {"error": f"ecobee read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    svc = _get_service()
    if svc is None:
        return {"error": "ecobee service not initialized "
                          "(missing API key or unauthorized — run ecobee_authorize)"}
    pyecobee = _pyecobee()
    t = _match_thermostat(svc, device)
    if t is None:
        return {"error": f"thermostat '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}
    nid = getattr(t, "identifier", None)
    Selection      = getattr(pyecobee, "Selection")
    SelectionType  = getattr(pyecobee, "SelectionType")
    selection = Selection(selection_type=SelectionType.THERMOSTATS.value,
                          selection_match=nid)

    # HVAC mode change
    mode = kwargs.get("mode")
    if kwargs.get("on") is False:
        mode = "off"
    if kwargs.get("on") is True and not mode:
        mode = "auto"
    if mode:
        try:
            # pyecobee (sfanous) has no set_hvac_mode; mode changes go
            # through update_thermostats with a Thermostat(settings=...) diff.
            Thermostat = getattr(pyecobee, "Thermostat")
            Settings   = getattr(pyecobee, "Settings")
            svc.update_thermostats(
                selection,
                thermostat=Thermostat(settings=Settings(hvac_mode=mode)),
            )
            applied["mode"] = mode
        except Exception as e:
            return {"error": f"ecobee set_hvac_mode failed: {e}",
                    "partial": applied}

    # Temperature hold — apply as a manual hold targeting both setpoints.
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        target = int(kwargs["temperature"])
        try:
            # pyecobee set_hold takes whole degrees F (validated 45-120) and
            # selection must be passed by keyword — temps are positional-first.
            svc.set_hold(cool_hold_temp=target, heat_hold_temp=target,
                         selection=selection)
            applied["temperature"] = target
        except Exception as e:
            return {"error": f"ecobee set_hold failed: {e}", "partial": applied}

    return {"ok": True, "applied": applied}


# ── two-step non-blocking PIN flow ────────────────────────────────
def _do_request_pin() -> tuple[str, str]:
    """Shared: returns (pin, error). On success error is empty string and
    the authorization_token is persisted to _TOKEN_PATH so a later call to
    ecobee_complete_setup can exchange it for access+refresh tokens."""
    pyecobee = _pyecobee()
    if pyecobee is None:
        return "", "pyecobee not installed, sir — `pip install pyecobee` first."
    cfg = _read_config()
    if not cfg.get("api_key"):
        return "", ("Need an API key first, sir. Create a free Ecobee developer "
                    "app at ecobee.com/developers and put it in "
                    "data/sh_ecobee_config.json as {\"api_key\": \"...\"}.")
    EcobeeService = getattr(pyecobee, "EcobeeService", None)
    if EcobeeService is None:
        return "", "pyecobee.EcobeeService missing — pip upgrade pyecobee."
    try:
        svc = EcobeeService(thermostat_name="", application_key=cfg["api_key"])
    except Exception as e:
        return "", f"Ecobee service construction failed: {e}"
    try:
        resp = svc.authorize()
        pin = getattr(resp, "ecobee_pin", "?")
    except Exception as e:
        return "", f"Ecobee authorize call failed: {e}"
    # Persist the authorization_token so ecobee_complete_setup can pick it up.
    _save_tokens(svc)
    return pin, ""


def ecobee_request_pin(_: str = "") -> str:
    """Voice action — non-blocking step 1 of authorization. Prints + returns
    the PIN immediately. User pastes it into the Ecobee web portal, then
    triggers ecobee_complete_setup to exchange for access/refresh tokens."""
    pin, err = _do_request_pin()
    if err:
        return err
    print()
    print(f"  [sh-ecobee] Open https://www.ecobee.com/consumerportal/")
    print(f"             -> My Apps -> Add Application -> enter PIN: {pin}")
    print("              Then say 'ecobee complete setup' to finish.")
    return (f"Ecobee PIN: {pin}, sir. Add it under My Apps at "
            f"ecobee.com/consumerportal, then say 'ecobee complete setup'.")


def ecobee_complete_setup(_: str = "") -> str:
    """Voice action — non-blocking step 2. Exchanges the cached
    authorization_token for access + refresh tokens. Run after the user has
    pasted the PIN from ecobee_request_pin into the Ecobee web portal."""
    pyecobee = _pyecobee()
    if pyecobee is None:
        return "pyecobee not installed, sir — `pip install pyecobee` first."
    cfg = _read_config()
    api_key = cfg.get("api_key")
    if not api_key:
        return ("Need an API key first, sir — set it in "
                "data/sh_ecobee_config.json.")
    tokens = _load_tokens()
    auth_token = tokens.get("authorization_token")
    if not auth_token:
        return ("No pending authorization, sir — run ecobee_request_pin "
                "first to get a fresh PIN.")
    EcobeeService = getattr(pyecobee, "EcobeeService", None)
    if EcobeeService is None:
        return "pyecobee.EcobeeService missing — pip upgrade pyecobee."
    try:
        svc = EcobeeService(
            thermostat_name=cfg.get("thermostat_name", ""),
            application_key=api_key,
            authorization_token=auth_token,
        )
        svc.request_tokens()
    except Exception as e:
        return (f"Ecobee token request failed: {e}. The PIN may have "
                "expired (Ecobee gives ~9 minutes) or you may not have "
                "added the app yet — run ecobee_request_pin again.")
    _save_tokens(svc)
    with _lock:
        _state["service"]    = svc
        _state["fetched_at"] = time.monotonic()
    return "Ecobee authorized, sir — tokens persisted."


def ecobee_authorize(_: str = "") -> str:
    """Voice-action alias: explains the two-step flow without blocking."""
    return ("Ecobee authorization is two steps, sir: say 'ecobee request "
            "pin' to get a PIN, paste it into ecobee.com/consumerportal "
            "under My Apps, then say 'ecobee complete setup' to finish. "
            "For an interactive prompt instead, run "
            "`python -m skills.sh_ecobee` in the JARVIS console.")


def _run_ecobee_wizard_interactive() -> str:
    """Full interactive PIN flow with input() prompt. ONLY call from the
    __main__ guard — never from voice context (would freeze the audio
    loop)."""
    pin, err = _do_request_pin()
    if err:
        return err
    print()
    print(f"  [sh-ecobee] Open https://www.ecobee.com/consumerportal/")
    print(f"             -> My Apps -> Add Application -> enter PIN: {pin}")
    print("              Then press <enter> here to exchange the PIN for tokens.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return "Authorization cancelled, sir."
    return ecobee_complete_setup()


def register(actions: dict) -> None:
    actions["ecobee_list_devices"]   = lambda _="": f"{len(list_devices())} Ecobee thermostat(s) configured."
    actions["ecobee_request_pin"]    = ecobee_request_pin
    actions["ecobee_complete_setup"] = ecobee_complete_setup
    actions["ecobee_authorize"]      = ecobee_authorize


if __name__ == "__main__":
    # Interactive PIN flow. Voice actions route here via the alias hint to
    # keep input() off the main audio loop.
    out = _run_ecobee_wizard_interactive()
    print()
    print(out)
