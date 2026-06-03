"""
sh_ring — Ring camera / doorbell controller skill.

Wraps `ring_doorbell` for read-only status (battery, last motion, last
ding) plus the small writable surface Ring exposes (siren, lights on
chimes/floodlights, in-home chime on/off). Ring intentionally doesn't
expose a "trigger doorbell" API, so set_state's actionable verbs are
limited to lights, sirens and chime mute.

Authentication uses Ring's username/password flow → a refresh token that
`ring_doorbell` persists in `data/sh_ring_token.json`. The first sign-in
requires 2FA; subsequent uses just rotate the refresh token.

Uniform set_state kwargs honored (subset, gated by device capabilities):
    on:    bool   (chime / floodlight devices → light on/off)
    siren: bool   (sound the siren on a stickup cam / floodlight)
    chime: bool   (chime device → enable/disable doorbell announce)

Any unsupported kwarg is silently ignored; the action returns the keys
that actually landed.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data")
_TOKEN_PATH  = os.path.join(_DATA_DIR, "sh_ring_token.json")
_CONFIG_PATH = os.path.join(_DATA_DIR, "sh_ring_config.json")

_lock = threading.Lock()
_state: dict[str, Any] = {"ring": None, "fetched_at": 0.0,
                           "devices_cache": {}}
_RING_TTL = 300.0


def _ring_doorbell():
    try:
        import ring_doorbell  # type: ignore
        return ring_doorbell
    except Exception:
        return None


def _read_token() -> dict:
    if not os.path.exists(_TOKEN_PATH):
        return {}
    try:
        with open(_TOKEN_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_token(tok: dict) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(_TOKEN_PATH, tok)
    except Exception as e:
        print(f"  [sh-ring] token save failed: {e}")


def _read_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def is_available() -> bool:
    return _ring_doorbell() is not None and bool(_read_token())


def _run_with_timeout(fn, timeout: float = 8.0) -> dict:
    """Run a blocking callable in a daemon thread with a hard timeout.

    Returns {"ok", "error", "timed_out"}. `ring.update_data()` is a blocking
    HTTPS fetch with no timeout of its own; on the voice dispatch thread a
    stalled request would hang the turn, so we cap it. A timed-out worker is
    left as a daemon thread and the caller retries on the next call."""
    result: dict[str, Any] = {"ok": False, "error": None, "timed_out": False}

    def _worker() -> None:
        try:
            fn()
            result["ok"] = True
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_worker, name="sh_ring-update", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["timed_out"] = True
    return result


# ── client ────────────────────────────────────────────────────────
def _get_ring() -> Any:
    ring_doorbell = _ring_doorbell()
    if ring_doorbell is None:
        return None
    with _lock:
        r = _state["ring"]
        if r is not None and (time.monotonic() - _state["fetched_at"]) < _RING_TTL:
            return r
    token = _read_token()
    if not token:
        return None
    try:
        Auth = getattr(ring_doorbell, "Auth", None)
        Ring = getattr(ring_doorbell, "Ring", None)
        if Auth is None or Ring is None:
            return None
        auth = Auth("JARVIS/1.0", token, _save_token)
        ring = Ring(auth)
        res = _run_with_timeout(ring.update_data, timeout=8.0)
        if res["timed_out"]:
            print("  [sh-ring] update_data timed out after 8s — "
                  "not caching; will retry next call.")
            return None
        if res["error"] is not None:
            print(f"  [sh-ring] auth failed: {res['error']}")
            return None
    except Exception as e:
        print(f"  [sh-ring] auth failed: {e}")
        return None
    with _lock:
        _state["ring"]       = ring
        _state["fetched_at"] = time.monotonic()
    return ring


def _enumerate(ring: Any) -> dict[str, Any]:
    """Return a {device_id → device_obj} dict spanning every device type
    ring_doorbell exposes (doorbells, stickup cams, chimes, floodlights)."""
    out: dict[str, Any] = {}
    if ring is None:
        return out
    try:
        devices = ring.devices()
    except Exception as e:
        print(f"  [sh-ring] device enum failed: {e}")
        return out
    if isinstance(devices, dict):
        for kind, items in devices.items():
            for it in items or []:
                try:
                    out[str(getattr(it, "id", id(it)))] = it
                except Exception:
                    pass
    return out


# ── public API ────────────────────────────────────────────────────
def list_devices() -> list[dict]:
    ring = _get_ring()
    if ring is None:
        return []
    out: list[dict] = []
    for did, d in _enumerate(ring).items():
        try:
            name = getattr(d, "name", "") or getattr(d, "id_str", "") or did
            kind = getattr(d, "family", "") or d.__class__.__name__.lower()
            caps = ["camera"] if "doorbell" in kind.lower() or "cam" in kind.lower() \
                else (["chime"] if "chime" in kind.lower() else [])
            if hasattr(d, "lights"):
                caps.append("on_off")
            if hasattr(d, "siren"):
                caps.append("siren")
            out.append({
                "name": name,
                "brand": "Ring",
                "type": "camera" if "camera" in caps else "chime",
                "capabilities": sorted(set(caps)),
                "native_id": did,
            })
        except Exception:
            continue
    return out


def _match(ring: Any, device: dict) -> Any:
    nid = device.get("native_id")
    devices = _enumerate(ring)
    if nid and nid in devices:
        return devices[nid]
    name_low = (device.get("name") or "").lower().strip()
    for d in devices.values():
        if (getattr(d, "name", "") or "").lower().strip() == name_low:
            return d
    return None


def get_state(device: dict) -> dict:
    ring = _get_ring()
    if ring is None:
        return {"error": "ring client not authorized"}
    d = _match(ring, device)
    if d is None:
        return {"error": f"ring device '{device.get('name')}' not found"}
    try:
        return {
            "battery": getattr(d, "battery_life", None),
            "online":  getattr(d, "connection_status", None),
            "last_motion": getattr(d, "last_motion", None) if hasattr(d, "last_motion") else None,
            "lights":  getattr(d, "lights", None) if hasattr(d, "lights") else None,
        }
    except Exception as e:
        return {"error": f"ring state read failed: {e}"}


def set_state(device: dict, **kwargs) -> dict:
    ring = _get_ring()
    if ring is None:
        return {"error": "ring client not authorized "
                          "(run ring_authorize to create data/sh_ring_token.json)"}
    d = _match(ring, device)
    if d is None:
        return {"error": f"ring device '{device.get('name')}' not found"}

    applied: dict[str, Any] = {}
    try:
        # 'on' on a floodlight/chime → toggle the light or chime motion alert.
        if "on" in kwargs and kwargs["on"] is not None:
            if hasattr(d, "lights"):
                try:
                    d.lights = "on" if kwargs["on"] else "off"
                    applied["lights"] = bool(kwargs["on"])
                except Exception as e:
                    return {"error": f"ring light toggle failed: {e}"}
        if kwargs.get("siren") is not None and hasattr(d, "siren"):
            try:
                d.siren = "on" if kwargs["siren"] else "off"
                applied["siren"] = bool(kwargs["siren"])
            except Exception as e:
                return {"error": f"ring siren failed: {e}",
                        "partial": applied}
        if kwargs.get("chime") is not None and hasattr(d, "existing_doorbell_type_enabled"):
            try:
                # No clean enable/disable; closest is mute via existence_alerts.
                # Gate on the attribute actually used below — the old `test_sound`
                # gate could be absent on a device that *does* expose
                # existing_doorbell_type_enabled, silently dropping the chime set.
                d.existing_doorbell_type_enabled = bool(kwargs["chime"])
                applied["chime"] = bool(kwargs["chime"])
            except Exception:
                pass
    except Exception as e:
        return {"error": f"ring set_state failed: {e}", "partial": applied}

    if not applied:
        return {"error": "ring device doesn't expose any of the requested controls",
                "requested": list(kwargs.keys())}
    return {"ok": True, "applied": applied}


_CLI_HINT = (
    "Sir, Ring sign-in needs an interactive terminal for the password "
    "(and 2FA code on first use). Please run "
    "`python -m skills.sh_ring` in the JARVIS console, or pass credentials "
    "inline as `ring_authorize, email|password` (add `|code` for 2FA)."
)


def _do_fetch_token(email: str, password: str, code: str = "") -> str:
    """Shared non-interactive sign-in. Returns success/error message."""
    ring_doorbell = _ring_doorbell()
    if ring_doorbell is None:
        return "ring_doorbell not installed, sir — `pip install ring_doorbell`."
    Auth = getattr(ring_doorbell, "Auth", None)
    if Auth is None:
        return "ring_doorbell.Auth missing — upgrade the library."
    try:
        auth = Auth("JARVIS/1.0", None, _save_token)
    except Exception as e:
        return f"Ring auth construction failed: {e}"
    try:
        if code:
            auth.fetch_token(email, password, code)
        else:
            auth.fetch_token(email, password)
    except Exception as e:
        if code:
            return f"Ring sign-in failed: {e}"
        return ("2FA_REQUIRED: " + str(e))
    return "Ring authorized, sir — token persisted to data/sh_ring_token.json."


def ring_authorize(arg: str = "") -> str:
    """Voice action — non-blocking. Accepts credentials inline as
    `email|password` (add `|2fa_code` for 2FA-protected accounts); with no
    arg, falls back to cached config or returns a CLI hint so input()
    never freezes the voice loop."""
    ring_doorbell = _ring_doorbell()
    if ring_doorbell is None:
        return "ring_doorbell not installed, sir — `pip install ring_doorbell`."

    parts = [p.strip() for p in (arg or "").split("|")] if "|" in (arg or "") else []
    email = password = code = ""
    if len(parts) >= 2:
        email, password = parts[0], parts[1]
        if len(parts) >= 3:
            code = parts[2]
    else:
        cfg = _read_config()
        email    = (cfg.get("email") or "").strip()
        password = (cfg.get("password") or "").strip()
        code     = (cfg.get("code") or "").strip()

    if not email or not password:
        return _CLI_HINT

    result = _do_fetch_token(email, password, code)
    if result.startswith("2FA_REQUIRED:"):
        return ("Ring requires a 2FA code, sir — re-run "
                "`ring_authorize, email|password|123456` with the code from "
                "your phone, or run `python -m skills.sh_ring` for an "
                "interactive prompt.")
    return result


def _run_ring_wizard_interactive(arg: str = "") -> str:
    """Full interactive sign-in with input()/getpass(). ONLY call from the
    __main__ guard — never from voice context (would freeze the audio
    loop)."""
    ring_doorbell = _ring_doorbell()
    if ring_doorbell is None:
        return "ring_doorbell not installed, sir — `pip install ring_doorbell`."

    if "|" in (arg or ""):
        parts = [p.strip() for p in arg.split("|")]
        email = parts[0] if len(parts) >= 1 else ""
        password = parts[1] if len(parts) >= 2 else ""
    else:
        cfg = _read_config()
        email    = (cfg.get("email") or "").strip()
        password = (cfg.get("password") or "").strip()

    if not email or not password:
        try:
            print()
            email = input("    Ring email: ").strip()
            import getpass
            password = getpass.getpass("    Ring password: ")
        except (EOFError, KeyboardInterrupt):
            return "No credentials provided, sir."
        if not email or not password:
            return "No credentials provided, sir."

    result = _do_fetch_token(email, password)
    if result.startswith("2FA_REQUIRED:"):
        try:
            code = input("    Ring 2FA code: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "2FA prompt cancelled."
        if not code:
            return "No 2FA code provided, sir."
        result = _do_fetch_token(email, password, code)
    return result


def register(actions: dict) -> None:
    actions["ring_list_devices"] = lambda _="": f"{len(list_devices())} Ring device(s)."
    actions["ring_authorize"]    = ring_authorize


if __name__ == "__main__":  # pragma: no cover - interactive sign-in CLI entry (input()/getpass()); run by hand, not under unittest
    # Interactive sign-in path. Voice actions route here via the CLI
    # hint to keep input() / getpass() off the main audio loop.
    import sys
    cli_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    out = _run_ring_wizard_interactive(cli_arg)
    print()
    print(out)
