"""
network_deco — TP-Link Deco mesh router integration.

Provides the network-awareness layer the smart-home discovery skill needs
and unlocks voice commands like:

    'JARVIS, who's on my WiFi?'
    'JARVIS, is the printer online?'
    'JARVIS, what's using all the bandwidth?'
    'JARVIS, kick the guest network'

It authenticates to the user's TP-Link Deco mesh router using
`tplinkrouterc6u` (https://pypi.org/project/tplinkrouterc6u/), pulls the
full client list, Deco-node topology, guest-network and parental-control
state, and caches a snapshot at `data/deco_network.json` refreshed every
5 minutes by a background thread.

Config
------
    DECO_HOST     — defaults to 192.168.1.1 (the user's confirmed Deco
                    LAN IP). On startup we ARP-scan briefly to verify a
                    Deco-OUI device actually answers at that address and
                    fall back to other matches if not.
    DECO_PASSWORD — read from the env var of the same name. If unset we
                    look in data/deco_config.json (written once by the
                    user). The skill is a clean no-op until a password
                    is set.

Persistence
-----------
    data/deco_config.json   — {host, password} (created on first run)
    data/deco_network.json  — last good snapshot (clients, topology,
                              guest_enabled, parental_profiles, summary)

Optional dependency
-------------------
    tplinkrouterc6u  — pip install tplinkrouterc6u
                       Falls back to a graceful 'not installed' message
                       and a Playwright-scrape hint (research-9's
                       browser_agent skill at
                       http://192.168.1.1/webpages/index.html#networkMap
                       can replace the API path on unsupported firmware).

Registered actions
------------------
    who_is_on_wifi          — spoken roll-call of currently-online clients
    network_clients         — alias
    is_printer_online       — checks the cached snapshot for the Bambu IP
                              (192.168.1.65) or any host whose name/mac
                              looks like a printer
    is_device_online        — generic '<name>' lookup
    network_usage           — top bandwidth users (last poll)
    bandwidth_hogs          — alias
    kick_guest_network      — disables the guest SSID
    enable_guest_network    — re-enables it
    deco_topology           — Deco node parent/child layout
    deco_status             — one-line health summary
    deco_refresh            — force-refresh the snapshot

All public functions degrade gracefully — missing library, missing
password, or unreachable router each produce informative dicts/strings
rather than raising.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any

try:
    from core.atomic_io import _atomic_write_json
except Exception:  # core may be importable late at boot
    import tempfile

    def _atomic_write_json(path: str, data: Any, *, indent: int = 2) -> None:  # type: ignore
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise


_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR     = os.path.join(_PROJECT_DIR, "data")
_CONFIG_PATH  = os.path.join(_DATA_DIR, "deco_config.json")
_SNAPSHOT_PATH = os.path.join(_DATA_DIR, "deco_network.json")

DECO_HOST_DEFAULT = os.environ.get("DECO_HOST", "192.168.1.1")
# Subnet to ARP-scan for Deco-OUI nodes — derived from the resolved host
# (env DECO_SUBNET_PREFIX overrides) so it follows whatever LAN you're on.
DECO_SUBNET_PREFIX = os.environ.get("DECO_SUBNET_PREFIX") or DECO_HOST_DEFAULT.rsplit(".", 1)[0] + "."
POLL_INTERVAL_SECONDS = 5 * 60       # cache topology snapshots every 5m
INITIAL_DELAY_SECONDS = 15           # let the rest of JARVIS settle first

# OUIs that identify TP-Link Deco mesh hardware. Used by the boot-time
# ARP verification so we can fall back to whatever Deco IP is actually
# alive if the user changed their subnet without updating config.
_DECO_OUIS = {
    "50C7BF",   # TP-Link
    "B0BE76",   # TP-Link
    "1C61B4",   # TP-Link Deco
    "501A59",   # TP-Link
    "B04E26",   # TP-Link
    "AC84C6",   # TP-Link
    "9C5322",   # TP-Link
}

# Heuristics for is_printer_online. If none of the user's snapshot
# devices match by IP or MAC we fall through to substring matching on
# hostname. The Bambu H2D is at 192.168.1.65 per the user's notes.
_PRINTER_IPS = {ip for ip in (os.environ.get("BAMBU_PRINTER_IP", "").strip(),) if ip}
_PRINTER_NAME_HINTS = ("bambu", "printer", "hp ", "epson", "canon",
                       "brother", "x1c", "x1-c", "h2d", "p1s", "p1p")


_lock = threading.Lock()
_state: dict[str, Any] = {
    "snapshot":        None,
    "snapshot_at":     0.0,
    "router_handle":   None,
    "router_host":     None,
    "router_class":    None,
    "auth_ok":         False,
    "last_error":      None,
    "missing_dep":     False,
}
_stop_evt = threading.Event()
_poll_thread: list[threading.Thread] = []


_JSON_SCALARS = (str, int, float, bool, type(None))


def _json_sanitize(obj: Any, _seen: set[int] | None = None) -> Any:
    """Return a copy of `obj` containing only JSON-safe values.

    Non-scalar values (e.g. aiohttp.Connection refs leaking out of the
    tplinkrouterc6u response objects) are coerced via str() so the
    snapshot dict structure is preserved for downstream consumers.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int, float)):
        return obj
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return f"<circular {type(obj).__name__}>"
    _seen.add(oid)
    try:
        if isinstance(obj, dict):
            return {
                (k if isinstance(k, str) else str(k)): _json_sanitize(v, _seen)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [_json_sanitize(v, _seen) for v in obj]
        try:
            return str(obj)
        except Exception:
            return f"<unserialisable {type(obj).__name__}>"
    finally:
        _seen.discard(oid)


# ── dependency import ──────────────────────────────────────────────
def _tplink():
    """Lazy import of tplinkrouterc6u. Returns the top-level module or
    None if the dep isn't installed."""
    try:
        import tplinkrouterc6u  # type: ignore
        return tplinkrouterc6u
    except Exception:
        return None


def is_available() -> bool:
    if _tplink() is None:
        return False
    pw = _password()
    return bool(pw)


# ── config ─────────────────────────────────────────────────────────
def _read_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_config(cfg: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    _atomic_write_json(_CONFIG_PATH, cfg)


# ── credential encryption (Windows DPAPI) ──────────────────────────────
# The router admin password used to sit in data/deco_config.json as plaintext
# (2026-05-30 audit). DPAPI (CryptProtectData) encrypts it bound to THIS
# Windows user + machine, so the ciphertext is useless if the file is copied
# elsewhere, and the plaintext never touches disk again. Falls back gracefully
# to plaintext if pywin32 is unavailable, so the skill never hard-breaks.
import base64 as _base64


def _dpapi_encrypt(plaintext: str) -> str | None:
    """Encrypt with Windows DPAPI; return base64 ciphertext, or None if DPAPI
    isn't available (caller then keeps plaintext as a last resort)."""
    if not plaintext:
        return None
    try:
        import win32crypt
        blob = win32crypt.CryptProtectData(
            plaintext.encode("utf-8"), "jarvis-deco", None, None, None, 0)
        return _base64.b64encode(blob).decode("ascii")
    except Exception:
        return None


def _dpapi_decrypt(b64: str) -> str | None:
    """Decrypt DPAPI base64 ciphertext back to plaintext, or None on failure."""
    if not b64:
        return None
    try:
        import win32crypt
        blob = _base64.b64decode(b64.encode("ascii"))
        _desc, data = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return data.decode("utf-8")
    except Exception:
        return None


def _password() -> str | None:
    # 1) Env var wins (never persisted).
    pw = os.environ.get("DECO_PASSWORD") or ""
    if pw:
        return pw
    cfg = _read_config()
    # 2) Encrypted form (preferred on-disk representation).
    enc = cfg.get("password_dpapi")
    if enc:
        dec = _dpapi_decrypt(enc)
        if dec:
            return dec
        # Ciphertext present but undecryptable (copied from another machine /
        # user, or pywin32 missing) — fall through to any plaintext.
    # 3) Legacy plaintext — use it, but AUTO-MIGRATE to encrypted so it stops
    #    living on disk in the clear. Best-effort: a failed migration just
    #    leaves the plaintext as-is rather than breaking the skill.
    plain = cfg.get("password")
    if plain:
        try:
            enc_new = _dpapi_encrypt(plain)
            if enc_new:
                cfg.pop("password", None)
                cfg["password_dpapi"] = enc_new
                _write_config(cfg)
                print("  [deco] migrated plaintext password to DPAPI-encrypted "
                      "store (data/deco_config.json)")
        except Exception as _e:
            print(f"  [deco] password encryption migration skipped: {_e}")
        return plain
    return None


def _host() -> str:
    cfg = _read_config()
    return (os.environ.get("DECO_HOST")
            or cfg.get("host")
            or DECO_HOST_DEFAULT)


# ── ARP helpers ────────────────────────────────────────────────────
_ARP_LINE = re.compile(
    r"\s*([0-9]+(?:\.[0-9]+){3})\s+([0-9A-Fa-f-:]{11,17})\s+(\w+)"
)


def _arp_table() -> list[dict]:
    """Run `arp -a` on Windows and return [{ip, mac, oui}, ...].
    Decodes utf-8 first, then cp1252 if the interface name has non-ASCII."""
    try:
        raw = subprocess.check_output(["arp", "-a"], stderr=subprocess.STDOUT,
                                      timeout=8,
                                      creationflags=(subprocess.CREATE_NO_WINDOW
                                                     if sys.platform == "win32"
                                                     else 0))
    except Exception:
        return []
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            txt = raw.decode("cp1252")
        except Exception:
            return []
    out: list[dict] = []
    for line in txt.splitlines():
        m = _ARP_LINE.match(line)
        if not m:
            continue
        ip, mac, kind = m.group(1), m.group(2).upper(), m.group(3)
        # normalise mac: 00-11-22-33-44-55 → 001122334455
        oui = re.sub(r"[^0-9A-F]", "", mac)[:6]
        if not oui:
            continue
        out.append({"ip": ip, "mac": mac, "oui": oui, "kind": kind})
    return out


def _verify_deco_host(host: str) -> str:
    """Confirm `host` actually has a TP-Link OUI on the LAN. If not,
    return the first Deco-OUI IP we see on the configured subnet, or
    the original host if nothing matched."""
    arp = _arp_table()
    for entry in arp:
        if entry["ip"] == host and entry["oui"] in _DECO_OUIS:
            return host
    # `host` didn't match — look for any Deco on the subnet.
    for entry in arp:
        if entry["oui"] in _DECO_OUIS and entry["ip"].startswith(DECO_SUBNET_PREFIX):
            print(f"  [deco] config host {host} didn't ARP-resolve to a "
                  f"Deco OUI; using {entry['ip']} instead.")
            return entry["ip"]
    # Give up; return the original — the auth call will fail loudly.
    return host


# ── router handle ──────────────────────────────────────────────────
def _make_router(host: str, password: str) -> tuple[Any, str | None]:
    """Construct a tplinkrouterc6u router handle for `host`. Returns
    (router, class_name) or (None, None) on failure. Prefers the
    library's helper that auto-selects the right concrete class for the
    user's firmware; falls back to TPLinkDecoRouter / TplinkRouter."""
    tp = _tplink()
    if tp is None:
        return (None, None)
    # Best path: factory helper that picks the right subclass.
    for fname in ("TplinkRouterProvider", "RouterProvider"):
        factory = getattr(tp, fname, None)
        if factory is not None and hasattr(factory, "get_client"):
            try:
                r = factory.get_client(f"http://{host}", password)
                return (r, factory.__name__)
            except Exception as e:
                _state["last_error"] = f"{fname}.get_client: {e}"
    # Direct-class fallbacks, Deco first.
    for cname in ("TPLinkDecoRouter", "TplinkDecoRouter",
                  "TPLinkMRRouter",   "TplinkRouter",
                  "TplinkC6Router",   "TplinkC1200Router"):
        cls = getattr(tp, cname, None)
        if cls is None:
            continue
        try:
            r = cls(f"http://{host}", password)
            return (r, cname)
        except Exception as e:
            _state["last_error"] = f"{cname}(...): {e}"
            continue
    return (None, None)


def _authorize(router: Any) -> bool:
    """Try each common login method name (libraries have renamed it
    repeatedly across versions)."""
    for m in ("authorize", "login", "connect"):
        fn = getattr(router, m, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception as e:
                _state["last_error"] = f"{m}(): {e}"
    return False


def _logout(router: Any) -> None:
    for m in ("logout", "disconnect"):
        fn = getattr(router, m, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


# ── snapshot collection ────────────────────────────────────────────
def _serialise_device(d: Any) -> dict:
    """Normalise a tplinkrouterc6u device-object into a plain dict.
    The library exposes Device/Client/SmartDevice instances with
    overlapping but version-dependent attribute names; we copy whatever
    is present and label it conservatively."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        # Some firmwares return raw dicts; merge them in.
        out.update(d)
    for attr, key in (
        ("hostname",       "name"),
        ("name",           "name"),
        ("alias",          "name"),
        ("ipaddr",         "ip"),
        ("ip",             "ip"),
        ("ip_address",     "ip"),
        ("macaddr",        "mac"),
        ("mac",            "mac"),
        ("mac_address",    "mac"),
        ("signal",         "signal"),
        ("signal_strength","signal"),
        ("rssi",           "signal"),
        ("connection",     "connection"),
        ("conn_type",      "connection"),
        ("link_type",      "connection"),
        ("interface",      "interface"),
        ("parent_mac",     "parent_mac"),
        ("master_mac",     "parent_mac"),
        ("node_mac",       "parent_mac"),
        ("up",             "tx_bytes"),
        ("down",           "rx_bytes"),
        ("up_speed",       "up_speed"),
        ("down_speed",     "down_speed"),
        ("up_total",       "up_total"),
        ("down_total",     "down_total"),
        ("online",         "online"),
        ("active",         "online"),
        ("last_seen",      "last_seen"),
        ("last_active",    "last_seen"),
        ("wire_type",      "wire_type"),
        ("type",           "kind"),
    ):
        val = getattr(d, attr, None)
        if val is not None and key not in out:
            out[key] = val
    if "mac" in out and isinstance(out["mac"], str):
        out["mac"] = out["mac"].upper().replace("-", ":")
    return out


def _summarise_topology(devices: list[dict]) -> dict:
    """Aggregate online/offline counts and discover Deco nodes from the
    client list. Deco nodes report themselves as wired clients with a
    TP-Link OUI."""
    online  = sum(1 for d in devices if d.get("online") not in (False, 0, "0"))
    offline = len(devices) - online
    wireless = sum(1 for d in devices
                   if str(d.get("connection") or d.get("wire_type") or "").lower()
                      in ("wireless", "wifi", "2.4ghz", "5ghz", "6ghz"))
    nodes = []
    for d in devices:
        mac = (d.get("mac") or "").replace(":", "")[:6].upper()
        if mac in _DECO_OUIS:
            nodes.append({
                "mac":  d.get("mac"),
                "ip":   d.get("ip"),
                "name": d.get("name") or "deco-node",
            })
    return {
        "clients_total": len(devices),
        "online":         online,
        "offline":        offline,
        "wireless":       wireless,
        "wired":          max(0, len(devices) - wireless),
        "deco_nodes":     nodes,
    }


def _collect_via(router: Any) -> dict:
    """Pull clients + guest + parental state via whichever method names
    the live library version exposes. Each block is independently
    try/except'd — one missing endpoint must not nuke the snapshot."""
    snap: dict[str, Any] = {
        "fetched_at": time.time(),
        "host":       _state.get("router_host"),
        "class":      _state.get("router_class"),
    }

    # Clients
    raw_devices: list[Any] = []
    for m in ("get_devices", "get_clients", "get_client_list",
              "get_online_devices"):
        fn = getattr(router, m, None)
        if callable(fn):
            try:
                r = fn()
                if r:
                    raw_devices = list(r) if not isinstance(r, list) else r
                    break
            except Exception as e:
                snap.setdefault("errors", []).append(f"{m}: {e}")
    # Status / full_info often carry devices too — useful fallback.
    if not raw_devices:
        for m in ("get_status", "get_full_info"):
            fn = getattr(router, m, None)
            if callable(fn):
                try:
                    s = fn()
                    devs = getattr(s, "devices", None) or getattr(s, "clients", None)
                    if devs:
                        raw_devices = list(devs)
                        break
                except Exception as e:
                    snap.setdefault("errors", []).append(f"{m}: {e}")

    devices = [_serialise_device(d) for d in raw_devices]
    snap["devices"] = devices
    snap["topology"] = _summarise_topology(devices)

    # Guest network
    for m in ("get_guest_wifi", "get_guest_network", "get_wifi_guest"):
        fn = getattr(router, m, None)
        if callable(fn):
            try:
                g = fn()
                snap["guest_network"] = g if isinstance(g, dict) else {
                    "enabled": getattr(g, "enable", getattr(g, "enabled", None)),
                    "ssid":    getattr(g, "ssid",   None),
                }
                break
            except Exception as e:
                snap.setdefault("errors", []).append(f"{m}: {e}")

    # Parental control profiles
    for m in ("get_parental_control", "get_parental", "get_parental_profiles"):
        fn = getattr(router, m, None)
        if callable(fn):
            try:
                p = fn()
                snap["parental_profiles"] = p if isinstance(p, list) else [p]
                break
            except Exception as e:
                snap.setdefault("errors", []).append(f"{m}: {e}")

    return snap


def _refresh_snapshot(force: bool = False) -> dict | None:
    """Authenticate if needed, collect a fresh snapshot, persist it."""
    pw = _password()
    if not pw:
        _state["last_error"] = "no DECO_PASSWORD set"
        return None
    if _tplink() is None:
        _state["missing_dep"] = True
        _state["last_error"] = (
            "tplinkrouterc6u not installed — pip install tplinkrouterc6u "
            "(or fall back to the Playwright web-UI scrape at "
            "http://192.168.1.1/webpages/index.html#networkMap)")
        return None

    host = _state.get("router_host") or _verify_deco_host(_host())
    router = _state.get("router_handle")
    if router is None or _state.get("router_host") != host:
        router, cname = _make_router(host, pw)
        if router is None:
            _state["last_error"] = (
                f"could not construct router handle for {host}: "
                f"{_state.get('last_error')}")
            return None
        if not _authorize(router):
            return None
        _state["router_handle"] = router
        _state["router_host"]   = host
        _state["router_class"]  = cname
        _state["auth_ok"]       = True

    try:
        snap = _collect_via(router)
    except Exception as e:
        _state["auth_ok"]      = False
        _state["router_handle"] = None
        _state["last_error"]   = f"snapshot: {e}"
        return None

    with _lock:
        _state["snapshot"]    = snap
        _state["snapshot_at"] = time.time()
    try:
        _atomic_write_json(_SNAPSHOT_PATH, _json_sanitize(snap))
    except Exception as e:
        print(f"  [deco] snapshot write failed: {e}")
    return snap


def _load_cached_snapshot() -> dict | None:
    if not os.path.exists(_SNAPSHOT_PATH):
        return None
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _current_snapshot() -> dict | None:
    with _lock:
        snap = _state.get("snapshot")
    if snap:
        return snap
    return _load_cached_snapshot()


# ── background poller ─────────────────────────────────────────────
def _poll_loop() -> None:
    # Stagger the first poll so we don't fight the rest of boot.
    if _stop_evt.wait(INITIAL_DELAY_SECONDS):
        return
    while not _stop_evt.is_set():
        try:
            _refresh_snapshot()
        except Exception as e:
            print(f"  [deco] poll loop error: {e}")
        if _stop_evt.wait(POLL_INTERVAL_SECONDS):
            return


def _start_monitor() -> bool:
    if _poll_thread and _poll_thread[0].is_alive():
        return True
    if _tplink() is None:
        print("  [deco] tplinkrouterc6u not installed — monitor disabled. "
              "pip install tplinkrouterc6u")
        return False
    if not _password():
        print("  [deco] no DECO_PASSWORD set — monitor disabled. "
              "Set env var DECO_PASSWORD or write "
              "data/deco_config.json: {\"password\": \"...\"}.")
        return False
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, daemon=True, name="deco-monitor")
    t.start()
    _poll_thread.append(t)
    print(f"  [deco] monitor active — polling {_host()} every "
          f"{POLL_INTERVAL_SECONDS:.0f}s")
    return True


# ── action helpers ────────────────────────────────────────────────
def _device_name(d: dict) -> str:
    return (d.get("name") or d.get("hostname") or d.get("alias")
            or d.get("ip") or d.get("mac") or "unknown")


def _is_online(d: dict) -> bool:
    v = d.get("online")
    if v is None:
        return True
    return v not in (False, 0, "0", "false", "False")


def _fmt_bytes(n: Any) -> str:
    try:
        n = float(n)
    except Exception:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ── registered actions ────────────────────────────────────────────
def _act_who_is_on_wifi(_: str = "") -> str:
    snap = _current_snapshot() or _refresh_snapshot()
    if not snap:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or "Check DECO_PASSWORD."))
    devices = [d for d in snap.get("devices", []) if _is_online(d)]
    if not devices:
        return "Nobody's on the WiFi at the moment, sir."
    names = sorted({_device_name(d) for d in devices})
    if len(names) > 12:
        head = ", ".join(names[:10])
        return (f"{len(names)} clients online, sir — {head}, and "
                f"{len(names) - 10} more.")
    return f"{len(names)} clients online, sir: " + ", ".join(names) + "."


def _act_is_printer_online(_: str = "") -> str:
    snap = _current_snapshot() or _refresh_snapshot()
    if not snap:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or "Check DECO_PASSWORD."))
    for d in snap.get("devices", []):
        ip   = d.get("ip") or ""
        name = (_device_name(d) or "").lower()
        looks_like_printer = (
            ip in _PRINTER_IPS
            or any(hint in name for hint in _PRINTER_NAME_HINTS)
        )
        if looks_like_printer:
            if _is_online(d):
                return f"The printer is online, sir — {_device_name(d)} at {ip}."
            return f"The printer appears offline, sir — last known at {ip}."
    return ("I don't see a printer in the Deco client list, sir. "
            "Try `is_device_online printer` with a more specific name.")


def _act_is_device_online(name: str = "") -> str:
    name = (name or "").strip().lower()
    if not name:
        return "Which device, sir?"
    snap = _current_snapshot() or _refresh_snapshot()
    if not snap:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or "Check DECO_PASSWORD."))
    matches = [d for d in snap.get("devices", [])
               if name in (_device_name(d) or "").lower()
               or name in (d.get("ip") or "")
               or name in (d.get("mac") or "").lower()]
    if not matches:
        return f"I don't see anything matching '{name}' on the network, sir."
    online = [d for d in matches if _is_online(d)]
    if online:
        d = online[0]
        return f"{_device_name(d)} is online, sir, at {d.get('ip') or '?'}."
    d = matches[0]
    return f"{_device_name(d)} appears offline, sir — last known at {d.get('ip') or '?'}."


def _act_network_usage(_: str = "") -> str:
    snap = _current_snapshot() or _refresh_snapshot()
    if not snap:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or "Check DECO_PASSWORD."))
    def total(d: dict) -> float:
        try:
            return float(d.get("up_total") or 0) + float(d.get("down_total") or 0)
        except Exception:
            return 0.0
    def speed(d: dict) -> float:
        try:
            return float(d.get("up_speed") or 0) + float(d.get("down_speed") or 0)
        except Exception:
            return 0.0
    ranked = sorted(snap.get("devices", []), key=total, reverse=True)
    ranked = [d for d in ranked if total(d) > 0]
    if ranked:
        top = ranked[:5]
        parts = [f"{_device_name(d)} ({_fmt_bytes(total(d))})" for d in top]
        return "Top bandwidth users, sir: " + "; ".join(parts) + "."
    # No byte totals on this firmware — fall back to instantaneous
    # up/down speeds, which the Deco does report per client.
    ranked = sorted(snap.get("devices", []), key=speed, reverse=True)
    ranked = [d for d in ranked if speed(d) > 0]
    if not ranked:
        return ("The Deco isn't reporting per-client byte totals on this "
                "firmware, sir. Topology and online state only.")
    top = ranked[:5]
    parts = [f"{_device_name(d)} ({_fmt_bytes(speed(d))}/s)" for d in top]
    return "Top bandwidth users right now, sir: " + "; ".join(parts) + "."


def _set_guest_network(enabled: bool) -> str:
    pw = _password()
    if not pw:
        return ("I'd need the Deco password to do that, sir. Set "
                "DECO_PASSWORD or fill in data/deco_config.json.")
    router = _state.get("router_handle")
    if router is None:
        _refresh_snapshot()
        router = _state.get("router_handle")
    if router is None:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or ""))
    for m in ("set_guest_wifi", "set_guest_network", "set_wifi_guest"):
        fn = getattr(router, m, None)
        if not callable(fn):
            continue
        for args in ({"enable": enabled},
                     {"enabled": enabled},
                     (enabled,)):
            try:
                if isinstance(args, tuple):
                    fn(*args)
                else:
                    fn(**args)
                return ("Guest network "
                        + ("enabled" if enabled else "disabled")
                        + ", sir.")
            except TypeError:
                continue
            except Exception as e:
                _state["last_error"] = f"{m}: {e}"
    return ("The guest-network toggle isn't exposed by this firmware, sir "
            "— I'd need the Playwright web-UI fallback for that.")


def _act_kick_guest_network(_: str = "") -> str:
    return _set_guest_network(False)


def _act_enable_guest_network(_: str = "") -> str:
    return _set_guest_network(True)


def _act_deco_topology(_: str = "") -> str:
    snap = _current_snapshot() or _refresh_snapshot()
    if not snap:
        return ("I can't reach the Deco right now, sir. "
                + (_state.get("last_error") or "Check DECO_PASSWORD."))
    topo = snap.get("topology") or {}
    nodes = topo.get("deco_nodes") or []
    if not nodes:
        return (f"{topo.get('clients_total', 0)} clients online, sir, but "
                "the Deco didn't expose individual nodes on this firmware.")
    listing = ", ".join(n.get("name", "deco-node") for n in nodes)
    return (f"{len(nodes)} Deco node(s) on the mesh, sir: {listing}. "
            f"{topo.get('online', 0)} of {topo.get('clients_total', 0)} "
            "clients online.")


def _act_deco_status(_: str = "") -> str:
    snap = _current_snapshot()
    if not snap:
        snap = _refresh_snapshot()
    if not snap:
        return ("Deco link is down, sir. "
                + (_state.get("last_error") or ""))
    age = max(0.0, time.time() - float(snap.get("fetched_at") or 0))
    topo = snap.get("topology") or {}
    return (f"Deco mesh nominal, sir — {topo.get('online', 0)} of "
            f"{topo.get('clients_total', 0)} clients online, "
            f"snapshot {int(age)}s old.")


def _act_deco_refresh(_: str = "") -> str:
    snap = _refresh_snapshot(force=True)
    if not snap:
        return ("Refresh failed, sir. "
                + (_state.get("last_error") or "Unknown error."))
    topo = snap.get("topology") or {}
    return (f"Deco snapshot refreshed, sir — {topo.get('online', 0)} of "
            f"{topo.get('clients_total', 0)} clients online.")


# ── registration ──────────────────────────────────────────────────
def register(actions: dict) -> None:
    actions["who_is_on_wifi"]       = _act_who_is_on_wifi
    actions["who_is_on_the_wifi"]   = _act_who_is_on_wifi
    actions["network_clients"]      = _act_who_is_on_wifi
    actions["list_wifi_clients"]    = _act_who_is_on_wifi
    actions["is_printer_online"]    = _act_is_printer_online
    actions["printer_online"]       = _act_is_printer_online
    actions["is_device_online"]     = _act_is_device_online
    actions["device_online"]        = _act_is_device_online
    actions["network_usage"]        = _act_network_usage
    actions["bandwidth_hogs"]       = _act_network_usage
    actions["whats_using_bandwidth"] = _act_network_usage
    actions["kick_guest_network"]   = _act_kick_guest_network
    actions["disable_guest_network"] = _act_kick_guest_network
    actions["enable_guest_network"] = _act_enable_guest_network
    actions["deco_topology"]        = _act_deco_topology
    actions["network_topology"]     = _act_deco_topology
    actions["deco_status"]          = _act_deco_status
    actions["deco_refresh"]         = _act_deco_refresh
    actions["refresh_network"]      = _act_deco_refresh

    # Kick the background monitor if the dep + password are both present.
    _start_monitor()
