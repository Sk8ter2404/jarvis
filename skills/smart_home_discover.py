"""
Smart-home discovery wizard via Alexa.

One-time wizard that signs into Amazon Alexa, enumerates every smart
home device the user's Alexa account controls (lights, locks, plugs,
thermostats, cameras, scenes), cross-references each entry with a LAN
ARP scan to recover IP + MAC where possible, and writes a canonical
device catalog at `data/smart_home_devices.json` for the smart-home
router (`core/smart_home_router.py`, research-4b) to dispatch per-brand
direct API calls without ongoing Alexa dependency.

After the first successful run the cookie is cached at
`data/alexa_cookie.json` (metadata) plus `data/alexa_cookie.pickle`
(the alexapy-format jar) — subsequent runs reuse the cookie until it
expires.

Registered actions
------------------
    smart_home_discover     — run the wizard end-to-end
    discover_smart_home     — alias for the LLM router
    smart_home_setup        — alias
    refresh_smart_home      — alias
    smart_home_catalog      — speak / log a summary of the cached catalog
    smart_home_purge_cookie — delete the cached cookie (forces re-login)

Voice trigger hints (handled by the LLM via these action names):
    'JARVIS, discover smart home devices'
    'JARVIS, set up smart home'
    'JARVIS, refresh smart home catalog'

Optional dependency:
    alexapy   ← REQUIRED  (pip install alexapy)
              Falls back to a graceful 'not installed' message and
              leaves the rest of JARVIS untouched.

For brands found by Alexa but without a JARVIS controller skill yet,
the wizard appends a self-implementing task to `jarvis_todo.md` so the
upgrade pipeline can build the missing `skills/sh_<brand>.py` next
pass — matches the [TODO: build skill for brand X] hand-off promised
by research-4b.
"""
from __future__ import annotations

import asyncio
import datetime
import getpass
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


_PROJECT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR           = os.path.join(_PROJECT_DIR, "data")
_COOKIE_JSON_PATH   = os.path.join(_DATA_DIR, "alexa_cookie.json")
_COOKIE_PICKLE_PATH = os.path.join(_DATA_DIR, "alexa_cookie.pickle")
_CATALOG_PATH       = os.path.join(_DATA_DIR, "smart_home_devices.json")
_TODO_PATH          = os.path.join(_PROJECT_DIR, "jarvis_todo.md")

# Cookie freshness — Amazon cookies last ~1 year, but we warn here so
# the user can refresh proactively rather than discovering an expired
# session mid-command.
_COOKIE_SUGGEST_REFRESH_DAYS = 300

# 2026-07-08: hard ceiling for a voice-context catalog refresh. The alexapy /
# aiohttp fetch runs on the assistant dispatch thread; a stalled Amazon endpoint
# with no timeout would wedge the whole voice loop. 45s is generous for a warm
# cookie fetch yet bounded enough to fail fast and speak a timeout notice.
_DISCOVERY_TIMEOUT_SEC = 45.0

# Brand string → controller skill name. alexapy returns free-form
# manufacturer strings ('Philips Hue', 'Signify Netherlands B.V.',
# 'TP-Link', 'tp-link Tapo'); _controller_skill() does substring
# matching against the lowercased manufacturer.
_BRAND_TO_SKILL = {
    "philips hue":  "sh_hue",
    "signify":      "sh_hue",
    "hue":          "sh_hue",
    "tp-link":      "sh_kasa",
    "tplink":       "sh_kasa",
    "kasa":         "sh_kasa",
    "tapo":         "sh_kasa",
    "lifx":         "sh_lifx",
    "govee":        "sh_govee",
    "ecobee":       "sh_ecobee",
    "nest":         "sh_nest",
    "google nest":  "sh_nest",
    "ring":         "sh_ring",
}

# OUI prefixes (upper, no separators) for the LAN cross-reference.
# Not exhaustive — enough to auto-label the most common smart-home
# device manufacturers on the user's LAN. Anything not in this table
# falls through to controller_skill = null + a missing-skill task.
_OUI_TO_BRAND = {
    "001788": "Philips Hue",
    "ECB5FA": "Philips Hue",
    "D073D5": "LIFX",
    "501A59": "TP-Link",
    "50C7BF": "TP-Link",
    "B0BE76": "TP-Link",
    "B04E26": "TP-Link",
    "1C61B4": "TP-Link",
    "18B430": "Nest",
    "641666": "Nest",
    "AC5D5C": "Ring",
    "FC65DE": "Ring",
    "446132": "ecobee",
    "5C32C6": "ecobee",
    "A4C138": "Govee",
}

# Alexa capability namespace → short tag consumed by sh_router.
# Anything not in this table is passed through with the 'Alexa.'
# prefix stripped and lowercased so unknown caps still appear.
_CAPABILITY_TAGS = {
    "Alexa.PowerController":             "on_off",
    "Alexa.BrightnessController":        "dim",
    "Alexa.ColorController":             "color",
    "Alexa.ColorTemperatureController":  "color_temperature",
    "Alexa.PercentageController":        "percentage",
    "Alexa.LockController":              "lock",
    "Alexa.ThermostatController":        "thermostat",
    "Alexa.TemperatureSensor":           "temperature",
    "Alexa.MotionSensor":                "motion",
    "Alexa.ContactSensor":               "contact",
    "Alexa.CameraStreamController":      "camera",
    "Alexa.SceneController":             "scene",
    "Alexa.RangeController":             "range",
    "Alexa.ModeController":              "mode",
    "Alexa.SecurityPanelController":     "security",
}


# ── lazy alexapy import ─────────────────────────────────────────────
def _alexapy():
    """Lazy import so a missing alexapy install can't crash skill loading."""
    try:
        import alexapy  # type: ignore
        return alexapy
    except Exception as e:
        print(f"  [sh-discover] alexapy not installed ({e}); "
              "install with `pip install alexapy` to enable the wizard.")
        return None


def is_available() -> bool:
    return _alexapy() is not None


# ── speech bridge ───────────────────────────────────────────────────
def _bc():
    """Lazy import of bobert_companion. Imported inside a function so
    py_compile / pytest don't drag the full audio stack into the
    test process."""
    return importlib.import_module("bobert_companion")


def _say(text: str) -> None:
    """Speak via JARVIS TTS; print on any failure."""
    try:
        _bc()._speak(text)
    except Exception:
        print(f"  [sh-discover] {text}")


# ── atomic write helpers ────────────────────────────────────────────
def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise


# ── catalog read / write ────────────────────────────────────────────
class CatalogWipeRefused(Exception):
    """Raised when a save would erase a populated catalog with an empty one."""


def _save_catalog(catalog: dict) -> None:
    """Persist the catalog — but REFUSE to overwrite a populated catalog with an
    empty one. A failed Alexa fetch (auth expiry, the 2026-07-04 LAN-DNS outage,
    any transient) returns 0 devices; _fetch_devices_async swallows the error and
    hands back an empty {echo:[],smarthome:[],groups:[]} dict, which _build_catalog
    turns into a 0-device catalog. Without this guard the wizard saved that over a
    good catalog and _merge_with_existing_catalog dropped every device — so voice
    control went dead until a *successful* re-run (owner hit this live; the on-disk
    catalog was wiped to device_count=0). We keep the last-known-good file instead
    and raise so the caller can report the failure honestly."""
    try:
        n = int(catalog.get("device_count", 0) or 0)
    except (TypeError, ValueError):
        n = len(catalog.get("devices") or [])
    if n <= 0:
        existing = _load_catalog()
        prev_n = 0
        if isinstance(existing, dict):
            try:
                prev_n = int(existing.get("device_count", 0) or 0)
            except (TypeError, ValueError):
                prev_n = len(existing.get("devices") or [])
        if prev_n > 0:
            raise CatalogWipeRefused(
                f"refused to overwrite {prev_n} known device(s) with an empty "
                f"catalog — the device fetch came back empty (auth/network?)")
    # Keep a one-deep backup of the prior good catalog before overwriting, so a
    # bad save is recoverable even if this guard is ever bypassed.
    try:
        if n > 0 and os.path.exists(_CATALOG_PATH):
            import shutil
            shutil.copy2(_CATALOG_PATH, _CATALOG_PATH + ".bak")
    except Exception:
        pass
    _atomic_write_json(_CATALOG_PATH, catalog)


def _load_catalog() -> dict | None:
    if not os.path.exists(_CATALOG_PATH):
        return None
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Fields that the user may hand-edit in data/smart_home_devices.json.
# A re-run of the wizard preserves any non-empty existing value for these
# fields, so a manual rename or controller override survives. All other
# fields are refreshed from the live discovery.
_USER_OVERRIDE_FIELDS = ("name", "controller_skill", "lan_ip", "lan_mac")


def _merge_with_existing_catalog(fresh: dict) -> dict:
    """Merge a freshly-built catalog with the existing on-disk catalog,
    preserving user overrides to the fields listed in
    `_USER_OVERRIDE_FIELDS`. Devices are matched by `alexa_entity_id`;
    devices absent from the existing file are kept as-is from `fresh`,
    and devices absent from `fresh` (i.e. removed in Alexa) are dropped."""
    existing = _load_catalog()
    if not existing or not isinstance(existing.get("devices"), list):
        return fresh
    by_id: dict[str, dict] = {}
    for d in existing["devices"]:
        if isinstance(d, dict):
            eid = d.get("alexa_entity_id")
            if eid:
                by_id[eid] = d
    if not by_id:
        return fresh
    for d in fresh.get("devices", []):
        eid = d.get("alexa_entity_id")
        if not eid:
            continue
        prev = by_id.get(eid)
        if not prev:
            continue
        for k in _USER_OVERRIDE_FIELDS:
            old = prev.get(k)
            if old not in (None, "", []):
                d[k] = old
    return fresh


# ── cookie persistence ──────────────────────────────────────────────
def _save_cookie_meta(email: str, status: dict, cookie_jar_obj: Any) -> None:
    """Pickle the alexapy cookie jar to its own file and write a small
    human-readable JSON metadata file the user can inspect."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(_COOKIE_PICKLE_PATH, "wb") as f:
            pickle.dump(cookie_jar_obj, f)
    except Exception as e:
        print(f"  [sh-discover] cookie pickle failed: {e}")
    meta = {
        "version": 1,
        "email": email,
        "saved_at": time.time(),
        "saved_at_iso": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "pickle_path": os.path.relpath(_COOKIE_PICKLE_PATH, _PROJECT_DIR),
        "status_keys": sorted(list((status or {}).keys())),
    }
    _atomic_write_json(_COOKIE_JSON_PATH, meta)


def _load_cookie_meta() -> dict | None:
    if not os.path.exists(_COOKIE_JSON_PATH):
        return None
    try:
        with open(_COOKIE_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cookie_is_stale(meta: dict, max_days: int = _COOKIE_SUGGEST_REFRESH_DAYS) -> bool:
    saved = float(meta.get("saved_at") or 0)
    if not saved:
        return True
    return (time.time() - saved) / 86400 > max_days


# ── LAN ARP scan ────────────────────────────────────────────────────
_ARP_LINE = re.compile(
    r"\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+"
    r"([0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}"
    r"[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2})"
)


def _scan_lan_arp() -> list[dict]:
    """Run `arp -a` and parse out a list of {ip, mac, oui, brand_oui_hint}.
    Quiet on failure — the wizard still emits a catalog with empty LAN
    fields when the scan fails."""
    try:
        raw = subprocess.check_output(
            ["arp", "-a"],
            stderr=subprocess.DEVNULL,
            timeout=10,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
    except Exception as e:
        print(f"  [sh-discover] arp -a failed: {e}")
        return []
    # Windows arp output is locale-dependent; force a permissive decode
    # so a non-ASCII interface name can't crash the parse.
    text = raw.decode("utf-8", errors="replace")
    if "Interface:" not in text and "Physical" not in text:
        text = raw.decode("cp1252", errors="replace")
    rows: list[dict] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _ARP_LINE.search(line)
        if not m:
            continue
        ip = m.group(1)
        if ip in seen:
            continue
        seen.add(ip)
        mac = m.group(2).upper().replace("-", ":")
        oui = mac.replace(":", "")[:6]
        rows.append({
            "ip": ip,
            "mac": mac,
            "oui": oui,
            "brand_oui_hint": _OUI_TO_BRAND.get(oui),
        })
    return rows


def _match_arp_entry(brand: str, arp_table: list[dict]) -> tuple[str, str] | None:
    """Best-effort LAN cross-reference for a given brand string. Returns
    (ip, mac) on a match, None otherwise. Multiple matches → first one
    (the user can confirm in the catalog by hand)."""
    if not brand or not arp_table:
        return None
    b = brand.lower()
    for row in arp_table:
        hint = (row.get("brand_oui_hint") or "").lower()
        if hint and hint in b:
            return (row["ip"], row["mac"])
    return None


# ── brand / capability helpers ──────────────────────────────────────
def _normalise_brand(raw: Any) -> str:
    if not raw:
        return ""
    return re.sub(r"\s+", " ", str(raw)).strip()


def _controller_skill(brand: str) -> str | None:
    if not brand:
        return None
    b = brand.lower()
    for key, skill in _BRAND_TO_SKILL.items():
        if key in b:
            return skill
    return None


def _capability_tags(raw_caps: Any) -> list[str]:
    """alexapy returns Amazon capability descriptors in several shapes
    depending on endpoint and library version. Flatten to a sorted set
    of short tags so downstream skills don't need to care."""
    if not raw_caps:
        return []
    tags: list[str] = []
    if isinstance(raw_caps, dict):
        for v in raw_caps.values():
            tags.extend(_capability_tags(v))
    elif isinstance(raw_caps, list):
        for item in raw_caps:
            if isinstance(item, dict):
                iface = item.get("interface") or item.get("namespace") or ""
                tag = _CAPABILITY_TAGS.get(iface)
                if tag:
                    tags.append(tag)
                elif iface.startswith("Alexa."):
                    tags.append(iface.replace("Alexa.", "").lower())
            elif isinstance(item, str):
                tags.append(_CAPABILITY_TAGS.get(item)
                            or item.replace("Alexa.", "").lower())
    elif isinstance(raw_caps, str):
        tags.append(_CAPABILITY_TAGS.get(raw_caps)
                    or raw_caps.replace("Alexa.", "").lower())
    return sorted({t for t in tags if t})


# ── async runner ────────────────────────────────────────────────────
def _run_async(coro, timeout: float | None = None):
    """Run a coroutine to completion regardless of whether the current
    thread already owns an event loop. If we're inside one we delegate
    to a worker thread so we don't deadlock.

    When `timeout` is set the coroutine is bounded by asyncio.wait_for and, on
    the worker-thread path, the join is bounded too — so a stalled Amazon /
    Playwright endpoint can't hang the caller (e.g. the voice dispatch thread)
    forever. A TimeoutError is raised in that case. (2026-07-08)"""
    async def _bounded():
        if timeout is None:
            return await coro
        return await asyncio.wait_for(coro, timeout)

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if not running:
        return asyncio.run(_bounded())

    box: dict = {}
    def _go() -> None:
        try:
            box["v"] = asyncio.run(_bounded())
        except Exception as e:
            box["err"] = e
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    # Bound the join to the coroutine's own budget plus a small grace margin so
    # a coro that ignores cancellation (blocked in a native call) still can't
    # wedge the dispatch thread indefinitely. (2026-07-08)
    t.join(None if timeout is None else timeout + 5.0)
    if t.is_alive():
        raise TimeoutError("discovery timed out")
    if "err" in box:
        raise box["err"]
    return box.get("v")


# ── alexapy interaction ─────────────────────────────────────────────
def _construct_login(email: str, password: str) -> Any:
    """Build an AlexaLogin, tolerating API drift between alexapy versions.
    Newer versions accept a callable for `outputpath`; older versions
    want a directory path string."""
    alexapy = _alexapy()
    if alexapy is None:
        return None
    AlexaLogin = alexapy.AlexaLogin

    def _out(txt: str) -> None:
        print(f"  [sh-discover/alexapy] {txt}")

    candidates = (
        dict(url="amazon.com", email=email, password=password,
             outputpath=_out, debug=False),
        dict(url="amazon.com", email=email, password=password,
             outputpath=_DATA_DIR, debug=False),
        dict(url="amazon.com", email=email, password=password),
    )
    last_err: Exception | None = None
    for kwargs in candidates:
        try:
            return AlexaLogin(**kwargs)
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break
    print(f"  [sh-discover] could not construct AlexaLogin: {last_err}")
    return None


class _LoginNeedsPlaywright(Exception):
    """Raised by `_login_async` when alexapy returns an empty or
    unrecognised status dict — Amazon's response shape has drifted past
    what alexapy 1.29.22 parses. The caller is expected to fall through
    to `_login_via_playwright` for a browser-driven sign-in."""


async def _login_async(email: str, password: str) -> Any:
    """Drive the alexapy login through CAPTCHA / claimspicker / OTP
    until status.login_successful is True (or fatal). Each step is
    surfaced to the console — the wizard is expected to be running
    in a terminal for the first sign-in."""
    login = _construct_login(email, password)
    if login is None:
        return None

    await login.login()

    safety_loop_limit = 8
    for _ in range(safety_loop_limit):
        status = dict(login.status or {})
        if status.get("login_successful"):
            return login
        if status.get("captcha_required"):
            captcha_url = (status.get("captcha_image_url")
                           or status.get("captcha_url") or "?")
            print(f"  [sh-discover] CAPTCHA required. Open this URL in a "
                  f"browser to view: {captcha_url}")
            try:
                ans = input("    Type the captcha characters: ").strip()
            except EOFError:
                return None
            await login.login(data={"captcha": ans})
            continue
        if status.get("claimspicker_required"):
            options = status.get("claimspicker_options") or {}
            print("  [sh-discover] Amazon wants you to pick where to receive the OTP:")
            for k, v in options.items():
                print(f"    {k}. {v}")
            try:
                pick = input("    Choose option key: ").strip()
            except EOFError:
                return None
            await login.login(data={"claimsoption": pick})
            continue
        if status.get("authselect_required"):
            options = status.get("authselect_options") or {}
            print("  [sh-discover] Amazon offered multiple auth methods:")
            for k, v in options.items():
                print(f"    {k}. {v}")
            try:
                pick = input("    Choose option key: ").strip()
            except EOFError:
                return None
            await login.login(data={"authselectoption": pick})
            continue
        if status.get("verificationcode_required"):
            try:
                code = input("    Enter the 2FA code Amazon just sent: ").strip()
            except EOFError:
                return None
            await login.login(data={"verificationcode": code})
            continue
        if status.get("securitycode_required"):
            try:
                code = input("    Enter your Amazon security code: ").strip()
            except EOFError:
                return None
            await login.login(data={"securitycode": code})
            continue
        if status.get("login_failed"):
            print(f"  [sh-discover] Login failed: "
                  f"{status.get('error') or '(no error message)'}")
            return None
        # Unknown step — alexapy 1.29.22 returns an empty/unknown dict
        # when Amazon's response shape doesn't match what it parses. We
        # signal the wizard to fall through to the Playwright fallback
        # rather than dying here.
        print(f"  [sh-discover] Unrecognised login state: "
              f"{sorted(status.keys())}; falling back to browser sign-in.")
        raise _LoginNeedsPlaywright()
    print("  [sh-discover] Exceeded login step limit; "
          "falling back to browser sign-in.")
    raise _LoginNeedsPlaywright()


# ── Playwright fallback ─────────────────────────────────────────────
# alexapy 1.29.22 can't parse Amazon's current OAuth login response and
# bails with an empty status dict. The fallback drives a real Chromium
# window so the user can sign in manually (CAPTCHA + 2FA included);
# JARVIS then captures the resulting session cookies for both the
# immediate device-fetch and a pickle re-used on subsequent runs.

_PLAYWRIGHT_LOGIN_TIMEOUT_S = 300.0  # 5 min — user may hit CAPTCHA / 2FA
# Canonical Amazon OpenID sign-in URL. A BARE ".../ap/signin" (no query) 404s
# to Amazon's "Looking for Something?" dog page; the full OpenID query lands on
# the real sign-in form and, on success, redirects to return_to (the amazon.com
# homepage) — which leaves the /ap/ auth path, the signal the capture loop waits
# for. 2026-05-30.
_PLAYWRIGHT_SIGNIN_URL = (
    # ALEXA OAuth sign-in (assoc_handle=amzn_dp_project_dee_web, return_to=
    # alexa.amazon.com) — NOT the generic amazon.com login. The generic login
    # yields amazon.com cookies that alexapy's /api/devices-v2 call rejects (it
    # redirects back to this exact Alexa sign-in), so the device fetch failed
    # with a JSON-parse error. Signing in through the Alexa flow produces the
    # Alexa-scoped session cookies alexapy actually needs. 2026-05-30.
    "https://www.amazon.com/ap/signin"
    "?openid.return_to=https%3A%2F%2Falexa.amazon.com%2Fspa%2Findex.html"
    "&openid.assoc_handle=amzn_dp_project_dee"
    "&openid.mode=checkid_setup"
    "&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
    "&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
    "&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
)


def _convert_playwright_cookies(pw_cookies: list[dict]) -> Any:
    """Turn the list of dicts returned by Playwright's `context.cookies()`
    into a requests.cookies.RequestsCookieJar — the format alexapy's
    `load_cookie()` accepts when unpickled and the same shape our
    existing pickle/restore path consumes."""
    from requests.cookies import RequestsCookieJar
    jar = RequestsCookieJar()
    for c in pw_cookies or []:
        try:
            name = c.get("name") or ""
            value = c.get("value") or ""
            if not name:
                continue
            kwargs: dict[str, Any] = {
                "domain": c.get("domain") or "",
                "path": c.get("path") or "/",
                "secure": bool(c.get("secure")),
            }
            exp = c.get("expires")
            if isinstance(exp, (int, float)) and exp > 0:
                kwargs["expires"] = int(exp)
            if c.get("httpOnly"):
                kwargs["rest"] = {"HttpOnly": ""}
            jar.set(name, value, **kwargs)
        except Exception:
            continue
    return jar


async def _login_via_playwright(
    email: str,
    timeout_seconds: float = _PLAYWRIGHT_LOGIN_TIMEOUT_S,
) -> tuple[Any, list[dict]] | None:
    """Open a HEADED Chromium so the user can complete Amazon sign-in
    interactively. Polls every 2 s for the URL to leave /ap/signin —
    Amazon's success signal — then dumps cookies and shuts the browser.

    Returns (RequestsCookieJar, raw_playwright_cookies) on success or
    None on any failure (missing Playwright, no display available,
    timeout, user closed the window before completing)."""
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        print(f"  [sh-discover] Playwright not available ({e}); "
              "falling through to install hint.")
        return None

    print()
    print(f"  [sh-discover] Opening a Chromium window so you can sign in to Amazon.")
    print(f"                Complete sign-in normally (CAPTCHA / 2FA included);")
    print(f"                JARVIS will capture the session cookies automatically")
    print(f"                once Amazon redirects you off the /ap/signin page.")
    print(f"                Timeout: {int(timeout_seconds)}s.")
    print()

    try:
        async with async_playwright() as pw:
            # Prefer the user's installed Google Chrome (channel="chrome").
            # Playwright's BUNDLED Chromium throws "spawn UNKNOWN" on some
            # Windows setups; the system Chrome the user runs daily launches
            # cleanly. Try Chrome -> bundled Chromium -> Edge so this works
            # across machines. 2026-05-30.
            browser = None
            _launch_errs = []
            for _chan in ("chrome", None, "msedge"):
                try:
                    if _chan is None:
                        browser = await pw.chromium.launch(headless=False)
                    else:
                        browser = await pw.chromium.launch(
                            headless=False, channel=_chan)
                    print(f"  [sh-discover] Sign-in browser: "
                          f"{_chan or 'bundled chromium'}.")
                    break
                except Exception as _le:
                    _launch_errs.append(f"{_chan or 'bundled'}: {_le}")
                    browser = None
            if browser is None:
                print("  [sh-discover] Could not launch any headed browser for "
                      "sign-in:\n      " + "\n      ".join(_launch_errs))
                return None
            try:
                context = await browser.new_context()
                page = await context.new_page()
                try:
                    await page.goto(_PLAYWRIGHT_SIGNIN_URL)
                except Exception as e:
                    print(f"  [sh-discover] Failed to load the Amazon sign-in page: {e}")
                    return None

                loop = asyncio.get_event_loop()
                deadline = loop.time() + timeout_seconds
                pw_cookies: list[dict] = []
                while True:
                    if loop.time() > deadline:
                        print("  [sh-discover] Sign-in timed out before Amazon "
                              "redirected off /ap/signin.")
                        return None
                    if page.is_closed():
                        # User closed the window — try grabbing cookies
                        # anyway in case they completed login first.
                        try:
                            pw_cookies = await context.cookies()
                        except Exception:
                            pw_cookies = []
                        break
                    try:
                        current_url = page.url or ""
                    except Exception:
                        current_url = ""
                    # Capture once we've left ALL Amazon auth pages (/ap/signin,
                    # /ap/mfa, /ap/cvf, /ap/register, ...) and landed on a real
                    # amazon page — so a 2FA / captcha step doesn't trigger a
                    # premature, half-authenticated cookie capture.
                    if (current_url and "/ap/" not in current_url
                            and "amazon." in current_url.lower()):
                        try:
                            pw_cookies = await context.cookies()
                        except Exception:
                            pw_cookies = []
                        break
                    await asyncio.sleep(2.0)
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"  [sh-discover] Playwright sign-in failed: {e}")
        return None

    if not pw_cookies:
        print("  [sh-discover] Playwright did not return any cookies; "
              "treating sign-in as cancelled.")
        return None
    jar = _convert_playwright_cookies(pw_cookies)
    print(f"  [sh-discover] Captured {len(pw_cookies)} cookie(s) from the "
          "Amazon sign-in.")
    return (jar, pw_cookies)


def _build_login_from_playwright_cookies(email: str,
                                         pw_cookies: list[dict]) -> Any:
    """Best-effort: construct a fresh AlexaLogin and inject the
    Playwright-captured cookies into its aiohttp session so the
    immediate `_fetch_devices_async()` call has a populated jar.

    Returns None if alexapy isn't installed or the injection fails —
    the wizard then completes without an in-process device fetch, but
    the cookie pickle is still cached so future runs can re-use it via
    `_restore_login_from_cookie()`."""
    alexapy = _alexapy()
    if alexapy is None:
        return None
    login = _construct_login(email, "")
    if login is None:
        return None
    try:
        from http.cookies import SimpleCookie
        from yarl import URL as YURL
        session = getattr(login, "_session", None) or getattr(login, "session", None)
        if session is None:
            return None
        jar = getattr(session, "cookie_jar", None)
        if jar is None:
            return None
        for c in pw_cookies:
            try:
                name = c.get("name") or ""
                value = c.get("value") or ""
                if not name:
                    continue
                sc = SimpleCookie()
                sc[name] = value
                m = sc[name]
                if c.get("domain"):
                    m["domain"] = c["domain"]
                if c.get("path"):
                    m["path"] = c["path"]
                if c.get("secure"):
                    m["secure"] = True
                if c.get("httpOnly"):
                    m["httponly"] = True
                domain = (c.get("domain") or "").lstrip(".") or "amazon.com"
                jar.update_cookies(sc, YURL(f"https://{domain}"))
            except Exception:
                continue
    except Exception as e:
        print(f"  [sh-discover] Cookie injection into alexapy session failed: {e}")
        return None
    return login


class _PlaywrightLoginShim:
    """Duck-typed stand-in for an AlexaLogin so `_extract_cookie_jar`
    and `_save_cookie_meta` can persist a Playwright-captured cookie
    jar through the existing code path unmodified."""
    def __init__(self, email: str, cookies_jar: Any) -> None:
        self.email = email
        self._cookies = cookies_jar
        self.status = {"login_successful": True, "playwright": True}


async def _fetch_devices_async(login: Any) -> dict:
    """Best-effort fetch of every device list alexapy exposes. Each call
    is wrapped individually because alexapy's API surface drifts and an
    older install may be missing one of these methods."""
    alexapy = _alexapy()
    if alexapy is None or login is None:
        return {}
    AlexaAPI = alexapy.AlexaAPI

    out: dict = {"echo": [], "smarthome": [], "groups": []}
    try:
        if hasattr(AlexaAPI, "get_devices"):
            out["echo"] = await AlexaAPI.get_devices(login) or []
    except Exception as e:
        print(f"  [sh-discover] get_devices failed: {e}")
    try:
        if hasattr(AlexaAPI, "get_smarthome_devices"):
            out["smarthome"] = await AlexaAPI.get_smarthome_devices(login) or []
        elif hasattr(AlexaAPI, "get_appliances"):
            out["smarthome"] = await AlexaAPI.get_appliances(login) or []
    except Exception as e:
        print(f"  [sh-discover] get_smarthome_devices failed: {e}")
    try:
        if hasattr(AlexaAPI, "get_smarthome_groups"):
            out["groups"] = await AlexaAPI.get_smarthome_groups(login) or []
        elif hasattr(AlexaAPI, "get_groups"):
            out["groups"] = await AlexaAPI.get_groups(login) or []
    except Exception as e:
        print(f"  [sh-discover] get_smarthome_groups failed: {e}")
    return out


# ── catalog assembly ────────────────────────────────────────────────
def _entity_room(entity: dict, echos: list[dict]) -> str:
    """Best-effort room assignment. Smart-home device payloads sometimes
    carry a roomName directly; otherwise we look up the linked Echo's
    accountName via applianceDetails.alexaDeviceId."""
    if not entity:
        return ""
    for k in ("room", "roomName", "applianceLocation"):
        v = entity.get(k)
        if isinstance(v, str) and v:
            return v
    appliance_id = (entity.get("applianceDetails") or {}).get("alexaDeviceId")
    if appliance_id and echos:
        for e in echos:
            if (e.get("serialNumber") == appliance_id
                    or e.get("deviceSerialNumber") == appliance_id):
                return e.get("accountName") or e.get("deviceName") or ""
    return ""


def _entity_groups(entity_id: str, groups: list[dict]) -> list[str]:
    if not entity_id or not groups:
        return []
    out: list[str] = []
    for g in groups:
        members = (g.get("applianceIds") or g.get("entityIds")
                   or g.get("members") or [])
        if isinstance(members, list) and entity_id in members:
            name = g.get("name") or g.get("groupName") or ""
            if name:
                out.append(name)
    return out


def _entity_type(caps: list[str], brand: str, entity: dict) -> str:
    """Pick a coarse type string for the catalog. Light/lock/thermostat
    /camera/scene are the high-value targets the LLM needs to dispatch
    against; everything else falls through to displayCategories[0] or
    'unknown'."""
    if "lock" in caps:
        return "lock"
    if "thermostat" in caps:
        return "thermostat"
    if "camera" in caps:
        return "camera"
    if "scene" in caps:
        return "scene"
    lighty = any(t in caps for t in ("color", "color_temperature", "dim"))
    if lighty:
        return "light"
    if "on_off" in caps:
        lower_brand = brand.lower()
        if any(b in lower_brand for b in ("hue", "lifx", "tplink", "tp-link",
                                           "kasa", "govee", "tapo")):
            return "light"
        return "plug"
    cats = entity.get("displayCategories")
    if isinstance(cats, list) and cats:
        return str(cats[0]).lower()
    return "unknown"


def _entity_to_record(entity: dict, echos: list[dict],
                      groups: list[dict], arp_table: list[dict]) -> dict:
    """Convert one Alexa smart-home device dict into the canonical record."""
    appliance = entity.get("applianceDetails") or {}
    name = (entity.get("friendlyName")
            or entity.get("name")
            or appliance.get("friendlyName")
            or "(unnamed)")
    brand = _normalise_brand(
        entity.get("manufacturerName")
        or entity.get("brand")
        or appliance.get("manufacturerName")
        or ""
    )
    model = (entity.get("modelName")
             or entity.get("model")
             or appliance.get("modelName")
             or "")
    raw_caps = (entity.get("capabilities")
                or entity.get("supportedActions")
                or appliance.get("capabilities")
                or [])
    caps = _capability_tags(raw_caps)
    dtype = _entity_type(caps, brand, entity)

    entity_id = (entity.get("entityId")
                 or entity.get("applianceId")
                 or appliance.get("applianceId")
                 or "")
    room = _entity_room(entity, echos)
    grp = _entity_groups(entity_id, groups)
    lan = _match_arp_entry(brand, arp_table)
    controller = _controller_skill(brand)

    return {
        "name": name,
        "brand": brand,
        "model": model,
        "type": dtype,
        "capabilities": caps,
        "alexa_entity_id": entity_id,
        "alexa_room": room,
        "alexa_groups": grp,
        "lan_ip": lan[0] if lan else "",
        "lan_mac": lan[1] if lan else "",
        "controller_skill": controller,
    }


def _build_catalog(devices: dict, arp_table: list[dict]) -> dict:
    echos = devices.get("echo") or []
    smarthome = devices.get("smarthome") or []
    groups = devices.get("groups") or []
    records = [_entity_to_record(e, echos, groups, arp_table)
               for e in smarthome if isinstance(e, dict)]
    # Stable sort by room then name so diffs against the file are clean
    # and manual inspection is room-grouped.
    records.sort(key=lambda r: ((r.get("alexa_room") or "").lower(),
                                (r.get("name") or "").lower()))
    return {
        "version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "device_count": len(records),
        "echo_count": len(echos),
        "group_count": len(groups),
        "arp_seen": len(arp_table),
        "devices": records,
    }


def _queue_missing_skill_tasks(catalog: dict) -> int:
    """Append one self-implementing task per unknown brand to
    jarvis_todo.md so the upgrade pipeline can build the missing
    `skills/sh_<brand>.py` next run. Idempotent — checks for an existing
    marker before appending."""
    if not os.path.exists(_TODO_PATH):
        return 0
    missing_brands: set[str] = set()
    for d in catalog.get("devices", []):
        if d.get("controller_skill"):
            continue
        b = (d.get("brand") or "").strip()
        if b:
            missing_brands.add(b)
    if not missing_brands:
        return 0
    try:
        with open(_TODO_PATH, "r", encoding="utf-8") as f:
            existing = f.read()
    except Exception:
        return 0
    today = datetime.date.today().isoformat()
    added: list[str] = []
    for brand in sorted(missing_brands):
        marker = f"[sh-discover] Build controller skill for brand '{brand}'"
        if marker in existing:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", brand.lower()).strip("_") or "unknown"
        added.append(
            f"- [ ] **{today} sh-discover** - {marker}. "
            f"Add `skills/sh_{slug}.py` following the sh_hue / sh_kasa template "
            f"(uniform set_state / get_state / list_devices). Brand discovered "
            f"by the Alexa wizard but no JARVIS controller skill exists yet."
        )
    if not added:
        return 0
    try:
        with open(_TODO_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(added) + "\n")
    except Exception as e:
        print(f"  [sh-discover] todo append failed: {e}")
        return 0
    return len(added)


# ── cookie restore ──────────────────────────────────────────────────
def _extract_cookie_jar(login: Any) -> Any:
    """Pull whatever cookie-bearing object alexapy exposes; layout shifts
    between versions, so we accept several attribute names."""
    for attr in ("_cookies", "_session", "session"):
        val = getattr(login, attr, None)
        if val is None:
            continue
        jar = getattr(val, "cookie_jar", None)
        if jar is not None:
            return jar
        return val
    return None


def _restore_login_from_cookie() -> Any:
    """Try to rebuild a usable AlexaLogin from the cached pickle.
    Returns the login on success, None on any failure (the wizard then
    falls back to a fresh interactive login)."""
    alexapy = _alexapy()
    if alexapy is None:
        return None
    meta = _load_cookie_meta()
    if not meta or not os.path.exists(_COOKIE_PICKLE_PATH):
        return None
    # SECURITY: pickle.load() executes arbitrary code on a crafted file and
    # is a known risk if data/ is writable by another process. The real fix
    # is migrating the cookie cache to JSON; until then we (a) refuse to load
    # anything that doesn't resolve to the exact expected path inside the
    # project data dir (defends against symlink/path swaps) and (b) catch all
    # exceptions so a corrupt/malicious file degrades to a fresh login
    # instead of executing or crashing.
    try:
        real_path = os.path.realpath(_COOKIE_PICKLE_PATH)
        real_data_dir = os.path.realpath(_DATA_DIR)
        if (
            real_path != os.path.realpath(os.path.join(real_data_dir,
                                                        "alexa_cookie.pickle"))
            or os.path.commonpath([real_path, real_data_dir]) != real_data_dir
            or not os.path.isfile(real_path)
        ):
            print("  [sh-discover] cookie pickle path failed safety check; "
                  "skipping load.")
            return None
        with open(real_path, "rb") as f:
            cookies = pickle.load(f)
    except Exception as e:
        print(f"  [sh-discover] cookie pickle load failed: {e}")
        return None
    try:
        login = _construct_login(meta.get("email") or "", "")
        if login is None:
            return None
        # alexapy's internal cookie attribute name has drifted; assign
        # to every plausible slot we know about.
        for attr in ("_cookies", "cookies"):
            if hasattr(login, attr):
                try:
                    setattr(login, attr, cookies)
                except Exception:
                    pass
        if hasattr(login, "reset"):
            try:
                _run_async(login.reset())
            except Exception:
                pass
        return login
    except Exception as e:
        print(f"  [sh-discover] cookie restore failed: {e}; will re-login.")
        return None


# ── action implementations ──────────────────────────────────────────
_wizard_lock = threading.Lock()


def _prompt_credentials() -> tuple[str, str] | None:
    """Console prompt for Amazon email + password. The password is read
    through getpass so it never echoes to the terminal."""
    print()
    print("  [sh-discover] First-time Alexa sign-in. Your credentials go")
    print("                directly to Amazon via alexapy; the resulting")
    print("                cookie is cached at data/alexa_cookie.json.")
    print("                No plaintext password is stored on disk.")
    print()
    try:
        email = input("    Amazon email: ").strip()
        if not email:
            return None
        password = getpass.getpass("    Amazon password: ")
        if not password:
            return None
    except (EOFError, KeyboardInterrupt):
        return None
    return (email, password)


_CLI_HINT = (
    "Sir, the smart-home discovery wizard needs an interactive terminal "
    "for Amazon sign-in. Please run "
    "`python -m skills.smart_home_discover` in the JARVIS console."
)


def smart_home_discover(arg: str = "") -> str:
    """Voice action — non-blocking. Defers the interactive sign-in flow
    to a direct CLI invocation so input() never freezes the voice loop.
    If a cached cookie is present we still refresh the catalog inline
    (no input() needed); otherwise we return the CLI hint immediately."""
    if _alexapy() is None:
        return ("Smart-home discovery is offline, sir — install alexapy "
                "with `pip install alexapy` and try again.")

    force_refresh = any(w in (arg or "").lower()
                        for w in ("force", "reauth", "fresh"))

    meta = _load_cookie_meta()
    if force_refresh or not meta or not os.path.exists(_COOKIE_PICKLE_PATH):
        _say(_CLI_HINT)
        return _CLI_HINT

    if not _wizard_lock.acquire(blocking=False):
        return "Smart-home discovery wizard is already running, sir."
    try:
        _say("One moment, sir — refreshing the smart home catalog.")
        try:
            devices = _run_async(_restore_and_fetch_async(),
                                 timeout=_DISCOVERY_TIMEOUT_SEC)
        except TimeoutError:
            # 2026-07-08: a stalled Amazon endpoint must not wedge the voice
            # dispatch thread — bail with a spoken notice, catalog untouched.
            msg = ("Smart-home discovery timed out talking to Amazon, sir — "
                   "the catalog is unchanged. Try again in a moment.")
            _say(msg)
            return msg
        except Exception as e:
            print(f"  [sh-discover] fetch traceback:\n{traceback.format_exc()}")
            return f"Catalog refresh failed during device fetch: {e}"
        if devices is None:
            _say(_CLI_HINT)
            return _CLI_HINT

        arp_table = _scan_lan_arp()
        catalog = _build_catalog(devices, arp_table)
        if not force_refresh:
            catalog = _merge_with_existing_catalog(catalog)
        try:
            _save_catalog(catalog)
        except CatalogWipeRefused as e:
            print(f"  [sh-discover] {e}")
            msg = ("The device fetch came back empty, sir — likely an Alexa "
                   "sign-in or network hiccup. I've kept your existing catalog "
                   "rather than wiping it. Try again once you're reconnected.")
            _say(msg)
            return msg
        queued = _queue_missing_skill_tasks(catalog)
        return _summarise_catalog(catalog, arp_table, queued, speak=True)
    finally:
        _wizard_lock.release()


def _summarise_catalog(catalog: dict, arp_table: list[dict],
                       queued: int, speak: bool) -> str:
    n = catalog["device_count"]
    unknown = sum(1 for d in catalog["devices"] if not d["controller_skill"])
    cross = sum(1 for d in catalog["devices"] if d["lan_ip"])
    bits = [
        f"Catalog complete, sir: {n} smart-home device(s) across "
        f"{catalog['echo_count']} Echo speaker(s) and "
        f"{catalog['group_count']} group(s)."
    ]
    if unknown:
        bits.append(
            f"{unknown} brand(s) have no controller skill yet; "
            f"I've queued {queued} build task(s) in the upgrade pipeline."
        )
    else:
        bits.append("Every brand has a controller skill mapped.")
    if arp_table:
        bits.append(f"{cross} device(s) cross-referenced to a LAN address.")
    summary = " ".join(bits)
    if speak:
        _say(summary)
    return summary


def _load_cookie_pickle_safe() -> Any:
    """Sync, loop-free: load the cached cookie pickle with the SAME path-safety
    checks as _restore_login_from_cookie. Returns the cookies object or None.
    The AlexaLogin itself is built separately, INSIDE a running event loop —
    see _restore_and_fetch_async (the Python 3.14 fix)."""
    if not os.path.exists(_COOKIE_PICKLE_PATH):
        return None
    try:
        real_path = os.path.realpath(_COOKIE_PICKLE_PATH)
        real_data_dir = os.path.realpath(_DATA_DIR)
        if (real_path != os.path.realpath(os.path.join(real_data_dir,
                                                       "alexa_cookie.pickle"))
                or os.path.commonpath([real_path, real_data_dir]) != real_data_dir
                or not os.path.isfile(real_path)):
            print("  [sh-discover] cookie pickle path failed safety check; "
                  "skipping load.")
            return None
        with open(real_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [sh-discover] cookie pickle load failed: {e}")
        return None


async def _restore_and_fetch_async() -> dict | None:
    """Construct the AlexaLogin from the cached cookie AND fetch the device
    catalog — ALL inside one running event loop.

    This is the fix for the Python 3.14 'no running event loop' failure:
    alexapy/aiohttp build their ClientSession at CONSTRUCTION time and now
    REQUIRE a running loop, and that session is bound to whichever loop created
    it. The old code built the login in one (sync/closed) context and fetched
    in another loop, so it broke both ways. Running construct + reset + fetch
    under a single asyncio.run() (via _run_async) keeps them on one loop.

    Returns the raw device dict on success, or None when no usable cached
    cookie is available (caller then falls back to a fresh sign-in)."""
    if _alexapy() is None:
        return None
    cookies = _load_cookie_pickle_safe()
    if cookies is None:
        return None
    meta = _load_cookie_meta() or {}
    login = _construct_login(meta.get("email") or "", "")   # built INSIDE loop
    if login is None:
        return None
    # Authenticate using the cached cookies the PROPER way: alexapy.login()
    # validates them and populates its internal session state (csrf token,
    # customer id keyed by email). The old code only set login._cookies and
    # never called login(), so get_devices later blew up with a KeyError on the
    # account email. Try login(cookies=...) first; fall back to the legacy
    # attribute injection if this alexapy build doesn't accept the kwarg.
    authed = False
    try:
        await login.login(cookies=cookies)
        authed = bool(dict(getattr(login, "status", {}) or {})
                      .get("login_successful"))
    except TypeError:
        # Older alexapy: login() has no cookies kwarg.
        pass
    except Exception as e:
        print(f"  [sh-discover] cookie login() failed: {e}")
    if not authed:
        for attr in ("_cookies", "cookies"):
            if hasattr(login, attr):
                try:
                    setattr(login, attr, cookies)
                except Exception:
                    pass
        if hasattr(login, "reset"):
            try:
                await login.reset()
            except Exception:
                pass
        try:
            await login.login(cookies=cookies)
        except Exception:
            pass
    return await _fetch_devices_async(login)


def _run_wizard_interactive(arg: str = "") -> str:
    """Full wizard with terminal input(). ONLY call from the __main__
    guard — never from voice context (would freeze the audio loop)."""
    if not _wizard_lock.acquire(blocking=False):
        return "Smart-home discovery wizard is already running, sir."
    try:
        if _alexapy() is None:
            return ("Smart-home discovery is offline, sir — install alexapy "
                    "with `pip install alexapy` and try again.")

        force_refresh = any(w in (arg or "").lower()
                            for w in ("force", "reauth", "fresh"))
        print("  [sh-discover] Looking up your smart home devices.")

        login = None
        devices = None
        meta = _load_cookie_meta()
        if meta and not force_refresh:
            if _cookie_is_stale(meta):
                age = int((time.time() - meta.get("saved_at", 0)) / 86400)
                print(f"  [sh-discover] cached cookie is {age} days old "
                      "— a refresh may be needed soon.")
            # Build the login AND fetch the catalog in ONE event loop (Py3.14
            # needs the alexapy/aiohttp session created + used on the same
            # running loop). This reuses the cached cookie — no re-sign-in.
            try:
                devices = _run_async(_restore_and_fetch_async())
            except Exception as e:
                print(f"  [sh-discover] cached-cookie fetch failed: {e}")
                devices = None
            if devices is not None:
                print("  [sh-discover] Reused cached Amazon login — no fresh "
                      "sign-in needed.")

        creds_email = (meta or {}).get("email", "")
        if devices is None:
            creds = _prompt_credentials()
            if creds is None:
                return "Wizard cancelled — no credentials provided."
            creds_email = creds[0]
            used_playwright = False
            try:
                login = _run_async(_login_async(*creds))
            except _LoginNeedsPlaywright:
                # alexapy 1.29.22 can't parse Amazon's current login
                # response. Drive Chromium directly so the user can sign
                # in and we can capture the session cookies.
                pw_result = _run_async(_login_via_playwright(creds_email))
                if pw_result is None:
                    return ("Wizard failed — Amazon sign-in did not complete "
                            "(Playwright fallback unavailable or cancelled).")
                jar, raw_pw_cookies = pw_result
                shim = _PlaywrightLoginShim(creds_email, jar)
                try:
                    _save_cookie_meta(creds_email, dict(shim.status), jar)
                except Exception as e:
                    print(f"  [sh-discover] cookie save failed: {e}")
                used_playwright = True
                # The cookie pickle is now saved — enumerate via the SAME
                # single-loop path the cached case uses (construct + login +
                # fetch all inside one event loop). Avoids the old
                # "no running event loop" failure from building the AlexaLogin
                # outside a loop, so the user gets devices immediately instead
                # of a "please re-run" message.
                try:
                    devices = _run_async(_restore_and_fetch_async())
                except Exception as e:
                    print(f"  [sh-discover] post-signin fetch failed: {e}")
                    devices = None
                login = None
            except Exception as e:
                print(f"  [sh-discover] login traceback:\n{traceback.format_exc()}")
                return f"Wizard failed during sign-in: {e}"
            if used_playwright and devices is None:
                return ("Amazon sign-in captured, sir, but the device fetch "
                        "came back empty — the cached Alexa cookie may not be "
                        "scoped correctly. Cookies are at "
                        f"{os.path.relpath(_COOKIE_JSON_PATH, _PROJECT_DIR)}.")
            if login is None and not used_playwright:
                return "Wizard failed — Amazon sign-in did not complete."
            if not used_playwright:
                try:
                    jar = _extract_cookie_jar(login)
                    _save_cookie_meta(creds_email,
                                      dict(login.status or {}), jar)
                except Exception as e:
                    print(f"  [sh-discover] cookie save failed: {e}")

        if devices is None:
            # Fresh sign-in path only — the cached path already fetched above.
            print("  [sh-discover] Signed in. Enumerating devices.")
            try:
                devices = _run_async(_fetch_devices_async(login))
            except Exception as e:
                print(f"  [sh-discover] fetch traceback:\n{traceback.format_exc()}")
                return f"Wizard failed during device fetch: {e}"

        arp_table = _scan_lan_arp()
        catalog = _build_catalog(devices, arp_table)
        if not force_refresh:
            catalog = _merge_with_existing_catalog(catalog)
        try:
            _save_catalog(catalog)
        except CatalogWipeRefused as e:
            print(f"  [sh-discover] {e}")
            return ("The device fetch came back empty — I kept the existing "
                    "catalog rather than overwriting it with nothing. Check the "
                    "Alexa sign-in / network and re-run.")
        queued = _queue_missing_skill_tasks(catalog)
        return _summarise_catalog(catalog, arp_table, queued, speak=False)
    finally:
        _wizard_lock.release()


def smart_home_catalog(_: str = "") -> str:
    """Speak a short summary of the cached catalog."""
    cat = _load_catalog()
    if not cat:
        return ("No smart-home catalog yet, sir. "
                "Say 'discover smart home devices' to run the wizard.")
    n = cat.get("device_count", 0)
    rooms: dict[str, int] = {}
    for d in cat.get("devices", []):
        r = d.get("alexa_room") or "(unassigned)"
        rooms[r] = rooms.get(r, 0) + 1
    top = sorted(rooms.items(), key=lambda x: -x[1])[:5]
    parts = [f"{count} in {room}" for room, count in top]
    return f"{n} smart-home devices, sir: " + ", ".join(parts) + "."


def smart_home_purge_cookie(_: str = "") -> str:
    """Remove the cached Alexa cookie so the next wizard run does a
    fresh interactive sign-in."""
    removed = 0
    for p in (_COOKIE_JSON_PATH, _COOKIE_PICKLE_PATH):
        try:
            if os.path.exists(p):
                os.unlink(p)
                removed += 1
        except Exception as e:
            print(f"  [sh-discover] purge failed for {p}: {e}")
    return (f"Alexa cookie cleared ({removed} file(s) removed), sir."
            if removed else
            "No cached Alexa cookie to clear, sir.")


def register(actions: dict) -> None:
    actions["smart_home_discover"]     = smart_home_discover
    actions["discover_smart_home"]     = smart_home_discover
    actions["smart_home_setup"]        = smart_home_discover
    actions["refresh_smart_home"]      = smart_home_discover
    actions["smart_home_catalog"]      = smart_home_catalog
    actions["list_smart_home_devices"] = smart_home_catalog
    actions["smart_home_purge_cookie"] = smart_home_purge_cookie
    actions["forget_alexa_login"]      = smart_home_purge_cookie


if __name__ == "__main__":
    # Interactive sign-in path. Voice actions route here via the CLI
    # hint to keep input() / getpass() off the main audio loop.
    import sys
    cli_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    result = _run_wizard_interactive(cli_arg)
    print()
    print(result)
