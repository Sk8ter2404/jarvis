"""
Smart-home discovery — Alexa + live LAN.

Populates the canonical device catalog at `data/smart_home_devices.json`
from TWO sources, each reported honestly per-source:

  1. Live LAN discovery via the brand controller skills
     (sh_kasa / sh_tuya / sh_govee / sh_hue `list_devices()`), bounded
     by per-source and overall deadlines — needs no Amazon account.
  2. The Amazon Alexa graph (via alexapy + a cached cookie), which adds
     rooms, groups and entity ids when a cached sign-in is available.

Records from both sources are merged (a LAN record enriches its Alexa
twin instead of duplicating it), cross-referenced against a LAN ARP
scan to recover IP + MAC where possible, and consumed by the smart-home
router (`core/smart_home_router.py`, research-4b) which dispatches
per-brand direct API calls without ongoing Alexa dependency.

The FIRST Amazon sign-in is a one-time console wizard (run
`python -m skills.smart_home_discover`, or `... reauth` to force a new one).
It opens a browser at Amazon's phone-app OAuth page (alexapy's
`AlexaLogin(oauth_login=True).start_url`), the owner signs in there
(password / CAPTCHA / 2FA never touch JARVIS), and the authorization code
Amazon redirects back with registers JARVIS as an Alexa device. The refresh
token is saved to `data/alexa_oauth.json` — later runs restore from it with no
browser. The pre-2026-10 cookie cache (`data/alexa_cookie.json` +
`data/alexa_cookie.pickle`) is still tried as a fallback. The voice action
never blocks on a terminal: without a cached sign-in it still runs the LAN
phase and reports the Alexa source as unavailable.

Smart-home appliances (lights, fans, plugs, thermostats, cameras...) come
from `AlexaAPI.get_network_details`; `get_devices` is only the Echo hardware.
Secrets (cookie jars, tokens) are scrubbed from every log line by
`_redact` / `_install_log_redaction`.

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
    alexapy   ← needed only for the Alexa source (pip install alexapy).
              When absent the LAN sources still run; the summary
              reports Alexa discovery as unavailable and the rest of
              JARVIS is untouched.

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
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from typing import Any


_PROJECT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# This module is a runtime-state WRITER; a private join here is how a
# staging-isolated sweep overwrote the LIVE smart-home catalog.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_COOKIE_JSON_PATH   = os.path.join(_DATA_DIR, "alexa_cookie.json")
_COOKIE_PICKLE_PATH = os.path.join(_DATA_DIR, "alexa_cookie.pickle")
_CATALOG_PATH       = os.path.join(_DATA_DIR, "smart_home_devices.json")
_TODO_PATH          = os.path.join(_PROJECT_DIR, "jarvis_todo.md")
# Alexa OAuth device-registration material (access/refresh token, mac_dms,
# PKCE verifier, device uuid). SECRET; `data/*` is gitignored. The path is
# resolved at call time from _DATA_DIR (see _oauth_path) so a test or staging
# redirect of _DATA_DIR can never touch the owner's live file.
_OAUTH_FILE_NAME    = "alexa_oauth.json"

# Cookie freshness — Amazon cookies last ~1 year, but we warn here so
# the user can refresh proactively rather than discovering an expired
# session mid-command.
_COOKIE_SUGGEST_REFRESH_DAYS = 300

# 2026-07-08: hard ceiling for a voice-context catalog refresh. The alexapy /
# aiohttp fetch runs on the assistant dispatch thread; a stalled Amazon endpoint
# with no timeout would wedge the whole voice loop. 45s is generous for a warm
# cookie fetch yet bounded enough to fail fast and speak a timeout notice.
_DISCOVERY_TIMEOUT_SEC = 45.0

# Live-LAN discovery sources — each is a brand controller skill exposing the
# uniform `list_devices()` API. The catalog is built from these PLUS the Alexa
# fetch, so a dead Alexa cookie can no longer leave the catalog empty forever.
_LAN_SOURCES = ("sh_kasa", "sh_tuya", "sh_govee", "sh_hue")
_LAN_SOURCE_LABELS = {"sh_kasa": "Kasa", "sh_tuya": "Tuya",
                      "sh_govee": "Govee", "sh_hue": "Hue"}
# Bounds for the LAN phase (voice dispatch thread — must never wedge). The
# per-source ceiling covers the worst internal budgets (kasa 5s UDP broadcast,
# govee 1.5s LAN + 6s cloud, hue 4s meethue + 3s bridge connect, tuya ~0);
# the overall deadline is shared by all source threads joined in parallel.
# Stragglers are abandoned as daemon threads — the established pattern of
# `_run_async` below and sh_hue._threaded_connect.
_LAN_SOURCE_TIMEOUT_SEC = 12.0
_LAN_TOTAL_TIMEOUT_SEC = 15.0

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


# ── secret redaction (2026-10-12) ───────────────────────────────────
# SECURITY BUG (owner-reported): when a cookie login failed, alexapy's
# `_catch_all_exceptions` logged WARNING "...(args, kwargs)..." with the FULL
# requests cookie jar repr (live Amazon session tokens) and JARVIS tees
# stderr into logs/session_*.log. Everything below keeps secrets out of any log:
#   * _redact()               pure text scrubber (jars, Cookie reprs, Atza|...
#                             tokens, named session / OAuth values),
#   * _RedactingFilter        logging.Filter applying it to a record,
#   * _install_log_redaction  idempotently hangs the filter on the "alexapy"
#                             logger tree, every root / alexapy handler and the
#                             logging.lastResort stderr handler, AND wraps the
#                             LogRecord factory for alexapy / aiohttp records.
#                             Logger-level filters do NOT apply to child
#                             loggers ("alexapy.helpers") and alexapy's loggers
#                             are created lazily at import, so the factory is
#                             the part that actually guarantees coverage.
# Every print() in this module that can carry an exception from alexapy goes
# through _redact() as well.
_REDACTED = "[REDACTED]"
_SECRET_NAMES = (
    "openid.oa2.authorization_code", "openid.oa2.access_token",
    "authorization_code", "code_verifier", "code_challenge",
    "access_token", "refresh_token", "mac_dms", "id_token",
    "sess-at-main", "sess-at-acbus", "at-main", "at-acbus", "x-main",
    "x-acbus", "ubid-main", "ubid-acbus", "session-token", "session-id-time",
    "session-id", "sst-main", "lc-main", "csrf-token", "csrf_token", "csrf",
)
_NAMED_SECRET_RE = re.compile(
    r"(?P<name>(?<![\w.-])(?:"
    + "|".join(re.escape(n) for n in sorted(_SECRET_NAMES, key=len, reverse=True))
    + r")(?![\w-]))"
    r"""(?P<sep>["']?\s*[:=]\s*)(?P<q>["']?)(?P<val>(?!\[REDACTED\])[^\s"',;&>)\]}]+)""",
    re.IGNORECASE)
_REDACT_RULES: tuple = (
    # whole jar repr: <RequestsCookieJar[<Cookie a=b for .x/>, ...]>
    (re.compile(r"<?RequestsCookieJar\[.*?\]>", re.DOTALL),
     "<RequestsCookieJar[" + _REDACTED + "]>"),
    # truncated / unterminated jar: redact the rest of the line
    (re.compile(r"RequestsCookieJar\[(?!\[REDACTED\])[^\n]*"),
     "RequestsCookieJar[" + _REDACTED + "]"),
    # aiohttp / http.cookiejar reprs
    (re.compile(r"<Cookie\s+([^=\s>]+)=\S*(\s+for\s)"),
     r"<Cookie \1=" + _REDACTED + r"\2"),
    (re.compile(r"""(\bvalue\s*=\s*)(['"]).*?\2"""),
     r"\1'" + _REDACTED + "'"),
    # raw Cookie / Set-Cookie / Authorization headers
    (re.compile(r"(?i)\b((?:Set-)?Cookie|Authorization)(\s*:\s*)[^\r\n]+"),
     r"\1\2" + _REDACTED),
    # name=value / "name": "value" for the known session + OAuth secrets
    (_NAMED_SECRET_RE, r"\g<name>\g<sep>\g<q>" + _REDACTED),
    # Amazon access / refresh token shapes, wherever else they appear
    (re.compile(r"\bAt(?:za|nr|n)\|[^\s\"',;)>\]}]+"), _REDACTED),
)


def _redact(text: Any) -> str:
    """Return `text` with cookie jars and token / session values replaced by
    [REDACTED]. Pure, never raises."""
    try:
        out = text if isinstance(text, str) else str(text)
        for rx, repl in _REDACT_RULES:
            out = rx.sub(repl, out)
        return out
    except Exception:
        return _REDACTED


class _RedactingFilter(logging.Filter):
    """logging.Filter that scrubs secrets out of a record (message, args and
    any attached traceback) before any handler formats it. Always passes the
    record through: it redacts, it never drops."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            clean = _redact(msg)
            if clean != msg:
                record.msg = clean
                record.args = ()
            if record.exc_info and not record.exc_text:
                try:
                    record.exc_text = logging.Formatter().formatException(
                        record.exc_info)
                except Exception:
                    pass
            if record.exc_text:
                record.exc_text = _redact(record.exc_text)
        except Exception:
            pass
        return True


_REDACT_LOGGER_PREFIXES = ("alexapy", "aiohttp")


def _install_log_redaction() -> None:
    """Idempotently route every alexapy / aiohttp log record through
    _RedactingFilter (see the block comment above). Safe to call repeatedly;
    never raises."""
    try:
        flt = getattr(logging, "_jarvis_alexa_redactor", None)
        if flt is None:
            flt = _RedactingFilter()
            logging._jarvis_alexa_redactor = flt        # type: ignore[attr-defined]
            prev_factory = logging.getLogRecordFactory()

            def _factory(*a: Any, **kw: Any) -> logging.LogRecord:
                rec = prev_factory(*a, **kw)
                try:
                    if str(rec.name).startswith(_REDACT_LOGGER_PREFIXES):
                        flt.filter(rec)
                except Exception:
                    pass
                return rec

            logging.setLogRecordFactory(_factory)
        # Loggers: the parent plus every child that already exists.
        names = ["alexapy"] + [
            n for n in list(logging.root.manager.loggerDict)
            if str(n).startswith("alexapy.")]
        for n in names:
            lg = logging.getLogger(n)
            if flt not in lg.filters:
                lg.addFilter(flt)
            for h in lg.handlers:
                if flt not in h.filters:
                    h.addFilter(flt)
        # Handlers: root's, plus the last-resort stderr handler that prints
        # WARNING+ when nothing else is configured (the path of the leak).
        handlers = list(logging.root.handlers)
        if logging.lastResort is not None:
            handlers.append(logging.lastResort)
        for h in handlers:
            if flt not in h.filters:
                h.addFilter(flt)
    except Exception:
        pass


_install_log_redaction()


# ── lazy alexapy import ─────────────────────────────────────────────
def _alexapy():
    """Lazy import so a missing alexapy install can't crash skill loading."""
    try:
        import alexapy  # type: ignore
        _install_log_redaction()    # its child loggers exist now
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
# fields are refreshed from the live discovery. EXCEPTION: for a record the
# LIVE LAN scan just produced (`source` starts with "lan:"), the fresh
# `lan_ip`/`lan_mac` win — a scan answer is ground truth and DHCP moves IPs,
# so preserving a stale manual address would break direct control.
_USER_OVERRIDE_FIELDS = ("name", "controller_skill", "lan_ip", "lan_mac")


def _merge_with_existing_catalog(fresh: dict) -> dict:
    """Merge a freshly-built catalog with the existing on-disk catalog,
    preserving user overrides to the fields listed in
    `_USER_OVERRIDE_FIELDS`. Devices are matched by `alexa_entity_id`
    (Alexa records) or `device_id` (LAN records); devices absent from the
    existing file are kept as-is from `fresh`, and devices absent from
    `fresh` (i.e. removed at the source) are dropped."""
    existing = _load_catalog()
    if not existing or not isinstance(existing.get("devices"), list):
        return fresh
    by_id: dict[str, dict] = {}
    for d in existing["devices"]:
        if isinstance(d, dict):
            eid = d.get("alexa_entity_id") or d.get("device_id")
            if eid:
                by_id[eid] = d
    if not by_id:
        return fresh
    for d in fresh.get("devices", []):
        eid = d.get("alexa_entity_id") or d.get("device_id")
        if not eid:
            continue
        prev = by_id.get(eid)
        if not prev:
            continue
        fresh_is_lan = str(d.get("source") or "").startswith("lan:")
        for k in _USER_OVERRIDE_FIELDS:
            if fresh_is_lan and k in ("lan_ip", "lan_mac"):
                continue    # live scan wins for addresses (see comment above)
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
                iface = (item.get("interface") or item.get("namespace")
                         or item.get("interfaceName") or "")
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
def _construct_login(email: str, password: str, *, oauth: dict | None = None,
                     uuid: str | None = None,
                     oauth_login: bool | None = None) -> Any:
    """Build an AlexaLogin, tolerating API drift between alexapy versions.
    Newer versions accept a callable for `outputpath`; older versions
    want a directory path string.

    `oauth` / `uuid` / `oauth_login` (2026-10-12) are the device-registration
    extras: `oauth` is the dict AlexaLogin restores its tokens from
    (access_token, refresh_token, mac_dms, expires_in, code_verifier,
    code_challenge, authorization_code), `uuid` is the stable device serial the
    registration was made under (the PKCE `deviceid` derives from it), and
    `oauth_login=True` selects the phone-app OAuth start_url. Left unset the
    constructor call is byte-for-byte what it always was. MUST be called from
    inside a running event loop (the aiohttp session binds to it)."""
    alexapy = _alexapy()
    if alexapy is None:
        return None
    AlexaLogin = alexapy.AlexaLogin

    def _out(name: str) -> str:
        # alexapy's `outputpath` is a PATH BUILDER, not a log sink: it calls
        # outputpath(".storage/alexa_media.<email>.pickle") and string-methods
        # the RESULT. The old callback printed and returned None, so
        # finalize_login's eager `self._cookiefile[0].replace(...)` raised
        # "'NoneType' object has no attribute 'replace'" and every login died
        # at the very last step (and each construct printed the account email
        # into the log). Return a real path under a directory we deliberately
        # never create: alexapy's own cookie-file save then fails quietly
        # (OSError, swallowed) instead of writing a second copy of the session
        # to disk — our persistence is data/alexa_oauth.json.
        return os.path.join(_DATA_DIR, "alexapy", str(name))

    extra: dict[str, Any] = {}
    if oauth:
        extra["oauth"] = dict(oauth)
    if uuid:
        extra["uuid"] = uuid
    if oauth_login is not None:
        extra["oauth_login"] = bool(oauth_login)

    candidates = (
        dict(url="amazon.com", email=email, password=password,
             outputpath=_out, debug=False, **extra),
        dict(url="amazon.com", email=email, password=password,
             outputpath=_DATA_DIR, debug=False, **extra),
        dict(url="amazon.com", email=email, password=password, **extra),
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
    print(f"  [sh-discover] could not construct AlexaLogin: {_redact(last_err)}")
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


# ── Playwright fallback (RETIRED 2026-10-12) ────────────────────────
# RETIRED: nothing in the wizard calls `_login_via_playwright` any more. It
# signs in through the alexa.amazon.com web app, which Amazon shut down; the
# cookies it captures never carry an Alexa session. The live sign-in is the
# OAuth flow further down (`_oauth_signin_and_fetch_async`). Kept, with the
# URL below, only so its tests and any out-of-tree caller keep working.
#
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


async def _launch_headed_browser(pw: Any) -> Any:
    """Launch a HEADED browser for an interactive Amazon sign-in, or None.

    Prefers the user's installed Google Chrome (channel="chrome"):
    Playwright's BUNDLED Chromium throws "spawn UNKNOWN" on some Windows
    setups, while the system Chrome the user runs daily launches cleanly.
    Chain: Chrome -> bundled Chromium -> Edge so this works across machines.
    2026-05-30; hoisted into a helper 2026-10-12 so the OAuth sign-in shares it."""
    browser = None
    launch_errs: list[str] = []
    for chan in ("chrome", None, "msedge"):
        try:
            if chan is None:
                browser = await pw.chromium.launch(headless=False)
            else:
                browser = await pw.chromium.launch(headless=False, channel=chan)
            print(f"  [sh-discover] Sign-in browser: "
                  f"{chan or 'bundled chromium'}.")
            break
        except Exception as le:
            launch_errs.append(f"{chan or 'bundled'}: {le}")
            browser = None
    if browser is None:
        print("  [sh-discover] Could not launch any headed browser for "
              "sign-in:\n      " + "\n      ".join(launch_errs))
    return browser


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
    print("  [sh-discover] Opening a Chromium window so you can sign in to Amazon.")
    print("                Complete sign-in normally (CAPTCHA / 2FA included);")
    print("                JARVIS will capture the session cookies automatically")
    print("                once Amazon redirects you off the /ap/signin page.")
    print(f"                Timeout: {int(timeout_seconds)}s.")
    print()

    try:
        async with async_playwright() as pw:
            browser = await _launch_headed_browser(pw)
            if browser is None:
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


def _inject_pw_cookies(login: Any, pw_cookies: list[dict]) -> bool:
    """Copy Playwright-captured cookies into `login`'s aiohttp cookie jar.
    True when the jar was reachable (individual bad cookies are skipped);
    False if the session / jar is missing or injection blew up."""
    try:
        from http.cookies import SimpleCookie
        from yarl import URL as YURL
        session = getattr(login, "_session", None) or getattr(login, "session", None)
        if session is None:
            return False
        jar = getattr(session, "cookie_jar", None)
        if jar is None:
            return False
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
        print(f"  [sh-discover] Cookie injection into alexapy session failed: "
              f"{_redact(e)}")
        return False
    return True


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
    if not _inject_pw_cookies(login, pw_cookies):
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


# ── OAuth device-registration sign-in (2026-10-12) ──────────────────
# WHY: the old fallback signed in through the RETIRED alexa.amazon.com web app
# (assoc_handle=amzn_dp_project_dee). Amazon shut that app down, so the browser
# only yielded plain www.amazon.com cookies with no Alexa csrf, alexapy's
# login(cookies=...) then died with "'NoneType' object has no attribute
# 'replace'" and the devices call bounced to /ap/signin: zero Alexa devices.
#
# NOW: the flow the Alexa phone app (and alexapy's own HA config-flow proxy)
# uses. AlexaLogin(oauth_login=True).start_url is an /ap/register page tied to
# a PKCE challenge + a device id. The owner signs in there (password, CAPTCHA,
# 2FA all in the real browser; the password never touches JARVIS). Amazon then
# redirects to /ap/maplanding?openid.oa2.authorization_code=<code>. We take
# the code, set login.authorization_code, and run alexapy's own post-maplanding
# sequence (AlexaLogin.test_loggedin): get_tokens (register this "device" with
# the code + code_verifier -> access / refresh token + mac_dms) ->
# register_capabilities -> exchange_token_for_cookies -> get_csrf -> identity
# check -> finalize_login (status login_successful). The refresh token is then
# persisted to data/alexa_oauth.json so the owner signs in ONCE: later runs
# restore from it with no browser.
_OAUTH_SIGNIN_TIMEOUT_S = 300.0     # 5 min: password + CAPTCHA + 2FA
# AlexaLogin only runs its "signed-in email == this email" check when the
# identity contains an '@'. We never ask for the email (the owner types it into
# Amazon's own page), so use a label without one: the check is skipped and a
# mistyped / different-from-expected address can't fail the sign-in.
_OAUTH_IDENTITY = "jarvis-owner"
_OAUTH_CODE_PARAM = "openid.oa2.authorization_code"
_MAPLANDING_PATH = "/ap/maplanding"
# The login attributes AlexaLogin(oauth=...) restores (alexalogin.py __init__).
_OAUTH_RESTORE_KEYS = ("access_token", "refresh_token", "mac_dms", "expires_in",
                       "code_verifier", "code_challenge", "authorization_code")
_REAUTH_HINT = "python -m skills.smart_home_discover reauth"


def _oauth_path() -> str:
    """Where the OAuth material lives; follows the CURRENT _DATA_DIR."""
    return os.path.join(_DATA_DIR, _OAUTH_FILE_NAME)


def _save_oauth(login: Any, identity: str = _OAUTH_IDENTITY) -> bool:
    """Persist everything AlexaLogin needs to restore a registered device.
    Never prints a value. Returns False (and writes nothing) when the login
    holds no refresh token, i.e. there is nothing restorable."""
    refresh = getattr(login, "refresh_token", None)
    if not refresh:
        return False
    rec: dict[str, Any] = {
        "version": 1,
        "identity": identity,
        "saved_at": time.time(),
        "saved_at_iso": datetime.datetime.now(datetime.timezone.utc)
                        .replace(tzinfo=None).isoformat() + "Z",
        "uuid": getattr(login, "uuid", None),
        "deviceid": getattr(login, "deviceid", None),
    }
    for key in _OAUTH_RESTORE_KEYS:
        val = getattr(login, key, None)
        if val is not None:
            rec[key] = val
    try:
        _atomic_write_json(_oauth_path(), rec)
    except Exception as e:
        print(f"  [sh-discover] could not save the Alexa sign-in: {_redact(e)}")
        return False
    return True


def _load_oauth() -> dict | None:
    """The saved OAuth record, or None when absent / unreadable / unusable
    (no refresh token)."""
    path = _oauth_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except Exception:
        return None
    if not isinstance(rec, dict):
        return None
    if not isinstance(rec.get("refresh_token"), str) or not rec["refresh_token"]:
        return None
    return rec


def _authorized_amazon_url(url: str) -> urllib.parse.SplitResult | None:
    """Parsed `url` iff it is an https amazon.com page (the only place the
    OAuth redirect may come from), else None."""
    try:
        u = urllib.parse.urlsplit(str(url or ""))
        host = (u.hostname or "").lower()
        if u.scheme == "https" and (host == "amazon.com"
                                    or host.endswith(".amazon.com")):
            return u
    except Exception:
        pass
    return None


def _authorization_code_from_url(url: str) -> str | None:
    """The `openid.oa2.authorization_code` carried by an amazon.com URL, or
    None. Host-checked so a page we don't trust can't feed us a code."""
    u = _authorized_amazon_url(url)
    if u is None:
        return None
    vals = urllib.parse.parse_qs(u.query).get(_OAUTH_CODE_PARAM) or []
    code = (vals[0] if vals else "").strip()
    if not code or len(code) > 2048 or re.search(r"\s", code):
        return None
    return code


def _is_maplanding_url(url: str) -> bool:
    u = _authorized_amazon_url(url)
    return u is not None and u.path.rstrip("/") == _MAPLANDING_PATH


async def _close_login(login: Any) -> None:
    """Release the login's aiohttp connector. Never raises."""
    close = getattr(login, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        pass


async def _browser_oauth_code(
    start_url: str,
    timeout_seconds: float = _OAUTH_SIGNIN_TIMEOUT_S,
) -> tuple[str, list[dict]] | None:
    """Open a HEADED browser at the OAuth `start_url` and wait for Amazon's
    redirect to /ap/maplanding. Returns (authorization_code, browser_cookies)
    or None on every failure: Playwright missing, no launchable browser, page
    won't load, timeout, window closed first, or a landing without a code
    (sign-in cancelled / refused). Every outcome prints a plain reason; the
    code and cookies are never printed."""
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        print(f"  [sh-discover] Playwright not available ({e}); install "
              "with `pip install playwright` to sign in.")
        return None

    print()
    print("  [sh-discover] Opening a browser window so you can sign in to Amazon.")
    print("                Sign in normally (password, CAPTCHA, 2FA); JARVIS")
    print("                never sees your password. When Amazon finishes it")
    print("                lands on a mostly-blank page and this window closes.")
    print(f"                Timeout: {int(timeout_seconds)}s.")
    print()

    seen: dict[str, Any] = {"code": None, "landed": False}

    def _see(url: str) -> None:
        if seen["code"]:
            return
        code = _authorization_code_from_url(url)
        if code:
            seen["code"] = code
        elif _is_maplanding_url(url):
            seen["landed"] = True

    reason = "timeout"
    pw_cookies: list[dict] = []
    try:
        async with async_playwright() as pw:
            browser = await _launch_headed_browser(pw)
            if browser is None:
                return None
            try:
                context = await browser.new_context()
                page = await context.new_page()
                # Catch the redirect from navigation events too, not only the
                # poll below: maplanding is a transit page and a slow poll
                # could otherwise miss the URL that carries the code.
                for evt, attr in (("framenavigated", "url"), ("request", "url")):
                    try:
                        page.on(evt, lambda obj, _a=attr: _see(
                            str(getattr(obj, _a, "") or "")))
                    except Exception:
                        pass
                try:
                    await page.goto(start_url)
                except Exception as e:
                    if not seen["code"]:
                        print("  [sh-discover] Failed to load the Amazon "
                              f"sign-in page: {_redact(e)}")
                        return None
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout_seconds
                while True:
                    try:
                        _see(page.url or "")
                    except Exception:
                        pass
                    if seen["code"]:
                        reason = "ok"
                        break
                    if seen["landed"]:
                        reason = "landed_without_code"
                        break
                    try:
                        closed = bool(page.is_closed())
                    except Exception:
                        closed = True
                    if closed:
                        reason = "closed"
                        break
                    if loop.time() > deadline:
                        reason = "timeout"
                        break
                    await asyncio.sleep(1.0)
                if reason == "ok":
                    try:
                        pw_cookies = await context.cookies()
                    except Exception:
                        pw_cookies = []
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as e:
        if not seen["code"]:
            print(f"  [sh-discover] Browser sign-in failed: {_redact(e)}")
            return None
        reason = "ok"
    if reason == "ok" and seen["code"]:
        print("  [sh-discover] Amazon sign-in finished; registering this "
              "JARVIS as an Alexa device.")
        return (seen["code"], list(pw_cookies or []))
    msgs = {
        "timeout": "Sign-in timed out before Amazon finished.",
        "closed": ("The sign-in window was closed before Amazon finished; "
                   "nothing was saved."),
        "landed_without_code": ("Amazon ended the sign-in without granting "
                                "access (cancelled or refused); nothing was "
                                "saved."),
    }
    print("  [sh-discover] " + msgs.get(reason, "Sign-in did not complete."))
    return None


async def _complete_oauth_login(login: Any,
                                pw_cookies: list[dict] | None = None) -> bool:
    """Run alexapy's own post-maplanding sequence on a login whose
    `authorization_code` is set. AlexaLogin.test_loggedin() is that sequence:
    get_tokens -> register_capabilities -> exchange_token_for_cookies ->
    get_csrf -> /api/users/me identity check -> check_domain ->
    finalize_login. True only when it succeeded AND the login reports
    login_successful."""
    if pw_cookies:
        # alexapy's own proxy flow copies the browser's cookies into the
        # login session before registering; mirror it (best-effort).
        _inject_pw_cookies(login, pw_cookies)
    try:
        ok = bool(await login.test_loggedin())
    except Exception as e:
        print(f"  [sh-discover] Alexa device registration raised: {_redact(e)}")
        return False
    status = dict(getattr(login, "status", None) or {})
    if ok and status.get("login_successful"):
        return True
    if not getattr(login, "refresh_token", None):
        print("  [sh-discover] Amazon did not register the device "
              "(no refresh token came back).")
    else:
        print("  [sh-discover] Registered, but the Alexa session could not be "
              "verified.")
    return False


async def _oauth_refresh_session(login: Any) -> bool:
    """Rebuild a working session from a restored refresh token: refresh the
    access token, exchange the refresh token for fresh Alexa cookies, find the
    csrf cookie, and mark the login successful. The first two are required;
    a missing csrf only costs write calls, not the device listing."""
    if not getattr(login, "refresh_token", None):
        return False
    if not await login.refresh_access_token():
        print("  [sh-discover] Amazon refused the saved sign-in (token "
              "refresh failed).")
        return False
    if not await login.exchange_token_for_cookies():
        print("  [sh-discover] Amazon would not exchange the saved sign-in "
              "for session cookies.")
        return False
    try:
        csrf_ok = bool(await login.get_csrf())
    except Exception:
        csrf_ok = False
    if not csrf_ok:
        print("  [sh-discover] No Alexa csrf token found; device listing may "
              "still work, controlling devices may not.")
    fin = getattr(login, "finalize_login", None)
    try:
        if fin is None:
            raise AttributeError("finalize_login")
        await fin()
    except Exception:
        login.status = {"login_successful": True}
    if not dict(getattr(login, "status", None) or {}).get("login_successful"):
        login.status = {"login_successful": True}
    return True


async def _oauth_restore_and_fetch_async() -> dict | None:
    """Restore the saved OAuth sign-in and fetch the device catalog, all in
    ONE running event loop (the Python 3.14 rule; see
    _restore_and_fetch_async). Returns the raw device dict, or None when there
    is no saved sign-in or Amazon refused it (the caller then falls back to
    the legacy cookie pickle, then to a browser sign-in). Never deletes the
    saved file: a transient network failure must not cost the owner his
    sign-in."""
    if _alexapy() is None:
        return None
    rec = _load_oauth()
    if rec is None:
        return None
    identity = rec.get("identity") or _OAUTH_IDENTITY
    oauth = {k: rec[k] for k in _OAUTH_RESTORE_KEYS if rec.get(k) is not None}
    login = _construct_login(identity, "", oauth=oauth,
                             uuid=rec.get("uuid"), oauth_login=True)
    if login is None:
        return None
    try:
        if not await _oauth_refresh_session(login):
            return None
        _save_oauth(login, identity)        # the access token / expiry moved
        return await _fetch_devices_async(login)
    except Exception as e:
        print(f"  [sh-discover] saved Alexa sign-in failed: {_redact(e)}")
        return None
    finally:
        await _close_login(login)


async def _oauth_login_async() -> Any:
    """Restore the saved OAuth sign-in and hand back the LOGGED-IN, still-open
    AlexaLogin (2026-10-13; the smart-home router drives devices with it).
    Must run inside the event loop the caller will keep USING the login on:
    the aiohttp session binds to the loop that created it, so the caller owns
    that loop and the login's lifetime (close it with _close_login). None when
    there is no saved sign-in or Amazon refused it. Never deletes the saved
    file; never prints a token."""
    if _alexapy() is None:
        return None
    rec = _load_oauth()
    if rec is None:
        return None
    identity = rec.get("identity") or _OAUTH_IDENTITY
    oauth = {k: rec[k] for k in _OAUTH_RESTORE_KEYS if rec.get(k) is not None}
    login = _construct_login(identity, "", oauth=oauth,
                             uuid=rec.get("uuid"), oauth_login=True)
    if login is None:
        return None
    try:
        if not await _oauth_refresh_session(login):
            await _close_login(login)
            return None
        _save_oauth(login, identity)        # the access token / expiry moved
        return login
    except Exception as e:
        print(f"  [sh-discover] saved Alexa sign-in failed: {_redact(e)}")
        await _close_login(login)
        return None


async def _oauth_signin_and_fetch_async(
    timeout_seconds: float = _OAUTH_SIGNIN_TIMEOUT_S,
) -> dict | None:
    """Interactive first sign-in: browser -> authorization code -> device
    registration -> persist -> fetch devices, ALL in one running event loop.
    Returns the raw device dict, or None if the owner didn't finish (or
    Amazon refused)."""
    if _alexapy() is None:
        return None
    login = _construct_login(_OAUTH_IDENTITY, "", oauth_login=True)
    if login is None:
        return None
    try:
        got = await _browser_oauth_code(str(login.start_url), timeout_seconds)
        if got is None:
            return None
        code, pw_cookies = got
        login.authorization_code = code
        if not await _complete_oauth_login(login, pw_cookies):
            return None
        if _save_oauth(login):
            print("  [sh-discover] Saved the Alexa sign-in to "
                  f"data/{_OAUTH_FILE_NAME} - you will not need to sign "
                  "in again.")
        return await _fetch_devices_async(login)
    except Exception as e:
        print(f"  [sh-discover] Alexa sign-in failed: {_redact(e)}")
        return None
    finally:
        await _close_login(login)


async def _restore_any_and_fetch_async() -> dict | None:
    """Cached-sign-in restore in priority order: (1) the OAuth refresh token
    (data/alexa_oauth.json), then (2) the legacy cookie pickle. None when
    neither is usable."""
    if _load_oauth() is not None:
        try:
            devices = await _oauth_restore_and_fetch_async()
        except Exception as e:
            print(f"  [sh-discover] saved Alexa sign-in failed: {_redact(e)}")
            devices = None
        if devices is not None:
            return devices
    return await _restore_and_fetch_async()


async def _fetch_devices_async(login: Any) -> dict:
    """Best-effort fetch of every device list alexapy exposes. Each call
    is wrapped individually because alexapy's API surface drifts and an
    older install may be missing one of these methods.

    Smart-home appliances (lights, fans, plugs, thermostats, cameras, ...)
    come from `AlexaAPI.get_network_details` (alexapy >= 1.29: the Alexa
    GraphQL `endpoints` list, one `legacyAppliance` dict each - applianceId,
    entityId, friendlyName, manufacturerName, modelName, capabilities,
    applianceTypes). `get_devices` is ONLY the Echo / Fire TV hardware list:
    the pre-2026-10-12 code looked for get_smarthome_devices / get_appliances,
    which no alexapy has ever shipped, so the smart-home list was always empty.
    The old names stay as fallbacks for other alexapy builds."""
    alexapy = _alexapy()
    if alexapy is None or login is None:
        return {}
    AlexaAPI = alexapy.AlexaAPI

    out: dict = {"echo": [], "smarthome": [], "groups": []}
    try:
        if hasattr(AlexaAPI, "get_devices"):
            out["echo"] = await AlexaAPI.get_devices(login) or []
    except Exception as e:
        print(f"  [sh-discover] get_devices failed: {_redact(e)}")
    for name in ("get_network_details", "get_smarthome_devices",
                 "get_appliances"):
        fn = getattr(AlexaAPI, name, None)
        if not callable(fn):
            continue
        try:
            got = await fn(login) or []
        except Exception as e:
            print(f"  [sh-discover] {name} failed: {_redact(e)}")
            continue
        if got:
            out["smarthome"] = got
            break
    if not out["smarthome"] and callable(getattr(AlexaAPI, "get_devices_gql", None)):
        # get_network_details is get_devices_gql + unwrapping `legacyAppliance`;
        # try the raw query if only the wrapper failed.
        try:
            raw = await AlexaAPI.get_devices_gql(login) or []
            out["smarthome"] = [
                el["legacyAppliance"] for el in raw
                if isinstance(el, dict)
                and isinstance(el.get("legacyAppliance"), dict)]
        except Exception as e:
            print(f"  [sh-discover] get_devices_gql failed: {_redact(e)}")
    # alexapy has no smart-home GROUP listing; kept for builds that do.
    try:
        if hasattr(AlexaAPI, "get_smarthome_groups"):
            out["groups"] = await AlexaAPI.get_smarthome_groups(login) or []
        elif hasattr(AlexaAPI, "get_groups"):
            out["groups"] = await AlexaAPI.get_groups(login) or []
    except Exception as e:
        print(f"  [sh-discover] get_smarthome_groups failed: {_redact(e)}")
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
    # Amazon's own appliance classification (GraphQL `legacyAppliance`) beats
    # the brand guess below: a SmartLife FAN has the same on/off capability as
    # a plug but must not be filed as one.
    atypes = [str(t).upper() for t in (entity.get("applianceTypes") or [])
              if isinstance(t, str)]
    for amazon_type, our_type in (("LIGHT", "light"), ("FAN", "fan"),
                                  ("SMARTPLUG", "plug")):
        if amazon_type in atypes:
            return our_type
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
    # GraphQL `legacyAppliance` rows (alexapy get_network_details) carry
    # applianceTypes instead, e.g. ["LIGHT"] or ["ALEXA_VOICE_ENABLED"].
    types = entity.get("applianceTypes")
    if isinstance(types, list):
        for t in types:
            t = str(t).lower()
            if t and t not in ("alexa_voice_enabled", "other"):
                return t
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


# ── live-LAN catalog sources ────────────────────────────────────────
def _lan_record(source: str, dev: dict, arp_table: list[dict]) -> dict:
    """Convert one brand-skill `list_devices()` dict into a canonical catalog
    record. Every key starting with "_" is DROPPED — sh_tuya's `_tuya` carries
    the secret local key, and credentials must never be written into the
    catalog. `device_id` is stable across scans: keyed on the ARP-resolved MAC
    when the record's lan_ip matched a row, else on the source + name slug."""
    rec: dict[str, Any] = {}
    for k, v in dev.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        rec[k] = v
    name = str(rec.get("name") or "").strip() or "(unnamed)"
    rec["name"] = name
    rec["brand"] = _normalise_brand(rec.get("brand"))
    rec["model"] = rec.get("model") or ""
    rec["type"] = rec.get("type") or "unknown"
    rec["capabilities"] = rec.get("capabilities") or []
    rec["lan_ip"] = str(rec.get("lan_ip") or "").strip()
    rec["alexa_entity_id"] = ""
    rec["alexa_room"] = ""
    rec["alexa_groups"] = []
    rec["controller_skill"] = source
    rec["source"] = f"lan:{source}"
    mac = ""
    for row in arp_table:
        if rec["lan_ip"] and row.get("ip") == rec["lan_ip"]:
            mac = row.get("mac") or ""
            break
    rec["lan_mac"] = mac
    if mac:
        rec["device_id"] = "mac:" + mac
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"
        rec["device_id"] = f"lan:{source}:{slug}"
    return rec


def _lan_discover_all(arp_table: list[dict]) -> tuple[list[dict], dict]:
    """Scan every `_LAN_SOURCES` brand skill in parallel daemon threads,
    bounded by `_LAN_SOURCE_TIMEOUT_SEC` per source under one overall
    `_LAN_TOTAL_TIMEOUT_SEC` deadline. A straggler is abandoned (daemon —
    same pattern as `_run_async`); a raising source degrades to its own
    error entry without touching the others.

    Returns (records, status) where status maps each source name to
    {"ok": bool, "count": int, "error": str?}."""
    # Lazy import so the two modules keep no import-time cycle (the router
    # already lazy-imports this module inside its own functions). The loader
    # lookup rule lives in ONE place — core.smart_home_router._skill_module.
    try:
        from core.smart_home_router import _skill_module
    except Exception as e:
        err = f"skill loader unavailable: {e}"
        return [], {s: {"ok": False, "count": 0, "error": err}
                    for s in _LAN_SOURCES}

    status: dict[str, dict] = {}
    records: list[dict] = []
    results: dict[str, Any] = {}
    threads: dict[str, threading.Thread] = {}
    for source in _LAN_SOURCES:
        try:
            mod = _skill_module(source)
        except Exception:
            mod = None
        fn = getattr(mod, "list_devices", None) if mod is not None else None
        if not callable(fn):
            status[source] = {"ok": False, "count": 0,
                              "error": "skill unavailable"}
            continue

        def _worker(source: str = source, fn: Any = fn) -> None:
            try:
                results[source] = list(fn() or [])
            except Exception as e:
                results[source] = e

        t = threading.Thread(target=_worker, daemon=True,
                             name=f"sh-discover-lan-{source}")
        t.start()
        threads[source] = t

    deadline = time.monotonic() + _LAN_TOTAL_TIMEOUT_SEC
    for source, t in threads.items():
        remaining = max(0.0, deadline - time.monotonic())
        t.join(min(_LAN_SOURCE_TIMEOUT_SEC, remaining))
        got = results.get(source)
        if t.is_alive():
            status[source] = {"ok": False, "count": 0, "error": "timed out"}
            print(f"  [sh-discover] {source} LAN scan timed out; abandoned.")
            continue
        if isinstance(got, Exception):
            status[source] = {"ok": False, "count": 0, "error": str(got)}
            print(f"  [sh-discover] {source} LAN scan failed: {got}")
            continue
        recs = [_lan_record(source, d, arp_table)
                for d in (got or []) if isinstance(d, dict)]
        records.extend(recs)
        status[source] = {"ok": True, "count": len(recs)}
        print(f"  [sh-discover] {source}: {len(recs)} device(s) on the LAN.")
    return records, status


def _alexa_status_from_devices(devices: dict | None) -> dict:
    """Honest per-source status for a completed (non-raising) Alexa fetch."""
    n = len((devices or {}).get("smarthome") or [])
    if n:
        return {"ok": True, "count": n}
    return {"ok": False, "count": 0, "console": True,
            "error": ("the device fetch returned no devices — the cached "
                      "Amazon sign-in may be expired")}


def _dedupe_key(rec: dict) -> tuple[str, str]:
    """Cross-source identity: same controller skill (or brand) + same name.
    The normal case is Alexa mirroring the LAN alias, so a LAN record lands on
    its Alexa twin and enriches it instead of duplicating."""
    return ((rec.get("controller_skill")
             or (rec.get("brand") or "").lower() or ""),
            (rec.get("name") or "").strip().lower())


def _build_catalog_v2(alexa_devices: dict | None, alexa_status: dict,
                      lan_records: list[dict], lan_status: dict,
                      arp_table: list[dict], force: bool = False) -> dict:
    """Assemble the merged Alexa + LAN catalog.

    Alexa records go through the UNCHANGED `_entity_to_record`/`_build_catalog`
    machinery; LAN records that share a `_dedupe_key` with an Alexa record
    enrich it (fill empty lan_ip/lan_mac/device_id) instead of duplicating.
    Non-force runs UNION-preserve prior on-disk `lan:*` devices whose identity
    wasn't re-found — one missed UDP broadcast must not evict a plug (sh_kasa
    re-resolves by name at control time); `force` rebuilds clean.

    Output keeps the v1 shape plus an additive `sources` map (the router reads
    catalog fields only via .get, so version 2 is backward-compatible)."""
    base = _build_catalog(
        alexa_devices if isinstance(alexa_devices, dict) else {}, arp_table)
    records: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    for r in base["devices"]:
        r.setdefault("source", "alexa")
        records.append(r)
        by_key[_dedupe_key(r)] = r
    for lr in lan_records:
        key = _dedupe_key(lr)
        prev = by_key.get(key)
        if prev is not None:
            for f in ("lan_ip", "lan_mac", "device_id"):
                if not prev.get(f) and lr.get(f):
                    prev[f] = lr[f]
            continue
        by_key[key] = lr
        records.append(lr)
    if not force:
        prior = _load_catalog()
        prior_devices = (prior or {}).get("devices") \
            if isinstance(prior, dict) else None
        if isinstance(prior_devices, list):
            present_ids = {r.get("device_id") for r in records
                           if r.get("device_id")}
            for d in prior_devices:
                if not isinstance(d, dict):
                    continue
                if not str(d.get("source") or "").startswith("lan:"):
                    continue
                if d.get("device_id") in present_ids \
                        or _dedupe_key(d) in by_key:
                    continue
                records.append(d)
                by_key[_dedupe_key(d)] = d
                if d.get("device_id"):
                    present_ids.add(d.get("device_id"))
    records.sort(key=lambda r: ((r.get("alexa_room") or "").lower(),
                                (r.get("name") or "").lower()))
    sources = {"alexa": dict(alexa_status or {})}
    for s in _LAN_SOURCES:
        sources[s] = dict((lan_status or {}).get(s)
                          or {"ok": False, "count": 0, "error": "not scanned"})
    return {
        "version": 2,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "device_count": len(records),
        "echo_count": base["echo_count"],
        "group_count": base["group_count"],
        "arp_seen": len(arp_table),
        "sources": sources,
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


# Spoken only as the Alexa-unavailable suffix (status["console"] True) and by
# the explicit reauth path — the LAN phase runs regardless, so this hint no
# longer gates the whole discovery.
_CLI_HINT = (
    "Sir, the smart-home discovery wizard needs an interactive terminal "
    "for Amazon sign-in. Please run "
    "`python -m skills.smart_home_discover` in the JARVIS console."
)


def _alexa_voice_phase(force_refresh: bool) -> tuple[dict | None, dict]:
    """Bounded, non-interactive Alexa fetch for the VOICE path. Returns
    (devices_or_None, status). Never prompts, never raises — every failure
    degrades to an honest status entry so the LAN phase still runs."""
    if _alexapy() is None:
        return None, {"ok": False, "count": 0, "console": False,
                      "error": ("alexapy is not installed — Alexa discovery "
                                "is offline (pip install alexapy)")}
    meta = _load_cookie_meta()
    if (_load_oauth() is None
            and (not meta or not os.path.exists(_COOKIE_PICKLE_PATH))):
        return None, {"ok": False, "count": 0, "console": True,
                      "error": "no cached Amazon sign-in"}
    try:
        # OAuth refresh token first, legacy cookie pickle second.
        devices = _run_async(_restore_any_and_fetch_async(),
                             timeout=_DISCOVERY_TIMEOUT_SEC)
    except TimeoutError:
        # 2026-07-08: a stalled Amazon endpoint must not wedge the voice
        # dispatch thread — the fetch is bounded and reported, not fatal.
        return None, {"ok": False, "count": 0, "console": False,
                      "error": "timed out talking to Amazon"}
    except Exception as e:
        print(f"  [sh-discover] fetch traceback:\n"
              f"{_redact(traceback.format_exc())}")
        return None, {"ok": False, "count": 0, "console": False,
                      "error": f"the device fetch failed ({_redact(e)})"}
    if devices is None:
        return None, {"ok": False, "count": 0, "console": True,
                      "error": "the cached sign-in was unusable"}
    if force_refresh:
        # An explicit reauth request means the user distrusts the cookie —
        # report what the fetch produced but keep the console pointer.
        st = _alexa_status_from_devices(devices)
        if not st.get("ok"):
            st["console"] = True
        return devices, st
    return devices, _alexa_status_from_devices(devices)


def smart_home_discover(arg: str = "") -> str:
    """Voice action — non-blocking. Runs the live LAN discovery (Kasa /
    Tuya / Govee / Hue) plus, when a cached Amazon cookie is available,
    the bounded Alexa fetch — no interactive terminal required. The first
    Amazon sign-in still needs the console wizard (input()/getpass must
    never run on the voice loop); a missing or expired cookie is reported
    per-source instead of aborting the whole scan."""
    force_refresh = any(w in (arg or "").lower()
                        for w in ("force", "reauth", "fresh"))
    if not _wizard_lock.acquire(blocking=False):
        return "Smart-home discovery wizard is already running, sir."
    try:
        _say("One moment, sir — refreshing the smart home catalog.")
        alexa_devices, alexa_status = _alexa_voice_phase(force_refresh)
        arp_table = _scan_lan_arp()
        lan_records, lan_status = _lan_discover_all(arp_table)
        catalog = _build_catalog_v2(alexa_devices, alexa_status,
                                    lan_records, lan_status,
                                    arp_table, force=force_refresh)
        if not force_refresh:
            catalog = _merge_with_existing_catalog(catalog)
        try:
            _save_catalog(catalog)
        except CatalogWipeRefused as e:
            print(f"  [sh-discover] {e}")
            msg = ("Discovery came back empty from both Alexa and the LAN, "
                   "sir — I've kept your existing catalog rather than wiping "
                   "it. Try again once the network settles.")
            _say(msg)
            return msg
        queued = _queue_missing_skill_tasks(catalog)
        return _summarise_catalog(catalog, arp_table, queued, speak=True)
    finally:
        _wizard_lock.release()


def _summarise_catalog(catalog: dict, arp_table: list[dict],
                       queued: int, speak: bool) -> str:
    n = catalog["device_count"]
    unknown = sum(1 for d in catalog["devices"]
                  if not d.get("controller_skill"))
    cross = sum(1 for d in catalog["devices"] if d.get("lan_ip"))
    sources = catalog.get("sources") or {}
    bits: list[str] = []
    if sources:
        # v2 catalog — honest per-source reporting (Alexa + live LAN).
        bits.append(f"Catalog updated, sir: {n} smart-home device(s).")
        found: list[str] = []
        failed: list[str] = []
        for src in _LAN_SOURCES:
            st = sources.get(src) or {}
            label = _LAN_SOURCE_LABELS.get(src, src)
            if st.get("ok"):
                if int(st.get("count") or 0) > 0:
                    found.append(f"{int(st['count'])} {label}")
            elif st.get("error") not in (None, "", "skill unavailable"):
                failed.append(f"{label} ({st['error']})")
        if found:
            bits.append("Found " + ", ".join(found) + " devices on the LAN.")
        if failed:
            bits.append("The LAN scan failed for " + ", ".join(failed) + ".")
        alexa = sources.get("alexa") or {}
        if alexa.get("ok"):
            bits.append(f"Alexa contributed "
                        f"{int(alexa.get('count') or 0)} device(s).")
        else:
            bits.append("Alexa discovery unavailable — "
                        f"{alexa.get('error') or 'not attempted'}.")
            if alexa.get("console"):
                bits.append(_CLI_HINT)
    else:
        # v1 catalog (Alexa-only build) — legacy phrasing preserved.
        bits.append(
            f"Catalog complete, sir: {n} smart-home device(s) across "
            f"{catalog['echo_count']} Echo speaker(s) and "
            f"{catalog['group_count']} group(s)."
        )
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
        print(f"  [sh-discover] cookie login() failed: {_redact(e)}")
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


async def _cookie_login_async() -> Any:
    """Legacy cookie-pickle sign-in as an open AlexaLogin, built on the
    CURRENT running loop (see _oauth_login_async). None when there is no
    usable cached cookie."""
    if _alexapy() is None:
        return None
    cookies = _load_cookie_pickle_safe()
    if cookies is None:
        return None
    meta = _load_cookie_meta() or {}
    login = _construct_login(meta.get("email") or "", "")
    if login is None:
        return None
    authed = False
    try:
        await login.login(cookies=cookies)
        authed = bool(dict(getattr(login, "status", {}) or {})
                      .get("login_successful"))
    except TypeError:
        pass                                # older alexapy: no cookies kwarg
    except Exception as e:
        print(f"  [sh-discover] cookie login() failed: {_redact(e)}")
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
    return login


def _devices_usable(devices: dict | None) -> bool:
    """A restored sign-in only counts if the fetch actually returned
    something. alexapy swallows a dead session into empty lists, and an empty
    dict must not be mistaken for "signed in, zero devices" - that would skip
    the browser sign-in the owner needs."""
    if not isinstance(devices, dict):
        return False
    return bool(devices.get("smarthome") or devices.get("echo"))


def _wizard_alexa_phase(force_refresh: bool) -> tuple[dict | None, dict]:
    """Interactive-console Alexa phase of the wizard. Order (2026-10-12):

      1. saved OAuth refresh token (data/alexa_oauth.json) - no browser,
      2. legacy cookie pickle (data/alexa_cookie.pickle) - kept as a fallback,
      3. BROWSER sign-in through Amazon's phone-app OAuth page (the only way
         to get a first / new sign-in; replaces the retired alexa.amazon.com
         web sign-in and the email+password prompt, which Amazon's current
         login no longer lets alexapy drive).

    `force_refresh` (the `reauth` CLI word) skips 1 and 2. Returns
    (devices_or_None, status) and NEVER aborts the wizard: a cancelled or
    failed Amazon sign-in is recorded honestly so the LAN phase still runs and
    the CLI wizard can land a LAN-only catalog."""
    devices = None
    meta = _load_cookie_meta()
    if not force_refresh and (_load_oauth() is not None or meta):
        if meta and _load_oauth() is None and _cookie_is_stale(meta):
            age = int((time.time() - meta.get("saved_at", 0)) / 86400)
            print(f"  [sh-discover] cached cookie is {age} days old "
                  "— a refresh may be needed soon.")
        # Restore AND fetch in ONE event loop (Py3.14: the alexapy / aiohttp
        # session must be created and used on the same running loop).
        try:
            devices = _run_async(_restore_any_and_fetch_async(),
                                 timeout=_DISCOVERY_TIMEOUT_SEC * 2)
        except Exception as e:
            print(f"  [sh-discover] cached sign-in failed: {_redact(e)}")
            devices = None
        if devices is not None and not _devices_usable(devices):
            print("  [sh-discover] The cached Amazon sign-in came back with "
                  "no devices; a fresh sign-in is needed.")
            devices = None
        if devices is not None:
            print("  [sh-discover] Reused cached Amazon login — no fresh "
                  "sign-in needed.")

    if devices is None:
        try:
            devices = _run_async(
                _oauth_signin_and_fetch_async(),
                timeout=_OAUTH_SIGNIN_TIMEOUT_S + 120.0)
        except Exception as e:
            print(f"  [sh-discover] sign-in traceback:\n"
                  f"{_redact(traceback.format_exc())}")
            return None, {"ok": False, "count": 0, "console": False,
                          "error": f"failed during sign-in ({_redact(e)})"}
        if devices is None:
            return None, {"ok": False, "count": 0, "console": False,
                          "error": ("Amazon sign-in did not complete "
                                    "(cancelled, timed out, or no browser "
                                    "available) - re-run `" + _REAUTH_HINT
                                    + "` to try again")}
    return devices, _alexa_status_from_devices(devices)


def _run_wizard_interactive(arg: str = "") -> str:
    """Full wizard with terminal input(). ONLY call from the __main__
    guard — never from voice context (would freeze the audio loop)."""
    if not _wizard_lock.acquire(blocking=False):
        return "Smart-home discovery wizard is already running, sir."
    try:
        force_refresh = any(w in (arg or "").lower()
                            for w in ("force", "reauth", "fresh"))
        print("  [sh-discover] Looking up your smart home devices.")

        if _alexapy() is None:
            devices, alexa_status = None, {
                "ok": False, "count": 0, "console": False,
                "error": ("alexapy is not installed — Alexa discovery is "
                          "offline (pip install alexapy)")}
        else:
            devices, alexa_status = _wizard_alexa_phase(force_refresh)

        arp_table = _scan_lan_arp()
        lan_records, lan_status = _lan_discover_all(arp_table)
        catalog = _build_catalog_v2(devices, alexa_status,
                                    lan_records, lan_status,
                                    arp_table, force=force_refresh)
        if not force_refresh:
            catalog = _merge_with_existing_catalog(catalog)
        try:
            _save_catalog(catalog)
        except CatalogWipeRefused as e:
            print(f"  [sh-discover] {e}")
            return ("Discovery came back empty from both Alexa and the LAN — "
                    "I kept the existing catalog rather than overwriting it "
                    "with nothing. Check the Alexa sign-in / network and "
                    "re-run.")
        queued = _queue_missing_skill_tasks(catalog)
        return _summarise_catalog(catalog, arp_table, queued, speak=False)
    finally:
        _wizard_lock.release()


def smart_home_catalog(_: str = "") -> str:
    """Read out the device catalog — list_smart_home_devices / smart_home_catalog.

    2026-10-01: this read the stale Alexa view (the header's device_count,
    rooms from Alexa only, so every LAN plug was "(unassigned)") and said
    "0 smart-home devices, sir: ." for an empty catalog. The read-out is now
    core.smart_home_router.catalog_summary — the same one smart_home_devices
    speaks — over this skill's own fresh read of the catalog file."""
    cat = _load_catalog()
    try:
        from core.smart_home_router import catalog_summary
    except Exception:   # pragma: no cover - core is in-tree
        devices = (cat or {}).get("devices") if isinstance(cat, dict) else None
        if not devices:
            return ("No smart-home devices are set up, sir — the catalog is "
                    "empty. Say 'discover smart home devices' to scan for them.")
        return f"{len(devices)} smart-home devices are in the catalog, sir."
    return catalog_summary(cat)


def smart_home_purge_cookie(_: str = "") -> str:
    """Remove the cached Alexa sign-in (OAuth refresh token + legacy cookie)
    so the next wizard run does a fresh interactive browser sign-in."""
    removed = 0
    for p in (_COOKIE_JSON_PATH, _COOKIE_PICKLE_PATH, _oauth_path()):
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
    cli_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    result = _run_wizard_interactive(cli_arg)
    print()
    print(result)
