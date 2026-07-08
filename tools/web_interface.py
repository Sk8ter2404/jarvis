"""Live web interface for JARVIS — a local-LAN dashboard + text command channel.

WHAT THIS IS
============
A tiny, dependency-light HTTP server that lets the owner (1) SEE what JARVIS is
doing from a browser — a live tail of the session log, plus a status strip
(online dot / version / awake state / uptime / model routing / VRAM / air-mouse
armed) — and (2) TALK TO HIM BY TEXT: a typed command is fed through the *exact
same* file-based inject channel a spoken command uses, so a typed "what time is
it" behaves identically to the spoken one. The page also carries QUICK-ACTION
BUTTONS (preset commands POSTed to /api/say) and an AUTO-REFRESH toggle that
freezes the 1s status/log polling so the view can be read without it scrolling.

STATUS FIELDS (build_status)
============================
version / state / running / routing / vram (gpu_lines+gpu_bar) / now_playing /
last_spoken / last_transcript / model, plus ``uptime`` (seconds since the newest
log's first timestamp, or None when not derivable) and ``air_mouse`` (a
``{"armed": bool, "engaged": bool}`` dict — present ONLY when the air-mouse skill
is loaded in THIS process, omitted entirely otherwise so its presence is truthful).

WHY STDLIB http.server (not Flask)
==================================
Flask already lives in this environment (the AirTag tracker runs on it at :8443),
but hard-depending on it here would (a) risk clashing with that server's import
state and (b) break bare-CI / cloud-only installs that don't ship Flask. So this
module is PURE STDLIB (http.server + socketserver + threading + json). We *probe*
for Flask lazily only to note its availability in logs — we never import-fail on
its absence and never actually route through it. One code path, everywhere.

HOW THE INJECT CHANNEL WORKS (reused verbatim from the voice loop)
==================================================================
JARVIS's main loop calls ``_drain_injected_command()`` at the top of every
iteration (bobert_companion.py). That function atomically renames
``injected_commands.json`` to ``.consuming``, pops the FIRST list item, and
requeues the tail. Each item is either a bare string or ``{"text": "...", ...}``.
We APPEND to that same file with the same atomic write-temp-then-os.replace
pattern the run-jarvis driver uses, so a typed command enters the loop exactly
as a mic turn would. The REPLY is read back by tailing the session log
(``logs/session_*.log``) from the byte offset captured at inject time, watching
for the ``JARVIS:`` / ``[action]`` lines the loop prints for that turn — mirroring
driver.py's ``wait_for_reply``. If no reply lands inside the timeout we return
``accepted: true`` (the command still ran; we just didn't capture spoken text).

SECURITY MODEL
==============
The endpoint can INJECT COMMANDS JARVIS EXECUTES, so binding it off-box is a real
exposure. The server therefore:
  • binds 127.0.0.1 by default (loopback — unreachable off the machine),
  • REFUSES TO START on a non-local bind (0.0.0.0 / a LAN IP) when the token is
    empty — ``create_server`` raises ``InsecureBindError`` with a clear reason,
  • when a token IS set, requires it on EVERY request via an Authorization: Bearer
    header, an X-Auth-Token header, or a ?token=… query param; a mismatch is 401.
The GET / dashboard page is served WITHOUT a token (it's just static HTML/JS that
then supplies the token on its API calls) ONLY on a local bind; on a non-local
bind even the page requires the token, so a bare browser hit can't fingerprint us.

TESTABILITY / HEADLESS-CI CONTRACT
==================================
Everything here is stdlib and OS-neutral (no win32, no real JARVIS needed):
  • ``create_server`` takes explicit ``inject_path`` / ``log_dir`` / ``hud_state_path``
    so a test can point them at a temp dir and bind 127.0.0.1:0 (ephemeral port).
  • The status/log/gpu sources all DEGRADE GRACEFULLY when the file/JARVIS is
    absent (missing log → empty tail; missing hud_state → unknown state; gpu
    import failure → omitted). Nothing here raises into a request handler.
  • The reply-wait is injected as ``reply_reader`` so a test can stub it (no live
    log to tail) — the default reader tails the newest session log.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── Optional Flask probe (informational only — we NEVER route through it) ───
# Documented in the module header: Flask is present in this env for the AirTag
# tracker, but hard-depending on it would break bare CI. We note availability so
# a boot log can say "(flask present, using stdlib anyway)" but the server is
# always the stdlib ThreadingHTTPServer below.
try:  # pragma: no cover - trivial import probe; result only affects a log line
    import flask as _flask  # noqa: F401
    FLASK_AVAILABLE = True
except Exception:  # pragma: no cover
    FLASK_AVAILABLE = False


# ── project paths (defaults; overridable per-instance for tests / blue-green) ─
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_THIS_DIR)          # tools/ -> project root
DEFAULT_INJECT_PATH = os.path.join(PROJECT_DIR, "injected_commands.json")
DEFAULT_LOG_DIR = os.path.join(PROJECT_DIR, "logs")
DEFAULT_HUD_STATE_PATH = os.path.join(PROJECT_DIR, "hud_state.json")
# The full-control-panel data sources: the live camera preview frame (written a
# few times a second by the main loop while the camera is on) and the machine-
# generated action inventory. Overridable per-instance so a test points them at a
# temp dir (mirroring inject_path/log_dir/hud_state_path).
DEFAULT_CAMERA_PREVIEW_PATH = os.path.join(PROJECT_DIR, "data", ".hud_camera_preview.jpg")
DEFAULT_ACTION_INDEX_PATH = os.path.join(PROJECT_DIR, "docs", "ACTION_INDEX.md")
# A preview frame older than this is treated as "camera off" (stale) and served
# as a 404 so the panel shows its placeholder rather than a frozen last frame.
_CAMERA_PREVIEW_STALE_S = 5.0

# Reply-wait bounds. The main loop can take a while on a cloud LLM turn, so we
# allow a generous ceiling but poll cheaply. A caller (the POST handler) passes a
# per-request timeout; these just bound it so a hostile ?timeout can't hang a
# worker thread forever.
_REPLY_TIMEOUT_DEFAULT = 30.0
_REPLY_TIMEOUT_MAX = 120.0
_LOG_TAIL_MAX_LINES = 2000        # hard cap on ?lines= so a request can't slurp a huge log


class InsecureBindError(RuntimeError):
    """Raised by ``create_server`` when asked to bind a non-local address with an
    empty token. Refusing to start (rather than binding wide-open) is the whole
    point of the security contract, so this is a hard error the caller logs."""


# ── local-bind detection ────────────────────────────────────────────────────
_LOCAL_BINDS = {"127.0.0.1", "localhost", "::1", "127.0.0.1/32"}


def is_local_bind(bind: str) -> bool:
    """True when ``bind`` is a loopback address that nothing off-box can reach.
    Everything else (0.0.0.0, a LAN IP, a hostname) is treated as EXPOSED and so
    requires a token. Kept deliberately strict — an unknown value is 'exposed'."""
    return (bind or "").strip().lower() in _LOCAL_BINDS


# ── data sources (all graceful — a missing JARVIS/file degrades, never raises) ─

def _newest_log(log_dir: str) -> str | None:
    """Path to the most-recently-modified session_*.log, or None."""
    try:
        files = glob.glob(os.path.join(log_dir, "session_*.log"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None


def tail_log(log_dir: str, lines: int) -> dict:
    """Return the last ``lines`` lines of the newest session log as a dict::

        {"log": "<basename or ''>", "lines": [...], "running": bool}

    ``running`` is a best-effort liveness flag: the newest log was written to
    within the last 20 s (the loop logs whisper/vad activity constantly). Never
    raises — a missing logs dir yields an empty tail with running=False."""
    lines = max(1, min(int(lines or 50), _LOG_TAIL_MAX_LINES))
    lg = _newest_log(log_dir)
    if not lg:
        return {"log": "", "lines": [], "running": False}
    try:
        with open(lg, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-lines:]
    except Exception:
        return {"log": os.path.basename(lg), "lines": [], "running": False}
    try:
        running = (time.time() - os.path.getmtime(lg)) < 20.0
    except Exception:
        running = False
    return {
        "log": os.path.basename(lg),
        "lines": [ln.rstrip("\n") for ln in tail],
        "running": running,
    }


def _read_hud_state(hud_state_path: str) -> dict:
    """Best-effort read of hud_state.json (empty dict on any error). Mirrors
    tray._read_hud_state — the same file the tray + HUD consume."""
    try:
        with open(hud_state_path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _read_version() -> str:
    """The shareable release string (VERSION file via core.version). Falls back
    to reading the VERSION file directly, then to 'unknown' — so a partial tree
    (or a test that imported us bare) still answers."""
    try:
        from core.version import version_string
        return version_string()
    except Exception:
        try:
            with open(os.path.join(PROJECT_DIR, "VERSION"), encoding="utf-8") as f:
                return f.read().strip() or "unknown"
        except Exception:
            return "unknown"


def _awake_state(hud: dict) -> str:
    """Map hud_state.json's sleep/standby flags into a single word for the strip.
    ``state`` is the canonical label the main loop writes ('Idle'/'Standby'/…);
    fall back to the boolean flags if it's absent."""
    state = hud.get("state")
    if isinstance(state, str) and state.strip():
        return state.strip()
    if hud.get("sleep_mode") or hud.get("standby_mode"):
        return "Asleep"
    return "Unknown"


def _gpu_summary() -> dict:
    """One-shot VRAM/model summary via core.gpu_usage, or a graceful stub.
    Returns ``{"lines": [...], "bar": "..."}``; on any failure (module missing on
    a cloud-only box, no GPU) returns empty lines so the strip shows 'GPU: n/a'."""
    try:
        from core import gpu_usage
        snap = gpu_usage.gpu_snapshot()
        return {"lines": gpu_usage.usage_lines(snap), "bar": gpu_usage.usage_bar(14, snap)}
    except Exception:
        return {"lines": [], "bar": ""}


def build_status(hud_state_path: str, log_dir: str) -> dict:
    """Assemble the /api/status payload from every (graceful) source."""
    hud = _read_hud_state(hud_state_path)
    lg = _newest_log(log_dir)
    running = False
    if lg:
        try:
            running = (time.time() - os.path.getmtime(lg)) < 20.0
        except Exception:
            running = False
    gpu = _gpu_summary()
    status = {
        "version": _read_version(),
        "state": _awake_state(hud),
        "running": running,
        "model": hud.get("last_intent_tag", ""),   # best-effort; routing is the real model view
        "routing": (_gpu_summary_routing(gpu)),
        "now_playing": hud.get("now_playing", ""),
        "last_spoken": hud.get("last_spoken", ""),
        "last_transcript": hud.get("last_transcript", ""),
        "gpu_lines": gpu["lines"],
        "gpu_bar": gpu["bar"],
        # Uptime is None (→ omitted client-side) when no timestamped log exists; a
        # float of seconds otherwise. Kept as raw seconds so the client formats it.
        "uptime": _uptime_seconds(log_dir),
        "ts": time.time(),
    }
    # air-mouse ARMED/ENGAGED is only present when the skill is loaded in THIS
    # process (see _air_mouse_status). Add the field ONLY when reachable so a bare
    # web process / headless CI simply omits it — the strip renders nothing for it
    # rather than a misleading "disarmed". This keeps the field's PRESENCE meaningful.
    am = _air_mouse_status()
    if am is not None:
        status["air_mouse"] = am
    return status


def _gpu_summary_routing(gpu: dict) -> str:
    """The routing line is the last entry of usage_lines() (chat→…  vision→…).
    Pulled out so the status payload carries a compact 'which brain' string."""
    lines = gpu.get("lines") or []
    for ln in reversed(lines):
        if "→" in ln:
            return ln
    return ""


def _air_mouse_status() -> dict | None:
    """The live air-mouse ARMED/ENGAGED flags, or None when not cheaply reachable.

    WHY sys.modules (no new import)
    ===============================
    The web interface runs IN-PROCESS with the JARVIS main loop (skills/web_interface
    imports tools.web_interface and calls create_server() inside the running process),
    so the air-mouse skill — when loaded — already lives in ``sys.modules`` under the
    key ``skill_kinect_air_mouse``. We therefore read its thread-safe getter the
    EXACT way bobert_companion._air_mouse_state_for_preview() does: fetch the module
    object from sys.modules (never import it — importing the skill standalone is
    heavy and would drag Kinect/pyautogui deps into a bare-CI web process) and call
    its ``get_air_mouse_state()`` if present. That getter returns a COPY of
    ``{'engaged': bool, 'armed': bool, 'hand': str|None, 'grip': str, ...}``.

    Returns a trimmed ``{"armed": bool, "engaged": bool}`` when the skill is loaded
    and readable, or ``None`` when it isn't — headless CI, a cloud-only box with no
    Kinect, or simply the air-mouse skill not being part of this build. ``None`` lets
    build_status OMIT the field entirely (the strip then shows nothing for it) rather
    than lying with a fabricated "disarmed", which would be indistinguishable from a
    genuinely-disarmed air-mouse. NEVER raises — every failure path degrades to None.
    """
    try:
        sk = sys.modules.get("skill_kinect_air_mouse")
        getter = getattr(sk, "get_air_mouse_state", None) if sk else None
        if callable(getter):
            st = getter()
            if isinstance(st, dict):
                return {"armed": bool(st.get("armed")),
                        "engaged": bool(st.get("engaged"))}
    except Exception:
        pass
    return None


# The main loop timestamps each log line "[HH:MM:SS] …" (see _capture_utterance /
# the loop's print wrapper). The FIRST line of a session log is written at boot, so
# its clock time is the boot time. We can't recover the date cheaply (the filename
# carries it, but a wall-clock delta is enough for an "up 2h13m" chip), so we treat
# the first timestamp as today-relative and compute a same-day delta, clamping a
# negative delta (crossed midnight) to 0 rather than reporting a bogus ~24h uptime.
_LOG_TS_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")


def _uptime_seconds(log_dir: str) -> float | None:
    """Best-effort session uptime in seconds, derived from the newest log's first
    timestamped line, or None when not derivable.

    We parse the "[HH:MM:SS]" prefix of the first line that HAS one (the loop's
    banner lines may precede the first timestamp) and subtract it from the current
    wall clock's H:M:S. This is intentionally cheap and approximate — it's a status
    nicety, not a billing meter. Returns None (→ field omitted) when there is no log,
    no parseable timestamp, or anything at all goes wrong. NEVER raises."""
    lg = _newest_log(log_dir)
    if not lg:
        return None
    try:
        first_hms = None
        with open(lg, encoding="utf-8", errors="replace") as f:
            # Only scan a bounded head of the file — the boot timestamp is in the
            # first handful of lines; we never want to read a multi-MB log to find it.
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                m = _LOG_TS_RE.search(line)
                if m:
                    first_hms = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    break
        if first_hms is None:
            return None
        now = time.localtime()
        boot_s = first_hms[0] * 3600 + first_hms[1] * 60 + first_hms[2]
        now_s = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
        delta = now_s - boot_s
        # Crossed midnight (or a clock skew) → negative; clamp to 0 rather than
        # reporting a spurious ~day of uptime. A same-day session reads correctly.
        return float(delta) if delta >= 0 else 0.0
    except Exception:
        return None


# ── inject channel (append with the SAME atomic pattern the loop drains) ─────

def inject_command(text: str, inject_path: str) -> None:
    """Append ``{"text": text, "ts": ...}`` to the inject queue atomically.

    Read-modify-write under a fresh temp + os.replace so a concurrent
    ``_drain_injected_command`` (which claims the file by renaming it) never sees
    a half-written array. If the queue was mid-consume (renamed away) we simply
    start a fresh list — the loop will drain ours next pass. Matches driver.py's
    ``inject`` and staging_instance's writer."""
    items: list = []
    try:
        if os.path.exists(inject_path):
            with open(inject_path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    items = decoded
    except Exception:
        items = []
    items.append({"text": text, "ts": time.time()})
    _dir = os.path.dirname(os.path.abspath(inject_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp", prefix=".webinject_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, inject_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def wait_for_reply(text: str, log_dir: str, timeout: float) -> dict:
    """Tail the newest session log from its current end and return JARVIS's reply
    lines for this utterance. Mirrors run-jarvis/driver.py's wait_for_reply, but
    trimmed to what the web UI needs.

    Returns ``{"status": "ok"|"accepted"|"no_log", "lines": [...]}``:
      • ok       — we captured JARVIS:/[action] line(s) for this turn.
      • accepted — injected, but no reply text landed within the timeout (the
                   command still ran; we just didn't see spoken output — e.g. a
                   pure side-effect action, or JARVIS is asleep and dropped it).
      • no_log   — no session log exists (JARVIS isn't running); the command was
                   still queued and will fire when it next boots."""
    lg = _newest_log(log_dir)
    if not lg:
        return {"status": "no_log", "lines": []}
    try:
        pos = os.path.getsize(lg)
    except Exception:
        return {"status": "no_log", "lines": []}
    snippet = text[:30].lower()
    saw_inject = False
    lines: list[str] = []
    deadline = time.time() + max(1.0, min(float(timeout), _REPLY_TIMEOUT_MAX))
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except Exception:
            continue
        for line in chunk.splitlines():
            low = line.lower()
            # The loop prints "  [inject] <text>" (see _capture_utterance) when it
            # picks up our command — anchor on it so we don't scrape an unrelated
            # concurrent turn's output. If we never see the anchor we still return
            # whatever JARVIS:/[action] lines appeared (best-effort).
            if "[inject]" in low and snippet and snippet in low:
                saw_inject = True
                continue
            if "jarvis:" in low or "[action]" in low:
                lines.append(line.rstrip())
        if saw_inject and lines:
            # Give a beat for a trailing spoken follow-up line to land, then stop.
            time.sleep(1.0)
            try:
                with open(lg, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    for line in f.read().splitlines():
                        if "jarvis:" in line.lower() or "[action]" in line.lower():
                            lines.append(line.rstrip())
            except Exception:
                pass
            return {"status": "ok", "lines": lines}
    return {"status": "ok" if lines else "accepted", "lines": lines}


# ── settings bridge (the FULL settings control panel) ───────────────────────
#
# WHY THIS EXISTS
# ===============
# The owner wants to "do it all from the web interface" — every user-facing
# config knob, including the wake-word mode toggle. Rather than re-declare those
# knobs here (they'd drift from the real config), we treat tools/settings_window's
# ``SCHEMA`` as the SINGLE SOURCE OF TRUTH: it already enumerates every
# user-facing knob keyed by name → {tab, label, type, default, help, choices}.
# We import it, render every persisted key, READ the CURRENT effective value from
# ``core.config`` (the constants _apply_user_settings() overrode at import), and
# WRITE changes back to data/user_settings.json (the same file the Settings GUI
# and _apply_user_settings() share).
#
# DEPENDENCY-LIGHT / GRACEFUL, like the rest of this module
# ========================================================
# Both imports are LAZY (inside the functions) and TOLERANT: a bare-CI import of
# web_interface never needs settings_window or core.config, and if either can't
# load we degrade (empty schema / schema-default value) instead of raising into a
# request handler — mirroring how _read_version / _gpu_summary already probe
# core.* lazily. settings_window's schema half is stdlib-only (the tkinter GUI is
# below its "# ── GUI ──" divider and imported lazily there), so importing it for
# SCHEMA/coerce_value costs nothing on a headless box.


def _load_settings_schema():
    """Import tools.settings_window and return ``(SCHEMA, coerce_value)``, or
    ``({}, None)`` if it can't load. Lazy + tolerant: importing web_interface must
    not require settings_window, and a broken/absent schema degrades to an empty
    panel rather than breaking every request. Never raises."""
    try:
        from tools import settings_window as sw
        return sw.SCHEMA, sw.coerce_value
    except Exception:
        return {}, None


def _config_value(key: str, default):
    """The CURRENT effective value of a config constant, read LIVE from
    ``core.config`` (the value _apply_user_settings() left after merging
    user_settings.json over the module defaults at import). Falls back to the
    schema ``default`` when core.config can't be imported (bare CI) or the
    constant is absent. Lazy import so a headless web process never drags the
    monolith's config in unless a settings request actually asks for it.

    NOTE: core.config is imported ONCE and then cached in sys.modules, so its
    constants reflect the values as of THAT import — which is boot time in a live
    JARVIS. A settings write does NOT mutate the running process's constants (see
    the restart caveat in _write_settings), so what we report here is the
    effective value the CURRENTLY-RUNNING loop is using, which is exactly the
    truthful thing to show."""
    try:
        from core import config as _config
        return getattr(_config, key, default)
    except Exception:
        return default


# Knobs whose VALUE is a secret. We render a row for them (so the owner can SET
# one from the panel) but we NEVER echo the current value back in the GET payload
# — the settings snapshot returns "" + secret/is_set flags instead of the live
# token. Reading /api/settings is already token-gated, so this isn't the only
# barrier, but echoing a live secret into a visible text field (shoulder-surfing,
# an accidental screen-share, a proxy log) is a footgun with no upside. The write
# path is unaffected: a POST can still set a new token.
_SECRET_SETTING_KEYS = frozenset({"WEB_INTERFACE_TOKEN"})


def build_settings_schema() -> dict:
    """Assemble the /api/settings GET payload: every PERSISTED schema knob with
    its current effective value. Shape::

        {"settings": [ {name, tab, label, type, choices, help, value, default,
                        secret?, is_set?}, ... ],
         "tabs": ["voice","ai","privacy","integrations","advanced"],
         "note": "…applies on restart…"}

    Read LIVE each call (no caching) so the panel always reflects the file/loop
    state. Status-only rows (keys starting with "_status_", type "status") are
    SKIPPED — they expose integration presence in the GUI but carry no persisted
    value and (deliberately) never surface a secret, so they have no place in a
    write-capable web panel. SECRET knobs (``_SECRET_SETTING_KEYS``) are rendered
    but their value is REDACTED to "" (with ``secret: True`` and ``is_set``) so
    the live token never leaves the process. Never raises: an unloadable schema
    yields an empty list."""
    schema, _coerce = _load_settings_schema()
    items: list[dict] = []
    tabs: list[str] = []
    for name, spec in schema.items():
        typ = spec.get("type")
        # Only real persisted knobs — skip the read-only integration status rows.
        if typ == "status" or name.startswith("_"):
            continue
        tab = spec.get("tab", "advanced")
        if tab not in tabs:
            tabs.append(tab)
        default = spec.get("default")
        row = {
            "name": name,
            "tab": tab,
            "label": spec.get("label", name),
            "type": typ,
            # choices only present for enum/combo/routing — omit when absent so
            # the client can rely on truthiness.
            "choices": spec.get("choices"),
            "help": spec.get("help", ""),
            # The CURRENT effective value (live from core.config), falling back to
            # the schema default when the constant isn't readable.
            "value": _config_value(name, default),
            "default": default,
        }
        if name in _SECRET_SETTING_KEYS:
            # Redact: report only WHETHER a value is set, never the value itself.
            # The client renders a password field and (on an empty save) leaves
            # the existing secret untouched — see saveSetting.
            live = row["value"]
            row["is_set"] = bool(live and str(live).strip())
            row["value"] = ""
            row["default"] = ""
            row["secret"] = True
        items.append(row)
    return {
        "settings": items,
        "tabs": tabs,
        "note": SETTINGS_RESTART_NOTE,
    }


# The honest caveat we return on every write. _apply_user_settings() runs ONCE at
# core.config import (boot), so a saved value overrides the module constant only
# on the NEXT JARVIS start. We do NOT claim a live-apply we can't guarantee —
# some knobs are re-read live by their consumers, but many are import-time, so the
# safe, truthful blanket statement is "applies on restart".
SETTINGS_RESTART_NOTE = ("Saved. Most settings take effect the next time JARVIS "
                         "restarts.")


class SettingsWriteError(ValueError):
    """Raised by _coerce_setting/_write_settings on an unknown key or a value that
    can't be coerced to the schema type. The POST handler turns it into a 400 with
    this message — a clear, actionable error rather than a silent drop."""


def _coerce_setting(name: str, value, schema: dict, coerce_value) -> object:
    """Validate ``name`` against the schema and coerce ``value`` to its declared
    type, raising ``SettingsWriteError`` on an unknown key or a bad value.

    We reuse settings_window.coerce_value for the actual type conversion so the
    web panel and the GUI apply IDENTICAL coercion rules (bool truthiness, enum
    membership, int/float parsing, text→list, routing merge). But coerce_value is
    deliberately LENIENT — it falls back to the default rather than raising — so a
    web caller that fat-fingers an enum would silently write the default and think
    it succeeded. That's wrong for an API, so we add a STRICT pre-check for the two
    cases a user most wants an error on:
      • unknown key                → 400 (typo / stale client)
      • enum value not in choices  → 400 (invalid choice)
    int/float that won't parse also 400 (coerce_value would swallow it to the
    default). Everything else defers to coerce_value's tolerant conversion."""
    spec = schema.get(name)
    if spec is None or spec.get("type") == "status" or name.startswith("_"):
        raise SettingsWriteError(f"unknown setting: {name!r}")
    typ = spec.get("type")
    # Enum: must be one of the declared choices — reject rather than default.
    if typ == "enum":
        choices = spec.get("choices") or []
        if str(value) not in choices:
            raise SettingsWriteError(
                f"invalid value for {name!r}: {value!r} is not one of {choices}")
    # int/float: coerce_value swallows a bad parse to the default, so pre-validate
    # here to surface a real 400 instead of a silent wrong write.
    if typ in ("int", "float"):
        try:
            (int if typ == "int" else float)(value)
        except (TypeError, ValueError):
            raise SettingsWriteError(
                f"invalid {typ} value for {name!r}: {value!r}")
    if coerce_value is None:                       # schema loaded but no coercer
        raise SettingsWriteError("settings coercion unavailable")
    # List/dict-valued settings ('routing', or a 'text'/list knob): the web panel
    # renders them as a single text input holding a JSON string. coerce_value's
    # 'text' branch would splitlines() that JSON into a bogus 1-element list, and
    # its 'routing' branch would reset a non-dict to the default — silently
    # corrupting the save (2026-07-08 finding). Parse the JSON back to the real
    # container FIRST so coerce_value gets the list/dict it accepts. Non-JSON
    # strings (a genuine newline-separated 'text' list) fall through untouched.
    if isinstance(value, str):
        default = spec.get("default")
        wants_list = typ == "text" or isinstance(default, list)
        wants_dict = typ == "routing" or isinstance(default, dict)
        s = value.strip()
        if (wants_list and s.startswith("[")) or (wants_dict and s.startswith("{")):
            try:
                parsed = json.loads(s)
                if (isinstance(parsed, list) and wants_list) or \
                   (isinstance(parsed, dict) and wants_dict):
                    value = parsed
            except (ValueError, TypeError):
                pass
    return coerce_value(spec, value)


def _write_settings(updates: dict, path: str) -> dict:
    """MERGE ``updates`` (name→value) into the user_settings.json at ``path``,
    atomically, preserving every other key already in the file.

    Returns the ``{name: coerced_value, ...}`` actually applied. Raises
    ``SettingsWriteError`` if ANY update is invalid (unknown key / bad type) —
    validation happens for ALL updates BEFORE we touch disk, so a bad key in a
    batch never leaves a half-applied file.

    ATOMICITY / PRESERVATION
    ========================
    We read the existing file, overlay ONLY the validated keys, and write via a
    fresh temp file + os.replace in the SAME directory — the identical crash-safe
    pattern as inject_command() above and settings_window.atomic_write_json /
    _apply_user_settings' reader. A concurrent reader (the Settings GUI, or JARVIS
    booting) therefore never observes a half-written document, and any key we
    didn't touch (including keys no schema knows about, e.g. a newer JARVIS's
    extra knobs) is preserved verbatim. We do NOT rewrite the full default
    template — a targeted merge is the whole contract ("preserve all other keys").

    RESTART CAVEAT: this writes the FILE only. core.config's live constants were
    set at import and are not mutated here, so the change reaches the running loop
    on its next restart (see SETTINGS_RESTART_NOTE)."""
    schema, coerce_value = _load_settings_schema()
    if not schema:
        raise SettingsWriteError("settings schema unavailable")
    # 1) Validate + coerce EVERYTHING first (fail closed before any disk write).
    applied: dict = {}
    for name, value in updates.items():
        applied[name] = _coerce_setting(name, value, schema, coerce_value)
    # 2) Read the current file (tolerant: missing/corrupt → start from {}), so we
    #    MERGE over it and preserve keys we don't manage.
    current: dict = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    current = decoded
    except Exception:
        current = {}          # a corrupt file is overwritten with a valid merge
    current.update(applied)
    # 3) Atomic write (temp in the same dir + os.replace) — never a partial file.
    _dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp", prefix=".websettings_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise
    return applied


def _default_user_settings_path() -> str:
    """The live user_settings.json path — resolved from settings_window so the web
    panel writes the EXACT file the Settings GUI and core.config._apply_user_settings
    read (data/user_settings.json under the project root, honouring the
    JARVIS_SETTINGS_PATH redirect). Falls back to the known relative location if
    settings_window can't be imported, so create_server always has a concrete
    default. Never raises."""
    try:
        from tools import settings_window as sw
        return sw.settings_path()
    except Exception:
        return os.path.join(PROJECT_DIR, "data", "user_settings.json")


# ── control-panel data sources (System / Actions / Voice / Camera / Memory) ──
#
# Every function here is READ-ONLY and GRACEFUL — the exact contract the rest of
# this module honours: a missing tool / module / file degrades to an empty-but-
# valid payload, never an exception into a request handler. The GET endpoints
# that call them (do_GET) each add their own try/except belt on top. All heavy
# imports (psutil, core.voice_clone, core.long_term_memory) are LAZY so importing
# web_interface stays dependency-light and a bare-CI/cloud box just degrades.


def _smi_num(v):
    """Parse an nvidia-smi CSV cell to a number, or None. '[N/A]'/'' → None. A
    value containing a dot becomes a float (power.draw is fractional); otherwise
    an int. Never raises."""
    try:
        s = str(v).strip()
        if not s or s.lower().startswith("[n/a") or s.lower() == "n/a":
            return None
        return float(s) if "." in s else int(s)
    except Exception:
        return None


def _nvidia_smi_gpus() -> list:
    """Per-GPU stats via a single nvidia-smi CSV query, or ``[]`` on ANY failure
    (no GPU, no driver, cloud-only box, command missing/timeout). Never raises.
    Uses CREATE_NO_WINDOW on Windows (read via getattr so the CI Linux-sim, which
    deletes that attribute, doesn't trip an AttributeError)."""
    import subprocess
    no_window = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 if sys.platform == "win32" else 0)
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4.0,
            creationflags=no_window,
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    gpus: list = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx = _smi_num(parts[0])
        gpus.append({
            "index":        idx if idx is not None else 0,
            "name":         parts[1],
            "mem_used_mb":  _smi_num(parts[2]),
            "mem_total_mb": _smi_num(parts[3]),
            "util_pct":     _smi_num(parts[4]),
            "temp_c":       _smi_num(parts[5]),
            "power_w":      _smi_num(parts[6]),
        })
    return gpus


def _disks_info() -> list:
    """Free/total (GB) per mounted disk. Prefers psutil (every real partition);
    falls back to shutil.disk_usage on the project's own drive when psutil is
    absent (e.g. the Linux-CI sim, which BLOCKS psutil). Never raises."""
    disks: list = []
    try:
        import psutil
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except Exception:
                continue           # empty CD drive / permission → skip
            disks.append({
                "drive":    part.mountpoint,
                "free_gb":  round(u.free / 1e9, 1),
                "total_gb": round(u.total / 1e9, 1),
            })
        return disks
    except Exception:
        pass
    # Degraded fallback: just the drive JARVIS lives on.
    try:
        import shutil
        u = shutil.disk_usage(PROJECT_DIR)
        drive = os.path.splitdrive(PROJECT_DIR)[0] or os.path.abspath(os.sep)
        disks.append({"drive": drive or "/",
                      "free_gb": round(u.free / 1e9, 1),
                      "total_gb": round(u.total / 1e9, 1)})
    except Exception:
        pass
    return disks


def _system_info(hud_state_path: str, log_dir: str) -> dict:
    """The /api/system payload: GPUs (nvidia-smi), CPU/RAM (psutil), disks, plus
    version/uptime/routing reused from the status sources. EVERY field is always
    present with a safe default (None / []) so the client can rely on the shape
    even when a source is unavailable. Read-only; never raises."""
    cpu_pct = ram_used = ram_total = None
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)   # rolling avg between polls
        vm = psutil.virtual_memory()
        ram_used = round((vm.total - vm.available) / 1e9, 1)
        ram_total = round(vm.total / 1e9, 1)
    except Exception:
        pass
    return {
        "gpus":         _nvidia_smi_gpus(),
        "cpu_pct":      cpu_pct,
        "ram_used_gb":  ram_used,
        "ram_total_gb": ram_total,
        "disks":        _disks_info(),
        "version":      _read_version(),
        "uptime":       _uptime_seconds(log_dir),
        "routing":      _gpu_summary_routing(_gpu_summary()),
    }


def _norm_speak(cell: str) -> str:
    """Normalise a speak-class table cell ('**VERBATIM**' / '*INFORMATIVE*' /
    'neither') to a bare token: 'VERBATIM' | 'INFORMATIVE' | 'neither'."""
    up = (cell or "").replace("*", "").strip().upper()
    if up == "VERBATIM":
        return "VERBATIM"
    if up == "INFORMATIVE":
        return "INFORMATIVE"
    return "neither"


def _parse_action_index(path: str) -> dict:
    """Parse docs/ACTION_INDEX.md's 'Full index' table into
    ``{"actions": [{"name", "spoken"}], "count": N}``. Aliases sharing a handler
    (a comma-separated first cell) are EXPANDED so every dispatchable name is its
    own sendable row. Unreadable/absent file → ``{"actions": [], "count": 0}``.
    Never raises.

    Row shape:  ``| `name1`, `name2` | `handler:line` | **VERBATIM** | ex? | tests |``
    We keep only rows whose first cell wraps its name(s) in backticks — that skips
    the header ('action(s)'), the ``|---|`` separator, and the Summary table
    (prose first cell, no backticks)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return {"actions": [], "count": 0}
    actions: list = []
    seen: set = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name_cell = cells[0]
        if "`" not in name_cell:
            continue
        spoken = _norm_speak(cells[2])
        for nm in name_cell.split(","):
            nm = nm.strip().strip("`").strip()
            if nm and nm not in seen:
                seen.add(nm)
                actions.append({"name": nm, "spoken": spoken})
    return {"actions": actions, "count": len(actions)}


def _voices_info() -> dict:
    """The /api/voices payload: enrolled voice-clone profiles (name/source and a
    ``usable`` flag straight from the consent gate) plus the active profile, the
    master switch, and the base TTS backend/voice — read live from core. Degrades
    to empty/defaults on any import failure (bare CI). READ-ONLY: it lists profile
    metadata only and never loads the cloning model."""
    profiles: list = []
    active = ""
    enabled = False
    tts_backend = ""
    tts_voice = ""
    try:
        from core import voice_clone
        for meta in voice_clone.list_profiles():
            try:
                usable = bool(voice_clone.profile_is_usable(meta))
            except Exception:
                usable = False
            profiles.append({
                "name":   meta.get("name", ""),
                "source": meta.get("source", ""),
                "usable": usable,
            })
    except Exception:
        profiles = []
    try:
        from core import config as _config
        active = getattr(_config, "VOICE_CLONE_PROFILE", "") or ""
        enabled = bool(getattr(_config, "VOICE_CLONE_ENABLED", False))
        tts_backend = getattr(_config, "TTS_BACKEND", "") or ""
        tts_voice = getattr(_config, "TTS_VOICE", "") or ""
    except Exception:
        pass
    return {
        "profiles":    profiles,
        "active":      active,
        "enabled":     enabled,
        "tts_backend": tts_backend,
        "tts_voice":   tts_voice,
    }


def _read_json_list(path: str) -> list:
    """Read a JSON file expected to hold a list; ``[]`` on any error. Cheap; no
    deps; never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _read_memory() -> dict:
    """The /api/memory payload: long-term semantic FACTS + recent EPISODES with
    counts. Deliberately READ-ONLY and CHEAP — we NEVER call the module's
    ensure_loaded() (which would rebuild the BM25 index / run first-boot
    migration) and never touch the embedder. Instead we read the already-loaded
    in-memory facts when a live JARVIS has them (instant, zero side effect) and
    otherwise read the JSON mirror + episode JSONL straight off disk. Degrades to
    empty on any failure. Shape::

        {"facts": [{text, source, tags, updated_at}],
         "episodes": [{text, role, iso}],  # newest-first, capped
         "counts": {"facts": N, "episodes": M}}
    """
    facts: list = []
    episodes: list = []
    try:
        from core import long_term_memory as ltm
    except Exception:
        return {"facts": [], "episodes": [],
                "counts": {"facts": 0, "episodes": 0}}
    # FACTS — prefer the loaded in-memory dict (a live JARVIS has it), else the
    # on-disk mirror. Both are pure reads; neither triggers a load/rebuild.
    try:
        raw_facts = None
        if getattr(ltm, "_loaded", False):
            lock = getattr(ltm, "_lock", None)
            if lock is not None:
                with lock:
                    raw_facts = list(getattr(ltm, "_facts", {}).values())
            else:
                raw_facts = list(getattr(ltm, "_facts", {}).values())
        if raw_facts is None:
            raw_facts = _read_json_list(getattr(ltm, "_FACTS_JSON", ""))
        for fentry in raw_facts:
            if isinstance(fentry, dict) and str(fentry.get("text", "")).strip():
                facts.append({
                    "text":       str(fentry.get("text", "")),
                    "source":     fentry.get("source", ""),
                    "tags":       fentry.get("tags", []),
                    "updated_at": fentry.get("updated_at"),
                })
    except Exception:
        facts = []
    # EPISODES — tail the JSONL log directly (read-only). Count all lines; keep
    # only the most recent 50 (newest-first) for display so a long history stays a
    # cheap payload.
    ep_count = 0
    try:
        ep_path = getattr(ltm, "_EPISODE_LOG", "")
        raw_lines: list = []
        if ep_path and os.path.exists(ep_path):
            with open(ep_path, encoding="utf-8", errors="replace") as f:
                raw_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        ep_count = len(raw_lines)
        for ln in reversed(raw_lines[-50:]):
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if isinstance(e, dict) and str(e.get("text", "")).strip():
                episodes.append({
                    "text": str(e.get("text", "")),
                    "role": e.get("role", ""),
                    "iso":  e.get("iso", ""),
                })
    except Exception:
        episodes = []
    return {
        "facts":    facts,
        "episodes": episodes,
        "counts":   {"facts": len(facts), "episodes": ep_count},
    }


# ── the request handler ──────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Routes: GET / (dashboard), GET /api/status, GET /api/log/tail, GET
    /api/settings, POST /api/say, POST /api/settings. The owning server pins
    config onto the class instance via the ``config`` attribute set in
    ``create_server`` (a small dict) so handlers are stateless beyond it."""

    # Silence the default per-request stderr logging — it would spam the session
    # log we're tailing. (The base class calls this for every request.)
    def log_message(self, fmt, *args):  # noqa: A003 - matches base signature
        return

    # ── auth ────────────────────────────────────────────────────────────────
    def _token(self) -> str:
        return self.server.config.get("token", "")  # type: ignore[attr-defined]

    def _request_token(self, query: dict) -> str:
        """Pull a caller-supplied token from (in priority) the Authorization
        Bearer header, an X-Auth-Token header, or a ?token= query param."""
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        xat = self.headers.get("X-Auth-Token")
        if xat:
            return xat.strip()
        q = query.get("token")
        if q:
            return q[0]
        return ""

    def _authorized(self, query: dict, *, is_page: bool) -> bool:
        """Auth gate. With no token configured (only reachable on a LOCAL bind —
        create_server enforces that) everything is allowed. With a token set, an
        API request must present it. The dashboard PAGE is allowed token-free on a
        local bind (convenience) but requires the token on an exposed bind."""
        token = self._token()
        if not token:
            return True
        if is_page and self.server.config.get("local_bind", True):  # type: ignore[attr-defined]
            return True
        return self._request_token(query) == token

    # ── anti-CSRF / anti-DNS-rebinding for state-changing POSTs ──────────────
    @staticmethod
    def _host_of(value: str) -> str:
        """Bare lowercase hostname from a Host/Origin/Referer header value —
        scheme, path and port stripped, IPv6 brackets kept (``[::1]``).
        ``"http://localhost:8766/x"`` → ``"localhost"``; ``"127.0.0.1:8766"`` →
        ``"127.0.0.1"``; ``"[::1]:8766"`` → ``"[::1]"``. Empty on junk."""
        if not value:
            return ""
        v = value.strip()
        if "://" in v:
            v = v.split("://", 1)[1]
        v = v.split("/", 1)[0]            # drop any path
        if v.startswith("["):            # IPv6 literal: keep the [..] intact
            return v.split("]", 1)[0].lower() + "]"
        if ":" in v:                     # strip :port on a plain host/IPv4
            v = v.rsplit(":", 1)[0]
        return v.lower()

    def _served_hosts(self) -> set:
        """Hostnames this server legitimately answers to: loopback plus the
        configured bind. A request whose Host/Origin is outside this set is either
        a DNS-rebinding attempt (foreign Host resolved to us) or a cross-site POST
        (foreign Origin)."""
        hosts = {"localhost", "127.0.0.1", "[::1]", "::1"}
        bind = str(self.server.config.get("bind", "127.0.0.1")).strip().lower()  # type: ignore[attr-defined]
        if bind and bind not in ("0.0.0.0", "::"):
            hosts.add(bind)
        return hosts

    def _state_change_allowed(self) -> tuple:
        """Anti-DNS-rebinding + anti-CSRF guard, applied to EVERY request (GET and
        POST). Blocks two browser-driven attacks:

          * DNS rebinding — a page on evil.com rebinds it to 127.0.0.1; caught
            because the Host header is ``evil.com``, not a host we answer to.
          * Cross-site request (CSRF) — a page on evil.com fetch()es our URL;
            caught because the Origin/Referer host is ``evil.com``.

        On a LOCAL (loopback) bind the ONLY host we legitimately answer to is
        loopback, so this is enforced regardless of any token. Crucially it does
        NOT short-circuit when a token is set: on a local bind the dashboard page
        is served token-free and bakes the token into its JS, so a rebinding page
        could read the token and then present it — the token cannot be the
        rebinding boundary here (2026-07-08 finding). It also now covers GET, so a
        rebinding page can't read the token-baked page, the session log, or the
        settings/system/memory snapshots.

        On a NON-local (exposed) bind a token is mandatory + unforgeable (enforced
        by _authorized) and the owner may legitimately reach the server by LAN IP /
        hostname, which the loopback allowlist would reject — so there the token is
        the boundary and we do not Host-restrict.

        Non-browser clients (curl, PowerShell, the driver) send no Origin and a
        loopback Host, so they pass untouched. Returns ``(ok, reason)``."""
        if not self.server.config.get("local_bind", True):  # type: ignore[attr-defined]
            return True, ""                       # exposed bind: token is the boundary
        served = self._served_hosts()
        host = self._host_of(self.headers.get("Host", ""))
        if host and host not in served:           # foreign Host → DNS rebinding
            return False, "host"
        origin = self.headers.get("Origin", "")
        if origin:
            if self._host_of(origin) not in served:
                return False, "origin"
        else:
            # Some browsers omit Origin on same-origin GET/POST; fall back to Referer.
            ref = self.headers.get("Referer", "")
            if ref and self._host_of(ref) not in served:
                return False, "referer"
        return True, ""

    def _forbidden(self, reason: str) -> None:
        self._send_json({"error": f"cross-origin request refused ({reason})"},
                        code=403)

    # ── tiny response helpers ────────────────────────────────────────────────
    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No caching of live status/log/reply payloads.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_html(self, text: str, code: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        """Send a raw binary body (e.g. the camera preview JPEG). Mirrors
        _send_html/_send_json but for arbitrary bytes, with no-store caching so a
        stale frame is never served from a browser cache."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _unauthorized(self) -> None:
        self._send_json({"error": "unauthorized"}, code=401)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        # Anti-rebinding/CSRF guard on GET too — a foreign Host on a local bind is
        # a DNS-rebinding page that would otherwise read the token-baked dashboard,
        # the session log, or the settings/system/memory snapshots. No-op for
        # non-browser clients and on an exposed (token-protected) bind.
        ok, why = self._state_change_allowed()
        if not ok:
            return self._forbidden(why)

        if path == "/":
            if not self._authorized(query, is_page=True):
                return self._unauthorized()
            return self._send_html(_dashboard_html(cfg.get("token", "")))

        if path == "/api/status":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json(build_status(cfg["hud_state_path"], cfg["log_dir"]))

        if path == "/api/log/tail":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                n = int(query.get("lines", ["50"])[0])
            except (TypeError, ValueError):
                n = 50
            return self._send_json(tail_log(cfg["log_dir"], n))

        if path == "/api/settings":
            # The FULL settings snapshot: every schema knob + its CURRENT effective
            # value, read live. Gated the same as every other API route (token when
            # one is set) — reading the config is less sensitive than writing it,
            # but there's no reason to leak it token-free on an exposed bind.
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json(build_settings_schema())

        # ── control-panel endpoints (System / Actions / Voice / Memory + the
        #    camera preview image). Each is auth-gated exactly like /api/status,
        #    read-only, and wraps its data source in try/except so a source
        #    failure becomes a safe JSON error rather than a 500 traceback. ──
        if path == "/api/system":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(
                    _system_info(cfg["hud_state_path"], cfg["log_dir"]))
            except Exception as e:
                return self._send_json({"error": f"system read failed: {e}",
                                        "gpus": [], "disks": []}, code=500)

        if path == "/api/actions":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(
                    _parse_action_index(cfg["action_index_path"]))
            except Exception as e:
                return self._send_json({"error": f"actions read failed: {e}",
                                        "actions": [], "count": 0}, code=500)

        if path == "/api/voices":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(_voices_info())
            except Exception as e:
                return self._send_json({"error": f"voices read failed: {e}",
                                        "profiles": []}, code=500)

        if path == "/api/memory":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(_read_memory())
            except Exception as e:
                return self._send_json({"error": f"memory read failed: {e}",
                                        "facts": [], "episodes": [],
                                        "counts": {"facts": 0, "episodes": 0}},
                                       code=500)

        if path == "/api/camera-preview":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            # Serve the live preview JPEG, or 404 when it's missing OR stale
            # (older than ~5 s = camera off). No side effects; never raises.
            try:
                p = cfg.get("camera_preview_path", "")
                if not p or not os.path.exists(p):
                    return self._send_json({"error": "no preview"}, code=404)
                try:
                    age = time.time() - os.path.getmtime(p)
                except Exception:
                    age = 1e9
                if age > _CAMERA_PREVIEW_STALE_S:
                    return self._send_json({"error": "no preview"}, code=404)
                with open(p, "rb") as f:
                    data = f.read()
                return self._send_bytes(data, "image/jpeg")
            except Exception:
                return self._send_json({"error": "no preview"}, code=404)

        return self._send_json({"error": "not found"}, code=404)

    # ── POST ─────────────────────────────────────────────────────────────────
    def _read_body(self, cap: int = 64 * 1024) -> bytes:
        """Read the request body up to ``cap`` bytes (never raises). Shared by the
        /api/say and /api/settings handlers so the body-length parsing + size cap
        live in ONE place."""
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return b""
        try:
            return self.rfile.read(min(length, cap))
        except Exception:
            return b""

    def do_POST(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        # Both POST routes below change state (run a command / write config). On a
        # local bind, refuse a browser-driven cross-origin or DNS-rebound request
        # BEFORE doing anything. Applied unconditionally (any POST path).
        ok, why = self._state_change_allowed()
        if not ok:
            return self._forbidden(why)

        # POST /api/settings — WRITE settings. A settings write is POWERFUL (it can
        # flip WEB_INTERFACE_BIND/TOKEN, enable ambient listening, etc.), so it is
        # gated by the SAME auth as every other route: when a token is configured it
        # is REQUIRED here (a settings write is never allowed token-free on an
        # exposed bind; on a local bind with no token there's no token to require,
        # exactly like /api/say). See _handle_post_settings.
        if path == "/api/settings":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._handle_post_settings(cfg)

        if path != "/api/say":
            return self._send_json({"error": "not found"}, code=404)
        if not self._authorized(query, is_page=False):
            return self._unauthorized()

        # Parse the JSON body {"text": "...", "timeout": <optional seconds>}.
        raw = self._read_body()
        text = ""
        req_timeout = _REPLY_TIMEOUT_DEFAULT
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            text = str(data.get("text", "")).strip()
            # Parse timeout SEPARATELY — a non-numeric timeout must fall back to the
            # default, NEVER discard an otherwise-valid command (2026-07-08 finding).
            if "timeout" in data:
                try:
                    req_timeout = float(data["timeout"])
                except (TypeError, ValueError):
                    req_timeout = _REPLY_TIMEOUT_DEFAULT
        if not text:
            return self._send_json({"error": "empty text"}, code=400)

        # Inject via the SAME channel the voice loop drains, then wait for a reply.
        try:
            inject_command(text, cfg["inject_path"])
        except Exception as e:
            return self._send_json({"error": f"inject failed: {e}"}, code=500)

        reader = cfg.get("reply_reader") or wait_for_reply
        try:
            res = reader(text, cfg["log_dir"], req_timeout)
        except Exception as e:
            # The command was injected and will run; we just couldn't tail a reply.
            return self._send_json({"accepted": True, "reply": "", "status": f"reply_error: {e}"})

        status = res.get("status", "accepted")
        lines = res.get("lines", []) or []
        return self._send_json({
            "accepted": True,
            "status": status,
            "reply": "\n".join(lines),
            "reply_lines": lines,
        })

    def _handle_post_settings(self, cfg: dict) -> None:
        """Handle POST /api/settings — validate + merge one or more settings into
        the user_settings.json file, atomically.

        BODY SHAPE (both accepted):
          • a single update:  {"name": "WAKE_WORD_AUTOSTART", "value": true}
          • a batch:          {"settings": {"WAKE_WORD_AUTOSTART": true,
                                            "TTS_BACKEND": "edge"}}
        RESPONSES:
          • 200 {"ok": true, "applied": {name: coerced_value, ...}, "note": "…"}
          • 400 on empty body / unknown key / bad type (clear message)
          • 500 if the atomic file write itself fails
        The ``note`` is the honest restart caveat (SETTINGS_RESTART_NOTE) — we do
        NOT claim a live-apply we can't guarantee."""
        raw = self._read_body()
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return self._send_json({"error": "invalid JSON body"}, code=400)
        if not isinstance(data, dict):
            return self._send_json({"error": "body must be a JSON object"},
                                   code=400)
        # Normalise the two accepted shapes into one {name: value} dict.
        updates: dict = {}
        if isinstance(data.get("settings"), dict):
            updates = dict(data["settings"])
        elif "name" in data:
            updates = {str(data["name"]): data.get("value")}
        if not updates:
            return self._send_json(
                {"error": "no settings to apply — send {name, value} or "
                          "{settings: {name: value, ...}}"}, code=400)
        # Validate + merge (raises SettingsWriteError → 400 on a bad key/value).
        try:
            applied = _write_settings(updates, cfg["user_settings_path"])
        except SettingsWriteError as e:
            return self._send_json({"error": str(e)}, code=400)
        except Exception as e:                      # a disk write failure etc.
            return self._send_json({"error": f"settings write failed: {e}"},
                                   code=500)
        return self._send_json({
            "ok": True,
            "applied": applied,
            "note": SETTINGS_RESTART_NOTE,
        })


# ── the dashboard page (single inline dark, arc-reactor-cyan HTML/JS) ────────

def _dashboard_html(token: str) -> str:
    """Return the full dashboard page. Inline CSS/JS only (no external fetches),
    dark theme with arc-reactor cyan accents. Polls /api/status + /api/log/tail
    once a second and POSTs typed commands to /api/say. The token (if any) is
    baked into the JS so the page's own API calls carry it."""
    # html.escape the token so a token with quotes can't break out of the JS
    # string literal (it's a shared secret, not attacker-controlled, but cheap
    # to be correct).
    tok = html.escape(token or "", quote=True)
    # NOTE: literal braces in the CSS/JS are doubled because this is an f-string.
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JARVIS — Live</title>
<style>
  :root {{ --cyan:#22d3ee; --cyan-dim:#0e7490; --bg:#05080d; --panel:#0b1420;
           --edge:#123; --text:#cfe9f2; --muted:#5b7a86; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:radial-gradient(1200px 600px at 50% -10%, #0a1a26 0%, var(--bg) 60%);
          color:var(--text); font:14px/1.5 ui-monospace,Menlo,Consolas,monospace; }}
  header {{ display:flex; align-items:center; gap:14px; padding:14px 18px;
            border-bottom:1px solid var(--edge); position:sticky; top:0;
            background:rgba(5,8,13,.9); backdrop-filter:blur(6px); }}
  .reactor {{ width:22px; height:22px; border-radius:50%;
              background:radial-gradient(circle at 50% 50%, #eafcff, var(--cyan) 45%, var(--cyan-dim) 70%, #04222b 100%);
              box-shadow:0 0 12px var(--cyan), 0 0 28px var(--cyan-dim); }}
  h1 {{ font-size:15px; margin:0; letter-spacing:.28em; color:var(--cyan); }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:16px 18px; }}
  .strip {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }}
  .chip {{ background:var(--panel); border:1px solid var(--edge); border-radius:8px;
           padding:8px 12px; min-width:120px; }}
  .chip .k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; }}
  .chip .v {{ color:var(--cyan); font-size:15px; margin-top:2px; word-break:break-word; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
          background:#666; vertical-align:middle; }}
  .dot.on {{ background:#2ee6a6; box-shadow:0 0 8px #2ee6a6; }}
  #log {{ background:#04070c; border:1px solid var(--edge); border-radius:8px;
          height:52vh; overflow:auto; padding:10px 12px; white-space:pre-wrap;
          font-size:12.5px; color:#a9c7d1; }}
  #log .a {{ color:var(--cyan); }}
  #log .j {{ color:#eafcff; }}
  form {{ display:flex; gap:8px; margin-top:14px; }}
  input[type=text] {{ flex:1; background:#04070c; border:1px solid var(--edge);
           color:var(--text); border-radius:8px; padding:11px 12px; font:inherit; }}
  input[type=text]:focus {{ outline:none; border-color:var(--cyan); box-shadow:0 0 0 1px var(--cyan-dim); }}
  button {{ background:var(--cyan-dim); color:#eafcff; border:1px solid var(--cyan);
            border-radius:8px; padding:0 18px; font:inherit; cursor:pointer; }}
  button:hover {{ background:var(--cyan); color:#04222b; }}
  button:disabled {{ opacity:.5; cursor:default; }}
  /* Quick-action row: a horizontal, wrapping strip of preset-command buttons.
     They reuse the base button look but are smaller/pill-shaped and ghosted
     (transparent fill) so the primary Send button stays the visual anchor. */
  .actions {{ display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 6px; }}
  .actions button {{ padding:7px 13px; font-size:12.5px; border-radius:999px;
            background:transparent; color:var(--cyan); }}
  .actions button:hover {{ background:var(--cyan); color:#04222b; }}
  /* Auto-refresh toggle in the header — a small inline checkbox + label. */
  .toggle {{ display:inline-flex; align-items:center; gap:6px; color:var(--muted);
            font-size:12px; cursor:pointer; user-select:none; }}
  .toggle input {{ accent-color:var(--cyan); cursor:pointer; }}
  #reply {{ margin-top:10px; color:#eafcff; min-height:1.4em; }}
  .muted {{ color:var(--muted); }}
  /* ── Settings panel ──────────────────────────────────────────────────────
     A second "view" under the live dashboard. The nav toggles which of the two
     sections (live / settings) is visible; only one shows at a time so the page
     stays a single self-contained screen. Same dark arc-reactor-cyan palette. */
  nav.views {{ display:flex; gap:8px; margin-left:18px; }}
  nav.views button {{ padding:6px 14px; font-size:12.5px; border-radius:999px;
            background:transparent; color:var(--cyan); }}
  nav.views button.active {{ background:var(--cyan); color:#04222b; }}
  .view[hidden] {{ display:none; }}
  /* The prominent wake-word switch sits at the top of Settings so it's the first
     thing the owner sees (the headline "do it all from the web" control). */
  .wakebanner {{ background:linear-gradient(90deg, #0b1a26, var(--panel));
            border:1px solid var(--cyan-dim); border-radius:10px;
            padding:14px 16px; margin-bottom:16px; display:flex;
            align-items:center; gap:14px; flex-wrap:wrap; }}
  .wakebanner .lbl {{ color:var(--cyan); font-size:14px; letter-spacing:.06em; }}
  .wakebanner .hint {{ color:var(--muted); font-size:12px; flex-basis:100%; }}
  /* Each tab-group of settings is a titled card; rows stack inside it. */
  .sgroup {{ background:var(--panel); border:1px solid var(--edge);
            border-radius:10px; padding:6px 14px 12px; margin-bottom:14px; }}
  .sgroup > h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.16em;
            color:var(--muted); margin:12px 2px 8px; }}
  .srow {{ display:flex; align-items:flex-start; gap:12px; padding:9px 2px;
            border-top:1px solid #0c1a24; flex-wrap:wrap; }}
  .srow:first-of-type {{ border-top:none; }}
  .srow .meta {{ flex:1 1 260px; min-width:220px; }}
  .srow .meta .name {{ color:var(--text); }}
  .srow .meta .help {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .srow .ctl {{ flex:0 0 auto; display:flex; align-items:center; gap:8px; }}
  .srow .ctl input[type=text], .srow .ctl input[type=number], .srow .ctl select {{
            background:#04070c; border:1px solid var(--edge); color:var(--text);
            border-radius:8px; padding:8px 10px; font:inherit; min-width:160px; }}
  .srow .ctl input:focus, .srow .ctl select:focus {{ outline:none;
            border-color:var(--cyan); box-shadow:0 0 0 1px var(--cyan-dim); }}
  .srow .ctl input[type=checkbox] {{ width:18px; height:18px;
            accent-color:var(--cyan); cursor:pointer; }}
  .srow .save {{ padding:7px 12px; font-size:12px; }}
  .srow .saved {{ color:#2ee6a6; font-size:12px; min-width:1em; }}
  #settingsNote {{ color:var(--muted); font-size:12px; margin:2px 2px 14px; }}
  /* ── Control-panel tabs (System / Actions / Voice / Camera / Memory) ───────
     All reuse the shared palette + the .view[hidden] show/hide mechanic. */
  /* System: a responsive grid of GPU cards + a stat row. */
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
            gap:12px; margin-bottom:14px; }}
  .card {{ background:var(--panel); border:1px solid var(--edge);
           border-radius:10px; padding:12px 14px; }}
  .card h3 {{ font-size:13px; margin:0 0 8px; color:var(--cyan);
             word-break:break-word; }}
  .card .kv {{ display:flex; justify-content:space-between; gap:10px;
              font-size:12.5px; padding:2px 0; color:#a9c7d1; }}
  .card .kv b {{ color:var(--text); font-weight:normal; }}
  /* A thin VRAM/usage bar: a filled inner track sized by percentage. */
  .bar {{ height:8px; border-radius:6px; background:#04070c;
          border:1px solid var(--edge); overflow:hidden; margin:6px 0 8px; }}
  .bar > i {{ display:block; height:100%; background:linear-gradient(90deg,
          var(--cyan-dim), var(--cyan)); }}
  /* A shared search box for the Actions + Memory lists. */
  .search {{ width:100%; background:#04070c; border:1px solid var(--edge);
          color:var(--text); border-radius:8px; padding:10px 12px; font:inherit;
          margin-bottom:10px; }}
  .search:focus {{ outline:none; border-color:var(--cyan);
          box-shadow:0 0 0 1px var(--cyan-dim); }}
  /* A scrollable list panel (Actions / Memory facts / episodes). */
  .listbox {{ background:#04070c; border:1px solid var(--edge); border-radius:8px;
          max-height:56vh; overflow:auto; }}
  .lrow {{ display:flex; align-items:center; gap:10px; padding:8px 12px;
          border-top:1px solid #0c1a24; }}
  .lrow:first-child {{ border-top:none; }}
  .lrow .nm {{ flex:1; color:var(--text); word-break:break-word; cursor:pointer; }}
  .lrow .nm:hover {{ color:var(--cyan); }}
  .lrow .txt {{ flex:1; color:#a9c7d1; word-break:break-word; }}
  /* A small speak-class chip on each action row. */
  .schip {{ font-size:10.5px; letter-spacing:.06em; padding:2px 8px;
          border-radius:999px; border:1px solid var(--edge); color:var(--muted);
          white-space:nowrap; }}
  .schip.verbatim {{ color:#2ee6a6; border-color:#12604a; }}
  .schip.informative {{ color:var(--cyan); border-color:var(--cyan-dim); }}
  .lrow .send {{ padding:5px 12px; font-size:12px; border-radius:999px;
          background:transparent; color:var(--cyan); }}
  .lrow .send:hover {{ background:var(--cyan); color:#04222b; }}
  .count {{ color:var(--muted); font-size:12px; margin:2px 2px 10px; }}
  /* Voice: a wrapping row of profile buttons + an info strip. */
  .voicebtns {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }}
  .voicebtns button {{ padding:8px 14px; font-size:12.5px; border-radius:999px;
          background:transparent; color:var(--cyan); }}
  .voicebtns button:hover {{ background:var(--cyan); color:#04222b; }}
  .voicebtns button.off {{ color:#f0a; border-color:#a05; }}
  /* Camera: the preview image + an off placeholder. */
  #camImg {{ max-width:100%; border:1px solid var(--edge); border-radius:10px;
          background:#04070c; display:none; }}
  #camOff {{ background:var(--panel); border:1px dashed var(--cyan-dim);
          border-radius:10px; padding:40px 16px; text-align:center;
          color:var(--muted); }}
</style></head><body>
<header><div class="reactor"></div><h1>J.A.R.V.I.S.</h1>
  <!-- View switcher: LIVE dashboard vs the full SETTINGS control panel. Only one
       view is shown at a time so the page stays one self-contained screen. -->
  <nav class="views">
    <button id="navLive" class="active" type="button">Live</button>
    <button id="navSystem" type="button">System</button>
    <button id="navActions" type="button">Actions</button>
    <button id="navVoice" type="button">Voice</button>
    <button id="navCamera" type="button">Camera</button>
    <button id="navMemory" type="button">Memory</button>
    <button id="navSettings" type="button">Settings</button>
  </nav>
  <label class="toggle" style="margin-left:auto" title="Pause/resume live status + log polling">
    <input id="autorefresh" type="checkbox" checked> auto-refresh
  </label>
  <span id="conn" class="muted">connecting…</span>
</header>
<div class="wrap">
  <!-- ── LIVE VIEW (unchanged dashboard: status strip / quick actions / log /
       command box) ────────────────────────────────────────────────────────── -->
  <section id="viewLive" class="view">
    <div class="strip" id="strip"></div>
    <!-- Quick-action buttons are injected here from the QUICK_ACTIONS array below,
         so presets are edited in ONE data-driven place (no per-button markup). -->
    <div class="actions" id="actions"></div>
    <div id="log" class="muted">loading log…</div>
    <form id="say">
      <input id="text" type="text" autocomplete="off" placeholder="Type a command for JARVIS…" autofocus>
      <button id="send" type="submit">Send</button>
    </form>
    <div id="reply"></div>
  </section>

  <!-- ── SETTINGS VIEW (the full control panel) ──────────────────────────────
       The wake-word switch is pinned in the banner at the top; every other knob
       is rendered from /api/settings, grouped by tab, into #settingsGroups. The
       whole thing is built client-side from the schema so the panel never drifts
       from settings_window.SCHEMA (the single source of truth). -->
  <section id="viewSettings" class="view" hidden>
    <div class="wakebanner">
      <label class="toggle" style="color:var(--cyan)">
        <input id="wakeToggle" type="checkbox"> <span class="lbl">Wake-word mode (start in standby)</span>
      </label>
      <button id="wakeSave" class="save" type="button">Save</button>
      <span id="wakeSaved" class="saved"></span>
      <span class="hint">Boot silent and wait for &ldquo;JARVIS&rdquo; instead of always-listening
        (WAKE_WORD_AUTOSTART). Toggle the neural detector + full standby knobs below too.</span>
    </div>
    <div id="settingsNote" class="muted">loading settings…</div>
    <div id="settingsGroups"></div>
  </section>

  <!-- ── SYSTEM VIEW (live hardware: GPUs / CPU / RAM / disks) ──────────────
       Auto-refreshes on a ~2s interval while visible (see showView/systemTimer). -->
  <section id="viewSystem" class="view" hidden>
    <div id="sysMeta" class="count">loading system…</div>
    <div class="cards" id="sysGpus"></div>
    <div class="cards" id="sysHost"></div>
  </section>

  <!-- ── ACTIONS VIEW (the "access everything" tab) ────────────────────────
       A search box filters the full ~500-action inventory; each row can be Sent
       straight to /api/say, or its name clicked to drop into the Live command box. -->
  <section id="viewActions" class="view" hidden>
    <div id="actionsCount" class="count">loading actions…</div>
    <input id="actionsSearch" class="search" type="text" autocomplete="off"
           placeholder="Search actions… (name)">
    <div id="actionsList" class="listbox"></div>
  </section>

  <!-- ── VOICE VIEW (active voice + one button per usable clone profile) ────
       Read-then-command: each button POSTs a spoken phrase to /api/say (no new
       write endpoint), so switching a voice goes through the same channel. -->
  <section id="viewVoice" class="view" hidden>
    <div id="voiceInfo" class="count">loading voices…</div>
    <div id="voiceBtns" class="voicebtns"></div>
  </section>

  <!-- ── CAMERA VIEW (live preview frame, refreshed ~1s while visible) ──────
       The img points at /api/camera-preview; on error (missing/stale = camera
       off) the placeholder is shown instead. -->
  <section id="viewCamera" class="view" hidden>
    <div class="count">Live camera preview (updates while the camera is on).</div>
    <img id="camImg" alt="camera preview">
    <div id="camOff">camera off / no preview</div>
  </section>

  <!-- ── MEMORY VIEW (long-term facts + recent episodes, searchable) ───────── -->
  <section id="viewMemory" class="view" hidden>
    <div id="memCount" class="count">loading memory…</div>
    <input id="memSearch" class="search" type="text" autocomplete="off"
           placeholder="Search facts…">
    <div id="memFacts" class="listbox"></div>
  </section>
</div>
<script>
const TOKEN = "{tok}";
function hdr() {{ const h = {{'Content-Type':'application/json'}}; if (TOKEN) h['X-Auth-Token']=TOKEN; return h; }}
function q(u) {{ return TOKEN ? (u + (u.includes('?')?'&':'?') + 'token=' + encodeURIComponent(TOKEN)) : u; }}
const strip = document.getElementById('strip');
const logEl = document.getElementById('log');
const conn  = document.getElementById('conn');

function chip(k, v) {{ const d=document.createElement('div'); d.className='chip';
  d.innerHTML = '<div class="k">'+k+'</div><div class="v"></div>';
  d.querySelector('.v').textContent = (v===''||v==null) ? '—' : v; return d; }}

// Format a raw uptime-in-seconds float into a compact "2h13m" / "4m" / "45s".
function fmtUptime(secs) {{
  if (secs==null || isNaN(secs)) return '';
  secs = Math.max(0, Math.floor(secs));
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = secs%60;
  if (h) return h+'h'+String(m).padStart(2,'0')+'m';
  if (m) return m+'m'+String(s).padStart(2,'0')+'s';
  return s+'s';
}}

async function refreshStatus() {{
  try {{
    const r = await fetch(q('/api/status'), {{headers:hdr()}});
    if (r.status===401) {{ conn.textContent='unauthorized — token required'; return; }}
    const s = await r.json();
    // Clearer online/offline indicator: a coloured dot + explicit word. The dot
    // is green (glowing) when the loop is live, grey when offline.
    conn.textContent = ''; conn.innerHTML =
      '<span class="dot '+(s.running?'on':'')+'"></span>'+(s.running?'live':'offline');
    strip.innerHTML='';
    // A leading status chip whose whole value is the online/offline dot, so the
    // strip itself carries a clear liveness signal (not just the header corner).
    const st = chip('online', s.running?'live':'offline');
    st.querySelector('.v').innerHTML =
      '<span class="dot '+(s.running?'on':'')+'"></span>'+(s.running?'live':'offline');
    strip.appendChild(st);
    strip.appendChild(chip('version', s.version));
    strip.appendChild(chip('state', s.state));
    // Uptime — only when the server could derive it (field present + non-null).
    if (s.uptime!=null) strip.appendChild(chip('uptime', fmtUptime(s.uptime)));
    strip.appendChild(chip('model / routing', s.routing || s.model));
    const g = (s.gpu_lines&&s.gpu_lines.length) ? s.gpu_lines[s.gpu_lines.length- (s.routing?2:1)] : '';
    strip.appendChild(chip('vram', (s.gpu_bar||'') + (g? '  '+g : '')));
    // Air-mouse chip — ONLY present when build_status could read the skill in-process
    // (s.air_mouse is omitted otherwise). Shows ARMED (+engaged) vs disarmed.
    if (s.air_mouse) {{
      const am = s.air_mouse.armed
        ? ('armed' + (s.air_mouse.engaged ? ' · engaged' : ''))
        : 'disarmed';
      strip.appendChild(chip('air-mouse', am));
    }}
    if (s.now_playing) strip.appendChild(chip('now playing', s.now_playing));
    if (s.last_spoken) strip.appendChild(chip('last said', s.last_spoken));
  }} catch(e) {{ conn.textContent = 'connection lost'; }}
}}

let pinned = true;
logEl.addEventListener('scroll', () => {{
  pinned = (logEl.scrollTop + logEl.clientHeight) >= (logEl.scrollHeight - 24);
}});
async function refreshLog() {{
  try {{
    const r = await fetch(q('/api/log/tail?lines=200'), {{headers:hdr()}});
    if (r.status===401) return;
    const d = await r.json();
    const frag = (d.lines||[]).map(l => {{
      const cls = /\\[action\\]/i.test(l) ? 'a' : (/jarvis:/i.test(l) ? 'j' : '');
      const esc = l.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      return cls ? '<span class="'+cls+'">'+esc+'</span>' : esc;
    }}).join('\\n');
    logEl.innerHTML = frag || '<span class="muted">(no log yet)</span>';
    if (pinned) logEl.scrollTop = logEl.scrollHeight;
  }} catch(e) {{}}
}}

const form = document.getElementById('say');
const textIn = document.getElementById('text');
const sendBtn = document.getElementById('send');
const replyEl = document.getElementById('reply');
const actionsEl = document.getElementById('actions');

// ── QUICK-ACTION PRESETS ──────────────────────────────────────────────────
// Data-driven so presets are trivial to edit HERE (one array) without touching
// markup or handlers. `label` is the button text; `cmd` is the exact phrase POSTed
// to /api/say — the SAME inject channel a spoken command uses, so "mouse control on"
// behaves identically typed, clicked, or spoken.
const QUICK_ACTIONS = [
  {{label:'Arm mouse control', cmd:'mouse control on'}},
  {{label:'Release mouse',     cmd:'mouse control off'}},
  {{label:"What's my status",  cmd:'system status'}},
  {{label:'Go to sleep',       cmd:'go to sleep'}},
  {{label:'Wake up',           cmd:'wake up'}},
];

// The ONE code path every command goes through — the typed form and every quick
// button both call this. Disables the sender, shows a pending marker, POSTs the
// phrase, renders the reply (or a queued/accepted note), and nudges the log.
async function sendCommand(text, opts) {{
  text = (text||'').trim(); if (!text) return;
  opts = opts || {{}};
  const btn = opts.button || null;
  sendBtn.disabled = true; if (btn) btn.disabled = true;
  replyEl.textContent = '…';
  try {{
    const r = await fetch(q('/api/say'), {{method:'POST', headers:hdr(),
      body: JSON.stringify({{text}})}});
    const d = await r.json();
    if (r.status===401) replyEl.textContent = 'unauthorized';
    else if (d.reply) replyEl.textContent = d.reply;
    else if (d.status==='no_log') replyEl.innerHTML = '<span class="muted">queued — JARVIS is not running; it will run on next boot.</span>';
    else replyEl.innerHTML = '<span class="muted">accepted (no spoken reply captured).</span>';
  }} catch(e) {{ replyEl.textContent = 'send failed'; }}
  finally {{ sendBtn.disabled=false; if (btn) btn.disabled=false; refreshLog(); }}
}}

// Render the quick-action buttons from QUICK_ACTIONS. Each POSTs its preset phrase
// via the shared sendCommand(); the returned reply lands in the existing #reply area.
QUICK_ACTIONS.forEach(a => {{
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = a.label;
  b.addEventListener('click', () => sendCommand(a.cmd, {{button:b}}));
  actionsEl.appendChild(b);
}});

form.addEventListener('submit', async (ev) => {{
  ev.preventDefault();
  await sendCommand(textIn.value);
  textIn.value=''; textIn.focus();
}});

// ── AUTO-REFRESH TOGGLE ────────────────────────────────────────────────────
// The checkbox (default ON) gates the 1s/1.5s polls so the user can FREEZE the
// view to read the log/status. We keep the intervals running but make each tick a
// no-op while paused (cheaper + simpler than clearing/re-creating timers), and do
// one immediate refresh when it's switched back on so the view catches up at once.
const autoEl = document.getElementById('autorefresh');
function autoOn() {{ return autoEl.checked; }}
autoEl.addEventListener('change', () => {{ if (autoOn()) {{ refreshStatus(); refreshLog(); }} }});

// ── SETTINGS CONTROL PANEL ─────────────────────────────────────────────────
// The full "do it all from the web" panel. It's built ENTIRELY from /api/settings
// (which serves settings_window.SCHEMA + live values), so it never drifts from the
// real config. Each control saves INDEPENDENTLY via POST /api/settings {{name,value}}
// and shows a per-row confirmation. A save writes the file; the effect lands on the
// next JARVIS restart (the note the server returns says so).
const navLive = document.getElementById('navLive');
const navSettings = document.getElementById('navSettings');
const viewLive = document.getElementById('viewLive');
const viewSettings = document.getElementById('viewSettings');
const settingsGroups = document.getElementById('settingsGroups');
const settingsNote = document.getElementById('settingsNote');
const wakeToggle = document.getElementById('wakeToggle');
const wakeSave = document.getElementById('wakeSave');
const wakeSaved = document.getElementById('wakeSaved');

// Friendly tab titles for the group headings (fallback to the raw key).
const TAB_TITLES = {{ voice:'Voice / Audio', ai:'AI / Models',
  privacy:'Privacy / Ambient', integrations:'Integrations', advanced:'Advanced' }};
// The wake-word knob the banner switch drives — the headline control the owner
// asked for. START_IN_STANDBY is the "Alexa-style wake-word mode" toggle;
// WAKE_WORD_AUTOSTART (the neural detector) is surfaced as a normal row below.
const WAKE_KEY = 'START_IN_STANDBY';
let settingsLoaded = false;

// Element refs for the five control-panel tabs.
const navSystem  = document.getElementById('navSystem');
const navActions = document.getElementById('navActions');
const navVoice   = document.getElementById('navVoice');
const navCamera  = document.getElementById('navCamera');
const navMemory  = document.getElementById('navMemory');
const viewSystem  = document.getElementById('viewSystem');
const viewActions = document.getElementById('viewActions');
const viewVoice   = document.getElementById('viewVoice');
const viewCamera  = document.getElementById('viewCamera');
const viewMemory  = document.getElementById('viewMemory');

// One registry drives show/hide + nav-active for EVERY view. Settings keeps its
// own settingsLoaded flag (the wake-word save resets it to force a reload), so it
// is dispatched specially below; the other lazy tabs use per-tab loaded flags.
// System + Camera additionally run a while-visible refresh timer (stopped on
// leave) so their live data updates without touching the other tabs.
const VIEWS = {{
  live:     {{nav:navLive,     view:viewLive}},
  system:   {{nav:navSystem,   view:viewSystem}},
  actions:  {{nav:navActions,  view:viewActions}},
  voice:    {{nav:navVoice,    view:viewVoice}},
  camera:   {{nav:navCamera,   view:viewCamera}},
  memory:   {{nav:navMemory,   view:viewMemory}},
  settings: {{nav:navSettings, view:viewSettings}},
}};
let currentView = 'live';
let systemTimer = null, cameraTimer = null;
let actionsLoaded = false, voiceLoaded = false, memoryLoaded = false;

function stopViewTimers() {{
  if (systemTimer) {{ clearInterval(systemTimer); systemTimer = null; }}
  if (cameraTimer) {{ clearInterval(cameraTimer); cameraTimer = null; }}
}}

function showView(which) {{
  if (!VIEWS[which]) which = 'live';
  currentView = which;
  Object.keys(VIEWS).forEach(k => {{
    const on = (k === which);
    VIEWS[k].view.hidden = !on;
    VIEWS[k].nav.classList.toggle('active', on);
  }});
  stopViewTimers();
  if (which === 'settings') {{ if (!settingsLoaded) loadSettings(); }}
  else if (which === 'system') {{
    loadSystem();
    systemTimer = setInterval(() => {{ if (autoOn()) loadSystem(); }}, 2000);
  }}
  else if (which === 'actions') {{ if (!actionsLoaded) {{ loadActions(); actionsLoaded = true; }} }}
  else if (which === 'voice')   {{ if (!voiceLoaded)   {{ loadVoices();  voiceLoaded  = true; }} }}
  else if (which === 'memory')  {{ if (!memoryLoaded)  {{ loadMemory();  memoryLoaded = true; }} }}
  else if (which === 'camera')  {{
    refreshCamera();
    cameraTimer = setInterval(() => {{ if (autoOn()) refreshCamera(); }}, 1000);
  }}
}}
navLive.addEventListener('click', () => showView('live'));
navSystem.addEventListener('click', () => showView('system'));
navActions.addEventListener('click', () => showView('actions'));
navVoice.addEventListener('click', () => showView('voice'));
navCamera.addEventListener('click', () => showView('camera'));
navMemory.addEventListener('click', () => showView('memory'));
navSettings.addEventListener('click', () => showView('settings'));

// Build ONE control for a schema item, returning {{el, read}} where read() yields
// the value to POST. bool→checkbox, enum→select, combo→text+datalist, int/float→
// number, everything else→text.
function buildControl(it) {{
  const t = it.type;
  // Secret knobs (e.g. the web token) never receive the live value from the
  // server — it's redacted. Render a password field; an EMPTY save means "keep
  // the current secret" (the click handler skips the POST), so a blank field
  // can't wipe an existing token. Type something to replace it.
  if (it.secret) {{
    const inp=document.createElement('input'); inp.type='password';
    inp.autocomplete='new-password';
    inp.placeholder = it.is_set ? '•••••• (set — type to replace)' : '(not set)';
    return {{el:inp, read:()=>inp.value, secret:true}};
  }}
  if (t === 'bool') {{
    const cb = document.createElement('input'); cb.type='checkbox';
    cb.checked = !!it.value; return {{el:cb, read:()=>cb.checked}};
  }}
  if (t === 'enum') {{
    const sel = document.createElement('select');
    (it.choices||[]).forEach(c => {{ const o=document.createElement('option');
      o.value=c; o.textContent=c; if (String(it.value)===String(c)) o.selected=true;
      sel.appendChild(o); }});
    return {{el:sel, read:()=>sel.value}};
  }}
  if (t === 'int' || t === 'float') {{
    const inp=document.createElement('input'); inp.type='number';
    if (t==='float') inp.step='any';
    inp.value = (it.value==null?'':it.value);
    return {{el:inp, read:()=> t==='int'?parseInt(inp.value,10):parseFloat(inp.value)}};
  }}
  // combo (free text + suggestions), str, device, text, routing → a text input.
  // combo gets a datalist of its suggested choices; the user can still type any.
  const inp=document.createElement('input'); inp.type='text';
  let val = it.value;
  if (val && typeof val === 'object') val = JSON.stringify(val);   // routing/list → shown as JSON
  inp.value = (val==null?'':val);
  if (t === 'combo' && (it.choices||[]).length) {{
    const dl=document.createElement('datalist'); const id='dl_'+it.name;
    dl.id=id; (it.choices||[]).forEach(c=>{{const o=document.createElement('option');
      o.value=c; dl.appendChild(o);}}); inp.setAttribute('list', id);
    const frag=document.createDocumentFragment(); frag.appendChild(inp); frag.appendChild(dl);
    return {{el:frag, read:()=>inp.value, focusEl:inp}};
  }}
  return {{el:inp, read:()=>inp.value}};
}}

// POST a single {{name,value}} and reflect the outcome in `saved` (a small span).
async function saveSetting(name, value, saved) {{
  saved.textContent='…'; saved.style.color='var(--muted)';
  try {{
    const r = await fetch(q('/api/settings'), {{method:'POST', headers:hdr(),
      body: JSON.stringify({{name, value}})}});
    const d = await r.json();
    if (r.ok && d.ok) {{ saved.textContent='saved ✓'; saved.style.color='#2ee6a6';
      if (d.note) settingsNote.textContent = d.note; }}
    else {{ saved.textContent = (d.error||'error'); saved.style.color='#f85149'; }}
  }} catch(e) {{ saved.textContent='failed'; saved.style.color='#f85149'; }}
}}

// Render the whole panel from an /api/settings payload: group by tab, one card per
// tab, one row per knob (meta + control + per-row Save button + confirmation).
function renderSettings(payload) {{
  settingsGroups.innerHTML='';
  const items = payload.settings || [];
  settingsNote.textContent = payload.note || '';
  const tabs = payload.tabs && payload.tabs.length ? payload.tabs
    : Array.from(new Set(items.map(i=>i.tab)));
  tabs.forEach(tab => {{
    const inTab = items.filter(i => i.tab === tab);
    if (!inTab.length) return;
    const group=document.createElement('div'); group.className='sgroup';
    const h=document.createElement('h2'); h.textContent = TAB_TITLES[tab]||tab;
    group.appendChild(h);
    inTab.forEach(it => {{
      const row=document.createElement('div'); row.className='srow';
      const meta=document.createElement('div'); meta.className='meta';
      meta.innerHTML = '<div class="name"></div>'+(it.help?'<div class="help"></div>':'');
      meta.querySelector('.name').textContent = it.label + '  ('+it.name+')';
      if (it.help) meta.querySelector('.help').textContent = it.help;
      const ctl=document.createElement('div'); ctl.className='ctl';
      const c = buildControl(it);
      ctl.appendChild(c.el);
      const saveBtn=document.createElement('button'); saveBtn.className='save';
      saveBtn.type='button'; saveBtn.textContent='Save';
      const saved=document.createElement('span'); saved.className='saved';
      saveBtn.addEventListener('click', () => {{
        const v = c.read();
        // Empty save on a secret = "keep the current value" — never POST "" and
        // wipe an existing token by accident.
        if (c.secret && (v===''||v==null)) {{
          saved.textContent='unchanged'; saved.style.color='var(--muted)'; return;
        }}
        saveSetting(it.name, v, saved);
      }});
      ctl.appendChild(saveBtn); ctl.appendChild(saved);
      row.appendChild(meta); row.appendChild(ctl);
      group.appendChild(row);
      // Mirror the wake-word row into the top banner switch so the headline
      // toggle and its row stay in sync (the banner is the prominent shortcut).
      if (it.name === WAKE_KEY) wakeToggle.checked = !!it.value;
    }});
    settingsGroups.appendChild(group);
  }});
}}

async function loadSettings() {{
  try {{
    const r = await fetch(q('/api/settings'), {{headers:hdr()}});
    if (r.status===401) {{ settingsNote.textContent='unauthorized — token required'; return; }}
    const d = await r.json();
    renderSettings(d);
    settingsLoaded = true;
  }} catch(e) {{ settingsNote.textContent='could not load settings'; }}
}}

// The prominent banner switch: saves START_IN_STANDBY directly, then reloads the
// panel so every mirrored row reflects the new value.
wakeSave.addEventListener('click', async () => {{
  await saveSetting(WAKE_KEY, wakeToggle.checked, wakeSaved);
  settingsLoaded = false; loadSettings();
}});

// ── SYSTEM TAB ──────────────────────────────────────────────────────────────
// Live hardware view: a card per GPU (VRAM bar + temp/util/power), a CPU/RAM
// card, and a card per disk. Auto-refreshed ~2s while visible (showView).
const sysMeta = document.getElementById('sysMeta');
const sysGpus = document.getElementById('sysGpus');
const sysHost = document.getElementById('sysHost');
function pctOf(u, t) {{ return (u!=null && t) ? Math.max(0, Math.min(100, Math.round(100*u/t))) : 0; }}
function kvRow(k, v) {{ const d=document.createElement('div'); d.className='kv';
  d.innerHTML='<span></span><b></b>';
  d.querySelector('span').textContent=k; d.querySelector('b').textContent=v; return d; }}
function renderSystem(s) {{
  sysMeta.textContent = 'version ' + (s.version||'?')
    + (s.uptime!=null ? '  ·  up ' + fmtUptime(s.uptime) : '')
    + (s.routing ? '  ·  ' + s.routing : '');
  sysGpus.innerHTML='';
  (s.gpus||[]).forEach(g => {{
    const c=document.createElement('div'); c.className='card';
    const p=pctOf(g.mem_used_mb, g.mem_total_mb);
    const h=document.createElement('h3');
    h.textContent='GPU ' + g.index + ' · ' + (g.name||''); c.appendChild(h);
    const bar=document.createElement('div'); bar.className='bar';
    const fill=document.createElement('i'); fill.style.width=p+'%'; bar.appendChild(fill);
    c.appendChild(bar);
    c.appendChild(kvRow('VRAM', (g.mem_used_mb!=null?g.mem_used_mb:'?') + ' / '
      + (g.mem_total_mb!=null?g.mem_total_mb:'?') + ' MB (' + p + '%)'));
    c.appendChild(kvRow('Util', g.util_pct!=null ? g.util_pct + '%' : 'n/a'));
    c.appendChild(kvRow('Temp', g.temp_c!=null ? g.temp_c + '°C' : 'n/a'));
    c.appendChild(kvRow('Power', g.power_w!=null ? g.power_w + ' W' : 'n/a'));
    sysGpus.appendChild(c);
  }});
  if (!(s.gpus||[]).length) {{
    const c=document.createElement('div'); c.className='card';
    const h=document.createElement('h3'); h.textContent='GPU'; c.appendChild(h);
    c.appendChild(kvRow('status', 'no nvidia-smi / no GPU')); sysGpus.appendChild(c);
  }}
  sysHost.innerHTML='';
  const hc=document.createElement('div'); hc.className='card';
  const hh=document.createElement('h3'); hh.textContent='CPU / Memory'; hc.appendChild(hh);
  hc.appendChild(kvRow('CPU', s.cpu_pct!=null ? s.cpu_pct + '%' : 'n/a'));
  hc.appendChild(kvRow('RAM', (s.ram_used_gb!=null && s.ram_total_gb!=null)
    ? s.ram_used_gb + ' / ' + s.ram_total_gb + ' GB' : 'n/a'));
  sysHost.appendChild(hc);
  (s.disks||[]).forEach(dk => {{
    const c=document.createElement('div'); c.className='card';
    const h=document.createElement('h3'); h.textContent='Disk ' + (dk.drive||''); c.appendChild(h);
    const used=(dk.total_gb!=null && dk.free_gb!=null) ? (dk.total_gb - dk.free_gb) : null;
    const p=pctOf(used, dk.total_gb);
    const bar=document.createElement('div'); bar.className='bar';
    const fill=document.createElement('i'); fill.style.width=p+'%'; bar.appendChild(fill);
    c.appendChild(bar);
    c.appendChild(kvRow('free', (dk.free_gb!=null?dk.free_gb:'?') + ' / '
      + (dk.total_gb!=null?dk.total_gb:'?') + ' GB'));
    sysHost.appendChild(c);
  }});
}}
async function loadSystem() {{
  try {{
    const r = await fetch(q('/api/system'), {{headers:hdr()}});
    if (r.status===401) {{ sysMeta.textContent='unauthorized — token required'; return; }}
    renderSystem(await r.json());
  }} catch(e) {{ sysMeta.textContent='could not load system info'; }}
}}

// ── ACTIONS TAB ─────────────────────────────────────────────────────────────
// The "access everything" list: search filters the full ~500-action inventory.
// Each row's name → drops into the Live command box (edit before send); its Send
// button POSTs the action name straight to /api/say via the shared sendCommand.
const actionsCount = document.getElementById('actionsCount');
const actionsSearch = document.getElementById('actionsSearch');
const actionsList = document.getElementById('actionsList');
let ALL_ACTIONS = [];
function speakChipClass(sp) {{
  const s=(sp||'').toUpperCase();
  if (s==='VERBATIM') return 'schip verbatim';
  if (s==='INFORMATIVE') return 'schip informative';
  return 'schip';
}}
function renderActions(filter) {{
  const f=(filter||'').trim().toLowerCase();
  actionsList.innerHTML=''; let shown=0;
  const frag=document.createDocumentFragment();
  for (const a of ALL_ACTIONS) {{
    if (f && a.name.toLowerCase().indexOf(f)===-1) continue;
    if (shown>=400) break;   // cap the DOM; refine the search to see more
    const row=document.createElement('div'); row.className='lrow';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=a.name;
    nm.title='Click to edit in the Live command box';
    nm.addEventListener('click', () => {{ textIn.value=a.name; showView('live'); textIn.focus(); }});
    const chip=document.createElement('span'); chip.className=speakChipClass(a.spoken);
    chip.textContent=a.spoken;
    const send=document.createElement('button'); send.type='button';
    send.className='send'; send.textContent='Send';
    send.addEventListener('click', () => sendCommand(a.name, {{button:send}}));
    row.appendChild(nm); row.appendChild(chip); row.appendChild(send);
    frag.appendChild(row); shown++;
  }}
  actionsList.appendChild(frag);
  actionsCount.textContent = ALL_ACTIONS.length + ' actions'
    + (f ? '  ·  ' + shown + ' shown' : '');
}}
actionsSearch.addEventListener('input', () => renderActions(actionsSearch.value));
async function loadActions() {{
  try {{
    const r = await fetch(q('/api/actions'), {{headers:hdr()}});
    if (r.status===401) {{ actionsCount.textContent='unauthorized — token required'; return; }}
    const d = await r.json();
    ALL_ACTIONS = (d.actions||[]).slice().sort((a,b)=>a.name.localeCompare(b.name));
    renderActions('');
  }} catch(e) {{ actionsCount.textContent='could not load actions'; }}
}}

// ── VOICE TAB ───────────────────────────────────────────────────────────────
// Read-then-command: shows the active voice + a button per USABLE clone profile
// (POSTs "switch to the <name> voice") and a "normal voice" button (POSTs "voice
// cloning off"), all through the same /api/say inject channel.
const voiceInfo = document.getElementById('voiceInfo');
const voiceBtns = document.getElementById('voiceBtns');
function renderVoices(d) {{
  const usable=(d.profiles||[]).filter(p=>p.usable);
  const active = (d.enabled && d.active) ? d.active
    : ('normal (' + (d.tts_voice||d.tts_backend||'default') + ')');
  voiceInfo.textContent = 'Active voice: ' + active
    + '  ·  backend ' + (d.tts_backend||'?')
    + '  ·  ' + usable.length + ' usable profile(s)';
  voiceBtns.innerHTML='';
  usable.forEach(p => {{
    const b=document.createElement('button'); b.type='button';
    b.textContent='Use ' + p.name + (p.source? ' ('+p.source+')':'');
    b.addEventListener('click', () => sendCommand('switch to the ' + p.name + ' voice', {{button:b}}));
    voiceBtns.appendChild(b);
  }});
  if (!usable.length) {{
    const note=document.createElement('span'); note.className='muted';
    note.style.alignSelf='center'; note.textContent='No usable clone profiles enrolled.  ';
    voiceBtns.appendChild(note);
  }}
  const off=document.createElement('button'); off.type='button'; off.className='off';
  off.textContent='Normal voice (cloning off)';
  off.addEventListener('click', () => sendCommand('voice cloning off', {{button:off}}));
  voiceBtns.appendChild(off);
}}
async function loadVoices() {{
  try {{
    const r = await fetch(q('/api/voices'), {{headers:hdr()}});
    if (r.status===401) {{ voiceInfo.textContent='unauthorized — token required'; return; }}
    renderVoices(await r.json());
  }} catch(e) {{ voiceInfo.textContent='could not load voices'; }}
}}

// ── CAMERA TAB ──────────────────────────────────────────────────────────────
// The <img> points at /api/camera-preview (cache-busted each poll). On load it
// shows; on error (404 = missing/stale = camera off) the placeholder shows.
const camImg = document.getElementById('camImg');
const camOff = document.getElementById('camOff');
camImg.addEventListener('load',  () => {{ camImg.style.display='block'; camOff.style.display='none'; }});
camImg.addEventListener('error', () => {{ camImg.style.display='none';  camOff.style.display='block'; }});
function refreshCamera() {{ camImg.src = q('/api/camera-preview?t=' + Date.now()); }}

// ── MEMORY TAB ──────────────────────────────────────────────────────────────
// Fact + episode counts and a searchable, scrollable list of long-term facts.
const memCount = document.getElementById('memCount');
const memSearch = document.getElementById('memSearch');
const memFacts = document.getElementById('memFacts');
let ALL_FACTS = [];
function renderFacts(filter) {{
  const f=(filter||'').trim().toLowerCase();
  memFacts.innerHTML=''; let shown=0;
  const frag=document.createDocumentFragment();
  for (const fact of ALL_FACTS) {{
    if (f && (fact.text||'').toLowerCase().indexOf(f)===-1) continue;
    if (shown>=400) break;
    const row=document.createElement('div'); row.className='lrow';
    const txt=document.createElement('div'); txt.className='txt'; txt.textContent=fact.text;
    row.appendChild(txt);
    if (fact.source) {{ const chip=document.createElement('span'); chip.className='schip';
      chip.textContent=fact.source; row.appendChild(chip); }}
    frag.appendChild(row); shown++;
  }}
  memFacts.appendChild(frag);
  if (!ALL_FACTS.length) memFacts.innerHTML=
    '<div class="lrow"><span class="muted">no facts stored</span></div>';
}}
memSearch.addEventListener('input', () => renderFacts(memSearch.value));
async function loadMemory() {{
  try {{
    const r = await fetch(q('/api/memory'), {{headers:hdr()}});
    if (r.status===401) {{ memCount.textContent='unauthorized — token required'; return; }}
    const d = await r.json();
    ALL_FACTS = d.facts||[];
    const c = d.counts||{{}};
    memCount.textContent = (c.facts!=null?c.facts:ALL_FACTS.length) + ' facts · '
      + (c.episodes!=null?c.episodes:0) + ' episodes';
    renderFacts('');
  }} catch(e) {{ memCount.textContent='could not load memory'; }}
}}

refreshStatus(); refreshLog();
setInterval(() => {{ if (autoOn()) refreshStatus(); }}, 1500);
setInterval(() => {{ if (autoOn()) refreshLog(); }}, 1000);
</script></body></html>"""


# ── server factory + lifecycle ───────────────────────────────────────────────

def _port_actively_served(host: str, port: int, timeout: float = 0.35) -> bool:
    """True when something is ALREADY accepting connections on host:port — the
    signal that binding here would co-bind a live listener (see the SO_REUSEADDR
    note in create_server). A quick loopback connect: it succeeds only against an
    ACTIVE listener, so a free port or a TIME_WAIT socket from our own restart
    both read False (safe to bind). Probes the loopback even for a wildcard bind
    (0.0.0.0), since a wildcard listener still answers on 127.0.0.1. Never raises."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((probe_host, int(port))) == 0
    except Exception:
        return False


def create_server(*, bind: str, port: int, token: str = "",
                  inject_path: str = DEFAULT_INJECT_PATH,
                  log_dir: str = DEFAULT_LOG_DIR,
                  hud_state_path: str = DEFAULT_HUD_STATE_PATH,
                  user_settings_path: str | None = None,
                  camera_preview_path: str = DEFAULT_CAMERA_PREVIEW_PATH,
                  action_index_path: str = DEFAULT_ACTION_INDEX_PATH,
                  reply_reader=None) -> ThreadingHTTPServer:
    """Build (but do not serve) a ThreadingHTTPServer for the web interface.

    SECURITY GATE: refuses to construct a server on a NON-LOCAL bind when the
    token is empty (raises InsecureBindError) — the caller must supply a token to
    expose it on the LAN. A local (loopback) bind needs no token.

    ``reply_reader`` lets a test stub the log-tail reply wait (default:
    ``wait_for_reply``). All paths default to the live project files but are
    injectable so a test can point them at a temp dir and bind 127.0.0.1:0.

    ``user_settings_path`` is where POST /api/settings MERGES its writes; it
    defaults (when None) to the live data/user_settings.json resolved from
    settings_window — the same file the Settings GUI and core.config read — but a
    test points it at a throwaway file so a settings write can never clobber the
    real one (mirroring inject_path/log_dir/hud_state_path)."""
    bind = (bind or "127.0.0.1").strip()
    local = is_local_bind(bind)
    if not local and not (token or "").strip():
        raise InsecureBindError(
            f"refusing to start the web interface: bind={bind!r} is not loopback "
            f"but WEB_INTERFACE_TOKEN is empty. Set a token to expose it on the "
            f"LAN, or bind 127.0.0.1 for localhost-only (no token needed)."
        )
    # PRE-BIND PROBE (Windows SO_REUSEADDR footgun): ThreadingHTTPServer sets
    # allow_reuse_address=True, and on Windows that lets a *second* socket bind
    # a port another process is ALREADY actively LISTENing on. The two sockets
    # then split incoming connections non-deterministically — so if a stale
    # instance (or a leaked test process) is squatting the port, JARVIS binds
    # "successfully" but half the requests land on the dead socket and hang
    # (observed live 2026-07-07: a leftover `-m unittest` process held 8766 and
    # the dashboard was unreachable). We can't tell that apart from a healthy
    # bind after the fact, so REFUSE up front when the port is already being
    # served: a real port>0 that answers a loopback connect is in use. We only
    # probe a concrete port (port 0 = ephemeral, always free) and only treat an
    # ACTIVE listener as a conflict — a TIME_WAIT socket from our own recent
    # restart doesn't answer connect(), so a normal reboot still rebinds. The
    # skill's _start() turns the resulting OSError into an honest spoken
    # "port in use" instead of a silent half-broken server.
    if int(port) > 0 and _port_actively_served(bind, int(port)):
        raise OSError(
            f"web interface port {port} on {bind} is already being served by "
            f"another process (a stale JARVIS or a leaked test server) — refusing "
            f"to co-bind it. Free the port (stop the other process) and retry."
        )
    httpd = ThreadingHTTPServer((bind, int(port)), _Handler)
    # Pin per-server config onto the instance so the stateless handler reads it.
    # user_settings_path resolves to the live data/user_settings.json when the
    # caller didn't override it — done HERE (not as a def-time default) so the
    # JARVIS_SETTINGS_PATH redirect is honoured at server-construction time.
    httpd.config = {  # type: ignore[attr-defined]
        "token": (token or "").strip(),
        "bind": bind,          # for the anti-CSRF/rebinding Host+Origin allowlist
        "local_bind": local,
        "inject_path": inject_path,
        "log_dir": log_dir,
        "hud_state_path": hud_state_path,
        "user_settings_path": user_settings_path or _default_user_settings_path(),
        # Full-control-panel sources: the live camera preview frame and the
        # machine-generated action inventory. Injectable so a test points them at
        # a temp path (mirroring inject_path/log_dir/hud_state_path).
        "camera_preview_path": camera_preview_path or DEFAULT_CAMERA_PREVIEW_PATH,
        "action_index_path": action_index_path or DEFAULT_ACTION_INDEX_PATH,
        "reply_reader": reply_reader,
    }
    return httpd


def serve_in_thread(httpd: ThreadingHTTPServer) -> threading.Thread:
    """Run ``httpd.serve_forever()`` on a daemon thread and return it. The caller
    stops it with ``httpd.shutdown()`` (which unblocks serve_forever) then
    ``httpd.server_close()``. Daemon so a hung request never blocks JARVIS exit."""
    t = threading.Thread(target=httpd.serve_forever, name="web-interface",
                         daemon=True)
    t.start()
    return t
