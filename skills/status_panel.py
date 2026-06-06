"""
Status panel skill — JARVIS "suit diagnostics" multi-line readout.

Distinct from skills/system_pulse which is a single-sentence aggregator
with a proactive abnormality alerter. status_panel is the deliberate,
manually-invoked "give me everything" report — the verbal equivalent of
Tony asking JARVIS to bring up the suit telemetry HUD.

Actions registered:
  status_panel       — primary name
  system_status      — common verbal phrasing ("JARVIS, system status")
  suit_diagnostics   — flavour alias

Data pulled (each item degrades gracefully if its source is unavailable):
  • CPU + RAM            via psutil
  • GPU utilization      via nvidia-smi
  • Network latency      via `ping 1.1.1.1` (single packet, 1500 ms budget)
  • Claude credit balance from credits_state.json (skipped if >24h stale)
  • Bambu print %         from sibling skill_bambu_monitor (sys.modules)
  • Primary focused app  via pygetwindow.getActiveWindow().title
  • Apple Music track    via iTunes COM — but ONLY if iTunes is already
                          running (we never auto-launch from status_panel —
                          the readout should be cheap and side-effect-free)

HUD widget:
  Every STATUS_PANEL_HUD_REFRESH_SECONDS (default 20 s) the focused-window
  short name and the ping-to-Cloudflare latency are written into
  hud_state.json under the key `status_panel_strip`. Non-redundant with
  system_pulse's `pulse_strip` (GPU/BAT/UP/APPS/NET) — this line surfaces
  the foreground context and round-trip latency the rings don't show.
"""
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.status_panel` or run directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402


def _show_card_safe() -> None:
    """Pop the transient status card. Imported lazily so the skill keeps
    working if hud_card.py is missing or fails to import."""
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
    try:
        hud_card = importlib.import_module("hud_card")
        hud_card.show_card("status")
    except Exception as e:
        print(f"  [status-panel] hud_card.show_card failed: {e}")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is a guaranteed dep (dev + CI); import never fails
    _HAS_PSUTIL = False

try:
    import pygetwindow as gw   # type: ignore
    _HAS_GW = True
except Exception:
    _HAS_GW = False


# ─── cadences / thresholds ───────────────────────────────────────────────
STATUS_PANEL_HUD_REFRESH_SECONDS = 20
PING_TIMEOUT_MS                  = 1500
GPU_UTIL_BUSY_PCT                = 95.0
CREDITS_LOW_DOLLARS              = 5.0
CREDITS_STATE_MAX_AGE_SECONDS    = 24 * 3600

# Foreground titles to ignore when picking the "primary focused app" —
# these are system surfaces that aren't useful as status context.
_IGNORE_FOREGROUND = {
    "Program Manager", "Default IME", "MSCTFIME UI", "",
    "Windows Input Experience", "Task Switching",
}

# Strip common app suffixes off window titles so the spoken readout reads
# clean ("Visual Studio Code" instead of "bobert_companion.py - jarvis - Visual Studio Code").
_APP_SUFFIX_HINTS = [
    "Google Chrome", "Microsoft Edge", "Mozilla Firefox", "Visual Studio Code",
    "Windows PowerShell", "PowerShell", "Command Prompt", "Notepad++",
    "Notepad", "Microsoft Word", "Microsoft Excel", "Microsoft PowerPoint",
    "Slack", "Microsoft Teams", "Discord", "Spotify", "Apple Music",
    "Bambu Studio", "Autodesk Fusion 360", "Fusion 360", "OrcaSlicer",
    "Blender", "SolidWorks", "FreeCAD", "OpenSCAD", "Tinkercad",
    "File Explorer", "Settings",
]


_SPEECH_QUEUE   = os.path.join(_PROJECT_DIR, "pending_speech.json")
_HUD_STATE_FILE = os.path.join(_PROJECT_DIR, "hud_state.json")
_CREDITS_STATE  = os.path.join(_PROJECT_DIR, "credits_state.json")

_speech_lock = threading.Lock()

# Latest full readout cached by the HUD loop so the voice action can answer
# instantly instead of re-running nvidia-smi + ping (2-5 s) on the main voice
# thread. {"text": str, "ts": float}; None until the loop has run once.
_readout_cache_lock = threading.Lock()
_readout_cache: dict | None = None
# A status readout is fine up to this age; the loop refreshes every
# STATUS_PANEL_HUD_REFRESH_SECONDS (20 s) so the cache is at most that stale
# plus one build, but we bound it explicitly in case the loop stalls.
READOUT_CACHE_MAX_AGE_SECONDS = 25.0


# ─── speech queue (same atomic pattern as the other skills) ──────────────

def _enqueue_speech(message: str) -> None:
    """Append a spoken alert to pending_speech.json for the main loop.

    Routes through bobert_companion.proactive_announce() so this skill shares
    one write path with every other pending_speech.json co-writer
    (bambu_monitor, night_owl_mode, screen_watch, …) and they don't race each
    other. Falls back to a local atomic write only when the parent module
    isn't loaded yet (import-time registration / unit tests) or the announcer
    call fails — so a broken parent import can't silence a status readout."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="status_panel")
            return
    except Exception:
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            print(f"  [status-panel] speech-queue write failed ({e}); status: {message}")


# ─── HUD strip publishing (merge so we don't clobber pulse_strip) ────────

def _resolve_services():
    """Return a ``core.services.JarvisServices`` for this skill, or ``None``.

    M2 Phase 1 reference migration. The loader now injects a typed
    ``services`` facade alongside the legacy ``skill_utils`` dict
    (bobert_companion.load_skills). We prefer that object; if only the dict was
    injected (e.g. an older monolith, or the isolated test harness that pins
    ``skill_utils={...}``), we wrap it on the fly via ``from_skill_utils`` so the
    call site is identical either way. Returns ``None`` only if neither was
    injected (referencing the globals raises ``NameError``) — caller no-ops,
    matching the old "HUD is best-effort" degradation exactly.
    """
    svc = globals().get("services")
    if svc is not None:
        return svc
    try:
        utils = skill_utils  # type: ignore[name-defined]
    except NameError:
        return None
    if not isinstance(utils, dict):
        return None
    try:
        from core.services import JarvisServices
    except Exception:
        return None
    return JarvisServices.from_skill_utils(utils)


def _publish_hud_strip(strip: str) -> None:
    """Merge `status_panel_strip` into HUD state via the canonical writer.
    Both this skill and `system_pulse` publish into hud_state.json — going
    through bobert_companion's _write_hud_state (now reached via the typed
    `services.write_hud_state`) means we share the same _hud_state_lock + cache,
    so neither side's strip gets clobbered."""
    svc = _resolve_services()
    if svc is None:
        return
    try:
        svc.write_hud_state(status_panel_strip=strip,
                            status_panel_updated_at=time.time())
    except Exception:
        pass


# ─── metric collectors ───────────────────────────────────────────────────

def _read_cpu_ram() -> tuple[float, float]:
    if not _HAS_PSUTIL:
        return 0.0, 0.0
    try:
        cpu = psutil.cpu_percent(interval=0.4)
        ram = psutil.virtual_memory().percent
        return cpu, ram
    except Exception:
        return 0.0, 0.0


def _read_gpu_util_pct() -> float | None:
    """Best-effort GPU utilization percent (0-100), via nvidia-smi.

    Utilization is what the diagnostics readout surfaces now (temps are reliably
    fine on this rig). Only nvidia-smi exposes a load percentage — psutil sensor
    blocks report *temperature*, not utilization, so they're not consulted here.
    Returns the busiest GPU's load, or None when nvidia-smi is unavailable."""
    try:
        exe = shutil.which("nvidia-smi")
        if exe:
            out = subprocess.run(
                [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
                creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            )
            lines = (out.stdout or "").strip().splitlines()
            utils = [int(v.strip()) for v in lines if v.strip().isdigit()]
            if utils:
                return float(max(utils))
    except Exception:
        pass
    return None


_PING_TIME_RE = re.compile(r"time[=<](\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)


def _read_ping_ms(host: str = "1.1.1.1") -> float | None:
    """Single ping to `host`. Returns round-trip time in ms or None on failure.

    Uses Windows `ping -n 1 -w <ms>` so the call has a hard upper bound and
    no console window flashes."""
    try:
        if sys.platform == "win32":
            cmd = ["ping", "-n", "1", "-w", str(PING_TIMEOUT_MS), host]
            cflags = subprocess.CREATE_NO_WINDOW
        else:
            # POSIX fallback for completeness; -W timeout is in seconds.
            cmd = ["ping", "-c", "1", "-W", str(max(1, PING_TIMEOUT_MS // 1000)), host]
            cflags = 0
        out = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=(PING_TIMEOUT_MS / 1000.0) + 1.0,
            creationflags=cflags,
        )
        m = _PING_TIME_RE.search(out.stdout or "")
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _read_credit_balance() -> float | None:
    """Last-known Anthropic credit balance from credits_state.json.
    Skipped silently if the file is missing or the entry is >24h stale."""
    if not os.path.exists(_CREDITS_STATE):
        return None
    try:
        with open(_CREDITS_STATE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return None
    bal = data.get("balance")
    ts  = data.get("checked_at", 0.0)
    if bal is None:
        return None
    if (time.time() - ts) > CREDITS_STATE_MAX_AGE_SECONDS:
        return None
    try:
        return float(bal)
    except (TypeError, ValueError):
        return None


def _read_bambu_percent() -> tuple[int, str] | None:
    """Return (percent, gcode_state) if Bambu monitor knows about a print,
    else None. Doesn't import the skill — looks it up across sys.modules so
    a missing/failed skill simply yields no Bambu line in the readout."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return None
    try:
        with getattr(mod, "_state_lock"):
            state = dict(getattr(mod, "_state"))
    except Exception:
        return None
    if not state or state.get("last_update", 0.0) == 0.0:
        return None
    gcode = (state.get("gcode_state") or "").upper()
    try:
        pct = int(state.get("mc_percent") or 0)
    except Exception:
        pct = 0
    return pct, gcode


def _shorten_foreground_title(title: str) -> str:
    """Pick the most-recognisable app name out of a noisy window title.
    e.g. 'bobert_companion.py - jarvis - Visual Studio Code' → 'Visual Studio Code'."""
    if not title:
        return ""
    t = title.strip()
    # If the title ends with a known app marker, prefer that.
    for hint in _APP_SUFFIX_HINTS:
        if t.lower().endswith(hint.lower()):
            return hint
        if f" - {hint.lower()}" in t.lower() or f" — {hint.lower()}" in t.lower():
            return hint
    # Otherwise return the right-most segment after " - " (or the whole title
    # if no separator), capped at 40 chars so the readout stays tight.
    parts = re.split(r"\s+[-—]\s+", t)
    candidate = parts[-1] if parts else t
    return candidate[:40]


def _read_foreground_app() -> str | None:
    """Title of the currently-focused window, shortened to an app name.
    Returns None on a desktop-only/system-only focus."""
    if not _HAS_GW:
        return None
    try:
        win = gw.getActiveWindow()
        if win is None:
            return None
        title = (getattr(win, "title", "") or "").strip()
        if title in _IGNORE_FOREGROUND:
            return None
        return _shorten_foreground_title(title) or None
    except Exception:
        return None


def _read_apple_music_track() -> str | None:
    """Currently-playing iTunes/Apple-Music track in the form
    "'Earth Song' by Michael Jackson", or None if iTunes isn't running or
    nothing is playing. NEVER launches iTunes — this is a passive read."""
    if sys.platform != "win32":
        return None
    if not _HAS_PSUTIL:
        # Without psutil we can't tell whether iTunes is already running, and
        # we refuse to auto-launch it from a status read — bail.
        return None
    try:
        running = any(
            (p.info.get("name") or "").lower() == "itunes.exe"
            for p in psutil.process_iter(["name"])
        )
    except Exception:
        return None
    if not running:
        return None
    try:
        import win32com.client  # type: ignore
        # GetActiveObject only binds to an already-running COM server, so it
        # cannot spawn iTunes.exe — unlike Dispatch, which would launch
        # iTunes via its COM server if the psutil check happened to race
        # against an iTunes shutdown. Belt-and-braces: we already verified
        # iTunes is running above, but use the launch-incapable API anyway.
        app = win32com.client.GetActiveObject("iTunes.Application")
        t = app.CurrentTrack
        if t is None:
            return None
        name   = (t.Name   or "").strip()
        artist = (t.Artist or "").strip()
        if not name:
            return None
        if artist:
            return f"'{name}' by {artist}"
        return f"'{name}'"
    except Exception:
        return None


# ─── readout formatter ───────────────────────────────────────────────────

def _gpu_phrase(gpu_pct: float | None) -> str:
    if gpu_pct is None:
        return "GPU telemetry unavailable, sir."
    verb = "working hard at" if gpu_pct >= GPU_UTIL_BUSY_PCT else "loafing at"
    # Spec called out the "reactor — I mean, GPU" easter-egg phrasing.
    # Fire it occasionally (~1 in 5) so it lands as a quirk, not a tic.
    if random.random() < 0.20:
        return f"reactor — I mean, GPU — {verb} {gpu_pct:.0f} percent."
    return f"GPU {verb} {gpu_pct:.0f} percent."


def _build_readout() -> str:
    cpu, ram      = _read_cpu_ram()
    gpu_pct       = _read_gpu_util_pct()
    ping_ms       = _read_ping_ms()
    credits       = _read_credit_balance()
    bambu         = _read_bambu_percent()
    foreground    = _read_foreground_app()
    music_track   = _read_apple_music_track()

    # Opener — nominal unless any reading is in clearly-concerning territory.
    concerning = (
        (cpu and cpu >= 90)
        or (ram and ram >= 90)
        or (gpu_pct is not None and gpu_pct >= GPU_UTIL_BUSY_PCT)
        or (credits is not None and credits < CREDITS_LOW_DOLLARS)
        or (bambu is not None and bambu[1] == "FAILED")
    )
    opener = "Slight problem, sir." if concerning else "All systems nominal, sir."

    lines: list[str] = [opener]

    # CPU + RAM are always reported — this is the headline.
    lines.append(f"CPU at {cpu:.0f} percent, RAM at {ram:.0f} percent.")

    # GPU
    lines.append(_gpu_phrase(gpu_pct))

    # Network — graceful skip if ping failed entirely
    if ping_ms is not None:
        lines.append(f"Network response time {ping_ms:.0f} milliseconds to Cloudflare.")
    else:
        lines.append("Network response time unavailable, sir.")

    # Claude credits
    if credits is not None:
        if credits < CREDITS_LOW_DOLLARS:
            lines.append(f"You have only ${credits:.2f} left in Claude credits — a top-up may be wise.")
        else:
            lines.append(f"You have ${credits:.2f} in Claude credits.")

    # Bambu printer
    if bambu is not None:
        pct, gcode = bambu
        if gcode == "RUNNING" and 0 < pct < 100:
            lines.append(f"Bambu printer at {pct} percent through the current print.")
        elif gcode == "FINISH":
            lines.append("Bambu printer finished its last print.")
        elif gcode == "FAILED":
            lines.append("I'm afraid the Bambu print has failed.")
        # else: idle — don't add a line, keeps readout tight

    # Foreground app
    if foreground:
        lines.append(f"Foreground is {foreground}.")

    # Apple Music
    if music_track:
        lines.append(f"Apple Music playing {music_track}.")

    # Closing
    lines.append("Shall I continue?")

    return " ".join(lines)


# ─── HUD strip formatter ─────────────────────────────────────────────────

def _build_hud_strip() -> str:
    """Compact one-line view for the HUD — focused app + ping latency.
    These are the two pieces system_pulse's `pulse_strip` doesn't show."""
    bits: list[str] = []
    fg = _read_foreground_app()
    if fg:
        # Truncate aggressively — the HUD line is narrow.
        bits.append(f"WIN {fg[:22]}")
    ping_ms = _read_ping_ms()
    if ping_ms is not None:
        bits.append(f"PING {ping_ms:.0f}ms")
    return "  ·  ".join(bits)


# ─── background threads ──────────────────────────────────────────────────

def _hud_publish_loop() -> None:
    while True:
        try:
            strip = _build_hud_strip()
            if strip:
                _publish_hud_strip(strip)
        except Exception as e:
            print(f"  [status-panel] hud publish error: {e}")
        # Refresh the cached full readout in the same cadence so the voice
        # action can return it without re-running nvidia-smi + ping on the
        # main voice thread. Built here (off the voice path) where a 2-5 s
        # collect is harmless.
        try:
            text = _build_readout()
            if text:
                global _readout_cache
                with _readout_cache_lock:
                    _readout_cache = {"text": text, "ts": time.time()}
        except Exception as e:
            print(f"  [status-panel] readout cache error: {e}")
        time.sleep(STATUS_PANEL_HUD_REFRESH_SECONDS)


# ─── action registration ─────────────────────────────────────────────────

def register(actions):
    def status_panel(_: str = "") -> str:
        try:
            text = None
            with _readout_cache_lock:
                cached = _readout_cache
            if (cached
                    and (time.time() - cached.get("ts", 0.0))
                    <= READOUT_CACHE_MAX_AGE_SECONDS):
                # Serve the loop's recent readout instead of re-running
                # nvidia-smi + ping (2-5 s) on the main voice thread.
                text = cached.get("text")
            if not text:
                # Cache empty/stale (first call before the loop has run) —
                # fall back to a live compute.
                text = _build_readout()
            _show_card_safe()
            return text
        except Exception as e:
            return f"status panel failed: {e}"

    actions["status_panel"]     = status_panel
    # Common verbal phrasings the LLM may emit
    actions["system_status"]    = status_panel
    actions["suit_diagnostics"] = status_panel

    # HUD widget thread (no-op if psutil/pygetwindow are missing — strip will
    # come out empty and we just don't publish). Guard against duplicate loops
    # on skill reload (load_skills re-execs the module → fresh globals, so only
    # an OS-thread name check survives).
    if not any(t.name == "status-panel-hud" and t.is_alive()
               for t in threading.enumerate()):
        threading.Thread(target=_hud_publish_loop, daemon=True,
                         name="status-panel-hud").start()
    print(
        f"  [status-panel] HUD strip refresh every {STATUS_PANEL_HUD_REFRESH_SECONDS}s; "
        f"actions: status_panel / system_status / suit_diagnostics"
    )
