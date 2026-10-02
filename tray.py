#!/usr/bin/env python3
"""
JARVIS system-tray applet — arc-reactor icon + 4 status pips + grouped menu.

Spawned as a subprocess by bobert_companion.py at startup (mirrors the
hud / reticle launcher pattern). Reads hud_state.json sibling to
bobert_companion.py to drive icon state.

Icon layout (redesigned for legibility at Windows' 16/24 px rasterisations —
the old 4-corner-pip design collapsed to indistinct ~4 px dots when the shell
downscaled the 64 px canvas):

  • Base       — assets/jarvis_icon.png (cyan arc-reactor). Matches the HUD's
                 visual identity. Falls back to a procedural reactor disc if
                 the asset is missing.
  • PRIMARY signal = full-icon TINT. The whole reactor is recoloured by the
                 listening state so it reads at any size where tiny pips don't:
                   green = awake · gray = standby/sleep · RED = muted.
  • Speaking   — a bold blue HALO/ring pulses around the reactor while JARVIS
                 is talking (a large glow survives downscaling).
  • Queue      — when the overnight-upgrade queue is non-empty, a LARGE
                 high-contrast badge (dark disc + bright digit) sits in the
                 bottom-right corner so a single digit is readable at 24 px.
                 It is dropped gracefully (too small to read) at 16 px.
  • Bambu H2D  — a small but bold orange print-mark in the top-right corner
                 only while a print is running (secondary signal, not a pip).

Right-click menu — read-only status lines, then the two things people reach
for (Dashboard = the LEFT-CLICK default, Settings…), the four listen/voice
toggles, the grouped submenus, and the lifecycle verbs last. Toggle items show
a checkmark when active; items with no meaning right now (e.g. "Run Upgrade
Now" while overnight upgrades are switched off) are greyed out. Destructive
one-clicks (Reset Memory, Forget Last Hour, the paid Claude switch, Restart,
Shut Down, Quit Tray Only) need a SECOND click within CONFIRM_WINDOW_S — no
modal popup; the item's label turns into "⚠ Click again: …" meanwhile.

    ● <status lines>          (disabled MenuItems — same info as the tooltip)
    ─────
    Open Dashboard           (default / left-click; status balloon if it's off)
    Settings…                (opens tools/settings_window, first tab)
    ─────
    Pause Listening  Mute Mic  Mute TTS  Ambient Mode        [✓ = on]
    ─────
    Audio ▶ / Apple Music ▶ / AI ▶ / Memory ▶ / Diagnostics ▶ / Power tools ▶
    ─────
    Open HUD / Show Today's Summary / Queue Task… / About JARVIS
    ─────
    Restart JARVIS / Shut Down JARVIS / Quit Tray Only

Every command whose effect is not a checkmark (stats, tests, backups, memory
exports, model switches …) sends a request id; the monolith answers in
tray_results.json and the tray shows the answer as a balloon (short) or opens
it as a text file under logs\\tray_results\\ (long). The tray's own prints and
errors go to logs\\tray.log; a failed Settings launch goes to
logs\\settings_window.log and raises a balloon.

IPC contract (bobert_companion.py's drainer depends on this):
  • READS  hud_state.json    (state, tts_amplitude, mic_muted, bambu_active,
                              sleep_mode/standby_mode, ambient_listening, …)
  • READS  jarvis_todo.md    (count of unchecked '- [ ]' lines → queue badge,
                              shown only while overnight upgrades are on)
  • READS  logs/             (today's session_*.log files → summary dialog)
  • READS  tray_results.json — {"results": [{rid, cmd, text, final, ts}, …]}
                                written atomically by the monolith.
  • WRITES tray_commands.json — JSON list of {cmd, ts, cid[, rid], …} pending
                                commands. bobert_companion.py drains and
                                dispatches (cid de-duplicates a re-appended
                                entry; rid asks for a tray_results answer).

CLI:
  python tray.py --parent-pid 12345 [--icon-path PATH]
"""
import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import types

# CREATE_NO_WINDOW safety net — the tray runs as pythonw; any console helper
# it spawns without a flag pops a visible ghost window (2026-07-10).
try:
    from core.no_window_subprocess import install as _install_no_window
    _install_no_window()
except Exception:
    pass

try:
    import pystray
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
except Exception as e:  # pragma: no cover - import-time hard-dep guard; tests inject a fake pystray so the real import never fails here
    print(f"[tray] missing dependency: {e}")
    print("[tray] install with:  pip install pystray pillow")
    sys.exit(1)

# tkinter is stdlib but can be absent on stripped-down Python builds
try:
    import tkinter as tk
    from tkinter import simpledialog
    _HAS_TK = True
except Exception:  # pragma: no cover - import-time optional-dep guard (tkinter absent on stripped Python); not reachable once the module has imported
    _HAS_TK = False

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - import-time optional-dep guard (psutil absent); not reachable once the module has imported
    _HAS_PSUTIL = False

PROJECT_DIR        = os.path.dirname(os.path.abspath(__file__))
HUD_STATE_FILE     = os.path.join(PROJECT_DIR, "hud_state.json")
TRAY_COMMANDS_FILE = os.path.join(PROJECT_DIR, "tray_commands.json")
TODO_FILE          = os.path.join(PROJECT_DIR, "jarvis_todo.md")
LOGS_DIR           = os.path.join(PROJECT_DIR, "logs")
HUD_SCRIPT         = os.path.join(PROJECT_DIR, "hud", "jarvis_hud.py")
ASSETS_DIR         = os.path.join(PROJECT_DIR, "assets")
DEFAULT_ICON_PATH  = os.path.join(ASSETS_DIR, "jarvis_icon.png")
DATA_DIR           = os.path.join(PROJECT_DIR, "data")
CHANGELOG_FILE     = os.path.join(PROJECT_DIR, "CHANGELOG.md")
# Two DIFFERENT version sources — keeping them straight is what stops the
# About dialog from disagreeing with GitHub:
#   • RELEASE_VERSION_FILE — the top-level VERSION file: the shareable release
#     string that also backs core/version.py, the git tag and the GitHub
#     release. This is the PRIMARY "Version:" line the user sees.
#   • VERSION_FILE (data/version.json) — the self-upgrade pipeline's INTERNAL
#     counter, bumped a patch every overnight run (e.g. 1.0.17). Shown only as
#     a clearly-labelled "Upgrade build" so it can't be mistaken for the
#     release version.
RELEASE_VERSION_FILE = os.path.join(PROJECT_DIR, "VERSION")
VERSION_FILE       = os.path.join(DATA_DIR, "version.json")
INSTANCES_FILE     = os.path.join(DATA_DIR, "instances.json")
PIPELINE_LOCK_FILE = os.path.join(PROJECT_DIR, "pipeline_lock.json")
OVERNIGHT_FLAG     = os.path.join(PROJECT_DIR, ".overnight_active")
MEMORY_FACTS_FILE  = os.path.join(DATA_DIR, "long_term_memory", "facts.json")
SETTINGS_WINDOW    = os.path.join(PROJECT_DIR, "tools", "settings_window.py")
# Launched as a MODULE (python -m tools.settings_window) with cwd=PROJECT_DIR so
# the window's lazy `from core import …` resolves. Launching the script path put
# tools\ (not the project root) on sys.path[0], and every core import inside the
# window failed silently (2026-09-30 audit).
SETTINGS_MODULE    = "tools.settings_window"
SHOW_LOG_PS1       = os.path.join(PROJECT_DIR, "_show_log.ps1")
# Answers to tray requests (see _send_request / _poll_results). The monolith
# writes it atomically; the tray only reads it.
TRAY_RESULTS_FILE  = os.path.join(PROJECT_DIR, "tray_results.json")
# Where the tray's own stdout/stderr go. The tray runs windowless (pythonw or a
# CREATE_NO_WINDOW console), so without this every print and traceback in this
# file vanished — failures were invisible (2026-09-30 audit).
TRAY_LOG_FILE      = os.path.join(LOGS_DIR, "tray.log")
SETTINGS_LOG_FILE  = os.path.join(LOGS_DIR, "settings_window.log")
TRAY_RESULTS_DIR   = os.path.join(LOGS_DIR, "tray_results")
CRASH_TRACES_LOG   = os.path.join(LOGS_DIR, "crash_traces.log")
# The release notes live on GitHub (CHANGELOG.md is the self-upgrade pipeline's
# own log and stopped at the last overnight run). Same owner/repo override as
# core/update_checker.py and core/bug_reporter.py.
RELEASES_URL_FMT   = "https://github.com/{owner}/{repo}/releases"
# Apple Music in the owner's browser, not the Store app. Override per machine
# with JARVIS_APPLE_MUSIC_URL.
APPLE_MUSIC_WEB_URL = "https://music.apple.com/"
LOCAL_LLM_TAGS_URL = "http://127.0.0.1:11434/api/tags"

TICK_SECONDS = 0.20   # 5 Hz animation tick — fast enough for the speaking
                      # dot pulse, slow enough to avoid hammering the Windows
                      # shell with icon updates.
SIZE = 64             # tray-icon canvas (Windows scales 16/24/32/40/48 from this)

# Menu freshness. pystray (win32) builds the native menu ONCE and rebuilds it
# only right after a click — BEFORE the monolith's 2 Hz drainer has applied the
# toggle — so every checkmark ran one click behind (2026-09-30 audit, proven on
# the installed backend). The animation loop now rebuilds whenever what the
# menu would show changes, at most this often.
MENU_REFRESH_MIN_S = 0.5
# Second-click confirmation window for destructive one-click items.
CONFIRM_WINDOW_S   = 10.0
# How often the results file is polled, and how long a request waits for its
# answer before it is forgotten.
RESULTS_POLL_S     = 0.5
RESULT_WAIT_S      = 15 * 60.0
# Balloon limits: Windows caps szInfo at 256 and szInfoTitle at 64 WCHARs
# (ctypes raises on overflow). Longer answers open as a text file instead.
NOTIFY_MAX_CHARS   = 250
NOTIFY_TITLE_MAX   = 63
NOTIFY_MAX_LINES   = 3
# A Settings window that exits non-zero within this long failed to open.
SETTINGS_LAUNCH_GRACE_S = 20.0
# The tray log is trimmed (rolled to tray.log.1) when it passes this size.
TRAY_LOG_MAX_BYTES = 1_000_000
# Now-playing header: a cached value refreshed off-thread, never read inline.
NOW_PLAYING_REFRESH_S = 5.0
NOW_PLAYING_TIMEOUT_S = 2.0
# Installed local models for the picker, refreshed off-thread.
LOCAL_MODELS_REFRESH_S = 60.0
LOCAL_MODELS_TIMEOUT_S = 2.0

# ── Signal palette ────────────────────────────────────────────────────────
# The listen colour is now the PRIMARY signal: the whole reactor is tinted
# toward it (see _tint_image), so it reads at any rasterisation size. The
# speaking colour drives a pulsing halo; the queue colour the badge disc;
# the bambu colour a small corner print-mark.
LISTEN_GREEN = (60, 210, 90)      # awake
LISTEN_GRAY  = (140, 140, 150)    # standby / sleep
LISTEN_RED   = (220, 40, 40)      # muted

SPEAK_BLUE   = (60, 150, 255)
SPEAK_DIM    = (25, 45, 85)       # very dim base when not speaking

QUEUE_YELLOW = (235, 200, 30)
QUEUE_DIM    = (55, 50, 18)       # dim when queue is empty

BAMBU_ORANGE = (235, 130, 30)
BAMBU_WHITE  = (220, 220, 220)    # idle

# Tint strength — how strongly the listen colour recolours the reactor.
# Awake/standby get a gentle wash so the arc-reactor identity survives;
# muted is pushed harder so "RED = muted" is unmistakable even at 16 px.
TINT_STRENGTH       = 0.45
TINT_STRENGTH_MUTED = 0.62
# Badge geometry as a fraction of the canvas — deliberately large so a
# single digit survives Windows' downscale to 24 px.
BADGE_FRAC = 0.46

# Backwards-compat — older code paths still reference COLORS["idle"] etc.
# Keeps the module import-safe if anything outside this file pokes at the
# table; the new renderer doesn't read from it.
COLORS = {
    "idle":      ((48, 100, 180),  (90, 180, 255)),
    "listening": ((180, 180, 200), (255, 255, 255)),
    "thinking":  ((180, 120, 0),   (255, 200, 60)),
    "speaking":  ((220, 170, 30),  (255, 220, 80)),
    "standby":   ((60, 40, 100),   (155, 140, 255)),
    "alert":     ((200, 0, 0),     (255, 80, 80)),
    "bambu":     ((220, 110, 0),   (255, 170, 60)),
}


# ─── Base icon (arc-reactor PNG) ─────────────────────────────────────────
# Loaded once at startup, resized to SIZE×SIZE, then re-used as the
# background layer for every animation frame. If the asset is missing or
# the file is corrupt, _base_icon stays None and _render_icon falls back
# to the procedural 4-dot grid — the tray must never crash because of a
# missing icon file (parent watchdog regression risk).
_base_icon: "Image.Image | None" = None
_icon_path: str = DEFAULT_ICON_PATH


def _load_base_icon(path: str) -> None:
    """Try to load the JARVIS arc-reactor PNG. Sets _base_icon on success;
    leaves it None on any error so the renderer falls through to the
    procedural fallback."""
    global _base_icon
    _base_icon = None
    if not path or not os.path.exists(path):
        print(f"[tray] icon asset not found at {path} — using procedural fallback")
        return
    try:
        img = Image.open(path).convert("RGBA")
        # Resize to the tray canvas — LANCZOS keeps the arc-reactor ring crisp
        # even when Windows downscales further for 16/24/32 px rasterisations.
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        _base_icon = img
        print(f"[tray] base icon loaded from {path}")
    except Exception as e:
        print(f"[tray] failed to load icon {path} ({e}) — using fallback")
        _base_icon = None


# While the animation loop evaluates the menu (to decide whether it changed and
# to rebuild it) every menu lambda reads the ONE hud_state snapshot that tick
# already parsed, instead of re-reading the file ~40 times. Thread-local, so the
# pystray message thread (which rebuilds after a click) still reads the file.
_hud_snapshot = threading.local()


def _read_hud_state() -> dict:
    """Best-effort read of hud_state.json. Returns empty dict on any error."""
    snap = getattr(_hud_snapshot, "data", None)
    if snap is not None:
        return snap
    try:
        with open(HUD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# One lock around the read-modify-write of tray_commands.json. Menu callbacks run
# on the pystray thread AND on worker threads (dialogs, confirmations); two
# unlocked writers could both read the same list and the later os.replace
# silently dropped the earlier click.
_send_lock = threading.Lock()
_cid_counter = [0]


def _next_id(prefix: str) -> str:
    """Process-unique id: '<prefix><pid>-<n>'. Used for command ids (cid, the
    monolith's de-duplication key) and request ids (rid, the results key)."""
    with _send_lock:
        _cid_counter[0] += 1
        n = _cid_counter[0]
    return f"{prefix}{os.getpid()}-{n}"


def _send_command(cmd: str, **kwargs) -> None:
    """Append a command to tray_commands.json using the same atomic
    temp+rename pattern the other JSON inboxes (pending_speech.json etc.)
    use. Bobert drains this on a 0.5s background timer.

    Every command carries a unique ``cid``. The drainer claims the inbox with
    an os.replace, so a claim landing between our read and our replace makes
    us write the already-claimed commands BACK — the drainer skips any cid it
    has already dispatched instead of running them twice."""
    payload = {"cmd": cmd, "ts": time.time(), "cid": _next_id("c")}
    payload.update(kwargs)
    try:
        with _send_lock:
            existing = []
            if os.path.exists(TRAY_COMMANDS_FILE):
                try:
                    with open(TRAY_COMMANDS_FILE, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if raw:
                        decoded, _ = json.JSONDecoder().raw_decode(raw)
                        if isinstance(decoded, list):
                            existing = decoded
                except Exception:
                    existing = []
            existing.append(payload)
            fd, tmp = tempfile.mkstemp(dir=PROJECT_DIR, suffix=".tmp",
                                       prefix="tray_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(existing, f)
                os.replace(tmp, TRAY_COMMANDS_FILE)
            except Exception:
                try: os.remove(tmp)
                except Exception: pass
                raise
        print(f"[tray] sent command: {cmd}")
    except Exception as e:
        print(f"[tray] command write failed ({cmd}): {e}")


# ─── Tray log (stdout/stderr → logs\tray.log) ───────────────────────────────

class _Tee:
    """File-like that writes to the tray log AND the original stream (when there
    is one — pythonw has none). Never raises: logging must not kill the tray."""

    def __init__(self, log_file, original):
        self._log = log_file
        self._orig = original

    def write(self, s):
        for target in (self._log, self._orig):
            if target is None:
                continue
            try:
                target.write(s)
            except Exception:
                pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        for target in (self._log, self._orig):
            if target is None:
                continue
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self):
        return False


_tray_log_handle = [None]


def _setup_tray_logging(path: str = "") -> bool:
    """Route the tray's prints, logging and uncaught tracebacks to
    logs\\tray.log. The tray is windowless, so before this every '[tray] …'
    line and every exception went nowhere. Rolls the log to tray.log.1 once it
    passes TRAY_LOG_MAX_BYTES. Returns True when the log is attached."""
    path = path or TRAY_LOG_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) > TRAY_LOG_MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except Exception:
        return False
    _tray_log_handle[0] = fh
    sys.stdout = _Tee(fh, sys.stdout)
    sys.stderr = _Tee(fh, sys.stderr)
    try:
        root = logging.getLogger()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
    except Exception:
        pass

    def _thread_hook(args):
        try:
            logging.error("[tray] uncaught exception in thread %s",
                          getattr(args.thread, "name", "?"),
                          exc_info=(args.exc_type, args.exc_value,
                                    args.exc_traceback))
        except Exception:
            pass
    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass
    print(f"[tray] ---- log opened {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(pid {os.getpid()}) ----")
    return True


# ─── Balloon notifications ──────────────────────────────────────────────────

_icon_ref: list = [None]      # the live pystray.Icon, set by main()


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _notify(message: str, title: str = "JARVIS") -> bool:
    """Show a tray balloon (Windows 'notification area' toast) and log it.
    Only ever called as the answer to something the owner just clicked — never
    as an unsolicited alert. Returns True when the balloon was handed to the
    shell. Never raises."""
    message = _clip(str(message or "").strip() or "(no details)",
                    NOTIFY_MAX_CHARS)
    title = _clip(title or "JARVIS", NOTIFY_TITLE_MAX)
    print(f"[tray] notify: {title}: {message}")
    icon = _icon_ref[0]
    if icon is None:
        return False
    try:
        if getattr(icon, "HAS_NOTIFICATION", True) is False:
            return False
        icon.notify(message, title)
        return True
    except Exception as e:
        print(f"[tray] notify failed: {e}")
        return False


# ─── Request / result round-trip (tray_results.json) ────────────────────────
#
# A request is a command whose answer the owner should SEE (a stat, a test, a
# backup). _send_request tags it with a rid; the monolith echoes the rid into
# tray_results.json (interim "started" lines have final=False); _poll_results
# (animation loop) shows each answer once. Only rids this tray sent are shown,
# so a voice-triggered action or a previous session's answers never pop up.

_pending: dict = {}           # rid -> {"cmd", "label", "sent_at"}
_pending_lock = threading.Lock()
_results_state = {"checked_at": 0.0, "mtime": None, "read_at": 0.0,
                  "shown": set()}


def _send_request(cmd: str, label: str = "", **kwargs) -> str:
    """Send ``cmd`` and register it for a visible answer. Returns the rid."""
    rid = _next_id("r")
    with _pending_lock:
        _pending[rid] = {"cmd": cmd, "label": label or cmd,
                         "sent_at": time.time()}
    _send_command(cmd, rid=rid, **kwargs)
    return rid


def _result_file_path(cmd: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in str(cmd or "result"))[:60] or "result"
    return os.path.join(TRAY_RESULTS_DIR, f"{safe}.txt")


def _show_result(label: str, cmd: str, text: str) -> None:
    """A short answer becomes a balloon; a long / multi-line one is written to
    logs\\tray_results\\<cmd>.txt and opened, with a one-line balloon."""
    text = str(text or "").strip() or "Done."
    lines = text.splitlines()
    if len(text) <= NOTIFY_MAX_CHARS and len(lines) <= NOTIFY_MAX_LINES:
        _notify(text, f"JARVIS — {label}")
        return
    path = _result_file_path(cmd)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{label} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(text + "\n")
    except Exception as e:
        print(f"[tray] could not write {path}: {e}")
        _notify(lines[0] if lines else text, f"JARVIS — {label}")
        return
    _open_path(path, os.path.basename(path))
    first = lines[0] if lines else ""
    _notify(f"{_clip(first, 160)}\n(full output opened: "
            f"logs\\tray_results\\{os.path.basename(path)})",
            f"JARVIS — {label}")


def _poll_results(now: float | None = None) -> int:
    """Show every new answer to one of OUR pending requests. Cheap: re-reads
    tray_results.json only when its mtime moved, at most every RESULTS_POLL_S.
    Returns how many answers were shown. Never raises."""
    now = time.time() if now is None else now
    if (now - _results_state["checked_at"]) < RESULTS_POLL_S:
        return 0
    _results_state["checked_at"] = now
    with _pending_lock:
        for rid in [r for r, p in _pending.items()
                    if (now - p.get("sent_at", now)) > RESULT_WAIT_S]:
            _pending.pop(rid, None)
        if not _pending:
            return 0
    try:
        st = os.stat(TRAY_RESULTS_FILE)
    except OSError:
        return 0
    # (mtime, size) — Windows file times can tick as coarsely as ~16 ms, so
    # two quick writes (an interim line then the final) may share an mtime;
    # a pending request also forces a re-read every 2 s regardless.
    sig = (st.st_mtime_ns, st.st_size)
    if (sig == _results_state["mtime"]
            and (now - _results_state.get("read_at", 0.0)) < 2.0):
        return 0
    try:
        with open(TRAY_RESULTS_FILE, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return 0          # mid-write or corrupt: try again next poll
    _results_state["mtime"] = sig
    _results_state["read_at"] = now
    entries = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        return 0
    shown = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("rid") or "")
        key = (rid, str(entry.get("seq", "")), bool(entry.get("final", True)))
        if key in _results_state["shown"]:
            continue
        with _pending_lock:
            req = _pending.get(rid)
            if req is None:
                continue
            _results_state["shown"].add(key)
            if entry.get("final", True):
                _pending.pop(rid, None)
        _show_result(req["label"], req["cmd"], entry.get("text", ""))
        shown += 1
    if len(_results_state["shown"]) > 512:
        _results_state["shown"] = set(list(_results_state["shown"])[-256:])
    return shown


# ─── Second-click confirmation (no modal popups) ────────────────────────────

_confirm_armed: dict = {}     # key -> expiry (epoch s)
_confirm_lock = threading.Lock()


def _confirm_text(key: str, label: str) -> str:
    """Menu label for a confirm-gated item: '⚠ Click again: <label>' while the
    first click is armed, the plain label otherwise."""
    with _confirm_lock:
        exp = _confirm_armed.get(key, 0.0)
    return f"⚠ Click again: {label}" if exp > time.time() else label


def _confirmed(key: str, label: str, hint: str = "") -> bool:
    """True on the SECOND click within CONFIRM_WINDOW_S (and disarms). The first
    click arms the gate, relabels the item, and says so in a balloon."""
    now = time.time()
    with _confirm_lock:
        exp = _confirm_armed.get(key, 0.0)
        if exp > now:
            _confirm_armed.pop(key, None)
            return True
        _confirm_armed[key] = now + CONFIRM_WINDOW_S
    _notify(f"Click “{label}” again within {int(CONFIRM_WINDOW_S)} s to "
            f"confirm.{(' ' + hint) if hint else ''}", "JARVIS — confirm")
    return False


# ─── Values published by the running JARVIS (hud_state) ─────────────────────

def _jarvis_ready(data: dict | None = None) -> bool:
    """True once the parent JARVIS has started its tray command drainer (it
    publishes tray_ready_pid). Until then the tray says 'starting…' — the tray
    is launched early in boot so it is there while JARVIS loads."""
    data = _read_hud_state() if data is None else data
    ready = data.get("tray_ready_pid")
    if not ready:
        return False
    parent = _parent_pid[0]
    try:
        return (not parent) or int(ready) == int(parent)
    except Exception:
        return False


def _upgrades_enabled(data: dict | None = None) -> bool:
    """Overnight self-upgrades switched on? Published by the monolith as
    hud_state.overnight_upgrade_enabled. Absent (an older JARVIS) = off, the
    safe reading: 'Run Upgrade Now' greys out instead of sleeping JARVIS."""
    data = _read_hud_state() if data is None else data
    return bool(data.get("overnight_upgrade_enabled"))


def _dashboard_url(data: dict | None = None) -> str:
    """The web dashboard's local URL when the web interface is running, else ''."""
    data = _read_hud_state() if data is None else data
    try:
        port = int(data.get("web_port") or 0)
    except Exception:
        port = 0
    return f"http://127.0.0.1:{port}/" if port > 0 else ""


# ─── Now-playing header (cached, refreshed off-thread with a timeout) ───────

_np_cache = {"text": "", "at": 0.0, "busy": False}
_np_lock = threading.Lock()


def _now_playing_lookup() -> str:
    """One blocking now-playing read (SMTC first, then the Apple Music window
    bridge). Runs on a worker thread only — see _now_playing_label."""
    try:
        from core.media_now_playing import now_playing_text as _smtc_np
        _np = _smtc_np()
        if _np:
            return f"♪ {_np}"
    except Exception:
        pass
    amapp = _apple_music_app()
    if amapp is None:
        return "Apple Music: unavailable"
    try:
        running = amapp.is_running()
    except Exception:
        running = False
    if not running:
        return "Apple Music: closed"
    try:
        np = amapp.now_playing()
    except Exception:
        np = None
    if np:
        np = np if len(np) <= 60 else np[:57].rstrip() + "…"
        return f"Apple Music: {np}"
    return "Apple Music: idle"


def _refresh_now_playing(timeout: float = NOW_PLAYING_TIMEOUT_S) -> str:
    """Run one lookup on a daemon thread, waiting at most ``timeout``. A hung
    WinRT / window query can no longer stall the menu: the cache keeps its old
    value and the stuck worker is abandoned (it is a daemon)."""
    box: dict = {}

    def _work():
        try:
            box["text"] = _now_playing_lookup()
        except Exception:
            box["text"] = "Apple Music: unavailable"
    t = threading.Thread(target=_work, name="tray-now-playing", daemon=True)
    t.start()
    t.join(timeout)
    with _np_lock:
        if "text" in box:
            _np_cache["text"] = box["text"]
        _np_cache["at"] = time.time()
        _np_cache["busy"] = False
        return _np_cache["text"] or "♪ …"


def _now_playing_label() -> str:
    """The cached header text; kicks a background refresh when stale. Never
    blocks the caller (menu rebuilds run on the tray's UI thread)."""
    with _np_lock:
        text = _np_cache["text"]
        stale = (time.time() - _np_cache["at"]) > NOW_PLAYING_REFRESH_S
        start = stale and not _np_cache["busy"]
        if start:
            _np_cache["busy"] = True
    if start:
        threading.Thread(target=_refresh_now_playing, name="tray-np-refresh",
                         daemon=True).start()
    return text or "♪ …"


# ─── Installed local models (for the model picker) ──────────────────────────

_models_cache = {"tags": [], "at": 0.0, "busy": False}
_models_lock = threading.Lock()


def _fetch_local_models(timeout: float = LOCAL_MODELS_TIMEOUT_S) -> list:
    """Installed Ollama chat-model tags, sorted. [] when Ollama is down."""
    try:
        import urllib.request
        with urllib.request.urlopen(LOCAL_LLM_TAGS_URL, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return []
    tags = []
    for m in (doc.get("models") or []) if isinstance(doc, dict) else []:
        name = str((m or {}).get("name") or (m or {}).get("model") or "")
        if name and "embed" not in name.lower():
            tags.append(name)
    return sorted(set(tags))


def _refresh_local_models() -> list:
    tags = _fetch_local_models()
    with _models_lock:
        if tags or not _models_cache["tags"]:
            _models_cache["tags"] = tags
        _models_cache["at"] = time.time()
        _models_cache["busy"] = False
        return list(_models_cache["tags"])


def _local_models() -> list:
    """Cached tag list; refreshes off-thread when stale. Never blocks."""
    with _models_lock:
        tags = list(_models_cache["tags"])
        stale = (time.time() - _models_cache["at"]) > LOCAL_MODELS_REFRESH_S
        start = stale and not _models_cache["busy"]
        if start:
            _models_cache["busy"] = True
    if start:
        threading.Thread(target=_refresh_local_models,
                         name="tray-models-refresh", daemon=True).start()
    return tags


# ─── Icon rendering ──────────────────────────────────────────────────────

def _blend(dark: tuple, bright: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(dark[i] + (bright[i] - dark[i]) * t) for i in range(3))


# Cached fonts keyed by requested size — Pillow's truetype loader is
# surprisingly hot when re-invoked at 5 Hz, so we keep one font per size.
_FONT_CACHE: dict[int, "ImageFont.ImageFont"] = {}


def _get_font(size: int):
    """Pillow font for the queue-count digit. Falls back to bitmap default."""
    font = _FONT_CACHE.get(size)
    if font is not None:
        return font
    try:
        from PIL import ImageFont as _IF
        # Arial ships with Windows; if it isn't found Pillow's loader walks
        # through fallbacks before we land on the bitmap default.
        for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "segoeui.ttf"):
            try:
                font = _IF.truetype(name, size)
                break
            except Exception:
                continue
        if font is None:
            font = _IF.load_default()
    except Exception:
        font = None
    if font is not None:
        _FONT_CACHE[size] = font
    return font


def _compute_signal_colors(state: str, frame: int, tts_amplitude: float,
                           queue_count: int, muted: bool,
                           bambu_active: bool) -> dict:
    """Resolve every status signal the renderers need into one dict.

    Shared by the arc-reactor renderer and the procedural fallback so both
    show identical colours for a given state. Besides the four base colours
    it also resolves the derived values the redesign keys off:

      • ``listen``       — the tint colour (PRIMARY signal); the whole
                           reactor is recoloured toward it.
      • ``tint_strength``— how hard to push the tint (muted pushes harder so
                           red is unmistakable at 16 px).
      • ``speak``        — pulsing halo colour; ``speak_t`` is the 0..1 pulse
                           level (0 when quiet) so the halo can fade in/out.
      • ``queue``/``queue_count`` — badge disc colour + the integer to draw.
      • ``bambu``/``bambu_active`` — corner print-mark colour + whether to
                           draw it at all.
    """
    raw = str(state or "").lower()

    if muted:
        listen_rgb = LISTEN_RED
        tint_strength = TINT_STRENGTH_MUTED
    elif raw in ("standby", "sleeping", "sleep"):
        listen_rgb = LISTEN_GRAY
        tint_strength = TINT_STRENGTH
    else:
        listen_rgb = LISTEN_GREEN
        tint_strength = TINT_STRENGTH

    is_speaking = (raw == "speaking") or ((tts_amplitude or 0.0) > 0.02)
    if is_speaking:
        # Pulse between 0.55 and 1.0 brightness at ~1.25 Hz (period 4 frames
        # @ 5 Hz tick); ride the TTS amplitude envelope when published.
        t = 0.55 + 0.45 * (math.sin(frame * 2 * math.pi / 4) + 1) / 2
        t = max(t, min(1.0, 0.55 + (tts_amplitude or 0.0) * 0.5))
        speak_rgb = _blend(SPEAK_DIM, SPEAK_BLUE, t)
        speak_t = t
    else:
        speak_rgb = SPEAK_DIM
        speak_t = 0.0

    count = max(0, int(queue_count or 0))
    queue_rgb = QUEUE_YELLOW if count > 0 else QUEUE_DIM

    bambu_rgb = BAMBU_ORANGE if bambu_active else BAMBU_WHITE

    return {
        "listen": listen_rgb,
        "tint_strength": tint_strength,
        "speak":  speak_rgb,
        "speak_t": speak_t,
        "queue":  queue_rgb,
        "bambu":  bambu_rgb,
        "queue_count": count,
        "bambu_active": bambu_active,
    }


def _tint_image(base: "Image.Image", rgb: tuple, strength: float) -> "Image.Image":
    """Recolour ``base`` toward ``rgb`` while preserving its shape + shading.

    The arc-reactor's own luminance is kept (so the ring/disc detail and the
    transparent surround survive); only the hue is washed toward the signal
    colour. This is the PRIMARY status channel — a full-icon tint stays
    legible at 16 px where the old corner pips dissolved into ~4 px mush.
    Falls back to returning a copy on any error so the renderer never raises.
    """
    try:
        strength = max(0.0, min(1.0, strength))
        src = base if base.mode == "RGBA" else base.convert("RGBA")
        r, g, b, a = src.split()
        # Per-pixel luminance of the original drives the brightness of the
        # tinted result, so highlights stay bright and shadows stay dark.
        lum = src.convert("L")
        tinted_rgb = Image.new("RGB", src.size, rgb)
        # Multiply the flat tint by the luminance ramp -> shaded tint.
        shaded = ImageChops.multiply(
            tinted_rgb, Image.merge("RGB", (lum, lum, lum)))
        orig_rgb = Image.merge("RGB", (r, g, b))
        mixed = Image.blend(orig_rgb, shaded, strength)
        mr, mg, mb = mixed.split()
        return Image.merge("RGBA", (mr, mg, mb, a))
    except Exception:
        return base.copy()


def _draw_speaking_halo(img: "Image.Image", speak_t: float, rgb: tuple) -> None:
    """Draw a soft pulsing ring just inside the canvas edge while speaking.

    A large halo (not a tiny dot) is the point — it reads as "JARVIS is
    talking" even after Windows squashes the icon to 16 px. ``speak_t`` is
    the 0..1 pulse level; at 0 we draw nothing. Mutates ``img`` in place.
    """
    if speak_t <= 0.0:
        return
    try:
        alpha = int(90 + 150 * max(0.0, min(1.0, speak_t)))   # 90..240
        ring = Image.new("RGBA", img.size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        w = max(2, int(SIZE * 0.09))            # bold stroke
        inset = max(1, int(SIZE * 0.04))
        rd.ellipse([inset, inset, SIZE - 1 - inset, SIZE - 1 - inset],
                   outline=rgb + (alpha,), width=w)
        # Blur so the ring reads as a glow rather than a hard circle, and so
        # it survives downscaling without aliasing into a dotted line.
        ring = ring.filter(ImageFilter.GaussianBlur(max(1, int(SIZE * 0.03))))
        img.alpha_composite(ring)
    except Exception:
        # A halo is pure polish — never let it break the icon.
        pass


def _draw_queue_badge(img: "Image.Image", count: int, queue_rgb: tuple) -> None:
    """Draw a LARGE bottom-right count badge (dark disc + bright digit).

    Sized at ``BADGE_FRAC`` of the canvas with a near-opaque dark disc behind
    a high-contrast digit so a single character is still readable once the
    shell downscales to 24 px (it simply becomes too small to resolve at
    16 px — an acceptable, graceful degradation). No-op when count <= 0.
    Mutates ``img`` in place.
    """
    if count <= 0:
        return
    try:
        d = ImageDraw.Draw(img)
        bd = max(12, int(SIZE * BADGE_FRAC))
        x = SIZE - bd
        y = SIZE - bd
        # Dark disc with a bright rim in the queue colour -> pops off any base.
        d.ellipse([x, y, x + bd, y + bd], fill=(15, 15, 18, 235),
                  outline=queue_rgb + (255,), width=max(2, int(bd * 0.10)))
        text = str(count) if count < 100 else "99+"
        # One digit gets a big glyph; "99+" needs to be smaller to fit.
        frac = 0.66 if len(text) <= 1 else (0.5 if len(text) == 2 else 0.4)
        font = _get_font(max(8, int(bd * frac)))
        if font is not None:
            try:
                bbox = d.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x + (bd - tw) / 2 - bbox[0]
                ty = y + (bd - th) / 2 - bbox[1]
                d.text((tx, ty), text, fill=queue_rgb + (255,), font=font)
            except Exception:
                pass
    except Exception:
        pass


def _draw_bambu_mark(img: "Image.Image") -> None:
    """Draw a small bold orange print-mark in the TOP-RIGHT corner.

    Only called when a Bambu print is active, so its mere presence is the
    signal (secondary to the listen tint). Kept compact but solid + rimmed
    so it doesn't vanish at small sizes. Mutates ``img`` in place.
    """
    try:
        d = ImageDraw.Draw(img)
        m = max(8, int(SIZE * 0.30))
        x1 = SIZE - m
        y0 = 0
        # Down-pointing triangle (a nozzle laying a line) — distinct from the
        # round queue badge so the two corners never read as the same thing.
        d.polygon([(x1, y0), (SIZE - 1, y0), ((x1 + SIZE - 1) / 2, m)],
                  fill=BAMBU_ORANGE + (255,), outline=(20, 20, 20, 255))
    except Exception:
        pass


ALERT_RED = (235, 45, 45)


def _draw_alert_mark(img: "Image.Image") -> None:
    """A bold red ring around the whole icon while a system alert (sustained
    CPU/RAM pressure) is active — hud_state.alert_active. A ring, not a dot, so
    it survives the 16 px downscale. Mutates ``img`` in place."""
    try:
        d = ImageDraw.Draw(img)
        w = max(3, int(SIZE * 0.08))
        d.ellipse([1, 1, SIZE - 2, SIZE - 2], outline=ALERT_RED + (255,),
                  width=w)
    except Exception:
        pass


def _draw_tts_muted_mark(img: "Image.Image") -> None:
    """A small dark disc with a red slash in the BOTTOM-LEFT corner while
    JARVIS's voice is muted (Mute TTS) — distinct from the full red tint, which
    means the MICROPHONE is muted. Mutates ``img`` in place."""
    try:
        d = ImageDraw.Draw(img)
        m = max(10, int(SIZE * 0.36))
        y = SIZE - m
        d.ellipse([0, y, m, SIZE - 1], fill=(15, 15, 18, 235),
                  outline=(200, 200, 200, 255), width=max(1, int(m * 0.08)))
        pad = int(m * 0.22)
        d.line([(pad, SIZE - 1 - pad), (m - pad, y + pad)],
               fill=ALERT_RED + (255,), width=max(2, int(m * 0.16)))
    except Exception:
        pass


def _draw_status_marks(img: "Image.Image", signals: dict) -> None:
    if signals.get("alert"):
        _draw_alert_mark(img)
    if signals.get("tts_muted"):
        _draw_tts_muted_mark(img)


def _render_icon_with_base(base: "Image.Image", signals: dict) -> Image.Image:
    """Render the arc-reactor icon with the redesigned status overlays.

    Pipeline: tint the whole reactor by the listen state (primary signal) →
    pulse a speaking halo → stamp the large queue badge (if any) → mark a
    Bambu print (if active) → alert ring / voice-muted mark (if any).
    """
    img = _tint_image(base, signals["listen"], signals["tint_strength"])
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    _draw_speaking_halo(img, signals["speak_t"], signals["speak"])
    if signals["bambu_active"]:
        _draw_bambu_mark(img)
    _draw_queue_badge(img, signals["queue_count"], signals["queue"])
    _draw_status_marks(img, signals)
    return img


def _render_reactor_disc(rgb: tuple) -> "Image.Image":
    """Procedural stand-in for the arc-reactor PNG, recoloured to ``rgb``.

    Used when the asset can't be loaded. Mirrors the real design: a glowing
    tinted disc (so the listen-state tint still reads) instead of the legacy
    4-dot grid, keeping the fallback visually consistent with the base path.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = SIZE / 2
    outer = SIZE * 0.46
    # Outer dark ring -> bright tinted ring -> dim core -> bright centre,
    # echoing the reactor's concentric look so the fallback isn't jarring.
    d.ellipse([c - outer, c - outer, c + outer, c + outer],
              fill=(18, 22, 28, 255))
    r2 = SIZE * 0.40
    d.ellipse([c - r2, c - r2, c + r2, c + r2],
              outline=rgb + (255,), width=max(2, int(SIZE * 0.07)))
    r3 = SIZE * 0.24
    d.ellipse([c - r3, c - r3, c + r3, c + r3],
              fill=_blend((10, 12, 16), rgb, 0.35) + (255,))
    r4 = SIZE * 0.12
    d.ellipse([c - r4, c - r4, c + r4, c + r4],
              fill=_blend(rgb, (255, 255, 255), 0.4) + (255,))
    return img


def _render_icon_procedural(signals: dict) -> Image.Image:
    """Fallback renderer used when the arc-reactor PNG can't be loaded.

    Builds a procedural reactor disc tinted by the listen state, then runs
    the SAME overlay stack as the base path (halo / badge / bambu mark) so
    status reads identically whether or not the asset is present.
    """
    img = _render_reactor_disc(signals["listen"])
    _draw_speaking_halo(img, signals["speak_t"], signals["speak"])
    if signals["bambu_active"]:
        _draw_bambu_mark(img)
    _draw_queue_badge(img, signals["queue_count"], signals["queue"])
    _draw_status_marks(img, signals)
    return img


def _render_icon(state: str, frame: int, mic_level: float = 0.0,
                 tts_amplitude: float = 0.0, queue_count: int = 0,
                 muted: bool = False, bambu_active: bool = False,
                 alert: bool = False, tts_muted: bool = False) -> Image.Image:
    """Render one tray-icon frame.

    Two modes — picks based on whether the arc-reactor base asset loaded:

      • Base loaded    — tint the arc-reactor PNG by listen state and overlay
                         the speaking halo + queue badge + bambu mark.
      • Base missing   — render a procedural tinted reactor disc and run the
                         same overlay stack, so status still reads with no
                         asset present.

    Returns an RGBA PIL Image suitable for assignment to pystray.Icon.icon.
    Never raises — bad inputs degrade to the fallback rather than crash
    the animation loop (parent watchdog regression risk).
    """
    try:
        signals = _compute_signal_colors(
            state, frame, tts_amplitude, queue_count, muted, bambu_active,
        )
        signals["alert"] = bool(alert)
        signals["tts_muted"] = bool(tts_muted)
    except Exception:
        # Worst-case: synthesise neutral signals so we still render something.
        signals = {
            "listen": LISTEN_GRAY, "tint_strength": TINT_STRENGTH,
            "speak": SPEAK_DIM, "speak_t": 0.0,
            "queue":  QUEUE_DIM,   "bambu": BAMBU_WHITE,
            "queue_count": 0, "bambu_active": False,
        }

    if _base_icon is not None:
        try:
            return _render_icon_with_base(_base_icon, signals)
        except Exception:
            logging.exception("[tray] base-icon composite failed — falling back")
    try:
        return _render_icon_procedural(signals)
    except Exception:
        # Absolute last resort: a flat tinted square so icon assignment never
        # receives a non-image. Keeps the animation loop alive no matter what.
        logging.exception("[tray] procedural render failed — flat fallback")
        return Image.new("RGBA", (SIZE, SIZE),
                         tuple(signals.get("listen", LISTEN_GRAY)) + (255,))


# ─── Parent-process watchdog ─────────────────────────────────────────────

_parent_pid = [0]
_stop_event = threading.Event()


def _parent_alive() -> bool:
    pid = _parent_pid[0]
    if not pid:
        return True
    # AUTHORITATIVE check first (2026-07-12): psutil.pid_exists reads TRUE
    # for a DEAD-but-unreaped Windows process — this tray outlived its
    # terminated parent by 25 minutes (duplicate tray icon). See
    # core.parent_watch (WaitForSingleObject, signaled on termination).
    try:
        from core.parent_watch import parent_is_alive
        return parent_is_alive(pid)
    except Exception:
        pass
    if _HAS_PSUTIL:
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _standby_from(data: dict) -> bool:
    """Is JARVIS in standby/sleep? The monolith's published sleep_mode /
    standby_mode flags are the truth. The transient ``state`` label is only a
    fallback for an older JARVIS that never publishes the flags: in standby the
    main loop keeps setting state='idle', so reading the label made "Pause
    Listening" show unchecked while JARVIS was in fact paused (2026-09-30)."""
    if "sleep_mode" in data or "standby_mode" in data:
        return bool(data.get("sleep_mode") or data.get("standby_mode"))
    raw = str(data.get("state") or "").lower()
    return raw in ("standby", "sleeping", "sleep")


def _classify_state(state_data: dict) -> dict:
    """Map hud_state.json into the icon renderer's inputs.

    Returns a dict with: state, mic_level, tts_amplitude, muted,
    bambu_active, alert, tts_muted. ``state`` is forced to "standby" when the
    published standby/sleep flags say so. The animator combines this with the
    cached queue_count (read from jarvis_todo.md on a slower cadence).
    """
    state = str(state_data.get("state") or "").lower()
    if _standby_from(state_data) and state != "speaking":
        state = "standby"
    return {
        "state":         state,
        "mic_level":     float(state_data.get("mic_level") or 0.0),
        "tts_amplitude": float(state_data.get("tts_amplitude") or 0.0),
        "muted":         bool(state_data.get("mic_muted")
                              or state_data.get("muted")),
        "bambu_active":  bool(state_data.get("bambu_active")),
        "alert":         bool(state_data.get("alert_active")),
        "tts_muted":     bool(state_data.get("tts_muted")),
    }


# Queue count is recomputed on a slower cadence (every ~2s) so a chatty
# editor saving jarvis_todo.md mid-write doesn't make the icon flicker.
_QUEUE_RECHECK_SECONDS = 2.0
_queue_cache = {"count": 0, "at": 0.0}


def _count_pending_tasks() -> int:
    """Count unchecked '- [ ]' lines in jarvis_todo.md. Cheap (<5ms on a
    160-line file) but we still cache for 2 seconds to keep the animation
    loop allocation-free in steady state."""
    now = time.time()
    if (now - _queue_cache["at"]) < _QUEUE_RECHECK_SECONDS:
        return int(_queue_cache["count"])
    count = 0
    try:
        if os.path.exists(TODO_FILE):
            with open(TODO_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("- [ ]"):
                        count += 1
    except Exception:
        count = int(_queue_cache["count"])   # keep last good value on read error
    _queue_cache["count"] = count
    _queue_cache["at"]    = now
    return count


# ─── Menu callbacks ──────────────────────────────────────────────────────

def _on_open_hud(icon, item):
    _send_command("open_hud")

def _on_open_logs(icon, item):
    """Power tools → Open Logs Folder (session logs, tray.log,
    settings_window.log, tray_results\\)."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        os.startfile(LOGS_DIR)   # Windows-only; tray spec is Windows anyway
    except Exception as e:
        print(f"[tray] open logs failed: {e}")

def _on_restart(icon, item):
    if _confirmed("restart", "Restart JARVIS"):
        _send_command("restart")


def _on_dashboard(icon, item):
    """Default (left-click) action: open the web dashboard when the web
    interface is running, otherwise show a one-line status balloon."""
    data = _read_hud_state()
    url = _dashboard_url(data)
    if url:
        try:
            import webbrowser
            webbrowser.open(url)
            return
        except Exception as e:
            print(f"[tray] open dashboard failed: {e}")
    _notify(_status_summary(data) + " · Dashboard is off — say "
            "'start the web interface'.", "JARVIS")


def _status_summary(data: dict | None = None) -> str:
    """Plain-English one-liner of what the icon shows (tooltip + balloon)."""
    data = _read_hud_state() if data is None else data
    if not _jarvis_ready(data):
        return "Starting…"
    s = _classify_state(data)
    if s["muted"]:
        listen = "Mic muted"
    elif s["state"] == "standby":
        listen = "Paused (say 'JARVIS')"
    else:
        listen = "Listening"
    bits = [listen]
    if s["state"] == "speaking" or s["tts_amplitude"] > 0.02:
        bits.append("speaking")
    if s["tts_muted"]:
        bits.append("voice muted")
    if s["alert"]:
        bits.append("SYSTEM ALERT")
    if s["bambu_active"]:
        bits.append("printing")
    if _upgrades_enabled(data):
        n = _count_pending_tasks()
        if n:
            bits.append(f"{n} queued")
    return " · ".join(bits)

def _append_queued_task(text: str) -> None:
    """Append a `- [ ]` entry to jarvis_todo.md (mirrors _act_queue_task)."""
    text = (text or "").strip()
    if not text:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M")
        entry = f"- [ ] **{ts}** [tray] — {text}\n"
        if not os.path.exists(TODO_FILE):
            with open(TODO_FILE, "w", encoding="utf-8") as f:
                f.write(
                    "# JARVIS Task Queue\n\n"
                    "Things the user wants Claude Code to build, fix, "
                    "or investigate later.\nTick items as you complete "
                    "them; archive when the file gets big.\n\n"
                )
        with open(TODO_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"[tray] queued: {text[:80]}")
    except Exception as e:
        print(f"[tray] queue task failed: {e}")


# ─── Spawned-dialog lifecycle tracking ──────────────────────────────────────
# Modal dialogs (queue-task, dossier, about, summary) run in short-lived Python
# subprocesses. Before v2.0.23 only the tray icon was stopped on shutdown, so a
# dialog left open orphaned its subprocess — it outlived JARVIS. Track every
# live dialog Popen here and reap them on quit / parent-death. 2026-07-08.
_dialog_procs: "list[subprocess.Popen]" = []
_dialog_procs_lock = threading.Lock()


def _terminate_dialog_procs() -> None:
    """Terminate every still-open spawned dialog subprocess so a modal dialog
    left on screen can't outlive JARVIS. Called on tray quit and when the parent
    JARVIS process is first seen gone. Never raises. 2026-07-08."""
    with _dialog_procs_lock:
        procs = list(_dialog_procs)
        _dialog_procs.clear()
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _tracked_dialog_run(args, *, capture_output=False, text=False,
                        timeout=None, creationflags=0):
    """``subprocess.run`` work-alike that registers the child in
    ``_dialog_procs`` for the life of the call, so a shutdown mid-dialog can
    reap the orphan. Returns an object exposing ``.stdout`` / ``.returncode``
    like ``subprocess.run``. On timeout the child is killed (mirrors run's
    contract) before the TimeoutExpired propagates. 2026-07-08."""
    pipe = subprocess.PIPE if capture_output else None
    # Spawn AND register atomically under the lock (2026-07-14 bug-hunt #26).
    # The old order (Popen, THEN take the lock to append) left a TOCTOU window:
    # a concurrent _terminate_dialog_procs — which snapshots+clears the list
    # under this same lock — could run between spawn and register and never see
    # the new child, orphaning a modal dialog past shutdown. Holding the lock
    # across Popen makes the reaper either miss the spawn entirely or see the
    # registered child; there is no in-between.
    with _dialog_procs_lock:
        proc = subprocess.Popen(args, stdout=pipe, stderr=pipe, text=text,
                                creationflags=creationflags)
        _dialog_procs.append(proc)
    try:
        out, _err = proc.communicate(timeout=timeout)
    except Exception:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        raise
    finally:
        with _dialog_procs_lock:
            try:
                _dialog_procs.remove(proc)
            except ValueError:
                pass
    return types.SimpleNamespace(stdout=out, returncode=proc.returncode)


def _run_queue_task_dialog() -> int:
    """Subprocess entry point: run the tkinter input dialog on THIS
    process's main thread, then print the entered text to stdout.

    Spawned by `_on_queue_task` so the GUI never touches a daemon thread
    in the tray process (tkinter on Windows requires main-thread Tcl)."""
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    # Don't promise an overnight upgrade that is switched off: the task list
    # (jarvis_todo.md) is then simply a to-do list for Claude Code.
    prompt = ("Describe the task to queue for the next overnight upgrade:"
              if _upgrades_enabled() else
              "Describe a task to add to JARVIS's task list (jarvis_todo.md)\n"
              "— overnight upgrades are off, so hand it to Claude Code:")
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        text = simpledialog.askstring(
            "Queue Task — JARVIS",
            prompt,
            parent=root,
        )
    finally:
        try: root.destroy()
        except Exception: pass
    if text and text.strip():
        sys.stdout.write(text.strip())
        sys.stdout.flush()
    return 0


def _on_queue_task(icon, item):
    """Show a small input dialog and append the result to jarvis_todo.md.

    The dialog runs in a separate Python subprocess so tkinter executes
    on that subprocess's main thread — calling `tk.Tk()` from a daemon
    thread on Windows can hang or crash the tray. We still wrap the
    subprocess call in a daemon thread so the pystray menu callback
    returns immediately."""

    def _spawn_and_collect():
        try:
            # CREATE_NO_WINDOW so the subprocess doesn't flash a console
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__),
                 "--queue-task-dialog"],
                capture_output=True, text=True, timeout=600,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] queue task dialog subprocess failed: {e}")
            return
        text = (proc.stdout or "").strip()
        if not text:
            return
        _append_queued_task(text)

    threading.Thread(target=_spawn_and_collect, daemon=True).start()


def _on_pause_listening(icon, item):
    """Toggle JARVIS's standby flag. Sends the inverse command based on the
    current state so a single menu entry can act as a true ✓ toggle."""
    if _is_standby():
        _send_command("force_wake")
    else:
        _send_command("enter_standby")

def _on_mute_tts(icon, item):
    """Toggle TTS mute — JARVIS still thinks/acts but stays silent."""
    _send_command("mute_tts_toggle")

def _on_mute_mic(icon, item):
    """Toggle the microphone mute. Bobert drops input while muted — including a
    capture already in progress — and mirrors the new flag back to
    hud_state.mic_muted, which also drives the icon's red listen tint."""
    _send_command("mic_mute_toggle")

def _on_ambient_mode(icon, item):
    """Toggle ambient mode (continuous-listen background mode)."""
    _send_command("ambient_mode_toggle")

def _on_force_upgrade(icon, item):
    """Spec verb 'force upgrade now' — kick the overnight engine. The item is
    greyed out while overnight upgrades are switched off; the monolith refuses
    too (it used to put JARVIS to sleep for an upgrade that never ran)."""
    if not _upgrades_enabled():
        _notify("Overnight upgrades are switched off, so there is nothing to "
                "run.", "JARVIS — Run Upgrade Now")
        return
    _send_request("trigger_overnight", "Run Upgrade Now")

def _on_shutdown_jarvis(icon, item):
    if _confirmed("shutdown_jarvis", "Shut Down JARVIS"):
        _send_command("shutdown_jarvis")

# ── Power tools submenu callbacks ────────────────────────────────────────

def _on_stop_pipeline(icon, item):
    _send_request("stop_pipeline", "Stop Running Pipeline")

def _on_force_backup(icon, item):
    _send_request("force_backup", "Force Backup")

def _on_reload_skills(icon, item):
    _send_request("reload_skills", "Reload Skills")

def _on_run_smoke_test(icon, item):
    _send_request("run_smoke_test", "Smoke Test")

def _on_pause_daemons(icon, item):
    _send_command("pause_daemons_toggle")

def _on_open_live_log(icon, item):
    threading.Thread(target=_open_live_log_viewer, daemon=True).start()

def _on_open_crashes(icon, item):
    threading.Thread(target=_open_event_viewer_crashes, daemon=True).start()

def _releases_url() -> str:
    owner = os.environ.get("JARVIS_GITHUB_OWNER", "").strip() or "Sk8ter2404"
    repo = os.environ.get("JARVIS_GITHUB_REPO", "").strip() or "jarvis"
    return RELEASES_URL_FMT.format(owner=owner, repo=repo)

def _on_open_changelog(icon, item):
    """Release notes. CHANGELOG.md is the self-upgrade pipeline's own log (it
    stopped at the last overnight run), so the real notes are the GitHub
    releases; the local file is the fallback when no browser is available."""
    try:
        import webbrowser
        if webbrowser.open(_releases_url()):
            return
    except Exception as e:
        print(f"[tray] open releases page failed: {e}")
    if os.path.exists(CHANGELOG_FILE):
        _open_path(CHANGELOG_FILE, "CHANGELOG.md")
    else:
        print("[tray] CHANGELOG.md not found")

# ── AI submenu callbacks ────────────────────────────────────────────────

def _on_switch_anthropic(icon, item):
    """Claude is the PAID cloud brain (the owner runs local-first), so the
    switch needs a second click."""
    if _confirmed("switch_anthropic", "Switch to Claude (cloud, paid)",
                  "It bills per conversation."):
        _send_request("switch_llm", "Switch to Claude", backend="anthropic")

def _on_switch_local(icon, item):
    # Send the "ollama" sentinel, NOT a hard-coded model tag (2026-07-21
    # audit): the tray's old 'qwen2.5:14b'/'llama3.1:8b' literals drifted from
    # the tags actually installed (qwen2.5:14b-instruct-q5_K_M etc.), pinning
    # the resolver cache at a tag Ollama 404s on every turn. _act_switch_llm's
    # "ollama" branch resolves the live local default via _get_local_llm_model,
    # so the tray can never name a model the box doesn't have.
    _send_request("switch_llm", "Switch to Local LLM", backend="ollama")

def _switch_to_model(tag: str):
    """Menu action for one installed local model in the picker. The tag comes
    from Ollama's own /api/tags, so it is always a model the box has."""
    def _action(icon, item):
        _send_request("switch_llm", f"Switch to {tag}", backend=tag)
    return _action

def _local_model_items():
    """The 'Local model' picker: one radio item per installed Ollama model
    (cached, refreshed off-thread), the active one checked."""
    tags = _local_models()
    if not tags:
        return (pystray.MenuItem("(no local models found — is Ollama running?)",
                                 None, enabled=False),)
    return tuple(
        pystray.MenuItem(tag, _switch_to_model(tag),
                         checked=(lambda t: (lambda i: _active_llm_backend()
                                             == t.lower()))(tag),
                         radio=True)
        for tag in tags[:25])

def _on_toggle_debug_mode(icon, item):
    _send_command("debug_mode_toggle")

def _on_show_llm_stats(icon, item):
    _send_request("show_llm_stats", "LLM Call Stats")

# ── Audio Controls submenu callbacks ────────────────────────────────────
# Each toggle flips one runtime flag in bobert (audio_master / aec / ns /
# agc). Bobert mirrors the new state back to hud_state.json and the animation
# loop rebuilds the menu, so the checkmark is right the next time it opens.
# The sub-layer toggles still apply only when the master "Audio Processing"
# toggle is on — turning the master off bypasses the processor entirely.

def _on_toggle_audio_processing(icon, item):
    _send_command("audio_processing_toggle")

def _on_toggle_echo_cancel(icon, item):
    _send_command("audio_echo_cancel_toggle")

def _on_toggle_noise_suppress(icon, item):
    _send_command("audio_noise_suppress_toggle")

def _on_toggle_agc(icon, item):
    _send_command("audio_agc_toggle")

# ── Memory submenu callbacks ────────────────────────────────────────────

def _on_open_memory_file(icon, item):
    if os.path.exists(MEMORY_FACTS_FILE):
        _open_path(MEMORY_FACTS_FILE, "facts.json")
    else:
        # Fall back to the legacy single-file location, then to the dir.
        legacy = os.path.join(PROJECT_DIR, "memory.json")
        if os.path.exists(legacy):
            _open_path(legacy, "memory.json")
        else:
            _open_path(os.path.dirname(MEMORY_FACTS_FILE), "memory dir")

def _on_show_dossier(icon, item):
    """Show JARVIS's full known-facts dossier on the user (read-only)."""
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__), "--dossier-dialog"],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] dossier dialog subprocess failed: {e}")
    threading.Thread(target=_spawn, daemon=True).start()

def _on_recent_facts(icon, item):
    _send_request("show_recent_facts", "Recent Facts (last 24h)")

def _on_reset_memory(icon, item):
    if _confirmed("reset_memory", "Reset Memory…",
                  "Everything JARVIS learned is wiped (a backup is kept)."):
        _send_request("reset_memory", "Reset Memory")

def _on_export_memory(icon, item):
    _send_request("export_memory", "Export Memory")

def _on_forget_last_hour(icon, item):
    if _confirmed("forget_last_hour", "Forget Last Hour"):
        _send_request("forget_last_hour", "Forget Last Hour")

# ── Diagnostics submenu callbacks ────────────────────────────────────────

def _on_run_diagnostic(icon, item):
    _send_request("run_diagnostic", "Diagnostic")

def _on_show_last_diagnostic(icon, item):
    _send_request("show_last_diagnostic", "Last Diagnostic Run")

def _on_test_mic(icon, item):
    _send_request("test_mic", "Test Mic")

def _on_test_tts(icon, item):
    _send_request("test_tts", "Test TTS")

def _on_test_vision(icon, item):
    _send_request("test_vision", "Test Vision")

def _on_test_each_skill(icon, item):
    _send_request("test_each_skill", "Test Each Skill")

def _on_latency_benchmark(icon, item):
    _send_request("latency_benchmark", "Latency Benchmark")

# ── Settings ─────────────────────────────────────────────────────────────

def _on_open_settings(icon, item):
    """Top-level "Settings…": open the settings window on its first tab. (It
    used to be only a submenu HEADER — clicking "Settings" did nothing.)"""
    threading.Thread(target=_open_settings_window, args=("",),
                     name="tray-settings", daemon=True).start()

# ── About dialog ────────────────────────────────────────────────────────

def _read_release_version() -> str:
    """The shareable RELEASE version — single source of truth (top-level
    VERSION file) that also backs core/version.py, the git tag and the GitHub
    release. Kept SEPARATE from the self-upgrade pipeline's CHANGELOG counter
    below so the About dialog's primary 'Version:' line always matches what
    GitHub shows. Returns 'unknown' if the file is missing (defensive only —
    VERSION is tracked, so a real checkout always has it)."""
    try:
        with open(RELEASE_VERSION_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "unknown"
    except Exception:
        return "unknown"


def _read_version_and_upgrade() -> tuple[str, str]:
    """Parse CHANGELOG.md for the latest version + timestamp.

    Header format written by the pipeline runner is:
        ## v1.0.6 — 2026-05-28 22:33
    Returns (version, last_upgrade_at). Falls back to data/version.json so
    the dialog still has something to show if CHANGELOG.md got truncated."""
    version = "unknown"
    upgrade_at = "unknown"
    try:
        if os.path.exists(CHANGELOG_FILE):
            with open(CHANGELOG_FILE, "r", encoding="utf-8",
                      errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("## v") and "—" in s:
                        # "## v1.0.6 — 2026-05-28 22:33"
                        try:
                            after_hash = s[3:].strip()       # "v1.0.6 — ..."
                            ver, _, rest = after_hash.partition("—")
                            version = ver.strip()
                            upgrade_at = rest.strip()
                        except Exception:  # pragma: no cover - defensive; str slice/partition/strip on a matched line cannot raise
                            pass
                        break
    except Exception:
        pass
    if version == "unknown":
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, "r", encoding="utf-8") as f:
                    vj = json.load(f) or {}
                version = "v" + str(vj.get("version") or "?")
                upgrade_at = str(vj.get("last_upgrade_at") or upgrade_at)
        except Exception:
            pass
    return version, upgrade_at


def _release_timestamp() -> float | None:
    """When the release on disk was made (epoch s), or None: the v<VERSION>
    tag / the commit that set VERSION, the VERSION mtime outside a checkout
    (core.version.release_timestamp, one copy for the tray and the spoken
    version answer)."""
    try:
        from core.version import release_timestamp
        return release_timestamp(PROJECT_DIR)
    except Exception:
        return None


def _last_updated_text(pipeline_at: str) -> str:
    """'YYYY-MM-DD HH:MM' of the newest of the release on disk and the
    self-upgrade pipeline's own last run, or '' when neither is known.

    The 'Last upgrade' line used to be the pipeline's CHANGELOG header alone,
    which no git release writes: it read 2026-05-30 07:03 four months and
    ~140 releases later (2026-10-02)."""
    from datetime import datetime
    best = None
    ts = _release_timestamp()
    if ts is not None:
        try:
            best = datetime.fromtimestamp(ts)
        except (OverflowError, OSError, ValueError):
            best = None
    raw = (pipeline_at or "").strip()
    if raw and raw != "unknown":
        try:
            p = datetime.fromisoformat(raw)
            if p.tzinfo is not None:
                p = p.astimezone().replace(tzinfo=None)
            if best is None or p > best:
                best = p
        except ValueError:
            if best is None:
                return raw
    return best.strftime("%Y-%m-%d %H:%M") if best is not None else ""


def _parent_started_at() -> float:
    """Start time (epoch s) of the LIVE parent JARVIS process, or 0.0. psutil
    reads it straight from the OS, so it can't be stale."""
    pid = _parent_pid[0]
    if not pid or not _HAS_PSUTIL:
        return 0.0
    try:
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return 0.0


def _read_uptime_seconds() -> float:
    """Uptime of the JARVIS this tray belongs to, or 0.0 when unknown.

    Order: the live parent process's OS start time; the boot time the running
    JARVIS published (hud_state boot_started_at, only when its jarvis_pid is our
    parent — or there is no parent to compare against); the instances.json
    entry whose pid IS our parent. The old code took the FIRST prod entry in
    data\\instances.json — a registry that keeps dead PIDs — and reported the
    uptime of a JARVIS that died weeks ago (2026-09-30 audit)."""
    now = time.time()
    started = _parent_started_at()
    if started > 0:
        return max(0.0, now - started)
    parent = _parent_pid[0]
    try:
        data = _read_hud_state()
        boot = float(data.get("boot_started_at") or 0.0)
        owner = data.get("jarvis_pid")
        same = (not parent) or (owner is not None and int(owner) == int(parent))
        if boot > 0 and same:
            return max(0.0, now - boot)
    except Exception:
        pass
    if parent:
        try:
            with open(INSTANCES_FILE, "r", encoding="utf-8") as f:
                inst = json.load(f) or {}
            for key, entry in inst.items():
                if not isinstance(entry, dict):
                    continue
                pid = entry.get("pid", key)
                if str(pid) == str(parent) and entry.get("started_at"):
                    return max(0.0, now - float(entry["started_at"]))
        except Exception:
            pass
    return 0.0


def _running_version(data: dict | None = None) -> str:
    """The version the RUNNING JARVIS booted with (hud_state.jarvis_version,
    published at boot), when it belongs to our parent; '' otherwise."""
    data = _read_hud_state() if data is None else data
    ver = str(data.get("jarvis_version") or "").strip()
    if not ver:
        return ""
    parent = _parent_pid[0]
    owner = data.get("jarvis_pid")
    try:
        if parent and owner is not None and int(owner) != int(parent):
            return ""
    except Exception:
        return ""
    return ver


def _git_commit() -> str:
    """Short commit of the tree on disk, or '' (no git, not a checkout, slow)."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        r = subprocess.run(["git", "-C", PROJECT_DIR, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=3,
                           creationflags=flags)
        out = (r.stdout or "").strip()
        return out if r.returncode == 0 and out else ""
    except Exception:
        return ""


def _format_uptime(seconds: float) -> str:
    s = int(max(0.0, seconds))
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    mins, _ = divmod(s, 60)
    if days:
        return f"{days}d {hours}h {mins}m"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _about_lines() -> list[str]:
    on_disk = _read_release_version()
    data = _read_hud_state()
    running = _running_version(data)
    # PRIMARY line = what is actually RUNNING (published at boot). The VERSION
    # file can be ahead of it after a `git pull` without a restart; say so
    # rather than claim a version the process isn't on.
    release = running or on_disk
    build, upgrade_at = _read_version_and_upgrade()
    up_s = _read_uptime_seconds()
    uptime = _format_uptime(up_s) if up_s > 0 else "unknown"
    lines = [
        "J.A.R.V.I.S.",
        "",
        f"Version:       {release}",      # matches GitHub + the git tag
    ]
    if running and on_disk not in ("unknown", running):
        lines.append(f"On disk:       {on_disk} (restart to apply)")
    commit = _git_commit()
    if commit:
        lines.append(f"Commit:        {commit}")
    # When the code on disk last changed: the release's git date or the
    # pipeline's last run, whichever is newer (_last_updated_text).
    updated = _last_updated_text(upgrade_at)
    if updated:
        lines.append(f"Last updated:  {updated}")
    # The self-upgrade pipeline's internal counter (e.g. v1.0.17) + its
    # timestamp — shown only when there's real upgrade history AND it differs
    # from the release version. A fresh clone (no pipeline runs) just shows the
    # release version, never a confusing 'Upgrade build: unknown'. Its own
    # date rides on this line, so it can't pass for the last update.
    if build and build not in ("unknown", release, f"v{release}"):
        ran = (f", last run {upgrade_at}"
               if upgrade_at and upgrade_at != "unknown" else "")
        lines.append(f"Upgrade build: {build}{ran}"
                     + ("" if _upgrades_enabled(data) else " (upgrades off)"))
    lines += [
        f"Uptime:        {uptime}",
        "",
        "Personal AI assistant.",
        "Left-click the tray icon for the dashboard, right-click for the menu.",
    ]
    return lines


def _run_about_dialog() -> int:
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    body = "\n".join(_about_lines())
    root = tk.Tk()
    try:
        root.title("About JARVIS")
        root.attributes("-topmost", True)
        root.geometry("480x320")   # room for the On disk / Commit lines
        try:
            text = tk.Text(root, wrap="word", font=("Consolas", 11),
                           bg="#0d1117", fg="#c9d1d9", padx=14, pady=12)
            text.pack(fill="both", expand=True)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0


def _on_about(icon, item):
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            # --parent-pid so the dialog process reads the uptime/version of
            # THIS tray's JARVIS (it has no other way to know which one).
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__), "--about-dialog",
                 "--parent-pid", str(_parent_pid[0] or 0)],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] about dialog subprocess failed: {e}")
    threading.Thread(target=_spawn, daemon=True).start()


# ── "Show what JARVIS knows about me" dossier dialog ────────────────────

def _dossier_lines() -> list[str]:
    """Read data/long_term_memory/facts.json and lay it out human-readably.
    This is a read-only dossier — the menu item "Reset Memory" handles
    edits via the bobert action so we don't accidentally drift schemas."""
    lines: list[str] = ["What JARVIS knows about you", ""]
    facts_path = MEMORY_FACTS_FILE
    if not os.path.exists(facts_path):
        legacy = os.path.join(PROJECT_DIR, "memory.json")
        if os.path.exists(legacy):
            facts_path = legacy
        else:
            lines.append("(no memory file found yet)")
            return lines
    try:
        with open(facts_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception as e:
        lines.append(f"(could not read memory: {e})")
        return lines

    # facts.json can be either a list of fact dicts or {"facts": [...]}.
    facts: list = []
    if isinstance(data, list):
        facts = data
    elif isinstance(data, dict):
        if isinstance(data.get("facts"), list):
            facts = data["facts"]
        else:
            # Fall through to a generic key:value dump.
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            return lines

    if not facts:
        lines.append("(no facts learned yet)")
        return lines

    lines.append(f"{len(facts)} fact(s) on file.")
    lines.append("")
    # Show the most-recent ~60; the rest would overflow the dialog.
    for entry in facts[-60:]:
        if isinstance(entry, dict):
            text = (entry.get("text")
                    or entry.get("fact")
                    or entry.get("content")
                    or json.dumps(entry))
        else:
            text = str(entry)
        trimmed = text if len(text) <= 160 else text[:157] + "…"
        lines.append(f"  • {trimmed}")
    return lines


def _run_dossier_dialog() -> int:
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    body = "\n".join(_dossier_lines())
    root = tk.Tk()
    try:
        root.title("What JARVIS Knows About Me")
        root.attributes("-topmost", True)
        root.geometry("680x520")
        try:
            frame = tk.Frame(root, bg="#0d1117")
            frame.pack(fill="both", expand=True)
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            text = tk.Text(frame, wrap="word", font=("Consolas", 10),
                           bg="#0d1117", fg="#c9d1d9", padx=10, pady=10,
                           yscrollcommand=scrollbar.set)
            text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0

def _on_open_todo(icon, item):
    """Open jarvis_todo.md in the user's default markdown editor."""
    if not os.path.exists(TODO_FILE):
        # Create a minimal file so os.startfile doesn't error on a fresh
        # install before any task has been queued.
        try:
            with open(TODO_FILE, "w", encoding="utf-8") as f:
                f.write("# JARVIS Task Queue\n\n")
        except Exception as e:
            print(f"[tray] could not create todo: {e}")
            return
    try:
        os.startfile(TODO_FILE)   # Windows-only; tray spec is Windows anyway
    except Exception as e:
        print(f"[tray] open todo failed: {e}")


def _today_summary_lines() -> list[str]:
    """Build the text shown in the 'Show Today's Summary' dialog.

    Reads today's session_YYYY-MM-DD_*.log files and surfaces:
      • session count + total log size
      • pending vs. completed task counts from jarvis_todo.md
      • last few completed tasks (today only) so the user can see what
        was actually finished in this calendar day
    """
    lines: list[str] = []
    today = time.strftime("%Y-%m-%d")
    lines.append(f"J.A.R.V.I.S. — {today}")
    lines.append("")

    # ── Session activity ──
    try:
        if os.path.isdir(LOGS_DIR):
            sessions = [
                f for f in os.listdir(LOGS_DIR)
                if f.startswith(f"session_{today}_") and f.endswith(".log")
            ]
            total_bytes = 0
            for f in sessions:
                try:
                    total_bytes += os.path.getsize(os.path.join(LOGS_DIR, f))
                except Exception:
                    pass
            kb = total_bytes / 1024
            lines.append(f"Sessions today:   {len(sessions)}  ({kb:.1f} KB logged)")
        else:
            lines.append("Sessions today:   (logs/ not found)")
    except Exception as e:
        lines.append(f"Sessions today:   (error: {e})")

    # ── Task queue health ──
    pending = 0
    completed_today = []
    try:
        if os.path.exists(TODO_FILE):
            with open(TODO_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("- [ ]"):
                        pending += 1
                    elif s.startswith("- [x]") and today in line:
                        # Strip the leading '- [x] ' so the summary reads cleanly.
                        completed_today.append(s[5:].strip())
    except Exception as e:
        lines.append(f"Todo:             (error: {e})")
    else:
        lines.append(f"Pending tasks:    {pending}")
        lines.append(f"Completed today:  {len(completed_today)}")

    if completed_today:
        lines.append("")
        lines.append("Recent completions:")
        # Show the last 5 — newest at the bottom of the file, so reverse.
        for entry in reversed(completed_today[-5:]):
            # Trim each entry to ~140 chars so the dialog doesn't span the
            # whole screen on a long ✓ DONE summary.
            trimmed = entry if len(entry) <= 140 else entry[:137] + "…"
            lines.append(f"  • {trimmed}")

    return lines


def _run_summary_dialog() -> int:
    """Subprocess entry point: show today's summary in a tkinter window.

    Same pattern as the queue-task dialog — runs in its own subprocess so
    tkinter gets the main thread. Read-only dialog with an OK button.
    """
    if not _HAS_TK:
        sys.stderr.write("tkinter not available\n")
        return 2
    lines = _today_summary_lines()
    body = "\n".join(lines) or "(no activity today)"
    root = tk.Tk()
    try:
        root.title("Today's Summary — JARVIS")
        root.attributes("-topmost", True)
        # Sized to fit ~14 lines of monospaced text; user can scroll if longer.
        root.geometry("560x360")
        try:
            text = tk.Text(root, wrap="word", font=("Consolas", 10),
                           bg="#0d1117", fg="#c9d1d9", padx=10, pady=10)
            text.pack(fill="both", expand=True)
            text.insert("1.0", body)
            text.configure(state="disabled")
        except Exception:
            tk.Label(root, text=body, justify="left").pack(padx=10, pady=10)
        tk.Button(root, text="OK", command=root.destroy,
                  width=10).pack(pady=(0, 10))
        root.mainloop()
    finally:
        try: root.destroy()
        except Exception: pass
    return 0


def _on_show_today_summary(icon, item):
    """Spawn the summary dialog in a subprocess (tkinter wants its own
    main thread, same constraint as the queue-task dialog)."""
    def _spawn():
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            _tracked_dialog_run(
                [sys.executable, os.path.abspath(__file__),
                 "--summary-dialog"],
                timeout=1800,
                creationflags=creationflags,
            )
        except Exception as e:
            print(f"[tray] summary dialog subprocess failed: {e}")

    threading.Thread(target=_spawn, daemon=True).start()


def _on_quit(icon, item):
    """Quit Tray Only — JARVIS keeps running. Needs a second click; the way
    back is the voice action 'show the tray icon' (show_tray)."""
    if not _confirmed("quit_tray", "Quit Tray Only",
                      "JARVIS keeps running; say 'JARVIS, show the tray icon' "
                      "to bring it back."):
        return
    _stop_event.set()
    _terminate_dialog_procs()   # reap any open modal dialog before we go. 2026-07-08
    try: icon.stop()
    except Exception: pass


# ─── Menu freshness ──────────────────────────────────────────────────────

def _menu_signature(menu, depth: int = 0) -> tuple:
    """What the native menu would show for ``menu`` right now — text, checked,
    enabled, radio/default flags and submenus, for every VISIBLE item — as a
    hashable tuple. Walks pystray's public API only, so it describes exactly
    what pystray itself would build. Any failure degrades to a sentinel that
    compares unequal to the last good signature (so the menu is rebuilt)."""
    if menu is None or depth > 6:
        return ()
    out = []
    try:
        for item in menu:
            if item is pystray.Menu.SEPARATOR:
                out.append("--")
                continue
            sub = getattr(item, "submenu", None)
            out.append((str(item.text), bool(item.checked), bool(item.enabled),
                        bool(getattr(item, "default", False)),
                        _menu_signature(sub, depth + 1) if sub else None))
    except Exception as e:
        return (("error", repr(e), time.time()),)
    return tuple(out)


_menu_state = {"sig": None, "at": 0.0}
_menu_open = threading.Event()


def _install_menu_open_guard(icon) -> bool:
    """While the native popup menu is on screen (pystray win32 runs it inside
    its WM_NOTIFY handler, via the modal TrackPopupMenuEx) a rebuild would
    DestroyMenu the very HMENU being displayed. Wrap that handler to raise
    _menu_open for its duration. Private-API, so feature-detected: on another
    backend (or a pystray without _message_handlers) it is a no-op."""
    handlers = getattr(icon, "_message_handlers", None)
    if not isinstance(handlers, dict):
        return False
    for code, fn in list(handlers.items()):
        if getattr(fn, "__name__", "") != "_on_notify":
            continue

        def _guarded(*args, _fn=fn, **kwargs):
            _menu_open.set()
            try:
                return _fn(*args, **kwargs)
            finally:
                _menu_open.clear()
        _guarded.__name__ = "_on_notify"
        handlers[code] = _guarded
        return True
    return False


def _refresh_menu_if_changed(icon, data: dict, now: float | None = None) -> bool:
    """Rebuild the native menu when what it would show has changed since the
    last build — so a toggle applied by the monolith's drainer (up to ~0.5 s
    after the click) shows its checkmark the next time the menu opens.
    Debounced to MENU_REFRESH_MIN_S and skipped while the menu is open.
    ``data`` is this tick's hud_state; every menu lambda reads it (see
    _hud_snapshot). Returns True when update_menu() ran."""
    now = time.time() if now is None else now
    menu = getattr(icon, "menu", None)
    if menu is None:
        return False
    _hud_snapshot.data = data
    try:
        sig = _menu_signature(menu)
        if sig == _menu_state["sig"]:
            return False
        if _menu_open.is_set():
            return False
        if (now - _menu_state["at"]) < MENU_REFRESH_MIN_S:
            return False
        try:
            icon.update_menu()
        except Exception as e:
            print(f"[tray] update_menu failed: {e}")
            return False
        _menu_state["sig"] = sig
        _menu_state["at"] = now
        return True
    finally:
        _hud_snapshot.data = None


# ─── Animation thread ────────────────────────────────────────────────────

def _tooltip(data: dict) -> str:
    """Plain-English tooltip (Windows caps it at 127 chars)."""
    return _clip(f"JARVIS — {_status_summary(data)}", 127)


_icon_state = {"key": None, "title": None}


def _animate(icon: "pystray.Icon") -> None:
    frame = 0
    while not _stop_event.is_set():
        try:
            if not _parent_alive():
                print("[tray] parent JARVIS process exited — closing tray")
                _terminate_dialog_procs()   # don't orphan an open dialog. 2026-07-08
                try: icon.stop()
                except Exception: pass
                return
            data = _read_hud_state()
            s = _classify_state(data)
            # The badge counts the overnight-upgrade queue; with upgrades off
            # jarvis_todo.md is a developer backlog and pinned it at 99+.
            queue_count = _count_pending_tasks() if _upgrades_enabled(data) else 0
            try:
                signals = _compute_signal_colors(
                    s["state"], frame, s["tts_amplitude"], queue_count,
                    s["muted"], s["bambu_active"])
                # Only push a NEW image when it would look different: a static
                # icon used to be re-rendered and re-sent to the shell 5x a
                # second (~2.5 % of a core). Pulsing (speaking) still animates
                # because speak_t changes every frame.
                key = (signals["listen"], signals["tint_strength"],
                       round(signals["speak_t"], 3), queue_count,
                       s["bambu_active"], s["alert"], s["tts_muted"])
                if key != _icon_state["key"]:
                    icon.icon = _render_icon(
                        s["state"],
                        frame,
                        mic_level=s["mic_level"],
                        tts_amplitude=s["tts_amplitude"],
                        queue_count=queue_count,
                        muted=s["muted"],
                        bambu_active=s["bambu_active"],
                        alert=s["alert"],
                        tts_muted=s["tts_muted"],
                    )
                    _icon_state["key"] = key
                title = _tooltip(data)
                if title != _icon_state["title"]:
                    icon.title = title
                    _icon_state["title"] = title
            except Exception:
                logging.exception("[tray] icon update failed")
            try:
                _poll_results()
            except Exception:
                logging.exception("[tray] result poll failed")
            try:
                _refresh_menu_if_changed(icon, data)
            except Exception:
                logging.exception("[tray] menu refresh failed")
            frame += 1
            _stop_event.wait(TICK_SECONDS)
        except Exception:
            # Never let a single bad iteration kill the animation thread —
            # log and keep spinning so the tray stays responsive.
            logging.exception("[tray] _animate iteration failed")
            _stop_event.wait(TICK_SECONDS)


# ─── Entry point ─────────────────────────────────────────────────────────

def _on_open_project_folder(icon, item):
    """Open the JARVIS project root in Explorer — handy for poking at
    config files (notification_rules.json, hud_config.json, etc.)."""
    try:
        os.startfile(PROJECT_DIR)
    except Exception as e:
        print(f"[tray] open project folder failed: {e}")


# ── Menu status header ───────────────────────────────────────────────────
# Disabled MenuItems at the top of the menu surface the same 4 signals
# the icon shows, but in words — useful for the "what's that pulsing
# blue dot mean again?" moment. Read at menu-open time (pystray invokes
# the lambdas on each right-click, so they always reflect current state).

def _status_text_starting() -> str:
    return "● JARVIS: starting…"


def _status_text_listen() -> str:
    data = _read_hud_state()
    if bool(data.get("mic_muted") or data.get("muted")):
        return "● Listening: muted"
    if _standby_from(data):
        return "● Listening: standby"
    return "● Listening: awake"


def _status_text_tts() -> str:
    data = _read_hud_state()
    raw = str(data.get("state") or "").lower()
    amp = float(data.get("tts_amplitude") or 0.0)
    if raw == "speaking" or amp > 0.02:
        return "● TTS: speaking"
    return "● TTS: quiet"


def _status_text_queue() -> str:
    return f"● Queue: {_count_pending_tasks()} task(s)"


def _status_text_bambu() -> str:
    data = _read_hud_state()
    return "● Bambu: printing" if bool(data.get("bambu_active")) else "● Bambu: idle"


# ── Apple Music tray controls ────────────────────────────────────────────
# JARVIS hosts Apple Music controls in ITS tray because the UWP Apple Music
# app has NO system tray of its own. Transport goes through the SAME command
# IPC as every other tray verb (_send_command -> bobert's drainer -> the
# existing media_playpause / media_next / media_prev / open_apple_music
# ACTIONS, which drive OS media keys + an AUMID launch). We do NOT script the
# app's UI from here — that automation is policy-restricted.
#
# The now-playing LABEL is the one thing read in-process: the tray imports the
# lazy audio.apple_music_app bridge and calls now_playing() (window-title
# parse). That bridge never raises and degrades to None when pygetwindow /
# psutil are missing, so the label still renders ("Apple Music: idle/closed").

def _apple_music_app():
    """Late-bound, best-effort handle to the audio.apple_music_app bridge, or
    None. Imported lazily (not at tray-module import) so a stripped install
    without the audio package — or without the bridge's optional deps — still
    builds the tray; the menu items just degrade to a no-op/'unknown'. Cached on
    the module so repeated menu opens don't re-import. Never raises."""
    cached = globals().get("_apple_music_app_mod")
    if cached is not None:
        return cached if cached is not _AM_UNAVAILABLE else None
    mod = sys.modules.get("audio.apple_music_app")
    if mod is None:
        try:
            from audio import apple_music_app as mod  # type: ignore
        except Exception:
            globals()["_apple_music_app_mod"] = _AM_UNAVAILABLE
            return None
    globals()["_apple_music_app_mod"] = mod
    return mod


# Sentinel so a failed import is remembered (and not retried every menu open)
# without colliding with a genuine None "not looked up yet".
_AM_UNAVAILABLE = object()


def _status_text_apple_music() -> str:
    """Now-playing header for the Apple Music submenu. Returns the CACHED label
    (see _now_playing_label / _now_playing_lookup): the SMTC + window-bridge
    read used to run inline in every menu rebuild with no timeout, on the
    tray's UI thread, so one hung WinRT call froze the whole menu. Never
    raises."""
    try:
        return _now_playing_label()
    except Exception:
        return "Apple Music: unavailable"


def _on_apple_music_playpause(icon, item):
    """Play/Pause the Apple Music app via the OS media-key path. Routes through
    the same command IPC as the rest of the tray: bobert's drainer dispatches
    `media_playpause` to ACTIONS['media_playpause'] (-> _media_key_with_focus)."""
    _send_command("media_playpause")


def _on_apple_music_next(icon, item):
    """Next track via the OS media-key path (ACTIONS['media_next'])."""
    _send_command("media_next")


def _on_apple_music_prev(icon, item):
    """Previous track via the OS media-key path (ACTIONS['media_prev'])."""
    _send_command("media_prev")


def _apple_music_url() -> str:
    return (os.environ.get("JARVIS_APPLE_MUSIC_URL", "").strip()
            or APPLE_MUSIC_WEB_URL)


def _on_open_apple_music(icon, item):
    """Open Apple Music in the default browser (the web player). The owner
    listens in Chrome; the old route launched the Store app by AUMID, which he
    doesn't use. Override the URL with JARVIS_APPLE_MUSIC_URL. The voice action
    'open Apple Music' opens the same page (core/actions.py's
    _APPLE_MUSIC_WEB_URL, kept equal to APPLE_MUSIC_WEB_URL by a test) through
    JARVIS's real-browser opener (2026-10-01)."""
    url = _apple_music_url()
    try:
        import webbrowser
        if webbrowser.open(url):
            return
    except Exception as e:
        print(f"[tray] open Apple Music web failed: {e}")
    _notify(f"Couldn't open {url} in the browser.", "JARVIS — Apple Music")


def _is_standby() -> bool:
    """Pause Listening checkmark + toggle direction: the published
    sleep_mode/standby_mode flags (see _standby_from), not the state label."""
    return _standby_from(_read_hud_state())


# ── Toggle state helpers ─────────────────────────────────────────────────
# All read hud_state.json. The fields below are written by bobert_companion.py
# when it processes the matching tray command (mute_tts_toggle,
# ambient_mode_toggle, …). Until those backend handlers exist the field will
# simply be absent and the toggle stays unchecked — that's intentional, the
# tray must not assume the backend supports a feature before it lands.

def _is_listen_paused() -> bool:
    return _is_standby()


def _is_tts_muted() -> bool:
    return bool(_read_hud_state().get("tts_muted"))


def _is_mic_muted() -> bool:
    """Mic-mute checkmark source. bobert publishes hud_state.mic_muted when it
    processes the mic_mute_toggle command; absent until then -> unchecked."""
    return bool(_read_hud_state().get("mic_muted"))


def _is_ambient_mode() -> bool:
    """Ambient Mode checkmark. The monolith publishes ``ambient_listening`` —
    whether the ambient-listen daemon is REALLY running — once a second.
    ``ambient_mode_active`` (the toggle's own cell) stayed False while
    AMBIENT_LISTEN_ENABLED auto-started the daemon at boot, so the checkmark
    lied; it remains the fallback for an older JARVIS."""
    data = _read_hud_state()
    if "ambient_listening" in data:
        return bool(data.get("ambient_listening"))
    return bool(data.get("ambient_mode_active"))


def _is_debug_mode() -> bool:
    return bool(_read_hud_state().get("debug_mode"))


def _is_daemons_paused() -> bool:
    return bool(_read_hud_state().get("daemons_paused"))


def _active_llm_backend() -> str:
    """Returns 'anthropic' / 'qwen' / 'llama' / 'other' or '' if unknown."""
    return str(_read_hud_state().get("llm_backend") or "").lower()


# Audio Controls submenu readers — default to True (on) when the field is
# absent so a fresh hud_state.json (or one written by an older bobert that
# doesn't publish audio_* flags yet) still shows the pipeline as enabled,
# matching the actual default of AUDIO_PROCESSING_ENABLED = True.
def _audio_field(name: str) -> bool:
    data = _read_hud_state()
    if name not in data:
        return True
    return bool(data.get(name))


def _is_audio_processing_enabled() -> bool:
    return _audio_field("audio_processing_enabled")


def _is_echo_cancel_enabled() -> bool:
    return _audio_field("echo_cancel_enabled")


def _is_noise_suppress_enabled() -> bool:
    return _audio_field("noise_suppress_enabled")


def _is_agc_enabled() -> bool:
    return _audio_field("agc_enabled")


def _is_pipeline_running() -> bool:
    """Detect an in-flight overnight/upgrade pipeline. We accept either the
    documented sentinel (pipeline_lock.json) or the existing .overnight_active
    flag bobert already writes today."""
    try:
        return os.path.exists(PIPELINE_LOCK_FILE) or os.path.exists(OVERNIGHT_FLAG)
    except Exception:
        return False


def _open_path(path: str, label: str = "") -> None:
    """Best-effort Windows-shell open. Used by every 'Open X' menu item."""
    try:
        os.startfile(path)
    except Exception as e:
        print(f"[tray] open {label or path} failed: {e}")


def _open_event_viewer_crashes() -> None:
    """Open JARVIS's own native-crash log (logs\\crash_traces.log — the
    faulthandler dumps the monolith writes) when it exists; otherwise the
    Windows Event Viewer (Application log), where APPCRASH records live."""
    if os.path.exists(CRASH_TRACES_LOG):
        _open_path(CRASH_TRACES_LOG, "crash_traces.log")
        return
    try:
        # eventvwr.msc opens to the Application log; user can pivot.
        # os.startfile uses the shell association for .msc (mmc.exe) and is
        # more robust than Popen(..., shell=True), which with a list arg
        # only honours argv[0].
        os.startfile("eventvwr.msc")
    except Exception as e:
        print(f"[tray] open event viewer failed: {e}")


def _open_live_log_viewer() -> None:
    """Spawn _show_log.ps1 in a visible PowerShell window."""
    if not os.path.exists(SHOW_LOG_PS1):
        print(f"[tray] live log viewer not found: {SHOW_LOG_PS1}")
        return
    try:
        # New visible PowerShell window so the user actually sees the log.
        #
        # EXPLICIT creationflags are LOAD-BEARING (2026-07-14 audit). This
        # process installs core.no_window_subprocess, which patches
        # Popen.__init__ to inject CREATE_NO_WINDOW whenever a caller passes
        # NEITHER creationflags NOR startupinfo — the net that killed the ghost
        # windows. This call passed neither, so the net hid the very window the
        # feature exists to show: clicking "Live log" did nothing visible and
        # left a HIDDEN -NoExit PowerShell running forever (a new orphan per
        # click). Passing an explicit flag opts out of the patch.
        _flags = (subprocess.CREATE_NEW_CONSOLE
                  if sys.platform == "win32" else 0)
        subprocess.Popen(
            ["powershell.exe", "-NoExit",
             "-ExecutionPolicy", "Bypass",
             "-File", SHOW_LOG_PS1],
            creationflags=_flags,
            close_fds=True,
        )
    except Exception as e:
        print(f"[tray] live log spawn failed: {e}")


def _settings_launch_argv(tab: str = "") -> list:
    """The exact argv the tray launches the Settings window with. A MODULE
    launch from cwd=PROJECT_DIR puts the project root on sys.path, so the
    window's `from core import …` works; the old script-path launch put
    tools\\ there instead and every core import inside the window failed."""
    args = [sys.executable, "-m", SETTINGS_MODULE]
    if tab:
        args += ["--tab", tab]
    return args


def _tail(path: str, max_bytes: int = 4000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _last_error_line(text: str) -> str:
    """The last line the child wrote (the tray's own '---- launch' banners
    are skipped)."""
    lines = [ln.strip() for ln in (text or "").splitlines()
             if ln.strip() and not ln.strip().startswith("---- ")]
    return lines[-1] if lines else ""


def _watch_settings_launch(proc, log_path: str, started_at: float,
                           grace: float = SETTINGS_LAUNCH_GRACE_S):
    """Wait up to ``grace`` s for the Settings window process. Exiting non-zero
    inside the grace means it never opened: log it and raise a balloon naming
    the log. Still running after the grace = it opened; stop watching.
    Returns the exit code, or None while it is still running."""
    try:
        rc = proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"[tray] settings window wait failed: {e}")
        return None
    if rc:
        err = _last_error_line(_tail(log_path))
        print(f"[tray] settings window exited with code {rc} after "
              f"{time.time() - started_at:.1f}s — {err or 'no error text'}")
        detail = (_clip(err, 120) + " — ") if err else ""
        _notify(f"Settings didn't open (exit {rc}). {detail}details in "
                f"logs\\{os.path.basename(log_path)}", "JARVIS — Settings")
    return rc


def _open_settings_window(tab: str = "") -> None:
    """Open the Settings window (tools/settings_window.py) on ``tab`` ('' = its
    first tab). stdout/stderr go to logs\\settings_window.log, and a launch that
    fails — or a window that dies at start-up — raises a balloon instead of
    failing silently. If the window isn't installed, fall back to opening the
    raw user_settings.json so settings stay editable."""
    if os.path.exists(SETTINGS_WINDOW):
        log = None
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                os.makedirs(os.path.dirname(SETTINGS_LOG_FILE), exist_ok=True)
                log = open(SETTINGS_LOG_FILE, "ab")
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"\n---- {stamp} launch tab={tab or '(first)'} ----\n"
                          .encode("utf-8"))
                log.flush()
            except Exception as e:
                print(f"[tray] settings log unavailable ({e}) — launching "
                      "without it")
                log = None
            out = log if log is not None else subprocess.DEVNULL
            started = time.time()
            proc = subprocess.Popen(_settings_launch_argv(tab), cwd=PROJECT_DIR,
                                    stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT,
                                    creationflags=creationflags, close_fds=True)
            print(f"[tray] settings window launched (pid "
                  f"{getattr(proc, 'pid', '?')}, tab={tab or 'first'})")
            if log is not None:
                try:
                    log.close()   # the child holds its own handle
                except Exception:
                    pass
                log = None
            _watch_settings_launch(proc, SETTINGS_LOG_FILE, started)
            return
        except Exception as e:
            print(f"[tray] settings window spawn failed: {e}")
            _notify(f"Settings couldn't start: {_clip(str(e), 160)} — opening "
                    "user_settings.json instead.", "JARVIS — Settings")
        finally:
            if log is not None:
                try:
                    log.close()
                except Exception:
                    pass
    # Fallback: open the JSON directly so settings remain user-editable.
    fallback = os.path.join(DATA_DIR, "user_settings.json")
    if os.path.exists(fallback):
        _open_path(fallback, "user_settings.json")
    else:
        print("[tray] settings window not installed and no user_settings.json"
              " — install tools/settings_window.py to enable Settings menu")


def _build_menu():
    """The whole right-click menu. Every dynamic part is a lambda over
    hud_state (read through _read_hud_state, which the animation loop pins to
    one snapshot per tick), so _menu_signature can tell when it changed."""
    # ── Submenu: Power tools ──
    # Stop Running Pipeline is greyed when no upgrade is mid-flight, and Run
    # Upgrade Now while overnight upgrades are switched off, so the menu never
    # promises a no-op. (Reset Local LLM Cache is gone: there is no in-process
    # LLM cache to reset — both servers cache on their side.)
    power_menu = pystray.Menu(
        pystray.MenuItem("Run Upgrade Now",        _on_force_upgrade,
                         enabled=lambda i: _upgrades_enabled()),
        pystray.MenuItem("Stop Running Pipeline",  _on_stop_pipeline,
                         enabled=lambda i: _is_pipeline_running()),
        pystray.MenuItem("Force Backup Now",       _on_force_backup),
        pystray.MenuItem("Reload All Skills",      _on_reload_skills),
        pystray.MenuItem("Run Smoke Test",         _on_run_smoke_test),
        pystray.MenuItem("Pause All Daemons",      _on_pause_daemons,
                         checked=lambda i: _is_daemons_paused()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open JARVIS Folder",     _on_open_project_folder),
        pystray.MenuItem("Open Logs Folder",       _on_open_logs),
        pystray.MenuItem("Open Task Queue",        _on_open_todo),
        pystray.MenuItem("Open Live Log Viewer",   _on_open_live_log),
        pystray.MenuItem("Open Crash Reports",     _on_open_crashes),
        pystray.MenuItem("Open Release Notes",     _on_open_changelog),
    )

    # ── Submenu: AI ──
    # Checkmarks reflect hud_state.llm_backend so bobert remains the source
    # of truth on which backend is actually serving requests. The local model
    # picker lists what Ollama actually has installed (no hard-coded tags).
    ai_menu = pystray.Menu(
        pystray.MenuItem(
            lambda i: _confirm_text("switch_anthropic",
                                    "Switch to Claude (cloud, paid)"),
            _on_switch_anthropic,
            checked=lambda i: _active_llm_backend() == "anthropic"),
        pystray.MenuItem("Switch to Local LLM (default)",    _on_switch_local,
                         checked=lambda i: _active_llm_backend()
                         not in ("", "anthropic")),
        pystray.MenuItem("Local Model", pystray.Menu(_local_model_items)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Debug Mode",                       _on_toggle_debug_mode,
                         checked=lambda i: _is_debug_mode()),
        pystray.MenuItem("Show LLM Call Stats",              _on_show_llm_stats),
    )

    # ── Submenu: Audio Controls ──
    # Sub-layer toggles (echo/NS/AGC) are greyed when the master switch is off
    # so the menu doesn't promise per-layer control bobert would short-circuit.
    # Thresholds and device pickers live in Settings… (one entry, top level).
    audio_menu = pystray.Menu(
        pystray.MenuItem("Audio Processing",   _on_toggle_audio_processing,
                         checked=lambda i: _is_audio_processing_enabled()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Echo Cancellation",  _on_toggle_echo_cancel,
                         checked=lambda i: _is_echo_cancel_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
        pystray.MenuItem("Noise Suppression",  _on_toggle_noise_suppress,
                         checked=lambda i: _is_noise_suppress_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
        pystray.MenuItem("Gain Normalization", _on_toggle_agc,
                         checked=lambda i: _is_agc_enabled(),
                         enabled=lambda i: _is_audio_processing_enabled()),
    )

    # ── Submenu: Apple Music ──
    # A cached now-playing header (refreshed off-thread) above the transport
    # verbs. Transport uses OS media keys (via the command IPC); "Open Apple
    # Music" opens the web player in the default browser.
    apple_music_menu = pystray.Menu(
        pystray.MenuItem(lambda i: _status_text_apple_music(), None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Play / Pause", _on_apple_music_playpause),
        pystray.MenuItem("Next",         _on_apple_music_next),
        pystray.MenuItem("Previous",     _on_apple_music_prev),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Apple Music", _on_open_apple_music),
    )

    # ── Submenu: Memory ──
    memory_menu = pystray.Menu(
        pystray.MenuItem("Open Long-Term Memory",            _on_open_memory_file),
        pystray.MenuItem("Show What JARVIS Knows About Me…", _on_show_dossier),
        pystray.MenuItem("Recent Facts Learned (last 24h)",  _on_recent_facts),
        pystray.MenuItem("Export Memory (JSON)",             _on_export_memory),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(lambda i: _confirm_text("forget_last_hour",
                                                 "Forget Last Hour"),
                         _on_forget_last_hour),
        pystray.MenuItem(lambda i: _confirm_text("reset_memory",
                                                 "Reset Memory…"),
                         _on_reset_memory),
    )

    # ── Submenu: Diagnostics ──
    diag_menu = pystray.Menu(
        pystray.MenuItem("Run Diagnostic Now",       _on_run_diagnostic),
        pystray.MenuItem("Show Last Diagnostic Run", _on_show_last_diagnostic),
        pystray.MenuItem("Test Mic",                 _on_test_mic),
        pystray.MenuItem("Test TTS",                 _on_test_tts),
        pystray.MenuItem("Test Vision",              _on_test_vision),
        pystray.MenuItem("Test Each Skill",          _on_test_each_skill),
        pystray.MenuItem("Latency Benchmark",        _on_latency_benchmark),
    )

    return pystray.Menu(
        # ── status header (read-only) ──
        pystray.MenuItem(lambda i: _status_text_starting(), None, enabled=False,
                         visible=lambda i: not _jarvis_ready()),
        pystray.MenuItem(lambda i: _status_text_listen(), None, enabled=False),
        pystray.MenuItem(lambda i: _status_text_tts(),    None, enabled=False),
        pystray.MenuItem(lambda i: _status_text_queue(),  None, enabled=False,
                         visible=lambda i: _upgrades_enabled()),
        pystray.MenuItem(lambda i: _status_text_bambu(),  None, enabled=False),
        pystray.Menu.SEPARATOR,
        # ── the two things people reach for (Dashboard = left-click) ──
        pystray.MenuItem("Open Dashboard", _on_dashboard, default=True),
        pystray.MenuItem("Settings…",      _on_open_settings),
        pystray.Menu.SEPARATOR,
        # ── frequent toggles ──
        pystray.MenuItem("Pause Listening", _on_pause_listening,
                         checked=lambda i: _is_listen_paused()),
        pystray.MenuItem("Mute Mic",        _on_mute_mic,
                         checked=lambda i: _is_mic_muted()),
        pystray.MenuItem("Mute TTS",        _on_mute_tts,
                         checked=lambda i: _is_tts_muted()),
        pystray.MenuItem("Ambient Mode",    _on_ambient_mode,
                         checked=lambda i: _is_ambient_mode()),
        pystray.Menu.SEPARATOR,
        # ── grouped submenus ──
        pystray.MenuItem("Audio",       audio_menu),
        pystray.MenuItem("Apple Music", apple_music_menu),
        pystray.MenuItem("AI",          ai_menu),
        pystray.MenuItem("Memory",      memory_menu),
        pystray.MenuItem("Diagnostics", diag_menu),
        pystray.MenuItem("Power tools", power_menu),
        pystray.Menu.SEPARATOR,
        # ── views + notes ──
        pystray.MenuItem("Open HUD",             _on_open_hud),
        pystray.MenuItem("Show Today's Summary", _on_show_today_summary),
        pystray.MenuItem("Queue Task…",          _on_queue_task),
        pystray.MenuItem("About JARVIS",         _on_about),
        pystray.Menu.SEPARATOR,
        # ── lifecycle (each needs a second click) ──
        pystray.MenuItem(lambda i: _confirm_text("restart", "Restart JARVIS"),
                         _on_restart),
        pystray.MenuItem(lambda i: _confirm_text("shutdown_jarvis",
                                                 "Shut Down JARVIS"),
                         _on_shutdown_jarvis),
        pystray.MenuItem(lambda i: _confirm_text("quit_tray", "Quit Tray Only"),
                         _on_quit),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="JARVIS parent PID — tray exits if this dies")
    parser.add_argument("--icon-path", type=str, default=DEFAULT_ICON_PATH,
                        help="Path to the arc-reactor PNG. Falls back to a "
                             "procedural 4-dot grid if missing.")
    parser.add_argument("--queue-task-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: tk dialog mode
    parser.add_argument("--summary-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: tk summary dialog
    parser.add_argument("--about-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: About JARVIS dialog
    parser.add_argument("--dossier-dialog", action="store_true",
                        help=argparse.SUPPRESS)  # internal: facts dossier dialog
    args = parser.parse_args()
    # The About dialog needs to know WHICH JARVIS it describes (uptime).
    _parent_pid[0] = args.parent_pid
    if args.queue_task_dialog:
        sys.exit(_run_queue_task_dialog())
    if args.summary_dialog:
        sys.exit(_run_summary_dialog())
    if args.about_dialog:
        sys.exit(_run_about_dialog())
    if args.dossier_dialog:
        sys.exit(_run_dossier_dialog())
    _setup_tray_logging()
    global _icon_path
    _icon_path = args.icon_path
    _load_base_icon(_icon_path)

    # Boot icon: render once with empty state so the tray has something to
    # show before the first animate() tick arrives. The tray is launched early
    # in JARVIS's boot, so it starts out saying so.
    icon = pystray.Icon(
        "jarvis-tray",
        icon=_render_icon("idle", 0),
        title="JARVIS — Starting…",
        menu=_build_menu(),
    )
    _icon_ref[0] = icon
    _install_menu_open_guard(icon)

    anim = threading.Thread(target=_animate, args=(icon,), daemon=True)
    anim.start()

    print(f"[tray] started (parent pid {_parent_pid[0] or 'unknown'})")
    try:
        icon.run()
    finally:
        _stop_event.set()
        print("[tray] exited")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint; never run under unittest
    main()
