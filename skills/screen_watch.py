"""
screen_watch skill — gentle wellness nudge for long single-window sessions.

Every SCREEN_WATCH_POLL_SECONDS (default 5 min), captures a small thumbnail of
the currently focused window only (cheap — just the window's bbox via mss),
downsamples + hashes it, and tracks how long the user has been on the same
window with no keyboard/mouse input. After STARE_THRESHOLD_SECONDS (default
25 min) of "same window AND system idle", JARVIS gently surfaces:

    "You've been on this for a while, sir. Would you like a 5-minute timer
    to stretch, or shall I leave you to it?"

Window identity is (window_title, content_hash) — a Chrome tab switch or a
different VS Code file counts as activity, not as the same stare. The hash
is computed from a 16×16 grayscale downsample of the focused-window region,
so minor cursor / blinking-caret movement doesn't trip a false reset.

Gates (all must allow before firing):
  • bobert_companion._sleep_mode[0] / _standby_mode[0] must be False
  • face_tracker gaze state must NOT be "away" (None / unknown is OK — we
    don't suppress when the tracker hasn't established gaze, since cameras
    may be unavailable)
  • System idle (Win32 GetLastInputInfo) must be >= STARE_THRESHOLD_SECONDS
  • The user must have been on the same window for >= STARE_THRESHOLD_SECONDS
  • A nudge for THIS window-identity must not have fired within
    NUDGE_COOLDOWN_SECONDS (default 60 min)

Actions registered:
  screen_watch_status — verbally report current focus, stare duration, idle
                        seconds, and whether a nudge is gated.
"""
import importlib
import json
import logging
import os
import sys
import threading
import time
import hashlib

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.screen_watch` or run directly.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────
SCREEN_WATCH_POLL_SECONDS    = 300.0    # 5 minutes
STARE_THRESHOLD_SECONDS      = 25 * 60  # 25 minutes
NUDGE_COOLDOWN_SECONDS       = 60 * 60  # 1 hour between nudges for same window
THUMB_SIZE                   = 16       # 16×16 grayscale downsample for hashing
MIN_WINDOW_PIXELS            = 64       # ignore degenerate (<64x64) windows

# Window titles we never want to nudge on — full-screen video, games, lock
# screen, screensaver, the empty desktop, JARVIS's own HUD. Substring match
# against the lowercased title.
IGNORE_TITLE_FRAGMENTS = (
    "lock screen",
    "screensaver",
    "task switching",
    "program manager",
    "windows default lock screen",
    "j.a.r.v.i.s",           # the HUD's own window
    "jarvis hud",
)

NUDGE_TEXT = (
    "You've been on this for a while, sir. Would you like a 5-minute timer "
    "to stretch, or shall I leave you to it?"
)

_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# ── State ─────────────────────────────────────────────────────────────────
_state_lock         = threading.Lock()
_current_identity   = [None]      # (title, hash) tuple or None
_stare_started_at   = [0.0]       # epoch — when current_identity first observed
_last_nudge_at      = [0.0]       # epoch — most recent nudge time
_last_nudged_id     = [None]      # the (title, hash) that triggered the last nudge

_speech_lock = threading.Lock()


def _enqueue_speech(message: str) -> None:
    """Append a spoken alert to pending_speech.json for the main loop.

    Routes through bobert_companion.proactive_announce() so this skill shares
    one write path with every other pending_speech.json co-writer
    (bambu_monitor, night_owl_mode, status_panel, …) and they don't race each
    other. Falls back to a local atomic write only when the parent module
    isn't loaded yet (import-time registration / unit tests) or the announcer
    call fails. Night-owl mode replaces this function with a sink, so the
    proactive_announce path here is bypassed when suppression is active —
    which is the desired behavior."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="screen_watch")
            return
    except Exception:
        # Fall through to local atomic write — never let a broken parent
        # import silence a wellness nudge.
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
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Print to console so the nudge isn't lost.
            print(f"  [screen_watch] speech-queue write failed ({e}); nudge: {message}")


# ── Win32 idle-time probe ─────────────────────────────────────────────────

def _get_system_idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input system-wide (Windows).
    Returns 0.0 on non-Windows / on failure (so we never falsely fire)."""
    try:
        from ctypes import Structure, c_uint, sizeof, windll, byref
        class _LASTINPUTINFO(Structure):
            _fields_ = [("cbSize", c_uint), ("dwTime", c_uint)]
        info = _LASTINPUTINFO()
        info.cbSize = sizeof(info)
        if not windll.user32.GetLastInputInfo(byref(info)):
            return 0.0
        millis = windll.kernel32.GetTickCount() - info.dwTime
        if millis < 0:
            return 0.0
        return millis / 1000.0
    except Exception:
        return 0.0


# ── Focused-window + thumbnail capture ────────────────────────────────────

def _get_focused_window():
    """Return (title, (left, top, width, height)) of the currently focused
    window, or (None, None) on failure / no active window."""
    try:
        import pygetwindow as gw
    except ImportError:
        return None, None
    try:
        w = gw.getActiveWindow()
    except Exception:
        return None, None
    if w is None:
        return None, None
    title = (w.title or "").strip()
    if not title:
        return None, None
    try:
        left, top, width, height = w.left, w.top, w.width, w.height
    except Exception:
        return None, None
    if width < MIN_WINDOW_PIXELS or height < MIN_WINDOW_PIXELS:
        return None, None
    # Clamp negative coords (minimized windows on Windows often report -32000)
    if left < -10000 or top < -10000:
        return None, None
    return title, (int(left), int(top), int(width), int(height))


def _hash_window_thumbnail(bbox: tuple) -> str | None:
    """Capture the focused window's region via mss, downsample to THUMB_SIZE
    grayscale, hash with md5. Returns hex digest or None on failure."""
    left, top, width, height = bbox
    try:
        import mss
        from PIL import Image
    except ImportError:
        return None
    try:
        with mss.mss() as sct:
            raw = sct.grab({"left": left, "top": top,
                            "width": width, "height": height})
            img = Image.frombytes("RGB", raw.size, raw.rgb)
        thumb = img.convert("L").resize((THUMB_SIZE, THUMB_SIZE),
                                        Image.Resampling.LANCZOS)
        return hashlib.md5(thumb.tobytes()).hexdigest()
    except Exception:
        return None


# ── Gate checks ───────────────────────────────────────────────────────────

def _is_sleeping_or_standby() -> bool:
    """True when JARVIS is asleep or in standby — must not nudge."""
    mod = sys.modules.get("bobert_companion")
    if mod is None:
        return False
    try:
        return bool(mod._sleep_mode[0]) or bool(mod._standby_mode[0])
    except Exception:
        return False


def _user_is_away() -> bool:
    """True when the face tracker has explicitly seen the user as 'away'.
    False when looking at a monitor OR when tracker state is unknown (so a
    missing camera doesn't suppress the nudge)."""
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return False
    snap_func = getattr(mod, "_snapshot_state", None)
    if snap_func is None:
        return False
    try:
        snap = snap_func()
    except Exception:
        return False
    if not snap.get("last_sample_at"):
        return False   # tracker hasn't established anything yet
    return snap.get("current_monitor") == "away"


def _title_is_ignored(title: str) -> bool:
    low = title.lower()
    return any(frag in low for frag in IGNORE_TITLE_FRAGMENTS)


# ── Poll loop ─────────────────────────────────────────────────────────────

def _poll_once() -> None:
    """One iteration of the watch loop. Updates state, fires nudge if due."""
    title, bbox = _get_focused_window()
    now = time.time()

    if title is None or bbox is None:
        # No usable focused window — clear identity so a future focus restart
        # gets a fresh stare timer.
        with _state_lock:
            _current_identity[0] = None
            _stare_started_at[0] = 0.0
        return

    if _title_is_ignored(title):
        with _state_lock:
            _current_identity[0] = None
            _stare_started_at[0] = 0.0
        return

    h = _hash_window_thumbnail(bbox)
    if h is None:
        return   # capture failed — leave state untouched

    identity = (title, h)
    with _state_lock:
        if _current_identity[0] != identity:
            _current_identity[0] = identity
            _stare_started_at[0] = now
            return   # just changed — nothing to fire on this tick

        stare_duration = now - _stare_started_at[0]

    if stare_duration < STARE_THRESHOLD_SECONDS:
        return

    # Stare-duration met. Check the remaining gates.
    if _is_sleeping_or_standby():
        return
    if _user_is_away():
        return
    if _get_system_idle_seconds() < STARE_THRESHOLD_SECONDS:
        return   # user IS using the window, just not switching it

    # Cooldown: don't repeat for the same identity within NUDGE_COOLDOWN_SECONDS
    with _state_lock:
        if (_last_nudged_id[0] == identity
                and (now - _last_nudge_at[0]) < NUDGE_COOLDOWN_SECONDS):
            return
        _last_nudge_at[0] = now
        _last_nudged_id[0] = identity

    print(f"  [screen_watch] firing stretch nudge — focused on '{title}' for "
          f"{stare_duration / 60:.1f} min with system idle "
          f"{_get_system_idle_seconds() / 60:.1f} min")
    _enqueue_speech(NUDGE_TEXT)


def _poll_loop() -> None:
    # Initial settle delay so we don't fire 0 seconds after startup
    time.sleep(SCREEN_WATCH_POLL_SECONDS)
    while True:
        try:
            _poll_once()
            time.sleep(SCREEN_WATCH_POLL_SECONDS)
        except Exception:
            logging.exception("  [screen_watch] poll loop error")
            # Still sleep on failure so we don't busy-loop on a persistent error
            try:
                time.sleep(SCREEN_WATCH_POLL_SECONDS)
            except Exception:
                logging.exception("  [screen_watch] sleep after error failed")


# ── Action handler ────────────────────────────────────────────────────────

def _fmt_minutes(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def register(actions):
    def screen_watch_status(_: str = "") -> str:
        with _state_lock:
            ident = _current_identity[0]
            started = _stare_started_at[0]
        now = time.time()
        if not ident:
            return "I haven't established a focused window yet, sir."
        title, _h = ident
        stare = now - started if started else 0.0
        idle  = _get_system_idle_seconds()
        gates = []
        if _is_sleeping_or_standby():
            gates.append("sleep mode")
        if _user_is_away():
            gates.append("user away")
        if idle < STARE_THRESHOLD_SECONDS:
            gates.append(f"idle only {_fmt_minutes(idle)}")
        if stare < STARE_THRESHOLD_SECONDS:
            gates.append(f"stare only {_fmt_minutes(stare)}")
        gate_str = ", ".join(gates) if gates else "all gates clear"
        return (f"Focused on '{title}' for {_fmt_minutes(stare)}, "
                f"system idle {_fmt_minutes(idle)} — {gate_str}, sir.")

    actions["screen_watch_status"] = screen_watch_status

    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    print(f"  [screen_watch] watcher active — polling focused window every "
          f"{SCREEN_WATCH_POLL_SECONDS:.0f}s, nudge threshold "
          f"{STARE_THRESHOLD_SECONDS // 60} min")
