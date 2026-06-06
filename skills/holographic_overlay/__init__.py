"""
JARVIS holographic overlay skill (package).

This skill grew enough siblings (fullscreen overlay, workshop canvas,
bambu corner, workshop HUD, workshop print monitor, arc-reactor status
HUD, holo HUD v2, stark status ring) that it now ships as a package so
auxiliary renderer scripts can live alongside the manager. The package
__init__ still owns all of the existing register() / action wiring —
unchanged from when this was a flat module — and adds one new
sub-renderer at `skills/holographic_overlay/hud_v2.py` (the Stark-style
top-monitor status ring; spec 2026-05-30 05:23 overnight).

Manages multiple on-screen reactor surfaces:

  1. hud/jarvis_holo.py — a fullscreen translucent always-on-top
     arc-reactor HUD on the top monitor. Spec: jarvis_todo.md 2026-05-27
     09:18 (holographic_overlay). Off by default.

  2. hud/holo_workshop_canvas.py — a compact 3D-style rotating arc
     reactor anchored to the corner of the top monitor. Spec:
     jarvis_todo.md 2026-05-27 10:38 (holo_workshop_canvas). Auto-shows
     when JARVIS is thinking/speaking so the user has a constant visual
     presence indicator; also exposed via `arc_reactor on/off/pulse`
     voice commands. Distinct from the fullscreen overlay because that
     one is a big visual commitment — the workshop canvas is the lighter,
     always-on companion.

Voice triggers — overlay (fullscreen):
  • show_holographic_overlay / show_holo / hud_on / holographic_on
        → spawn the fullscreen overlay subprocess on the top monitor.
  • hide_holographic_overlay / hide_holo / hud_off / dismiss_holo
        → terminate the fullscreen overlay subprocess.
  • toggle_holographic_overlay / toggle_holo  → flip between the two.

Voice triggers — workshop canvas (compact reactor):
  • arc_reactor [on|off|pulse]   → dispatch by argument (default "on").
  • arc_reactor_on / arc_reactor_off / arc_reactor_pulse → direct aliases.
  • holo_workshop_canvas         → alias for arc_reactor on.

Auto-show behavior:
  HOLO_WORKSHOP_AUTO_ON_THINK (config flag, default True): a background
  watcher thread polls hud_state.json and launches the workshop canvas
  the first time JARVIS enters thinking/speaking, then keeps it visible
  for the configured grace period after JARVIS goes idle. This gives the
  reactor that authentic "appears when JARVIS is working" feel without
  the user having to ask for it.

Both renderer subprocesses auto-exit when their parent JARVIS process
dies, so we don't strictly need to manage their lifecycle on shutdown.

Voice triggers — arc-reactor status HUD (2026-05-29 16:35 overnight):
  • arc_reactor_status / arc_reactor_status_on / status_hud / status_ring
        → spawn the four-quadrant CPU/RAM/GPU/NET ring + Bambu inner ring
          on the top monitor (click-through, top-left corner).
  • arc_reactor_status_off / hide_status_hud  → dismiss the widget.
  • arc_reactor_status_toggle / pulse_hud → flip between the two.
  Optional auto-launch via HOLO_ARC_REACTOR_STATUS_AUTO_LAUNCH (off by
  default — replaces the text status_panel_strip visually but is purely
  additive, so the user opts in).

Voice triggers — Stark status ring (2026-05-30 05:23 overnight — hud_v2):
  • stark_status_ring / hud_v2 / status_ring_v2 / show_status_ring_v2
        → spawn the top-center reactor ring with live CPU/RAM/GPU, current
          track, next calendar event, Bambu print %, and a glowing
          speech-state core (idle/listening/thinking/speaking).
  • hide_status_ring_v2 / hud_v2_off / stark_status_ring_off
        → dismiss the widget.
  • status_ring_v2_toggle  → flip between the two states.
  Optional auto-launch via HOLO_STARK_STATUS_RING_AUTO_LAUNCH (off by
  default — additive sibling to the existing arc-reactor surfaces).
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time

_OVERLAY_PROCESS = None
_OVERLAY_LOCK = threading.Lock()
# This module lives at skills/holographic_overlay/__init__.py — three
# dirname() hops up to the project root so every sibling helper path
# (hud/*, *.json control files) still resolves correctly.
_PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_OVERLAY_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "jarvis_holo.py")

# Holographic HUD v2 — Iron Man arc-reactor ring (PyQt6 + QGraphicsView).
# Spec: jarvis_todo.md 2026-05-29 09:18 (holographic_hud_v2). Runs as its
# own subprocess so v1 (tkinter, full-screen cinematic) and v2 (PyQt6,
# permanent arc-reactor panel) can be engaged independently.
_HOLO_HUD_V2_PROCESS = None
_HOLO_HUD_V2_LOCK = threading.Lock()
_HOLO_HUD_V2_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "holographic_hud_v2.py")
# Spec asks for a permanent ring graphic; sized to roughly half the height
# of the top monitor so the ring + transcript area + intent/action lines
# fit without crowding the workspace.
_HOLO_HUD_V2_W = 640
_HOLO_HUD_V2_H = 720
_HOLO_HUD_V2_MARGIN = 40  # px from the top of the top monitor

# Arc-reactor status HUD — PyQt6 four-quadrant ring widget that
# replaces the generic status_panel_strip with system_pulse data
# (CPU/RAM/GPU/network) rendered as four outer arcs around a central
# core plus an inner Bambu print-progress ring. Spec: jarvis_todo.md
# 2026-05-29 16:35 (overnight). Independent subprocess so the user can
# toggle it on its own without disturbing the sibling reactors.
_ARC_STATUS_PROCESS = None
_ARC_STATUS_LOCK = threading.Lock()
_ARC_STATUS_SCRIPT = os.path.join(
    _PROJECT_DIR, "hud", "arc_reactor_status_hud.py",
)
_ARC_STATUS_CONTROL_FILE = os.path.join(
    _PROJECT_DIR, "arc_reactor_status_state.json",
)
# Compact disc — 320×320 reads clearly without crowding the top edge.
_ARC_STATUS_W = 320
_ARC_STATUS_H = 320
# workshop_hud + bambu_h2d_overlay both anchor to the top-right of the
# top monitor; pin this widget to the top-left to avoid stacking
# collisions (regression-risk note in the planner brief).
_ARC_STATUS_MARGIN = 24  # px from the top-left corner of the top monitor

# Workshop canvas (compact reactor) — independent lifecycle from the
# fullscreen overlay. Tracked separately so the user can have one open
# without the other.
_WORKSHOP_PROCESS = None
_WORKSHOP_LOCK = threading.Lock()
_WORKSHOP_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "holo_workshop_canvas.py")
_WORKSHOP_STATE_FILE = os.path.join(_PROJECT_DIR, "holo_workshop_state.json")
_HUD_STATE_FILE = os.path.join(_PROJECT_DIR, "hud_state.json")

# Bambu H2D overlay — an extension of this holographic_overlay skill that
# pins a compact print-status widget to the top-right corner of the top
# monitor whenever bambu_monitor reports an active print. Separate process
# so its tkinter loop is isolated from the fullscreen overlay's.
_BAMBU_OVERLAY_PROCESS = None
_BAMBU_OVERLAY_LOCK = threading.Lock()
_BAMBU_OVERLAY_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "bambu_h2d_overlay.py")
_BAMBU_OVERLAY_STATE_FILE = os.path.join(_PROJECT_DIR, "bambu_overlay_state.json")
# Widget dimensions per the spec (~280×140, top-right of top monitor).
_BAMBU_OVERLAY_W = 280
_BAMBU_OVERLAY_H = 140
_BAMBU_OVERLAY_MARGIN = 20  # px gap from the top-right corner of the top monitor
# After a print transitions to FINISH / FAILED / IDLE we keep the widget
# up briefly so the user gets a chance to glance at the final state
# before it disappears.
_BAMBU_OVERLAY_LINGER_S = 30.0
# Whether the bambu-overlay watcher thread has been started yet.
_BAMBU_WATCHER_STARTED = False
_BAMBU_WATCHER_STOP = threading.Event()
# When True, the watcher will not auto-spawn or auto-dismiss the widget —
# used so a manual `bambu_overlay_off` command sticks rather than getting
# re-summoned by the next active-print poll. The flag clears the next
# time the user re-issues `bambu_overlay_on` (or printer goes idle so the
# user can re-engage the auto mode at the next print).
_BAMBU_OVERLAY_USER_OFF = False

# Bambu chamber-camera HUD — a movable PyQt6 panel showing the printer's
# built-in camera feed with a print-status footer. Spec: HUD Bambu-printer-
# camera view (2026-06). The frame itself is pulled over the LAN by
# core/bambu_camera.py (RTSPS port 322 for the H2D, port-6000 JPEG fallback
# for P1/A1); this widget is a pure view of data/bambu_camera_frame.jpg.
# Gated by the HUD_BAMBU_CAMERA config flag (default True).
_BAMBU_CAMERA_HUD_PROCESS = None
_BAMBU_CAMERA_HUD_LOCK = threading.Lock()
_BAMBU_CAMERA_HUD_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "bambu_camera_hud.py")
_BAMBU_CAMERA_HUD_CONTROL_FILE = os.path.join(
    _PROJECT_DIR, "bambu_camera_hud_state.json",
)
# A 4:3-ish panel large enough to read the build plate. Anchored top-right
# of the top monitor, below the bambu corner overlay if that's also alive.
_BAMBU_CAMERA_HUD_W = 360
_BAMBU_CAMERA_HUD_H = 300
_BAMBU_CAMERA_HUD_MARGIN = 20  # px gap from the top-right corner
# Linger after a print finishes so the user catches the final frame.
_BAMBU_CAMERA_HUD_LINGER_S = 30.0
_BAMBU_CAMERA_WATCHER_STARTED = False
_BAMBU_CAMERA_WATCHER_STOP = threading.Event()
# Manual-off latch so `bambu_camera_off` sticks against the auto-show watcher;
# cleared when the printer goes idle so the next print can re-arm auto-show.
_BAMBU_CAMERA_USER_OFF = False

# Workshop HUD — persistent top-right corner widget (arc reactor +
# CPU/RAM bars + bambu progress + rotating "monitoring" status line).
# Spec: jarvis_todo.md 2026-05-27 12:15 (workshop_hud).
_WORKSHOP_HUD_PROCESS = None
_WORKSHOP_HUD_LOCK = threading.Lock()
_WORKSHOP_HUD_SCRIPT = os.path.join(_PROJECT_DIR, "hud", "workshop_hud.py")
_WORKSHOP_HUD_CONTROL_FILE = os.path.join(_PROJECT_DIR, "workshop_hud_state.json")
# Slim & tall — sits to the right of the work area without crowding it.
_WORKSHOP_HUD_W = 260
_WORKSHOP_HUD_H = 290
_WORKSHOP_HUD_MARGIN = 20  # px from the top-right corner of the top monitor

# Workshop print monitor HUD — Stark-style top-center panel summarising
# the H2D print state whenever the user is in the workshop OR a print is
# active. Spec: jarvis_todo.md 2026-05-29 23:27 (overnight). Sibling to
# the existing bambu_h2d_overlay but visually distinct (cyan-accent Stark
# panel) and triggered by a richer set of conditions (CAD-window focus
# OR active print) so the user always has a glanceable surface for the
# print state while at the workshop bench.
_WORKSHOP_PRINT_MONITOR_PROCESS = None
_WORKSHOP_PRINT_MONITOR_LOCK = threading.Lock()
_WORKSHOP_PRINT_MONITOR_SCRIPT = os.path.join(
    _PROJECT_DIR, "hud", "workshop_print_monitor.py",
)
_WORKSHOP_PRINT_MONITOR_CONTROL_FILE = os.path.join(
    _PROJECT_DIR, "workshop_print_monitor_state.json",
)
# Wider than the bambu overlay so the Stark panel reads as a top-monitor
# HUD instead of a sticky corner widget. 400×200 keeps it from crowding
# either the top-left (arc_reactor_status_hud, 320×320) or top-right
# (workshop_hud 260×290 + bambu_h2d_overlay 280×140) columns.
_WORKSHOP_PRINT_MONITOR_W = 400
_WORKSHOP_PRINT_MONITOR_H = 200
_WORKSHOP_PRINT_MONITOR_MARGIN = 18  # px from the top of the top monitor
# Linger window after the print finishes / CAD window closes — gives the
# user a beat to see the final state before the panel retires.
_WORKSHOP_PRINT_MONITOR_LINGER_S = 30.0
_WORKSHOP_PRINT_MONITOR_WATCHER_STARTED = False
_WORKSHOP_PRINT_MONITOR_WATCHER_STOP = threading.Event()
# When True, a manual `workshop_print_monitor_off` sticks even if the
# watcher sees an active print / open CAD window. Cleared automatically
# the next time both conditions are clear so the next workshop session
# re-arms auto-show.
_WORKSHOP_PRINT_MONITOR_USER_OFF = False

# Default canvas size and corner offset. ~320×320 reads as a clear visual
# without crowding the work area on the top monitor.
_WORKSHOP_W = 320
_WORKSHOP_H = 320
_WORKSHOP_MARGIN = 40  # pixels from the bottom-right of the top monitor

# Auto-show watcher state. Single-shot start guarded by _WATCHER_STARTED.
_WATCHER_STARTED = False
_WATCHER_STOP = threading.Event()
# Seconds of consecutive idle/sleep state before the auto-launched
# workshop canvas hides itself. Buffers brief gaps between
# thinking → speaking → idle so the reactor doesn't blink.
_AUTO_HIDE_GRACE_S = 3.0


def _get_monitor_rect():
    """Resolve the top-monitor rect from bobert_companion.MONITORS. Falls
    back to a 1440p default if the parent module isn't loaded or the
    'top' key is missing."""
    try:
        import bobert_companion as _bc
        mon = _bc.MONITORS.get("top") if hasattr(_bc, "MONITORS") else None
        if mon and len(mon) >= 4:
            return int(mon[0]), int(mon[1]), int(mon[2]), int(mon[3])
    except Exception:
        pass
    return 0, 0, 2560, 1440


def _overlay_is_alive() -> bool:
    proc = _OVERLAY_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _launch_overlay() -> tuple[bool, str]:
    """Spawn the overlay subprocess. Returns (success, message)."""
    global _OVERLAY_PROCESS
    with _OVERLAY_LOCK:
        if _overlay_is_alive():
            return True, "Holographic overlay is already engaged, sir."
        if not os.path.exists(_OVERLAY_SCRIPT):
            return False, f"I'm afraid the overlay script is missing — {_OVERLAY_SCRIPT}."
        x, y, w, h = _get_monitor_rect()
        try:
            _OVERLAY_PROCESS = subprocess.Popen(
                [sys.executable, _OVERLAY_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _OVERLAY_PROCESS = None
            return False, f"I'm afraid the overlay failed to launch, sir — {e}."
        return True, "Holographic interface online, sir."


def _shutdown_overlay() -> tuple[bool, str]:
    """Terminate the overlay subprocess if it's running."""
    global _OVERLAY_PROCESS
    with _OVERLAY_LOCK:
        if not _overlay_is_alive():
            _OVERLAY_PROCESS = None
            return True, "The overlay isn't currently engaged, sir."
        try:
            _OVERLAY_PROCESS.terminate()
            try:
                _OVERLAY_PROCESS.wait(timeout=2.0)
            except Exception:
                # terminate() didn't take — escalate to kill, then ALWAYS
                # wait() so the OS handle is released even when kill()
                # itself raises (e.g. process exited between the wait
                # timeout and the kill call → zombie on Windows).
                try:
                    _OVERLAY_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _OVERLAY_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _OVERLAY_PROCESS = None
        return True, "Holographic interface dismissed, sir."


def _act_show(_: str = "") -> str:
    ok, msg = _launch_overlay()
    return msg if ok else f"REFUSED: {msg}"


def _act_hide(_: str = "") -> str:
    ok, msg = _shutdown_overlay()
    return msg if ok else f"REFUSED: {msg}"


def _act_toggle(_: str = "") -> str:
    if _overlay_is_alive():
        return _act_hide("")
    return _act_show("")


def _act_status(_: str = "") -> str:
    if _overlay_is_alive():
        x, y, w, h = _get_monitor_rect()
        return (f"The holographic overlay is engaged on the top monitor, sir — "
                f"{w}x{h} at ({x}, {y}).")
    return "The holographic overlay is not currently engaged, sir."


# ──────────────────────────────────────────────────────────────────────────
#  Workshop canvas — compact 3D-style reactor (separate subprocess from
#  the fullscreen overlay above).
# ──────────────────────────────────────────────────────────────────────────


def _workshop_is_alive() -> bool:
    proc = _WORKSHOP_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_workshop_state(**updates):
    """Atomic-write the workshop state file so a live canvas subprocess
    can pick up mode changes without restart."""
    try:
        existing = {}
        if os.path.exists(_WORKSHOP_STATE_FILE):
            try:
                with open(_WORKSHOP_STATE_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _WORKSHOP_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _WORKSHOP_STATE_FILE)
    except Exception:
        # State file is a nice-to-have for mid-flight mode switching.
        # If we can't write it, the existing CLI default still applies.
        pass


def _resolve_workshop_geometry() -> tuple[int, int, int, int]:
    """Anchor the canvas to the bottom-right corner of the top monitor.
    Falls back to a sensible default rect if MONITORS is unavailable."""
    mx, my, mw, mh = _get_monitor_rect()
    w = min(_WORKSHOP_W, max(160, mw // 4))
    h = min(_WORKSHOP_H, max(160, mh // 4))
    x = mx + mw - w - _WORKSHOP_MARGIN
    y = my + mh - h - _WORKSHOP_MARGIN
    return x, y, w, h


def _launch_workshop(mode: str = "on") -> tuple[bool, str]:
    """Spawn the workshop canvas subprocess in the given mode."""
    global _WORKSHOP_PROCESS
    mode = (mode or "on").lower()
    if mode not in ("on", "pulse"):
        mode = "on"
    with _WORKSHOP_LOCK:
        if _workshop_is_alive():
            # Update the mode in the live state file so the running
            # canvas can switch presentation without a restart.
            _write_workshop_state(mode=mode)
            label = "pulsing" if mode == "pulse" else "engaged"
            return True, f"Arc reactor {label}, sir."
        if not os.path.exists(_WORKSHOP_SCRIPT):
            return False, (f"I'm afraid the workshop canvas script is "
                           f"missing — {_WORKSHOP_SCRIPT}.")
        x, y, w, h = _resolve_workshop_geometry()
        # Reset the state file so a stale "off" entry from the previous
        # session doesn't immediately tell the new subprocess to exit.
        _write_workshop_state(mode=mode, force_visible=False)
        try:
            _WORKSHOP_PROCESS = subprocess.Popen(
                [sys.executable, _WORKSHOP_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid()),
                 "--mode", mode],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _WORKSHOP_PROCESS = None
            return False, f"I'm afraid the arc reactor failed to engage, sir — {e}."
        if mode == "pulse":
            return True, "Arc reactor online, pulsing for your benefit, sir."
        return True, "Arc reactor online, sir."


def _shutdown_workshop() -> tuple[bool, str]:
    """Terminate the workshop canvas subprocess if it's running."""
    global _WORKSHOP_PROCESS
    with _WORKSHOP_LOCK:
        # Signal a clean exit via the state file so the next tick the
        # canvas reads it and closes itself; we also send terminate() as
        # a belt-and-braces measure for cases where the canvas missed a
        # tick.
        _write_workshop_state(mode="off", force_visible=False)
        if not _workshop_is_alive():
            _WORKSHOP_PROCESS = None
            return True, "The arc reactor isn't currently engaged, sir."
        try:
            _WORKSHOP_PROCESS.terminate()
            try:
                _WORKSHOP_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _WORKSHOP_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _WORKSHOP_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _WORKSHOP_PROCESS = None
        return True, "Arc reactor disengaged, sir."


def _act_arc_reactor(args: str = "") -> str:
    """Dispatch `arc_reactor on|off|pulse`. Defaults to `on` when called
    bare — matches the spec's `arc_reactor on/off/pulse voice commands`."""
    arg = (args or "").strip().lower()
    # Accept a few natural phrasings.
    if arg in ("", "on", "engage", "show", "start"):
        ok, msg = _launch_workshop("on")
    elif arg in ("off", "disengage", "hide", "stop", "dismiss"):
        ok, msg = _shutdown_workshop()
    elif arg in ("pulse", "pulsing", "throb"):
        ok, msg = _launch_workshop("pulse")
    else:
        # Unknown arg — be permissive and treat it as "on" rather than
        # bouncing the user with a refusal. The LLM occasionally hands us
        # state-y phrases ("active", "engage now"); err on the side of
        # doing the most-expected thing.
        ok, msg = _launch_workshop("on")
    return msg if ok else f"REFUSED: {msg}"


def _act_arc_reactor_on(_: str = "") -> str:
    ok, msg = _launch_workshop("on")
    return msg if ok else f"REFUSED: {msg}"


def _act_arc_reactor_off(_: str = "") -> str:
    ok, msg = _shutdown_workshop()
    return msg if ok else f"REFUSED: {msg}"


def _act_arc_reactor_pulse(_: str = "") -> str:
    ok, msg = _launch_workshop("pulse")
    return msg if ok else f"REFUSED: {msg}"


def _read_jarvis_state() -> str:
    """Read the current JARVIS state out of hud_state.json. Returns
    'idle' on any failure so the watcher leans towards inaction."""
    if not os.path.exists(_HUD_STATE_FILE):
        return "idle"
    try:
        with open(_HUD_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("state") or "idle").lower()
    except Exception:
        return "idle"


def _auto_show_watcher():
    """Background thread: launches the workshop canvas when JARVIS first
    enters thinking/speaking, then hides it after _AUTO_HIDE_GRACE_S of
    sustained idle/sleep. Polls at 2 Hz — plenty for visual feedback
    without burning CPU."""
    last_active_at = 0.0
    while not _WATCHER_STOP.wait(0.5):
        try:
            state = _read_jarvis_state()
            active = state in ("thinking", "speaking")
            now = time.time()
            if active:
                last_active_at = now
                if not _workshop_is_alive():
                    # Lean on the existing launcher's locking so two near-
                    # simultaneous triggers don't double-spawn.
                    _launch_workshop("on")
            else:
                # If the canvas is up AND we've been idle long enough, retire
                # it. We only auto-hide canvases WE auto-launched — to detect
                # that, we check whether the state file's mode is "on" (the
                # default we wrote on auto-launch); if the user explicitly
                # set "pulse" or anything else, leave it alone.
                if _workshop_is_alive() and last_active_at > 0:
                    if (now - last_active_at) >= _AUTO_HIDE_GRACE_S:
                        mode = "on"
                        try:
                            if os.path.exists(_WORKSHOP_STATE_FILE):
                                with open(_WORKSHOP_STATE_FILE, "r",
                                          encoding="utf-8") as f:
                                    mode = (json.load(f).get("mode")
                                            or "on").lower()
                        except Exception:
                            pass
                        if mode == "on":
                            _shutdown_workshop()
                        # Reset so we don't keep re-firing the shutdown.
                        last_active_at = 0.0
        except Exception:
            logging.exception(
                "_auto_show_watcher iteration failed; continuing"
            )


def _maybe_start_auto_watcher():
    """Start the auto-show watcher exactly once. Reads the config flag
    HOLO_WORKSHOP_AUTO_ON_THINK lazily so the skill module imports
    cleanly even before bobert_companion finishes initialising."""
    global _WATCHER_STARTED
    if _WATCHER_STARTED:
        return
    enabled = True
    try:
        import bobert_companion as _bc
        enabled = bool(getattr(_bc, "HOLO_WORKSHOP_AUTO_ON_THINK", True))
    except Exception:
        pass
    if not enabled:
        return
    t = threading.Thread(
        target=_auto_show_watcher, name="HoloWorkshopWatcher", daemon=True,
    )
    t.start()
    _WATCHER_STARTED = True


# ──────────────────────────────────────────────────────────────────────────
#  Bambu H2D corner overlay — active-print HUD extension.
# ──────────────────────────────────────────────────────────────────────────


def _bambu_overlay_is_alive() -> bool:
    proc = _BAMBU_OVERLAY_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _resolve_bambu_overlay_geometry() -> tuple[int, int, int, int]:
    """Anchor the overlay to the top-right corner of the top monitor.

    Falls back to the (0,0)-anchored default rect when MONITORS isn't
    loaded — produces a sensible on-screen placement even before
    bobert_companion finishes initialising.

    When the workshop HUD is also engaged it owns the very top of the
    top-right column, so we slide the bambu widget down by the HUD's
    footprint so the two stack cleanly instead of colliding.
    """
    mx, my, mw, mh = _get_monitor_rect()
    w = _BAMBU_OVERLAY_W
    h = _BAMBU_OVERLAY_H
    x = mx + mw - w - _BAMBU_OVERLAY_MARGIN
    y = my + _BAMBU_OVERLAY_MARGIN
    if _workshop_hud_is_alive():
        y += _WORKSHOP_HUD_H + 12
    return x, y, w, h


def _launch_bambu_overlay() -> tuple[bool, str]:
    """Spawn the bambu overlay widget subprocess. Idempotent."""
    global _BAMBU_OVERLAY_PROCESS
    with _BAMBU_OVERLAY_LOCK:
        if _bambu_overlay_is_alive():
            return True, "The print overlay is already engaged, sir."
        if not os.path.exists(_BAMBU_OVERLAY_SCRIPT):
            return False, (f"I'm afraid the print overlay script is "
                           f"missing — {_BAMBU_OVERLAY_SCRIPT}.")
        x, y, w, h = _resolve_bambu_overlay_geometry()
        try:
            _BAMBU_OVERLAY_PROCESS = subprocess.Popen(
                [sys.executable, _BAMBU_OVERLAY_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _BAMBU_OVERLAY_PROCESS = None
            return False, f"I'm afraid the print overlay failed to launch, sir — {e}."
        return True, "Print overlay engaged, sir."


def _shutdown_bambu_overlay() -> tuple[bool, str]:
    """Terminate the bambu overlay subprocess if it's running."""
    global _BAMBU_OVERLAY_PROCESS
    with _BAMBU_OVERLAY_LOCK:
        if not _bambu_overlay_is_alive():
            _BAMBU_OVERLAY_PROCESS = None
            return True, "The print overlay isn't currently engaged, sir."
        try:
            _BAMBU_OVERLAY_PROCESS.terminate()
            try:
                _BAMBU_OVERLAY_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _BAMBU_OVERLAY_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _BAMBU_OVERLAY_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _BAMBU_OVERLAY_PROCESS = None
        return True, "Print overlay dismissed, sir."


def _read_bambu_overlay_state() -> dict:
    if not os.path.exists(_BAMBU_OVERLAY_STATE_FILE):
        return {}
    try:
        with open(_BAMBU_OVERLAY_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _bambu_is_active(state: dict) -> bool:
    """Whether the printer is currently in a state where the overlay
    should be visible. RUNNING / PAUSE = active print. PREPARE = print
    about to start so we want the user to see ETA dialling in. FINISH /
    FAILED handled by the linger window so the user sees the final
    state before the widget retires."""
    gs = (state.get("gcode_state") or "").upper()
    return gs in ("RUNNING", "PAUSE", "PREPARE")


def _bambu_overlay_watcher() -> None:
    """Background thread: spawns the overlay when a print becomes active
    and tears it down after _BAMBU_OVERLAY_LINGER_S of sustained idle /
    finish. Polls at 1 Hz — the bambu push cadence is roughly per-minute
    so anything tighter is wasted CPU."""
    last_active_at = 0.0
    saw_active_once = False
    while not _BAMBU_WATCHER_STOP.wait(1.0):
        try:
            state = _read_bambu_overlay_state()
            active = _bambu_is_active(state)
            now = time.time()
            if active:
                last_active_at = now
                saw_active_once = True
                # If the user explicitly turned the overlay off this print is
                # still "active" — we still respect the off until they ask
                # back. (Cleared automatically when the printer goes idle so
                # the next print re-engages auto mode.)
                if not _BAMBU_OVERLAY_USER_OFF and not _bambu_overlay_is_alive():
                    _launch_bambu_overlay()
            else:
                # Linger so the user sees FINISH / FAILED before it goes.
                if saw_active_once and _bambu_overlay_is_alive():
                    if (now - last_active_at) >= _BAMBU_OVERLAY_LINGER_S:
                        _shutdown_bambu_overlay()
                        saw_active_once = False
                        # Printer went idle long enough — re-enable auto-show
                        # for the next print.
                        _clear_user_off()
        except Exception:
            logging.exception("bambu overlay watcher iteration failed")


def _clear_user_off() -> None:
    """Helper so the watcher can reset the manual-off flag without
    importing module globals at every site."""
    global _BAMBU_OVERLAY_USER_OFF
    _BAMBU_OVERLAY_USER_OFF = False


def _maybe_start_bambu_watcher() -> None:
    """Start the bambu overlay watcher exactly once. Respects the
    BAMBU_OVERLAY_AUTO_WHILE_PRINTING flag (default True)."""
    global _BAMBU_WATCHER_STARTED
    if _BAMBU_WATCHER_STARTED:
        return
    enabled = True
    try:
        import bobert_companion as _bc
        enabled = bool(getattr(_bc, "BAMBU_OVERLAY_AUTO_WHILE_PRINTING",
                               True))
    except Exception:
        pass
    if not enabled:
        return
    t = threading.Thread(
        target=_bambu_overlay_watcher,
        name="BambuOverlayWatcher",
        daemon=True,
    )
    t.start()
    _BAMBU_WATCHER_STARTED = True


def _act_bambu_overlay_on(_: str = "") -> str:
    global _BAMBU_OVERLAY_USER_OFF
    _BAMBU_OVERLAY_USER_OFF = False
    ok, msg = _launch_bambu_overlay()
    return msg if ok else f"REFUSED: {msg}"


def _act_bambu_overlay_off(_: str = "") -> str:
    global _BAMBU_OVERLAY_USER_OFF
    _BAMBU_OVERLAY_USER_OFF = True
    ok, msg = _shutdown_bambu_overlay()
    return msg if ok else f"REFUSED: {msg}"


def _act_bambu_overlay_toggle(_: str = "") -> str:
    if _bambu_overlay_is_alive():
        return _act_bambu_overlay_off("")
    return _act_bambu_overlay_on("")


def _act_bambu_overlay_status(_: str = "") -> str:
    state = _read_bambu_overlay_state()
    gs = (state.get("gcode_state") or "—").upper()
    if _bambu_overlay_is_alive():
        return (f"Print overlay is engaged on the top monitor, sir — "
                f"printer state {gs}.")
    if _BAMBU_OVERLAY_USER_OFF:
        return ("The print overlay is dismissed (manual), sir. "
                "Say `bambu overlay on` to re-engage it.")
    return f"The print overlay is dormant, sir — printer state {gs}."


# ──────────────────────────────────────────────────────────────────────────
#  Bambu chamber-camera HUD — movable PyQt6 panel showing the printer's
#  built-in camera feed. The frame is pulled over the LAN by
#  core/bambu_camera.py; this manager owns the widget subprocess + the
#  background frame grabber's lifecycle.
# ──────────────────────────────────────────────────────────────────────────


def _bambu_camera_enabled() -> bool:
    """Master HUD_BAMBU_CAMERA flag (default True). Read lazily so the skill
    imports cleanly before bobert_companion finishes initialising."""
    try:
        import bobert_companion as _bc
        return bool(getattr(_bc, "HUD_BAMBU_CAMERA", True))
    except Exception:
        return True


def _start_camera_grabber() -> None:
    """Spin up the LAN frame grabber (core/bambu_camera.py) so the widget has
    frames to render. Idempotent and best-effort — a missing module or import
    error just means the widget shows its 'camera offline' placeholder."""
    try:
        import importlib
        cam = importlib.import_module("core.bambu_camera")
        cam.start_grabber()
    except Exception as e:
        print(f"  [bambu-camera] grabber start skipped: {e}")


def _stop_camera_grabber() -> None:
    """Stop the LAN frame grabber. Best-effort."""
    try:
        import importlib
        cam = importlib.import_module("core.bambu_camera")
        cam.stop_grabber()
    except Exception:
        pass


def _bambu_camera_hud_is_alive() -> bool:
    proc = _BAMBU_CAMERA_HUD_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_bambu_camera_hud_control(**updates) -> None:
    """Atomic-write the camera-HUD control file so the running widget polls a
    mode change (e.g. retire) without us racing terminate()."""
    try:
        existing = {}
        if os.path.exists(_BAMBU_CAMERA_HUD_CONTROL_FILE):
            try:
                with open(_BAMBU_CAMERA_HUD_CONTROL_FILE, "r",
                          encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _BAMBU_CAMERA_HUD_CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _BAMBU_CAMERA_HUD_CONTROL_FILE)
    except Exception:
        # Control file is a nice-to-have; terminate() is the fallback.
        pass


def _resolve_bambu_camera_hud_geometry() -> tuple[int, int, int, int]:
    """Anchor the camera panel to the top-right of the top monitor. When the
    bambu corner overlay is also alive it owns the very top-right, so slide
    the camera panel below it so the two stack cleanly."""
    mx, my, mw, _mh = _get_monitor_rect()
    w = _BAMBU_CAMERA_HUD_W
    h = _BAMBU_CAMERA_HUD_H
    x = mx + mw - w - _BAMBU_CAMERA_HUD_MARGIN
    y = my + _BAMBU_CAMERA_HUD_MARGIN
    if _bambu_overlay_is_alive():
        y += _BAMBU_OVERLAY_H + 12
    return x, y, w, h


def _launch_bambu_camera_hud() -> tuple[bool, str]:
    """Spawn the camera HUD subprocess (and the frame grabber). Idempotent."""
    global _BAMBU_CAMERA_HUD_PROCESS
    if not _bambu_camera_enabled():
        return False, ("The printer camera is disabled in settings, sir "
                       "(HUD_BAMBU_CAMERA).")
    with _BAMBU_CAMERA_HUD_LOCK:
        if _bambu_camera_hud_is_alive():
            _write_bambu_camera_hud_control(mode="on")
            return True, "The printer camera is already on screen, sir."
        if not os.path.exists(_BAMBU_CAMERA_HUD_SCRIPT):
            return False, (f"I'm afraid the camera HUD script is "
                           f"missing — {_BAMBU_CAMERA_HUD_SCRIPT}.")
        # Start the LAN grabber first so a frame is on its way by the time
        # the widget paints, then clear any stale 'off' control entry.
        _start_camera_grabber()
        _write_bambu_camera_hud_control(mode="on")
        x, y, w, h = _resolve_bambu_camera_hud_geometry()
        try:
            _BAMBU_CAMERA_HUD_PROCESS = subprocess.Popen(
                [sys.executable, _BAMBU_CAMERA_HUD_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _BAMBU_CAMERA_HUD_PROCESS = None
            return False, f"I'm afraid the camera HUD failed to launch, sir — {e}."
        return True, "Printer camera engaged, sir."


def _shutdown_bambu_camera_hud(stop_grabber: bool = True) -> tuple[bool, str]:
    """Terminate the camera HUD subprocess (and optionally the grabber)."""
    global _BAMBU_CAMERA_HUD_PROCESS
    with _BAMBU_CAMERA_HUD_LOCK:
        _write_bambu_camera_hud_control(mode="off")
        alive = _bambu_camera_hud_is_alive()
        if not alive:
            _BAMBU_CAMERA_HUD_PROCESS = None
        else:
            try:
                _BAMBU_CAMERA_HUD_PROCESS.terminate()
                try:
                    _BAMBU_CAMERA_HUD_PROCESS.wait(timeout=2.0)
                except Exception:
                    # See _shutdown_overlay — kill() may race the process
                    # exiting; wait() after kill() ensures handle release.
                    try:
                        _BAMBU_CAMERA_HUD_PROCESS.kill()
                    except Exception:
                        pass
                    try:
                        _BAMBU_CAMERA_HUD_PROCESS.wait(timeout=0.1)
                    except Exception:
                        pass
            except Exception:
                pass
            _BAMBU_CAMERA_HUD_PROCESS = None
        # Stop pulling frames over the LAN once nothing's displaying them.
        if stop_grabber:
            _stop_camera_grabber()
        if not alive:
            return True, "The printer camera isn't currently on screen, sir."
        return True, "Printer camera dismissed, sir."


def _bambu_camera_watcher() -> None:
    """Background thread: auto-show the camera while a print is active and
    retire it after _BAMBU_CAMERA_HUD_LINGER_S of sustained idle/finish.
    Mirrors _bambu_overlay_watcher. Only runs when
    BAMBU_CAMERA_AUTO_WHILE_PRINTING is on."""
    last_active_at = 0.0
    saw_active_once = False
    while not _BAMBU_CAMERA_WATCHER_STOP.wait(1.0):
        try:
            state = _read_bambu_overlay_state()
            active = _bambu_is_active(state)
            now = time.time()
            if active:
                last_active_at = now
                saw_active_once = True
                if (not _BAMBU_CAMERA_USER_OFF
                        and not _bambu_camera_hud_is_alive()):
                    _launch_bambu_camera_hud()
            else:
                if saw_active_once and _bambu_camera_hud_is_alive():
                    if (now - last_active_at) >= _BAMBU_CAMERA_HUD_LINGER_S:
                        _shutdown_bambu_camera_hud()
                        saw_active_once = False
                        _clear_camera_user_off()
        except Exception:
            logging.exception("bambu camera watcher iteration failed")


def _clear_camera_user_off() -> None:
    global _BAMBU_CAMERA_USER_OFF
    _BAMBU_CAMERA_USER_OFF = False


def _maybe_start_bambu_camera_watcher() -> None:
    """Start the camera auto-show watcher exactly once. Respects both
    HUD_BAMBU_CAMERA (master) and BAMBU_CAMERA_AUTO_WHILE_PRINTING (default
    False — the camera is opt-in, never pops up unbidden)."""
    global _BAMBU_CAMERA_WATCHER_STARTED
    if _BAMBU_CAMERA_WATCHER_STARTED:
        return
    if not _bambu_camera_enabled():
        return
    auto = False
    try:
        import bobert_companion as _bc
        auto = bool(getattr(_bc, "BAMBU_CAMERA_AUTO_WHILE_PRINTING", False))
    except Exception:
        pass
    if not auto:
        return
    t = threading.Thread(
        target=_bambu_camera_watcher,
        name="BambuCameraWatcher",
        daemon=True,
    )
    t.start()
    _BAMBU_CAMERA_WATCHER_STARTED = True


def _act_bambu_camera_on(_: str = "") -> str:
    global _BAMBU_CAMERA_USER_OFF
    _BAMBU_CAMERA_USER_OFF = False
    ok, msg = _launch_bambu_camera_hud()
    return msg if ok else f"REFUSED: {msg}"


def _act_bambu_camera_off(_: str = "") -> str:
    global _BAMBU_CAMERA_USER_OFF
    _BAMBU_CAMERA_USER_OFF = True
    ok, msg = _shutdown_bambu_camera_hud()
    return msg if ok else f"REFUSED: {msg}"


def _act_bambu_camera_toggle(_: str = "") -> str:
    if _bambu_camera_hud_is_alive():
        return _act_bambu_camera_off("")
    return _act_bambu_camera_on("")


def _act_bambu_camera_status(_: str = "") -> str:
    if not _bambu_camera_enabled():
        return ("The printer camera is disabled in settings, sir "
                "(HUD_BAMBU_CAMERA).")
    # Surface the grabber's view of the feed when we can read it.
    feed = ""
    try:
        import importlib
        cam = importlib.import_module("core.bambu_camera")
        st = cam.get_status()
        if st.get("ok"):
            feed = f" Feed is live via the {st.get('path') or 'local'} path."
        elif st.get("last_error"):
            feed = f" Feed status: {st.get('last_error')}."
    except Exception:
        pass
    if _bambu_camera_hud_is_alive():
        x, y, w, h = _resolve_bambu_camera_hud_geometry()
        return (f"The printer camera is on screen, sir — {w}x{h} at "
                f"({x}, {y}).{feed}")
    if _BAMBU_CAMERA_USER_OFF:
        return ("The printer camera is dismissed (manual), sir. "
                "Say `show the printer camera` to bring it back.")
    return f"The printer camera is off, sir.{feed}"


# ──────────────────────────────────────────────────────────────────────────
#  Workshop HUD — persistent top-right corner widget with arc reactor
#  power %, CPU/RAM bars, Bambu progress, rotating monitoring status.
# ──────────────────────────────────────────────────────────────────────────


def _workshop_hud_is_alive() -> bool:
    proc = _WORKSHOP_HUD_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_workshop_hud_control(**updates) -> None:
    """Atomic-write the workshop-HUD control file. The running widget
    polls this every tick so we can ask it to retire cleanly without
    a kill signal racing the tkinter loop."""
    try:
        existing = {}
        if os.path.exists(_WORKSHOP_HUD_CONTROL_FILE):
            try:
                with open(_WORKSHOP_HUD_CONTROL_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _WORKSHOP_HUD_CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _WORKSHOP_HUD_CONTROL_FILE)
    except Exception:
        # Control file is a nice-to-have — terminate() is the fallback.
        pass


def _resolve_workshop_hud_geometry() -> tuple[int, int, int, int]:
    """Anchor the workshop HUD to the very top-right of the top monitor.
    Spec calls for "persistent top-right corner widget on the top monitor"."""
    mx, my, mw, mh = _get_monitor_rect()
    w = _WORKSHOP_HUD_W
    h = _WORKSHOP_HUD_H
    x = mx + mw - w - _WORKSHOP_HUD_MARGIN
    y = my + _WORKSHOP_HUD_MARGIN
    return x, y, w, h


def _launch_workshop_hud() -> tuple[bool, str]:
    """Spawn the workshop HUD subprocess. Idempotent."""
    global _WORKSHOP_HUD_PROCESS
    with _WORKSHOP_HUD_LOCK:
        if _workshop_hud_is_alive():
            _write_workshop_hud_control(mode="on")
            return True, "The workshop HUD is already engaged, sir."
        if not os.path.exists(_WORKSHOP_HUD_SCRIPT):
            return False, (f"I'm afraid the workshop HUD script is "
                           f"missing — {_WORKSHOP_HUD_SCRIPT}.")
        # Clear any stale "off" entry so the new subprocess doesn't
        # immediately read it and exit.
        _write_workshop_hud_control(mode="on")
        x, y, w, h = _resolve_workshop_hud_geometry()
        try:
            _WORKSHOP_HUD_PROCESS = subprocess.Popen(
                [sys.executable, _WORKSHOP_HUD_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _WORKSHOP_HUD_PROCESS = None
            return False, f"I'm afraid the workshop HUD failed to launch, sir — {e}."
        return True, "Workshop HUD engaged, sir."


def _shutdown_workshop_hud() -> tuple[bool, str]:
    """Terminate the workshop HUD subprocess if it's running."""
    global _WORKSHOP_HUD_PROCESS
    with _WORKSHOP_HUD_LOCK:
        # Belt-and-braces: ask the widget to self-exit via its control
        # file AND send terminate() to cover any case where it missed
        # the latest poll.
        _write_workshop_hud_control(mode="off")
        if not _workshop_hud_is_alive():
            _WORKSHOP_HUD_PROCESS = None
            return True, "The workshop HUD isn't currently engaged, sir."
        try:
            _WORKSHOP_HUD_PROCESS.terminate()
            try:
                _WORKSHOP_HUD_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _WORKSHOP_HUD_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _WORKSHOP_HUD_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _WORKSHOP_HUD_PROCESS = None
        return True, "Workshop HUD dismissed, sir."


def _act_workshop_hud_on(_: str = "") -> str:
    ok, msg = _launch_workshop_hud()
    return msg if ok else f"REFUSED: {msg}"


def _act_workshop_hud_off(_: str = "") -> str:
    ok, msg = _shutdown_workshop_hud()
    return msg if ok else f"REFUSED: {msg}"


def _act_workshop_hud_toggle(_: str = "") -> str:
    if _workshop_hud_is_alive():
        return _act_workshop_hud_off("")
    return _act_workshop_hud_on("")


def _act_workshop_hud_status(_: str = "") -> str:
    if _workshop_hud_is_alive():
        x, y, w, h = _resolve_workshop_hud_geometry()
        return (f"The workshop HUD is engaged on the top monitor, sir — "
                f"{w}x{h} at ({x}, {y}).")
    return "The workshop HUD is not currently engaged, sir."


# ──────────────────────────────────────────────────────────────────────────
#  Workshop print monitor HUD — top-center Stark panel that surfaces
#  while a print is active OR a CAD app is focused.
# ──────────────────────────────────────────────────────────────────────────


def _workshop_print_monitor_is_alive() -> bool:
    proc = _WORKSHOP_PRINT_MONITOR_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_workshop_print_monitor_control(**updates) -> None:
    """Atomic-write the workshop-print-monitor control file. The widget
    polls this every tick so the watcher can ask it to retire cleanly
    without racing terminate()."""
    try:
        existing = {}
        if os.path.exists(_WORKSHOP_PRINT_MONITOR_CONTROL_FILE):
            try:
                with open(_WORKSHOP_PRINT_MONITOR_CONTROL_FILE, "r",
                          encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _WORKSHOP_PRINT_MONITOR_CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _WORKSHOP_PRINT_MONITOR_CONTROL_FILE)
    except Exception:
        # Control file is best-effort — terminate() is the fallback.
        pass


def _resolve_workshop_print_monitor_geometry() -> tuple[int, int, int, int]:
    """Anchor the print monitor to the top-center of the top monitor.
    Sibling widgets own the corners (workshop_hud + bambu_h2d_overlay on
    the top-right, arc_reactor_status_hud on the top-left), so the
    center strip is the only collision-free region."""
    mx, my, mw, _mh = _get_monitor_rect()
    w = _WORKSHOP_PRINT_MONITOR_W
    h = _WORKSHOP_PRINT_MONITOR_H
    x = mx + (mw - w) // 2
    y = my + _WORKSHOP_PRINT_MONITOR_MARGIN
    return x, y, w, h


def _launch_workshop_print_monitor() -> tuple[bool, str]:
    """Spawn the workshop print monitor subprocess. Idempotent."""
    global _WORKSHOP_PRINT_MONITOR_PROCESS
    with _WORKSHOP_PRINT_MONITOR_LOCK:
        if _workshop_print_monitor_is_alive():
            _write_workshop_print_monitor_control(mode="on")
            return True, "The workshop print monitor is already engaged, sir."
        if not os.path.exists(_WORKSHOP_PRINT_MONITOR_SCRIPT):
            return False, (
                f"I'm afraid the workshop print monitor script is "
                f"missing — {_WORKSHOP_PRINT_MONITOR_SCRIPT}."
            )
        # Clear any stale "off" entry so the new subprocess doesn't
        # immediately read it and exit.
        _write_workshop_print_monitor_control(mode="on")
        x, y, w, h = _resolve_workshop_print_monitor_geometry()
        try:
            _WORKSHOP_PRINT_MONITOR_PROCESS = subprocess.Popen(
                [sys.executable, _WORKSHOP_PRINT_MONITOR_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _WORKSHOP_PRINT_MONITOR_PROCESS = None
            return False, (
                f"I'm afraid the workshop print monitor failed to "
                f"launch, sir — {e}."
            )
        return True, "Workshop print monitor engaged, sir."


def _shutdown_workshop_print_monitor() -> tuple[bool, str]:
    """Terminate the workshop print monitor if it's running."""
    global _WORKSHOP_PRINT_MONITOR_PROCESS
    with _WORKSHOP_PRINT_MONITOR_LOCK:
        # Belt-and-braces: ask the widget to self-exit via its control
        # file AND send terminate() as backup for missed polls.
        _write_workshop_print_monitor_control(mode="off")
        if not _workshop_print_monitor_is_alive():
            _WORKSHOP_PRINT_MONITOR_PROCESS = None
            return True, "The workshop print monitor isn't currently engaged, sir."
        try:
            _WORKSHOP_PRINT_MONITOR_PROCESS.terminate()
            try:
                _WORKSHOP_PRINT_MONITOR_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _WORKSHOP_PRINT_MONITOR_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _WORKSHOP_PRINT_MONITOR_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _WORKSHOP_PRINT_MONITOR_PROCESS = None
        return True, "Workshop print monitor dismissed, sir."


def _workshop_mode_is_active() -> bool:
    """Read workshop_mode._workshop_active without taking its lock.

    The poll loop is read-only here — we tolerate a stale value for one
    tick rather than acquire workshop_mode._state_lock and block its
    own poll thread. Returns False on any failure (module not loaded,
    attribute missing) so the watcher leans towards inaction."""
    try:
        # workshop_mode is registered as a top-level skill module, but
        # its actual module name in sys.modules depends on the loader.
        # Try the most common forms.
        mod = sys.modules.get("skills.workshop_mode") or sys.modules.get(
            "workshop_mode"
        )
        if mod is None:
            try:
                import importlib
                mod = importlib.import_module("skills.workshop_mode")
            except Exception:
                try:
                    import importlib
                    mod = importlib.import_module("workshop_mode")
                except Exception:
                    mod = None
        if mod is None:
            return False
        active_cell = getattr(mod, "_workshop_active", None)
        if active_cell is None:
            return False
        try:
            return bool(active_cell[0])
        except (TypeError, IndexError):
            return bool(active_cell)
    except Exception:
        return False


def _bambu_print_is_active_for_monitor() -> bool:
    """True when bambu_monitor reports the H2D is in a state worth
    surfacing in the workshop print monitor.

    Reads via bambu_overlay_state.json — that's the file bambu_monitor
    already writes atomically on every MQTT report, and avoids us
    importing the live `_state` dict (which would mean threading a lock
    from a foreign module into this watcher)."""
    state = _read_bambu_overlay_state()
    if not state:
        return False
    return _bambu_is_active(state)


def _workshop_print_monitor_watcher() -> None:
    """Background thread: spawn the workshop print monitor when EITHER
    bambu_monitor reports an active print OR workshop_mode flags a CAD
    window, and tear it down after _WORKSHOP_PRINT_MONITOR_LINGER_S of
    sustained idle. Polls at 1 Hz — bambu push cadence is ~per-minute
    and workshop_mode polls at 30s, so anything tighter is wasted CPU."""
    last_active_at = 0.0
    saw_active_once = False
    while not _WORKSHOP_PRINT_MONITOR_WATCHER_STOP.wait(1.0):
        try:
            printer_active = _bambu_print_is_active_for_monitor()
            workshop_active = _workshop_mode_is_active()
            active = printer_active or workshop_active
            now = time.time()
            if active:
                last_active_at = now
                saw_active_once = True
                if (not _WORKSHOP_PRINT_MONITOR_USER_OFF
                        and not _workshop_print_monitor_is_alive()):
                    _launch_workshop_print_monitor()
            else:
                # Linger so the user catches the final state before the
                # widget retires.
                if saw_active_once and _workshop_print_monitor_is_alive():
                    if (now - last_active_at) >= _WORKSHOP_PRINT_MONITOR_LINGER_S:
                        _shutdown_workshop_print_monitor()
                        saw_active_once = False
                        # Re-arm auto-show for the next session.
                        _clear_workshop_print_monitor_user_off()
        except Exception:
            logging.exception(
                "workshop print monitor watcher iteration failed"
            )


def _clear_workshop_print_monitor_user_off() -> None:
    global _WORKSHOP_PRINT_MONITOR_USER_OFF
    _WORKSHOP_PRINT_MONITOR_USER_OFF = False


def _maybe_start_workshop_print_monitor_watcher() -> None:
    """Start the watcher exactly once. Respects the
    WORKSHOP_PRINT_MONITOR_AUTO_LAUNCH flag (default True)."""
    global _WORKSHOP_PRINT_MONITOR_WATCHER_STARTED
    if _WORKSHOP_PRINT_MONITOR_WATCHER_STARTED:
        return
    enabled = True
    try:
        import bobert_companion as _bc
        enabled = bool(
            getattr(_bc, "WORKSHOP_PRINT_MONITOR_AUTO_LAUNCH", True)
        )
    except Exception:
        pass
    if not enabled:
        return
    t = threading.Thread(
        target=_workshop_print_monitor_watcher,
        name="WorkshopPrintMonitorWatcher",
        daemon=True,
    )
    t.start()
    _WORKSHOP_PRINT_MONITOR_WATCHER_STARTED = True


def _act_workshop_print_monitor_on(_: str = "") -> str:
    global _WORKSHOP_PRINT_MONITOR_USER_OFF
    _WORKSHOP_PRINT_MONITOR_USER_OFF = False
    ok, msg = _launch_workshop_print_monitor()
    return msg if ok else f"REFUSED: {msg}"


def _act_workshop_print_monitor_off(_: str = "") -> str:
    global _WORKSHOP_PRINT_MONITOR_USER_OFF
    _WORKSHOP_PRINT_MONITOR_USER_OFF = True
    ok, msg = _shutdown_workshop_print_monitor()
    return msg if ok else f"REFUSED: {msg}"


def _act_workshop_print_monitor_toggle(_: str = "") -> str:
    if _workshop_print_monitor_is_alive():
        return _act_workshop_print_monitor_off("")
    return _act_workshop_print_monitor_on("")


def _act_workshop_print_monitor_status(_: str = "") -> str:
    printer_active = _bambu_print_is_active_for_monitor()
    workshop_active = _workshop_mode_is_active()
    if _workshop_print_monitor_is_alive():
        x, y, w, h = _resolve_workshop_print_monitor_geometry()
        ctx = []
        if printer_active:
            ctx.append("printer active")
        if workshop_active:
            ctx.append("workshop mode")
        ctx_str = (f" ({', '.join(ctx)})" if ctx else "")
        return (
            f"The workshop print monitor is engaged on the top monitor, "
            f"sir — {w}x{h} at ({x}, {y}){ctx_str}."
        )
    if _WORKSHOP_PRINT_MONITOR_USER_OFF:
        return (
            "The workshop print monitor is dismissed (manual), sir. "
            "Say `workshop print monitor on` to re-engage it."
        )
    if printer_active or workshop_active:
        return (
            "The workshop print monitor is initialising, sir."
        )
    return "The workshop print monitor is dormant, sir."


# ──────────────────────────────────────────────────────────────────────────
#  Holographic HUD v2 — PyQt6 arc-reactor ring (separate subprocess).
# ──────────────────────────────────────────────────────────────────────────


def _holo_hud_v2_is_alive() -> bool:
    proc = _HOLO_HUD_V2_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _resolve_holo_hud_v2_geometry() -> tuple[int, int, int, int]:
    """Center the HUD horizontally on the top monitor and anchor it just
    below the top edge. Falls back to a sensible rect when MONITORS isn't
    loaded yet."""
    mx, my, mw, mh = _get_monitor_rect()
    w = min(_HOLO_HUD_V2_W, max(400, int(mw * 0.45)))
    h = min(_HOLO_HUD_V2_H, max(420, int(mh * 0.6)))
    x = mx + (mw - w) // 2
    y = my + _HOLO_HUD_V2_MARGIN
    return x, y, w, h


def _launch_holo_hud_v2() -> tuple[bool, str]:
    """Spawn the PyQt6 HUD v2 subprocess. Idempotent."""
    global _HOLO_HUD_V2_PROCESS
    with _HOLO_HUD_V2_LOCK:
        if _holo_hud_v2_is_alive():
            return True, "The holographic HUD v2 is already engaged, sir."
        if not os.path.exists(_HOLO_HUD_V2_SCRIPT):
            return False, (f"I'm afraid the HUD v2 script is missing — "
                           f"{_HOLO_HUD_V2_SCRIPT}.")
        x, y, w, h = _resolve_holo_hud_v2_geometry()
        try:
            _HOLO_HUD_V2_PROCESS = subprocess.Popen(
                [sys.executable, _HOLO_HUD_V2_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _HOLO_HUD_V2_PROCESS = None
            return False, f"I'm afraid the HUD v2 failed to launch, sir — {e}."
        return True, "Holographic HUD v2 online, sir."


def _shutdown_holo_hud_v2() -> tuple[bool, str]:
    global _HOLO_HUD_V2_PROCESS
    with _HOLO_HUD_V2_LOCK:
        if not _holo_hud_v2_is_alive():
            _HOLO_HUD_V2_PROCESS = None
            return True, "The holographic HUD v2 isn't currently engaged, sir."
        try:
            _HOLO_HUD_V2_PROCESS.terminate()
            try:
                _HOLO_HUD_V2_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _HOLO_HUD_V2_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _HOLO_HUD_V2_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _HOLO_HUD_V2_PROCESS = None
        return True, "Holographic HUD v2 dismissed, sir."


def _act_holo_hud_v2_on(_: str = "") -> str:
    ok, msg = _launch_holo_hud_v2()
    return msg if ok else f"REFUSED: {msg}"


def _act_holo_hud_v2_off(_: str = "") -> str:
    ok, msg = _shutdown_holo_hud_v2()
    return msg if ok else f"REFUSED: {msg}"


def _act_holo_hud_v2_toggle(_: str = "") -> str:
    if _holo_hud_v2_is_alive():
        return _act_holo_hud_v2_off("")
    return _act_holo_hud_v2_on("")


def _act_holo_hud_v2_status(_: str = "") -> str:
    if _holo_hud_v2_is_alive():
        x, y, w, h = _resolve_holo_hud_v2_geometry()
        return (f"The holographic HUD v2 is engaged on the top monitor, sir — "
                f"{w}x{h} at ({x}, {y}).")
    return "The holographic HUD v2 is not currently engaged, sir."


# ──────────────────────────────────────────────────────────────────────────
#  Arc-reactor status HUD — four-quadrant ring (CPU/RAM/GPU/NET) + Bambu
#  inner ring, replacing the generic status_panel strip with an
#  MCU-authentic visual.
# ──────────────────────────────────────────────────────────────────────────


def _arc_status_is_alive() -> bool:
    proc = _ARC_STATUS_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_arc_status_control(**updates) -> None:
    """Atomic-write the arc-reactor-status control file so the running
    widget polls a mode change without us racing terminate()."""
    try:
        existing = {}
        if os.path.exists(_ARC_STATUS_CONTROL_FILE):
            try:
                with open(_ARC_STATUS_CONTROL_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _ARC_STATUS_CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _ARC_STATUS_CONTROL_FILE)
    except Exception:
        # Control file is a nice-to-have; terminate() is the fallback.
        pass


def _resolve_arc_status_geometry() -> tuple[int, int, int, int]:
    """Anchor the arc-reactor status HUD to the top-left of the top
    monitor. The workshop HUD + bambu overlay occupy the top-right
    column, so the left edge is the only collision-free corner."""
    mx, my, _mw, _mh = _get_monitor_rect()
    w = _ARC_STATUS_W
    h = _ARC_STATUS_H
    x = mx + _ARC_STATUS_MARGIN
    y = my + _ARC_STATUS_MARGIN
    return x, y, w, h


def _launch_arc_status() -> tuple[bool, str]:
    """Spawn the arc-reactor status HUD subprocess. Idempotent."""
    global _ARC_STATUS_PROCESS
    with _ARC_STATUS_LOCK:
        if _arc_status_is_alive():
            _write_arc_status_control(mode="on")
            return True, "The arc reactor status HUD is already engaged, sir."
        if not os.path.exists(_ARC_STATUS_SCRIPT):
            return False, (f"I'm afraid the arc reactor status HUD script is "
                           f"missing — {_ARC_STATUS_SCRIPT}.")
        # Clear any stale "off" entry so the new subprocess doesn't
        # immediately read it and exit.
        _write_arc_status_control(mode="on")
        x, y, w, h = _resolve_arc_status_geometry()
        try:
            _ARC_STATUS_PROCESS = subprocess.Popen(
                [sys.executable, _ARC_STATUS_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _ARC_STATUS_PROCESS = None
            return False, (f"I'm afraid the arc reactor status HUD failed to "
                           f"launch, sir — {e}.")
        return True, "Arc reactor status HUD online, sir."


def _shutdown_arc_status() -> tuple[bool, str]:
    """Terminate the arc-reactor status HUD if it's running."""
    global _ARC_STATUS_PROCESS
    with _ARC_STATUS_LOCK:
        # Belt-and-braces: ask the widget to self-exit via its control
        # file AND send terminate() to cover any case where it missed
        # the latest poll.
        _write_arc_status_control(mode="off")
        if not _arc_status_is_alive():
            _ARC_STATUS_PROCESS = None
            return True, "The arc reactor status HUD isn't currently engaged, sir."
        try:
            _ARC_STATUS_PROCESS.terminate()
            try:
                _ARC_STATUS_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _ARC_STATUS_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _ARC_STATUS_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _ARC_STATUS_PROCESS = None
        return True, "Arc reactor status HUD dismissed, sir."


def _act_arc_status_on(_: str = "") -> str:
    ok, msg = _launch_arc_status()
    return msg if ok else f"REFUSED: {msg}"


def _act_arc_status_off(_: str = "") -> str:
    ok, msg = _shutdown_arc_status()
    return msg if ok else f"REFUSED: {msg}"


def _act_arc_status_toggle(_: str = "") -> str:
    if _arc_status_is_alive():
        return _act_arc_status_off("")
    return _act_arc_status_on("")


def _act_arc_status_status(_: str = "") -> str:
    if _arc_status_is_alive():
        x, y, w, h = _resolve_arc_status_geometry()
        return (f"The arc reactor status HUD is engaged on the top monitor, sir — "
                f"{w}x{h} at ({x}, {y}).")
    return "The arc reactor status HUD is not currently engaged, sir."


# ──────────────────────────────────────────────────────────────────────────
#  Stark status ring (hud_v2) — top-center arc reactor with CPU/RAM/GPU,
#  now-playing, next calendar event, Bambu %, and a glowing speech-state
#  core. Spec: jarvis_todo.md 2026-05-30 05:23 (overnight). Sibling to
#  arc_reactor_status_hud (top-left) but visually distinct — full Stark
#  reactor disc with a four-region peripheral layout (track top-center,
#  calendar bottom-center, bambu inner ring, speech-state pulse core).
# ──────────────────────────────────────────────────────────────────────────
_STARK_STATUS_PROCESS = None
_STARK_STATUS_LOCK = threading.Lock()
# Renderer lives inside the package next to this __init__.py.
_STARK_STATUS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hud_v2.py",
)
_STARK_STATUS_CONTROL_FILE = os.path.join(
    _PROJECT_DIR, "stark_status_ring_state.json",
)
# Wider than the arc_reactor_status_hud disc so the text rows (track,
# calendar) read cleanly. Slim height keeps it from masking the workspace.
_STARK_STATUS_W = 460
_STARK_STATUS_H = 340
# Anchored top-center of the top monitor. workshop_print_monitor (also
# top-center) is 400×200 and only surfaces during workshop sessions; this
# ring is for general status and stacks below the print monitor when both
# happen to be alive (offset added in _resolve_stark_status_geometry).
_STARK_STATUS_MARGIN = 18


def _stark_status_is_alive() -> bool:
    proc = _STARK_STATUS_PROCESS
    if proc is None:
        return False
    return proc.poll() is None


def _write_stark_status_control(**updates) -> None:
    """Atomic-write the control file so the renderer can self-exit
    cleanly without us racing terminate()."""
    try:
        existing = {}
        if os.path.exists(_STARK_STATUS_CONTROL_FILE):
            try:
                with open(_STARK_STATUS_CONTROL_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(updates)
        tmp = _STARK_STATUS_CONTROL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        os.replace(tmp, _STARK_STATUS_CONTROL_FILE)
    except Exception:
        # Control file is a nice-to-have; terminate() is the fallback.
        pass


def _resolve_stark_status_geometry() -> tuple[int, int, int, int]:
    """Top-center of the top monitor. Stacks below the workshop print
    monitor if that sibling is alive so the two don't collide."""
    mx, my, mw, _mh = _get_monitor_rect()
    w = _STARK_STATUS_W
    h = _STARK_STATUS_H
    x = mx + (mw - w) // 2
    y = my + _STARK_STATUS_MARGIN
    if _workshop_print_monitor_is_alive():
        y += _WORKSHOP_PRINT_MONITOR_H + 12
    return x, y, w, h


def _launch_stark_status() -> tuple[bool, str]:
    """Spawn the hud_v2 renderer subprocess. Idempotent."""
    global _STARK_STATUS_PROCESS
    with _STARK_STATUS_LOCK:
        if _stark_status_is_alive():
            _write_stark_status_control(mode="on")
            return True, "The Stark status ring is already engaged, sir."
        if not os.path.exists(_STARK_STATUS_SCRIPT):
            return False, (f"I'm afraid the Stark status ring script is "
                           f"missing — {_STARK_STATUS_SCRIPT}.")
        _write_stark_status_control(mode="on")
        x, y, w, h = _resolve_stark_status_geometry()
        try:
            _STARK_STATUS_PROCESS = subprocess.Popen(
                [sys.executable, _STARK_STATUS_SCRIPT,
                 "--x", str(x), "--y", str(y),
                 "--width", str(w), "--height", str(h),
                 "--parent-pid", str(os.getpid())],
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0),
                close_fds=True,
            )
        except Exception as e:
            _STARK_STATUS_PROCESS = None
            return False, f"I'm afraid the Stark status ring failed to launch, sir — {e}."
        return True, "Stark status ring online, sir."


def _shutdown_stark_status() -> tuple[bool, str]:
    global _STARK_STATUS_PROCESS
    with _STARK_STATUS_LOCK:
        _write_stark_status_control(mode="off")
        if not _stark_status_is_alive():
            _STARK_STATUS_PROCESS = None
            return True, "The Stark status ring isn't currently engaged, sir."
        try:
            _STARK_STATUS_PROCESS.terminate()
            try:
                _STARK_STATUS_PROCESS.wait(timeout=2.0)
            except Exception:
                # See _shutdown_overlay — kill() may race the process
                # exiting; wait() after kill() ensures handle release.
                try:
                    _STARK_STATUS_PROCESS.kill()
                except Exception:
                    pass
                try:
                    _STARK_STATUS_PROCESS.wait(timeout=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        _STARK_STATUS_PROCESS = None
        return True, "Stark status ring dismissed, sir."


def _act_stark_status_on(_: str = "") -> str:
    ok, msg = _launch_stark_status()
    return msg if ok else f"REFUSED: {msg}"


def _act_stark_status_off(_: str = "") -> str:
    ok, msg = _shutdown_stark_status()
    return msg if ok else f"REFUSED: {msg}"


def _act_stark_status_toggle(_: str = "") -> str:
    if _stark_status_is_alive():
        return _act_stark_status_off("")
    return _act_stark_status_on("")


def _act_stark_status_status(_: str = "") -> str:
    if _stark_status_is_alive():
        x, y, w, h = _resolve_stark_status_geometry()
        return (f"The Stark status ring is engaged on the top monitor, sir — "
                f"{w}x{h} at ({x}, {y}).")
    return "The Stark status ring is not currently engaged, sir."


def register(actions: dict):
    """Wire show/hide/toggle actions for the fullscreen overlay AND the
    workshop canvas arc-reactor commands, then optionally auto-launch
    either or both."""
    # ── fullscreen overlay (the original heavy-weight HUD) ──
    actions["show_holographic_overlay"]   = _act_show
    actions["show_holo"]                  = _act_show
    actions["hud_on"]                     = _act_show
    actions["holographic_on"]             = _act_show

    actions["hide_holographic_overlay"]   = _act_hide
    actions["hide_holo"]                  = _act_hide
    actions["hud_off"]                    = _act_hide
    actions["dismiss_holo"]               = _act_hide
    actions["holographic_off"]            = _act_hide

    actions["toggle_holographic_overlay"] = _act_toggle
    actions["toggle_holo"]                = _act_toggle

    actions["holographic_status"]         = _act_status

    # ── workshop canvas (compact 3D-style reactor) ──
    # `arc_reactor` is now the workshop-canvas dispatcher (`on|off|pulse`)
    # per the spec; the fullscreen overlay still has its own aliases
    # above for users who want the cinematic version.
    actions["arc_reactor"]                = _act_arc_reactor
    actions["arc_reactor_on"]             = _act_arc_reactor_on
    actions["arc_reactor_off"]            = _act_arc_reactor_off
    actions["arc_reactor_pulse"]          = _act_arc_reactor_pulse
    actions["holo_workshop_canvas"]       = _act_arc_reactor_on
    actions["holo_workshop"]              = _act_arc_reactor_on
    actions["workshop_canvas"]            = _act_arc_reactor_on

    # ── optional auto-launch of the fullscreen overlay ──
    auto = False
    try:
        import bobert_companion as _bc
        auto = bool(getattr(_bc, "HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH", False))
    except Exception:
        pass
    if auto:
        ok, msg = _launch_overlay()
        print(f"  [holographic_overlay] auto-launch: {msg}")

    # ── auto-show watcher for the workshop canvas ──
    # Default ON: surfaces the reactor whenever JARVIS is thinking or
    # speaking. The spec explicitly asks for this visual presence beyond
    # the static status strip.
    _maybe_start_auto_watcher()

    # ── bambu H2D overlay extension ──
    # Spec (2026-05-27 11:26 bambu_h2d_overlay): pin a layer/ETA/temp
    # widget to the top-right of the top monitor while a print is
    # active. Manual on/off and toggle for explicit control; the watcher
    # thread auto-shows/auto-hides based on bambu_monitor's state.
    actions["bambu_h2d_overlay"]    = _act_bambu_overlay_toggle
    actions["bambu_overlay"]        = _act_bambu_overlay_toggle
    actions["bambu_overlay_on"]     = _act_bambu_overlay_on
    actions["show_bambu_overlay"]   = _act_bambu_overlay_on
    actions["bambu_overlay_off"]    = _act_bambu_overlay_off
    actions["hide_bambu_overlay"]   = _act_bambu_overlay_off
    actions["bambu_overlay_toggle"] = _act_bambu_overlay_toggle
    actions["bambu_overlay_status"] = _act_bambu_overlay_status
    _maybe_start_bambu_watcher()

    # ── bambu chamber-camera HUD ──
    # Spec (HUD Bambu-printer-camera view, 2026-06): a movable PyQt6 panel
    # showing the printer's built-in camera feed (RTSPS for the H2D, JPEG
    # fallback for P1/A1) with a print-status footer. Gated by the
    # HUD_BAMBU_CAMERA config flag. On-demand by default; auto-shows while a
    # print is active only when BAMBU_CAMERA_AUTO_WHILE_PRINTING is set.
    actions["bambu_camera"]          = _act_bambu_camera_toggle
    actions["bambu_camera_on"]       = _act_bambu_camera_on
    actions["bambu_camera_off"]      = _act_bambu_camera_off
    actions["bambu_camera_toggle"]   = _act_bambu_camera_toggle
    actions["bambu_camera_status"]   = _act_bambu_camera_status
    # Friendlier natural-language aliases — the LLM phrases this a dozen ways.
    actions["printer_camera"]        = _act_bambu_camera_toggle
    actions["show_printer_camera"]   = _act_bambu_camera_on
    actions["show_bambu_camera"]     = _act_bambu_camera_on
    actions["hide_printer_camera"]   = _act_bambu_camera_off
    actions["hide_bambu_camera"]     = _act_bambu_camera_off
    actions["print_camera"]          = _act_bambu_camera_toggle
    actions["printer_cam"]           = _act_bambu_camera_toggle
    actions["show_print_camera"]     = _act_bambu_camera_on
    actions["camera_hud"]            = _act_bambu_camera_toggle
    _maybe_start_bambu_camera_watcher()

    # ── workshop HUD (persistent corner widget) ──
    # Spec (2026-05-27 12:15 workshop_hud): a slim always-on-top widget
    # pinned to the top-right of the top monitor with arc-reactor power,
    # CPU/RAM bars, Bambu progress, and a rotating monitoring status
    # line. `hide_hud` voice command is the user-facing dismiss alias
    # called out by the spec.
    actions["workshop_hud"]         = _act_workshop_hud_toggle
    actions["show_workshop_hud"]    = _act_workshop_hud_on
    actions["workshop_hud_on"]      = _act_workshop_hud_on
    actions["workshop_hud_off"]     = _act_workshop_hud_off
    actions["workshop_hud_toggle"]  = _act_workshop_hud_toggle
    actions["workshop_hud_status"]  = _act_workshop_hud_status
    actions["hide_workshop_hud"]    = _act_workshop_hud_off
    # NOTE: do NOT register "hide_hud" here — that name is owned by the
    # bobert_companion `_act_hide_hud` action which hides the main status-bar
    # HUD overlay (see bobert_companion.py:7544). Skills load AFTER the main
    # ACTIONS dict is built, so registering "hide_hud" here used to silently
    # overwrite the main one and route 'hide the HUD' voice commands to the
    # workshop-HUD closer instead. The dedicated workshop alias is above.

    workshop_hud_auto = True
    try:
        import bobert_companion as _bc
        workshop_hud_auto = bool(getattr(_bc, "WORKSHOP_HUD_AUTO_LAUNCH", True))
    except Exception:
        pass
    if workshop_hud_auto:
        ok, msg = _launch_workshop_hud()
        print(f"  [holographic_overlay] workshop_hud auto-launch: {msg}")

    # ── workshop print monitor (top-center Stark panel) ──
    # Spec (2026-05-29 23:27 overnight): a persistent visual surface for
    # the H2D print state whenever the user is in the workshop OR a
    # print is active. The watcher polls bambu_overlay_state.json +
    # workshop_mode._workshop_active at 1 Hz and auto-shows the widget.
    actions["workshop_print_monitor"]        = _act_workshop_print_monitor_toggle
    actions["workshop_print_monitor_on"]     = _act_workshop_print_monitor_on
    actions["show_workshop_print_monitor"]   = _act_workshop_print_monitor_on
    actions["workshop_print_monitor_off"]    = _act_workshop_print_monitor_off
    actions["hide_workshop_print_monitor"]   = _act_workshop_print_monitor_off
    actions["workshop_print_monitor_toggle"] = _act_workshop_print_monitor_toggle
    actions["workshop_print_monitor_status"] = _act_workshop_print_monitor_status
    # Friendlier voice aliases — the LLM occasionally rephrases "workshop
    # print monitor" as "print HUD" / "workshop hud monitor" / etc.
    actions["print_hud"]                     = _act_workshop_print_monitor_toggle
    actions["print_hud_on"]                  = _act_workshop_print_monitor_on
    actions["print_hud_off"]                 = _act_workshop_print_monitor_off
    actions["workshop_print_hud"]            = _act_workshop_print_monitor_toggle
    actions["workshop_print_hud_on"]         = _act_workshop_print_monitor_on
    actions["workshop_print_hud_off"]        = _act_workshop_print_monitor_off

    _maybe_start_workshop_print_monitor_watcher()

    # ── holographic HUD v2 (PyQt6 arc-reactor ring) ──
    # Spec (2026-05-29 09:18 holographic_hud_v2): a permanent Iron Man
    # arc-reactor ring showing CPU/RAM, current intent, last action,
    # ambient state, and a rolling 5-line transcript. Independent of v1.
    actions["holographic_hud_v2"]        = _act_holo_hud_v2_toggle
    actions["holo_hud_v2"]               = _act_holo_hud_v2_toggle
    actions["holo_hud_v2_on"]            = _act_holo_hud_v2_on
    actions["show_holo_hud_v2"]          = _act_holo_hud_v2_on
    actions["holo_hud_v2_off"]           = _act_holo_hud_v2_off
    actions["hide_holo_hud_v2"]          = _act_holo_hud_v2_off
    actions["holo_hud_v2_toggle"]        = _act_holo_hud_v2_toggle
    actions["holo_hud_v2_status"]        = _act_holo_hud_v2_status
    actions["arc_reactor_hud"]           = _act_holo_hud_v2_on

    holo_hud_v2_auto = False
    try:
        import bobert_companion as _bc
        holo_hud_v2_auto = bool(getattr(_bc, "HOLO_HUD_V2_AUTO_LAUNCH", False))
    except Exception:
        pass
    if holo_hud_v2_auto:
        ok, msg = _launch_holo_hud_v2()
        print(f"  [holographic_overlay] holo_hud_v2 auto-launch: {msg}")

    # ── arc-reactor status HUD (4-quadrant ring + Bambu inner ring) ──
    # Spec (2026-05-29 16:35 overnight): replaces the generic
    # status_panel_strip with system_pulse data drawn as four arcs
    # around a central core + an inner Bambu print-progress ring.
    # Independent toggle; doesn't disturb jarvis_hud.py or status_panel.
    actions["arc_reactor_status_hud"]    = _act_arc_status_toggle
    actions["arc_reactor_status"]        = _act_arc_status_toggle
    actions["arc_reactor_status_on"]     = _act_arc_status_on
    actions["arc_reactor_status_off"]    = _act_arc_status_off
    actions["arc_reactor_status_toggle"] = _act_arc_status_toggle
    actions["arc_reactor_status_status"] = _act_arc_status_status
    actions["status_hud"]                = _act_arc_status_toggle
    actions["status_hud_on"]             = _act_arc_status_on
    actions["show_status_hud"]           = _act_arc_status_on
    actions["status_hud_off"]            = _act_arc_status_off
    actions["hide_status_hud"]           = _act_arc_status_off
    actions["status_ring"]               = _act_arc_status_toggle
    actions["status_ring_on"]            = _act_arc_status_on
    actions["status_ring_off"]           = _act_arc_status_off
    actions["pulse_hud"]                 = _act_arc_status_toggle
    actions["pulse_hud_on"]              = _act_arc_status_on
    actions["pulse_hud_off"]             = _act_arc_status_off

    arc_status_auto = False
    try:
        import bobert_companion as _bc
        arc_status_auto = bool(
            getattr(_bc, "HOLO_ARC_REACTOR_STATUS_AUTO_LAUNCH", False)
        )
    except Exception:
        pass
    if arc_status_auto:
        ok, msg = _launch_arc_status()
        print(f"  [holographic_overlay] arc_reactor_status auto-launch: {msg}")

    # ── Stark status ring (hud_v2) — top-center reactor with CPU/RAM/GPU,
    #    now-playing, next calendar event, Bambu %, and a glowing
    #    speech-state core. Spec: jarvis_todo.md 2026-05-30 05:23
    #    (overnight). Independent toggle so it stacks cleanly alongside
    #    the existing arc-reactor surfaces.
    actions["stark_status_ring"]        = _act_stark_status_toggle
    actions["stark_status_ring_on"]     = _act_stark_status_on
    actions["stark_status_ring_off"]    = _act_stark_status_off
    actions["stark_status_ring_toggle"] = _act_stark_status_toggle
    actions["stark_status_ring_status"] = _act_stark_status_status
    actions["hud_v2"]                   = _act_stark_status_toggle
    actions["hud_v2_on"]                = _act_stark_status_on
    actions["show_hud_v2"]              = _act_stark_status_on
    actions["hud_v2_off"]               = _act_stark_status_off
    actions["hide_hud_v2"]              = _act_stark_status_off
    actions["hud_v2_toggle"]            = _act_stark_status_toggle
    actions["status_ring_v2"]           = _act_stark_status_toggle
    actions["status_ring_v2_on"]        = _act_stark_status_on
    actions["status_ring_v2_off"]       = _act_stark_status_off
    actions["show_status_ring_v2"]      = _act_stark_status_on
    actions["hide_status_ring_v2"]      = _act_stark_status_off

    stark_status_auto = False
    try:
        import bobert_companion as _bc
        stark_status_auto = bool(
            getattr(_bc, "HOLO_STARK_STATUS_RING_AUTO_LAUNCH", False)
        )
    except Exception:
        pass
    if stark_status_auto:
        ok, msg = _launch_stark_status()
        print(f"  [holographic_overlay] stark_status_ring auto-launch: {msg}")
