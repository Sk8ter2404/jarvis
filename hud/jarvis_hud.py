#!/usr/bin/env python3
"""
JARVIS HUD — animated arc-reactor status ring (corner overlay).

Run as a subprocess spawned by bobert_companion.py at startup. Reads
hud_state.json (sibling to bobert_companion.py) at the animation tick
rate to display:

  • center arc-reactor disc — color + pulse driven by JARVIS state
    (spec contract — jarvis_todo.md 2026-05-27 07:31, arc_reactor_hud):
      Idle      → slow cyan pulse
      Listening → faster pulse, WHITE
      Thinking  → spinning BLUE ticks at high spin rate
      Speaking  → ring BRIGHTENS WITH TTS AMPLITUDE (blue→bright-blue)
      Standby   → very slow violet drift (preserved from prior pass)
      ALERT     → RED FLASH (CPU≥90 / RAM≥90 / explicit `alert` field)

      When an action is executing, the center label shows a spinner
      that "spells out" the action name char-by-char.
  • four RADIAL readouts around the ring:
      N (top)    — CPU %
      W (left)   — RAM %
      E (right)  — current focused window (shortened)
      S (bottom) — last heard user transcript (truncated)
  • outer ring  — CPU %
  • middle ring — RAM %
  • inner ring  — live mic input level (0.0–1.0)
  • rotating SHIELD-style tick marks around each ring
  • activity_ring (top-left corner) — small arc-reactor-style "what is
    JARVIS doing RIGHT NOW" glyph (spec contract — jarvis_todo.md
    2026-05-27 10:04, activity_ring). Distinct from the main ring; the
    visual signature changes per state so a quick glance at the corner
    tells the user the mode without parsing the full HUD:
      idle      → slow BLUE pulse (breathing circle)
      listening → CYAN waveform drawn across the disc
      thinking  → BLUE arc segment rotating around the ring
      speaking  → BLUE concentric pulses radiating outward (amp-modulated)
      acting    → AMBER lightning bolt glyph in the center
      standby   → very slow violet drift (consistent with main ring)
      ALERT     → red flash with '!' centre glyph
  • ticker zone below the rings:
      line 1 — currently-running action (with spinner) OR last action ago
      line 2 — overnight-mode countdown (h:mm) when overnight is active
      line 3 — pulse_strip from skills/system_pulse.py
      line 4 — status_panel_strip from skills/status_panel.py

The window is click-through on its transparent background (Win32
-transparentcolor) AND draggable by grabbing the ring/text — the
drag handlers preserve the corner-anchor default until the user
moves it.

Closes cleanly when its parent process exits (the parent_pid argument lets
us detect that).

CLI:
  python hud/jarvis_hud.py --x 0 --y 0 --width 2560 --parent-pid 12345
"""
import argparse
import json
import math
import os
import sys
import time
import tkinter as tk

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pygetwindow as _gw
    _HAS_GW = True
except Exception:
    _HAS_GW = False

# ──────────────────────────────────────────────────────────────────────────
#  Layout
# ──────────────────────────────────────────────────────────────────────────
HUD_W      = 400            # canvas width (widened to fit E/W radial text)
HUD_H      = 460            # canvas height
RING_CX    = HUD_W / 2      # x-center of the ring
RING_CY    = 180            # y-center of the ring (leaves room below for ticker)
ANCHOR_PAD = 8              # px gap from monitor edge in the corner
TICK_MS    = 50             # animation tick (~20 fps)

R_OUTER    = 142            # CPU ring radius
R_MID      = 118            # RAM ring radius
R_INNER    = 96             # mic ring radius
R_CORE     = 60             # arc-reactor core radius
R_CORE_IN  = 26             # inner hub radius

# activity_ring — compact state indicator pinned to the top-left corner.
# Lives off the canvas's outer rings so it never collides with the boot
# breadcrumbs (centered, y≈10) or the explicit_alert slot (top center, y=8).
ACT_CX     = 36
ACT_CY     = 36
ACT_R      = 18             # outer radius of the activity glyph

# Bottom ticker zone — sits below the S radial readout
S_READOUT_Y         = RING_CY + R_OUTER + 14    # last_transcript (S radial)
TICKER_Y_TOP        = S_READOUT_Y + 24          # current/last action line
TICKER_Y_BOTTOM     = TICKER_Y_TOP + 18         # overnight countdown
PULSE_STRIP_Y       = TICKER_Y_TOP + 36         # system_pulse skill widget
STATUS_PANEL_Y      = TICKER_Y_TOP + 54         # status_panel skill widget

# ──────────────────────────────────────────────────────────────────────────
#  Palette
#    Spec mandates (per task arc_reactor_hud):
#      idle=cyan, listening=white, thinking=blue ticks,
#      speaking=blue brightening with TTS amplitude, error=red FLASH.
#    Standby (violet) preserved from prior pass since it reads
#    distinctly without conflicting with the spec.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY        = "#010101"   # near-black, used as -transparentcolor on Win32
PANEL_COLOR   = "#04080d"
CYAN          = "#4cc9ff"
CYAN_DIM      = "#1b4a66"
CYAN_BRIGHT   = "#9ee7ff"
TEXT_COLOR    = "#cfeefb"
DIM_TEXT      = "#5d8aa3"
WHITE         = "#f0f4f8"   # spec: "listening=faster white"
WHITE_DIM     = "#7a8995"
BLUE          = "#3da5ff"   # spec: "thinking=spinning blue ticks"
BLUE_DIM      = "#1c5994"
BLUE_BRIGHT   = "#a8d9ff"   # spec: "speaking=ring brightens with TTS amplitude"
GOLD          = "#ffd166"   # retained for running-action label + timer/pulse accents
GOLD_DIM      = "#a67b1f"
GOLD_GLOW     = "#e9b94a"
GREEN         = "#9ff9c4"
VIOLET        = "#9b8cff"
ALERT         = "#ff5b5b"
ALERT_DIM     = "#5a1414"   # darker than before so the flash reads stronger

STATE_FILE_NAME   = "hud_state.json"
CONFIG_FILE_NAME  = "hud_config.json"    # persisted position + scale
CONTROL_FILE_NAME = "jarvis_hud_control.json"  # user-driven hide flag (own file)
PROJECT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE       = os.path.join(PROJECT_DIR, STATE_FILE_NAME)
CONFIG_FILE      = os.path.join(PROJECT_DIR, CONFIG_FILE_NAME)
CONTROL_FILE     = os.path.join(PROJECT_DIR, CONTROL_FILE_NAME)

# Shared atomic JSON writer. This HUD runs as a standalone subprocess, so
# `core` may not be on the path yet — add PROJECT_DIR and fall back to a
# local mkstemp+replace if the import still fails, so a missing module can
# never crash the overlay.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
try:
    from core.atomic_io import _atomic_write_json
except Exception:  # pragma: no cover - exercised only without the core pkg
    import tempfile

    def _atomic_write_json(path, data, *, indent=2):
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    # AUTHORITATIVE liveness first (2026-07-14, audit finding): psutil.pid_exists
    # reads TRUE for a DEAD-but-unreaped Windows process — a kernel-stuck
    # "terminating forever" row keeps its PID until reboot — so an overlay that
    # trusts it outlives its dead parent (a HUD + tray once survived by 25
    # minutes). core.parent_watch asks the kernel (GetExitCodeProcess +
    # WaitForSingleObject). Fail-open: if the helper is unavailable we fall
    # through to the historical checks below rather than tearing the overlay down.
    try:
        from core.parent_watch import parent_is_alive
        return parent_is_alive(pid)
    except Exception:
        pass
    if _HAS_PSUTIL:
        # pid_exists can raise on Windows for a transient handle/permission
        # error; treat an unknowable parent as alive so a hiccup can't freeze
        # the render loop (matches the PyQt HUDs' guard).
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _sf(value, default: float = 0.0) -> float:
    """Guarded float() — returns `default` on any non-numeric value.

    The shared hud_state.json is written by another process; a malformed or
    unexpectedly-typed field must never raise inside a Tkinter `after()`
    callback, or the reschedule at the end of tick() would never run and the
    overlay would freeze permanently."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_hud_config() -> dict:
    """Load persisted HUD geometry (x, y, scale). Returns {} if missing/corrupt."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_hud_config(data: dict) -> None:
    """Atomic temp+rename write so a partial flush can't corrupt the file."""
    try:
        _atomic_write_json(CONFIG_FILE, data)
    except Exception:
        pass


def _read_hud_control() -> dict:
    """Load the dedicated HUD control file (user-driven hide flag).

    Kept separate from hud_state.json — the main process rewrites that
    canonical snapshot continuously, so a read-modify-write against it from
    this subprocess both races the writer (torn write) and is instantly
    overwritten on the next tick. The control file is owned by this HUD."""
    if not os.path.exists(CONTROL_FILE):
        return {}
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_hud_control(data: dict) -> None:
    """Atomic write of the HUD control file via the shared helper."""
    try:
        _atomic_write_json(CONTROL_FILE, data)
    except Exception:
        pass


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, c)) for c in rgb))


def _mix(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two hex colors. t=0 → c1, t=1 → c2."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    ))


def _state_color(jarvis_state: str) -> str:
    """Base ring color per spec — used when alert isn't overriding everything.
    Speaking returns the bright variant; brightness modulation by TTS amplitude
    happens at the call site so we can mix between BLUE and BLUE_BRIGHT."""
    s = (jarvis_state or "").lower()
    if s == "listening":
        return WHITE
    if s == "thinking":
        return BLUE
    if s == "speaking":
        return BLUE_BRIGHT
    if s == "standby":
        return VIOLET
    return CYAN  # idle / unknown


# State → (ring spin rate, halo arc spin rate, pulse frequency, pulse amplitude).
#   idle      → drifts gently, soft cyan pulse (spec: "slow cyan pulse")
#   listening → FASTER pulse than idle (spec: "faster white")
#   thinking  → fast spin AND fast pulse (spec: "spinning blue ticks")
#   speaking  → moderate spin, amplitude-driven brightness (handled at draw site)
#   standby   → near-still
#   alert     → fast strobe (spec: "red flash")
_STATE_ANIM = {
    "idle":      (1.0 / 55.0, 3, 0.12, 3.0),
    "listening": (1.0 / 22.0, 5, 0.32, 4.5),
    "thinking":  (1.0 /  9.0, 12, 0.34, 6.0),
    "speaking":  (1.0 / 22.0, 5, 0.24, 5.0),
    "standby":   (1.0 / 80.0, 2, 0.05, 1.4),
    "alert":     (1.0 / 14.0, 8, 0.55, 7.0),
}


# Window-title shorteners — keep the E radial readout legible without
# truncating to a meaningless prefix. Order matters (longest match wins).
_WINDOW_SHORTCUTS = [
    ("visual studio code", "VSCode"),
    ("vs code",            "VSCode"),
    ("bambu studio",       "Bambu"),
    ("autodesk fusion",    "Fusion"),
    ("fusion 360",         "Fusion"),
    ("microsoft teams",    "Teams"),
    ("microsoft edge",     "Edge"),
    ("google chrome",      "Chrome"),
    ("file explorer",      "Files"),
    ("windows terminal",   "Term"),
    ("powershell",         "PSh"),
    ("orcaslicer",         "Orca"),
    ("prusaslicer",        "Prusa"),
    ("solidworks",         "SolidW"),
    ("freecad",            "FreeCAD"),
    ("openscad",           "SCAD"),
    ("onenote",            "OneNote"),
    ("blender",            "Blender"),
    ("photoshop",          "PShop"),
    ("illustrator",        "Ai"),
    ("notepad",            "Notepad"),
    ("explorer",           "Files"),
    ("discord",            "Discord"),
    ("spotify",            "Spotify"),
    ("firefox",            "Firefox"),
    ("chrome",             "Chrome"),
    ("teams",              "Teams"),
    ("slack",              "Slack"),
    ("itunes",             "iTunes"),
    ("excel",              "Excel"),
    ("word",               "Word"),
    ("outlook",            "Outlook"),
]


def _shorten_window(title: str, limit: int = 8) -> str:
    if not title:
        return ""
    lt = title.lower()
    for needle, label in _WINDOW_SHORTCUTS:
        if needle in lt:
            return label
    # Fallback: take first token, strip separators, truncate to limit
    first = title
    for sep in (" — ", " - ", " | ", " – "):
        if sep in first:
            first = first.split(sep)[-1].strip() or first
    first = first.strip()
    if not first:
        return "?"
    return first[:limit]


def _ago(seconds: float) -> str:
    if seconds < 1:
        return "now"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _fmt_countdown(secs: int) -> str:
    if secs <= 0:
        return "0m"
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class HUD:
    def __init__(self, x: int, y: int, width: int, parent_pid: int,
                 role: str = "prod"):
        self.parent_pid = parent_pid
        # blue-green-2: "staging" → render a cyan-bordered STAGING badge in
        # the top-right corner and a small test-progress overlay so the user
        # can watch the ceremony on the LEFT monitor. "prod" → existing
        # behaviour (no badge).
        self.role       = "staging" if role == "staging" else "prod"
        self.frame      = 0
        self.last_cpu   = 0.0
        self.last_ram   = 0.0
        self.last_mic   = 0.0   # smoothed mic level
        self.last_amp   = 0.0   # smoothed TTS amplitude (for speaking brightness)
        # Focused window is polled at 1 Hz inside the HUD itself rather than
        # being published over the JSON state file — pygetwindow is local
        # and lightweight, and skipping the IPC keeps the writer side simple.
        self._focused_window      = ""
        self._focused_check_at    = 0.0

        # Center spinner reveal-state — track when the action changed so we
        # can re-spell-out the name from scratch each time.
        self._action_at_start     = ""
        self._action_reveal_frame = 0

        # Drag bookkeeping — populated by _on_drag_start.
        self._drag_origin_x = 0
        self._drag_origin_y = 0

        # User-driven scale factor — applied to HUD_W/HUD_H via mouse wheel
        # (Ctrl+wheel for fine control). 1.0 = launch-time default size.
        # Clamped 0.5–2.5 so it can't accidentally vanish or fill the screen.
        self._scale = 1.0
        self._scale_min = 0.5
        self._scale_max = 2.5

        # Whether the user actually moved the window during this drag (set by
        # B1-Motion). _on_drag_end only persists when this is True so a plain
        # click on the ring doesn't rewrite the same coords every time.
        self._drag_moved = False

        # Hidden state — driven by the `visible` field in hud_state.json (so
        # the main script can ask JARVIS to hide the HUD without killing the
        # subprocess) AND by the user-driven menu hide persisted in the
        # dedicated control file. `_user_hidden` mirrors the control file so a
        # menu hide survives ticks; it's cleared when JARVIS issues an
        # explicit show (visible flips False->True). `_prev_state_visible`
        # tracks that transition.
        self._hidden = False
        self._user_hidden = bool(_read_hud_control().get("hidden"))
        self._prev_state_visible = None
        # Set by _on_close so the tick() wrapper stops rescheduling after the
        # Tk root is destroyed (a post-destroy after() would raise).
        self._closing = False

        # Load persisted geometry from hud_config.json. We read this BEFORE
        # creating the Tk window so the canvas comes up at the right size and
        # in the right place from frame 0, rather than visibly snapping after
        # the user already sees the default-anchored position.
        default_win_x = x + width - HUD_W - ANCHOR_PAD
        default_win_y = y + ANCHOR_PAD
        cfg = _read_hud_config()
        try:
            self._scale = max(self._scale_min,
                              min(self._scale_max, float(cfg.get("scale", 1.0))))
        except (TypeError, ValueError):
            self._scale = 1.0
        try:
            win_x = int(cfg.get("x", default_win_x))
            win_y = int(cfg.get("y", default_win_y))
        except (TypeError, ValueError):
            win_x, win_y = default_win_x, default_win_y
        # Scaled window dimensions from persisted scale
        start_w = int(HUD_W * self._scale)
        start_h = int(HUD_H * self._scale)

        self.root = tk.Tk()
        self.root.title("JARVIS HUD")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Windows: use -transparentcolor so the dark background reads as a
        # floating overlay (click-through on the keyed color). Falls back to
        # plain alpha on other platforms.
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.88)
            except Exception:
                pass

        # Window geometry — saved coords (or top-right anchor of the configured
        # monitor) plus scaled width/height from persisted scale.
        self.root.geometry(f"{start_w}x{start_h}+{win_x}+{win_y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=start_w, height=start_h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Spec: "Click-through, always-on-top, draggable."
        # transparentcolor already makes the keyed background pixels
        # click-through; the ring/text pixels still receive events, so
        # binding to <Button-1> on the canvas lets the user grab the
        # ring itself to drag the window. <Double-Button-1> re-anchors
        # to the top-right so the user can recover the default position
        # without restarting.
        self.canvas.bind("<Button-1>",        self._on_drag_start)
        self.canvas.bind("<B1-Motion>",       self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", self._on_drag_reset)
        # Resize: Ctrl+mouse wheel scales the whole HUD smoothly. Plain wheel
        # is ignored so accidental scrolls over the HUD don't change its size.
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_resize)
        # Right-click context menu — hide / show / reset / resize options.
        self.canvas.bind("<Button-3>",        self._on_right_click)
        # Keyboard shortcut: Ctrl+0 resets to anchor + scale 1.0
        self.root.bind("<Control-Key-0>",     self._on_drag_reset)
        # Reset anchor — ALWAYS the default top-right at scale 1.0, NOT the
        # saved position. Double-click / Ctrl+0 means "go back to the original
        # spot", not "go back to wherever I last dropped it".
        self._anchor_geom = f"{HUD_W}x{HUD_H}+{default_win_x}+{default_win_y}"
        self._anchor_x = default_win_x
        self._anchor_y = default_win_y

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Prime psutil so the first cpu_percent() returns a real value
        if _HAS_PSUTIL:
            try: psutil.cpu_percent(interval=None)
            except Exception: pass

        # Per-state phase accumulator so the ring keeps spinning smoothly
        # across state transitions instead of snapping back to phase=0 when
        # the spin rate changes.
        self._phase      = 0.0
        self._halo_phase = 0.0

        self.tick()

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass

    # ─── drag handling ──────────────────────────────────────────────────
    def _on_drag_start(self, event):
        # Record click offset relative to current window origin so the
        # window doesn't snap its top-left to the cursor on first motion.
        self._drag_origin_x = event.x_root - self.root.winfo_x()
        self._drag_origin_y = event.y_root - self.root.winfo_y()
        self._drag_moved = False

    def _on_drag_motion(self, event):
        new_x = event.x_root - self._drag_origin_x
        new_y = event.y_root - self._drag_origin_y
        # Don't include the WxH each time — bare +X+Y is a position-only move.
        self.root.geometry(f"+{new_x}+{new_y}")
        self._drag_moved = True

    def _on_drag_end(self, _event=None):
        # Persist the new corner only if the user actually dragged (a plain
        # click on the ring shouldn't rewrite the same coords every time).
        if self._drag_moved:
            self._drag_moved = False
            self._save_geometry()

    def _on_drag_reset(self, _event=None):
        # Return to the launch-time top-right anchor + scale 1.0 on double-click
        # (or Ctrl+0). Resets both position AND size in one gesture, and also
        # resizes the canvas widget so the inner display matches.
        self._scale = 1.0
        try:
            self.root.geometry(self._anchor_geom)
            self.canvas.config(width=HUD_W, height=HUD_H)
        except Exception:
            pass
        # Persist the reset so the next launch comes up at the default anchor.
        self._save_geometry()

    # ─── resize handling (Ctrl + mouse wheel) ───────────────────────────
    def _on_wheel_resize(self, event):
        # Each wheel notch ~120 units on Windows. 10% per notch is a noticeable
        # but not jarring step. Clamp to [_scale_min, _scale_max].
        step = 0.10 * (1 if event.delta > 0 else -1)
        self._set_scale(self._scale + step, anchor_x=event.x_root, anchor_y=event.y_root)

    def _set_scale(self, new_scale: float, anchor_x: int = -1, anchor_y: int = -1):
        new_scale = max(self._scale_min, min(self._scale_max, new_scale))
        if abs(new_scale - self._scale) < 0.005:
            return
        self._scale = new_scale
        new_w = int(HUD_W * new_scale)
        new_h = int(HUD_H * new_scale)
        # Keep the corner under the cursor stable when resizing — feels natural.
        cur_x = self.root.winfo_x()
        cur_y = self.root.winfo_y()
        try:
            self.root.geometry(f"{new_w}x{new_h}+{cur_x}+{cur_y}")
            # Tell Tk's canvas to re-scale its coordinate system so the drawn
            # rings actually grow/shrink. self.canvas.scale() rescales the
            # display list; we set its size to match the new window dims.
            self.canvas.config(width=new_w, height=new_h)
            self.canvas.scale("all", 0, 0, new_scale / 1.0, new_scale / 1.0)
            # On the next tick the canvas is redrawn from scratch, so the
            # scale() above mostly affects this frame — but it prevents a
            # one-frame flicker between the old and new layout.
        except Exception:
            pass
        # Persist the new scale so it survives the next restart.
        self._save_geometry()

    # ─── persistence (hud_config.json) ──────────────────────────────────
    def _save_geometry(self):
        """Atomic write of current window x/y + scale to hud_config.json."""
        try:
            _write_hud_config({
                "x": self.root.winfo_x(),
                "y": self.root.winfo_y(),
                "scale": round(self._scale, 3),
            })
        except Exception:
            # Persistence is best-effort — a failed write just means the
            # next launch starts at the default anchor.
            pass

    # ─── right-click context menu ────────────────────────────────────────
    def _on_right_click(self, event):
        try:
            menu = tk.Menu(self.root, tearoff=0, bg=PANEL_COLOR, fg=TEXT_COLOR,
                           activebackground=CYAN_DIM, activeforeground=WHITE)
            menu.add_command(label="Reset position + size  (Ctrl+0)",
                             command=lambda: self._on_drag_reset(None))
            menu.add_separator()
            menu.add_command(label="Smaller  (Ctrl+wheel down)",
                             command=lambda: self._set_scale(self._scale - 0.15))
            menu.add_command(label="Larger  (Ctrl+wheel up)",
                             command=lambda: self._set_scale(self._scale + 0.15))
            menu.add_separator()
            menu.add_command(label="Hide HUD  (JARVIS will re-show it on request)",
                             command=self._hide_via_menu)
            menu.add_command(label="Close HUD permanently",
                             command=self._on_close)
            menu.tk_popup(event.x_root, event.y_root)
        except Exception:
            pass

    def _hide_via_menu(self):
        # Local, user-driven hide. Persist the choice in this HUD's OWN
        # control file — never hud_state.json, which the main process
        # rewrites continuously. Writing the hide flag there used to (a) race
        # the main writer (torn write of the canonical snapshot) and (b) get
        # instantly overwritten on the next tick, so the hide was both unsafe
        # and ineffective. The control file is read with precedence in tick()
        # and is cleared when JARVIS issues an explicit show.
        self._hidden = True
        self._user_hidden = True
        try:
            self.root.withdraw()
        except Exception:
            pass
        _write_hud_control({"hidden": True})

    def _poll_focused_window(self):
        """Refresh the cached focused-window title at most once per second."""
        if not _HAS_GW:
            return
        now = time.time()
        if (now - self._focused_check_at) < 1.0:
            return
        self._focused_check_at = now
        try:
            w = _gw.getActiveWindow()
            self._focused_window = (w.title if w else "") or ""
        except Exception:
            # pygetwindow on Windows occasionally raises on closing windows
            pass

    # ─── drawing helpers ────────────────────────────────────────────────
    def _bbox(self, radius: float):
        return (RING_CX - radius, RING_CY - radius,
                RING_CX + radius, RING_CY + radius)

    def _ring_track(self, radius: float, color: str = CYAN_DIM, width: int = 1):
        """Faint full-circle base for a data ring."""
        x1, y1, x2, y2 = self._bbox(radius)
        self.canvas.create_oval(x1, y1, x2, y2, outline=color, width=width)

    def _gauge_arc(self, radius: float, pct: float, color: str, width: int = 3):
        """Arc that sweeps clockwise from 12 o'clock proportional to pct (0–100)."""
        pct = max(0.0, min(100.0, pct))
        if pct <= 0.01:
            return
        x1, y1, x2, y2 = self._bbox(radius)
        # tkinter arcs measure angle counter-clockwise from 3 o'clock;
        # start=90 puts us at 12 o'clock, negative extent sweeps clockwise.
        extent = -3.6 * pct
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=90, extent=extent,
            style="arc", outline=color, width=width,
        )

    def _rotating_ticks(self, radius: float, count: int, color: str,
                        phase: float, tick_len: float = 4.0, width: int = 1):
        """SHIELD-style tick marks that rotate slowly around a ring."""
        for i in range(count):
            theta = (i / count) * 2 * math.pi + phase
            x1 = RING_CX + radius * math.cos(theta)
            y1 = RING_CY + radius * math.sin(theta)
            x2 = RING_CX + (radius + tick_len) * math.cos(theta)
            y2 = RING_CY + (radius + tick_len) * math.sin(theta)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _arc_segment(self, radius: float, start_deg: float, extent_deg: float,
                     color: str, width: int = 2):
        x1, y1, x2, y2 = self._bbox(radius)
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=start_deg, extent=extent_deg,
            style="arc", outline=color, width=width,
        )

    def _text(self, x: float, y: float, msg: str, color: str = TEXT_COLOR,
              size: int = 9, weight: str = "normal", anchor: str = "center"):
        self.canvas.create_text(
            x, y, text=msg, fill=color, anchor=anchor,
            font=("Consolas", size, weight),
        )

    # ─── boot-sequence power-up animation ───────────────────────────────
    # When boot_sequence.py is running its 4–5s spoken intro, the HUD
    # paints a concentric-rings power-up overlay INSTEAD of the normal
    # status rings. boot_phase clears when the boot routine completes,
    # AND the HUD self-clears the visual after duration + 0.5s so a
    # crashed parent can't strand the overlay.
    def _draw_boot_animation(self, started_at: float, duration: float):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, HUD_W, HUD_H, fill=BG_KEY, outline="")

        elapsed = time.time() - started_at
        progress = max(0.0, min(1.0, elapsed / max(0.1, duration)))

        # Rings fill sequentially from inside out. Slight overlap between
        # ring stages so the animation flows rather than stepping.
        # (radius, start_progress, end_progress, tick_count)
        rings = [
            (R_CORE + 6,  0.00, 0.30,  8),
            (R_INNER,     0.18, 0.55, 12),
            (R_MID,       0.40, 0.75, 18),
            (R_OUTER,     0.60, 0.95, 24),
        ]
        for r, p_start, p_end, tick_count in rings:
            rp = (progress - p_start) / max(0.01, p_end - p_start)
            rp = max(0.0, min(1.0, rp))
            if rp <= 0:
                continue
            self._ring_track(r, CYAN_DIM, 1)
            sweep_pct = 100.0 * rp
            self._gauge_arc(r, sweep_pct, CYAN_BRIGHT, width=4)
            # Counter-rotating shimmer ticks scaled by ring fill — gives
            # the rings the "spinning up to speed" feel.
            self._rotating_ticks(
                r, count=tick_count, color=CYAN,
                phase=self.frame * 0.18 * rp,
                tick_len=3.0, width=1,
            )

        # Central hub grows in size and brightens as the boot progresses.
        hub_pulse = 0.5 * (1 + math.sin(self.frame * 0.45))
        hub_radius = R_CORE_IN * (0.45 + 0.55 * progress) + 1.5 * hub_pulse
        hub_color = _mix(CYAN_DIM, CYAN_BRIGHT, progress)
        x1, y1, x2, y2 = self._bbox(hub_radius)
        self.canvas.create_oval(
            x1, y1, x2, y2, outline=hub_color, width=2,
            fill=PANEL_COLOR,
        )
        # Final flash — when progress hits the last 10%, the hub gets a
        # bright halo to mark "online".
        if progress >= 0.90:
            flash_r = hub_radius + 6 + 3 * hub_pulse
            fx1, fy1, fx2, fy2 = self._bbox(flash_r)
            self.canvas.create_oval(
                fx1, fy1, fx2, fy2, outline=CYAN_BRIGHT, width=2,
            )

        # Center label scrolls through three stages so the boot reads as an
        # actual sequence rather than a single status — INITIALISING up to
        # ~38%, DIAGNOSTICS through the middle, ONLINE for the final flash.
        # Matches the iron_man_boot.py contract (jarvis_todo.md 2026-05-27
        # 10:04, iron_man_boot).
        if progress < 0.38:
            label = "INITIALISING"
            label_color = CYAN
        elif progress < 0.78:
            label = "DIAGNOSTICS"
            label_color = CYAN
        else:
            label = "ONLINE"
            label_color = CYAN_BRIGHT
        self._text(RING_CX, RING_CY - 4, label,
                   color=label_color, size=11, weight="bold")
        self._text(RING_CX, RING_CY + 12, f"{int(progress * 100)}%",
                   color=DIM_TEXT, size=8)

        # Stage breadcrumbs — three small tags above the ring that brighten
        # one at a time as each stage activates. Gives the user a visual
        # 'three-step' read even on a quick glance.
        stage_y = RING_CY - R_OUTER - 28
        stages = [
            ("INIT",  progress >= 0.00),
            ("DIAG",  progress >= 0.38),
            ("ONLINE", progress >= 0.78),
        ]
        spacing = 68  # px between stage labels
        total = spacing * (len(stages) - 1)
        for i, (tag, lit) in enumerate(stages):
            sx = RING_CX - total / 2 + i * spacing
            sc = CYAN_BRIGHT if lit else CYAN_DIM
            self._text(sx, stage_y, tag, color=sc, size=8, weight="bold")

        # Bottom strip — a hint that this is the boot sequence (helps the
        # user distinguish from a normal alert / state change).
        self._text(RING_CX, TICKER_Y_TOP,
                   "J.A.R.V.I.S.  power-up sequence",
                   color=DIM_TEXT, size=8)

    # ─── activity_ring (top-left state glyph) ───────────────────────────
    # Compact "what is JARVIS doing RIGHT NOW" indicator. Drawn after the
    # main ring so it overlays cleanly if the user drags the HUD around.
    # Driven entirely from the published hud_state.json (state, active_action,
    # tts_amplitude, mic_level) — no separate IPC.
    def _draw_activity_ring(self, jarvis_state: str, active_action: str,
                            alert_active: bool, alert_color: str):
        cx, cy, r = ACT_CX, ACT_CY, ACT_R

        # Faint base ring sits behind everything so the glyph still reads as
        # a "reactor" disc even when the state is idle / no animation.
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=CYAN_DIM, width=1,
        )

        s = (jarvis_state or "").lower()

        if alert_active:
            # Red flash — synced to the main ring's 5 Hz cadence so the
            # whole HUD pulses together.
            flash_on = (self.frame // 4) % 2 == 0
            c = alert_color if flash_on else ALERT_DIM
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=c, width=2, fill=PANEL_COLOR,
            )
            self._text(cx, cy - 1, "!", color=c, size=14, weight="bold")

        elif active_action:
            # Acting → amber lightning bolt. The bolt shimmers (brightness +
            # width modulated by frame) so the glyph reads as "energy" rather
            # than a static icon. Spec: "acting=amber lightning glyph".
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=GOLD, width=2, fill=PANEL_COLOR,
            )
            shimmer = (self.frame // 3) % 2 == 0
            bolt_color = GOLD_GLOW if shimmer else GOLD
            bolt_width = 3 if shimmer else 2
            bolt_pts = [
                (cx - 4, cy - 10),
                (cx + 2, cy - 2),
                (cx - 3, cy + 1),
                (cx + 4, cy + 10),
            ]
            for i in range(len(bolt_pts) - 1):
                x1, y1 = bolt_pts[i]
                x2, y2 = bolt_pts[i + 1]
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=bolt_color, width=bolt_width, capstyle="round",
                )

        elif s == "listening":
            # CYAN waveform — small sinusoid drawn across the disc width,
            # tapered at the ends so it stays inside the circle. Phase
            # advances with frame so the wave appears to scroll.
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=CYAN, width=2, fill=PANEL_COLOR,
            )
            inner = r - 4
            pts: list[float] = []
            n = 20
            for i in range(n):
                t = i / (n - 1)
                x = cx - inner + t * 2 * inner
                envelope = 1.0 - abs(2 * t - 1)  # 0 at edges, 1 at middle
                y = cy + math.sin(t * 4 * math.pi + self.frame * 0.45) * 6 * envelope
                pts.extend([x, y])
            self.canvas.create_line(*pts, fill=CYAN_BRIGHT, width=2, smooth=True)

        elif s == "thinking":
            # Rotating ring — fast BLUE arc segment chasing around the disc,
            # plus a counter-rotating inner tick ring for depth. Spec:
            # "thinking=rotating ring".
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=BLUE_DIM, width=1, fill=PANEL_COLOR,
            )
            seg_start = (self.frame * 9) % 360
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=seg_start, extent=90,
                style="arc", outline=BLUE_BRIGHT, width=3,
            )
            inner_r = r - 7
            for i in range(6):
                theta = (i / 6.0) * 2 * math.pi - self.frame * 0.22
                ix1 = cx + (inner_r - 4) * math.cos(theta)
                iy1 = cy + (inner_r - 4) * math.sin(theta)
                ix2 = cx + inner_r * math.cos(theta)
                iy2 = cy + inner_r * math.sin(theta)
                self.canvas.create_line(ix1, iy1, ix2, iy2, fill=BLUE, width=1)

        elif s == "speaking":
            # Outgoing waveform — concentric rings expanding/fading. Each
            # ring is at a different phase so they appear to ripple outward
            # continuously. Brightness scales with TTS amplitude. Spec:
            # "speaking=outgoing waveform".
            amp = self.last_amp
            base = _mix(BLUE, BLUE_BRIGHT, amp)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=base, width=2, fill=PANEL_COLOR,
            )
            period = 30
            for i in range(3):
                p = ((self.frame + i * (period // 3)) % period) / float(period)
                pr = 3 + p * (r - 4)
                fade = 1.0 - p
                c = _mix(PANEL_COLOR, base, fade * (0.55 + 0.45 * amp))
                self.canvas.create_oval(
                    cx - pr, cy - pr, cx + pr, cy + pr,
                    outline=c, width=1,
                )

        elif s == "standby":
            # Very slow violet drift — matches the main-ring standby palette.
            pulse = 0.5 * (1 + math.sin(self.frame * 0.04))
            inner_r = (r - 6) + 2 * pulse
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=VIOLET, width=1, fill=PANEL_COLOR,
            )
            self.canvas.create_oval(
                cx - inner_r, cy - inner_r,
                cx + inner_r, cy + inner_r,
                outline=VIOLET, width=1,
            )

        else:
            # Idle → slow BLUE pulse. Breathing inner disc filled with a
            # color mix that brightens at the peak of each pulse. Spec:
            # "idle=slow blue pulse".
            pulse = 0.5 * (1 + math.sin(self.frame * 0.12))
            inner_r = (r - 9) + 4 * pulse
            color = _mix(BLUE_DIM, BLUE, pulse)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=BLUE, width=2, fill=PANEL_COLOR,
            )
            self.canvas.create_oval(
                cx - inner_r, cy - inner_r,
                cx + inner_r, cy + inner_r,
                outline=color, width=2, fill=color,
            )

        # Tiny label so a new viewer can identify the glyph without docs.
        self._text(cx, cy + r + 8, "ACT", color=DIM_TEXT, size=7, weight="bold")

    # ─── blue-green-2: STAGING badge + test progress overlay ────────────
    def _draw_staging_overlay(self, test_state: dict):
        """Render a small cyan-bordered 'STAGING vX.Y.Z' badge in the
        upper-right corner of the HUD and a one-line test-progress strip
        beneath it. No-ops when self.role != 'staging' (caller already
        gates on that)."""
        if self.role != "staging":
            return

        # Badge text. Pull version from hud_state ('staging_version') if
        # the writer published one; otherwise fall back to a generic label.
        try:
            ver = test_state.get("version") if isinstance(test_state, dict) else None
        except Exception:
            ver = None
        if not ver:
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    _data = json.load(f) or {}
                ver = _data.get("staging_version") or _data.get("version")
            except Exception:
                ver = None
        badge_text = f"STAGING  {ver}" if ver else "STAGING"

        # Position: top-right corner, kept clear of the existing alert slot
        # (which lives at top-center, y=8). Width is measured naively from
        # the text length so different version strings sit neatly.
        pad_x = 8
        pad_y = 4
        badge_w = max(70, 7 * len(badge_text) + 2 * pad_x)
        badge_h = 18
        x2 = HUD_W - 6
        x1 = x2 - badge_w
        y1 = 4
        y2 = y1 + badge_h
        # Filled panel + cyan border. Filling with PANEL_COLOR (not BG_KEY)
        # avoids the transparency keying so the badge text sits over a
        # solid disc rather than a click-through hole.
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=CYAN, width=2, fill=PANEL_COLOR,
        )
        self._text(
            (x1 + x2) // 2, (y1 + y2) // 2,
            badge_text, color=CYAN_BRIGHT, size=8, weight="bold",
        )

        # Test progress strip — drawn directly below the badge. Reads
        # current_test_case / tests_passed / tests_remaining from the
        # published test_state dict.
        if not isinstance(test_state, dict) or not test_state:
            return
        try:
            current = str(test_state.get("current_test_case") or "")[:32]
            passed  = int(test_state.get("tests_passed") or 0)
            remain  = int(test_state.get("tests_remaining") or 0)
        except (TypeError, ValueError):
            return
        total = max(1, passed + remain)
        # Status line: "3/5 ▸ are you the new one"
        status_line = f"{passed}/{total}"
        if current:
            status_line += f"  {current}"
        status_y = y2 + 6
        self._text(
            x2, status_y, status_line,
            color=TEXT_COLOR, size=8, anchor="e",
        )
        # Mini-progress bar beneath the status line — a 2-px-tall cyan bar
        # filling left→right with the fraction passed/total. Keeps the
        # ceremony readable from across the room.
        bar_x1 = x1
        bar_x2 = x2
        bar_y  = status_y + 8
        self.canvas.create_rectangle(
            bar_x1, bar_y, bar_x2, bar_y + 2,
            outline="", fill=CYAN_DIM,
        )
        fill_w = int((bar_x2 - bar_x1) * (passed / float(total)))
        if fill_w > 0:
            self.canvas.create_rectangle(
                bar_x1, bar_y, bar_x1 + fill_w, bar_y + 2,
                outline="", fill=CYAN_BRIGHT,
            )

    # ─── main paint ─────────────────────────────────────────────────────
    def tick(self):
        # Wrapper: the entire render runs in a guarded body, and the after()
        # reschedule lives in finally so a single bad value or transient error
        # can NEVER strand this overlay on a frozen frame. The body returns the
        # delay (ms) for the next tick, or None when it has closed the window.
        if self._closing:
            return
        delay = TICK_MS
        try:
            d = self._tick_body()
            if d is not None:
                delay = d
        except Exception:
            # Swallow and keep the loop alive at the normal cadence.
            pass
        if not self._closing:
            try:
                self.root.after(delay, self.tick)
            except Exception:
                pass

    def _tick_body(self):
        if not _is_parent_alive(self.parent_pid):
            self._on_close()
            return None

        state              = _read_state()

        # ── visibility check ────────────────────────────────────────────
        # Two independent hide sources, combined:
        #   1. The main script's `visible` field in hud_state.json (JARVIS
        #      `hide_hud`/`show_hud`).
        #   2. The user's menu "Hide HUD" persisted in our OWN control file
        #      (read each tick so it survives, and never written into the
        #      shared hud_state.json).
        # An explicit JARVIS show — `visible` rising False->True — clears the
        # user hide too, honouring the menu's "JARVIS will re-show it on
        # request" promise.
        state_visible = bool(state.get("visible", True))
        if (self._prev_state_visible is False and state_visible
                and self._user_hidden):
            self._user_hidden = False
            _write_hud_control({"hidden": False})
        self._prev_state_visible = state_visible
        # Re-read the control file so a hide issued by another process (or a
        # show that just cleared it) is reflected promptly.
        if not self._user_hidden:
            self._user_hidden = bool(_read_hud_control().get("hidden"))
        want_visible = state_visible and not self._user_hidden
        if not want_visible and not self._hidden:
            self._hidden = True
            try: self.root.withdraw()
            except Exception: pass
        elif want_visible and self._hidden:
            self._hidden = False
            try: self.root.deiconify()
            except Exception: pass
        if self._hidden:
            # Slow tick rate while hidden — 500 ms is plenty to notice when
            # the user asks JARVIS to show the HUD again.
            return 500

        jarvis_state       = state.get("state", "Idle")
        now_playing        = state.get("now_playing", "")
        timers             = state.get("timers", []) or []
        active_action      = state.get("active_action", "") or ""
        recent_action      = state.get("recent_action", "") or ""
        # Every float parse below is guarded (via _sf) so a non-numeric value
        # in the shared state file degrades to a sane fallback for this frame
        # instead of raising out of the callback.
        recent_action_at   = _sf(state.get("recent_action_at"), 0.0)
        overnight_expiry   = _sf(state.get("overnight_expiry"), 0.0)
        mic_level_raw      = _sf(state.get("mic_level"), 0.0)
        tts_amp_raw        = _sf(state.get("tts_amplitude"), 0.0)
        last_transcript    = state.get("last_transcript", "") or ""
        last_transcript_at = _sf(state.get("last_transcript_at"), 0.0)
        explicit_alert     = state.get("alert", "") or ""

        # ── boot-sequence override ──
        # When boot_sequence.play_boot_sequence() is running, take over the
        # canvas with the power-up animation. Self-clears 0.5s after the
        # advertised duration so a crashed parent can't strand the overlay.
        boot_phase       = state.get("boot_phase", "") or ""
        boot_started_at  = _sf(state.get("boot_started_at"), 0.0)
        boot_duration    = _sf(state.get("boot_duration"), 0.0)
        if (boot_phase == "powering" and boot_started_at > 0
                and boot_duration > 0
                and (time.time() - boot_started_at) <= (boot_duration + 0.5)):
            self._draw_boot_animation(boot_started_at, boot_duration)
            self.frame += 1
            return TICK_MS

        # CPU / RAM live from psutil; cache last value if read fails
        if _HAS_PSUTIL:
            try:
                self.last_cpu = psutil.cpu_percent(interval=None)
                self.last_ram = psutil.virtual_memory().percent
            except Exception:
                pass

        self._poll_focused_window()

        # Smooth mic + TTS amplitude so the gauges aren't jittery
        target_mic = max(0.0, min(1.0, mic_level_raw))
        self.last_mic = 0.55 * self.last_mic + 0.45 * target_mic
        target_amp = max(0.0, min(1.0, tts_amp_raw))
        self.last_amp = 0.45 * self.last_amp + 0.55 * target_amp

        # ── clear canvas ──
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, HUD_W, HUD_H, fill=BG_KEY, outline="",
        )

        # ── alert detection (spec: "error=red flash") ──
        # CPU/RAM thresholds match the system_monitor skill's alert bar;
        # explicit_alert lets future skills publish a one-shot red state.
        alert_active = (
            bool(explicit_alert)
            or self.last_cpu >= 90.0
            or self.last_ram >= 90.0
        )
        # Red FLASH — alternate between ALERT and ALERT_DIM at ~5 Hz so the
        # alert visually pulses instead of sitting at steady red.
        flash_on = (self.frame // 4) % 2 == 0  # 4 frames @ 50ms = 200ms ≈ 5 Hz toggle
        alert_main_color = ALERT if flash_on else ALERT_DIM

        # ── per-state animation parameters ──
        anim_key = "alert" if alert_active else (jarvis_state or "idle").lower()
        spin_step, halo_step, pulse_freq, pulse_amp = _STATE_ANIM.get(
            anim_key, _STATE_ANIM["idle"]
        )

        # Advance accumulators so state changes don't snap the ring back to 0
        self._phase      = (self._phase + spin_step) % (2 * math.pi)
        self._halo_phase = (self._halo_phase + halo_step) % 360
        phase = self._phase
        pulse = 0.5 * (1 + math.sin(self.frame * pulse_freq))

        # ── color routing ──
        # ALERT wins everything. Otherwise state color drives the core; the
        # ring gauges still flip individually red when they cross 90% so the
        # offending dimension is identifiable at a glance even if the whole
        # HUD is already in alert mode.
        s = (jarvis_state or "").lower()
        if alert_active:
            core_color = alert_main_color
            halo_color = alert_main_color
            ring_color = alert_main_color
        else:
            base_state_color = _state_color(jarvis_state)
            # Speaking: brighten color with TTS amplitude (spec). When amp is
            # 0 the ring sits at BLUE; as amp rises the color is mixed toward
            # BLUE_BRIGHT and the pulse_amp scale is boosted so the hub
            # visibly grows with louder speech.
            if s == "speaking":
                core_color = _mix(BLUE, BLUE_BRIGHT, self.last_amp)
                pulse_amp = pulse_amp * (0.6 + 0.8 * self.last_amp)
            else:
                core_color = base_state_color
            halo_color = core_color
            ring_color = CYAN

        cpu_alert = self.last_cpu >= 90.0
        ram_alert = self.last_ram >= 90.0
        cpu_color = alert_main_color if cpu_alert else (ring_color if alert_active else CYAN)
        ram_color = alert_main_color if ram_alert else (ring_color if alert_active else CYAN)

        # ── outer CPU ring ──
        self._ring_track(R_OUTER, ALERT_DIM if alert_active else CYAN_DIM, 1)
        self._gauge_arc(R_OUTER, self.last_cpu, cpu_color, width=3)
        self._rotating_ticks(R_OUTER, count=24,
                             color=ALERT_DIM if alert_active else CYAN_DIM,
                             phase=phase)

        # ── middle RAM ring ──
        self._ring_track(R_MID, ALERT_DIM if alert_active else CYAN_DIM, 1)
        self._gauge_arc(R_MID, self.last_ram, ram_color, width=3)
        self._rotating_ticks(R_MID, count=18,
                             color=ALERT_DIM if alert_active else CYAN_DIM,
                             phase=-phase * 0.7)

        # ── inner MIC ring ──
        self._ring_track(R_INNER, ALERT_DIM if alert_active else CYAN_DIM, 1)
        mic_pct = min(100.0, self.last_mic * 100.0)
        if alert_active:
            mic_color = alert_main_color
        elif s == "listening":
            mic_color = WHITE
        else:
            mic_color = CYAN
        self._gauge_arc(R_INNER, mic_pct, mic_color, width=3)
        self._rotating_ticks(R_INNER, count=12,
                             color=ALERT_DIM if alert_active else CYAN_DIM,
                             phase=phase * 1.4, tick_len=3.0)

        # ── thinking: prominent spinning BLUE ticks around the core ──
        # Spec: "thinking=spinning blue ticks". Render a dedicated ring of
        # short ticks just outside the core, spinning fast in BLUE_BRIGHT,
        # so the visual signature is unmistakable even when the ring
        # gauges are also moving.
        if s == "thinking" and not alert_active:
            self._rotating_ticks(R_CORE + 4, count=16,
                                 color=BLUE_BRIGHT,
                                 phase=phase * 2.2,
                                 tick_len=6.0, width=2)
            # Counter-rotating dim companion ticks for depth
            self._rotating_ticks(R_CORE + 12, count=8,
                                 color=BLUE_DIM,
                                 phase=-phase * 1.4,
                                 tick_len=4.0, width=1)

        # ── speaking glow halo — outer rings brighten with TTS amplitude ──
        if s == "speaking" and not alert_active:
            # Halo radius and inner-glow brightness both scale with amplitude
            glow_r = R_CORE + 8 + 6 * self.last_amp + 2 * pulse
            outer_glow_color = _mix(BLUE_DIM, BLUE_BRIGHT, self.last_amp)
            inner_glow_color = _mix(BLUE, BLUE_BRIGHT, min(1.0, self.last_amp * 1.4))
            self._arc_segment(glow_r,     0, 360, inner_glow_color, width=1)
            self._arc_segment(glow_r + 4, 0, 360, outer_glow_color, width=1)

        # Rotating halo arcs around the core
        for i in range(4):
            seg_start = (i * 90 + self._halo_phase) % 360
            self._arc_segment(R_CORE, seg_start, 60, halo_color, width=2)

        # ── inner hub — radius pulses on the per-state schedule ──
        hub_radius = R_CORE_IN + pulse_amp * pulse
        x1, y1, x2, y2 = self._bbox(hub_radius)
        self.canvas.create_oval(
            x1, y1, x2, y2, outline=core_color, width=2, fill=PANEL_COLOR,
        )

        # ── center spinner + typewriter for the active action ────────────
        # When an action is running, we reveal the name char-by-char over
        # ~ N frames while a unicode spinner glyph rotates next to it.
        # When idle, the center reads back the JARVIS state label.
        if active_action:
            if active_action != self._action_at_start:
                self._action_at_start     = active_action
                self._action_reveal_frame = self.frame
            chars_revealed = min(
                len(active_action),
                (self.frame - self._action_reveal_frame) // 2 + 1,
            )
            revealed = active_action[:chars_revealed].upper()
            spinner_glyphs = "◐◓◑◒"
            spin_glyph = spinner_glyphs[self.frame // 3 % len(spinner_glyphs)]
            label = f"{spin_glyph} {revealed[:14]}"
            label_color = alert_main_color if alert_active else GOLD
        else:
            self._action_at_start     = ""
            self._action_reveal_frame = 0
            label = (jarvis_state or "Idle").upper()
            label_color = alert_main_color if alert_active else core_color

        self._text(RING_CX, RING_CY - 4, label,
                   color=label_color, size=10, weight="bold")
        self._text(RING_CX, RING_CY + 10, time.strftime("%H:%M:%S"),
                   color=DIM_TEXT, size=8)

        # ── FOUR RADIAL READOUTS ─────────────────────────────────────────
        # N: CPU%, W: RAM%, E: focused window, S: last heard transcript
        self._text(RING_CX, RING_CY - R_OUTER - 12,
                   f"CPU {self.last_cpu:>4.0f}%",
                   color=cpu_color, size=9, weight="bold")
        self._text(RING_CX - R_OUTER - 6, RING_CY,
                   f"RAM\n{self.last_ram:>3.0f}%",
                   color=ram_color, size=9, weight="bold", anchor="e")

        # E radial — focused window.
        win_label = _shorten_window(self._focused_window, limit=8) or "—"
        win_color = TEXT_COLOR if self._focused_window else DIM_TEXT
        self._text(RING_CX + R_OUTER + 6, RING_CY,
                   f"WIN\n{win_label}",
                   color=win_color, size=9, weight="bold", anchor="w")

        # S radial — last user transcript.
        if last_transcript:
            disp = last_transcript[:42]
            if len(last_transcript) > 42:
                disp = disp + "…"
            transcript_text = f'▸ "{disp}"'
            age = time.time() - last_transcript_at if last_transcript_at else 0
            t_color = TEXT_COLOR if age < 300 else DIM_TEXT
        else:
            transcript_text = "▸ awaiting input"
            t_color = DIM_TEXT
        self._text(RING_CX, S_READOUT_Y, transcript_text,
                   color=t_color, size=9, anchor="n")

        # ── inside-ring strip (now-playing / next timer, when no action) ──
        if not active_action:
            if now_playing:
                self._text(RING_CX, RING_CY + R_OUTER - 8,
                           f"♪ {now_playing[:28]}",
                           color=DIM_TEXT, size=9)
            elif timers:
                t = timers[0]
                secs = int(t.get("remaining", 0))
                if secs > 0:
                    if secs < 60:
                        rem = f"{secs}s"
                    elif secs < 3600:
                        rem = f"{secs // 60}m{secs % 60:02d}s"
                    else:
                        rem = f"{secs // 3600}h{(secs % 3600) // 60:02d}m"
                    tlabel = (t.get("label", "") or "")[:18]
                    self._text(RING_CX, RING_CY + R_OUTER - 8,
                               f"⏱ {rem} {tlabel}".rstrip(),
                               color=GOLD, size=9)

        # ── ticker line 1: current/last action ───────────────────────────
        if active_action:
            spinner = "◐◓◑◒"[self.frame // 4 % 4]
            ticker_msg   = f"{spinner} running: {active_action[:22]}"
            ticker_color = GOLD
        elif recent_action and recent_action_at > 0:
            ago = time.time() - recent_action_at
            ticker_msg = f"• last: {recent_action[:18]} ({_ago(ago)})"
            ticker_color = TEXT_COLOR if ago < 300 else DIM_TEXT
        else:
            ticker_msg   = "• awaiting input"
            ticker_color = DIM_TEXT
        self._text(RING_CX, TICKER_Y_TOP, ticker_msg,
                   color=ticker_color, size=9)

        # ── ticker line 2: overnight countdown ───────────────────────────
        if overnight_expiry > 0:
            secs_left = int(overnight_expiry - time.time())
            if secs_left > 0:
                self._text(RING_CX, TICKER_Y_BOTTOM,
                           f"⏳ OVERNIGHT  {_fmt_countdown(secs_left)}",
                           color=VIOLET, size=9, weight="bold")
            else:
                self._text(RING_CX, TICKER_Y_BOTTOM,
                           "⏳ OVERNIGHT  expired",
                           color=GOLD_DIM, size=9)

        # ── pulse strip: system_pulse skill widget ───────────────────────
        pulse_strip = state.get("pulse_strip", "") or ""
        if pulse_strip:
            self._text(RING_CX, PULSE_STRIP_Y, pulse_strip,
                       color=DIM_TEXT, size=8)

        # ── status panel strip: status_panel skill widget ────────────────
        status_panel_strip = state.get("status_panel_strip", "") or ""
        if status_panel_strip:
            self._text(RING_CX, STATUS_PANEL_Y, status_panel_strip,
                       color=DIM_TEXT, size=8)

        # ── explicit alert message, if any ───────────────────────────────
        # Renders in a small dedicated slot above the ring so a CPU/RAM
        # alert isn't the only signal — published skills can name the issue.
        if explicit_alert:
            self._text(RING_CX, 8, explicit_alert[:48],
                       color=alert_main_color, size=8, weight="bold", anchor="n")

        # ── activity_ring (corner state glyph) ───────────────────────────
        # Drawn last so it always sits on top of any ring/text overlap if
        # the user shrinks the canvas. Driven from the same published state
        # the rest of the HUD already reads — no extra IPC required.
        self._draw_activity_ring(
            jarvis_state, active_action, alert_active, alert_main_color,
        )

        # ── blue-green-2: STAGING badge + test progress (left monitor) ───
        if self.role == "staging":
            ts = state.get("test_state") or {}
            self._draw_staging_overlay(ts if isinstance(ts, dict) else {})

        # Advance the frame counter. Rescheduling is handled by the tick()
        # wrapper's finally-style tail so it runs even if this body raised.
        self.frame += 1
        return TICK_MS

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--parent-pid", type=int, default=0)
    # Blue/green: a staging-launched HUD reads from data_staging/hud_state.json
    # so its rendering can't blink the prod HUD. Default empty preserves the
    # prod path (STATE_FILE module constant above).
    parser.add_argument("--state-file", type=str, default="",
                        help="override hud_state.json path "
                             "(used by blue/green staging instances)")
    # blue-green-2: per-role rendering. "staging" enables the cyan
    # STAGING badge + test-progress overlay; everything else (default
    # "prod") renders as before.
    parser.add_argument("--role", type=str, default="prod",
                        choices=("prod", "staging"),
                        help="blue/green role badge to render in this HUD")
    args = parser.parse_args()

    if args.state_file:
        global STATE_FILE
        STATE_FILE = args.state_file

    hud = HUD(args.x, args.y, args.width, args.parent_pid, role=args.role)
    hud.run()


if __name__ == "__main__":
    main()
