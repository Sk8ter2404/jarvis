#!/usr/bin/env python3
"""
JARVIS Air-Cursor Overlay — an animated targeting reticle that follows the
Kinect air-mouse cursor.

Spawned as a click-THROUGH subprocess by skills/kinect_air_mouse.py while the
air-mouse is enabled. Renders a glowing JARVIS reticle at the cursor position the
air-mouse publishes, so the owner always sees where their hand is pointing.

WHY A SMALL FOLLOW-WINDOW (not a full virtual-desktop surface)
==============================================================
v1.64.0 stretched this overlay across the ENTIRE virtual desktop (all monitors).
A full-desktop tkinter window relies on Win32 colour-keying (``-transparentcolor``)
to stay transparent + click-through; if that colour-key is ever lost the keyed
background paints as a SOLID OPAQUE BLOCK — and at full-desktop size that block
blacks out every monitor (the v1.64.0 "black screen over everything" report).

The previous code made this failure certain: after Tk set up the colour-key it
ALSO did ``SetWindowLongW(GWL_EXSTYLE, ... | WS_EX_LAYERED ...)`` on the window.
Re-asserting WS_EX_LAYERED on a window whose colour-key was already installed by
Tk *invalidates that colour-key* (a layered window with neither
SetLayeredWindowAttributes nor UpdateLayeredWindow has NO defined alpha/key and
Windows composites it fully opaque). Result: an opaque near-black layer the exact
size of the virtual desktop — exactly what the owner saw.

The robust fix is to stop painting a desktop-sized surface at all. We now use a
SMALL window (``WINDOW_SIZE`` px) repositioned to the cursor each tick. It still
uses the proven ``-transparentcolor`` key (same as hud/jarvis_reticle.py, which
works), and the Win32 backstop now *correctly re-establishes* the colour-key via
SetLayeredWindowAttributes after touching the ex-style. So even in the worst case
the only thing that could ever show is a tiny ``WINDOW_SIZE`` patch under the
cursor — never a full-screen blackout.

LOOK
====
  • TRACKING (open hand): a JARVIS-cyan concentric ring + a few rotating arc
    segments + a centre dot + a soft glow + a gentle breathing pulse.
  • GRAB (closed hand → right-click / drag): the reticle SNAPS inward, flashes
    GOLD/amber, and the ring contracts — a visible "lock". Held through a drag.
  • A faint MOTION TRAIL of recent positions so fast moves are traceable.
  • Hidden whenever the air-mouse is off or no hand is tracked.

CLICK-THROUGH (must NOT intercept real clicks)
==============================================
Like hud/jarvis_reticle.py, the window is frameless + always-on-top and uses
Tk's ``-transparentcolor`` so the keyed background is fully transparent AND
click-through on Win32 (the drawn cyan/gold pixels stay visible but the window
never steals a click — critical, since the air-mouse is literally driving the
real cursor underneath this overlay). As a backstop we also set
WS_EX_LAYERED | WS_EX_TRANSPARENT (+ re-key via SetLayeredWindowAttributes) so
input still falls through even if a future Tk drops ``-transparentcolor``'s
click-through behaviour. If colour-keying is unavailable we fall back to a low
global alpha (degraded, but a tiny window, never a fullscreen block).

STATE
=====
The air-mouse writes ``air_cursor_state.json`` (sibling to bobert_companion.py)
each tick:
    {"x": int, "y": int,                  # virtual-desktop pixel
     "state": "track"|"grab"|"hidden",
     "color": "cyan"|"gold",
     "visible": bool, "ts": <epoch>}
This overlay reads it every animation tick, smoothly eases the reticle toward
the target, and animates the rings/arcs locally (so it keeps spinning/pulsing
between the air-mouse's position updates).

LIFECYCLE
=========
Closes cleanly when its parent process exits — the ``--parent-pid`` argument
lets us detect that without IPC plumbing (mirrors jarvis_reticle.py). Also
self-exits if the state file goes stale for a long time (the air-mouse stopped
publishing) so a crashed parent can't strand a click-through layer.

CLI (the --x/--y/--width/--height span is accepted for spawn-contract
compatibility with jarvis_reticle.py but only used to clamp/centre the small
follow-window; the window itself is WINDOW_SIZE px, not the full span):
  python hud/jarvis_air_cursor.py --x -2560 --y -1440 --width 7680 \
                                  --height 2880 --parent-pid 12345
"""
import argparse
import json
import math
import os
import time
import tkinter as tk

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────
#  Appearance — palette matches hud/jarvis_reticle.py / jarvis_hud.py so the
#  air-cursor reads as part of the same coherent JARVIS overlay system.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY        = "#010101"   # near-black, keyed transparent on Win32

# Tracking (open hand) — cyan.
CYAN          = "#4cc9ff"
CYAN_BRIGHT   = "#9ee7ff"
CYAN_DIM      = "#1b4a66"

# Grab (closed hand) — gold / amber lock.
GOLD          = "#ffb347"
GOLD_BRIGHT   = "#ffe0a0"
GOLD_DIM      = "#7a5a23"

# TWO-HAND reticles (Part 3): the owner wants to SEE two circle cursors, one per
# hand, while both hands are engaged. BLUE normally; PURPLE while a window is being
# actively grabbed/resized. (These are overlay-drawn circles — the real OS mouse
# stays single; we just draw two reticles.)
BLUE          = "#3aa0ff"
BLUE_BRIGHT   = "#9fd0ff"
BLUE_DIM      = "#1c4f80"
PURPLE        = "#b06cff"
PURPLE_BRIGHT = "#e0c0ff"
PURPLE_DIM    = "#502080"

TRAIL_COLOR   = "#2f7fa3"   # faint cyan for the motion trail

# Geometry (pixels).
RING_RADIUS_TRACK = 26      # outer ring radius while tracking
RING_RADIUS_GRAB  = 16      # ring contracts on grab (a visible lock)
INNER_OFFSET      = 6       # inner ring sits this far inside the outer
CENTER_DOT_R      = 3
ARC_SEGMENTS      = 3       # rotating arc segments around the ring
ARC_SWEEP_DEG     = 38      # angular length of each arc segment
GLOW_RADIUS_TRACK = 40
GLOW_RADIUS_GRAB  = 30

# The small follow-window is large enough to hold the glow ring + the rotating
# arcs (which sit a few px outside it) with a little breathing room. The reticle
# is always drawn at the window's CENTRE; the WINDOW moves, not the drawing.
# 2*(GLOW_RADIUS_TRACK=40 + arc pad ~5 + pulse ~3) = ~96, so 132 leaves margin.
WINDOW_SIZE       = 132     # px square
WINDOW_HALF       = WINDOW_SIZE // 2

TICK_MS           = 16      # ~60 fps so the spin/pulse + easing look smooth
EASE              = 0.45    # cursor-follow easing (0..1); higher = snappier
SPIN_SPEED        = 0.06    # radians/tick the arc segments rotate
PULSE_SPEED       = 0.18    # breathing pulse rate
GRAB_FLASH_TICKS  = 8       # how many ticks the gold flash brightens on grab
TRAIL_MAX         = 8       # number of trail dots retained
TRAIL_MIN_MOVE    = 6       # only drop a new trail dot after moving this many px

# If the air-mouse stops publishing (file mtime/ts goes stale) for this long,
# hide the reticle; if it stays stale much longer, exit (parent likely gone).
STATE_STALE_HIDE_S = 0.6
STATE_STALE_EXIT_S = 120.0
# Orphan guard (mirrors jarvis_reticle / holo HUD): with --parent-pid 0/absent,
# self-exit after this long so a parentless click-through layer can't strand.
ORPHAN_MAX_LIFETIME_S = 1800.0

STATE_FILE_NAME = "air_cursor_state.json"
PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE      = os.path.join(PROJECT_DIR, STATE_FILE_NAME)

# Win32 extended-window-style bits (module-level so tests can assert the
# click-through contract without constructing a Tk root).
GWL_EXSTYLE       = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TRANSPARENT = 0x00000020   # makes the window click-through (hit-test pass)
WS_EX_TOOLWINDOW  = 0x00000080   # keep it out of the alt-tab / taskbar list
WS_EX_NOACTIVATE  = 0x08000000   # never steal focus from the app underneath
LWA_COLORKEY      = 0x00000001   # SetLayeredWindowAttributes: key a colour out


def _click_through_exstyle(cur: int) -> int:
    """OR the click-through / layered / no-activate / tool-window bits onto an
    existing GWL_EXSTYLE value. Pure (int in, int out) so the transparency +
    click-through contract is unit-testable with no Tk root / no Win32.

    WS_EX_LAYERED is required for the colour-key (and for click-through to be
    rock-solid); WS_EX_TRANSPARENT is what actually makes hit-testing fall
    through to the window underneath — together they guarantee the air-mouse's
    real cursor underneath this overlay is never blocked."""
    return (cur
            | WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE)


def _colorref(hex_color: str) -> int:
    """``#rrggbb`` → Win32 COLORREF (0x00bbggrr). Used as the colour-key passed
    to SetLayeredWindowAttributes so the keyed background composites fully
    transparent instead of as an opaque block. Pure; testable."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def _is_parent_alive(pid: int, start_time: "float | None" = None) -> bool:
    """True while the spawning JARVIS is still alive.

    PID-RECYCLE GUARD (P0-2): a bare pid_exists() can read TRUE for a STRANGER
    that the OS handed our dead parent's recycled PID — leaving this click-through
    overlay stranded on top of an unrelated process. When ``start_time`` (the
    parent's psutil create_time() captured at spawn, passed via --parent-start) is
    given, we additionally require the live process at ``pid`` to report the SAME
    create_time (within a small epsilon): a recycled PID has a DIFFERENT start
    time and so reads as DEAD. Without a start_time (e.g. the air-mouse spawn,
    which doesn't pass one) we fall back to the historical PID-exists-only check.
    A transient/unknowable lookup is treated as ALIVE so a hiccup can't strand the
    overlay."""
    if pid <= 0:
        return True
    if _HAS_PSUTIL:
        # pid_exists can raise on Windows for a transient handle/permission
        # error; treat an unknowable parent as alive (matches the other HUDs)
        # so a hiccup can't strand this click-through layer.
        try:
            if not psutil.pid_exists(pid):
                return False
        except Exception:
            return True
        if start_time is not None:
            try:
                # A recycled PID reads as DEAD: same number, different birth time.
                if abs(psutil.Process(pid).create_time() - start_time) > 1.0:
                    return False
            except Exception:
                # Can't read the live process's start time (gone between the
                # exists-check and here, or access denied) — treat as alive so a
                # transient race can't strand the overlay.
                return True
        return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class AirCursorOverlay:
    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int,
                 parent_start: "float | None" = None):
        self.parent_pid = parent_pid
        # Parent's create_time() captured at spawn (--parent-start), for the
        # PID-recycle guard in _is_parent_alive. None when the spawner didn't
        # pass one (the air-mouse path) → PID-exists-only fallback.
        self.parent_start = parent_start
        # The virtual-desktop span the air-mouse operates over. We don't paint a
        # window this big any more — we only use it to clamp the small follow-
        # window so its top-left stays a sane on-desktop coordinate.
        self.origin_x   = x
        self.origin_y   = y
        self.span_w     = w
        self.span_h     = h
        self._started_at = time.time()

        # Animation state.
        self.frame       = 0
        self.cur_x       = None      # eased reticle position (virtual-desktop px)
        self.cur_y       = None
        self.target_x    = None
        self.target_y    = None
        self.state       = "hidden"
        self.color_name  = "cyan"
        self.visible     = False
        self._last_state_ts = 0.0
        self._grab_flash = 0         # countdown of the gold-flash brighten
        self._was_grab   = False
        self.trail       = []        # recent (vx, vy) virtual-desktop points
        self._prev_visible = False
        self._win_x      = None      # last placed window top-left (to avoid
        self._win_y      = None      # redundant geometry() churn)

        # TWO-HAND mode (Part 3): when both hands are engaged the air-mouse / two-
        # hand skill publish two hand points; we draw TWO circle reticles (BLUE,
        # PURPLE while resizing). The FIRST hand reuses this window; the SECOND uses
        # a lazily-built identical small click-through window so the two reticles can
        # sit far apart (even on different monitors) without ever painting a large
        # surface (the full-desktop-blackout failure mode). One-hand mode keeps the
        # single cyan reticle and hides the second window.
        self.two_hand     = False        # both-hands mode active this frame
        self.two_resizing = False        # a window is being actively resized → PURPLE
        self.hand_pts     = None         # [(x, y), (x, y)] virtual-desktop px, or None
        self.root2        = None         # the 2nd hand's window (lazy)
        self.canvas2      = None
        self._win2_x      = None
        self._win2_y      = None
        self._has_colorkey2 = False

        self.root = tk.Tk()
        self.root.title("JARVIS Air Cursor")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Color-key the background fully transparent + click-through on Win32.
        # This is the SAME mechanism hud/jarvis_reticle.py uses successfully.
        self._has_colorkey = False
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
            self._has_colorkey = True
        except tk.TclError:
            # Fallback: a low global alpha. Less ideal (the whole window dims)
            # but it's only a small WINDOW_SIZE square, never a fullscreen block;
            # still topmost + frameless + click-through via WS_EX_TRANSPARENT.
            try:
                self.root.attributes("-alpha", 0.78)
            except Exception:
                pass

        # Start the small window centred on the desktop; tick() repositions it
        # to the cursor every frame. (Off-screen until first visible frame.)
        start_x = x + max(0, (w - WINDOW_SIZE) // 2)
        start_y = y + max(0, (h - WINDOW_SIZE) // 2)
        self.root.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}+{start_x}+{start_y}")
        self._win_x, self._win_y = start_x, start_y

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=WINDOW_SIZE, height=WINDOW_SIZE,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Belt-and-suspenders click-through: on Windows set the layered +
        # transparent extended styles AND (critically) re-establish the colour-
        # key via SetLayeredWindowAttributes. Re-asserting WS_EX_LAYERED without
        # re-keying is exactly what turned the old full-desktop overlay opaque.
        self._make_click_through_win32()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.tick()

    def _make_click_through_win32(self, win=None, has_colorkey=None):
        win = self.root if win is None else win
        if has_colorkey is None:
            has_colorkey = self._has_colorkey
        try:
            if os.name != "nt":
                return
            import ctypes
            win.update_idletasks()
            # The real top-level window is the parent of the Tk toplevel's
            # window id (Tk wraps the toplevel in a frame on Win32). Walk up from
            # the TOPLEVEL — not the canvas — so we style the actual HWND Tk
            # colour-keyed; styling a child would leave the toplevel opaque.
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(win.winfo_id())
            if not hwnd:
                hwnd = win.winfo_id()
            cur = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, _click_through_exstyle(cur))
            # Re-key the background transparent. Tk's -transparentcolor already
            # did this, but we just touched the ex-style with WS_EX_LAYERED, so
            # re-assert the colour-key to be safe — a layered window with no
            # SetLayeredWindowAttributes/UpdateLayeredWindow composites OPAQUE.
            # (When -transparentcolor was unavailable we skip the key and rely on
            # the -alpha fallback set above.)
            if has_colorkey:
                user32.SetLayeredWindowAttributes(
                    hwnd, _colorref(BG_KEY), 0, LWA_COLORKEY)
        except Exception:
            # Color-keying already gives transparency + click-through; this is
            # only a backstop. Worst case it's a tiny window, never fullscreen.
            pass

    def _on_close(self):
        for w in (getattr(self, "root2", None), getattr(self, "root", None)):
            try:
                if w is not None:
                    w.destroy()
            except Exception:
                pass

    def _clamp_window_xy(self, vx: float, vy: float) -> "tuple[int, int]":
        """Top-left for a WINDOW_SIZE window centred at virtual-desktop (vx, vy),
        clamped onto the desktop span (defensive against bogus state values)."""
        wx = int(round(vx)) - WINDOW_HALF
        wy = int(round(vy)) - WINDOW_HALF
        min_x, min_y = self.origin_x, self.origin_y
        max_x = self.origin_x + max(0, self.span_w - WINDOW_SIZE)
        max_y = self.origin_y + max(0, self.span_h - WINDOW_SIZE)
        return min(max(wx, min_x), max_x), min(max(wy, min_y), max_y)

    def _place_window(self, vx: float, vy: float):
        """Move the small window so its CENTRE sits at virtual-desktop (vx, vy),
        clamped to stay within the desktop span. No-op when it wouldn't move (so
        we don't thrash geometry() at 60fps when the cursor is still)."""
        wx, wy = self._clamp_window_xy(vx, vy)
        if wx == self._win_x and wy == self._win_y:
            return
        self._win_x, self._win_y = wx, wy
        self.root.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}+{wx}+{wy}")

    # ── TWO-HAND second reticle window (Part 3) ────────────────────────────
    def _ensure_second_window(self):
        """Lazily build the SECOND hand's small click-through window (identical to
        the first: WINDOW_SIZE, colour-keyed transparent, topmost, click-through).
        Built only the first time two-hand mode is entered so single-hand mode pays
        nothing. NEVER raises — a failure just means the 2nd reticle won't show."""
        if self.root2 is not None:
            return
        try:
            self.root2 = tk.Toplevel(self.root)
            self.root2.title("JARVIS Air Cursor 2")
            self.root2.configure(bg=BG_KEY)
            self.root2.overrideredirect(True)
            self.root2.attributes("-topmost", True)
            self._has_colorkey2 = False
            try:
                self.root2.attributes("-transparentcolor", BG_KEY)
                self._has_colorkey2 = True
            except tk.TclError:
                try:
                    self.root2.attributes("-alpha", 0.78)
                except Exception:
                    pass
            sx = self.origin_x + max(0, (self.span_w - WINDOW_SIZE) // 2)
            sy = self.origin_y + max(0, (self.span_h - WINDOW_SIZE) // 2)
            self.root2.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}+{sx}+{sy}")
            self._win2_x, self._win2_y = sx, sy
            self.canvas2 = tk.Canvas(
                self.root2, bg=BG_KEY, width=WINDOW_SIZE, height=WINDOW_SIZE,
                highlightthickness=0, bd=0)
            self.canvas2.pack(fill="both", expand=True)
            self._make_click_through_win32(self.root2, self._has_colorkey2)
        except Exception:
            self.root2 = None
            self.canvas2 = None

    def _place_window2(self, vx: float, vy: float):
        if self.root2 is None:
            return
        wx, wy = self._clamp_window_xy(vx, vy)
        if wx == self._win2_x and wy == self._win2_y:
            return
        self._win2_x, self._win2_y = wx, wy
        try:
            self.root2.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}+{wx}+{wy}")
        except Exception:
            pass

    def _hide_second_window(self):
        if self.root2 is None:
            return
        try:
            if self.root2.state() != "withdrawn":
                self.root2.withdraw()
        except Exception:
            pass

    # ── drawing ──────────────────────────────────────────────────────────
    def _palette(self):
        """(outer, inner, glow, dot) colours for the current state."""
        if self.state == "grab":
            return GOLD, GOLD_BRIGHT, GOLD, GOLD_BRIGHT
        return CYAN, CYAN_BRIGHT, CYAN, CYAN_BRIGHT

    def _two_hand_palette(self):
        """(outer, inner, dim) for a TWO-HAND reticle: PURPLE while actively
        resizing a window, else BLUE."""
        if self.two_resizing:
            return PURPLE, PURPLE_BRIGHT, PURPLE_DIM
        return BLUE, BLUE_BRIGHT, BLUE_DIM

    def _draw_two_hand_circle(self, canvas, cx: float, cy: float):
        """Draw ONE two-hand reticle circle (a clean concentric ring + crosshair +
        centre dot) at the window-local (cx, cy). BLUE normally, PURPLE while
        resizing. Drawn into whichever canvas (hand 1 or hand 2) is passed."""
        outer, inner, dim = self._two_hand_palette()
        pulse = 0.5 * (1.0 + math.sin(self.frame * PULSE_SPEED))
        radius = RING_RADIUS_TRACK + 2.0 * pulse
        # Soft dim glow ring.
        glow_r = GLOW_RADIUS_TRACK
        if glow_r > radius:
            canvas.create_oval(cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r,
                               outline=dim, width=1)
        # Outer + inner ring.
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                           outline=outer, width=3)
        inner_r = max(3, radius - INNER_OFFSET)
        canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
                           outline=inner, width=1)
        # Short crosshair ticks for a precise centre.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            canvas.create_line(
                cx + dx * (inner_r - 1), cy + dy * (inner_r - 1),
                cx + dx * (inner_r - 1 - 7), cy + dy * (inner_r - 1 - 7),
                fill=inner, width=1)
        # Centre dot.
        canvas.create_oval(cx - CENTER_DOT_R, cy - CENTER_DOT_R,
                           cx + CENTER_DOT_R, cy + CENTER_DOT_R,
                           fill=inner, outline="")

    def _draw_reticle(self, cx: float, cy: float):
        is_grab = (self.state == "grab")
        outer_color, inner_color, glow_color, dot_color = self._palette()

        # Breathing pulse (gentle while tracking; a tighter, brighter pulse on
        # grab). 0..1.
        pulse = 0.5 * (1.0 + math.sin(self.frame * PULSE_SPEED))

        # Ring radius: contracts on grab. Add a small grab-flash inward snap on
        # the first few ticks after the lock for a satisfying "click".
        base_r = RING_RADIUS_GRAB if is_grab else RING_RADIUS_TRACK
        if is_grab and self._grab_flash > 0:
            snap = (self._grab_flash / float(GRAB_FLASH_TICKS))  # 1→0
            base_r -= int(4 * snap)                              # extra inward
        # A subtle radius breathing.
        radius = base_r + (1.5 if is_grab else 2.5) * pulse

        # ── soft glow (concentric translucent-ish rings; tk has no real alpha
        #    on a canvas oval, so emulate a glow with a couple of dim rings) ──
        glow_r = (GLOW_RADIUS_GRAB if is_grab else GLOW_RADIUS_TRACK)
        glow_dim = GOLD_DIM if is_grab else CYAN_DIM
        for i, gr in enumerate((glow_r, glow_r - 8)):
            if gr <= radius:
                continue
            self.canvas.create_oval(
                cx - gr, cy - gr, cx + gr, cy + gr,
                outline=glow_dim, width=1,
            )

        # ── outer ring ──
        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline=outer_color, width=2,
        )
        # ── inner ring ──
        inner_r = max(3, radius - INNER_OFFSET)
        self.canvas.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
            outline=inner_color, width=1,
        )

        # ── rotating arc segments around the outer ring ──
        arc_r = radius + 5
        spin = math.degrees(self.frame * SPIN_SPEED)
        # Counter-rotate on grab for a distinct "locking" motion.
        if is_grab:
            spin = -spin * 1.6
        bbox = (cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r)
        for i in range(ARC_SEGMENTS):
            start = (spin + i * (360.0 / ARC_SEGMENTS)) % 360.0
            self.canvas.create_arc(
                *bbox, start=start, extent=ARC_SWEEP_DEG,
                style="arc", outline=outer_color, width=2,
            )

        # ── short crosshair ticks (precise centre) ──
        tick_len = 6 if is_grab else 8
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.canvas.create_line(
                cx + dx * (inner_r - 1), cy + dy * (inner_r - 1),
                cx + dx * (inner_r - 1 - tick_len),
                cy + dy * (inner_r - 1 - tick_len),
                fill=inner_color, width=1,
            )

        # ── centre dot (brighter on grab / on the pulse peak) ──
        dot_r = CENTER_DOT_R + (1 if (is_grab and self._grab_flash > 0) else 0)
        self.canvas.create_oval(
            cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
            fill=dot_color, outline="",
        )

    def _draw_trail(self, cx: float, cy: float):
        """Faint fading dots behind the reticle so fast moves are traceable.
        Trail points are stored in virtual-desktop coords; project them into the
        window's local frame (relative to the eased centre, which is drawn at the
        window centre). Points outside the small window simply clip."""
        n = len(self.trail)
        if n < 2 or self.cur_x is None:
            return
        for i, (tx, ty) in enumerate(self.trail[:-1]):
            # Older points (front of list) are smaller + dimmer.
            frac = (i + 1) / float(n)
            r = max(1, int(2 * frac))
            # Virtual-desktop → window-local: centre (cur_x,cur_y) maps to
            # (cx,cy) == window centre, so offset each trail point by the same
            # delta.
            lx = cx + (tx - self.cur_x)
            ly = cy + (ty - self.cur_y)
            self.canvas.create_oval(
                lx - r, ly - r, lx + r, ly + r,
                fill=TRAIL_COLOR if self.state != "grab" else GOLD_DIM,
                outline="",
            )

    @staticmethod
    def _parse_hand_pts(raw):
        """Parse the published two-hand points [{x,y},{x,y}] → [(x,y),(x,y)] in
        virtual-desktop px, or None when malformed. Tolerant of bad data."""
        try:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None
            out = []
            for h in raw[:2]:
                if isinstance(h, dict):
                    out.append((int(h.get("x", 0)), int(h.get("y", 0))))
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    out.append((int(h[0]), int(h[1])))
                else:
                    return None
            return out
        except (TypeError, ValueError):
            return None

    # ── per-frame update ──────────────────────────────────────────────────
    def _refresh_state(self):
        """Pull the latest published cursor state + decide visibility. Returns
        False when the overlay should CLOSE (parent dead / long stale / orphan
        cap)."""
        now = time.time()
        if not _is_parent_alive(self.parent_pid, self.parent_start):
            return False
        # Orphan cap: no real parent and we've lived too long → exit.
        if self.parent_pid <= 0 and (now - self._started_at) > ORPHAN_MAX_LIFETIME_S:
            return False

        data = _read_state()
        ts = 0.0
        try:
            ts = float(data.get("ts", 0) or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts > 0:
            self._last_state_ts = ts

        age = now - self._last_state_ts if self._last_state_ts > 0 else 1e9
        # If the publisher has gone quiet for a long time, the air-mouse (and
        # likely JARVIS) is gone — exit so we don't linger forever.
        if age > STATE_STALE_EXIT_S:
            return False

        fresh = age <= STATE_STALE_HIDE_S
        want_visible = bool(data.get("visible")) and fresh
        state = str(data.get("state", "hidden") or "hidden").lower()
        if state not in ("track", "grab"):
            want_visible = False

        # ── TWO-HAND mode (Part 3): the air-mouse / two-hand skill publishes
        #    {"two_hand": True, "hands": [{x,y},{x,y}], "resizing": bool}. Parse the
        #    two hand points; we draw TWO circle reticles (BLUE, PURPLE while
        #    resizing) instead of the single cyan reticle. A stale/false frame falls
        #    back to single-hand. ───────────────────────────────────────────────
        self.two_hand = False
        self.two_resizing = False
        self.hand_pts = None
        if fresh and bool(data.get("two_hand")):
            pts = self._parse_hand_pts(data.get("hands"))
            if pts is not None:
                self.two_hand = True
                self.two_resizing = bool(data.get("resizing"))
                self.hand_pts = pts
                self.visible = True
                # The single-hand reticle is suppressed while two-hand is active.
                self.state = "hidden"
                self.color_name = str(data.get("color", "blue") or "blue").lower()
                self._was_grab = False
                return True

        # Latch grab-flash on the rising edge into "grab".
        if state == "grab" and not self._was_grab:
            self._grab_flash = GRAB_FLASH_TICKS
        self._was_grab = (state == "grab")

        self.state = state if want_visible else "hidden"
        self.visible = want_visible
        self.color_name = str(data.get("color", "cyan") or "cyan").lower()

        if want_visible:
            try:
                ax = int(data.get("x", 0))
                ay = int(data.get("y", 0))
            except (TypeError, ValueError):
                ax = ay = None
            if ax is not None:
                # (x, y) is the SAME absolute virtual-desktop pixel the air-mouse
                # hands to SetCursorPos, so the reticle tracks the real cursor
                # 1:1 — the X un-mirror and the full-virtual-desktop scaling live
                # in skills/kinect_air_mouse.py's ReachBox.map(); do NOT re-apply
                # either here or the reticle would desync from the cursor it is
                # meant to follow. We keep the reticle in virtual-desktop coords
                # and MOVE THE WINDOW to it (see _place_window).
                self.target_x = ax
                self.target_y = ay
                # Snap on (re)acquire so we don't sweep across the screen.
                if not self._prev_visible or self.cur_x is None:
                    self.cur_x = self.target_x
                    self.cur_y = self.target_y
                    self.trail = []
        return True

    def _draw_two_hand_into(self, canvas):
        """Repaint one two-hand window's canvas with a single reticle at its centre
        (the window is moved to the hand; the reticle is always centred)."""
        canvas.delete("all")
        canvas.create_rectangle(0, 0, WINDOW_SIZE, WINDOW_SIZE,
                                fill=BG_KEY, outline="")
        self._draw_two_hand_circle(canvas, WINDOW_HALF, WINDOW_HALF)

    def _tick_two_hand(self):
        """Render the two hand reticles: window 1 (this overlay's existing window) at
        hand 0, window 2 (the lazily-built companion) at hand 1. Each draws a single
        BLUE (PURPLE while resizing) circle at its centre."""
        (h0x, h0y), (h1x, h1y) = self.hand_pts[0], self.hand_pts[1]
        # Hand 1 → the primary window.
        try:
            if self.root.state() == "withdrawn":
                self.root.deiconify()
                self.root.attributes("-topmost", True)
        except Exception:
            pass
        self._place_window(h0x, h0y)
        self._draw_two_hand_into(self.canvas)
        # Hand 2 → the companion window (build on first use).
        self._ensure_second_window()
        if self.root2 is not None and self.canvas2 is not None:
            try:
                if self.root2.state() == "withdrawn":
                    self.root2.deiconify()
                    self.root2.attributes("-topmost", True)
            except Exception:
                pass
            self._place_window2(h1x, h1y)
            self._draw_two_hand_into(self.canvas2)

    def tick(self):
        if not self._refresh_state():
            self._on_close()
            return

        self.frame += 1
        if self._grab_flash > 0:
            self._grab_flash -= 1

        # ── TWO-HAND mode: draw TWO circle reticles (one per hand) and skip the
        #    single-hand path. Each hand gets its own small click-through window so
        #    the two circles can sit far apart (even on different monitors) without
        #    ever painting a large surface. ────────────────────────────────────
        if self.two_hand and self.hand_pts is not None:
            self._tick_two_hand()
            self._prev_visible = True
            self.root.after(TICK_MS, self.tick)
            return
        # Left two-hand mode (or never in it): hide the 2nd window so a stale 2nd
        # reticle can't linger.
        self._hide_second_window()

        # Idle short-circuit: nothing visible now and nothing last frame either
        # → hide the window and back off to a slow poll. (A small window costs
        # far less than the old full-desktop redraw, but withdrawing it while
        # idle keeps it from sitting topmost over the desktop doing nothing.)
        if not self.visible and not self._prev_visible:
            self._prev_visible = False
            try:
                if self.root.state() != "withdrawn":
                    self.root.withdraw()
            except Exception:
                pass
            self.root.after(120, self.tick)
            return

        # Ease the reticle toward the target (in virtual-desktop coords).
        if self.visible and self.target_x is not None:
            if self.cur_x is None:
                self.cur_x, self.cur_y = self.target_x, self.target_y
            else:
                self.cur_x = _lerp(self.cur_x, self.target_x, EASE)
                self.cur_y = _lerp(self.cur_y, self.target_y, EASE)
            # Update the trail when we've moved enough.
            if (not self.trail or
                    abs(self.cur_x - self.trail[-1][0]) +
                    abs(self.cur_y - self.trail[-1][1]) >= TRAIL_MIN_MOVE):
                self.trail.append((self.cur_x, self.cur_y))
                if len(self.trail) > TRAIL_MAX:
                    self.trail.pop(0)

        if self.visible and self.cur_x is not None:
            # Make sure the (possibly withdrawn) window is shown, then move it to
            # the cursor and draw the reticle at the window's centre.
            try:
                if self.root.state() == "withdrawn":
                    self.root.deiconify()
                    self.root.attributes("-topmost", True)
            except Exception:
                pass
            self._place_window(self.cur_x, self.cur_y)

            # Repaint. The reticle is ALWAYS at the window centre; the window
            # moved to the cursor. We only ever paint a small WINDOW_SIZE canvas.
            self.canvas.delete("all")
            # Keyed background so old frames don't smear. NOTE: this fill is the
            # COLOUR-KEYED transparent colour (BG_KEY), never an opaque black —
            # Win32 keys it out via -transparentcolor + SetLayeredWindowAttributes.
            self.canvas.create_rectangle(
                0, 0, WINDOW_SIZE, WINDOW_SIZE, fill=BG_KEY, outline="",
            )
            self._draw_trail(WINDOW_HALF, WINDOW_HALF)
            self._draw_reticle(WINDOW_HALF, WINDOW_HALF)

        self._prev_visible = self.visible
        self.root.after(TICK_MS, self.tick)

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
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--parent-pid", type=int, default=0)
    # Parent's psutil create_time() at spawn — enables the PID-recycle guard in
    # _is_parent_alive. Optional (<=0 / absent → PID-exists-only fallback) so a
    # spawner that doesn't pass it (the air-mouse) still works unchanged.
    parser.add_argument("--parent-start", type=float, default=0.0)
    args = parser.parse_args()

    overlay = AirCursorOverlay(
        args.x, args.y, args.width, args.height, args.parent_pid,
        parent_start=(args.parent_start if args.parent_start > 0 else None),
    )
    overlay.run()


if __name__ == "__main__":
    main()
