#!/usr/bin/env python3
"""
JARVIS Air-Cursor Overlay — an animated targeting reticle that follows the
Kinect air-mouse cursor.

Spawned as a click-THROUGH subprocess by skills/kinect_air_mouse.py while the
air-mouse is enabled. Spans the entire multi-monitor virtual desktop and renders
a glowing JARVIS reticle at the cursor position the air-mouse publishes, so the
owner always sees where their hand is pointing.

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
real cursor underneath this overlay). If color-keying is unavailable we fall
back to a low global alpha (degraded, but never blocks input via WS_EX flags
where the platform exposes them).

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
publishing) so a crashed parent can't strand a fullscreen layer.

CLI:
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


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if _HAS_PSUTIL:
        # pid_exists can raise on Windows for a transient handle/permission
        # error; treat an unknowable parent as alive (matches the other HUDs)
        # so a hiccup can't strand this fullscreen layer.
        try:
            return psutil.pid_exists(pid)
        except Exception:
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
    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int):
        self.parent_pid = parent_pid
        self.origin_x   = x
        self.origin_y   = y
        self.width      = w
        self.height     = h
        self._started_at = time.time()

        # Animation state.
        self.frame       = 0
        self.cur_x       = None      # eased reticle position (canvas-local)
        self.cur_y       = None
        self.target_x    = None
        self.target_y    = None
        self.state       = "hidden"
        self.color_name  = "cyan"
        self.visible     = False
        self._last_state_ts = 0.0
        self._grab_flash = 0         # countdown of the gold-flash brighten
        self._was_grab   = False
        self.trail       = []        # recent (cx, cy) canvas points
        self._prev_visible = False

        self.root = tk.Tk()
        self.root.title("JARVIS Air Cursor")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Color-key the background fully transparent + click-through on Win32.
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            # Fallback: a low global alpha. Less ideal (whole window dims) but
            # better than an opaque block; still topmost + frameless.
            try:
                self.root.attributes("-alpha", 0.78)
            except Exception:
                pass

        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=w, height=h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Belt-and-suspenders click-through: on Windows, also set the layered +
        # transparent extended window styles so input falls through even if a
        # future Tk drops -transparentcolor's click-through behaviour. Pure
        # best-effort; the overlay works without it via the color-key above.
        self._make_click_through_win32()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.tick()

    def _make_click_through_win32(self):
        try:
            if os.name != "nt":
                return
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.canvas.winfo_id())
            if not hwnd:
                hwnd = self.canvas.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW = 0x00000080
            cur = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                cur | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
        except Exception:
            # Color-keying already gives click-through; this is only a backstop.
            pass

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── drawing ──────────────────────────────────────────────────────────
    def _palette(self):
        """(outer, inner, glow, dot) colours for the current state."""
        if self.state == "grab":
            return GOLD, GOLD_BRIGHT, GOLD, GOLD_BRIGHT
        return CYAN, CYAN_BRIGHT, CYAN, CYAN_BRIGHT

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

    def _draw_trail(self):
        """Faint fading dots behind the reticle so fast moves are traceable."""
        n = len(self.trail)
        if n < 2:
            return
        for i, (tx, ty) in enumerate(self.trail[:-1]):
            # Older points (front of list) are smaller + dimmer.
            frac = (i + 1) / float(n)
            r = max(1, int(2 * frac))
            self.canvas.create_oval(
                tx - r, ty - r, tx + r, ty + r,
                fill=TRAIL_COLOR if self.state != "grab" else GOLD_DIM,
                outline="",
            )

    # ── per-frame update ──────────────────────────────────────────────────
    def _refresh_state(self):
        """Pull the latest published cursor state + decide visibility. Returns
        False when the overlay should CLOSE (parent dead / long stale / orphan
        cap)."""
        now = time.time()
        if not _is_parent_alive(self.parent_pid):
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
                # Virtual-desktop → canvas-local.
                self.target_x = ax - self.origin_x
                self.target_y = ay - self.origin_y
                # Snap on (re)acquire so we don't sweep across the screen.
                if not self._prev_visible or self.cur_x is None:
                    self.cur_x = self.target_x
                    self.cur_y = self.target_y
                    self.trail = []
        return True

    def tick(self):
        if not self._refresh_state():
            self._on_close()
            return

        self.frame += 1
        if self._grab_flash > 0:
            self._grab_flash -= 1

        # Idle short-circuit: nothing visible now and nothing last frame either
        # → skip the full-virtual-desktop clear+repaint and back off to a slow
        # poll (the same CPU-runaway guard jarvis_reticle uses). A 60fps redraw
        # of an empty 7680×2880 canvas would otherwise burn a core.
        if not self.visible and not self._prev_visible:
            self._prev_visible = False
            self.root.after(120, self.tick)
            return

        # Ease the reticle toward the target.
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

        # Repaint.
        self.canvas.delete("all")
        # Keyed background so old frames don't smear.
        self.canvas.create_rectangle(
            0, 0, self.width, self.height, fill=BG_KEY, outline="",
        )
        if self.visible and self.cur_x is not None:
            # Skip if somehow off our canvas (defensive against bogus data).
            if (-64 <= self.cur_x <= self.width + 64 and
                    -64 <= self.cur_y <= self.height + 64):
                self._draw_trail()
                self._draw_reticle(self.cur_x, self.cur_y)

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
    args = parser.parse_args()

    overlay = AirCursorOverlay(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    overlay.run()


if __name__ == "__main__":
    main()
