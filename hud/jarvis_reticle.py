#!/usr/bin/env python3
"""
JARVIS Reticle Overlay — full-virtual-screen translucent target reticle.

Spawned as a subprocess by bobert_companion.py at startup. Spans the entire
multi-monitor virtual desktop and draws a 2-second translucent target reticle
at the click/type coordinates published by JARVIS whenever it executes a
UI-automation action (ui_click, ui_type, ui_press, ui_hotkey, ui_scroll,
_act_focus_window).

The host publishes reticle events to ``hud_reticles.json`` (sibling to
bobert_companion.py). Each event has ``x``, ``y`` (virtual-desktop pixel
coordinates — may be negative on the left monitor), an optional ``label``,
and a ``created_at`` epoch timestamp. The overlay reads the file at every
animation tick and draws every reticle whose age is < ``RETICLE_TTL``.

Closes cleanly when its parent process exits — the ``--parent-pid`` argument
lets us detect that without IPC plumbing.

CLI:
  python hud/jarvis_reticle.py --x -2560 --y -1440 --width 7680 --height 2880 \
                               --parent-pid 12345
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────
#  Appearance
#  Palette matches hud/jarvis_hud.py so the visual language reads as a
#  single coherent overlay system.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY        = "#010101"   # near-black, keyed transparent on Win32
RING_COLOR    = "#4cc9ff"   # cyan (matches HUD CYAN)
RING_GLOW     = "#9ee7ff"   # bright inner (matches HUD CYAN_BRIGHT)
RING_FADE     = "#1b4a66"   # dim cyan during the fade-out tail
RING_FADE_2   = "#5d8aa3"   # dim mid (matches HUD DIM_TEXT)
LABEL_COLOR   = "#cfeefb"   # matches HUD TEXT_COLOR
LABEL_FADE    = "#5d8aa3"

# Priority-1 palette — reticles with color=="red" use this so a boss-mode
# alert reads as visually distinct from a normal UI-automation click.
RED_RING_COLOR  = "#ff3344"
RED_RING_GLOW   = "#ffb0b0"
RED_RING_FADE   = "#5a1620"
RED_RING_FADE_2 = "#8a3a44"
RED_LABEL_COLOR = "#ffd2d6"
RED_LABEL_FADE  = "#8a3a44"

RETICLE_TTL       = 2.0     # spec: "2-second translucent target reticle"
GROW_DURATION     = 0.35    # outer ring grows for the first 0.35s, then holds
FADE_TAIL_SECS    = 0.5     # last 0.5s fade to the dim palette
START_RADIUS      = 14
MAX_RADIUS        = 46
CROSSHAIR_LEN     = 14
TICK_MS           = 33      # ~30 fps animation
LABEL_FONT        = ("Consolas", 9, "bold")

STATE_FILE_NAME = "hud_reticles.json"
PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE      = os.path.join(PROJECT_DIR, STATE_FILE_NAME)


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if _HAS_PSUTIL:
        # pid_exists can raise on Windows for a transient handle/permission
        # error. The reticle spans the full virtual desktop, so a frozen
        # frame here is the worst case — treat an unknowable parent as alive
        # (matches the PyQt HUDs' guard) so a hiccup can't strand it.
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_reticles() -> list:
    if not os.path.exists(STATE_FILE):
        return []
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = data.get("reticles", [])
        return out if isinstance(out, list) else []
    except Exception:
        return []


class ReticleOverlay:
    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int):
        self.parent_pid = parent_pid
        self.origin_x   = x
        self.origin_y   = y
        self.width      = w
        self.height     = h

        self.root = tk.Tk()
        self.root.title("JARVIS Reticle")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # On Windows, -transparentcolor makes the keyed background fully
        # transparent AND click-through. Drawn ring pixels (cyan) remain
        # opaque, which is desired so the reticle is visible.
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            # Fallback: plain alpha. Less ideal — entire window is
            # semi-transparent — but better than nothing.
            try:
                self.root.attributes("-alpha", 0.85)
            except Exception:
                pass

        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=w, height=h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.tick()

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def _draw_reticle(self, x: int, y: int, age: float, label: str,
                      color: str = ""):
        # Outer ring grows over the first GROW_DURATION seconds, then holds.
        grow = min(1.0, age / GROW_DURATION)
        radius = START_RADIUS + (MAX_RADIUS - START_RADIUS) * grow

        # Last FADE_TAIL_SECS dims toward the fade palette
        fading = age > (RETICLE_TTL - FADE_TAIL_SECS)

        # Palette selection. The default (cyan) is unchanged so every existing
        # UI-automation reticle still renders identically. Boss-mode entries
        # come through with color=="red" and pick up the priority-1 palette.
        if color == "red":
            outer_color = RED_RING_FADE if fading else RED_RING_COLOR
            inner_color = RED_RING_FADE_2 if fading else RED_RING_GLOW
            text_color  = RED_LABEL_FADE if fading else RED_LABEL_COLOR
        else:
            outer_color = RING_FADE if fading else RING_COLOR
            inner_color = RING_FADE_2 if fading else RING_GLOW
            text_color  = LABEL_FADE if fading else LABEL_COLOR

        # Crosshair (always at fixed length so the center is precise)
        self.canvas.create_line(
            x - CROSSHAIR_LEN, y, x + CROSSHAIR_LEN, y,
            fill=outer_color, width=1,
        )
        self.canvas.create_line(
            x, y - CROSSHAIR_LEN, x, y + CROSSHAIR_LEN,
            fill=outer_color, width=1,
        )

        # Two concentric rings (thick outer + thin inner for the gauge look)
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline=outer_color, width=2,
        )
        inner_r = max(4, radius - 5)
        self.canvas.create_oval(
            x - inner_r, y - inner_r, x + inner_r, y + inner_r,
            outline=inner_color, width=1,
        )

        # Center dot
        self.canvas.create_oval(
            x - 2, y - 2, x + 2, y + 2,
            fill=inner_color, outline="",
        )

        if label:
            self.canvas.create_text(
                x, y + radius + 8,
                text=label[:24],
                fill=text_color, font=LABEL_FONT, anchor="n",
            )

    def tick(self):
        if not _is_parent_alive(self.parent_pid):
            self._on_close()
            return

        now = time.time()
        entries = _read_reticles()

        # Filter to live entries up-front so the idle path can short-circuit.
        live: list = []
        for r in entries:
            try:
                t0 = float(r.get("created_at", 0) or 0)
            except (TypeError, ValueError):
                continue
            age = now - t0
            if age < 0 or age > RETICLE_TTL:
                continue
            live.append((r, age))

        # task-13:22 CPU runaway fix: when there's nothing on-screen and
        # there was nothing last frame either, skip the 7680×2880 canvas
        # delete+repaint and back off to a slow idle poll. A 30fps full-
        # screen redraw of an empty canvas was burning a full CPU core
        # (observed PID 15964 at 99% on 2026-05-29).
        if not hasattr(self, "_prev_live_count"):
            self._prev_live_count = 0
        if not live and self._prev_live_count == 0:
            # Both this frame and last had no reticles — idle poll at 4 Hz
            # instead of redrawing 30 times a second.
            self.root.after(250, self.tick)
            return
        self._prev_live_count = len(live)

        self.canvas.delete("all")
        # Repaint the keyed background so old reticles don't smear
        self.canvas.create_rectangle(
            0, 0, self.width, self.height, fill=BG_KEY, outline="",
        )

        for r, age in live:
            try:
                rx_abs = int(r.get("x", 0))
                ry_abs = int(r.get("y", 0))
            except (TypeError, ValueError):
                continue
            # Translate virtual-desktop coords into our canvas-local coords.
            cx = rx_abs - self.origin_x
            cy = ry_abs - self.origin_y
            # Skip reticles that fall outside our canvas (shouldn't happen
            # for valid clicks, but defensive against bogus data).
            if cx < -64 or cy < -64 or cx > self.width + 64 or cy > self.height + 64:
                continue
            color = str(r.get("color", "") or "").lower()
            self._draw_reticle(
                cx, cy, age, str(r.get("label", "") or ""), color=color,
            )

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

    overlay = ReticleOverlay(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    overlay.run()


if __name__ == "__main__":
    main()
