#!/usr/bin/env python3
"""
Bambu H2D print-status overlay widget — corner pin for the top monitor.

Spec: jarvis_todo.md 2026-05-27 11:26 (bambu_h2d_overlay extension to
holographic_overlay).

While a print is active per skills/bambu_monitor.py this widget pins a
~280x140 always-on-top translucent panel to the top-right of the top
monitor showing:
  • filename
  • layer N of TOTAL
  • progress bar with % complete
  • estimated time remaining
  • nozzle / bed temperatures
  • a tiny synthetic progress thumbnail (silhouette of the build plate
    with a rising fill that tracks % complete — we don't fetch the real
    .3mf preview because pulling it from the printer over FTP would
    require credentials beyond what bambu_monitor uses and would block
    the tkinter loop)
The entire panel colour-shifts cyan → amber → red as `risk_level` (set
by bambu_monitor from chamber temp swings, AMS errors, print_error
non-zero) climbs.

State source: bambu_overlay_state.json (sibling to bobert_companion.py),
written atomically by skills/bambu_monitor.py on every MQTT update.

Click-through: Win32 -transparentcolor on the BG_KEY pixels. Drawn
panel/text pixels remain opaque, but the panel itself is small (~280x140)
so it doesn't realistically obstruct work.

CLI:
  python hud/bambu_h2d_overlay.py --x 2260 --y -1420 --width 280 \
                                  --height 140 --parent-pid 12345
"""
import argparse
import json
import math
import os
import sys
import time
import tkinter as tk

# hud/ is not a package root — put the project dir on sys.path so
# `from core.parent_watch import ...` resolves (2026-07-14).
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except Exception:
    pass

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


TICK_MS = 200  # 5 fps — the data refreshes once a minute, anything faster is wasted CPU

BG_KEY        = "#010101"
PANEL_DARK    = "#04080d"
PANEL_RIM     = "#0a1820"
CYAN          = "#4cc9ff"
CYAN_DIM      = "#1b4a66"
CYAN_BRIGHT   = "#9ee7ff"
AMBER         = "#ffb347"
AMBER_DIM     = "#7a4a1a"
RED           = "#ff5b5b"
RED_DIM       = "#5a1414"
TEXT          = "#cfeefb"
DIM_TEXT      = "#5d8aa3"

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE     = os.path.join(PROJECT_DIR, "bambu_overlay_state.json")


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
        return psutil.pid_exists(pid)
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
            return json.load(f)
    except Exception:
        return {}


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        *(max(0, min(255, c)) for c in rgb)
    )


def _mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    ))


def _format_minutes(minutes) -> str:
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return "<1m"
    if m < 60:
        return f"{m}m"
    h = m // 60
    rem = m % 60
    if rem == 0:
        return f"{h}h"
    return f"{h}h {rem}m"


def _format_temp(t) -> str:
    try:
        v = float(t)
    except (TypeError, ValueError):
        return "—"
    if v < 1:
        return "—"
    return f"{int(round(v))}°"


def _shorten_filename(name: str, max_len: int = 28) -> str:
    if not name:
        return "(unnamed print)"
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _risk_palette(risk: int):
    """Return (accent, accent_dim) pair for the given risk_level.
        0 → cyan (nominal)
        1 → amber (warning — temp wobble or non-fatal flag)
        2 → red   (failure-risk — print_error non-zero, AMS jam, etc.)
    Any other value falls back to cyan."""
    if risk >= 2:
        return RED, RED_DIM
    if risk == 1:
        return AMBER, AMBER_DIM
    return CYAN, CYAN_DIM


class BambuOverlay:
    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int):
        self.parent_pid = parent_pid
        self.w, self.h = w, h
        self.frame = 0

        self.root = tk.Tk()
        self.root.title("JARVIS Bambu H2D Overlay")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.92)
            except Exception:
                pass

        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=w, height=h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Double-right-click anywhere to dismiss (matches the workshop canvas).
        self.canvas.bind("<Double-Button-3>", lambda _e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.tick()

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def _draw_panel(self, accent: str, accent_dim: str):
        # Rounded-ish rectangle approximated with a regular rect + corner
        # accents. The fill is dark with a thin accent border so the
        # overlay reads against any background.
        self.canvas.create_rectangle(
            2, 2, self.w - 2, self.h - 2,
            outline=PANEL_RIM, fill=PANEL_DARK, width=1,
        )
        # Top accent strip — colour-coded by risk so a glance tells the
        # user "is this print healthy?".
        self.canvas.create_rectangle(
            2, 2, self.w - 2, 5,
            outline="", fill=accent,
        )
        # Subtle inner border to give the panel an arc-reactor finish.
        self.canvas.create_rectangle(
            5, 8, self.w - 5, self.h - 5,
            outline=accent_dim, width=1,
        )

    def _draw_progress_bar(self, pct: float, accent: str, accent_dim: str):
        # Bar geometry — left/right margins matched to the panel padding.
        margin_x = 12
        bar_y    = 60
        bar_h    = 8
        bar_x1   = margin_x
        bar_x2   = self.w - margin_x
        bar_w    = bar_x2 - bar_x1
        # Track
        self.canvas.create_rectangle(
            bar_x1, bar_y, bar_x2, bar_y + bar_h,
            outline=accent_dim, fill=PANEL_DARK, width=1,
        )
        # Fill — clamp to [0,100] and convert to pixel width.
        pct_clamped = max(0.0, min(100.0, pct))
        fill_w = int(bar_w * (pct_clamped / 100.0))
        if fill_w > 0:
            self.canvas.create_rectangle(
                bar_x1 + 1, bar_y + 1, bar_x1 + fill_w, bar_y + bar_h - 1,
                outline="", fill=accent,
            )
        # Percent label sits centered on the bar.
        self.canvas.create_text(
            self.w / 2, bar_y + bar_h / 2,
            text=f"{pct_clamped:.0f}%",
            fill=TEXT, font=("Consolas", 7, "bold"),
        )

    def _draw_thumbnail(self, pct: float, accent: str, accent_dim: str):
        """A tiny silhouette of the build plate with a rising fill
        proportional to % complete. Sits in the lower-left of the panel."""
        cx_start = 12
        cy_top   = 80
        size     = 48
        # Build-plate trapezoid (mimics a 256x256 H2D bed seen from front).
        self.canvas.create_polygon(
            cx_start,        cy_top + size,
            cx_start + size, cy_top + size,
            cx_start + size - 4, cy_top + 6,
            cx_start + 4,    cy_top + 6,
            outline=accent_dim, fill=PANEL_DARK, width=1,
        )
        # Rising fill: clipped to the trapezoid by simple linear interp on the
        # vertical edges. Tkinter has no clip primitive, so we approximate
        # with a polygon whose top edge is at fill_y.
        pct_clamped = max(0.0, min(100.0, pct))
        # Map % to a fill height inside the trapezoid (leave a 1px gap so the
        # outline stays visible).
        usable_h = size - 6 - 2
        fill_h   = int(usable_h * (pct_clamped / 100.0))
        if fill_h > 0:
            fill_y    = cy_top + size - 1 - fill_h
            # Linearly interpolate the trapezoid x-edges at fill_y.
            t = (fill_y - (cy_top + 6)) / (size - 6) if (size - 6) else 0.0
            t = max(0.0, min(1.0, t))
            left_x  = (cx_start + 4) + (cx_start - (cx_start + 4)) * (1 - t)
            right_x = (cx_start + size - 4) + ((cx_start + size) - (cx_start + size - 4)) * (1 - t)
            self.canvas.create_polygon(
                left_x,        fill_y,
                right_x,       fill_y,
                cx_start + size - 1, cy_top + size - 1,
                cx_start + 1,        cy_top + size - 1,
                outline="", fill=accent,
            )

    def _draw_text_block(self, state: dict, accent: str, accent_dim: str):
        """Right column: layer, ETA, temps. Stays put regardless of the
        thumbnail in the lower left."""
        col_x = 72  # right of the thumbnail
        # Filename on the top line (above the progress bar, full width).
        fname = _shorten_filename(state.get("filename") or "", max_len=34)
        self.canvas.create_text(
            12, 18, anchor="w",
            text=fname,
            fill=TEXT, font=("Segoe UI", 9, "bold"),
        )
        # State chip on the right edge of the title line.
        gcode = (state.get("gcode_state") or "").upper()
        chip  = gcode if gcode else "—"
        if gcode == "RUNNING":
            chip = "PRINTING"
        elif gcode == "PAUSE":
            chip = "PAUSED"
        elif gcode == "FINISH":
            chip = "DONE"
        elif gcode == "FAILED":
            chip = "FAILED"
        self.canvas.create_text(
            self.w - 12, 18, anchor="e",
            text=chip,
            fill=accent, font=("Consolas", 7, "bold"),
        )

        # Layer / total — directly under the filename.
        layer = state.get("layer_num")
        total = state.get("total_layer")
        if layer and total:
            layer_str = f"Layer {int(layer)} / {int(total)}"
        elif layer:
            layer_str = f"Layer {int(layer)}"
        else:
            layer_str = "Layer —"
        self.canvas.create_text(
            12, 36, anchor="w",
            text=layer_str,
            fill=DIM_TEXT, font=("Segoe UI", 8),
        )

        # ETA on the right of the layer line.
        eta = _format_minutes(state.get("mc_remaining"))
        eta_str = f"ETA {eta}" if eta else "ETA —"
        self.canvas.create_text(
            self.w - 12, 36, anchor="e",
            text=eta_str,
            fill=DIM_TEXT, font=("Segoe UI", 8),
        )

        # Temps — beside the thumbnail.
        nozzle = _format_temp(state.get("nozzle_temper"))
        bed    = _format_temp(state.get("bed_temper"))
        chamber = state.get("chamber_temper")
        self.canvas.create_text(
            col_x, 82, anchor="w",
            text=f"Nozzle {nozzle}",
            fill=TEXT, font=("Segoe UI", 8),
        )
        self.canvas.create_text(
            col_x, 100, anchor="w",
            text=f"Bed    {bed}",
            fill=TEXT, font=("Segoe UI", 8),
        )
        chamber_str = _format_temp(chamber)
        if chamber_str != "—":
            self.canvas.create_text(
                col_x, 118, anchor="w",
                text=f"Chmbr  {chamber_str}",
                fill=DIM_TEXT, font=("Segoe UI", 8),
            )

        # Risk note (only when risk > 0).
        risk = int(state.get("risk_level", 0) or 0)
        if risk >= 1:
            note = state.get("risk_note") or ("Risk detected" if risk >= 2 else "Watching")
            self.canvas.create_text(
                self.w / 2, self.h - 8, anchor="s",
                text=str(note)[:42],
                fill=accent, font=("Segoe UI", 7, "italic"),
            )

    def tick(self):
        if not _is_parent_alive(self.parent_pid):
            self._on_close()
            return

        self.frame += 1
        self.canvas.delete("all")

        state = _read_state()
        try:
            pct = float(state.get("mc_percent") or 0)
        except (TypeError, ValueError):
            pct = 0.0
        try:
            risk = int(state.get("risk_level") or 0)
        except (TypeError, ValueError):
            risk = 0

        # Animate a subtle pulse on the accent border by mixing the
        # accent with its dim partner — feels alive without being noisy.
        accent_base, accent_dim = _risk_palette(risk)
        pulse_t = 0.5 + 0.5 * math.sin(self.frame * 0.18)
        accent = _mix(accent_dim, accent_base, 0.6 + 0.4 * pulse_t)

        self._draw_panel(accent, accent_dim)
        self._draw_text_block(state, accent, accent_dim)
        self._draw_progress_bar(pct, accent, accent_dim)
        self._draw_thumbnail(pct, accent, accent_dim)

        self.root.after(TICK_MS, self.tick)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=2260)
    parser.add_argument("--y", type=int, default=-1420)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--height", type=int, default=140)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    app = BambuOverlay(args.x, args.y, args.width, args.height, args.parent_pid)
    try:
        app.root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
