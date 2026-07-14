#!/usr/bin/env python3
"""
Workshop print monitor HUD — Stark-style top-center panel for the top
monitor.

Spec: jarvis_todo.md 2026-05-29 23:27 (overnight). A persistent visual
surface that surfaces while the user is in the workshop OR a print is
active, summarising the H2D's nozzle/bed/layer/ETA/progress in an
MCU-authentic cyan panel that complements the existing workshop_hud
(top-right) and arc_reactor_status_hud (top-left).

The skill-side watcher (`_workshop_print_monitor_watcher` in
holographic_overlay.py) spawns this subprocess when either
`bambu_monitor` reports an active print or `workshop_mode` flags a CAD
window as focused, and signals shutdown via the control file
`workshop_print_monitor_state.json` (`mode=off`).

State sources:
  • bambu_overlay_state.json — atomic-written by bambu_monitor on every
    MQTT report. Owns nozzle / bed / chamber / layer / total / pct / ETA /
    print_error / risk_level / filename.
  • workshop_print_monitor_state.json — control file. The watcher writes
    `mode=off` here so the widget self-exits without racing terminate().

Filament-remaining is intentionally a placeholder (—): bambu_monitor's
`_state` does not surface filament-remaining (planner regression-risk
note), and fetching the real .3mf preview over FTP would block tkinter
and require credentials beyond what bambu_monitor uses.

CLI:
  python hud/workshop_print_monitor.py --x 1080 --y -1430 --width 400 \
                                       --height 200 --parent-pid 12345
"""
import argparse
import json
import math
import os
import sys
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


TICK_MS = 250  # 4 fps — bambu MQTT pushes ~1/min, anything faster is wasted CPU.

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
CONTROL_FILE   = os.path.join(PROJECT_DIR, "workshop_print_monitor_state.json")


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
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _control_says_off() -> bool:
    data = _read_json(CONTROL_FILE)
    return (data.get("mode") or "").lower() == "off"


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


def _shorten(text: str, max_len: int) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _risk_palette(risk: int):
    """Return (accent, accent_dim) pair for the given risk_level."""
    if risk >= 2:
        return RED, RED_DIM
    if risk == 1:
        return AMBER, AMBER_DIM
    return CYAN, CYAN_DIM


class WorkshopPrintMonitor:
    """Stark-style HUD panel pinned to the top-center of the top monitor.

    Layout (400×200 default):
      ┌─[ accent strip ]───────────────────────────────────────────────┐
      │ J.A.R.V.I.S. ▸ WORKSHOP PRINT MONITOR        STATE     · live │
      │ filename.3mf                                                   │
      │ Layer 47 / 312                              ETA 18m            │
      │ ┌── synthetic thumbnail ──┐  Nozzle 220°    Bed 60°            │
      │ │                         │  Chamber 36°   Filament —          │
      │ │  ▓▓▓▓▓▓▓▓               │                                    │
      │ │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓         │  [████████░░░░░░░░░░░] 42%         │
      │ └─────────────────────────┘                                    │
      └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int):
        self.parent_pid = parent_pid
        self.w, self.h = w, h
        self.frame = 0

        self.root = tk.Tk()
        self.root.title("JARVIS Workshop Print Monitor")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.94)
            except Exception:
                pass

        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=w, height=h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Double-right-click anywhere to dismiss (matches sibling overlays).
        self.canvas.bind("<Double-Button-3>", lambda _e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.tick()

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── drawing primitives ────────────────────────────────────────────────

    def _draw_panel(self, accent: str, accent_dim: str):
        # Outer chassis — Stark-style angled corner cuts approximated with
        # a clipped polygon so the panel reads as a sci-fi readout, not a
        # plain rectangle.
        cut = 16
        self.canvas.create_polygon(
            cut,          2,
            self.w - 2,   2,
            self.w - 2,   self.h - cut,
            self.w - cut, self.h - 2,
            2,            self.h - 2,
            2,            cut,
            outline=PANEL_RIM, fill=PANEL_DARK, width=1,
        )
        # Top accent strip — colour-coded by risk so a glance tells the
        # user "is this print healthy?".
        self.canvas.create_polygon(
            cut + 1,      3,
            self.w - 3,   3,
            self.w - 3,   7,
            cut + 1,      7,
            outline="", fill=accent,
        )
        # Inner border — gives the panel an arc-reactor finish.
        inset = 6
        self.canvas.create_polygon(
            cut + inset,           inset + 2,
            self.w - 2 - inset,    inset + 2,
            self.w - 2 - inset,    self.h - cut - inset,
            self.w - cut - inset,  self.h - 2 - inset,
            2 + inset,             self.h - 2 - inset,
            2 + inset,             cut + inset,
            outline=accent_dim, fill="", width=1,
        )

    def _draw_header(self, state: dict, accent: str, accent_dim: str):
        # Title row — left: identity tag, right: state chip + live tick.
        self.canvas.create_text(
            22, 16, anchor="w",
            text="J . A . R . V . I . S .  ▸  WORKSHOP PRINT MONITOR",
            fill=TEXT, font=("Consolas", 8, "bold"),
        )

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
        elif gcode == "PREPARE":
            chip = "PREPARING"
        # Live tick — alternating ●/○ next to the state chip so the user
        # can tell at a glance whether the widget is still receiving data.
        tick_char = "●" if (self.frame % 4 < 2) else "○"
        chip_text = f"{chip}  {tick_char}"
        self.canvas.create_text(
            self.w - 22, 16, anchor="e",
            text=chip_text,
            fill=accent, font=("Consolas", 8, "bold"),
        )

        # Filename underneath the title row.
        fname = _shorten(state.get("filename") or "(no active print)", 48)
        self.canvas.create_text(
            22, 34, anchor="w",
            text=fname,
            fill=DIM_TEXT, font=("Segoe UI", 9),
        )

    def _draw_thumbnail(self, pct: float, accent: str, accent_dim: str):
        """A small synthetic build-plate silhouette in the lower-left with
        a rising fill that tracks % complete. Planner explicitly called
        out that the real .3mf preview is out of scope."""
        size_x = 90
        size_y = 70
        x0 = 22
        y0 = self.h - size_y - 18
        # Build-plate trapezoid mimicking the H2D bed from the front.
        self.canvas.create_polygon(
            x0,            y0 + size_y,
            x0 + size_x,   y0 + size_y,
            x0 + size_x - 8, y0 + 8,
            x0 + 8,        y0 + 8,
            outline=accent_dim, fill=PANEL_DARK, width=1,
        )
        # Rising fill — clipped to the trapezoid via vertical interp on
        # the left/right edges (tkinter has no clip primitive).
        pct_clamped = max(0.0, min(100.0, pct))
        usable_h = size_y - 8 - 2
        fill_h   = int(usable_h * (pct_clamped / 100.0))
        if fill_h > 0:
            fill_y = y0 + size_y - 1 - fill_h
            t = (fill_y - (y0 + 8)) / (size_y - 8) if (size_y - 8) else 0.0
            t = max(0.0, min(1.0, t))
            left_x  = (x0 + 8) + (x0 - (x0 + 8)) * (1 - t)
            right_x = (x0 + size_x - 8) + ((x0 + size_x) - (x0 + size_x - 8)) * (1 - t)
            self.canvas.create_polygon(
                left_x,            fill_y,
                right_x,           fill_y,
                x0 + size_x - 1,   y0 + size_y - 1,
                x0 + 1,            y0 + size_y - 1,
                outline="", fill=accent,
            )
        # "Thumbnail" caption underneath.
        self.canvas.create_text(
            x0 + size_x / 2, y0 + size_y + 8, anchor="n",
            text="PREVIEW",
            fill=DIM_TEXT, font=("Consolas", 6),
        )
        return x0 + size_x  # right edge so callers can lay text to the right

    def _draw_layer_eta(self, state: dict, accent: str, accent_dim: str):
        layer = state.get("layer_num")
        total = state.get("total_layer")
        if layer and total:
            layer_str = f"Layer {int(layer)} / {int(total)}"
        elif layer:
            layer_str = f"Layer {int(layer)}"
        else:
            layer_str = "Layer —"
        self.canvas.create_text(
            22, 56, anchor="w",
            text=layer_str,
            fill=TEXT, font=("Segoe UI", 10, "bold"),
        )
        eta = _format_minutes(state.get("mc_remaining"))
        eta_str = f"ETA  {eta}" if eta else "ETA  —"
        self.canvas.create_text(
            self.w - 22, 56, anchor="e",
            text=eta_str,
            fill=TEXT, font=("Segoe UI", 10, "bold"),
        )

    def _draw_telemetry(self, state: dict, accent: str, accent_dim: str,
                        text_left_x: int):
        """Right column beside the thumbnail: nozzle / bed / chamber /
        filament. Planner: filament-remaining is not tracked by
        bambu_monitor, so it's always a `—` placeholder."""
        nozzle = _format_temp(state.get("nozzle_temper"))
        bed    = _format_temp(state.get("bed_temper"))
        chamber = _format_temp(state.get("chamber_temper"))
        col_x = text_left_x + 22
        row_y = self.h - 88
        line_h = 18
        self.canvas.create_text(
            col_x, row_y, anchor="w",
            text=f"Nozzle    {nozzle}",
            fill=TEXT, font=("Consolas", 9),
        )
        self.canvas.create_text(
            col_x, row_y + line_h, anchor="w",
            text=f"Bed       {bed}",
            fill=TEXT, font=("Consolas", 9),
        )
        chamber_color = DIM_TEXT if chamber == "—" else TEXT
        self.canvas.create_text(
            col_x, row_y + 2 * line_h, anchor="w",
            text=f"Chamber   {chamber}",
            fill=chamber_color, font=("Consolas", 9),
        )
        # Filament-remaining placeholder — bambu_monitor doesn't track it.
        self.canvas.create_text(
            col_x, row_y + 3 * line_h, anchor="w",
            text="Filament  —",
            fill=DIM_TEXT, font=("Consolas", 9),
        )

    def _draw_progress_bar(self, pct: float, accent: str, accent_dim: str):
        margin_x = 22
        bar_y    = self.h - 22
        bar_h    = 8
        bar_x1   = margin_x
        bar_x2   = self.w - margin_x
        bar_w    = bar_x2 - bar_x1
        self.canvas.create_rectangle(
            bar_x1, bar_y, bar_x2, bar_y + bar_h,
            outline=accent_dim, fill=PANEL_DARK, width=1,
        )
        pct_clamped = max(0.0, min(100.0, pct))
        fill_w = int(bar_w * (pct_clamped / 100.0))
        if fill_w > 0:
            self.canvas.create_rectangle(
                bar_x1 + 1, bar_y + 1, bar_x1 + fill_w, bar_y + bar_h - 1,
                outline="", fill=accent,
            )
        self.canvas.create_text(
            self.w / 2, bar_y + bar_h / 2,
            text=f"{pct_clamped:.0f}%",
            fill=TEXT, font=("Consolas", 7, "bold"),
        )

    def _draw_risk_note(self, state: dict, accent: str):
        risk = int(state.get("risk_level", 0) or 0)
        if risk >= 1:
            note = state.get("risk_note") or (
                "Risk detected" if risk >= 2 else "Watching"
            )
            self.canvas.create_text(
                self.w / 2, self.h - 36, anchor="s",
                text=str(note)[:60],
                fill=accent, font=("Segoe UI", 7, "italic"),
            )

    # ── main loop ─────────────────────────────────────────────────────────

    def tick(self):
        if not _is_parent_alive(self.parent_pid):
            self._on_close()
            return
        if _control_says_off():
            self._on_close()
            return

        self.frame += 1
        self.canvas.delete("all")

        state = _read_json(STATE_FILE)
        try:
            pct = float(state.get("mc_percent") or 0)
        except (TypeError, ValueError):
            pct = 0.0
        try:
            risk = int(state.get("risk_level") or 0)
        except (TypeError, ValueError):
            risk = 0

        # Pulse the accent against its dim partner — feels alive without
        # blinking distractingly.
        accent_base, accent_dim = _risk_palette(risk)
        pulse_t = 0.5 + 0.5 * math.sin(self.frame * 0.18)
        accent = _mix(accent_dim, accent_base, 0.6 + 0.4 * pulse_t)

        self._draw_panel(accent, accent_dim)
        self._draw_header(state, accent, accent_dim)
        self._draw_layer_eta(state, accent, accent_dim)
        thumb_right = self._draw_thumbnail(pct, accent, accent_dim)
        self._draw_telemetry(state, accent, accent_dim, thumb_right)
        self._draw_progress_bar(pct, accent, accent_dim)
        self._draw_risk_note(state, accent)

        self.root.after(TICK_MS, self.tick)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=1080)
    parser.add_argument("--y", type=int, default=-1430)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    app = WorkshopPrintMonitor(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    try:
        app.root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
