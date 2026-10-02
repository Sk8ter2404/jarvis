#!/usr/bin/env python3
"""
JARVIS holographic globe — a slowly rotating wireframe Earth on any monitor.

The Iron-Man "pull up the globe" moment: an orthographic wireframe Earth
(graticule every 15 degrees + simplified coastlines, far side culled) inside a
soft cyan glow ring. JARVIS can drop pins on it — each one a pulsing amber
ring with a label — joined in order by raised great-circle arcs, and the globe
eases round to face every new pin before it resumes its slow spin.

Launched and stopped by skills/globe.py (show_globe / hide_globe / globe_pin /
globe_clear), exactly like the holographic overlays in
skills/holographic_overlay: one subprocess, --x/--y/--width/--height/
--parent-pid CLI, CREATE_NO_WINDOW. Nothing launches it at boot.

Control file:
  globe_hud_state.json at the project root — written ONLY by the skill:
    {"mode": "on"|"off", "pins": [{"lat", "lon", "label"}, ...], "seq": n}
  "off" closes the globe; a new "seq" with pins turns the globe to the last
  pin. Polled by mtime four times a second (a stat, not a read).

Look (same palette + window treatment as hud/jarvis_holo.py):
  • frameless, always-on-top; Win32 -transparentcolor keys the background
    out so only the globe itself is on screen (other platforms fall back to
    window alpha).
  • Escape (after a click gives it focus) or a double-right-click closes it.

CPU: the static layers (glow ring, disc, captions) are drawn once; canvas
items are reused (coords updates, never delete-and-redraw); the wireframe is
only re-projected once the spin has moved it a whole pixel, and the parallels
only when the tilt changes (a spin about the pole leaves them unchanged).
Ticks are capped at ~30 fps and drop to the one-pixel cadence when nothing
pulses.

Geometry lives in hud/globe_geometry.py (pure, tested without a display).

CLI:
  python hud/globe_hud.py --x 840 --y -1173 --width 893 --height 893 \
                          --parent-pid 12345
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk

# hud/ is not a package root — put the project dir on sys.path so
# `from core.parent_watch import ...` resolves, and this directory so the
# sibling geometry module imports however the script was started.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except Exception:
    pass

import globe_geometry as geo

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ──────────────────────────────────────────────────────────────────────────
#  Timing
# ──────────────────────────────────────────────────────────────────────────
TICK_MS          = 33      # ~30 fps cap
IDLE_TICK_MAX_MS = 250     # slowest tick when nothing pulses or tweens
SPIN_DEG_PER_S   = 3.0     # one revolution every two minutes
DEFAULT_TILT     = 18.0    # view latitude while spinning (look from above)
TILT_RETURN_DEG_PER_S = 12.0
TWEEN_MIN_S      = 0.9     # rotate-to-pin duration, scaled by the distance
TWEEN_MAX_S      = 2.2
HOLD_S           = 6.0     # stay on a new pin this long before spinning on
PULSE_PERIOD_S   = 1.6
CONTROL_POLL_S   = 0.25
PARENT_POLL_S    = 1.0
MAX_PINS         = 12
# With --parent-pid 0/absent there is no parent to track: self-exit after
# this long rather than linger as a parentless topmost window (mirrors
# hud/holographic_hud_v2.py).
ORPHAN_MAX_LIFETIME_S = 1800.0

# Layout fractions of min(width, height).
R_GLOBE_FRAC = 0.38

# ──────────────────────────────────────────────────────────────────────────
#  Palette — hud/jarvis_holo.py's, so the globe reads as the same system.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY       = "#010101"  # transparentcolor target on Win32
PANEL_COLOR  = "#04080d"
CYAN         = "#4cc9ff"
CYAN_DIM     = "#1b4a66"
CYAN_BRIGHT  = "#9ee7ff"
TEXT_COLOR   = "#cfeefb"
DIM_TEXT     = "#5d8aa3"
AMBER        = "#ffb347"
AMBER_DIM    = "#7a5520"
AMBER_BRIGHT = "#ffe0a0"
GOLD         = "#ffd166"

PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_FILE = os.path.join(PROJECT_DIR, "globe_hud_state.json")
# Made with Natural Earth - public domain (see the file's "_source").
COASTLINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "globe_coastline.json")


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    # AUTHORITATIVE liveness first — see hud/jarvis_holo.py: psutil.pid_exists
    # reads TRUE for a dead-but-unreaped Windows process. Fail-open to the
    # historical checks if the helper is unavailable.
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
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_pins(data: dict) -> list:
    """The control file's pins as [(lat, lon, label)], dropping anything
    malformed or out of range, newest last, at most MAX_PINS."""
    pins = []
    raw = data.get("pins") if isinstance(data, dict) else None
    for p in raw if isinstance(raw, list) else []:
        try:
            lat = float(p["lat"])
            lon = float(p["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        label = str(p.get("label") or "").strip()[:40]
        pins.append((lat, lon, label))
    return pins[-MAX_PINS:]


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t))


def _fmt_latlon(lat: float, lon: float) -> str:
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}  "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


class GlobeHUD:
    def __init__(self, x: int, y: int, width: int, height: int, parent_pid: int):
        self.parent_pid = parent_pid
        self._closing = False
        self.w, self.h = width, height
        self.cx = self.w / 2
        self.cy = self.h / 2
        self.R = min(self.w, self.h) * R_GLOBE_FRAC
        self._dpp = geo.degrees_per_pixel(self.R)

        now = time.monotonic()
        self._started_at = now
        self._last_tick = now
        self._next_control_poll = 0.0
        self._next_parent_poll = now + PARENT_POLL_S
        self._control_mtime = None
        self._seq = None

        # View state machine: "spin" → (new pin) "tween" → "hold" → "spin".
        self.view_lat = DEFAULT_TILT
        self.view_lon = 0.0
        self._mode = "spin"
        self._tween_from = (0.0, 0.0)
        self._tween_to = (0.0, 0.0)
        self._tween_t0 = 0.0
        self._tween_len = 1.0
        self._hold_until = 0.0
        # Views the layers were last projected for (None = must redraw).
        self._drawn_view = None
        self._drawn_par_lat = None

        try:
            lines = _read_json(COASTLINE_FILE).get("lines") or []
            self._coast = geo.coastline_polylines(lines)
        except Exception:
            self._coast = []
        self._meridians = geo.meridians()
        self._parallels = geo.parallels()

        self.pins = []          # [(lat, lon, label)]
        self._pin_xyz = []
        self._arcs = []         # polylines of sphere points
        self._pin_screen = []   # [(sx, sy) | None] at the last projection

        self.root = tk.Tk()
        self.root.title("JARVIS Globe")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        # Win32 keyed transparency — pixels drawn in BG_KEY vanish (and are
        # click-through). Falls back to global alpha on other platforms.
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.85)
            except Exception:
                pass
        self.root.geometry(f"{self.w}x{self.h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=self.w, height=self.h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Escape needs keyboard focus, which a frameless topmost window only
        # gets when clicked — so a click on the globe takes focus (it never
        # steals it on launch). Double-right-click dismisses outright, as on
        # the fullscreen holo overlay.
        self.canvas.bind("<Button-1>", lambda _e: self._focus())
        self.root.bind("<Escape>", lambda _e: self._on_close())
        self.canvas.bind("<Double-Button-3>", lambda _e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_static()
        self._poll_control(force=True)
        self.tick()

    # ─── lifecycle ──────────────────────────────────────────────────────
    def _focus(self):
        try:
            self.root.focus_force()
        except Exception:
            pass

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass

    # ─── static layers + item pools ─────────────────────────────────────
    def _build_static(self):
        c = self.canvas
        cx, cy, R = self.cx, self.cy, self.R
        # Soft glow ring: touching concentric rings fading outward (tkinter
        # has no blur, same trick as the holo overlay's halo). Drawn once.
        steps = 9
        for i in reversed(range(steps)):
            r = R + 2 + 3 * i
            color = _mix(CYAN, PANEL_COLOR, 0.25 + 0.75 * i / steps)
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=color, width=3)
        # The disc body: gives the hologram a surface and makes it clickable.
        c.create_oval(cx - R, cy - R, cx + R, cy + R,
                      fill=PANEL_COLOR, outline=CYAN, width=2)

        # One invisible anchor per layer, in z-order. A pooled item is
        # lowered just under its layer's anchor, so layers keep their stacking
        # however late their items are created.
        self._anchor = {}
        for layer in ("par", "mer", "coast", "arc", "ring", "dot", "label"):
            self._anchor[layer] = c.create_line(0, 0, 0, 0, state="hidden")
        self._pool = {k: [] for k in ("par", "mer", "coast", "arc")}
        self._shown = {k: 0 for k in self._pool}
        self._style = {
            "par":   {"fill": CYAN_DIM, "width": 1},
            "mer":   {"fill": CYAN_DIM, "width": 1},
            "coast": {"fill": CYAN, "width": 1},
            "arc":   {"fill": GOLD, "width": 2},
        }
        self._pin_items = []    # [(ring, dot, label)]

        c.create_text(cx, 20, text="J . A . R . V . I . S .   —   GLOBAL VIEW",
                      fill=DIM_TEXT, font=("Consolas", 11, "bold"))
        self._readout = c.create_text(
            cx, self.h - 42, text="", fill=AMBER,
            font=("Consolas", 12, "bold"))
        c.create_text(cx, self.h - 16,
                      text="esc / double-right-click to dismiss",
                      fill=DIM_TEXT, font=("Consolas", 9))

    def _set_runs(self, layer: str, runs: list):
        """Show `runs` on the layer's pooled line items: reuse, grow, and
        hide the surplus — never delete-and-recreate. Invariant: the first
        self._shown[layer] items of the pool are visible, the rest hidden."""
        c = self.canvas
        pool = self._pool[layer]
        while len(pool) < len(runs):
            item = c.create_line(0, 0, 0, 0, state="hidden",
                                 **self._style[layer])
            c.tag_lower(item, self._anchor[layer])
            pool.append(item)
        for item, run in zip(pool, runs):
            c.coords(item, run)
        if len(runs) < self._shown[layer]:
            for item in pool[len(runs):self._shown[layer]]:
                c.itemconfigure(item, state="hidden")
        elif len(runs) > self._shown[layer]:
            for item in pool[self._shown[layer]:len(runs)]:
                c.itemconfigure(item, state="normal")
        self._shown[layer] = len(runs)

    # ─── pins ───────────────────────────────────────────────────────────
    def _set_pins(self, pins: list):
        c = self.canvas
        self.pins = pins
        self._pin_xyz = [geo.latlon_to_xyz(lat, lon) for lat, lon, _ in pins]
        self._arcs = []
        for (a_lat, a_lon, _), (b_lat, b_lon, _) in zip(pins, pins[1:]):
            dist = geo.angular_distance(a_lat, a_lon, b_lat, b_lon)
            lift = 0.03 + 0.12 * dist / 180.0
            self._arcs.append(geo.great_circle_arc(
                a_lat, a_lon, b_lat, b_lon, step_deg=2.0, lift=lift))
        while len(self._pin_items) < len(pins):
            ring = c.create_oval(0, 0, 0, 0, outline=AMBER, width=2)
            dot = c.create_oval(0, 0, 0, 0, fill=AMBER_BRIGHT, outline=AMBER)
            label = c.create_text(0, 0, text="", anchor="w", fill=TEXT_COLOR,
                                  font=("Consolas", 10, "bold"))
            c.tag_lower(ring, self._anchor["ring"])
            c.tag_lower(dot, self._anchor["dot"])
            c.tag_lower(label, self._anchor["label"])
            self._pin_items.append((ring, dot, label))
        for i, items in enumerate(self._pin_items):
            if i < len(pins):
                c.itemconfigure(items[2], text=pins[i][2].upper())
            else:
                for item in items:
                    c.itemconfigure(item, state="hidden")
        if pins:
            lat, lon, label = pins[-1]
            name = f"{label.upper()}  ·  " if label else ""
            c.itemconfigure(self._readout, text=f"{name}{_fmt_latlon(lat, lon)}")
        else:
            c.itemconfigure(self._readout, text="")
        self._pin_screen = [None] * len(pins)
        self._drawn_view = None          # arcs changed: re-project

    def _focus_on(self, lat: float, lon: float, now: float):
        """Start the smooth rotate-to-pin."""
        target = (max(-70.0, min(70.0, lat)), geo.wrap_lon(lon))
        start = (self.view_lat, self.view_lon)
        dist = geo.angular_distance(start[0], start[1], target[0], target[1])
        self._tween_from = start
        self._tween_to = target
        self._tween_t0 = now
        self._tween_len = TWEEN_MIN_S + (TWEEN_MAX_S - TWEEN_MIN_S) * dist / 180.0
        self._mode = "tween"

    # ─── control file ───────────────────────────────────────────────────
    def _poll_control(self, force: bool = False):
        try:
            mtime = os.stat(CONTROL_FILE).st_mtime_ns
        except OSError:
            mtime = None
        if not force and mtime == self._control_mtime:
            return
        self._control_mtime = mtime
        data = _read_json(CONTROL_FILE)
        if (data.get("mode") or "").lower() == "off":
            self._on_close()
            return
        pins = parse_pins(data)
        seq = data.get("seq")
        if pins != self.pins:
            self._set_pins(pins)
        if seq != self._seq:
            self._seq = seq
            if pins:
                self._focus_on(pins[-1][0], pins[-1][1], time.monotonic())
        if not pins and self._mode != "spin":
            self._mode = "spin"

    # ─── per-tick view update ───────────────────────────────────────────
    def _advance_view(self, now: float, dt: float):
        if self._mode == "tween":
            t = (now - self._tween_t0) / max(0.01, self._tween_len)
            self.view_lat, self.view_lon = geo.tween_view(
                self._tween_from, self._tween_to, t)
            if t >= 1.0:
                self._mode = "hold"
                self._hold_until = now + HOLD_S
        elif self._mode == "hold":
            if now >= self._hold_until:
                self._mode = "spin"
        else:
            self.view_lon = geo.wrap_lon(self.view_lon + SPIN_DEG_PER_S * dt)
            self.view_lat = geo.approach(
                self.view_lat, DEFAULT_TILT, TILT_RETURN_DEG_PER_S * dt)

    def _redraw_globe(self):
        basis = geo.view_basis(self.view_lat, self.view_lon)
        cx, cy, R = self.cx, self.cy, self.R
        if (self._drawn_par_lat is None
                or abs(self.view_lat - self._drawn_par_lat) >= self._dpp * 0.5):
            self._set_runs("par", geo.project_runs(self._parallels, basis, cx, cy, R))
            self._drawn_par_lat = self.view_lat
        self._set_runs("mer", geo.project_runs(self._meridians, basis, cx, cy, R))
        self._set_runs("coast", geo.project_runs(self._coast, basis, cx, cy, R))
        self._set_runs("arc", geo.project_runs(self._arcs, basis, cx, cy, R))
        c = self.canvas
        for i, xyz in enumerate(self._pin_xyz):
            pos = geo.to_screen(xyz, basis, cx, cy, R)
            was = self._pin_screen[i]
            self._pin_screen[i] = pos
            ring, dot, label = self._pin_items[i]
            if pos is None:
                if was is not None or self._drawn_view is None:
                    for item in (ring, dot, label):
                        c.itemconfigure(item, state="hidden")
                continue
            sx, sy = pos
            c.coords(dot, sx - 4, sy - 4, sx + 4, sy + 4)
            c.coords(label, sx + 12, sy - 12)
            if was is None:
                for item in (ring, dot, label):
                    c.itemconfigure(item, state="normal")
        self._drawn_view = (self.view_lat, self.view_lon)

    def _pulse(self, now: float):
        c = self.canvas
        n = len(self.pins)
        for i, pos in enumerate(self._pin_screen):
            if pos is None:
                continue
            phase = ((now - self._started_at) / PULSE_PERIOD_S + i * 0.17) % 1.0
            r = 6.0 + 16.0 * phase
            sx, sy = pos
            ring = self._pin_items[i][0]
            c.coords(ring, sx - r, sy - r, sx + r, sy + r)
            hot = AMBER_BRIGHT if i == n - 1 else AMBER
            c.itemconfigure(ring, outline=_mix(hot, AMBER_DIM, phase))

    def _needs_redraw(self) -> bool:
        if self._drawn_view is None:
            return True
        dlat = abs(self.view_lat - self._drawn_view[0])
        dlon = abs(geo.shortest_lon_delta(self._drawn_view[1], self.view_lon))
        return dlat >= self._dpp or dlon >= self._dpp

    # ─── main loop ──────────────────────────────────────────────────────
    def tick(self):
        # Render in a guarded body and ALWAYS reschedule (unless closing) so
        # one bad value can never strand the window on a frozen frame.
        if self._closing:
            return
        delay = TICK_MS
        try:
            delay = self._tick_body()
        except Exception:
            pass
        finally:
            if not self._closing:
                try:
                    self.root.after(int(delay or TICK_MS), self.tick)
                except Exception:
                    pass

    def _tick_body(self) -> int:
        now = time.monotonic()
        dt = min(0.5, now - self._last_tick)
        self._last_tick = now

        if now >= self._next_parent_poll:
            self._next_parent_poll = now + PARENT_POLL_S
            if not _is_parent_alive(self.parent_pid):
                self._on_close()
                return TICK_MS
            if (self.parent_pid <= 0
                    and now - self._started_at > ORPHAN_MAX_LIFETIME_S):
                self._on_close()
                return TICK_MS
        if now >= self._next_control_poll:
            self._next_control_poll = now + CONTROL_POLL_S
            self._poll_control()
            if self._closing:
                return TICK_MS

        self._advance_view(now, dt)
        if self._needs_redraw():
            self._redraw_globe()
        if self.pins:
            self._pulse(now)

        if self.pins or self._mode == "tween":
            return TICK_MS
        # Nothing pulsing: wake again when the spin has moved a pixel.
        ms = 1000.0 * self._dpp / SPIN_DEG_PER_S
        return int(max(TICK_MS, min(IDLE_TICK_MAX_MS, ms)))

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    hud = GlobeHUD(args.x, args.y, args.width, args.height, args.parent_pid)
    hud.run()


if __name__ == "__main__":
    main()
