#!/usr/bin/env python3
"""
JARVIS holographic workshop canvas — a compact 3D-style rotating arc
reactor anchored to the corner of the top monitor. Distinct from the
existing fullscreen jarvis_holo.py overlay: this one is small, focused,
and meant to give JARVIS a constant visual "presence" during thinking
and speaking states without consuming the whole screen.

Spec: jarvis_todo.md 2026-05-27 10:38 (holo_workshop_canvas).

Rendered elements:
  • Rotating "blade" ring — six thin curved blades that orbit the core
    on a tilted axis. The tilt is faked by stretching the orbit ellipse
    vertically and modulating each blade's brightness based on its
    apparent depth (a blade on the "near" side glows brighter than one
    on the "far" side). This gives a convincing pseudo-3D rotation
    without an actual 3D engine.
  • Concentric depth-shaded core — three nested rings in graduated cyan
    so the hub reads as a glowing orb rather than a flat ring.
  • Pulsing outer ring — radius and brightness scale with the live TTS
    amplitude (state["tts_amplitude"]) so JARVIS visibly "speaks" in the
    canvas during responses.
  • Status label — the lower-case state (thinking / speaking / idle).

State sources:
  • hud_state.json — written by bobert_companion. Provides `state` and
    `tts_amplitude`. Read at the tick rate.
  • holo_workshop_state.json (sibling of hud_state.json) — controls the
    canvas mode without restart. {"mode": "on" | "pulse" | "off",
    "force_visible": bool}. The skill writes this on voice commands.

Click-through:
  • Win32: -transparentcolor BG_KEY → most of the canvas is keyed so
    clicks pass through to whatever is underneath. The reactor itself
    blocks clicks, but it's small.
  • Other platforms: window-level alpha fallback.

Auto-exits when its parent JARVIS process dies.

CLI:
  python hud/holo_workshop_canvas.py --x 2000 --y -1200 --width 320
                                     --height 320 --parent-pid 12345
                                     [--mode on|pulse]
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


TICK_MS = 40  # 25 fps — keeps the pseudo-3D rotation smooth

BG_KEY        = "#010101"
CYAN          = "#4cc9ff"
CYAN_DIM      = "#1b4a66"
CYAN_BRIGHT   = "#9ee7ff"
CYAN_DEEP     = "#0b2a3d"
TEXT_COLOR    = "#cfeefb"
DIM_TEXT      = "#5d8aa3"
AMBER         = "#ffb347"
AMBER_BRIGHT  = "#ffe0a0"
GOLD          = "#ffd166"
VIOLET        = "#9b8cff"

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_STATE_FILE = os.path.join(PROJECT_DIR, "hud_state.json")
WORKSHOP_FILE  = os.path.join(PROJECT_DIR, "holo_workshop_state.json")


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
        # the overlay (matches the PyQt HUDs' guard).
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


class WorkshopCanvas:
    def __init__(self, x: int, y: int, width: int, height: int,
                 parent_pid: int, mode: str = "on"):
        self.parent_pid = parent_pid
        self._closing = False
        self.x, self.y = x, y
        self.w, self.h = width, height
        self.frame = 0
        self.cli_mode = mode if mode in ("on", "pulse") else "on"

        ref = min(self.w, self.h)
        self.cx = self.w / 2
        self.cy = self.h / 2 - 6  # nudge up slightly to leave room for label
        # Core (hub) and ring radii — keyed off canvas size so the reactor
        # rescales cleanly when the user picks a different window size.
        self.R_PULSE_BASE = ref * 0.36   # outer pulsing ring (TTS amp)
        self.R_BLADES     = ref * 0.27   # rotating blade orbit
        self.R_CORE_OUT   = ref * 0.13
        self.R_CORE_MID   = ref * 0.09
        self.R_CORE_HUB   = ref * 0.05

        # Smoothed sensors so the pulse doesn't strobe.
        self.last_amp = 0.0
        self.last_mic = 0.0
        # Phase accumulators — kept across state changes so transitions
        # don't snap the rotation.
        self._phase_blades = 0.0
        self._phase_pulse  = 0.0

        # Last seen non-idle state — used to keep the canvas warm for
        # ~2 s after JARVIS goes idle so it doesn't blink off the instant
        # speaking ends.
        self._last_active_at = 0.0

        self.root = tk.Tk()
        self.root.title("JARVIS Workshop Canvas")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.88)
            except Exception:
                pass

        self.root.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=self.w, height=self.h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Double-right-click to dismiss (matches the fullscreen overlay's
        # escape hatch convention).
        self.canvas.bind("<Double-Button-3>", lambda _e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.tick()

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass

    # ─── helpers ────────────────────────────────────────────────────────
    def _bbox(self, r):
        return (self.cx - r, self.cy - r, self.cx + r, self.cy + r)

    def _bbox_ellipse(self, rx, ry):
        return (self.cx - rx, self.cy - ry, self.cx + rx, self.cy + ry)

    def _circle(self, r, color, width=1, fill=""):
        x1, y1, x2, y2 = self._bbox(r)
        if fill:
            self.canvas.create_oval(x1, y1, x2, y2, outline=color,
                                    width=width, fill=fill)
        else:
            self.canvas.create_oval(x1, y1, x2, y2, outline=color,
                                    width=width)

    def _arc_seg(self, r, start_deg, extent_deg, color, width=2):
        x1, y1, x2, y2 = self._bbox(r)
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=start_deg, extent=extent_deg,
            style="arc", outline=color, width=width,
        )

    # ─── pseudo-3D rotating blades ──────────────────────────────────────
    def _draw_blades(self, base_color, intensity: float):
        """Six blades orbiting on a tilted axis. The axis tilt is faked by
        stretching the orbit vertically (rx > ry) and modulating each
        blade's color by sin(theta) so blades on the 'far' side dim and
        blades on the 'near' side brighten — gives the eye a convincing
        sense of 3D rotation without leaving tkinter's 2D canvas."""
        BLADES = 6
        # Tilt aspect: ry/rx — 0.45 reads as a steeply tilted axis. The
        # tilt itself slowly wobbles so the reactor feels alive.
        tilt = 0.45 + 0.06 * math.sin(self.frame * 0.018)
        rx = self.R_BLADES
        ry = self.R_BLADES * tilt
        # Each blade is a short arc segment drawn as a small ellipse arc
        # centered on its angular position. We draw blades back-to-front
        # so near blades visually overlap far blades.
        positions = []
        for i in range(BLADES):
            theta = self._phase_blades + (i / BLADES) * 2 * math.pi
            # Depth signal: sin(theta) > 0 → near side.
            depth = math.sin(theta)
            positions.append((theta, depth))
        # Sort far-to-near so painter's algorithm places near blades on top.
        positions.sort(key=lambda p: p[1])
        for theta, depth in positions:
            x = self.cx + rx * math.cos(theta)
            y = self.cy + ry * math.sin(theta)
            # Blade size scales with depth so near blades read larger.
            size = 14 + 6 * (depth + 1) * 0.5
            # Brightness: -1 (far) → CYAN_DEEP, +1 (near) → CYAN_BRIGHT.
            depth_t = (depth + 1) * 0.5  # 0..1
            color = _mix(CYAN_DEEP, base_color, 0.25 + 0.75 * depth_t)
            color = _mix(color, CYAN_BRIGHT, 0.3 * intensity * depth_t)
            # Blade is a short curved line — draw as an arc seg on a small
            # circle centered at (x, y). The arc's orientation follows the
            # tangent of the orbit so the blade reads as flowing.
            tangent_deg = math.degrees(math.atan2(
                -ry * math.cos(theta), rx * math.sin(theta)
            ))
            x1 = x - size
            y1 = y - size * 0.5
            x2 = x + size
            y2 = y + size * 0.5
            self.canvas.create_arc(
                x1, y1, x2, y2,
                start=tangent_deg - 60, extent=120,
                style="arc", outline=color, width=2,
            )

    # ─── main paint ─────────────────────────────────────────────────────
    def tick(self):
        # Wrapper: render in a guarded body and ALWAYS reschedule in finally
        # (unless we're closing) so one bad value or transient error can't
        # leave the overlay frozen on a stale frame.
        if getattr(self, "_closing", False):
            return
        try:
            self._tick_body()
        except Exception:
            pass
        finally:
            if not getattr(self, "_closing", False):
                try:
                    self.root.after(TICK_MS, self.tick)
                except Exception:
                    pass

    def _tick_body(self):
        if not _is_parent_alive(self.parent_pid):
            self._on_close()
            return

        state = _read_json(HUD_STATE_FILE)
        workshop = _read_json(WORKSHOP_FILE)
        # mode precedence: workshop file > CLI default. "off" hides.
        mode = (workshop.get("mode") or self.cli_mode or "on").lower()
        force_visible = bool(workshop.get("force_visible", False))

        if mode == "off" and not force_visible:
            # The skill asked us to hide. Tkinter doesn't have a great
            # 'invisible but still polling' state, so we just shutdown.
            self._on_close()
            return

        jarvis_state = (state.get("state") or "Idle").lower()
        # Guard the float parses so a non-numeric value can't escape this
        # Tkinter callback and prevent the after() reschedule from running.
        try:
            tts_amp  = float(state.get("tts_amplitude", 0.0) or 0.0)
        except (TypeError, ValueError):
            tts_amp = 0.0
        try:
            mic_level = float(state.get("mic_level", 0.0) or 0.0)
        except (TypeError, ValueError):
            mic_level = 0.0

        # Smooth amp/mic so the pulse breathes instead of jittering.
        self.last_amp = 0.55 * self.last_amp + 0.45 * max(0.0, min(1.0, tts_amp))
        self.last_mic = 0.55 * self.last_mic + 0.45 * max(0.0, min(1.0, mic_level))

        # Track when JARVIS was last active so we can keep the reactor on
        # for ~2s after speaking ends instead of hard-cutting.
        if jarvis_state in ("thinking", "speaking", "listening"):
            self._last_active_at = time.time()

        # Per-state animation pace.
        if jarvis_state == "thinking":
            blade_step = 0.10
            base_color = GOLD
            intensity  = 0.9
        elif jarvis_state == "speaking":
            blade_step = 0.06 + 0.10 * self.last_amp
            base_color = _mix(AMBER, AMBER_BRIGHT, self.last_amp)
            intensity  = 0.6 + 0.4 * self.last_amp
        elif jarvis_state == "listening":
            blade_step = 0.045
            base_color = _mix(CYAN, AMBER, 0.25 + 0.4 * self.last_mic)
            intensity  = 0.4 + 0.4 * self.last_mic
        elif jarvis_state == "standby":
            blade_step = 0.012
            base_color = VIOLET
            intensity  = 0.25
        else:  # idle / unknown
            blade_step = 0.022
            base_color = CYAN
            intensity  = 0.35

        self._phase_blades = (self._phase_blades + blade_step) % (2 * math.pi)
        self._phase_pulse  = (self._phase_pulse + 0.14) % (2 * math.pi)

        # ── clear canvas ──
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, self.w, self.h, fill=BG_KEY, outline="",
        )

        # ── outer pulsing ring — TTS amplitude driven ──
        # In "pulse" mode the amplitude swing is exaggerated for a more
        # dramatic effect (Tony asks JARVIS to "show off"); ambient mode
        # keeps it subtler.
        breath = 0.5 * (1 + math.sin(self._phase_pulse))
        if mode == "pulse":
            amp_scale = 0.6 + 1.4 * self.last_amp
        else:
            amp_scale = 0.4 + 0.6 * self.last_amp
        # When idle, the ring still gently breathes so the reactor never
        # looks completely dead.
        if jarvis_state == "speaking":
            radius = self.R_PULSE_BASE + 10 * amp_scale + 4 * breath
            ring_color = _mix(AMBER, AMBER_BRIGHT, self.last_amp)
            ring_w = 3
        elif jarvis_state == "thinking":
            radius = self.R_PULSE_BASE + 4 * breath
            ring_color = GOLD
            ring_w = 2
        elif jarvis_state == "listening":
            radius = self.R_PULSE_BASE + 3 * breath + 5 * self.last_mic
            ring_color = _mix(CYAN, AMBER, 0.4 + 0.3 * self.last_mic)
            ring_w = 2
        else:
            radius = self.R_PULSE_BASE + 2 * breath
            ring_color = CYAN_DIM
            ring_w = 1
        self._circle(radius, ring_color, width=ring_w)
        # An inner dim track behind the pulsing ring so the breathing is
        # visible by reference even at low amplitude.
        self._circle(self.R_PULSE_BASE - 8, CYAN_DIM, width=1)

        # ── rotating blade ring (pseudo-3D) ──
        self._draw_blades(base_color, intensity)

        # ── concentric depth-shaded core ──
        # Three nested rings stepping toward CYAN_BRIGHT so the hub reads
        # like a glowing orb rather than a flat washer.
        # Hub color follows JARVIS state.
        if jarvis_state == "speaking":
            hub_color = _mix(AMBER, AMBER_BRIGHT, self.last_amp)
        elif jarvis_state == "thinking":
            hub_color = GOLD
        elif jarvis_state == "listening":
            hub_color = _mix(CYAN, AMBER, 0.3 + 0.4 * self.last_mic)
        elif jarvis_state == "standby":
            hub_color = VIOLET
        else:
            hub_color = CYAN
        self._circle(self.R_CORE_OUT, hub_color, width=2)
        self._circle(self.R_CORE_MID, _mix(hub_color, CYAN_DEEP, 0.4),
                     width=2)
        # Inner hub: filled, pulsing slightly so the reactor "throbs".
        hub_pulse = 0.5 * (1 + math.sin(self._phase_pulse))
        hub_r = self.R_CORE_HUB + 2 * hub_pulse + 3 * self.last_amp
        x1, y1, x2, y2 = self._bbox(hub_r)
        self.canvas.create_oval(
            x1, y1, x2, y2,
            outline=_mix(hub_color, CYAN_BRIGHT, 0.4),
            fill=_mix(hub_color, "#000000", 0.55),
            width=2,
        )

        # ── status label below the reactor ──
        label = jarvis_state.upper() if jarvis_state else "IDLE"
        label_color = hub_color if jarvis_state != "idle" else DIM_TEXT
        self.canvas.create_text(
            self.cx, self.cy + self.R_PULSE_BASE + 22,
            text=label, fill=label_color, anchor="center",
            font=("Consolas", 11, "bold"),
        )
        # Tiny mode chip in the top-right of the canvas so the user can
        # confirm which mode they're in (ambient vs pulse).
        chip = "PULSE" if mode == "pulse" else "ON"
        self.canvas.create_text(
            self.w - 10, 10, text=chip, fill=DIM_TEXT, anchor="ne",
            font=("Consolas", 8, "bold"),
        )

        # ── advance frame counter ──
        # Rescheduling is handled by the tick() wrapper's finally so it runs
        # even if this body raised partway through the render.
        self.frame += 1

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=2160)
    parser.add_argument("--y", type=int, default=-1100)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--mode", type=str, default="on",
                        choices=["on", "pulse"])
    args = parser.parse_args()

    canvas = WorkshopCanvas(
        args.x, args.y, args.width, args.height,
        args.parent_pid, mode=args.mode,
    )
    canvas.run()


if __name__ == "__main__":
    main()
