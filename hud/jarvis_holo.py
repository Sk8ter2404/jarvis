#!/usr/bin/env python3
"""
JARVIS holographic overlay — fullscreen arc-reactor HUD on the top monitor.

Distinct from the existing corner HUD (hud/jarvis_hud.py): this overlay
fills the entire top monitor with a large cinematic arc reactor — the
"visually present the way MCU JARVIS was" overlay specified in
jarvis_todo.md 2026-05-27 09:18 (holographic_overlay).

Rendered elements:
  • Large center arc reactor (concentric rings, pulsing hub).
  • Outer rotating ring with the current active action name flowing as text.
  • Four status arcs around the perimeter:
      N (top)    — CPU %
      E (right)  — RAM %
      S (bottom) — Network (up + down KB/s)
      W (left)   — Anthropic credits balance
  • Center text panel below the reactor showing what JARVIS just HEARD
    (▸ user transcript) and what JARVIS just SAID (◂ reply).
  • Soft amber glow that pulses around the reactor:
      Listening → gentle amber breathing
      Speaking  → brighter amber, modulated by tts_amplitude

State sources:
  • hud_state.json (sibling of bobert_companion.py) — written by the main
    process. Read at the tick rate.
  • credits_state.json — written by skills/credits_monitor.py.
  • psutil — CPU / RAM / net io counters polled locally so we don't push
    high-frequency sensor data through the JSON state file.

Click-through:
  • Win32: -transparentcolor on the keyed background pixels — drawn pixels
    block clicks but most of the canvas is keyed/transparent.
  • Other platforms: window-level alpha fallback (not strictly click-
    through but still translucent).

Closes cleanly when its parent (the launcher) exits.

CLI:
  python hud/jarvis_holo.py --x 0 --y -1440 --width 2560 --height 1440
                             --parent-pid 12345
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

# ──────────────────────────────────────────────────────────────────────────
#  Layout — sized off the monitor passed via CLI args.
# ──────────────────────────────────────────────────────────────────────────
TICK_MS = 50            # ~20 fps

# Reactor sizing fractions (of min(width, height))
R_OUTER_FRAC  = 0.30
R_RING_CPU    = 0.27
R_RING_RAM    = 0.25
R_RING_NET    = 0.23
R_RING_CREDIT = 0.21
# now_doing ring — sits between the credits gauge (0.21) and the core (0.10)
# so it doesn't fight the four cardinal status gauges for screen real estate.
# jarvis_todo.md 2026-05-29 18:05 (now_doing realtime status ring).
R_NOW_DOING   = 0.185
R_CORE_FRAC   = 0.10
R_HUB_FRAC    = 0.045
R_GLOW_FRAC   = 0.34

# ──────────────────────────────────────────────────────────────────────────
#  Palette — cyan for the rings, amber for the listen/speak glow.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY       = "#010101"  # transparentcolor target on Win32
PANEL_COLOR  = "#04080d"
CYAN         = "#4cc9ff"
CYAN_DIM     = "#1b4a66"
CYAN_BRIGHT  = "#9ee7ff"
TEXT_COLOR   = "#cfeefb"
DIM_TEXT     = "#5d8aa3"
AMBER        = "#ffb347"  # spec: "soft amber glow that pulses"
AMBER_DIM    = "#7a5520"
AMBER_BRIGHT = "#ffe0a0"
GOLD         = "#ffd166"
ALERT        = "#ff5b5b"
ALERT_DIM    = "#5a1414"

STATE_FILE_NAME = "hud_state.json"
CREDITS_FILE    = "credits_state.json"
PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE      = os.path.join(PROJECT_DIR, STATE_FILE_NAME)
CREDITS_PATH    = os.path.join(PROJECT_DIR, CREDITS_FILE)


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if _HAS_PSUTIL:
        # On Windows pid_exists can raise on a transient handle/permission
        # error. Treat an unknowable parent as alive so a momentary hiccup
        # never tears down the overlay (matches the PyQt HUDs' guard).
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
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, c)) for c in rgb))


def _mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    ))


def _fmt_rate(bytes_per_sec: float) -> str:
    """Render a byte/sec rate in a compact form (KB/s, MB/s)."""
    if bytes_per_sec < 1024:
        return f"{int(bytes_per_sec)} B/s"
    if bytes_per_sec < 1024 * 1024:
        return f"{bytes_per_sec / 1024:.0f} KB/s"
    return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"


class HoloHUD:
    def __init__(self, x: int, y: int, width: int, height: int, parent_pid: int):
        self.parent_pid = parent_pid
        self._closing   = False
        self.frame      = 0
        self.x, self.y = x, y
        self.w, self.h = width, height

        # Reactor anchor — center of the top monitor.
        self.cx = self.w / 2
        self.cy = self.h / 2
        ref = min(self.w, self.h)
        self.R_OUTER  = ref * R_OUTER_FRAC
        self.R_CPU    = ref * R_RING_CPU
        self.R_RAM    = ref * R_RING_RAM
        self.R_NET    = ref * R_RING_NET
        self.R_CREDIT = ref * R_RING_CREDIT
        self.R_NOW_DOING = ref * R_NOW_DOING
        self.R_CORE   = ref * R_CORE_FRAC
        self.R_HUB    = ref * R_HUB_FRAC
        self.R_GLOW   = ref * R_GLOW_FRAC

        # Live psutil samples — polled in tick().
        self.last_cpu     = 0.0
        self.last_ram    = 0.0
        self.last_net_up = 0.0  # bytes/sec
        self.last_net_dn = 0.0  # bytes/sec
        self.last_amp    = 0.0  # smoothed TTS amplitude
        self.last_mic    = 0.0  # smoothed mic level
        self._prev_net   = None  # (sent, recv, ts) for delta calc

        # Per-state phase accumulators so spin doesn't snap on transitions.
        self._phase      = 0.0
        self._halo_phase = 0.0

        # Most-recently applied window alpha — only re-poke -alpha when the
        # requested value actually changes, so tkinter isn't asked to
        # re-render every frame.
        self._last_alpha_applied = None

        self.root = tk.Tk()
        self.root.title("JARVIS Holographic")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Win32 keyed transparency — pixels drawn in BG_KEY become click-
        # through. Falls back to global alpha on other platforms.
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.78)
            except Exception:
                pass

        # Cover the entire passed-in monitor rect.
        self.root.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=self.w, height=self.h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Double-right-click anywhere to dismiss — the overlay is fullscreen
        # and tkinter can't easily make it 100% click-through, so this
        # provides an escape hatch if the user gets stuck.
        self.canvas.bind("<Double-Button-3>", lambda _e: self._on_close())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if _HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        self.tick()

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except Exception:
            pass

    # ─── drawing helpers ────────────────────────────────────────────────
    def _bbox(self, r):
        return (self.cx - r, self.cy - r, self.cx + r, self.cy + r)

    def _ring_track(self, radius, color=CYAN_DIM, width=1):
        x1, y1, x2, y2 = self._bbox(radius)
        self.canvas.create_oval(x1, y1, x2, y2, outline=color, width=width)

    def _gauge_arc(self, radius, pct, color, width=3, start_deg=90):
        pct = max(0.0, min(100.0, pct))
        if pct <= 0.01:
            return
        x1, y1, x2, y2 = self._bbox(radius)
        extent = -3.6 * pct  # clockwise sweep
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=start_deg, extent=extent,
            style="arc", outline=color, width=width,
        )

    def _arc_segment(self, radius, start_deg, extent_deg, color, width=2):
        x1, y1, x2, y2 = self._bbox(radius)
        self.canvas.create_arc(
            x1, y1, x2, y2,
            start=start_deg, extent=extent_deg,
            style="arc", outline=color, width=width,
        )

    def _rotating_ticks(self, radius, count, color, phase,
                        tick_len=6.0, width=1):
        for i in range(count):
            theta = (i / count) * 2 * math.pi + phase
            x1 = self.cx + radius * math.cos(theta)
            y1 = self.cy + radius * math.sin(theta)
            x2 = self.cx + (radius + tick_len) * math.cos(theta)
            y2 = self.cy + (radius + tick_len) * math.sin(theta)
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _text_at(self, x, y, msg, color=TEXT_COLOR, size=11,
                 weight="normal", anchor="center"):
        self.canvas.create_text(
            x, y, text=msg, fill=color, anchor=anchor,
            font=("Consolas", size, weight),
        )

    def _curved_text(self, radius, start_deg, text, color, size=11,
                     weight="bold", char_arc_deg=2.6):
        """Render `text` as individual characters laid out along an arc.
        start_deg is measured the same way tkinter measures angles (CCW
        from 3 o'clock). Used for the outer rotating action ring."""
        if not text:
            return
        # Lay out each character at its own angle. char_arc_deg is the
        # angular spacing between glyph centers.
        for i, ch in enumerate(text):
            angle_deg = start_deg - i * char_arc_deg
            theta = math.radians(angle_deg)
            x = self.cx + radius * math.cos(theta)
            # Tkinter's y is inverted relative to math convention.
            y = self.cy - radius * math.sin(theta)
            # Rotate each character so it tangentially follows the curve.
            text_angle = (angle_deg - 90) % 360
            try:
                self.canvas.create_text(
                    x, y, text=ch, fill=color, anchor="center",
                    font=("Consolas", size, weight),
                    angle=text_angle,
                )
            except tk.TclError:
                # Some older Tk builds don't support angle= — fall back to
                # an un-rotated character.
                self.canvas.create_text(
                    x, y, text=ch, fill=color, anchor="center",
                    font=("Consolas", size, weight),
                )

    # ─── network rate ───────────────────────────────────────────────────
    def _refresh_network(self):
        if not _HAS_PSUTIL:
            return
        try:
            counters = psutil.net_io_counters()
            now = time.time()
        except Exception:
            return
        sent, recv = counters.bytes_sent, counters.bytes_recv
        if self._prev_net is None:
            self._prev_net = (sent, recv, now)
            return
        prev_sent, prev_recv, prev_ts = self._prev_net
        dt = max(0.001, now - prev_ts)
        # Smooth a bit so the gauge isn't flickery.
        new_up = (sent - prev_sent) / dt
        new_dn = (recv - prev_recv) / dt
        self.last_net_up = 0.4 * self.last_net_up + 0.6 * max(0.0, new_up)
        self.last_net_dn = 0.4 * self.last_net_dn + 0.6 * max(0.0, new_dn)
        self._prev_net = (sent, recv, now)

    # ─── main paint ─────────────────────────────────────────────────────
    def tick(self):
        # Wrapper: render in a guarded body and ALWAYS reschedule in finally
        # (unless we're closing) so a single bad value or transient error can
        # never strand this full-screen overlay on a frozen frame.
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

        state = _read_json(STATE_FILE)
        credits = _read_json(CREDITS_PATH)

        jarvis_state    = (state.get("state") or "Idle").lower()
        active_action   = state.get("active_action", "") or ""
        now_doing       = state.get("now_doing", "") or ""
        last_transcript = state.get("last_transcript", "") or ""
        last_spoken     = state.get("last_spoken", "") or ""
        # Guard the float parses: a non-numeric value in the shared state file
        # must not raise out of this Tkinter callback. If it did, the
        # `after()` reschedule at the end of tick() would never run and the
        # full-screen overlay would freeze permanently. Bad value -> 0.0.
        try:
            mic_level   = float(state.get("mic_level", 0.0) or 0.0)
        except (TypeError, ValueError):
            mic_level = 0.0
        try:
            tts_amp     = float(state.get("tts_amplitude", 0.0) or 0.0)
        except (TypeError, ValueError):
            tts_amp = 0.0

        # Night-owl mode dims the overlay. 0.0 / missing == normal opacity;
        # any positive value < 1 dims the window. Applied to the root via
        # -alpha so the whole reactor (including the click-through keyed
        # pixels) softens past midnight. Only restated when it changes.
        try:
            night_owl_dim = float(state.get("night_owl_dim", 0.0) or 0.0)
        except (TypeError, ValueError):
            night_owl_dim = 0.0
        if 0.0 < night_owl_dim < 1.0:
            target_alpha = night_owl_dim
        else:
            target_alpha = 1.0
        if self._last_alpha_applied != target_alpha:
            try:
                self.root.attributes("-alpha", target_alpha)
                self._last_alpha_applied = target_alpha
            except Exception:
                pass

        # ── live psutil sensors (local poll, not state file) ──
        if _HAS_PSUTIL:
            try:
                self.last_cpu = psutil.cpu_percent(interval=None)
                self.last_ram = psutil.virtual_memory().percent
            except Exception:
                pass
        self._refresh_network()

        # Smooth mic + amp for non-jittery glow.
        self.last_amp = 0.55 * self.last_amp + 0.45 * max(0.0, min(1.0, tts_amp))
        self.last_mic = 0.55 * self.last_mic + 0.45 * max(0.0, min(1.0, mic_level))

        # ── advance per-state animation phases ──
        is_alert = self.last_cpu >= 90.0 or self.last_ram >= 90.0
        if jarvis_state == "thinking":
            spin_step = 1.0 / 11.0
            halo_step = 11
        elif jarvis_state == "listening":
            spin_step = 1.0 / 26.0
            halo_step = 6
        elif jarvis_state == "speaking":
            spin_step = 1.0 / 22.0
            halo_step = 5
        elif jarvis_state == "standby":
            spin_step = 1.0 / 90.0
            halo_step = 2
        else:  # idle / unknown
            spin_step = 1.0 / 60.0
            halo_step = 3
        if is_alert:
            spin_step *= 1.6

        self._phase      = (self._phase + spin_step) % (2 * math.pi)
        self._halo_phase = (self._halo_phase + halo_step) % 360

        # ── clear canvas ──
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, self.w, self.h, fill=BG_KEY, outline="",
        )

        # ── soft amber glow halo (spec contract) ──
        # The glow lives OUTSIDE the data rings so it doesn't fight the
        # gauges for attention. Amplitude on speaking, gentle breathing on
        # listening, dim ember on every other state so the reactor never
        # looks fully cold.
        if jarvis_state == "listening":
            # Gentle breathing in amber — pulse_freq slow, large amplitude.
            breath = 0.5 * (1 + math.sin(self.frame * 0.10))
            glow_alpha = 0.4 + 0.6 * breath + 0.3 * self.last_mic
            glow_alpha = max(0.0, min(1.0, glow_alpha))
            glow_color = _mix(AMBER_DIM, AMBER, glow_alpha)
            # Outer concentric glow rings — drawing 3 ovals so the glow has
            # some thickness even though tkinter has no real blur.
            for i, dr in enumerate((0, 6, 12)):
                self._arc_segment(
                    self.R_GLOW + dr, 0, 360,
                    color=_mix(AMBER_DIM, glow_color, 1.0 - i * 0.25),
                    width=2,
                )
        elif jarvis_state == "speaking":
            # Brighter amber pulse modulated by TTS amplitude — radius and
            # color brightness both ride the audio so it visibly throbs.
            breath = 0.5 * (1 + math.sin(self.frame * 0.22))
            glow_color = _mix(AMBER, AMBER_BRIGHT, self.last_amp)
            radius_lift = 4 + 10 * self.last_amp + 4 * breath
            for i, dr in enumerate((0, 5, 10)):
                self._arc_segment(
                    self.R_GLOW + radius_lift + dr, 0, 360,
                    color=_mix(AMBER_DIM, glow_color, 1.0 - i * 0.25),
                    width=2,
                )
        elif jarvis_state == "thinking":
            # Subtle cooler shimmer — keeps the reactor feeling "warm" but
            # not amber-glowing.
            ember = 0.5 * (1 + math.sin(self.frame * 0.16))
            self._arc_segment(
                self.R_GLOW, 0, 360,
                color=_mix(AMBER_DIM, AMBER, 0.25 * ember),
                width=1,
            )
        # idle / standby: no halo (lets the reactor sit at rest).

        # ── status rings (CPU / RAM / Net / Credits) ──
        # Color flips ALERT on high CPU/RAM individually so the offending
        # dimension is identifiable.
        cpu_color = ALERT if self.last_cpu >= 90 else CYAN
        ram_color = ALERT if self.last_ram >= 90 else CYAN

        self._ring_track(self.R_CPU, CYAN_DIM, 1)
        self._gauge_arc(self.R_CPU, self.last_cpu, cpu_color, width=3)
        self._rotating_ticks(self.R_CPU, count=24, color=CYAN_DIM,
                             phase=self._phase)

        self._ring_track(self.R_RAM, CYAN_DIM, 1)
        self._gauge_arc(self.R_RAM, self.last_ram, ram_color, width=3)
        self._rotating_ticks(self.R_RAM, count=18, color=CYAN_DIM,
                             phase=-self._phase * 0.7)

        # Network: scale the gauge against a soft cap of 5 MB/s so a normal
        # browsing session paints a visible slice without 100Mbit/s
        # downloads pegging it.
        NET_CAP_BPS = 5 * 1024 * 1024
        net_total = self.last_net_up + self.last_net_dn
        net_pct = min(100.0, 100.0 * net_total / NET_CAP_BPS)
        self._ring_track(self.R_NET, CYAN_DIM, 1)
        self._gauge_arc(self.R_NET, net_pct, CYAN, width=3)
        self._rotating_ticks(self.R_NET, count=14, color=CYAN_DIM,
                             phase=self._phase * 1.3)

        # Credits: scale against a $20 baseline (the per-month topup amount
        # the user typically sees). Red when balance < $5.
        balance = credits.get("balance")
        if isinstance(balance, (int, float)) and balance >= 0:
            credit_pct = min(100.0, balance / 20.0 * 100.0)
            credit_color = ALERT if balance < 5 else GOLD
        else:
            credit_pct = 0
            credit_color = CYAN_DIM
        self._ring_track(self.R_CREDIT, CYAN_DIM, 1)
        self._gauge_arc(self.R_CREDIT, credit_pct, credit_color, width=3)
        self._rotating_ticks(self.R_CREDIT, count=10, color=CYAN_DIM,
                             phase=-self._phase * 1.5)

        # ── now_doing ring — realtime status surface ──
        # Curves the now_doing string (LISTENING / THINKING (model) /
        # EXECUTING: see_screen / SPEAKING / IDLE) around its own dedicated
        # ring between the credits gauge and the core. Color keys to state:
        # amber when JARVIS is actively engaged (listening/thinking/speaking
        # /executing) so the user can answer "are you working?" at a glance
        # (jarvis_todo.md 2026-05-29 18:05).
        if now_doing.startswith("EXECUTING"):
            nd_color = AMBER_BRIGHT
        elif jarvis_state in ("listening", "thinking", "speaking"):
            nd_color = AMBER
        elif jarvis_state == "standby":
            nd_color = "#9b8cff"  # violet — matches hub
        else:
            nd_color = CYAN
        self._ring_track(self.R_NOW_DOING, _mix(nd_color, CYAN_DIM, 0.55), 1)
        nd_text = (now_doing or jarvis_state).upper()
        # Counter-spin a touch so the now_doing label drifts opposite to the
        # outer action ring — makes the two layers feel distinct without
        # adding another moving part.
        nd_spin_deg = math.degrees(-self._phase * 0.6) % 360
        nd_start_deg = 90 + nd_spin_deg
        # Repeat twice so even short labels ("IDLE") visibly rotate.
        nd_repeated = f"  {nd_text}  ·  {nd_text}  "
        self._curved_text(
            self.R_NOW_DOING + 10, nd_start_deg, nd_repeated[:56],
            color=nd_color, size=10, weight="bold", char_arc_deg=3.6,
        )

        # ── outer rotating ring — current action name flowing as text ──
        # The text follows the ring's arc, rotating at the same rate as the
        # reactor spin so it feels mechanically coupled.
        outer_color = AMBER if (active_action and jarvis_state != "idle") else CYAN_DIM
        self._ring_track(self.R_OUTER, outer_color, 1)
        # Action text or state label rotates around the ring.
        if active_action:
            ring_text = active_action.upper()
            ring_color = AMBER
        else:
            ring_text = (jarvis_state or "idle").upper()
            ring_color = CYAN
        # Repeat the text 3× around the ring with separator dots so the
        # rotation is visible even on short strings.
        repeated = f"  {ring_text}  •  {ring_text}  •  {ring_text}  "
        spin_deg = math.degrees(self._phase * 1.0) % 360
        # Start angle measured tkinter-style (CCW from 3 o'clock).
        start_deg = 90 + spin_deg
        # Limit length so we don't run off the visible arc.
        self._curved_text(
            self.R_OUTER + 14, start_deg, repeated[:64],
            color=ring_color, size=11, weight="bold",
        )

        # Outer ring tick marks underneath
        self._rotating_ticks(
            self.R_OUTER, count=36,
            color=AMBER_DIM if active_action else CYAN_DIM,
            phase=self._phase, tick_len=10.0, width=1,
        )

        # ── core: arc reactor hub ──
        # Color of the hub follows the JARVIS state — cyan idle, amber when
        # listening/speaking, gold thinking — so a glance tells you where
        # JARVIS is in its cycle even without reading the rings.
        if jarvis_state == "listening":
            hub_color = _mix(AMBER_DIM, AMBER, 0.4 + 0.6 * self.last_mic)
        elif jarvis_state == "speaking":
            hub_color = _mix(AMBER, AMBER_BRIGHT, self.last_amp)
        elif jarvis_state == "thinking":
            hub_color = GOLD
        elif jarvis_state == "standby":
            hub_color = "#9b8cff"  # violet
        else:
            hub_color = CYAN

        # Core ring
        self._ring_track(self.R_CORE, hub_color, 2)
        # Rotating halo arcs around the core (4 segments)
        for i in range(4):
            seg_start = (i * 90 + self._halo_phase) % 360
            self._arc_segment(self.R_CORE - 6, seg_start, 60,
                              color=hub_color, width=2)
        # Inner pulsing hub
        pulse = 0.5 * (1 + math.sin(self.frame * 0.18))
        hub_radius = self.R_HUB + pulse * 4
        x1, y1, x2, y2 = self._bbox(hub_radius)
        self.canvas.create_oval(
            x1, y1, x2, y2, outline=hub_color, width=3, fill=PANEL_COLOR,
        )

        # ── center label inside hub ──
        center_label = (jarvis_state or "Idle").upper()
        self._text_at(self.cx, self.cy - 6, center_label,
                      color=hub_color, size=14, weight="bold")
        self._text_at(self.cx, self.cy + 12, time.strftime("%H:%M:%S"),
                      color=DIM_TEXT, size=10)

        # ── status numbers near each cardinal ring ──
        # Labels float just inside their respective rings so the data is
        # readable without crowding the rotating ticks.
        self._text_at(
            self.cx, self.cy - self.R_CPU - 18,
            f"CPU  {self.last_cpu:>4.0f}%",
            color=cpu_color, size=12, weight="bold",
        )
        self._text_at(
            self.cx + self.R_RAM + 18, self.cy,
            f"RAM\n{self.last_ram:>3.0f}%",
            color=ram_color, size=12, weight="bold", anchor="w",
        )
        net_str = f"NET\n↑ {_fmt_rate(self.last_net_up)}\n↓ {_fmt_rate(self.last_net_dn)}"
        self._text_at(
            self.cx, self.cy + self.R_NET + 28, net_str,
            color=CYAN, size=11, weight="bold", anchor="n",
        )
        if isinstance(balance, (int, float)) and balance >= 0:
            credits_str = f"CREDITS\n${balance:0.2f}"
            credits_text_color = ALERT if balance < 5 else GOLD
        else:
            credits_str = "CREDITS\n—"
            credits_text_color = DIM_TEXT
        self._text_at(
            self.cx - self.R_CREDIT - 18, self.cy, credits_str,
            color=credits_text_color, size=12, weight="bold", anchor="e",
        )

        # ── center text panel below the reactor ──
        # Two lines: what JARVIS just HEARD, what JARVIS just SAID. These
        # populate as the conversation flows so the overlay reads back the
        # most recent exchange even when the user isn't looking at the
        # console.
        panel_y = self.cy + self.R_OUTER + 60
        # User transcript
        if last_transcript:
            disp = last_transcript[:90]
            if len(last_transcript) > 90:
                disp = disp + "…"
            self._text_at(
                self.cx, panel_y,
                f'▸  {disp}',
                color=TEXT_COLOR, size=14, anchor="n",
            )
        else:
            self._text_at(
                self.cx, panel_y, "▸  awaiting input",
                color=DIM_TEXT, size=14, anchor="n",
            )
        # JARVIS reply
        if last_spoken:
            disp = last_spoken[:110]
            if len(last_spoken) > 110:
                disp = disp + "…"
            self._text_at(
                self.cx, panel_y + 32,
                f'◂  {disp}',
                color=AMBER, size=14, anchor="n",
            )

        # ── top-of-screen banner: J.A.R.V.I.S. — Holographic Interface ──
        self._text_at(
            self.cx, 32,
            "J . A . R . V . I . S .   —   HOLOGRAPHIC INTERFACE",
            color=DIM_TEXT, size=13, weight="bold",
        )
        # Bottom-of-screen banner: dismiss hint.
        self._text_at(
            self.cx, self.h - 22,
            "double-right-click anywhere to dismiss",
            color=DIM_TEXT, size=9,
        )

        # Advance the frame counter. Rescheduling is handled by the tick()
        # wrapper's finally so it runs even if this body raised.
        self.frame += 1

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

    hud = HoloHUD(args.x, args.y, args.width, args.height, args.parent_pid)
    hud.run()


if __name__ == "__main__":
    main()
