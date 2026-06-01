#!/usr/bin/env python3
"""
JARVIS Stark-style status ring (holographic_overlay/hud_v2.py) — a
transparent, always-on-top PyQt6 reactor disc pinned to the top monitor.

Spec: jarvis_todo.md 2026-05-30 05:23 (overnight). Replaces the bare debug
HUD with a proper Stark-style status ring. Draws every 500 ms:

  • Outer arc-reactor ring split into three quadrant arcs
      ▸ top-left   (9 o'clock → 12 o'clock)  CPU %     cyan → amber → red
      ▸ top-right  (12 o'clock → 3 o'clock)  RAM %     cyan → amber → red
      ▸ bottom     (3 o'clock → 9 o'clock)   GPU °C    cyan → amber → red
    so the four-quadrant layout of arc_reactor_status_hud.py is preserved
    but the bottom semicircle is reserved for the calendar/track text
    rows that sit underneath the disc.
  • Inner Bambu progress ring — only drawn when a print is RUNNING /
    PAUSE / PREPARE (read from bambu_overlay_state.json). Otherwise the
    slot is invisible and the reactor renders one ring lighter.
  • Central core hub — pulses with the current speech state:
      ▸ idle      → dim cyan, slow heartbeat
      ▸ listening → amber, modulated by mic_level
      ▸ thinking  → bright cyan, fast heartbeat
      ▸ speaking  → amber, modulated by tts_amplitude
      ▸ standby/sleep → very dim cyan
  • Top-center text row — current track ("Artist — Title") from
    apple_music_intel (iTunes COM / window title fallback). Empty slot
    when nothing is playing — renderer still draws cleanly.
  • Bottom-center text row — next calendar event from ms_graph
    (subject + relative time, e.g. "Standup in 14 min"). When no token
    is configured or Graph is unreachable, the slot stays blank — the
    spec calls for a silent degrade rather than an error banner.
  • Stat chips at the cardinals — `CPU 47%`, `RAM 62%`, `GPU 71°` —
    same number-readable affordance as arc_reactor_status_hud.

Data sources:
  • hud_state.json (project root) — state, tts_amplitude, mic_level,
    pulse_strip fallback for GPU temp when nvidia-smi is unavailable.
  • psutil — CPU % / RAM % each tick.
  • nvidia-smi — GPU temperature, cached 4 s.
  • skill_apple_music_intel._sample_now_playing — current track via the
    iTunes COM bridge with window-title fallback.
  • skill_ms_graph.get_first_meeting — next calendar event. Silent
    degrade when no token is configured.
  • bambu_overlay_state.json — printer state + mc_percent.

Subprocess lifecycle: auto-exits when its parent (the launcher) dies
OR when stark_status_ring_state.json flips to mode=off. The launcher
writes that control file to ask the widget to retire cleanly without
racing terminate().

Click-through: WA_TransparentForMouseEvents — this is an information
layer, never an interactive surface.

CLI:
  python skills/holographic_overlay/hud_v2.py --x 1000 --y -1400 \
      --width 460 --height 340 --parent-pid 12345
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QRadialGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QGraphicsView, QGraphicsScene,
        QGraphicsDropShadowEffect,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False


TICK_MS = 500  # spec — 500 ms refresh cadence

# Slow-data refresh cadences (in ticks). The renderer ticks every 500 ms;
# calendar lookups are an HTTP roundtrip and track sampling pokes the
# iTunes COM bridge, so we don't want them on every frame.
TRACK_REFRESH_TICKS    = 12     # ~6 s
CALENDAR_REFRESH_TICKS = 240    # ~2 min
GPU_CACHE_SECONDS      = 4.0

# Palette — kept in lockstep with arc_reactor_status_hud so the two
# surfaces read as one system.
CYAN         = QColor(76, 201, 255)      if _HAS_PYQT6 else None  # #4cc9ff
CYAN_DIM     = QColor(27, 74, 102)       if _HAS_PYQT6 else None  # #1b4a66
CYAN_BRIGHT  = QColor(158, 231, 255)     if _HAS_PYQT6 else None  # #9ee7ff
TEXT_COLOR   = QColor(207, 238, 251)     if _HAS_PYQT6 else None  # #cfeefb
DIM_TEXT     = QColor(93, 138, 163)      if _HAS_PYQT6 else None  # #5d8aa3
AMBER        = QColor(255, 179, 71)      if _HAS_PYQT6 else None  # #ffb347
AMBER_BRIGHT = QColor(255, 224, 160)     if _HAS_PYQT6 else None  # #ffe0a0
RED          = QColor(255, 91, 91)       if _HAS_PYQT6 else None  # #ff5b5b
GREEN_SOFT   = QColor(120, 235, 168)     if _HAS_PYQT6 else None  # #78eba8
PANEL_DARK   = QColor(4, 8, 13, 215)     if _HAS_PYQT6 else None  # translucent
PANEL_RIM    = QColor(10, 24, 32, 230)   if _HAS_PYQT6 else None

# Stark-style state-coloured ring is the spec's centrepiece; the chosen
# accents are the same ones the rest of the JARVIS HUD family already
# uses for state transitions.

# Walking up: hud_v2.py → holographic_overlay/ → skills/ → <project root>.
PROJECT_DIR        = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
HUD_STATE_FILE     = os.path.join(PROJECT_DIR, "hud_state.json")
BAMBU_STATE_FILE   = os.path.join(PROJECT_DIR, "bambu_overlay_state.json")
CONTROL_FILE       = os.path.join(PROJECT_DIR, "stark_status_ring_state.json")

# Thresholds (same as arc_reactor_status_hud — keep them aligned so the
# colour transitions happen together).
CPU_WARN_PCT      = 75.0
CPU_CRIT_PCT      = 90.0
RAM_WARN_PCT      = 75.0
RAM_CRIT_PCT      = 90.0
GPU_WARN_C        = 70.0
GPU_CRIT_C        = 82.0


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
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


def _sample_now_playing_safe() -> dict | None:
    """Best-effort `_sample_now_playing` call — silent on every failure
    (missing skill module, iTunes COM hiccup, etc.) so a renderer crash
    is impossible."""
    try:
        mod = sys.modules.get("skill_apple_music_intel")
        if mod is None:
            try:
                mod = importlib.import_module("skill_apple_music_intel")
            except Exception:
                mod = None
        if mod is None or not hasattr(mod, "_sample_now_playing"):
            return None
        return mod._sample_now_playing()
    except Exception:
        return None


def _get_first_meeting_safe() -> dict | None:
    """Best-effort `ms_graph.get_first_meeting` — silent on every
    failure (no token, urllib timeout, schema drift). Returns None when
    Graph isn't configured so the renderer simply leaves the row blank."""
    try:
        mod = sys.modules.get("skill_ms_graph")
        if mod is None:
            try:
                mod = importlib.import_module("skill_ms_graph")
            except Exception:
                mod = None
        if mod is None:
            return None
        if hasattr(mod, "is_configured"):
            try:
                if not mod.is_configured():
                    return None
            except Exception:
                return None
        if not hasattr(mod, "get_first_meeting"):
            return None
        return mod.get_first_meeting(when="today") or \
               mod.get_first_meeting(when="next_14_days")
    except Exception:
        return None


def _format_meeting(evt: dict | None) -> tuple[str, str]:
    """Turn an event dict into (subject, when-string). when-string is
    'in 12 min', 'now', or 'in 3 h' — depending on lead time. Both
    strings can be empty when nothing's coming up."""
    if not evt:
        return "", ""
    subject = (evt.get("subject") or "").strip()
    start = evt.get("start")
    if not isinstance(start, datetime.datetime):
        return subject, ""
    now = datetime.datetime.now()
    if start.tzinfo is not None:
        # Defensive — ms_graph returns naive local but if a caller swaps
        # it for an aware dt we still produce a sensible delta.
        start = start.astimezone().replace(tzinfo=None)
    delta = start - now
    secs = int(delta.total_seconds())
    if secs <= -60:
        return subject, "now"
    if secs < 60:
        return subject, "now"
    if secs < 3600:
        return subject, f"in {secs // 60} min"
    if secs < 86400:
        hours = secs / 3600.0
        return subject, f"in {hours:.1f} h"
    days = secs // 86400
    return subject, f"in {days} d"


class StarkStatusRingScene(QGraphicsScene):
    """Stark-style reactor scene — renders the ring, the speech-state
    core, and the two text rows in a single drawBackground pass."""

    def __init__(self, width: int, height: int, parent_pid: int):
        super().__init__(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self.parent_pid = parent_pid
        self.frame = 0

        self._recompute_layout()

        # Live sample buffers.
        self.cpu_pct        = 0.0
        self.ram_pct        = 0.0
        self.gpu_temp_c: float | None = None
        self.state          = "idle"
        self.tts_amp        = 0.0
        self.mic_level      = 0.0
        # Bambu state — only drawn when active.
        self.bambu_active   = False
        self.bambu_percent  = 0
        self.bambu_gcode    = ""
        # Now-playing — track + artist; both blank when nothing's playing.
        self.track_title    = ""
        self.track_artist   = ""
        # Next calendar event — subject + relative-time string.
        self.cal_subject    = ""
        self.cal_when       = ""
        # GPU caching — nvidia-smi is slow.
        self._gpu_cached_at = 0.0

        if _HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)  # prime
            except Exception:
                pass

    def _recompute_layout(self) -> None:
        # Reactor disc sized off the shorter axis so the ring stays
        # circular on widescreen and tall panels alike. The text rows
        # eat top/bottom margins — disc shrinks to fit.
        ref = min(self.w, self.h)
        self.cx = self.w / 2.0
        # Push the disc slightly above centre so the bottom calendar row
        # has breathing room.
        self.cy = self.h * 0.5
        self.R_OUTER = ref * 0.36
        self.R_INNER_BAMBU = ref * 0.28
        self.R_CORE = ref * 0.18
        self.R_HUB  = ref * 0.10
        self.R_GLOW = ref * 0.48

    def resize_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self._recompute_layout()

    # ─── sensor reads ──────────────────────────────────────────────────
    def _read_gpu_temp(self) -> float | None:
        """Best-effort GPU temperature in Celsius — cached 4 s so the
        nvidia-smi subprocess doesn't dominate the renderer."""
        now = time.time()
        if (now - self._gpu_cached_at) < GPU_CACHE_SECONDS:
            return self.gpu_temp_c
        self._gpu_cached_at = now
        try:
            exe = shutil.which("nvidia-smi")
            if exe:
                out = subprocess.run(
                    [exe, "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0,
                    creationflags=(subprocess.CREATE_NO_WINDOW
                                   if sys.platform == "win32" else 0),
                )
                temps = []
                for v in (out.stdout or "").strip().splitlines():
                    v = v.strip()
                    if v.isdigit():
                        temps.append(int(v))
                if temps:
                    return float(max(temps))
        except Exception:
            pass
        # Fallback — parse the pulse_strip "GPU 33C" hint if present.
        hud = _read_json(HUD_STATE_FILE)
        strip = (hud.get("pulse_strip") or "")
        if "GPU " in strip:
            try:
                seg = strip.split("GPU ", 1)[1]
                num = ""
                for ch in seg:
                    if ch.isdigit() or ch == ".":
                        num += ch
                    else:
                        break
                if num:
                    return float(num)
            except Exception:
                pass
        return None

    # ─── data refresh (called by the QTimer) ───────────────────────────
    def refresh_data(self) -> bool:
        """Pull the latest state + sensor readings. Returns False when
        the parent died or the control file flipped to off so the
        owning window can close itself."""
        if not _is_parent_alive(self.parent_pid):
            return False
        if _control_says_off():
            return False

        hud = _read_json(HUD_STATE_FILE)
        self.state = (hud.get("state") or "Idle").lower()
        try:
            self.tts_amp = float(hud.get("tts_amplitude") or 0.0)
        except (TypeError, ValueError):
            self.tts_amp = 0.0
        try:
            self.mic_level = float(hud.get("mic_level") or 0.0)
        except (TypeError, ValueError):
            self.mic_level = 0.0

        if _HAS_PSUTIL:
            try:
                self.cpu_pct = float(psutil.cpu_percent(interval=None))
                self.ram_pct = float(psutil.virtual_memory().percent)
            except Exception:
                pass
        self.gpu_temp_c = self._read_gpu_temp()

        bambu = _read_json(BAMBU_STATE_FILE)
        gs = (bambu.get("gcode_state") or "").upper()
        self.bambu_gcode = gs
        self.bambu_active = gs in ("RUNNING", "PAUSE", "PREPARE")
        try:
            self.bambu_percent = int(bambu.get("mc_percent") or 0)
        except (TypeError, ValueError):
            self.bambu_percent = 0

        # Slow refreshes (track / calendar) — staggered so a single tick
        # never blocks on two external calls in a row.
        if (self.frame % TRACK_REFRESH_TICKS) == 0:
            sample = _sample_now_playing_safe()
            if sample:
                self.track_title = (sample.get("title") or "").strip()
                self.track_artist = (sample.get("artist") or "").strip()
            else:
                self.track_title = ""
                self.track_artist = ""
        if ((self.frame + TRACK_REFRESH_TICKS // 2)
                % CALENDAR_REFRESH_TICKS) == 0:
            evt = _get_first_meeting_safe()
            self.cal_subject, self.cal_when = _format_meeting(evt)

        self.frame += 1
        self.update()
        return True

    # ─── helpers ───────────────────────────────────────────────────────
    def _accent_for_state(self) -> QColor:
        s = self.state
        if s == "listening":
            return AMBER
        if s == "speaking":
            return AMBER_BRIGHT
        if s == "thinking":
            return CYAN_BRIGHT
        if s in ("standby", "sleep"):
            return CYAN_DIM
        return CYAN

    @staticmethod
    def _color_for_metric(value: float, warn: float, crit: float) -> QColor:
        if value >= crit:
            return RED
        if value >= warn:
            return AMBER
        return CYAN

    @staticmethod
    def _fraction(value: float, full_at: float) -> float:
        if full_at <= 0:
            return 0.0
        return max(0.0, min(1.0, value / full_at))

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return text[: max_chars - 1].rstrip() + "…"

    # ─── painting ──────────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.fillRect(rect, QColor(0, 0, 0, 0))

        # ── 1. Panel backdrop ────────────────────────────────────────
        panel_rect = QRectF(2.0, 2.0, self.w - 4.0, self.h - 4.0)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.setPen(QPen(PANEL_RIM, 1))
        painter.drawRoundedRect(panel_rect, 14.0, 14.0)

        cx, cy = self.cx, self.cy
        accent = self._accent_for_state()

        # ── 2. Outer cyan/amber glow halo (state-tinted) ─────────────
        glow = QRadialGradient(QPointF(cx, cy), self.R_GLOW)
        gcol = QColor(accent)
        gcol.setAlpha(160)
        glow.setColorAt(0.55, QColor(0, 0, 0, 0))
        glow.setColorAt(0.85, gcol)
        gouter = QColor(accent)
        gouter.setAlpha(0)
        glow.setColorAt(1.0, gouter)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), self.R_GLOW, self.R_GLOW)

        # ── 3. Outer ring track ──────────────────────────────────────
        outer_rect = QRectF(
            cx - self.R_OUTER, cy - self.R_OUTER,
            2 * self.R_OUTER, 2 * self.R_OUTER,
        )
        track_pen = QPen(CYAN_DIM, 2)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(outer_rect)

        # ── 4. Three quadrant arcs (CPU / RAM / GPU) ─────────────────
        # The bottom semicircle would visually fight the calendar text
        # row below the disc; we leave it as the faint cyan track so
        # the disc reads as Stark's three-rotor reactor.
        cpu_frac = self._fraction(self.cpu_pct, 100.0)
        ram_frac = self._fraction(self.ram_pct, 100.0)
        gpu_val  = self.gpu_temp_c if self.gpu_temp_c is not None else 0.0
        gpu_frac = max(0.0, min(1.0, (gpu_val - 30.0) / max(1.0, 95.0 - 30.0)))
        gpu_arc_color = (
            self._color_for_metric(gpu_val, GPU_WARN_C, GPU_CRIT_C)
            if self.gpu_temp_c is not None else CYAN_DIM
        )

        # Qt drawArc: 0° = 3 o'clock, CCW positive, angles in 1/16 °.
        # CPU — top-left wedge   (12 → 9 o'clock counterclockwise)
        # RAM — top-right wedge  (12 → 3 o'clock counterclockwise from anchor)
        # GPU — bottom wedge spanning 3 → 9 o'clock across the bottom semicircle
        gap_deg = 4.0
        arc_width = max(5.0, self.R_OUTER * 0.10)
        quadrants = [
            # name, frac, anchor_deg, span_deg, color
            ("CPU", cpu_frac, 180 - gap_deg, -(90 - 2 * gap_deg),
                self._color_for_metric(self.cpu_pct, CPU_WARN_PCT, CPU_CRIT_PCT)),
            ("RAM", ram_frac, 90 - gap_deg, -(90 - 2 * gap_deg),
                self._color_for_metric(self.ram_pct, RAM_WARN_PCT, RAM_CRIT_PCT)),
            ("GPU", gpu_frac, -gap_deg, -(180 - 2 * gap_deg), gpu_arc_color),
        ]

        for _, frac, anchor_deg, span_deg, color in quadrants:
            # Faint "remaining" trace across the whole usable wedge.
            faint_pen = QPen(CYAN_DIM, arc_width * 0.55)
            faint_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(faint_pen)
            start_16 = int(anchor_deg * 16)
            painter.drawArc(outer_rect, start_16, int(span_deg * 16))

            # Filled portion.
            fill_span_deg = span_deg * frac
            fill_pen = QPen(color, arc_width)
            fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(fill_pen)
            painter.drawArc(outer_rect, start_16, int(fill_span_deg * 16))

        # ── 5. Inner Bambu ring (only when a print is active) ────────
        if self.bambu_active:
            bambu_rect = QRectF(
                cx - self.R_INNER_BAMBU, cy - self.R_INNER_BAMBU,
                2 * self.R_INNER_BAMBU, 2 * self.R_INNER_BAMBU,
            )
            bambu_track = QPen(CYAN_DIM, 2)
            painter.setPen(bambu_track)
            painter.drawEllipse(bambu_rect)
            bambu_frac = max(0.0, min(1.0, self.bambu_percent / 100.0))
            bambu_color = (AMBER if self.bambu_gcode in ("PAUSE", "PREPARE")
                           else GREEN_SOFT)
            bambu_pen = QPen(bambu_color, max(3.0, self.R_OUTER * 0.06))
            bambu_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(bambu_pen)
            bambu_span = -int(360 * 16 * bambu_frac)
            painter.drawArc(bambu_rect, 90 * 16, bambu_span)

        # ── 6. Decorative rotating tick ring ─────────────────────────
        spin = (self.frame * 0.04) % (2 * math.pi)
        tick_radius = self.R_OUTER * 0.62
        tick_pen = QPen(CYAN_DIM, 1)
        painter.setPen(tick_pen)
        ticks = 28
        for i in range(ticks):
            theta = (i / ticks) * 2 * math.pi + spin
            inner = tick_radius
            outer = tick_radius + 4
            x1 = cx + inner * math.cos(theta)
            y1 = cy + inner * math.sin(theta)
            x2 = cx + outer * math.cos(theta)
            y2 = cy + outer * math.sin(theta)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # ── 7. Core ring + glowing speech-state hub ──────────────────
        # Heartbeat speed is faster when JARVIS is actively thinking so
        # the user reads the disc as "alive". Brightness modulates with
        # the live audio amplitude during speak/listen — bigger pulse,
        # bigger glow.
        beat_speed = 0.20
        if self.state == "thinking":
            beat_speed = 0.35
        elif self.state in ("standby", "sleep"):
            beat_speed = 0.08
        pulse = 0.5 * (1 + math.sin(self.frame * beat_speed))

        core_rect = QRectF(
            cx - self.R_CORE, cy - self.R_CORE,
            2 * self.R_CORE, 2 * self.R_CORE,
        )
        core_pen = QPen(accent, 2)
        painter.setPen(core_pen)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.drawEllipse(core_rect)

        if self.state == "speaking":
            hub_brightness = 0.55 + 0.45 * max(0.0, min(1.0, self.tts_amp))
        elif self.state == "listening":
            hub_brightness = 0.45 + 0.55 * max(0.0, min(1.0, self.mic_level))
        elif self.state == "thinking":
            hub_brightness = 0.55 + 0.45 * pulse
        elif self.state in ("standby", "sleep"):
            hub_brightness = 0.20 + 0.15 * pulse
        else:
            hub_brightness = 0.40 + 0.30 * pulse
        # Outer glow ring (slightly larger than the hub) — gives the
        # speech-state core its halo.
        halo_color = QColor(accent)
        halo_color.setAlpha(int(120 * hub_brightness + 35))
        halo_pen = QPen(halo_color, max(3.0, self.R_HUB * 0.4))
        painter.setPen(halo_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        halo_r = self.R_HUB * (1.15 + 0.15 * pulse)
        painter.drawEllipse(QPointF(cx, cy), halo_r, halo_r)

        hub_color = QColor(accent)
        hub_color.setAlpha(int(180 * hub_brightness + 60))
        painter.setBrush(QBrush(hub_color))
        painter.setPen(Qt.PenStyle.NoPen)
        hub_r = self.R_HUB * (0.85 + 0.15 * pulse)
        painter.drawEllipse(QPointF(cx, cy), hub_r, hub_r)

        # ── 8. Centre state label ────────────────────────────────────
        state_label = (self.state.upper() if self.state else "IDLE")[:9]
        state_font = QFont("Consolas", 10, QFont.Weight.Bold)
        painter.setFont(state_font)
        painter.setPen(QPen(TEXT_COLOR))
        painter.drawText(
            QRectF(cx - 60.0, cy - 8.0, 120.0, 16.0),
            int(Qt.AlignmentFlag.AlignCenter),
            state_label,
        )

        # ── 9. Bambu inline label (only when active) ─────────────────
        if self.bambu_active:
            if self.bambu_gcode == "PAUSE":
                bambu_label = f"PAUSED {self.bambu_percent:>3d}%"
                bambu_color = AMBER
            elif self.bambu_gcode == "PREPARE":
                bambu_label = "PREPARE…"
                bambu_color = AMBER
            else:
                bambu_label = f"PRINT {self.bambu_percent:>3d}%"
                bambu_color = GREEN_SOFT
            bambu_font = QFont("Consolas", 8, QFont.Weight.Bold)
            painter.setFont(bambu_font)
            painter.setPen(QPen(bambu_color))
            painter.drawText(
                QRectF(cx - 60.0, cy + 9.0, 120.0, 13.0),
                int(Qt.AlignmentFlag.AlignCenter),
                bambu_label,
            )

        # ── 10. Stat chips at the cardinals ──────────────────────────
        chip_font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(chip_font)

        # CPU chip — left of the disc
        cpu_color = self._color_for_metric(self.cpu_pct, CPU_WARN_PCT, CPU_CRIT_PCT)
        painter.setPen(QPen(cpu_color))
        cpu_x = max(8.0, cx - self.R_OUTER - 90.0)
        painter.drawText(
            QRectF(cpu_x, cy - 18.0, 80.0, 14.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"CPU {self.cpu_pct:>4.0f}%",
        )
        # RAM chip — left of the disc, just below CPU
        ram_color = self._color_for_metric(self.ram_pct, RAM_WARN_PCT, RAM_CRIT_PCT)
        painter.setPen(QPen(ram_color))
        painter.drawText(
            QRectF(cpu_x, cy + 4.0, 80.0, 14.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"RAM {self.ram_pct:>4.0f}%",
        )
        # GPU chip — right of the disc
        if self.gpu_temp_c is not None:
            gpu_color = self._color_for_metric(
                self.gpu_temp_c, GPU_WARN_C, GPU_CRIT_C,
            )
            gpu_text = f"GPU {self.gpu_temp_c:>3.0f}°C"
        else:
            gpu_color = DIM_TEXT
            gpu_text = "GPU — °C"
        painter.setPen(QPen(gpu_color))
        gpu_x = min(self.w - 88.0, cx + self.R_OUTER + 10.0)
        painter.drawText(
            QRectF(gpu_x, cy - 8.0, 80.0, 16.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            gpu_text,
        )

        # ── 11. Top-center now-playing row ───────────────────────────
        # Width-budget — leave the corners for the stat chips.
        text_w = max(160.0, self.w - 80.0)
        track_max_chars = max(18, int(text_w / 7.5))
        if self.track_title or self.track_artist:
            if self.track_artist and self.track_title:
                track_line = f"♪ {self.track_artist} — {self.track_title}"
            else:
                track_line = "♪ " + (self.track_title or self.track_artist)
            track_line = self._truncate(track_line, track_max_chars)
            track_font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
            painter.setFont(track_font)
            painter.setPen(QPen(CYAN_BRIGHT))
            painter.drawText(
                QRectF((self.w - text_w) / 2.0, 8.0, text_w, 18.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                track_line,
            )
        else:
            # Placeholder — keep the row visible so the disc doesn't
            # appear to "jump" when a track starts/stops.
            track_font = QFont("Segoe UI", 9, QFont.Weight.Normal)
            painter.setFont(track_font)
            painter.setPen(QPen(DIM_TEXT))
            painter.drawText(
                QRectF((self.w - text_w) / 2.0, 8.0, text_w, 18.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                "♪ —",
            )

        # ── 12. Bottom-center next-meeting row ───────────────────────
        meeting_max_chars = max(18, int(text_w / 7.0))
        if self.cal_subject:
            if self.cal_when:
                meeting_line = f"⏵ {self.cal_subject} · {self.cal_when}"
            else:
                meeting_line = f"⏵ {self.cal_subject}"
            meeting_line = self._truncate(meeting_line, meeting_max_chars)
            meeting_font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
            painter.setFont(meeting_font)
            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                QRectF((self.w - text_w) / 2.0, self.h - 26.0, text_w, 18.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                meeting_line,
            )
        else:
            meeting_font = QFont("Segoe UI", 9, QFont.Weight.Normal)
            painter.setFont(meeting_font)
            painter.setPen(QPen(DIM_TEXT))
            painter.drawText(
                QRectF((self.w - text_w) / 2.0, self.h - 26.0, text_w, 18.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                "⏵ —",
            )


class StarkStatusRingWindow(QWidget):
    """Frameless translucent always-on-top window hosting the scene.

    Click-through (WA_TransparentForMouseEvents) — information-only,
    never an interactive surface."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 parent_pid: int):
        super().__init__()
        self.parent_pid = parent_pid

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Click-through — wrapped in try/except because some Windows
        # builds quietly ignore this flag and would raise.
        try:
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True,
            )
        except Exception:
            pass
        self.setWindowTitle("JARVIS Stark Status Ring")

        self.setGeometry(x, y, width, height)

        self.scene = StarkStatusRingScene(width, height, parent_pid)
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, width, height)
        self.view.setStyleSheet("background: transparent; border: 0;")
        self.view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.view.setFrameShape(self.view.Shape.NoFrame)
        self.view.viewport().setAutoFillBackground(False)

        # Cyan-glow drop shadow — Qt's built-in shadow with zero offset.
        glow_fx = QGraphicsDropShadowEffect(self)
        glow_fx.setColor(CYAN_BRIGHT)
        glow_fx.setBlurRadius(36)
        glow_fx.setOffset(0, 0)
        self.view.setGraphicsEffect(glow_fx)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _on_tick(self) -> None:
        alive = self.scene.refresh_data()
        if not alive:
            self.timer.stop()
            QApplication.instance().quit()


def _print_install_hint() -> None:
    print(
        "[hud_v2] PyQt6 is not installed — this HUD requires PyQt6. "
        "Install with:  pip install PyQt6",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=1050)
    parser.add_argument("--y", type=int, default=-1420)
    parser.add_argument("--width", type=int, default=460)
    parser.add_argument("--height", type=int, default=340)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PYQT6:
        _print_install_hint()
        return 2

    app = QApplication(sys.argv[:1])
    win = StarkStatusRingWindow(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
