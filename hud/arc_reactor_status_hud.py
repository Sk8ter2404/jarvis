#!/usr/bin/env python3
"""
JARVIS arc-reactor status HUD — system_pulse data as a circular ring (PyQt6).

Spec: jarvis_todo.md 2026-05-29 16:35 (overnight — arc-reactor-style HUD
widget). Replaces the generic `status_panel_strip` text strip with
something visually JARVIS-authentic: an arc-reactor disc whose outer
ring is split into four quadrant arcs (CPU / RAM / GPU util / network
bandwidth) and whose inner ring tracks the current Bambu print progress
when a print is active.

Renders, refreshed every 500 ms, into a 320×320 panel pinned to the
top-left of the top monitor (the workshop_hud already owns the top-right
corner — stacking on the opposite edge avoids occlusion):

  • Outer ring split into four quadrant arcs:
      ▸ top   (12→3 o'clock)  CPU %      cyan → amber → red
      ▸ right ( 3→6 o'clock)  RAM %      cyan → amber → red
      ▸ bottom( 6→9 o'clock)  GPU %      cyan → amber → red
      ▸ left  ( 9→12 o'clock) NET Mbps   cyan → amber → red
  • Inner ring (just inside the outer track) — Bambu print %, only
    drawn when a print is active (RUNNING / PAUSE / PREPARE).
  • Central core — pulsing solid hub with the current JARVIS state
    label (IDLE / LISTENING / THINKING / SPEAKING / STANDBY / SLEEP).
  • Four quadrant chip labels at the cardinals — `CPU 47%`, `RAM 62%`,
    `GPU 71%`, `NET 0.4 MB/s` — so the user can read the numbers
    without having to interpret the arc fill.

Data sources:
  • hud_state.json — JARVIS state, ambient flags, the formatted
    `pulse_strip` we already publish (used as a fallback if local
    sensor reads fail).
  • psutil — CPU % / RAM % sampled locally on every tick; network
    bandwidth computed as a delta between consecutive ticks so we
    don't have to wait 15 s for the next system_pulse publish.
  • nvidia-smi — GPU utilization, cached briefly because spawning a
    subprocess each 500 ms tick would dominate this widget's CPU footprint
    (but only ~1 s, since load swings fast and a long cache would alias it).
  • bambu_overlay_state.json — current print's mc_percent and
    gcode_state, written atomically by skills/bambu_monitor.py.

Subprocess lifecycle: auto-exits when its parent (the launcher) dies
OR when the control file `arc_reactor_status_state.json` flips to
mode=off. The launcher writes the control file so it can ask the
widget to retire cleanly without racing terminate().

Click-through: WA_TransparentForMouseEvents is set so the widget never
steals focus or clicks — it's an information layer only.

CLI:
  python hud/arc_reactor_status_hud.py --x 0 --y -1420 --width 320 \
      --height 320 --parent-pid 12345
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

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
    class _QtMissing:  # stub: subclassable + callable so module-scope Qt refs don't NameError without PyQt6; main() exits 2 before any real use
        def __init__(self, *a, **k): pass
    Qt = QTimer = QRectF = QPointF = QPainter = QColor = QPen = QBrush = QFont = QRadialGradient = QApplication = QWidget = QGraphicsView = QGraphicsScene = QGraphicsDropShadowEffect = _QtMissing


TICK_MS = 500  # spec: 500 ms refresh cadence

# Palette — same cyan reactor + amber alert hues as the sibling HUDs
# so all three surfaces read as one visual system.
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

PROJECT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_STATE_FILE     = os.path.join(PROJECT_DIR, "hud_state.json")
BAMBU_STATE_FILE   = os.path.join(PROJECT_DIR, "bambu_overlay_state.json")
CONTROL_FILE       = os.path.join(PROJECT_DIR, "arc_reactor_status_state.json")

# Thresholds for the colour transitions. Kept in sync with the
# system_pulse abnormality thresholds so the arc visibly turns amber/red
# at the same point JARVIS would proactively comment.
CPU_WARN_PCT      = 75.0
CPU_CRIT_PCT      = 90.0
RAM_WARN_PCT      = 75.0
RAM_CRIT_PCT      = 90.0
GPU_WARN_PCT      = 70.0
GPU_CRIT_PCT      = 90.0
# Network in MB/s — saturating a 1 Gb link is ~125 MB/s. Anything past
# the warn threshold reads as "doing something serious".
NET_WARN_MBPS     = 10.0
NET_CRIT_MBPS     = 50.0
NET_SCALE_MBPS    = 100.0  # arc fills to 100 % at 100 MB/s sustained

# GPU utilization readings via nvidia-smi are slow (subprocess + parse), but
# load swings 0→100→0 between ticks, so we cache only briefly — a long cache
# would alias/freeze the displayed value while still sparing most ticks a spawn.
GPU_CACHE_SECONDS = 1.0


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


class ArcReactorStatusScene(QGraphicsScene):
    """Renders the four-quadrant arc reactor + inner Bambu ring via
    drawBackground so a single full redraw on each timer tick produces
    the whole frame — no per-item bookkeeping."""

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
        self.gpu_util_pct: float | None = None
        self.net_mbps       = 0.0
        self.state          = "idle"
        self.tts_amp        = 0.0
        self.mic_level      = 0.0
        # Bambu state — only drawn when active.
        self.bambu_active   = False
        self.bambu_percent  = 0
        self.bambu_gcode    = ""
        # Network-rate tracking — psutil counters are cumulative; we
        # compute the per-tick delta to get instantaneous bandwidth.
        self._last_net_bytes: int | None = None
        self._last_net_at: float | None = None
        # GPU caching to keep nvidia-smi out of the hot path.
        self._gpu_cached_at = 0.0
        self._gpu_sampling = False   # guards against overlapping sampler threads

        if _HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)  # prime
            except Exception:
                pass
            try:
                io = psutil.net_io_counters()
                self._last_net_bytes = int(io.bytes_recv + io.bytes_sent)
                self._last_net_at = time.time()
            except Exception:
                pass

    def _recompute_layout(self) -> None:
        # The reactor is the whole widget — outer ring at ~38 % of the
        # min dimension, inner Bambu ring tucked inside.
        ref = min(self.w, self.h)
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        self.R_OUTER = ref * 0.38
        self.R_INNER_BAMBU = ref * 0.30   # bambu progress ring
        self.R_CORE = ref * 0.20
        self.R_HUB  = ref * 0.10
        self.R_GLOW = ref * 0.50

    def resize_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self._recompute_layout()

    # ─── sensor reads ──────────────────────────────────────────────────
    def _read_gpu_util(self) -> float | None:
        """Return the cached GPU utilization percent (0-100), never blocking.

        nvidia-smi can stall for up to its 2 s timeout per spawn, so it must not
        run on the Qt GUI/paint thread. On a cache miss we kick the actual
        sampling onto a short-lived background thread and immediately return the
        previously-cached value; the worker writes the fresh reading back when
        it lands. The displayed value and GPU_CACHE_SECONDS TTL are unchanged —
        _gpu_cached_at is stamped here (sample start) exactly as before so the
        refresh cadence stays the same and overlapping spawns are avoided."""
        now = time.time()
        if (now - self._gpu_cached_at) < GPU_CACHE_SECONDS:
            return self.gpu_util_pct
        self._gpu_cached_at = now
        if not self._gpu_sampling:
            self._gpu_sampling = True
            threading.Thread(target=self._gpu_sample_worker, daemon=True).start()
        return self.gpu_util_pct

    def _gpu_sample_worker(self) -> None:
        """Background worker: do the blocking nvidia-smi sample and publish the
        result back to the cached attribute the paint loop reads."""
        try:
            self.gpu_util_pct = self._sample_gpu_util()
        finally:
            self._gpu_sampling = False

    def _sample_gpu_util(self) -> float | None:
        """Blocking GPU-utilization sample (runs OFF the GUI thread)."""
        try:
            exe = shutil.which("nvidia-smi")
            if exe:
                out = subprocess.run(
                    [exe, "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0,
                    creationflags=(subprocess.CREATE_NO_WINDOW
                                   if sys.platform == "win32" else 0),
                )
                utils = []
                for v in (out.stdout or "").strip().splitlines():
                    v = v.strip()
                    if v.isdigit():
                        utils.append(int(v))
                if utils:
                    return float(max(utils))
        except Exception:
            pass
        # Fallback — parse the pulse_strip "GPU 33%" hint if present (digit-scan
        # stops at the first non-digit, so the trailing "%" terminates cleanly).
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

    def _read_net_mbps(self) -> float:
        """Instantaneous network throughput across all interfaces, in
        MB/s. Computed from the delta of psutil's cumulative byte
        counters since the last tick."""
        if not _HAS_PSUTIL:
            return 0.0
        try:
            io = psutil.net_io_counters()
            now = time.time()
            total = int(io.bytes_recv + io.bytes_sent)
            if self._last_net_bytes is None or self._last_net_at is None:
                self._last_net_bytes = total
                self._last_net_at = now
                return 0.0
            dt = max(1e-3, now - self._last_net_at)
            db = max(0, total - self._last_net_bytes)
            self._last_net_bytes = total
            self._last_net_at = now
            return (db / dt) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    # ─── data refresh (called by the QTimer) ───────────────────────────
    def refresh_data(self) -> bool:
        """Pull the latest state + sensor readings. Returns False when
        the parent died or the control file flipped to off so the owning
        window can close itself."""
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
        self.net_mbps = self._read_net_mbps()
        self.gpu_util_pct = self._read_gpu_util()

        bambu = _read_json(BAMBU_STATE_FILE)
        gs = (bambu.get("gcode_state") or "").upper()
        self.bambu_gcode = gs
        self.bambu_active = gs in ("RUNNING", "PAUSE", "PREPARE")
        try:
            self.bambu_percent = int(bambu.get("mc_percent") or 0)
        except (TypeError, ValueError):
            self.bambu_percent = 0

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
        """Map a metric value to a 0.0 → 1.0 arc fill fraction."""
        if full_at <= 0:
            return 0.0
        return max(0.0, min(1.0, value / full_at))

    # ─── painting ──────────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.fillRect(rect, QColor(0, 0, 0, 0))

        # ── 1. Panel backdrop (subtle so the reactor is the focus) ────
        panel_rect = QRectF(2.0, 2.0, self.w - 4.0, self.h - 4.0)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.setPen(QPen(PANEL_RIM, 1))
        painter.drawRoundedRect(panel_rect, 12.0, 12.0)

        cx, cy = self.cx, self.cy
        accent = self._accent_for_state()

        # ── 2. Outer cyan glow halo ───────────────────────────────────
        glow = QRadialGradient(QPointF(cx, cy), self.R_GLOW)
        gcol = QColor(accent)
        gcol.setAlpha(150)
        glow.setColorAt(0.55, QColor(0, 0, 0, 0))
        glow.setColorAt(0.85, gcol)
        gouter = QColor(accent)
        gouter.setAlpha(0)
        glow.setColorAt(1.0, gouter)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), self.R_GLOW, self.R_GLOW)

        # ── 3. Outer ring track (full circle dim cyan) ────────────────
        outer_rect = QRectF(
            cx - self.R_OUTER, cy - self.R_OUTER,
            2 * self.R_OUTER, 2 * self.R_OUTER,
        )
        track_pen = QPen(CYAN_DIM, 2)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(outer_rect)

        # ── 4. Four quadrant arcs (CPU / RAM / GPU / NET) ─────────────
        # Qt drawArc: 0° = 3 o'clock, CCW positive, angles in 1/16 °.
        # We define each quadrant's start angle (anchor) and fill the
        # arc clockwise inside that 90° wedge proportional to the
        # metric's fraction. Inside the wedge, the leftover (unfilled)
        # portion shows a faint cyan "remaining" trace.
        cpu_frac = self._fraction(self.cpu_pct, 100.0)
        ram_frac = self._fraction(self.ram_pct, 100.0)
        gpu_val  = self.gpu_util_pct if self.gpu_util_pct is not None else 0.0
        # GPU utilization maps 0 % → 0 up to 100 % → full, a flat 0-100 scale.
        gpu_frac = max(0.0, min(1.0, gpu_val / 100.0))
        net_frac = self._fraction(self.net_mbps, NET_SCALE_MBPS)

        # Anchors (Qt: 0° = 3 o'clock, CCW positive). Each quadrant
        # spans 90° and the arc fills clockwise from the anchor.
        # ▸ CPU  — top quadrant (12 o'clock → 3 o'clock)        : start 90°,  span -90°
        # ▸ RAM  — right quadrant ( 3 o'clock → 6 o'clock)       : start  0°,  span -90°
        # ▸ GPU  — bottom quadrant ( 6 o'clock → 9 o'clock)      : start 270°, span -90°
        # ▸ NET  — left quadrant ( 9 o'clock → 12 o'clock)       : start 180°, span -90°
        gpu_arc_color = (
            self._color_for_metric(gpu_val, GPU_WARN_PCT, GPU_CRIT_PCT)
            if self.gpu_util_pct is not None else CYAN_DIM
        )
        quadrants = [
            ("CPU", (cpu_frac, 90,
                self._color_for_metric(self.cpu_pct, CPU_WARN_PCT, CPU_CRIT_PCT))),
            ("RAM", (ram_frac, 0,
                self._color_for_metric(self.ram_pct, RAM_WARN_PCT, RAM_CRIT_PCT))),
            ("GPU", (gpu_frac, 270, gpu_arc_color)),
            ("NET", (net_frac, 180,
                self._color_for_metric(self.net_mbps, NET_WARN_MBPS, NET_CRIT_MBPS))),
        ]

        # Small angular gap between quadrants so the eye reads them as
        # four distinct arcs rather than one continuous ring.
        gap_deg = 4.0
        usable_span_deg = 90.0 - 2 * gap_deg  # per quadrant

        arc_width = max(5.0, self.R_OUTER * 0.10)
        for _, (frac, anchor_deg, color) in quadrants:
            # Faint "remaining" trace across the full usable wedge.
            faint_pen = QPen(CYAN_DIM, arc_width * 0.55)
            faint_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(faint_pen)
            start_16 = int((anchor_deg - gap_deg) * 16)
            painter.drawArc(outer_rect, start_16, -int(usable_span_deg * 16))

            # Filled portion.
            fill_span_deg = usable_span_deg * frac
            fill_pen = QPen(color, arc_width)
            fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(fill_pen)
            painter.drawArc(outer_rect, start_16, -int(fill_span_deg * 16))

        # ── 5. Inner Bambu ring (only when a print is active) ─────────
        if self.bambu_active:
            bambu_rect = QRectF(
                cx - self.R_INNER_BAMBU, cy - self.R_INNER_BAMBU,
                2 * self.R_INNER_BAMBU, 2 * self.R_INNER_BAMBU,
            )
            bambu_track = QPen(CYAN_DIM, 2)
            painter.setPen(bambu_track)
            painter.drawEllipse(bambu_rect)
            # Print progress arc — sweeps clockwise from 12 o'clock,
            # full 360°. Pause/Prepare get a softer amber tint so the
            # user can tell the print isn't actively pushing layers.
            bambu_frac = max(0.0, min(1.0, self.bambu_percent / 100.0))
            if self.bambu_gcode in ("PAUSE", "PREPARE"):
                bambu_color = AMBER
            else:
                bambu_color = GREEN_SOFT
            bambu_pen = QPen(bambu_color, max(3.0, self.R_OUTER * 0.06))
            bambu_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(bambu_pen)
            bambu_span = -int(360 * 16 * bambu_frac)
            painter.drawArc(bambu_rect, 90 * 16, bambu_span)

        # ── 6. Rotating tick ring — animated "machinery" decoration ───
        spin = (self.frame * 0.05) % (2 * math.pi)
        tick_radius = self.R_OUTER * 0.62
        tick_pen = QPen(CYAN_DIM, 1)
        painter.setPen(tick_pen)
        ticks = 24
        for i in range(ticks):
            theta = (i / ticks) * 2 * math.pi + spin
            inner = tick_radius
            outer = tick_radius + 4
            x1 = cx + inner * math.cos(theta)
            y1 = cy + inner * math.sin(theta)
            x2 = cx + outer * math.cos(theta)
            y2 = cy + outer * math.sin(theta)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # ── 7. Core ring + pulsing hub ────────────────────────────────
        pulse = 0.5 * (1 + math.sin(self.frame * 0.20))
        core_rect = QRectF(
            cx - self.R_CORE, cy - self.R_CORE,
            2 * self.R_CORE, 2 * self.R_CORE,
        )
        core_pen = QPen(accent, 2)
        painter.setPen(core_pen)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.drawEllipse(core_rect)

        if self.state == "speaking":
            hub_brightness = 0.5 + 0.5 * max(0.0, min(1.0, self.tts_amp))
        elif self.state == "listening":
            hub_brightness = 0.4 + 0.6 * max(0.0, min(1.0, self.mic_level))
        else:
            hub_brightness = 0.5 + 0.4 * pulse
        hub_color = QColor(accent)
        hub_color.setAlpha(int(180 * hub_brightness + 60))
        painter.setBrush(QBrush(hub_color))
        painter.setPen(Qt.PenStyle.NoPen)
        hub_r = self.R_HUB * (0.9 + 0.15 * pulse)
        painter.drawEllipse(QPointF(cx, cy), hub_r, hub_r)

        # ── 8. Centre state label ─────────────────────────────────────
        state_label = (self.state.upper() if self.state else "IDLE")[:9]
        state_font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(state_font)
        painter.setPen(QPen(TEXT_COLOR))
        painter.drawText(
            QRectF(cx - 60.0, cy - 8.0, 120.0, 16.0),
            int(Qt.AlignmentFlag.AlignCenter),
            state_label,
        )

        # ── 9. Quadrant chip labels at the cardinal corners ───────────
        # Top, right, bottom, left — clipped to the panel so a wide
        # value never spills off the rounded backdrop. Numbers are
        # drawn in the same colour the matching arc is painted in so
        # the eye links chip → arc instantly.
        chip_font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(chip_font)

        # CPU — top centre
        cpu_color = self._color_for_metric(self.cpu_pct, CPU_WARN_PCT, CPU_CRIT_PCT)
        painter.setPen(QPen(cpu_color))
        painter.drawText(
            QRectF(cx - 60.0, max(8.0, cy - self.R_OUTER - 22.0), 120.0, 16.0),
            int(Qt.AlignmentFlag.AlignCenter),
            f"CPU {self.cpu_pct:>4.0f}%",
        )
        # RAM — right side, vertically centred
        ram_color = self._color_for_metric(self.ram_pct, RAM_WARN_PCT, RAM_CRIT_PCT)
        painter.setPen(QPen(ram_color))
        ram_x = min(self.w - 70.0, cx + self.R_OUTER + 6.0)
        painter.drawText(
            QRectF(ram_x, cy - 8.0, max(40.0, self.w - ram_x - 6.0), 16.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"RAM {self.ram_pct:>4.0f}%",
        )
        # GPU — bottom centre
        if self.gpu_util_pct is not None:
            gpu_color = self._color_for_metric(
                self.gpu_util_pct, GPU_WARN_PCT, GPU_CRIT_PCT,
            )
            gpu_text = f"GPU {self.gpu_util_pct:>3.0f}%"
        else:
            gpu_color = DIM_TEXT
            gpu_text = "GPU — %"
        painter.setPen(QPen(gpu_color))
        painter.drawText(
            QRectF(cx - 60.0, min(self.h - 22.0, cy + self.R_OUTER + 6.0),
                   120.0, 16.0),
            int(Qt.AlignmentFlag.AlignCenter),
            gpu_text,
        )
        # NET — left side, vertically centred
        net_color = self._color_for_metric(
            self.net_mbps, NET_WARN_MBPS, NET_CRIT_MBPS,
        )
        painter.setPen(QPen(net_color))
        if self.net_mbps >= 10.0:
            net_text = f"NET {self.net_mbps:>4.1f}MB/s"
        else:
            net_text = f"NET {self.net_mbps:>4.2f}MB/s"
        net_w = 110.0
        net_x = max(6.0, cx - self.R_OUTER - net_w - 4.0)
        painter.drawText(
            QRectF(net_x, cy - 8.0, net_w, 16.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            net_text,
        )

        # ── 10. Bambu print label (only when active) ─────────────────
        if self.bambu_active:
            # Sit the chip just below the centre state label, inside
            # the core, so it reads as the print's status hub.
            bambu_label = f"PRINT {self.bambu_percent:>3d}%"
            if self.bambu_gcode == "PAUSE":
                bambu_label = f"PAUSED {self.bambu_percent:>3d}%"
            elif self.bambu_gcode == "PREPARE":
                bambu_label = "PREPARE…"
            bambu_font = QFont("Consolas", 8, QFont.Weight.Bold)
            painter.setFont(bambu_font)
            bambu_color = AMBER if self.bambu_gcode in ("PAUSE", "PREPARE") else GREEN_SOFT
            painter.setPen(QPen(bambu_color))
            painter.drawText(
                QRectF(cx - 60.0, cy + 8.0, 120.0, 14.0),
                int(Qt.AlignmentFlag.AlignCenter),
                bambu_label,
            )


class ArcReactorStatusWindow(QWidget):
    """Frameless translucent always-on-top window hosting the scene.

    Click-through (WA_TransparentForMouseEvents) — this widget is an
    information layer, never an interactive surface."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 parent_pid: int):
        super().__init__()
        self.parent_pid = parent_pid

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Click-through — mouse events pass right through to whatever's
        # underneath. Some Windows builds quietly ignore this flag, so
        # we wrap it in a try/except: a Windows version that doesn't
        # honour it just means the user can click the panel, which is
        # mildly annoying but not a feature break.
        try:
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True,
            )
        except Exception:
            pass
        self.setWindowTitle("JARVIS Arc Reactor Status HUD")

        self.setGeometry(x, y, width, height)

        self.scene = ArcReactorStatusScene(width, height, parent_pid)
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

        # Cyan-glow drop shadow on the view — Qt's built-in shadow with
        # zero offset reads as a glow.
        glow_fx = QGraphicsDropShadowEffect(self)
        glow_fx.setColor(CYAN_BRIGHT)
        glow_fx.setBlurRadius(32)
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
        "[arc_reactor_status_hud] PyQt6 is not installed — this HUD "
        "requires PyQt6. Install with:  pip install PyQt6",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=20)
    parser.add_argument("--y", type=int, default=-1420)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PYQT6:
        _print_install_hint()
        return 2

    app = QApplication(sys.argv[:1])
    win = ArcReactorStatusWindow(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
