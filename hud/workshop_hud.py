#!/usr/bin/env python3
"""
JARVIS workshop HUD — persistent top-right corner widget (PyQt6).

Spec: jarvis_todo.md 2026-05-29 11:26 (holographic_hud_v2 workshop_hud
replacement). Reimplements the previous tkinter workshop HUD with PyQt6
+ QGraphicsView so the small corner widget gets the same MCU-aesthetic
cyan-glow treatment as the fullscreen holographic_hud_v2 overlay.

Renders, refreshed every 500 ms, into a slim 260×290 panel pinned to
the top-right of the top monitor:

  • Arc-reactor ring graphic — outer ring split into CPU% (left arc)
    and RAM% (right arc) filling clockwise/counter-clockwise from the
    12 o'clock anchor; turns amber/red as load climbs.
  • Pulsing core — solid hub whose brightness pulses with the TTS
    amplitude when JARVIS is speaking, or with mic level when listening.
  • Centre state label — IDLE / LISTENING / THINKING / SPEAKING /
    STANDBY / SLEEP — read from hud_state.json `state`.
  • Intent tag — last `[intent:xxx]` chip from hud_state.json
    `last_intent_tag`.
  • Last action — `active_action` if a skill is running, otherwise the
    most recent `recent_action`.
  • Rolling 5-line transcript — most-recent-last, faded for older lines,
    sourced from `transcript_history` in hud_state.json.

Data sources:
  • hud_state.json — bobert_companion is the canonical writer.
  • psutil — CPU% / RAM% sampled locally; system_pulse only publishes
    a pre-formatted string, not the raw numbers we need to drive the
    arc fills.

Control file:
  workshop_hud_state.json at the project root — `{"mode": "on"|"off"}`.
  The skill-side launcher writes "off" to ask the running widget to
  exit cleanly without racing terminate().

CLI (preserved from the tkinter version so the launcher in
skills/holographic_overlay.py keeps working unchanged):
  python hud/workshop_hud.py --x 2300 --y -1420 --width 260 \
                             --height 290 --parent-pid 12345
"""
import argparse
import json
import math
import os
import sys
import tempfile
from collections import deque

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

# PyQt6 is the required backend per the 2026-05-29 spec. Fall back with
# a friendly install hint if it isn't present so the launcher just sees
# a fast-exiting subprocess instead of a stack trace.
try:
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPoint
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont,
        QRadialGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QGraphicsView, QGraphicsScene,
        QGraphicsDropShadowEffect, QMenu,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    class _QtMissing:  # stub: subclassable + callable so module-scope Qt refs (base classes, QColor(...)) don't NameError without PyQt6; main() exits 2 before any real use
        def __init__(self, *a, **k): pass
    Qt = QTimer = QRectF = QPointF = QPoint = QPainter = QColor = QPen = QBrush = QFont = QRadialGradient = QApplication = QWidget = QGraphicsView = QGraphicsScene = QGraphicsDropShadowEffect = QMenu = _QtMissing


TICK_MS = 500  # Spec: 500 ms refresh cadence.

# Palette — same cyan reactor + amber alert hues as holographic_hud_v2
# so the small corner widget and the big centre HUD read as a single
# visual system.
CYAN         = QColor(76, 201, 255)      if _HAS_PYQT6 else None  # #4cc9ff
CYAN_DIM     = QColor(27, 74, 102)       if _HAS_PYQT6 else None  # #1b4a66
CYAN_BRIGHT  = QColor(158, 231, 255)     if _HAS_PYQT6 else None  # #9ee7ff
TEXT_COLOR   = QColor(207, 238, 251)     if _HAS_PYQT6 else None  # #cfeefb
DIM_TEXT     = QColor(93, 138, 163)      if _HAS_PYQT6 else None  # #5d8aa3
AMBER        = QColor(255, 179, 71)      if _HAS_PYQT6 else None  # #ffb347
AMBER_BRIGHT = QColor(255, 224, 160)     if _HAS_PYQT6 else None  # #ffe0a0
RED          = QColor(255, 91, 91)       if _HAS_PYQT6 else None  # #ff5b5b
PANEL_DARK   = QColor(4, 8, 13, 220)     if _HAS_PYQT6 else None  # translucent base
PANEL_RIM    = QColor(10, 24, 32, 230)   if _HAS_PYQT6 else None

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_STATE_FILE = os.path.join(PROJECT_DIR, "hud_state.json")
CONTROL_FILE   = os.path.join(PROJECT_DIR, "workshop_hud_state.json")
GEOM_STATE_DIR = os.path.join(PROJECT_DIR, "data")
GEOM_STATE_FILE = os.path.join(GEOM_STATE_DIR, "workshop_hud_state.json")


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


class WorkshopHudScene(QGraphicsScene):
    """Renders the arc reactor + textual readouts via drawBackground so
    a single full redraw on each timer tick produces the whole frame —
    no need to juggle individual QGraphicsItems."""

    def __init__(self, width: int, height: int, parent_pid: int):
        super().__init__(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self.parent_pid = parent_pid
        self.frame = 0

        # Layout reference dimensions — recomputed on resize so the
        # ring and text scale together if the user later drags-to-
        # resize the window.
        self._recompute_layout()

        # Latest sampled / read state.
        self.cpu_pct       = 0.0
        self.ram_pct       = 0.0
        self.tts_amp       = 0.0
        self.mic_level     = 0.0
        self.state         = "idle"
        self.active_action = ""
        self.recent_action = ""
        self.intent_tag    = ""
        self.transcripts: deque[str] = deque(maxlen=5)

        if _HAS_PSUTIL:
            try:
                # Prime the counter so the first interval=None read isn't 0.
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def _recompute_layout(self) -> None:
        # Reactor sits in the upper half of the widget so the transcript
        # has room beneath without crowding. Sizes chosen to read clearly
        # at 260×290 while still scaling cleanly to ~2x via the launcher.
        self.cx = self.w / 2.0
        # Title strip ~18 px, then reactor centred ~78 px below the top.
        self.reactor_cy = 78.0
        self.R_OUTER = min(self.w, self.h) * 0.21
        self.R_FILL  = self.R_OUTER * 0.80
        self.R_CORE  = self.R_OUTER * 0.46
        self.R_HUB   = self.R_OUTER * 0.22
        self.R_GLOW  = self.R_OUTER * 1.30

    def resize_scene(self, width: int, height: int) -> None:
        self.setSceneRect(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self._recompute_layout()

    # ─── data refresh (called by the QTimer) ────────────────────────────
    def refresh_data(self) -> bool:
        """Pull the latest state + sensor readings. Returns False when
        the parent died or the control file flipped to off, so the
        owning window can close itself."""
        if not _is_parent_alive(self.parent_pid):
            return False
        if _control_says_off():
            return False

        state = _read_json(HUD_STATE_FILE)
        self.state         = (state.get("state") or "Idle").lower()
        self.active_action = (state.get("active_action") or "").strip()
        self.recent_action = (state.get("recent_action") or "").strip()
        self.intent_tag    = (state.get("last_intent_tag") or "").strip()
        try:
            self.tts_amp = float(state.get("tts_amplitude") or 0.0)
        except (TypeError, ValueError):
            self.tts_amp = 0.0
        try:
            self.mic_level = float(state.get("mic_level") or 0.0)
        except (TypeError, ValueError):
            self.mic_level = 0.0

        history = state.get("transcript_history") or []
        if isinstance(history, list):
            self.transcripts = deque(
                [str(t) for t in history[-5:]], maxlen=5
            )
        else:
            self.transcripts = deque(maxlen=5)

        if _HAS_PSUTIL:
            try:
                self.cpu_pct = float(psutil.cpu_percent(interval=None))
                self.ram_pct = float(psutil.virtual_memory().percent)
            except Exception:
                pass

        self.frame += 1
        self.update()
        return True

    # ─── helpers ────────────────────────────────────────────────────────
    def _accent_for_state(self) -> QColor:
        high_load = self.cpu_pct >= 90.0 or self.ram_pct >= 90.0
        if high_load:
            return RED
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
    def _bar_color(pct: float) -> QColor:
        if pct >= 90:
            return RED
        if pct >= 75:
            return AMBER
        return CYAN

    @staticmethod
    def _shorten(text: str, max_len: int) -> str:
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "…"

    # ─── painting ───────────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Clear to fully transparent — the host widget is translucent.
        painter.fillRect(rect, QColor(0, 0, 0, 0))

        # ── 1. Panel backdrop ─────────────────────────────────────────
        panel_rect = QRectF(2.0, 2.0, self.w - 4.0, self.h - 4.0)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.setPen(QPen(PANEL_RIM, 1))
        painter.drawRoundedRect(panel_rect, 8.0, 8.0)

        # Title bar accent stripe along the top.
        accent = self._accent_for_state()
        accent_top = QColor(accent)
        accent_top.setAlpha(200)
        painter.setBrush(QBrush(accent_top))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            QRectF(2.0, 2.0, self.w - 4.0, 4.0), 4.0, 4.0,
        )

        # ── 2. Title text + live tick dot ─────────────────────────────
        title_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(TEXT_COLOR))
        painter.drawText(
            QRectF(10.0, 8.0, self.w - 60.0, 16.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "J . A . R . V . I . S .",
        )
        tick_char = "●" if (self.frame % 2 == 0) else "○"
        painter.setPen(QPen(accent))
        painter.drawText(
            QRectF(self.w - 22.0, 8.0, 12.0, 16.0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            tick_char,
        )

        # ── 3. Outer glow halo ────────────────────────────────────────
        cx, cy = self.cx, self.reactor_cy
        glow = QRadialGradient(QPointF(cx, cy), self.R_GLOW)
        gcol = QColor(accent)
        gcol.setAlpha(140)
        glow.setColorAt(0.55, QColor(0, 0, 0, 0))
        glow.setColorAt(0.85, gcol)
        gcol_outer = QColor(accent)
        gcol_outer.setAlpha(0)
        glow.setColorAt(1.0, gcol_outer)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), self.R_GLOW, self.R_GLOW)

        # ── 4. Outer ring track + CPU/RAM split arcs ──────────────────
        outer_rect = QRectF(
            cx - self.R_OUTER, cy - self.R_OUTER,
            2 * self.R_OUTER, 2 * self.R_OUTER,
        )
        track_pen = QPen(CYAN_DIM, 2)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(outer_rect)

        # Qt's drawArc: 0° = 3 o'clock, CCW positive, angles in 1/16°.
        # CPU sweeps left half (CCW from 12 o'clock); RAM sweeps right
        # half (CW from 12 o'clock).
        cpu_color = self._bar_color(self.cpu_pct)
        ram_color = self._bar_color(self.ram_pct)
        arc_pen = QPen(cpu_color, 5)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        cpu_span = int(180 * 16 * (max(0.0, min(100.0, self.cpu_pct)) / 100.0))
        painter.drawArc(outer_rect, 90 * 16, cpu_span)

        arc_pen.setColor(ram_color)
        painter.setPen(arc_pen)
        ram_span = -int(180 * 16 * (max(0.0, min(100.0, self.ram_pct)) / 100.0))
        painter.drawArc(outer_rect, 90 * 16, ram_span)

        # ── 5. Rotating tick ring just inside the fill ring ───────────
        spin = (self.frame * 0.06) % (2 * math.pi)
        tick_radius = self.R_FILL + 2
        tick_pen = QPen(CYAN_DIM, 1)
        painter.setPen(tick_pen)
        ticks = 24
        for i in range(ticks):
            theta = (i / ticks) * 2 * math.pi + spin
            inner = tick_radius
            outer = tick_radius + 5
            x1 = cx + inner * math.cos(theta)
            y1 = cy + inner * math.sin(theta)
            x2 = cx + outer * math.cos(theta)
            y2 = cy + outer * math.sin(theta)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # ── 6. Mid ring — breathing pulse ─────────────────────────────
        pulse = 0.5 * (1 + math.sin(self.frame * 0.22))
        mid_color = QColor(accent)
        mid_color.setAlpha(int(110 + 90 * pulse))
        mid_pen = QPen(mid_color, 2)
        painter.setPen(mid_pen)
        painter.drawEllipse(QPointF(cx, cy), self.R_FILL, self.R_FILL)

        # ── 7. Core ring + pulsing hub ────────────────────────────────
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

        # ── 8. CPU% / RAM% numeric chips at the sides of the ring ─────
        chip_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(chip_font)
        painter.setPen(QPen(cpu_color))
        painter.drawText(
            QRectF(4.0, cy - 9.0, max(40.0, cx - self.R_OUTER - 8.0), 18.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"CPU {self.cpu_pct:>4.0f}%",
        )
        painter.setPen(QPen(ram_color))
        painter.drawText(
            QRectF(cx + self.R_OUTER + 4.0, cy - 9.0,
                   max(40.0, self.w - (cx + self.R_OUTER) - 8.0), 18.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"RAM {self.ram_pct:>4.0f}%",
        )

        # ── 9. Centre state label (inside the core) ───────────────────
        state_label = (self.state.upper() if self.state else "IDLE")[:9]
        state_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(state_font)
        painter.setPen(QPen(TEXT_COLOR))
        painter.drawText(
            QRectF(cx - 50.0, cy - 7.0, 100.0, 14.0),
            int(Qt.AlignmentFlag.AlignCenter),
            state_label,
        )

        # ── 10. Intent + action rows ──────────────────────────────────
        # Lay everything beneath the reactor (below cy + R_OUTER) on a
        # fixed grid so the typography never collides with the ring.
        text_y = cy + self.R_OUTER + 14.0

        intent_display = (self.intent_tag or "—").upper()
        intent_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(intent_font)
        painter.setPen(QPen(AMBER if self.intent_tag else DIM_TEXT))
        painter.drawText(
            QRectF(8.0, text_y, self.w - 16.0, 14.0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            f"INTENT · {self._shorten(intent_display, 22)}",
        )

        action_y = text_y + 16.0
        action_text = (self.active_action or self.recent_action or "—")
        action_color = (CYAN_BRIGHT if (self.active_action or self.recent_action)
                        else DIM_TEXT)
        action_font = QFont("Consolas", 8, QFont.Weight.Normal)
        painter.setFont(action_font)
        painter.setPen(QPen(action_color))
        painter.drawText(
            QRectF(8.0, action_y, self.w - 16.0, 14.0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            f"ACTION · {self._shorten(action_text, 24)}",
        )

        # Divider line above the transcript so the eye knows where the
        # "what JARVIS just heard" panel begins.
        divider_y = action_y + 18.0
        div_pen = QPen(CYAN_DIM, 1)
        painter.setPen(div_pen)
        painter.drawLine(QPointF(12.0, divider_y),
                         QPointF(self.w - 12.0, divider_y))

        # ── 11. Rolling 5-line transcript ─────────────────────────────
        # Newest line at the bottom; older lines fade out so the eye
        # is drawn to the freshest utterance.
        line_h = 12.0
        transcript_y0 = divider_y + 6.0
        transcript_font = QFont("Consolas", 7, QFont.Weight.Normal)
        painter.setFont(transcript_font)
        lines = list(self.transcripts)
        while len(lines) < 5:
            lines.insert(0, "")
        for i, line in enumerate(lines):
            age = (len(lines) - 1) - i  # 0 = newest at the bottom
            alpha = 255 - min(180, age * 45)
            color = QColor(TEXT_COLOR)
            color.setAlpha(alpha)
            painter.setPen(QPen(color))
            disp = self._shorten((line or "").strip(), 38)
            painter.drawText(
                QRectF(8.0, transcript_y0 + i * line_h,
                       self.w - 16.0, line_h + 2.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                disp,
            )


class WorkshopHudWindow(QWidget):
    """Frameless transparent always-on-top window hosting the
    QGraphicsView that renders the arc reactor. Interactive (drag-to-
    move, double-click to reset, right-click for a menu) — unlike the
    holographic_hud_v2 fullscreen overlay this widget is a small UI
    surface the user actually wants to grab and move."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 parent_pid: int):
        super().__init__()
        self.parent_pid = parent_pid
        self._anchor_x = x
        self._anchor_y = y
        self._anchor_w = width
        self._anchor_h = height
        self._drag_origin: QPoint | None = None
        self._drag_moved = False

        # Restore the last persisted position if the user moved the
        # widget in a previous session. Falls back to the CLI args.
        sx, sy = self._load_persisted_geometry()
        if sx is not None:
            x = sx
        if sy is not None:
            y = sy

        # Frameless, always-on-top, transparent — matches the v2 holo HUD
        # so both surfaces have the same compositor footprint.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # NOTE: deliberately do NOT set WA_TransparentForMouseEvents —
        # the workshop HUD is an interactive surface (drag, menu, close).
        self.setWindowTitle("JARVIS Workshop HUD")

        self.setGeometry(x, y, width, height)

        self.scene = WorkshopHudScene(width, height, parent_pid)
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
        # Forward viewport mouse events to this window so drag-to-move
        # works no matter where in the panel the user grabs.
        self.view.viewport().installEventFilter(self)

        # Cyan-glow drop shadow on the view — Qt's built-in shadow with
        # offset (0, 0) reads as a glow, no real GPU shader needed. This
        # is the "cyan glow shader" called out by the spec.
        glow_fx = QGraphicsDropShadowEffect(self)
        glow_fx.setColor(CYAN_BRIGHT)
        glow_fx.setBlurRadius(28)
        glow_fx.setOffset(0, 0)
        self.view.setGraphicsEffect(glow_fx)

        # 500 ms refresh per the spec.
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ─── lifecycle ──────────────────────────────────────────────────────
    def _on_tick(self) -> None:
        alive = self.scene.refresh_data()
        if not alive:
            # Parent JARVIS died OR control file flipped to off — close
            # the subprocess cleanly so the launcher can re-spawn later.
            self.timer.stop()
            QApplication.instance().quit()

    # ─── persisted geometry ─────────────────────────────────────────────
    def _load_persisted_geometry(self) -> tuple[int | None, int | None]:
        try:
            if not os.path.exists(GEOM_STATE_FILE):
                return None, None
            with open(GEOM_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            x = int(data["x"]) if "x" in data else None
            y = int(data["y"]) if "y" in data else None
            return x, y
        except Exception:
            return None, None

    def _save_geometry(self) -> None:
        """Atomic write of current x/y so the next bounce restores the
        widget where the user dragged it."""
        try:
            os.makedirs(GEOM_STATE_DIR, exist_ok=True)
            data = {
                "x": int(self.x()),
                "y": int(self.y()),
            }
            fd, tmp = tempfile.mkstemp(
                dir=GEOM_STATE_DIR, prefix=".workshop_", suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, GEOM_STATE_FILE)
            except Exception:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            # Persistence is best-effort.
            pass

    # ─── mouse handling (drag, double-click reset, right-click menu) ───
    def eventFilter(self, obj, event):
        # The QGraphicsView's viewport intercepts mouse events before
        # they reach this window, so we filter at the viewport level
        # and forward the ones we care about.
        et = event.type()
        try:
            from PyQt6.QtCore import QEvent
        except ImportError:
            return False
        if obj is self.view.viewport():
            if et == QEvent.Type.MouseButtonPress:
                self._on_mouse_press(event)
                return False  # let the view see it too
            if et == QEvent.Type.MouseMove:
                self._on_mouse_move(event)
                return False
            if et == QEvent.Type.MouseButtonRelease:
                self._on_mouse_release(event)
                return False
            if et == QEvent.Type.MouseButtonDblClick:
                self._on_mouse_double_click(event)
                return False
            if et == QEvent.Type.ContextMenu:
                self._on_context_menu(event)
                return True
        return super().eventFilter(obj, event)

    def _on_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            self._drag_moved = False

    def _on_mouse_move(self, event) -> None:
        if self._drag_origin is None:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        new_pos = event.globalPosition().toPoint() - self._drag_origin
        self.move(new_pos)
        self._drag_moved = True

    def _on_mouse_release(self, event) -> None:
        if self._drag_origin is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_moved:
                self._save_geometry()
            self._drag_origin = None
            self._drag_moved = False

    def _on_mouse_double_click(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Snap back to the launch-time anchor.
            self.setGeometry(
                self._anchor_x, self._anchor_y,
                self._anchor_w, self._anchor_h,
            )
            self._save_geometry()
        elif event.button() == Qt.MouseButton.RightButton:
            # Double-right-click dismisses (matches sibling overlays).
            QApplication.instance().quit()

    def _on_context_menu(self, event) -> None:
        try:
            menu = QMenu(self)
            menu.addAction(
                "Reset position (double-click)",
                lambda: self._on_mouse_double_click_synthetic(),
            )
            menu.addSeparator()
            menu.addAction("Close HUD", QApplication.instance().quit)
            menu.exec(event.globalPos())
        except Exception:
            pass

    def _on_mouse_double_click_synthetic(self) -> None:
        self.setGeometry(
            self._anchor_x, self._anchor_y,
            self._anchor_w, self._anchor_h,
        )
        self._save_geometry()


def _print_install_hint() -> None:
    print(
        "[workshop_hud] PyQt6 is not installed — this HUD requires "
        "PyQt6. Install with:  pip install PyQt6",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=2300)
    parser.add_argument("--y", type=int, default=-1420)
    parser.add_argument("--width", type=int, default=260)
    parser.add_argument("--height", type=int, default=290)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PYQT6:
        _print_install_hint()
        return 2

    app = QApplication(sys.argv[:1])
    win = WorkshopHudWindow(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
