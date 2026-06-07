#!/usr/bin/env python3
"""
JARVIS holographic HUD v2 — Iron Man arc-reactor ring (PyQt6).

Spec: jarvis_todo.md 2026-05-29 09:18 (holographic_hud_v2). Upgrades the
existing holographic_overlay skill with a *permanent* arc-reactor-style
ring graphic rendered with PyQt6 + QGraphicsView (cyan glow shaders to
match the MCU aesthetic) instead of the tkinter canvases the v1 overlays
use. Refreshes every 500 ms.

Renders, on the top monitor:
  • A large circular arc reactor whose outer ring is split into two
    half-arcs filling with CPU% (left) and RAM% (right).
  • Central core showing the current ambient/wake state
    (Listening / Thinking / Speaking / Idle / Standby / Sleep).
  • Current intent tag (e.g. ``dry_wit``, ``urgent``) below the core.
  • Last action executed underneath the intent tag.
  • Rolling 5-line transcript scrolled in below the reactor.

Data sources:
  • hud_state.json — written by bobert_companion (state, mic_level,
    tts_amplitude, active_action, recent_action, last_intent_tag,
    transcript_history list).
  • psutil — CPU% / RAM% polled locally so we don't push high-frequency
    sensor data through the JSON state file.

Subprocess lifecycle: auto-exits when its parent (the launcher) dies.

CLI:
  python hud/holographic_hud_v2.py --x 0 --y -1440 --width 2560 \
                                   --height 1440 --parent-pid 12345
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# PyQt6 is the canonical backend per spec. If it isn't installed the
# script prints a friendly install hint and exits — the launcher treats a
# fast-exiting subprocess as "not engaged" so no other JARVIS surface
# breaks.
try:
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
        QRadialGradient, QLinearGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QGraphicsView, QGraphicsScene,
        QGraphicsDropShadowEffect,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    class _QtMissing:  # stub: subclassable + callable so module-scope Qt refs (incl. the unguarded QColor(...) constants below) don't NameError without PyQt6; main() exits 2 before any real use
        def __init__(self, *a, **k): pass
    Qt = QTimer = QRectF = QPointF = QPainter = QColor = QPen = QBrush = QFont = QPainterPath = QRadialGradient = QLinearGradient = QApplication = QWidget = QGraphicsView = QGraphicsScene = QGraphicsDropShadowEffect = _QtMissing


# ──────────────────────────────────────────────────────────────────────────
#  Layout / palette
# ──────────────────────────────────────────────────────────────────────────
TICK_MS = 500  # spec: refresh every 500ms

# Reactor sizing (fractions of min(window_w, window_h)).
R_OUTER_FRAC = 0.30
R_FILL_FRAC  = 0.27
R_CORE_FRAC  = 0.12
R_HUB_FRAC   = 0.055
R_GLOW_FRAC  = 0.34

# Palette — cyan reactor + amber for listen/speak + red for high-load.
CYAN         = QColor(76, 201, 255)      # #4cc9ff
CYAN_DIM     = QColor(27, 74, 102)       # #1b4a66
CYAN_BRIGHT  = QColor(158, 231, 255)     # #9ee7ff
TEXT_COLOR   = QColor(207, 238, 251)     # #cfeefb
DIM_TEXT     = QColor(93, 138, 163)      # #5d8aa3
AMBER        = QColor(255, 179, 71)      # #ffb347
AMBER_BRIGHT = QColor(255, 224, 160)     # #ffe0a0
RED          = QColor(255, 91, 91)       # #ff5b5b
PANEL_DARK   = QColor(4, 8, 13, 200)     # translucent

PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_STATE_FILE  = os.path.join(PROJECT_DIR, "hud_state.json")
# Dedicated control file (never hud_state.json — that's the main process's
# canonical snapshot). The launcher / any skill can write {"mode":"off"} here
# to dismiss this overlay. This is the escape hatch for a click-through,
# frameless window that otherwise has no dismiss gesture.
CONTROL_FILE    = os.path.join(PROJECT_DIR, "holographic_hud_v2_state.json")
# When launched with no real parent (--parent-pid 0/absent) the overlay would
# otherwise stay up forever (it's click-through, so the user can't even close
# it). Self-exit after this many seconds with no parent so it can never become
# an unkillable fullscreen layer. The launcher always passes a real PID, so a
# correctly-supervised HUD is never affected by this cap.
ORPHAN_MAX_LIFETIME_S = 1800.0  # 30 min


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
            return json.load(f)
    except Exception:
        return {}


def _control_says_off() -> bool:
    """True when the dedicated control file requests dismissal.

    Reads a *separate* file (never hud_state.json) so a skill can hide this
    overlay without racing the main process's continuous rewrites of the
    canonical snapshot."""
    data = _read_json(CONTROL_FILE)
    return (data.get("mode") or "").lower() == "off"


class ArcReactorScene(QGraphicsScene):
    """Custom scene that paints the arc reactor + textual readouts via
    drawBackground so we get a single full-redraw on each timer tick
    without juggling individual QGraphicsItem objects."""

    def __init__(self, width: int, height: int, parent_pid: int):
        super().__init__(0.0, 0.0, float(width), float(height))
        self.w = float(width)
        self.h = float(height)
        self.parent_pid = parent_pid
        self.frame = 0

        ref = min(self.w, self.h)
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        self.R_OUTER = ref * R_OUTER_FRAC
        self.R_FILL  = ref * R_FILL_FRAC
        self.R_CORE  = ref * R_CORE_FRAC
        self.R_HUB   = ref * R_HUB_FRAC
        self.R_GLOW  = ref * R_GLOW_FRAC

        # Live samples + smoothed channels.
        self.cpu_pct      = 0.0
        self.ram_pct      = 0.0
        self.tts_amp      = 0.0
        self.mic_level    = 0.0
        self.state        = "idle"
        self.active_action = ""
        self.recent_action = ""
        self.intent_tag   = ""
        self.transcripts: deque[str] = deque(maxlen=5)
        self.last_spoken  = ""
        self._started_at  = time.time()

        if _HAS_PSUTIL:
            try:
                # Prime the cpu_percent counter so the first non-zero
                # reading is meaningful.
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    # ─── data refresh (called by the QTimer) ────────────────────────────
    def refresh_data(self) -> bool:
        """Pull the latest state + sensor readings. Returns False when the
        overlay should close: the parent JARVIS process has died, the control
        file asked it to hide, or an orphaned (no-parent) overlay has exceeded
        its max lifetime. The timer stops and the window closes on False."""
        if not _is_parent_alive(self.parent_pid):
            return False
        # Dismissable even though the window is click-through: a skill can
        # write {"mode":"off"} to the control file.
        if _control_says_off():
            return False
        # Safety net for --parent-pid 0/absent: without a real parent to track,
        # the overlay would otherwise be unkillable except via Task Manager.
        if self.parent_pid <= 0 and (
                time.time() - self._started_at) > ORPHAN_MAX_LIFETIME_S:
            return False

        state = _read_json(HUD_STATE_FILE)
        self.state        = (state.get("state") or "Idle").lower()
        self.active_action = (state.get("active_action") or "").strip()
        self.recent_action = (state.get("recent_action") or "").strip()
        self.intent_tag    = (state.get("last_intent_tag") or "").strip()
        # Guard the float parses — a non-numeric value in the shared state
        # file must not raise inside the QTimer slot (an unhandled exception
        # there can abort the Qt event loop and kill the overlay). Mirror the
        # try/except used by the sibling PyQt HUDs; a bad value degrades to
        # the last-known channel value rather than crashing.
        try:
            self.tts_amp   = float(state.get("tts_amplitude") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            self.mic_level = float(state.get("mic_level") or 0.0)
        except (TypeError, ValueError):
            pass
        self.last_spoken   = (state.get("last_spoken") or "").strip()

        history = state.get("transcript_history") or []
        if isinstance(history, list):
            # Replace contents in-order. deque(maxlen) handles the cap.
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
        self.update()  # request repaint of background
        return True

    # ─── painting ───────────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Transparent fill — the QGraphicsView is built with a transparent
        # viewport so the background needs to stay clear.
        painter.fillRect(rect, QColor(0, 0, 0, 0))

        cx, cy = self.cx, self.cy
        state = self.state

        is_high_load = self.cpu_pct >= 90.0 or self.ram_pct >= 90.0
        if state == "listening":
            accent = AMBER
        elif state == "speaking":
            accent = AMBER_BRIGHT
        elif state == "thinking":
            accent = CYAN_BRIGHT
        elif state in ("standby", "sleep"):
            accent = CYAN_DIM
        else:
            accent = CYAN
        if is_high_load:
            accent = RED

        # ── 1. Outer glow halo ──
        # Multi-step radial gradient gives the cinematic cyan glow
        # without needing a real GPU shader.
        glow = QRadialGradient(QPointF(cx, cy), self.R_GLOW)
        gcol = QColor(accent)
        gcol.setAlpha(180)
        glow.setColorAt(0.55, QColor(0, 0, 0, 0))
        glow.setColorAt(0.85, gcol)
        gcol2 = QColor(accent)
        gcol2.setAlpha(0)
        glow.setColorAt(1.0, gcol2)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), self.R_GLOW, self.R_GLOW)

        # ── 2. Outer ring (CPU left, RAM right) ──
        # Track ring first (dim full circle).
        pen = QPen(CYAN_DIM)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        outer_rect = QRectF(
            cx - self.R_OUTER, cy - self.R_OUTER,
            2 * self.R_OUTER, 2 * self.R_OUTER,
        )
        painter.drawEllipse(outer_rect)

        # CPU sweep on the left half (90° → 270° anchor at top).
        cpu_color = RED if self.cpu_pct >= 90 else CYAN
        ram_color = RED if self.ram_pct >= 90 else CYAN
        pen.setWidth(6)
        pen.setColor(cpu_color)
        painter.setPen(pen)
        # Qt drawArc: angles in 1/16 degrees, 0° = 3 o'clock, CCW positive.
        cpu_span = int(180 * 16 * (self.cpu_pct / 100.0))
        painter.drawArc(outer_rect, 90 * 16, cpu_span)

        pen.setColor(ram_color)
        painter.setPen(pen)
        ram_span = -int(180 * 16 * (self.ram_pct / 100.0))
        painter.drawArc(outer_rect, 90 * 16, ram_span)

        # Rotating tick ring just inside the fill ring — gives the
        # animated MCU "spinning machinery" feel.
        spin = (self.frame * 0.04) % (2 * math.pi)
        tick_radius = self.R_FILL + 4
        pen.setColor(CYAN_DIM)
        pen.setWidth(1)
        painter.setPen(pen)
        ticks = 36
        for i in range(ticks):
            theta = (i / ticks) * 2 * math.pi + spin
            inner = tick_radius
            outer = tick_radius + 8
            x1 = cx + inner * math.cos(theta)
            y1 = cy + inner * math.sin(theta)
            x2 = cx + outer * math.cos(theta)
            y2 = cy + outer * math.sin(theta)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # ── 3. Mid ring — system "pulse" ring with breathing alpha ──
        pulse = 0.5 * (1 + math.sin(self.frame * 0.18))
        mid_color = QColor(accent)
        mid_color.setAlpha(int(120 + 100 * pulse))
        pen.setColor(mid_color)
        pen.setWidth(3)
        painter.setPen(pen)
        mid_rect = QRectF(
            cx - self.R_FILL, cy - self.R_FILL,
            2 * self.R_FILL, 2 * self.R_FILL,
        )
        painter.drawEllipse(mid_rect)

        # ── 4. Core ring ──
        core_rect = QRectF(
            cx - self.R_CORE, cy - self.R_CORE,
            2 * self.R_CORE, 2 * self.R_CORE,
        )
        pen.setColor(accent)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(PANEL_DARK))
        painter.drawEllipse(core_rect)

        # Core hub — pulsing solid disc. Brightness reflects TTS amplitude
        # when speaking so the reactor visibly "throbs" with JARVIS' voice.
        if state == "speaking":
            hub_brightness = 0.5 + 0.5 * max(0.0, min(1.0, self.tts_amp))
        elif state == "listening":
            hub_brightness = 0.4 + 0.6 * max(0.0, min(1.0, self.mic_level))
        else:
            hub_brightness = 0.5 + 0.4 * pulse
        hub_color = QColor(accent)
        hub_color.setAlpha(int(180 * hub_brightness + 60))
        painter.setBrush(QBrush(hub_color))
        painter.setPen(Qt.PenStyle.NoPen)
        hub_r = self.R_HUB * (0.9 + 0.15 * pulse)
        painter.drawEllipse(QPointF(cx, cy), hub_r, hub_r)

        # ── 5. Centre state label ──
        state_label = state.upper() if state else "IDLE"
        font = QFont("Consolas", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(TEXT_COLOR))
        text_rect = QRectF(cx - 120, cy - 12, 240, 24)
        painter.drawText(
            text_rect, int(Qt.AlignmentFlag.AlignCenter), state_label,
        )

        # ── 6. CPU / RAM numerals at cardinal positions ──
        font = QFont("Consolas", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(cpu_color))
        painter.drawText(
            QRectF(cx - self.R_OUTER - 80, cy - 14, 70, 28),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"CPU {self.cpu_pct:>4.0f}%",
        )
        painter.setPen(QPen(ram_color))
        painter.drawText(
            QRectF(cx + self.R_OUTER + 10, cy - 14, 80, 28),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"RAM {self.ram_pct:>4.0f}%",
        )

        # ── 7. Intent tag (below the reactor, above the action) ──
        intent_y = cy + self.R_OUTER + 30
        font = QFont("Consolas", 13, QFont.Weight.Bold)
        painter.setFont(font)
        intent_display = (self.intent_tag or "—").upper()
        painter.setPen(QPen(AMBER if self.intent_tag else DIM_TEXT))
        painter.drawText(
            QRectF(0, intent_y, self.w, 22),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            f"INTENT  ·  {intent_display}",
        )

        # ── 8. Last action executed ──
        action_y = intent_y + 26
        action_text = self.active_action or self.recent_action or "—"
        font = QFont("Consolas", 12, QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QPen(CYAN_BRIGHT if (self.active_action or
                                            self.recent_action) else DIM_TEXT))
        painter.drawText(
            QRectF(0, action_y, self.w, 22),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            f"ACTION  ·  {action_text[:80]}",
        )

        # ── 9. Rolling 5-line transcript scroll ──
        transcript_y0 = action_y + 38
        font = QFont("Consolas", 11, QFont.Weight.Normal)
        painter.setFont(font)
        lines = list(self.transcripts)
        # Pad so the layout doesn't jump when the deque is partially full.
        while len(lines) < 5:
            lines.insert(0, "")
        line_h = 22
        for i, line in enumerate(lines):
            # Faded older lines, brighter newest line at the bottom.
            age = (len(lines) - 1) - i  # 0 = newest at the bottom
            alpha = 255 - min(180, age * 45)
            color = QColor(TEXT_COLOR)
            color.setAlpha(alpha)
            painter.setPen(QPen(color))
            disp = (line or "").strip()
            if len(disp) > 110:
                disp = disp[:108] + "…"
            painter.drawText(
                QRectF(0, transcript_y0 + i * line_h, self.w, line_h),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                disp,
            )

        # ── 10. Top banner ──
        font = QFont("Consolas", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(DIM_TEXT))
        painter.drawText(
            QRectF(0, 18, self.w, 24),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            "J . A . R . V . I . S .   —   HOLOGRAPHIC HUD v2",
        )


class HoloHUDV2Window(QWidget):
    """Frameless transparent always-on-top window hosting the
    QGraphicsView that renders the arc reactor."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 parent_pid: int):
        super().__init__()
        self.parent_pid = parent_pid

        # Frameless, always-on-top, transparent — same intent as v1 but
        # using PyQt's native flags so the QGraphicsView paints onto a
        # truly transparent surface (no colour-keying needed).
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Clicks pass straight through to whatever is underneath — the
        # HUD is an information layer, not an interactive surface. Falls
        # back gracefully if the platform doesn't support it.
        try:
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True,
            )
        except Exception:
            pass

        self.setGeometry(x, y, width, height)

        self.scene = ArcReactorScene(width, height, parent_pid)
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

        # QGraphicsDropShadowEffect on the view gives the reactor a
        # genuine cyan bloom — this is the "cyan glow shader" called out
        # by the spec, implemented via Qt's built-in shadow-as-glow trick.
        glow_fx = QGraphicsDropShadowEffect(self)
        glow_fx.setColor(CYAN_BRIGHT)
        glow_fx.setBlurRadius(48)
        glow_fx.setOffset(0, 0)
        self.view.setGraphicsEffect(glow_fx)

        # 500 ms refresh per the spec.
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _on_tick(self) -> None:
        alive = self.scene.refresh_data()
        if not alive:
            # Parent JARVIS died, the control file flipped to "off", or an
            # orphaned overlay hit its lifetime cap — close ourselves so the
            # subprocess exits cleanly.
            self._quit()

    def _quit(self) -> None:
        try:
            self.timer.stop()
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def keyPressEvent(self, event) -> None:
        # Keyboard escape hatch: the window is click-through (mouse events pass
        # straight through), so a key is the only in-window dismiss gesture.
        # Esc or Q closes the overlay if it ever holds focus.
        try:
            key = event.key()
            if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
                self._quit()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


def _print_install_hint() -> None:
    print(
        "[holographic_hud_v2] PyQt6 is not installed — "
        "this HUD requires PyQt6. Install with:  pip install PyQt6",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PYQT6:
        _print_install_hint()
        return 2

    app = QApplication(sys.argv[:1])
    win = HoloHUDV2Window(args.x, args.y, args.width, args.height,
                          args.parent_pid)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
