#!/usr/bin/env python3
"""
JARVIS Bambu chamber-camera HUD — live printer-camera view (PyQt6).

A movable, always-on-top panel that shows the printer's built-in camera
feed with a slim print-status footer (filename / layer / % / ETA), so the
user can watch the H2D print from the HUD. Spec: HUD Bambu-printer-camera
view (2026-06).

This widget is a pure *view*. The frame itself is fetched over the LAN by
``core/bambu_camera.py`` (RTSPS for the H2D / X-class on port 322, with a
port-6000 JPEG-stills fallback) and written to
``data/bambu_camera_frame.jpg``; the print-status line is read from
``bambu_overlay_state.json`` (written by skills/bambu_monitor.py). The HUD
never touches the network — it just polls those two files and repaints.

Behaviour:
  • When a fresh frame exists, it's drawn scaled-to-fit (aspect preserved)
    with a cyan reticle frame and a translucent status footer.
  • When the frame is missing / stale, a "CAMERA OFFLINE" placeholder is
    shown over the last-known print status, with a one-line reason pulled
    from the camera grabber's status sidecar (data/bambu_camera_state.json)
    — e.g. "LAN Only Liveview off?" or "printer asleep". So the feature
    degrades to the status readout instead of going blank.
  • Draggable (click-drag the body) and closable (double-right-click), like
    the other repositionable HUD surfaces. A live REC-style dot pulses while
    frames are arriving.

Lifecycle: auto-exits when the parent (JARVIS) PID dies OR when the control
file ``bambu_camera_hud_state.json`` flips to mode=off — identical to the
sibling HUD renderers so the existing launcher convention works.

CLI:
  python hud/bambu_camera_hud.py --x 2200 --y -1400 --width 360 \
      --height 300 --parent-pid 12345
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPoint
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    )
    from PyQt6.QtWidgets import QApplication, QWidget
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    class _QtMissing:  # stub: subclassable + callable so module-scope Qt refs don't NameError without PyQt6; main() exits 2 before any real use
        def __init__(self, *a, **k): pass
    Qt = QTimer = QRectF = QPoint = QPainter = QColor = QPen = QBrush = QFont = QImage = QPixmap = QApplication = QWidget = _QtMissing


TICK_MS = 250  # 4 fps repaint — the grabber refreshes ~1-2 fps, so this is plenty

PROJECT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_FILE       = os.path.join(PROJECT_DIR, "data", "bambu_camera_frame.jpg")
CAMERA_STATE_FILE = os.path.join(PROJECT_DIR, "data", "bambu_camera_state.json")
BAMBU_STATE_FILE = os.path.join(PROJECT_DIR, "bambu_overlay_state.json")
CONTROL_FILE     = os.path.join(PROJECT_DIR, "bambu_camera_hud_state.json")

# A frame older than this is treated as "no live feed" and the placeholder
# is shown. Matches core/bambu_camera.FRAME_STALE_SECONDS in spirit (kept
# local so the HUD has no import dependency on the grabber).
FRAME_STALE_SECONDS = 20.0

MIN_W, MIN_H = 240, 200

# Stark cyan palette — matches the other HUD surfaces so the look is one system.
if _HAS_PYQT6:
    CYAN        = QColor(76, 201, 255)
    CYAN_DIM    = QColor(27, 74, 102)
    CYAN_BRIGHT = QColor(158, 231, 255)
    TEXT_FG     = QColor(207, 238, 251)
    DIM_FG      = QColor(120, 160, 184)
    AMBER       = QColor(255, 179, 71)
    RED         = QColor(255, 91, 91)
    GREEN       = QColor(120, 235, 168)
    PANEL_BG    = QColor(4, 8, 13, 220)
    FOOTER_BG   = QColor(2, 6, 10, 200)
    PANEL_RIM   = QColor(76, 201, 255, 110)


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
    return (_read_json(CONTROL_FILE).get("mode") or "").lower() == "off"


def _format_minutes(minutes) -> str:
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return "<1m"
    if m < 60:
        return f"{m}m"
    h, rem = divmod(m, 60)
    return f"{h}h" if rem == 0 else f"{h}h {rem}m"


def _shorten(name: str, max_len: int = 30) -> str:
    if not name:
        return ""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _frame_age(camera_state: dict) -> float:
    """Seconds since the frame file was last written. Prefers the grabber's
    own ``last_frame_at`` timestamp (more accurate than mtime under a network
    share); falls back to the file mtime."""
    ts = 0.0
    try:
        ts = float(camera_state.get("last_frame_at") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0.0:
        try:
            ts = os.path.getmtime(FRAME_FILE)
        except OSError:
            return 1e9
    return max(0.0, time.time() - ts)


def _offline_reason(camera_state: dict, bambu_state: dict) -> str:
    """A short, human reason for the placeholder. Leans on the grabber's
    recorded error, then the printer's gcode state, then a generic line."""
    err = (camera_state.get("last_error") or "").strip()
    if err:
        low = err.lower()
        if "credential" in low:
            return "No printer credentials set"
        if "disabled" in low:
            return "Camera disabled (HUD_BAMBU_CAMERA)"
        if "no frame" in low or "path returned" in low:
            return "No camera signal — LAN Only Liveview on?"
        return err[:42]
    gs = (bambu_state.get("gcode_state") or "").upper()
    if not gs:
        return "Printer offline / asleep"
    if gs == "IDLE":
        return "Printer idle"
    return "Connecting to camera…"


class BambuCameraWindow(QWidget):
    """Frameless translucent always-on-top camera panel. Movable + closable
    (this one is NOT click-through — it's something the user looks at and
    repositions, like the unified HUD)."""

    def __init__(self, x: int, y: int, width: int, height: int, parent_pid: int):
        super().__init__()
        self.parent_pid = parent_pid
        self.frame = 0
        self._drag_offset: QPoint | None = None
        self._pixmap = None           # cached QPixmap of the current frame
        self._pixmap_mtime = 0.0      # mtime the cache was built from
        self._have_live_frame = False
        self._camera_state: dict = {}
        self._bambu_state: dict = {}

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("JARVIS Bambu Camera HUD")
        self.setMinimumSize(MIN_W, MIN_H)
        self.setGeometry(x, y, max(width, MIN_W), max(height, MIN_H))

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    # ── lifecycle ────────────────────────────────────────────────────────
    def _on_tick(self) -> None:
        if not _is_parent_alive(self.parent_pid) or _control_says_off():
            self.timer.stop()
            app = QApplication.instance()
            if app is not None:
                app.quit()
            return
        self.frame += 1
        self._camera_state = _read_json(CAMERA_STATE_FILE)
        self._bambu_state = _read_json(BAMBU_STATE_FILE)
        self._refresh_pixmap()
        self.update()

    def _refresh_pixmap(self) -> None:
        """(Re)load the frame file into a QPixmap when it has changed and is
        fresh. Sets ``_have_live_frame`` accordingly."""
        age = _frame_age(self._camera_state)
        fresh = age <= FRAME_STALE_SECONDS and os.path.exists(FRAME_FILE)
        self._have_live_frame = False
        if not fresh:
            return
        try:
            mtime = os.path.getmtime(FRAME_FILE)
        except OSError:
            return
        if self._pixmap is not None and mtime == self._pixmap_mtime:
            # Unchanged since last load — reuse the cached pixmap.
            self._have_live_frame = True
            return
        # Load the JPEG bytes ourselves and decode via QImage so a partially
        # written file (mid os.replace) just fails to decode rather than
        # throwing — the grabber writes atomically, but belt-and-braces.
        try:
            with open(FRAME_FILE, "rb") as f:
                data = f.read()
            img = QImage.fromData(data, "JPG")
            if img.isNull():
                return
            self._pixmap = QPixmap.fromImage(img)
            self._pixmap_mtime = mtime
            self._have_live_frame = not self._pixmap.isNull()
        except Exception:
            self._have_live_frame = False

    # ── drag-to-move + close ─────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None

    def mouseDoubleClickEvent(self, event) -> None:
        # Double right-click dismisses (matches the other corner widgets).
        if event.button() == Qt.MouseButton.RightButton:
            self.timer.stop()
            app = QApplication.instance()
            if app is not None:
                app.quit()

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        footer_h = 30
        view_rect = QRectF(0, 0, w, h - footer_h)

        # Panel backdrop.
        painter.setBrush(QBrush(PANEL_BG))
        painter.setPen(QPen(PANEL_RIM, 1))
        painter.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 10.0, 10.0)

        if self._have_live_frame and self._pixmap is not None:
            self._draw_frame(painter, view_rect)
        else:
            self._draw_placeholder(painter, view_rect)

        self._draw_reticle(painter, view_rect)
        self._draw_footer(painter, w, h, footer_h)
        painter.end()

    def _draw_frame(self, painter: QPainter, view_rect: QRectF) -> None:
        """Draw the camera pixmap scaled-to-fit (aspect preserved), centred."""
        pm = self._pixmap
        pw, ph = pm.width(), pm.height()
        if pw <= 0 or ph <= 0:
            self._draw_placeholder(painter, view_rect)
            return
        scale = min(view_rect.width() / pw, view_rect.height() / ph)
        dw, dh = pw * scale, ph * scale
        dx = view_rect.x() + (view_rect.width() - dw) / 2.0
        dy = view_rect.y() + (view_rect.height() - dh) / 2.0
        painter.drawPixmap(QRectF(dx, dy, dw, dh), pm,
                           QRectF(0, 0, pw, ph))

    def _draw_placeholder(self, painter: QPainter, view_rect: QRectF) -> None:
        """No live frame — fill with a dark field, a camera glyph, OFFLINE
        label, and the best-guess reason."""
        painter.setBrush(QBrush(QColor(6, 12, 18, 235)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(view_rect)

        cx = view_rect.center().x()
        cy = view_rect.center().y()

        # Simple camera glyph (body + lens) in dim cyan.
        body_w, body_h = 64.0, 40.0
        painter.setPen(QPen(CYAN_DIM, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            QRectF(cx - body_w / 2, cy - body_h / 2 - 6, body_w, body_h), 6, 6
        )
        painter.drawEllipse(QRectF(cx - 12, cy - 18, 24, 24))
        # A slash through it to read "no signal".
        painter.setPen(QPen(RED, 2))
        painter.drawLine(int(cx - body_w / 2), int(cy - body_h / 2 - 12),
                         int(cx + body_w / 2), int(cy + body_h / 2 + 2))

        painter.setPen(QPen(DIM_FG))
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(view_rect.x(), cy + 26, view_rect.width(), 18),
            int(Qt.AlignmentFlag.AlignCenter), "CAMERA OFFLINE",
        )
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(DIM_FG))
        reason = _offline_reason(self._camera_state, self._bambu_state)
        painter.drawText(
            QRectF(view_rect.x() + 6, cy + 44, view_rect.width() - 12, 16),
            int(Qt.AlignmentFlag.AlignCenter), reason,
        )

    def _draw_reticle(self, painter: QPainter, view_rect: QRectF) -> None:
        """Cyan corner brackets + a pulsing REC dot when frames are live."""
        painter.setPen(QPen(CYAN, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        x0, y0 = view_rect.x() + 6, view_rect.y() + 6
        x1, y1 = view_rect.right() - 6, view_rect.bottom() - 6
        seg = 16
        for (px, py, ddx, ddy) in (
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1),
        ):
            painter.drawLine(int(px), int(py), int(px + seg * ddx), int(py))
            painter.drawLine(int(px), int(py), int(px), int(py + seg * ddy))

        if self._have_live_frame:
            pulse = 0.5 + 0.5 * math.sin(self.frame * 0.25)
            dot = QColor(RED)
            dot.setAlpha(int(120 + 135 * pulse))
            painter.setBrush(QBrush(dot))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(x0 + 6, y0 + 4, 9, 9))
            painter.setPen(QPen(TEXT_FG))
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.drawText(int(x0 + 20), int(y0 + 12), "LIVE")

    def _draw_footer(self, painter: QPainter, w: int, h: int, footer_h: int) -> None:
        """Print-status strip: filename / layer / % / ETA, coloured by state."""
        footer_rect = QRectF(2, h - footer_h - 1, w - 4, footer_h)
        painter.setBrush(QBrush(FOOTER_BG))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(footer_rect, 8.0, 8.0)
        # Square off the top of the footer so it sits flush under the view.
        painter.drawRect(QRectF(2, h - footer_h - 1, w - 4, footer_h / 2))

        st = self._bambu_state
        gs = (st.get("gcode_state") or "").upper()
        accent = CYAN
        if gs == "FAILED" or st.get("print_error"):
            accent = RED
        elif gs in ("PAUSE", "PREPARE"):
            accent = AMBER
        elif gs == "RUNNING":
            accent = GREEN
        elif gs == "FINISH":
            accent = CYAN_BRIGHT

        # Left: filename or printer state.
        fname = _shorten(st.get("filename") or "", 26)
        left = fname or (gs.title() if gs else "No active print")
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(TEXT_FG))
        painter.drawText(
            QRectF(footer_rect.x() + 8, footer_rect.y(),
                   footer_rect.width() * 0.55, footer_rect.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            left,
        )

        # Right: layer / % / ETA.
        parts = []
        layer, total = st.get("layer_num"), st.get("total_layer")
        if layer and total:
            parts.append(f"L{int(layer)}/{int(total)}")
        pct = st.get("mc_percent")
        try:
            if pct is not None:
                parts.append(f"{int(pct)}%")
        except (TypeError, ValueError):
            pass
        eta = _format_minutes(st.get("mc_remaining"))
        if eta:
            parts.append(eta)
        right = "  ".join(parts)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QPen(accent))
        painter.drawText(
            QRectF(footer_rect.x() + footer_rect.width() * 0.45, footer_rect.y(),
                   footer_rect.width() * 0.55 - 8, footer_rect.height()),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            right,
        )


def _print_install_hint() -> None:
    print(
        "[bambu_camera_hud] PyQt6 is not installed — this HUD requires "
        "PyQt6. Install with:  pip install PyQt6",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=2200)
    parser.add_argument("--y", type=int, default=-1400)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if not _HAS_PYQT6:
        _print_install_hint()
        return 2

    app = QApplication(sys.argv[:1])
    win = BambuCameraWindow(
        args.x, args.y, args.width, args.height, args.parent_pid,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
