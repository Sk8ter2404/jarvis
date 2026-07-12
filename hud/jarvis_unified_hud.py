#!/usr/bin/env python3
"""
JARVIS Unified HUD — the single, fully-featured heads-up display (PyQt6).

This replaces the sprawl of overlapping overlays (fullscreen reticle, workshop
HUD, arc-reactor status ring, bambu corner, workshop print monitor, holo HUD
v1/v2, briefing card, workshop canvas) with ONE polished panel that the user
can actually move and size.

Why this exists (user, 2026-05-30): "the huds are messed up too theres too many
some i cant reposition and resize i want a fully upgraded one fully feature
packed." The old overlays were frameless windows with hard-coded geometry and
no drag/resize handlers — so several literally could not be moved. This one is:

  • DRAGGABLE  — click anywhere on the body and drag.
  • RESIZABLE  — a corner grip (and it reflows proportionally).
  • PERSISTENT — position + size are saved to unified_hud_geometry.json and
    restored on the next launch / JARVIS bounce.
  • FEATURE-PACKED — a state-reactive arc-reactor core, four live system arcs
    (CPU / RAM / GPU / NET), now-playing, weather + 3-day forecast, next
    calendar event, Bambu H2D print progress + ETA, unread mail / alerts, and
    a live transcript of the last few things heard + what JARVIS is doing.

Data sources (all already published elsewhere — this is a *view*):
  • hud_state.json          — state, now_playing, transcript_history, mic/tts
                              amplitude, now_doing, alert flags (set_state()).
  • bambu_overlay_state.json — gcode_state + mc_percent + ETA (bambu_monitor).
  • psutil + nvidia-smi      — CPU / RAM / NET locally; GPU util cached.
  • hud_card.py gatherers    — weather (wttr), 3-day forecast, calendar +
                              unread mail (Microsoft Graph). Reused, not
                              duplicated. Refreshed on a background thread.

Lifecycle: auto-exits when the parent (JARVIS) PID dies or when the control
file unified_hud_state.json flips to mode=off. CLI mirrors the sibling HUDs so
the existing launcher convention (--x --y --width --height --parent-pid) works,
but a saved geometry overrides the CLI default.

CLI:
  python hud/jarvis_unified_hud.py --x 2280 --y -1400 --width 420 \
      --height 560 --parent-pid 12345
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

# CREATE_NO_WINDOW safety net: this overlay's ~3s nvidia-smi utilization poll
# was popping a Windows Terminal ghost window PER POLL (~38 windows in 30 min,
# live 2026-07-10) — the overlay runs as pythonw with no console, so every
# unflagged console spawn allocates a visible one. hud/ isn't a package root,
# hence the parent-dir insert. Fail-open: a missing core module must never
# stop the HUD.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.no_window_subprocess import install as _install_no_window
    _install_no_window()
except Exception:
    pass

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPoint
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QRadialGradient,
        QLinearGradient, QPainterPath, QPixmap,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QSizeGrip, QPushButton,
    )
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    class _QtMissing:  # stub: subclassable + callable so module-scope Qt refs don't NameError without PyQt6; main() exits 2 before any real use
        def __init__(self, *a, **k): pass
    Qt = QTimer = QRectF = QPointF = QPoint = QPainter = QColor = QPen = QBrush = QFont = QRadialGradient = QLinearGradient = QPainterPath = QPixmap = QApplication = QWidget = QSizeGrip = QPushButton = _QtMissing


TICK_MS = 500                 # cheap data refresh + repaint cadence
# The camera-preview tile gets its OWN faster cadence, decoupled from the 500 ms
# state poll, so the live mirror looks smooth instead of stepping at 2 fps. Only
# the preview JPEG is re-read + the widget repainted on this tick (and only when
# the frame actually changed — see _refresh_camera_preview's return value), so
# the rest of the HUD state stays on the cheap TICK_MS loop. 75 ms ≈ 13.3 fps,
# comfortably above the writer's ~0.15 s (~6-7 fps) cadence so we never starve it.
CAMERA_TICK_MS = 75           # camera-preview-only refresh cadence (~13 fps)
SLOW_REFRESH_S = 600.0        # weather/calendar refresh cadence (background)
GPU_CACHE_SECONDS = 1.0       # utilization swings 0→100→0 fast; a long cache aliases it

PROJECT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_STATE_FILE   = os.path.join(PROJECT_DIR, "hud_state.json")
BAMBU_STATE_FILE = os.path.join(PROJECT_DIR, "bambu_overlay_state.json")
CONTROL_FILE     = os.path.join(PROJECT_DIR, "unified_hud_state.json")
GEOMETRY_FILE    = os.path.join(PROJECT_DIR, "unified_hud_geometry.json")

# Live camera preview: the main process (bobert_companion) writes a single,
# overwriting, downscaled JPEG of the primary face-tracking frame here whenever
# the camera is on + tracking. We show it in a corner. A file older than
# CAMERA_PREVIEW_STALE_S is treated as "camera off" (placeholder) so a missed
# delete on the writer side can't leave a frozen frame on screen. See the
# HUD_CAMERA_PREVIEW config flag + _hud_camera_preview_* in bobert_companion.py.
CAMERA_PREVIEW_FILE     = os.path.join(PROJECT_DIR, "data", ".hud_camera_preview.jpg")
CAMERA_PREVIEW_STALE_S  = 2.5     # > writer's ~0.15s cadence; tolerates hitches

MIN_W, MIN_H = 300, 380
# Upper bounds for the panel. Shared by BOTH the CLI-default clamp (a stray
# full-monitor --width must never make a monster bar) and the saved-geometry
# restore clamp, so a stale unified_hud_geometry.json from a since-removed huge
# monitor can't resurrect an oversized HUD either.
MAX_W, MAX_H = 900, 1100

# Stark cyan palette — matches the retired HUDs so the look is continuous.
if _HAS_PYQT6:
    CYAN        = QColor(76, 201, 255)     # #4cc9ff
    CYAN_DIM    = QColor(27, 74, 102)      # #1b4a66
    CYAN_BRIGHT = QColor(158, 231, 255)    # #9ee7ff
    TEXT_FG     = QColor(207, 238, 251)    # #cfeefb
    DIM_FG      = QColor(120, 160, 184)    # softer than the old #5d8aa3
    GOLD        = QColor(255, 209, 102)    # #ffd166
    AMBER       = QColor(255, 179, 71)     # #ffb347
    AMBER_BRT   = QColor(255, 224, 160)    # #ffe0a0
    RED         = QColor(255, 91, 91)      # #ff5b5b
    GREEN       = QColor(120, 235, 168)    # #78eba8
    PANEL_TOP   = QColor(8, 16, 24, 232)
    PANEL_BOT   = QColor(3, 7, 12, 238)
    PANEL_RIM   = QColor(76, 201, 255, 90)

# Metric thresholds (kept in sync with system_pulse so the arc turns amber/red
# at the same point JARVIS would proactively comment).
CPU_WARN, CPU_CRIT = 75.0, 90.0
RAM_WARN, RAM_CRIT = 75.0, 90.0
GPU_WARN, GPU_CRIT = 70.0, 90.0   # GPU utilization % (was 70/82 °C)
NET_WARN, NET_CRIT = 10.0, 50.0
NET_FULL_MBPS      = 100.0


# ─── small helpers ──────────────────────────────────────────────────────────
def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _is_parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    # AUTHORITATIVE check first (2026-07-12): psutil.pid_exists reads TRUE
    # for a DEAD-but-unreaped Windows process (any open handle keeps the row)
    # — this HUD outlived its terminated parent by 25 minutes and the owner
    # saw two of everything. core.parent_watch uses WaitForSingleObject on a
    # SYNCHRONIZE handle, which is signaled the instant the parent dies.
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


def _control_says_off() -> bool:
    return (_read_json(CONTROL_FILE).get("mode") or "").lower() == "off"


def _camera_preview_fresh_at(now: float) -> bool:
    """True iff the live camera-preview JPEG exists and was written within the
    last CAMERA_PREVIEW_STALE_S seconds. The writer removes the file when the
    camera is off / tracking is paused; this staleness gate is the belt-and-
    braces fallback for a missed delete, so the HUD shows the placeholder rather
    than a frozen frame. Pure (stat + time only) so it is unit-testable without
    Qt or a real camera. Any stat error → not fresh."""
    try:
        return (now - os.path.getmtime(CAMERA_PREVIEW_FILE)) <= CAMERA_PREVIEW_STALE_S
    except OSError:
        return False


def _load_saved_geometry() -> dict | None:
    g = _read_json(GEOMETRY_FILE)
    try:
        x, y, w, h = int(g["x"]), int(g["y"]), int(g["w"]), int(g["h"])
        if w >= MIN_W and h >= MIN_H:
            return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        pass
    return None


def _rects_overlap(ax: int, ay: int, aw: int, ah: int,
                   bx: int, by: int, bw: int, bh: int) -> bool:
    """True iff the two (x, y, w, h) rectangles share any area. A shared edge
    only (zero-area touch) does NOT count — the HUD must have real pixels on a
    screen to be reachable."""
    return (ax < bx + bw and bx < ax + aw and
            ay < by + bh and by < ay + ah)


def _validate_geometry(geo: dict, screens: list, default_xy: tuple) -> dict:
    """Make a restored window rect safe to show, given the available screen
    rects.

    A frameless, no-taskbar HUD that restores a stale geometry can otherwise
    strand fully OFF-screen after a monitor-layout change (the only recovery
    being to hand-delete unified_hud_geometry.json). This guards both ways:

      • the width/height are clamped to [MIN_*, MAX_*] (a since-removed huge
        monitor can't resurrect a monster bar), and
      • after clamping the size, if the rect overlaps NO available screen its
        top-left is reset to ``default_xy`` (the CLI anchor) so it lands back
        on-screen. A rect that still touches any screen is left where the user
        put it — straddling a bezel is intentional and preserved.

    Pure (arithmetic only) so it is unit-testable without Qt or a real display.
    ``screens`` is a list of (x, y, w, h); an empty list (no screens known)
    skips the on-screen test and only clamps the size.
    """
    x, y = int(geo["x"]), int(geo["y"])
    w = min(max(MIN_W, int(geo["w"])), MAX_W)
    h = min(max(MIN_H, int(geo["h"])), MAX_H)
    if screens:
        on_screen = any(
            _rects_overlap(x, y, w, h, int(sx), int(sy), int(sw), int(sh))
            for (sx, sy, sw, sh) in screens
        )
        if not on_screen:
            x, y = int(default_xy[0]), int(default_xy[1])
    return {"x": x, "y": y, "w": w, "h": h}


def _available_screen_rects() -> list:
    """Best-effort list of available (work-area) screen rects as (x, y, w, h).

    Prefers Qt's own view of the screens (so it matches exactly where the HUD
    can actually be placed) when a QApplication exists; otherwise falls back to
    the project's MONITORS layout from core.config. Returns [] if neither is
    available, in which case _validate_geometry only clamps the size."""
    rects: list = []
    if _HAS_PYQT6:
        try:
            app = QApplication.instance()
            if app is not None:
                for sc in app.screens():
                    r = sc.availableGeometry()
                    rects.append((r.x(), r.y(), r.width(), r.height()))
        except Exception:
            rects = []
    if rects:
        return rects
    # Fallback: the friendly MONITORS layout the rest of JARVIS uses. Imported
    # lazily so this lightweight HUD subprocess doesn't drag in core.config
    # unless it actually needs the fallback.
    try:
        if PROJECT_DIR not in sys.path:
            sys.path.insert(0, PROJECT_DIR)
        from core import config as _cfg  # type: ignore
        mons = getattr(_cfg, "MONITORS", None)
        if isinstance(mons, dict):
            for v in mons.values():
                try:
                    x, y, w, h = v
                    rects.append((int(x), int(y), int(w), int(h)))
                except Exception:
                    continue
    except Exception:
        pass
    return rects


# ─── background data cache (weather / calendar — slow, networked) ────────────
class _SlowData:
    """Weather + forecast + calendar + unread mail, refreshed off the UI thread
    via hud_card's gatherers so the 500 ms paint loop never blocks on the
    network."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.weather: dict | None = None
        self.forecast: list = []
        self.calendar: list = []
        self.unread_mail: int | None = None
        self._stop = threading.Event()
        self._hud_card = self._import_hud_card()

    @staticmethod
    def _import_hud_card():
        if PROJECT_DIR not in sys.path:
            sys.path.insert(0, PROJECT_DIR)
        try:
            import hud_card  # type: ignore
            return hud_card
        except Exception:
            return None

    def start(self) -> None:
        threading.Thread(target=self._loop, name="UnifiedHudSlowData",
                         daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._refresh_once()
            self._stop.wait(SLOW_REFRESH_S)

    def _refresh_once(self) -> None:
        # ONLY weather + the 3-day forecast are fetched here. Both come from
        # wttr.in (a plain HTTP GET) — safe in this lightweight subprocess —
        # and each is STORED THE INSTANT it succeeds.
        #
        # Calendar + unread mail are deliberately NOT gathered here. That path
        # (hud_card._gather_calendar -> import ms_graph) transitively imports
        # the 14k-line bobert_companion, whose early-boot singleton lock calls
        # sys.exit() in a child process. sys.exit raises SystemExit, which is a
        # BaseException — NOT an Exception — so it escaped the old
        # `except Exception` and KILLED this whole thread before the
        # already-fetched weather was ever stored. That's the bug behind
        # WEATHER reading "—" forever. The HUD now reads calendar/mail from
        # hud_state.json (written by the main process, where ms_graph is safe).
        hc = self._hud_card
        if hc is None:
            return
        try:
            w = hc._gather_weather_now()
            if w is not None:
                with self._lock:
                    self.weather = w
        except BaseException:
            pass
        try:
            f = hc._gather_forecast()
            if f is not None:
                with self._lock:
                    self.forecast = f
        except BaseException:
            pass

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "weather": self.weather,
                "forecast": list(self.forecast),
            }


# ─── the HUD widget ──────────────────────────────────────────────────────────
class UnifiedHud(QWidget):
    def __init__(self, parent_pid: int, slow: _SlowData):
        super().__init__()
        self.parent_pid = parent_pid
        self.slow = slow
        self.frame = 0
        self._drag_offset: QPoint | None = None

        # Live (cheap) sample buffers.
        self.state = "idle"
        self._want_visible = True   # driven by hud_state.json "visible" flag
        self._user_hidden = False   # set by the ✕ button (control-file "hidden")
        self.now_doing = ""
        self.now_playing = ""
        self.transcript: list[str] = []
        self.tts_amp = 0.0
        self.mic_level = 0.0
        self.alert_active = False
        # Calendar + mail come from the main process via hud_state.json (the
        # HUD subprocess must not import ms_graph — see _SlowData).
        self.next_event: dict | None = None
        self.unread_mail_count: int | None = None
        self.cpu = 0.0
        self.ram = 0.0
        self.gpu_util: float | None = None
        self.net_mbps = 0.0
        self.bambu_active = False
        self.bambu_pct = 0
        self.bambu_gcode = ""
        self.bambu_eta_min = 0
        self._last_net = None
        self._last_net_at = None
        self._gpu_cached_at = 0.0
        self._gpu_sampling = False   # guards against overlapping sampler threads
        # Live camera preview: cached QPixmap + the mtime it was loaded from, so
        # we only re-decode the JPEG when the writer has produced a new frame.
        self.cam_preview: "QPixmap | None" = None
        self._cam_preview_mtime = 0.0

        if _HAS_PSUTIL:
            try:
                psutil.cpu_percent(interval=None)
                io = psutil.net_io_counters()
                self._last_net = int(io.bytes_recv + io.bytes_sent)
                self._last_net_at = time.time()
            except Exception:
                pass

        # Frameless, translucent, always-on-top, no taskbar entry.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(MIN_W, MIN_H)
        self.setWindowTitle("JARVIS HUD")
        self.setMouseTracking(True)

        # Close button (top-right). Quits the subprocess; voice/tray relaunch.
        self.btn_close = QPushButton("✕", self)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self._on_close)
        self.btn_close.setStyleSheet(
            "QPushButton{color:#9ee7ff;background:rgba(10,28,38,180);"
            "border:1px solid rgba(76,201,255,120);border-radius:11px;"
            "font:bold 12px 'Segoe UI';}"
            "QPushButton:hover{color:#04080d;background:#ff5b5b;border:0;}"
        )

        # Resize grip, bottom-right.
        self.grip = QSizeGrip(self)
        self.grip.setStyleSheet("background: transparent;")

        # Debounced geometry-save timer.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(450)
        self._save_timer.timeout.connect(self._save_geometry)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

        # Dedicated fast tick for JUST the camera-preview tile, so the live
        # mirror is smooth without dragging the whole HUD up to 13 fps. It only
        # re-reads the preview JPEG (decode skipped unless its mtime changed) and
        # repaints when the frame actually changed — the rest of the HUD state
        # stays on the cheap TICK_MS loop above.
        self.cam_timer = QTimer(self)
        self.cam_timer.setInterval(CAMERA_TICK_MS)
        self.cam_timer.timeout.connect(self._on_camera_tick)
        self.cam_timer.start()

        self._reposition_chrome()

    # ── data refresh ────────────────────────────────────────────────────────
    def _read_gpu_util(self) -> float | None:
        """Return the cached GPU utilization, never blocking the caller.

        nvidia-smi can stall for up to its 2 s timeout per spawn, so it must
        not run on the Qt GUI/paint thread. On a cache miss we kick the actual
        sampling onto a short-lived background thread and immediately return the
        previously-cached value; the worker writes the fresh reading back when
        it lands. The displayed value and GPU_CACHE_SECONDS TTL are unchanged —
        _gpu_cached_at is stamped here (sample start) exactly as before so the
        refresh cadence stays the same and overlapping spawns are avoided."""
        now = time.time()
        if (now - self._gpu_cached_at) < GPU_CACHE_SECONDS:
            return self.gpu_util
        # Stamp the sample-start time up front so the next tick won't re-trigger
        # while this sample is in flight (preserves the old TTL behaviour).
        self._gpu_cached_at = now
        if not self._gpu_sampling:
            self._gpu_sampling = True
            threading.Thread(target=self._gpu_sample_worker, daemon=True).start()
        return self.gpu_util

    def _gpu_sample_worker(self) -> None:
        """Background worker: do the blocking nvidia-smi sample and publish the
        result back to the cached attribute the paint loop reads."""
        try:
            self.gpu_util = self._sample_gpu_util()
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
                utils = [int(v.strip()) for v in (out.stdout or "").splitlines()
                         if v.strip().isdigit()]
                if utils:
                    return float(max(utils))
        except Exception:
            pass
        # Fallback: parse "GPU 54%" out of the pulse_strip (digit-scan stops at
        # the first non-digit, so the trailing "%" terminates the number; the
        # producer emits a "GPU 54C" temperature token only when it too had no
        # utilization reading).
        strip = _read_json(HUD_STATE_FILE).get("pulse_strip") or ""
        if "GPU " in strip:
            num = ""
            for ch in strip.split("GPU ", 1)[1]:
                if ch.isdigit() or ch == ".":
                    num += ch
                else:
                    break
            if num:
                try:
                    return float(num)
                except ValueError:
                    pass
        return None

    def _read_net_mbps(self) -> float:
        if not _HAS_PSUTIL:
            return 0.0
        try:
            io = psutil.net_io_counters()
            now = time.time()
            total = int(io.bytes_recv + io.bytes_sent)
            if self._last_net is None:
                self._last_net, self._last_net_at = total, now
                return 0.0
            dt = max(1e-3, now - self._last_net_at)
            db = max(0, total - self._last_net)
            self._last_net, self._last_net_at = total, now
            return (db / dt) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _refresh_camera_preview(self) -> bool:
        """Load the live camera-preview JPEG into self.cam_preview, but only when
        it is fresh AND newer than what we last decoded (so a static frame isn't
        re-decoded every camera tick). A stale/absent file clears the pixmap so
        the corner falls back to the 'camera off' placeholder.

        Returns True iff what the tile shows changed (a new frame was decoded, or
        a previously-shown frame was cleared to the placeholder) — the fast
        camera tick uses this to repaint ONLY when there is something new, so an
        idle/off camera costs a stat() per tick and no repaint. The expensive
        JPEG decode is skipped whenever the file's mtime is unchanged."""
        # Skip ALL work (stat + decode) while the HUD is hidden. Both callers
        # (_on_camera_tick's ~75ms tick and _refresh's ~500ms tick) run regardless
        # of visibility, and the preview writer keeps producing frames based on
        # camera state — not HUD visibility — so a hidden HUD would otherwise keep
        # decoding each new ~6-7fps JPEG only to discard it (the repaint is already
        # visibility-gated). When the HUD is shown again the next tick decodes the
        # current frame. 2026-07-08.
        if not self.isVisible():
            return False
        now = time.time()
        if not _camera_preview_fresh_at(now):
            had = self.cam_preview is not None
            self.cam_preview = None
            self._cam_preview_mtime = 0.0
            return had   # changed only if we were showing a frame before
        try:
            mtime = os.path.getmtime(CAMERA_PREVIEW_FILE)
        except OSError:
            had = self.cam_preview is not None
            self.cam_preview = None
            return had
        if self.cam_preview is not None and mtime <= self._cam_preview_mtime:
            return False   # already showing this frame → skip decode + repaint
        pm = QPixmap(CAMERA_PREVIEW_FILE)
        if pm.isNull():
            # A half-written file (the writer uses atomic replace, so this is
            # rare) — keep the previous frame and retry next tick.
            return False
        self.cam_preview = pm
        self._cam_preview_mtime = mtime
        return True

    def _refresh(self) -> bool:
        if not _is_parent_alive(self.parent_pid) or _control_says_off():
            return False
        hud = _read_json(HUD_STATE_FILE)
        # Honour the existing show_hud / hide_hud / toggle_hud voice commands —
        # they flip this flag in hud_state.json (via _write_hud_state).
        self._want_visible = bool(hud.get("visible", True))
        # ✕-button hide persists in the control file; 'show HUD' clears it.
        self._user_hidden = bool(_read_json(CONTROL_FILE).get("hidden"))
        self.state = (hud.get("state") or "Idle").lower()
        self.now_doing = (hud.get("now_doing") or "").strip()
        self.now_playing = (hud.get("now_playing") or "").strip()
        th = hud.get("transcript_history")
        if isinstance(th, list):
            self.transcript = [str(t) for t in th][-4:]
        elif hud.get("last_transcript"):
            self.transcript = [str(hud.get("last_transcript"))]
        for attr, key in (("tts_amp", "tts_amplitude"), ("mic_level", "mic_level")):
            try:
                setattr(self, attr, float(hud.get(key) or 0.0))
            except (TypeError, ValueError):
                setattr(self, attr, 0.0)
        self.alert_active = bool(hud.get("alert_active"))
        ne = hud.get("next_event")
        self.next_event = ne if isinstance(ne, dict) else None
        try:
            um = hud.get("unread_mail")
            self.unread_mail_count = int(um) if um is not None else None
        except (TypeError, ValueError):
            self.unread_mail_count = None

        if _HAS_PSUTIL:
            try:
                self.cpu = float(psutil.cpu_percent(interval=None))
                self.ram = float(psutil.virtual_memory().percent)
            except Exception:
                pass
        self.net_mbps = self._read_net_mbps()
        self.gpu_util = self._read_gpu_util()

        bambu = _read_json(BAMBU_STATE_FILE)
        gs = (bambu.get("gcode_state") or "").upper()
        self.bambu_gcode = gs
        self.bambu_active = gs in ("RUNNING", "PAUSE", "PREPARE")
        for attr, key in (("bambu_pct", "mc_percent"),
                          ("bambu_eta_min", "mc_remaining")):
            try:
                setattr(self, attr, int(bambu.get(key) or 0))
            except (TypeError, ValueError):
                setattr(self, attr, 0)
        self._refresh_camera_preview()
        self.frame += 1
        return True

    def _on_tick(self) -> None:
        if not self._refresh():
            self.timer.stop()
            app = QApplication.instance()
            if app:
                app.quit()
            return
        # Apply visibility — hidden by EITHER the voice show/hide flag OR the
        # ✕ button. The timer keeps ticking while hidden so 'show HUD' (which
        # clears both) brings it right back.
        want = self._want_visible and not self._user_hidden
        if want and not self.isVisible():
            self.show()
        elif not want and self.isVisible():
            self.hide()
            return
        if self.isVisible():
            self.update()

    def _on_camera_tick(self) -> None:
        """Fast (~CAMERA_TICK_MS) tick for the camera-preview tile only. Re-reads
        the preview JPEG (the decode is skipped unless its mtime changed) and
        repaints ONLY when the displayed frame actually changed, so a live camera
        animates smoothly while an idle/off one costs a single stat() per tick and
        no repaint. Visibility is owned by _on_tick; we never show/hide here, and
        we skip the repaint entirely while the HUD is hidden."""
        changed = self._refresh_camera_preview()
        if changed and self.isVisible():
            self.update()

    # ── colour helpers ───────────────────────────────────────────────────────
    def _accent(self) -> QColor:
        return {
            "listening": GREEN,
            "thinking":  GOLD,
            "speaking":  CYAN_BRIGHT,
            "standby":   CYAN_DIM,
            "sleep":     CYAN_DIM,
        }.get(self.state, CYAN)

    @staticmethod
    def _metric_color(v: float, warn: float, crit: float) -> QColor:
        if v >= crit:
            return RED
        if v >= warn:
            return AMBER
        return CYAN

    # ── interaction: drag, geometry persistence ──────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (e.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())
            e.accept()

    def mouseMoveEvent(self, e) -> None:
        if (self._drag_offset is not None
                and e.buttons() & Qt.MouseButton.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e) -> None:
        self._drag_offset = None
        self._save_timer.start()

    def moveEvent(self, e) -> None:
        super().moveEvent(e)
        self._save_timer.start()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._reposition_chrome()
        self._save_timer.start()

    def _reposition_chrome(self) -> None:
        w, h = self.width(), self.height()
        self.btn_close.setGeometry(w - 30, 10, 22, 22)
        self.grip.setGeometry(w - 18, h - 18, 16, 16)

    def _save_geometry(self) -> None:
        try:
            g = self.geometry()
            tmp = GEOMETRY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"x": g.x(), "y": g.y(),
                           "w": g.width(), "h": g.height()}, f)
            os.replace(tmp, GEOMETRY_FILE)
        except Exception:
            pass

    def _write_control(self, **updates) -> None:
        try:
            ctrl = _read_json(CONTROL_FILE)
            ctrl.update(updates)
            # Unique per-write temp: core/actions.py:_set_unified_hud_hidden
            # writes this SAME control file from the main process. A fixed shared
            # ".tmp" name let the two processes truncate each other's half-
            # written temp and race the os.replace, corrupting the JSON. A
            # mkstemp temp keeps each writer's file whole (last replace wins).
            import tempfile
            _dir = os.path.dirname(os.path.abspath(CONTROL_FILE)) or "."
            fd, tmp = tempfile.mkstemp(dir=_dir, prefix=".uhud_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(ctrl, f)
                os.replace(tmp, CONTROL_FILE)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                raise
        except Exception:
            pass

    def _on_close(self) -> None:
        # HIDE, do not quit. Quitting would orphan the HUD with no way back
        # short of restarting JARVIS. Persist the hidden state so the
        # visibility sync keeps it down; saying 'show HUD' (which clears the
        # flag) or using the tray brings it right back.
        self._user_hidden = True
        self._write_control(hidden=True)
        self.hide()

    # ── painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        W, H = float(self.width()), float(self.height())
        s = max(0.78, min(1.7, W / 420.0))       # global scale factor
        pad = 14.0 * s

        # 1. Panel backdrop — vertical gradient + cyan rim + soft inner glow.
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, PANEL_TOP)
        grad.setColorAt(1.0, PANEL_BOT)
        path = QPainterPath()
        path.addRoundedRect(QRectF(1, 1, W - 2, H - 2), 16 * s, 16 * s)
        p.fillPath(path, QBrush(grad))
        p.setPen(QPen(PANEL_RIM, 1.4))
        p.drawPath(path)

        accent = self._accent()

        # 2. Title strip.
        title_h = 34.0 * s
        f = QFont("Segoe UI", 1)
        f.setPixelSize(int(17 * s))
        f.setBold(True)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0 * s)
        p.setFont(f)
        p.setPen(QPen(CYAN_BRIGHT))
        p.drawText(QRectF(pad, 8 * s, W - 2 * pad, title_h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   "J.A.R.V.I.S.")
        # State word, right-aligned (leaves room for the ✕ button).
        f2 = QFont("Consolas", 1)
        f2.setPixelSize(int(11 * s))
        f2.setBold(True)
        p.setFont(f2)
        p.setPen(QPen(accent))
        p.drawText(QRectF(pad, 8 * s, W - 2 * pad - 34 * s, title_h),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                   self.state.upper())
        # Divider.
        p.setPen(QPen(PANEL_RIM, 1))
        p.drawLine(QPointF(pad, title_h + 6 * s), QPointF(W - pad, title_h + 6 * s))

        # 3. Reactor disc (core + four metric arcs).
        reactor_top = title_h + 12 * s
        reactor_size = min(W - 2 * pad, H * 0.40)
        cx = W / 2.0
        cy = reactor_top + reactor_size / 2.0
        self._draw_reactor(p, cx, cy, reactor_size * 0.42, accent, s)

        # 3b. Live camera preview — picture-in-picture in the top-right corner,
        # in the empty space beside the circular reactor, below the ✕ button.
        self._draw_camera_preview(p, W, pad, title_h, reactor_top, reactor_size, s)

        y = reactor_top + reactor_size + 10 * s

        # 4. Vitals line (numbers, colour-coded to their arcs).
        y = self._draw_vitals(p, pad, y, W, s)

        # 5. Info rows.
        snap = self.slow.snapshot()
        y = self._draw_now_playing(p, pad, y, W, s)
        y = self._draw_weather(p, pad, y, W, s, snap)
        y = self._draw_calendar(p, pad, y, W, s)
        if self.bambu_active:
            y = self._draw_bambu(p, pad, y, W, s)
        y = self._draw_notifications(p, pad, y, W, s)

        # 6. Transcript panel (fills remaining space down to the grip).
        self._draw_transcript(p, pad, y, W, H, s)
        p.end()

    # ── live camera preview (picture-in-picture) ─────────────────────────────
    def _draw_camera_preview(self, p, W, pad, title_h, reactor_top,
                             reactor_size, s) -> None:
        """Draw the small live camera preview (or an 'off' placeholder) pinned to
        the top-right corner, in the empty space beside the round reactor. The
        frame itself is supplied by self.cam_preview (a QPixmap or None); None →
        the camera is off / paused / the file went stale, so we show the
        placeholder. The frame is scaled to COVER the tile (aspect kept, centre-
        cropped) so there are no letterbox bars and it never looks stretched."""
        # Tile geometry: a 4:3-ish tile tucked under the ✕ button, right-aligned.
        tile_w = max(72.0, min(118.0 * s, (W - 2 * pad) * 0.30))
        tile_h = tile_w * 0.72
        tx = W - pad - tile_w
        ty = title_h + 14 * s            # just below the title divider
        rect = QRectF(tx, ty, tile_w, tile_h)
        rad = 6.0 * s

        # Backdrop + cyan rim (always drawn, frame or placeholder).
        clip = QPainterPath()
        clip.addRoundedRect(rect, rad, rad)
        p.fillPath(clip, QBrush(QColor(4, 10, 16, 235)))

        live = self.cam_preview is not None and not self.cam_preview.isNull()
        if live:
            # Scale to COVER the tile (fill, center-crop) so there are no bars,
            # then clip to the rounded rect.
            pm = self.cam_preview.scaled(
                int(tile_w), int(tile_h),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            ox = tx - max(0.0, (pm.width() - tile_w) / 2.0)
            oy = ty - max(0.0, (pm.height() - tile_h) / 2.0)
            p.save()
            p.setClipPath(clip)
            p.drawPixmap(QPointF(ox, oy), pm)
            p.restore()

        # Rim.
        p.setPen(QPen(PANEL_RIM, 1.4)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, rad, rad)

        if live:
            # "● LIVE" badge, top-left of the tile.
            fb = QFont("Consolas", 1); fb.setPixelSize(int(8 * s)); fb.setBold(True)
            p.setFont(fb)
            p.setPen(QPen(RED))
            p.drawText(QRectF(tx + 4 * s, ty + 2 * s, tile_w, 11 * s),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                       "● LIVE")
        else:
            # Placeholder: dim camera glyph + label, centered.
            p.setPen(QPen(DIM_FG))
            fg = QFont("Segoe UI", 1); fg.setPixelSize(int(15 * s))
            p.setFont(fg)
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "▣")
            fl = QFont("Consolas", 1); fl.setPixelSize(int(7.5 * s)); fl.setBold(True)
            p.setFont(fl)
            p.drawText(QRectF(tx, ty + tile_h - 13 * s, tile_w, 11 * s),
                       int(Qt.AlignmentFlag.AlignCenter), "CAMERA OFF")

    # ── reactor ───────────────────────────────────────────────────────────────
    def _draw_reactor(self, p, cx, cy, R, accent, s) -> None:
        # Glow halo.
        glow = QRadialGradient(QPointF(cx, cy), R * 1.5)
        g0 = QColor(accent); g0.setAlpha(0)
        g1 = QColor(accent); g1.setAlpha(120)
        g2 = QColor(accent); g2.setAlpha(0)
        glow.setColorAt(0.45, g0)
        glow.setColorAt(0.82, g1)
        glow.setColorAt(1.0, g2)
        p.setBrush(QBrush(glow)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), R * 1.5, R * 1.5)

        outer = QRectF(cx - R, cy - R, 2 * R, 2 * R)
        p.setPen(QPen(CYAN_DIM, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(outer)

        # Four metric arcs. Qt: 0°=3 o'clock, CCW+, angle in 1/16°.
        gpu_v = self.gpu_util if self.gpu_util is not None else 0.0
        gpu_frac = max(0.0, min(1.0, gpu_v / 100.0))   # 0-100% utilization → flat 0-1
        quads = [
            (min(1.0, self.cpu / 100.0), 90,
             self._metric_color(self.cpu, CPU_WARN, CPU_CRIT)),
            (min(1.0, self.ram / 100.0), 0,
             self._metric_color(self.ram, RAM_WARN, RAM_CRIT)),
            (gpu_frac, 270,
             self._metric_color(gpu_v, GPU_WARN, GPU_CRIT)
             if self.gpu_util is not None else CYAN_DIM),
            (min(1.0, self.net_mbps / NET_FULL_MBPS), 180,
             self._metric_color(self.net_mbps, NET_WARN, NET_CRIT)),
        ]
        gap = 4.0
        span = 90.0 - 2 * gap
        aw = max(5.0, R * 0.11)
        for frac, anchor, col in quads:
            start16 = int((anchor - gap) * 16)
            faint = QPen(CYAN_DIM, aw * 0.5); faint.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(faint); p.drawArc(outer, start16, -int(span * 16))
            fp = QPen(col, aw); fp.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(fp); p.drawArc(outer, start16, -int(span * frac * 16))

        # Inner Bambu ring.
        if self.bambu_active:
            rb = R * 0.78
            br = QRectF(cx - rb, cy - rb, 2 * rb, 2 * rb)
            p.setPen(QPen(CYAN_DIM, 1.5)); p.drawArc(br, 0, 360 * 16)
            col = AMBER if self.bambu_gcode in ("PAUSE", "PREPARE") else GREEN
            bp = QPen(col, max(3.0, R * 0.06)); bp.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(bp)
            p.drawArc(br, 90 * 16, -int(360 * 16 * min(1.0, self.bambu_pct / 100.0)))

        # Rotating tick decoration.
        spin = (self.frame * 0.05) % (2 * math.pi)
        tr = R * 0.60
        p.setPen(QPen(CYAN_DIM, 1))
        for i in range(24):
            th = (i / 24) * 2 * math.pi + spin
            p.drawLine(QPointF(cx + tr * math.cos(th), cy + tr * math.sin(th)),
                       QPointF(cx + (tr + 4) * math.cos(th), cy + (tr + 4) * math.sin(th)))

        # Core + pulsing hub.
        pulse = 0.5 * (1 + math.sin(self.frame * 0.2))
        rc = R * 0.52
        p.setPen(QPen(accent, 2)); p.setBrush(QBrush(PANEL_BOT))
        p.drawEllipse(QRectF(cx - rc, cy - rc, 2 * rc, 2 * rc))
        if self.state == "speaking":
            bright = 0.5 + 0.5 * max(0.0, min(1.0, self.tts_amp))
        elif self.state == "listening":
            bright = 0.4 + 0.6 * max(0.0, min(1.0, self.mic_level))
        else:
            bright = 0.5 + 0.4 * pulse
        hub = QColor(accent); hub.setAlpha(int(180 * bright + 60))
        p.setBrush(QBrush(hub)); p.setPen(Qt.PenStyle.NoPen)
        rh = R * 0.26 * (0.9 + 0.15 * pulse)
        p.drawEllipse(QPointF(cx, cy), rh, rh)

        # Centre state label.
        f = QFont("Consolas", 1); f.setPixelSize(int(R * 0.22)); f.setBold(True)
        p.setFont(f); p.setPen(QPen(TEXT_FG))
        label = self.state.upper()
        if self.bambu_active:
            label = f"{self.bambu_pct}%"
        p.drawText(QRectF(cx - R, cy - R * 0.3, 2 * R, R * 0.6),
                   int(Qt.AlignmentFlag.AlignCenter), label[:9])

    # ── info rows ─────────────────────────────────────────────────────────────
    def _label(self, p, x, y, w, s, tag, value, col=None, value_col=None):
        """Draw a `TAG   value` row; returns the new y."""
        rh = 21.0 * s
        ft = QFont("Consolas", 1); ft.setPixelSize(int(10 * s)); ft.setBold(True)
        p.setFont(ft); p.setPen(QPen(col or DIM_FG))
        p.drawText(QRectF(x, y, 64 * s, rh),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), tag)
        fv = QFont("Segoe UI", 1); fv.setPixelSize(int(12 * s))
        p.setFont(fv); p.setPen(QPen(value_col or TEXT_FG))
        p.drawText(QRectF(x + 66 * s, y, w - x - 66 * s - 14 * s, rh),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   value)
        return y + rh

    def _draw_vitals(self, p, pad, y, W, s) -> float:
        rh = 22.0 * s
        cells = [
            ("CPU", f"{self.cpu:.0f}%", self._metric_color(self.cpu, CPU_WARN, CPU_CRIT)),
            ("RAM", f"{self.ram:.0f}%", self._metric_color(self.ram, RAM_WARN, RAM_CRIT)),
            ("GPU", (f"{self.gpu_util:.0f}%" if self.gpu_util is not None else "—"),
             self._metric_color(self.gpu_util or 0, GPU_WARN, GPU_CRIT)
             if self.gpu_util is not None else DIM_FG),
            ("NET", (f"{self.net_mbps:.1f}" if self.net_mbps >= 10 else f"{self.net_mbps:.2f}"),
             self._metric_color(self.net_mbps, NET_WARN, NET_CRIT)),
        ]
        cw = (W - 2 * pad) / 4.0
        ft = QFont("Consolas", 1); ft.setPixelSize(int(9 * s)); ft.setBold(True)
        fv = QFont("Consolas", 1); fv.setPixelSize(int(13 * s)); fv.setBold(True)
        for i, (tag, val, col) in enumerate(cells):
            cellx = pad + i * cw
            p.setFont(ft); p.setPen(QPen(DIM_FG))
            p.drawText(QRectF(cellx, y, cw, 12 * s),
                       int(Qt.AlignmentFlag.AlignCenter), tag)
            p.setFont(fv); p.setPen(QPen(col))
            p.drawText(QRectF(cellx, y + 10 * s, cw, 14 * s),
                       int(Qt.AlignmentFlag.AlignCenter), val)
        return y + rh + 8 * s

    def _draw_now_playing(self, p, pad, y, W, s, snap=None) -> float:
        track = self.now_playing or "—"
        return self._label(p, pad, y, W, s, "♪ NOW", track,
                            col=CYAN, value_col=TEXT_FG if self.now_playing else DIM_FG)

    def _draw_weather(self, p, pad, y, W, s, snap) -> float:
        w = snap.get("weather")
        if w:
            # Weather temp is stored in Celsius; sir reads it in Fahrenheit.
            try:
                _tf = f"{int(round(float(w.get('temp_c')) * 9 / 5 + 32))}°F"
            except (TypeError, ValueError):
                _tf = "?°F"
            txt = f"{w.get('emoji','')} {_tf}  {(w.get('desc') or '').title()}"
        else:
            txt = "—"
        y = self._label(p, pad, y, W, s, "WEATHER", txt, col=CYAN,
                        value_col=TEXT_FG if w else DIM_FG)
        fc = snap.get("forecast") or []
        if fc:
            def _toF(v):
                # Forecast highs/lows are stored in Celsius; show Fahrenheit.
                try:
                    return str(int(round(float(v) * 9 / 5 + 32)))
                except (TypeError, ValueError):
                    return "?"
            parts = [f"{d.get('emoji','')}{d.get('label','')[:3]} "
                     f"{_toF(d.get('high_c'))}/{_toF(d.get('low_c'))}" for d in fc[:3]]
            fv = QFont("Segoe UI", 1); fv.setPixelSize(int(10 * s))
            p.setFont(fv); p.setPen(QPen(DIM_FG))
            p.drawText(QRectF(pad + 66 * s, y, W - pad - 66 * s - 14 * s, 18 * s),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                       "   ".join(parts))
            y += 18 * s
        return y

    def _draw_calendar(self, p, pad, y, W, s) -> float:
        c = self.next_event
        if c:
            txt = f"{c.get('time','')}  {(c.get('subject') or '')[:34]}"
        else:
            txt = "Nothing upcoming"
        return self._label(p, pad, y, W, s, "NEXT", txt, col=GOLD,
                           value_col=TEXT_FG if c else DIM_FG)

    def _draw_bambu(self, p, pad, y, W, s) -> float:
        rh = 30.0 * s
        col = AMBER if self.bambu_gcode in ("PAUSE", "PREPARE") else GREEN
        ft = QFont("Consolas", 1); ft.setPixelSize(int(10 * s)); ft.setBold(True)
        p.setFont(ft); p.setPen(QPen(DIM_FG))
        p.drawText(QRectF(pad, y, 64 * s, 18 * s),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), "PRINT")
        # Progress bar.
        bx = pad + 66 * s
        bw = W - bx - 14 * s
        bar_h = 9 * s
        by = y + 5 * s
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(CYAN_DIM))
        p.drawRoundedRect(QRectF(bx, by, bw, bar_h), bar_h / 2, bar_h / 2)
        p.setBrush(QBrush(col))
        p.drawRoundedRect(QRectF(bx, by, bw * min(1.0, self.bambu_pct / 100.0), bar_h),
                          bar_h / 2, bar_h / 2)
        fv = QFont("Consolas", 1); fv.setPixelSize(int(10 * s)); fv.setBold(True)
        p.setFont(fv); p.setPen(QPen(col))
        eta = ""
        if self.bambu_eta_min > 0:
            h, m = divmod(self.bambu_eta_min, 60)
            eta = f"  ·  {h}h {m}m left" if h else f"  ·  {m}m left"
        p.drawText(QRectF(bx, by + bar_h + 1 * s, bw, 14 * s),
                   int(Qt.AlignmentFlag.AlignLeft),
                   f"{self.bambu_gcode.title()}  {self.bambu_pct}%{eta}")
        return y + rh

    def _draw_notifications(self, p, pad, y, W, s) -> float:
        unread = self.unread_mail_count
        bits = []
        if isinstance(unread, int) and unread > 0:
            bits.append(f"{unread} unread email{'s' if unread != 1 else ''}")
        if self.alert_active:
            bits.append("⚠ alert active")
        txt = "  ·  ".join(bits) if bits else "All clear"
        col = GOLD if bits else GREEN
        return self._label(p, pad, y, W, s, "INBOX", txt, col=DIM_FG, value_col=col)

    def _draw_transcript(self, p, pad, y, W, H, s) -> float:
        top = y + 4 * s
        bottom = H - 16 * s
        if bottom - top < 28 * s:
            return y
        p.setPen(QPen(PANEL_RIM, 1))
        p.drawLine(QPointF(pad, top), QPointF(W - pad, top))
        ft = QFont("Consolas", 1); ft.setPixelSize(int(9 * s)); ft.setBold(True)
        p.setFont(ft); p.setPen(QPen(DIM_FG))
        p.drawText(QRectF(pad, top + 3 * s, W - 2 * pad, 14 * s),
                   int(Qt.AlignmentFlag.AlignLeft), "HEARD")
        fv = QFont("Segoe UI", 1); fv.setPixelSize(int(11 * s))
        p.setFont(fv)
        ty = top + 18 * s
        for line in self.transcript[-3:]:
            if ty > bottom - 14 * s:
                break
            p.setPen(QPen(TEXT_FG))
            p.drawText(QRectF(pad + 4 * s, ty, W - 2 * pad - 4 * s, 16 * s),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                       f"“{line[:46]}”")
            ty += 16 * s
        # Now-doing footer.
        if self.now_doing and ty <= bottom - 14 * s:
            p.setPen(QPen(self._accent()))
            fd = QFont("Consolas", 1); fd.setPixelSize(int(9 * s)); fd.setBold(True)
            p.setFont(fd)
            p.drawText(QRectF(pad, bottom - 14 * s, W - 2 * pad, 14 * s),
                       int(Qt.AlignmentFlag.AlignLeft),
                       f"▸ {self.now_doing[:50]}")
        return bottom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, default=2280)
    ap.add_argument("--y", type=int, default=-1400)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--parent-pid", type=int, default=0)
    # The shared launcher in bobert_companion also passes --role / --state-file
    # (blue-green plumbing) and may omit --height. parse_known_args() lets us
    # ignore anything we don't use rather than erroring out of existence.
    args, _unknown = ap.parse_known_args()

    if not _HAS_PYQT6:
        print("[unified_hud] PyQt6 is required:  pip install PyQt6",
              file=sys.stderr)
        return 2

    # Clamp the launcher-provided default so a stray full-monitor width can
    # never produce a monster bar.
    default_x, default_y = args.x, args.y
    w = min(max(MIN_W, args.width), MAX_W)
    h = min(max(MIN_H, args.height), MAX_H)
    x, y = default_x, default_y

    app = QApplication(sys.argv[:1])

    # Saved geometry (from a prior drag/resize) overrides the CLI default, but
    # must be validated FIRST: clamp its size and, if a monitor-layout change
    # left it stranded off every screen, snap it back to the CLI anchor so a
    # stale unified_hud_geometry.json can't hide the frameless HUD off-screen.
    # Done after the QApplication exists so _available_screen_rects sees the
    # real Qt screens (it falls back to MONITORS otherwise).
    geo = _load_saved_geometry()
    if geo:
        valid = _validate_geometry(geo, _available_screen_rects(),
                                   (default_x, default_y))
        x, y, w, h = valid["x"], valid["y"], valid["w"], valid["h"]
    slow = _SlowData()
    slow.start()
    hud = UnifiedHud(args.parent_pid, slow)
    hud.setGeometry(x, y, w, h)
    hud.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
