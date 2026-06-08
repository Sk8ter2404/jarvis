"""
kinect_air_mouse skill — a Kinect v2 "air-mouse" for JARVIS.

THE FEATURE
===========
Point an OPEN hand at the screen to move the cursor; CLOSE the hand to
RIGHT-click; hold it closed to drag. Concretely:

  • OPEN hand          → move the cursor (no button held).
  • close → open fast  → a RIGHT-click (button down then up, cursor parked).
  • close, move, open  → a RIGHT-drag (button stays down while the hand is
                         closed, releases on re-open).

A glowing JARVIS targeting reticle (hud/jarvis_air_cursor.py, a separate
click-THROUGH overlay process) follows the cursor — cyan + gently pulsing while
TRACKING an open hand, snapping inward to a GOLD lock on grab/drag — so the
owner always sees where their hand is pointing.

This module is the LIVE WIRING; the testable core is pure and lives right here
alongside it (no sensor, no real mouse, no Qt needed to exercise it):

  • ReachBox + map_hand_to_cursor() — turn a hand position (camera-space metres)
    into an absolute VIRTUAL-DESKTOP pixel (spanning ALL monitors), clamped to
    the desktop bounds.
  • EMA — exponential smoothing to fight the Kinect's hand-joint jitter.
  • GripDebounter — the open/closed state machine: requires N consecutive frames
    of a new grip before it flips, so a single flickered frame never fires a
    stray right-click.
  • AirMouseController — ties those together into a per-frame decision:
    (cursor_xy, button: "down"|"up"|None, overlay_state: "track"|"grab"|"hidden").

V1 MAPPING (deliberately simple + robust)
=========================================
The pointing hand's (x, y) is mapped from a CALIBRATED comfortable reach-box in
front of the user onto the ENTIRE virtual desktop (every monitor, including any
left of / above the primary, which have a negative virtual-screen origin). This
is robust and needs no calibration ritual — it just maps "hand left↔right /
up↔down within arm's reach" to "cursor left↔right / up↔down across all screens",
NON-mirrored (hand right → cursor right). It is NOT ray-projection.

  v2 (deferred, noted here so it isn't lost): project the actual arm RAY
  (shoulder→hand, via audio/kinect_pointing.arm_direction) onto each monitor's
  screen plane for true "point AT the pixel" aiming across the whole multi-
  monitor virtual desktop, plus per-user reach-box calibration and fine-tuning
  of the smoothing / dead-zone. v1 ships the simple mapping so it's usable today.

EVERYTHING is opt-in + safe (mirrors skills/kinect_gestures.py):
  • Gated by core.config.KINECT_AIR_MOUSE_ENABLED (default False), RE-READ each
    tick so a Settings toggle takes effect with no restart.
  • A staging / test instance NEVER moves the mouse (JARVIS_STAGING /
    bobert_companion._is_staging()) — the poll loop self-gates every tick.
  • All sensor contact is via audio/kinect_bridge (accessors never raise); a
    missing / disabled sensor degrades to a quiet no-op.
  • DEAD-MAN: the instant the hand isn't tracked, any held button is RELEASED and
    cursor motion stops — a closed hand that leaves the frame can never strand
    the right button down.

LEFT vs RIGHT click: this is a ONE-LINE swap — change AIR_MOUSE_BUTTON below
from "right" to "left". v1 uses RIGHT per the owner's spec.

Voice actions:
  air_mouse_on / air_mouse_off — toggle KINECT_AIR_MOUSE_ENABLED live, persisted
                                 via the same Settings writer kinect_gestures /
                                 model_picker use.
  air_mouse_status             — is the air-mouse on + can I see your hand.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional


# ─── tunables ────────────────────────────────────────────────────────────
AIR_MOUSE_POLL_HZ = 30.0                      # cursor update rate (~30 Hz)
AIR_MOUSE_POLL_INTERVAL = 1.0 / AIR_MOUSE_POLL_HZ
INITIAL_DELAY_SECONDS = 6.0                   # let the monolith + bridge come up
_THREAD_NAME = "kinect-air-mouse-skill"

# Which mouse button the closed-hand grab actuates. ONE-LINE swap to "left" for
# a left-click air-mouse; v1 ships RIGHT per the owner's spec.
AIR_MOUSE_BUTTON = "right"

# EMA smoothing factor for the cursor (0..1). LOWER = smoother but laggier;
# HIGHER = snappier but jitterier. 0.35 is a comfortable middle that tames the
# Kinect hand-joint jitter without feeling like the cursor is dragging through
# molasses. Tunable; v2 may auto-adapt it to hand speed.
AIR_MOUSE_EMA_ALPHA = 0.35

# How many CONSECUTIVE frames a new grip (open↔closed) must persist before the
# state machine accepts it. At ~30 Hz, 3 frames ≈ 100 ms — long enough that a
# single flickered Kinect hand-state frame can't fire a stray right-click, short
# enough that an intentional close still feels instant.
AIR_MOUSE_GRIP_DEBOUNCE_FRAMES = 3

# The comfortable reach-box in front of the user, in camera-space METRES, that
# maps onto the whole virtual desktop. Centred roughly on where a seated user's
# hand naturally sits when pointing at the screen. x: sensor-RIGHT is +; the box
# is wider than tall to match a 16:9 screen. y: sensor-UP is +; centred near
# shoulder height. These are the v1 defaults; v2 makes them per-user calibrated.
#   half-width  → ±X metres from centre maps to the desktop's left/right edges
#   half-height → ±Y metres from centre maps to the desktop's top/bottom edges
REACH_CENTER_X = 0.0      # metres (centred on the sensor's optical axis)
REACH_CENTER_Y = 0.30     # metres above the sensor (≈ seated shoulder height)
REACH_HALF_W = 0.35       # ±0.35 m horizontal reach → full desktop width
REACH_HALF_H = 0.22       # ±0.22 m vertical reach → full desktop height

# Default geometry used only as a fallback when the real virtual-desktop bounds
# can't be read (headless / win32 absent). The live bounds are resolved at
# runtime by _virtual_screen_bounds().
_DEFAULT_SCREEN_W = 2560
_DEFAULT_SCREEN_H = 1440

# How often the live poll loop re-reads the virtual-desktop bounds, so that
# hot-plugging a monitor / changing the display layout is picked up without a
# restart (the metrics are otherwise cached so we don't hit win32 every tick).
VIRTUAL_BOUNDS_REFRESH_SECONDS = 5.0

# Overlay state-file (sibling to bobert_companion.py — same convention the
# reticle / holo-HUD use). The poller writes the live cursor + grip; the overlay
# process reads it each tick to draw the reticle.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIR_CURSOR_STATE_FILE = os.path.join(PROJECT_DIR, "air_cursor_state.json")


# ══════════════════════════════════════════════════════════════════════════
#  PURE CORE (no sensor, no mouse, no Qt — unit-tested directly)
# ══════════════════════════════════════════════════════════════════════════

class ReachBox:
    """The comfortable reach-box → VIRTUAL-DESKTOP mapping.

    Maps a hand (x, y) in camera-space metres onto an absolute virtual-desktop
    pixel that spans EVERY monitor. The desktop is described by its top-left
    origin (origin_x, origin_y) and its (width, height); the origin is NEGATIVE
    for monitors arranged left-of / above the primary, so a fully left monitor is
    reachable too (SetCursorPos accepts these virtual coordinates directly). A
    hand at the box centre lands at the desktop centre, the box edges land at the
    desktop edges, and anything beyond is CLAMPED to the desktop bounds (so a hand
    that overshoots the box parks the cursor at the edge rather than flying off).

    X is NON-mirrored: the Kinect color/body image is itself mirror-flipped
    relative to the user, so the user's hand moving to THEIR right reads as +x and
    we map +x straight to a larger cursor x — hand right → cursor right, hand left
    → cursor left, natural and un-mirrored. y increases UP in camera space but
    screen y increases DOWN, so y is inverted.

    Back-compat: the 2-positional form ``ReachBox(width, height)`` keeps the old
    primary-only behaviour with a (0, 0) origin; pass origin_x / origin_y to span
    the whole virtual desktop."""

    def __init__(self, width: int, height: int,
                 origin_x: int = 0, origin_y: int = 0,
                 center_x: float = REACH_CENTER_X,
                 center_y: float = REACH_CENTER_Y,
                 half_w: float = REACH_HALF_W,
                 half_h: float = REACH_HALF_H):
        # Kept named screen_w / screen_h for back-compat with existing callers;
        # these are the virtual-desktop extents (all monitors), not just primary.
        self.screen_w = int(width)
        self.screen_h = int(height)
        self.origin_x = int(origin_x)
        self.origin_y = int(origin_y)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        # Guard against a zero/negative half-extent (divide-by-zero); floor it.
        self.half_w = max(1e-3, float(half_w))
        self.half_h = max(1e-3, float(half_h))

    def map(self, hand_x: float, hand_y: float) -> tuple[int, int]:
        """(hand_x, hand_y) metres → (px, py) absolute VIRTUAL-DESKTOP pixel,
        clamped to the desktop bounds (origin .. origin+extent-1)."""
        # Normalise to -1..+1 within the box.
        nx = (float(hand_x) - self.center_x) / self.half_w
        ny = (float(hand_y) - self.center_y) / self.half_h
        # X is NON-mirrored (the camera image is already mirror-flipped, so +x =
        # hand-right = cursor-right). Invert y (camera-up → screen-down).
        ny = -ny
        # -1..+1 → 0..1 → absolute virtual-desktop pixel (origin + offset).
        fx = (nx + 1.0) * 0.5
        fy = (ny + 1.0) * 0.5
        px = self.origin_x + int(round(fx * (self.screen_w - 1)))
        py = self.origin_y + int(round(fy * (self.screen_h - 1)))
        # Clamp to the desktop so an overshoot parks at the edge.
        px = max(self.origin_x, min(self.origin_x + self.screen_w - 1, px))
        py = max(self.origin_y, min(self.origin_y + self.screen_h - 1, py))
        return px, py


class EMA:
    """Exponential moving average for a single channel. Heavily smooths the
    jittery Kinect hand position. seed() / reset() so a fresh hand (after the
    hand left the frame) snaps to the new position instead of sweeping the
    cursor across the screen from the stale last value."""

    def __init__(self, alpha: float = AIR_MOUSE_EMA_ALPHA):
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self._value: Optional[float] = None

    def reset(self) -> None:
        self._value = None

    def update(self, x: float) -> float:
        x = float(x)
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


class GripDebouncer:
    """Debounce the OPEN↔CLOSED hand transition.

    Feed the raw grip string each frame ("open" / "closed" / "lasso" /
    "unknown"); the *stable* grip only changes after the new grip has been seen
    for `frames` consecutive ticks. "unknown"/"lasso" never flip the stable
    state (they're treated as "no new evidence") so a momentary tracking dropout
    holds the last good grip rather than spuriously releasing a drag — the
    dead-man (hand UNTRACKED) is what releases, not a single ambiguous frame.

    `stable` starts at "open" so the first real close is a clean down-edge."""

    def __init__(self, frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES,
                 initial: str = "open"):
        self.frames = max(1, int(frames))
        self._stable = initial
        self._candidate: Optional[str] = None
        self._count = 0

    @property
    def stable(self) -> str:
        return self._stable

    def reset(self, initial: str = "open") -> None:
        self._stable = initial
        self._candidate = None
        self._count = 0

    def update(self, raw: str) -> str:
        """Feed a raw grip; return the (possibly unchanged) stable grip."""
        raw = (raw or "unknown").lower()
        # Only OPEN/CLOSED carry a vote. Ambiguous frames hold the current
        # stable grip and reset any in-flight candidate streak.
        if raw not in ("open", "closed"):
            self._candidate = None
            self._count = 0
            return self._stable
        if raw == self._stable:
            # Already stable here — clear any partial streak toward the other.
            self._candidate = None
            self._count = 0
            return self._stable
        # raw differs from stable: build/extend the candidate streak.
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1
        if self._count >= self.frames:
            self._stable = raw
            self._candidate = None
            self._count = 0
        return self._stable


# Per-frame decision returned by AirMouseController.update().
class AirMouseDecision:
    """What the live loop should DO this frame.

      cursor:  (px, py) | None   — where to put the cursor (None = don't move)
      button:  "down" | "up" | None — actuate the grab button (edge only; None
               means no change this frame)
      overlay: "track" | "grab" | "hidden" — the reticle state to publish
               (cyan-track / gold-grab / hidden)
      grip:    the debounced stable grip ("open"/"closed") — for diagnostics
    """
    __slots__ = ("cursor", "button", "overlay", "grip")

    def __init__(self, cursor, button, overlay, grip):
        self.cursor = cursor
        self.button = button
        self.overlay = overlay
        self.grip = grip

    def __repr__(self):   # pragma: no cover - debug aid
        return (f"AirMouseDecision(cursor={self.cursor}, button={self.button!r}, "
                f"overlay={self.overlay!r}, grip={self.grip!r})")


class AirMouseController:
    """The pure per-frame brain. Holds the smoothing + debounce state and turns
    each (hand_pos, raw_grip, tracked) sample into an AirMouseDecision. NO I/O —
    the live loop applies the decision (move cursor, press button, publish
    overlay state). Re-buildable cheaply; reset() on disable / hand-loss.

    Button semantics (RIGHT by default):
      • stable grip OPEN  → cursor moves; overlay "track"; no button change.
      • OPEN → CLOSED edge → emit button "down"; overlay flips to "grab".
      • stays CLOSED       → cursor STILL moves (so a closed hand DRAGS); overlay
                             holds "grab". (close→move→open = a right-DRAG.)
      • CLOSED → OPEN edge  → emit button "up"; overlay back to "track".
        (a quick close→open with no move = a right-CLICK.)
    """

    def __init__(self, reach: ReachBox,
                 alpha: float = AIR_MOUSE_EMA_ALPHA,
                 debounce_frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES):
        self.reach = reach
        self._ema_x = EMA(alpha)
        self._ema_y = EMA(alpha)
        self._grip = GripDebouncer(debounce_frames, initial="open")
        self._button_down = False

    def reset(self) -> None:
        """Drop all smoothing + grip state. Used by the dead-man and on disable
        so the next hand starts clean (no cursor sweep from a stale value, no
        phantom button edge)."""
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip.reset(initial="open")
        # NB: this does NOT itself emit a button-up — the caller (dead-man) is
        # responsible for releasing a held button. We only clear our own view.
        self._button_down = False

    @property
    def button_is_down(self) -> bool:
        return self._button_down

    def release_decision(self) -> AirMouseDecision:
        """The DEAD-MAN decision: hand lost. If a button was held, command it
        UP; hide the overlay; clear smoothing so the next acquisition snaps.
        Idempotent — once released, repeated calls just keep the overlay
        hidden with no button edge."""
        button = "up" if self._button_down else None
        self._button_down = False
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip.reset(initial="open")
        return AirMouseDecision(cursor=None, button=button,
                                overlay="hidden", grip="open")

    def update(self, hand_xy, raw_grip: str, tracked: bool) -> AirMouseDecision:
        """Advance one frame.

        hand_xy: (x, y) camera-space metres, or None when no hand sample.
        raw_grip: the raw bridge grip ("open"/"closed"/"lasso"/"unknown").
        tracked: True when the bridge reported a tracked body this frame.

        When NOT tracked (or no hand sample), this returns the dead-man
        release decision — a held button is released and the overlay hides."""
        if not tracked or hand_xy is None:
            return self.release_decision()

        # Smooth the position, then map to a pixel.
        sx = self._ema_x.update(hand_xy[0])
        sy = self._ema_y.update(hand_xy[1])
        cursor = self.reach.map(sx, sy)

        # Debounce the grip; detect button edges off the STABLE grip.
        stable = self._grip.update(raw_grip)
        button = None
        want_down = (stable == "closed")
        if want_down and not self._button_down:
            button = "down"
            self._button_down = True
        elif not want_down and self._button_down:
            button = "up"
            self._button_down = False

        overlay = "grab" if self._button_down else "track"
        return AirMouseDecision(cursor=cursor, button=button,
                                overlay=overlay, grip=stable)


def overlay_color_for(overlay_state: str) -> str:
    """Map an overlay state to the reticle's accent colour name. "grab" →
    "gold" (the locked state), everything else → "cyan" (the tracking state).
    Pure helper shared by the live publisher and the unit test so the colour
    contract is asserted against the same source the overlay reads."""
    return "gold" if overlay_state == "grab" else "cyan"


# ══════════════════════════════════════════════════════════════════════════
#  LIVE WIRING (sensor, mouse, overlay, staging gate, config flag)
# ══════════════════════════════════════════════════════════════════════════

def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_gestures)."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — the air-mouse must NEVER move the
    real cursor there. Matches the monolith's own gate plus the raw env var so
    the check holds even before the monolith is importable."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def _air_mouse_enabled() -> bool:
    """The master gate for the live loop: opt-in flag ON and not staging."""
    return _cfg_flag("KINECT_AIR_MOUSE_ENABLED") and not _is_staging()


# ─── primary-monitor geometry ──────────────────────────────────────────────
def _primary_screen_size() -> tuple[int, int]:
    """The PRIMARY monitor's (width, height) in pixels. Tries win32 first, then
    a configured MONITORS entry, then a safe default. Never raises."""
    # 1) win32 — the real primary-monitor metrics (SM_CXSCREEN / SM_CYSCREEN).
    try:
        import win32api
        import win32con
        w = int(win32api.GetSystemMetrics(win32con.SM_CXSCREEN))
        h = int(win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    # 2) A configured MONITORS dict (core.config) — use the "middle"/primary-ish
    #    entry's (w, h) when present.
    try:
        from core import config as _cfg
        monitors = getattr(_cfg, "MONITORS", None)
        if isinstance(monitors, dict):
            for key in ("middle", "primary", "main"):
                ent = monitors.get(key)
                if isinstance(ent, (list, tuple)) and len(ent) >= 4:
                    return int(ent[2]), int(ent[3])
    except Exception:
        pass
    # 3) Fallback.
    return _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H


# ─── mouse actuation (win32api primary, pyautogui fallback) ────────────────
def _set_cursor_pos(px: int, py: int) -> bool:
    """Move the OS cursor to an absolute primary-monitor pixel. win32api first
    (lowest latency), pyautogui as a fallback. Returns True on success. Never
    raises — a failed move is a silent no-op (the next frame retries)."""
    try:
        import win32api
        win32api.SetCursorPos((int(px), int(py)))
        return True
    except Exception:
        pass
    try:
        import pyautogui
        # FAILSAFE off: the air-mouse legitimately parks the cursor in a screen
        # corner (reach-box clamp), which pyautogui's default failsafe treats as
        # an abort. We do our own clamping, so disable it.
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(int(px), int(py))
        return True
    except Exception:
        return False


def _mouse_button(action: str) -> bool:
    """Press ('down') or release ('up') AIR_MOUSE_BUTTON at the current cursor
    position. win32api SendInput-style events first, pyautogui fallback. Returns
    True on success; never raises."""
    button = AIR_MOUSE_BUTTON
    # win32api path: event flags per button + up/down.
    try:
        import win32api
        import win32con
        if button == "left":
            flag = (win32con.MOUSEEVENTF_LEFTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_LEFTUP)
        else:  # right
            flag = (win32con.MOUSEEVENTF_RIGHTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_RIGHTUP)
        win32api.mouse_event(flag, 0, 0, 0, 0)
        return True
    except Exception:
        pass
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        if action == "down":
            pyautogui.mouseDown(button=button)
        else:
            pyautogui.mouseUp(button=button)
        return True
    except Exception:
        return False


# ─── overlay state publishing + spawn ──────────────────────────────────────
def _publish_overlay_state(decision: AirMouseDecision, visible: bool) -> None:
    """Write the live cursor + reticle state to AIR_CURSOR_STATE_FILE for the
    overlay process. Atomic-ish (write then it's a tiny file); best-effort and
    silent on failure — the overlay just renders the last good frame.

    Shape: {"x": int, "y": int, "state": "track"|"grab"|"hidden",
            "color": "cyan"|"gold", "ts": <epoch>, "visible": bool}"""
    try:
        import json
        if decision.cursor is not None:
            x, y = decision.cursor
        else:
            x, y = -10000, -10000   # off-screen sentinel; overlay hides anyway
        state = decision.overlay if visible else "hidden"
        data = {
            "x": int(x), "y": int(y),
            "state": state,
            "color": overlay_color_for(state),
            "visible": bool(visible and state != "hidden"),
            "ts": time.time(),
        }
        with open(AIR_CURSOR_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _clear_overlay_state() -> None:
    """Publish a hidden/blank overlay state (used when the air-mouse turns off or
    the hand is lost) so the reticle disappears promptly."""
    try:
        import json
        with open(AIR_CURSOR_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"x": -10000, "y": -10000, "state": "hidden",
                       "color": "cyan", "visible": False, "ts": time.time()}, f)
    except Exception:
        pass


_overlay_process = [None]   # module-list so the loop can (re)assign without global


def _overlay_alive() -> bool:
    proc = _overlay_process[0]
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def _spawn_overlay() -> None:
    """Spawn hud/jarvis_air_cursor.py as a click-through overlay subprocess sized
    to the virtual desktop, mirroring _launch_reticle_overlay() in the monolith
    (same --x/--y/--width/--height/--parent-pid contract + CREATE_NO_WINDOW).
    Silent on failure so a missing tkinter / odd geometry never breaks the loop.
    Only ever called from the live loop, never in staging/test."""
    if _overlay_alive():
        return
    try:
        import subprocess
        overlay_path = os.path.join(PROJECT_DIR, "hud", "jarvis_air_cursor.py")
        if not os.path.exists(overlay_path):
            return
        vx, vy, vw, vh = _virtual_screen_bounds()
        parent_pid = os.getpid()
        # Prefer the monolith's PID so the overlay dies with JARVIS, not with a
        # transient skill thread (which shares this process anyway, but be
        # explicit/robust if a future reload changes that).
        bc = _bc()
        if bc is not None:
            try:
                parent_pid = int(getattr(bc, "_MAIN_PID", parent_pid) or parent_pid)
            except Exception:
                parent_pid = os.getpid()
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        except Exception:
            flags = 0
        _overlay_process[0] = subprocess.Popen(
            [sys.executable, overlay_path,
             "--x", str(vx), "--y", str(vy),
             "--width", str(vw), "--height", str(vh),
             "--parent-pid", str(parent_pid)],
            creationflags=flags, close_fds=True,
        )
        print(f"  [air-mouse] cursor overlay launched "
              f"({vw}x{vh} @ {vx},{vy}, pid {_overlay_process[0].pid})")
    except Exception as e:
        print(f"  [air-mouse] overlay launch failed: {e}")
        _overlay_process[0] = None


def _shutdown_overlay() -> None:
    """Terminate the overlay subprocess (best-effort)."""
    proc = _overlay_process[0]
    _overlay_process[0] = None
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass


# Win32 GetSystemMetrics indices for the VIRTUAL desktop (all monitors). Used
# both via win32con and, as a no-pywin32 fallback, via ctypes user32 directly.
_SM_XVIRTUALSCREEN = 76     # left edge of the virtual desktop (NEGATIVE if a
_SM_YVIRTUALSCREEN = 77     #   monitor sits left of / above the primary)
_SM_CXVIRTUALSCREEN = 78    # full virtual-desktop width  (sum across monitors)
_SM_CYVIRTUALSCREEN = 79    # full virtual-desktop height


def _virtual_screen_bounds() -> tuple[int, int, int, int]:
    """(x, y, w, h) of the whole virtual desktop spanning ALL monitors. Prefer
    the monolith's helper (single source of truth); fall back to win32, then to
    ctypes user32.GetSystemMetrics, then to the primary size. x/y are NEGATIVE
    when a monitor is arranged left-of / above the primary — SetCursorPos accepts
    these directly, so the whole desktop is reachable."""
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_virtual_screen_bounds", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        import win32api
        import win32con
        vx = int(win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN))
        vy = int(win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN))
        vw = int(win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN))
        vh = int(win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN))
        if vw > 0 and vh > 0:
            return vx, vy, vw, vh
    except Exception:
        pass
    # pywin32 absent but we may still be on real Windows: ask user32 directly.
    try:
        import ctypes
        gsm = ctypes.windll.user32.GetSystemMetrics
        vx = int(gsm(_SM_XVIRTUALSCREEN))
        vy = int(gsm(_SM_YVIRTUALSCREEN))
        vw = int(gsm(_SM_CXVIRTUALSCREEN))
        vh = int(gsm(_SM_CYVIRTUALSCREEN))
        if vw > 0 and vh > 0:
            return vx, vy, vw, vh
    except Exception:
        pass
    w, h = _primary_screen_size()
    return 0, 0, w, h


# Cached virtual-desktop bounds + the time we last refreshed them, so the live
# loop doesn't hit win32 every tick but still notices a display-layout change.
_VBOUNDS_CACHE: list = [None, 0.0]   # [(x, y, w, h) | None, last_refresh_ts]


def _cached_virtual_bounds(refresh: bool = False) -> tuple[int, int, int, int]:
    """The virtual-desktop bounds, cached. Re-reads when `refresh` is True, when
    nothing is cached yet, or when VIRTUAL_BOUNDS_REFRESH_SECONDS have elapsed —
    so hot-plugging a monitor is picked up without a restart."""
    cached, last = _VBOUNDS_CACHE
    now = time.time()
    if (cached is None or refresh
            or (now - last) >= VIRTUAL_BOUNDS_REFRESH_SECONDS):
        cached = _virtual_screen_bounds()
        _VBOUNDS_CACHE[0] = cached
        _VBOUNDS_CACHE[1] = now
    return cached


def _reach_box_for_virtual_desktop(refresh: bool = False) -> "ReachBox":
    """Build a ReachBox mapped across the WHOLE virtual desktop (all monitors),
    using the cached virtual-screen bounds."""
    vx, vy, vw, vh = _cached_virtual_bounds(refresh=refresh)
    return ReachBox(vw, vh, origin_x=vx, origin_y=vy)


# ─── the per-tick read → decide → act path (unit-tested via _poll_once) ────
def _hand_sample(bridge) -> tuple[Optional[tuple], str, bool]:
    """Read the pointing hand from the bridge: (hand_xy, raw_grip, tracked).

    hand_xy is the nearest body's pointing-hand (x, y) in camera-space metres
    (None when no body / no usable hand joint). raw_grip is that body's grip for
    the SAME hand. tracked is whether a body was in view. Never raises — any
    failure degrades to (None, "unknown", False) which the controller treats as
    a dead-man release."""
    try:
        if not bridge.get_enabled():
            return None, "unknown", False
        ok, _reason = bridge.available()
        if not ok:
            return None, "unknown", False
        bodies = bridge.get_bodies()
    except Exception:
        return None, "unknown", False
    if not bodies:
        return None, "unknown", False

    # Nearest body (same ranking the rest of the stack uses).
    def _key(b):
        d = b.get("distance_m") if isinstance(b, dict) else None
        return d if isinstance(d, (int, float)) and d > 0 else float("inf")
    try:
        body = min((b for b in bodies if isinstance(b, dict)), key=_key)
    except (TypeError, ValueError):
        return None, "unknown", False

    joints = body.get("joints") or {}
    # Choose the pointing hand: prefer whichever hand's grip is most decisive
    # (closed beats open beats unknown), else the right hand. We read the SAME
    # side's hand joint so the cursor follows the hand we're gripping with.
    right_grip = (body.get("hand_right") or "unknown").lower()
    left_grip = (body.get("hand_left") or "unknown").lower()

    def _rank(g):
        return {"closed": 2, "open": 1}.get(g, 0)
    side = "right" if _rank(right_grip) >= _rank(left_grip) else "left"
    grip = right_grip if side == "right" else left_grip

    hand = joints.get(f"hand_{side}") or joints.get(f"wrist_{side}")
    if not hand or len(hand) < 2:
        # No usable hand joint on the chosen side — try the other side's joint.
        other = "left" if side == "right" else "right"
        hand = joints.get(f"hand_{other}") or joints.get(f"wrist_{other}")
        if hand and len(hand) >= 2:
            grip = left_grip if other == "left" else right_grip
    if not hand or len(hand) < 2:
        return None, grip, True   # body tracked but no hand joint this frame
    return (float(hand[0]), float(hand[1])), grip, True


def _apply_decision(decision: AirMouseDecision) -> None:
    """Perform the side effects of a decision: move the cursor and actuate the
    button. Pure-core stays I/O-free; THIS is where the real mouse is touched.
    Best-effort; never raises out to the loop."""
    if decision.cursor is not None:
        _set_cursor_pos(decision.cursor[0], decision.cursor[1])
    if decision.button in ("down", "up"):
        _mouse_button(decision.button)


def _poll_once(ctrl: AirMouseController, bridge) -> Optional[AirMouseDecision]:
    """One air-mouse tick: read the hand, decide, and (only when enabled +
    not staging) ACT — move the cursor, actuate the button, publish the overlay
    state. Returns the decision (for tests) or None when the bridge is absent.
    NEVER raises.

    GATING: the controller is ALWAYS advanced (so its smoothing/grip state stays
    current and a re-enable doesn't see a huge gap), but the SIDE EFFECTS (mouse
    move, button, visible overlay) only happen when KINECT_AIR_MOUSE_ENABLED is
    on AND not staging. Flipping the flag off therefore stops the cursor moving
    instantly and releases any held button via the dead-man path."""
    if bridge is None:
        return None
    hand_xy, raw_grip, tracked = _hand_sample(bridge)
    try:
        decision = ctrl.update(hand_xy, raw_grip, tracked)
    except Exception:
        # A controller error must not strand a held button — force a release.
        try:
            decision = ctrl.release_decision()
        except Exception:
            return None

    enabled = _air_mouse_enabled()
    if not enabled:
        # Gated OFF mid-session: make sure no button is left held and the
        # overlay is hidden. ctrl.update already returned a (possibly
        # button-up) decision if it had been holding; honour a pending 'up'
        # so a flag flip during a drag still releases, but never a 'down'.
        if decision.button == "up":
            _mouse_button("up")
        _clear_overlay_state()
        return decision

    # Enabled + not staging: act.
    _apply_decision(decision)
    visible = tracked and decision.overlay != "hidden"
    _publish_overlay_state(decision, visible)
    # Keep the reticle overlay process alive while enabled.
    if visible and not _overlay_alive():
        _spawn_overlay()
    return decision


def _poll_loop() -> None:  # pragma: no cover - non-terminating daemon; each tick delegates to _poll_once, which is unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    bridge = _bridge()
    if bridge is None:
        print("  [air-mouse] kinect_bridge unavailable — poller exiting")
        return
    # Map across the WHOLE virtual desktop (all monitors), not just primary.
    ctrl = AirMouseController(_reach_box_for_virtual_desktop(refresh=True))
    was_enabled = False
    last_bounds_refresh = time.time()
    while True:
        try:
            bridge = _bridge() or bridge
            enabled = _air_mouse_enabled()
            now = time.time()
            if enabled and not was_enabled:
                # Just turned on — re-read the virtual-desktop bounds (the user
                # may have changed displays) and start fresh so the first hand
                # snaps to where it's pointing.
                ctrl.reach = _reach_box_for_virtual_desktop(refresh=True)
                last_bounds_refresh = now
                ctrl.reset()
            elif enabled and (now - last_bounds_refresh) >= VIRTUAL_BOUNDS_REFRESH_SECONDS:
                # Periodic refresh while running so a hot-plugged / rearranged
                # monitor is picked up live. _cached_virtual_bounds() only hits
                # win32 when the interval has actually elapsed, so this is cheap;
                # the ReachBox is rebuilt only when the bounds changed.
                vb = _cached_virtual_bounds(refresh=True)
                last_bounds_refresh = now
                cur = ctrl.reach
                if (vb[0], vb[1], vb[2], vb[3]) != (
                        cur.origin_x, cur.origin_y, cur.screen_w, cur.screen_h):
                    ctrl.reach = ReachBox(vb[2], vb[3], origin_x=vb[0], origin_y=vb[1])
            if was_enabled and not enabled:
                # Just turned off — tear the overlay down + clear its state.
                _shutdown_overlay()
                _clear_overlay_state()
            was_enabled = enabled
            _poll_once(ctrl, bridge)
        except Exception as e:
            print(f"  [air-mouse] poll error: {e}")
        time.sleep(AIR_MOUSE_POLL_INTERVAL)


# ─── persistence (reuse the hardened Settings writer) ──────────────────────
def _persist_setting(key: str, value) -> bool:
    """Write {key: value} into the settings file WITHOUT clobbering the owner's
    other settings — the EXACT path model_picker / kinect_gestures use
    (settings_window.load_settings + save_settings, which honour
    JARVIS_SETTINGS_PATH so tests can't touch the real file). Best-effort."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current)
        return True
    except Exception:
        return False


def _set_enabled(on: bool) -> bool:
    """Flip KINECT_AIR_MOUSE_ENABLED live (core.config) and persist it."""
    try:
        import core.config as _cfg
        _cfg.KINECT_AIR_MOUSE_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("KINECT_AIR_MOUSE_ENABLED", bool(on))


# ─── sensor-readiness (honest spoken reasons) ──────────────────────────────
def _sensor_ready() -> tuple[bool, str]:
    """(True, "") when the Kinect is enabled AND available; else (False, why)."""
    kb = _bridge()
    if kb is None:
        return False, "the Kinect bridge isn't loaded"
    try:
        if not kb.get_enabled():
            return False, "the Kinect is switched off"
        ok, reason = kb.available()
        if not ok:
            return False, (reason or "the Kinect is unavailable")
    except Exception:
        return False, "the Kinect is unavailable"
    return True, ""


def _hand_in_view() -> Optional[bool]:
    """True/False if the Kinect can currently see a usable hand; None when the
    sensor is off/absent so the caller can phrase it honestly."""
    kb = _bridge()
    if kb is None:
        return None
    try:
        if not kb.get_enabled():
            return None
        ok, _reason = kb.available()
        if not ok:
            return None
        states = kb.get_hand_states()
        if not states.get("tracked"):
            return False
        return (states.get("right") in ("open", "closed")
                or states.get("left") in ("open", "closed"))
    except Exception:
        return None


# ─── actions ─────────────────────────────────────────────────────────────
def air_mouse_on(_: str = "") -> str:
    """Turn the air-mouse on (live + persisted)."""
    if _cfg_flag("KINECT_AIR_MOUSE_ENABLED"):
        already = "The air-mouse is already on, sir."
    else:
        already = None
    persisted = _set_enabled(True)
    ready, why = _sensor_ready()
    sensor_note = "" if ready else f" Note {why} — enable the Kinect so I can see your hand."
    if already:
        return already + sensor_note
    msg = ("Air-mouse on, sir — point an open hand at the screen to move the "
           "cursor, close your hand to right-click, and hold it closed to drag.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + sensor_note


def air_mouse_off(_: str = "") -> str:
    """Turn the air-mouse off (live + persisted). Also clears the overlay so the
    reticle disappears immediately."""
    if not _cfg_flag("KINECT_AIR_MOUSE_ENABLED"):
        return "The air-mouse is already off, sir."
    persisted = _set_enabled(False)
    # Make sure nothing is left held and the reticle is gone right away.
    try:
        _mouse_button("up")
    except Exception:
        pass
    _shutdown_overlay()
    _clear_overlay_state()
    msg = "Air-mouse off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def air_mouse_status(_: str = "") -> str:
    """Report whether the air-mouse is on + whether a hand is in view.
    'is the air-mouse on' / 'air-mouse status'."""
    enabled = _cfg_flag("KINECT_AIR_MOUSE_ENABLED")
    how = ("point an open hand to move the cursor, close to right-click, hold "
           "closed to drag")
    if not enabled:
        return (f"The air-mouse is off, sir — say 'turn on the air-mouse' to "
                f"enable it. Once on, {how}.")
    in_view = _hand_in_view()
    if in_view is None:
        return ("The air-mouse is on, sir, but the Kinect is off or "
                "unavailable, so I can't see your hand right now.")
    if in_view:
        return f"The air-mouse is on and I can see your hand, sir — {how}."
    return ("The air-mouse is on, sir, but I don't see a hand in the Kinect's "
            f"view at the moment. Raise an open hand toward the screen and {how}.")


# ─── registration ────────────────────────────────────────────────────────
def register(actions):
    actions["air_mouse_on"] = air_mouse_on
    actions["air_mouse_off"] = air_mouse_off
    actions["air_mouse_status"] = air_mouse_status

    # Guard against duplicate pollers on skill reload (same OS-thread-name check
    # kinect_gestures / face_tracker use). The loop self-gates on
    # KINECT_AIR_MOUSE_ENABLED + staging each tick, so it's cheap to leave
    # running even when disabled.
    if any(th.name == _THREAD_NAME and th.is_alive()
           for th in threading.enumerate()):
        print("  [air-mouse] poller already running — skipping duplicate (reload)")
    else:
        t = threading.Thread(target=_poll_loop, daemon=True, name=_THREAD_NAME)
        t.start()
        print(f"  [air-mouse] air-mouse poller active (~{AIR_MOUSE_POLL_HZ:.0f} Hz; "
              "opt-in via KINECT_AIR_MOUSE_ENABLED, off by default)")
