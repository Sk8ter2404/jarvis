"""Hand mouse (air mouse v2): drive the cursor with a pinch, seen by the webcam
hand tracker (tools/hand_tracker_service.py, ws://127.0.0.1:8793).

This module is PURE logic plus one thin Windows input class. Nothing here
imports Windows modules at import time, and every OS effect goes through the
`MouseOutput` interface, so tests drive the whole state machine with a fake and
the real cursor is never touched while developing.

Owner context: the Kinect air mouse was turned OFF on 2026-10-13 because it kept
stealing his cursor. So everything here is deliberate and conservative.

WHAT TRIGGERS WHAT  (the gesture states come from tools/handtrack_gestures.py,
the single source of truth; nothing is re-derived here)
  ENGAGE      the engine's `pinch_hold` event: a pinch held PINCH_HOLD_S (0.6 s)
              while the pinch point spans <= 1.0 hand sizes (that is the "held
              fairly still" test, with the engine's own hysteresis: pinch on at
              strength 0.75, off below 0.50). Only while the hand mouse is
              enabled and nothing blocks it. Engaging never clicks or drags: the
              pinch that engaged is CONSUMED, the cursor only starts following
              once that pinch is released.
  MOVE        cursor = pinch point (midpoint of thumb tip and index tip) mapped
              from a central box of the camera view (BOX, normalised mirrored
              frame) to the screen, through a One Euro filter (adaptive low
              pass, dt-aware so it works at 12 fps and at 30) and a dead zone.
  CLICK       a quick pinch and release (held < DRAG_HOLD_S, 0.40 s) = ONE left
              click, at the position the cursor had when the pinch began (the
              cursor is frozen for the duration of a pinch, so closing the
              fingers cannot nudge the click).
  DRAG        a pinch held >= DRAG_HOLD_S presses the left button; moving the
              hand drags; releasing the pinch releases the button. Because the
              engaging pinch is consumed, a drag can only start from a SECOND,
              deliberate pinch.
  DISENGAGE   an open palm facing the camera held STILL for palm_release_s (0.4 s;
              moving open hand is the normal follow pose, so it must hold still;
              HAND_MOUSE_PALM_RELEASE_S = 0 turns this off) | the
              hand gone from the frame HAND_GONE_S (1.0 s) | the tracker
              stopping or disconnecting | "hand mouse off" / "hand tracking off"
              | a blocker (game mode, fullscreen app in front, in a call) |
              the real mouse being moved by hand (the cursor is somewhere other
              than where we put it: YIELD_PX). Any held button is released.
  NOT DONE    right click (not trivial and safe), multi-monitor spanning.

MULTI-MONITOR: the box maps to the PRIMARY monitor. To use another one set
core.config.HAND_MOUSE_SCREEN_RECT = (left, top, width, height) in virtual
desktop pixels, e.g. (-1920, 0, 1920, 1080) for a 1080p monitor left of the
primary. (Windows reports those through the same coordinate space SetCursorPos
uses.) DPI: the numbers are in the same pixel space SetCursorPos takes, so a
DPI-aware JARVIS gets physical pixels.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

LOG = "  [hand-mouse]"

# landmark indices (MediaPipe)
THUMB_TIP = 4
INDEX_TIP = 8

# phases
IDLE = "idle"              # enabled, waiting for the engage pinch-hold
ENGAGING = "engaging"      # engaged; waiting for the engaging pinch to be released
ENGAGED = "engaged"        # following the pinch point
# sub-states while ENGAGED
FREE = "free"              # no pinch: the cursor follows the hand
PENDING = "pending"        # a pinch began: cursor frozen, click or drag undecided
DRAG = "drag"              # left button held, the cursor follows the hand


@dataclass
class Params:
    """Every tunable, in one place (defaults are the documented thresholds)."""
    box: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.75)
    screen_rect: Optional[Tuple[int, int, int, int]] = None   # None = primary
    min_cutoff: float = 1.0        # One Euro: cutoff (Hz) for a still hand
    beta: float = 0.008            # One Euro: cutoff gain per px/s of speed
    d_cutoff: float = 1.0          # One Euro: cutoff of the speed estimate
    deadzone_px: float = 8.0       # a still hand keeps the cursor still
    palm_release_s: float = 0.4    # open palm held this long disengages (0 = never)
    palm_still: float = 0.03       # ...and the palm stayed within this (normalised)
    hand_gone_s: float = 1.0       # hand out of the frame this long disengages
    drag_hold_s: float = 0.40      # pinch held this long presses the button
    freeze_lookback_s: float = 0.15  # the click lands where the cursor WAS
    block_check_s: float = 0.5     # how often the blockers are re-checked
    yield_px: float = 40.0         # real mouse moved this far from where we left it
    max_dt_s: float = 0.5          # a longer gap between frames restarts the filter

    @classmethod
    def from_config(cls, get: Callable[[str, object], object]) -> "Params":
        p = cls()
        box = get("HAND_MOUSE_BOX", None)
        try:
            if box is not None and len(box) == 4:
                b = tuple(float(v) for v in box)
                if b[2] - b[0] > 0.05 and b[3] - b[1] > 0.05:
                    p.box = b
        except Exception:
            pass
        try:
            v = get("HAND_MOUSE_PALM_RELEASE_S", None)
            if v is not None and float(v) >= 0:
                p.palm_release_s = float(v)
        except Exception:
            pass
        rect = get("HAND_MOUSE_SCREEN_RECT", None)
        try:
            if rect is not None and len(rect) == 4 and int(rect[2]) > 0 and int(rect[3]) > 0:
                p.screen_rect = tuple(int(v) for v in rect)
        except Exception:
            pass
        return p


# ── the output interface (the only place real input can come from) ────────
class MouseOutput:
    """What the hand mouse needs from the OS. Tests pass a fake."""

    def move(self, x: int, y: int) -> None:
        raise NotImplementedError

    def button(self, down: bool) -> None:
        """Press (True) or release (False) the LEFT button at the cursor."""
        raise NotImplementedError

    def position(self) -> Optional[Tuple[int, int]]:
        """Where the OS cursor is now (None = unknown)."""
        return None

    def screen_rect(self) -> Tuple[int, int, int, int]:
        """(left, top, width, height) of the primary monitor."""
        raise NotImplementedError


class WindowsMouse(MouseOutput):
    """ctypes SetCursorPos / GetCursorPos / SendInput. Constructing it touches
    nothing; each call does one OS call. Windows only."""

    _LEFTDOWN = 0x0002
    _LEFTUP = 0x0004
    # dwExtraInfo tag so other JARVIS input watchers can tell this apart from a
    # physical mouse ("JHM!").
    EXTRA_TAG = 0x4A484D21

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ct = ctypes
        self._user32 = ctypes.windll.user32

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._MOUSEINPUT, self._INPUT = MOUSEINPUT, INPUT
        self._pt = wintypes.POINT

    def move(self, x: int, y: int) -> None:
        self._user32.SetCursorPos(int(x), int(y))

    def button(self, down: bool) -> None:
        flag = self._LEFTDOWN if down else self._LEFTUP
        inp = self._INPUT(type=0, u=self._INPUT._fields_[1][1](
            mi=self._MOUSEINPUT(0, 0, 0, flag, 0, self.EXTRA_TAG)))
        self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp))

    def position(self) -> Optional[Tuple[int, int]]:
        pt = self._pt()
        if self._user32.GetCursorPos(self._ct.byref(pt)):
            return int(pt.x), int(pt.y)
        return None

    def screen_rect(self) -> Tuple[int, int, int, int]:
        return (0, 0, int(self._user32.GetSystemMetrics(0)),
                int(self._user32.GetSystemMetrics(1)))


# ── the One Euro filter ───────────────────────────────────────────────────
class OneEuro:
    """One Euro filter (Casiez et al. 2012) for one axis. Its smoothing factor
    is computed from the REAL time between samples, so it behaves the same at
    12 fps and 30 fps."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    def seed(self, x: float, t: float) -> None:
        """Restart the filter at x (no speed memory)."""
        self._x, self._dx, self._t = x, 0.0, t

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            if self._x is None:
                self._x, self._t = x, t
            return self._x
        dt = t - self._t
        raw_d = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * raw_d + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x


def pinch_point(lm) -> Optional[Tuple[float, float]]:
    """Midpoint of thumb tip and index tip, normalised mirrored frame."""
    try:
        a, b = lm[THUMB_TIP], lm[INDEX_TIP]
        return (float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0
    except Exception:
        return None


# ── the state machine ─────────────────────────────────────────────────────
class HandMouse:
    """Feed it the tracker's messages; it drives `output`. All methods are
    thread safe and never raise."""

    def __init__(self, output: MouseOutput, *,
                 blocker: Optional[Callable[[], str]] = None,
                 cue: Optional[Callable[[str, str], None]] = None,
                 params: Optional[Params] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.out = output
        self.blocker = blocker
        self.cue = cue
        self.p = params or Params()
        self._log = log or (lambda s: print(f"{LOG} {s}"))
        self._lock = threading.RLock()
        self.enabled = False
        self.phase = IDLE
        self.sub = FREE
        self.last_reason = ""
        self.clicks = 0
        self.engagements = 0
        self._hands: dict = {}
        self._frame_at: Optional[float] = None
        self._last_block_log = -1e9
        self._reset_run()

    # ── lifecycle ────────────────────────────────────────────────────────
    def set_enabled(self, on: bool, now: Optional[float] = None) -> None:
        with self._lock:
            if not on:
                self.disengage("hand mouse off", now)
                self._hands = {}
                self._frame_at = None
            self.enabled = bool(on)

    def _reset_run(self) -> None:
        self.side = ""
        self._pressed = False
        self._last_seen = 0.0
        self._palm_since: Optional[float] = None
        self._next_block_check = 0.0
        self._rect = (0, 0, 1920, 1080)
        self._fx = OneEuro(self.p.min_cutoff, self.p.beta, self.p.d_cutoff)
        self._fy = OneEuro(self.p.min_cutoff, self.p.beta, self.p.d_cutoff)
        self._filt: Optional[Tuple[float, float]] = None
        self._cur: Optional[Tuple[float, float]] = None
        self._last_set: Optional[Tuple[int, int]] = None
        self._offset = (0.0, 0.0)
        self._hist: deque = deque()         # (t, x, y) of cursor moves
        self._fhist: deque = deque()        # (t, fx, fy) of the filtered point, per frame
        self._freeze_filt: Optional[Tuple[float, float]] = None
        self._down_t = 0.0
        self._frame_prev: Optional[float] = None
        self._palm_ref: Optional[Tuple[float, float]] = None

    def engaged(self) -> bool:
        return self.phase != IDLE

    def button_down(self) -> bool:
        return self._pressed

    def status(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "phase": self.phase, "sub": self.sub,
                    "pressed": self._pressed, "last_reason": self.last_reason,
                    "clicks": self.clicks, "engagements": self.engagements}

    def _blocked(self) -> str:
        if self.blocker is None:
            return ""
        try:
            return str(self.blocker() or "")
        except Exception:
            return ""

    def _emit_cue(self, kind: str, reason: str = "") -> None:
        if self.cue is None:
            return
        try:
            self.cue(kind, reason)
        except Exception:
            pass

    # ── input ────────────────────────────────────────────────────────────
    def on_gesture(self, ev: dict, now: Optional[float] = None) -> str:
        """A gesture event from the tracker. Returns what happened (logs and
        tests): 'engaged', 'ignored: <why>' or ''."""
        try:
            with self._lock:
                now = time.monotonic() if now is None else now
                if not self.enabled or ev.get("name") != "pinch_hold":
                    return ""
                if self.phase != IDLE:
                    return "ignored: already engaged"
                why = self._blocked()
                if why:
                    if now - self._last_block_log > 10.0:
                        self._last_block_log = now
                        self._log(f"pinch hold ignored: {why}")
                    return f"ignored: {why}"
                side = str(ev.get("side", ""))
                h = self._hands.get(side)
                if (h is None or self._frame_at is None or now - self._frame_at > 0.5
                        or h.get("state") != "pinch"):
                    return "ignored: not pinching in a live frame"
                self._reset_run()
                self.side = side
                self._last_seen = now
                self._rect = self._screen_rect()
                self.phase, self.sub = ENGAGING, FREE
                self._next_block_check = now + self.p.block_check_s
                self.engagements += 1
                self._log(f"engaged ({side} hand) - release the pinch, then move")
                self._emit_cue("engage", "")
                return "engaged"
        except Exception:
            return "ignored: error"

    def on_hands(self, msg: dict, now: Optional[float] = None) -> None:
        """A parsed {"type":"hands"} message."""
        try:
            with self._lock:
                if not self.enabled:
                    return
                now = time.monotonic() if now is None else now
                hands = {}
                for h in (msg.get("hands") or []):
                    if isinstance(h, dict):
                        side = str(h.get("side", ""))
                        if side not in hands or float(h.get("score", 0) or 0) > float(
                                hands[side].get("score", 0) or 0):
                            hands[side] = h
                self._hands = hands
                self._frame_at = now
                if self.phase != IDLE:
                    self._step(now)
        except Exception:
            pass

    def tick(self, now: Optional[float] = None) -> None:
        """Call now and then (the listener does, on every socket timeout):
        handles the hand vanishing without a frame saying so."""
        try:
            with self._lock:
                if not self.enabled or self.phase == IDLE:
                    return
                now = time.monotonic() if now is None else now
                if now - self._last_seen >= self.p.hand_gone_s:
                    self.disengage("the hand left the view", now)
        except Exception:
            pass

    def on_tracker_stopped(self, now: Optional[float] = None) -> None:
        with self._lock:
            self._hands = {}
            self._frame_at = None
            self.disengage("the tracker stopped", now)

    # ── disengage ────────────────────────────────────────────────────────
    def disengage(self, reason: str, now: Optional[float] = None) -> bool:
        """Leave the engaged state, releasing any held button. True if it was
        engaged. Never raises."""
        with self._lock:
            was = self.phase != IDLE
            if self._pressed:
                self._pressed = False
                try:
                    self.out.button(False)
                except Exception:
                    pass
            if not was:
                return False
            self._reset_run()
            self.phase, self.sub = IDLE, FREE
            self.last_reason = reason
            self._log(f"disengaged: {reason}")
            self._emit_cue("disengage", reason)
            return True

    # ── the engaged loop ─────────────────────────────────────────────────
    def _screen_rect(self) -> Tuple[int, int, int, int]:
        if self.p.screen_rect is not None:
            return self.p.screen_rect
        try:
            return tuple(int(v) for v in self.out.screen_rect())
        except Exception:
            return (0, 0, 1920, 1080)

    def _step(self, now: float) -> None:
        if now >= self._next_block_check:
            self._next_block_check = now + self.p.block_check_s
            why = self._blocked()
            if why:
                self.disengage(why, now)
                return
        h = self._hands.get(self.side)
        if h is None:
            if now - self._last_seen >= self.p.hand_gone_s:
                self.disengage("the hand left the view", now)
            return
        self._last_seen = now
        state = h.get("state")
        pinching = state == "pinch"

        # open palm held (and still) -> disengage
        palm = h.get("palm") or [0.0, 0.0]
        if (self.p.palm_release_s > 0 and state == "open" and h.get("palm_facing")):
            px, py = float(palm[0]), float(palm[1])
            if (self._palm_since is None or self._palm_ref is None
                    or math.hypot(px - self._palm_ref[0], py - self._palm_ref[1])
                    > self.p.palm_still):
                self._palm_since, self._palm_ref = now, (px, py)
            elif now - self._palm_since >= self.p.palm_release_s:
                self.disengage("open palm", now)
                return
        else:
            self._palm_since = None

        if self.phase == ENGAGING:
            if pinching:
                return                        # the engaging pinch is consumed
            self.phase, self.sub = ENGAGED, FREE
            self._log("following your pinch point")

        pt = pinch_point(h.get("lm"))
        if pt is None:
            return
        fx, fy = self._filter(pt, now)

        if self.sub == FREE:
            if pinching:
                self._begin_pinch(now, fx, fy)
                return
            self._follow(fx, fy)
        elif self.sub == PENDING:
            if not pinching:
                self._click()
                self._resume_from_freeze(now)
                self.sub = FREE
            elif now - self._down_t >= self.p.drag_hold_s:
                self._press()
                self._reanchor(fx, fy)
                self.sub = DRAG
        elif self.sub == DRAG:
            if not pinching:
                self._release()
                self._reanchor(fx, fy)
                self.sub = FREE
            else:
                self._follow(fx, fy)

    def _filter(self, pt, now: float) -> Tuple[float, float]:
        if self._filt is not None and self._frame_prev is not None and \
                now - self._frame_prev > self.p.max_dt_s:
            self._fx.reset()
            self._fy.reset()
        self._frame_prev = now
        l, t, w, h = self._rect
        x0, y0, x1, y1 = self.p.box
        u = min(1.0, max(0.0, (pt[0] - x0) / (x1 - x0)))
        v = min(1.0, max(0.0, (pt[1] - y0) / (y1 - y0)))
        sx = l + u * max(0, w - 1)
        sy = t + v * max(0, h - 1)
        self._filt = (self._fx(sx, now), self._fy(sy, now))
        self._fhist.append((now, self._filt[0], self._filt[1]))
        while self._fhist and now - self._fhist[0][0] > 1.0:
            self._fhist.popleft()
        return self._filt

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        l, t, w, h = self._rect
        return (min(l + max(0, w - 1), max(l, x)), min(t + max(0, h - 1), max(t, y)))

    def _follow(self, fx: float, fy: float) -> None:
        tx, ty = self._clamp(fx + self._offset[0], fy + self._offset[1])
        if self._cur is None:
            new = (tx, ty)
        else:
            dx, dy = tx - self._cur[0], ty - self._cur[1]
            d = math.hypot(dx, dy)
            if d <= self.p.deadzone_px:
                return
            k = (d - self.p.deadzone_px) / d
            new = (self._cur[0] + dx * k, self._cur[1] + dy * k)
        self._place(new)

    def _place(self, pos: Tuple[float, float]) -> bool:
        """Move the OS cursor (when the integer pixel changes). False when the
        real mouse was moved by hand - the hand mouse then disengages."""
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        if self._last_set is not None and (ix, iy) == self._last_set:
            self._cur = pos
            return True
        if self._last_set is not None:
            try:
                real = self.out.position()
            except Exception:
                real = None
            if real is not None and math.hypot(
                    real[0] - self._last_set[0], real[1] - self._last_set[1]) > self.p.yield_px:
                self.disengage("you moved the mouse", None)
                return False
        try:
            self.out.move(ix, iy)
        except Exception:
            pass
        self._last_set = (ix, iy)
        self._cur = pos
        now = self._frame_prev if self._frame_prev is not None else 0.0
        self._hist.append((now, pos[0], pos[1]))
        while self._hist and now - self._hist[0][0] > 1.0:
            self._hist.popleft()
        return True

    def _resume_from_freeze(self, now: float) -> None:
        """After a click: the pinch point drifted while the fingers closed, and
        that drift is not hand motion. Put the filter back where it was when the
        pinch began so the cursor carries on from the click spot."""
        if self._freeze_filt is None:
            return
        self._fx.seed(self._freeze_filt[0], now)
        self._fy.seed(self._freeze_filt[1], now)
        self._filt = self._freeze_filt

    def _reanchor(self, fx: float, fy: float) -> None:
        if self._cur is not None:
            self._offset = (self._cur[0] - fx, self._cur[1] - fy)

    def _begin_pinch(self, now: float, fx: float, fy: float) -> None:
        """A pinch began: freeze the cursor where it was just BEFORE the
        fingers started closing (the pinch point drifts while they do)."""
        self._down_t = now
        self.sub = PENDING
        self._freeze_filt = (fx, fy)
        for t, x, y in reversed(self._fhist):
            if now - t >= self.p.freeze_lookback_s:
                self._freeze_filt = (x, y)
                break
        else:
            if self._fhist:
                self._freeze_filt = (self._fhist[0][1], self._fhist[0][2])
        target = self._cur
        for t, x, y in reversed(self._hist):
            if now - t >= self.p.freeze_lookback_s:
                target = (x, y)
                break
        else:
            if self._hist:
                target = (self._hist[0][1], self._hist[0][2])
        if target is not None and self._cur is not None and target != self._cur:
            self._place(target)

    def _click(self) -> None:
        try:
            self.out.button(True)
            self.out.button(False)
        except Exception:
            pass
        self.clicks += 1

    def _press(self) -> None:
        try:
            self.out.button(True)
        except Exception:
            pass
        self._pressed = True

    def _release(self) -> None:
        self._pressed = False
        try:
            self.out.button(False)
        except Exception:
            pass
