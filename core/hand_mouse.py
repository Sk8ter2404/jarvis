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
  SCROLL      (2026-10-20, owner: "I also need a way to scroll") while ENGAGED,
              with ONE usable hand in view: close a FIST and hold it
              scroll_hold_s (0.2 s), then move it up / down - the page follows
              the hand like grabbing it (natural scrolling; scroll_natural =
              False flips it). The amount is proportional to the vertical
              distance from where the fist closed, past a dead zone of
              scroll_deadzone_hs hand sizes, capped at scroll_max_rate wheel
              units/s. The cursor does not move while scrolling; opening the
              hand ends it. Wheel events go out through MouseOutput.wheel (the
              same SendInput path as the clicks). HAND_MOUSE_SCROLL_ENABLED.
  TWO HANDS   while the webcam two-hand window grab (core/hand_two_hand.py) is
              active the hand mouse stands down: it disengages, releasing any
              button, and cannot engage (the Kinect air mouse does the same).
  NOT DONE    right click (not trivial and safe), multi-monitor spanning.

RECOGNITION RULES (2026-10-20, owner: "I don't want false triggers" - typing,
talking with his hands, scratching his face, a classmate's hand in frame).
Every threshold is a Params field (named, commented, tunable):
  * a hand counts only when it is USABLE: tracker score >= min_score_on and
    size >= min_size_on (a fraction of the frame height; too far / too small =
    ignored) to become usable, and it stays usable down to the looser
    min_score_off / min_size_off (hysteresis: entering is stricter than
    staying). A hand that just came into view is ignored for settle_s.
  * every hand STATE (pinch / fist / open ...) is debounced again here: a new
    state must persist confirm_min_frames frames AND confirm_s seconds, so the
    frame count scales with the measured frame rate. A one-frame pinch is no
    click, a one-frame fist is no scroll.
  * engaging needs the tracker's pinch_hold AND a pinch confirmed here for
    engage_pinch_s, from a usable, settled hand; never within input_quiet_s of
    the owner's own keyboard / mouse input (MouseOutput.idle_s), nor right
    after "you moved the mouse".
  * ONE owner hand: the engaged hand is locked (its side and its tracker id). A
    different hand - the other side, a new track id, or a hand reappearing far
    from where the owner's vanished - never drives the cursor.

RETICLE: reticle() describes what the on-screen reticle (hud/
jarvis_air_cursor.py, the Kinect air mouse's overlay) should show; the skill
publishes it (skills/hand_tracking.py). This module never writes files.

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
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17

# phases
IDLE = "idle"              # enabled, waiting for the engage pinch-hold
ENGAGING = "engaging"      # engaged; waiting for the engaging pinch to be released
ENGAGED = "engaged"        # following the pinch point
# sub-states while ENGAGED
FREE = "free"              # no pinch: the cursor follows the hand
PENDING = "pending"        # a pinch began: cursor frozen, click or drag undecided
DRAG = "drag"              # left button held, the cursor follows the hand
SCROLL = "scroll"          # fist held: vertical hand motion scrolls, cursor frozen

WHEEL_DELTA = 120          # one wheel notch (Windows WHEEL_DELTA)


@dataclass
class Params:
    """Every tunable, in one place (defaults are the documented thresholds)."""
    box: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.75)
    screen_rect: Optional[Tuple[int, int, int, int]] = None   # None = primary
    min_cutoff: float = 1.0        # One Euro: cutoff (Hz) for a still hand
    # 2026-10-20: 0.008 -> 0.015. Simulated at 18 fps with tracking noise this
    # cuts the follow lag by about a third (8/12/16 -> 6/8/10 px at 150/500/
    # 1500 px/s) and leaves a still hand's jitter the same (the dead zone
    # below absorbs it); 0.02 no longer smooths enough at
    # 12 fps (SmoothingTests). Most of the lag was the clumpy frame delivery,
    # fixed in core/handtrack_frames.FramePump.
    beta: float = 0.015            # One Euro: cutoff gain per px/s of speed
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
    # ── recognition (2026-10-20: no false triggers) ─────────────────────────
    # MediaPipe's handedness score is ~0.95+ for a clear hand; a half-hidden or
    # blurred one drops well below. Enter at 0.80, stay down to 0.60.
    min_score_on: float = 0.80     # a hand becomes usable at this tracker score...
    min_score_off: float = 0.60    # ...and stays usable down to this one
    # Hand size = max(wrist->middle-MCP, 1.2 x index-MCP->pinky-MCP) over the
    # frame HEIGHT (the gesture engine's "hs"). Owner at a laptop (~0.6 m) is
    # ~0.15-0.20; ~1.2 m away ~0.09; a classmate 2 m back ~0.05.
    min_size_on: float = 0.10      # a hand becomes usable at this size...
    min_size_off: float = 0.075    # ...and stays usable down to this one
    settle_s: float = 0.40         # a hand that just came into view is ignored this long
    track_gap_s: float = 0.30      # unseen longer than this = a NEW hand when it comes back
    confirm_s: float = 0.10        # a new hand state must persist this long...
    confirm_min_frames: int = 2    # ...and this many consecutive frames
    engage_pinch_s: float = 0.30   # engaging also needs a pinch confirmed here this long
    input_quiet_s: float = 1.0     # no engage within this long of real keyboard / mouse input
    reacquire_hs: float = 1.5      # the owner's hand may come back within this many hand sizes
    # ── scroll (2026-10-20) ─────────────────────────────────────────────────
    scroll_enabled: bool = True    # HAND_MOUSE_SCROLL_ENABLED
    scroll_hold_s: float = 0.20    # a fist held this long (after its own debounce) starts scrolling
    scroll_deadzone_hs: float = 0.25   # vertical travel (hand sizes) that scrolls nothing
    scroll_gain: float = 480.0     # wheel units per hand size beyond the dead zone (4 notches)
    scroll_max_rate: float = 2400.0    # wheel units/s at most (20 notches/s)
    scroll_min_step: int = 20      # smaller wheel steps are held back until they add up
    scroll_natural: bool = True    # True: the page follows the hand (hand down = wheel up)

    @classmethod
    def from_config(cls, get: Callable[[str, object], object]) -> "Params":
        p = cls()
        for key, attr in (("HAND_MOUSE_MIN_SCORE", "min_score_on"),
                          ("HAND_MOUSE_MIN_HAND_SIZE", "min_size_on"),
                          ("HAND_MOUSE_SETTLE_S", "settle_s"),
                          ("HAND_MOUSE_INPUT_QUIET_S", "input_quiet_s")):
            try:
                v = get(key, None)
                if v is not None and float(v) >= 0:
                    setattr(p, attr, float(v))
            except Exception:
                pass
        # Keep the exit thresholds looser than the (possibly tuned) entry ones.
        p.min_score_off = min(p.min_score_off, p.min_score_on * 0.75)
        p.min_size_off = min(p.min_size_off, p.min_size_on * 0.75)
        try:
            v = get("HAND_MOUSE_SCROLL_ENABLED", None)
            if v is not None:
                p.scroll_enabled = bool(v)
        except Exception:
            pass
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

    def wheel(self, delta: int) -> None:
        """Turn the vertical mouse wheel by ``delta`` (WHEEL_DELTA = one notch;
        positive = away from the user = scroll up)."""
        raise NotImplementedError

    def idle_s(self) -> Optional[float]:
        """Seconds since the last keyboard / mouse input on this machine
        (None = unknown). Engaging is refused right after real input."""
        return None


class WindowsMouse(MouseOutput):
    """ctypes SetCursorPos / GetCursorPos / SendInput. Constructing it touches
    nothing; each call does one OS call. Windows only."""

    _LEFTDOWN = 0x0002
    _LEFTUP = 0x0004
    _WHEEL = 0x0800
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
        self._send(flag, 0)

    def wheel(self, delta: int) -> None:
        # mouseData is a DWORD carrying a SIGNED wheel amount.
        self._send(self._WHEEL, int(delta) & 0xFFFFFFFF)

    def _send(self, flag: int, data: int) -> None:
        inp = self._INPUT(type=0, u=self._INPUT._fields_[1][1](
            mi=self._MOUSEINPUT(0, 0, data, flag, 0, self.EXTRA_TAG)))
        self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp))

    def idle_s(self) -> Optional[float]:
        # GetLastInputInfo also counts injected input; that is fine here: it is
        # only consulted while IDLE, when the hand mouse injects nothing.
        class LASTINPUTINFO(self._ct.Structure):
            _fields_ = [("cbSize", self._ct.c_uint), ("dwTime", self._ct.c_uint)]
        li = LASTINPUTINFO()
        li.cbSize = self._ct.sizeof(li)
        if not self._user32.GetLastInputInfo(self._ct.byref(li)):
            return None
        now = self._ct.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        return ((now - li.dwTime) & 0xFFFFFFFF) / 1000.0

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


def hand_size(h: dict, fw: float = 640.0, fh: float = 480.0) -> float:
    """The hand's size as a fraction of the frame HEIGHT: the gesture engine's
    "hs" (max(wrist->middle MCP, 1.2 x index MCP->pinky MCP)) computed from the
    normalised landmarks. A "size" key from the tracker wins. 0.0 when it
    cannot be measured (which is never usable)."""
    try:
        v = h.get("size")
        if v is not None:
            return max(0.0, float(v))
    except Exception:
        pass
    try:
        lm = h.get("lm")
        fw = float(fw) if fw else 640.0
        fh = float(fh) if fh else 480.0

        def d(i, j):
            return math.hypot((float(lm[i][0]) - float(lm[j][0])) * fw,
                              (float(lm[i][1]) - float(lm[j][1])) * fh)
        return max(d(WRIST, MIDDLE_MCP), 1.2 * d(INDEX_MCP, PINKY_MCP)) / fh
    except Exception:
        return 0.0


class _Confirm:
    """Debounce for one hand's state: a new state must persist `min_frames`
    consecutive frames AND `min_s` seconds before it is believed, so the frame
    count needed grows with the frame rate (2-3 at 15 fps, 4 at 30)."""

    def __init__(self):
        self.stable: Optional[str] = None
        self.since = 0.0               # when the stable state began
        self._cand: Optional[str] = None
        self._cand_since = 0.0
        self._n = 0

    def update(self, raw: str, now: float, min_s: float, min_frames: int) -> str:
        if self.stable is None:        # first sighting (the settle time covers it)
            self.stable, self.since = raw, now
            return raw
        if raw == self.stable:
            self._cand, self._n = None, 0
            return self.stable
        if raw != self._cand:
            self._cand, self._cand_since, self._n = raw, now, 0
        self._n += 1
        if self._n >= min_frames and now - self._cand_since >= min_s:
            self.stable, self.since = raw, self._cand_since
            self._cand, self._n = None, 0
        return self.stable


class _SideTrack:
    """What the hand mouse knows about the hand on one side of the frame."""

    def __init__(self, now: float, hid):
        self.first = now               # came into view (settle_s counts from here)
        self.last = now
        self.id = hid                  # the tracker's track id
        self.usable = False            # score + size, with hysteresis
        self.conf = _Confirm()
        self.state = "unknown"         # the confirmed state
        self.raw = "unknown"
        self.pos: Optional[Tuple[float, float]] = None   # palm, normalised
        self.size = 0.0

    def note(self, h: dict, now: float, size: float, p: "Params") -> None:
        self.last = now
        self.size = size
        try:
            score = float(h.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if self.usable:
            self.usable = score >= p.min_score_off and size >= p.min_size_off
        else:
            self.usable = score >= p.min_score_on and size >= p.min_size_on
        self.raw = str(h.get("state", "unknown") or "unknown")
        self.state = self.conf.update(self.raw, now, p.confirm_s,
                                      max(1, int(p.confirm_min_frames)))
        palm = h.get("palm")
        try:
            self.pos = (float(palm[0]), float(palm[1]))
        except Exception:
            pt = pinch_point(h.get("lm"))
            if pt is not None:
                self.pos = pt

    def ready(self, now: float, p: "Params") -> bool:
        """Usable, settled and seen in the last half second."""
        return (self.usable and now - self.first >= p.settle_s
                and now - self.last <= 0.5)


# ── the state machine ─────────────────────────────────────────────────────
class HandMouse:
    """Feed it the tracker's messages; it drives `output`. All methods are
    thread safe and never raise."""

    def __init__(self, output: MouseOutput, *,
                 blocker: Optional[Callable[[], str]] = None,
                 cue: Optional[Callable[[str, str], None]] = None,
                 params: Optional[Params] = None,
                 log: Optional[Callable[[str], None]] = None,
                 two_hand: Optional[Callable[[], bool]] = None):
        self.out = output
        self.blocker = blocker
        self.cue = cue
        self.two_hand = two_hand        # True while the two-hand window grab drives
        self.p = params or Params()
        self._log = log or (lambda s: print(f"{LOG} {s}"))
        self._lock = threading.RLock()
        self.enabled = False
        self.phase = IDLE
        self.sub = FREE
        self.last_reason = ""
        self.clicks = 0
        self.engagements = 0
        self.scrolls = 0               # scroll gestures started
        self.wheel_total = 0           # wheel units sent (signed)
        self._hands: dict = {}
        self._tracks: dict = {}        # side -> _SideTrack
        self._frame_at: Optional[float] = None
        self._last_block_log = -1e9
        self._quiet_until = -1e9       # no engage before this (after "you moved the mouse")
        self._reset_run()

    # ── lifecycle ────────────────────────────────────────────────────────
    def set_enabled(self, on: bool, now: Optional[float] = None) -> None:
        with self._lock:
            if not on:
                self.disengage("hand mouse off", now)
                self._hands = {}
                self._tracks = {}
                self._frame_at = None
            self.enabled = bool(on)

    def _reset_run(self) -> None:
        self.side = ""
        self._owner: Optional[_SideTrack] = None
        self._owner_pos: Optional[Tuple[float, float]] = None
        self._owner_size = 0.0
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
        self._engage_pos: Optional[Tuple[int, int]] = None
        self._scroll_y0 = 0.0
        self._scroll_hs = 0.0
        self._scroll_sent = 0.0         # wheel units sent this scroll (signed)
        self._scroll_t: Optional[float] = None

    def engaged(self) -> bool:
        return self.phase != IDLE

    def button_down(self) -> bool:
        return self._pressed

    def status(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "phase": self.phase, "sub": self.sub,
                    "pressed": self._pressed, "last_reason": self.last_reason,
                    "clicks": self.clicks, "engagements": self.engagements,
                    "scrolls": self.scrolls}

    def reticle(self) -> dict:
        """What the on-screen reticle should show: {"visible", "x", "y",
        "state"} with state "hidden" (idle), "prime" (engaged, waiting for the
        engaging pinch to open: a full priming ring), "track" (following) or
        "grab" (a pinch / drag / scroll is held). Never raises."""
        try:
            with self._lock:
                if not self.enabled or self.phase == IDLE:
                    return {"visible": False, "x": -10000, "y": -10000,
                            "state": "hidden"}
                pos = self._last_set or self._engage_pos
                if pos is None:
                    return {"visible": False, "x": -10000, "y": -10000,
                            "state": "hidden"}
                if self.phase == ENGAGING:
                    state = "prime"
                elif self.sub in (PENDING, DRAG, SCROLL):
                    state = "grab"
                else:
                    state = "track"
                return {"visible": True, "x": int(pos[0]), "y": int(pos[1]),
                        "state": state}
        except Exception:
            return {"visible": False, "x": -10000, "y": -10000, "state": "hidden"}

    def _blocked(self) -> str:
        if self.blocker is None:
            return ""
        try:
            return str(self.blocker() or "")
        except Exception:
            return ""

    def _two_hand_on(self) -> bool:
        if self.two_hand is None:
            return False
        try:
            return bool(self.two_hand())
        except Exception:
            return False

    def _recent_input(self, now: float) -> bool:
        if now < self._quiet_until:
            return True
        try:
            idle = self.out.idle_s()
        except Exception:
            idle = None
        return idle is not None and idle < self.p.input_quiet_s

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
                if self._two_hand_on():
                    return "ignored: two-hand mode"
                side = str(ev.get("side", ""))
                h = self._hands.get(side)
                st = self._tracks.get(side)
                if (h is None or st is None or self._frame_at is None
                        or now - self._frame_at > 0.5 or st.raw != "pinch"):
                    return "ignored: not pinching in a live frame"
                if not st.ready(now, self.p):
                    return "ignored: hand too small, unsure or just arrived"
                if st.state != "pinch" or now - st.conf.since < self.p.engage_pinch_s:
                    return "ignored: pinch not held long enough"
                if self._recent_input(now):
                    return "ignored: you are using the keyboard or mouse"
                self._reset_run()
                self.side = side
                self._owner = st
                self._owner_pos, self._owner_size = st.pos, st.size
                self._last_seen = now
                self._rect = self._screen_rect()
                try:
                    pos = self.out.position()
                    self._engage_pos = (int(pos[0]), int(pos[1])) if pos else None
                except Exception:
                    self._engage_pos = None
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
                self._note_tracks(hands, msg, now)
                self._hands = hands
                self._frame_at = now
                if self.phase != IDLE:
                    self._step(now)
        except Exception:
            pass

    def _note_tracks(self, hands: dict, msg: dict, now: float) -> None:
        fw, fh = msg.get("w") or 640, msg.get("h") or 480
        for side, h in hands.items():
            st = self._tracks.get(side)
            hid = h.get("id")
            if (st is None or now - st.last > self.p.track_gap_s
                    or (hid is not None and st.id is not None and hid != st.id)):
                st = _SideTrack(now, hid)        # a NEW hand on this side
                self._tracks[side] = st
            st.note(h, now, hand_size(h, fw, fh), self.p)

    def hands_in_view(self) -> int:
        """How many hands in the latest frame pass the score / size bar (settled
        or not): a second hand that big blocks a scroll, since two fists belong
        to the two-hand window grab."""
        return sum(1 for side in self._hands
                   if side in self._tracks and self._tracks[side].usable)

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
            self._tracks = {}
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

    def _owner_hand(self, now: float):
        """(hand dict, track) of the OWNER's hand in this frame, or (None, None).
        A new track on the owner's side is adopted only when it comes back where
        the owner's hand vanished; anything else is someone else's hand."""
        h = self._hands.get(self.side)
        st = self._tracks.get(self.side)
        if h is None or st is None:
            return None, None
        if st is not self._owner:
            near = (st.pos is not None and self._owner_pos is not None
                    and math.hypot(st.pos[0] - self._owner_pos[0],
                                   st.pos[1] - self._owner_pos[1])
                    <= self.p.reacquire_hs * max(self._owner_size, st.size, 0.05))
            if not (near and st.usable):
                return None, None
            self._owner = st                     # the owner's hand, back again
        if not st.usable:
            return None, None
        self._owner_pos, self._owner_size = st.pos, st.size
        return h, st

    def _step(self, now: float) -> None:
        if now >= self._next_block_check:
            self._next_block_check = now + self.p.block_check_s
            why = self._blocked()
            if why:
                self.disengage(why, now)
                return
        if self._two_hand_on():
            self.disengage("two-hand mode", now)
            return
        h, st = self._owner_hand(now)
        if h is None:
            if now - self._last_seen >= self.p.hand_gone_s:
                self.disengage("the hand left the view", now)
            return
        self._last_seen = now
        state = st.state                          # confirmed (debounced) state
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
                # the drag hold counts from when the pinch really began (its
                # debounce already took part of it)
                self._down_t = min(now, st.conf.since)
                return
            if state == "fist" and self.p.scroll_enabled:
                # the cursor holds still while a fist may become a scroll
                if (st.raw == "fist" and now - st.conf.since >= self.p.scroll_hold_s
                        and self.hands_in_view() == 1):
                    self._begin_scroll(now, st)
                return
            if st.raw in ("pinch", "fist") and st.raw != state:
                return                        # a pinch / fist forming: hold still
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
        elif self.sub == SCROLL:
            if state != "fist" or self.hands_in_view() > 1:
                self._end_scroll(fx, fy)
            else:
                self._scroll_step(now, st)

    # ── scroll ───────────────────────────────────────────────────────────
    def _begin_scroll(self, now: float, st: _SideTrack) -> None:
        self.sub = SCROLL
        self._scroll_y0 = st.pos[1] if st.pos else 0.0
        self._scroll_hs = max(st.size, 1e-3)
        self._scroll_sent = 0.0
        self._scroll_t = now
        self.scrolls += 1
        self._log("scrolling - move your fist up or down, open it to stop")

    def _scroll_step(self, now: float, st: _SideTrack) -> None:
        if st.pos is None:
            return
        dt = min(0.1, max(0.0, now - (self._scroll_t or now)))
        self._scroll_t = now
        dy = (st.pos[1] - self._scroll_y0) / self._scroll_hs     # hand sizes, + = down
        mag = abs(dy) - self.p.scroll_deadzone_hs
        target = 0.0 if mag <= 0 else math.copysign(mag * self.p.scroll_gain, dy)
        if not self.p.scroll_natural:
            target = -target
        step = target - self._scroll_sent
        cap = self.p.scroll_max_rate * dt
        step = max(-cap, min(cap, step))
        n = int(step)
        if abs(n) < self.p.scroll_min_step:
            return
        try:
            self.out.wheel(n)
        except Exception:
            pass
        self._scroll_sent += n
        self.wheel_total += n

    def _end_scroll(self, fx: float, fy: float) -> None:
        self.sub = FREE
        self._scroll_t = None
        # the hand moved while scrolling: carry on from where the cursor is
        self._reanchor(fx, fy)

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
                now = self._frame_prev if self._frame_prev is not None else 0.0
                self.disengage("you moved the mouse", None)
                # and no engaging again while he is using it
                self._quiet_until = now + self.p.input_quiet_s
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
