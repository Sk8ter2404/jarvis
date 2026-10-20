"""Webcam TWO-HAND window grab (2026-10-20): move / resize a window with both
hands, seen by the MediaPipe hand tracker - the webcam twin of the Kinect's
skills/kinect_two_hand.py.

Only with AIR_MOUSE_ENGINE = "hands" and HAND_TWO_HAND_ENABLED (default OFF;
core/config.py). skills/hand_tracking.py builds one TwoHandGrab and feeds it
the tracker's "hands" messages.

THE GESTURE
  GRAB    BOTH hands usable (core.hand_mouse's recognition rules: tracker score,
          hand size, settle time, debounced state) and BOTH closed into FISTS,
          held for kinect_two_hand.GRAB_HOLD_SEC (0.5 s) -> the top-level window
          UNDER THE MIDPOINT of the two hands is grabbed. Fists, not pinches:
          a held pinch is the hand mouse's engage gesture, and the tracker's
          fist state (all four fingers curled) is the one it reports most
          reliably for two hands at once.
  MOVE    move both fists -> the window follows the midpoint.
  RESIZE  spread them -> it grows; bring them together -> it shrinks (about
          its centre, proportional to the hand distance).
  RELEASE open EITHER hand, or lose either hand -> it ends (after the
          controller's short dead-man grace), the window stays where it is.

WHAT IS REUSED, NOT COPIED (skills/kinect_two_hand.py): the TwoHandController
state machine (hold timer, distance / rect smoothing, dead-band, step limit,
keep-on-screen clamp, dead-man, open-hands release), the window targeting with
its shell / desktop / taskbar skip-list (_target_rect_at, the under-the-point
variant of its foreground grab), SetWindowPos (_set_window_pos), the dual
reticle publisher (_publish_two_hand_overlay) and the overlay spawn
(_ensure_overlay_alive - never in staging). The webcam only replaces the
sensor read: hand points come from the tracker, mapped to the screen through
the hand mouse's box (core.hand_mouse.Params.box), like the cursor.

While active() is True the single-hand cursor (core/hand_mouse.py) stands
down - exactly as the Kinect air mouse does for the Kinect two-hand mode.
Every OS effect goes through the injected `win` object, so tests drive the
whole thing with fakes. Never raises.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Tuple

from core import hand_mouse as hm

LOG = "  [two-hand]"
ACTIVE_TTL_S = 0.5          # no hands message for this long = not active
GRIPS = {"fist": "closed", "open": "open"}   # tracker state -> controller grip


def _kinect_two_hand():
    """skills/kinect_two_hand.py (the loaded skill instance when there is one)."""
    import sys
    mod = sys.modules.get("skill_kinect_two_hand")
    if mod is not None:
        return mod
    import importlib
    return importlib.import_module("skills.kinect_two_hand")


class WindowsBackend:
    """The real effects, all from skills/kinect_two_hand.py. Constructing it
    touches nothing."""

    def __init__(self, kt=None):
        self._kt = kt

    @property
    def kt(self):
        if self._kt is None:
            self._kt = _kinect_two_hand()
        return self._kt

    def controller(self):
        return self.kt.TwoHandController()

    def window_at(self, x: float, y: float):
        return self.kt._target_rect_at(x, y)

    def set_window_pos(self, hwnd, rect) -> bool:
        return bool(self.kt._set_window_pos(hwnd, rect))

    def screen_rect(self) -> Tuple[int, int, int, int]:
        """The primary monitor (the hand mouse box maps onto it)."""
        try:
            w, h = self.kt._air_mouse_mod()._primary_screen_size()
            return (0, 0, int(w), int(h))
        except Exception:
            return (0, 0, 1920, 1080)

    def bounds(self) -> Tuple[int, int, int, int]:
        try:
            am = self.kt._air_mouse_mod()
            return tuple(int(v) for v in am._cached_virtual_bounds())
        except Exception:
            return (0, 0, 1920, 1080)

    def publish(self, decision, edge: list) -> None:
        """The dual reticle (None = clear it); spawns the overlay while active."""
        if decision is None:
            decision = self.kt.TwoHandDecision(False, None, False, "idle")
        self.kt._publish_two_hand_overlay(decision, edge=edge)
        if decision.active:
            self.kt._ensure_overlay_alive()


class TwoHandGrab:
    """Feed it the tracker's "hands" messages; it moves / resizes windows."""

    def __init__(self, win=None, *, params: Optional[hm.Params] = None,
                 blocker: Optional[Callable[[], str]] = None,
                 recent_input: Optional[Callable[[], bool]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.win = win or WindowsBackend()
        self.p = params or hm.Params()
        self.blocker = blocker           # game mode / fullscreen / call / per-app
        self.recent_input = recent_input  # the owner is typing / using the mouse
        self._log = log or (lambda s: print(f"{LOG} {s}"))
        self._lock = threading.RLock()
        self.ctrl = self.win.controller()
        self._tracks: dict = {}          # side -> hand_mouse._SideTrack
        self._hwnd = 0
        self._edge = [False]             # our own dual-reticle edge flag
        self._active = False
        self._at = -1e9
        self.grabs = 0
        self.last: Optional[object] = None

    # ── state ────────────────────────────────────────────────────────────
    def active(self, now: Optional[float] = None) -> bool:
        """True while both fists are up (holding or grabbed) and fresh: the
        single-hand cursor must stand down."""
        now = time.monotonic() if now is None else now
        return self._active and now - self._at <= ACTIVE_TTL_S

    def stop(self) -> None:
        """Drop any grab and clear the reticles (feature off / tracker gone)."""
        with self._lock:
            try:
                self.ctrl.reset()
            except Exception:
                pass
            self._hwnd = 0
            self._tracks = {}
            was = self._active
            self._active = False
            if was or self._edge[0]:
                self._publish(None)

    # ── input ────────────────────────────────────────────────────────────
    def on_hands(self, msg: dict, now: Optional[float] = None):
        """One parsed {"type":"hands"} message. Returns the controller's
        decision (tests / logs), or None. Never raises."""
        try:
            with self._lock:
                return self._step(msg, time.monotonic() if now is None else now)
        except Exception:
            return None

    def _blocked(self) -> bool:
        for fn in (self.blocker, self.recent_input):
            if fn is None:
                continue
            try:
                if fn():
                    return True
            except Exception:
                continue
        return False

    def _screen_pt(self, pos, rect) -> Tuple[float, float]:
        """Mirrored normalised frame point -> screen pixel, through the hand
        mouse's box (the same mapping as the cursor)."""
        l, t, w, h = rect
        x0, y0, x1, y1 = self.p.box
        u = min(1.0, max(0.0, (pos[0] - x0) / (x1 - x0)))
        v = min(1.0, max(0.0, (pos[1] - y0) / (y1 - y0)))
        return l + u * max(0, w - 1), t + v * max(0, h - 1)

    def _step(self, msg: dict, now: float):
        fw, fh = msg.get("w") or 640, msg.get("h") or 480
        hands = {}
        for h in (msg.get("hands") or []):
            if isinstance(h, dict):
                side = str(h.get("side", ""))
                if side in ("Left", "Right") and (side not in hands or float(
                        h.get("score", 0) or 0) > float(hands[side].get("score", 0) or 0)):
                    hands[side] = h
        for side, h in hands.items():
            st = self._tracks.get(side)
            hid = h.get("id")
            if (st is None or now - st.last > self.p.track_gap_s
                    or (hid is not None and st.id is not None and hid != st.id)):
                st = hm._SideTrack(now, hid)
                self._tracks[side] = st
            st.note(h, now, hm.hand_size(h, fw, fh), self.p)
        lt, rt = self._tracks.get("Left"), self._tracks.get("Right")
        present = ("Left" in hands and "Right" in hands and lt is not None
                   and rt is not None and lt.pos is not None and rt.pos is not None)
        ready = present and lt.ready(now, self.p) and rt.ready(now, self.p)
        both_fists = ready and lt.state == "fist" and rt.state == "fist"
        grabbed = self.ctrl.is_grabbed
        hand_dist = mid = pts = None
        if present and (both_fists or grabbed):
            rect = self.p.screen_rect
            if rect is None:
                try:
                    rect = tuple(self.win.screen_rect())
                except Exception:
                    rect = (0, 0, 1920, 1080)
            pl, pr = self._screen_pt(lt.pos, rect), self._screen_pt(rt.pos, rect)
            pts = (pl, pr)
            mid = ((pl[0] + pr[0]) / 2.0, (pl[1] + pr[1]) / 2.0)
            hand_dist = math.hypot((lt.pos[0] - rt.pos[0]) * fw,
                                   (lt.pos[1] - rt.pos[1]) * fh)
        grab_allowed = not self._blocked()
        focused = None
        if both_fists and grab_allowed and not grabbed and self.ctrl.phase == "holding":
            try:
                hwnd, focused = self.win.window_at(mid[0], mid[1])
            except Exception:
                hwnd, focused = 0, None
            if focused is not None:
                self._hwnd = hwnd
        grips = (GRIPS.get(lt.state, "unknown") if lt else "unknown",
                 GRIPS.get(rt.state, "unknown") if rt else "unknown")
        try:
            bounds = self.win.bounds()
        except Exception:
            bounds = (0, 0, 1920, 1080)
        decision = self.ctrl.update(both_engaged=both_fists, hand_dist=hand_dist,
                                    midpoint=mid, focused_rect=focused,
                                    bounds=bounds, hands=pts,
                                    left_grip=grips[0], right_grip=grips[1],
                                    grab_allowed=grab_allowed)
        if decision.resizing and not grabbed:
            self.grabs += 1
            self._log("grabbed the window between your hands - move or "
                      "spread them, open a hand to let go")
        if grabbed and not decision.resizing:
            self._log("let go of the window")
        if decision.resizing and decision.rect is not None and self._hwnd:
            try:
                self.win.set_window_pos(self._hwnd, decision.rect)
            except Exception:
                pass
        if not decision.resizing:
            self._hwnd = 0
        self._active = bool(decision.active)
        self._at = now
        self.last = decision
        self._publish(decision)
        return decision

    def _publish(self, decision) -> None:
        try:
            self.win.publish(decision, self._edge)
        except Exception:
            pass
