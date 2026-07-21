#!/usr/bin/env python3
"""
JARVIS Kinect TWO-HAND pinch-to-resize windows.

When BOTH hands are raised above the shoulder (the SAME raise-to-engage lift gate
the single-hand air-mouse uses), this poller takes over and lets the owner GRAB
the foreground window with both hands and resize / move it like a giant photo on a
touchscreen:

  • GRAB    — both hands engaged and HELD for ~0.2 s. Captures the focused window's
              rect + the initial 3D hand-distance.
  • RESIZE  — SPREAD the hands apart → the window GROWS; PINCH them together → it
              SHRINKS. The scale is the live hand-distance / the grab hand-distance
              (EMA-smoothed so it is NOT jittery), applied ABOUT THE WINDOW CENTRE
              so the window grows/shrinks in place.
  • MOVE    — slide both hands together (their MIDPOINT, screen-projected) → the
              window translates by the same screen delta.
  • RELEASE — a hand drops below the engage line / opens / the body is lost → the
              gesture finishes and the window is left where it is.

Both the smoothed hand-distance AND the resulting rect are EMA-smoothed, then the
window is moved with a single Win32 SetWindowPos per tick. The rect is clamped to a
sane minimum size and kept on-screen.

While TWO-HAND mode is active this poller publishes a heartbeat back to the
single-hand air-mouse (skills/kinect_air_mouse.set_two_hand_active) so the air-
mouse STANDS DOWN and the two don't fight over the cursor; and it publishes the two
hands' screen-projected positions to the air-cursor overlay so the HUD can draw the
TWO blue (purple while resizing) reticle circles (hud/jarvis_air_cursor.py).

GATING / SAFETY
  • Opt-in via KINECT_TWO_HAND_ENABLED (default True). Off → the poller idles.
  • NEVER acts on the staging/test instance (mirrors the air-mouse staging gate).
  • Targets ONLY a normal top-level foreground window; the shell / desktop /
    taskbar are skipped by class + title so a stray two-hand raise over the desktop
    can't resize the taskbar.
  • Pure geometry + the controller state machine are hardware-free and unit-tested
    directly; the Win32 calls are injected so the resize logic is tested with a
    mock Win32. NEVER raises out of the poll tick.

This reuses the single-hand air-mouse's sensor plumbing (skills/kinect_air_mouse:
_hand_sample / ReachBox / _reach_box_for_virtual_desktop / _dist3 / EMA / the lift
gate) as the single source of truth for reading the Kinect, so the engage gate +
mirror + projection match the air-mouse exactly.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════
#  TUNABLES
# ══════════════════════════════════════════════════════════════════════════
TWO_HAND_POLL_HZ = 30.0                       # match the air-mouse cadence (~30 Hz)
TWO_HAND_POLL_INTERVAL = 1.0 / TWO_HAND_POLL_HZ
INITIAL_DELAY_SECONDS = 6.5                   # after the monolith + bridge come up
#   (a touch after the air-mouse's 6.0 so the air-mouse module is importable first)
_THREAD_NAME = "kinect-two-hand-skill"

# GRAB hold: both hands must stay engaged this long before we capture the window +
# baseline distance, so a momentary two-hand raise (reaching past the sensor) does
# not snatch the window. ~0.2 s ≈ 6 frames at 30 Hz.
GRAB_HOLD_SEC = 0.20

# TWO-HAND DEAD-MAN (FILTER 3): symmetric to the air-mouse's DISENGAGE_GRACE_SEC.
# A grab may only LATCH on frames where BOTH hands are FULLY Tracked (TrackingState
# >= 2). If no fully-tracked both-hands frame is seen within this window WHILE
# grabbed — i.e. the grab is being held alive only by INFERRED/guessed hand joints —
# the controller force-reset()s, releasing the window. So a grab can't keep resizing
# on phantom hands the Kinect isn't actually seeing. ~0.30 s matches the air-mouse.
GRAB_DEADMAN_SEC = 0.30

# Distance + rect smoothing. The hand-distance EMA tames the Kinect's jittery hand
# joints (the owner's complaint about the old behaviour); the rect EMA keeps the
# window from twitching frame-to-frame. Higher = snappier / less smooth.
DIST_EMA_ALPHA = 0.40
RECT_EMA_ALPHA = 0.45

# A scale dead-band: hand-distance ratios within ±this of 1.0 are treated as "no
# resize" so tiny hand jitter while merely MOVING the window doesn't slowly creep
# its size. Below/above it the window scales.
SCALE_DEADBAND = 0.04

# Clamp: never shrink a window below this, and keep at least this many px on-screen
# so a grabbed window can't be lost off the desktop edge.
MIN_WINDOW_W = 240
MIN_WINDOW_H = 160
KEEP_ON_SCREEN_MARGIN = 32      # at least this much of the window stays on-desktop

# A sane upper bound on the per-tick scale step so a 1-frame distance glitch can't
# explode the window; the EMA already tames this, this is a hard backstop.
MAX_SCALE_STEP = 0.18           # ≤ ±18 % size change per tick

# OPEN-HANDS RELEASE debounce. The module docstring always promised "a hand …
# opens → release", but the per-hand grips _hand_sample already reads were
# thrown away (the poller's `_lg, _rg`), so the ONLY way to let go was to drop
# both arms below the engage line or lose tracking — the owner's "two-hand still
# doesn't like to unlatch" complaint. Now opening BOTH hands releases, but only
# a CONFIDENT sustained double-OPEN counts: a grip must read "open" on BOTH hands
# continuously for this long before the grab drops. "closed"/"lasso" and the
# flaky "unknown" the SDK emits for an occluded hand never force a release, and a
# 1–2 frame OPEN flicker mid-resize is swallowed — so this can only END a grab
# the owner is deliberately opening out of, never drop one they mean to hold.
OPEN_RELEASE_SEC = 0.15         # ~4–5 frames at 30 Hz


# ══════════════════════════════════════════════════════════════════════════
#  PURE GEOMETRY (no Win32, no sensor — unit-tested directly)
# ══════════════════════════════════════════════════════════════════════════
class Rect:
    """A window rectangle in virtual-desktop pixels: (left, top, right, bottom),
    right/bottom EXCLUSIVE (Win32 GetWindowRect convention). Immutable-ish value
    object with width/height/centre helpers. Pure."""

    __slots__ = ("left", "top", "right", "bottom")

    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    def as_tuple(self) -> "tuple[int, int, int, int]":
        return (self.left, self.top, self.right, self.bottom)

    # SetWindowPos takes (x, y, w, h) — the top-left + size.
    def as_xywh(self) -> "tuple[int, int, int, int]":
        return (self.left, self.top, self.width, self.height)

    def __eq__(self, other) -> bool:
        return isinstance(other, Rect) and self.as_tuple() == other.as_tuple()

    def __repr__(self) -> str:   # pragma: no cover - debug aid only
        return (f"Rect(l={self.left}, t={self.top}, r={self.right}, "
                f"b={self.bottom}, {self.width}x{self.height})")


def scale_rect_about_center(rect: "Rect", scale: float,
                            *, min_w: int = MIN_WINDOW_W,
                            min_h: int = MIN_WINDOW_H) -> "Rect":
    """Scale `rect` by `scale` ABOUT ITS CENTRE (the centre stays put; the window
    grows/shrinks in place). The new size is floored at (min_w, min_h) so a hard
    pinch can't collapse the window. Pure.

    A scale of 1.0 returns the same size; >1 grows (SPREAD), <1 shrinks (PINCH)."""
    s = max(1e-3, float(scale))
    cx, cy = rect.cx, rect.cy
    new_w = max(int(min_w), int(round(rect.width * s)))
    new_h = max(int(min_h), int(round(rect.height * s)))
    left = int(round(cx - new_w / 2.0))
    top = int(round(cy - new_h / 2.0))
    return Rect(left, top, left + new_w, top + new_h)


def translate_rect(rect: "Rect", dx: float, dy: float) -> "Rect":
    """Move `rect` by (dx, dy) pixels. Pure."""
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    return Rect(rect.left + dx_i, rect.top + dy_i,
                rect.right + dx_i, rect.bottom + dy_i)


def clamp_rect_on_screen(rect: "Rect", bounds: "tuple[int, int, int, int]",
                         *, margin: int = KEEP_ON_SCREEN_MARGIN) -> "Rect":
    """Nudge `rect` so at least `margin` px of it stays within the virtual-desktop
    `bounds` (origin_x, origin_y, width, height) on every edge — the window can't be
    dragged/shrunk fully off the desktop. Only TRANSLATES (size is preserved); the
    size floor is handled by scale_rect_about_center. Pure."""
    ox, oy, w, h = bounds
    desk_left, desk_top = int(ox), int(oy)
    desk_right, desk_bottom = int(ox) + int(w), int(oy) + int(h)
    m = int(margin)
    rw, rh = rect.width, rect.height
    left, top = rect.left, rect.top
    # The window's left can't go so far right that < margin remains on the left edge,
    # nor so far left that the window's right edge is < margin onto the desktop.
    min_left = desk_left - (rw - m)      # right edge keeps `m` px on-desktop
    max_left = desk_right - m            # left edge keeps `m` px on-desktop
    if min_left > max_left:              # window wider than desktop → pin to origin
        left = desk_left
    else:
        left = max(min_left, min(max_left, left))
    min_top = desk_top - (rh - m)
    max_top = desk_bottom - m
    if min_top > max_top:
        top = desk_top
    else:
        top = max(min_top, min(max_top, top))
    return Rect(left, top, left + rw, top + rh)


def clamp_scale_step(prev_scale: float, target_scale: float,
                     *, max_step: float = MAX_SCALE_STEP) -> float:
    """Limit how far the (already EMA-smoothed) scale may move toward its target in
    one tick — a hard backstop so a 1-frame distance glitch can't explode/collapse
    the window even if the EMA briefly tracks it. Pure."""
    prev = max(1e-3, float(prev_scale))
    target = max(1e-3, float(target_scale))
    lo = prev * (1.0 - max_step)
    hi = prev * (1.0 + max_step)
    return max(lo, min(hi, target))


def apply_deadband(ratio: float, *, deadband: float = SCALE_DEADBAND) -> float:
    """Snap a hand-distance ratio within ±deadband of 1.0 to exactly 1.0, so tiny
    hand jitter while MOVING the window doesn't slowly creep its size. Pure."""
    r = float(ratio)
    if abs(r - 1.0) <= float(deadband):
        return 1.0
    return r


# ══════════════════════════════════════════════════════════════════════════
#  TWO-HAND CONTROLLER — the grab → resize/move → release state machine.
#  PURE: fed (hands engaged?, hand-distance, hand-midpoint screen-pixel, focused
#  window rect, desktop bounds) each tick; emits the rect the window SHOULD have
#  (or None when not grabbing). No sensor, no Win32, no clock except an injected
#  monotonic — so the whole resize logic is unit-tested directly.
# ══════════════════════════════════════════════════════════════════════════
class TwoHandDecision:
    """One tick's outcome. `active` is True whenever BOTH hands are engaged (so the
    air-mouse should stand down even during the pre-grab hold); `rect` is the window
    rect to apply this tick (None when not resizing/moving a window); `resizing` is
    True once a window is actually grabbed (drives the PURPLE reticle); `hands` is
    the two screen points to draw the reticles at (or None)."""

    __slots__ = ("active", "rect", "resizing", "phase", "hands")

    def __init__(self, active: bool, rect: "Optional[Rect]", resizing: bool,
                 phase: str, hands: "Optional[tuple]" = None):
        self.active = bool(active)
        self.rect = rect
        self.resizing = bool(resizing)
        self.phase = phase            # "idle" | "holding" | "grabbed"
        self.hands = hands            # ((lx, ly), (rx, ry)) screen px, or None


class TwoHandController:
    """The grab/resize/move/release state machine. Holds the captured window rect +
    baseline hand-distance + midpoint while grabbed, EMA-smooths the scale + rect,
    and emits the rect the window should have each tick.

    Inject `clock` (monotonic) for the grab-hold timing in tests."""

    def __init__(self, *, grab_hold_sec: float = GRAB_HOLD_SEC,
                 dist_alpha: float = DIST_EMA_ALPHA,
                 rect_alpha: float = RECT_EMA_ALPHA,
                 min_w: int = MIN_WINDOW_W, min_h: int = MIN_WINDOW_H,
                 keep_margin: int = KEEP_ON_SCREEN_MARGIN,
                 deadman_sec: float = GRAB_DEADMAN_SEC,
                 open_release_sec: float = OPEN_RELEASE_SEC,
                 clock=time.monotonic):
        self._grab_hold_sec = max(0.0, float(grab_hold_sec))
        self._dist_alpha = max(0.0, min(1.0, float(dist_alpha)))
        self._rect_alpha = max(0.0, min(1.0, float(rect_alpha)))
        self._min_w = int(min_w)
        self._min_h = int(min_h)
        self._keep_margin = int(keep_margin)
        self._deadman_sec = max(0.0, float(deadman_sec))
        self._open_release_sec = max(0.0, float(open_release_sec))
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Drop all grab state (used on release / disable)."""
        self._phase = "idle"          # "idle" | "holding" | "grabbed"
        self._engaged_since: Optional[float] = None
        self._grab_dist: Optional[float] = None     # baseline 3D hand-distance
        self._grab_rect: Optional[Rect] = None      # window rect at grab time
        self._grab_mid: Optional[tuple] = None      # midpoint screen-px at grab
        self._smoothed_dist: Optional[float] = None
        self._scale: float = 1.0                    # last applied scale
        self._smoothed_rect: Optional[Rect] = None
        # FILTER 3 (dead-man): clock of the last frame where BOTH hands were FULLY
        # Tracked. While grabbed, if this ages past _deadman_sec the grab is force-
        # released (it's being held alive only by inferred/guessed hands).
        self._last_confirmed_grab_at: Optional[float] = None
        # FILTER 6 (body-id pin): the id of the body that took the grab; a change
        # under us is treated as a loss (release) rather than a silent retarget.
        self._grab_body_id = None
        # OPEN-HANDS RELEASE: clock of the first frame in the current run of
        # BOTH-hands-open while grabbed. None whenever the hands aren't both
        # confidently open; when it ages past _open_release_sec the grab drops.
        self._both_open_since: Optional[float] = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_grabbed(self) -> bool:
        return self._phase == "grabbed"

    def _ema(self, prev: Optional[float], x: float, alpha: float) -> float:
        if prev is None:
            return float(x)
        return alpha * float(x) + (1.0 - alpha) * prev

    def _ema_rect(self, prev: Optional["Rect"], target: "Rect") -> "Rect":
        if prev is None:
            return target
        a = self._rect_alpha
        return Rect(
            int(round(a * target.left + (1 - a) * prev.left)),
            int(round(a * target.top + (1 - a) * prev.top)),
            int(round(a * target.right + (1 - a) * prev.right)),
            int(round(a * target.bottom + (1 - a) * prev.bottom)),
        )

    def update(self, *, both_engaged: bool, hand_dist: "Optional[float]",
               midpoint: "Optional[tuple]", focused_rect: "Optional[Rect]",
               bounds: "tuple[int, int, int, int]",
               hands: "Optional[tuple]" = None, body_id=None,
               left_grip: str = "unknown",
               right_grip: str = "unknown") -> "TwoHandDecision":
        """Advance one frame.

        both_engaged: True when BOTH hands clear the raise-to-engage lift gate AND
                      are FULLY Tracked (FILTER 2 — _both_hands_engaged now requires
                      the TrackingState floor), i.e. a CONFIRMED both-hands frame.
        hand_dist:    the live 3D distance between the two hands (metres), or None.
        midpoint:     the two-hand midpoint projected to a screen pixel (x, y), or
                      None — used for the MOVE translation.
        focused_rect: the foreground window's rect at GRAB time (only read on the
                      grab edge; None → nothing to grab).
        bounds:       (origin_x, origin_y, w, h) of the virtual desktop, for the
                      keep-on-screen clamp.
        hands:        the two hands' screen points for the reticles (passed through).
        body_id:      the id of the controlling body (FILTER 6). Pinned on the grab
                      edge; a change while grabbed releases (no silent retarget).
        left_grip/right_grip: the per-hand grips ("open"/"closed"/"lasso"/"unknown",
                      lower-case). Opening BOTH hands releases the grab (debounced by
                      OPEN_RELEASE_SEC); a both-open frame also refuses to LATCH a new
                      grab, so you grab with fists and let go by opening your palms.
                      Default "unknown" preserves the pre-grip behaviour for callers
                      (and tests) that don't pass grips.

        Returns a TwoHandDecision: `rect` is the window rect to apply this tick (None
        when not actively resizing/moving), `resizing` True once grabbed."""
        now = self._clock()

        # ── OPEN-HANDS RELEASE (highest priority while grabbed). Opening BOTH
        #    hands is the natural "drop the giant photo" gesture. Only a CONFIDENT
        #    sustained double-OPEN counts: "closed"/"lasso"/"unknown" never trip it,
        #    and the _open_release_sec debounce swallows a 1–2 frame OPEN flicker
        #    mid-resize — so a deliberate hold is never dropped, only a deliberate
        #    release. Checked before the geometry paths so an opening hand wins even
        #    while it is still geometrically up and feeding a distance. ────────────
        both_open = (str(left_grip).lower() == "open"
                     and str(right_grip).lower() == "open")
        if self._phase == "grabbed" and both_open:
            if self._both_open_since is None:
                self._both_open_since = now
            elif (now - self._both_open_since) >= self._open_release_sec:
                self.reset()
                return TwoHandDecision(active=False, rect=None, resizing=False,
                                       phase="idle", hands=hands)
        else:
            self._both_open_since = None

        # ── HARD RELEASE: the hands are GONE this tick (a hand dropped below the
        #    line / opened, or the body was lost), so there's no hand data at all.
        #    Finish the gesture immediately — a deliberate hand-drop releases the
        #    window at once (no grace; the FILTER 3 grace below is ONLY for a brief
        #    INFERRED flicker while the hands are still geometrically up). ─────────
        if hand_dist is None or midpoint is None:
            self.reset()
            return TwoHandDecision(active=False, rect=None, resizing=False,
                                   phase="idle", hands=hands)

        # ── DEAD-MAN GRACE (FILTER 3, symmetric to the air-mouse tracking grace):
        #    the hands are still geometrically present but this frame is NOT a
        #    CONFIRMED fully-tracked both-hands frame (one hand flickered to an
        #    INFERRED joint, so _both_hands_engaged read False). While GRABBED, HOLD
        #    the current rect rather than drop the window — but ONLY until
        #    _deadman_sec since the last fully-tracked confirmation. Past that the
        #    grab is being kept alive by phantom/inferred hands → force release. A
        #    grab can therefore never LATCH or persist on inferred frames beyond the
        #    dead-man window. (Holding pre-grab just stays idle/holding → release.)
        if not both_engaged:
            if (self._phase == "grabbed" and self._deadman_sec > 0.0
                    and self._last_confirmed_grab_at is not None
                    and (now - self._last_confirmed_grab_at) <= self._deadman_sec):
                return TwoHandDecision(active=True, rect=self._smoothed_rect,
                                       resizing=True, phase="grabbed", hands=hands)
            self.reset()
            return TwoHandDecision(active=False, rect=None, resizing=False,
                                   phase="idle", hands=hands)

        # CONFIRMED both-hands frame (both fully Tracked + raised): stamp it for the
        # FILTER 3 dead-man so a later brief dropout is graced from HERE.
        self._last_confirmed_grab_at = now

        # ── BODY-ID PIN (FILTER 6): while grabbed, a change of the controlling body
        #    id (a closer 2nd person took the nearest slot) is a tracking-loss —
        #    release rather than seamlessly retarget the resize onto the interloper.
        #    body_id None (caller didn't supply one) disables the pin. ─────────────
        if (self._phase == "grabbed" and self._grab_body_id is not None
                and body_id is not None and body_id != self._grab_body_id):
            self.reset()
            return TwoHandDecision(active=False, rect=None, resizing=False,
                                   phase="idle", hands=hands)

        # ── Both hands engaged. Start (or continue) the grab-hold timer. ────────
        if self._phase == "idle":
            self._phase = "holding"
            self._engaged_since = now
            self._smoothed_dist = float(hand_dist)
            return TwoHandDecision(active=True, rect=None, resizing=False,
                                   phase="holding", hands=hands)

        if self._phase == "holding":
            # Keep smoothing the distance through the hold so the baseline captured
            # at grab is already settled (no jump on the first resize frame).
            self._smoothed_dist = self._ema(self._smoothed_dist, hand_dist,
                                            self._dist_alpha)
            held = (self._engaged_since is not None
                    and (now - self._engaged_since) >= self._grab_hold_sec)
            if not held:
                return TwoHandDecision(active=True, rect=None, resizing=False,
                                       phase="holding", hands=hands)
            # ── GRAB EDGE: capture the window rect + baseline distance + midpoint.
            if focused_rect is None:
                # No grabbable window under the hands — stay "holding" (active, so
                # the air-mouse still stands down) but grab nothing.
                return TwoHandDecision(active=True, rect=None, resizing=False,
                                       phase="holding", hands=hands)
            if both_open:
                # You grab with FISTS, not open palms: a both-open frame parks in
                # "holding" instead of snatching the window. This keeps the
                # open-hands RELEASE above unambiguous (no grab→instant-release
                # oscillation when the owner raises open hands) — close the fists
                # to latch. "unknown"/one-open still grabs, so nothing regresses
                # for callers that don't report grips.
                return TwoHandDecision(active=True, rect=None, resizing=False,
                                       phase="holding", hands=hands)
            self._phase = "grabbed"
            self._grab_dist = max(1e-3, float(self._smoothed_dist or hand_dist))
            self._grab_rect = focused_rect
            self._grab_mid = (float(midpoint[0]), float(midpoint[1]))
            self._smoothed_rect = focused_rect
            self._scale = 1.0
            self._grab_body_id = body_id   # FILTER 6: pin the controlling body
            # First grabbed frame: leave the window exactly where it is.
            return TwoHandDecision(active=True, rect=focused_rect, resizing=True,
                                   phase="grabbed", hands=hands)

        # ── GRABBED: resize about the centre + translate by the midpoint delta. ─
        self._smoothed_dist = self._ema(self._smoothed_dist, hand_dist,
                                        self._dist_alpha)
        # SCALE = smoothed live distance / the baseline at grab (with a dead-band so
        # a still hand while MOVING doesn't creep the size), step-limited + EMA'd.
        raw_ratio = (self._smoothed_dist or self._grab_dist) / self._grab_dist
        target_scale = apply_deadband(raw_ratio)
        target_scale = clamp_scale_step(self._scale, target_scale)
        self._scale = target_scale

        # Resize the ORIGINAL grabbed rect about its centre (absolute from the
        # baseline, so the size always reflects the current spread — no drift).
        resized = scale_rect_about_center(self._grab_rect, self._scale,
                                          min_w=self._min_w, min_h=self._min_h)
        # MOVE: translate by how far the hand-midpoint has moved since grab.
        dx = float(midpoint[0]) - self._grab_mid[0]
        dy = float(midpoint[1]) - self._grab_mid[1]
        moved = translate_rect(resized, dx, dy)
        # Keep it on-screen, then EMA the final rect so the window glides (not jitter).
        clamped = clamp_rect_on_screen(moved, bounds, margin=self._keep_margin)
        self._smoothed_rect = self._ema_rect(self._smoothed_rect, clamped)
        # Re-clamp after the EMA so smoothing can't nudge it back off-screen.
        out = clamp_rect_on_screen(self._smoothed_rect, bounds,
                                   margin=self._keep_margin)
        self._smoothed_rect = out
        return TwoHandDecision(active=True, rect=out, resizing=True,
                               phase="grabbed", hands=hands)


# ══════════════════════════════════════════════════════════════════════════
#  AIR-MOUSE plumbing bridge (reuse its sensor read + projection + state)
# ══════════════════════════════════════════════════════════════════════════
def _air_mouse_mod():
    """The loaded single-hand air-mouse skill module (skill_kinect_air_mouse), or
    None. Prefer the instance load_skills() already imported; fall back to a direct
    import so we work under the test harness / a standalone import too. NEVER
    raises."""
    mod = sys.modules.get("skill_kinect_air_mouse")
    if mod is not None:
        return mod
    # Fall back to importing the file directly (e.g. running this module alone).
    try:
        return importlib.import_module("skills.kinect_air_mouse")
    except Exception:
        try:
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "kinect_air_mouse.py")
            spec = importlib.util.spec_from_file_location(
                "skill_kinect_air_mouse", path)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules["skill_kinect_air_mouse"] = m
                spec.loader.exec_module(m)
                return m
        except Exception:
            # Roll back the pre-exec insert — a failed exec must not leave a
            # half-initialized module for the next sys.modules.get() to trust
            # (same rule as load_skills' failure path).
            sys.modules.pop("skill_kinect_air_mouse", None)
            return None
    return None


def _bridge():
    """The live kinect_bridge (via the air-mouse module's resolver), or None."""
    am = _air_mouse_mod()
    if am is None:
        return None
    try:
        return am._bridge()
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh each
    call so a Settings toggle takes effect with no restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — two-hand resize must NEVER move a real
    window there. Reuses the air-mouse staging gate (same env + monolith check)."""
    am = _air_mouse_mod()
    if am is not None:
        try:
            return bool(am._is_staging())
        except Exception:
            pass
    return os.environ.get("JARVIS_STAGING", "").strip() == "1"


def _two_hand_enabled() -> bool:
    """The master gate: opt-in flag ON (default True) and not staging."""
    return _cfg_flag("KINECT_TWO_HAND_ENABLED", True) and not _is_staging()


def _publish_air_mouse_standdown(active: bool) -> None:
    """Tell the single-hand air-mouse to STAND DOWN (or resume) — it reads this and
    suppresses its cursor while two-hand mode drives. Best-effort."""
    am = _air_mouse_mod()
    if am is None:
        return
    try:
        am.set_two_hand_active(bool(active))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  SENSOR READ — both hands' 3D positions + the engage gate (reuse the air-mouse)
# ══════════════════════════════════════════════════════════════════════════
def _both_hands_engaged(am, left_ext, right_ext, thresholds) -> bool:
    """True when BOTH arms clear the air-mouse's raise-to-engage lift gate (the same
    height hysteresis the single-hand cursor uses). `engaged=False` asks the strict
    ENGAGE bar (not the looser stay-engaged), so a deliberate two-hand RAISE is
    required to enter the mode.

    HAND-STATE FLOOR (FILTER 2): BOTH hand joints must be sensor-TRACKED
    (TrackingState >= 2 — finite, non-zero) before any grab / _dist3 / midpoint math
    runs. An INFERRED (state 1) 2nd hand — the noisy guess the SDK emits for an
    occluded / phantom hand — can NOT grab a window or feed a jittery distance: it
    reads not-engaged. (The shoulder-ref tracking floor is covered transitively —
    an untracked shoulder ref leaves lift_m None via the air-mouse FILTER 1, so
    is_extended() already returns False.) NEVER raises."""
    try:
        if left_ext is None or right_ext is None:
            return False
        if left_ext.hand is None or right_ext.hand is None:
            return False
        if not (_joint_tracked(am, left_ext.hand)
                and _joint_tracked(am, right_ext.hand)):
            return False
        return bool(
            left_ext.is_extended(engaged=False, thresholds=thresholds)
            and right_ext.is_extended(engaged=False, thresholds=thresholds))
    except Exception:
        return False


def _joint_tracked(am, joint) -> bool:
    """Whether `joint` is sensor-TRACKED, via the air-mouse's pure joint_well_tracked
    helper (single source of truth — TrackingState >= 2, finite, non-zero). Falls
    back to True if an older air-mouse build lacks the helper, so two-hand still
    works against it (degrades to the prior behaviour). NEVER raises."""
    try:
        fn = getattr(am, "joint_well_tracked", None)
        if callable(fn):
            return bool(fn(joint))
    except Exception:
        pass
    return True


def _controlling_body_id(am):
    """The id of the NEAREST body from the most recent am._hand_sample() (FILTER 6
    body-id pin). Read off the air-mouse module's _last_body_id stash so the two
    skills agree on which body is in control. None when unavailable / no body.
    NEVER raises."""
    try:
        holder = getattr(am, "_last_body_id", None)
        if isinstance(holder, list) and holder:
            return holder[0]
    except Exception:
        pass
    return None


def _hand_distance(am, left_ext, right_ext) -> "Optional[float]":
    """The 3D distance between the two hand joints (metres), via the air-mouse's
    _dist3. None when either hand is missing. NEVER raises."""
    try:
        return am._dist3(left_ext.hand, right_ext.hand)
    except Exception:
        return None


def _project_hand(reach, hand) -> "Optional[tuple]":
    """Project a hand (x, y, z, ...) to a virtual-desktop pixel via the air-mouse's
    fixed-centre ReachBox (the SAME projection the single-hand cursor uses, so the
    reticles land where the air-mouse would put the cursor). None on failure."""
    try:
        return reach.map(float(hand[0]), float(hand[1]))
    except Exception:
        return None


def _midpoint(p_left, p_right) -> "Optional[tuple]":
    """Screen-pixel midpoint of the two projected hand points. None if either is
    missing."""
    if p_left is None or p_right is None:
        return None
    return ((p_left[0] + p_right[0]) / 2.0, (p_left[1] + p_right[1]) / 2.0)


# ══════════════════════════════════════════════════════════════════════════
#  WIN32 LAYER — foreground-window get/move (injected so the core is testable)
# ══════════════════════════════════════════════════════════════════════════
# Window styles / class names we must NOT resize: the desktop shell, the taskbar,
# and tool/popup windows. Skipping these by class + title means a stray two-hand
# raise over the desktop can never grab the taskbar or the wallpaper.
_SKIP_CLASSES = {
    "Progman",            # the desktop "Program Manager"
    "WorkerW",            # the desktop wallpaper host
    "Shell_TrayWnd",      # the primary taskbar
    "Shell_SecondaryTrayWnd",   # secondary-monitor taskbars
    "Button",             # the Start button
    "DV2ControlHost",     # the old Start menu host
    "Windows.UI.Core.CoreWindow",   # Start / search / action-center surfaces
    "ForegroundStaging",
    "XamlExplorerHostIslandWindow",
    "MultitaskingViewFrame",        # task view / alt-tab
}
_SKIP_TITLE_SUBSTR = ("Program Manager",)


def _get_foreground_hwnd():   # pragma: no cover - thin ctypes wrapper
    """The HWND of the current foreground window, or 0. Win32 only."""
    try:
        import ctypes
        return int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        return 0


def _window_class_name(hwnd) -> str:   # pragma: no cover - thin ctypes wrapper
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)
        return buf.value or ""
    except Exception:
        return ""


def _window_title(hwnd) -> str:   # pragma: no cover - thin ctypes wrapper
    try:
        import ctypes
        n = int(ctypes.windll.user32.GetWindowTextLengthW(hwnd))
        if n <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value or ""
    except Exception:
        return ""


def is_resizable_target(class_name: str, title: str, *,
                        is_window: bool = True, visible: bool = True,
                        minimized: bool = False) -> bool:
    """True when a foreground window is a normal, grabbable top-level window — i.e.
    NOT the shell / desktop / taskbar / a minimized or hidden window. PURE (class +
    title + flags in, bool out) so the skip-list is unit-tested with no Win32."""
    if not is_window or not visible or minimized:
        return False
    cls = (class_name or "").strip()
    if cls in _SKIP_CLASSES:
        return False
    ttl = title or ""
    for sub in _SKIP_TITLE_SUBSTR:
        if sub and sub in ttl:
            return False
    return True


def _get_window_rect(hwnd) -> "Optional[Rect]":   # pragma: no cover - ctypes
    """GetWindowRect(hwnd) → Rect, or None. Win32 only."""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ok = ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
        if not ok:
            return None
        return Rect(int(r.left), int(r.top), int(r.right), int(r.bottom))
    except Exception:
        return None


# SetWindowPos flags: don't change Z-order, don't activate (so grabbing doesn't
# steal focus / pop the window forward distractingly), and apply async-friendly.
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_NOOWNERZORDER = 0x0200
_SWP_FLAGS = _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_NOOWNERZORDER


def _set_window_pos(hwnd, rect: "Rect") -> bool:   # pragma: no cover - ctypes
    """Move + size a window to `rect` via SetWindowPos. Win32 only. Returns True on
    success. NEVER raises."""
    try:
        import ctypes
        x, y, w, h = rect.as_xywh()
        ok = ctypes.windll.user32.SetWindowPos(
            int(hwnd), 0, int(x), int(y), int(w), int(h), _SWP_FLAGS)
        return bool(ok)
    except Exception:
        return False


def _foreground_target_rect() -> "tuple[object, Optional[Rect]]":
    """(hwnd, Rect) for the current foreground window when it's a grabbable target,
    else (hwnd, None). Win32 only; (0, None) off Windows / on error. NEVER raises.
    This is the GRAB-time capture of what's under the hands."""
    try:
        hwnd = _get_foreground_hwnd()
        if not hwnd:
            return 0, None
        cls = _window_class_name(hwnd)
        ttl = _window_title(hwnd)
        try:
            import ctypes
            u = ctypes.windll.user32
            is_win = bool(u.IsWindow(hwnd))
            visible = bool(u.IsWindowVisible(hwnd))
            minimized = bool(u.IsIconic(hwnd))
        except Exception:
            is_win = visible = True
            minimized = False
        if not is_resizable_target(cls, ttl, is_window=is_win,
                                   visible=visible, minimized=minimized):
            return hwnd, None
        return hwnd, _get_window_rect(hwnd)
    except Exception:
        return 0, None


# ══════════════════════════════════════════════════════════════════════════
#  DUAL-RETICLE state publishing (Part 3 — two circles, blue/purple)
# ══════════════════════════════════════════════════════════════════════════
# Module-list (lock-free GIL mutation, per house style): True while our last
# published frame was an ACTIVE two-hand frame. Lets us clear the two-hand keys
# exactly ONCE on the active→inactive edge instead of re-writing a hidden frame
# every ~30 Hz tick — which fought the air-mouse poller (also ~30 Hz on the same
# AIR_CURSOR_STATE_FILE) and strobed the reticle.
_two_hand_overlay_was_active = [False]


def _ensure_overlay_alive() -> None:
    """Make sure the air-cursor overlay SUBPROCESS (hud/jarvis_air_cursor.py) is
    actually running. The air-mouse's own poll keeps it alive, but ONLY while
    KINECT_AIR_MOUSE_ENABLED is on — and two-hand is ON by default while the
    air-mouse is OFF, so in the shipped config nobody would ever spawn the window
    that draws our dual reticles. Reuse the air-mouse's _overlay_alive /
    _spawn_overlay (single source of truth for the spawn contract); skip silently
    if the module or helpers are missing, and NEVER spawn on the staging/test
    instance (same gate the air-mouse respects). NEVER raises."""
    try:
        if _is_staging():
            return
        am = _air_mouse_mod()
        if am is None:
            return
        alive = getattr(am, "_overlay_alive", None)
        spawn = getattr(am, "_spawn_overlay", None)
        if not callable(alive) or not callable(spawn):
            return
        if not alive():
            spawn()
    except Exception:
        pass


def _atomic_write_overlay_json(path: str, data: dict) -> None:
    """Write `data` to `path` via a temp file + os.replace, mirroring the
    air-mouse's hardened writer (2026-07-14 bug-hunt #23 — this was the stale
    duplicate that still did a plain open()/json.dump, which the 60 Hz reader
    could catch mid-write as a torn frame). Never raises."""
    import json as _json
    import tempfile
    d = os.path.dirname(path) or "."
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".2hand_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        if tmp is not None:
            try:
                os.remove(tmp)
            except Exception:
                pass


def _publish_two_hand_overlay(decision: "TwoHandDecision") -> None:
    """Publish the TWO hands' screen points + the resize state to the air-cursor
    overlay's state file so the HUD draws two reticle circles (BLUE while engaged,
    PURPLE while actively resizing a window). Reuses the air-mouse's
    AIR_CURSOR_STATE_FILE (the overlay already reads it). Best-effort + silent.

    Shape (extends the single-hand schema additively — old single-hand keys kept so
    nothing else breaks):
        {"visible": bool, "ts": epoch,
         "two_hand": True,
         "hands": [{"x": int, "y": int}, {"x": int, "y": int}],
         "resizing": bool,
         "color": "purple"|"blue"}"""
    am = _air_mouse_mod()
    if am is None:
        return
    try:
        import json
        path = am.AIR_CURSOR_STATE_FILE
        hands = decision.hands
        if not decision.active or not hands or hands[0] is None or hands[1] is None:
            # Not in two-hand mode → the air-mouse owns the file for the
            # single-hand reticle. Clear our two-hand keys exactly ONCE on the
            # active→inactive edge (a hidden two-hand frame); after that write
            # nothing, so we don't fight the air-mouse's ~30 Hz writes and
            # strobe the reticle. The air-mouse's next single-hand frame
            # overwrites this cleared frame within a tick.
            if _two_hand_overlay_was_active[0]:
                data = {"visible": False, "two_hand": False, "ts": time.time()}
                _atomic_write_overlay_json(path, data)
                _two_hand_overlay_was_active[0] = False
            return
        _two_hand_overlay_was_active[0] = True
        (lx, ly), (rx, ry) = hands
        color = "purple" if decision.resizing else "blue"
        data = {
            "visible": True,
            "two_hand": True,
            "resizing": bool(decision.resizing),
            "color": color,
            # Keep the single-hand keys present + sane so any reader that only knows
            # the old schema still sees a valid (first-hand) point rather than crash.
            "x": int(round(lx)), "y": int(round(ly)),
            "state": "grab" if decision.resizing else "track",
            "hands": [
                {"x": int(round(lx)), "y": int(round(ly))},
                {"x": int(round(rx)), "y": int(round(ry))},
            ],
            "ts": time.time(),
        }
        _atomic_write_overlay_json(path, data)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  LIVE POLL — read both hands → decide → move the foreground window
# ══════════════════════════════════════════════════════════════════════════
_grab_hwnd = [0]    # module-list: the hwnd captured on the current grab (or 0)


def _poll_once(ctrl: "TwoHandController",
               *, foreground_target=_foreground_target_rect,
               set_window_pos=_set_window_pos) -> "Optional[TwoHandDecision]":
    """One two-hand tick: read both hands via the air-mouse plumbing, run the
    controller, and (only when enabled + not staging) ACT — move/size the foreground
    window, publish the air-mouse stand-down + the dual reticle. Returns the decision
    (for tests) or None when the sensor plumbing is absent. NEVER raises.

    `foreground_target` / `set_window_pos` are injected so the resize wiring is
    tested with a mock Win32."""
    am = _air_mouse_mod()
    if am is None:
        return None
    bridge = _bridge()
    if bridge is None:
        return None
    try:
        left_ext, right_ext, _lg, _rg, tracked = am._hand_sample(bridge)
    except Exception:
        left_ext = right_ext = None
        _lg = _rg = "unknown"
        tracked = False

    thresholds = None
    try:
        thresholds = am._reach_thresholds()
    except Exception:
        thresholds = None

    both = bool(tracked) and _both_hands_engaged(am, left_ext, right_ext, thresholds)
    # The controlling body's id (stashed by am._hand_sample above) for the FILTER 6
    # pin — so a closer 2nd person can't steal the grab mid-resize. None if absent.
    body_id = _controlling_body_id(am)

    # Project both hands to screen + compute the distance/midpoint. We do this both
    # when CONFIRMED (`both`) AND, while already GRABBED, whenever both hand joints
    # are still geometrically PRESENT (even if this frame's tracking is inferred) —
    # so the controller's FILTER 3 dead-man grace receives hand data during a brief
    # inferred flicker (and can HOLD the rect), rather than seeing it as 'hands gone'
    # and hard-releasing. If the hands are truly gone, hand_dist/mid stay None and
    # the controller releases at once.
    hands_present = (left_ext is not None and right_ext is not None
                     and left_ext.hand is not None and right_ext.hand is not None)
    try:
        reach = am._reach_box_for_virtual_desktop()
    except Exception:
        reach = None
    p_left = p_right = None
    hand_dist = None
    mid = None
    if (both or (ctrl.is_grabbed and hands_present)) and reach is not None:
        p_left = _project_hand(reach, left_ext.hand)
        p_right = _project_hand(reach, right_ext.hand)
        hand_dist = _hand_distance(am, left_ext, right_ext)
        mid = _midpoint(p_left, p_right)
    hands = (p_left, p_right) if (p_left is not None and p_right is not None) else None

    # Desktop bounds for the keep-on-screen clamp.
    try:
        bounds = am._cached_virtual_bounds()
    except Exception:
        bounds = (0, 0, 2560, 1440)

    enabled = _two_hand_enabled()
    if not enabled:
        # Gated off → make sure we're not holding the air-mouse down and we drop any
        # in-flight grab. Publish ONE inactive frame so the reticle/standdown clear.
        if ctrl.phase != "idle" or _grab_hwnd[0]:
            ctrl.reset()
            _grab_hwnd[0] = 0
        _publish_air_mouse_standdown(False)
        return TwoHandDecision(active=False, rect=None, resizing=False,
                               phase="idle", hands=None)

    # On the GRAB edge we need the foreground window rect. We only query Win32 when
    # both hands are up AND we don't already hold a grab (cheap; avoids a per-tick
    # foreground query while idle).
    focused_rect = None
    if both and not ctrl.is_grabbed:
        try:
            hwnd, focused_rect = foreground_target()
        except Exception:
            hwnd, focused_rect = 0, None
        if focused_rect is not None:
            _grab_hwnd[0] = hwnd

    decision = ctrl.update(both_engaged=both, hand_dist=hand_dist, midpoint=mid,
                           focused_rect=focused_rect, bounds=bounds, hands=hands,
                           body_id=body_id, left_grip=_lg, right_grip=_rg)

    # STAND DOWN the single-hand air-mouse whenever two-hand mode is active (incl.
    # the pre-grab hold) so the two never fight the cursor.
    _publish_air_mouse_standdown(decision.active)

    # ACT: move/size the grabbed window.
    if decision.resizing and decision.rect is not None and _grab_hwnd[0]:
        try:
            set_window_pos(_grab_hwnd[0], decision.rect)
        except Exception:
            pass
    if not decision.resizing:
        _grab_hwnd[0] = 0

    # Publish the dual reticle (two circles; purple while resizing) — and make
    # sure the overlay process that DRAWS it exists: with the air-mouse disabled
    # (the shipped default) nothing else would ever spawn it.
    _publish_two_hand_overlay(decision)
    if decision.active:
        _ensure_overlay_alive()
    return decision


def _poll_loop() -> None:  # pragma: no cover - non-terminating daemon; each tick delegates to _poll_once, which is unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    am = _air_mouse_mod()
    if am is None:
        print("  [two-hand] air-mouse plumbing unavailable — poller exiting")
        return
    if _bridge() is None:
        print("  [two-hand] kinect_bridge unavailable — poller exiting")
        return
    ctrl = TwoHandController()
    was_active = False
    while True:
        try:
            decision = _poll_once(ctrl)
            active = bool(decision and decision.active)
            if active and not was_active:
                print("  [two-hand] both hands raised — TWO-HAND mode engaged "
                      "(air-mouse standing down)")
            elif was_active and not active:
                print("  [two-hand] released — TWO-HAND mode off")
            was_active = active
        except Exception as e:
            print(f"  [two-hand] poll error: {e}")
        time.sleep(TWO_HAND_POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════
#  Registration — auto-starts the poller (gated, off-cheap), like the air-mouse
# ══════════════════════════════════════════════════════════════════════════
def register(actions):
    # No spoken actions for now — the mode is gesture-only. (The air-mouse on/off
    # commands govern the Kinect cursor stack as a whole.) Just start the poller.
    if any(th.name == _THREAD_NAME and th.is_alive()
           for th in threading.enumerate()):
        print("  [two-hand] poller already running — skipping duplicate (reload)")
        return
    t = threading.Thread(target=_poll_loop, daemon=True, name=_THREAD_NAME)
    t.start()
    print(f"  [two-hand] pinch-to-resize poller active (~{TWO_HAND_POLL_HZ:.0f} Hz; "
          "both-hands raise to grab; KINECT_TWO_HAND_ENABLED, on by default)")
