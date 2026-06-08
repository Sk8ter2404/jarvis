"""
kinect_air_mouse skill — a Kinect v2 "air-mouse" for JARVIS.

THE FEATURE
===========
RAISE an OPEN hand toward the screen to move the cursor; CLOSE the hand to
LEFT-click; hold it closed to drag; LOWER the hand to let go of the cursor.
Concretely:

  • hand RAISED, OPEN   → move the cursor (no button held).
  • close → open fast   → a LEFT-click (button down then up, cursor parked).
  • close, move, open   → a left-DRAG (button stays down while the hand is
                          closed, releases on re-open).
  • hand LOWERED / lost → DISENGAGE — the cursor is released so the PHYSICAL
                          mouse works again, any held button is let go, and the
                          reticle hides. Raise the hand to re-engage.

A glowing JARVIS targeting reticle (hud/jarvis_air_cursor.py, a separate
click-THROUGH overlay process) follows the cursor — cyan + gently pulsing while
TRACKING a raised open hand, snapping inward to a GOLD lock on grab/drag, and
HIDDEN while disengaged — so the owner always sees where their hand is pointing.

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
  • DEAD-MAN / ENGAGE GATE: the cursor is driven ONLY while the controlling hand
    is actively RAISED above a body reference (the spine-mid / chest, with a
    fall-back to waist / elbow / shoulder) AND the body+hand are tracked. The
    moment the owner LOWERS the hand below that reference — or the hand/body goes
    untracked for more than DISENGAGE_GRACE_SEC (~0.3 s) — the air-mouse
    DISENGAGES: it stops calling SetCursorPos entirely (releasing control so the
    PHYSICAL mouse works again), releases any held button, and hides the reticle.
    A raise/drop hysteresis band keeps it from flickering at the threshold, and a
    closed hand that drops or leaves the frame can never strand the button down.

LEFT vs RIGHT click: this is a ONE-LINE swap — change AIR_MOUSE_BUTTON below
from "left" to "right". The owner wants the PRIMARY (LEFT) button.

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

# Which mouse button the closed-hand grab actuates. The owner wants the PRIMARY
# (LEFT) button: a quick close→open is a normal left-CLICK, and close-move-open
# is a left-DRAG. _mouse_button() maps this to MOUSEEVENTF_LEFTDOWN/LEFTUP (win32)
# or pyautogui's left button. Swap to "right" here for a right-click air-mouse.
AIR_MOUSE_BUTTON = "left"

# EMA smoothing factor for the cursor (0..1). LOWER = smoother but laggier;
# HIGHER = snappier but jitterier. Tunable; v2 may auto-adapt it to hand speed.
#
# TUNED 2026-06-08 (owner: "works but laggy, make it snappier"): 0.35 → 0.55.
#   At the 30 Hz poll rate the EMA's response time-constant is ~1/alpha frames.
#   0.35 → ≈2.9-frame constant (~95 ms to reach 63 % of a step) which read as a
#   cursor "dragging behind" the hand. 0.55 → ≈1.8-frame constant (~60 ms): the
#   cursor catches the hand markedly faster (≈35 ms less lag) while still
#   averaging roughly two frames, so the Kinect hand-joint jitter is still tamed
#   (a lone jittered sample only moves the cursor ~55 % of the way, not 100 %).
#   Nudge back toward 0.4-0.45 if it feels twitchy; toward 0.6-0.7 for even less
#   lag at the cost of more jitter.
AIR_MOUSE_EMA_ALPHA = 0.55

# How many CONSECUTIVE frames a new grip (open↔closed) must persist before the
# state machine accepts it. This is the anti-stray-click guard: a single
# flickered Kinect hand-state frame must never fire a click.
#
# TUNED 2026-06-08 (owner: "click should fire promptly"): 3 → 2.
#   At 30 Hz, 3 frames ≈ 100 ms of latency before a close registers as a click;
#   2 frames ≈ 67 ms — the click fires ~33 ms sooner so it feels prompt/instant.
#   TWO consecutive frames still rejects a lone 1-frame flicker (the actual
#   failure mode the Kinect produces), so accidental clicks are still prevented;
#   we only gave up the third confirmation frame, which was belt-and-suspenders.
#   Raise back to 3 if any stray clicks appear; 2 is the snappy-but-safe floor.
AIR_MOUSE_GRIP_DEBOUNCE_FRAMES = 2

# ─── dead-man ENGAGE gate (the hand must be RAISED to drive the cursor) ──────
# The air-mouse only controls the cursor while the controlling hand is held UP,
# above a body reference (the spine-mid / chest height by default, see
# _engage_reference_y). Raising the hand ENGAGES; lowering it (or losing the
# hand) DISENGAGES — releasing the OS cursor so the real mouse works again. Two
# thresholds give HYSTERESIS so a hand hovering right at the line can't flap the
# engage state on/off every frame:
#   • to ENGAGE   the hand must rise to >= ref_y + ENGAGE_RAISE_M  (clearly up)
#   • to STAY engaged it may sag only to  ref_y + DISENGAGE_DROP_M (a lower bar);
#     dropping below that DISENGAGES.
# Camera-space y increases UPWARD (metres), so "above the reference" = a LARGER
# y. The gap between the two is the dead-band. Defaults: engage ~8 cm above the
# chest reference, release once it sags to ~2 cm above it.
AIR_MOUSE_ENGAGE_RAISE_M = 0.08     # rise this far above ref_y to ENGAGE
AIR_MOUSE_DISENGAGE_DROP_M = 0.02   # sag below ref_y + this to DISENGAGE

# Grace window for a TRACKING dropout. A single lost/ambiguous frame (the Kinect
# briefly drops the body or hand joint) must NOT instantly disengage and re-snap
# — that would make the cursor jump and a drag stutter. While engaged, a dropout
# is tolerated for up to this long (button stays held, no cursor motion since
# there's no sample); past it the dead-man fully releases. ~0.3 s per the spec.
AIR_MOUSE_DISENGAGE_GRACE_SEC = 0.30

# The comfortable reach-box in front of the user, in camera-space METRES, that
# maps onto the whole virtual desktop. Centred roughly on where a seated user's
# hand naturally sits when pointing at the screen. x: sensor-RIGHT is +; the box
# is wider than tall to match a 16:9 screen. y: sensor-UP is +; centred near
# shoulder height. These are the v1 defaults; v2 makes them per-user calibrated.
#   half-width  → ±X metres from centre maps to the desktop's left/right edges
#   half-height → ±Y metres from centre maps to the desktop's top/bottom edges
#
# TUNED 2026-06-08 (owner: "shouldn't need huge arm swings to cross the screen").
#   The smaller the box, the LESS hand travel maps to the full desktop, i.e.
#   higher sensitivity. The old ±0.35 m / ±0.22 m box demanded a ~70 cm-wide
#   sweep edge-to-edge — a whole-arm shoulder swing. A natural pointing arc with
#   the elbow tucked (forearm pivoting at the elbow/wrist) is only ~±25 cm
#   horizontal / ~±15 cm vertical, so:
#     REACH_HALF_W 0.35 → 0.26  (full desktop width  in a ~52 cm hand sweep)
#     REACH_HALF_H 0.22 → 0.16  (full desktop height in a ~32 cm hand sweep)
#   That maps a comfortable forearm arc to the whole virtual desktop — small
#   hand moves now cover the screen. The ~1.6 W:H ratio is kept ≈16:9 so x and y
#   sensitivity stay proportionate (no axis feels twitchier than the other).
#   The EMA + debounce above keep this from feeling jittery despite the higher
#   gain. Widen these (toward the old values) if it feels too sensitive; shrink
#   further for even less travel.
REACH_CENTER_X = 0.0      # metres (centred on the sensor's optical axis)
REACH_CENTER_Y = 0.30     # metres above the sensor (≈ seated shoulder height)
REACH_HALF_W = 0.26       # ±0.26 m horizontal reach → full desktop width
REACH_HALF_H = 0.16       # ±0.16 m vertical reach → full desktop height

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
    """The pure per-frame brain. Holds the smoothing + debounce + ENGAGE state
    and turns each (hand_pos, raw_grip, tracked, ref_y) sample into an
    AirMouseDecision. NO I/O — the live loop applies the decision (move cursor,
    press button, publish overlay state). Re-buildable cheaply; reset() on
    disable / hand-loss.

    DEAD-MAN ENGAGE GATE (the cursor only moves while the hand is held UP):
      The controller is ENGAGED only while the controlling hand is RAISED above
      the body reference ref_y (chest/spine-mid by default) AND a body+hand are
      tracked. A raise/drop hysteresis band (AIR_MOUSE_ENGAGE_RAISE_M /
      AIR_MOUSE_DISENGAGE_DROP_M) stops it flapping at the line. While
      DISENGAGED — hand lowered below the reference, OR untracked beyond the
      ~0.3 s grace — the decision carries cursor=None (so the live loop calls NO
      SetCursorPos and the PHYSICAL mouse is free), releases any held button, and
      hides the overlay. Re-raising the hand re-engages and the EMA re-snaps.

    Button semantics (LEFT by default), only while ENGAGED:
      • stable grip OPEN  → cursor moves; overlay "track"; no button change.
      • OPEN → CLOSED edge → emit button "down"; overlay flips to "grab".
      • stays CLOSED       → cursor STILL moves (so a closed hand DRAGS); overlay
                             holds "grab". (close→move→open = a left-DRAG.)
      • CLOSED → OPEN edge  → emit button "up"; overlay back to "track".
        (a quick close→open with no move = a left-CLICK.)
    """

    def __init__(self, reach: ReachBox,
                 alpha: float = AIR_MOUSE_EMA_ALPHA,
                 debounce_frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES,
                 engage_raise_m: float = AIR_MOUSE_ENGAGE_RAISE_M,
                 disengage_drop_m: float = AIR_MOUSE_DISENGAGE_DROP_M,
                 grace_sec: float = AIR_MOUSE_DISENGAGE_GRACE_SEC,
                 clock=time.monotonic):
        self.reach = reach
        self._ema_x = EMA(alpha)
        self._ema_y = EMA(alpha)
        self._grip = GripDebouncer(debounce_frames, initial="open")
        self._button_down = False
        # Engage gate state.
        self._engage_raise_m = float(engage_raise_m)
        self._disengage_drop_m = float(disengage_drop_m)
        self._grace_sec = max(0.0, float(grace_sec))
        self._clock = clock          # injectable monotonic clock (tests)
        self._engaged = False        # is the air-mouse currently driving the cursor
        self._last_engaged_at = 0.0  # clock() of the last RAISED+tracked frame

    def reset(self) -> None:
        """Drop all smoothing + grip + engage state. Used by the dead-man and on
        disable so the next hand starts clean (no cursor sweep from a stale
        value, no phantom button edge, freshly DISENGAGED)."""
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip.reset(initial="open")
        # NB: this does NOT itself emit a button-up — the caller (dead-man) is
        # responsible for releasing a held button. We only clear our own view.
        self._button_down = False
        self._engaged = False
        self._last_engaged_at = 0.0

    @property
    def button_is_down(self) -> bool:
        return self._button_down

    @property
    def engaged(self) -> bool:
        """True while the air-mouse is actively driving the cursor (hand raised
        + tracked). False while disengaged (hand lowered / lost) — in which state
        update() returns cursor=None so the physical mouse is free."""
        return self._engaged

    def _is_raised(self, hand_y: float, ref_y: float) -> bool:
        """Apply the raise/drop HYSTERESIS to decide if the hand counts as
        'raised' this frame. When currently DISENGAGED the hand must clear the
        higher ENGAGE bar (ref_y + engage_raise_m); once ENGAGED it only has to
        stay above the lower DISENGAGE bar (ref_y + disengage_drop_m). Camera y
        is UP-positive, so 'above' = greater-than. The gap between the two bars
        is the dead-band that prevents threshold flicker."""
        if self._engaged:
            return hand_y >= (ref_y + self._disengage_drop_m)
        return hand_y >= (ref_y + self._engage_raise_m)

    def release_decision(self) -> AirMouseDecision:
        """The DEAD-MAN / disengaged decision: if a button was held, command it
        UP; hide the overlay; clear smoothing + grip + engage so the next
        acquisition snaps. cursor=None so the live loop issues NO SetCursorPos
        and the physical mouse is free. Idempotent — once released, repeated
        calls just keep the overlay hidden with no button edge."""
        button = "up" if self._button_down else None
        self._button_down = False
        self._engaged = False
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip.reset(initial="open")
        return AirMouseDecision(cursor=None, button=button,
                                overlay="hidden", grip="open")

    def update(self, hand_xy, raw_grip: str, tracked: bool,
               ref_y: Optional[float] = None) -> AirMouseDecision:
        """Advance one frame.

        hand_xy: (x, y) camera-space metres, or None when no hand sample.
        raw_grip: the raw bridge grip ("open"/"closed"/"lasso"/"unknown").
        tracked: True when the bridge reported a tracked body this frame.
        ref_y:   the body engage-reference height in camera-space metres (chest /
                 spine-mid, see _engage_reference_y), or None when it couldn't be
                 read. The hand must be RAISED above this (with hysteresis) to
                 drive the cursor.

        DISENGAGES (returns cursor=None — no SetCursorPos — and releases any held
        button) when the body/hand is NOT tracked, when there's no hand sample,
        when no reference is available, or when the hand is LOWERED below the
        reference. A brief tracking dropout while ENGAGED is tolerated for up to
        the grace window (button held, cursor parked) before the full release."""
        # ── tracking-loss path, with a short grace so a 1-frame dropout doesn't
        #    disengage + re-snap (a held drag must survive a flicker). ──────────
        if not tracked or hand_xy is None:
            if (self._engaged and self._grace_sec > 0.0
                    and (self._clock() - self._last_engaged_at) <= self._grace_sec):
                # Brief dropout: hold. No sample → no cursor motion; keep any held
                # button and the current overlay. Do NOT refresh the engage clock,
                # so a sustained dropout still ages out into a full release.
                overlay = "grab" if self._button_down else "track"
                return AirMouseDecision(cursor=None, button=None,
                                        overlay=overlay, grip=self._grip.stable)
            return self.release_decision()

        # ── engage gate: the hand must be RAISED above the body reference. No
        #    reference (couldn't read a spine/elbow joint) → can't confirm a
        #    raise → disengage (fail SAFE to releasing the real mouse). ─────────
        if ref_y is None or not self._is_raised(float(hand_xy[1]), float(ref_y)):
            return self.release_decision()

        # ── ENGAGED. On the rising edge (was disengaged) snap the smoothing to
        #    the new hand position so the cursor doesn't sweep from a stale value.
        if not self._engaged:
            self._ema_x.reset()
            self._ema_y.reset()
        self._engaged = True
        self._last_engaged_at = self._clock()

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
#  PREVIEW FEEDBACK: hand-circle colour + thread-safe air-mouse state (B2)
# ══════════════════════════════════════════════════════════════════════════
# The HUD's Kinect skeleton preview (bobert_companion._compose_kinect_preview)
# draws a translucent circle around the controlling hand joint, coloured by the
# LIVE air-mouse state so the owner SEES when the cursor is active / clicking:
#   • ENGAGED + open hand   → BLUE   (cursor active, tracking)
#   • ENGAGED + closed hand → ORANGE (left-click / drag held)
#   • disengaged / off      → faint GREY idle (or no circle)
# The colour logic is a PURE helper (no cv2 / sensor) so the geometry module and
# the unit test assert the exact same contract; the preview just calls it.

# BGR triples (OpenCV order) for the three hand-circle states. Stark, saturated
# colours that read on the small downscaled preview tile.
HAND_CIRCLE_COLOR_ENGAGED = (255, 160, 32)   # BGR ≈ bright BLUE  (#20a0ff)
HAND_CIRCLE_COLOR_CLOSED  = (32, 170, 255)   # BGR ≈ ORANGE/amber (#ffaa20)
HAND_CIRCLE_COLOR_IDLE    = (150, 150, 150)  # dim GREY (disengaged idle hint)


def hand_circle_color_for(engaged: bool, grip: str) -> "tuple[int, int, int] | None":
    """The hand-circle BGR colour for the preview, by air-mouse state. PURE +
    hardware-free so the preview and the test share one source of truth.

      • engaged + grip "closed"        → ORANGE  (left-click / drag active)
      • engaged + any other grip       → BLUE    (cursor engaged, tracking)
      • not engaged                    → GREY    (faint idle hint)

    Returns a (B, G, R) tuple. The caller decides idle→draw-faint-or-skip; we
    return the idle grey so the colour mapping itself stays total + testable."""
    if engaged:
        if (grip or "").lower() == "closed":
            return HAND_CIRCLE_COLOR_CLOSED
        return HAND_CIRCLE_COLOR_ENGAGED
    return HAND_CIRCLE_COLOR_IDLE


# Thread-safe snapshot of the LIVE air-mouse engage state + current grip, written
# by _poll_once each tick and read by the HUD preview compositor (a DIFFERENT
# thread — the face-tracking loop). A tiny lock keeps the (engaged, grip, ts)
# triple consistent. The preview reads it to decide the hand-circle colour and
# whether to draw the circle at all; stale reads simply paint the last state.
_air_mouse_state_lock = threading.Lock()
_air_mouse_state: dict = {"engaged": False, "grip": "open", "ts": 0.0}


def _set_air_mouse_state(engaged: bool, grip: str) -> None:
    """Publish the live engage state + grip for the preview (thread-safe)."""
    with _air_mouse_state_lock:
        _air_mouse_state["engaged"] = bool(engaged)
        _air_mouse_state["grip"] = (grip or "open")
        _air_mouse_state["ts"] = time.time()


def get_air_mouse_state() -> dict:
    """Thread-safe snapshot {'engaged': bool, 'grip': str, 'ts': float} of the
    air-mouse. Read by the HUD skeleton preview to colour the hand circle
    (engaged→blue, closed→orange, off→grey). Returns a COPY so the caller can't
    mutate the shared dict. Never raises."""
    with _air_mouse_state_lock:
        return dict(_air_mouse_state)


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
# Joint names, in priority order, used as the ENGAGE REFERENCE the hand must be
# raised ABOVE to drive the cursor. spine_mid (mid-chest) first so a resting hand
# at the side / in the lap reads as DOWN while a hand lifted toward the screen
# reads as UP; fall back through waist (spine_base) and the arm joints if the
# torso isn't tracked, and finally the side-specific elbow. All are normally
# present on a tracked Kinect skeleton; if none are, _hand_sample returns ref=None
# and the controller fails safe to DISENGAGED.
_ENGAGE_REF_JOINTS = ("spine_mid", "spine_base", "spine_shoulder",
                      "shoulder_left", "shoulder_right")


def _engage_reference_y(joints: dict, side: str) -> Optional[float]:
    """The camera-space y (metres, UP-positive) the hand must clear to ENGAGE.
    Prefers the chest/spine, then the controlling arm's elbow as a final
    fall-back, reading the y component of whichever reference joint is present.
    Returns None when no reference joint can be read (→ controller disengages).
    NEVER raises."""
    try:
        for name in _ENGAGE_REF_JOINTS:
            j = joints.get(name)
            if j and len(j) >= 2:
                return float(j[1])
        # Last resort: the elbow of the gripping arm (a low bar, but better than
        # no reference at all — a hand below its own elbow is clearly lowered).
        elbow = joints.get(f"elbow_{side}")
        if elbow and len(elbow) >= 2:
            return float(elbow[1])
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _hand_sample(bridge) -> tuple[Optional[tuple], str, bool, Optional[float]]:
    """Read the pointing hand from the bridge:
    (hand_xy, raw_grip, tracked, ref_y).

    hand_xy is the nearest body's pointing-hand (x, y) in camera-space metres
    (None when no body / no usable hand joint). raw_grip is that body's grip for
    the SAME hand. tracked is whether a body was in view. ref_y is the engage
    reference height (chest/spine, see _engage_reference_y) the hand must be
    raised above to drive the cursor, or None when it can't be read. Never raises
    — any failure degrades to (None, "unknown", False, None) which the controller
    treats as a dead-man release."""
    try:
        if not bridge.get_enabled():
            return None, "unknown", False, None
        ok, _reason = bridge.available()
        if not ok:
            return None, "unknown", False, None
        bodies = bridge.get_bodies()
    except Exception:
        return None, "unknown", False, None
    if not bodies:
        return None, "unknown", False, None

    # Nearest body (same ranking the rest of the stack uses).
    def _key(b):
        d = b.get("distance_m") if isinstance(b, dict) else None
        return d if isinstance(d, (int, float)) and d > 0 else float("inf")
    try:
        body = min((b for b in bodies if isinstance(b, dict)), key=_key)
    except (TypeError, ValueError):
        return None, "unknown", False, None

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
        hand2 = joints.get(f"hand_{other}") or joints.get(f"wrist_{other}")
        if hand2 and len(hand2) >= 2:
            hand = hand2
            grip = left_grip if other == "left" else right_grip
            side = other
    # The engage reference is read off the SAME tracked body regardless of
    # whether a hand joint was found, so a body in view with the hand lowered out
    # of the joint set still disengages cleanly.
    ref_y = _engage_reference_y(joints, side)
    if not hand or len(hand) < 2:
        return None, grip, True, ref_y   # body tracked but no hand joint
    return (float(hand[0]), float(hand[1])), grip, True, ref_y


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
    hand_xy, raw_grip, tracked, ref_y = _hand_sample(bridge)
    try:
        decision = ctrl.update(hand_xy, raw_grip, tracked, ref_y)
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
        # Publish DISENGAGED to the preview: with the air-mouse off the HUD
        # hand-circle must show idle/grey (or nothing), never a live blue/orange.
        _set_air_mouse_state(False, "open")
        return decision

    # Enabled + not staging: act.
    _apply_decision(decision)
    visible = tracked and decision.overlay != "hidden"
    _publish_overlay_state(decision, visible)
    # Publish the LIVE engage state + grip for the HUD skeleton preview's hand
    # circle (B2): engaged→blue, closed→orange, disengaged→grey. Read off the
    # controller (engaged) + the debounced decision (grip) so the preview colour
    # matches what the cursor/reticle are actually doing this frame.
    try:
        _set_air_mouse_state(bool(ctrl.engaged), decision.grip)
    except Exception:
        pass
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
    msg = ("Air-mouse on, sir — raise an open hand toward the screen to move the "
           "cursor, close your hand to click, hold it closed to drag, and lower "
           "your hand to release the cursor.")
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
    how = ("raise an open hand to move the cursor, close to click, hold closed "
           "to drag, and lower your hand to release the cursor")
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
