"""
kinect_air_mouse skill — a Kinect v2 "air-mouse" for JARVIS.

THE FEATURE  (REACH-TO-ENGAGE model)
====================================
EXTEND an arm OUT toward the screen/Kinect — a deliberate REACH — to take the
cursor; the cursor follows the EXTENDED hand. CLOSE a hand to click, and which
hand clicks which button is HAND-SPECIFIC. When you RELAX the arm (pull the hand
back toward your body / bend the elbow) the air-mouse DISENGAGES and gestures
take over. A hand merely raised / visible does NOTHING — only a reach engages.

  • arm EXTENDED out (hand pushed FORWARD in depth and/or the arm straightened)
                          → ENGAGE: the cursor follows that hand.
  • LEFT hand closes      → LEFT mouse button (down on close, up on open; hold-
                            closed + move = a LEFT-drag).
  • RIGHT hand closes     → RIGHT mouse button (right-click; hold = right-drag).
                            Either hand can click regardless of which one is
                            driving the cursor.
  • arm RELAXED / lost    → DISENGAGE — the cursor is released so the PHYSICAL
                            mouse works again, any held button is let go, the
                            reticle hides, and GESTURES re-arm. Reach out again
                            to re-engage.

So: reach out = mouse (left/right hand = left/right click), relax = gestures,
hand merely up = nothing.

A glowing JARVIS targeting reticle (hud/jarvis_air_cursor.py, a separate
click-THROUGH overlay process) follows the cursor — cyan + gently pulsing while
TRACKING the extended hand, snapping inward to a GOLD lock on a grab/drag, and
HIDDEN while disengaged.

This module is the LIVE WIRING; the testable core is pure and lives right here
alongside it (no sensor, no real mouse, no Qt needed to exercise it):

  • ReachBox + map_hand_to_cursor() — turn a hand position (camera-space metres)
    into an absolute VIRTUAL-DESKTOP pixel (spanning ALL monitors), clamped to
    the desktop bounds.
  • EMA — exponential smoothing to fight the Kinect's hand-joint jitter.
  • GripDebouncer — the per-hand open/closed state machine: requires N
    consecutive frames of a new grip before it flips, tolerates 1-frame Unknown
    dropouts (carries the last confident grip), and treats Lasso as closed — so a
    single flickered frame never fires a stray click and a fist reliably clicks /
    an open hand reliably releases.
  • ArmExtension thresholds — the REACH gate: forward-depth + arm-straightness,
    with engage/disengage HYSTERESIS so a hand hovering at the threshold can't
    flap, plus a short tracking-loss GRACE so a 1-frame dropout doesn't strand a
    held button.
  • AirMouseController — ties those together into a per-frame decision:
    cursor_xy, per-hand button edges (left/right down|up), overlay state, the
    controlling hand, and the per-hand grips — for the HUD preview hand-circle.

V1 MAPPING (deliberately simple + robust)
=========================================
The EXTENDED hand's (x, y) is mapped from a CALIBRATED comfortable reach-box in
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
  • DEAD-MAN / ENGAGE GATE: the cursor is driven ONLY while an arm is EXTENDED
    OUT toward the sensor (a deliberate reach — see ArmExtension below) AND the
    body+hand are tracked. The moment the owner RELAXES the arm (pulls the hand
    back / bends the elbow) — or the hand/body goes untracked for more than
    DISENGAGE_GRACE_SEC (~0.3 s) — the air-mouse DISENGAGES: it stops calling
    SetCursorPos entirely (releasing control so the PHYSICAL mouse works again),
    releases any held button, and hides the reticle. Engage/disengage hysteresis
    keeps it from flickering at the threshold, and a closed hand that drops or
    leaves the frame can never strand the button down.

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
# per-hand state machine accepts it. This is the anti-stray-click guard: a single
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

# ─── REACH-TO-ENGAGE gate (an arm EXTENDED OUT drives the cursor) ────────────
# The air-mouse only controls the cursor while an arm is EXTENDED toward the
# sensor — a deliberate reach, NOT a hand merely raised. Extension is judged on
# TWO cues from the bridge's arm_extension() geometry (forward-depth +
# straightness); EITHER cue clearing its threshold counts as extended (the owner
# can reach by pushing forward OR by straightening the arm). Two threshold pairs
# give HYSTERESIS so an arm hovering right at the line can't flap engage on/off:
#   • to ENGAGE   the reach must clear the HIGHER bar (clearly extended)
#   • to STAY engaged it may relax only to the LOWER bar; past it → DISENGAGE.
#
# FORWARD-DEPTH (forward_reach_m = torso_z - hand_z, metres; >0 = hand in front):
#   engage at ~+0.12 m in front of the torso (a clear reach toward the screen),
#   stay engaged until it falls back to ~+0.06 m. A relaxed hand at the side /
#   resting on the desk sits at or behind the torso (≈0 or negative) → disengaged.
#
# LOOSENED 2026-06-08 (owner: "doesn't enable when the arm extends"): 0.20/0.10
#   → 0.12/0.06, and straightness 0.85/0.72 → 0.78/0.66. The v1.68 bars were tuned
#   at a different seating distance and never tripped for the owner's real reach,
#   so the air-mouse would not engage at all. These looser bars engage more
#   readily out of the box; the CALIBRATION routine ('calibrate air mouse',
#   persisted as KINECT_REACH_* in user_settings.json and read each tick by
#   _reach_thresholds()) then auto-fits the owner's actual relaxed→extended span
#   so the gate matches their body precisely.
AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M = 0.12    # reach this far forward to ENGAGE
AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M = 0.06  # relax below this to DISENGAGE
#
# ARM-STRAIGHTNESS (shoulder→hand chord / summed bone length; 0..1, 1 = straight):
#   engage when the arm is ≥ ~0.78 straight (nearly extended), stay engaged until
#   it bends back below ~0.66. A relaxed, elbow-bent arm folds well under this.
AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE = 0.78     # arm this straight to ENGAGE
AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE = 0.66  # bend below this to DISENGAGE

# ─── persisted per-body CALIBRATION (data/user_settings.json) ────────────────
# 'calibrate air mouse' / 'calibrate reach' captures the owner's RELAXED +
# EXTENDED forward-reach + straightness and writes the fitted thresholds under
# these keys; the live gate reads them every tick (via _reach_thresholds()),
# falling back to the looser defaults above when unset. The engage bar is placed
# ~60 % of the way relaxed→extended and the disengage bar ~40 %, so the gate fits
# the owner's actual range instead of a fixed guess (auto-fits their body).
SETTING_REACH_ENGAGE = "KINECT_REACH_ENGAGE"              # forward engage (m)
SETTING_REACH_DISENGAGE = "KINECT_REACH_DISENGAGE"        # forward disengage (m)
SETTING_STRAIGHT_ENGAGE = "KINECT_STRAIGHT_ENGAGE"        # straightness engage
SETTING_STRAIGHT_DISENGAGE = "KINECT_STRAIGHT_DISENGAGE"  # straightness disengage
CALIB_ENGAGE_FRACTION = 0.60     # engage bar this far relaxed→extended
CALIB_DISENGAGE_FRACTION = 0.40  # disengage bar this far relaxed→extended

# ─── HAND MIRROR (selfie-view correction) ────────────────────────────────────
# The Kinect color/skeleton stream the owner sees is MIRRORED (selfie view), so
# the owner's REAL left hand appears on the RIGHT of the image and vice-versa.
# The owner reported clicks + the controlling-hand circle landing on the WRONG
# side. With KINECT_HAND_MIRROR True (the owner's default) the air-mouse SWAPS the
# bridge's left↔right hands — BOTH the grip strings AND the per-arm extension/
# joints — so the owner's REAL left hand → LEFT button + left-side circle and
# their REAL right hand → RIGHT button + right-side circle. Flip this False
# (Settings GUI / user_settings.json) if a future build un-mirrors the stream.
KINECT_HAND_MIRROR_DEFAULT = True

# Grace window for a TRACKING dropout. A single lost/ambiguous frame (the Kinect
# briefly drops the body or hand joint) must NOT instantly disengage and re-snap
# — that would make the cursor jump and a drag stutter. While engaged, a dropout
# is tolerated for up to this long (button stays held, no cursor motion since
# there's no sample); past it the dead-man fully releases. ~0.3 s per the spec.
AIR_MOUSE_DISENGAGE_GRACE_SEC = 0.30

# ─── controlling-hand HYSTERESIS (ISSUE 3: both-hands stability) ─────────────
# With BOTH hands raised the cursor must NOT thrash between them frame-to-frame.
# Once a hand is driving the cursor it STAYS the controlling hand until the OTHER
# hand is BOTH clearly more extended (its reach_score leads by at least
# HAND_SWITCH_MARGIN) AND has been so for HAND_SWITCH_FRAMES consecutive frames.
# A brief wobble where the idle hand momentarily out-reaches by a hair can never
# flip control. (The L/R clicks are tracked for BOTH hands regardless — only the
# CURSOR-driving hand is sticky.)
HAND_SWITCH_MARGIN = 0.25     # challenger must lead the holder's score by this
HAND_SWITCH_FRAMES = 6        # …for this many consecutive frames before it wins

# The comfortable reach-box in front of the user, in camera-space METRES, that
# maps onto the whole virtual desktop. Centred roughly on where a seated user's
# hand naturally sits when reaching at the screen. x: sensor-RIGHT is +; the box
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
    """Debounce the OPEN↔CLOSED hand transition for ONE hand.

    Feed the raw grip string each frame ("open" / "closed" / "lasso" /
    "unknown"); the *stable* grip only changes after the new grip has been seen
    for `frames` consecutive ticks. ROBUST-CLOSE rules:
      • "lasso" (the two-finger pointer) is treated as CLOSED — a half-curled
        fist often reads as Lasso, and the owner means it as a click.
      • "unknown" / "nottracked" never flip the stable state (they're "no new
        evidence"): a 1-frame grip dropout HOLDS the last confident grip rather
        than spuriously releasing a drag. The dead-man (hand UNTRACKED) is what
        releases a held button, not a single ambiguous frame.
    Hysteresis falls out of the consecutive-frame requirement: a fist must be
    seen `frames` times to latch CLOSED, and an open hand `frames` times to
    latch OPEN, so neither flickers.

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

    @property
    def is_closed(self) -> bool:
        return self._stable == "closed"

    def reset(self, initial: str = "open") -> None:
        self._stable = initial
        self._candidate = None
        self._count = 0

    @staticmethod
    def _canon(raw: str) -> Optional[str]:
        """Canonicalise a raw bridge grip to "open"/"closed", or None for an
        ambiguous frame that carries no vote. Lasso → closed (a half-fist the
        owner means as a click); unknown / nottracked → None (hold)."""
        g = (raw or "unknown").lower()
        if g == "closed" or g == "lasso":
            return "closed"
        if g == "open":
            return "open"
        return None   # unknown / nottracked → no evidence this frame

    def update(self, raw: str) -> str:
        """Feed a raw grip; return the (possibly unchanged) stable grip."""
        vote = self._canon(raw)
        # Ambiguous frames hold the current stable grip and reset any in-flight
        # candidate streak (so a flicker mid-streak doesn't count toward a flip).
        if vote is None:
            self._candidate = None
            self._count = 0
            return self._stable
        if vote == self._stable:
            # Already stable here — clear any partial streak toward the other.
            self._candidate = None
            self._count = 0
            return self._stable
        # vote differs from stable: build/extend the candidate streak.
        if vote == self._candidate:
            self._count += 1
        else:
            self._candidate = vote
            self._count = 1
        if self._count >= self.frames:
            self._stable = vote
            self._candidate = None
            self._count = 0
        return self._stable


# Per-frame decision returned by AirMouseController.update().
class AirMouseDecision:
    """What the live loop should DO this frame.

      cursor:  (px, py) | None   — where to put the cursor (None = don't move)
      left:    "down" | "up" | None  — actuate the LEFT button (edge only; None
               means no change). Fired by the LEFT hand closing/opening.
      right:   "down" | "up" | None  — actuate the RIGHT button (edge only).
               Fired by the RIGHT hand closing/opening.
      overlay: "track" | "grab" | "hidden" — the reticle state to publish
               (cyan-track / gold-grab / hidden). "grab" while EITHER button held.
      hand:    "left" | "right" | None — which hand is driving the cursor (for
               the preview hand-circle), None while disengaged.
      grip:    the controlling hand's debounced stable grip ("open"/"closed") —
               drives the preview circle colour + diagnostics.
    """
    __slots__ = ("cursor", "left", "right", "overlay", "hand", "grip")

    def __init__(self, cursor, left, right, overlay, hand, grip):
        self.cursor = cursor
        self.left = left
        self.right = right
        self.overlay = overlay
        self.hand = hand
        self.grip = grip

    @property
    def button_edges(self):
        """The (button, action) edges this frame, e.g. [("left","down")]. Order:
        left then right. Used by the live loop to actuate the real mouse."""
        out = []
        if self.left in ("down", "up"):
            out.append(("left", self.left))
        if self.right in ("down", "up"):
            out.append(("right", self.right))
        return out

    def __repr__(self):   # pragma: no cover - debug aid
        return (f"AirMouseDecision(cursor={self.cursor}, left={self.left!r}, "
                f"right={self.right!r}, overlay={self.overlay!r}, "
                f"hand={self.hand!r}, grip={self.grip!r})")


class ArmExtension:
    """One arm's extension cues + the engage hysteresis test. A thin value object
    fed the bridge's arm_extension() dict (forward_reach_m + straightness) so the
    controller can ask "is this arm reaching?" with the right (engage vs stay-
    engaged) bar. PURE — no sensor. EITHER cue clearing its bar counts as
    extended, so the owner can reach by pushing the hand forward OR by
    straightening the arm."""

    __slots__ = ("side", "forward_m", "straightness", "hand")

    def __init__(self, side: str, forward_m: Optional[float],
                 straightness: Optional[float], hand=None):
        self.side = side
        self.forward_m = forward_m
        self.straightness = straightness
        self.hand = hand   # (x, y, z, state) of the controlling hand, or None

    @classmethod
    def from_bridge(cls, ext: dict) -> "ArmExtension":
        ext = ext or {}
        return cls(ext.get("side", ""), ext.get("forward_reach_m"),
                   ext.get("straightness"), ext.get("hand"))

    def is_extended(self, *, engaged: bool, thresholds: "Optional[dict]" = None,
                    fwd_engage: float = AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
                    fwd_disengage: float = AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M,
                    straight_engage: float = AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
                    straight_disengage: float = AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE
                    ) -> bool:
        """Is this arm EXTENDED enough to (stay) engaged? Applies HYSTERESIS:
        when currently DISENGAGED the reach must clear the HIGHER engage bar;
        once ENGAGED it only has to stay above the LOWER disengage bar. EITHER
        the forward-depth OR the straightness cue clearing its bar suffices.

        `thresholds` (the live CALIBRATED bars from _reach_thresholds(), keys
        fwd_engage / fwd_disengage / straight_engage / straight_disengage) wins
        over the keyword defaults when given, so the owner's calibration fits the
        gate to their body without re-plumbing every caller."""
        if thresholds:
            fwd_engage = thresholds.get("fwd_engage", fwd_engage)
            fwd_disengage = thresholds.get("fwd_disengage", fwd_disengage)
            straight_engage = thresholds.get("straight_engage", straight_engage)
            straight_disengage = thresholds.get("straight_disengage",
                                                straight_disengage)
        fwd_bar = fwd_disengage if engaged else fwd_engage
        straight_bar = straight_disengage if engaged else straight_engage
        fwd_ok = (self.forward_m is not None and self.forward_m >= fwd_bar)
        straight_ok = (self.straightness is not None
                       and self.straightness >= straight_bar)
        return bool(fwd_ok or straight_ok)

    def reach_score(self) -> float:
        """A scalar "how extended" used to pick the MORE-extended arm when both
        are reaching. Combines normalised forward-depth + straightness; missing
        cues contribute 0 so a partially-tracked arm still ranks. Higher = more
        extended / more dominant."""
        fwd = self.forward_m if self.forward_m is not None else 0.0
        # Normalise forward metres against the engage bar so it's comparable to
        # the 0..1 straightness; clamp negatives (hand behind body) to 0.
        fwd_n = max(0.0, fwd / AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M)
        straight = self.straightness if self.straightness is not None else 0.0
        return fwd_n + max(0.0, straight)


def extended_arms(left: "ArmExtension", right: "ArmExtension", *, engaged: bool,
                  thresholds: "Optional[dict]" = None) -> "list[ArmExtension]":
    """The arms currently EXTENDED enough to (stay) engaged — each with a usable
    hand joint and clearing its reach bar (engage hysteresis). PURE."""
    return [a for a in (left, right)
            if a is not None and a.hand is not None
            and a.is_extended(engaged=engaged, thresholds=thresholds)]


def choose_controlling_arm(left: "ArmExtension", right: "ArmExtension",
                           *, engaged: bool, thresholds: "Optional[dict]" = None,
                           current_side: "Optional[str]" = None,
                           margin: float = HAND_SWITCH_MARGIN
                           ) -> "Optional[ArmExtension]":
    """Pick which arm drives the cursor: among the EXTENDED arms (engage / stay-
    engage hysteresis), the MORE-extended (higher reach_score). Returns None when
    neither is extended (→ disengage). PURE.

    HAND-HYSTERESIS (ISSUE 3): when `current_side` (the hand already driving) is
    still among the extended candidates, it is KEPT unless the other arm leads it
    by at least `margin` — i.e. a tie / marginal lead never flips control. The
    controller adds the multi-FRAME requirement on top; this gives the per-frame
    stickiness (the holder wins ties)."""
    candidates = extended_arms(left, right, engaged=engaged, thresholds=thresholds)
    if not candidates:
        return None
    best = max(candidates, key=lambda a: a.reach_score())
    if current_side is not None:
        holder = next((a for a in candidates if a.side == current_side), None)
        if holder is not None and holder is not best:
            # The holder is still extended but isn't the top score: keep it unless
            # the challenger leads by the margin (sticky tie-break).
            if best.reach_score() - holder.reach_score() < margin:
                return holder
    return best


def _median(values: "list[float]") -> "Optional[float]":
    """Median of a list of floats, or None when empty. Pure; robust to the odd
    outlier sample the Kinect throws (better than a mean for calibration)."""
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return (float(xs[mid - 1]) + float(xs[mid])) / 2.0


def compute_reach_thresholds(
        relaxed_forward: "Optional[float]", extended_forward: "Optional[float]",
        relaxed_straight: "Optional[float]", extended_straight: "Optional[float]",
        *, engage_fraction: float = CALIB_ENGAGE_FRACTION,
        disengage_fraction: float = CALIB_DISENGAGE_FRACTION) -> dict:
    """Fit the reach-gate thresholds from a captured RELAXED + EXTENDED pose.

    Each bar is placed a FRACTION of the way relaxed→extended: the engage bar
    ~60 % (so a reach a little short of full extension still engages) and the
    disengage bar ~40 % (so a small sag doesn't drop it) — engage strictly above
    disengage, giving the hysteresis. Computed independently for the forward-reach
    cue and the straightness cue. A cue whose pose pair is missing/degenerate
    (None, or extended not clearly beyond relaxed) FALLS BACK to that cue's module
    default, so a partial capture still yields a safe, usable gate. PURE — no
    sensor; the live action feeds it the captured medians.

    Returns {fwd_engage, fwd_disengage, straight_engage, straight_disengage}."""
    ef = min(max(float(engage_fraction), 0.0), 1.0)
    df = min(max(float(disengage_fraction), 0.0), 1.0)

    def _bars(relaxed, extended, default_engage, default_disengage, min_span):
        # Need both ends AND a real span (extended clearly beyond relaxed) to fit;
        # otherwise keep the defaults rather than emit a nonsense (inverted/tiny)
        # threshold from a flubbed capture.
        if (relaxed is None or extended is None
                or (extended - relaxed) < min_span):
            return default_engage, default_disengage
        span = extended - relaxed
        return relaxed + ef * span, relaxed + df * span

    fwd_e, fwd_d = _bars(relaxed_forward, extended_forward,
                         AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
                         AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M, 0.05)
    str_e, str_d = _bars(relaxed_straight, extended_straight,
                         AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
                         AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE, 0.05)
    return {"fwd_engage": fwd_e, "fwd_disengage": fwd_d,
            "straight_engage": str_e, "straight_disengage": str_d}


class AirMouseController:
    """The pure per-frame brain. Holds the smoothing, the PER-HAND grip
    debouncers + button state, and the REACH engage state, turning each
    (left_ext, right_ext, left_grip, right_grip, tracked) sample into an
    AirMouseDecision. NO I/O — the live loop applies the decision (move cursor,
    press buttons, publish overlay state). Re-buildable cheaply; reset() on
    disable / hand-loss.

    REACH ENGAGE GATE (the cursor only moves while an arm is EXTENDED OUT):
      The controller is ENGAGED only while at least one arm is extended toward the
      sensor (forward-depth and/or arm-straightness clearing its bar, see
      ArmExtension) AND a body+hand are tracked. Engage/stay-engage hysteresis
      stops it flapping at the line. The cursor follows the MORE-extended arm
      (choose_controlling_arm). While DISENGAGED — both arms relaxed, OR untracked
      beyond the ~0.3 s grace — the decision carries cursor=None (so the live loop
      calls NO SetCursorPos and the PHYSICAL mouse is free), releases any held
      button, and hides the overlay. Re-extending an arm re-engages, EMA re-snaps.

    PER-HAND clicks (HAND-SPECIFIC), evaluated EVERY engaged frame for BOTH hands
    regardless of which one drives the cursor:
      • LEFT hand  OPEN→CLOSED → emit LEFT  "down"; CLOSED→OPEN → LEFT  "up".
      • RIGHT hand OPEN→CLOSED → emit RIGHT "down"; CLOSED→OPEN → RIGHT "up".
      A held-closed hand keeps its button down while the cursor moves (a drag);
      a quick close→open with no move is a click. The overlay shows "grab" while
      EITHER button is held."""

    def __init__(self, reach: ReachBox,
                 alpha: float = AIR_MOUSE_EMA_ALPHA,
                 debounce_frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES,
                 grace_sec: float = AIR_MOUSE_DISENGAGE_GRACE_SEC,
                 clock=time.monotonic,
                 switch_margin: float = HAND_SWITCH_MARGIN,
                 switch_frames: int = HAND_SWITCH_FRAMES):
        self.reach = reach
        self._ema_x = EMA(alpha)
        self._ema_y = EMA(alpha)
        # One debouncer + one button-down flag PER HAND so each hand drives its
        # own (left/right) mouse button independently.
        self._grip_left = GripDebouncer(debounce_frames, initial="open")
        self._grip_right = GripDebouncer(debounce_frames, initial="open")
        self._left_down = False
        self._right_down = False
        # Engage gate state.
        self._grace_sec = max(0.0, float(grace_sec))
        self._clock = clock          # injectable monotonic clock (tests)
        self._engaged = False        # is the air-mouse currently driving the cursor
        self._hand: Optional[str] = None   # which hand is driving ("left"/"right")
        self._last_engaged_at = 0.0  # clock() of the last EXTENDED+tracked frame
        # Controlling-hand HYSTERESIS (ISSUE 3): the challenger must out-reach the
        # holder by `switch_margin` for `switch_frames` consecutive frames before
        # the cursor switches hands, so two raised hands can't thrash the cursor.
        self._switch_margin = max(0.0, float(switch_margin))
        self._switch_frames = max(1, int(switch_frames))
        self._challenge_side: Optional[str] = None   # the side currently challenging
        self._challenge_count = 0                    # its consecutive-lead streak

    def reset(self) -> None:
        """Drop all smoothing + grip + engage state. Used by the dead-man and on
        disable so the next reach starts clean (no cursor sweep from a stale
        value, no phantom button edge, freshly DISENGAGED)."""
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip_left.reset(initial="open")
        self._grip_right.reset(initial="open")
        # NB: this does NOT itself emit a button-up — the caller (dead-man) is
        # responsible for releasing held buttons. We only clear our own view.
        self._left_down = False
        self._right_down = False
        self._engaged = False
        self._hand = None
        self._last_engaged_at = 0.0
        self._challenge_side = None
        self._challenge_count = 0

    @property
    def button_is_down(self) -> bool:
        """True while EITHER mouse button is held (left or right)."""
        return self._left_down or self._right_down

    @property
    def left_is_down(self) -> bool:
        return self._left_down

    @property
    def right_is_down(self) -> bool:
        return self._right_down

    @property
    def engaged(self) -> bool:
        """True while the air-mouse is actively driving the cursor (an arm
        extended + tracked). False while disengaged (arm relaxed / lost) — in
        which state update() returns cursor=None so the physical mouse is free."""
        return self._engaged

    @property
    def hand(self) -> Optional[str]:
        """Which hand is driving the cursor ("left"/"right"), or None while
        disengaged. Read by the preview to draw the circle on the right hand."""
        return self._hand

    def release_decision(self) -> AirMouseDecision:
        """The DEAD-MAN / disengaged decision: if a button was held, command it
        UP (per hand); hide the overlay; clear smoothing + grips + engage so the
        next acquisition snaps. cursor=None so the live loop issues NO
        SetCursorPos and the physical mouse is free. Idempotent — once released,
        repeated calls just keep the overlay hidden with no button edge."""
        left = "up" if self._left_down else None
        right = "up" if self._right_down else None
        self._left_down = False
        self._right_down = False
        self._engaged = False
        self._hand = None
        self._challenge_side = None
        self._challenge_count = 0
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip_left.reset(initial="open")
        self._grip_right.reset(initial="open")
        return AirMouseDecision(cursor=None, left=left, right=right,
                                overlay="hidden", hand=None, grip="open")

    def _hand_button_edge(self, debouncer: GripDebouncer, raw_grip: str,
                          down_attr: str) -> Optional[str]:
        """Advance ONE hand's debouncer and emit its button edge. `down_attr` is
        the instance attribute name holding that hand's button-down flag
        ("_left_down"/"_right_down"). Returns "down"/"up"/None."""
        stable = debouncer.update(raw_grip)
        held = getattr(self, down_attr)
        want_down = (stable == "closed")
        if want_down and not held:
            setattr(self, down_attr, True)
            return "down"
        if not want_down and held:
            setattr(self, down_attr, False)
            return "up"
        return None

    def _select_controlling_arm(self, left_ext, right_ext,
                                thresholds: "Optional[dict]"):
        """Pick the cursor-driving arm with HAND-HYSTERESIS (ISSUE 3). Keeps the
        current controlling hand unless the OTHER arm out-reaches it by the margin
        for `switch_frames` consecutive frames; a brief or marginal lead never
        flips control. Updates the challenger streak as a side effect and returns
        the chosen ArmExtension (or None when neither arm is extended)."""
        candidates = extended_arms(left_ext, right_ext, engaged=self._engaged,
                                   thresholds=thresholds)
        if not candidates:
            self._challenge_side = None
            self._challenge_count = 0
            return None
        best = max(candidates, key=lambda a: a.reach_score())
        # Is the hand currently driving still a live candidate?
        holder = next((a for a in candidates if a.side == self._hand), None)
        if holder is None:
            # The driving hand relaxed/left (or we weren't engaged): take the most-
            # extended arm outright and reset any challenge.
            self._challenge_side = None
            self._challenge_count = 0
            return best
        if best is holder:
            # The holder is still the most extended — no challenge in progress.
            self._challenge_side = None
            self._challenge_count = 0
            return holder
        # A DIFFERENT arm out-scores the holder. Require a sustained, clear lead.
        lead = best.reach_score() - holder.reach_score()
        if lead >= self._switch_margin:
            if self._challenge_side == best.side:
                self._challenge_count += 1
            else:
                self._challenge_side = best.side
                self._challenge_count = 1
            if self._challenge_count >= self._switch_frames:
                # Sustained clear lead → switch hands.
                self._challenge_side = None
                self._challenge_count = 0
                return best
        else:
            # Lead too small this frame → challenge resets (no thrash).
            self._challenge_side = None
            self._challenge_count = 0
        return holder

    def update(self, left_ext, right_ext, left_grip: str, right_grip: str,
               tracked: bool, thresholds: "Optional[dict]" = None
               ) -> AirMouseDecision:
        """Advance one frame.

        left_ext / right_ext: the per-arm ArmExtension (or None when that arm's
            joints couldn't be read) describing forward-reach + straightness.
        left_grip / right_grip: the raw bridge grips for each hand
            ("open"/"closed"/"lasso"/"unknown").
        tracked: True when the bridge reported a tracked body this frame.
        thresholds: the live reach-gate bars from _reach_thresholds() (the owner's
            CALIBRATION, or the looser defaults). None → the module defaults.

        DISENGAGES (returns cursor=None — no SetCursorPos — and releases any held
        button) when the body/hand is NOT tracked, or when NEITHER arm is extended
        (both relaxed). A brief tracking dropout while ENGAGED is tolerated for up
        to the grace window (button held, cursor parked) before the full release.
        The cursor follows the more-extended arm with HAND-HYSTERESIS (no thrash
        between two raised hands); per-hand close→click is evaluated for BOTH hands
        every engaged frame regardless of which one drives the cursor."""
        # ── tracking-loss path, with a short grace so a 1-frame dropout doesn't
        #    disengage + re-snap (a held drag must survive a flicker). ──────────
        if not tracked:
            if (self._engaged and self._grace_sec > 0.0
                    and (self._clock() - self._last_engaged_at) <= self._grace_sec):
                # Brief dropout: hold. No sample → no cursor motion; keep any held
                # button and the current overlay. Do NOT refresh the engage clock,
                # so a sustained dropout still ages out into a full release.
                overlay = "grab" if self.button_is_down else "track"
                return AirMouseDecision(cursor=None, left=None, right=None,
                                        overlay=overlay, hand=self._hand,
                                        grip=self._controlling_grip())
            return self.release_decision()

        # ── reach gate: pick the controlling arm (engage hysteresis + sticky
        #    hand-hysteresis). No arm extended → disengage (fail SAFE). ─────────
        arm = self._select_controlling_arm(left_ext, right_ext, thresholds)
        if arm is None:
            return self.release_decision()

        # ── ENGAGED. On the rising edge (was disengaged) snap the smoothing to
        #    the new hand position so the cursor doesn't sweep from a stale value.
        if not self._engaged:
            self._ema_x.reset()
            self._ema_y.reset()
        self._engaged = True
        self._hand = arm.side
        self._last_engaged_at = self._clock()

        # Smooth the controlling hand's position, then map to a pixel.
        hand = arm.hand
        sx = self._ema_x.update(float(hand[0]))
        sy = self._ema_y.update(float(hand[1]))
        cursor = self.reach.map(sx, sy)

        # Per-hand clicks: evaluate BOTH hands every engaged frame so either hand
        # can click regardless of which drives the cursor. LEFT hand → LEFT
        # button, RIGHT hand → RIGHT button.
        left_edge = self._hand_button_edge(self._grip_left, left_grip, "_left_down")
        right_edge = self._hand_button_edge(self._grip_right, right_grip, "_right_down")

        overlay = "grab" if self.button_is_down else "track"
        return AirMouseDecision(cursor=cursor, left=left_edge, right=right_edge,
                                overlay=overlay, hand=arm.side,
                                grip=self._controlling_grip())

    def _controlling_grip(self) -> str:
        """The stable grip of whichever hand is driving the cursor (for the
        preview circle colour). Falls back to "open" when disengaged."""
        if self._hand == "left":
            return self._grip_left.stable
        if self._hand == "right":
            return self._grip_right.stable
        return "open"


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
# draws a translucent circle around the controlling (extended) hand joint,
# coloured by the LIVE air-mouse state so the owner SEES when the cursor is active
# / clicking:
#   • ENGAGED + open hand   → BLUE   (cursor active, tracking)
#   • ENGAGED + closed hand → ORANGE (click / drag held)
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

      • engaged + grip "closed"        → ORANGE  (click / drag active)
      • engaged + any other grip       → BLUE    (cursor engaged, tracking)
      • not engaged                    → GREY    (faint idle hint)

    Returns a (B, G, R) tuple. The caller decides idle→draw-faint-or-skip; we
    return the idle grey so the colour mapping itself stays total + testable."""
    if engaged:
        if (grip or "").lower() == "closed":
            return HAND_CIRCLE_COLOR_CLOSED
        return HAND_CIRCLE_COLOR_ENGAGED
    return HAND_CIRCLE_COLOR_IDLE


# Thread-safe snapshot of the LIVE air-mouse engage state + which hand + grip,
# written by _poll_once each tick and read by the HUD preview compositor (a
# DIFFERENT thread — the face-tracking loop). A tiny lock keeps the
# (engaged, hand, grip, ts) tuple consistent. The preview reads it to decide the
# hand-circle colour + which hand to draw it on; stale reads paint the last state.
_air_mouse_state_lock = threading.Lock()
_air_mouse_state: dict = {"engaged": False, "hand": None, "grip": "open",
                          "ts": 0.0}


def _set_air_mouse_state(engaged: bool, grip: str,
                         hand: "str | None" = None) -> None:
    """Publish the live engage state + which hand + grip for the preview
    (thread-safe)."""
    with _air_mouse_state_lock:
        _air_mouse_state["engaged"] = bool(engaged)
        _air_mouse_state["hand"] = hand
        _air_mouse_state["grip"] = (grip or "open")
        _air_mouse_state["ts"] = time.time()


def get_air_mouse_state() -> dict:
    """Thread-safe snapshot {'engaged': bool, 'hand': str|None, 'grip': str,
    'ts': float} of the air-mouse. Read by the HUD skeleton preview to colour the
    hand circle (engaged→blue, closed→orange, off→grey) and place it on the
    controlling hand. Returns a COPY so the caller can't mutate the shared dict.
    Never raises."""
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


def _speak(text: str) -> None:
    """Speak a line mid-action (used by the calibration walk-through) via the
    monolith's TTS. Best-effort + silent — mirrors kinect_gestures._speak. NEVER
    raises; a headless/test instance just no-ops."""
    bc = _bc()
    if bc is None:
        return
    try:
        fn = getattr(bc, "_speak", None) or getattr(bc, "speak", None)
        if callable(fn):
            fn(text)
    except Exception:
        pass


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _saved_settings() -> dict:
    """The owner's persisted settings dict (data/user_settings.json) via the same
    reader model_picker / kinect_gestures use — honours JARVIS_SETTINGS_PATH so a
    test never touches the real file. Returns {} on any failure. NEVER raises."""
    try:
        from tools import settings_window as sw
        cur = sw.load_settings()
        return cur if isinstance(cur, dict) else {}
    except Exception:
        return {}


def _saved_float(settings: dict, key: str) -> "Optional[float]":
    """A persisted float by key, or None when absent / unparseable."""
    try:
        v = settings.get(key)
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _reach_thresholds() -> dict:
    """The LIVE reach-gate thresholds the engage test uses, read fresh each call:
    persisted CALIBRATION values (KINECT_REACH_* in user_settings.json) when the
    owner has calibrated, else the looser module defaults. Returns
    {fwd_engage, fwd_disengage, straight_engage, straight_disengage}. A partially
    written calibration (only some keys) falls back per-field to the default, so a
    half-finished calibration can never strand the gate. NEVER raises."""
    s = _saved_settings()
    fwd_e = _saved_float(s, SETTING_REACH_ENGAGE)
    fwd_d = _saved_float(s, SETTING_REACH_DISENGAGE)
    str_e = _saved_float(s, SETTING_STRAIGHT_ENGAGE)
    str_d = _saved_float(s, SETTING_STRAIGHT_DISENGAGE)
    return {
        "fwd_engage": fwd_e if fwd_e is not None
        else AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
        "fwd_disengage": fwd_d if fwd_d is not None
        else AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M,
        "straight_engage": str_e if str_e is not None
        else AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
        "straight_disengage": str_d if str_d is not None
        else AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE,
    }


def _hand_mirror_enabled() -> bool:
    """Whether to SWAP the bridge's left↔right hands (selfie-view correction, see
    KINECT_HAND_MIRROR_DEFAULT). Read fresh each call from core.config (Settings
    GUI override) so the owner can flip it live; defaults True."""
    return _cfg_flag("KINECT_HAND_MIRROR", KINECT_HAND_MIRROR_DEFAULT)


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


def _mouse_button(action: str, button: str = "left") -> bool:
    """Press ('down') or release ('up') the LEFT or RIGHT mouse button at the
    current cursor position. win32api mouse_event flags first, pyautogui fallback.
    `button` ∈ {"left","right"} — the LEFT hand closing maps to "left", the RIGHT
    hand to "right". Returns True on success; never raises."""
    button = (button or "left").lower()
    # win32api path: event flags per button + up/down.
    try:
        import win32api
        import win32con
        if button == "right":
            flag = (win32con.MOUSEEVENTF_RIGHTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_RIGHTUP)
        else:  # left (primary)
            flag = (win32con.MOUSEEVENTF_LEFTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_LEFTUP)
        win32api.mouse_event(flag, 0, 0, 0, 0)
        return True
    except Exception:
        pass
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        btn = "right" if button == "right" else "left"
        if action == "down":
            pyautogui.mouseDown(button=btn)
        else:
            pyautogui.mouseUp(button=btn)
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
def _dist3(a, b) -> Optional[float]:
    """Euclidean 3D distance between two (x, y, z, ...) joints, or None. Local
    mirror of the bridge helper, used only by the no-bridge-helper fallback."""
    try:
        if not a or not b or len(a) < 3 or len(b) < 3:
            return None
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except (TypeError, ValueError):
        return None


def _local_arm_extension(joints: dict, side: str) -> dict:
    """Local fallback for the bridge's arm_extension() — same forward-depth +
    straightness math — used only if the loaded bridge lacks the helper (older
    build). NEVER raises. (audio.kinect_bridge.arm_extension is the canonical
    one; this keeps the air-mouse working against any bridge.)"""
    out = {"side": side, "hand": None, "forward_reach_m": None,
           "straightness": None, "shoulder_hand_m": None, "arm_len_m": None}
    try:
        shoulder = joints.get(f"shoulder_{side}")
        elbow = joints.get(f"elbow_{side}")
        hand = joints.get(f"hand_{side}") or joints.get(f"wrist_{side}")
        out["hand"] = hand
        body_ref = None
        for name in ("spine_mid", "spine_shoulder", "spine_base"):
            j = joints.get(name)
            if j and len(j) >= 3 and float(j[2]) > 0:
                body_ref = j
                break
        if body_ref is None and shoulder and len(shoulder) >= 3:
            body_ref = shoulder
        if (body_ref is not None and hand and len(hand) >= 3
                and float(hand[2]) > 0 and float(body_ref[2]) > 0):
            out["forward_reach_m"] = float(body_ref[2]) - float(hand[2])
        chord = _dist3(shoulder, hand)
        upper = _dist3(shoulder, elbow)
        fore = _dist3(elbow, hand)
        out["shoulder_hand_m"] = chord
        if upper is not None and fore is not None:
            arm_len = upper + fore
            out["arm_len_m"] = arm_len
            if chord is not None and arm_len > 1e-3:
                out["straightness"] = min(1.0, chord / arm_len)
    except (TypeError, ValueError, KeyError):
        pass
    return out


def _arm_extension(bridge, joints: dict, side: str) -> "ArmExtension":
    """Build the ArmExtension for one side. Prefers the bridge's arm_extension()
    geometry helper (single source of truth for forward-reach + straightness);
    falls back to computing it locally so the air-mouse still works against an
    older bridge build that lacks the helper. NEVER raises."""
    try:
        fn = getattr(bridge, "arm_extension", None)
        if callable(fn):
            return ArmExtension.from_bridge(fn(joints, side))
    except Exception:
        pass
    try:
        return ArmExtension.from_bridge(_local_arm_extension(joints, side))
    except Exception:
        return ArmExtension(side, None, None, None)


def _hand_sample(bridge) -> tuple["Optional[ArmExtension]", "Optional[ArmExtension]",
                                  str, str, bool]:
    """Read the per-arm extension + per-hand grips from the bridge:
    (left_ext, right_ext, left_grip, right_grip, tracked).

    left_ext / right_ext are the ArmExtension (forward-reach + straightness +
    controlling-hand joint) for each arm of the NEAREST body, or None when that
    arm's joints aren't usable. left_grip / right_grip are that body's raw grips.
    tracked is whether a body was in view. NEVER raises — any failure degrades to
    (None, None, "unknown", "unknown", False) which the controller treats as a
    dead-man release (no arm extended / not tracked → disengaged)."""
    none_result = (None, None, "unknown", "unknown", False)
    try:
        if not bridge.get_enabled():
            return none_result
        ok, _reason = bridge.available()
        if not ok:
            return none_result
        bodies = bridge.get_bodies()
    except Exception:
        return none_result
    if not bodies:
        return none_result

    # Nearest body (same ranking the rest of the stack uses).
    def _key(b):
        d = b.get("distance_m") if isinstance(b, dict) else None
        return d if isinstance(d, (int, float)) and d > 0 else float("inf")
    try:
        body = min((b for b in bodies if isinstance(b, dict)), key=_key)
    except (TypeError, ValueError):
        return none_result

    joints = body.get("joints") or {}
    left_grip = (body.get("hand_left") or "unknown").lower()
    right_grip = (body.get("hand_right") or "unknown").lower()
    left_ext = _arm_extension(bridge, joints, "left")
    right_ext = _arm_extension(bridge, joints, "right")
    # ISSUE 1 — selfie-view correction: the Kinect stream is MIRRORED, so the
    # owner's REAL left hand is what the SDK labels "right" (and vice-versa). When
    # KINECT_HAND_MIRROR is on, SWAP the two hands here — BOTH the grips AND the
    # per-arm extensions (relabelling each .side) — so everything downstream (the
    # per-hand L/R clicks, choose_controlling_arm, the published which-hand, the
    # preview circle's prefer_side) treats the owner's REAL left hand as LEFT.
    if _hand_mirror_enabled():
        left_ext, right_ext = (_relabel_arm_side(right_ext, "left"),
                               _relabel_arm_side(left_ext, "right"))
        left_grip, right_grip = right_grip, left_grip
    return left_ext, right_ext, left_grip, right_grip, True


def _relabel_arm_side(ext: "Optional[ArmExtension]",
                      side: str) -> "Optional[ArmExtension]":
    """Return `ext` with its .side relabelled (used by the mirror swap so a
    swapped arm reports the side it now drives). None passes through. The geometry
    (forward-reach / straightness / hand joint) is unchanged — only the label, so
    the published which-hand + preview circle land on the correct side."""
    if ext is None:
        return None
    return ArmExtension(side, ext.forward_m, ext.straightness, ext.hand)


# ─── ISSUE 2a: CALIBRATION capture ──────────────────────────────────────────
CALIBRATE_CAPTURE_SECONDS = 3.0           # hold each pose this long
CALIBRATE_POLL_HZ = 15.0                  # sample cadence while capturing
CALIBRATE_POLL_INTERVAL = 1.0 / CALIBRATE_POLL_HZ
CALIBRATE_MAX_SECONDS = 4.0               # hard wall-time cap per pose (slack)


def _capture_reach(bridge, seconds: float = CALIBRATE_CAPTURE_SECONDS,
                   sleep_fn=time.sleep, now_fn=time.monotonic
                   ) -> "tuple[Optional[float], Optional[float], int]":
    """Sample the MORE-extended arm's forward-reach + straightness for ~`seconds`
    and return their MEDIANS plus the usable-frame count:
    (median_forward_m, median_straightness, n_samples).

    Reads the same _hand_sample() the live gate uses (so the mirror swap +
    geometry are identical), taking, per frame, the more-extended arm's cues.
    Median (not mean) shrugs off the odd Kinect outlier. A wedged sensor can't
    hang the voice loop — capped at CALIBRATE_MAX_SECONDS. NEVER raises."""
    fwd_samples: list = []
    straight_samples: list = []
    n = 0
    try:
        deadline = now_fn() + min(float(seconds), CALIBRATE_MAX_SECONDS)
        while now_fn() < deadline:
            left_ext, right_ext, _lg, _rg, tracked = _hand_sample(bridge)
            if tracked:
                arms = [a for a in (left_ext, right_ext) if a is not None]
                arm = (max(arms, key=lambda a: a.reach_score())
                       if arms else None)
                if arm is not None:
                    n += 1
                    if arm.forward_m is not None:
                        fwd_samples.append(float(arm.forward_m))
                    if arm.straightness is not None:
                        straight_samples.append(float(arm.straightness))
            sleep_fn(CALIBRATE_POLL_INTERVAL)
    except Exception:
        pass
    return _median(fwd_samples), _median(straight_samples), n


def _persist_reach_thresholds(th: dict) -> bool:
    """Write the four fitted reach thresholds to user_settings.json (KINECT_REACH_*
    / KINECT_STRAIGHT_*) via the hardened settings writer. All-or-nothing-ish: each
    key is written; returns True iff every write reported success. NEVER raises."""
    ok = True
    ok = _persist_setting(SETTING_REACH_ENGAGE, float(th["fwd_engage"])) and ok
    ok = _persist_setting(SETTING_REACH_DISENGAGE, float(th["fwd_disengage"])) and ok
    ok = _persist_setting(SETTING_STRAIGHT_ENGAGE,
                          float(th["straight_engage"])) and ok
    ok = _persist_setting(SETTING_STRAIGHT_DISENGAGE,
                          float(th["straight_disengage"])) and ok
    return ok


def _apply_decision(decision: AirMouseDecision) -> None:
    """Perform the side effects of a decision: move the cursor and actuate the
    per-hand buttons. Pure-core stays I/O-free; THIS is where the real mouse is
    touched. Best-effort; never raises out to the loop."""
    if decision.cursor is not None:
        _set_cursor_pos(decision.cursor[0], decision.cursor[1])
    for button, action in decision.button_edges:
        _mouse_button(action, button)


# ─── ISSUE 2b: live reach DEBUG LOG (~2 Hz) ─────────────────────────────────
# While the air-mouse is enabled, print the live reach numbers at ~2 Hz so the
# owner can SEE what the gate sees and tune / confirm a calibration, e.g.
#   [air-mouse] reach=0.18 straight=0.91 hand=right engaged=False
# The most-extended arm's cues are logged (that's the one the gate is judging).
_AIR_MOUSE_DEBUG_INTERVAL = 0.5             # seconds between debug lines (~2 Hz)
_air_mouse_debug_last = [0.0]               # module-list so the throttle persists


def _format_reach_debug(left_ext, right_ext, tracked: bool, ctrl) -> str:
    """The ~2 Hz debug line for the live reach values. Picks the more-extended
    arm's forward-reach + straightness (the cue the gate is judging) and reports
    the live which-hand + engaged. PURE-ish (reads ctrl state); NEVER raises."""
    try:
        arms = [a for a in (left_ext, right_ext) if a is not None]
        arm = max(arms, key=lambda a: a.reach_score()) if arms else None
        if arm is None:
            reach_s, straight_s, hand_s = "n/a", "n/a", "none"
        else:
            reach_s = ("%.2f" % arm.forward_m) if arm.forward_m is not None else "n/a"
            straight_s = ("%.2f" % arm.straightness
                          if arm.straightness is not None else "n/a")
            hand_s = arm.side or "?"
        return ("  [air-mouse] reach=%s straight=%s hand=%s engaged=%s tracked=%s"
                % (reach_s, straight_s, hand_s, bool(ctrl.engaged), bool(tracked)))
    except Exception:
        return "  [air-mouse] reach=? straight=? hand=? engaged=?"


def _maybe_debug_log(left_ext, right_ext, tracked: bool, ctrl,
                     now: "Optional[float]" = None) -> bool:
    """Emit the reach debug line if the throttle window has elapsed. Returns True
    iff a line was printed (for the test). NEVER raises."""
    try:
        t = time.monotonic() if now is None else float(now)
        if (t - _air_mouse_debug_last[0]) < _AIR_MOUSE_DEBUG_INTERVAL:
            return False
        _air_mouse_debug_last[0] = t
        print(_format_reach_debug(left_ext, right_ext, tracked, ctrl))
        return True
    except Exception:
        return False


def _poll_once(ctrl: AirMouseController, bridge) -> Optional[AirMouseDecision]:
    """One air-mouse tick: read the arms, decide, and (only when enabled +
    not staging) ACT — move the cursor, actuate the per-hand buttons, publish the
    overlay state. Returns the decision (for tests) or None when the bridge is
    absent. NEVER raises.

    GATING: the controller is ALWAYS advanced (so its smoothing/grip state stays
    current and a re-enable doesn't see a huge gap), but the SIDE EFFECTS (mouse
    move, buttons, visible overlay) only happen when KINECT_AIR_MOUSE_ENABLED is
    on AND not staging. Flipping the flag off therefore stops the cursor moving
    instantly and releases any held button via the dead-man path."""
    if bridge is None:
        return None
    left_ext, right_ext, left_grip, right_grip, tracked = _hand_sample(bridge)
    # Live CALIBRATED reach bars (owner's persisted KINECT_REACH_* or the looser
    # defaults), read fresh each tick so a re-calibration takes effect with no
    # restart. The controller applies them in its engage hysteresis.
    thresholds = _reach_thresholds()
    try:
        decision = ctrl.update(left_ext, right_ext, left_grip, right_grip,
                               tracked, thresholds=thresholds)
    except Exception:
        # A controller error must not strand a held button — force a release.
        try:
            decision = ctrl.release_decision()
        except Exception:
            return None

    enabled = _air_mouse_enabled()
    # ISSUE 2b: while enabled, surface the live reach numbers at ~2 Hz for tuning
    # / confirming a calibration. Throttled + best-effort; only when enabled so a
    # disabled poller stays quiet.
    if enabled:
        _maybe_debug_log(left_ext, right_ext, tracked, ctrl)
    if not enabled:
        # Gated OFF mid-session: make sure no button is left held and the
        # overlay is hidden. ctrl.update already returned a (possibly
        # button-up) decision if it had been holding; honour pending 'up's
        # (per hand) so a flag flip during a drag still releases, but never a
        # 'down'.
        for button, action in decision.button_edges:
            if action == "up":
                _mouse_button("up", button)
        _clear_overlay_state()
        # Publish DISENGAGED to the preview: with the air-mouse off the HUD
        # hand-circle must show idle/grey (or nothing), never a live blue/orange.
        _set_air_mouse_state(False, "open", None)
        return decision

    # Enabled + not staging: act.
    _apply_decision(decision)
    visible = tracked and decision.overlay != "hidden"
    _publish_overlay_state(decision, visible)
    # Publish the LIVE engage state + which hand + grip for the HUD skeleton
    # preview's hand circle (B2): engaged→blue, closed→orange, disengaged→grey,
    # drawn on the controlling hand. Read off the controller (engaged + hand) +
    # the decision (grip) so the preview colour matches the cursor/reticle.
    try:
        _set_air_mouse_state(bool(ctrl.engaged), decision.grip, ctrl.hand)
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
                # may have changed displays) and start fresh so the first reach
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
    msg = ("Air-mouse on, sir — reach an arm out toward the screen to take the "
           "cursor, close your left hand to left-click or your right hand to "
           "right-click, hold a hand closed to drag, and relax your arm to "
           "release the cursor.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + sensor_note


def air_mouse_off(_: str = "") -> str:
    """Turn the air-mouse off (live + persisted). Also clears the overlay so the
    reticle disappears immediately."""
    if not _cfg_flag("KINECT_AIR_MOUSE_ENABLED"):
        return "The air-mouse is already off, sir."
    persisted = _set_enabled(False)
    # Make sure nothing is left held (BOTH buttons) and the reticle is gone.
    try:
        _mouse_button("up", "left")
        _mouse_button("up", "right")
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
    how = ("reach an arm out to take the cursor, close your left hand to "
           "left-click or your right hand to right-click, hold to drag, and "
           "relax your arm to release the cursor")
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
            f"view at the moment. Reach an arm out toward the screen and {how}.")


def calibrate_air_mouse(_: str = "") -> str:
    """ISSUE 2a — the CALIBRATION walk-through ('calibrate air mouse' / 'calibrate
    reach'). Speaks the owner through it: hold an arm EXTENDED for ~3 s (capture
    forward-reach + straightness medians), then RELAX it down for ~3 s (capture
    relaxed medians), fit the engage/disengage thresholds ~60 %/40 % of the way
    relaxed→extended, and persist them (KINECT_REACH_* / KINECT_STRAIGHT_* in
    user_settings.json) so the live gate auto-fits the owner's body. Honest on
    every failure — never claims to have calibrated something it didn't capture.

    Runs synchronously (like point_calibrate): it speaks each prompt, captures,
    then returns the spoken summary."""
    if _is_staging():
        return "Not while I'm in staging, sir."
    ready, why = _sensor_ready()
    if not ready:
        return (f"I can't calibrate the air-mouse, sir — {why}. Enable the "
                "Kinect and try again.")
    bridge = _bridge()
    if bridge is None:
        return "I can't calibrate the air-mouse, sir — the Kinect bridge isn't loaded."

    # 1) EXTENDED pose.
    _speak("Let's calibrate the air-mouse, sir. Extend your arm out toward the "
           "screen and hold it there.")
    ext_fwd, ext_straight, ext_n = _capture_reach(bridge)
    if ext_n == 0:
        return ("I couldn't see your arm while you reached, sir — make sure "
                "you're in the Kinect's view and try calibrating again.")

    # 2) RELAXED pose.
    _speak("Got it. Now relax your arm down to your side.")
    rel_fwd, rel_straight, rel_n = _capture_reach(bridge)
    if rel_n == 0:
        return ("I lost track of you while you relaxed, sir — please try "
                "calibrating again.")

    # 3) Fit + persist the thresholds.
    th = compute_reach_thresholds(rel_fwd, ext_fwd, rel_straight, ext_straight)
    persisted = _persist_reach_thresholds(th)
    # Did we actually fit either cue from the capture (vs. fall back to defaults)?
    fitted_fwd = (rel_fwd is not None and ext_fwd is not None
                  and (ext_fwd - rel_fwd) >= 0.05)
    fitted_straight = (rel_straight is not None and ext_straight is not None
                       and (ext_straight - rel_straight) >= 0.05)
    if not (fitted_fwd or fitted_straight):
        return ("I couldn't tell your reach apart from your relax, sir — extend "
                "fully then drop your arm all the way, and calibrate again.")
    msg = ("Air-mouse calibrated, sir — reach out past about %.2f metres or "
           "straighten your arm to take the cursor, and it releases when you "
           "relax back." % th["fwd_engage"])
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ─── registration ────────────────────────────────────────────────────────
def register(actions):
    actions["air_mouse_on"] = air_mouse_on
    actions["air_mouse_off"] = air_mouse_off
    actions["air_mouse_status"] = air_mouse_status
    actions["calibrate_air_mouse"] = calibrate_air_mouse

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
