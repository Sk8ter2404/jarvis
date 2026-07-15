"""
air_control — pure, hardware-free "AIR CONTROL" engine for the Kinect v2.

WHY THIS MODULE EXISTS
======================
Movie-style spatial hand control: the owner reaches a hand out toward the
sensor, and that hand drives the mouse cursor across the whole virtual desktop;
a closed fist GRABS and drags whatever's under the cursor (a window title bar,
a slider) — including across monitors — opening the hand releases; a quick
close-then-open is a CLICK; a "lasso" (pointing) hand scrolls. Dropping the
hand lets go of the cursor entirely.

Turning the raw skeleton stream (audio/kinect_bridge.get_bodies(), which now
also carries per-hand OPEN/CLOSED/LASSO state) into mouse operations is a
self-contained problem with NOTHING to do with the sensor, threading, or
pyautogui. So — exactly like audio/kinect_pointing.py — it lives here as a PURE
engine (stdlib only, ZERO third-party imports, ZERO Kinect contact):

  • `AirControlEngine.update(bodies, desktop)` consumes one frame's worth of
    bodies plus the virtual-desktop bounds and returns ONE `AirOp` describing
    what the live skill should do to the mouse this tick (move / grab-down /
    drag-move / release-up / click / scroll / idle). The engine owns ALL the
    state: which hand is engaged, the EMA-smoothed cursor target, whether a fist
    is currently held (a drag in progress), and the timing/travel needed to tell
    a CLICK from a DRAG.

  • Engagement is the safety + UX model: the controlling hand only drives the
    mouse when ENGAGED — raised above the waist AND pushed forward toward the
    sensor (hand.z meaningfully less than the shoulder.z, i.e. the arm extended).
    Lower or retract the hand and the engine DISENGAGES (releasing any held
    button). "Reach out to take the cursor, drop your hand to let go."

  • The cursor mapping is BODY-RELATIVE: the hand position is mapped within an
    interaction box anchored to the body (centred on spine_shoulder, in front of
    the chest) onto the FULL virtual desktop, so it works regardless of where the
    user stands. Then it's EMA-smoothed to kill Kinect jitter and clamped to the
    desktop bounds.

EVERY magic number is a named module constant (interaction-box size/offset,
smoothing, engage z-threshold, grab/click thresholds, scroll gain) so the live
behaviour is tunable without touching the algorithm — and so the tests assert
against the same numbers the code uses. NOTHING here raises on a malformed /
partial frame: a missing joint, a None body, an untracked hand all degrade to
an IDLE op (cursor released).

THE op the skill applies (see skills/air_control.py):
    AirOp(kind, x, y, scroll_amount, engaged, hand_state, reason)
  kind ∈ {"idle", "move", "down", "up", "click", "scroll"}.
  • "move"  → pyautogui.moveTo(x, y)          (cursor follows an OPEN hand)
  • "down"  → pyautogui.mouseDown()           (fist just closed — grab)
  • "up"    → pyautogui.mouseUp()             (fist opened after a DRAG — drop)
  • "click" → mouseDown()+mouseUp() at (x,y)  (quick close+open, little travel)
  • "scroll"→ pyautogui.scroll(scroll_amount) (lasso hand moved vertically)
  • "idle"  → no-op (no body / hand not engaged / nothing to do)
A "down"/"click" op also carries the (x, y) to pin the action at the grab spot.

FRAME SHAPE (one entry per audio.kinect_bridge.get_bodies() element — the
bridge is THE contract; verified against _parse_body_frame 2026-07-07)
===================================================================
    {"id": int,
     "joints": {name: (x, y, z, tracking_state), ...},   # metres, camera space
     "head": (x, y, z) | None,
     "distance_m": float | None,
     "facing": bool | None,
     "facing_yaw_deg": float | None,
     "hand_right": "open"|"closed"|"lasso"|"unknown",
     "hand_left":  ... (same)}

NB the hand-state keys are "hand_right"/"hand_left" (NOT "*_state" — an earlier
draft of this engine guessed wrong and read keys the bridge never emits), and
the bridge collapses the SDK's NotTracked into "unknown" (its _HAND_STATE_NAMES
maps both 0 Unknown and 1 NotTracked to "unknown"), so "nottracked" never
actually arrives — it's accepted below purely as defensive normalisation.
There are NO per-hand confidence keys.

Camera space (per the SDK / the bridge docstring): x increases to the sensor's
RIGHT, y increases UP, z increases with depth (forward, away from the sensor).
A joint's tracking_state is 0 not-tracked, 1 inferred, 2 tracked.
"""
from __future__ import annotations

import time
from typing import Any, NamedTuple, Optional


# ─── hand-state names (mirror the bridge's mapping of the PyKinectV2 enum) ──
HAND_OPEN = "open"
HAND_CLOSED = "closed"
HAND_LASSO = "lasso"
HAND_UNKNOWN = "unknown"
HAND_NOTTRACKED = "nottracked"


# ─── op kinds ───────────────────────────────────────────────────────────────
OP_IDLE = "idle"
OP_MOVE = "move"
OP_DOWN = "down"
OP_UP = "up"
OP_CLICK = "click"
OP_SCROLL = "scroll"


# ════════════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS — every magic number lives here, named, so the live feel
# can be calibrated without touching the algorithm. The tests assert against
# THESE same symbols, so a re-tune updates the constant and the test together.
# ════════════════════════════════════════════════════════════════════════════

# ── Engagement (the safety gate: "reach out to take control") ──────────────
# A joint counts toward engagement / mapping only when its TrackingState is
# >= this (2 == fully tracked). Inferred/untracked joints are too noisy to
# drive a cursor with.
MIN_TRACKING_STATE = 2

# The hand must be pushed FORWARD of the shoulder by at least this many metres
# (shoulder.z − hand.z, since z grows with depth → a forward hand has the
# SMALLER z) for the arm to count as "extended toward the sensor". ~0.18 m is a
# clear, deliberate reach without demanding a fully locked elbow. RAISE this if
# a relaxed arm keeps grabbing the cursor; LOWER it if you have to over-reach.
AIR_ENGAGE_FORWARD_M = 0.18

# The hand must also be raised at least this far ABOVE the waist (the spine_mid
# / spine_base height) — hand.y − waist.y >= this — so an arm hanging forward at
# the hip doesn't engage. ~0.05 m = "at or above the navel".
AIR_ENGAGE_ABOVE_WAIST_M = 0.05

# UPPER height bound — the "raising your hands ≠ mouse" gate (2026-07-15). A
# reach-to-control keeps the hand at/BELOW shoulder level; raising or stretching
# your arms lifts them ABOVE the shoulders. So a hand raised more than this far
# above the shoulder (hand.y − shoulder.y) does NOT start engagement, even if it
# is also forward + above the waist — which was the sole cause of the "false
# triggers when I raise my hands" reports (the old gate had forward + above-waist
# but NO upper bound). Applies to FRESH engagement only; a brief rise while
# already controlling won't drop you (that's the forward/waist hysteresis).
# Set a touch above the interaction-box top (~0.125 m above the shoulder) so a
# deliberate reach toward the TOP of the screen can still START control, while a
# clearly-raised/overhead arm (0.2 m+ above the shoulder) cannot. Once engaged,
# the hand may go higher (mapping clamps to the top edge) — the cap is fresh-
# engage only. RAISE to allow higher reaches; LOWER to be stricter.
AIR_ENGAGE_MAX_ABOVE_SHOULDER_M = 0.15

# Hysteresis: once engaged, the hand may relax back to within this (smaller)
# forward distance before we DISENGAGE, so the cursor doesn't flicker on/off at
# the exact threshold. Must be < AIR_ENGAGE_FORWARD_M.
AIR_DISENGAGE_FORWARD_M = 0.10

# ── Body-relative interaction box (the mapping domain) ─────────────────────
# The box is anchored to the body and mapped onto the FULL virtual desktop.
# Width/height in metres of the comfortable reach envelope in front of the
# chest: ~0.55 m wide × ~0.35 m tall is a relaxed forearm sweep. WIDEN for
# lower sensitivity (more hand travel per screen), NARROW for a flick-of-the-
# wrist feel.
AIR_BOX_WIDTH_M = 0.55
AIR_BOX_HEIGHT_M = 0.35

# Where the box centre sits relative to spine_shoulder (the chest anchor), in
# metres. x offset is 0 (centred left-right on the body). The box centre sits
# a little BELOW the shoulder (hands rest below shoulder height) — a positive
# value here is subtracted from spine_shoulder.y so the comfortable neutral
# hand position lands near screen-centre rather than near the top edge.
AIR_BOX_CENTER_DROP_M = 0.10

# ── Cursor smoothing (kill Kinect jitter) ──────────────────────────────────
# EMA factor: next = prev + AIR_SMOOTHING*(target − prev). Lower = smoother but
# laggier; higher = snappier but jitterier. ~0.35 is a good Kinect default.
AIR_SMOOTHING = 0.35

# ── Click vs drag discrimination ───────────────────────────────────────────
# A fist that closes then opens again within this many milliseconds AND with
# less than AIR_CLICK_TRAVEL_PX of cursor travel while closed is treated as a
# CLICK (down+up at the spot). A longer hold, or more travel, was a DRAG (the
# mouseDown already happened, so opening just releases — no extra click).
AIR_CLICK_MS = 350.0
AIR_CLICK_TRAVEL_PX = 25.0

# Grab debounce: ignore a hand-state flip that doesn't persist for at least this
# many milliseconds, so a single mis-classified frame (Kinect briefly reads
# OPEN↔CLOSED) can't spuriously start/stop a drag. Applied to the OPEN/CLOSED
# transition that drives grab.
AIR_GRAB_DEBOUNCE_MS = 80.0

# ── Scroll (lasso hand) ─────────────────────────────────────────────────────
# While the controlling hand is in the LASSO state, VERTICAL hand motion
# scrolls instead of moving the cursor (keeping scroll separate from move so
# they don't fight). The hand's vertical delta IN METRES since the last scroll
# frame is multiplied by this gain to get pyautogui scroll "clicks". Hand UP =
# scroll up (positive). RAISE for faster scrolling. NB: lasso is the documented,
# tunable scroll trigger — if the Kinect's lasso classification proves
# unreliable in practice, the skill can be pointed at a different trigger
# without changing this engine.
AIR_SCROLL_GAIN = 600.0

# Dead-zone (metres) of vertical hand motion below which we emit no scroll, so a
# perfectly still lasso hand doesn't dribble ±1 scroll clicks from jitter.
AIR_SCROLL_DEADZONE_M = 0.01


# ─── the op the engine emits each tick ─────────────────────────────────────
class AirOp(NamedTuple):
    """One tick's mouse instruction for the live skill to apply.

    kind          — OP_IDLE / OP_MOVE / OP_DOWN / OP_UP / OP_CLICK / OP_SCROLL.
    x, y          — target cursor coords (virtual-desktop pixels) for move /
                    down / click; None for idle / up / scroll.
    scroll_amount — integer pyautogui scroll clicks for OP_SCROLL (0 otherwise).
    engaged       — whether a controlling hand is engaged this tick (for status).
    hand_state    — the controlling hand's state name (for status / debugging).
    reason        — short human string (status line / tests / live calibration).
    """
    kind: str
    x: Optional[int] = None
    y: Optional[int] = None
    scroll_amount: int = 0
    engaged: bool = False
    hand_state: Optional[str] = None
    reason: str = ""


# ─── tiny helpers (plain tuples; no numpy) ─────────────────────────────────
def _joint_ok(j: Optional[tuple]) -> bool:
    """True when a (x, y, z, state) joint tuple is present and tracked."""
    if not j or len(j) < 4:
        return False
    try:
        return int(j[3]) >= MIN_TRACKING_STATE
    except Exception:
        return False


def _clamp(v: float, lo: float, hi: float) -> float:
    if hi < lo:                       # degenerate bounds → pin to lo
        return lo
    return lo if v < lo else (hi if v > hi else v)


def _waist_y(joints: dict) -> Optional[float]:
    """Best available 'waist' height: spine_mid, then spine_base, then hips."""
    for name in ("spine_mid", "spine_base"):
        j = joints.get(name)
        if _joint_ok(j):
            return float(j[1])
    # Average the two hips if present (tracked or not — last resort).
    hl, hr = joints.get("hip_left"), joints.get("hip_right")
    ys = [float(h[1]) for h in (hl, hr) if h and len(h) >= 2]
    if ys:
        return sum(ys) / len(ys)
    return None


def _hand_forwardness(joints: dict, side: str) -> Optional[float]:
    """How far the hand is pushed forward of the shoulder, in metres:
    shoulder.z − hand.z (z grows with depth, so a forward/extended hand yields a
    POSITIVE value). Requires both the hand and shoulder tracked. None when we
    can't tell."""
    hand = joints.get(f"hand_{side}")
    shoulder = joints.get(f"shoulder_{side}")
    if not _joint_ok(hand) or not _joint_ok(shoulder):
        return None
    return float(shoulder[2]) - float(hand[2])


def _hand_height_above_waist(joints: dict, side: str,
                             waist_y: Optional[float]) -> Optional[float]:
    """hand.y − waist.y in metres, or None if we can't tell."""
    hand = joints.get(f"hand_{side}")
    if not _joint_ok(hand) or waist_y is None:
        return None
    return float(hand[1]) - waist_y


def _hand_above_shoulder(joints: dict, side: str) -> Optional[float]:
    """hand.y − shoulder.y in metres (POSITIVE = the hand is raised above the
    shoulder). Used to reject raised/stretched arms from STARTING engagement — a
    reach-to-control sits at/below shoulder height. None when we can't tell (a
    missing shoulder must never block engagement — fail open)."""
    hand = joints.get(f"hand_{side}")
    shoulder = joints.get(f"shoulder_{side}")
    if not _joint_ok(hand) or not _joint_ok(shoulder):
        return None
    return float(hand[1]) - float(shoulder[1])


def _engagement_score(joints: dict, side: str,
                      waist_y: Optional[float]) -> Optional[float]:
    """A single comparable 'how engaged is this hand' score (higher = more
    engaged), or None if the hand isn't usable. Combines forwardness and
    height so the MORE extended/raised hand is picked as the controller.
    Returns None unless BOTH signals are available."""
    fwd = _hand_forwardness(joints, side)
    above = _hand_height_above_waist(joints, side, waist_y)
    if fwd is None or above is None:
        return None
    return fwd + max(0.0, above) * 0.5


# ─── the engine ─────────────────────────────────────────────────────────────
class AirControlEngine:
    """Stateful air-control engine. ONE instance per live loop; `update()` is
    called once per Kinect frame and returns the AirOp to apply.

    PURE: it never imports pyautogui or touches the sensor — it only reads the
    bodies dict the caller passes and reports what the mouse should do. All
    mutable state (engaged hand, EMA target, drag-in-progress, click timing)
    lives on the instance.

    `default_hand` ("right"|"left") breaks ties when both hands are equally
    engaged; the more-engaged hand always wins regardless. `now_fn` is injected
    so tests can drive time deterministically (no real clock, no sleeps).
    """

    def __init__(self, default_hand: str = "right", now_fn=None) -> None:
        self.default_hand = "left" if str(default_hand).lower() == "left" else "right"
        self._now = now_fn or time.monotonic

        # EMA-smoothed cursor target (virtual-desktop px); None until first move.
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None

        # Engagement / which hand is controlling right now.
        self._engaged: bool = False
        self._active_side: Optional[str] = None

        # Grab (drag) state machine.
        self._button_down: bool = False          # is a mouse button held (drag)?
        self._grab_start_t: float = 0.0          # when the fist closed
        self._grab_start_xy: tuple = (0, 0)      # cursor at fist-close (px)
        self._grab_max_travel: float = 0.0       # max travel since fist-close (px)
        # Debounce: the candidate next grab-state + when it was first seen.
        self._pending_closed: Optional[bool] = None
        self._pending_since: float = 0.0
        self._closed_stable: bool = False        # debounced fist-closed verdict

        # Scroll state: last lasso hand y (metres) to difference against.
        self._scroll_last_hand_y: Optional[float] = None

    # ── public state (for the status action) ────────────────────────────────
    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def active_side(self) -> Optional[str]:
        return self._active_side

    @property
    def button_down(self) -> bool:
        return self._button_down

    def cursor_target(self) -> Optional[tuple]:
        """The current smoothed cursor target (x, y) ints, or None."""
        if self._ema_x is None or self._ema_y is None:
            return None
        return (int(round(self._ema_x)), int(round(self._ema_y)))

    # ── helpers the skill calls on stop / loss of tracking ──────────────────
    def release(self) -> Optional[AirOp]:
        """Force-release any held button (used when the mode is turned off mid-
        drag, or when tracking is lost). Returns an OP_UP if a button was down,
        else None. Resets the grab/engagement state either way so the next
        engagement starts clean."""
        was_down = self._button_down
        self._reset_engagement()
        if was_down:
            return AirOp(OP_UP, engaged=False, reason="released (stopped)")
        return None

    def _reset_engagement(self) -> None:
        self._engaged = False
        self._active_side = None
        self._button_down = False
        self._closed_stable = False
        self._pending_closed = None
        self._scroll_last_hand_y = None
        # Keep _ema_* so re-engaging near the same spot doesn't jump.

    # ── the per-frame entry point ────────────────────────────────────────────
    def update(self, bodies: Any, desktop: tuple) -> AirOp:
        """Process one frame. `bodies` is audio.kinect_bridge.get_bodies()'s
        list (or None); `desktop` is (vx, vy, vw, vh) virtual-desktop bounds in
        px. Returns the AirOp to apply this tick. NEVER raises.

        Idle (no body / hand not engaged) releases any held button and returns
        OP_UP once, then OP_IDLE — so the cursor is never left grabbed when the
        hand drops or tracking is lost."""
        try:
            return self._update(bodies, desktop)
        except Exception:   # pragma: no cover - defensive: never raise into the loop
            # On any unexpected error, fail safe: drop the cursor.
            return self.release() or AirOp(OP_IDLE, reason="error")

    def _update(self, bodies: Any, desktop: tuple) -> AirOp:
        body = self._nearest_body(bodies)
        if body is None:
            # No one tracked → disengage (releasing a held button once).
            return self._idle_or_release("no body tracked")

        joints = body.get("joints") or {}
        if not isinstance(joints, dict) or not joints:
            return self._idle_or_release("no joints")

        waist_y = _waist_y(joints)
        side = self._pick_controlling_hand(joints, waist_y)
        if side is None:
            return self._idle_or_release("no hand engaged")

        # ── engagement with hysteresis ──────────────────────────────────────
        fwd = _hand_forwardness(joints, side)
        above = _hand_height_above_waist(joints, side, waist_y)
        if fwd is None or above is None:
            return self._idle_or_release("hand not tracked")

        # Engage threshold is higher than the disengage threshold (hysteresis):
        # to START controlling, reach past AIR_ENGAGE_FORWARD_M + above waist;
        # once engaged, stay engaged until the hand falls back inside
        # AIR_DISENGAGE_FORWARD_M (or drops below the waist).
        if self._engaged and self._active_side == side:
            still = (fwd >= AIR_DISENGAGE_FORWARD_M
                     and above >= AIR_ENGAGE_ABOVE_WAIST_M)
            engaged_now = still
        else:
            # FRESH engagement also rejects a raised/stretched arm: the hand must
            # not be more than AIR_ENGAGE_MAX_ABOVE_SHOULDER_M above the shoulder.
            # This is the fix for "false triggers when I raise my hands" — those
            # cleared forward + above-waist but sat well above shoulder height. A
            # missing shoulder joint fails OPEN (None → no block) so tracking gaps
            # never make the cursor un-grabbable. NOT applied to the hysteresis
            # branch above, so a brief rise mid-control won't drop you.
            above_sh = _hand_above_shoulder(joints, side)
            engaged_now = (fwd >= AIR_ENGAGE_FORWARD_M
                           and above >= AIR_ENGAGE_ABOVE_WAIST_M
                           and (above_sh is None
                                or above_sh <= AIR_ENGAGE_MAX_ABOVE_SHOULDER_M))

        if not engaged_now:
            return self._idle_or_release("hand retracted")

        # We are engaged on `side`.
        newly = not (self._engaged and self._active_side == side)
        self._engaged = True
        self._active_side = side
        if newly:
            # Fresh engagement: clear grab/scroll latches so we don't inherit a
            # stale held button or scroll origin from a prior hand.
            self._button_down = False
            self._closed_stable = False
            self._pending_closed = None
            self._scroll_last_hand_y = None

        hand_state = self._hand_state(body, side)

        # Map the hand to a smoothed cursor target regardless of state (so the
        # cursor is positioned correctly the instant a grab/click fires).
        tx, ty = self._map_hand_to_cursor(joints, side, desktop)
        self._apply_ema(tx, ty)
        cx, cy = self.cursor_target()   # type: ignore[misc]

        # ── LASSO → scroll (kept separate from move) ────────────────────────
        if hand_state == HAND_LASSO:
            return self._handle_scroll(joints, side, cx, cy, hand_state)
        # Leaving lasso: forget the scroll origin so re-entering re-seeds it.
        self._scroll_last_hand_y = None

        # ── OPEN / CLOSED → move / grab / drag / click ──────────────────────
        return self._handle_grab(hand_state, cx, cy)

    # ── body / hand selection ────────────────────────────────────────────────
    def _nearest_body(self, bodies: Any) -> Optional[dict]:
        """The nearest tracked body (smallest distance_m), or None. Mirrors the
        bridge's 'nearest body' convention used elsewhere."""
        if not bodies or not isinstance(bodies, (list, tuple)):
            return None
        best = None
        best_d = float("inf")
        for b in bodies:
            if not isinstance(b, dict):
                continue
            d = b.get("distance_m")
            try:
                d = float(d) if d is not None else float("inf")
            except Exception:
                d = float("inf")
            if d < best_d:
                best_d = d
                best = b
        # If nobody had a distance, just take the first dict body.
        if best is None:
            for b in bodies:
                if isinstance(b, dict):
                    return b
        return best

    def _pick_controlling_hand(self, joints: dict,
                               waist_y: Optional[float]) -> Optional[str]:
        """Whichever hand is MORE engaged (more extended/raised) is the
        controller; the default hand wins exact ties. Returns the side string or
        None if neither hand yields a usable engagement score.

        If we're already engaged on a side and that side is still usable, prefer
        to KEEP it (don't hand control to the other arm just because it crept
        slightly more forward this frame) — only switch when the current side is
        no longer scoreable."""
        scores = {}
        for side in ("right", "left"):
            s = _engagement_score(joints, side, waist_y)
            if s is not None:
                scores[side] = s
        if not scores:
            return None
        # Sticky: keep the active side while it's still scoreable.
        if self._active_side in scores:
            return self._active_side
        if len(scores) == 1:
            return next(iter(scores))
        # Both available → most engaged; default hand breaks an exact tie.
        r, l = scores["right"], scores["left"]
        if r > l:
            return "right"
        if l > r:
            return "left"
        return self.default_hand

    def _hand_state(self, body: dict, side: str) -> str:
        """The controlling hand's state name from the body dict, normalised.

        The bridge's key is "hand_right"/"hand_left" (see the FRAME SHAPE note
        in the module docstring — the bridge is the contract). We ALSO accept
        the legacy "hand_<side>_state" spelling as a fallback so a hand-rolled
        test fixture written against the old draft still degrades gracefully
        rather than silently reading HAND_UNKNOWN forever. A missing/non-string
        value → HAND_UNKNOWN (which the grab logic treats as 'not closed', so a
        bad read can never START a drag)."""
        val = body.get(f"hand_{side}")
        if not isinstance(val, str):
            val = body.get(f"hand_{side}_state")
        if not isinstance(val, str):
            return HAND_UNKNOWN
        v = val.strip().lower()
        if v in (HAND_OPEN, HAND_CLOSED, HAND_LASSO, HAND_NOTTRACKED):
            return v
        return HAND_UNKNOWN

    # ── cursor mapping (body-relative box → virtual desktop) ────────────────
    def _map_hand_to_cursor(self, joints: dict, side: str,
                            desktop: tuple) -> tuple:
        """Map the hand position inside the body-relative interaction box onto
        the virtual desktop. Returns (x, y) UNsmoothed target px (the EMA is
        applied by the caller). Y is INVERTED (hand up → cursor up).

        The box is centred on spine_shoulder (chest anchor), AIR_BOX_WIDTH_M
        wide × AIR_BOX_HEIGHT_M tall, dropped AIR_BOX_CENTER_DROP_M below the
        shoulder so a relaxed hand sits near screen-centre. The hand's offset
        from the box centre, normalised to [-0.5, 0.5] across the box, maps
        linearly to [vx, vx+vw] / [vy, vy+vh]."""
        vx, vy, vw, vh = desktop
        hand = joints.get(f"hand_{side}")
        anchor = (joints.get("spine_shoulder") or joints.get("spine_mid")
                  or joints.get("neck"))
        if not hand or anchor is None:
            # Shouldn't happen (engagement already required the hand), but stay
            # safe: park at desktop centre.
            return (vx + vw // 2, vy + vh // 2)

        ax, ay = float(anchor[0]), float(anchor[1])
        hx, hy = float(hand[0]), float(hand[1])
        cx_m = ax                          # box centre x = chest x
        cy_m = ay - AIR_BOX_CENTER_DROP_M  # box centre y = a bit below shoulder

        # Normalised position in the box, [-0.5 .. 0.5] (clamped to box edges).
        nx = _clamp((hx - cx_m) / AIR_BOX_WIDTH_M, -0.5, 0.5)
        ny = _clamp((hy - cy_m) / AIR_BOX_HEIGHT_M, -0.5, 0.5)

        # Map to desktop. x: left edge of box → left of desktop. y INVERTED:
        # hand HIGH (ny → +0.5) maps to the TOP of the desktop (vy).
        sx = vx + (nx + 0.5) * vw
        sy = vy + (0.5 - ny) * vh
        # Clamp into bounds (defensive; nx/ny are already clamped).
        sx = _clamp(sx, vx, vx + vw - 1)
        sy = _clamp(sy, vy, vy + vh - 1)
        return (sx, sy)

    def _apply_ema(self, tx: float, ty: float) -> None:
        """Advance the EMA-smoothed cursor toward the raw target (tx, ty)."""
        if self._ema_x is None or self._ema_y is None:
            self._ema_x, self._ema_y = float(tx), float(ty)
            return
        a = AIR_SMOOTHING
        self._ema_x += a * (float(tx) - self._ema_x)
        self._ema_y += a * (float(ty) - self._ema_y)

    # ── grab / click / drag state machine (OPEN / CLOSED) ───────────────────
    def _debounced_closed(self, raw_closed: bool) -> bool:
        """Debounce the raw fist-closed signal: a flip only takes effect after
        it persists AIR_GRAB_DEBOUNCE_MS. Returns the current STABLE verdict."""
        now = self._now()
        if raw_closed == self._closed_stable:
            # Already matches the stable state → clear any pending flip.
            self._pending_closed = None
            return self._closed_stable
        # raw differs from stable → it's a candidate flip.
        if self._pending_closed != raw_closed:
            self._pending_closed = raw_closed
            self._pending_since = now
            return self._closed_stable
        # Same pending candidate as last frame → has it held long enough?
        if (now - self._pending_since) * 1000.0 >= AIR_GRAB_DEBOUNCE_MS:
            self._closed_stable = raw_closed
            self._pending_closed = None
        return self._closed_stable

    def _handle_grab(self, hand_state: str, cx: int, cy: int) -> AirOp:
        """OPEN/CLOSED logic. CLOSED == grab/drag; OPEN == move (or release a
        held grab, possibly as a CLICK). Unknown/not-tracked hand → treat as
        'not closed' (so we never start a drag on a bad read) and just move."""
        raw_closed = (hand_state == HAND_CLOSED)
        closed = self._debounced_closed(raw_closed)

        if closed and not self._button_down:
            # Fist just closed → GRAB (mouseDown) at the current spot.
            self._button_down = True
            self._grab_start_t = self._now()
            self._grab_start_xy = (cx, cy)
            self._grab_max_travel = 0.0
            return AirOp(OP_DOWN, x=cx, y=cy, engaged=True,
                        hand_state=hand_state, reason="grab (fist closed)")

        if closed and self._button_down:
            # Still closed → DRAG: keep moving with the button held. Track the
            # furthest the cursor has wandered from the grab point (click test).
            dx = cx - self._grab_start_xy[0]
            dy = cy - self._grab_start_xy[1]
            travel = (dx * dx + dy * dy) ** 0.5
            if travel > self._grab_max_travel:
                self._grab_max_travel = travel
            return AirOp(OP_MOVE, x=cx, y=cy, engaged=True,
                        hand_state=hand_state, reason="drag (fist held)")

        if (not closed) and self._button_down:
            # Fist opened after a grab → RELEASE. Decide click vs drag: a SHORT
            # hold with LITTLE travel was a click; otherwise the mouseDown was a
            # drag and the mouseUp just drops it.
            held_ms = (self._now() - self._grab_start_t) * 1000.0
            self._button_down = False
            if (held_ms <= AIR_CLICK_MS
                    and self._grab_max_travel <= AIR_CLICK_TRAVEL_PX):
                gx, gy = self._grab_start_xy
                return AirOp(OP_CLICK, x=gx, y=gy, engaged=True,
                            hand_state=hand_state, reason="click (quick tap)")
            return AirOp(OP_UP, x=cx, y=cy, engaged=True,
                        hand_state=hand_state, reason="release (end drag)")

        # Open and not holding → ordinary cursor MOVE.
        return AirOp(OP_MOVE, x=cx, y=cy, engaged=True,
                    hand_state=hand_state, reason="move (open hand)")

    # ── scroll (LASSO) ───────────────────────────────────────────────────────
    def _handle_scroll(self, joints: dict, side: str, cx: int, cy: int,
                       hand_state: str) -> AirOp:
        """LASSO hand: vertical hand motion (metres) since the last scroll frame
        → a pyautogui scroll amount (hand UP = scroll up = positive). If a
        button was somehow held when we entered lasso, release it first (don't
        scroll mid-drag). Returns OP_SCROLL (possibly amount 0 inside the dead-
        zone) — never OP_MOVE, so scrolling can't fight the cursor."""
        if self._button_down:
            # Releasing first is safer than scrolling while dragging.
            self._button_down = False
            return AirOp(OP_UP, x=cx, y=cy, engaged=True,
                        hand_state=hand_state, reason="release (entered scroll)")
        hand = joints.get(f"hand_{side}")
        if not _joint_ok(hand):
            return AirOp(OP_IDLE, engaged=True, hand_state=hand_state,
                        reason="scroll: hand untracked")
        hy = float(hand[1])
        if self._scroll_last_hand_y is None:
            self._scroll_last_hand_y = hy
            return AirOp(OP_SCROLL, scroll_amount=0, engaged=True,
                        hand_state=hand_state, reason="scroll: armed")
        dy = hy - self._scroll_last_hand_y    # +ve = hand moved UP
        self._scroll_last_hand_y = hy
        if abs(dy) < AIR_SCROLL_DEADZONE_M:
            return AirOp(OP_SCROLL, scroll_amount=0, engaged=True,
                        hand_state=hand_state, reason="scroll: deadzone")
        amount = int(round(dy * AIR_SCROLL_GAIN))
        if amount == 0:
            return AirOp(OP_SCROLL, scroll_amount=0, engaged=True,
                        hand_state=hand_state, reason="scroll: sub-click")
        return AirOp(OP_SCROLL, scroll_amount=amount, engaged=True,
                    hand_state=hand_state,
                    reason=f"scroll {amount:+d} (lasso)")

    # ── idle / release-once ──────────────────────────────────────────────────
    def _idle_or_release(self, reason: str) -> AirOp:
        """Disengage. If a button was held, emit OP_UP once (so a dropped hand
        never strands a grabbed window); otherwise OP_IDLE. Resets engagement."""
        was_down = self._button_down
        self._reset_engagement()
        if was_down:
            return AirOp(OP_UP, engaged=False, reason=f"release ({reason})")
        return AirOp(OP_IDLE, engaged=False, reason=reason)
