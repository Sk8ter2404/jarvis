"""
kinect_gestures — pure, hardware-free gesture recognizer for the Kinect v2.

WHY THIS MODULE EXISTS
======================
The Kinect skeleton stream (audio/kinect_bridge.get_bodies()) gives us per-frame
joint positions in camera-space metres. Turning that raw stream into DISCRETE,
debounced gestures (wave / raise-hand / swipe) is a self-contained signal-
processing problem that has NOTHING to do with the sensor, threading, or the
voice loop. So it lives here as a single class with ZERO imports beyond the
stdlib and ZERO Kinect contact:

  • `GestureRecognizer` consumes a rolling ~1.0-1.5 s history of body frames
    (whatever shape get_bodies() returns) for the NEAREST tracked body and
    emits at most one gesture name per call.

  • Every threshold is a named module constant so the live behaviour is tunable
    without touching the algorithm, and so the tests can assert against the same
    numbers the recognizer uses.

  • It NEVER raises on a malformed / partial frame — a missing joint, a None
    body list, an untracked joint state all degrade to "no gesture this tick".

The daemon poll loop, the bridge call, the action mapping, and the staging gate
all live in skills/kinect_gestures.py. This module is what that skill's unit
tests drive directly with fabricated frames.

FRAME SHAPE (one entry per get_bodies() element)
================================================
    {"id": int,
     "joints": {name: (x, y, z, tracking_state), ...},   # metres, camera space
     "head": (x, y, z) | None,
     "distance_m": float | None,
     "facing": bool | None}

Camera space (per the SDK / the bridge docstring): x increases to the sensor's
RIGHT, y increases UP, z increases with depth (forward, away from the sensor).
A joint's tracking_state is 0 not-tracked, 1 inferred, 2 tracked.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional


# ─── gesture names (stable identifiers used by the skill's action map) ─────
WAVE = "wave"
RAISE_HAND = "raise_hand"
SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"
TWO_HAND_X = "two_hand_x"      # both wrists crossed in front of the chest (held)
POINT = "point"                # one arm held out, two-finger point, aimed steady

ALL_GESTURES = (WAVE, RAISE_HAND, SWIPE_LEFT, SWIPE_RIGHT, TWO_HAND_X, POINT)


# ─── tunable thresholds (named so they're adjustable + test-visible) ───────
# History window: how much past motion the recognizer keeps. Long enough to see
# a couple of wave reversals, short enough that an old motion ages out.
HISTORY_SECONDS = 1.5

# A joint counts only when its TrackingState is >= this (2 == fully tracked).
MIN_TRACKING_STATE = 2

# WAVE — a hand held up (above its elbow) oscillating side to side. We count
# horizontal-direction reversals of the hand's x over the recent window.
WAVE_WINDOW_SECONDS = 1.2          # reversals must occur within this span
WAVE_MIN_REVERSALS = 2             # ≥2 direction changes == a wave
WAVE_MIN_AMPLITUDE_M = 0.06        # ignore micro-jitter: each swing ≥ this (m)
WAVE_HAND_ABOVE_ELBOW_M = 0.02     # hand.y must clear elbow.y by this (m)

# RAISE_HAND — a hand sustained above the head (hand.y > head.y) for a dwell.
RAISE_HAND_DWELL_SECONDS = 0.8     # must stay up at least this long
RAISE_HAND_ABOVE_HEAD_M = 0.0      # hand.y above head.y by ≥ this (m)

# SWIPE — a hand above the waist travelling laterally fast with little vertical
# change. Measured over a short trailing window.
SWIPE_WINDOW_SECONDS = 0.5         # the lateral move must complete within this
SWIPE_MIN_DX_M = 0.40              # ≥ this much horizontal travel (m)
SWIPE_MAX_DY_M = 0.25              # ≤ this much vertical travel (keeps it flat)
SWIPE_ABOVE_WAIST_M = 0.0          # hand.y above spine_base.y by ≥ this (m)

# Debounce: after ANY gesture fires, suppress re-fires (of the same OR any
# gesture) for this long, so one physical wave can't machine-gun the action.
COOLDOWN_SECONDS = 2.0

# TWO_HAND_X (2026-10-12) - both wrists CROSSED in front of the upper chest and
# held: the "stop / close this" sign. Chosen because it is the one two-hand pose
# that clears every other Kinect mode: it sits BELOW the air-mouse's raise line
# (a hand above the shoulder + open palm takes the cursor in 0.3 s, which would
# pre-empt any overhead two-hand sign), it is not a pinch-to-resize grab (both
# hands up, fists), and a relaxed pose never makes it. The one that LOOKS like it
# is FOLDED ARMS (a TV-watching rest), so the pose also demands the hands be at
# the upper chest (not at the biceps) AND pushed forward of the torso - folded
# arms hug the body. Fires ONCE per pose: the arms must come apart before it can
# fire again, so holding the sign can never close a second window.
X_HOLD_SECONDS = 0.8               # unbroken hold before it fires
X_GAP_TOLERANCE_SECONDS = 0.2      # a tracking blip this short does not restart the hold
X_REARM_ABSENT_SECONDS = 0.5       # pose must be gone this long before it can fire again
X_MIN_CROSS_M = 0.06               # each wrist this far past the other (x), at least
X_MAX_HAND_GAP_M = 0.45            # |x| gap between the hands: they are together
X_MAX_HAND_DY_M = 0.25             # hands at about the same height
X_MIN_FORWARD_M = 0.18             # each hand this far in FRONT of the shoulder line (z)
X_MAX_BELOW_SHOULDER_M = 0.20      # hands no lower than this under the shoulder line
X_MAX_ABOVE_SHOULDER_M = 0.05      # ...and not above it (that is the air-mouse's raise)
X_MIN_SHOULDER_SPAN_M = 0.15       # shoulders must be measurable to know "crossed"

# POINT (2026-10-12) - one arm held out toward the screens with the Kinect's
# two-finger "Lasso" hand state, aimed steady. The recognizer only reports the AIM
# (azimuth / elevation); mapping that to a monitor is monitor_for_aim() below.
POINT_DWELL_SECONDS = 0.8          # aim must stay put this long
POINT_GAP_TOLERANCE_SECONDS = 0.3  # Lasso flickers; a blip this short keeps the hold
POINT_STEADY_AZIMUTH_DEG = 8.0     # max azimuth spread across the dwell
POINT_STEADY_ELEVATION_DEG = 10.0  # max elevation spread across the dwell
POINT_MIN_FORWARD_M = 0.25         # hand this far in front of the shoulder (z)
POINT_MIN_EXTENSION_M = 0.45       # shoulder->hand distance (arm is out, not folded)
POINT_MIN_ELEVATION_DEG = -25.0    # aim no lower than this (a desk reach is lower)
POINT_MAX_ELEVATION_DEG = 70.0
POINT_REARM_ABSENT_SECONDS = 0.6   # arm down this long -> a new point can fire
POINT_REAIM_DEG = 12.0             # ...or swing the aim this far to pick another monitor
POINT_MONITOR_BIN_DEG = 24.0       # azimuth width of one monitor in a row
POINT_TOP_ELEVATION_DEG = 28.0     # aimed this high -> the monitor stacked ABOVE the row


def _joint_ok(j: Optional[tuple]) -> bool:
    """True when a (x, y, z, state) joint tuple is present and tracked."""
    if not j or len(j) < 4:
        return False
    try:
        return int(j[3]) >= MIN_TRACKING_STATE
    except Exception:
        return False


def _xyz(j: Optional[tuple]) -> Optional[tuple]:
    """(x, y, z) of a tracked, finite, non-zero-fill joint, else None."""
    if not _joint_ok(j):
        return None
    try:
        x, y, z = float(j[0]), float(j[1]), float(j[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    if x == 0.0 and y == 0.0 and z == 0.0:     # NotTracked zero-fill
        return None
    return (x, y, z)


def two_hand_x_pose(body: Any) -> bool:
    """True when this body holds the crossed-wrists sign THIS frame (see the
    TWO_HAND_X notes). Orientation-agnostic: "crossed" is judged against the
    shoulders' own left/right order, so it holds whichever way the stream is
    mirrored. FAILS CLOSED on any missing joint. PURE; never raises."""
    try:
        joints = (body or {}).get("joints") or {}
        hl = _xyz(joints.get("hand_left"))
        hr = _xyz(joints.get("hand_right"))
        sl = _xyz(joints.get("shoulder_left"))
        sr = _xyz(joints.get("shoulder_right"))
        if hl is None or hr is None or sl is None or sr is None:
            return False
        span = sr[0] - sl[0]
        if abs(span) < X_MIN_SHOULDER_SPAN_M:
            return False
        sign = 1.0 if span > 0 else -1.0
        # Wrists crossed: the right hand has moved past the left one.
        if (hr[0] - hl[0]) * sign > -X_MIN_CROSS_M:
            return False
        if abs(hr[0] - hl[0]) > X_MAX_HAND_GAP_M:
            return False
        if abs(hr[1] - hl[1]) > X_MAX_HAND_DY_M:
            return False
        shoulder_y = (sl[1] + sr[1]) / 2.0
        shoulder_z = (sl[2] + sr[2]) / 2.0
        for h in (hl, hr):
            dy = h[1] - shoulder_y
            if dy < -X_MAX_BELOW_SHOULDER_M or dy > X_MAX_ABOVE_SHOULDER_M:
                return False
            if (shoulder_z - h[2]) < X_MIN_FORWARD_M:    # z grows away from the sensor
                return False
        return True
    except Exception:
        return False


def point_aim(body: Any, require_lasso: bool = True) -> Optional[dict]:
    """The aim of a held-out, pointing arm on this body THIS frame, or None.

    Returns {"side", "azimuth_deg", "elevation_deg", "extension_m"}. Azimuth is
    the sideways angle of the shoulder->hand ray measured from straight at the
    sensor (+ = the owner's right, the same +x the air-mouse maps to the larger
    cursor x); elevation is the upward angle of that ray. With require_lasso the
    hand must also read the Kinect's two-finger "lasso" state, which an air-mouse
    open palm and a fist both are not - that is what keeps a reach for the
    keyboard (or the air-mouse raise) from reading as a point. PURE; never raises."""
    try:
        if not isinstance(body, dict):
            return None
        joints = body.get("joints") or {}
        best = None
        for side in ("right", "left"):
            if require_lasso and str(body.get(f"hand_{side}") or "").lower() != "lasso":
                continue
            sh = _xyz(joints.get(f"shoulder_{side}"))
            hd = _xyz(joints.get(f"hand_{side}"))
            if sh is None or hd is None:
                continue
            dx, dy = hd[0] - sh[0], hd[1] - sh[1]
            fwd = sh[2] - hd[2]                       # z grows away from the sensor
            ext = math.sqrt(dx * dx + dy * dy + fwd * fwd)
            if fwd < POINT_MIN_FORWARD_M or ext < POINT_MIN_EXTENSION_M:
                continue
            elev = math.degrees(math.atan2(dy, math.hypot(dx, fwd)))
            if not (POINT_MIN_ELEVATION_DEG <= elev <= POINT_MAX_ELEVATION_DEG):
                continue
            cand = {"side": side, "azimuth_deg": math.degrees(math.atan2(dx, fwd)),
                    "elevation_deg": elev, "extension_m": ext}
            if best is None or ext > best["extension_m"]:
                best = cand
        return best
    except Exception:
        return None


def monitor_for_aim(azimuth_deg: float, elevation_deg: float, monitors: Any,
                    bin_deg: float = POINT_MONITOR_BIN_DEG) -> Optional[str]:
    """The monitor a pointing aim lands on, from the {name: (x, y, w, h)} layout
    (core.config.MONITORS shape), or None when the layout is empty. Aimed high
    (POINT_TOP_ELEVATION_DEG) picks the monitor stacked ABOVE the row if there is
    one; otherwise the row (left to right by x) is split into `bin_deg`-wide
    azimuth bins centred on straight ahead, the outer bins taking everything
    beyond them. No per-desk calibration. PURE; never raises."""
    try:
        rects = []
        for name, r in dict(monitors or {}).items():
            x, y, w, h = (float(v) for v in tuple(r)[:4])
            if w > 0 and h > 0:
                rects.append((str(name), x + w / 2.0, y + h / 2.0, h))
        if not rects:
            return None
        median_cy = sorted(r[2] for r in rects)[len(rects) // 2]
        above = [r for r in rects if r[2] < median_cy - r[3] / 2.0]
        row = [r for r in rects if r not in above] or rects
        if above and elevation_deg >= POINT_TOP_ELEVATION_DEG:
            return min(above, key=lambda r: r[2])[0]
        row.sort(key=lambda r: r[1])
        n = len(row)
        width = float(bin_deg) if bin_deg and bin_deg > 0 else POINT_MONITOR_BIN_DEG
        idx = int(math.floor(float(azimuth_deg) / width + n / 2.0))
        return row[max(0, min(n - 1, idx))][0]
    except Exception:
        return None


def _nearest_body(bodies: Any) -> Optional[dict]:
    """Pick the body we should read gestures off — the owner at the desk — from a
    get_bodies()-shaped list, or None.

    Ranking: FACING-the-sensor bodies first, then closest. "Closest" uses
    distance_m when present; bodies without a distance sort last so a body we CAN
    range always wins.

    GHOST FIX (2026-07-15): the bridge's _body_is_real gate already drops
    inferred/zero-fill phantoms, but a true reflection (TV/mirror/window) tracks
    cleanly at state 2 and can sit NEARER than the real owner — the old pure-
    distance pick would let that reflection STARVE the owner (only the ghost is
    ever sampled) or fire a WAVE/SWIPE/RAISE_HAND off it. The owner at the desk
    faces the sensor; an off-axis reflection typically does not, so preferring a
    facing body closes the reflection-hijack the distance sort can't."""
    if not bodies:
        return None
    try:
        candidates = [b for b in bodies if isinstance(b, dict) and b.get("joints")]
    except TypeError:
        return None
    if not candidates:
        return None

    def _key(b: dict):
        d = b.get("distance_m")
        dist = d if isinstance(d, (int, float)) and d > 0 else float("inf")
        # facing True → 0 (preferred), unknown/None → 1, explicitly not-facing → 2,
        # so a facing owner always outranks a non-facing reflection regardless of
        # distance, while an unknown-facing body still beats a known side-on one.
        facing = b.get("facing")
        facing_rank = 0 if facing is True else (2 if facing is False else 1)
        return (facing_rank, dist)

    return min(candidates, key=_key)


class GestureRecognizer:
    """Stateful recognizer fed one body-frame at a time.

    Usage (the skill's poll loop does exactly this):
        rec = GestureRecognizer()
        while ...:
            g = rec.update(kinect_bridge.get_bodies())   # 15-20 Hz
            if g: dispatch(g)

    `update()` appends the nearest body's hand/elbow/head/spine sample to a
    short rolling history, runs the three detectors, and returns the FIRST
    gesture that fires (or None). A successful fire starts a global cooldown so
    nothing re-fires until COOLDOWN_SECONDS have passed.

    `now_fn` is injectable so tests can drive deterministic timestamps without
    sleeping; it defaults to time.monotonic.
    """

    def __init__(self, now_fn: Callable[[], float] = time.monotonic):
        self._now = now_fn
        # Each sample: dict(t, hand_x, hand_y, elbow_y, head_y, spine_base_y,
        #                    above_elbow:bool, side:str). side ∈ {"left","right"}
        # records which hand we sampled (for diagnostics only).
        self._hist: list[dict] = []
        self._last_fire_at: float = 0.0
        self._last_gesture: Optional[str] = None
        # Which gestures are live (None = all). The skill narrows this from its
        # per-gesture settings each tick so a gesture that is switched off neither
        # fires nor burns the shared cooldown.
        self._enabled: Optional[frozenset] = None
        # POINT needs the Kinect's two-finger Lasso state unless the owner turned
        # that requirement off (the skill sets it from KINECT_POINT_REQUIRE_LASSO).
        self._require_lasso: bool = True
        # Extra facts about the gesture that last fired (POINT: azimuth /
        # elevation). {} for gestures that carry none.
        self.last_detail: dict = {}
        # TWO_HAND_X hold state (see X_HOLD_SECONDS).
        self._x_since: Optional[float] = None
        self._x_last_pose: float = 0.0
        self._x_absent_since: Optional[float] = None
        self._x_armed: bool = False
        # POINT hold state: [(t, azimuth, elevation, side)] of the unbroken aim.
        self._point_hist: list[tuple] = []
        self._point_absent_since: Optional[float] = None
        self._point_fired_az: Optional[float] = None

    # ── public API ────────────────────────────────────────────────────────
    def update(self, bodies: Any) -> Optional[str]:
        """Feed the latest get_bodies() result; return a gesture name or None.
        Never raises — a bad frame just yields None."""
        try:
            return self._update(bodies)
        except Exception:
            return None

    def reset(self) -> None:
        """Drop all history + cooldown (used on disable / between tests). The
        crossed-wrists and point detectors come back DISARMED: the pose must be
        seen to lapse once before either can fire, so a sign that was already held
        when the air-mouse / two-hand mode let go can never fire on release."""
        self._hist.clear()
        self._last_fire_at = 0.0
        self._last_gesture = None
        self.last_detail = {}
        self._x_since = None
        self._x_absent_since = None
        self._x_armed = False
        self._point_hist.clear()
        self._point_absent_since = None
        self._point_fired_az = None

    def set_enabled(self, gestures: Any) -> None:
        """Restrict recognition to these gesture names (None = every gesture)."""
        self._enabled = None if gestures is None else frozenset(gestures)

    def set_require_lasso(self, on: bool) -> None:
        self._require_lasso = bool(on)

    def is_enabled(self, gesture: str) -> bool:
        return self._enabled is None or gesture in self._enabled

    @property
    def last_gesture(self) -> Optional[str]:
        return self._last_gesture

    def in_cooldown(self, now: Optional[float] = None) -> bool:
        now = self._now() if now is None else now
        return (now - self._last_fire_at) < COOLDOWN_SECONDS

    # ── internals ───────────────────────────────────────────────────────────
    def _sample_hand(self, body: dict) -> Optional[dict]:
        """Extract the best raised-hand sample from a body, or None.

        Prefers whichever hand is higher relative to its elbow (so the gesturing
        arm is the one we track even if both hands are visible). Falls back to
        either tracked hand. Returns the per-tick sample dict (without `t`)."""
        joints = body.get("joints") or {}
        head = joints.get("head")
        spine_base = joints.get("spine_base")
        head_y = float(head[1]) if _joint_ok(head) else None
        spine_base_y = float(spine_base[1]) if _joint_ok(spine_base) else None

        best = None
        for side, hand_k, elbow_k in (
            ("right", "hand_right", "elbow_right"),
            ("left", "hand_left", "elbow_left"),
        ):
            hand = joints.get(hand_k)
            elbow = joints.get(elbow_k)
            if not _joint_ok(hand):
                continue
            hand_x = float(hand[0])
            hand_y = float(hand[1])
            elbow_y = float(elbow[1]) if _joint_ok(elbow) else None
            above_elbow = (elbow_y is not None
                           and hand_y > elbow_y + WAVE_HAND_ABOVE_ELBOW_M)
            sample = {
                "hand_x": hand_x, "hand_y": hand_y, "elbow_y": elbow_y,
                "head_y": head_y, "spine_base_y": spine_base_y,
                "above_elbow": above_elbow, "side": side,
            }
            # Rank candidate hands: an above-elbow hand beats one that isn't;
            # among equals, the higher hand wins (more likely the gesture arm).
            if best is None:
                best = sample
            else:
                better = (above_elbow and not best["above_elbow"]) or (
                    above_elbow == best["above_elbow"]
                    and hand_y > best["hand_y"]
                )
                if better:
                    best = sample
        return best

    def _update(self, bodies: Any) -> Optional[str]:
        now = self._now()
        body = _nearest_body(bodies)

        # Age out old samples regardless of whether this tick had a body.
        cutoff = now - HISTORY_SECONDS
        self._hist = [s for s in self._hist if s["t"] >= cutoff]

        self.last_detail = {}
        if body is None:
            self._x_since = None
            self._point_hist.clear()
            return None
        # The two-handed / pointing poses are tracked on EVERY frame (their holds
        # must see the pose lapse to re-arm), but only fire outside the cooldown.
        x_pose = self._track_two_hand_x(body, now)
        point_fire = self._track_point(body, now)
        if x_pose:
            # Hands together in front of the chest: no single-hand detector may
            # read the arms' travel into the sign as a swipe / wave.
            self._hist.clear()
            if (self._x_armed and self._x_since is not None
                    and now - self._x_since >= X_HOLD_SECONDS
                    and not self.in_cooldown(now)):
                self._x_armed = False
                self._x_since = None
                self._fire(TWO_HAND_X, now)
                return TWO_HAND_X
            return None
        if point_fire is not None and not self.in_cooldown(now):
            self._point_fired_az = point_fire["azimuth_deg"]
            self._point_hist.clear()
            self._fire(POINT, now)
            self.last_detail = dict(point_fire)
            return POINT
        sample = self._sample_hand(body)
        if sample is None:
            return None
        sample["t"] = now
        # SIDE LOCK (2026-10-04). The sampled hand is re-chosen every tick (the
        # higher one), so with BOTH hands up it alternates left/right - and the
        # history then holds a left-hand x next to a right-hand x ~0.4 m apart:
        # a phantom 0.4 m "swipe" or a run of fake wave "reversals" (a false
        # SWIPE cancels speech and a pending confirmation). A change of side
        # therefore CLEARS the history: every detector only ever sees one hand.
        if self._hist and self._hist[-1].get("side") != sample.get("side"):
            self._hist.clear()
        self._hist.append(sample)

        # Global debounce: nothing fires during the cooldown window.
        if self.in_cooldown(now):
            return None

        # Detector priority: raise-hand (a deliberate sustained pose) and wave
        # both read as "summon"; we check raise-hand first because a swipe and a
        # wave can both involve lateral motion, and an intentional hold is the
        # least ambiguous. Order: RAISE_HAND → WAVE → SWIPE.
        for detector in (self._detect_raise_hand,
                         self._detect_wave,
                         self._detect_swipe):
            gesture = detector(now)
            if gesture and self.is_enabled(gesture):
                self._fire(gesture, now)
                return gesture
        return None

    def _track_two_hand_x(self, body: dict, now: float) -> bool:
        """Advance the crossed-wrists hold state for this frame; True while the
        sign is held (and the gesture is enabled). Re-arms once the pose has been
        gone X_REARM_ABSENT_SECONDS."""
        pose = self.is_enabled(TWO_HAND_X) and two_hand_x_pose(body)
        if pose:
            self._x_absent_since = None
            self._x_last_pose = now
            if self._x_since is None:
                self._x_since = now
            return True
        if self._x_absent_since is None:
            self._x_absent_since = now
        if now - self._x_last_pose > X_GAP_TOLERANCE_SECONDS:
            self._x_since = None
        if now - self._x_absent_since >= X_REARM_ABSENT_SECONDS:
            self._x_armed = True
        return False

    def _track_point(self, body: dict, now: float) -> Optional[dict]:
        """Advance the point hold state; return the settled aim
        ({side, azimuth_deg, elevation_deg, extension_m}) once the arm has been
        held steady for POINT_DWELL_SECONDS, else None. One fire per point: the
        arm must come down POINT_REARM_ABSENT_SECONDS, or swing POINT_REAIM_DEG to
        another monitor, before it can fire again."""
        aim = (point_aim(body, require_lasso=self._require_lasso)
               if self.is_enabled(POINT) else None)
        if aim is None:
            if self._point_absent_since is None:
                self._point_absent_since = now
            if (self._point_hist
                    and now - self._point_hist[-1][0] > POINT_GAP_TOLERANCE_SECONDS):
                self._point_hist.clear()
            if now - self._point_absent_since >= POINT_REARM_ABSENT_SECONDS:
                self._point_fired_az = None
            return None
        self._point_absent_since = None
        if self._point_hist and (
                self._point_hist[-1][3] != aim["side"]
                or now - self._point_hist[-1][0] > POINT_GAP_TOLERANCE_SECONDS):
            self._point_hist.clear()
        self._point_hist.append((now, aim["azimuth_deg"], aim["elevation_deg"],
                                 aim["side"]))
        if len(self._point_hist) < 3:
            return None
        if now - self._point_hist[0][0] < POINT_DWELL_SECONDS:
            return None
        azs = [h[1] for h in self._point_hist]
        els = [h[2] for h in self._point_hist]
        if (max(azs) - min(azs) > POINT_STEADY_AZIMUTH_DEG
                or max(els) - min(els) > POINT_STEADY_ELEVATION_DEG):
            # Not steady: keep only the newest dwell's worth so a settled aim can
            # still complete after a swing.
            lo = now - POINT_DWELL_SECONDS
            self._point_hist = [h for h in self._point_hist if h[0] >= lo]
            return None
        mean_az = sum(azs) / len(azs)
        if (self._point_fired_az is not None
                and abs(mean_az - self._point_fired_az) < POINT_REAIM_DEG):
            return None
        return {"side": aim["side"], "azimuth_deg": mean_az,
                "elevation_deg": sum(els) / len(els),
                "extension_m": aim["extension_m"]}

    def _fire(self, gesture: str, now: float) -> None:
        self._last_fire_at = now
        self._last_gesture = gesture
        # Clear history so the motion that just fired can't immediately satisfy
        # another detector the instant the cooldown lapses.
        self._hist.clear()

    # ── detectors (each reads self._hist; returns a gesture name or None) ───
    def _window(self, now: float, seconds: float) -> list[dict]:
        lo = now - seconds
        return [s for s in self._hist if s["t"] >= lo]

    def _detect_raise_hand(self, now: float) -> Optional[str]:
        """A hand sustained above the head for ≥ RAISE_HAND_DWELL_SECONDS.

        Scans the TRAILING run of consecutive 'hand above head' samples in the
        full history (ending at the most recent sample): if that unbroken run
        spans at least the dwell, it's a deliberate raise. Using the trailing
        run (not a fixed window) means a dip below the head breaks the run, and
        the dwell is measured edge-to-edge so it doesn't depend on the poll rate
        lining a sample up exactly on the window boundary."""
        if len(self._hist) < 2:
            return None

        def _up(s) -> bool:
            return (s["head_y"] is not None
                    and s["hand_y"] > s["head_y"] + RAISE_HAND_ABOVE_HEAD_M)

        # The latest sample must currently be 'up' for the pose to be held now.
        if not _up(self._hist[-1]):
            return None
        # Walk backward collecting the unbroken trailing up-run.
        run = []
        for s in reversed(self._hist):
            if _up(s):
                run.append(s)
            else:
                break
        if len(run) < 2:
            return None
        span = run[0]["t"] - run[-1]["t"]   # newest - oldest in the run
        return RAISE_HAND if span >= RAISE_HAND_DWELL_SECONDS else None

    def _detect_wave(self, now: float) -> Optional[str]:
        """Hand above its elbow, oscillating side-to-side ≥ WAVE_MIN_REVERSALS
        times within WAVE_WINDOW_SECONDS, each swing ≥ WAVE_MIN_AMPLITUDE_M."""
        win = self._window(now, WAVE_WINDOW_SECONDS)
        up = [s for s in win if s["above_elbow"]]
        if len(up) < 3:
            return None
        # Count horizontal-direction reversals using a small amplitude gate so
        # jitter doesn't register as a swing. Walk the x series tracking the
        # current travel direction; a reversal is a sign flip AFTER the hand has
        # moved at least WAVE_MIN_AMPLITUDE_M since the last extreme.
        xs = [s["hand_x"] for s in up]
        reversals = 0
        direction = 0          # -1 left, +1 right, 0 unknown
        last_extreme = xs[0]
        for x in xs[1:]:
            dx = x - last_extreme
            if abs(dx) < WAVE_MIN_AMPLITUDE_M:
                continue
            step_dir = 1 if dx > 0 else -1
            if direction == 0:
                direction = step_dir
            elif step_dir != direction:
                reversals += 1
                direction = step_dir
            last_extreme = x
        return WAVE if reversals >= WAVE_MIN_REVERSALS else None

    def _detect_swipe(self, now: float) -> Optional[str]:
        """A hand above the waist crossing ≥ SWIPE_MIN_DX_M horizontally within
        SWIPE_WINDOW_SECONDS while moving ≤ SWIPE_MAX_DY_M vertically.
        Returns SWIPE_LEFT / SWIPE_RIGHT by travel direction (sensor frame)."""
        win = self._window(now, SWIPE_WINDOW_SECONDS)
        # Only consider samples where the hand is VERIFIABLY above the waist
        # (spine_base). An untracked waist (spine_base_y None — e.g. the desk
        # occluding the lower torso, the common seated case) means the sample
        # CANNOT qualify: fail CLOSED, matching the raise-hand/wave detectors'
        # reference-joint contract. The old `is None or` accepted every sample
        # whenever spine_base was untracked, so a flat desk-level reach for the
        # keyboard fired SWIPE and cancelled speech + pending confirmations
        # (2026-07-21 audit).
        above = [s for s in win
                 if s["spine_base_y"] is not None
                 and s["hand_y"] > s["spine_base_y"] + SWIPE_ABOVE_WAIST_M]
        if len(above) < 2:
            return None
        xs = [s["hand_x"] for s in above]
        ys = [s["hand_y"] for s in above]
        dx = xs[-1] - xs[0]
        dy = max(ys) - min(ys)
        if abs(dx) < SWIPE_MIN_DX_M:
            return None
        if dy > SWIPE_MAX_DY_M:
            return None
        # Camera-space x increases to the sensor's RIGHT. A hand moving toward
        # +x is a rightward swipe from the sensor's point of view.
        return SWIPE_RIGHT if dx > 0 else SWIPE_LEFT
