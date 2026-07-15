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

import time
from typing import Any, Callable, Optional


# ─── gesture names (stable identifiers used by the skill's action map) ─────
WAVE = "wave"
RAISE_HAND = "raise_hand"
SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"

ALL_GESTURES = (WAVE, RAISE_HAND, SWIPE_LEFT, SWIPE_RIGHT)


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


def _joint_ok(j: Optional[tuple]) -> bool:
    """True when a (x, y, z, state) joint tuple is present and tracked."""
    if not j or len(j) < 4:
        return False
    try:
        return int(j[3]) >= MIN_TRACKING_STATE
    except Exception:
        return False


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

    # ── public API ────────────────────────────────────────────────────────
    def update(self, bodies: Any) -> Optional[str]:
        """Feed the latest get_bodies() result; return a gesture name or None.
        Never raises — a bad frame just yields None."""
        try:
            return self._update(bodies)
        except Exception:
            return None

    def reset(self) -> None:
        """Drop all history + cooldown (used on disable / between tests)."""
        self._hist.clear()
        self._last_fire_at = 0.0
        self._last_gesture = None

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

        if body is None:
            return None
        sample = self._sample_hand(body)
        if sample is None:
            return None
        sample["t"] = now
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
            if gesture:
                self._fire(gesture, now)
                return gesture
        return None

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
        # Only consider samples where the hand is above the waist (spine_base).
        above = [s for s in win
                 if s["spine_base_y"] is None
                 or s["hand_y"] > s["spine_base_y"] + SWIPE_ABOVE_WAIST_M]
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
