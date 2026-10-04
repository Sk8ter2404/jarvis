"""
kinect_stabilizer - the ONE shared per-frame hand stabiliser every Kinect
consumer reads (2026-10-04, the owner's "hand tracking is unstable").

WHY THIS EXISTS
===============
Before this module nothing between the Kinect body frame and the mouse filtered
the data, debounced on REAL sensor frames, or used the SDK's grip confidence.
Each consumer (air-mouse, two-hand, gestures, pointing) re-derived "which body,
which hand, is it raised, is it closed" on its own, per POLL, from one raw
frame - and the live log of 2026-10-04 13:09-13:36 showed what that costs:

  * two-hand mode engaged 51 times in 28 min, 48 of them gone within a second
    (up to 6 engages in ONE second), because "both hands raised and fully
    Tracked" flipped on a single frame - and every engage stood the air-mouse
    down (button up, drag dropped);
  * the cursor hand flipped left/right, and lift went None for a single
    Inferred frame, releasing the cursor instantly;
  * grip open/closed/open inside a second while engaged (each one a click),
    with LOW-confidence and duplicate-read frames counted as votes.

THE FIX (fix once here; every consumer reads one shared result)
==============================================================
The bridge's body pump calls HandStabilizer.process() exactly ONCE per new
body frame and publishes the returned snapshot (kinect_bridge.get_tracked_frame).
Everything is time-based on real frame times, so a consumer polling twice per
frame cannot double-count, and a skipped frame cannot shorten a dwell.

  * OWNER BODY   - sticky by tracking id; held through a short loss; a
                   different body takes over only if the owner is gone past
                   the grace or another body is clearly nearer for a while.
  * HAND POSITION - Tracked wrist + the SMOOTHED hand-wrist offset (the wrist
                   is Tracked ~94% of frames when the hand joint is ~35%, it is
                   steadier, and closing the hand moves the hand joint but not
                   the wrist), else the Tracked hand, else the Inferred hand,
                   else HOLD the last value for the loss grace. Single-frame
                   jumps are rejected. A One Euro filter smooths it: heavy
                   smoothing at rest (jitter), light smoothing in motion
                   (latency).
  * LIFT         - hand height above the shoulder line, measured ONLY from a
                   Tracked hand / wrist (an Inferred guess never fabricates a
                   raise), held through the loss grace instead of dropping to
                   None on one bad frame.
  * RAISED       - per hand, with an enter dwell above the engage margin and
                   an exit dwell below the (lower) stay margin.
  * GRIP         - changes ONLY on High-confidence votes from a Tracked hand
                   joint, with separate time dwells to close (press) and to
                   open (release); Low confidence / Unknown / an untracked
                   joint is "no vote". A flicker shorter than the close dwell
                   can never press a button.
  * ACTIVE HAND  - sticky: the holder keeps it while raised; the other hand
                   takes over only when it is raised AND leads by a clear
                   margin for a dwell (or the holder is lowered).
  * TWO-HAND     - enter only after both hands stay raised for a dwell, exit
                   only after "not both" persists, and no re-entry for a
                   re-arm window after an exit: it cannot flap within a second.

PURE: stdlib only, no sensor, no clock (the caller passes frame times), no
config import at module level (params are read through an injectable
function). NEVER raises out of process(): a malformed frame degrades to "no
measurement this frame", which the grace windows then absorb.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

# The Kinect v2 body stream runs at 30 Hz.
FRAME_PERIOD_S = 1.0 / 30.0

# Every tunable, with its shipped default. core/config.py holds the SAME
# literals under the same names (the live, owner-overridable values; a test
# pins the two copies equal - the stale-duplicate bug class). Read live through
# load_params() each frame, so a Settings tweak needs no restart.
DEFAULTS: dict = {
    # One Euro filter on the hand position (metres, m/s).
    "KINECT_HAND_FILTER_MIN_CUTOFF_HZ": 0.7,
    "KINECT_HAND_FILTER_BETA": 12.0,
    "KINECT_HAND_FILTER_D_CUTOFF_HZ": 1.0,
    # Reject a single-frame hand jump larger than this (scaled by frames
    # elapsed), for at most this many consecutive frames.
    "KINECT_HAND_JUMP_REJECT_M": 0.25,
    "KINECT_HAND_JUMP_REJECT_FRAMES": 2,
    # Hold the last good hand value this long before "hand = none".
    "KINECT_HAND_LOSS_GRACE_SEC": 0.30,
    # A hand-wrist offset older than this is not trusted for the wrist stand-in.
    "KINECT_WRIST_OFFSET_MAX_AGE_SEC": 2.0,
    # Cutoff of the smoothing on that hand-wrist offset.
    "KINECT_HAND_OFFSET_CUTOFF_HZ": 1.0,
    # Grip: High-confidence votes only; dwell to CLOSE (press) and to OPEN
    # (release); a vote run is broken by a contrary vote or a gap this long.
    "KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE": True,
    "KINECT_GRIP_CLOSE_SEC": 0.09,
    "KINECT_GRIP_CLOSE_MIN_VOTES": 4,
    "KINECT_GRIP_OPEN_SEC": 0.03,
    "KINECT_GRIP_OPEN_MIN_VOTES": 2,
    "KINECT_GRIP_VOTE_GAP_SEC": 0.10,
    "KINECT_GRIP_LASSO_AS": "none",
    # Raised: engage above UP, stay down to DOWN (metres of lift).
    "KINECT_LIFT_UP_MARGIN": 0.07,
    "KINECT_LIFT_DOWN_MARGIN": -0.10,
    "KINECT_RAISE_ENTER_SEC": 0.10,
    "KINECT_RAISE_EXIT_SEC": 0.10,
    # Active hand: the challenger must lead by this much lift for this long.
    "KINECT_ACTIVE_HAND_SWITCH_LEAD_M": 0.15,
    "KINECT_ACTIVE_HAND_SWITCH_SEC": 0.40,
    # Two-hand mode: enter dwell, exit dwell, re-arm after an exit.
    "KINECT_TWO_HAND_ENTER_SEC": 0.25,
    "KINECT_TWO_HAND_EXIT_SEC": 0.20,
    "KINECT_TWO_HAND_REARM_SEC": 0.60,
    # Owner body: hold through a loss; switch to a nearer body only when it is
    # this much nearer for this long.
    "KINECT_OWNER_LOSS_GRACE_SEC": 0.30,
    "KINECT_OWNER_SWITCH_NEARER_M": 0.25,
    "KINECT_OWNER_SWITCH_SEC": 1.0,
}

_SIDES = ("left", "right")


def _cfg_reader(name: str, default):
    """Live core.config value, or `default`. NEVER raises."""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def load_params(reader: Optional[Callable[[str, Any], Any]] = None) -> dict:
    """Resolve every tunable through `reader(name, default)` (core.config by
    default), coerced to the default's type; a bad value keeps the default.
    NEVER raises."""
    rd = reader
    if rd is None:
        try:
            from core import config as _c
            rd = lambda n, d: getattr(_c, n, d)   # noqa: E731 - one import per call
        except Exception:
            rd = _cfg_reader
    out: dict = {}
    for name, default in DEFAULTS.items():
        try:
            v = rd(name, default)
            if isinstance(default, bool):
                if isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "yes", "on", "y")
                out[name] = bool(v)
            elif isinstance(default, int):
                out[name] = int(v)
            elif isinstance(default, float):
                f = float(v)
                out[name] = f if math.isfinite(f) else default
            else:
                out[name] = str(v)
        except Exception:
            out[name] = default
    return out


# ══════════════════════════════════════════════════════════════════════════
#  ONE EURO FILTER (Casiez, Roussel & Vogel, CHI 2012)
# ══════════════════════════════════════════════════════════════════════════
def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at `cutoff_hz` for step dt."""
    tau = 1.0 / (2.0 * math.pi * max(1e-6, float(cutoff_hz)))
    return 1.0 / (1.0 + tau / max(1e-6, float(dt)))


class OneEuro3:
    """One Euro filter on a 3-D point. The cutoff rises with the filtered SPEED
    (vector magnitude, so a diagonal move is treated like a straight one):
    cutoff = min_cutoff + beta * |v|. At rest it smooths hard (kills jitter);
    moving, it opens up (keeps latency low). Units: metres, seconds, Hz."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x", "_dx")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[list] = None
        self._dx = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        self._x = None
        self._dx = [0.0, 0.0, 0.0]

    @property
    def value(self) -> Optional[tuple]:
        return tuple(self._x) if self._x is not None else None

    def filter(self, p, dt: float) -> tuple:
        x = [float(p[0]), float(p[1]), float(p[2])]
        if self._x is None:
            self._x = x
            self._dx = [0.0, 0.0, 0.0]
            return tuple(x)
        dt = max(1e-3, float(dt))
        a_d = _alpha(self.d_cutoff, dt)
        for i in range(3):
            raw_d = (x[i] - self._x[i]) / dt
            self._dx[i] = a_d * raw_d + (1.0 - a_d) * self._dx[i]
        speed = math.sqrt(sum(d * d for d in self._dx))
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        for i in range(3):
            self._x[i] = a * x[i] + (1.0 - a) * self._x[i]
        return tuple(self._x)


# ══════════════════════════════════════════════════════════════════════════
#  GRIP (open / closed) - High confidence only, time-debounced, asymmetric
# ══════════════════════════════════════════════════════════════════════════
class GripFilter:
    """Per-hand stable grip ("open" | "closed"), changed only by a RUN of
    agreeing votes that spans the dwell for that direction.

    A vote is cast only when the hand joint is fully Tracked AND the SDK's
    classification is Open/Closed (Lasso per KINECT_GRIP_LASSO_AS) AND - when
    KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE - its confidence is not "low". Anything
    else is NO vote: it neither flips nor counts toward a flip. A run is broken
    by a contrary vote, or by a gap with no vote longer than the vote gap, and
    then must start again. To PRESS the run needs CLOSE_MIN_VOTES real frames
    (4: a closed reading seen for 4 frames = 133 ms) whose first-to-last span
    reaches CLOSE_SEC; to RELEASE, OPEN_MIN_VOTES frames spanning OPEN_SEC. So a
    closed flicker of 3 frames or fewer (100 ms or less) can never press a
    button, however late a frame's timestamp lands, and two reads of the same
    frame cannot count twice (the caller feeds one vote per real frame). Starts
    "open" so the first real close is a clean press edge."""

    __slots__ = ("stable", "_cand", "_first_t", "_last_t", "_votes",
                 "last_vote", "since")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.stable = "open"
        self._cand: Optional[str] = None
        self._first_t = 0.0
        self._last_t = 0.0
        self._votes = 0
        self.last_vote: Optional[str] = None
        self.since: Optional[float] = None

    @staticmethod
    def vote_for(state: str, conf: str, joint_tracked: bool,
                 params: dict) -> Optional[str]:
        """The vote ("open"/"closed") one frame casts, or None. PURE."""
        if not joint_tracked:
            return None
        s = (state or "unknown").lower()
        if s == "lasso":
            mapped = str(params.get("KINECT_GRIP_LASSO_AS", "none")).lower()
            s = mapped if mapped in ("open", "closed") else "none"
        if s not in ("open", "closed"):
            return None
        if (params.get("KINECT_GRIP_REQUIRE_HIGH_CONFIDENCE", True)
                and (conf or "").lower() == "low"):
            return None
        return s

    def update(self, vote: Optional[str], t: float, params: dict) -> str:
        self.last_vote = vote
        gap = float(params["KINECT_GRIP_VOTE_GAP_SEC"])
        if vote is None:
            # No evidence: hold. (A gap is judged when the next vote arrives.)
            return self.stable
        if vote == self.stable:
            self._cand = None
            return self.stable
        if vote == self._cand and t <= self._last_t:
            # The same frame fed twice (no time advanced): one frame, one vote.
            return self.stable
        if vote != self._cand or (t - self._last_t) > gap:
            self._cand = vote
            self._first_t = t
            self._votes = 0
        self._last_t = t
        self._votes += 1
        if vote == "closed":
            need = float(params["KINECT_GRIP_CLOSE_SEC"])
            votes = int(params["KINECT_GRIP_CLOSE_MIN_VOTES"])
        else:
            need = float(params["KINECT_GRIP_OPEN_SEC"])
            votes = int(params["KINECT_GRIP_OPEN_MIN_VOTES"])
        if (self._votes >= max(1, votes)
                and (self._last_t - self._first_t) >= need - 1e-9):
            self.stable = vote
            self.since = t
            self._cand = None
        return self.stable


# ══════════════════════════════════════════════════════════════════════════
#  ONE HAND - position, lift, raised, grip
# ══════════════════════════════════════════════════════════════════════════
def _xyz(j) -> Optional[tuple]:
    try:
        x, y, z = float(j[0]), float(j[1]), float(j[2])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    if x == 0.0 and y == 0.0 and z == 0.0:
        return None
    return (x, y, z)


def _state(j) -> int:
    try:
        return int(j[3])
    except (TypeError, ValueError, IndexError):
        return 0


def _dist(a, b) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


class HandTrack:
    """The stabilised state of ONE hand (SDK side) of the owner body."""

    def __init__(self, side: str, params: dict):
        self.side = side
        self._pos_f = OneEuro3(params["KINECT_HAND_FILTER_MIN_CUTOFF_HZ"],
                               params["KINECT_HAND_FILTER_BETA"],
                               params["KINECT_HAND_FILTER_D_CUTOFF_HZ"])
        self.grip = GripFilter()
        self.reset()

    def reset(self) -> None:
        self._pos_f.reset()
        self.grip.reset()
        self.pos: Optional[tuple] = None
        self.source: Optional[str] = None     # hand | wrist | inferred | hold
        self.measured = False                 # a real position this frame
        self.lift: Optional[float] = None
        self.lift_measured = False            # lift measured (Tracked) this frame
        self.raised = False
        self.joint_tracked = False            # hand joint fully Tracked this frame
        self.state = "unknown"                # raw SDK hand state this frame
        self.conf = "unknown"                 # raw SDK confidence this frame
        self.ext: dict = {}
        self._last_t: Optional[float] = None
        self._last_meas_t: Optional[float] = None
        self._last_lift_t: Optional[float] = None
        self._last_raw: Optional[tuple] = None
        self._jumps = 0
        self._offset: Optional[tuple] = None
        self._offset_t: Optional[float] = None
        self._raise_cand_since: Optional[float] = None
        self._raw_above = False
        self._shoulder_y: Optional[float] = None

    def _sync_filter_params(self, params: dict) -> None:
        self._pos_f.min_cutoff = float(params["KINECT_HAND_FILTER_MIN_CUTOFF_HZ"])
        self._pos_f.beta = float(params["KINECT_HAND_FILTER_BETA"])
        self._pos_f.d_cutoff = float(params["KINECT_HAND_FILTER_D_CUTOFF_HZ"])

    def lost_for(self, t: float) -> Optional[float]:
        """Seconds since the last real position measurement (None = never)."""
        if self._last_meas_t is None:
            return None
        return max(0.0, t - self._last_meas_t)

    def update(self, joints: Optional[dict], state: str, conf: str, t: float,
               params: dict, reliable: Callable[[Any], bool],
               ext: Optional[dict] = None) -> None:
        self._sync_filter_params(params)
        dt = FRAME_PERIOD_S if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        grace = float(params["KINECT_HAND_LOSS_GRACE_SEC"])
        joints = joints or {}
        if ext:
            self.ext = dict(ext)
        side = self.side
        hand = joints.get(f"hand_{side}")
        wrist = joints.get(f"wrist_{side}")
        hand_ok = bool(reliable(hand))
        wrist_ok = bool(reliable(wrist))
        self.joint_tracked = hand_ok
        self.state = (state or "unknown").lower()
        self.conf = (conf or "unknown").lower()

        # Learn the hand-wrist offset whenever BOTH are Tracked - SMOOTHED, so one
        # bad "Tracked" hand frame between Inferred ones can't throw the hand
        # position by centimetres (measured: a single such frame moved the old
        # wrist stand-in 44 mm RMS while the wrist itself sat still).
        if hand_ok and wrist_ok:
            h, w = _xyz(hand), _xyz(wrist)
            if h is not None and w is not None:
                raw_off = (h[0] - w[0], h[1] - w[1], h[2] - w[2])
                fresh = (self._offset is None or self._offset_t is None
                         or (t - self._offset_t)
                         > float(params["KINECT_WRIST_OFFSET_MAX_AGE_SEC"]))
                if fresh:
                    self._offset = raw_off
                else:
                    a = _alpha(float(params["KINECT_HAND_OFFSET_CUTOFF_HZ"]), dt)
                    self._offset = tuple(a * raw_off[k] + (1.0 - a) * self._offset[k]
                                         for k in range(3))
                self._offset_t = t

        # ── measurement. The WRIST is the steadier joint (larger, constrained by
        # the forearm; Tracked 94% of frames when the hand joint was 35%), and a
        # closing hand moves the hand joint but not the wrist - so a Tracked wrist
        # plus the smoothed hand-wrist offset is preferred; then the Tracked hand;
        # then the raw Tracked wrist (no offset learned yet); then the Inferred
        # hand (a guess - it never measures lift).
        meas, src = None, None
        w = _xyz(wrist) if wrist_ok else None
        off_ok = (self._offset is not None and self._offset_t is not None
                  and (t - self._offset_t)
                  <= float(params["KINECT_WRIST_OFFSET_MAX_AGE_SEC"]))
        if w is not None and off_ok:
            meas = (w[0] + self._offset[0], w[1] + self._offset[1],
                    w[2] + self._offset[2])
            src = "wrist"
        elif hand_ok:
            meas, src = _xyz(hand), "hand"
        elif w is not None:
            meas, src = w, "wrist"
        elif hand is not None and _state(hand) >= 1:
            meas, src = _xyz(hand), "inferred"

        # A fresh acquisition (nothing measured within the grace) re-seeds.
        if meas is not None and (self._last_meas_t is None
                                 or (t - self._last_meas_t) > grace):
            self._pos_f.reset()
            self._last_raw = None
            self._jumps = 0

        # ── jump rejection: one or two impossible frames are dropped.
        if meas is not None and self._last_raw is not None:
            frames = max(1.0, dt / FRAME_PERIOD_S)
            if _dist(meas, self._last_raw) > float(
                    params["KINECT_HAND_JUMP_REJECT_M"]) * frames:
                if self._jumps < int(params["KINECT_HAND_JUMP_REJECT_FRAMES"]):
                    self._jumps += 1
                    meas, src = None, None
                else:
                    # It stayed there: it is real. Re-seed at the new place.
                    self._pos_f.reset()
                    self._jumps = 0
            else:
                self._jumps = 0

        if meas is not None:
            self.pos = self._pos_f.filter(meas, dt)
            self.source = src
            self.measured = True
            self._last_meas_t = t
            self._last_raw = meas
        else:
            self.measured = False
            if self._last_meas_t is not None and (t - self._last_meas_t) <= grace:
                self.source = "hold"          # keep the last good value
            else:
                self.pos = None
                self.source = None
                self._pos_f.reset()
                self._last_raw = None

        # ── lift: hand height above the shoulder line, from a TRACKED source.
        shoulder_ref = joints.get("spine_shoulder")
        if not reliable(shoulder_ref):
            shoulder_ref = joints.get(f"shoulder_{side}")
            if not reliable(shoulder_ref):
                shoulder_ref = None
        if shoulder_ref is not None:
            sy = float(shoulder_ref[1])
            # The shoulder line barely moves (0.6 mm jitter measured); a light
            # EMA only smooths the spine-shoulder <-> side-shoulder hand-off.
            a = _alpha(2.0, dt)
            self._shoulder_y = (sy if self._shoulder_y is None
                                else a * sy + (1.0 - a) * self._shoulder_y)
        if (src in ("hand", "wrist") and self.pos is not None
                and self._shoulder_y is not None and shoulder_ref is not None):
            self.lift = self.pos[1] - self._shoulder_y
            # The UNFILTERED lift of this frame decides the raise ENTRY below: a
            # two-frame twitch over the line must not count as a raise just
            # because the filter's decay keeps the smoothed value up longer.
            self._raw_above = (meas[1] - self._shoulder_y) >= float(
                params["KINECT_LIFT_UP_MARGIN"])
            self.lift_measured = True
            self._last_lift_t = t
        else:
            self.lift_measured = False
            if not (self._last_lift_t is not None
                    and (t - self._last_lift_t) <= grace
                    and self.pos is not None):
                self.lift = None

        # ── raised: enter dwell above UP (raw), exit dwell below DOWN
        # (filtered); a lost hand is down.
        down = float(params["KINECT_LIFT_DOWN_MARGIN"])
        if self.lift is None:
            self.raised = False
            self._raise_cand_since = None
        elif not self.raised:
            # ENTRY: the measured (raw) lift must stay over UP for the dwell; a
            # held (unmeasured) frame carries the last frame's verdict.
            if self._raw_above:
                if self._raise_cand_since is None:
                    self._raise_cand_since = t
                if (t - self._raise_cand_since) >= float(
                        params["KINECT_RAISE_ENTER_SEC"]) - 1e-9:
                    self.raised = True
                    self._raise_cand_since = None
            else:
                self._raise_cand_since = None
        else:
            if self.lift < down:
                if self._raise_cand_since is None:
                    self._raise_cand_since = t
                if (t - self._raise_cand_since) >= float(
                        params["KINECT_RAISE_EXIT_SEC"]) - 1e-9:
                    self.raised = False
                    self._raise_cand_since = None
            else:
                self._raise_cand_since = None

        # ── grip: one vote per real frame, High confidence only.
        if self.pos is None:
            # Hand gone past the grace: a stale grip must not survive into the
            # next acquisition (it would press on re-engage).
            self.grip.reset()
        else:
            self.grip.update(GripFilter.vote_for(self.state, self.conf,
                                                 hand_ok, params), t, params)

    def view(self, t: float) -> dict:
        """The public, read-only per-hand dict published in the snapshot."""
        return {
            "side": self.side,
            "pos": self.pos,
            "source": self.source,
            "measured": self.measured,
            "lost_s": self.lost_for(t),
            "lift": self.lift,
            "lift_measured": self.lift_measured,
            "raised": self.raised,
            "grip": self.grip.stable,
            "grip_vote": self.grip.last_vote,
            "state": self.state,
            "conf": self.conf,
            "joint_tracked": self.joint_tracked,
            "ext": dict(self.ext),
        }


# ══════════════════════════════════════════════════════════════════════════
#  BODY-LEVEL GATES - active hand, two-hand mode, owner body
# ══════════════════════════════════════════════════════════════════════════
class ActiveHand:
    """Sticky cursor hand. The holder keeps it while it stays raised; the other
    hand wins only when it is raised AND leads the holder's lift by the switch
    lead continuously for the switch dwell - or when the holder is lowered."""

    __slots__ = ("side", "_chall_since")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.side: Optional[str] = None
        self._chall_since: Optional[float] = None

    def update(self, hands: dict, t: float, params: dict) -> Optional[str]:
        raised = [s for s in _SIDES if hands[s].raised and hands[s].lift is not None]
        if self.side not in raised:
            self._chall_since = None
            if not raised:
                self.side = None
            else:
                self.side = max(raised, key=lambda s: hands[s].lift)
            return self.side
        other = "left" if self.side == "right" else "right"
        lead = float(params["KINECT_ACTIVE_HAND_SWITCH_LEAD_M"])
        if (other in raised
                and hands[other].lift - hands[self.side].lift >= lead):
            if self._chall_since is None:
                self._chall_since = t
            if (t - self._chall_since) >= float(
                    params["KINECT_ACTIVE_HAND_SWITCH_SEC"]) - 1e-9:
                self.side = other
                self._chall_since = None
        else:
            self._chall_since = None
        return self.side


class TwoHandGate:
    """Two-hand mode with an enter dwell, an exit dwell and a re-arm window.
    Engage edges are therefore at least EXIT + REARM + ENTER apart (1.05 s at
    the defaults): it cannot flap within a second, whatever one frame says."""

    __slots__ = ("active", "_both_since", "_not_both_since", "_last_exit_t")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.active = False
        self._both_since: Optional[float] = None
        self._not_both_since: Optional[float] = None
        self._last_exit_t: Optional[float] = None

    def update(self, both_raised: bool, t: float, params: dict) -> bool:
        if not self.active:
            rearmed = (self._last_exit_t is None
                       or (t - self._last_exit_t) >= float(
                           params["KINECT_TWO_HAND_REARM_SEC"]) - 1e-9)
            if both_raised and rearmed:
                if self._both_since is None:
                    self._both_since = t
                if (t - self._both_since) >= float(
                        params["KINECT_TWO_HAND_ENTER_SEC"]) - 1e-9:
                    self.active = True
                    self._both_since = None
                    self._not_both_since = None
            else:
                self._both_since = None
        else:
            if not both_raised:
                if self._not_both_since is None:
                    self._not_both_since = t
                if (t - self._not_both_since) >= float(
                        params["KINECT_TWO_HAND_EXIT_SEC"]) - 1e-9:
                    self.active = False
                    self._last_exit_t = t
                    self._not_both_since = None
            else:
                self._not_both_since = None
        return self.active


def _distance(body: Optional[dict]) -> float:
    try:
        d = body.get("distance_m")
        if isinstance(d, (int, float)) and d > 0 and math.isfinite(d):
            return float(d)
    except Exception:
        pass
    return float("inf")


def _nearest(bodies: list) -> Optional[dict]:
    cands = [b for b in bodies if isinstance(b, dict)]
    if not cands:
        return None
    return min(cands, key=_distance)


# ══════════════════════════════════════════════════════════════════════════
#  THE STABILISER (one per bridge; process() once per new body frame)
# ══════════════════════════════════════════════════════════════════════════
class HandStabilizer:
    """Owner body + both hands + the body-level gates, advanced once per frame.

    arm_extension_fn(joints, side) -> dict  (the bridge's canonical reach /
        straightness geometry, carried through for the air-mouse reach cue);
    joint_reliable_fn(joint) -> bool         (the bridge's Tracked-joint rule);
    params_fn() -> dict                      (load_params by default).
    Each is injected so the module stays pure and every rule has ONE home."""

    def __init__(self, *, arm_extension_fn: Optional[Callable] = None,
                 joint_reliable_fn: Optional[Callable] = None,
                 params_fn: Optional[Callable[[], dict]] = None):
        self._ext_fn = arm_extension_fn
        self._reliable = joint_reliable_fn or _default_reliable
        self._params_fn = params_fn or load_params
        self.params = self._params_fn()
        self.hands = {s: HandTrack(s, self.params) for s in _SIDES}
        self.active = ActiveHand()
        self.two_hand = TwoHandGate()
        self._owner_id = None
        self._owner_body: Optional[dict] = None
        self._owner_seen_t: Optional[float] = None
        self._chall_id = None
        self._chall_since: Optional[float] = None
        self.snapshot: Optional[dict] = None

    def _reset_owner_state(self) -> None:
        for h in self.hands.values():
            h.reset()
        self.active.reset()
        self.two_hand.reset()
        self._chall_id = None
        self._chall_since = None

    def _pick_owner(self, bodies: list, t: float) -> "tuple[Optional[dict], bool]":
        """(owner body, fresh-this-frame). Sticky by tracking id."""
        p = self.params
        by_id = {}
        for b in bodies:
            if isinstance(b, dict):
                by_id.setdefault(b.get("id"), b)
        nearest = _nearest(bodies)
        if self._owner_id is not None and self._owner_id in by_id:
            ob = by_id[self._owner_id]
            self._owner_seen_t = t
            self._owner_body = ob
            if (nearest is not None and nearest.get("id") != self._owner_id
                    and _distance(nearest) + float(
                        p["KINECT_OWNER_SWITCH_NEARER_M"]) <= _distance(ob)):
                if self._chall_id != nearest.get("id"):
                    self._chall_id = nearest.get("id")
                    self._chall_since = t
                elif (t - (self._chall_since or t)) >= float(
                        p["KINECT_OWNER_SWITCH_SEC"]) - 1e-9:
                    self._owner_id = nearest.get("id")
                    self._owner_body = nearest
                    self._reset_owner_state()
                    return nearest, True
            else:
                self._chall_id = None
                self._chall_since = None
            return ob, True
        # The owner is not in this frame: hold it through the loss grace.
        if (self._owner_id is not None and self._owner_seen_t is not None
                and (t - self._owner_seen_t) <= float(
                    p["KINECT_OWNER_LOSS_GRACE_SEC"])):
            return self._owner_body, False
        if nearest is None:
            if self._owner_id is not None:
                self._owner_id = None
                self._owner_body = None
                self._reset_owner_state()
            return None, False
        if nearest.get("id") != self._owner_id:
            self._owner_id = nearest.get("id")
            self._reset_owner_state()
        self._owner_body = nearest
        self._owner_seen_t = t
        return nearest, True

    def process(self, bodies, t: float, seq: int = 0) -> dict:
        """Advance one REAL body frame (bodies = the bridge's parsed list) at
        frame time `t` (monotonic seconds). Returns the published snapshot.
        NEVER raises: any internal failure yields an untracked snapshot."""
        try:
            return self._process(list(bodies or []), float(t), int(seq))
        except Exception:
            snap = {"seq": int(seq or 0), "t": float(t or 0.0), "owner_id": None,
                    "owner": None, "fresh": False, "tracked": False,
                    "active": None, "two_hand": False,
                    "hands": {s: HandTrack(s, DEFAULTS).view(0.0) for s in _SIDES}}
            self.snapshot = snap
            return snap

    def _process(self, bodies: list, t: float, seq: int) -> dict:
        try:
            self.params = self._params_fn()
        except Exception:
            pass
        p = self.params
        owner, fresh = self._pick_owner(bodies, t)
        joints = (owner.get("joints") or {}) if (owner is not None and fresh) else {}
        for side in _SIDES:
            h = self.hands[side]
            ext = None
            if fresh and owner is not None and self._ext_fn is not None:
                try:
                    ext = self._ext_fn(joints, side)
                except Exception:
                    ext = None
            state = owner.get(f"hand_{side}", "unknown") if fresh else "unknown"
            conf = owner.get(f"hand_{side}_conf", "unknown") if fresh else "unknown"
            h.update(joints, state, conf, t, p, self._reliable, ext)
        if owner is None:
            active, two = None, False
            self.active.reset()
            self.two_hand.reset()
        else:
            active = self.active.update(self.hands, t, p)
            two = self.two_hand.update(
                self.hands["left"].raised and self.hands["right"].raised, t, p)
        snap = {
            "seq": seq,
            "t": t,
            "owner_id": self._owner_id if owner is not None else None,
            "owner": owner,
            "fresh": bool(fresh),
            "tracked": owner is not None,
            "active": active,
            "two_hand": bool(two),
            "hands": {s: self.hands[s].view(t) for s in _SIDES},
        }
        self.snapshot = snap
        return snap


def _default_reliable(j) -> bool:
    """Fallback Tracked-joint rule (state >= 2, finite, not the zero fill), used
    only when the caller injects none. The bridge injects its own
    _joint_reliable so the live rule has one home."""
    return j is not None and _state(j) >= 2 and _xyz(j) is not None
