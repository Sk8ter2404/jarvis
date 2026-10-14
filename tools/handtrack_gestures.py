"""Hand-gesture recognition for the JARVIS hand tracker (pure Python).

No mediapipe, no numpy, no I/O: this module turns per-frame hand landmarks
into (a) a per-hand state and (b) discrete gesture EVENTS, so every threshold
can be unit-tested with synthetic landmarks (tests/test_handtrack_gestures.py)
and the page, the desktop trigger and any future air mouse all agree on what a
gesture is. tools/hand_tracker_service.py feeds it and serves the result.

INPUT (per frame)
    engine.update(t, hands, w, h)
      t      seconds (any monotonic clock)
      hands  list of {"side": "Left"|"Right", "score": 0-1, "lm": [[x, y, z]] * 21}
             lm are MediaPipe normalised image coordinates of the MIRRORED
             frame (the service flips the camera frame before detection, so
             "Left"/"Right" are the owner's real hands and x runs like a mirror).
      w, h   frame size in pixels (distances are measured in pixels so a 4:3
             frame does not skew them).
    -> (infos, events)
      infos   one dict per input hand: {"id", "side", "score", "state", "pinch",
              "palm": [x, y], "palm_facing"} (the service adds lm / world).
      events  [{"type": "gesture", "name", "side", "pos": [x, y], "value"?}]

UNITS
    "hs" = one hand size = max(wrist->middle-MCP, 1.2 x index-MCP->pinky-MCP)
    in pixels. Speeds are hs/s, so a gesture means the same near and far.

HAND STATES (with hysteresis; a state change must also persist STATE_DEBOUNCE_S)
    Per finger the ratio r = |wrist->tip| / |wrist->PIP|:
        extended   set when r > EXT_ON (1.25), cleared when r < EXT_OFF (1.10)
        curled     set when r < CURL_ON (0.98), cleared when r > CURL_OFF (1.15)
    pinch strength = clamp((PINCH_OPEN_D - d) / (PINCH_OPEN_D - PINCH_CLOSED_D))
        with d = |thumb tip -> index tip| / hs, PINCH_CLOSED_D 0.18, PINCH_OPEN_D 0.60
        pinch state: strength >= 0.75 on, < 0.50 off.
    state (first match): pinch, point (index extended, middle/ring/pinky curled),
        fist (all four curled), open (all four extended), else unknown.
    spread = fan angle between the index and pinky directions (MCP->tip),
        spread_ok set at >= 20 deg, cleared below 14 deg.
    palm_facing: the sign of cross(wrist->index MCP, wrist->pinky MCP) says
        whether the palm or the back of the hand faces the (mirrored) camera:
        for a mirrored "Right" hand the palm faces the camera when the cross
        product is positive (index MCP left of the pinky MCP with fingers up),
        for "Left" when it is negative. Needs |cross| >= 0.20 hs^2 (not edge-on).

EVENTS  (each fires once, then needs the gesture to end before it can re-arm)
    palm_hold    open palm facing the camera, fingers spread, held PALM_HOLD_S
                 (1.5 s) and still: over the last 0.6 s the palm centre spans
                 <= 0.5 hs and the hand axis turns <= 20 deg; since the start
                 of the hold the centre spans <= 1.2 hs. Any failure restarts
                 the clock (a wave never qualifies). Re-arms after 0.5 s
                 without the posture. value = seconds held.
    pinch_start / pinch_end   state transitions, pos = the pinch point.
    pinch_hold   pinch held PINCH_HOLD_S (0.6 s) with the pinch point spanning
                 <= 1.0 hs. value = seconds held.
    fist_grab / fist_release   state becomes / stops being fist (after the
                 debounce); also fires on release if the hand disappears.
    point_flick  index pointing and the wrist->index-tip direction sweeps
                 >= 30 deg within 0.18 s at >= 220 deg/s. value = signed
                 angular speed in deg/s (positive = clockwise on screen, i.e.
                 an upward finger swinging to the right). Cooldown 0.7 s.
                 At a low frame rate (the owner's webcam gives ~12-15 fps in
                 room light) the window widens to 2.5 frame intervals (at most
                 0.35 s) and 2 samples are enough; the 30 deg / 220 deg/s bar
                 is unchanged, so jitter still cannot fire it.
    explode      (a) two fists, apart-speed >= 4 hs/s and >= 1.2 hs wider in
                 0.4 s; or (b) two hands held together (<= 1.6 hs apart for
                 >= 0.3 s) then parting at >= 4 hs/s within 0.45 s. Never
                 within 0.8 s of a clap. Cooldown 1.2 s. value = hs/s.
    clap         two hands (neither a fist) closing at >= 4 hs/s from >= 1.0 hs
                 apart to <= 1.3 hs; or closing at >= 3.5 hs/s and then both
                 disappearing / one disappearing within 0.2 s from <= 1.9 hs
                 (palms edge-on often hide a hand on contact). Cooldown
                 0.8 s. value = hs/s.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

# ── landmark indices (MediaPipe hand) ──────────────────────────────────
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGERS = (  # (mcp, pip, tip)
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)
PALM_POINTS = (0, 5, 9, 13, 17)

# ── thresholds (documented in the module docstring) ────────────────────
EXT_ON, EXT_OFF = 1.25, 1.10
CURL_ON, CURL_OFF = 0.98, 1.15
PINCH_CLOSED_D, PINCH_OPEN_D = 0.18, 0.60
PINCH_ON, PINCH_OFF = 0.75, 0.50
FIST_PINCH_GUARD_D = 0.22
SPREAD_ON_DEG, SPREAD_OFF_DEG = 20.0, 14.0
FACING_MIN_CROSS = 0.20
STATE_DEBOUNCE_S = 0.10
TRACK_DROP_S = 0.25

PALM_HOLD_S = 1.5
PALM_STILL_WINDOW_S = 0.6
PALM_STILL_SPAN_HS = 0.5
PALM_STILL_TOTAL_HS = 1.2
PALM_STILL_AXIS_DEG = 20.0
PALM_GRACE_S = 0.2
PALM_REARM_S = 0.5

PINCH_HOLD_S = 0.6
PINCH_HOLD_SPAN_HS = 1.0

FLICK_WINDOW_S = 0.18
FLICK_SWEEP_DEG = 30.0
FLICK_OMEGA_DEG_S = 220.0
FLICK_COOLDOWN_S = 0.7
# Low frame rates (live 2026-10-14: 12 fps from the owner's webcam in room
# light, with ~0.1 s gaps while JARVIS's face pass runs). A 0.18 s window then
# holds 2 samples at best and "3 samples" never fires.
FLICK_WINDOW_FRAMES = 2.5        # window = max(FLICK_WINDOW_S, this x frame interval)
FLICK_WINDOW_MAX_S = 0.35
FLICK_SLOW_FRAME_S = 0.05        # slower than 20 fps -> 2 samples suffice

PAIR_WINDOW_S = 0.4
EXPLODE_RATE_HS_S = 4.0
EXPLODE_GAIN_HS = 1.2
EXPLODE_TOGETHER_HS = 1.6
EXPLODE_TOGETHER_S = 0.3
EXPLODE_PART_WINDOW_S = 0.45
EXPLODE_COOLDOWN_S = 1.2
EXPLODE_AFTER_CLAP_S = 0.8
CLAP_WINDOW_S = 0.25
CLAP_RATE_HS_S = 4.0
CLAP_FROM_HS = 1.0
CLAP_MEET_HS = 1.3
CLAP_LOST_RATE_HS_S = 3.5
CLAP_LOST_NEAR_HS = 1.9
CLAP_LOST_WITHIN_S = 0.2
CLAP_COOLDOWN_S = 0.8


# ── small geometry helpers ─────────────────────────────────────────────
def _px(lm, i, w, h) -> Tuple[float, float]:
    p = lm[i]
    return p[0] * w, p[1] * h


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _span(points) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


class _Track:
    """One tracked hand (id 0 or 1) and the little state its events need."""

    def __init__(self, tid: int, side: str, t: float):
        self.id = tid
        self.side = side
        self.first_t = t
        self.last_t = t
        self.hs = 0.0
        self.palm = (0.0, 0.0)           # px
        self.ext = [False] * 4
        self.curled = [False] * 4
        self.spread_ok = False
        self.pinch_on = False
        self.state = "unknown"
        self.cand = "unknown"
        self.cand_since = t
        self.hist: deque = deque()       # (t, cx, cy, axis_deg) in px / deg
        # palm_hold
        self.hold_start: Optional[float] = None
        self.hold_bad_since: Optional[float] = None
        self.hold_fired = False
        self.hold_clear_since: Optional[float] = None
        # pinch
        self.pinch_since: Optional[float] = None
        self.pinch_hist: deque = deque()  # (t, x, y) px of the pinch point
        self.pinch_hold_fired = False
        # fist
        self.grabbing = False
        self.grab_pos = (0.0, 0.0)
        # flick
        self.flick_hist: deque = deque()  # (t, angle_deg)
        self.flick_until = 0.0
        self.dt_ema = 0.0                 # smoothed frame interval (s)
        self.last_info: dict = {}


class GestureEngine:
    """Feed it every frame; it returns hand infos and gesture events."""

    def __init__(self, invert_palm: bool = False):
        self.invert_palm = invert_palm
        self._tracks: List[_Track] = []
        self._pair: deque = deque()      # (t, d_hs, mx, my, stateA, stateB, nofist)
        self._clap_until = 0.0
        self._explode_until = 0.0
        self._clap_block_explode_until = 0.0
        self._pair_w = 1
        self._pair_h = 1

    # ── public ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._tracks.clear()
        self._pair.clear()

    def update(self, t: float, hands: List[dict], w: int, h: int
               ) -> Tuple[List[dict], List[dict]]:
        events: List[dict] = []
        self._pair_w, self._pair_h = max(1, w), max(1, h)
        analyses = []
        for hd in hands:
            try:
                analyses.append(self._analyze(hd, w, h))
            except Exception:
                analyses.append(None)
        matched = self._match(t, hands, analyses)
        infos: List[dict] = []
        seen = set()
        for hd, a, tr in zip(hands, analyses, matched):
            if a is None or tr is None:
                infos.append({"id": -1, "side": hd.get("side", ""),
                              "score": float(hd.get("score", 0.0) or 0.0),
                              "state": "unknown", "pinch": 0.0,
                              "palm": [0.0, 0.0], "palm_facing": False})
                continue
            seen.add(tr.id)
            self._step_track(tr, t, a, hd, w, h, events)
            infos.append(tr.last_info)
        # tracks that did not appear this frame
        for tr in list(self._tracks):
            if tr.id in seen:
                continue
            if t - tr.last_t > TRACK_DROP_S:
                self._drop_track(tr, t, w, h, events)
        self._step_pair(t, w, h, events)
        return infos, events

    # ── analysis of one hand ────────────────────────────────────────────
    def _analyze(self, hd: dict, w: int, h: int) -> dict:
        lm = hd["lm"]
        pts = [_px(lm, i, w, h) for i in range(21)]
        wrist = pts[WRIST]
        hs = max(_dist(wrist, pts[MIDDLE_MCP]),
                 1.2 * _dist(pts[INDEX_MCP], pts[PINKY_MCP]), 1e-6)
        ratios = []
        for mcp, pip, tip in FINGERS:
            dp = _dist(wrist, pts[pip])
            ratios.append(_dist(wrist, pts[tip]) / dp if dp > 1e-6 else 1.0)
        pd = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / hs
        strength = _clamp01((PINCH_OPEN_D - pd) / (PINCH_OPEN_D - PINCH_CLOSED_D))
        # spread: fan angle index <-> pinky (MCP -> tip directions)
        ai = math.degrees(math.atan2(pts[INDEX_TIP][1] - pts[INDEX_MCP][1],
                                     pts[INDEX_TIP][0] - pts[INDEX_MCP][0]))
        ap = math.degrees(math.atan2(pts[PINKY_TIP][1] - pts[PINKY_MCP][1],
                                     pts[PINKY_TIP][0] - pts[PINKY_MCP][0]))
        spread = abs(_wrap_deg(ai - ap))
        v1 = (pts[INDEX_MCP][0] - wrist[0], pts[INDEX_MCP][1] - wrist[1])
        v2 = (pts[PINKY_MCP][0] - wrist[0], pts[PINKY_MCP][1] - wrist[1])
        cross = (v1[0] * v2[1] - v1[1] * v2[0]) / (hs * hs)
        side = hd.get("side", "")
        facing = False
        if abs(cross) >= FACING_MIN_CROSS and side in ("Left", "Right"):
            facing = (cross > 0) == (side == "Right")
            if self.invert_palm:
                facing = not facing
        axis = math.degrees(math.atan2(pts[MIDDLE_MCP][1] - wrist[1],
                                       pts[MIDDLE_MCP][0] - wrist[0]))
        cx = sum(pts[i][0] for i in PALM_POINTS) / len(PALM_POINTS)
        cy = sum(pts[i][1] for i in PALM_POINTS) / len(PALM_POINTS)
        tip_ang = math.degrees(math.atan2(pts[INDEX_TIP][1] - wrist[1],
                                          pts[INDEX_TIP][0] - wrist[0]))
        pinch_pt = ((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) / 2.0,
                    (pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) / 2.0)
        return {"hs": hs, "ratios": ratios, "pd": pd, "strength": strength,
                "spread": spread, "facing": facing, "axis": axis,
                "palm": (cx, cy), "tip_ang": tip_ang, "pinch_pt": pinch_pt}

    # ── track assignment ────────────────────────────────────────────────
    def _match(self, t: float, hands: List[dict], analyses: list) -> list:
        out: list = [None] * len(hands)
        cands = []
        for hi, a in enumerate(analyses):
            if a is None:
                continue
            for tr in self._tracks:
                if t - tr.last_t > TRACK_DROP_S:
                    continue
                scale = max(tr.hs, a["hs"], 1e-6)
                cost = _dist(a["palm"], tr.palm) / scale
                if tr.side != hands[hi].get("side"):
                    cost += 1.5
                cands.append((cost, hi, tr))
        cands.sort(key=lambda c: c[0])
        used_tracks = set()
        for cost, hi, tr in cands:
            if cost > 4.0 or out[hi] is not None or tr.id in used_tracks:
                continue
            out[hi] = tr
            used_tracks.add(tr.id)
        for hi, a in enumerate(analyses):
            if a is None or out[hi] is not None:
                continue
            taken = {tr.id for tr in self._tracks if t - tr.last_t <= TRACK_DROP_S}
            taken |= {tr.id for tr in out if tr is not None}
            free = [i for i in (0, 1) if i not in taken]
            if not free:
                continue
            tr = _Track(free[0], hands[hi].get("side", ""), t)
            # a stale track holding the id is replaced
            self._tracks = [x for x in self._tracks if x.id != tr.id]
            self._tracks.append(tr)
            tr.hs, tr.palm = a["hs"], a["palm"]
            out[hi] = tr
        return out

    def _drop_track(self, tr: _Track, t: float, w: int, h: int, events: list):
        if tr.grabbing:
            events.append(self._ev("fist_release", tr.side, tr.grab_pos, w, h))
            tr.grabbing = False
        if tr.pinch_on:
            events.append(self._ev("pinch_end", tr.side, tr.grab_pos, w, h))
            tr.pinch_on = False
        self._tracks = [x for x in self._tracks if x.id != tr.id]

    # ── one track, one frame ────────────────────────────────────────────
    def _step_track(self, tr: _Track, t: float, a: dict, hd: dict,
                    w: int, h: int, events: list) -> None:
        tr.side = hd.get("side", tr.side)
        if t > tr.last_t:
            d = min(t - tr.last_t, 0.5)
            tr.dt_ema = d if tr.dt_ema <= 0 else 0.8 * tr.dt_ema + 0.2 * d
        tr.last_t = t
        tr.hs = a["hs"] if tr.hs <= 0 else 0.7 * tr.hs + 0.3 * a["hs"]
        tr.palm = a["palm"]
        # finger flags with hysteresis
        for i, r in enumerate(a["ratios"]):
            if r > EXT_ON:
                tr.ext[i] = True
            elif r < EXT_OFF:
                tr.ext[i] = False
            if r < CURL_ON:
                tr.curled[i] = True
            elif r > CURL_OFF:
                tr.curled[i] = False
        if a["spread"] >= SPREAD_ON_DEG:
            tr.spread_ok = True
        elif a["spread"] < SPREAD_OFF_DEG:
            tr.spread_ok = False
        # pinch strength with hysteresis
        s = a["strength"]
        fist_like = all(tr.curled)
        if tr.pinch_on:
            pinch_now = s >= PINCH_OFF
        else:
            pinch_now = s >= PINCH_ON
        if fist_like and a["pd"] > FIST_PINCH_GUARD_D:
            pinch_now = False
        point_flags = bool(tr.ext[0] and tr.curled[1] and tr.curled[2]
                           and tr.curled[3])
        raw = ("pinch" if pinch_now else
               "point" if point_flags else
               "fist" if fist_like else
               "open" if all(tr.ext) else "unknown")
        # debounce the reported state
        if raw != tr.cand:
            tr.cand, tr.cand_since = raw, t
        if raw != tr.state and (t - tr.cand_since) >= STATE_DEBOUNCE_S:
            tr.state = raw
        elif tr.first_t == t:
            tr.state = raw                      # first sighting reports at once
        palm_n = [a["palm"][0] / w, a["palm"][1] / h]
        pinch_n = [a["pinch_pt"][0] / w, a["pinch_pt"][1] / h]
        tr.hist.append((t, a["palm"][0], a["palm"][1], a["axis"]))
        while tr.hist and t - tr.hist[0][0] > 2.5:
            tr.hist.popleft()
        shown = round(s, 3)
        tr.last_info = {"id": tr.id, "side": tr.side,
                        "score": float(hd.get("score", 0.0) or 0.0),
                        "state": tr.state, "pinch": shown,
                        "palm": [round(palm_n[0], 4), round(palm_n[1], 4)],
                        "palm_facing": bool(a["facing"])}
        tr.grab_pos = tuple(palm_n)

        # ---- pinch start / end / hold
        pinching = tr.state == "pinch"
        if pinching and not tr.pinch_on:
            tr.pinch_on = True
            tr.pinch_since = t
            tr.pinch_hist.clear()
            tr.pinch_hold_fired = False
            events.append(self._ev("pinch_start", tr.side, pinch_n, w, h))
        elif not pinching and tr.pinch_on:
            tr.pinch_on = False
            tr.pinch_since = None
            events.append(self._ev("pinch_end", tr.side, pinch_n, w, h))
        if tr.pinch_on:
            tr.pinch_hist.append((t, a["pinch_pt"][0], a["pinch_pt"][1]))
            held = t - (tr.pinch_since or t)
            if (not tr.pinch_hold_fired and held >= PINCH_HOLD_S
                    and _span([(x, y) for _, x, y in tr.pinch_hist])
                    <= PINCH_HOLD_SPAN_HS * tr.hs):
                tr.pinch_hold_fired = True
                events.append(self._ev("pinch_hold", tr.side, pinch_n, w, h,
                                       round(held, 2)))

        # ---- fist grab / release
        if tr.state == "fist" and not tr.grabbing:
            tr.grabbing = True
            events.append(self._ev("fist_grab", tr.side, palm_n, w, h))
        elif tr.state != "fist" and tr.grabbing:
            tr.grabbing = False
            events.append(self._ev("fist_release", tr.side, palm_n, w, h))

        # ---- palm hold
        self._step_palm_hold(tr, t, a, palm_n, w, h, events)

        # ---- point flick
        self._step_flick(tr, t, a, palm_n, w, h, events,
                         point_flags and not pinch_now)

    def _step_palm_hold(self, tr, t, a, palm_n, w, h, events):
        base = tr.state == "open" and a["facing"] and tr.spread_ok
        if not base:
            if tr.hold_bad_since is None:
                tr.hold_bad_since = t
            if t - tr.hold_bad_since > PALM_GRACE_S:
                tr.hold_start = None
            if tr.hold_clear_since is None:
                tr.hold_clear_since = t
            if t - tr.hold_clear_since >= PALM_REARM_S:
                tr.hold_fired = False
            return
        tr.hold_bad_since = None
        tr.hold_clear_since = None
        if tr.hold_start is None:
            tr.hold_start = t
        # stillness
        recent = [(x, y) for (tt, x, y, _ax) in tr.hist
                  if t - tt <= PALM_STILL_WINDOW_S]
        total = [(x, y) for (tt, x, y, _ax) in tr.hist if tt >= tr.hold_start]
        axes = [ax for (tt, _x, _y, ax) in tr.hist if t - tt <= PALM_STILL_WINDOW_S]
        still = True
        if len(recent) >= 2 and _span(recent) > PALM_STILL_SPAN_HS * tr.hs:
            still = False
        if still and len(total) >= 2 and _span(total) > PALM_STILL_TOTAL_HS * tr.hs:
            still = False
        if still and len(axes) >= 2:
            rel = [_wrap_deg(x - axes[0]) for x in axes]
            if max(rel) - min(rel) > PALM_STILL_AXIS_DEG:
                still = False
        if not still:
            tr.hold_start = t
            return
        if not tr.hold_fired and t - tr.hold_start >= PALM_HOLD_S:
            tr.hold_fired = True
            events.append(self._ev("palm_hold", tr.side, palm_n, w, h,
                                   round(t - tr.hold_start, 2)))

    def _step_flick(self, tr, t, a, palm_n, w, h, events, pointing):
        if not pointing:
            # a pause of the posture ends the sweep
            if tr.flick_hist and t - tr.flick_hist[-1][0] > 0.12:
                tr.flick_hist.clear()
            return
        ang = a["tip_ang"]
        if tr.flick_hist:
            # unwrap against the previous sample
            prev = tr.flick_hist[-1][1]
            ang = prev + _wrap_deg(ang - prev)
        tr.flick_hist.append((t, ang))
        window = min(FLICK_WINDOW_MAX_S,
                     max(FLICK_WINDOW_S, FLICK_WINDOW_FRAMES * tr.dt_ema))
        while tr.flick_hist and t - tr.flick_hist[0][0] > window:
            tr.flick_hist.popleft()
        need = 2 if tr.dt_ema >= FLICK_SLOW_FRAME_S else 3
        if len(tr.flick_hist) < need or t < tr.flick_until:
            return
        t0, a0 = tr.flick_hist[0]
        dt = t - t0
        sweep = tr.flick_hist[-1][1] - a0
        if dt <= 0.03:
            return
        omega = sweep / dt
        if abs(sweep) >= FLICK_SWEEP_DEG and abs(omega) >= FLICK_OMEGA_DEG_S:
            tr.flick_until = t + FLICK_COOLDOWN_S
            tr.flick_hist.clear()
            tip_n = palm_n
            events.append(self._ev("point_flick", tr.side, tip_n, w, h,
                                   round(omega, 1)))

    # ── two-hand gestures ───────────────────────────────────────────────
    def _live_pair(self, t) -> Optional[Tuple[_Track, _Track]]:
        live = [tr for tr in self._tracks if tr.last_t == t]
        return (live[0], live[1]) if len(live) == 2 else None

    def _step_pair(self, t, w, h, events):
        pair = self._live_pair(t)
        if pair is None:
            # Pair just broke: a fast closing that ends in a loss is a clap.
            if self._pair and 0 <= t - self._pair[-1][0] <= CLAP_LOST_WITHIN_S:
                self._maybe_clap_lost(t, events)
            elif self._pair and t - self._pair[-1][0] > 0.4:
                self._pair.clear()
            return
        a, b = pair
        hs = max((a.hs + b.hs) / 2.0, 1e-6)
        d = _dist(a.palm, b.palm) / hs
        mx = (a.palm[0] + b.palm[0]) / 2.0 / w
        my = (a.palm[1] + b.palm[1]) / 2.0 / h
        if self._pair and t - self._pair[-1][0] > 0.4:
            self._pair.clear()
        nofist = a.state != "fist" and b.state != "fist"
        self._pair.append((t, d, mx, my, a.state, b.state, nofist))
        while self._pair and t - self._pair[0][0] > 2.0:
            self._pair.popleft()
        self._check_clap(t, events)
        self._check_explode(t, events)

    def _window(self, t, span):
        return [e for e in self._pair if t - e[0] <= span]

    def _check_clap(self, t, events):
        if t < self._clap_until:
            return
        win = self._window(t, CLAP_WINDOW_S)
        if len(win) < 3:
            return
        start, now = win[0], win[-1]
        dt = now[0] - start[0]
        if dt <= 0.05 or not (start[6] and now[6]):
            return
        rate = (start[1] - now[1]) / dt
        if (start[1] >= CLAP_FROM_HS and now[1] <= CLAP_MEET_HS
                and rate >= CLAP_RATE_HS_S):
            self._fire_clap(t, now[2], now[3], rate, events)

    def _maybe_clap_lost(self, t, events):
        if t < self._clap_until:
            return
        win = self._window(self._pair[-1][0], CLAP_WINDOW_S)
        if len(win) >= 3:
            start, last = win[0], win[-1]
            dt = last[0] - start[0]
            if dt > 0.05 and start[6] and last[6]:
                rate = (start[1] - last[1]) / dt
                if last[1] <= CLAP_LOST_NEAR_HS and rate >= CLAP_LOST_RATE_HS_S:
                    self._fire_clap(t, last[2], last[3], rate, events)

    def _fire_clap(self, t, mx, my, rate, events):
        self._clap_until = t + CLAP_COOLDOWN_S
        self._clap_block_explode_until = t + EXPLODE_AFTER_CLAP_S
        self._pair.clear()
        events.append({"type": "gesture", "name": "clap", "side": "both",
                       "pos": [round(mx, 4), round(my, 4)],
                       "value": round(rate, 2)})

    def _check_explode(self, t, events):
        if t < self._explode_until or t < self._clap_block_explode_until:
            return
        if len(self._pair) < 3:
            return
        now = self._pair[-1]
        # (a) two fists flying apart
        win = self._window(t, PAIR_WINDOW_S)
        if len(win) >= 3:
            start = win[0]
            dt = now[0] - start[0]
            if dt > 0.05 and start[4] == "fist" and start[5] == "fist":
                rate = (now[1] - start[1]) / dt
                if rate >= EXPLODE_RATE_HS_S and now[1] - start[1] >= EXPLODE_GAIN_HS:
                    self._fire_explode(t, now, rate, events)
                    return
        # (b) hands held together, then parted fast
        entries = list(self._pair)
        i = len(entries) - 1
        while i >= 0 and entries[i][1] > EXPLODE_TOGETHER_HS:
            i -= 1
        if i < 0 or i == len(entries) - 1:
            return
        tog_end = entries[i]
        if now[0] - tog_end[0] > EXPLODE_PART_WINDOW_S:
            return
        j = i
        while j - 1 >= 0 and entries[j - 1][1] <= EXPLODE_TOGETHER_HS:
            j -= 1
        if tog_end[0] - entries[j][0] < EXPLODE_TOGETHER_S:
            return
        dt = now[0] - tog_end[0]
        if dt <= 0.03:
            return
        rate = (now[1] - tog_end[1]) / dt
        if rate >= EXPLODE_RATE_HS_S and now[1] - tog_end[1] >= EXPLODE_GAIN_HS:
            self._fire_explode(t, now, rate, events)

    def _fire_explode(self, t, now, rate, events):
        self._explode_until = t + EXPLODE_COOLDOWN_S
        self._pair.clear()
        events.append({"type": "gesture", "name": "explode", "side": "both",
                       "pos": [round(now[2], 4), round(now[3], 4)],
                       "value": round(rate, 2)})

    # ── event helper ────────────────────────────────────────────────────
    @staticmethod
    def _ev(name, side, pos, w, h, value=None) -> dict:
        ev = {"type": "gesture", "name": name, "side": side,
              "pos": [round(float(pos[0]), 4), round(float(pos[1]), 4)]}
        if value is not None:
            ev["value"] = value
        return ev


EVENT_NAMES = ("palm_hold", "pinch_start", "pinch_end", "pinch_hold",
               "fist_grab", "fist_release", "point_flick", "explode", "clap")
