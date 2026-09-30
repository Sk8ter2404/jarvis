"""core/face_presence.py — is someone REALLY at the desk, by the camera?

The proactive-remark gate asks "can a webcam see the owner?" through
bobert_companion.last_face_seen. Until 2026-09-30 that stamp was written by
ANY single-frame hit of the face-track loop's Haar cascade chain
(_detect_face: a strict frontal pass, a relaxed minNeighbors=3 pass, a profile
pass and a mirrored profile pass, minimum 40 px on a 1280x720 frame) — no
confidence, no size floor worth the name, no persistence, and no check that
the frame was new. One face-like blob per minute (a face on a TV across the
room, a poster, a passer-by, a frame a stalled source keeps re-serving) kept
"the owner is at his desk" true, and JARVIS talked to an empty chair
(2026-09-30 09:56 and 09:59, owner away).

Presence now needs evidence that survives all four:
  * FRESH frames only: a frame whose content fingerprint equals the previous
    frame from the same camera is a re-served / cached buffer (the Kinect
    shim's get_color_bgr(require_new=False) hands back the SAME buffer when
    the sensor stalls) and is not evidence of anything;
  * a QUALIFIED detection: found by a pass run at the cascade's STRICT
    neighbour count, minNeighbors=4 (the nearest thing Haar has to a
    confidence) — the frontal pass or the profile passes, never the relaxed
    minNeighbors=3 escalation pass (CONFIDENT_PASSES) — with a face at least
    MIN_FACE_FRAC of the frame width (someone within roughly two metres of
    the camera, not a face on a screen across the room);
  * SUSTAINED: MIN_HITS qualified hits within WINDOW_S, at least MIN_HIT_RATIO
    of the fresh frames in that window, spanning MIN_SPAN_S or more — a
    person facing the desk, not a one-frame blip.

Stdlib only: frames are duck-typed (numpy slicing + tobytes()); the monolith
owns the cameras, this module owns the arithmetic.
"""
from __future__ import annotations

import zlib
from collections import deque

MIN_FACE_FRAC = 0.07   # face box width / frame width (~90 px at 1280)
WINDOW_S      = 10.0   # the sustained-presence window
MIN_HITS      = 5      # qualified detections needed inside it
MIN_HIT_RATIO = 0.6    # ... as a share of the fresh frames inside it
MIN_SPAN_S    = 2.0    # first-to-last qualified hit inside it

# The _detect_face passes run at minNeighbors=4. "frontal_relaxed" (the
# minNeighbors=3 escalation) steers the eyes but is not presence evidence.
# The profile passes count: the desk webcam sits on a side monitor, so the
# owner facing the middle one shows it a turned head.
CONFIDENT_PASSES = frozenset({"frontal", "profile", "profile_mirror"})

_FINGERPRINT_STRIDE = 8   # every 8th pixel each way: 160x90 of 1280x720


def frame_fingerprint(frame):
    """A cheap content fingerprint of a frame (CRC32 of a strided sample), or
    None when it cannot be taken. Two frames with the same fingerprint are
    treated as the same buffer served twice. Never raises."""
    if frame is None:
        return None
    try:
        s = _FINGERPRINT_STRIDE
        sample = frame[::s, ::s]
        return zlib.crc32(sample.tobytes())
    except Exception:
        try:
            return zlib.crc32(bytes(frame))
        except Exception:
            return None


def qualifies(info, *, min_frac: float | None = None) -> bool:
    """A detection counts toward presence: a CONFIDENT_PASSES pass found it
    and the face is at least ``min_frac`` (default MIN_FACE_FRAC) of the
    frame width. ``info`` is the detector's detail dict ({"pass", "w_frac"})
    or None. Never raises."""
    try:
        if not isinstance(info, dict):
            return False
        if info.get("pass") not in CONFIDENT_PASSES:
            return False
        floor = MIN_FACE_FRAC if min_frac is None else float(min_frac)
        return float(info.get("w_frac") or 0.0) >= floor
    except Exception:
        return False


class SustainedFace:
    """Per-camera sustained-presence tracker. Feed it every frame the
    detector ran on; it answers "is a face sustained right now?"."""

    def __init__(self):
        self._frames: deque = deque()   # timestamps of fresh frames
        self._hits: deque = deque()     # timestamps of qualified fresh hits

    def _prune(self, now: float, window: float) -> None:
        edge = now - window
        while self._frames and self._frames[0] < edge:
            self._frames.popleft()
        while self._hits and self._hits[0] < edge:
            self._hits.popleft()

    def observe(self, now: float, *, fresh: bool, qualified: bool) -> bool:
        """Record one frame; True when THIS frame is a qualified fresh hit
        and the window now holds sustained presence. A re-served frame is
        ignored entirely (neither a hit nor a miss) and never confirms.
        Thresholds are read from the module at call time. Never raises."""
        try:
            now = float(now)
            window = float(WINDOW_S)
            self._prune(now, window)
            if not fresh:
                return False
            self._frames.append(now)
            if not qualified:
                return False
            self._hits.append(now)
            hits = len(self._hits)
            if hits < int(MIN_HITS):
                return False
            if hits / max(1, len(self._frames)) < float(MIN_HIT_RATIO):
                return False
            return (self._hits[-1] - self._hits[0]) >= float(MIN_SPAN_S)
        except Exception:
            return False

    def counts(self) -> tuple:
        """(fresh frames, qualified hits) currently in the window."""
        return (len(self._frames), len(self._hits))
