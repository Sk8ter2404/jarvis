"""
kinect_skeleton — pure, hardware-free skeleton→color-space projection helpers.

WHY THIS MODULE EXISTS
======================
PART A of the Kinect HUD upgrade draws the LIVE skeleton over the Kinect color
frame in the HUD preview. Turning the body stream (audio/kinect_bridge.get_bodies(),
which gives per-joint CAMERA-SPACE metres) into 2D pixel points on the 1920×1080
color frame, plus the list of BONES to stroke between adjacent joints, is a
self-contained geometry problem with NOTHING to do with cv2, pykinect2, threads,
or the HUD. So it lives here as pure functions with ZERO imports beyond the
stdlib and ZERO Kinect / cv2 contact — which makes it unit-testable on the CI
runner (where cv2 is a blocked dependency) with fabricated joints + a fake
mapper.

The two things the drawer (bobert_companion._hud_kinect_overlay_*) needs are:

  • BONES — the fixed adjacency list of the Kinect v2 25-joint skeleton, by the
    SAME friendly joint names audio/kinect_bridge emits. Stroke a line between
    each pair; dot every joint.

  • project_body_joints(joints, mapper, ...) — map each tracked joint's
    camera-space (x, y, z) to a color-space (px, py) pixel via the injected
    `mapper` callable (in production: PyKinectRuntime's
    MapCameraPointToColorSpace, fed each joint's CameraSpacePoint). Returns
    {name: (px, py)} for the joints that projected to a FINITE, on-frame-ish
    point, so the drawer can look bones up by name.

WHY AN INJECTED MAPPER (not body_joints_to_color_space directly)
================================================================
The installed pykinect2 0.1.0 `body_joints_to_color_space()` allocates a
`numpy.ndarray(..., dtype=numpy.object)` — and `numpy.object` was REMOVED in
numpy 1.24+, so that convenience method raises on this machine's modern numpy
(the same class of breakage the bridge's patch-loader works around). The robust
path is the per-joint mapper `runtime._mapper.MapCameraPointToColorSpace(point)`,
which returns a `ColorSpacePoint` with float `.x` / `.y`. bobert_companion wires
that mapper to this pure projector; the projector itself only ever calls the
callable it's handed, so the tests pass a trivial fake and never touch pykinect2.

COORDINATES
===========
`mapper(x, y, z)` is expected to return either an object with `.x`/`.y`
attributes (the real ColorSpacePoint) or a 2-tuple `(px, py)` — both are
accepted. The Kinect maps an un-seen / behind-camera joint to +/-inf or NaN;
those are dropped (a finite, vaguely on-frame point is required) so the drawer
never strokes a bone to infinity.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional


# ─── the Kinect v2 skeleton bone topology (friendly names per kinect_bridge) ──
# Each tuple is an adjacent (parent, child) joint pair to stroke a line between.
# Names MATCH audio/kinect_bridge._JOINT_NAMES exactly so a projected
# {name: (px, py)} dict can be looked up directly. This is the standard Kinect v2
# 25-joint hierarchy (spine + the two arms incl. hand-tips/thumbs + the two legs).
BONES: tuple[tuple[str, str], ...] = (
    # ── spine / head ──
    ("head", "neck"),
    ("neck", "spine_shoulder"),
    ("spine_shoulder", "spine_mid"),
    ("spine_mid", "spine_base"),
    # ── left arm ──
    ("spine_shoulder", "shoulder_left"),
    ("shoulder_left", "elbow_left"),
    ("elbow_left", "wrist_left"),
    ("wrist_left", "hand_left"),
    ("hand_left", "hand_tip_left"),
    ("wrist_left", "thumb_left"),
    # ── right arm ──
    ("spine_shoulder", "shoulder_right"),
    ("shoulder_right", "elbow_right"),
    ("elbow_right", "wrist_right"),
    ("wrist_right", "hand_right"),
    ("hand_right", "hand_tip_right"),
    ("wrist_right", "thumb_right"),
    # ── left leg ──
    ("spine_base", "hip_left"),
    ("hip_left", "knee_left"),
    ("knee_left", "ankle_left"),
    ("ankle_left", "foot_left"),
    # ── right leg ──
    ("spine_base", "hip_right"),
    ("hip_right", "knee_right"),
    ("knee_right", "ankle_right"),
    ("ankle_right", "foot_right"),
)

# Kinect color frame is fixed 1920×1080. Used to reject wildly off-frame
# projections (a joint the mapper places thousands of px outside the frame is
# almost certainly a bad/behind-camera point we don't want to draw a bone to).
COLOR_W = 1920
COLOR_H = 1080
# How far OUTSIDE the frame a projected point may sit and still be kept. A real
# joint can legitimately fall just off-frame (an arm raised past the edge), so we
# allow a generous margin rather than hard-clipping to [0, W]; only absurd points
# (NaN/inf, or many frame-widths away) are discarded.
_OFF_FRAME_MARGIN = 600


def _coerce_point(p: Any) -> Optional[tuple[float, float]]:
    """Normalise a mapper result to a finite (px, py) float tuple, or None.

    Accepts the real pykinect2 ``ColorSpacePoint`` (``.x`` / ``.y`` floats) OR a
    plain 2-sequence ``(px, py)``. Any non-finite coordinate (the Kinect maps an
    unseen / behind-camera joint to +/-inf or NaN) → None so the caller drops it.
    """
    if p is None:
        return None
    px: Any
    py: Any
    # ColorSpacePoint-style object with .x / .y attributes.
    if hasattr(p, "x") and hasattr(p, "y"):
        px, py = p.x, p.y
    else:
        # Sequence form (px, py).
        try:
            px, py = p[0], p[1]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        fx = float(px)
        fy = float(py)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return None
    return fx, fy


def _on_frame_ish(px: float, py: float,
                  width: int = COLOR_W, height: int = COLOR_H) -> bool:
    """True when (px, py) is inside the color frame plus a forgiving margin.

    Rejects only ABSURD points (a bad projection landing many frame-widths
    away); a joint legitimately just off the edge is kept so a raised arm still
    draws to the frame boundary."""
    return (-_OFF_FRAME_MARGIN <= px <= width + _OFF_FRAME_MARGIN
            and -_OFF_FRAME_MARGIN <= py <= height + _OFF_FRAME_MARGIN)


def project_body_joints(
    joints: dict,
    mapper: Callable[[float, float, float], Any],
    *,
    min_tracking_state: int = 1,
    width: int = COLOR_W,
    height: int = COLOR_H,
) -> dict[str, tuple[int, int]]:
    """Project a body's camera-space joints to color-space pixel points.

    ``joints`` is the per-body ``{name: (x, y, z, tracking_state)}`` dict
    audio/kinect_bridge.get_bodies() emits (camera-space metres). ``mapper`` is
    called ``mapper(x, y, z)`` per joint and must return a ColorSpacePoint-like
    object (``.x``/``.y``) or a ``(px, py)`` tuple — in production the bound
    PyKinectRuntime ``MapCameraPointToColorSpace`` on each joint's
    ``CameraSpacePoint`` (see module docstring on why we avoid the numpy.object
    convenience method).

    Returns ``{name: (px, py)}`` (ints) for every joint that:
      • is at least ``min_tracking_state`` (Kinect TrackingState: 0 NotTracked,
        1 Inferred, 2 Tracked — default >=1 so inferred joints still draw, which
        keeps the stick-figure whole when a hand briefly drops to inferred), and
      • projected to a FINITE point within the frame + margin.

    NEVER raises: a malformed joint tuple, a mapper that throws on one joint, or
    a None result all just omit that joint. The drawer treats a missing joint as
    "skip any bone that needs it".
    """
    out: dict[str, tuple[int, int]] = {}
    if not isinstance(joints, dict) or not callable(mapper):
        return out
    for name, j in joints.items():
        # joint tuple is (x, y, z, tracking_state); tolerate a bare (x,y,z).
        try:
            if j is None or len(j) < 3:
                continue
            state = int(j[3]) if len(j) >= 4 else min_tracking_state
            if state < min_tracking_state:
                continue
            x, y, z = float(j[0]), float(j[1]), float(j[2])
        except (TypeError, ValueError, IndexError):
            continue
        try:
            raw = mapper(x, y, z)
        except Exception:
            # A per-joint mapper hiccup must not abort the whole skeleton.
            continue
        pt = _coerce_point(raw)
        if pt is None:
            continue
        px, py = pt
        if not _on_frame_ish(px, py, width, height):
            continue
        out[name] = (int(round(px)), int(round(py)))
    return out


def iter_bone_segments(
    points: dict[str, tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Bone line segments to stroke, given projected ``{name: (px, py)}`` points.

    For each (parent, child) in :data:`BONES` where BOTH endpoints projected,
    yields ``((x1, y1), (x2, y2))``. A bone with a missing endpoint (joint not
    tracked / off-frame) is skipped, so a partially-visible body still draws the
    bones it can. Pure — the drawer just cv2.line()s each returned segment."""
    segs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in BONES:
        pa = points.get(a)
        pb = points.get(b)
        if pa is not None and pb is not None:
            segs.append((pa, pb))
    return segs
