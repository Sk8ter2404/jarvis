"""
kinect_pointing — pure, hardware-free arm-pointing math + a calibration store
for the Kinect v2.

WHY THIS MODULE EXISTS
======================
"Point-to-control": the owner points an arm at a real device (a lamp, a fan)
and says "turn that on", and JARVIS controls the right smart-home device. The
raw skeleton stream (audio/kinect_bridge.get_bodies()) gives per-frame joint
positions in camera-space metres; turning that into a normalised pointing RAY,
then matching that ray against a small set of CALIBRATED directions, is a
self-contained geometry problem with NOTHING to do with the sensor, threading,
or the smart-home dispatch. So it lives here as pure functions + a tiny JSON
store with ZERO Kinect contact and ZERO third-party imports (stdlib only):

  • `arm_direction(body)` picks the arm the user is pointing WITH (the most
    extended / most raised arm whose joints are actually tracked) and returns
    its origin (the shoulder) plus a unit direction vector, or None.

  • `angle_between(d1, d2)` is the angular distance between two directions, in
    degrees — the metric both calibration and resolution use.

  • `PointingStore` persists `{target_name: {"dir":[x,y,z], "device": <bound
    smart-home device name>, "ts": ...}}` to a SEPARATE gitignored json
    (data/kinect_pointing.json by default) with an atomic write, and resolves a
    live pointing direction to the closest calibrated target within an angular
    threshold.

Every threshold is a named module constant so the live behaviour is tunable
without touching the algorithm, and so the tests assert against the same
numbers the code uses. NOTHING here raises on a malformed / partial frame — a
missing joint, a None body, an untracked joint all degrade to "no direction".

FRAME SHAPE (one entry per audio.kinect_bridge.get_bodies() element)
===================================================================
    {"id": int,
     "joints": {name: (x, y, z, tracking_state), ...},   # metres, camera space
     "head": (x, y, z) | None,
     "distance_m": float | None,
     "facing": bool | None}

Joint names are the bridge's snake_case keys: shoulder_right, elbow_right,
wrist_right, hand_right, hand_tip_right (and the _left mirror), head,
spine_base/mid/shoulder, neck.

Camera space (per the SDK / the bridge docstring): x increases to the sensor's
RIGHT, y increases UP, z increases with depth (forward, away from the sensor).
A joint's tracking_state is 0 not-tracked, 1 inferred, 2 tracked.
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from typing import Any, Optional

try:
    import json
except Exception:   # pragma: no cover - json is stdlib; defensive only
    json = None     # type: ignore


# ─── tunable thresholds (named so they're adjustable + test-visible) ───────
# A joint counts toward a pointing ray only when its TrackingState is >= this
# (2 == fully tracked). Inferred/untracked joints are too noisy to aim with.
MIN_TRACKING_STATE = 2

# Resolution: a live pointing direction matches a calibrated target only when
# the angle between them is <= this many degrees. ~18° is a comfortable cone:
# wide enough to forgive arm wobble + calibration drift, tight enough that two
# devices in different directions don't both match. Pick the CLOSEST when
# several fall inside the cone.
RESOLVE_MAX_ANGLE_DEG = 18.0

# Calibration: while sampling, reject any single frame whose direction sits
# more than this far from the running mean — a steadiness gate so a flailing
# arm or a mid-sample twitch can't poison the stored vector.
CALIBRATE_STEADY_MAX_ANGLE_DEG = 25.0

# Minimum number of accepted (steady, tracked) frames a calibration must gather
# before it will store a direction. Below this we don't trust the average.
CALIBRATE_MIN_SAMPLES = 5

# How "extended" an arm must be to be considered a deliberate point: the
# shoulder→hand distance, in metres. A relaxed arm at the side is ~0.5-0.6 m;
# an extended point is longer. Used only to PICK between two tracked arms, never
# as a hard reject (a short but clearly-tracked arm still yields a ray).
ARM_EXTENSION_GOOD_M = 0.45


# ─── tiny 3D vector helpers (plain tuples; no numpy) ───────────────────────
Vec3 = tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Optional[Vec3]:
    """Unit vector, or None for a (near-)zero vector (no direction)."""
    n = _norm(v)
    if n <= 1e-9:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _joint_ok(j: Optional[tuple]) -> bool:
    """True when a (x, y, z, state) joint is FULLY tracked (state >= MIN_TRACKING_
    STATE) AND carries a real, finite, non-zero-fill position.

    GHOST FIX (2026-07-15, CONFIRMED): the old check tested ONLY the tracking
    state, so a state-2 joint still carrying a NaN/±Inf coordinate (an SDK glitch
    kinect_bridge._joint_reliable already guards against) — or the NotTracked
    (0,0,0) zero-fill — produced a garbage pointing ray, which store.resolve()
    then matched at 0.0° against the FIRST calibrated target and fired
    'turn that on' on the wrong smart-home device. Mirror _joint_reliable here so
    every joint feeding a ray (tip / origin / extension) is filtered at this one
    chokepoint."""
    if not j or len(j) < 4:
        return False
    try:
        if int(j[3]) < MIN_TRACKING_STATE:
            return False
        x, y, z = float(j[0]), float(j[1]), float(j[2])
        if x == 0.0 and y == 0.0 and z == 0.0:   # NotTracked zero-fill
            return False
        if not (x == x and y == y and z == z):    # NaN != itself
            return False
        inf = float("inf")
        if x in (inf, -inf) or y in (inf, -inf) or z in (inf, -inf):
            return False
        return True
    except (TypeError, ValueError):
        return False


def _xyz(j: tuple) -> Vec3:
    return (float(j[0]), float(j[1]), float(j[2]))


# ─── public geometry API ───────────────────────────────────────────────────
def angle_between(d1: Optional[Vec3], d2: Optional[Vec3]) -> Optional[float]:
    """Angle between two direction vectors, in DEGREES (0..180). Returns None
    if either is missing or zero-length. Inputs need not be pre-normalised."""
    if d1 is None or d2 is None:
        return None
    u1 = _normalize((float(d1[0]), float(d1[1]), float(d1[2])))
    u2 = _normalize((float(d2[0]), float(d2[1]), float(d2[2])))
    if u1 is None or u2 is None:
        return None
    # Clamp for float error so acos never sees |x|>1.
    c = max(-1.0, min(1.0, _dot(u1, u2)))
    return math.degrees(math.acos(c))


def _arm_ray(joints: dict, side: str) -> Optional[tuple[Vec3, Vec3, float]]:
    """For one side ('right'/'left'), build the best pointing ray:
        (origin, unit_dir, extension_m)
    where origin is the shoulder and dir points shoulder→hand. Prefers the
    longest reliable segment: shoulder→hand_tip, then shoulder→hand, then
    elbow→hand. Returns None if the required joints aren't tracked.

    `extension_m` is the shoulder→hand distance (deliberate-point heuristic);
    when the shoulder is unavailable we fall back to an elbow→hand ray and
    report its length as the extension proxy."""
    j = joints
    shoulder = j.get(f"shoulder_{side}")
    elbow = j.get(f"elbow_{side}")
    hand = j.get(f"hand_{side}")
    hand_tip = j.get(f"hand_tip_{side}")
    wrist = j.get(f"wrist_{side}")

    # The aim point (far end of the ray): prefer the fingertip, then the hand,
    # then the wrist — whichever is tracked and furthest down the arm.
    tip = None
    for cand in (hand_tip, hand, wrist):
        if _joint_ok(cand):
            tip = _xyz(cand)
            break
    if tip is None:
        return None

    # The origin (near end): prefer the shoulder (full-arm ray = best aim),
    # else the elbow (forearm ray). Require it tracked.
    if _joint_ok(shoulder):
        origin = _xyz(shoulder)
    elif _joint_ok(elbow):
        origin = _xyz(elbow)
    else:
        return None

    direction = _normalize(_sub(tip, origin))
    if direction is None:
        return None

    # Extension = shoulder→hand distance when we have both; else the ray length.
    if _joint_ok(shoulder) and _joint_ok(hand):
        extension = _norm(_sub(_xyz(hand), _xyz(shoulder)))
    else:
        extension = _norm(_sub(tip, origin))
    return (origin, direction, extension)


def arm_direction(body: Any) -> Optional[tuple[Vec3, Vec3]]:
    """Compute the arm pointing ray for one get_bodies() body.

    Returns (origin, unit_dir) — origin is the pointing shoulder, dir is the
    normalised shoulder→hand direction in camera space — or None if neither arm
    is tracked well enough to aim with.

    Arm selection: build a candidate ray for each side (right first), then pick
    the arm the user is most plausibly POINTING with — the most extended /
    most raised one. Concretely we score each candidate by how extended it is
    (shoulder→hand distance) plus a bonus for the hand being raised above the
    shoulder (a deliberate point is rarely hanging at the side). The right arm
    wins ties (most people point right-handed), matching "fall back to the left
    arm only if the right isn't the one being used"."""
    if not isinstance(body, dict):
        return None
    joints = body.get("joints") or {}
    if not isinstance(joints, dict) or not joints:
        return None

    best = None          # (score, origin, dir)
    for side in ("right", "left"):
        ray = _arm_ray(joints, side)
        if ray is None:
            continue
        origin, direction, extension = ray
        # Raised-hand bonus: hand.y above shoulder.y → a more deliberate point.
        # dir[1] is the y-component of the unit direction; >0 means the ray aims
        # upward (hand above origin). Scale modestly so extension still leads.
        raised_bonus = max(0.0, direction[1]) * 0.5
        score = extension + raised_bonus
        # Right arm wins exact ties (checked first → only replace on strict >).
        if best is None or score > best[0]:
            best = (score, origin, direction)
    if best is None:
        return None
    return (best[1], best[2])


# ─── calibration store ──────────────────────────────────────────────────────
def _default_store_path() -> str:
    """data/kinect_pointing.json under the project root. Honours the
    JARVIS_POINTING_PATH env override so tests (and a relocated install) can
    point it at a throwaway file WITHOUT ever touching the real one."""
    env = os.environ.get("JARVIS_POINTING_PATH")
    if env:
        return env
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project, "data", "kinect_pointing.json")


def _atomic_write_json(path: str, data: Any) -> None:
    """Mirror skills/smart_home_discover._atomic_write_json: write to a temp
    file in the same dir, then os.replace() so a reader never sees a half-
    written file and a crash mid-write can't corrupt the store."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _now() -> float:
    return time.time()


class PointingStore:
    """The calibrated direction→target map, persisted to a gitignored json.

    Shape on disk:
        {"version": 1,
         "targets": {
            "<target_name>": {"dir": [x, y, z], "device": "<bound name>",
                              "ts": <unix>},
            ...}}

    `target_name` is the name the user SAYS (e.g. "desk lamp"); `device` is the
    real smart-home device name it was bound to at calibration time (may equal
    target_name, or be empty if no catalog match — resolved loosely at control
    time by the skill). Directions are stored normalised.

    Every method is best-effort: a missing / corrupt file reads as an empty map
    rather than raising, so a first-run calibrate just creates it.
    """

    def __init__(self, path: Optional[str] = None,
                 now_fn=_now) -> None:
        self.path = path or _default_store_path()
        self._now = now_fn

    # ── load / save ─────────────────────────────────────────────────────────
    def _load(self) -> dict:
        try:
            if not os.path.exists(self.path):
                return {"version": 1, "targets": {}}
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "targets": {}}
            data.setdefault("version", 1)
            tgt = data.get("targets")
            if not isinstance(tgt, dict):
                data["targets"] = {}
            return data
        except Exception:
            return {"version": 1, "targets": {}}

    def _save(self, data: dict) -> bool:
        try:
            _atomic_write_json(self.path, data)
            return True
        except Exception:
            return False

    # ── queries ─────────────────────────────────────────────────────────────
    def list_targets(self) -> list[dict]:
        """Every calibrated target as a list of dicts:
            [{"name": ..., "device": ..., "dir": [x,y,z], "ts": ...}, ...]
        sorted by name. Empty list when nothing is calibrated."""
        data = self._load()
        out: list[dict] = []
        for name, rec in (data.get("targets") or {}).items():
            if not isinstance(rec, dict):
                continue
            out.append({
                "name": name,
                "device": rec.get("device") or name,
                "dir": list(rec.get("dir") or []),
                "ts": rec.get("ts"),
            })
        out.sort(key=lambda r: (r.get("name") or "").lower())
        return out

    def get(self, name: str) -> Optional[dict]:
        """The stored record for one target name (case-insensitive), or None."""
        if not name:
            return None
        data = self._load()
        targets = data.get("targets") or {}
        # Exact key first, then case-insensitive.
        rec = targets.get(name)
        if rec is None:
            low = name.strip().lower()
            for k, v in targets.items():
                if k.lower() == low:
                    rec = v
                    break
        if not isinstance(rec, dict):
            return None
        return rec

    def device_for(self, name: str) -> Optional[str]:
        """The bound smart-home device name for a target (falls back to the
        target name itself when no explicit binding was stored)."""
        rec = self.get(name)
        if rec is None:
            return None
        return rec.get("device") or name

    # ── mutations ───────────────────────────────────────────────────────────
    def put(self, name: str, direction: Vec3,
            device: Optional[str] = None) -> bool:
        """Store / overwrite a target's calibrated direction (normalised on
        write). `device` is the bound smart-home device name; when omitted the
        target name doubles as the device name. Returns True on a durable save.
        """
        if not name or not name.strip():
            return False
        unit = _normalize((float(direction[0]), float(direction[1]),
                           float(direction[2])))
        if unit is None:
            return False
        data = self._load()
        targets = data.setdefault("targets", {})
        targets[name.strip()] = {
            "dir": [unit[0], unit[1], unit[2]],
            "device": (device or name).strip(),
            "ts": self._now(),
        }
        return self._save(data)

    def remove_target(self, name: str) -> bool:
        """Forget a target. Returns True if one was removed (case-insensitive
        match), False if there was nothing by that name."""
        if not name:
            return False
        data = self._load()
        targets = data.get("targets") or {}
        if name in targets:
            del targets[name]
            self._save(data)
            return True
        low = name.strip().lower()
        for k in list(targets.keys()):
            if k.lower() == low:
                del targets[k]
                self._save(data)
                return True
        return False

    # ── resolution ──────────────────────────────────────────────────────────
    def resolve(self, direction: Optional[Vec3],
                max_angle_deg: float = RESOLVE_MAX_ANGLE_DEG
                ) -> Optional[str]:
        """Return the calibrated target NAME whose stored direction is closest
        to `direction` AND within `max_angle_deg`; None if nothing qualifies.

        Closest-wins: among every target inside the cone, the one with the
        smallest angular distance is returned, so two calibrated directions
        that both fall inside the (generous) cone disambiguate to the nearer."""
        if direction is None:
            return None
        best_name: Optional[str] = None
        best_angle = float("inf")
        for rec in self.list_targets():
            stored = rec.get("dir")
            if not stored or len(stored) < 3:
                continue
            ang = angle_between(direction, (stored[0], stored[1], stored[2]))
            if ang is None:
                continue
            if ang <= max_angle_deg and ang < best_angle:
                best_angle = ang
                best_name = rec.get("name")
        return best_name


# ─── calibration sampling (pure: caller supplies the frame source) ─────────
def average_direction(directions: list[Optional[Vec3]],
                      steady_max_angle_deg: float = CALIBRATE_STEADY_MAX_ANGLE_DEG
                      ) -> Optional[Vec3]:
    """Average a list of per-frame pointing directions into one steady unit
    vector, rejecting outliers, or None if too few survive.

    Two-pass: (1) take the unit mean of all valid directions as a provisional
    centre, (2) keep only directions within `steady_max_angle_deg` of it and
    re-average. This drops the odd twitch frame without needing the caller to
    gate frames itself. Returns None when fewer than CALIBRATE_MIN_SAMPLES
    survive — the signal we use to say "hold still / I couldn't get a steady
    read"."""
    valid = []
    for d in directions:
        if d is None:
            continue
        u = _normalize((float(d[0]), float(d[1]), float(d[2])))
        if u is not None:
            valid.append(u)
    if len(valid) < CALIBRATE_MIN_SAMPLES:
        return None

    def _mean(vs: list[Vec3]) -> Optional[Vec3]:
        sx = sum(v[0] for v in vs)
        sy = sum(v[1] for v in vs)
        sz = sum(v[2] for v in vs)
        return _normalize((sx, sy, sz))

    centre = _mean(valid)
    if centre is None:
        return None
    kept = [v for v in valid
            if (angle_between(v, centre) or 999.0) <= steady_max_angle_deg]
    if len(kept) < CALIBRATE_MIN_SAMPLES:
        return None
    return _mean(kept)
