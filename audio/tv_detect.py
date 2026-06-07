"""
tv_detect — pure, hardware-free "is a TV screen visibly ON?" math + a calibration
store for the region of the camera frame the TV occupies.

WHY THIS MODULE EXISTS
======================
JARVIS already suppresses ambient-learning ingestion of TV chatter via AUDIO
(SMTC now-playing + the spectral room-music detector + a local-LLM content judge
+ voice-ID). This module adds an INDEPENDENT VISUAL signal: if a camera sees a
bright, flickering rectangle (a powered-on TV/monitor), that's strong evidence
the room audio is the TV — so ambient ingestion should be vetoed even when every
audio gate misses (a muted TV, an unrecognised stream, a show the content judge
can't classify).

The decision is a self-contained image-statistics + hysteresis problem with
NOTHING to do with the camera, threading, or the ambient seam. So it lives here
as pure functions + a tiny JSON store with ZERO camera contact and (importantly)
NO hard dependency on OpenCV — the per-frame statistics are computed with numpy
alone so the logic runs on the CI runner (which installs numpy but BLOCKS cv2).
The live wiring (reading the monolith's cached frame, the rolling window, the
voice actions, the ambient OR-signal) is skills/tv_detect.py.

THE SIGNAL
==========
Two per-frame statistics over the calibrated region (whole frame if uncalibrated):

  • BRIGHTNESS — mean luma (0..255). A dark room / a camera pointed at a wall
    reads low; a lit TV panel reads high. `frame_brightness()`.

  • TEMPORAL VARIANCE — the mean absolute luma change between THIS frame and the
    PREVIOUS one (0..255), i.e. how much the picture is moving/flickering.
    A static scene (an off TV, a still poster, an empty couch) reads ~0; live
    video content reads high. `frame_temporal_delta()`.

A single bright OR a single moving frame is not enough — a lamp is bright but
static; a person walking past a dark TV moves but isn't bright. The verdict
(`TVDecider`) requires BOTH, SUSTAINED over a short rolling window, with simple
hysteresis (it takes a few qualifying frames to switch ON and a few quiet frames
to switch OFF) so a one-frame glint or a single dropped frame can't flip it.

Every threshold is a named module constant so the live behaviour is tunable
without touching the algorithm, and so the tests assert against the same numbers
the code uses. NOTHING here raises on a malformed / partial / None frame — it
degrades to "no reading" (None) and the decider treats a gap as a non-qualifying
frame.

FRAME SHAPE
===========
A frame is whatever the face-tracker cached: an OpenCV BGR ``numpy.ndarray`` of
shape (H, W, 3) dtype uint8 (a grayscale (H, W) frame is also accepted). Pixel
order (BGR vs RGB) is irrelevant to luma here — we use symmetric-enough BT.601
weights and the brightness/delta thresholds are channel-order agnostic.

REGION
======
A calibration rectangle is stored NORMALISED (fractions 0..1 of width/height) so
it survives a camera-resolution change: ``{"x":..,"y":..,"w":..,"h":..}`` with
the top-left at (x, y). ``crop_region(frame, region)`` clamps it to the frame and
returns the sub-array; a missing / degenerate region falls back to the whole
frame (the documented uncalibrated behaviour).
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Optional

try:
    import json
except Exception:   # pragma: no cover - json is stdlib; defensive only
    json = None     # type: ignore

try:
    import numpy as _np
except Exception:   # pragma: no cover - numpy is a hard dep on CI + dev
    _np = None       # type: ignore


# ─── tunable thresholds (named so they're adjustable + test-visible) ───────
# BRIGHTNESS floor (mean luma, 0..255): a region must be at least this bright to
# count as a lit screen. A dark living room reads well under this; a powered TV
# panel sits comfortably above it. Deliberately not too high — a dim
# movie/scene still clears it, and the temporal-variance gate is what actually
# separates "a lit wall" from "a playing screen".
BRIGHTNESS_ON_MIN = 60.0

# TEMPORAL-VARIANCE floor (mean abs luma delta between consecutive frames,
# 0..255): the region must be CHANGING at least this much frame-to-frame to count
# as live video. A static scene (off TV, a poster, an empty couch) reads ~0–1;
# real playing content (cuts, motion, flicker) reads well above this. This is the
# signal that stops a bright-but-static lamp/window from reading as a TV.
TEMPORAL_DELTA_ON_MIN = 4.0

# Hysteresis window: how many of the most-recent readings to consider, and how
# many of them must qualify (bright AND moving) to declare the TV ON. Requiring a
# MAJORITY over a few frames means a single glint or one dropped/blurred frame
# can't trip or untrip the verdict. With ~1 reading/second (the skill's poll
# cadence) this is a few seconds of evidence either way.
DECISION_WINDOW = 5            # consider the last N readings
ON_QUALIFYING_FRAMES = 3       # >= this many qualifying → ON (switch-on threshold)
OFF_QUALIFYING_FRAMES = 1      # <= this many qualifying → OFF (switch-off threshold)

# A reading older than this many seconds is stale (the camera stopped feeding
# frames) and is ignored by the decider — a frozen last-frame must not keep the
# TV verdict latched ON forever. The live skill also won't read a stale frame,
# but the decider enforces it too so the math is self-contained + testable.
READING_MAX_AGE_S = 8.0

# Below this many CONSECUTIVE frames the decider has not yet seen enough history
# to make any positive call — it reports OFF (fail-safe: never SUPPRESS learning
# on thin evidence). One frame alone also can't produce a temporal delta.
MIN_FRAMES_FOR_DECISION = 2


# ─── per-frame statistics (pure; numpy only — NO cv2) ──────────────────────
def _as_luma(frame: Any) -> "Optional[_np.ndarray]":
    """Return a 2-D float luma image (0..255) for a BGR/grayscale frame, or None
    if numpy is missing or the frame is unusable. NEVER raises.

    Luma uses BT.601 weights on the first three channels. Because a TV is
    detected by *brightness + change*, not colour fidelity, the exact channel
    order (BGR vs RGB) does not matter — the weights are close enough that a
    swap shifts the mean by a negligible amount well inside the threshold
    margins."""
    if _np is None or frame is None:
        return None
    try:
        arr = _np.asarray(frame)
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            return arr.astype(_np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            b = arr[:, :, 0].astype(_np.float32)
            g = arr[:, :, 1].astype(_np.float32)
            r = arr[:, :, 2].astype(_np.float32)
            # BT.601 luma. (Treating the first channel as blue per OpenCV BGR;
            # the symmetric green weight dominates so an RGB frame differs only
            # trivially.)
            return 0.114 * b + 0.587 * g + 0.299 * r
        return None
    except Exception:
        return None


def crop_region(frame: Any, region: "Optional[dict]") -> Any:
    """Crop `frame` to a NORMALISED region dict {"x","y","w","h"} (each 0..1,
    fractions of width/height, top-left origin). Returns the whole frame when
    `region` is None/empty/degenerate or anything goes wrong — that whole-frame
    fallback IS the documented uncalibrated behaviour. Never raises."""
    if frame is None or not region or _np is None:
        return frame
    try:
        arr = _np.asarray(frame)
        if arr.ndim < 2:
            return frame
        h, w = arr.shape[0], arr.shape[1]
        if h <= 0 or w <= 0:
            return frame
        rx = float(region.get("x", 0.0))
        ry = float(region.get("y", 0.0))
        rw = float(region.get("w", 1.0))
        rh = float(region.get("h", 1.0))
        # Clamp to [0,1] and to a sane rectangle.
        x0 = max(0, min(w - 1, int(round(rx * w))))
        y0 = max(0, min(h - 1, int(round(ry * h))))
        x1 = max(x0 + 1, min(w, int(round((rx + rw) * w))))
        y1 = max(y0 + 1, min(h, int(round((ry + rh) * h))))
        sub = arr[y0:y1, x0:x1]
        if sub.size == 0:
            return frame
        return sub
    except Exception:
        return frame


def frame_brightness(frame: Any, region: "Optional[dict]" = None) -> Optional[float]:
    """Mean luma (0..255) of the (optionally region-cropped) frame, or None if it
    can't be computed. Higher = brighter. Never raises."""
    luma = _as_luma(crop_region(frame, region))
    if luma is None:
        return None
    try:
        return float(luma.mean())
    except Exception:
        return None


def frame_temporal_delta(frame_a: Any, frame_b: Any,
                         region: "Optional[dict]" = None) -> Optional[float]:
    """Mean absolute luma difference (0..255) between two frames over the
    (optional) region — the flicker/motion signal. Returns None if either frame
    is unusable or their cropped shapes differ (e.g. the camera changed
    resolution mid-stream), which the decider treats as 'no reading' rather than
    a spurious full-frame change. Never raises."""
    la = _as_luma(crop_region(frame_a, region))
    lb = _as_luma(crop_region(frame_b, region))
    if la is None or lb is None:
        return None
    try:
        if la.shape != lb.shape:
            return None
        return float(_np.abs(la - lb).mean())
    except Exception:
        return None


def frame_qualifies(brightness: Optional[float],
                    temporal_delta: Optional[float],
                    bright_min: float = BRIGHTNESS_ON_MIN,
                    delta_min: float = TEMPORAL_DELTA_ON_MIN) -> bool:
    """A single reading 'qualifies' as a lit, playing screen iff it is BOTH
    bright enough AND changing enough. A missing statistic disqualifies it
    (can't confirm a TV from absence). Pure + side-effect free."""
    if brightness is None or temporal_delta is None:
        return False
    return (brightness >= bright_min) and (temporal_delta >= delta_min)


# ─── rolling decision with hysteresis (pure; caller feeds readings) ────────
class TVDecider:
    """Sustained-evidence verdict over a rolling window of per-frame readings.

    Feed it one ``observe(brightness, temporal_delta, ts)`` per polled frame; ask
    ``is_on()`` for the current latched verdict. Hysteresis: it takes
    ``ON_QUALIFYING_FRAMES`` qualifying readings within the last
    ``DECISION_WINDOW`` to switch ON, and drops to OFF once qualifying readings
    fall to ``OFF_QUALIFYING_FRAMES`` or fewer — so a momentary glint or a single
    dropped frame can't flip the decision. Stale readings (older than
    ``READING_MAX_AGE_S``) are discarded so a frozen camera can't latch ON.

    Pure: holds only its own small ring of (qualifies, ts) booleans + the latched
    state. No clock of its own — the caller passes ``ts`` (and ``now`` to the
    queries) so tests are deterministic.
    """

    def __init__(self, window: int = DECISION_WINDOW,
                 on_frames: int = ON_QUALIFYING_FRAMES,
                 off_frames: int = OFF_QUALIFYING_FRAMES,
                 max_age_s: float = READING_MAX_AGE_S,
                 bright_min: float = BRIGHTNESS_ON_MIN,
                 delta_min: float = TEMPORAL_DELTA_ON_MIN) -> None:
        self.window = max(1, int(window))
        self.on_frames = max(1, int(on_frames))
        self.off_frames = max(0, int(off_frames))
        self.max_age_s = float(max_age_s)
        self.bright_min = float(bright_min)
        self.delta_min = float(delta_min)
        self._readings: list[tuple[bool, float]] = []   # (qualifies, ts)
        self._on = False

    def observe(self, brightness: Optional[float],
                temporal_delta: Optional[float],
                ts: Optional[float] = None) -> bool:
        """Record one reading and re-evaluate the latched verdict. Returns the
        verdict AFTER this reading. ``ts`` defaults to wall-clock now."""
        now = time.time() if ts is None else float(ts)
        q = frame_qualifies(brightness, temporal_delta,
                             self.bright_min, self.delta_min)
        self._readings.append((q, now))
        # Keep only the most-recent `window` readings.
        if len(self._readings) > self.window:
            self._readings = self._readings[-self.window:]
        self._reevaluate(now)
        return self._on

    def _fresh(self, now: float) -> list[bool]:
        """The qualify-flags of readings still within max_age_s of `now`."""
        return [q for (q, ts) in self._readings
                if (now - ts) <= self.max_age_s]

    def _reevaluate(self, now: float) -> None:
        fresh = self._fresh(now)
        if len(fresh) < MIN_FRAMES_FOR_DECISION:
            # Not enough recent evidence to assert anything — fail safe to OFF.
            self._on = False
            return
        qualifying = sum(1 for q in fresh if q)
        if not self._on:
            if qualifying >= self.on_frames:
                self._on = True
        else:
            if qualifying <= self.off_frames:
                self._on = False

    def is_on(self, now: Optional[float] = None) -> bool:
        """The current latched verdict. Passing ``now`` lets a caller (or test)
        age out stale readings WITHOUT recording a new one — so a verdict can
        decay to OFF when frames stop arriving."""
        if now is not None:
            self._reevaluate(float(now))
        return self._on

    def reset(self) -> None:
        self._readings.clear()
        self._on = False

    def debug_state(self, now: Optional[float] = None) -> dict:
        """A small dict of the current evidence — for status reporting / logs."""
        n = time.time() if now is None else float(now)
        fresh = self._fresh(n)
        return {
            "on": self._on,
            "fresh_readings": len(fresh),
            "qualifying": sum(1 for q in fresh if q),
            "window": self.window,
            "on_frames": self.on_frames,
            "off_frames": self.off_frames,
        }


# ─── calibration store (gitignored json, atomic write) ─────────────────────
def _default_store_path() -> str:
    """data/tv_region.json under the project root. Honours the
    JARVIS_TV_REGION_PATH env override so tests (and a relocated install) can
    point it at a throwaway file WITHOUT ever touching the real one. Mirrors
    audio/kinect_pointing.py's store-path contract."""
    env = os.environ.get("JARVIS_TV_REGION_PATH")
    if env:
        return env
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project, "data", "tv_region.json")


def _atomic_write_json(path: str, data: Any) -> None:
    """Write to a temp file in the same dir, then os.replace() so a reader never
    sees a half-written file and a crash mid-write can't corrupt the store.
    Mirrors audio/kinect_pointing._atomic_write_json."""
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


def normalize_region(x: float, y: float, w: float, h: float
                     ) -> "Optional[dict]":
    """Clamp a normalised rectangle to a valid in-frame region, or None if it's
    degenerate (zero/negative area after clamping). Inputs are fractions 0..1 of
    width/height with the top-left at (x, y). Pure."""
    try:
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = float(w)
        h = float(h)
    except Exception:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    # Clamp the far edge into the frame, then recompute width/height.
    x1 = min(1.0, x + w)
    y1 = min(1.0, y + h)
    w = x1 - x
    h = y1 - y
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


class TVRegionStore:
    """The calibrated TV-region rectangle, persisted to a gitignored json.

    Shape on disk:
        {"version": 1,
         "region": {"x":.., "y":.., "w":.., "h":..} | null,
         "ts": <unix> | null}

    The region is NORMALISED (fractions of frame width/height) so it survives a
    resolution change. A null/absent region means 'uncalibrated' → the detector
    uses the whole frame. Every method is best-effort: a missing / corrupt file
    reads as 'uncalibrated' rather than raising, so a first-run calibrate just
    creates it. Mirrors audio/kinect_pointing.PointingStore.
    """

    def __init__(self, path: Optional[str] = None, now_fn=_now) -> None:
        self.path = path or _default_store_path()
        self._now = now_fn

    # ── load / save ─────────────────────────────────────────────────────────
    def _load(self) -> dict:
        try:
            if not os.path.exists(self.path):
                return {"version": 1, "region": None, "ts": None}
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "region": None, "ts": None}
            data.setdefault("version", 1)
            data.setdefault("region", None)
            data.setdefault("ts", None)
            return data
        except Exception:
            return {"version": 1, "region": None, "ts": None}

    def _save(self, data: dict) -> bool:
        try:
            _atomic_write_json(self.path, data)
            return True
        except Exception:
            return False

    # ── queries ─────────────────────────────────────────────────────────────
    def get_region(self) -> "Optional[dict]":
        """The calibrated normalised region dict, or None when uncalibrated /
        the stored value is malformed (→ whole-frame fallback)."""
        rec = self._load().get("region")
        if not isinstance(rec, dict):
            return None
        try:
            return normalize_region(rec.get("x", 0.0), rec.get("y", 0.0),
                                    rec.get("w", 1.0), rec.get("h", 1.0))
        except Exception:
            return None

    def is_calibrated(self) -> bool:
        return self.get_region() is not None

    # ── mutations ───────────────────────────────────────────────────────────
    def put_region(self, x: float, y: float, w: float, h: float) -> bool:
        """Store a normalised region (clamped). Returns True on a durable save,
        False if the rectangle was degenerate or the write failed."""
        region = normalize_region(x, y, w, h)
        if region is None:
            return False
        data = self._load()
        data["region"] = region
        data["ts"] = self._now()
        return self._save(data)

    def clear(self) -> bool:
        """Forget the calibrated region (back to whole-frame). True if saved."""
        data = self._load()
        data["region"] = None
        data["ts"] = self._now()
        return self._save(data)
