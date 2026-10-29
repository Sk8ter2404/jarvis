"""face_detect_fast - the cheaper at-desk face detector path, and its shadow
comparison (2026-10-28).

WHY. The face-track loop's Haar cascade passes are its whole cost: on a
1280x720 frame with nobody in it all four passes run (frontal, relaxed
frontal, profile, mirrored profile) and take ~90 ms wall and ~1.2 CPU-s
across OpenCV's worker threads. While a face is seen that runs on every
frame, so the camera loop (and the HUD preview it writes) is held to the
detector's pace. The presence gate (core/face_presence.py) only counts
faces at least MIN_FACE_FRAC (7 %) of the frame width - ~90 px at 1280,
~45 px at 640 - and the frontal cascade's own window is 24 px, so running
the SAME passes on a frame downscaled to FACE_DETECT_FAST_WIDTH, with the
smallest face searched for held at FACE_DETECT_FAST_MIN_FRAC of the width,
should find the same people for a fraction of the work.

"Should" is the point of the three modes (FACE_DETECT_FAST_MODE):
  off     today's full-resolution passes, nothing else.
  shadow  today's passes decide; on 1 in FACE_DETECT_SHADOW_EVERY_N detector
          runs the cheap path ALSO runs on the same frame and only the
          agreement is kept - same / different verdict, the time each took.
          Every FACE_DETECT_SHADOW_REPORT_N samples one log line says so.
  on      the cheap path decides.

PRIVACY. Nothing here keeps, writes or logs an image, a crop, a box or a
face position: the shadow keeps counts and timings only.

Pure apart from the cv2 module the caller passes in (no cv2 / numpy import
here), so the arithmetic and the stats run on the light-deps CI runner.
Never raises into the face-track loop.
"""
from __future__ import annotations

import statistics
import threading

MODES = ("off", "shadow", "on")

# The cascade parameters _detect_face has always used, at full resolution.
SCALE_FACTOR = 1.05
STRICT = {"minNeighbors": 4, "min_px": 40}
RELAXED = {"minNeighbors": 3, "min_px": 60}

# Kept in step with core/face_presence.CONFIDENT_PASSES (re-read at runtime).
_CONFIDENT_FALLBACK = frozenset({"frontal", "profile", "profile_mirror"})


def normalise_mode(value) -> str:
    """'off' / 'shadow' / 'on'. Anything else (a typo, None, a number) is
    'off' - today's detector - never a guess at the cheaper one."""
    try:
        v = str(value).strip().lower()
    except Exception:
        return "off"
    if v in ("true", "1", "yes"):
        return "on"
    return v if v in MODES else "off"


def _min_size(px: int, scale: float, floor: int) -> tuple:
    n = max(1, int(round(px * scale)), int(floor))
    return (n, n)


def cascade_passes(cv2, gray, frontal, profile, *, scale: float = 1.0,
                   floor_px: int = 0):
    """Run _detect_face's four passes on `gray` (already equalised).
    `scale` = gray width / the original frame width; each pass's minSize is
    the full-resolution one times `scale`, and never below `floor_px`.
    At scale 1.0 with floor 0 the calls are exactly the historic ones.
    Returns (boxes, pass_name) in `gray`'s own pixels; boxes may be empty."""
    faces = frontal.detectMultiScale(
        gray, scaleFactor=SCALE_FACTOR, minNeighbors=STRICT["minNeighbors"],
        minSize=_min_size(STRICT["min_px"], scale, floor_px))
    found_by = "frontal"
    if len(faces) == 0:
        faces = frontal.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR,
            minNeighbors=RELAXED["minNeighbors"],
            minSize=_min_size(RELAXED["min_px"], scale, floor_px))
        found_by = "frontal_relaxed"
    if len(faces) == 0 and profile is not None:
        prof = profile.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR,
            minNeighbors=STRICT["minNeighbors"],
            minSize=_min_size(STRICT["min_px"], scale, floor_px))
        if len(prof) > 0:
            faces = prof
            found_by = "profile"
        else:
            gw = gray.shape[1]
            mirror = cv2.flip(gray, 1)
            prof_m = profile.detectMultiScale(
                mirror, scaleFactor=SCALE_FACTOR,
                minNeighbors=STRICT["minNeighbors"],
                minSize=_min_size(STRICT["min_px"], scale, floor_px))
            if len(prof_m) > 0:
                faces = [(gw - x - fw, y, fw, fh) for (x, y, fw, fh) in prof_m]
                found_by = "profile_mirror"
    return faces, found_by


def detect(cv2, frame_bgr, frontal, profile, *, width: int = 0,
           min_frac: float = 0.0):
    """One detection on a BGR frame. width <= 0 (or >= the frame's width)
    = full resolution, today's path. Otherwise the grey frame is shrunk to
    `width` (INTER_AREA) before equalising and the passes run there, with
    no face smaller than `min_frac` of the width searched for.
    Returns (box, pass_name, (w, h)) with box = (x, y, fw, fh) of the
    largest face in ORIGINAL frame pixels, or (None, None, (w, h))."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    floor = 0
    try:
        width = int(width or 0)
    except Exception:
        width = 0
    if 0 < width < w:
        scale = width / float(w)
        gray = cv2.resize(gray, (width, max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
        try:
            floor = int(round(max(0.0, float(min_frac or 0.0)) * width))
        except Exception:
            floor = 0
    gray = cv2.equalizeHist(gray)
    faces, found_by = cascade_passes(cv2, gray, frontal, profile,
                                     scale=scale, floor_px=floor)
    if len(faces) == 0:
        return None, None, (w, h)
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if scale != 1.0:
        x, y, fw, fh = (x / scale, y / scale, fw / scale, fh / scale)
    return (x, y, fw, fh), found_by, (w, h)


def verdicts(box, found_by, size, *, min_frac=None) -> tuple:
    """(any_face, presence_qualified) for one detector result, using the
    presence gate's own rule. Never raises."""
    if box is None:
        return False, False
    try:
        from core import face_presence as _fp
        info = {"pass": found_by, "w_frac": float(box[2]) / float(size[0])}
        return True, bool(_fp.qualifies(info, min_frac=min_frac))
    except Exception:
        try:
            ok = (found_by in _CONFIDENT_FALLBACK
                  and float(box[2]) / float(size[0]) >= 0.07)
        except Exception:
            ok = False
        return True, ok


class ShadowStats:
    """Agreement counters for the shadow mode. Counts and timings only.
    Thread-safe; the face-track producer is the only writer in practice."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = 0
        self.total = self._blank()
        self.window = self._blank()

    @staticmethod
    def _blank() -> dict:
        return {"n": 0, "same": 0, "full_only": 0, "cheap_only": 0,
                "face_same": 0, "full_ms": [], "cheap_ms": [], "errors": 0}

    def due(self, every_n) -> bool:
        """True on 1 in `every_n` calls (the first call included). <= 0 or
        junk = never."""
        try:
            n = int(every_n)
        except Exception:
            return False
        if n <= 0:
            return False
        with self._lock:
            self._calls += 1
            return (self._calls - 1) % n == 0

    def record(self, full: tuple, cheap: tuple, full_ms: float,
               cheap_ms: float) -> None:
        """`full` / `cheap` = (any_face, qualified) from verdicts()."""
        with self._lock:
            for st in (self.total, self.window):
                st["n"] += 1
                if bool(full[1]) == bool(cheap[1]):
                    st["same"] += 1
                elif full[1]:
                    st["full_only"] += 1
                else:
                    st["cheap_only"] += 1
                if bool(full[0]) == bool(cheap[0]):
                    st["face_same"] += 1
                st["full_ms"].append(float(full_ms))
                st["cheap_ms"].append(float(cheap_ms))
                # Keep memory bounded on a long run: medians from the newest.
                for k in ("full_ms", "cheap_ms"):
                    if len(st[k]) > 2000:
                        del st[k][:1000]

    def note_error(self) -> None:
        with self._lock:
            self.total["errors"] += 1
            self.window["errors"] += 1

    def window_size(self) -> int:
        with self._lock:
            return self.window["n"]

    @staticmethod
    def _med(xs) -> float:
        return statistics.median(xs) if xs else 0.0

    def report(self, width, min_frac) -> str:
        """One log line for the samples since the last report, plus the
        running totals; resets the window. Numbers only."""
        with self._lock:
            w, t = self.window, self.total
            line = (
                f"[face-detect shadow] last {w['n']}: presence verdict same "
                f"{w['same']}/{w['n']} (full-only {w['full_only']}, cheap-only "
                f"{w['cheap_only']}), any-face same {w['face_same']}/{w['n']}; "
                f"median full {self._med(w['full_ms']):.1f} ms vs cheap@{width}px "
                f"{self._med(w['cheap_ms']):.1f} ms | total {t['n']}: same "
                f"{t['same']}, full-only {t['full_only']}, cheap-only "
                f"{t['cheap_only']}, median {self._med(t['full_ms']):.1f} vs "
                f"{self._med(t['cheap_ms']):.1f} ms, errors {t['errors']} "
                f"(min face {float(min_frac or 0) * 100:g}% of width)")
            self.window = self._blank()
            return line


def policy_line(mode, width, min_frac, every_n, report_n) -> str:
    """The boot line for the detector path (one line, numbers only)."""
    m = normalise_mode(mode)
    if m == "off":
        return ("[cpu] face-detect path: full resolution only "
                "(FACE_DETECT_FAST_MODE=off)")
    cheap = (f"{int(width)} px wide, faces from {float(min_frac) * 100:g}% "
             f"of the width")
    if m == "on":
        return f"[cpu] face-detect path: CHEAP ({cheap}) decides"
    return (f"[cpu] face-detect path: full resolution decides; shadow runs the "
            f"cheap path ({cheap}) on 1 in {int(every_n)} detections and logs "
            f"agreement every {int(report_n)} samples (counts and timings only)")
