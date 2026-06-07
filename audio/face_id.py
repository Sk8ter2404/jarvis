"""
face_id — lazy, graceful OpenCV face recognition engine for JARVIS.

WHY THIS MODULE EXISTS
======================
JARVIS already knows HOW MANY bodies are in the room (the Kinect skeleton
tracker, audio/kinect_bridge.py) and WHICH MONITOR you're looking at (the two
monitor webcams). What it could not do is say WHO is at the desk. This module
adds identity: it detects faces in a webcam frame, embeds each into a 128-float
vector, and matches that against a small enrolled gallery — so "who's at the
desk" can answer "that's you, sir" or "someone I don't recognise".

It uses OpenCV's BUILT-IN face modules — YuNet (detector) + SFace (recognizer),
both small ONNX models — so there is NO new pip dependency (cv2 is already
installed) and NO dlib. The two ONNX files download once to a gitignored
data/models/ folder on first use.

DESIGN — mirrors audio/kinect_bridge.py + core/voice_id.py
==========================================================
  * LAZY + GRACEFUL: nothing heavy is imported at module load. cv2 / numpy /
    urllib all import inside functions, after gates. Every public function
    returns a graceful sentinel and NEVER raises into the caller — a missing
    model, an absent cv2, a download failure, or a frame with no face all
    degrade to "[], 0, None, (False, reason)" rather than crashing the voice
    loop.
  * SINGLETONS behind a lock: the YuNet detector and SFace recognizer are each
    opened once and cached. The detector's input size is reset per frame
    (det.setInputSize) because YuNet needs the exact (W, H).
  * PATCHABLE PATHS: MODELS_DIR / ENROLL_PATH / the two model URLs are
    module-level constants so tests point them at a tmp dir and NEVER touch the
    real data/. The cosine math has a pure-numpy fallback so the
    detect→embed→match logic is unit-testable WITHOUT a live cv2 recognizer.

PRIVACY
=======
Face embeddings are BIOMETRIC PII. They live ONLY in data/face_enroll.json,
which is gitignored (the data/* glob) and must NEVER be committed or shipped.
The recognizer runs entirely on-device; no frame or embedding leaves the
machine. Enrollment is always an explicit user action ("learn my face").

THE PIPELINE (the exact OpenCV usage, validated live on cv2 4.13.0)
===================================================================
  det = cv2.FaceDetectorYN_create(yunet_path, "", (W, H)); det.setInputSize((W,H))
  ok, faces = det.detect(bgr)        # faces: Nx15 float (bbox + 5 landmarks +
                                     #        score) or None
  rec = cv2.FaceRecognizerSF_create(sface_path, "")
  aligned = rec.alignCrop(bgr, faces[i])     # 112x112 aligned face
  feat    = rec.feature(aligned)             # 1x128 float embedding
  cos     = rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)  # higher == closer;
                                             # same person if cos >= ~0.363
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


# ─── patchable paths + model coordinates ───────────────────────────────────
# All module-level so tests can repoint them at a tmp dir. NEVER let a test
# touch the real data/ — patch MODELS_DIR + ENROLL_PATH first.

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
MODELS_DIR    = os.path.join(_PROJECT_ROOT, "data", "models")
ENROLL_PATH   = os.path.join(_PROJECT_ROOT, "data", "face_enroll.json")

# YuNet face DETECTOR (~232 KB) and SFace face RECOGNIZER (~38 MB). Pinned to
# the exact OpenCV-Zoo model files validated for this cv2 build.
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/" + YUNET_FILENAME)
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/" + SFACE_FILENAME)

# Network: one-shot download with a generous timeout. Kept modest so a stalled
# fetch fails to a graceful "(False, reason)" rather than hanging the action.
_DOWNLOAD_TIMEOUT_SEC = 60.0
# A downloaded ONNX must be at least this big to be considered real (guards
# against a saved HTML error page / truncated file masquerading as the model).
_MIN_MODEL_BYTES = 50 * 1024            # 50 KB — YuNet is ~232 KB, SFace ~38 MB


# ─── singletons (cached behind a lock) ─────────────────────────────────────
_lock = threading.RLock()
_detector: list[Any] = [None]          # cv2.FaceDetectorYN
_recognizer: list[Any] = [None]        # cv2.FaceRecognizerSF
_logged_fetch = [False]                # one-line "downloading models" log guard


def _yunet_path() -> str:
    return os.path.join(MODELS_DIR, YUNET_FILENAME)


def _sface_path() -> str:
    return os.path.join(MODELS_DIR, SFACE_FILENAME)


# ─── model readiness (download-once, never raise) ──────────────────────────

def _file_ok(path: str) -> bool:
    """True if `path` exists and is at least _MIN_MODEL_BYTES (so a truncated /
    error-page download isn't mistaken for a real model)."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= _MIN_MODEL_BYTES
    except OSError:
        return False


def _download(url: str, dest: str) -> tuple[bool, str]:
    """Fetch `url` to `dest` atomically (temp + os.replace), validating the
    size. Returns (True, "") or (False, reason). NEVER raises."""
    try:
        import urllib.request
    except Exception as e:   # pragma: no cover - urllib is stdlib; defensive
        return False, f"urllib unavailable: {type(e).__name__}"
    tmp = dest + ".part"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS-face-id"})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SEC) as r:
            data = r.read()
        if not data or len(data) < _MIN_MODEL_BYTES:
            return False, (f"downloaded file too small "
                           f"({len(data) if data else 0} bytes)")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        return True, ""
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False, f"{type(e).__name__}: {e}"


def _models_ready() -> tuple[bool, str]:
    """Ensure both ONNX model files exist (downloading on first use). Returns
    (True, "") when both are present, else (False, reason). NEVER raises.

    Cached models skip the network entirely. A failed download leaves the
    engine OFF (no exception) so a bad connection never breaks the voice loop.
    """
    want = [(_yunet_path(), YUNET_URL, "YuNet detector"),
            (_sface_path(), SFACE_URL, "SFace recognizer")]
    missing = [(p, u, n) for (p, u, n) in want if not _file_ok(p)]
    if not missing:
        return True, ""
    if not _logged_fetch[0]:
        _logged_fetch[0] = True
        names = ", ".join(n for (_p, _u, n) in missing)
        print(f"  [face-id] fetching face models on first use ({names}) — "
              f"one-time download to data/models/ …")
    for path, url, name in missing:
        ok, reason = _download(url, path)
        if not ok:
            return False, f"could not download {name}: {reason}"
    # Re-verify (a 200 that wrote something too small is caught by _file_ok).
    if all(_file_ok(p) for (p, _u, _n) in want):
        return True, ""
    return False, "models downloaded but failed validation"


# ─── cv2 detector / recognizer singletons ──────────────────────────────────

def _get_detector():
    """Cached cv2.FaceDetectorYN, or None if cv2/model unavailable. The input
    size is a placeholder here — detect() resets it per frame. NEVER raises."""
    with _lock:
        if _detector[0] is not None:
            return _detector[0]
        ok, _reason = _models_ready()
        if not ok:
            return None
        try:
            import cv2
            det = cv2.FaceDetectorYN_create(_yunet_path(), "", (320, 320))
        except Exception:
            return None
        _detector[0] = det
        return det


def _get_recognizer():
    """Cached cv2.FaceRecognizerSF, or None if cv2/model unavailable. NEVER
    raises."""
    with _lock:
        if _recognizer[0] is not None:
            return _recognizer[0]
        ok, _reason = _models_ready()
        if not ok:
            return None
        try:
            import cv2
            rec = cv2.FaceRecognizerSF_create(_sface_path(), "")
        except Exception:
            return None
        _recognizer[0] = rec
        return rec


def is_available() -> tuple[bool, str]:
    """(True, "") when cv2 imports AND both models are ready; else
    (False, reason). The reason is human-readable for the status action.
    NEVER raises."""
    try:
        import importlib.util
        if importlib.util.find_spec("cv2") is None:
            return False, "OpenCV (cv2) not installed"
    except Exception as e:
        return False, f"OpenCV (cv2) not available: {type(e).__name__}"
    ok, reason = _models_ready()
    if not ok:
        return False, reason
    return True, ""


# ─── detection ─────────────────────────────────────────────────────────────

def detect_faces(bgr) -> list:
    """Detect faces in a BGR ndarray with YuNet. Returns a list of face rows
    (each a length-15 sequence: bbox x,y,w,h + 5 landmarks + score), or [] if
    none / unavailable. NEVER raises."""
    if bgr is None:
        return []
    det = _get_detector()
    if det is None:
        return []
    try:
        h, w = bgr.shape[:2]
        det.setInputSize((int(w), int(h)))
        ok, faces = det.detect(bgr)
        if not ok or faces is None:
            return []
        return [row for row in faces]
    except Exception:   # pragma: no cover - defensive: mid-detect cv2 glitch
        return []


def _bbox_of(face_row) -> list:
    """Integer [x, y, w, h] from a YuNet face row's first four values."""
    try:
        return [int(round(float(face_row[0]))), int(round(float(face_row[1]))),
                int(round(float(face_row[2]))), int(round(float(face_row[3])))]
    except Exception:   # pragma: no cover - defensive
        return [0, 0, 0, 0]


def _face_area(face_row) -> float:
    try:
        return float(face_row[2]) * float(face_row[3])
    except Exception:   # pragma: no cover - defensive
        return 0.0


def _largest_face(faces: list):
    """The biggest face row (by bbox area) — the one nearest the camera, which
    is who we enroll/identify. None for an empty list."""
    if not faces:
        return None
    return max(faces, key=_face_area)


# ─── embedding ─────────────────────────────────────────────────────────────

def embed(bgr, face_row):
    """alignCrop + feature → a 128-float embedding (numpy 1-D array) for one
    detected face, or None if the recognizer is unavailable. NEVER raises."""
    if bgr is None or face_row is None:
        return None
    rec = _get_recognizer()
    if rec is None:
        return None
    try:
        import numpy as np
        aligned = rec.alignCrop(bgr, _as_face_array(face_row))
        feat = rec.feature(aligned)
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
        return arr if arr.size else None
    except Exception:   # pragma: no cover - defensive: mid-embed cv2 glitch
        return None


def _as_face_array(face_row):
    """cv2.alignCrop wants the face row as a (1, 15) float32 ndarray. Accept a
    list/tuple/ndarray and coerce."""
    import numpy as np
    arr = np.asarray(face_row, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ─── similarity (cv2.match, with a pure-numpy fallback for testability) ─────

def cosine(a, b) -> float:
    """Cosine similarity between two embeddings, HIGHER == more similar (the
    SFace FR_COSINE convention). Prefers the live recognizer's own match() so
    the score matches OpenCV's exactly; falls back to a numpy cosine so the
    match math is unit-testable without a live recognizer. NEVER raises —
    returns -1.0 (maximally dissimilar) on any failure."""
    if a is None or b is None:
        return -1.0
    rec = _recognizer[0]
    if rec is not None:
        try:
            import cv2
            import numpy as np
            fa = np.asarray(a, dtype=np.float32).reshape(1, -1)
            fb = np.asarray(b, dtype=np.float32).reshape(1, -1)
            return float(rec.match(fa, fb, cv2.FaceRecognizerSF_FR_COSINE))
        except Exception:   # pragma: no cover - fall through to numpy
            pass
    return _numpy_cosine(a, b)


def _numpy_cosine(a, b) -> float:
    """Pure cosine similarity, no cv2. -1.0 on a zero vector or any failure."""
    try:
        import numpy as np
        va = np.asarray(a, dtype=np.float64).reshape(-1)
        vb = np.asarray(b, dtype=np.float64).reshape(-1)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na <= 0.0 or nb <= 0.0:
            return -1.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:   # pragma: no cover - defensive
        return -1.0


def _match_threshold() -> float:
    """Live FACE_ID_MATCH_THRESHOLD from core.config (re-read each call so a
    Settings change applies without a restart). Falls back to SFace's 0.363."""
    try:
        from core import config as _cfg
        return float(getattr(_cfg, "FACE_ID_MATCH_THRESHOLD", 0.363))
    except Exception:
        return 0.363


# ─── enrollment store (data/face_enroll.json — biometric PII, gitignored) ──

def _load_store() -> dict:
    """Read the enrollment JSON, or a fresh empty store. NEVER raises."""
    try:
        with open(ENROLL_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("people"), list):
            return data
    except FileNotFoundError:
        pass
    except Exception:
        # Corrupt store — start clean rather than crash; the user can re-enroll.
        pass
    return {"people": []}


def _save_store(store: dict) -> bool:
    """Atomically persist the enrollment store. Prefers core.atomic_io; falls
    back to a temp+replace inline. Returns True on success. NEVER raises."""
    try:
        os.makedirs(os.path.dirname(ENROLL_PATH), exist_ok=True)
    except Exception:
        return False
    try:
        from core.atomic_io import _atomic_write_json
        _atomic_write_json(ENROLL_PATH, store, indent=2)
        return True
    except Exception:
        pass
    # Inline fallback (no core dependency).
    import tempfile
    dir_ = os.path.dirname(os.path.abspath(ENROLL_PATH)) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
        os.replace(tmp, ENROLL_PATH)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False


def _person_entry(store: dict, name: str) -> Optional[dict]:
    key = (name or "").strip().lower()
    for p in store.get("people", []):
        if str(p.get("name", "")).strip().lower() == key:
            return p
    return None


def list_enrolled() -> list:
    """[{"name": str, "count": int}] — every enrolled person and how many
    embeddings each has. NEVER raises."""
    store = _load_store()
    out = []
    for p in store.get("people", []):
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        embs = p.get("embeddings") or []
        out.append({"name": name, "count": len(embs)})
    return out


def forget(name: str) -> bool:
    """Delete one person's enrollment. True if a person was removed. NEVER
    raises."""
    key = (name or "").strip().lower()
    if not key:
        return False
    store = _load_store()
    people = store.get("people", [])
    kept = [p for p in people if str(p.get("name", "")).strip().lower() != key]
    if len(kept) == len(people):
        return False
    store["people"] = kept
    return _save_store(store)


def enroll(name: str, frames) -> int:
    """Enroll `name` from an iterable of BGR frames. For EACH frame: detect the
    LARGEST face, embed it, and append the embedding to this person's gallery.
    Returns the number of good captures added (>= 1 means success). 0 means no
    usable face was found in any frame / the engine is unavailable. NEVER
    raises.

    Embeddings ACCUMULATE across calls (append=True semantics) so a user can
    add more angles over time. The store is biometric PII — see module docs."""
    name = (name or "").strip()
    if not name or frames is None:
        return 0
    captured: list[list] = []
    for fr in frames:
        if fr is None:
            continue
        faces = detect_faces(fr)
        face = _largest_face(faces)
        if face is None:
            continue
        feat = embed(fr, face)
        if feat is None:
            continue
        try:
            captured.append([float(x) for x in feat])
        except Exception:   # pragma: no cover - defensive
            continue
    if not captured:
        return 0
    store = _load_store()
    entry = _person_entry(store, name)
    if entry is None:
        entry = {"name": name, "embeddings": [], "ts": time.time()}
        store.setdefault("people", []).append(entry)
    entry.setdefault("embeddings", [])
    entry["embeddings"].extend(captured)
    entry["ts"] = time.time()
    if not _save_store(store):
        return 0
    return len(captured)


def _is_unknown_name(name) -> bool:
    """True if a recognize() result names nobody (unknown / blank / None)."""
    return str(name or "").strip().lower() in ("", "unknown")


def _largest_unknown_face(faces: list, results: list):
    """The biggest face row whose recognise() verdict is UNKNOWN — i.e. the
    nearest person we do NOT already know. `results` is recognize()'s output for
    the SAME frame, in the SAME order as `faces` (both iterate detect_faces() in
    order). A face with no paired result is treated as unknown (recognition fell
    short, so it's certainly not someone we recognise). None when every visible
    face is already someone we know (or there are no faces)."""
    best = None
    best_area = -1.0
    for i, face in enumerate(faces):
        name = results[i].get("name") if i < len(results) else None
        if not _is_unknown_name(name):
            continue   # skip a recognised person (owner or a known guest)
        area = _face_area(face)
        if area > best_area:
            best_area = area
            best = face
    return best


def enroll_unknown(name: str, frames) -> dict:
    """Enroll `name` from an iterable of BGR frames, capturing only the largest
    UNKNOWN face per frame — the nearest person we do NOT already recognise. This
    is the guest path: it must NOT grab the owner (or any already-enrolled
    person) just because they happen to be closest to the camera. Recognised
    faces are skipped; among the rest the biggest is enrolled.

    Returns {"added": int, "saw_face": bool, "saw_unknown": bool}:
      * added       — good captures appended (>= 1 means success),
      * saw_face    — at least one face was detected in some frame,
      * saw_unknown — at least one UNKNOWN face was seen (vs. all recognised).
    The caller uses saw_face/saw_unknown to tell "everyone here is already known"
    apart from "I couldn't see a face". NEVER raises. Embeddings ACCUMULATE
    across calls, like enroll()."""
    name = (name or "").strip()
    if not name or frames is None:
        return {"added": 0, "saw_face": False, "saw_unknown": False}
    captured: list[list] = []
    saw_face = False
    saw_unknown = False
    for fr in frames:
        if fr is None:
            continue
        faces = detect_faces(fr)
        if faces:
            saw_face = True
        results = recognize(fr)
        face = _largest_unknown_face(faces, results)
        if face is None:
            continue
        saw_unknown = True
        feat = embed(fr, face)
        if feat is None:
            continue
        try:
            captured.append([float(x) for x in feat])
        except Exception:   # pragma: no cover - defensive
            continue
    if not captured:
        return {"added": 0, "saw_face": saw_face, "saw_unknown": saw_unknown}
    store = _load_store()
    entry = _person_entry(store, name)
    if entry is None:
        entry = {"name": name, "embeddings": [], "ts": time.time()}
        store.setdefault("people", []).append(entry)
    entry.setdefault("embeddings", [])
    entry["embeddings"].extend(captured)
    entry["ts"] = time.time()
    if not _save_store(store):
        return {"added": 0, "saw_face": saw_face, "saw_unknown": saw_unknown}
    return {"added": len(captured), "saw_face": saw_face,
            "saw_unknown": saw_unknown}


# ─── recognition ───────────────────────────────────────────────────────────

def recognize(bgr) -> list:
    """Recognise every detected face in a BGR frame. Returns a list of:
        {"name": str|"unknown", "score": float, "bbox": [x, y, w, h]}
    For each detected face we embed it, take the BEST cosine against ALL of
    each enrolled person's embeddings, and name them if that best >= the match
    threshold (else "unknown"). Empty list if no face / engine unavailable.
    NEVER raises."""
    faces = detect_faces(bgr)
    if not faces:
        return []
    store = _load_store()
    people = [p for p in store.get("people", [])
              if str(p.get("name", "")).strip() and (p.get("embeddings"))]
    thr = _match_threshold()
    out = []
    for face in faces:
        feat = embed(bgr, face)
        bbox = _bbox_of(face)
        if feat is None:
            out.append({"name": "unknown", "score": -1.0, "bbox": bbox})
            continue
        best_name = "unknown"
        best_score = -1.0
        for p in people:
            pname = str(p.get("name", "")).strip()
            for emb in (p.get("embeddings") or []):
                score = cosine(feat, emb)
                if score > best_score:
                    best_score = score
                    best_name = pname
        if best_score >= thr:
            out.append({"name": best_name, "score": round(best_score, 4),
                        "bbox": bbox})
        else:
            out.append({"name": "unknown", "score": round(best_score, 4),
                        "bbox": bbox})
    return out
