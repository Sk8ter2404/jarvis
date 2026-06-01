"""
Multi-user voice identification (speaker recognition) for JARVIS.

On every utterance, the main loop hands the raw float32 mono mic audio to
`identify_speaker(audio, sample_rate)` which returns the best-match
enrolled-speaker name and a confidence score. The caller can then load
that user's personalised memory namespace, music preferences, and access
permissions (e.g. "only the owner can run sudo commands").

Backend
-------
Resemblyzer's `VoiceEncoder` produces 256-dim speaker embeddings on the
GPU/CPU in well under a second per utterance. We compare a new utterance's
embedding against every enrolled speaker's averaged embedding using cosine
similarity. The best match above CONFIDENCE_THRESHOLD wins; everything
else collapses to UNKNOWN_SPEAKER (returned as None) so callers can decide
how to handle strangers (e.g. refuse sudo, deny memory writes).

Storage
-------
  data/voiceprints/<name>.npy        — averaged 256-dim embedding per speaker
  data/voiceprints/<name>.json       — sidecar metadata (sample_count, ts,
                                       last_seen, permissions)
  data/voiceprints/_index.json       — list of enrolled speakers + active user

Graceful degradation
--------------------
- If `resemblyzer` is not installed, `is_available()` returns False and
  `identify_speaker()` returns (None, 0.0) — the rest of JARVIS continues
  exactly as before (single-user mode).
- If no voiceprints are enrolled, identification short-circuits to the
  same single-user fallback regardless of `resemblyzer` install status.

Public API
----------
  is_available()                                      → bool
  list_enrolled()                                     → list[str]
  enroll_from_audio(name, audio, sample_rate, *,
                    append=True, permissions=None)    → dict result
  identify_speaker(audio, sample_rate)                → (name|None, score)
  get_active_speaker()                                → str|None  (sticky)
  set_active_speaker(name)                            → bool
  forget_speaker(name)                                → bool
  permissions_for(name)                               → dict
  can(name, capability)                               → bool

The module is import-safe — it never imports heavy deps at module load,
only when an action actually needs them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import numpy as np


# ── config ──────────────────────────────────────────────────────────────────

_HERE          = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_HERE)
_VOICE_DIR     = os.path.join(_PROJECT_ROOT, "data", "voiceprints")
_INDEX_FILE    = os.path.join(_VOICE_DIR, "_index.json")

# Resemblyzer ships at 16 kHz mono. Anything else gets resampled in
# `_to_resemblyzer_audio` via librosa if available, otherwise via a cheap
# numpy stride trick (good enough for 24 kHz→16 kHz which is the only
# realistic mismatch since edge-tts uses 24 kHz and the mic uses 16 kHz).
TARGET_SR = 16000

# Cosine-similarity floor. Resemblyzer embeddings for the SAME speaker
# typically sit at 0.78–0.92; DIFFERENT speakers sit at 0.45–0.65. 0.72
# gives a comfortable margin without producing false positives on a quiet
# 1-second utterance (where the embedding is noisier).
CONFIDENCE_THRESHOLD = 0.72

# Minimum number of seconds of audio before we even attempt identification.
# Anything shorter than 0.6 s gives unreliable embeddings.
MIN_IDENTIFY_SECONDS = 0.6

# Default permissions assigned to a brand-new enrollment — conservative.
# The first speaker enrolled is auto-promoted to the "owner" permissions
# set below so the user doesn't have to manually edit JSON before voice
# commands work.
_DEFAULT_PERMISSIONS = {
    "sudo": False,
    "shell": False,
    "memory_write": False,
    "smart_home": True,
    "music": True,
}

_OWNER_PERMISSIONS = {
    "sudo": True,
    "shell": True,
    "memory_write": True,
    "smart_home": True,
    "music": True,
}


# ── module state ────────────────────────────────────────────────────────────

_state_lock = threading.RLock()
_encoder = None                       # lazy-loaded VoiceEncoder
_encoder_error: Optional[str] = None
_voiceprints: dict[str, np.ndarray] = {}      # name → averaged embedding
_voicemeta:   dict[str, dict] = {}            # name → sidecar metadata
_active_speaker: Optional[str] = None         # last identified speaker
_loaded = False


# ── utility ─────────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    try:
        os.makedirs(_VOICE_DIR, exist_ok=True)
    except OSError:
        pass


def _atomic_write_json(path: str, data) -> None:
    """Use core.atomic_io if importable, otherwise inline same-volume rename."""
    try:
        from core import atomic_io
        atomic_io._atomic_write_json(path, data)
        return
    except Exception:
        pass
    import tempfile
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _slug(name: str) -> str:
    """Sanitise a speaker name into a filesystem-safe slug.

    Keeps unicode letters/digits but collapses everything else to '_'.
    Two different inputs that slug to the same string would collide — we
    accept that, because callers should already be passing canonical names.
    """
    out = []
    for ch in (name or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "speaker"


def _to_resemblyzer_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert an arbitrary mono float32 buffer to 16 kHz float32 in [-1, 1]."""
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1) if arr.shape[1] < arr.shape[0] else arr.mean(axis=0)
    # Normalise — resemblyzer expects roughly unit-scale samples. Most JARVIS
    # callers already provide float32 in [-1, 1], but int16-derived buffers
    # arrive with values up to 32k.
    mx = float(np.max(np.abs(arr))) if arr.size else 0.0
    if mx > 1.5:
        arr = arr / 32768.0
    if sample_rate == TARGET_SR or arr.size == 0:
        return arr.astype(np.float32, copy=False)
    # Prefer librosa if available (sinc-interp), else fall back to linear.
    try:
        import librosa
        return librosa.resample(arr.astype(np.float32),
                                orig_sr=sample_rate,
                                target_sr=TARGET_SR).astype(np.float32, copy=False)
    except Exception:
        ratio = TARGET_SR / float(sample_rate)
        n_out = max(1, int(round(arr.size * ratio)))
        x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, arr).astype(np.float32, copy=False)


# ── encoder ─────────────────────────────────────────────────────────────────

def _load_encoder():
    """Lazy-init Resemblyzer's VoiceEncoder. Returns the encoder or None."""
    global _encoder, _encoder_error
    if _encoder is not None:
        return _encoder
    try:
        from resemblyzer import VoiceEncoder
    except Exception as e:
        _encoder_error = (
            f"resemblyzer not installed ({e}); voice ID disabled. "
            "Install with `pip install resemblyzer`."
        )
        return None
    try:
        _encoder = VoiceEncoder(verbose=False)
    except TypeError:
        # Older resemblyzer signatures don't accept `verbose`.
        try:
            _encoder = VoiceEncoder()
        except Exception as e:
            _encoder_error = f"VoiceEncoder init failed: {e}"
            return None
    except Exception as e:
        _encoder_error = f"VoiceEncoder init failed: {e}"
        return None
    return _encoder


def _embed(audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Return a 256-dim float32 unit-norm embedding, or None on failure."""
    enc = _load_encoder()
    if enc is None:
        return None
    wav = _to_resemblyzer_audio(audio, sample_rate)
    if wav.size < int(TARGET_SR * MIN_IDENTIFY_SECONDS):
        return None
    try:
        # Resemblyzer's `embed_utterance` does its own preprocessing
        # (normalisation, VAD trim), but it expects pre-resampled 16 kHz
        # audio. We've already resampled in _to_resemblyzer_audio.
        emb = enc.embed_utterance(wav)
    except Exception as e:
        print(f"  [voice_id] embed failed: {e}")
        return None
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(emb))
    if n > 0:
        emb = emb / n
    return emb


# ── persistence ─────────────────────────────────────────────────────────────

def _load_index() -> dict:
    try:
        with open(_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        return {}


def _save_index() -> None:
    _ensure_dir()
    payload = {
        "speakers": sorted(_voiceprints.keys()),
        "active": _active_speaker,
        "updated_ts": time.time(),
    }
    try:
        _atomic_write_json(_INDEX_FILE, payload)
    except Exception as e:
        print(f"  [voice_id] index save failed: {e}")


def _embedding_path(name: str) -> str:
    return os.path.join(_VOICE_DIR, f"{_slug(name)}.npy")


def _meta_path(name: str) -> str:
    return os.path.join(_VOICE_DIR, f"{_slug(name)}.json")


def _load_all() -> None:
    """Populate _voiceprints / _voicemeta from disk. Idempotent."""
    global _loaded, _active_speaker
    with _state_lock:
        if _loaded:
            return
        _voiceprints.clear()
        _voicemeta.clear()
        _ensure_dir()
        if not os.path.isdir(_VOICE_DIR):
            _loaded = True
            return
        for fname in os.listdir(_VOICE_DIR):
            if not fname.endswith(".npy") or fname.startswith("_"):
                continue
            slug = fname[:-4]
            npy_path = os.path.join(_VOICE_DIR, fname)
            try:
                emb = np.load(npy_path).astype(np.float32, copy=False)
            except Exception as e:
                print(f"  [voice_id] failed to load {fname}: {e}")
                continue
            emb = emb.reshape(-1)
            n = float(np.linalg.norm(emb))
            if n > 0:
                emb = emb / n
            _voiceprints[slug] = emb
            meta_p = os.path.join(_VOICE_DIR, f"{slug}.json")
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    _voicemeta[slug] = json.load(f) or {}
            except (FileNotFoundError, json.JSONDecodeError):
                _voicemeta[slug] = {
                    "name": slug,
                    "sample_count": 1,
                    "enrolled_ts": time.time(),
                    "permissions": dict(_DEFAULT_PERMISSIONS),
                }
            except Exception:
                _voicemeta[slug] = {"name": slug}
        idx = _load_index()
        a = idx.get("active")
        if isinstance(a, str) and a in _voiceprints:
            _active_speaker = a
        _loaded = True


# ── public API ──────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True iff resemblyzer can be imported. Does NOT require enrollments —
    callers fall back to single-user mode when no one is enrolled."""
    return _load_encoder() is not None


def encoder_status() -> dict:
    _load_all()
    enc = _load_encoder()
    return {
        "encoder_loaded": enc is not None,
        "encoder_error": _encoder_error,
        "enrolled": sorted(_voiceprints.keys()),
        "active_speaker": _active_speaker,
        "threshold": CONFIDENCE_THRESHOLD,
    }


def list_enrolled() -> list[str]:
    _load_all()
    with _state_lock:
        return sorted(_voiceprints.keys())


def get_active_speaker() -> Optional[str]:
    _load_all()
    return _active_speaker


def set_active_speaker(name: Optional[str]) -> bool:
    """Manually set the sticky active speaker (used by callers that want to
    pin the session to one user without per-utterance reclassification).
    Pass None to clear."""
    global _active_speaker
    _load_all()
    with _state_lock:
        if name is None:
            _active_speaker = None
            _save_index()
            return True
        slug = _slug(name)
        if slug not in _voiceprints:
            return False
        _active_speaker = slug
        _save_index()
        return True


def forget_speaker(name: str) -> bool:
    """Delete an enrolled voiceprint (and its metadata sidecar)."""
    global _active_speaker
    _load_all()
    slug = _slug(name)
    with _state_lock:
        if slug not in _voiceprints:
            return False
        for p in (_embedding_path(slug), _meta_path(slug)):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"  [voice_id] could not delete {p}: {e}")
        _voiceprints.pop(slug, None)
        _voicemeta.pop(slug, None)
        if _active_speaker == slug:
            _active_speaker = None
        _save_index()
        return True


def enroll_from_audio(
    name: str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    append: bool = True,
    permissions: Optional[dict] = None,
) -> dict:
    """Enroll (or extend) a speaker's voiceprint from a raw audio buffer.

    `append=True` averages the new embedding into any existing voiceprint
    for that name, weighted by sample_count. This lets the user say
    "JARVIS, learn my voice" multiple times to harden the print without
    overwriting it.

    Returns a dict: {ok, name, sample_count, dim, error?}.
    """
    _load_all()
    slug = _slug(name)
    if not slug:
        return {"ok": False, "error": "empty name"}
    if not is_available():
        return {"ok": False, "error": _encoder_error or "resemblyzer not available"}
    emb = _embed(audio, sample_rate)
    if emb is None:
        return {"ok": False, "error":
                f"could not compute embedding — need at least "
                f"{MIN_IDENTIFY_SECONDS:.1f}s of clear speech"}

    with _state_lock:
        existing = _voiceprints.get(slug) if append else None
        # Always read existing meta from the sidecar so a "re-enroll from
        # scratch" (append=False) preserves enrolled_ts, permissions, and
        # the user's preferred display name. Only the embedding and
        # sample_count get reset; updated_ts is refreshed below.
        meta = _voicemeta.get(slug, {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        if existing is not None:
            prev_n = int(meta.get("sample_count", 1) or 1)
            blended = existing * prev_n + emb
            blended = blended / float(prev_n + 1)
            n = float(np.linalg.norm(blended))
            if n > 0:
                blended = blended / n
            new_emb = blended.astype(np.float32, copy=False)
            sample_count = prev_n + 1
        else:
            new_emb = emb.astype(np.float32, copy=False)
            sample_count = 1

        # First enrollment in the whole system gets owner permissions.
        is_first_speaker = (len(_voiceprints) == 0) and (slug not in _voiceprints)
        if permissions is None:
            permissions = dict(meta.get("permissions") or {}) if meta else {}
            if not permissions:
                permissions = dict(_OWNER_PERMISSIONS if is_first_speaker
                                   else _DEFAULT_PERMISSIONS)

        _voiceprints[slug] = new_emb
        _voicemeta[slug] = {
            "name": name.strip() or slug,
            "slug": slug,
            "sample_count": sample_count,
            "enrolled_ts": float(meta.get("enrolled_ts") or time.time()),
            "updated_ts": time.time(),
            "permissions": permissions,
        }

        _ensure_dir()
        try:
            np.save(_embedding_path(slug), new_emb)
        except Exception as e:
            return {"ok": False, "error": f"could not save embedding: {e}"}
        try:
            _atomic_write_json(_meta_path(slug), _voicemeta[slug])
        except Exception as e:
            print(f"  [voice_id] meta save failed: {e}")
        _save_index()

    return {
        "ok": True,
        "name": _voicemeta[slug]["name"],
        "sample_count": sample_count,
        "dim": int(new_emb.shape[0]),
    }


def identify_speaker(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[Optional[str], float]:
    """Return (display_name, score) for the best-matching enrolled speaker.

    Returns (None, score) if:
      - no speakers are enrolled (single-user fallback)
      - resemblyzer is missing
      - the audio is too short for a reliable embedding
      - no enrolled speaker exceeds CONFIDENCE_THRESHOLD

    The score is the cosine similarity of the best match (0 if no match
    could be computed) so callers can log close-calls / debug threshold
    tuning.
    """
    global _active_speaker
    _load_all()
    with _state_lock:
        if not _voiceprints:
            return None, 0.0
    if not is_available():
        return None, 0.0

    emb = _embed(audio, sample_rate)
    if emb is None:
        return None, 0.0

    best_slug: Optional[str] = None
    best_score = -1.0
    with _state_lock:
        for slug, ref in _voiceprints.items():
            score = float(np.dot(emb, ref))
            if score > best_score:
                best_score = score
                best_slug = slug

    if best_slug is None or best_score < CONFIDENCE_THRESHOLD:
        return None, max(best_score, 0.0)

    with _state_lock:
        _active_speaker = best_slug
        meta = _voicemeta.get(best_slug, {})
        meta["last_seen_ts"] = time.time()
        meta["last_seen_score"] = best_score
        _voicemeta[best_slug] = meta
        display = meta.get("name") or best_slug
        meta_copy = dict(meta)
        path_copy = _meta_path(best_slug)
    # Persist OUTSIDE the lock. This runs on the per-utterance hot path, so a
    # slow/locked filesystem here would stall every other voice_id caller
    # (list_enrolled / permissions_for / can / enroll) that also takes
    # _state_lock. The in-memory _voicemeta is already updated under the lock;
    # the disk sidecar is just best-effort persistence. 2026-05-30 audit.
    try:
        _atomic_write_json(path_copy, meta_copy)
    except Exception:
        pass
    return display, best_score


def permissions_for(name: Optional[str]) -> dict:
    """Return the permissions dict for `name`, or owner-level for an
    unrecognised user when no speakers are enrolled (single-user mode).
    Falls back to the locked-down default permissions when speakers ARE
    enrolled but the caller is unknown — that keeps a stranger's voice
    from being able to run sudo commands once enrollments exist."""
    _load_all()
    if name is None:
        with _state_lock:
            if not _voiceprints:
                return dict(_OWNER_PERMISSIONS)
            return dict(_DEFAULT_PERMISSIONS)
    slug = _slug(name)
    with _state_lock:
        meta = _voicemeta.get(slug)
        if meta and isinstance(meta.get("permissions"), dict):
            return dict(meta["permissions"])
        # Enrolled but no permissions block → conservative default.
        if slug in _voiceprints:
            return dict(_DEFAULT_PERMISSIONS)
    return dict(_DEFAULT_PERMISSIONS)


def can(name: Optional[str], capability: str) -> bool:
    perms = permissions_for(name)
    return bool(perms.get(capability, False))


def grant(name: str, capability: str, value: bool = True) -> bool:
    """Set a permission flag on an enrolled speaker. Returns True on success."""
    _load_all()
    slug = _slug(name)
    with _state_lock:
        if slug not in _voiceprints:
            return False
        meta = _voicemeta.get(slug) or {}
        perms = dict(meta.get("permissions") or _DEFAULT_PERMISSIONS)
        perms[capability] = bool(value)
        meta["permissions"] = perms
        meta["updated_ts"] = time.time()
        _voicemeta[slug] = meta
        try:
            _atomic_write_json(_meta_path(slug), meta)
        except Exception as e:
            print(f"  [voice_id] grant save failed: {e}")
            return False
    return True


# ── single-user fallback helper ─────────────────────────────────────────────

def memory_namespace_for(name: Optional[str]) -> str:
    """Return the per-user memory namespace tag. Single-user mode (no
    enrollments) always returns 'default' so existing memory files stay
    untouched until the user opts in by enrolling a voice."""
    _load_all()
    with _state_lock:
        if not _voiceprints:
            return "default"
    if not name:
        return "guest"
    return _slug(name)
