"""
Local voice cloning skill for JARVIS — Coqui XTTS-v2.

Replaces (or runs alongside) the edge-tts backend with a locally-cloned voice
rendered on the RTX 3090. Once the user records a ~10-second sample WAV,
XTTS-v2 clones that voice and synthesises every reply in it.

Public API (used by bobert_companion's synthesise() and the voice-toggle
pre-router):

  is_available()                → True if XTTS deps + a usable voice sample
                                  exist on disk. Cheap; safe to call every
                                  utterance.

  render(text, rate, pitch)     → (audio: np.ndarray float32 mono, sr: int)
                                  Loads the model on first call and caches it.
                                  Raises on hard failure so the caller can
                                  fall back to edge-tts. `rate` ("+0%") and
                                  `pitch` ("+0Hz") are accepted for parity
                                  with edge-tts and applied post-hoc via a
                                  best-effort librosa time-stretch / pitch-
                                  shift when those libraries are present —
                                  silently ignored otherwise so prosody-
                                  preset misses are still better than no
                                  audio.

  set_backend(name)             → tries to switch bobert_companion.TTS_BACKEND
                                  to any name in _VALID_BACKENDS ('edge' |
                                  'pyttsx3' | 'xtts' | 'kokoro'). Returns
                                  a spoken confirmation string. If 'xtts' is
                                  requested but the XTTS deps / sample aren't
                                  present, the backend is left on its previous
                                  value and a diagnostic string is returned.

  maybe_switch_backend(utt)     → voice pre-router: matches phrases like
                                  'use my voice' / 'switch to edge voice' /
                                  'back to the default voice' and dispatches
                                  set_backend(). Returns the spoken reply or
                                  None if the utterance didn't match.

Actions registered (so the LLM can also reach the toggle):

  set_tts_backend <name>        — switch backend at runtime
  list_tts_backends             — report current + available backends
  enroll_xtts_sample <path>     — point XTTS at a new sample WAV without
                                  restarting JARVIS

Config (lives on bobert_companion, override via env on boot):
  TTS_BACKEND            "edge" | "pyttsx3" | "xtts" | "kokoro"   (default "edge")
  XTTS_VOICE_SAMPLE      absolute path to a ~10 s WAV (24 kHz mono is best)
  XTTS_LANGUAGE          ISO-639-1 hint, default "en"

Latency note: cold-start model load is several seconds on first call. After
that XTTS-v2 finishes a one-sentence reply in ~700 ms on the 3090. To hit
the <400 ms first-syllable target promised on the wish-list the caller
would need to use the streaming `inference_stream()` API — see the
`render_stream()` helper at the bottom of this file for an opt-in path
the playback layer can switch to later. The default `render()` is
non-streaming for compatibility with the existing synchronous synthesise()
in bobert_companion.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
#  CONFIG ACCESSORS
#
#  bobert_companion is the source of truth for both TTS_BACKEND and the
#  XTTS_VOICE_SAMPLE path so the user can flip them at runtime via the
#  skill bridge and have the next utterance pick up the change. Env vars
#  are honoured at first read so headless deployments can set the path
#  without editing source.
# ──────────────────────────────────────────────────────────────────────────

_VALID_BACKENDS = ("edge", "pyttsx3", "xtts", "kokoro")


def _bobert():
    """Return the bobert_companion module, or None if it isn't imported yet
    (e.g. running this file as `python -m skills.custom_voice`)."""
    return sys.modules.get("bobert_companion") or sys.modules.get("__main__")


def get_backend() -> str:
    bc = _bobert()
    if bc is not None:
        val = getattr(bc, "TTS_BACKEND", None)
        if isinstance(val, str) and val.lower() in _VALID_BACKENDS:
            return val.lower()
    env = os.environ.get("TTS_BACKEND", "").strip().lower()
    if env in _VALID_BACKENDS:
        return env
    return "edge"


def get_sample_path() -> str:
    bc = _bobert()
    if bc is not None:
        p = getattr(bc, "XTTS_VOICE_SAMPLE", "") or ""
        if p:
            return os.path.abspath(p)
    p = os.environ.get("XTTS_VOICE_SAMPLE", "").strip()
    return os.path.abspath(p) if p else ""


def get_language() -> str:
    bc = _bobert()
    if bc is not None:
        lang = getattr(bc, "XTTS_LANGUAGE", "") or ""
        if lang:
            return lang
    return os.environ.get("XTTS_LANGUAGE", "").strip() or "en"


def _set_backend_on_bobert(name: str) -> None:
    bc = _bobert()
    if bc is not None:
        setattr(bc, "TTS_BACKEND", name)


# ──────────────────────────────────────────────────────────────────────────
#  AVAILABILITY CHECK
# ──────────────────────────────────────────────────────────────────────────

_HAS_TTS_LIB: Optional[bool] = None
_TTS_IMPORT_ERROR: Optional[str] = None


def _probe_tts_lib() -> bool:
    """One-shot import probe — cached so we don't spend ~200 ms per turn
    re-importing `TTS` just to check whether XTTS is installed."""
    global _HAS_TTS_LIB, _TTS_IMPORT_ERROR
    if _HAS_TTS_LIB is not None:
        return _HAS_TTS_LIB
    try:
        import TTS  # noqa: F401 — only here to confirm the package resolves
        _HAS_TTS_LIB = True
    except Exception as e:
        _HAS_TTS_LIB = False
        _TTS_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    return _HAS_TTS_LIB


def is_available() -> bool:
    """True iff the Coqui TTS library imports cleanly AND a voice sample
    file exists at XTTS_VOICE_SAMPLE. Both are prerequisites to ever
    returning audio."""
    if not _probe_tts_lib():
        return False
    sample = get_sample_path()
    return bool(sample) and os.path.isfile(sample)


def availability_reason() -> str:
    """Human-readable reason XTTS isn't usable right now. Empty string when
    it IS usable. Used by the spoken error path so the user gets a useful
    hint rather than a silent fallback."""
    if not _probe_tts_lib():
        hint = f" ({_TTS_IMPORT_ERROR})" if _TTS_IMPORT_ERROR else ""
        return ("Coqui TTS isn't installed, sir — "
                f"pip install TTS to enable voice cloning{hint}")
    sample = get_sample_path()
    if not sample:
        return ("No voice sample, sir — set XTTS_VOICE_SAMPLE to a ~10-second "
                "WAV of the voice you'd like me to clone, then try again.")
    if not os.path.isfile(sample):
        return (f"I can't find the voice sample at '{sample}', sir. "
                f"Check the XTTS_VOICE_SAMPLE path.")
    return ""


# ──────────────────────────────────────────────────────────────────────────
#  MODEL CACHE
#
#  XTTS-v2 takes several seconds to load and pins ~3 GB of VRAM. Cache the
#  loaded instance on the module so subsequent renders reuse it. A lock
#  protects against two TTS calls racing to construct the model when the
#  first utterance after boot fans out — without it the second caller would
#  see an half-initialised cache and crash.
# ──────────────────────────────────────────────────────────────────────────

_xtts_model = None        # cached coqui TTS instance
_xtts_model_lock = threading.Lock()
_xtts_load_error: Optional[str] = None
_xtts_loaded_sample: str = ""   # remember which sample we trained the cache on


def _load_xtts_model():
    """Construct the Coqui TTS XTTS-v2 wrapper and stash it on the module.

    Returns the loaded model instance, or raises with a useful message on
    failure. Re-uses an existing cache when present and the requested
    sample hasn't changed."""
    global _xtts_model, _xtts_load_error, _xtts_loaded_sample
    sample = get_sample_path()
    # If we already have a model AND the sample hasn't changed, keep it.
    if _xtts_model is not None and _xtts_loaded_sample == sample:
        return _xtts_model
    with _xtts_model_lock:
        if _xtts_model is not None and _xtts_loaded_sample == sample:
            return _xtts_model
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore[import-not-found]
        except Exception as e:
            _xtts_load_error = f"TTS import failed: {type(e).__name__}: {e}"
            raise RuntimeError(_xtts_load_error) from e

        # Prefer CUDA when available; the wish-list explicitly targets the
        # 3090. Coqui's `gpu=True` flag does the device move internally.
        gpu = False
        try:
            import torch  # type: ignore[import-not-found]
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False

        try:
            model = CoquiTTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=gpu,
            )
        except Exception as e:
            # A CUDA-OOM here can leave a half-allocated model pinned in
            # VRAM; empty the cache + null the cached handle so retries (and
            # the edge-tts fallback) don't keep re-OOMing on dead state.
            if gpu and _is_cuda_oom(e):
                _drop_gpu_model_cache()
            _xtts_load_error = (f"XTTS-v2 load failed: {type(e).__name__}: {e}. "
                                f"First-run downloads ~2 GB of model files; "
                                f"check disk space + network.")
            raise RuntimeError(_xtts_load_error) from e

        _xtts_model = model
        _xtts_loaded_sample = sample
        _xtts_load_error = None
        print(f"  [xtts] model loaded (gpu={gpu}); cloning voice from {sample}")
        return _xtts_model


def _invalidate_model_cache() -> None:
    """Drop the cached XTTS model — used after enroll_xtts_sample() so the
    next render picks up the new speaker WAV."""
    global _xtts_model, _xtts_loaded_sample
    with _xtts_model_lock:
        _xtts_model = None
        _xtts_loaded_sample = ""


def _is_cuda_oom(exc: Exception) -> bool:
    """True when an exception looks like a CUDA out-of-memory error. The
    3090 is shared with whisper + qwen, so XTTS can lose the VRAM race; we
    detect that by message rather than by type so it works even when torch
    isn't importable here."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in msg or "cuda" in msg


def _drop_gpu_model_cache() -> None:
    """On a CUDA-OOM, free the half-allocated VRAM and null the cached
    model so the next attempt (or the edge-tts fallback) starts clean
    instead of re-OOMing on the same dead cache. Best-effort: a missing
    torch must never break the fallback path."""
    global _xtts_model, _xtts_loaded_sample
    _xtts_model = None
    _xtts_loaded_sample = ""
    try:
        import torch  # type: ignore[import-not-found]
        torch.cuda.empty_cache()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  PARSE EDGE-TTS-STYLE RATE / PITCH STRINGS
# ──────────────────────────────────────────────────────────────────────────

_RATE_RE = re.compile(r"^\s*([+\-]?\d+(?:\.\d+)?)\s*%\s*$")
_PITCH_RE = re.compile(r"^\s*([+\-]?\d+(?:\.\d+)?)\s*Hz\s*$", re.IGNORECASE)


def _parse_rate(rate: str) -> float:
    """'+5%' → 1.05.   '-10%' → 0.90. Anything unparseable → 1.0."""
    m = _RATE_RE.match(rate or "")
    if not m:
        return 1.0
    try:
        pct = float(m.group(1))
    except ValueError:  # pragma: no cover - unreachable: _RATE_RE only matches valid float syntax, so float() never raises
        return 1.0
    return max(0.5, min(2.0, 1.0 + pct / 100.0))


def _parse_pitch_semitones(pitch: str) -> float:
    """Edge-tts pitch is given in Hz; convert to semitones around 200 Hz
    (~male speaking fundamental) so librosa pitch-shift gets a sane value.
    '+4Hz' → +0.34 semitones. Returns 0.0 on parse failure."""
    m = _PITCH_RE.match(pitch or "")
    if not m:
        return 0.0
    try:
        hz = float(m.group(1))
    except ValueError:  # pragma: no cover - unreachable: _PITCH_RE only matches valid float syntax, so float() never raises
        return 0.0
    if hz == 0.0:
        return 0.0
    base = 200.0
    new = max(40.0, base + hz)
    return 12.0 * float(np.log2(new / base))


# ──────────────────────────────────────────────────────────────────────────
#  RENDER
# ──────────────────────────────────────────────────────────────────────────

def render(text: str, rate: str = "+0%", pitch: str = "+0Hz") -> Tuple[np.ndarray, int]:
    """Synthesise `text` in the cloned voice. Returns (audio float32, sr).

    Raises RuntimeError on any failure — bobert_companion.synthesise()
    catches that and falls back to edge-tts."""
    if not text or not text.strip():
        return np.zeros(1, dtype=np.float32), 24000

    reason = availability_reason()
    if reason:
        raise RuntimeError(reason)

    sample = get_sample_path()
    language = get_language()

    model = _load_xtts_model()

    # The Coqui high-level API accepts `tts()` for in-memory synthesis. It
    # returns a list[float] at 24 kHz for XTTS-v2.
    try:
        out = model.tts(text=text, speaker_wav=sample, language=language)
    except Exception as e:
        # OOM during render leaves the cached model in a bad VRAM state —
        # clear it so the next utterance reloads clean rather than re-OOMing.
        # The raised RuntimeError lets bobert_companion fall back to edge-tts.
        if _is_cuda_oom(e):
            _drop_gpu_model_cache()
        raise RuntimeError(f"XTTS render failed: {type(e).__name__}: {e}") from e

    audio = np.asarray(out, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    sr = 24000   # XTTS-v2 default sample rate

    # Optional prosody match — only when librosa is present, otherwise the
    # raw audio still goes out (still better than silence). Time-stretch
    # first so pitch-shift's pitch correction works against final length.
    rate_mul = _parse_rate(rate)
    semitones = _parse_pitch_semitones(pitch)
    if rate_mul != 1.0 or semitones != 0.0:
        try:
            import librosa  # type: ignore[import-not-found]
            if rate_mul != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=rate_mul)
            if semitones != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
            audio = audio.astype(np.float32)
        except Exception:
            # librosa missing or broken — fall through with the raw audio.
            pass

    return audio, sr


# ──────────────────────────────────────────────────────────────────────────
#  STREAMING RENDER (opt-in; not used by the default synthesise() path)
# ──────────────────────────────────────────────────────────────────────────

def render_stream(text: str, language: Optional[str] = None):
    """Yield (chunk, sr) pairs as XTTS-v2 produces them. Use this from a
    streaming playback layer to land first audio under ~400 ms.

    The default synthesise() in bobert_companion is synchronous and waits
    for the full buffer, so this helper is opt-in — call it from a future
    realtime voice path. Raises RuntimeError on availability failure."""
    reason = availability_reason()
    if reason:
        raise RuntimeError(reason)

    sample = get_sample_path()
    lang = language or get_language()
    model = _load_xtts_model()

    # Reach past the high-level wrapper for the underlying XTTS model
    # which exposes inference_stream() / synthesize() with chunking.
    underlying = getattr(model, "synthesizer", None)
    xtts = getattr(underlying, "tts_model", None) if underlying is not None else None
    if xtts is None or not hasattr(xtts, "inference_stream"):
        # Fallback: call the non-streaming render once and yield it.
        audio, sr = render(text)
        yield audio, sr
        return

    try:
        gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
            audio_path=[sample]
        )
    except Exception as e:
        raise RuntimeError(f"XTTS conditioning failed: {type(e).__name__}: {e}") from e

    sr = 24000
    try:
        for chunk in xtts.inference_stream(
            text,
            lang,
            gpt_cond_latent,
            speaker_embedding,
        ):
            # Tensor → numpy float32 mono
            arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else -1).astype(np.float32)
            yield arr, sr
    except Exception as e:
        raise RuntimeError(f"XTTS stream failed: {type(e).__name__}: {e}") from e


# ──────────────────────────────────────────────────────────────────────────
#  RUNTIME BACKEND TOGGLE
# ──────────────────────────────────────────────────────────────────────────

def set_backend(name: str) -> str:
    """Switch bobert_companion.TTS_BACKEND. Refuses to switch to 'xtts'
    when its deps / sample aren't ready, so the user doesn't get silence
    after saying 'use my voice'."""
    n = (name or "").strip().lower()
    if n not in _VALID_BACKENDS:
        return (f"'{name}' isn't a TTS backend, sir. "
                f"Options: {', '.join(_VALID_BACKENDS)}.")
    if n == "xtts":
        reason = availability_reason()
        if reason:
            return reason
        # Warm the model on a background thread so the first reply doesn't
        # eat the cold-start latency. If the load fails we'll still surface
        # it on the next render and fall back to edge.
        def _warm():
            try:
                _load_xtts_model()
            except Exception as e:
                print(f"  [xtts] background warm-up failed: {e}")
        threading.Thread(target=_warm, daemon=True).start()
    _set_backend_on_bobert(n)
    pretty = {"edge": "edge-tts", "pyttsx3": "the offline fallback voice",
              "xtts": "your cloned voice",
              "kokoro": "my local Kokoro voice"}[n]
    return f"Switched to {pretty}, sir."


# Voice-trigger phrases. Both directions are matched in a single regex so
# the pre-router stays one if-statement. Captured group 1 is non-empty for
# the "use my voice" / cloned direction; group 2 is non-empty for the
# "switch to edge" / default direction; group 3 captures explicit names
# like 'pyttsx3'.
_BACKEND_VOICE_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:"
        r"(?:use|switch\s+to|enable|turn\s+on|activate)\s+"
        r"(?:my|the\s+cloned|my\s+custom)\s+voice"                       # group1: cloned
        r"|"
        r"(?:use|switch\s+(?:to|back\s+to)|enable|turn\s+on|activate)\s+"
        r"(?:the\s+)?(edge(?:[-\s]tts)?|default|original|microsoft)\s+voice"  # group2: edge
        r"|"
        r"(?:use|switch\s+to|enable)\s+(?:the\s+)?(pyttsx3|offline|kokoro|local)\s+voice"  # group3: pyttsx3/kokoro
    r")"
    r"[.!?]*\s*$",
    re.IGNORECASE,
)


def maybe_switch_backend(utterance: str) -> Optional[str]:
    """Pre-route 'use my voice' / 'switch to edge voice' phrases straight
    to set_backend() so the LLM round-trip can be skipped.

    Returns the spoken confirmation when matched, else None."""
    if not utterance or not utterance.strip():
        return None
    m = _BACKEND_VOICE_RE.match(utterance.strip())
    if not m:
        return None
    text_low = utterance.lower()
    if any(p in text_low for p in ("my voice", "cloned voice", "custom voice")):
        return set_backend("xtts")
    if any(p in text_low for p in ("edge", "default voice", "original voice", "microsoft voice")):
        return set_backend("edge")
    if "kokoro" in text_low or "local voice" in text_low:
        return set_backend("kokoro")
    if "pyttsx" in text_low or "offline voice" in text_low:
        return set_backend("pyttsx3")
    return None


# ──────────────────────────────────────────────────────────────────────────
#  ACTIONS — exposed to the LLM in case it wants to call the toggle itself
# ──────────────────────────────────────────────────────────────────────────

def _act_set_tts_backend(arg: str) -> str:
    return set_backend(arg or "")


def _act_list_tts_backends(_: str = "") -> str:
    current = get_backend()
    sample = get_sample_path()
    xtts_state = "ready" if is_available() else availability_reason() or "unavailable"
    sample_note = f" (sample: {sample})" if sample else ""
    return (f"current backend: {current}. "
            f"available: {', '.join(_VALID_BACKENDS)}. "
            f"xtts status: {xtts_state}{sample_note}")


def _act_enroll_xtts_sample(arg: str) -> str:
    """Point XTTS at a different voice sample WAV without restarting."""
    path = (arg or "").strip().strip("'\"")
    if not path:
        return "format: enroll_xtts_sample, <path/to/sample.wav>"
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return f"no such file: {path}"
    bc = _bobert()
    if bc is None:
        return "bobert_companion isn't loaded, sir — can't persist the sample path"
    setattr(bc, "XTTS_VOICE_SAMPLE", path)
    _invalidate_model_cache()
    return (f"Voice sample updated, sir. I'll re-train on '{path}' "
            f"the next time you ask me to speak in that voice.")


def register(actions: dict):
    actions["set_tts_backend"]     = _act_set_tts_backend
    actions["list_tts_backends"]   = _act_list_tts_backends
    actions["enroll_xtts_sample"]  = _act_enroll_xtts_sample
