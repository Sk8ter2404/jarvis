"""
Local voice-cloning TTS backend for JARVIS — Resemble AI **Chatterbox**.

WHAT THIS IS
------------
A LAZY, degrade-cleanly wrapper around the optional ``chatterbox-tts`` model
(Resemble AI, MIT-licensed) so JARVIS can synthesise replies in a *cloned*
voice on the owner's RTX 3090. Chatterbox clones a voice from a short (~5 s)
consented reference clip and renders roughly real-time on a 3090.

It slots in FRONT of the existing edge-tts → pyttsx3 → SAPI5 ladder in
``bobert_companion.synthesise()``: when the clone is enabled + a profile is
selected + the package/model/GPU are all present, we render the sentence
through Chatterbox and hand back a full ``(np.ndarray float32 mono, sample_rate)``
waveform. On ANY failure (package missing, no CUDA, no profile, model load
crash, render error) ``synthesize()`` returns ``None`` so the caller falls
straight through to the normal ladder — a missed dep or an over-eager toggle
must NEVER silence JARVIS. This mirrors how ``skills/local_vision.py`` /
``LOCAL_VISION_FALLBACK`` treat an absent heavy model.

The waveform is returned WHOLE per sentence, so the v1.94.0 streaming-TTS and
v1.96.0 barge-in paths (which call ``_speak`` → ``synthesise`` once per
sentence) keep working unchanged — they never see a partial buffer.

═══════════════════════════════════════════════════════════════════════════
ETHICS / SCOPE  (read before extending this module)
═══════════════════════════════════════════════════════════════════════════
This feature supports EXACTLY TWO kinds of voice profile, both consented:

  (a) OWNER  — the owner's OWN voice, enrolled from his OWN reference
               recording that he explicitly consented to (source="owner").
  (b) CHARACTER — a "JARVIS" in-character British-butler voice built from a
               reference clip the owner provides, OR a bundled *non-celebrity*
               voice (source="character").

It DOES NOT and MUST NOT grow a path that clones a NAMED REAL PERSON — no
celebrity presets, no "the real voice actor", no scraping reference audio off
the internet. Enrollment REQUIRES an explicit ``consent: true`` flag written
into the profile's ``meta.json`` (see ``tools/enroll_voice.py``); a profile
WITHOUT that flag is refused here at load/selection time. Audio + profiles live
under a GITIGNORED directory and are NEVER committed.

═══════════════════════════════════════════════════════════════════════════
TESTABILITY
═══════════════════════════════════════════════════════════════════════════
The pure/stdlib-testable surface — the profile registry (``list_profiles`` /
``load_profile``), selection logic (``resolve_active_profile``), the consent
gate (``profile_is_usable``), and the fallback decision (``is_available``) —
is all separate small functions unit-tested WITHOUT importing chatterbox,
torch, or touching a GPU. The single heavy seam is ``_load_engine()`` (the only
place ``import chatterbox`` happens); tests mock it or its import failure.

Config knobs (core/config.py, all user_settings-overridable):
  VOICE_CLONE_ENABLED   bool  — master switch (default False / OFF)
  VOICE_CLONE_PROFILE   str   — active profile name ("" = none selected)
  VOICE_CLONE_MODEL     str   — engine id, currently only "chatterbox"

Latency note: the FIRST call pays a multi-second cold-start (model download on
the very first run, then load onto CUDA). After that a one-sentence reply is
~real-time on a 3090. The model is loaded once and cached for the process.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

# numpy is present wherever real synthesis runs (the monolith imports it at top
# level) and on the CI runner's light-dep set (pre-imported by the sim). Import
# defensively anyway so a bare-stdlib import of THIS module for the pure-logic
# tests never hard-fails on a box without numpy — the type is only needed on the
# heavy render path, which is mocked in tests.
try:
    import numpy as np           # type: ignore
except Exception:                # pragma: no cover - numpy is effectively always present
    np = None                    # type: ignore


# ─── profile storage layout ────────────────────────────────────────────────
# data/voice_profiles/<name>/reference.wav + meta.json. The directory is
# GITIGNORED (see .gitignore) so no audio and no owner-identifying profile
# metadata is ever committed. tools/enroll_voice.py is the ONLY writer.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(_PROJECT_ROOT, "data", "voice_profiles")

REFERENCE_WAV_NAME = "reference.wav"
META_NAME = "meta.json"

# The only sources we accept — mirrors the ETHICS scope above. A profile whose
# meta.json declares any other source is treated as not-usable (defensive: an
# unknown source is a profile we can't vouch for the provenance of).
_ALLOWED_SOURCES = ("owner", "character")

# Cached engine handle + which (model, profile) it was built for, so a profile
# switch rebuilds rather than serving the previous voice. Module-level single
# element lists follow the codebase's mutable-singleton pattern so a test (or a
# live profile switch) can reset them without rebinding the name.
_engine_cache: list = [None]          # the loaded chatterbox model, or None
_engine_key: list = [None]            # (model_id, profile_name) the cache is for


# ═══════════════════════════════════════════════════════════════════════════
# PURE / STDLIB-TESTABLE SURFACE  (no chatterbox, no torch, no GPU)
# ═══════════════════════════════════════════════════════════════════════════

def _meta_path(name: str) -> str:
    return os.path.join(PROFILES_DIR, name, META_NAME)


def _reference_path(name: str) -> str:
    return os.path.join(PROFILES_DIR, name, REFERENCE_WAV_NAME)


def load_profile(name: str) -> Optional[dict]:
    """Read ``data/voice_profiles/<name>/meta.json`` and return it as a dict,
    with the resolved reference-wav path folded in under ``reference_wav``.

    Returns ``None`` — never raises — when the profile dir/meta is missing or
    unreadable, so every caller can treat "no such profile" and "broken
    profile" identically (a defensive read; a corrupt meta must not crash the
    hot synth path). The consent/source GATE lives in ``profile_is_usable`` so
    this stays a dumb reader.
    """
    if not name or not isinstance(name, str):
        return None
    meta_path = _meta_path(name)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    # Fold in the on-disk name + reference path so downstream code doesn't
    # re-derive them. Trust the dir name over any stale ``name`` in the file.
    meta = dict(meta)
    meta["name"] = name
    meta["reference_wav"] = _reference_path(name)
    return meta


def list_profiles() -> list[dict]:
    """Return every enrolled profile as a meta-dict (see ``load_profile``),
    sorted by name. Missing dir → ``[]``. Sub-entries that fail to parse are
    silently skipped rather than aborting the whole listing — one broken
    profile must not hide the others.
    """
    out: list[dict] = []
    try:
        entries = sorted(os.listdir(PROFILES_DIR))
    except Exception:
        return out
    for entry in entries:
        if entry.startswith(".") or entry.startswith("_"):
            continue
        if not os.path.isdir(os.path.join(PROFILES_DIR, entry)):
            continue
        meta = load_profile(entry)
        if meta is not None:
            out.append(meta)
    return out


def profile_is_usable(meta: Optional[dict]) -> bool:
    """THE CONSENT GATE. A profile is usable ONLY if its meta declares
    ``consent == true`` AND an allowed ``source`` ("owner" | "character") AND
    its reference wav actually exists on disk.

    This is the single chokepoint that enforces the ethics scope: no consent
    flag → not usable, unknown/celebrity source → not usable. Kept pure +
    boolean so it's trivially unit-testable without the model.
    """
    if not isinstance(meta, dict):
        return False
    if meta.get("consent") is not True:          # must be a literal True, not truthy
        return False
    if meta.get("source") not in _ALLOWED_SOURCES:
        return False
    ref = meta.get("reference_wav")
    if not ref or not os.path.isfile(ref):
        return False
    return True


def resolve_active_profile(
    enabled: bool,
    profile_name: str,
) -> Optional[dict]:
    """Selection logic: given the master switch + the configured active-profile
    name, return the usable profile meta-dict to render with, or ``None``.

    ``None`` (→ fall back to the normal ladder) when: the feature is disabled,
    no profile name is set, the named profile doesn't exist, or it fails the
    consent gate. Pure — no model, no CUDA, no import of chatterbox.
    """
    if not enabled:
        return None
    if not profile_name:
        return None
    meta = load_profile(profile_name)
    if not profile_is_usable(meta):
        return None
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG SNAPSHOT  (read live so a user_settings flip / voice toggle applies)
# ═══════════════════════════════════════════════════════════════════════════

def _config():
    """Return core.config (or None). Imported lazily so this module stays
    importable in the pure-logic tests even if config's import chain isn't
    wanted; every consumer treats a None config as 'feature off'."""
    try:
        from core import config  # type: ignore
        return config
    except Exception:             # pragma: no cover - config is in-tree
        return None


def _cfg_enabled() -> bool:
    cfg = _config()
    return bool(getattr(cfg, "VOICE_CLONE_ENABLED", False)) if cfg else False


def _cfg_profile() -> str:
    cfg = _config()
    return str(getattr(cfg, "VOICE_CLONE_PROFILE", "") or "") if cfg else ""


def _cfg_model() -> str:
    cfg = _config()
    return str(getattr(cfg, "VOICE_CLONE_MODEL", "chatterbox") or "chatterbox") if cfg else "chatterbox"


def _cfg_device() -> str:
    """Optional torch device override for the clone engine. "" (default/unset)
    preserves the historical behaviour of ``"cuda"`` (device 0) when CUDA is
    present else ``"cpu"``. Set e.g. ``"cuda:1"`` to run the clone on a second,
    idle GPU instead of hard-pinning cuda:0 and fighting gemma for its last
    ~1GB. 2026-07-08."""
    cfg = _config()
    return str(getattr(cfg, "VOICE_CLONE_DEVICE", "") or "") if cfg else ""


# ═══════════════════════════════════════════════════════════════════════════
# HEAVY SEAM  (the ONLY place chatterbox / torch / CUDA are touched)
# ═══════════════════════════════════════════════════════════════════════════

# Minimum free VRAM we require on an EXPLICITLY chosen CUDA device before we
# load Chatterbox onto it. Below this we degrade to the normal TTS ladder rather
# than OOM-contend for gemma's headroom. ~2GB gives the model room to load. Only
# consulted when VOICE_CLONE_DEVICE names a cuda device (unset knob = old path,
# no gating). 2026-07-08.
_MIN_FREE_VRAM_BYTES = 2 * 1024 * 1024 * 1024


def _device_index(device: str) -> int:
    """Parse a torch CUDA device string ("cuda" | "cuda:1") to the integer index
    ``torch.cuda.mem_get_info`` wants. Bare "cuda" → 0. Never raises."""
    try:
        if ":" in device:
            return int(device.split(":", 1)[1])
    except Exception:
        pass
    return 0


def _free_vram_ok(device: str) -> bool:
    """True if ``device`` reports at least ``_MIN_FREE_VRAM_BYTES`` free VRAM via
    ``torch.cuda.mem_get_info``. Fails OPEN (returns True) on any probe failure —
    a torch build without mem_get_info must not wrongly suppress the feature; a
    real OOM at load is still caught downstream and falls back cleanly."""
    try:
        import torch  # type: ignore
    except Exception:
        return True
    try:
        free, _total = torch.cuda.mem_get_info(_device_index(device))
        return int(free) >= _MIN_FREE_VRAM_BYTES
    except Exception:
        return True


def _resolve_device() -> Optional[str]:
    """Decide which torch device to load Chatterbox onto.

    Unset ``VOICE_CLONE_DEVICE`` → exactly the previous behaviour ("cuda" when
    available else "cpu") with NO VRAM gating, so the happy path is unchanged.
    A knob that names a CUDA device is honoured AND free-VRAM checked; if that
    device is too full (or CUDA isn't available) we return ``None`` so the
    caller degrades to the normal ladder instead of OOM-contending. A "cpu"
    override always passes. 2026-07-08."""
    want = _cfg_device().strip()
    if not want:
        return "cuda" if _cuda_available() else "cpu"
    if want.lower().startswith("cpu"):
        return want
    if not _cuda_available():
        return None
    if not _free_vram_ok(want):
        return None
    return want

def _cuda_available() -> bool:
    """True only if torch is importable AND reports a CUDA device. Defensive:
    any import/attribute error → False (→ no clone → normal ladder). Never
    raises. Tests either mock this or mock the import failure it wraps."""
    try:
        import torch  # type: ignore
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _chatterbox_importable() -> bool:
    """True if the ``chatterbox`` package can be imported. Isolated so tests can
    mock JUST the import decision (via find_spec) without a real install. Uses
    importlib.util.find_spec so we don't pay the (heavy) import just to probe.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("chatterbox") is not None
    except Exception:
        return False


def _load_engine(profile: dict):
    """Load + cache the Chatterbox model for ``profile``. The SINGLE heavy
    import site. Returns the model handle (opaque) or raises on any failure so
    ``synthesize`` can convert that into a clean ``None`` fallback.

    Cached by (model_id, profile_name): a profile switch rebuilds so we never
    serve the previous voice. Cold start is several seconds; subsequent calls
    reuse the handle.

    NOTE: the exact Chatterbox load/generate API is intentionally wrapped
    thinly here and fully mocked in tests — CI never imports chatterbox. The
    call shape below matches the documented ``ChatterboxTTS.from_pretrained``
    entrypoint; if the upstream signature drifts, this is the one function to
    adjust and it fails closed (→ fallback) rather than crashing JARVIS.
    """
    model_id = _cfg_model()
    key = (model_id, profile.get("name"))
    if _engine_cache[0] is not None and _engine_key[0] == key:
        return _engine_cache[0]

    # Heavy import — deliberately local so importing core.voice_clone stays
    # cheap and the pure-logic tests never need chatterbox on the path.
    from chatterbox.tts import ChatterboxTTS  # type: ignore

    # Device selection honours the optional VOICE_CLONE_DEVICE knob (+ a
    # free-VRAM check on an explicitly chosen CUDA device) so we don't hard-pin
    # cuda:0. None → the device is too full / unusable: raise so synthesize()
    # converts it into a clean None fallback rather than OOM-contending. Unset
    # knob keeps the historical "cuda"/"cpu" pick. 2026-07-08.
    device = _resolve_device()
    if device is None:
        raise RuntimeError("voice-clone: chosen CUDA device lacks free VRAM")
    model = ChatterboxTTS.from_pretrained(device=device)

    _engine_cache[0] = model
    _engine_key[0] = key
    return model


def _reset_engine_cache() -> None:
    """Drop the cached model so the next synth reloads (used by a profile
    switch and by tests). Cheap; never raises."""
    _engine_cache[0] = None
    _engine_key[0] = None


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API  (called from bobert_companion.synthesise())
# ═══════════════════════════════════════════════════════════════════════════

def is_available() -> bool:
    """Cheap, defensive gate the synth hot-path calls every utterance:

        chatterbox importable  AND  a usable profile is selected  AND  CUDA.

    Every clause fails CLOSED (any exception → False → normal ladder). Ordered
    cheapest-first: config flags, then find_spec, then the CUDA probe last.
    Safe to call on a headless / no-GPU box (returns False) and safe to call
    with chatterbox absent (returns False).
    """
    try:
        if not _cfg_enabled():
            return False
        if resolve_active_profile(_cfg_enabled(), _cfg_profile()) is None:
            return False
        if not _chatterbox_importable():
            return False
        if not _cuda_available():
            return False
        return True
    except Exception:
        return False


def synthesize(text: str, profile: Optional[dict] = None) -> Optional[Tuple["np.ndarray", int]]:
    """Render ``text`` through the cloned voice and return
    ``(np.ndarray float32 mono, sample_rate)``, or ``None`` on ANY failure so
    the caller falls back to the normal edge-tts → pyttsx3 → SAPI5 ladder.

    ``profile`` may be passed explicitly (tests / callers that already resolved
    it); when omitted we resolve it from config. A ``None`` here is the
    contract the whole feature rests on: it must be impossible for a voice-clone
    problem to silence JARVIS.
    """
    try:
        if not text or not str(text).strip():
            return None
        if np is None:            # no numpy → can't produce the waveform contract
            return None
        if profile is None:
            profile = resolve_active_profile(_cfg_enabled(), _cfg_profile())
        # Re-check the consent gate even for a caller-supplied profile — never
        # render from a profile that isn't consented/usable.
        if not profile_is_usable(profile):
            return None
        if not _chatterbox_importable() or not _cuda_available():
            return None

        model = _load_engine(profile)
        ref_wav = profile.get("reference_wav")
        # Chatterbox: generate a waveform conditioned on the reference clip.
        # Return shape is a torch tensor / array-like; normalise to a mono
        # float32 numpy buffer + its sample rate. Wrapped thin + mocked in CI.
        wav = model.generate(str(text), audio_prompt_path=ref_wav)
        audio, sr = _to_mono_float32(wav, model)
        if audio is None or sr <= 0 or getattr(audio, "size", 0) == 0:
            return None
        return audio, sr
    except Exception as e:  # pragma: no cover - defensive; exercised via mocks
        print(f"  [voice-clone] render failed ({type(e).__name__}: {e}); "
              f"falling back to the normal TTS ladder")
        return None


def _to_mono_float32(wav, model) -> Tuple[Optional["np.ndarray"], int]:
    """Coerce whatever Chatterbox hands back (torch tensor / ndarray, possibly
    2-D) into a contiguous mono float32 numpy array + an int sample rate.

    Sample rate is read off the model (``model.sr``) with a 24 kHz fallback —
    24 kHz mirrors edge-tts' native rate so the downstream resample path is
    unchanged. Kept separate so the normalisation is unit-testable with a fake
    tensor and a fake model, no real chatterbox needed.
    """
    if np is None:
        return None, 0
    try:
        arr = wav
        # torch tensor → numpy (detach/cpu if those attrs exist; a plain
        # ndarray or list falls straight through to np.asarray).
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr, dtype=np.float32)
        # Chatterbox returns (1, N) or (N,) — squeeze to mono 1-D.
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            # Unexpected multi-channel: average down to mono rather than error.
            arr = arr.mean(axis=0).astype(np.float32)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    except Exception:
        return None, 0
    sr = 24000
    try:
        sr = int(getattr(model, "sr", 24000) or 24000)
    except Exception:
        sr = 24000
    return arr, sr
