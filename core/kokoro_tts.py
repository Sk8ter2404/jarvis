"""core/kokoro_tts.py — CPU-only Kokoro TTS backend (frees the 3090 for the brain).

WHY THIS EXISTS
---------------
The default voice used to be the Chatterbox clone, resident on the RTX 3090 (~5 GB)
— which capped how big a local brain could fit. Kokoro-82M runs entirely on the
14900K CPU (onnxruntime CPUExecutionProvider, torch-free) at ~4.5× real-time, so
moving everyday TTS here frees the GPU while keeping speech local + private +
offline. The consented Chatterbox clone stays available ON DEMAND (Axis 1, checked
before this backend in synthesise()).

FAIL-CLOSED CONTRACT (mirrors core/voice_clone)
-----------------------------------------------
`synthesize()` NEVER raises and returns None on ANY failure, so the caller falls
straight through the existing edge-tts → pyttsx3 → SAPI5 → silence ladder and
JARVIS is never silenced. Adversarial-review hardening (2026-07-15):
  * the ENTIRE engine construction (onnx session init + the espeak-ng dll load
    that runs inside the phonemizer) is wrapped fail-closed, not just create();
  * engine FAILURE is memoized (not only success) so a broken/corrupt model can't
    thrash a fresh load attempt on every single utterance;
  * a wall-clock timeout bounds a pathological long synth so it can't wedge the
    voice thread;
  * `is_available()` is cheap (find_spec + file existence) AND the default flip is
    gated elsewhere on one PROVEN in-process synth, so a missing/corrupt model
    quietly runs on edge instead of going mute.

CI SAFETY: the real `import kokoro_onnx` happens ONLY inside `_engine()`, guarded
by find_spec, so tools/run_tests_ci_sim.py never imports onnxruntime/kokoro.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

_LOCK = threading.Lock()
_ENGINE = [None]          # the Kokoro singleton once built
_FAILED = [False]         # True once construction has failed — do not retry-thrash

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.environ.get(
    "KOKORO_MODEL", os.path.join(_HERE, "..", "models", "kokoro", "kokoro-v1.0.onnx"))
_VOICES = os.environ.get(
    "KOKORO_VOICES", os.path.join(_HERE, "..", "models", "kokoro", "voices-v1.0.bin"))
# bm_george = British male "butler"; bf_emma = British female. Swap via env.
_VOICE = os.environ.get("KOKORO_VOICE", "bm_george")
_LANG = os.environ.get("KOKORO_LANG", "en-gb")
_SR = 24000               # Kokoro native sample rate (matches chatterbox/edge path)
# Generous wall-clock ceiling: at RTF ~0.22 even a 60 s reply synths in ~13 s, so
# 30 s only ever trips on a genuinely wedged engine — then we return None and the
# edge ladder speaks instead. Mirrors voice_clone's timeout discipline.
_SYNTH_TIMEOUT_S = float(os.environ.get("KOKORO_SYNTH_TIMEOUT_S", "30"))

# ── onnxruntime session tuning (2026-09-04 profile, TRACK 3) ─────────────
# kokoro_onnx builds its InferenceSession with DEFAULT SessionOptions. On this
# box that means ORT sizes the intra-op pool to the PHYSICAL core count (24 on
# the 14900K) and leaves spin-wait ON, so every worker that finishes its slice
# BUSY-WAITS on the next op instead of sleeping. Measured on this rig, one
# 2.3 s reply cost 644 ms wall and 13.7 CPU-seconds — 21 cores, of which most
# was spin, and the spinning threads were stealing cycles from the threads
# doing real work. A bounded pool with spinning OFF is faster AND ~4x cheaper:
#
#   config                short reply        long reply        CPU burned
#   default (today)       644 ms             1814 ms           21.2 cores
#   16 threads, no spin   491 ms  (-24%)     1268 ms  (-30%)    6.7-7.7 cores
#   8 threads,  no spin   586 ms  ( -9%)     1465 ms  (-19%)    3.9-4.2 cores
#   (3 passes x 5 reps each, one config per subprocess, medians)
#
# Output is unchanged: the tuned session is bit-identical on short replies and
# within 3e-6 (~-110 dBFS, far below 16-bit quantisation) on long ones, across
# voices/speeds — thread count only reorders float reductions.
#
# KOKORO_ONNX_THREADS=0 restores kokoro_onnx's stock construction entirely.
_ONNX_THREADS = os.environ.get("KOKORO_ONNX_THREADS", "16")
_ONNX_SPIN = (os.environ.get("KOKORO_ONNX_SPIN", "0").strip() or "0")


def _tuned_session(model_path: str):
    """An onnxruntime InferenceSession for Kokoro with a BOUNDED intra-op pool
    and spin-wait disabled. Returns None — meaning "let kokoro_onnx construct
    the session itself, exactly as before" — when onnxruntime is missing, the
    tunable is disabled, or ANYTHING at all goes wrong. Never raises. Imported
    lazily so the module load stays free of native deps (CI-safety contract)."""
    try:
        import onnxruntime as _rt
    except Exception:
        return None
    try:
        n = int(_ONNX_THREADS)
    except Exception:
        n = 16
    if n <= 0:                      # explicit opt-out
        return None
    try:
        n = max(1, min(n, os.cpu_count() or n))
        so = _rt.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        so.execution_mode = _rt.ExecutionMode.ORT_SEQUENTIAL
        # "0" = a worker that finishes its slice SLEEPS instead of burning a
        # core until the next op arrives. This is the change that matters.
        so.add_session_config_entry("session.intra_op.allow_spinning", _ONNX_SPIN)
        so.add_session_config_entry("session.inter_op.allow_spinning", _ONNX_SPIN)
        provider = os.environ.get("ONNX_PROVIDER") or "CPUExecutionProvider"
        sess = _rt.InferenceSession(model_path, sess_options=so,
                                    providers=[provider])
        return sess
    except Exception as e:
        print(f"  [kokoro] tuned ONNX session unavailable "
              f"({type(e).__name__}: {e}); using kokoro_onnx defaults")
        return None


def _build_engine(Kokoro):
    """Build the Kokoro engine, preferring our tuned session. Returns
    (engine, how) where `how` is a short string for the ready line. Falls back
    to kokoro_onnx's own construction on any problem, so this can only ever be
    as good as before, never worse."""
    sess = _tuned_session(_MODEL)
    if sess is not None and hasattr(Kokoro, "from_session"):
        try:
            eng = Kokoro.from_session(sess, _VOICES)
            n = getattr(sess, "_sess_options", None)
            n = getattr(n, "intra_op_num_threads", None) or _ONNX_THREADS
            spin = "on" if _ONNX_SPIN not in ("0", "false", "False") else "off"
            return eng, f"onnx intra_op={n}, spinning {spin}"
        except Exception as e:
            print(f"  [kokoro] from_session failed ({type(e).__name__}: {e}); "
                  f"falling back to stock construction")
    return Kokoro(_MODEL, _VOICES), "onnx defaults"



def _models_present() -> bool:
    try:
        return (os.path.exists(_MODEL) and os.path.getsize(_MODEL) > 1_000_000
                and os.path.exists(_VOICES) and os.path.getsize(_VOICES) > 100_000)
    except OSError:
        return False


def is_available() -> bool:
    """Cheap: the package is importable, the model files exist, and we haven't
    already failed to build the engine this process. Does NOT import kokoro_onnx
    (keeps CI + cold callers light). A True here does not guarantee a good synth —
    the default-flip is gated on a proven synth; a later failure fails closed."""
    if _FAILED[0]:
        return False
    try:
        import importlib.util as _u
        if _u.find_spec("kokoro_onnx") is None:
            return False
    except Exception:
        return False
    return _models_present()


def _engine():
    """Lazy CPU singleton. Builds the espeak-ng phonemizer wiring + the onnx
    Kokoro session ONCE. Any failure is memoized in _FAILED so we never retry —
    and returns None (caller falls back). The whole thing is fail-closed."""
    if _ENGINE[0] is not None:
        return _ENGINE[0]
    if _FAILED[0]:
        return None
    with _LOCK:
        if _ENGINE[0] is not None:
            return _ENGINE[0]
        if _FAILED[0]:
            return None
        try:
            if not _models_present():
                raise FileNotFoundError(f"kokoro model missing: {_MODEL}")
            # Force CPU everywhere; never touch the 3090.
            os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")
            # Point the phonemizer at the espeak-ng dll BUNDLED by espeakng_loader
            # (no separate Windows espeak install). This runs a LoadLibrary — a
            # prime cp314 failure point, so it is inside the fail-closed block.
            try:
                import espeakng_loader
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            except Exception as _pe:
                print(f"  [kokoro] espeak wiring warning ({type(_pe).__name__}: "
                      f"{_pe}); continuing — kokoro may still self-wire")
            from kokoro_onnx import Kokoro
            _ENGINE[0], _how = _build_engine(Kokoro)
            print(f"  [kokoro] CPU engine ready (voice={_VOICE}, {_LANG}) — {_how}")
            return _ENGINE[0]
        except Exception as e:
            _FAILED[0] = True
            print(f"  [kokoro] engine construction FAILED "
                  f"({type(e).__name__}: {e}); backend disabled → edge fallback")
            return None


def _render(text: str, speed: float, out: list) -> None:
    try:
        import numpy as np
        eng = _engine()
        if eng is None:
            return
        samples, sr = eng.create(text, voice=_VOICE, speed=float(speed), lang=_LANG)
        a = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).squeeze())
        if a.ndim > 1:                      # coerce any stereo down to mono
            a = a.mean(axis=0).astype(np.float32)
        if a.size:
            out.append((a, int(sr or _SR)))
    except Exception as e:
        print(f"  [kokoro] render failed ({type(e).__name__}: {e})")


def synthesize(text: str, speed: float = 1.0) -> Optional[Tuple["object", int]]:
    """Render `text` to (float32 mono ndarray, 24000). NEVER raises; returns None
    on empty text, unavailable engine, timeout, or any error — so the caller's
    edge/pyttsx3/SAPI/silence ladder takes over. Bounded by a wall-clock timeout."""
    t = (text or "").strip()
    if not t or not is_available():
        return None
    # JARVIS's own number/version normaliser (times, decimals, versions) if present
    # — Kokoro's g2p reads digits literally otherwise ("v2.0.83" → "vee two point…").
    try:
        from core.voice_clone import _normalize_numbers_for_speech as _norm
        t = _norm(t)
    except Exception:
        pass
    out: list = []
    th = threading.Thread(target=_render, args=(t, speed, out), daemon=True)
    th.start()
    th.join(_SYNTH_TIMEOUT_S)
    if th.is_alive():
        print(f"  [kokoro] synth exceeded {_SYNTH_TIMEOUT_S:.0f}s — falling back")
        return None
    return out[0] if out else None
