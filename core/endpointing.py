"""core/endpointing.py — where the owner's speech really ended (speed plan R1).

WHY THIS EXISTS
---------------
record_speech ends a capture after 21 silent 64 ms chunks (1,344 ms) in a row.
"Silent" means below the raw RMS gate, so a noise spike restarts the count:
counting chunks since the last loud one therefore ALWAYS gives exactly 1,344
ms and cannot measure the real wait. This module runs a speech detector over
the finished clip instead and reports how long the clip ran on after the last
speech window: ``tail_ms``, in AUDIO time (last speech window -> the clip's
last sample). The VAD break — the turn's t0 — is the wall-clock moment the
capture loop handled that last sample, which trails it by the capture lag
(the input stream's latency plus any chunks still queued behind the loop);
the [turn-timing] line prints that as ``cap_lag_ms``. EOS -> first answer
audio is then ``tail_ms + cap_lag_ms + first_play`` (core/turn_timing.py).

SileroVad opens its OWN onnxruntime session on faster-whisper's bundled
``assets/silero_vad_v6.onnx`` (intra-op 1, inter-op 1, spin-wait off). It never
touches faster-whisper's get_vad_model() singleton, which _transcribe_impl uses
under _stt_lock — sharing it would make this probe wait on, or race, the
owner's own decode.

Telemetry only. Nothing here can change a turn:
  * speech_tail_ms() never raises and returns None for anything it cannot
    measure; a load or run failure LATCHES the detector off for the session
    (``failed`` names why), so a broken install costs one attempt, not one per
    turn;
  * no I/O besides reading the model file once.

CI SAFETY: numpy and onnxruntime are imported lazily inside the methods (the
rule at core/kokoro_tts.py's CI SAFETY note), and the model is located with
importlib.util.find_spec, which never imports faster_whisper. Importing this
module therefore pulls in nothing beyond the stdlib (tests/test_endpointing.py
pins that). R7 (Smart Turn) extends this module; this is the Silero part only.
"""
from __future__ import annotations

import importlib.util
import os
import threading

SAMPLE_RATE = 16000        # Silero's rate; record_speech captures at 16 kHz
WINDOW = 512               # samples per Silero v6 window (32 ms)
CONTEXT = 64               # samples of the previous window each one carries
SPEECH_THRESHOLD = 0.5     # Silero's stock speech threshold (R7 uses it too)
TAIL_SCAN_S = 4.0          # the usual scan: the clip's last 4 s
MAX_SCAN_S = 30.0          # the fallback whole-clip scan never exceeds this
MODEL_FILE = "silero_vad_v6.onnx"


def bundled_model_path() -> "str | None":
    """faster-whisper's bundled Silero model, or None. find_spec only: the
    package is never imported. Never raises."""
    try:
        spec = importlib.util.find_spec("faster_whisper")
    except Exception:
        return None
    if spec is None:
        return None
    try:
        for d in (spec.submodule_search_locations or ()):
            p = os.path.join(d, "assets", MODEL_FILE)
            if os.path.isfile(p):
                return p
    except Exception:
        return None
    return None


def default_session(path: str):
    """An onnxruntime CPU session for the Silero model: one intra-op and one
    inter-op thread, spin-wait off, no arena — a ~2 MB side session that
    never competes with Kokoro or the ambient decoder for cores."""
    import onnxruntime as ort   # lazy: CI never installs it
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 4
    for key in ("session.intra_op.allow_spinning",
                "session.inter_op.allow_spinning"):
        try:
            so.add_session_config_entry(key, "0")
        except Exception:
            pass
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


class SileroVad:
    """Per-window speech probabilities for a 16 kHz mono clip, and the
    speech tail measured from them. Thread-safe (one lock serialises the
    lazy load and every run). Construction does nothing; the model loads on
    first use or warm().

    `model_path` defaults to bundled_model_path(); `session_factory(path)`
    defaults to default_session (tests pass a fake: anything with
    ``run(None, {"input", "h", "c"})`` returning ``(probs, h, c)``)."""

    def __init__(self, model_path: "str | None" = None, session_factory=None):
        self._path = model_path
        self._factory = session_factory if session_factory is not None \
            else default_session
        self._sess = None
        self._failed = ""
        self._lock = threading.Lock()

    @property
    def failed(self) -> str:
        """Why the detector latched off ('' while usable)."""
        return self._failed

    def _latch(self, why: str) -> None:
        self._sess = None
        self._failed = why or "failed"

    def _session(self):
        """Caller holds self._lock. The session, loading it once; None once
        latched off."""
        if self._failed:
            return None
        if self._sess is None:
            path = self._path if self._path else bundled_model_path()
            if not path:
                self._latch("silero model not found (faster-whisper's "
                            f"assets/{MODEL_FILE})")
                return None
            try:
                self._sess = self._factory(path)
            except Exception as e:
                self._latch(f"load failed: {type(e).__name__}: {e}")
                return None
            if self._sess is None:
                self._latch("load failed: no session")
        return self._sess

    def warm(self) -> bool:
        """Load the model and run one silent window (the first ORT run is the
        slow one). Raises RuntimeError when the detector is (or becomes)
        unusable, so a boot warmer reports it; True otherwise."""
        try:
            import numpy as np
        except Exception as e:
            self._latch(f"numpy unavailable: {e}")
            raise RuntimeError(self._failed)
        self._probs(np.zeros(WINDOW, dtype=np.float32), None)
        if self._failed:
            raise RuntimeError(self._failed)
        return True

    def _probs(self, x, ctx0):
        """Speech probability per WINDOW of `x` (float32, length a multiple
        of WINDOW). `ctx0`: the CONTEXT samples just before x, or None for
        zeros. None on any failure, which latches the detector off — the
        input was already validated by the caller."""
        import numpy as np
        try:
            win = x.reshape(-1, WINDOW)
            ctx = np.zeros((win.shape[0], CONTEXT), dtype=np.float32)
            if win.shape[0] > 1:
                ctx[1:] = win[:-1, -CONTEXT:]
            if ctx0 is not None and len(ctx0) == CONTEXT:
                ctx[0] = ctx0
            batch = np.ascontiguousarray(
                np.concatenate([ctx, win], axis=1), dtype=np.float32)
            h = np.zeros((1, 1, 128), dtype=np.float32)
            c = np.zeros((1, 1, 128), dtype=np.float32)
        except Exception:
            return None   # a shape problem in THIS clip, not the detector
        with self._lock:
            sess = self._session()
            if sess is None:
                return None
            try:
                out = sess.run(None, {"input": batch, "h": h, "c": c})
                probs = np.asarray(out[0], dtype=np.float32).reshape(-1)
            except Exception as e:
                self._latch(f"run failed: {type(e).__name__}: {e}")
                return None
        if probs.shape[0] != win.shape[0] or not np.all(np.isfinite(probs)):
            with self._lock:
                self._latch(f"bad output: {probs.shape[0]} probabilities for "
                            f"{win.shape[0]} windows")
            return None
        return probs

    def _last_speech_end(self, a, scan_samples: int) -> "int | None":
        """Sample index just past the last speech window within the clip's
        last `scan_samples` (window-aligned to the clip END, so no padding
        is needed); None when there is none or the detector failed."""
        import numpy as np
        n = len(a)
        scan = min(n, int(scan_samples)) // WINDOW * WINDOW
        if scan < WINDOW:
            return None
        start = n - scan
        ctx0 = a[start - CONTEXT:start] if start >= CONTEXT else None
        probs = self._probs(a[start:], ctx0)
        if probs is None:
            return None
        hits = np.nonzero(probs >= SPEECH_THRESHOLD)[0]
        if hits.size == 0:
            return None
        return start + (int(hits[-1]) + 1) * WINDOW

    def speech_tail_ms(self, audio, sample_rate: int = SAMPLE_RATE
                       ) -> "int | None":
        """Clip end minus the end of the last speech window, in ms.

        Scans the last TAIL_SCAN_S (about 125 windows; roughly 15-40 ms of
        one CPU core). Only when that holds no speech at all — the owner went
        quiet more than 4 s before the break, e.g. noise kept the RMS gate
        open — does it scan the clip's last MAX_SCAN_S. None when the clip is
        not 16 kHz mono float audio, holds no speech window, or the detector
        is unusable. Never raises."""
        try:
            if self._failed or int(sample_rate) != SAMPLE_RATE:
                return None
            import numpy as np
            a = np.asarray(audio, dtype=np.float32)
            if a.ndim != 1:
                if a.ndim == 2 and 1 in a.shape:
                    a = a.reshape(-1)
                else:
                    return None
            n = len(a)
            if n < WINDOW:
                return None
            end = self._last_speech_end(a, int(TAIL_SCAN_S * SAMPLE_RATE))
            if end is None and not self._failed \
                    and n > int(TAIL_SCAN_S * SAMPLE_RATE):
                end = self._last_speech_end(a, int(MAX_SCAN_S * SAMPLE_RATE))
            if end is None:
                return None
            return int(round((n - end) * 1000.0 / SAMPLE_RATE))
        except Exception:
            return None
