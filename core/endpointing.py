"""core/endpointing.py — where the owner's speech really ended (speed plan R1)
and whether the owner has finished the turn (R7).

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

The R1 part is telemetry only. Nothing in it can change a turn:
  * speech_tail_ms() never raises and returns None for anything it cannot
    measure; a load or run failure LATCHES the detector off for the session
    (``failed`` names why), so a broken install costs one attempt, not one per
    turn;
  * no I/O besides reading the model file once.

R7 (speed plan, Smart Turn v3.2) adds the one part that CAN change a turn:
EotDecider may end a capture before the 21-chunk hangover, in SMART_TURN_MODE
'on' only, and only when Silero (SileroStream, streamed over the live capture)
and Smart Turn (SmartTurn, P(the owner has finished) from the last 8 s) both
agree. Either model missing, failing or latched off = no early end, exactly
today's turn. SmartTurn follows the same latch-off rules as SileroVad, plus
two of its own (too slow, an out-of-range p).

CI SAFETY: numpy, onnxruntime and faster_whisper are imported lazily inside
the methods (the rule at core/kokoro_tts.py's CI SAFETY note), and the Silero
model is located with importlib.util.find_spec, which never imports
faster_whisper. Importing this module therefore pulls in nothing beyond the
stdlib (tests/test_endpointing.py pins that); EotDecider itself is pure Python
and never imports anything.

R6 (Parakeet, STT_ENGINE='parakeet') asks one more question of the same
detector, speech_in_head(): does the clip's first 0.8 s hold speech? Its
wake-word rescue (core/stt_parakeet.rescue_reason) uses the answer; nothing
else does, and None (cannot tell) makes the rescue run. Like
speech_tail_ms() it never raises and follows the same latch-off rules. It is
asked only when STT_ENGINE is 'parakeet'.
"""
from __future__ import annotations

import importlib.util
import math
import os
import threading
import time

SAMPLE_RATE = 16000        # Silero's rate; record_speech captures at 16 kHz
WINDOW = 512               # samples per Silero v6 window (32 ms)
CONTEXT = 64               # samples of the previous window each one carries
SPEECH_THRESHOLD = 0.5     # Silero's stock speech threshold (R7 uses it too)
TAIL_SCAN_S = 4.0          # the usual scan: the clip's last 4 s
MAX_SCAN_S = 30.0          # the fallback whole-clip scan never exceeds this
MODEL_FILE = "silero_vad_v6.onnx"

# ─── R7: Smart Turn end of turn ─────────────────────────────────────────
MODES = ("off", "shadow", "on")   # SMART_TURN_MODE
CHUNK_S = 1024 / SAMPLE_RATE      # one record_speech capture chunk (64 ms)
WINDOW_S = WINDOW / SAMPLE_RATE   # one Silero window (32 ms)
SPEECH_EXIT = 0.35         # hysteresis: speech enters at SPEECH_THRESHOLD
                           # and only leaves below this (Silero: threshold-0.15)
ST_SILERO_SILENCE_S = 0.2  # a Silero silence run before any check
ST_CHECK_GAP = 5           # capture chunks between two checks, at least
ST_MAX_CHECKS = 8          # Smart Turn checks per turn, at most
ST_SECONDS = 8             # Smart Turn hears the capture's last 8 s
ST_SAMPLES = ST_SECONDS * SAMPLE_RATE   # 128,000
ST_MELS = 80
ST_FRAMES = 800            # mel frames in 8 s at hop 160
ST_INPUT = "input_features"
ST_OUTPUT = "logits"       # [batch, 1], already a probability (sigmoid)
ST_BUDGET_S = 0.150        # one call (features + run) must fit inside this
ST_SLOW_LIMIT = 3          # runtime calls over budget in a row -> latch off
ST_INTRA_THREADS = 4


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


def default_session(path: str, intra: int = 1):
    """An onnxruntime CPU session for the Silero model: one intra-op and one
    inter-op thread, spin-wait off, no arena — a ~2 MB side session that
    never competes with Kokoro or the ambient decoder for cores. `intra`
    raises the intra-op threads (smart_turn_session)."""
    import onnxruntime as ort   # lazy: CI never installs it
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra)
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


def smart_turn_session(path: str):
    """default_session for Smart Turn: four intra-op threads (a ~9 MB
    Whisper-tiny encoder over 8 s must answer inside ST_BUDGET_S), still one
    inter-op thread and no spin-wait."""
    return default_session(path, intra=ST_INTRA_THREADS)


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

    def speech_in_head(self, audio, head_s: float = 0.8,
                       sample_rate: int = SAMPLE_RATE) -> "bool | None":
        """Does the clip's first `head_s` seconds hold speech? True when any
        whole window there scores at or above SPEECH_THRESHOLD, False when
        none does. None when the clip is not 16 kHz mono float audio, holds
        less than one window, or the detector is unusable. Speed plan R6's
        wake-word rescue (core/stt_parakeet.rescue_reason) asks this: did the
        owner start talking right away, as he does when he says "JARVIS"?
        About 25 windows for 0.8 s. Never raises."""
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
            n = min(len(a), int(float(head_s) * SAMPLE_RATE)) // WINDOW * WINDOW
            if n < WINDOW:
                return None
            probs = self._probs(np.ascontiguousarray(a[:n]), None)
            if probs is None:
                return None
            return bool(np.any(probs >= SPEECH_THRESHOLD))
        except Exception:
            return None

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


class SileroStream(SileroVad):
    """Streaming Silero over the live capture (R7): feed() each chunk as
    record_speech captures it and get one speech probability per completed
    WINDOW, with the recurrent state (h, c) and the CONTEXT samples carried
    across calls exactly as one long clip would carry them. Samples short of
    a whole window wait for the next chunk. reset() at the start of every
    turn.

    Its OWN onnxruntime session (one per instance, SileroVad's loader): the
    R1 tail probe scores a finished clip on a daemon while the next capture is
    already streaming, and the two must never share a recurrent state or wait
    on each other's run. The same latch-off rules as SileroVad; once latched,
    feed() returns None for the rest of the session."""

    def __init__(self, model_path: "str | None" = None, session_factory=None):
        super().__init__(model_path, session_factory)
        self.reset()

    def reset(self) -> None:
        """Forget the previous turn: zero state, zero context, nothing
        pending. Loads nothing and imports nothing (mode 'off' calls it)."""
        with self._lock:
            self._h = None        # None = zeros on the next run
            self._c = None
            self._ctx = None
            self._pend = None     # samples short of a whole window

    def feed(self, chunk, sample_rate: int = SAMPLE_RATE):
        """Speech probabilities (a list of floats; empty when `chunk` did not
        complete a window) for the windows `chunk` completes. None when the
        chunk is not 16 kHz mono audio or the detector is unusable (latched
        off: a load or run failure, or a bad output). Never raises."""
        try:
            if self._failed or int(sample_rate) != SAMPLE_RATE:
                return None
            import numpy as np
            a = np.asarray(chunk, dtype=np.float32)
            if a.ndim != 1:
                if a.ndim == 2 and 1 in a.shape:
                    a = a.reshape(-1)
                else:
                    return None
            with self._lock:
                sess = self._session()
                if sess is None:
                    return None
                if self._pend is not None and len(self._pend):
                    a = np.concatenate([self._pend, a])
                nwin = len(a) // WINDOW
                self._pend = a[nwin * WINDOW:].copy()
                if nwin == 0:
                    return []
                win = a[:nwin * WINDOW].reshape(nwin, WINDOW)
                ctx = np.zeros((nwin, CONTEXT), dtype=np.float32)
                if self._ctx is not None:
                    ctx[0] = self._ctx
                if nwin > 1:
                    ctx[1:] = win[:-1, -CONTEXT:]
                batch = np.ascontiguousarray(
                    np.concatenate([ctx, win], axis=1), dtype=np.float32)
                z = np.zeros((1, 1, 128), dtype=np.float32)
                h = self._h if self._h is not None else z
                c = self._c if self._c is not None else z
                try:
                    out = sess.run(None, {"input": batch, "h": h, "c": c})
                    probs = np.asarray(out[0], dtype=np.float32).reshape(-1)
                    h2 = np.asarray(out[1], dtype=np.float32)
                    c2 = np.asarray(out[2], dtype=np.float32)
                except Exception as e:
                    self._latch(f"run failed: {type(e).__name__}: {e}")
                    return None
                if (probs.shape[0] != nwin or not np.all(np.isfinite(probs))
                        or h2.shape != z.shape or c2.shape != z.shape):
                    self._latch(f"bad output: {probs.shape[0]} probabilities "
                                f"for {nwin} windows, state {h2.shape}")
                    return None
                self._h, self._c = h2, c2
                self._ctx = win[-1, -CONTEXT:].copy()
                return [float(p) for p in probs]
        except Exception:
            return None


def whisper_features():
    """faster-whisper's numpy log-mel extractor sized for Smart Turn (80
    mels, hop 160, n_fft 400, 8 s): a callable ST_SAMPLES floats -> [80, 800].
    padding=0: the input is already exactly 8 s (the default pads 160 more
    samples, one frame too many). Imports faster_whisper — call it lazily."""
    from faster_whisper.feature_extractor import FeatureExtractor
    fe = FeatureExtractor(feature_size=ST_MELS, sampling_rate=SAMPLE_RATE,
                          hop_length=160, chunk_length=ST_SECONDS, n_fft=400)
    return lambda x: fe(x, padding=0)


def smart_turn_input(a):
    """The model's waveform from a float32 16 kHz mono clip: its last
    ST_SECONDS, zero-padded on the LEFT to exactly ST_SAMPLES (the end of the
    turn sits at the end, as in training), then zero-mean / unit-variance over
    all ST_SAMPLES (Whisper's do_normalize, applied after the padding as the
    reference inference code does)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float32)[-ST_SAMPLES:]
    if len(a) < ST_SAMPLES:
        a = np.concatenate([np.zeros(ST_SAMPLES - len(a), np.float32), a])
    return ((a - a.mean()) / np.sqrt(a.var() + 1e-7)).astype(np.float32)


def _smart_turn_io_problem(sess) -> str:
    """'' when `sess` looks like Smart Turn v3: one input ST_INPUT shaped
    [batch, 80, 800] and a first output ST_OUTPUT shaped [batch, 1]; else
    what is wrong."""
    try:
        ins, outs = list(sess.get_inputs()), list(sess.get_outputs())
        if len(ins) != 1 or ins[0].name != ST_INPUT:
            return f"inputs {[i.name for i in ins]}, want ['{ST_INPUT}']"
        if list(ins[0].shape)[1:] != [ST_MELS, ST_FRAMES] \
                or len(ins[0].shape) != 3:
            return f"input shape {list(ins[0].shape)}"
        if not outs or outs[0].name != ST_OUTPUT:
            return f"outputs {[o.name for o in outs]}, want ['{ST_OUTPUT}']"
        if list(outs[0].shape)[1:] != [1] or len(outs[0].shape) != 2:
            return f"output shape {list(outs[0].shape)}"
    except Exception as e:
        return f"no I/O metadata: {type(e).__name__}: {e}"
    return ""


class SmartTurn:
    """Smart Turn v3.2 (pipecat-ai/smart-turn-v3): P(the owner has finished
    the turn) from the capture so far. Thread-safe like SileroVad (one lock
    serialises the lazy load and every run); construction does nothing.

    Latches OFF for the session (``failed`` says why; predict() then returns
    None, which EotDecider treats as "no early end") on: a load failure (no
    model file, onnxruntime or faster-whisper missing, I/O names or shapes
    that are not Smart Turn v3's); a run failure; a non-finite or
    out-of-range p; a warm call over ST_BUDGET_S; ST_SLOW_LIMIT runtime calls
    over ST_BUDGET_S in a row. The call that latches returns None too.

    `session_factory(path)` defaults to smart_turn_session, `features`
    (ST_SAMPLES floats -> [80, 800]) to whisper_features(), `clock` to
    time.perf_counter — tests pass fakes for all three."""

    def __init__(self, model_path: "str | None" = None, session_factory=None,
                 features=None, clock=None):
        self._path = model_path
        self._factory = session_factory if session_factory is not None \
            else smart_turn_session
        self._features = features
        self._clock = clock if clock is not None else time.perf_counter
        self._sess = None
        self._failed = ""
        self._slow = 0            # runtime calls over budget in a row
        self._lock = threading.Lock()

    @property
    def failed(self) -> str:
        """Why Smart Turn latched off ('' while usable)."""
        return self._failed

    def _latch(self, why: str) -> None:
        self._sess = None
        self._failed = why or "failed"

    def _session(self):
        """Caller holds self._lock. The session (and the feature extractor),
        loading and checking them once; None once latched off."""
        if self._failed:
            return None
        if self._sess is None:
            if not self._path:
                self._latch("no model path (SMART_TURN_MODEL)")
                return None
            try:
                if self._features is None:
                    self._features = whisper_features()
                sess = self._factory(self._path)
            except Exception as e:
                self._latch(f"load failed: {type(e).__name__}: {e}")
                return None
            if sess is None:
                self._latch("load failed: no session")
                return None
            why = _smart_turn_io_problem(sess)
            if why:
                self._latch(f"not a Smart Turn v3 model: {why}")
                return None
            self._sess = sess
        return self._sess

    def _run(self, a, kind: str):
        """p for clip `a` (float32 mono 16 kHz). `kind`: 'prime' (untimed),
        'warm' (one call over budget latches) or 'turn' (ST_SLOW_LIMIT in a
        row latch). None on any failure, which latches."""
        import numpy as np
        with self._lock:
            sess = self._session()
            if sess is None:
                return None
            t0 = self._clock()
            try:
                f = np.asarray(self._features(smart_turn_input(a)),
                               dtype=np.float32)
                if f.size != ST_MELS * ST_FRAMES \
                        or f.shape[-2:] != (ST_MELS, ST_FRAMES):
                    self._latch(f"bad features: shape {f.shape}")
                    return None
                out = sess.run(None, {ST_INPUT: np.ascontiguousarray(
                    f.reshape(1, ST_MELS, ST_FRAMES))})
                p = np.asarray(out[0], dtype=np.float64).reshape(-1)
            except Exception as e:
                self._latch(f"run failed: {type(e).__name__}: {e}")
                return None
            dt = self._clock() - t0
            if p.shape != (1,) or not 0.0 <= float(p[0]) <= 1.0:
                self._latch(f"bad output: {p[:2].tolist()} (shape {p.shape})")
                return None
            ms = int(round(dt * 1000))
            if kind == "warm" and dt > ST_BUDGET_S:
                self._latch(f"too slow: warm call {ms} ms > "
                            f"{int(ST_BUDGET_S * 1000)} ms")
                return None
            if kind == "turn":
                self._slow = self._slow + 1 if dt > ST_BUDGET_S else 0
                if self._slow >= ST_SLOW_LIMIT:
                    self._latch(f"too slow: {ST_SLOW_LIMIT} calls in a row "
                                f"over {int(ST_BUDGET_S * 1000)} ms "
                                f"(last {ms} ms)")
                    return None
            return float(p[0])

    def warm(self) -> bool:
        """Load the model, check its I/O names and shapes, and run it on 8 s
        of zeros twice: the first run is onnxruntime's slow one (allocation),
        the second is the timed warm call — over ST_BUDGET_S latches Smart
        Turn off (this box cannot run it inside a turn). Raises RuntimeError
        when unusable, so a boot warmer reports it; True otherwise."""
        try:
            import numpy as np
            z = np.zeros(ST_SAMPLES, dtype=np.float32)
        except Exception as e:
            self._latch(f"numpy unavailable: {e}")
            raise RuntimeError(self._failed)
        for kind in ("prime", "warm"):
            if self._run(z, kind) is None:
                raise RuntimeError(self._failed or "failed")
        return True

    def predict(self, audio, sample_rate: int = SAMPLE_RATE
                ) -> "float | None":
        """P(turn complete) in [0, 1] for the capture so far (16 kHz mono;
        only the last ST_SECONDS count). None when the clip is not 16 kHz
        mono audio or is empty (never latches), or Smart Turn is unusable.
        Never raises."""
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
            if a.size == 0:
                return None
            return self._run(a, "turn")
        except Exception:
            return None


def _windows(seconds: float, unit_s: float) -> int:
    """Whole units of `unit_s` that reach `seconds` (at least 1)."""
    return max(1, int(math.ceil(float(seconds) / unit_s - 1e-9)))


class EotRecord:
    """One finished turn's end-of-turn verdict (EotDecider.record()).

    eot        'st' when Smart Turn ended the turn, else the caller's reason
               ('rms': the fixed 21-chunk hangover).
    st_p       the deciding p (the would-fire p in shadow), else the last p
               measured this turn; None when no check produced one.
    st_n       Smart Turn checks run this turn.
    fire_ms    audio time from the turn's first fed chunk to the end of the
               chunk on which Smart Turn ended (or, in shadow, WOULD have
               ended) the turn; None when it never fired.
    resumed    1 when the capture heard voice again after fire_ms (its RMS
               silence count restarted): 'on' would have cut the turn there.
    actual_ms  audio time from the first fed chunk to the last one: where the
               capture really ended. On a resumed=0 shadow turn, actual_ms -
               fire_ms is what 'on' would have saved."""

    __slots__ = ("mode", "eot", "st_p", "st_n", "fire_ms", "resumed",
                 "actual_ms")

    def __init__(self, mode, eot, st_p, st_n, fire_ms, resumed, actual_ms):
        self.mode = mode
        self.eot = eot
        self.st_p = st_p
        self.st_n = st_n
        self.fire_ms = fire_ms
        self.resumed = resumed
        self.actual_ms = actual_ms

    def stats(self) -> dict:
        """The [turn-timing] fields (TurnTiming.note_stat names)."""
        return {"eot": self.eot,
                "st_p": None if self.st_p is None else round(self.st_p, 3),
                "st_n": self.st_n}

    def shadow_line(self) -> "str | None":
        """'[eot-shadow] fire_ms= p= resumed=0|1 actual_ms=' in shadow mode
        (every turn, fired or not: '-' = never fired / never measured); None
        in any other mode."""
        if self.mode != "shadow":
            return None
        fire = "-" if self.fire_ms is None else str(self.fire_ms)
        p = "-" if self.st_p is None else f"{self.st_p:.3f}"
        return (f"[eot-shadow] fire_ms={fire} p={p} "
                f"resumed={self.resumed} actual_ms={self.actual_ms}")


class EotDecider:
    """Whether to end the owner's turn before the fixed RMS hangover (R7).

    PURE: no model, no numpy, no I/O, no imports. The models come in as
    callables: `vad(chunk)` -> that chunk's per-window Silero speech
    probabilities (SileroStream.feed; None = Silero unavailable) and
    `predict()` -> Smart Turn's p for the capture so far (a closure over
    the clip calling SmartTurn.predict; None = unavailable). Mode 'off' calls
    neither, ever.

    The caller builds one per turn (or reset()s it), calls update(chunk,
    silence_n) for EVERY chunk it appends to the clip while recording, voiced
    or not, after updating its own silence count, and breaks when update()
    returns True. Its own `silence_n >= 21` break stays: today's worst case,
    reported as 'rms'. At the end, record() gives the [eot-shadow] line and
    the turn stats.

    A Smart Turn check runs only when ALL of these hold:
      * silence_n >= SMART_TURN_MIN_SILENCE_S of chunks (4 x 64 ms);
      * Silero is in a silence run of >= ST_SILERO_SILENCE_S (hysteresis:
        speech starts at p >= SPEECH_THRESHOLD, ends at p < SPEECH_EXIT);
      * Silero has heard >= SMART_TURN_MIN_SPEECH_S of speech this turn;
      * >= ST_CHECK_GAP chunks since the last check, and fewer than
        ST_MAX_CHECKS checks this turn so far.
    p >= threshold fires: 'on' ends the turn, 'shadow' only records it (and
    stops checking: 'on' would have ended there). Below threshold it keeps
    waiting. Silero unavailable (vad None, a None or out-of-range result at
    any point this turn) or Smart Turn unavailable (a None, raising or
    out-of-range predict) = it NEVER ends this turn early: RMS silence alone
    is never enough. Never raises."""

    def __init__(self, mode: str = "shadow", vad=None, predict=None, *,
                 threshold: float = 0.7, min_silence_s: float = 0.256,
                 min_speech_s: float = 1.0, chunk_s: float = CHUNK_S):
        m = str(mode).strip().lower()
        self.mode = m if m in MODES else "off"
        self._vad = vad
        self._predict = predict
        self._threshold = float(threshold)
        self._chunk_ms = float(chunk_s) * 1000.0
        self._min_sil = _windows(min_silence_s, float(chunk_s))
        self._sil_windows = _windows(ST_SILERO_SILENCE_S, WINDOW_S)
        self._speech_windows = _windows(min_speech_s, WINDOW_S)
        self.reset()

    def reset(self) -> None:
        """Start a new turn."""
        self._chunks = 0          # chunks fed this turn
        self._speech = False      # Silero hysteresis state
        self._voiced = 0          # windows in the speech state this turn
        self._sil_run = 0         # windows in the current silence run
        self._vad_ok = self._vad is not None
        self._st_ok = self._predict is not None
        self._last_check = None   # self._chunks at the last check
        self._n = 0               # Smart Turn checks this turn
        self._p = None            # the latest p
        self._fire = None         # self._chunks when p >= threshold
        self._resumed = 0

    def should_end(self) -> bool:
        """True once Smart Turn fired in mode 'on' (never in 'shadow')."""
        return self.mode == "on" and self._fire is not None

    def update(self, chunk, silence_n: int) -> bool:
        """Feed one captured chunk and the capture loop's RMS silence count
        after it; returns should_end(). Never raises."""
        try:
            if self.mode == "off" or self.should_end():
                return self.should_end()
            self._chunks += 1
            if self._fire is not None:
                if silence_n <= 0:
                    self._resumed = 1
                return False      # shadow: 'on' would have ended already
            if self._vad_ok:
                self._listen(chunk)
            if self._due(silence_n):
                self._check()
        except Exception:
            self._vad_ok = False
        return self.should_end()

    def _listen(self, chunk) -> None:
        try:
            probs = self._vad(chunk)
            if probs is None:
                self._vad_ok = False
                return
            for p in probs:
                p = float(p)
                if not 0.0 <= p <= 1.0:
                    self._vad_ok = False
                    return
                if self._speech:
                    self._speech = p >= SPEECH_EXIT
                else:
                    self._speech = p >= SPEECH_THRESHOLD
                if self._speech:
                    self._voiced += 1
                    self._sil_run = 0
                else:
                    self._sil_run += 1
        except Exception:
            self._vad_ok = False

    def _due(self, silence_n) -> bool:
        return (self._vad_ok and self._st_ok
                and silence_n >= self._min_sil
                and self._sil_run >= self._sil_windows   # Silero: silent
                and self._voiced >= self._speech_windows
                and self._n < ST_MAX_CHECKS
                and (self._last_check is None
                     or self._chunks - self._last_check >= ST_CHECK_GAP))

    def _check(self) -> None:
        self._last_check = self._chunks
        self._n += 1
        try:
            p = self._predict()
            p = None if p is None else float(p)
        except Exception:
            p = None
        if p is None or not 0.0 <= p <= 1.0:
            self._st_ok = False   # unavailable: no early end this turn
            return
        self._p = p
        if p >= self._threshold:
            self._fire = self._chunks

    def record(self, eot: str = "rms") -> EotRecord:
        """This turn's verdict so far (call it when the capture ended).
        `eot` names how the capture ended when Smart Turn did not end it."""
        return EotRecord(
            self.mode, "st" if self.should_end() else eot, self._p, self._n,
            None if self._fire is None
            else int(round(self._fire * self._chunk_ms)),
            self._resumed, int(round(self._chunks * self._chunk_ms)))
