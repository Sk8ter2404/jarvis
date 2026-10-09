"""core/wake_pregate.py - an openWakeWord score stream on the mic (the wake
pre-gate: A3 shadow, D1 trigger) and on what the PC plays (the loopback
veto, D2), 2026-10-05.

WHY THIS EXISTS
---------------
Over a video the capture that hears the owner usually started on the
video's loudness, so his "Jarvis" sits seconds into it and the positional
wake rule refuses the line. A wake-word DETECTOR scoring every 80 ms can
start the capture at the name instead (D1), and the same detector on the
loopback can tell when the video itself said "Jarvis" (D2).

The stock ``hey_jarvis_v0.1`` model is trained on "hey Jarvis"; the owner
mostly says "Jarvis, ...". Its recall for the plain name depends heavily on
the voice (synthetic set: 71 % at 0.5 for one voice, 12 % for three others),
so it is a SHADOW first (A3, WAKE_PREGATE_MODE 'shadow' - the setting ships
'off' since the 2026-10-09 review: loading it adds ~100 MB and a second
OpenMP runtime to the live process, so 'shadow' waits for its own canary): it
scores the mic
the main capture already hears (the record tap - no new stream) and logs
NUMBERS ONLY - for each accepted wake-word turn its highest score around the
capture start, and per media minute how many events each threshold would
have made. A week of that sets D1's threshold without recording anything.

Model files are on disk (openwakeword/resources/models). The ONNX graphs are
loaded explicitly (Model() defaults to tflite, which is not installed) on
the CPU execution provider with one intra-op and one inter-op thread each,
and openWakeWord's Silero gate (vad_threshold 0.5) zeroes scores without
speech. A load failure latches the detector off for the session with one
log line. Nothing here uses CUDA.

The worker is ONE never-exiting daemon thread per detector, fed through a
bounded queue that drops its oldest frame (a slow model can never stall the
capture callback that feeds it).
"""
from __future__ import annotations

import collections
import importlib.util
import os
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
FRAME = 1280                     # openWakeWord's 80 ms frame
MODEL_FILE = "hey_jarvis_v0.1.onnx"
MELSPEC_FILE = "melspectrogram.onnx"
EMBED_FILE = "embedding_model.onnx"
VAD_THRESHOLD = 0.5
QUEUE_MAX = 64                   # frames (~4 s of 1024-sample chunks)
GAIN_WINDOW_S = 2.0
# The capture's own auto-gain (core.config CAPTURE_AUTO_GAIN_*; the same
# defaults as bobert_companion.apply_capture_auto_gain).
GAIN_TARGET, GAIN_MAX, GAIN_FLOOR = 0.25, 10.0, 0.005


def model_dir() -> "str | None":
    """openWakeWord's bundled model folder, or None when not installed."""
    try:
        spec = importlib.util.find_spec("openwakeword")
        if spec is None or not spec.origin:
            return None
        d = os.path.join(os.path.dirname(spec.origin), "resources", "models")
        return d if os.path.isdir(d) else None
    except Exception:
        return None


def load_model():
    """openWakeWord's hey_jarvis model, ONNX on the CPU, one thread per
    session, Silero gate on. Raises when the package or files are
    missing."""
    d = model_dir()
    if d is None:
        raise RuntimeError("openwakeword is not installed")
    paths = [os.path.join(d, f) for f in (MODEL_FILE, MELSPEC_FILE,
                                         EMBED_FILE)]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    from openwakeword.model import Model
    return Model(wakeword_models=[paths[0]], inference_framework="onnx",
                 vad_threshold=VAD_THRESHOLD, melspec_model_path=paths[1],
                 embedding_model_path=paths[2], ncpu=1)


class StreamGain:
    """The capture auto-gain on a stream: the loudest frame RMS over the
    last GAIN_WINDOW_S stands in for the clip's peak; a quiet stream is
    lifted toward GAIN_TARGET (at most GAIN_MAX); silence (under
    GAIN_FLOOR) and loud audio are left alone - apply_capture_auto_gain's
    rule."""

    def __init__(self, target=GAIN_TARGET, max_gain=GAIN_MAX,
                 floor=GAIN_FLOOR, window_s=GAIN_WINDOW_S):
        self.target, self.max_gain, self.floor = target, max_gain, floor
        self._win = float(window_s)
        self._hist: "collections.deque[tuple[float, float]]" = \
            collections.deque()

    def gain(self, frame: np.ndarray, t: float) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) \
            if frame.size else 0.0
        self._hist.append((t, rms))
        while self._hist and t - self._hist[0][0] > self._win:
            self._hist.popleft()
        peak = max(r for _t, r in self._hist)
        if not (self.floor < peak < self.target):
            return 1.0
        return max(1.0, min(self.max_gain, self.target / max(peak, 1e-9)))


class Detector:
    """One model's 80 ms scores over a stream. feed(mono, t_end) -> a list
    of (t, score), one per complete 80 ms frame (t = monotonic time of the
    frame's last sample, from the chunk's arrival time). Not thread-safe:
    one worker owns it."""

    def __init__(self, model_factory=load_model, gain: bool = True):
        self._factory = model_factory
        self._model = None
        self.failed = ""
        self._buf = np.zeros(0, np.float32)
        self._gain = StreamGain() if gain else None

    def ready(self) -> bool:
        if self._model is not None:
            return True
        if self.failed:
            return False
        try:
            self._model = self._factory()
            return True
        except Exception as e:
            self.failed = f"{type(e).__name__}: {e}"[:200]
            return False

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        try:
            if self._model is not None:
                self._model.reset()
        except Exception:
            pass

    def feed(self, mono, t_end: float) -> list:
        if not self.ready():
            return []
        x = np.asarray(mono, np.float32).reshape(-1)
        if x.size == 0:
            return []
        g = self._gain.gain(x, t_end) if self._gain is not None else 1.0
        if g != 1.0:
            x = x * np.float32(g)
        self._buf = np.concatenate([self._buf, x])
        out = []
        n_frames = len(self._buf) // FRAME
        for i in range(n_frames):
            fr = self._buf[i * FRAME:(i + 1) * FRAME]
            pcm = np.clip(fr * 32767.0, -32768, 32767).astype(np.int16)
            pred = self._model.predict(pcm)
            try:
                score = float(next(iter(pred.values())))
            except Exception:
                score = 0.0
            # The frame ended (len(buf) - (i+1)*FRAME) samples before the
            # newest sample, which arrived at t_end.
            left = len(self._buf) - (i + 1) * FRAME
            out.append((t_end - left / SAMPLE_RATE, score))
        self._buf = self._buf[n_frames * FRAME:]
        return out


class PregateWorker:
    """A detector on its own never-exiting daemon thread.

    ``feed(mono, t)`` (any thread, cheap) queues a chunk; ``tap`` is a
    queue-like object for add_record_tap (it stamps each frame's arrival
    time). Every score goes to ``track`` (core/listen_media.ScoreTrack) and,
    when given, to ``trigger`` (core/listen_media.PregateTrigger) - a
    trigger calls ``on_trigger(t, score)``. ``active()`` False (the mode is
    'off') drops frames unscored."""

    def __init__(self, name: str, *, track, trigger=None, on_trigger=None,
                 detector=None, active=None, log=print, clock=time.monotonic):
        self.name = name
        self.track = track
        self.trigger = trigger
        self.on_trigger = on_trigger
        self.det = detector if detector is not None else Detector()
        self._active = active or (lambda: True)
        self._log = log
        self._clock = clock
        self._q: "collections.deque" = collections.deque()
        self._cv = threading.Condition()
        self.dropped = 0
        self.frames = 0
        self.cpu_s = 0.0
        self.audio_s = 0.0
        self._thread = None
        self._tlock = threading.Lock()
        self._said_failed = False
        self._stop = False
        self.tap = _StampedTap(self)

    def start(self) -> bool:
        with self._tlock:
            t = self._thread
            if t is not None and t.is_alive():
                return True
            try:
                t = threading.Thread(target=self._loop,
                                     name=f"wake-pregate-{self.name}",
                                     daemon=True)
                t.start()
                self._thread = t
                return True
            except Exception:
                return False

    def feed(self, mono, t: "float | None" = None) -> None:
        try:
            if t is None:
                t = self._clock()
            with self._cv:
                if len(self._q) >= QUEUE_MAX:
                    self._q.popleft()
                    self.dropped += 1
                self._q.append((mono, float(t)))
                self._cv.notify()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Let the worker end - tests and process exit only."""
        self._stop = True
        with self._cv:
            self._cv.notify_all()

    def _loop(self) -> None:
        while not self._stop:
            with self._cv:
                while not self._q:
                    if self._stop:
                        return
                    self._cv.wait(1.0)
                mono, t = self._q.popleft()
            try:
                if not self._active():
                    continue
                c0 = time.thread_time()
                scores = self.det.feed(mono, t)
                self.cpu_s += time.thread_time() - c0
                self.audio_s += len(mono) / SAMPLE_RATE
                if self.det.failed and not self._said_failed:
                    self._said_failed = True
                    try:
                        self._log(f"  [wake-pregate] {self.name} detector off "
                                  f"for this session ({self.det.failed})")
                    except Exception:
                        pass
                for ts, sc in scores:
                    self.frames += 1
                    self.track.add(ts, sc)
                    if self.trigger is not None and self.trigger.offer(ts, sc):
                        if self.on_trigger is not None:
                            try:
                                self.on_trigger(ts, sc)
                            except Exception:
                                pass
            except Exception:
                pass

    def status(self) -> dict:
        cps = (self.cpu_s / self.audio_s) if self.audio_s > 0 else None
        return {"frames": self.frames, "dropped": self.dropped,
                "failed": self.det.failed,
                "cpu_s_per_audio_s": None if cps is None else round(cps, 4)}


class _StampedTap:
    """What add_record_tap wants (put_nowait), stamping arrival time."""

    def __init__(self, worker: PregateWorker):
        self._w = worker

    def put_nowait(self, item) -> None:
        self._w.feed(item)

    def put(self, item, block=True, timeout=None) -> None:   # noqa: ARG002
        self._w.feed(item)
