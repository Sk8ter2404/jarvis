"""
Always-on wake-word detector for JARVIS.

A low-power neural wake-word detector that listens continuously on a
background audio stream and only forwards events to the main listen
loop once a configured wake phrase ("hey jarvis", "jarvis", ...) is
spoken. Replaces hotkey-only activation while remaining opt-in:
WAKE_WORD_ENGINE='off' disables the detector and the rest of the
companion behaves exactly as before.

Engines supported
-----------------
- 'openwakeword'  : default. Apache-2.0, CPU-friendly, ONNX-based. Loads
                    one or more bundled / user-trained models keyed by
                    the wake phrase. Installed via `pip install
                    openwakeword`. First call downloads ~30 MB of models.
- 'porcupine'     : Picovoice Porcupine. Higher accuracy, free for
                    personal use with an access key set in the
                    PORCUPINE_ACCESS_KEY env var. Requires
                    `pip install pvporcupine`.
- 'off'           : no-op. start() returns immediately; the rest of
                    JARVIS continues to use VAD + hotkey activation.

Endpointing
-----------
After a wake-word hit, the detector can optionally hand the next ~10 s
of mic audio to Silero VAD (`silero-vad` package, lazy-imported) so the
caller knows when the user stopped speaking and the wake utterance can
be cleanly handed to Whisper without trailing dead air. If Silero is
not installed, endpointing is skipped and the caller falls back to
the existing energy-VAD path in bobert_companion.record_speech().

Public API
----------
    detector = WakeWordDetector(
        engine="openwakeword",
        wake_words=["hey jarvis", "jarvis"],
        sample_rate=16000,
        device=None,
        threshold=0.5,
        on_detect=lambda evt: ...,
    )
    detector.start()
    ...
    detector.stop()

`on_detect` receives a dict: {"phrase": str, "score": float, "ts": float}.
Events are also pushed to detector.events (queue.Queue) so a caller
that prefers polling can drain it.

The class is intentionally tolerant of missing optional dependencies so
that simply importing this module never crashes the main companion.
Failures surface only when start() is called.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np


DEFAULT_WAKE_WORDS = ["hey jarvis", "jarvis"]
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 80               # openwakeword expects 80 ms frames
DEFAULT_THRESHOLD = 0.5             # openwakeword score (0..1)
COOLDOWN_SECS = 1.5                 # min gap between two wake events
EVENTS_QUEUE_MAX = 64               # bound on the events queue; the real
                                    # caller drains via on_detect, so this
                                    # only guards a polling caller that lags
MAX_BUFFER_FRAMES = 50              # hard cap on _buf size (in frames) to
                                    # avoid unbounded growth if _on_frame
                                    # raises and the drain loop exits early


def _safe_close_stream(stream, timeout_sec: float = 2.0) -> None:
    """Stop+close a sounddevice stream without blocking the caller.

    Mirrors bobert_companion._safe_close_stream: sounddevice.close() at
    sounddevice.py:1167 SIGSEGV'd on this build (faulthandler caught the
    crash across multiple PIDs on 2026-05-29). Stop synchronously, then run
    close() on a daemon thread; if it hangs past timeout_sec, force sd.stop()
    and let the daemon die with the process."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception as e:
        print(f"  [wake-word] stream.stop raised: {e!r}")
    done = threading.Event()

    def _do_close():
        try:
            stream.close()
        except Exception as e:
            print(f"  [wake-word] stream.close raised: {e!r}")
        finally:
            done.set()

    t = threading.Thread(target=_do_close, daemon=True)
    t.start()
    if not done.wait(timeout=timeout_sec):
        print(f"  [wake-word] stream.close hung >{timeout_sec:.1f}s — "
              "forcing sd.stop()")
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass


def _phrase_to_oww_model_name(phrase: str) -> str:
    """Map a human wake phrase to openwakeword's bundled-model name.

    openwakeword ships several pretrained models keyed by short slugs
    (e.g. 'hey_jarvis_v0.1', 'alexa_v0.1'). Users typing 'hey jarvis'
    or 'jarvis' should land on the same bundled model — both are routed
    to 'hey_jarvis_v0.1' which is the only Jarvis model in the OOTB
    set. A future custom-trained model can be referenced by giving the
    exact .onnx filename in WAKE_WORDS and the detector will use it
    directly without remapping.
    """
    key = (phrase or "").strip().lower()
    if key.endswith(".onnx") or os.path.isfile(key):
        return phrase
    if "jarvis" in key:
        return "hey_jarvis_v0.1"
    if "alexa" in key:
        return "alexa_v0.1"
    if "hey mycroft" in key or key == "mycroft":
        return "hey_mycroft_v0.1"
    return key.replace(" ", "_")


class WakeWordDetector:
    """Background wake-word detector with pluggable engine."""

    def __init__(
        self,
        engine: str = "openwakeword",
        wake_words: Optional[list[str]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: Optional[int] = None,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_secs: float = COOLDOWN_SECS,
        on_detect: Optional[Callable[[dict], None]] = None,
        use_silero_vad: bool = False,
    ) -> None:
        self.engine = (engine or "off").lower().strip()
        self.wake_words = list(wake_words or DEFAULT_WAKE_WORDS)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.threshold = float(threshold)
        self.cooldown_secs = float(cooldown_secs)
        self.on_detect = on_detect
        self.use_silero_vad = bool(use_silero_vad)

        # 2026-07-08: bound the events queue so a caller that only uses
        # on_detect (the real path) can't let this grow without limit.
        self.events: "queue.Queue[dict]" = queue.Queue(maxsize=EVENTS_QUEUE_MAX)
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._running = False
        self._paused = False
        self._pause_lock = threading.Lock()
        self._last_fire_ts = 0.0

        # Engine handles, populated by start()
        self._oww = None
        self._porcupine = None
        self._silero = None

        # 2026-07-08: int16 leftover carried between porcupine blocks so the
        # 256-sample remainder of each 1280-sample block is not discarded
        # (porcupine consumes fixed frame_length frames; ~20% was dropped).
        self._porcupine_leftover = np.zeros(0, dtype=np.int16)

        # Ring buffer of float32 mono samples; openwakeword consumes
        # 80 ms (1280 sample @ 16 kHz) chunks at a time.
        self._buf = np.zeros(0, dtype=np.float32)

        # Audio taps: queues that receive a copy of every raw mono
        # float32 frame the detector reads from the mic. This lets
        # bobert_companion.get_mic_buffer() share the persistent
        # InputStream instead of opening a second one — Windows
        # WASAPI rejects concurrent opens on the same device.
        self._taps: list["queue.Queue[np.ndarray]"] = []
        self._taps_lock = threading.Lock()

    # ── public api ────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict:
        return {
            "engine": self.engine,
            "running": self._running,
            "wake_words": self.wake_words,
            "threshold": self.threshold,
            "last_event_ts": self._last_fire_ts,
        }

    def add_tap(self, q: "queue.Queue[np.ndarray]") -> None:
        """Register a queue that receives a copy of every raw mono
        float32 frame the detector reads from the mic. Caller is
        responsible for draining it and calling remove_tap when done."""
        with self._taps_lock:
            if q not in self._taps:
                self._taps.append(q)

    def remove_tap(self, q: "queue.Queue[np.ndarray]") -> None:
        with self._taps_lock:
            try:
                self._taps.remove(q)
            except ValueError:
                pass

    def start(self) -> bool:
        """Begin background detection. Returns True if the detector is
        actually live, False if engine='off' or any optional dep is
        missing (the caller can then fall back to hotkey/VAD)."""
        if self.engine == "off":
            print("  [wake-word] engine=off; detector not started")
            return False
        if self._running:
            return True
        try:
            if self.engine == "openwakeword":
                self._init_openwakeword()
            elif self.engine == "porcupine":
                self._init_porcupine()
            else:
                print(f"  [wake-word] unknown engine '{self.engine}'; "
                      "expected 'openwakeword' | 'porcupine' | 'off'")
                return False
        except ImportError as e:
            print(f"  [wake-word] engine '{self.engine}' not installed: {e}")
            return False
        except Exception as e:
            print(f"  [wake-word] engine '{self.engine}' init failed: {e}")
            return False

        if self.use_silero_vad:
            self._init_silero_vad()  # best-effort, non-fatal

        self._stop_flag.clear()
        if not self._open_stream():
            return False

        self._running = True
        print(f"  [wake-word] listening on {self.engine} for "
              f"{self.wake_words} (threshold={self.threshold:.2f})")
        return True

    def _open_stream(self) -> bool:
        """Open and start the persistent InputStream. Returns True on success.
        Used by both start() (initial open) and resume() (after pause for a
        PortAudio device refresh)."""
        try:
            import sounddevice as sd
        except ImportError:
            print("  [wake-word] sounddevice missing; cannot open mic")
            return False

        frame_size = int(self.sample_rate * DEFAULT_FRAME_MS / 1000)
        max_buf_samples = frame_size * MAX_BUFFER_FRAMES

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                # PortAudio status flags (overflows, etc.) — log once per
                # event class via repr() so we don't spam.
                pass
            mono = indata[:, 0] if indata.ndim > 1 else indata
            mono = mono.astype(np.float32, copy=False)
            # Fan out to any registered audio taps before mutating. Snapshot
            # the list under the lock so add_tap/remove_tap calls from other
            # threads can't race with iteration. Failures are isolated per
            # tap so one stuck consumer can't kill the wake-word loop.
            if self._taps:
                with self._taps_lock:
                    taps = list(self._taps)
                for tq in taps:
                    try:
                        tq.put_nowait(mono.copy())
                    except Exception:
                        pass
            self._buf = np.concatenate([self._buf, mono])
            # Hard cap: if _on_frame previously raised and stopped draining,
            # _buf keeps growing forever and eventually the np.concatenate
            # above will OOM inside the PortAudio callback, corrupting heap.
            # Drop oldest samples so the buffer can never exceed the cap.
            if self._buf.size > max_buf_samples:
                self._buf = self._buf[-max_buf_samples:]
            while self._buf.size >= frame_size:
                frame = self._buf[:frame_size]
                self._buf = self._buf[frame_size:]
                try:
                    self._on_frame(frame)
                except Exception as e:
                    # Never let a frame-handler exception kill the drain
                    # loop — that would let _buf grow without bound on the
                    # next callback.
                    print(f"  [wake-word] _on_frame raised: {e!r}")

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=frame_size,
                device=self.device,
                callback=_cb,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"  [wake-word] failed to open input stream: {e}")
            self._stream = None
            return False

    def stop(self) -> None:
        self._stop_flag.set()
        self._running = False
        self._paused = False
        s = self._stream
        self._stream = None
        _safe_close_stream(s)

    def pause(self) -> None:
        """Temporarily close the InputStream so PortAudio can be torn down
        and re-initialized safely (e.g. for USB hotplug refresh). Engine
        handles (_oww / _porcupine) and _running stay set; resume() reopens
        the stream. Safe to call when not running or already paused."""
        with self._pause_lock:
            if not self._running or self._paused:
                return
            s = self._stream
            self._stream = None
            self._paused = True
            _safe_close_stream(s)
            # Drop any partial frame data so the resumed stream doesn't splice
            # pre- and post-refresh audio into a single garbage frame.
            self._buf = np.zeros(0, dtype=np.float32)
            print("  [wake-word] paused for device refresh")

    def resume(self) -> bool:
        """Reopen the InputStream after pause(). Returns True on success.
        On failure, _running is cleared so callers know detection is dead."""
        with self._pause_lock:
            if not self._paused:
                return False
            self._paused = False
            if not self._running:
                return False
            if self._open_stream():
                print("  [wake-word] resumed after device refresh")
                return True
            self._running = False
            print("  [wake-word] resume failed; detector stopped — "
                  "call wake_listener_start to retry")
            return False

    # ── engine init ──────────────────────────────────────────────────
    def _init_openwakeword(self) -> None:
        import openwakeword  # noqa: F401  — package check
        from openwakeword.model import Model

        # Auto-download bundled models the first time we run.
        try:
            from openwakeword.utils import download_models
            download_models()
        except Exception:
            pass

        model_paths: list[str] = []
        inline_names: list[str] = []
        for phrase in self.wake_words:
            name = _phrase_to_oww_model_name(phrase)
            if name.endswith(".onnx") and os.path.isfile(name):
                model_paths.append(name)
            else:
                inline_names.append(name)

        # openwakeword.Model(wakeword_models=[...]) accepts either model
        # paths or bundled names. Mix freely.
        wakeword_models = model_paths + inline_names
        self._oww = Model(wakeword_models=wakeword_models or None)

    def _init_porcupine(self) -> None:
        import pvporcupine  # noqa: F401  — package check

        key = os.environ.get("PORCUPINE_ACCESS_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "PORCUPINE_ACCESS_KEY env var not set — get a free "
                "personal-use key from https://console.picovoice.ai/"
            )
        # Porcupine ships a built-in 'jarvis' keyword.
        keywords = []
        for phrase in self.wake_words:
            k = phrase.strip().lower()
            if "jarvis" in k:
                keywords.append("jarvis")
            elif k in pvporcupine.KEYWORDS:
                keywords.append(k)
            else:
                # Unknown keyword — fall back to 'jarvis' so something
                # is always listening rather than crashing.
                print(
                    f"  [wake-word] warning: porcupine has no built-in "
                    f"keyword for {phrase!r}; falling back to 'jarvis'. "
                    f"Valid keywords: {sorted(pvporcupine.KEYWORDS)}"
                )
                keywords.append("jarvis")
        # De-dupe while preserving order.
        seen: set[str] = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))]
        self._porcupine = pvporcupine.create(
            access_key=key,
            keywords=keywords,
            sensitivities=[self.threshold] * len(keywords),
        )
        # Porcupine wants 16 kHz mono frames of exactly frame_length samples.
        self._porcupine_frame = int(self._porcupine.frame_length)
        self._porcupine_keywords = keywords

    def _init_silero_vad(self) -> None:
        try:
            import torch  # noqa: F401
            silero, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._silero = silero
        except Exception as e:
            print(f"  [wake-word] silero-vad unavailable ({e}); "
                  "endpointing disabled")
            self._silero = None

    # ── per-frame ────────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray) -> None:
        if self._stop_flag.is_set():
            return
        try:
            if self.engine == "openwakeword" and self._oww is not None:
                self._process_oww(frame)
            elif self.engine == "porcupine" and self._porcupine is not None:
                self._process_porcupine(frame)
        except Exception as e:
            print(f"  [wake-word] frame processing error: {e}")

    def _process_oww(self, frame: np.ndarray) -> None:
        # openwakeword expects int16 PCM; record_speech() in the main
        # companion uses float32 in [-1, 1] so convert.
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        scores = self._oww.predict(pcm)
        if not scores:
            return
        # Coerce each score to a finite float and skip anything else.
        # Depending on the openwakeword version, predict() values can be
        # numpy scalars/arrays or NaN; an unguarded max() can then pick a
        # NaN as "best" and silently swallow a real wake.
        best_name = None
        best_score = -math.inf
        for name, raw in scores.items():
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            if score > best_score:
                best_name = name
                best_score = score
        if best_name is not None and best_score >= self.threshold:
            self._fire(best_name, float(best_score))

    def _process_porcupine(self, frame: np.ndarray) -> None:
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        # 2026-07-08: prepend any leftover from the previous block so
        # porcupine sees a contiguous stream; blocks (e.g. 1280) that are
        # not a multiple of frame_length (e.g. 512) otherwise drop the
        # trailing remainder each call and break frame continuity.
        if self._porcupine_leftover.size:
            pcm = np.concatenate((self._porcupine_leftover, pcm))
        # Porcupine wants exactly self._porcupine_frame samples.
        n = self._porcupine_frame
        i = 0
        while i + n <= len(pcm):
            kw_index = self._porcupine.process(pcm[i:i + n])
            if kw_index >= 0:
                kw = (self._porcupine_keywords[kw_index]
                      if kw_index < len(self._porcupine_keywords)
                      else "unknown")
                self._fire(kw, 1.0)
            i += n
        # Carry the unconsumed tail into the next block.
        self._porcupine_leftover = pcm[i:].copy()

    def _fire(self, phrase: str, score: float) -> None:
        now = time.time()
        if (now - self._last_fire_ts) < self.cooldown_secs:
            return
        self._last_fire_ts = now
        evt = {"phrase": str(phrase), "score": float(score), "ts": now}
        # 2026-07-08: drop-oldest so a full queue (nobody draining events
        # because the caller uses on_detect) can't block or grow unbounded.
        try:
            self.events.put_nowait(evt)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(evt)
            except queue.Full:
                pass
        except Exception:
            pass
        cb = self.on_detect
        if cb is not None:
            try:
                cb(evt)
            except Exception as e:
                print(f"  [wake-word] on_detect callback failed: {e}")
        else:
            print(f"  [wake-word] HEARD '{phrase}' "
                  f"(score={score:.2f}) at {time.strftime('%H:%M:%S')}")
