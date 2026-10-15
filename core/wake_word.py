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
import sys
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
FOLLOW_INTERVAL_SECS = 4.0          # how often the follower asks the device
                                    # provider whether the mic moved
FOLLOW_BUSY_AFTER_FIRE_SECS = 20.0  # a wake just fired: the owner's command is
                                    # being captured - never re-open under it
FOLLOW_RETRY_SECS = 30.0            # wait before retrying a device that would
                                    # not open (the old stream keeps running)
MAX_BUFFER_FRAMES = 50              # hard cap on _buf size (in frames) to
                                    # avoid unbounded growth if _on_frame
                                    # raises and the drain loop exits early


def _safe_close_stream(stream, timeout_sec: float = 2.0) -> None:
    """Stop+close a sounddevice stream without blocking the caller.

    Mirrors bobert_companion._safe_close_stream: sounddevice.close() at
    sounddevice.py:1167 SIGSEGV'd on this build (faulthandler caught the
    crash across multiple PIDs on 2026-05-29). Stop synchronously, then run
    close() on a daemon thread; if it hangs past timeout_sec, ABANDON the
    daemon (it dies with the process) — never sd.stop(), see the H-3 note at
    the bottom of this function."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception as e:
        print(f"  [wake-word] stream.stop raised: {e!r}")
    done = threading.Event()
    # H-6 (2026-08-20) — THE STALE-DUPLICATE HALF. This copy abandons a native
    # Pa_CloseStream exactly like bobert_companion._safe_close_stream does, and
    # bobert_companion._refresh_devices calls det.pause() — which lands HERE —
    # immediately before its destructive sd._terminate(). So a wake-word close
    # that hung past timeout_sec was abandoned and then torn down underneath
    # (0xc0000374), by the very code path the gate protects. Register the
    # hand-off with the host's _pa_close_pending cell when the monolith is
    # loaded; degrade to a plain start() when it is not (standalone / test
    # import), where there is no _refresh_devices to defer in the first place.
    _bc = sys.modules.get("bobert_companion")
    _handoff = getattr(_bc, "_pa_close_handoff", None) if _bc is not None else None
    _retire = getattr(_bc, "_pa_close_done", None) if _bc is not None else None
    _gated = callable(_handoff) and callable(_retire)

    def _do_close():
        try:
            stream.close()
        except Exception as e:
            print(f"  [wake-word] stream.close raised: {e!r}")
        finally:
            if _gated:
                # Retire BEFORE waking the caller — a successful bounded wait
                # must never leave a phantom count deferring hotplug.
                try:
                    _retire()
                except Exception:
                    pass
            done.set()

    t = threading.Thread(target=_do_close, daemon=True)
    if _gated:
        # Bumps the count, then starts t (and retires again on a failed start,
        # so a thread that never ran can't defer the reinit forever).
        _handoff(t)
    else:
        t.start()
    if not done.wait(timeout=timeout_sec):
        # H-3 (2026-08-20) — THE STALE-DUPLICATE HALF of the same deletion in
        # bobert_companion._safe_close_stream. Module-level sd.stop() does NOT
        # touch the stream passed in: sounddevice 0.5.5 stops+closes whatever
        # `_last_callback` points at, which is only ever set by
        # sd.play()/sd.rec()/sd.playrec(). `s` here is always an explicitly
        # constructed InputStream, so the call could never free this handle —
        # but it CAN close the live TTS playback stream out from under
        # play_with_lipsync's reaper (two threads in Pa_CloseStream on one
        # WASAPI stream -> 0xc0000374). Abandon instead; the close stays
        # registered with the host's _pa_close_pending gate above.
        print(f"  [wake-word] stream.close hung >{timeout_sec:.1f}s — "
              "abandoning the close daemon (it dies with the process). "
              "sd.stop() is deliberately NOT called: it acts only on the last "
              "sd.play()/sd.rec() stream, so it cannot free this handle and "
              "would instead close a live TTS playback stream "
              "(double Pa_CloseStream -> 0xc0000374).")


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
        device_provider: Optional[Callable[[], Optional[int]]] = None,
        follow_interval_s: float = FOLLOW_INTERVAL_SECS,
    ) -> None:
        self.engine = (engine or "off").lower().strip()
        self.wake_words = list(wake_words or DEFAULT_WAKE_WORDS)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.threshold = float(threshold)
        self.cooldown_secs = float(cooldown_secs)
        self.on_detect = on_detect
        self.use_silero_vad = bool(use_silero_vad)
        # Asked for the input device at EVERY stream open (2026-10-02, see
        # _open_stream). None = always open on ``device``, as before.
        self.device_provider = device_provider
        # With a provider a small follower thread re-asks it every
        # ``follow_interval_s`` while the stream is live and moves the stream
        # when the answer changed (2026-10-15). 0 = never follow (open-time
        # resolution only, as before).
        self.follow_interval_s = float(follow_interval_s or 0.0)
        self._follower: Optional[threading.Thread] = None
        self._follow_fail: Optional[tuple] = None   # (target, monotonic ts)
        # Bumped on every stream swap: a callback of an older generation
        # drops its frames, so two streams never splice into one buffer.
        self._gen = 0

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
        # Taps that only listen in the background (the voice-ID rolling
        # buffer): they do not mean a capture is in progress.
        self._passive_taps: list["queue.Queue[np.ndarray]"] = []

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

    def mark_tap_passive(self, q: "queue.Queue[np.ndarray]") -> None:
        """Declare a registered tap a background listener: it never counts as
        'someone is capturing', so it does not hold off a mic follow."""
        with self._taps_lock:
            if q not in self._passive_taps:
                self._passive_taps.append(q)

    def remove_tap(self, q: "queue.Queue[np.ndarray]") -> None:
        with self._taps_lock:
            try:
                self._taps.remove(q)
            except ValueError:
                pass
            try:
                self._passive_taps.remove(q)
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
        self._start_follower()
        return True

    def _make_callback(self, gen: int):
        """The PortAudio callback for one stream generation. A callback whose
        generation is no longer current (its stream was swapped out by the
        follower) drops its frames, so two streams never splice."""
        frame_size = int(self.sample_rate * DEFAULT_FRAME_MS / 1000)
        max_buf_samples = frame_size * MAX_BUFFER_FRAMES

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            if gen != self._gen:
                return
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
        return _cb

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
        self._gen += 1
        _cb = self._make_callback(self._gen)

        # The mic is resolved at EVERY open when the owner passed a provider
        # (2026-10-02): start() and the resume() after a PortAudio device
        # refresh then open the mic the owner's capture uses NOW. ``device``
        # alone is fixed when the detector is built, and its None default is
        # not "the current default" either - PortAudio freezes the default
        # input at Pa_Initialize, so it is the mic picked at boot. A provider
        # that raises keeps the last device.
        if self.device_provider is not None:
            try:
                self.device = self.device_provider()
            except Exception as e:
                print(f"  [wake-word] input device lookup failed ({e!r}); "
                      f"keeping device={self.device!r}")

        try:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=frame_size,
                    device=self.device,
                    callback=_cb,
                )
            except Exception as e:
                # A provider-picked device that will not open (a stale index
                # after a hotplug) gets the main capture's one retry, on the
                # system default - what this detector always opened before
                # it had a provider, so following the mic is never worse.
                if self.device_provider is None or self.device is None:
                    raise
                print(f"  [wake-word] input device {self.device!r} would not "
                      f"open ({e}); retrying with the system default")
                self.device = None
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=frame_size,
                    device=None,
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
            # pre- and post-refresh audio into a single garbage frame. That
            # includes Porcupine's partial-frame carry (July audit A47): the
            # stream is already closed above, so no callback races this.
            self._buf = np.zeros(0, dtype=np.float32)
            self._porcupine_leftover = np.zeros(0, dtype=np.int16)
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

    # ── follow the current mic ───────────────────────────────────────
    # 2026-10-15. The provider is asked at every OPEN (above), but a live
    # stream keeps the device it opened on: when the Windows default input
    # moves WITHOUT a PortAudio teardown (bobert_companion._refresh_devices
    # adopts a Stream Deck / headset flip from the existing enumeration and
    # only pauses this detector on the destructive reinit path), the wake
    # stream kept listening on the old mic until restart. The follower asks
    # the provider every ``follow_interval_s`` on its own thread (the
    # provider may take the host's device lock, which the PortAudio callback
    # must never wait on) and moves the stream when the answer changed. It
    # opens the NEW stream first and swaps only if that worked, so a failing
    # open keeps the old stream; an explicit pin is just a provider that
    # keeps answering the same device, so it is never moved.
    def _start_follower(self) -> None:
        if (self.device_provider is None or self.follow_interval_s <= 0
                or (self._follower is not None and self._follower.is_alive())):
            return
        t = threading.Thread(target=self._follow_loop, name="wake-word-follow",
                             daemon=True)
        self._follower = t
        t.start()

    def _follow_loop(self) -> None:
        while not self._stop_flag.wait(self.follow_interval_s):
            self._follow_tick()

    def _capture_in_progress(self) -> bool:
        """The owner is mid-utterance: a wake fired moments ago (the command
        is being captured) or a foreground tap is reading this stream."""
        if (time.time() - self._last_fire_ts) < FOLLOW_BUSY_AFTER_FIRE_SECS:
            return True
        with self._taps_lock:
            return any(t not in self._passive_taps for t in self._taps)

    @staticmethod
    def _device_label(sd, idx) -> str:
        if idx is None:
            return "system default"
        try:
            return f"{idx} ({sd.query_devices(idx)['name']})"
        except Exception:
            return str(idx)

    def _follow_tick(self) -> bool:
        """One follower pass. True = the stream moved to a new device.
        Never raises."""
        try:
            return self._follow_tick_inner()
        except Exception as e:
            print(f"  [wake-word] input follow failed: {e!r}")
            return False

    def _follow_tick_inner(self) -> bool:
        if (self.device_provider is None or not self._running
                or self._paused or self._stop_flag.is_set()):
            return False
        try:
            target = self.device_provider()
        except Exception:
            return False            # keep the current stream
        if target == self.device:
            self._follow_fail = None
            return False
        if target is not None and getattr(self._stream, "device", None) == target:
            # Opened on "the default" and PortAudio already resolved that to
            # this very index: same mic, nothing to move.
            self.device = target
            return False
        failed = self._follow_fail
        if (failed is not None and failed[0] == target
                and time.monotonic() - failed[1] < FOLLOW_RETRY_SECS):
            return False
        if self._capture_in_progress():
            return False            # try again next tick, after the utterance
        try:
            import sounddevice as sd
        except ImportError:
            return False
        # Never wait behind pause()/resume() (the host's device refresh holds
        # this lock across its PortAudio teardown): skip, try next tick.
        if not self._pause_lock.acquire(blocking=False):
            return False
        try:
            if (not self._running or self._paused or self._stop_flag.is_set()
                    or self._stream is None):
                return False
            old_dev = self.device
            gen = self._gen + 1
            frame_size = int(self.sample_rate * DEFAULT_FRAME_MS / 1000)
            new = None
            try:
                new = sd.InputStream(
                    samplerate=self.sample_rate, channels=1, dtype="float32",
                    blocksize=frame_size, device=target,
                    callback=self._make_callback(gen))
                new.start()
            except Exception as e:
                self._follow_fail = (target, time.monotonic())
                print(f"  [wake-word] input {self._device_label(sd, target)} "
                      f"would not open ({e}); keeping "
                      f"{self._device_label(sd, old_dev)}, retrying later")
                if new is not None:
                    _safe_close_stream(new)
                return False
            # The utterance check again: opening can take a while.
            if (self._capture_in_progress() or self._stop_flag.is_set()
                    or not self._running or self._paused):
                _safe_close_stream(new)
                return False
            old = self._stream
            self._gen = gen          # the old callback now drops its frames
            self._stream = new
            self.device = target
            self._follow_fail = None
            self._buf = np.zeros(0, dtype=np.float32)
            self._porcupine_leftover = np.zeros(0, dtype=np.int16)
            _safe_close_stream(old)
            print(f"  [wake] input changed: "
                  f"{self._device_label(sd, old_dev)} -> "
                  f"{self._device_label(sd, target)}")
            return True
        finally:
            self._pause_lock.release()

    # ── engine init ──────────────────────────────────────────────────
    def _init_openwakeword(self) -> None:
        import openwakeword  # package check — dropped below: pyflakes ignores noqa
        del openwakeword
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
