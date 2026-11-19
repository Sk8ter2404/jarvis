"""core/loopback_ref.py - what the PC is playing, as the echo canceller's
reference (MEDIA_AEC_MODE, 2026-10-05).

WHY THIS EXISTS
---------------
core/audio_processor.MediaEchoCanceller subtracts the PC's own playback from
the owner's mic. Its reference is the LOOPBACK of the default render
endpoint (the desk speakers): exactly the samples Windows mixed for them.

PRIVACY (owner decision OD-2): the reference is what the PC plays. It lives
in RAM only, in a ring of at most RING_S seconds, and is never written to
disk, transcribed, logged or sent anywhere. Nothing here runs unless
MEDIA_AEC_MODE is 'shadow' or 'on' - the monolith calls start() only then.

HOW
---
soundcard 0.4.6 (WASAPI loopback through Media Foundation, NOT PortAudio - so
it sits outside the 0xc0000374 PortAudio crash class) records the default
speaker at 16 kHz (Windows converts the format: soundcard opens the stream
with AUTOCONVERTPCM), in 10 ms packets, on ONE never-exiting daemon thread
that initialises COM for itself. soundcard pads a silent endpoint with zeros
by the wall clock, so the ring's sample count follows time. pyaudiowpatch
(bundled PortAudio, so never beside it) is the fallback only when soundcard
cannot open the endpoint at all.

  * a stream error, or the default speaker changing (re-checked every
    RESOLVE_EVERY_S), reopens it; each reopen bumps ``gap_seq`` so the
    canceller re-measures the delay; failures back off 0.5 -> 5 s and are
    logged once per episode (no hot loop); the pyaudiowpatch fallback is
    tried at most every FALLBACK_RETRY_S;
  * ``age_s()`` is the reader's liveness: soundcard pads silence with zeros
    by the clock, so a reader that stops delivering (a stalled read) shows
    as a growing age, never as a stale "loud" last second;
  * ``index_at(t)`` maps a monotonic time to a ring index through the
    earliest-arrival time base over the last few seconds (the packet
    arrival jitter never moves it forward);
  * pause() closes the recorder (COM objects released in its __exit__)
    and parks the thread; start() resumes it.

Never raises out of its public methods.
"""
from __future__ import annotations

import collections
import importlib
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
RING_S = 30.0
PACKET = 160                    # 10 ms at 16 kHz
# The Windows capture BUFFER the soundcard recorder asks for (2026-10-18).
# soundcard turns `blocksize` straight into the WASAPI buffer duration, and it
# was PACKET: a 10 ms buffer, so any stall over 10 ms on this reader thread
# overflowed it. A night's log held 27,593 "data discontinuity in recording"
# warnings (about 46 a minute), each one audio the reference silently LOST -
# the echo canceller's timeline slipped (its delay estimate swung by hundreds
# of ms, median echo reduction 4 dB). 200 ms of headroom; reads stay PACKET.
RECORDER_BUFFER_FRAMES = 3200
# A capture gap that still happens is counted and logged at most this often.
GAP_REPORT_EVERY_S = 300.0
# TIMELINE SLIPS (3.0.26, diagnostics only). The canceller reads the ring by
# SAMPLE COUNT after it anchors, so the ring must hold exactly one sample per
# 1/16000 s of what the PC played. soundcard keeps that only approximately:
# while the endpoint delivers no packets it pads zeros for the idle time it
# measured, but only once the idle passes 4 device periods, so the last
# partial idle before the audio resumes is never padded (samples LOST), and
# a packet that reads empty is padded with a block of zeros at once (samples
# ADDED). Neither raises a discontinuity, so nothing re-anchors; the echo then
# sits somewhere else on the reference and only the canceller's GCC re-
# measure finds it - which is what the live log's delay swings of 100-300 ms
# and 6-16 realigns a minute look like. Each SLIP_BUCKET_S the earliest-
# arrival time base of the packets in that second is compared with the
# level it held; a move of SLIP_MIN_S or more that the next bucket confirms
# is one slip (signed: + = the ring ran ahead of the clock). Counted for the
# canceller's minute line; nothing changes.
SLIP_BUCKET_S = 1.0
SLIP_MIN_S = 0.008
SLIP_CONFIRM_S = 0.004          # the confirming bucket agrees within this
RESOLVE_EVERY_S = 5.0
TIMEBASE_WINDOW_S = 10.0
BACKOFF_MIN_S, BACKOFF_MAX_S = 0.5, 5.0
# The pyaudiowpatch fallback bundles a SECOND PortAudio: while soundcard keeps
# failing it is tried at most once a minute, not on every back-off step (each
# try initialises and terminates that PortAudio - 2026-10-09 review).
FALLBACK_RETRY_S = 60.0


def _com_init_mta(import_soundcard: bool = True) -> None:
    """COM for THIS thread (multithreaded). ORDER MATTERS: soundcard
    initialises COM at its first import, on the importing thread, and its
    own check treats CoInitializeEx's S_FALSE ("already initialised") as an
    error - so soundcard is imported FIRST (it initialises this thread), and
    only then is CoInitializeEx called (S_FALSE there is harmless; it covers
    a soundcard already imported by another thread). Never raises."""
    if import_soundcard:
        try:
            importlib.import_module("soundcard")
        except Exception:
            pass
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)   # COINIT_MULTITHREADED
    except Exception:
        pass


class _SoundcardSource:
    """The default speaker's loopback through soundcard. ``name`` / ``key``
    identify the endpoint (``key`` is its id) for change detection."""

    backend = "soundcard"

    def __init__(self):
        import soundcard as sc          # imported on the reader thread
        self._sc = sc
        spk = sc.default_speaker()
        self.name = str(getattr(spk, "name", "") or "")
        self.key = str(getattr(spk, "id", "") or self.name)
        self._mic = sc.get_microphone(self.key, include_loopback=True)
        self._rec = None

    def current_key(self) -> str:
        spk = self._sc.default_speaker()
        return str(getattr(spk, "id", "") or getattr(spk, "name", ""))

    def __enter__(self):
        self._rec = self._mic.recorder(samplerate=SAMPLE_RATE, channels=2,
                                       blocksize=RECORDER_BUFFER_FRAMES)
        self._rec.__enter__()
        return self

    def __exit__(self, *exc):
        rec, self._rec = self._rec, None
        if rec is not None:
            rec.__exit__(None, None, None)
        return False

    def read(self, n: int) -> np.ndarray:
        return self._rec.record(numframes=n)


class _PyAudioWPatchSource:
    """Fallback: pyaudiowpatch's WASAPI loopback of the default speaker.
    It bundles its own PortAudio, so it is used only when soundcard cannot
    open the endpoint."""

    backend = "pyaudiowpatch"

    def __init__(self):
        import pyaudiowpatch as pa
        self._pa_mod = pa
        self._pa = pa.PyAudio()
        try:
            dev = self._pa.get_default_wasapi_loopback()
        except Exception:
            self._pa.terminate()
            raise
        self.name = str(dev.get("name", ""))
        self.key = self.name
        self._dev = dev
        self._stream = None
        self._sr = int(dev.get("defaultSampleRate", 48000) or 48000)
        self._ch = max(1, int(dev.get("maxInputChannels", 2) or 2))

    def current_key(self) -> str:
        try:
            return str(self._pa.get_default_wasapi_loopback().get("name", ""))
        except Exception:
            return self.key

    def __enter__(self):
        self._stream = self._pa.open(
            format=self._pa_mod.paFloat32, channels=self._ch, rate=self._sr,
            input=True, input_device_index=int(self._dev["index"]),
            frames_per_buffer=int(self._sr // 100))
        return self

    def __exit__(self, *exc):
        st, self._stream = self._stream, None
        try:
            if st is not None:
                st.stop_stream()
                st.close()
        finally:
            try:
                self._pa.terminate()
            except Exception:
                pass
        return False

    def read(self, n: int) -> np.ndarray:
        frames = int(round(n * self._sr / SAMPLE_RATE))
        raw = self._stream.read(frames, exception_on_overflow=False)
        x = np.frombuffer(raw, dtype=np.float32).reshape(-1, self._ch)
        mono = x.mean(axis=1)
        if self._sr != SAMPLE_RATE:
            pos = np.linspace(0, len(mono) - 1, n)
            mono = np.interp(pos, np.arange(len(mono)), mono)
        return mono.reshape(-1, 1).astype(np.float32)


def _default_sources():
    return (_SoundcardSource, _PyAudioWPatchSource)


# soundcard reports each lost stretch of capture with a warning it forces to
# print EVERY time (warnings.simplefilter('always', ...)), on the thread that
# called record() - here, the reader thread. This hook counts those warnings
# instead of printing two log lines each, so the reader can mark the gap.
_DISCONTINUITY_TEXT = "data discontinuity in recording"
_DISCONTINUITIES = [0]
_ORIG_SHOWWARNING = [None]


def _showwarning_hook(message, category, filename, lineno, file=None,
                      line=None):
    if _DISCONTINUITY_TEXT in str(message):
        _DISCONTINUITIES[0] += 1
        return
    orig = _ORIG_SHOWWARNING[0]
    if orig is not None:
        orig(message, category, filename, lineno, file, line)


def _install_discontinuity_counter() -> None:
    """Route soundcard's discontinuity warnings to the counter (once).
    Every other warning is shown exactly as before. Never raises."""
    try:
        import warnings
        if warnings.showwarning is not _showwarning_hook:
            _ORIG_SHOWWARNING[0] = warnings.showwarning
            warnings.showwarning = _showwarning_hook
    except Exception:
        pass


class LoopbackReference:
    """The ring of what the PC played (16 kHz mono float32, RAM only).

    The reader thread is started by start(); tests drive ``write()``
    directly or inject ``sources`` (callables returning a source object
    with ``read(n)``, ``current_key()``, ``name``, ``key`` and the context
    manager protocol)."""

    def __init__(self, seconds: float = RING_S, sample_rate: int = SAMPLE_RATE,
                 clock=time.monotonic, sources=None, log=print):
        self.sr = int(sample_rate)
        self._n = int(seconds * self.sr)
        self._ring = np.zeros(self._n, np.float32)
        self._clock = clock
        self._sources = sources
        self._log = log
        self._cv = threading.Condition()
        self.n_written = 0
        self.gap_seq = 0
        self._tb = collections.deque()        # (t, t - n/sr) arrivals
        self._t0 = None
        self._last_write = None
        self.endpoint = ""
        self.backend = ""
        self._run = threading.Event()
        self._thread = None
        self._thread_lock = threading.Lock()
        self.failures = 0
        self.opens = 0
        self.capture_gaps = 0          # lost stretches of capture (see above)
        self._gaps_reported = 0
        # Timeline slips (SLIP_BUCKET_S): count, and the signed / absolute
        # total in ms (+ = ring ahead of the clock: samples added).
        self.slips = 0
        self.slip_ms_net = 0.0
        self.slip_ms_abs = 0.0
        self._slip_reset()
        self._fallback_at = None          # last pyaudiowpatch try (monotonic)
        self._episode_logged = False
        self.last_error = ""
        self._stop = False

    # ── the ring ─────────────────────────────────────────────────────────
    def write(self, block, t: "float | None" = None) -> None:
        """Append ``block`` (mono float32) that ARRIVED at monotonic ``t``
        (its last sample). Updates the time base. Never raises."""
        try:
            x = np.asarray(block, np.float32).reshape(-1)
            if x.size == 0:
                return
            if t is None:
                t = float(self._clock())
            with self._cv:
                n0 = self.n_written
                total = len(x)
                m = min(total, self._n)
                x = x[-m:]
                i = (n0 + total - m) % self._n
                end = i + m
                if end <= self._n:
                    self._ring[i:end] = x
                else:
                    cut = self._n - i
                    self._ring[i:] = x[:cut]
                    self._ring[:m - cut] = x[cut:]
                self.n_written = n0 + total
                base = float(t) - self.n_written / self.sr
                self._tb.append((float(t), base))
                while self._tb and float(t) - self._tb[0][0] > TIMEBASE_WINDOW_S:
                    self._tb.popleft()
                self._t0 = min(b for _t, b in self._tb)
                self._last_write = float(t)
                self._note_slip(float(t), base)
                self._cv.notify_all()
        except Exception:
            pass

    # ── timeline slips (diagnostics, SLIP_BUCKET_S) ──────────────────────
    def _slip_reset(self) -> None:
        self._slip_t0 = None           # this bucket's first arrival
        self._slip_min = None          # its earliest-arrival base
        self._slip_level = None        # the confirmed level
        self._slip_pending = None      # a move waiting for its confirmation

    def _note_slip(self, t: float, base: float) -> None:
        """One packet's base (t - n_written / sr), under self._cv. Never
        raises."""
        try:
            if self._slip_t0 is None:
                self._slip_t0, self._slip_min = t, base
                return
            if t - self._slip_t0 < SLIP_BUCKET_S:
                if base < self._slip_min:
                    self._slip_min = base
                return
            cur, self._slip_t0, self._slip_min = self._slip_min, t, base
            level = self._slip_level
            if level is None:
                self._slip_level = cur
                return
            move = cur - level              # + = later base = samples lost
            if abs(move) < SLIP_MIN_S:
                self._slip_pending = None
                self._slip_level = cur      # follows the clock drift
                return
            pend = self._slip_pending
            if pend is not None and abs(cur - pend) <= SLIP_CONFIRM_S:
                ms = -move * 1000.0         # + = the ring ran ahead
                self.slips += 1
                self.slip_ms_net += ms
                self.slip_ms_abs += abs(ms)
                self._slip_level = cur
                self._slip_pending = None
            else:
                self._slip_pending = cur    # one late bucket is jitter
        except Exception:
            pass

    def mark_gap(self) -> None:
        """A discontinuity (reopened stream): the time base restarts and the
        canceller re-measures."""
        with self._cv:
            self.gap_seq += 1
            self._tb.clear()
            self._t0 = None
            self._slip_reset()

    def read(self, start: int, n: int) -> "np.ndarray | None":
        """Samples [start, start+n), or None when any of them is not in the
        ring (not written yet, or already overwritten)."""
        try:
            start, n = int(start), int(n)
            with self._cv:
                if (n <= 0 or start < 0 or start + n > self.n_written
                        or start < self.n_written - self._n):
                    return None
                i = start % self._n
                end = i + n
                if end <= self._n:
                    return self._ring[i:end].copy()
                return np.concatenate([self._ring[i:],
                                       self._ring[:end - self._n]])
        except Exception:
            return None

    def index_at(self, t: float) -> "float | None":
        """The (fractional) ring index the PC played at monotonic ``t``;
        None before the first packet of this stream."""
        with self._cv:
            if self._t0 is None:
                return None
            return (float(t) - self._t0) * self.sr

    def wait_for(self, index: int, timeout: float) -> bool:
        """Wait (bounded) until ``index`` samples are written."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while self.n_written < int(index):
                rem = deadline - time.monotonic()
                if rem <= 0 or not self._run.is_set():
                    return self.n_written >= int(index)
                self._cv.wait(rem)
            return True

    def rms_recent(self, seconds: float = 1.0) -> float:
        """RMS of the last ``seconds`` written (0.0 before any)."""
        n = int(min(self.n_written, seconds * self.sr, self._n))
        if n <= 0:
            return 0.0
        x = self.read(self.n_written - n, n)
        if x is None or x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))

    def age_s(self) -> "float | None":
        """Seconds since the last packet (None: never)."""
        last = self._last_write
        return None if last is None else max(0.0, float(self._clock()) - last)

    # ── the reader thread ───────────────────────────────────────────────
    def start(self) -> bool:
        """Run the reader (idempotent). The thread is created once and
        never exits; this only wakes it."""
        with self._thread_lock:
            self._run.set()
            t = self._thread
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._loop, name="loopback-ref",
                                     daemon=True)
                self._thread = t
                try:
                    t.start()
                except Exception as e:
                    self._run.clear()
                    self._thread = None
                    self.last_error = f"{type(e).__name__}: {e}"
                    return False
            return True

    def pause(self) -> None:
        """Stop recording (the recorder is closed by the thread)."""
        self._run.clear()
        with self._cv:
            self._cv.notify_all()

    @property
    def running(self) -> bool:
        return self._run.is_set()

    def status(self) -> dict:
        return {"running": self.running, "backend": self.backend,
                "endpoint": self.endpoint, "n_written": self.n_written,
                "gap_seq": self.gap_seq, "opens": self.opens,
                "failures": self.failures, "capture_gaps": self.capture_gaps,
                "slips": self.slips,
                "slip_ms_net": round(self.slip_ms_net, 1),
                "slip_ms_abs": round(self.slip_ms_abs, 1),
                "age_s": self.age_s(),
                "last_error": self.last_error}

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass

    def _open(self):
        """The first source that opens. A fallback (every source after the
        first) is tried at most every FALLBACK_RETRY_S."""
        last = None
        for i, factory in enumerate(self._sources or _default_sources()):
            if i > 0:
                now = time.monotonic()
                if (self._fallback_at is not None
                        and now - self._fallback_at < FALLBACK_RETRY_S):
                    continue
                self._fallback_at = now
            try:
                return factory()
            except Exception as e:
                last = e
        raise last if last is not None else RuntimeError("no loopback source")

    def shutdown(self) -> None:
        """Stop recording and let the thread end - tests and process exit
        only."""
        self._stop = True
        self.pause()

    def _loop(self) -> None:
        # The real sources only (a test's fake sources import nothing).
        _com_init_mta(import_soundcard=self._sources is None)
        _install_discontinuity_counter()
        backoff = BACKOFF_MIN_S
        next_report = time.monotonic() + GAP_REPORT_EVERY_S
        while not self._stop:
            if not self._run.wait(timeout=0.2):
                continue
            src = None
            try:
                src = self._open()
                with src:
                    self.opens += 1
                    self.backend = getattr(src, "backend", "")
                    self.endpoint = getattr(src, "name", "")
                    self.mark_gap()
                    if self._episode_logged:
                        self._say(f"  [loopback] recording the PC's playback "
                                  f"again ({self.backend})")
                    self._episode_logged = False
                    backoff = BACKOFF_MIN_S
                    next_check = time.monotonic() + RESOLVE_EVERY_S
                    while self._run.is_set():
                        lost_before = _DISCONTINUITIES[0]
                        x = src.read(PACKET)
                        a = np.asarray(x, np.float32)
                        mono = a.mean(axis=1) if a.ndim == 2 else a.reshape(-1)
                        if _DISCONTINUITIES[0] != lost_before:
                            # Audio was lost before this packet: restart the
                            # time base so the canceller re-anchors, instead
                            # of carrying a silent slip.
                            self.capture_gaps += 1
                            self.mark_gap()
                        self.write(mono, float(self._clock()))
                        if time.monotonic() >= next_report:
                            next_report = time.monotonic() + GAP_REPORT_EVERY_S
                            new = self.capture_gaps - self._gaps_reported
                            if new > 0:
                                self._gaps_reported = self.capture_gaps
                                self._say(f"  [loopback] {new} capture gap(s) "
                                          f"in the last "
                                          f"{GAP_REPORT_EVERY_S / 60:.0f} min "
                                          f"(the echo canceller re-anchors at "
                                          f"each)")
                        if time.monotonic() >= next_check:
                            next_check = time.monotonic() + RESOLVE_EVERY_S
                            try:
                                if src.current_key() != src.key:
                                    break          # the default moved: reopen
                            except Exception:
                                pass
            except Exception as e:
                self.failures += 1
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                if not self._episode_logged:
                    self._episode_logged = True
                    self._say(f"  [loopback] cannot record the PC's playback "
                              f"({self.last_error}) - the echo canceller "
                              f"passes the mic through; retrying with "
                              f"back-off")
                time.sleep(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2.0)
            finally:
                self.mark_gap()
