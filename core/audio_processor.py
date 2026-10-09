"""
core/audio_processor.py

Real-time three-layer audio processing pipeline for JARVIS's input chain.

Applied in order:
    1. Echo cancellation — removes JARVIS's own TTS playback bleed-through
       so the mic doesn't hear itself.
    2. Noise suppression  — attenuates stationary background noise
       (fans, HVAC, keyboards, music, distant speakers).
    3. Gain normalization — auto-targets a steady RMS so a far-away
       whisper transcribes as cleanly as a close talker.

Each layer probes for a preferred backend (webrtc_audio_processing or
noisereduce) and falls back to a numpy implementation when the backend
isn't installed. If ANY stage raises, the processor returns the input
unchanged for that stage so a missing dep or processing error never
silences the pipeline.

Public surface:
    get_processor(sample_rate=16000)  → AudioProcessor singleton
    feed_playback(audio, sample_rate) → record TTS output (AEC reference)
    is_playback_recent(within=…)      → True if speakers were active

The processor is frame-agnostic (callers pass whatever chunk size they
already use) but operates internally at frame_ms granularity so per-call
latency stays well under 50 ms at 16 kHz/20 ms frames.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np


_DEBUG = bool(os.environ.get("AUDIO_PROCESSOR_DEBUG"))


def _dprint(msg: str) -> None:
    if _DEBUG:
        print(f"  [audio_processor] {msg}")


def _safe_exc(prefix: str, exc: BaseException) -> str:
    """Format an exception WITHOUT calling its __str__ unprotected.

    Discovered 2026-05-30 08:30: a numpy exception's __str__ method can
    SIGSEGV when the underlying ndarray ref-state is corrupted (which
    happens when near-silent audio chunks land in noisereduce after
    VAD_THRESHOLD relaxation). The faulthandler caught the crash at
    numpy/_core/_exceptions.py:47 in __str__, triggered from
    audio_processor.py:432 ``err = f"noisereduce: {e}"``.

    Calling ``str(e)`` or ``f"{e}"`` on such an exception during normal
    error handling SEGV's the whole process. This helper isolates the
    string conversion so a crashing __str__ degrades to a class-name
    label instead of taking the interpreter down.

    Use everywhere an exception from noisereduce / numpy / a C
    extension might be formatted into an error string."""
    cls = type(exc).__name__
    try:
        msg = str(exc)
    except BaseException:
        # If str() itself crashes (the SIGSEGV path), fall back to the
        # class name alone — at least we get a label instead of a death.
        return f"{prefix}: <{cls}: __str__ failed>"
    return f"{prefix}: {cls}: {msg}"


class AudioProcessor:
    """Three-layer real-time processor for mono float32 audio at a fixed
    sample rate. Thread-safe for process() + feed_playback() interleave."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        agc_target_rms: float = 0.05,
        agc_max_gain: float = 8.0,
        ns_strength: float = 0.7,
        aec_duck_gain: float = 0.7,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = max(1, int(self.sample_rate * self.frame_ms / 1000))

        self.agc_target_rms = float(agc_target_rms)
        self.agc_max_gain = float(agc_max_gain)
        self.ns_strength = float(ns_strength)
        self.aec_duck_gain = float(aec_duck_gain)

        # ── Config overrides (core/config.py) ─────────────────────────
        # Read AEC duck gain + flatness bounds from config so the user can
        # tune them without editing this module. Falls through to the
        # constructor defaults when config is unavailable (test contexts).
        self._agc_flatness_min: float = 0.20
        self._agc_flatness_max: float = 0.80
        try:
            from core import config as _cfg  # type: ignore
            self.aec_duck_gain = float(getattr(_cfg, "AEC_DUCK_GAIN", self.aec_duck_gain))
            self._agc_flatness_min = float(
                getattr(_cfg, "AGC_FLATNESS_MIN", self._agc_flatness_min))
            self._agc_flatness_max = float(
                getattr(_cfg, "AGC_FLATNESS_MAX", self._agc_flatness_max))
        except Exception:
            pass

        # ── Optional backends ─────────────────────────────────────────
        # webrtc_audio_processing: full APM (AEC3 + NS + AGC). Spotty on
        # Windows wheels; if the import or construction fails we silently
        # fall through to per-layer fallbacks.
        self._apm = None
        try:
            from webrtc_audio_processing import AudioProcessingModule as APM
            apm = APM(aec_type=2, enable_ns=True, agc_type=1)
            try:
                apm.set_stream_format(self.sample_rate, 1)
                apm.set_reverse_stream_format(self.sample_rate, 1)
            except Exception:
                pass
            self._apm = apm
            _dprint("webrtc_audio_processing online")
        except Exception as e:
            self._apm = None
            _dprint(f"webrtc_audio_processing unavailable ({e})")

        # noisereduce: spectral-subtraction NS, well-supported on pip.
        self._nr = None
        try:
            import noisereduce as nr  # noqa: F401
            self._nr = nr
            _dprint("noisereduce online")
        except Exception as e:
            self._nr = None
            _dprint(f"noisereduce unavailable ({e})")

        # ── Playback ring (AEC reference) ─────────────────────────────
        # ~2 s of recent speaker output. feed_playback() appends; the
        # AEC layer pulls the latest n_samples to use as the far-end
        # reference for whichever cancellation strategy is active.
        self._playback_buffer: "deque[tuple[float, np.ndarray]]" = deque()
        self._playback_lock = threading.Lock()
        self._last_playback_ts: float = 0.0

        # AGC state
        self._agc_running_rms: float = 0.0
        self._agc_smooth: float = 0.9
        self._agc_lock = threading.Lock()
        # Spectral peakedness gate: prevents broadband ambient noise (fans,
        # keyboards) from being amplified above the VAD threshold at cold
        # start. Flatness ≈ 0 for tonal/speech content, ≈ 1 for white noise.
        self._agc_flatness: float = 0.5
        self._agc_flatness_init: bool = False
        self._agc_flatness_smooth: float = 0.7
        # Sigmoid center: flatness above this → suppress gain; below → keep it.
        # Speech typically sits at 0.05-0.30; fan/keyboard noise at 0.4+.
        self._agc_flatness_center: float = 0.35
        self._agc_flatness_width: float = 0.05

        # NS fallback state — adaptive noise spectrum.
        self._ns_noise_mag: Optional[np.ndarray] = None
        self._ns_alpha: float = 0.95
        self._ns_lock = threading.Lock()

        # Stats
        # Reads from status() and writes from process()/_aec()/_ns()/
        # feed_playback() can race across threads. Protect _n_processed,
        # _n_aec_dropouts, and _last_error under _stats_lock so status()
        # never observes a partially-updated counter or a torn object
        # reference for _last_error on multi-core builds.
        self._stats_lock = threading.Lock()
        self._n_processed = 0
        self._n_aec_dropouts = 0
        self._n_aec_ducked = 0          # AEC-fallback duck firings (diagnostics)
        self._last_raw_rms: float = 0.0  # raw RMS of last process() input
        self._last_proc_rms: float = 0.0 # post-pipeline RMS of last output
        self._last_error: Optional[str] = None

        # ── RMS history ring (60 s) ───────────────────────────────────
        # Each entry is (timestamp, rms). Used by core.tts.detect_stress_from_rms()
        # to read recent peak loudness as a stress proxy. Bounded by both age
        # (60 s) and count so a runaway processor never grows it without limit.
        self._rms_history: "deque[tuple[float, float]]" = deque(maxlen=4096)
        self._rms_history_lock = threading.Lock()
        self._rms_history_window_s: float = 60.0

    # ── public ────────────────────────────────────────────────────────

    def status(self) -> dict:
        # Read _last_playback_ts under _playback_lock — float assignment
        # isn't atomic on 32-bit Python builds, so an unlocked read can
        # tear and return garbage.
        with self._playback_lock:
            last_ts = self._last_playback_ts
        with self._stats_lock:
            n_processed = self._n_processed
            n_aec_dropouts = self._n_aec_dropouts
            n_aec_ducked = self._n_aec_ducked
            last_raw_rms = self._last_raw_rms
            last_proc_rms = self._last_proc_rms
            last_error = self._last_error
        with self._agc_lock:
            agc_flatness = self._agc_flatness
            agc_running_rms = self._agc_running_rms
        return {
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_ms,
            "apm_available": self._apm is not None,
            "noisereduce_available": self._nr is not None,
            "last_playback_age_s": (time.time() - last_ts) if last_ts else None,
            "n_processed": n_processed,
            "n_aec_dropouts": n_aec_dropouts,
            "n_aec_ducked": n_aec_ducked,
            "aec_duck_gain": self.aec_duck_gain,
            "last_raw_rms": last_raw_rms,
            "last_proc_rms": last_proc_rms,
            "agc_running_rms": agc_running_rms,
            "agc_flatness": agc_flatness,
            "agc_flatness_bounds": (self._agc_flatness_min, self._agc_flatness_max),
            "last_error": last_error,
        }

    def feed_playback(self, audio: np.ndarray,
                      sample_rate: Optional[int] = None) -> None:
        """Record outgoing speaker audio so the AEC layer has a far-end
        reference. Safe to call from any thread.  Mismatched sample
        rates are linearly resampled to self.sample_rate so callers
        don't have to care.
        """
        try:
            if audio is None or audio.size == 0:
                return
            x = np.asarray(audio, dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(axis=1)
            sr_in = int(sample_rate or self.sample_rate)
            if sr_in != self.sample_rate and x.size > 1:
                n_out = max(1, int(round(x.size * self.sample_rate / sr_in)))
                xs_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
                xs_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
                x = np.interp(xs_new, xs_old, x).astype(np.float32, copy=False)
            ts = time.time()
            with self._playback_lock:
                self._playback_buffer.append((ts, x))
                self._last_playback_ts = ts
                cutoff = ts - 2.0
                while self._playback_buffer and self._playback_buffer[0][0] < cutoff:
                    self._playback_buffer.popleft()
        except Exception as e:
            err = f"feed_playback: {e}"
            with self._stats_lock:
                self._last_error = err
            _dprint(err)

    def is_playback_recent(self, within: float = 0.2) -> bool:
        """True when feed_playback() has fired in the last `within`
        seconds — the AEC fallback uses this to duck input during
        JARVIS's own playback."""
        # Float assignment isn't atomic on 32-bit Python builds; read the
        # timestamp under the lock so we never observe a torn value and
        # erroneously trip the AEC fallback. Lock is released before
        # time.time() so we don't hold it during the syscall.
        with self._playback_lock:
            last_ts = self._last_playback_ts
        return (time.time() - last_ts) < float(within)

    def process(
        self,
        audio: np.ndarray,
        *,
        enable_aec: bool = True,
        enable_ns: bool = True,
        enable_agc: bool = True,
        record_mic_stats: bool = True,
    ) -> np.ndarray:
        """Run the three-layer pipeline on a mono float32 chunk.

        `record_mic_stats=False` suppresses the raw-RMS / rms-history writes that
        feed the silent-mic health detector and the mic-only stress proxy. Set
        it False on the LOOPBACK (system-audio) path — otherwise media/TTS
        loudness pollutes those mic-only stats: a silent mic during loud playback
        would look "audible", masking a genuinely dead mic. 2026-07-14 bug-hunt.

        Any stage that raises is skipped — the chunk passes through
        whatever stages succeed. Never returns None for a non-empty
        input.
        """
        if audio is None or getattr(audio, "size", 0) == 0:
            return audio
        try:
            x = np.asarray(audio, dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(axis=1).astype(np.float32, copy=False)
        except Exception as e:
            with self._stats_lock:
                self._last_error = f"process pre-cast: {e}"
            return audio

        # Capture pre-processing RMS for diagnostics (silent-mic detection
        # consumes this via note_raw_rms() / seconds_since_audible_chunk()).
        # ONLY for the real mic — the loopback path passes record_mic_stats=False
        # so system-audio loudness can't masquerade as mic input (#17).
        if record_mic_stats:
            try:
                raw_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
                with self._stats_lock:
                    self._last_raw_rms = raw_rms
                note_raw_rms(raw_rms)
            except Exception:
                pass

        # All three stage wrappers use BaseException + _safe_exc so a
        # numpy / C-extension exception with a corrupted __str__ can't
        # crash the interpreter here either. The crash on 2026-05-30
        # 08:30 happened inside _ns's own try-block, but the outer ns
        # wrapper at the (now-gone) `err = f"ns: {e}"` line had the
        # exact same SIGSEGV exposure.
        if enable_aec:
            try:
                x = self._aec(x)
            except BaseException as e:
                err = _safe_exc("aec", e)
                with self._stats_lock:
                    self._last_error = err
                    self._n_aec_dropouts += 1
                _dprint(err)

        if enable_ns:
            try:
                x = self._ns(x)
            except BaseException as e:
                err = _safe_exc("ns", e)
                with self._stats_lock:
                    self._last_error = err
                _dprint(err)

        if enable_agc:
            try:
                x = self._agc(x)
            except BaseException as e:
                err = _safe_exc("agc", e)
                with self._stats_lock:
                    self._last_error = err
                _dprint(err)

        with self._stats_lock:
            self._n_processed += 1
        try:
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            with self._stats_lock:
                self._last_proc_rms = rms
            if record_mic_stats and rms > 0.0:
                ts = time.time()
                cutoff = ts - self._rms_history_window_s
                with self._rms_history_lock:
                    self._rms_history.append((ts, rms))
                    while self._rms_history and self._rms_history[0][0] < cutoff:
                        self._rms_history.popleft()
        except Exception as e:
            with self._stats_lock:
                self._last_error = f"rms history: {e}"
        return x

    def recent_peak_rms(self, within: float = 60.0) -> float:
        """Highest RMS observed in the last `within` seconds. Returns 0.0
        when no frames have been processed inside the window."""
        cutoff = time.time() - float(within)
        with self._rms_history_lock:
            recent = [r for ts, r in self._rms_history if ts >= cutoff]
        return max(recent) if recent else 0.0

    # ── layer 1: echo cancellation ────────────────────────────────────

    def _aec(self, audio: np.ndarray) -> np.ndarray:
        """Echo cancel against the recent playback ring. Two paths:

        1. webrtc APM (best, real AEC3). Needs reference + capture in
           10 ms frames at the APM's stream sample rate.
        2. Fallback: spectral ducking. When playback was recent, scale
           the input by aec_duck_gain to keep the mic from re-triggering
           on JARVIS's own voice. Not real cancellation but it prevents
           the worst self-trigger feedback loop.
        """
        if self._apm is not None:
            ref = self._reference_frame(len(audio))
            if ref is not None:
                try:
                    return self._apm_process(audio, ref)
                except Exception as e:
                    with self._stats_lock:
                        self._last_error = f"apm process: {e}"
                    # Fall through to ducking fallback

        if self.is_playback_recent(within=0.15):
            with self._stats_lock:
                self._n_aec_ducked += 1
            return (audio * float(self.aec_duck_gain)).astype(np.float32, copy=False)
        return audio

    def _reference_frame(self, n_samples: int) -> Optional[np.ndarray]:
        """Pull the most-recent n_samples of speaker output for AEC ref.
        Returns None when no playback is buffered."""
        with self._playback_lock:
            if not self._playback_buffer:
                return None
            recent_parts = [a for _, a in self._playback_buffer]
        try:
            cat = np.concatenate(recent_parts).astype(np.float32, copy=False)
        except Exception:
            return None
        if cat.size == 0:
            return None
        if cat.size >= n_samples:
            return cat[-n_samples:]
        pad = np.zeros(n_samples - cat.size, dtype=np.float32)
        return np.concatenate([pad, cat])

    def _apm_process(self, audio: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Feed the WebRTC APM in 10 ms frames at the configured rate."""
        apm = self._apm
        if apm is None:
            return audio
        # APM uses 10 ms frames internally; align our chunk into 10 ms
        # blocks, pad the tail, and stitch the cleaned blocks back.
        block = max(1, int(self.sample_rate * 0.010))
        n = audio.size
        n_blocks = (n + block - 1) // block
        # Pad both signals to the block boundary.
        pad_a = np.zeros(n_blocks * block - n, dtype=np.float32)
        a_buf = np.concatenate([audio, pad_a])
        if ref.size < a_buf.size:
            r_buf = np.concatenate([np.zeros(a_buf.size - ref.size,
                                             dtype=np.float32), ref])
        else:
            r_buf = ref[-a_buf.size:]
        out_blocks: list[np.ndarray] = []
        for i in range(n_blocks):
            s = i * block
            e = s + block
            a16 = (a_buf[s:e] * 32767.0).clip(-32768.0, 32767.0).astype(np.int16).tobytes()
            r16 = (r_buf[s:e] * 32767.0).clip(-32768.0, 32767.0).astype(np.int16).tobytes()
            try:
                apm.process_reverse_stream(r16)
                cleaned = apm.process_stream(a16)
            except Exception:
                cleaned = a16
            arr = np.frombuffer(cleaned, dtype=np.int16).astype(np.float32) / 32767.0
            out_blocks.append(arr)
        out = np.concatenate(out_blocks)[:n]
        return out.astype(np.float32, copy=False)

    # ── layer 2: noise suppression ────────────────────────────────────

    def _ns(self, audio: np.ndarray) -> np.ndarray:
        """Suppress stationary background noise.

        Path A (preferred): noisereduce.reduce_noise, stationary mode.
        Path B (fallback): adaptive spectral subtraction in numpy.

        Crash hardening 2026-05-30: noisereduce/numpy can SIGSEGV on
        malformed near-silent input (caught in faulthandler trace at
        08:30 mid daily-briefing, PID 73152). The fix has three layers:

          1. Input guard — skip noisereduce entirely for near-silent /
             near-empty chunks and route straight to spectral_subtract,
             which is numpy-only and well-defined on tiny inputs.
          2. Output validation — even if noisereduce returns, verify
             the result is finite and has matching dtype/shape before
             trusting it.
          3. Safe exception formatting — use _safe_exc() so a numpy
             exception with a corrupted __str__ can't crash the handler.
        """
        # ── Input guard. The 2026-05-30 crash was triggered by chunks
        # that passed VAD at 0.008 but were effectively silent. noisereduce
        # internally builds spectrograms; an all-zero or sub-eps input
        # can leave its internal numpy arrays in a state that SIGSEGV's
        # on the next __str__ if it raises. Spectral subtraction is
        # pure numpy and handles empty/silent input deterministically.
        if audio.size < 256:
            return self._spectral_subtract(audio)
        try:
            peak = float(np.max(np.abs(audio)))
        except Exception:
            return audio
        if peak < 1e-4:
            # Near-silent. noisereduce has no meaningful signal to
            # denoise here and is the historical crash trigger.
            return self._spectral_subtract(audio)

        if self._nr is not None:
            try:
                cleaned = self._nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=float(self.ns_strength),
                )
            except BaseException as e:
                # BaseException, not Exception — catches everything short
                # of SystemExit, including a corrupted exception that
                # would SIGSEGV in the handler if we tried to format it
                # naively.
                err = _safe_exc("noisereduce", e)
                with self._stats_lock:
                    self._last_error = err
                _dprint(err)
                # Fall through to spectral subtraction.
                return self._spectral_subtract(audio)
            # Output validation: noisereduce can return a shorter array
            # or one with NaN/Inf when input is degenerate. Reject any
            # of those and fall back rather than propagate them downstream.
            try:
                cleaned_arr = np.asarray(cleaned, dtype=np.float32)
                if (cleaned_arr.size == 0
                        or cleaned_arr.size != audio.size
                        or not np.all(np.isfinite(cleaned_arr))):
                    return self._spectral_subtract(audio)
                return cleaned_arr
            except BaseException as e:
                err = _safe_exc("noisereduce_output", e)
                with self._stats_lock:
                    self._last_error = err
                _dprint(err)
                return self._spectral_subtract(audio)

        return self._spectral_subtract(audio)

    def _spectral_subtract(self, audio: np.ndarray) -> np.ndarray:
        """Cheap one-frame FFT noise gate. Tracks the noise magnitude
        spectrum whenever the chunk RMS is low, then subtracts it from
        louder chunks. Strength is bounded so speech can't be zeroed."""
        if audio.size < 64:
            return audio
        try:
            spec = np.fft.rfft(audio)
        except Exception:
            return audio
        mag = np.abs(spec)
        phase = np.angle(spec)
        rms = float(np.sqrt(np.mean(audio * audio)))
        with self._ns_lock:
            if rms < 0.005:
                if (self._ns_noise_mag is None
                        or self._ns_noise_mag.shape != mag.shape):
                    self._ns_noise_mag = mag.copy()
                else:
                    self._ns_noise_mag = (
                        self._ns_alpha * self._ns_noise_mag
                        + (1.0 - self._ns_alpha) * mag
                    )
            noise = self._ns_noise_mag.copy() if self._ns_noise_mag is not None else None
        if noise is None or noise.shape != mag.shape:
            return audio
        # Over-subtraction factor scales with configured strength.
        k = 1.0 + 0.8 * float(self.ns_strength)
        cleaned_mag = np.maximum(mag - k * noise, 0.1 * mag)
        cleaned = cleaned_mag * np.exp(1j * phase)
        try:
            out = np.fft.irfft(cleaned, n=audio.size).astype(np.float32, copy=False)
        except Exception:
            return audio
        return out

    # ── layer 3: automatic gain control ───────────────────────────────

    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """Wiener entropy: geometric_mean / arithmetic_mean of |FFT|.

        Ranges from ~0 (pure tone / single harmonic) to ~1 (white noise).
        Returns 0.0 when the frame is too short or silent — callers treat
        that as "tonal" and apply full AGC gain.
        """
        if audio.size < 64:
            return 0.0
        try:
            mag = np.abs(np.fft.rfft(audio))
        except Exception:
            return 0.0
        # Drop the DC bin so a slow drift doesn't bias the metric tonal.
        if mag.size > 1:
            mag = mag[1:]
        if mag.size == 0:  # pragma: no cover - unreachable: the [1:] slice only runs when size>1, which always leaves >=1 bin; a 1-bin spectrum skips the slice entirely
            return 0.0
        mag = mag + 1e-10
        arith = float(np.mean(mag))
        if arith < 1e-9:
            return 0.0
        geo = float(np.exp(np.mean(np.log(mag))))
        return max(0.0, min(1.0, geo / arith))

    def _agc(self, audio: np.ndarray) -> np.ndarray:
        """Smooth gain to hold a target RMS. Bounded by agc_max_gain so
        a silent frame doesn't get amplified into pure noise.

        A spectral peakedness gate scales the gain back toward 1.0 when
        the input spectrum is broadband (high flatness) — without it,
        cold-start ambient noise (rms≈0.004) gets boosted past
        VAD_THRESHOLD and triggers false speech detection.
        """
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        if rms < 1e-6:
            return audio
        with self._agc_lock:
            if self._agc_running_rms <= 1e-6:
                self._agc_running_rms = rms
            else:
                self._agc_running_rms = (
                    self._agc_smooth * self._agc_running_rms
                    + (1.0 - self._agc_smooth) * rms
                )
            tracked = self._agc_running_rms
        if tracked < 1e-6:
            return audio  # pragma: no cover - unreachable defensive guard: tracked is an EMA of values each >=1e-6 (input rms passed the 1e-6 gate above; seeded running_rms is set to that rms), so the smoothed value can't fall below 1e-6
        gain = self.agc_target_rms / tracked
        max_g = float(self.agc_max_gain)
        if max_g > 0:
            gain = max(1.0 / max_g, min(max_g, gain))

        # Spectral peakedness gate. Only matters when we'd amplify
        # (gain > 1.0) — broadband noise mustn't be lifted into the VAD
        # band. Tonal frames (speech, music notes) pass through unchanged.
        if gain > 1.0 and audio.size >= 64:
            flat = self._spectral_flatness(audio)
            with self._agc_lock:
                if self._agc_flatness_init:
                    self._agc_flatness = (
                        self._agc_flatness_smooth * self._agc_flatness
                        + (1.0 - self._agc_flatness_smooth) * flat
                    )
                else:
                    # Cold-start: seed with the current frame so a fan in
                    # the room gates frame 1, not frame 5.
                    self._agc_flatness = flat
                    self._agc_flatness_init = True
                # Clamp to bounds so a long run of borderline-broadband
                # frames can't drift the smoothed estimate past the
                # sigmoid center and pin the gate closed permanently
                # (which would starve VAD on real speech once recovered).
                if self._agc_flatness < self._agc_flatness_min:
                    self._agc_flatness = self._agc_flatness_min
                elif self._agc_flatness > self._agc_flatness_max:
                    self._agc_flatness = self._agc_flatness_max
                smoothed = self._agc_flatness
            # Sigmoid → 1.0 for tonal (gate open), → 0.0 for broadband
            # (gate closed). Width=0.05 keeps the transition tight so
            # speech with mild noise still gets amplified.
            z = (smoothed - self._agc_flatness_center) / max(
                1e-6, self._agc_flatness_width
            )
            gate = 1.0 / (1.0 + float(np.exp(z)))
            gain = 1.0 + gate * (gain - 1.0)

        out = audio * float(gain)
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────

_singleton: Optional[AudioProcessor] = None
_singleton_lock = threading.Lock()


def get_processor(sample_rate: int = 16000) -> AudioProcessor:
    """Return the global processor, building it on first call.

    Re-builds when the sample-rate changes (eg. system audio loopback
    capture at 48 kHz vs the mic at 16 kHz) so each capture path gets a
    coherent noise profile."""
    global _singleton
    with _singleton_lock:
        if _singleton is None or _singleton.sample_rate != int(sample_rate):
            _singleton = AudioProcessor(sample_rate=int(sample_rate))
        return _singleton


def feed_playback(audio: np.ndarray,
                  sample_rate: Optional[int] = None) -> None:
    """Module-level helper so TTS callers don't have to import the class."""
    try:
        get_processor().feed_playback(audio, sample_rate=sample_rate)
    except Exception as e:
        _dprint(f"feed_playback shim failed: {e}")


def is_playback_recent(within: float = 0.2) -> bool:
    try:
        return get_processor().is_playback_recent(within=within)
    except Exception:
        return False


def recent_peak_rms(within: float = 60.0) -> float:
    """Module-level helper for core.tts stress detection. Returns 0.0 on
    any failure so a missing audio processor never blocks preset selection."""
    try:
        return get_processor().recent_peak_rms(within=within)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# VAD activity tracking (consumed by skills/self_diagnostic auto-queue)
# ──────────────────────────────────────────────────────────────────────
# record_speech() in bobert_companion calls note_vad_active() whenever the
# VAD threshold trips, and note_vad_poll() on every chunk it inspects. The
# self-diagnostic probe spots a "stall" — the input loop is actively
# polling chunks but no chunk has crossed the VAD floor — as the gap
# between the two timestamps growing past _VAD_STALL_THRESHOLD_S while
# JARVIS is supposed to be listening. Used to distinguish "user is just
# silent" (poll fresh + last_active stale + JARVIS asleep → fine) from "mic
# is dead / pipeline broke" (poll fresh + last_active stale + JARVIS awake
# for over a minute → fix request).
_vad_state_lock = threading.Lock()
_vad_state: dict = {
    "last_vad_active_ts":     0.0,
    "last_vad_poll_ts":       0.0,
    "vad_session_start":      0.0,
    "total_vad_trips":        0,
    # 2026-05-30 [self-heal]: silent-mic instrumentation. Tracks the most
    # recent chunk whose raw RMS crossed _AUDIBLE_RMS_FLOOR — distinguishes
    # "user is quiet" (mic alive, rms hovers above 1e-5) from "mic is
    # delivering literal zero-frames" (driver / privacy block / dead mic).
    "last_audible_chunk_ts":  0.0,
}

# Anything above this RMS counts as "the mic is alive". The hardware noise
# floor of a working capture device sits well above this even in a quiet
# room; a chunk under 1e-5 means the driver is handing us null samples.
_AUDIBLE_RMS_FLOOR = 1.0e-5


def note_vad_active(ts: Optional[float] = None) -> None:
    """Mark that the VAD just tripped (rms > threshold). Safe from any thread."""
    t = float(time.time() if ts is None else ts)
    with _vad_state_lock:
        _vad_state["last_vad_active_ts"] = t
        _vad_state["last_vad_poll_ts"]   = t
        _vad_state["total_vad_trips"]   += 1


def note_vad_poll(ts: Optional[float] = None) -> None:
    """Mark that the input loop inspected a chunk this tick (even if VAD
    didn't trip). Used to distinguish a stalled input pipeline from idle."""
    t = float(time.time() if ts is None else ts)
    with _vad_state_lock:
        _vad_state["last_vad_poll_ts"] = t
        if _vad_state["vad_session_start"] == 0.0:
            _vad_state["vad_session_start"] = t


def get_vad_state() -> dict:
    """Snapshot of the VAD activity counters. Returns a shallow copy so the
    caller can compute derived fields without holding the lock."""
    with _vad_state_lock:
        return dict(_vad_state)


def seconds_since_vad_active() -> float:
    """Seconds since the last VAD trip. Returns float('inf') if VAD has
    never tripped this session — callers should treat that as "no data
    yet" rather than "infinitely stalled"."""
    with _vad_state_lock:
        ts = _vad_state["last_vad_active_ts"]
    if ts <= 0.0:
        return float("inf")
    return max(0.0, time.time() - ts)


def note_raw_rms(rms: float, ts: Optional[float] = None) -> None:
    """Record the raw (pre-processing) RMS of a mic chunk. When rms crosses
    _AUDIBLE_RMS_FLOOR we update last_audible_chunk_ts; the silent-mic
    health check in bobert_companion.record_speech consumes that timestamp
    to distinguish a hardware-silent mic from a quiet user. Safe from any
    thread; called automatically by AudioProcessor.process() and may also
    be called directly by capture paths that bypass the processor (e.g.
    record_speech's raw VAD pre-check)."""
    t = float(time.time() if ts is None else ts)
    with _vad_state_lock:
        if float(rms) > _AUDIBLE_RMS_FLOOR:
            _vad_state["last_audible_chunk_ts"] = t


def seconds_since_audible_chunk() -> float:
    """Seconds since the mic last delivered a chunk with raw RMS above
    _AUDIBLE_RMS_FLOOR. When the mic has never produced audible audio
    *this* session but polling is active, returns time since the session
    started — so the silent-mic warning fires after MIC_SILENT_WARN_SECONDS
    even from cold start. Returns float('inf') when polling hasn't begun."""
    with _vad_state_lock:
        ts_audible = _vad_state["last_audible_chunk_ts"]
        ts_start = _vad_state["vad_session_start"]
    now = time.time()
    if ts_audible > 0.0:
        return max(0.0, now - ts_audible)
    if ts_start > 0.0:
        return max(0.0, now - ts_start)
    return float("inf")


# ──────────────────────────────────────────────────────────────────────
# Media echo cancellation (MEDIA_AEC_MODE, 2026-10-05)
# ──────────────────────────────────────────────────────────────────────
# The 0.7x duck in _aec() above only covers JARVIS's OWN voice for 150 ms.
# Over a video the desk mic hears the desk speakers at 0.013-0.063 RMS
# against a 0.008 capture threshold, so every capture filled with video and
# ran to 30 s. MediaEchoCanceller subtracts what the PC itself plays (the
# loopback of the default render endpoint, core/loopback_ref.py) from the
# mic, the way a phone cancels its own loudspeaker:
#
#   * partitioned-block frequency-domain NLMS (16 ms blocks x 20 partitions
#     = a 320 ms filter, step 0.5), as a background / foreground pair: the
#     background adapts every block, the output (foreground) filter is only
#     replaced by a background that is doing better - so the owner talking
#     over the video cannot drag the output filter off - and a background
#     that diverged is reset from the foreground;
#   * GCC-PHAT bulk delay on loud reference stretches, re-measured every
#     MEDIA_AEC_DELAY_EVERY_S and soon after any gap, and clock DRIFT from
#     the slope of those delays over MEDIA_AEC_DRIFT_WINDOW_S: a USB desk
#     mic and USB speakers run on separate crystals, and without
#     drift compensation the benchmark's ERLE fell to 11.5 dB at 20 ppm
#     (6.5 dB at 40). The reference is read at fractional positions
#     (windowed sinc) at the estimated rate;
#   * a residual-echo suppressor (STFT Wiener gain from the echo estimate,
#     the leak tracked by a running minimum). Its output ("sup") is for
#     DETECTION only - capture start / end, loudness, the pre-gate. Speech
#     recognition and voice-ID get the linear output ("lin"): the suppressor
#     cost Parakeet 0.938 -> 0.901 at -5 dB in the research benchmark.
#
# Guards: output louder than input by 3 dB for 1 s resets the filters
# (pass-through while it reconverges); ERLE under 10 dB for 60 s passes the
# mic through and says "AEC not converging" once; a silent reference passes
# the mic through and keeps the weights. Pure numpy; never raises out of
# process() (any failure passes the chunk through). The reference audio
# lives only in RAM (core/loopback_ref.py), never on disk or in a log.
MEDIA_AEC_BLOCK = 256               # 16 ms at 16 kHz
MEDIA_AEC_PARTITIONS = 20           # x 16 ms = a 320 ms filter
MEDIA_AEC_MU = 0.5
MEDIA_AEC_MARGIN_S = 0.030          # the reference leads the echo by this
MEDIA_AEC_DELAY_EVERY_S = 10.0
MEDIA_AEC_FIRST_DELAY_S = 1.0       # the first measurement after a gap
# Until the drift is known, measure twice a second: 20 ppm moves the echo
# 0.32 samples a second, and an NLMS chasing that caps ERLE near 7 dB on
# broadband audio, so the first estimate (>= 4 points over >= 3 s) cannot
# wait for the slow schedule. Later estimates refine over a longer span.
# Converged, the measurement is the filter's own direct-path tap (cheap and
# far steadier than a GCC peak, which a strong reflection can win).
MEDIA_AEC_FAST_EVERY_S = 0.5
MEDIA_AEC_FAST_FOR_S = 20.0
MEDIA_AEC_FIRST_DRIFT_SPAN_S = 1.0
MEDIA_AEC_DRIFT_SPAN_S = 30.0
MEDIA_AEC_SMALL_STEP = 2.0e-6       # a drift correction under 2 ppm
MEDIA_AEC_DELAY_WINDOW_S = 4.0      # each measurement looks this far back
MEDIA_AEC_DRIFT_WINDOW_S = 120.0
MEDIA_AEC_MAX_LAG_S = 0.400         # GCC searches the echo up to this late
MEDIA_AEC_MAX_LEAD_S = 0.200        # ... and this early
MEDIA_AEC_REF_ACTIVE_RMS = 1e-4     # quieter reference = silence
MEDIA_AEC_DIVERGE_DB = 3.0
MEDIA_AEC_DIVERGE_S = 1.0
MEDIA_AEC_POOR_ERLE_DB = 10.0
MEDIA_AEC_TRACK_ERLE_DB = 6.0       # the filter's peak is the echo path
MEDIA_AEC_POOR_S = 60.0
MEDIA_AEC_RECOVER_ERLE_DB = 12.0
_SINC_TAPS = 16


def _sinc_kernel(frac: np.ndarray) -> np.ndarray:
    """Kaiser-windowed sinc weights (len(frac), _SINC_TAPS) for reading a
    signal at integer index + frac (taps at -7..+8)."""
    k = np.arange(-_SINC_TAPS // 2 + 1, _SINC_TAPS // 2 + 1, dtype=np.float64)
    win = np.kaiser(_SINC_TAPS + 2, 8.0)[1:-1]
    x = k[None, :] - frac[:, None]
    return np.sinc(x * 0.95) * 0.95 * win[None, :]


def gcc_phat_lag(x: np.ndarray, d: np.ndarray, max_lag: int,
                 max_lead: int = 0) -> "tuple[float, float]":
    """(lag, peak): how many samples ``d`` trails ``x`` (sub-sample,
    parabolic peak; negative = leads, down to -max_lead) and the PHAT
    correlation peak (under ~0.05 = no reliable echo). Pure numpy.

    Hann-windowed, PHAT-beta 0.8: with a rectangular window and full
    whitening a strong TONE (music) leaks into many bins that all carry the
    same phase difference, and they add up to a false peak at lag 0 - the
    synthetic gated 220 Hz tone did exactly that, 1 measurement in 3."""
    x = np.asarray(x, np.float64)
    d = np.asarray(d, np.float64)
    if len(x) == len(d) and len(x) > 8:
        w = np.hanning(len(x))
        x = x * w
        d = d * w
    n = len(x) + len(d)
    L = 1 << int(np.ceil(np.log2(max(2, n))))
    G = np.fft.rfft(d, L) * np.conj(np.fft.rfft(x, L))
    G /= (np.abs(G) + 1e-12) ** 0.8
    cc = np.fft.irfft(G, L)
    lags = np.concatenate([cc[L - max_lead:] if max_lead > 0 else cc[:0],
                           cc[:max_lag + 1]])
    k = int(np.argmax(lags))
    peak = float(lags[k])
    frac = 0.0
    if 0 < k < len(lags) - 1:
        a, b, c = lags[k - 1], lags[k], lags[k + 1]
        den = a - 2 * b + c
        if den != 0:
            frac = float(0.5 * (a - c) / den)
    return float(k - max_lead) + frac, peak


class _PBFDAF:
    """Partitioned-block frequency-domain NLMS (overlap-save, constrained
    gradient) with a background / foreground pair. block(x, d) -> (e, y):
    the error (mic minus the echo estimate) and the echo estimate, both from
    the FOREGROUND filter."""

    def __init__(self, B=MEDIA_AEC_BLOCK, P=MEDIA_AEC_PARTITIONS,
                 mu=MEDIA_AEC_MU, ref_active=MEDIA_AEC_REF_ACTIVE_RMS):
        self.B, self.P, self.mu = int(B), int(P), float(mu)
        self.K = self.B + 1
        self.ref_active = float(ref_active)
        self.reset()

    def reset(self) -> None:
        P, K, B = self.P, self.K, self.B
        self.Wb = np.zeros((P, K), np.complex128)
        self.Wf = np.zeros((P, K), np.complex128)
        self.Xbuf = np.zeros((P, K), np.complex128)
        self.head = 0
        self.xprev = np.zeros(B)
        self.Pxx = np.full(K, 1e-6)
        self.Eb = 0.0
        self.Ef = 0.0
        self.zB = np.zeros(B)
        self.n_copy = 0
        self.n_reset = 0

    def shift(self, s: int) -> None:
        """Move both filters' taps ``s`` samples EARLIER (s > 0) or later
        (s < 0): the echo path is re-aligned by the same amount, so a
        converged filter stays converged (zeros enter at the far end)."""
        s = int(s)
        if s == 0:
            return
        B, P = self.B, self.P
        for W in (self.Wb, self.Wf):
            h = np.fft.irfft(W, axis=1)[:, :B].reshape(-1)     # P*B taps
            out = np.zeros_like(h)
            if s > 0:
                out[:len(h) - s] = h[s:] if s < len(h) else 0.0
            else:
                out[-s:] = h[:len(h) + s] if -s < len(h) else 0.0
            g = np.zeros((P, 2 * B))
            g[:, :B] = out.reshape(P, B)
            W[:] = np.fft.rfft(g, axis=1)

    def taps(self) -> np.ndarray:
        """The foreground filter as time-domain taps (P*B)."""
        return np.fft.irfft(self.Wf, axis=1)[:, :self.B].reshape(-1)

    def peak(self) -> "float | None":
        """The foreground filter's strongest tap (the direct path), to a
        sub-sample (sinc-interpolated around the peak); None when the
        filter is still empty."""
        h = np.fft.irfft(self.Wf, axis=1)[:, :self.B].reshape(-1)
        a = np.abs(h)
        k = int(np.argmax(a))
        if a[k] <= 1e-9:
            return None
        lo, hi = max(0, k - 8), min(len(h), k + 9)
        taps = h[lo:hi]
        grid = np.arange(-1.0, 1.0 + 1e-9, 1.0 / 32.0) + k
        n = np.arange(lo, hi)
        vals = np.abs(np.sinc(grid[:, None] - n[None, :]) @ taps)
        return float(grid[int(np.argmax(vals))])

    def _conv(self, W, idx):
        Y = np.einsum("pk,pk->k", W, self.Xbuf[idx])
        return np.fft.irfft(Y)[self.B:]

    def block(self, x: np.ndarray, d: np.ndarray):
        B = self.B
        X = np.fft.rfft(np.concatenate([self.xprev, x]))
        self.xprev = x
        self.head = (self.head + 1) % self.P
        self.Xbuf[self.head] = X
        idx = (self.head - np.arange(self.P)) % self.P
        Ps = np.sum(self.Xbuf.real ** 2 + self.Xbuf.imag ** 2, axis=0)
        self.Pxx = 0.5 * self.Pxx + 0.5 * Ps
        yb = self._conv(self.Wb, idx)
        eb = d - yb
        active = float(np.mean(x * x)) > self.ref_active ** 2
        if active:
            Eb = np.fft.rfft(np.concatenate([self.zB, eb]))
            norm = self.mu / (self.Pxx + 1e-6 * 2 * B + 1e-10)
            G = np.conj(self.Xbuf[idx]) * (Eb * norm)[None, :]
            g = np.fft.irfft(G, axis=1)
            g[:, B:] = 0.0                       # gradient constraint
            self.Wb += np.fft.rfft(g, axis=1)
        yf = self._conv(self.Wf, idx)
        ef = d - yf
        a = 0.85
        self.Eb = a * self.Eb + (1 - a) * float(np.dot(eb, eb))
        self.Ef = a * self.Ef + (1 - a) * float(np.dot(ef, ef))
        if active:
            if self.Eb < 0.7 * self.Ef:
                self.Wf[:] = self.Wb
                self.n_copy += 1
                self.Ef = self.Eb
                ef, yf = eb, yb
            elif self.Eb > 2.0 * self.Ef and self.Ef > 0:
                self.Wb[:] = self.Wf
                self.n_reset += 1
                self.Eb = self.Ef
        return ef, yf, active


def _onset(h: np.ndarray, frac: float = 0.5) -> "int | None":
    """The first tap at least ``frac`` of the strongest one: the direct path
    of a learnt echo path (a reflection can be the strongest tap, rarely
    the first strong one). None for an empty filter."""
    a = np.abs(h)
    m = float(a.max()) if a.size else 0.0
    if m <= 1e-9:
        return None
    return int(np.argmax(a >= frac * m))


def _path_shift(a: np.ndarray, b: np.ndarray,
                max_shift: int = 8) -> "float | None":
    """How far the learnt echo path moved between two snapshots of the
    filter's taps (sub-sample; + = later). Cross-correlates the WHOLE
    response, so a reflection overtaking the direct path does not read as a
    move. None when either snapshot is empty."""
    na, nb = float(np.dot(a, a)), float(np.dot(b, b))
    if na <= 1e-18 or nb <= 1e-18:
        return None
    L = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(b, L) * np.conj(np.fft.rfft(a, L)), L)
    lags = np.concatenate([c[L - max_shift:], c[:max_shift + 1]])
    k = int(np.argmax(lags))
    if lags[k] <= 0:
        return None
    frac = 0.0
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = lags[k - 1], lags[k], lags[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            frac = float(0.5 * (y0 - y2) / den)
    return float(k - max_shift) + frac


class _ResidualSuppressor:
    """Streaming residual-echo suppressor: 512-point sqrt-Hann STFT, hop
    256 (one block), a Wiener-style gain per bin from the echo estimate; the
    echo leak per bin is the running MINIMUM over ~1.5 s of S_ee / S_yy (the
    single-talk moments between the owner's words set it, so his speech does
    not inflate the estimate). Output lags input by one hop (16 ms)."""

    NFFT, HOP = 512, 256

    def __init__(self, beta=2.0, gmin=0.1, alpha=0.6, win_s=1.5,
                 sr=16000):
        self.beta, self.gmin, self.alpha = float(beta), float(gmin), float(alpha)
        self.win = np.sqrt(np.hanning(self.NFFT + 1)[:-1])
        self.W = max(3, int(win_s * sr / self.HOP))
        self.reset()

    def reset(self) -> None:
        K = self.NFFT // 2 + 1
        self.e_prev = np.zeros(self.HOP)
        self.y_prev = np.zeros(self.HOP)
        self.See = np.zeros(K)
        self.Syy = np.zeros(K)
        self.ratios = np.ones((self.W, K))
        self.ri = 0
        self.ola = np.zeros(self.HOP)

    def block(self, e: np.ndarray, y: np.ndarray) -> np.ndarray:
        fe = np.concatenate([self.e_prev, e]) * self.win
        fy = np.concatenate([self.y_prev, y]) * self.win
        self.e_prev, self.y_prev = e, y
        E = np.fft.rfft(fe)
        Y = np.fft.rfft(fy)
        a = self.alpha
        self.See = a * self.See + (1 - a) * (E.real ** 2 + E.imag ** 2)
        self.Syy = a * self.Syy + (1 - a) * (Y.real ** 2 + Y.imag ** 2)
        self.ratios[self.ri] = np.minimum(self.See / (self.Syy + 1e-12), 1.0)
        self.ri = (self.ri + 1) % self.W
        eta = self.ratios.min(axis=0)
        R = 1.5 * eta * self.Syy
        G = np.clip(1.0 - self.beta * R / (self.See + 1e-12), self.gmin, 1.0)
        fr = np.fft.irfft(E * G, self.NFFT) * self.win
        out = self.ola + fr[:self.HOP]
        self.ola = fr[self.HOP:].copy()
        return out


class MediaEchoCanceller:
    """Echo cancellation of what the PC plays, for the owner's mic.

    ``ref`` is the reference source (core/loopback_ref.LoopbackReference or
    a test double): ``read(start, n)`` -> the samples [start, start+n) or
    None when they are not in its ring, ``index_at(t)`` -> the (fractional)
    reference index the PC played at monotonic time t (None when unknown),
    ``n_written``, ``gap_seq`` (bumped on every discontinuity) and,
    optionally, ``wait_for(index, timeout)`` (block until the index is
    written; the reference may trail the mic by a packet).

    process(mic, t=None) -> (lin, sup): the mic chunk with the echo removed
    (lin: for STT and voice-ID) and the residual-suppressed copy (sup: for
    detection), both the chunk's length (sup trails lin by 16 ms). ``t`` =
    monotonic time the chunk arrived (its last sample). Consecutive chunks are
    one SESSION - the caller says when a new stream starts (new_session():
    a new capture stream, a reopened bus, dropped frames); a gap in the
    reference starts one too. A session is placed on the reference by the
    wall clock (the first chunk's ``t``) and then re-measured (GCC-PHAT)
    soon after. The filters and the delay / drift estimates carry over.
    Thread-safe; never raises (a failure passes the chunk through)."""

    def __init__(self, ref=None, sample_rate: int = 16000,
                 clock=time.monotonic, delay_prior_s: float = 0.040):
        self.sr = int(sample_rate)
        self.ref = ref
        self._clock = clock
        self._mu = threading.Lock()
        self.margin = int(round(MEDIA_AEC_MARGIN_S * self.sr))
        self.max_lag = int(round(MEDIA_AEC_MAX_LAG_S * self.sr))
        self.max_lead = int(round(MEDIA_AEC_MAX_LEAD_S * self.sr))
        self.f = _PBFDAF()
        self.res = _ResidualSuppressor(sr=self.sr)
        # Samples the echo trails the wall-clock reference by (speaker +
        # room + capture latency, less the loopback's): a prior until the
        # first GCC measurement replaces it.
        self.delay_wall = float(delay_prior_s) * self.sr
        self.eps = 0.0              # reference rate - 1 (clock drift)
        self._last_drift_step = 0.0
        self.stats = {"resets": 0, "realigns": 0, "delay_measures": 0,
                      "drift_updates": 0, "chunks": 0, "passthrough": 0,
                      "ref_late": 0, "sessions": 0, "not_converging": 0}
        self._hist_n = int(self.sr * MEDIA_AEC_DELAY_WINDOW_S)
        self._mic_hist = np.zeros(self._hist_n)
        self._x_hist = np.zeros(self._hist_n)
        self._erle_db = None
        self._erle_num = 0.0
        self._erle_den = 0.0
        self._div_blocks = 0
        self._poor_since = None
        self._passthrough = False
        self._said_poor = False
        self._blocks = 0            # blocks processed, all sessions
        self._session = False
        self._last_t = None
        self._pending = (np.zeros(0), np.zeros(0))
        self._fifo = np.zeros(0)
        self._new_session_state()

    # ── session / alignment ──────────────────────────────────────────────
    def _new_session_state(self) -> None:
        self._k = 0                    # session samples processed
        self._A = 0.0                  # reference index of session sample 0
        self._gap_seq = None
        self._next_measure = None
        self._hist_w = 0               # circular write index
        self._hist_fill = 0
        self._drift_pts = []           # (session sample, continuous lag)
        self._sess_shift = 0.0
        self._gcc_last = None
        self._h_prev = None            # the filter's taps at the last measure
        self._gcc_series = []          # per-path GCC lag series
        self._track_pos = 0.0          # how far the learnt path has moved

    def new_session(self) -> None:
        """The mic stream restarted (a new capture stream): re-anchor on the
        next chunk."""
        with self._mu:
            self._session = False

    def reset(self) -> None:
        """Forget everything learnt (filters, delay, drift)."""
        with self._mu:
            self.f.reset()
            self.res.reset()
            self._session = False
            self._erle_db = None
            self._passthrough = False
            self.eps = 0.0

    def _anchor(self, t_first: float) -> bool:
        """Place session sample 0 (heard at monotonic ``t_first``) on the
        reference. False when the reference cannot place it yet."""
        ref = self.ref
        if ref is None:
            return False
        try:
            at = ref.index_at(t_first)
        except Exception:
            at = None
        if at is None:
            return False
        self._new_session_state()
        self._A = float(at) - self.delay_wall + self.margin
        self._gap_seq = getattr(ref, "gap_seq", None)
        self._session = True
        self._next_measure = int(MEDIA_AEC_FIRST_DELAY_S * self.sr)
        self.f.xprev = np.zeros(self.f.B)
        self.stats["sessions"] += 1
        return True

    def _ref_block(self, k0: int, n: int):
        """The aligned reference for session samples [k0, k0+n): an array,
        "late" (not written yet, even after a short wait) or None (gone from
        the ring)."""
        ref = self.ref
        pos = self._A + (k0 + np.arange(n, dtype=np.float64)) * (1.0 + self.eps)
        half = _SINC_TAPS // 2
        i0 = int(np.floor(pos[0])) - half + 1
        i1 = int(np.floor(pos[-1])) + half + 1
        try:
            written = int(getattr(ref, "n_written", 0))
        except Exception:
            written = 0
        if i1 + 1 > written:
            wait = getattr(ref, "wait_for", None)
            if callable(wait):
                try:
                    wait(i1 + 1, 0.06)
                except Exception:
                    pass
            if i1 + 1 > int(getattr(ref, "n_written", 0)):
                return "late"
        seg = ref.read(i0, i1 - i0 + 1)
        if seg is None:
            return None
        seg = np.asarray(seg, np.float64)
        ip = np.floor(pos).astype(np.int64)
        frac = pos - ip
        if self.eps == 0.0 and float(np.max(np.abs(frac))) < 1e-9:
            return seg[ip - i0]
        w = _sinc_kernel(frac)
        kk = np.arange(-half + 1, half + 1)
        idx = np.clip((ip - i0)[:, None] + kk[None, :], 0, len(seg) - 1)
        return np.sum(seg[idx] * w, axis=1)

    def _push_hist(self, d: np.ndarray, x: np.ndarray) -> None:
        n, w = len(d), self._hist_w
        end = w + n
        if end <= self._hist_n:
            self._mic_hist[w:end] = d
            self._x_hist[w:end] = x
        else:
            cut = self._hist_n - w
            self._mic_hist[w:] = d[:cut]
            self._x_hist[w:] = x[:cut]
            self._mic_hist[:n - cut] = d[cut:]
            self._x_hist[:n - cut] = x[cut:]
        self._hist_w = end % self._hist_n
        self._hist_fill = min(self._hist_n, self._hist_fill + n)

    def _hist(self, n: int):
        order = np.roll(np.arange(self._hist_n), -self._hist_w)[-n:]
        return self._x_hist[order], self._mic_hist[order]

    def _realign(self, err: float) -> None:
        """Move the reference by ``err`` samples (+ = the echo is later than
        planned) so it leads the echo by the margin again. The filter learnt
        the path at the old alignment: its taps move with it (instead of
        starting over) and its reference history is re-read."""
        step = int(round(err))
        if step == 0:
            return
        self._A -= step
        self._sess_shift += step
        self.delay_wall += step
        self.f.shift(step)
        self._rebuild_ref_history()
        self._h_prev = None                # the drift series restart
        self._gcc_series = []
        self._drift_pts = []
        self.stats["realigns"] += 1
        self._hist_fill = 0            # the GCC history straddles the shift

    def _measure_delay(self) -> None:
        """Where is the echo, and is it moving?

        Until the drift is known (and while its corrections are still
        large): GCC-PHAT between the aligned reference and the mic over the
        last second, twice a second. Its lag series (sub-sample) gives the
        drift within a couple of seconds; a jump of more than 2 samples
        means a reflection won the peak, and the series restarts. The
        reference is re-aligned only when the filter needs it, and - unless
        the echo is out of the filter's reach - only when two measurements
        in a row agree.

        Once the drift is known and the filter has converged (ERLE >=
        MEDIA_AEC_TRACK_ERLE_DB): how far the WHOLE learnt path moved since
        the last look (_path_shift), every 10 s - steadier than any single
        peak - refines the drift, and the first strong tap re-aligns the
        reference if it wanders out of place."""
        converged = (self._erle_db is not None
                     and self._erle_db >= MEDIA_AEC_TRACK_ERLE_DB)
        quick = (self.stats["drift_updates"] == 0
                 or self._last_drift_step > MEDIA_AEC_SMALL_STEP)
        if converged and not quick:
            h = self.f.taps()
            self.stats["delay_measures"] += 1
            onset = (_onset(h) if self._erle_db >= MEDIA_AEC_POOR_ERLE_DB
                     else None)
            if onset is not None and (onset < self.margin / 3.0
                                      or onset > self.margin + 0.010 * self.sr):
                self._realign(onset - self.margin)
                return
            prev, self._h_prev = self._h_prev, h
            if prev is None:
                self._track_pos = 0.0
                self._drift_pts = [(float(self._k), 0.0)]
                return
            delta = _path_shift(prev, h)
            if delta is None or abs(delta) > 4.0:
                # The learnt path changed shape, not place: a new series.
                self._track_pos = 0.0
                self._drift_pts = [(float(self._k), 0.0)]
                return
            self._track_pos += delta
            self._drift_pts.append((float(self._k), self._track_pos))
            horizon = MEDIA_AEC_DRIFT_WINDOW_S * self.sr
            self._drift_pts = [q for q in self._drift_pts
                               if self._k - q[0] <= horizon]
            self._update_drift()
            return
        self._h_prev = None
        n = min(self._hist_fill,
                int((1.0 if quick else MEDIA_AEC_DELAY_WINDOW_S) * self.sr))
        if n < int(1.0 * self.sr):
            return
        x, d = self._hist(n)
        if float(np.sqrt(np.mean(x * x))) < 10 * MEDIA_AEC_REF_ACTIVE_RMS:
            return
        lag, pk = gcc_phat_lag(x, d, self.max_lag, self.max_lead)
        self.stats["delay_measures"] += 1
        if pk < 0.05:
            self._gcc_last = None
            return
        err = lag - self.margin            # + = the echo is later than planned
        # Re-align only when the filter needs it: an echo earlier than a
        # third of the margin risks an acausal path, one far later than the
        # margin starves the filter's tail.
        if lag < self.margin / 3.0 or lag > self.margin + 0.050 * self.sr:
            last, self._gcc_last = self._gcc_last, lag
            reachable = 0 <= lag <= self.f.P * self.f.B - self.f.B
            if reachable and (last is None
                              or abs(last - lag) > 0.002 * self.sr):
                return                     # confirm it on the next measure
            self._gcc_last = None
            self._realign(err)
            return
        self._gcc_last = None
        # The drift series: the lag on the session's clock (re-alignments
        # added back). In a room with strong reflections the PHAT peak hops
        # between paths (a few samples to a few ms apart), so each path
        # keeps its own series: a lag joins the series whose last value is
        # within 2 samples, else it starts one. The drift comes from the
        # longest series.
        q = lag + self._sess_shift
        k_mid = float(self._k) - n / 2.0
        for ser in self._gcc_series:
            if abs(q - ser[-1][1]) <= 2.0:
                ser.append((k_mid, q))
                break
        else:
            self._gcc_series.append([(k_mid, q)])
            del self._gcc_series[:-6]
        self._drift_pts = max(self._gcc_series, key=len)
        self._update_drift()

    def _rebuild_ref_history(self) -> None:
        """Refill the filter's reference spectra (its last P blocks) from
        the reference at the CURRENT alignment, so the taps moved by
        _PBFDAF.shift meet the reference they now expect."""
        f = self.f
        B, P = f.B, f.P
        k_end = self._k
        blocks = []
        for j in range(P + 1):
            k0 = k_end - (P + 1 - j) * B
            if k0 < 0:
                blocks.append(np.zeros(B))
                continue
            xr = self._ref_block(k0, B)
            blocks.append(np.zeros(B) if (xr is None or isinstance(xr, str))
                          else np.asarray(xr, np.float64))
        for j in range(1, P + 1):
            f.head = (f.head + 1) % P
            f.Xbuf[f.head] = np.fft.rfft(np.concatenate([blocks[j - 1],
                                                         blocks[j]]))
        f.xprev = blocks[-1]

    def _update_drift(self) -> None:
        pts = self._drift_pts
        # Quick estimates (a few seconds) until a step is small: an NLMS
        # that is still chasing the drift under-reads it, so the first
        # estimate falls short and the next few close the gap.
        quick = (self.stats["drift_updates"] == 0
                 or self._last_drift_step > MEDIA_AEC_SMALL_STEP)
        span = (MEDIA_AEC_FIRST_DRIFT_SPAN_S if quick
                else MEDIA_AEC_DRIFT_SPAN_S)
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < span * self.sr:
            return
        ks = np.array([p[0] for p in pts], np.float64)
        ls = np.array([p[1] for p in pts], np.float64)
        M = np.vstack([ks, np.ones_like(ks)]).T
        coef, *_ = np.linalg.lstsq(M, ls, rcond=None)
        res = ls - M @ coef
        keep = np.abs(res) < max(1.0, 3 * float(np.std(res)))
        if keep.sum() >= 4:
            coef, *_ = np.linalg.lstsq(M[keep], ls[keep], rcond=None)
        slope = float(coef[0])
        if abs(slope) < 1.0e-6:            # under 1 ppm: noise
            return
        # The lag grows by (eps - eps_true) a sample: eps_true = eps - slope.
        # A moves so the reference position at the current sample does not
        # jump: A + k(1+eps_old) == A' + k(1+eps_new).
        self._A += self._k * slope
        self.eps = float(np.clip(self.eps - slope, -300e-6, 300e-6))
        self._last_drift_step = abs(slope)
        self.stats["drift_updates"] += 1
        self._drift_pts = []
        self._gcc_series = []
        self._h_prev = None

    # ── processing ──────────────────────────────────────────────────────
    def process(self, mic, t: "float | None" = None):
        x = np.asarray(mic, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x, x
        try:
            with self._mu:
                return self._process(x, t)
        except BaseException as e:   # never break the capture loop
            self.stats["last_error"] = _safe_exc("media_aec", e)
            self._session = False
            return x, x

    def _process(self, chunk: np.ndarray, t):
        n = len(chunk)
        self.stats["chunks"] += 1
        if t is None:
            t = float(self._clock())
        self._last_t = t
        ref = self.ref
        gap_seq = getattr(ref, "gap_seq", None) if ref is not None else None
        if (not self._session
                or (gap_seq is not None and gap_seq != self._gap_seq)):
            self._fifo = np.zeros(0)
            self._pending = (np.zeros(0), np.zeros(0))
            if not self._anchor(t - n / self.sr):
                self._session = False
                self.stats["passthrough"] += 1
                return chunk, chunk
        d_all = np.concatenate([self._fifo, chunk.astype(np.float64)])
        B = self.f.B
        nb = len(d_all) // B
        lin = d_all[:nb * B].copy()
        sup = d_all[:nb * B].copy()
        for b in range(nb):
            s0 = b * B
            d = d_all[s0:s0 + B]
            xr = self._ref_block(self._k, B)
            if xr is None:
                # Gone from the ring: pass the rest through and re-anchor.
                self._session = False
                self.stats["passthrough"] += 1
                break
            if isinstance(xr, str):
                # The reference has not arrived yet: this block passes
                # through; the session (and its sample count) goes on.
                self.stats["ref_late"] += 1
                xr = np.zeros(B)
            e, y, active = self.f.block(xr, d)
            self._guard(d, e, active)
            out = d if self._passthrough else e
            lin[s0:s0 + B] = out
            sup[s0:s0 + B] = self.res.block(
                out, np.zeros(B) if self._passthrough else y)
            self._push_hist(d, xr)
            self._k += B
            self._blocks += 1
            if self._next_measure is not None and self._k >= self._next_measure:
                fast = (self.stats["drift_updates"] == 0
                        or self._last_drift_step > MEDIA_AEC_SMALL_STEP
                        or self._k < MEDIA_AEC_FAST_FOR_S * self.sr)
                self._next_measure = self._k + int(
                    (MEDIA_AEC_FAST_EVERY_S if fast
                     else MEDIA_AEC_DELAY_EVERY_S) * self.sr)
                self._measure_delay()
        self._fifo = d_all[nb * B:]
        # Out has the chunk's length: a chunk that is not a whole number of
        # blocks delays the output by the leftover (never with the 1024-
        # sample capture chunks).
        out_lin = np.concatenate([self._pending[0], lin])
        out_sup = np.concatenate([self._pending[1], sup])
        if len(out_lin) < n:
            pad = np.asarray(chunk[:n - len(out_lin)], np.float64)
            out_lin = np.concatenate([pad, out_lin])
            out_sup = np.concatenate([pad, out_sup])
        self._pending = (out_lin[n:], out_sup[n:])
        return (out_lin[:n].astype(np.float32),
                out_sup[:n].astype(np.float32))

    def _guard(self, d: np.ndarray, e: np.ndarray, active: bool) -> None:
        """The ERLE estimate, the divergence reset and the poor-convergence
        pass-through."""
        pd = float(np.dot(d, d))
        pe = float(np.dot(e, e))
        per_s = self.sr / len(d)
        if active and pd > 1e-10:
            self._erle_num = 0.95 * self._erle_num + 0.05 * pd
            self._erle_den = 0.95 * self._erle_den + 0.05 * pe
            self._erle_db = 10.0 * float(np.log10(
                max(self._erle_num, 1e-20) / max(self._erle_den, 1e-20)))
        # Output louder than input by 3 dB for 1 s: the filter diverged.
        if pd > 1e-10 and pe > pd * 10 ** (MEDIA_AEC_DIVERGE_DB / 10.0):
            self._div_blocks += 1
            if self._div_blocks >= MEDIA_AEC_DIVERGE_S * per_s:
                self.f.reset()
                self.res.reset()
                self._div_blocks = 0
                self._erle_num = self._erle_den = 0.0
                self._erle_db = None
                self.stats["resets"] += 1
        else:
            self._div_blocks = 0
        if not active or self._erle_db is None:
            return
        now_s = self._blocks / per_s
        if self._erle_db < MEDIA_AEC_POOR_ERLE_DB:
            if self._poor_since is None:
                self._poor_since = now_s
            elif (not self._passthrough
                  and now_s - self._poor_since >= MEDIA_AEC_POOR_S):
                self._passthrough = True
                self.stats["not_converging"] += 1
        else:
            self._poor_since = None
            if (self._passthrough
                    and self._erle_db >= MEDIA_AEC_RECOVER_ERLE_DB):
                self._passthrough = False

    def take_not_converging(self) -> bool:
        """True once per episode of the poor-ERLE pass-through (the caller
        logs "AEC not converging" once)."""
        with self._mu:
            if self._passthrough and not self._said_poor:
                self._said_poor = True
                return True
            if not self._passthrough:
                self._said_poor = False
            return False

    def erle_db(self) -> "float | None":
        with self._mu:
            return None if self._erle_db is None else float(self._erle_db)

    def status(self) -> dict:
        """Numbers only: ERLE, delay, drift, convergence and counters."""
        with self._mu:
            st = dict(self.stats)
            st.update({
                "erle_db": (None if self._erle_db is None
                            else round(float(self._erle_db), 1)),
                "delay_ms": round(self.delay_wall * 1000.0 / self.sr, 1),
                "drift_ppm": round(self.eps * 1e6, 2),
                "converging": not self._passthrough,
                "session": self._session,
                "fg_copies": self.f.n_copy, "bg_resets": self.f.n_reset,
            })
            return st


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover - manual smoke test, never run under unittest/import
    print("AudioProcessor smoke test")
    proc = get_processor(16000)
    print(f"  status: {proc.status()}")
    # Synthetic input: 1 s tone + noise.
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = (0.02 * np.random.randn(sr)).astype(np.float32)
    raw = tone + noise

    # No playback active → AEC should pass through.
    out = proc.process(raw)
    print(f"  raw rms={np.sqrt(np.mean(raw*raw)):.4f}  "
          f"processed rms={np.sqrt(np.mean(out*out)):.4f}")

    # Simulate active playback - AEC fallback should duck.
    proc.feed_playback(0.5 * tone, sample_rate=sr)
    out2 = proc.process(raw)
    print(f"  with playback active -> processed rms="
          f"{np.sqrt(np.mean(out2*out2)):.4f}")
    print(f"  final status: {proc.status()}")
