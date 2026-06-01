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
    ) -> np.ndarray:
        """Run the three-layer pipeline on a mono float32 chunk.

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
            if rms > 0.0:
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
        if mag.size == 0:
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
            return audio
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
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
