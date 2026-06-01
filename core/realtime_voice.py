"""
core/realtime_voice.py

Streaming voice pipeline for JARVIS — a parallel path to the existing
turn-based record→transcribe→synthesise→play loop in bobert_companion.

Backends
--------
- STT: RealtimeSTT (KoljaB) — Whisper-backed recorder that emits partial
       transcripts WHILE the user is still speaking and signals VAD
       start/stop on a callback API. Replaces the energy-VAD path in
       bobert_companion.record_speech() with sub-300 ms endpointing.
- TTS: RealtimeTTS (KoljaB) — TextToAudioStream that begins playback the
       moment the FIRST sentence is buffered, so the first syllable of a
       reply lands while the LLM is still producing the rest.

Both packages are lazy-imported. Importing this module is always safe;
failure surfaces only at start(). When either dep is missing, callers
should fall back to the legacy turn-based pipeline.

Barge-in
--------
RealtimeTTS exposes stream.stop() which drains its internal text→audio
queue and halts playback. The RealtimeSTT recorder's on_recording_start
callback fires the moment its VAD locks onto the user — that's our
trigger to call barge_in(), which:
    1. Calls stream.stop() to flush queued audio.
    2. Fires the optional on_barge_in() hook so the main loop can
       discard the in-flight LLM response, mark the conversation turn
       as interrupted, etc.

A second guard fires barge_in() on the first non-empty partial
transcript while the stream is playing — covers the rare case where
the VAD-start signal is suppressed (e.g. echo cancellation muted it
during playback).

VOICE_MODE knob
---------------
bobert_companion.VOICE_MODE selects which pipeline drives the main loop:

    'turn_based' — default, existing record_speech + synthesise path
    'realtime'   — engage this streaming pipeline; falls back silently
                   to 'turn_based' if is_available() returns False at
                   boot, so an uninstalled optional dep never kills
                   JARVIS

Hooking the pipeline into the listen loop is a separate wire-up step —
this module ships the engine and a clean public API; the existing
turn-based loop stays the default until the wire-up lands.

Public API
----------
    pipe = RealtimeVoicePipeline(
        on_user_utterance=lambda text: ...,     # final transcript
        on_partial_transcript=lambda text: ...,  # partials (optional)
        on_barge_in=lambda: ...,                 # user interrupted TTS
        stt_model='base', stt_language='en',
        tts_engine='system', tts_voice='en-GB-RyanNeural',
    )
    pipe.start()                                 # background threads up
    pipe.feed_response_chunk("Hello, sir. ")     # stream LLM tokens
    pipe.feed_response_chunk("Right away.")
    pipe.flush_response()                        # signal end-of-stream
    ...
    pipe.stop()

    is_available() → (True, '') | (False, reason)

The single playback queue is owned by the underlying TextToAudioStream;
barge_in() flushes it atomically.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional


# Tunables. Conservative defaults that match the existing turn-based path;
# tuned for a desktop mic with mild echo cancellation.
DEFAULT_STT_MODEL = "base"
DEFAULT_STT_LANGUAGE = "en"
DEFAULT_TTS_ENGINE = "system"               # 'system' | 'azure' | 'elevenlabs' | 'coqui'
DEFAULT_TTS_VOICE = "en-GB-RyanNeural"      # ignored by SystemEngine; used by Azure/EL
DEFAULT_SAMPLE_RATE = 16000

# RealtimeSTT VAD sensitivities — higher = trip on quieter speech.
DEFAULT_SILERO_SENSITIVITY = 0.4
DEFAULT_WEBRTC_SENSITIVITY = 2

# Partial-transcript barge-in guard: require this many chars in the partial
# before treating it as 'real speech' (Whisper sometimes emits a single
# punctuation glyph on the leading frames).
PARTIAL_BARGE_IN_MIN_CHARS = 3


def is_available() -> tuple[bool, str]:
    """Return (True, '') if both RealtimeSTT and RealtimeTTS are importable.
    Otherwise (False, reason) so the caller can log + fall back.

    Cheap — runs two imports. Safe to call at boot to decide which
    pipeline to use.
    """
    try:
        import RealtimeSTT  # noqa: F401
    except Exception as e:
        return False, f"RealtimeSTT missing ({e})"
    try:
        import RealtimeTTS  # noqa: F401
    except Exception as e:
        return False, f"RealtimeTTS missing ({e})"
    return True, ""


class RealtimeVoicePipeline:
    """Streaming STT + TTS with single-queue barge-in.

    One background AudioToTextRecorder for input + one TextToAudioStream
    for output. The recorder fires `on_user_utterance` once per finalised
    user turn; the caller is expected to drive the LLM and pipe its
    response stream into `feed_response_chunk` / `flush_response`.
    """

    def __init__(
        self,
        on_user_utterance: Optional[Callable[[str], None]] = None,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        on_vad_start: Optional[Callable[[], None]] = None,
        on_vad_stop: Optional[Callable[[], None]] = None,
        stt_model: str = DEFAULT_STT_MODEL,
        stt_language: str = DEFAULT_STT_LANGUAGE,
        tts_engine: str = DEFAULT_TTS_ENGINE,
        tts_voice: str = DEFAULT_TTS_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        input_device: Optional[int] = None,
        silero_sensitivity: float = DEFAULT_SILERO_SENSITIVITY,
        webrtc_sensitivity: int = DEFAULT_WEBRTC_SENSITIVITY,
    ) -> None:
        self.on_user_utterance = on_user_utterance
        self.on_partial_transcript = on_partial_transcript
        self.on_barge_in = on_barge_in
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop

        self.stt_model = stt_model
        self.stt_language = stt_language
        self.tts_engine_name = (tts_engine or DEFAULT_TTS_ENGINE).lower().strip()
        self.tts_voice = tts_voice
        self.sample_rate = int(sample_rate)
        self.input_device = input_device
        self.silero_sensitivity = float(silero_sensitivity)
        self.webrtc_sensitivity = int(webrtc_sensitivity)

        self._recorder = None
        self._tts_stream = None
        self._tts_engine = None
        self._stt_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._running = False

        # Playback flag — set when stream.play_async() is in flight, cleared
        # when the on_playback_end hook fires or on barge_in(). The main loop
        # reads it to decide whether a fresh partial counts as barge-in.
        self._playing = threading.Event()

        # Track the last partial we saw so callers can poll for live captions
        # without having to wire a callback.
        self._last_partial: str = ""
        self._last_utterance: str = ""
        self._last_utterance_ts: float = 0.0
        self._last_barge_in_ts: float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._running

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def status(self) -> dict:
        return {
            "running": self._running,
            "playing": self._playing.is_set(),
            "stt_model": self.stt_model,
            "stt_language": self.stt_language,
            "tts_engine": self.tts_engine_name,
            "tts_voice": self.tts_voice,
            "last_partial": self._last_partial,
            "last_utterance": self._last_utterance,
            "last_utterance_ts": self._last_utterance_ts,
            "last_barge_in_ts": self._last_barge_in_ts,
        }

    def start(self) -> bool:
        """Bring the pipeline up. Returns True on success, False on any
        init failure (missing dep, no mic, bad TTS engine name). Failure
        prints a diagnostic; caller should fall back to turn-based."""
        ok, why = is_available()
        if not ok:
            print(f"  [realtime-voice] unavailable: {why}")
            return False
        if self._running:
            return True
        if not self._init_tts():
            return False
        if not self._init_stt():
            self._teardown_tts()
            return False
        self._stop_flag.clear()
        self._stt_thread = threading.Thread(
            target=self._stt_loop, name="rtv-stt", daemon=True,
        )
        self._stt_thread.start()
        self._running = True
        print(
            f"  [realtime-voice] started "
            f"(stt={self.stt_model}/{self.stt_language}, "
            f"tts={self.tts_engine_name})"
        )
        return True

    def stop(self) -> None:
        self._stop_flag.set()
        self._running = False
        rec = self._recorder
        self._recorder = None
        if rec is not None:
            for name in ("stop", "shutdown"):
                fn = getattr(rec, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        self._teardown_tts()

    # ── output: stream LLM tokens out as audio ────────────────────────
    def feed_response_chunk(self, text: str) -> None:
        """Append `text` to the streaming TTS queue and (re)start playback.

        Safe to call from the LLM's streaming callback — RealtimeTTS
        buffers internally and only synthesises once it has a complete
        sentence (or the stream is flushed via flush_response()).
        """
        if not text:
            return
        s = self._tts_stream
        if s is None:
            return
        try:
            s.feed(text)
            if not s.is_playing():
                self._playing.set()
                # play_async() returns immediately; chunks stream from the
                # engine into the playback ring buffer on a worker thread.
                try:
                    s.play_async(on_audio_chunk=self._on_audio_chunk)
                except TypeError:
                    # Older RealtimeTTS releases don't accept the hook kwargs.
                    s.play_async()
        except Exception as e:
            print(f"  [realtime-voice] feed failed: {e}")
            self._playing.clear()

    def flush_response(self) -> None:
        """Mark end-of-LLM-stream. Lets RealtimeTTS flush its last partial
        sentence and drain the playback queue cleanly. Idempotent."""
        s = self._tts_stream
        if s is None:
            return
        # RealtimeTTS auto-flushes on sentence boundary; this is a no-op
        # hook so callers can be symmetric (feed*, then flush_response).
        # If the engine exposes a finalise() hook we call it best-effort.
        for name in ("finalize", "finalise", "end_of_stream", "flush"):
            fn = getattr(s, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

    def wait_for_playback(self, timeout: Optional[float] = None) -> bool:
        """Block until current playback drains or `timeout` elapses.
        Returns True if playback finished, False on timeout."""
        s = self._tts_stream
        if s is None:
            return True
        deadline = (time.time() + timeout) if timeout is not None else None
        while True:
            try:
                playing = bool(s.is_playing())
            except Exception:
                playing = False
            if not playing:
                self._on_playback_end()
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.02)

    def barge_in(self) -> None:
        """Stop TTS, flush the playback queue, fire on_barge_in callback."""
        self._last_barge_in_ts = time.time()
        s = self._tts_stream
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass
        self._playing.clear()
        cb = self.on_barge_in
        if cb is not None:
            try:
                cb()
            except Exception as e:
                print(f"  [realtime-voice] on_barge_in failed: {e}")

    # ── internals ─────────────────────────────────────────────────────
    def _on_audio_chunk(self, chunk) -> None:
        # Hook for HUD lipsync / level meter. The chunk is raw int16 PCM;
        # currently a no-op for the HUD path — the HUD reads its lipsync
        # waveform from the legacy synthesise() path. We DO route the
        # chunk into the noise-cancel-1 processor's playback ring so its
        # AEC layer has a far-end reference for any concurrent legacy
        # mic capture (e.g. the wake-word listener still using sd.InputStream).
        try:
            from core import audio_processor as _ap  # type: ignore
        except Exception:
            return
        try:
            import numpy as _np
            if isinstance(chunk, (bytes, bytearray, memoryview)):
                arr = _np.frombuffer(bytes(chunk), dtype=_np.int16)
                arr = arr.astype(_np.float32) / 32767.0
            else:
                arr = _np.asarray(chunk, dtype=_np.float32)
            _ap.feed_playback(arr, sample_rate=self.sample_rate)
        except Exception:
            pass

    def _init_tts(self) -> bool:
        try:
            from RealtimeTTS import TextToAudioStream
            engine = self._build_tts_engine()
            if engine is None:
                return False
            self._tts_engine = engine
            # on_audio_stream_stop is intentionally not passed: older
            # RealtimeTTS builds raise TypeError on the kwarg, and the
            # fallback path used to silently drop the hook, stranding
            # _playing in the set state. wait_for_playback() now drives
            # _on_playback_end() directly, so the hook is redundant.
            self._tts_stream = TextToAudioStream(engine)
            return True
        except Exception as e:
            print(f"  [realtime-voice] TTS init failed: {e}")
            return False

    def _build_tts_engine(self):
        name = self.tts_engine_name
        try:
            if name == "system":
                from RealtimeTTS import SystemEngine
                # SystemEngine uses Windows SAPI on Windows; voice kwarg
                # is a SAPI voice name, not an Edge neural voice. Passing
                # the Edge voice string is harmless — SAPI ignores
                # unknown names and falls back to the default voice.
                try:
                    return SystemEngine(voice=self.tts_voice)
                except TypeError:
                    return SystemEngine()
            if name == "azure":
                from RealtimeTTS import AzureEngine
                key = os.environ.get("AZURE_TTS_KEY", "").strip()
                region = os.environ.get("AZURE_TTS_REGION", "").strip()
                if not key:
                    print("  [realtime-voice] AZURE_TTS_KEY env var not set")
                    return None
                return AzureEngine(
                    speech_key=key,
                    speech_region=region or "eastus",
                    voice=self.tts_voice,
                )
            if name == "elevenlabs":
                from RealtimeTTS import ElevenlabsEngine
                key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
                if not key:
                    print("  [realtime-voice] ELEVENLABS_API_KEY env var not set")
                    return None
                try:
                    return ElevenlabsEngine(api_key=key, voice=self.tts_voice)
                except TypeError:
                    return ElevenlabsEngine(key, voice=self.tts_voice)
            if name == "coqui":
                from RealtimeTTS import CoquiEngine
                try:
                    return CoquiEngine(voice=self.tts_voice or None)
                except TypeError:
                    return CoquiEngine()
            print(f"  [realtime-voice] unknown TTS engine '{name}'")
            return None
        except ImportError as e:
            print(f"  [realtime-voice] engine '{name}' not installed: {e}")
            return None
        except Exception as e:
            print(f"  [realtime-voice] engine '{name}' load failed: {e}")
            return None

    def _teardown_tts(self) -> None:
        s = self._tts_stream
        self._tts_stream = None
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass
        eng = self._tts_engine
        self._tts_engine = None
        if eng is not None:
            shutdown = getattr(eng, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass

    def _init_stt(self) -> bool:
        try:
            from RealtimeSTT import AudioToTextRecorder
        except Exception as e:
            print(f"  [realtime-voice] RealtimeSTT import failed: {e}")
            return False

        kwargs = dict(
            model=self.stt_model,
            language=self.stt_language,
            spinner=False,
            use_microphone=True,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self._on_partial,
            on_realtime_transcription_stabilized=self._on_partial,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            silero_sensitivity=self.silero_sensitivity,
            webrtc_sensitivity=self.webrtc_sensitivity,
        )
        if self.input_device is not None:
            kwargs["input_device_index"] = int(self.input_device)
        try:
            self._recorder = AudioToTextRecorder(**kwargs)
        except TypeError:
            # Older RealtimeSTT releases drop unknown kwargs only when
            # explicit; pare back to the minimal set on TypeError.
            self._recorder = AudioToTextRecorder(
                model=self.stt_model,
                language=self.stt_language,
                spinner=False,
                use_microphone=True,
                enable_realtime_transcription=True,
                on_realtime_transcription_update=self._on_partial,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
            )
        except Exception as e:
            print(f"  [realtime-voice] STT recorder init failed: {e}")
            self._recorder = None
            return False
        return True

    def _stt_loop(self) -> None:
        """Pump finalised utterances out of RealtimeSTT until stop()."""
        rec = self._recorder
        if rec is None:
            return
        while not self._stop_flag.is_set():
            try:
                text = rec.text()
            except Exception as e:
                if self._stop_flag.is_set():
                    return
                print(f"  [realtime-voice] STT loop error: {e}")
                time.sleep(0.5)
                continue
            if not text or not text.strip():
                # Some RealtimeSTT builds return empty immediately when no
                # utterance is ready; sleep briefly so this path can't
                # busy-spin and peg a core.
                time.sleep(0.05)
                continue
            self._last_utterance = text
            self._last_utterance_ts = time.time()
            cb = self.on_user_utterance
            if cb is None:
                continue
            try:
                cb(text)
            except Exception as e:
                print(f"  [realtime-voice] on_user_utterance failed: {e}")

    def _on_partial(self, text: str) -> None:
        self._last_partial = text or ""
        cb = self.on_partial_transcript
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass
        # Partial-transcript barge-in fallback. The VAD-start hook is the
        # primary trigger; this catches the cases where echo cancellation
        # suppressed VAD-start during loud playback but Whisper still
        # produced characters.
        if (
            self._playing.is_set()
            and text
            and len(text.strip()) >= PARTIAL_BARGE_IN_MIN_CHARS
        ):
            self.barge_in()

    def _on_recording_start(self) -> None:
        cb = self.on_vad_start
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
        # Primary barge-in trigger.
        if self._playing.is_set():
            self.barge_in()

    def _on_recording_stop(self) -> None:
        cb = self.on_vad_stop
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _on_playback_end(self) -> None:
        self._playing.clear()


# ──────────────────────────────────────────────────────────────────────
#  Module-level convenience used by bobert_companion's boot path.
#
#  start_pipeline(...) honours the VOICE_MODE knob and returns either a
#  running pipeline or None (signal: stick with the turn-based loop).
# ──────────────────────────────────────────────────────────────────────

_singleton: Optional[RealtimeVoicePipeline] = None
_singleton_lock = threading.Lock()


def get_pipeline() -> Optional[RealtimeVoicePipeline]:
    """Return the live singleton pipeline, or None if not started."""
    return _singleton


def start_pipeline(
    *,
    voice_mode: str = "turn_based",
    on_user_utterance: Optional[Callable[[str], None]] = None,
    on_partial_transcript: Optional[Callable[[str], None]] = None,
    on_barge_in: Optional[Callable[[], None]] = None,
    stt_model: str = DEFAULT_STT_MODEL,
    stt_language: str = DEFAULT_STT_LANGUAGE,
    tts_engine: str = DEFAULT_TTS_ENGINE,
    tts_voice: str = DEFAULT_TTS_VOICE,
    input_device: Optional[int] = None,
) -> Optional[RealtimeVoicePipeline]:
    """Honour the VOICE_MODE knob and return a live pipeline (or None).

    Returns None when VOICE_MODE='turn_based' or when is_available()
    reports a missing dep. Caller should treat None as 'keep using the
    legacy turn-based loop' and only branch into the realtime code path
    when the return value is truthy.
    """
    global _singleton
    if (voice_mode or "turn_based").lower() != "realtime":
        return None
    with _singleton_lock:
        if _singleton is not None and _singleton.is_running():
            return _singleton
        pipe = RealtimeVoicePipeline(
            on_user_utterance=on_user_utterance,
            on_partial_transcript=on_partial_transcript,
            on_barge_in=on_barge_in,
            stt_model=stt_model,
            stt_language=stt_language,
            tts_engine=tts_engine,
            tts_voice=tts_voice,
            input_device=input_device,
        )
        if not pipe.start():
            return None
        _singleton = pipe
        return pipe


def stop_pipeline() -> None:
    """Stop the singleton pipeline if running."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            return
        try:
            _singleton.stop()
        except Exception:
            pass
        _singleton = None


# ──────────────────────────────────────────────────────────────────────
#  Self-test
#
#  Smoke check: import the module, confirm is_available() runs and the
#  pipeline can be constructed without side effects. Does NOT spin up a
#  mic — that would require a working audio device which CI doesn't have.
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RealtimeVoicePipeline smoke test")
    ok, why = is_available()
    print(f"  is_available() = {ok} ({why or 'ok'})")
    pipe = RealtimeVoicePipeline(
        on_user_utterance=lambda t: print(f"  user: {t!r}"),
        on_partial_transcript=lambda t: print(f"  …: {t!r}"),
        on_barge_in=lambda: print("  BARGE IN"),
    )
    print(f"  running={pipe.is_running()} playing={pipe.is_playing()}")
    print(f"  status={pipe.status()}")
    # We do NOT call start() here — that requires a mic + downloads
    # whisper-base. The smoke test only verifies the module imports and
    # the public API is wired.
    print("  OK")
